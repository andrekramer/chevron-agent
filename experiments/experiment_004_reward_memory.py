"""Experiment 004: delayed scalar reward and protected online memory.

This is a contextual-bandit RL bridge, not a supervised compatibility task.
Learned comparators are optimised with REINFORCE from delayed scalar reward
only. A negative reward does not reveal the correct one of four actions.
Latent category IDs and correct actions are retained by the runner solely for
auditing and never enter an agent observation or learning call.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from chevron_agent import (
    DirectPairMLP,
    ProjectedBilinearNullAttention,
    ProjectedCosineAssent,
)


CONDITIONS = (
    "content_attention_buffer",
    "direct_mlp_buffer",
    "chevron_buffer",
    "chevron_immediate",
    "chevron_coupled_write",
)

DISPLAY_NAMES = {
    "content_attention_buffer": "Content attention + buffer",
    "direct_mlp_buffer": "Direct MLP + buffer",
    "chevron_buffer": "Chevron + buffer",
    "chevron_immediate": "Chevron immediate write",
    "chevron_coupled_write": "Chevron coupled write",
}


@dataclass(frozen=True)
class ExperimentConfig:
    training_seeds: int = 2
    seed_offset: int = 0
    training_lifetimes: int = 60
    evaluation_lifetimes: int = 10
    groups: int = 4
    action_dim: int = 4
    content_dim: int = 12
    comparison_dim: int = 13
    direct_hidden_dim: int = 12
    stream_steps: int = 600
    shift_step: int = 200
    outcome_delay: int = 3
    novel_probability: float = 0.35
    evidence_noise: float = 0.15
    hard_noise: float = 0.32
    hard_noise_probability: float = 0.10
    novel_anchor_cosine: float = 0.55
    permanent_capacity: int = 12
    buffer_capacity: int = 2
    promotion_support: int = 2
    null_threshold: float = 0.80
    admitted_threshold: float = 0.25
    standard_similarity_threshold: float = 0.62
    standard_temperature: float = 0.10
    write_threshold_margin: float = 0.05
    value_update_rate: float = 0.35
    content_update_rate: float = 0.05
    policy_scale: float = 4.0
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    entropy_coefficient: float = 0.01
    retrospective_loss_weight: float = 0.0


@dataclass(frozen=True)
class AgentObservation:
    event_id: int
    family: int
    evidence: Tensor


@dataclass(frozen=True)
class AuditEvent:
    observation: AgentObservation
    category: int
    correct_action: int
    is_novel: bool


@dataclass(frozen=True)
class Lifetime:
    events: tuple[AuditEvent, ...]
    prototypes: Tensor
    correct_actions: Tensor
    initial_categories: tuple[int, ...]
    novel_categories: tuple[int, ...]


@dataclass
class MemorySlot:
    family: int
    content: Tensor
    action_values: Tensor
    last_used: int
    established: bool
    promotion_id: int | None = None
    origin_content: Tensor | None = None


@dataclass(frozen=True)
class ReadTrace:
    slot_mass: Tensor
    q: Tensor
    selected_slot: int
    should_candidate: bool
    read_assent: Tensor
    write_assent: Tensor


@dataclass
class PendingAction:
    observation: AgentObservation
    action: int
    log_probabilities: Tensor
    entropy: Tensor
    selected_action_support: Tensor
    trace: ReadTrace
    immediate_slot: int | None = None
    candidate_id: int | None = None


@dataclass
class CandidateCluster:
    candidate_id: int
    family: int
    content: Tensor
    observation_count: int
    pending_events: set[int]
    positive_action_counts: Tensor
    last_seen: int

class ProvisionalCandidateBank:
    """Capacity-limited unresolved clusters, separate from reward eligibility."""

    def __init__(self, capacity: int, match_similarity: float) -> None:
        if capacity <= 0:
            raise ValueError("candidate capacity must be positive")
        self.capacity = capacity
        self.match_similarity = match_similarity
        self.entries: list[CandidateCluster] = []
        self.next_id = 0

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, observation: AgentObservation) -> tuple[int, CandidateCluster | None]:
        matches = [
            float(entry.content @ observation.evidence)
            if entry.family == observation.family
            else -1.0
            for entry in self.entries
        ]
        evicted = None
        if matches and max(matches) >= self.match_similarity:
            entry = self.entries[int(torch.tensor(matches).argmax())]
            count = entry.observation_count + 1
            entry.content = _unit(
                (entry.content * entry.observation_count + observation.evidence.detach())
                / count
            )
            entry.observation_count = count
            entry.pending_events.add(observation.event_id)
            entry.last_seen = observation.event_id
            return entry.candidate_id, None

        if len(self.entries) >= self.capacity:
            index = min(range(len(self.entries)), key=lambda item: self.entries[item].last_seen)
            evicted = self.entries.pop(index)
        candidate_id = self.next_id
        self.next_id += 1
        self.entries.append(
            CandidateCluster(
                candidate_id=candidate_id,
                family=observation.family,
                content=observation.evidence.detach().clone(),
                observation_count=1,
                pending_events={observation.event_id},
                positive_action_counts=torch.zeros(4, dtype=torch.long),
                last_seen=observation.event_id,
            )
        )
        return candidate_id, evicted

    def resolve(
        self,
        candidate_id: int,
        event_id: int,
        action: int,
        reward: float,
        *,
        support: int,
    ) -> CandidateCluster | None:
        entry = next(
            (candidate for candidate in self.entries if candidate.candidate_id == candidate_id),
            None,
        )
        if entry is None or event_id not in entry.pending_events:
            return None
        entry.pending_events.remove(event_id)
        if reward > 0.0:
            entry.positive_action_counts[action] += 1
        if int(entry.positive_action_counts.max()) >= support:
            self.entries.remove(entry)
            return entry
        return None


@dataclass(frozen=True)
class Resolution:
    event_id: int
    promotion_id: int | None


@dataclass
class AgentMetrics:
    premature_writes: int = 0
    established_overwrites: int = 0
    false_candidates: int = 0
    buffer_evictions: int = 0
    promotions: int = 0
    permanent_writes: int = 0
    admitted_reads: int = 0
    write_permissions: int = 0
    read_gate_sum: float = 0.0
    write_gate_sum: float = 0.0
    gate_observations: int = 0


@dataclass
class LifetimeMetrics:
    condition: str
    training_seed: int
    lifetime_seed: int
    return_per_decision: float
    overall_accuracy: float
    final_old_accuracy: float
    final_new_accuracy: float
    old_probe_accuracy: float
    new_probe_accuracy: float
    category_coverage: float
    unresolved_q: float
    resolved_q: float
    residual_calibration: float
    premature_write_rate: float
    established_overwrite_rate: float
    established_drift: float
    false_candidate_rate: float
    buffer_evictions: float
    promotions: float
    promotion_precision: float
    read_write_margin: float
    retention_plasticity_score: float


def _unit(vector: Tensor) -> Tensor:
    return F.normalize(vector, dim=-1)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _mean_sd(values: list[float]) -> dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"mean": float("nan"), "population_sd": float("nan")}
    return {
        "mean": statistics.fmean(finite),
        "population_sd": statistics.pstdev(finite),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def retrospective_consistency_loss(
    selected_action_support: Tensor,
    reward: float,
) -> Tensor:
    """Score whether admitted memory predicted the delayed action outcome.

    The target is derived only from the chosen action's scalar reward.  Slot
    identities, latent categories, and the unobserved correct action are not
    inputs.  Positive reward reinforces admitted support for the selected
    action; non-positive reward suppresses it.
    """

    target = selected_action_support.new_tensor(1.0 if reward > 0.0 else 0.0)
    return F.binary_cross_entropy(
        selected_action_support.clamp(1e-5, 1.0 - 1e-5),
        target,
    )


def make_lifetime(config: ExperimentConfig, seed: int) -> Lifetime:
    generator = torch.Generator().manual_seed(seed)
    prototypes: list[Tensor] = []
    initial_categories: list[int] = []
    novel_categories: list[int] = []
    correct_actions = torch.empty(config.groups * 3, dtype=torch.long)

    for family in range(config.groups):
        first = _unit(torch.randn(config.content_dim, generator=generator))
        second = _unit(torch.randn(config.content_dim, generator=generator))
        while float(first @ second) > 0.35:
            second = _unit(torch.randn(config.content_dim, generator=generator))
        direction = torch.randn(config.content_dim, generator=generator)
        orthogonal = _unit(direction - (direction @ first) * first)
        novel = (
            config.novel_anchor_cosine * first
            + math.sqrt(1.0 - config.novel_anchor_cosine**2) * orthogonal
        )
        prototypes.extend((first, second, novel))
        action_order = torch.randperm(config.action_dim, generator=generator)
        correct_actions[3 * family] = action_order[0]
        correct_actions[3 * family + 1] = action_order[1]
        correct_actions[3 * family + 2] = action_order[2]
        initial_categories.extend((3 * family, 3 * family + 1))
        novel_categories.append(3 * family + 2)

    prototype_tensor = torch.stack(prototypes)
    events: list[AuditEvent] = []
    for step in range(config.stream_steps):
        use_novel = step >= config.shift_step and float(torch.rand((), generator=generator)) < config.novel_probability
        pool = novel_categories if use_novel else initial_categories
        category = pool[int(torch.randint(len(pool), (), generator=generator))]
        noise = (
            config.hard_noise
            if float(torch.rand((), generator=generator)) < config.hard_noise_probability
            else config.evidence_noise
        )
        evidence = _unit(
            prototype_tensor[category]
            + noise * torch.randn(config.content_dim, generator=generator)
        )
        observation = AgentObservation(event_id=step, family=category // 3, evidence=evidence)
        events.append(
            AuditEvent(
                observation=observation,
                category=category,
                correct_action=int(correct_actions[category]),
                is_novel=category in novel_categories,
            )
        )
    return Lifetime(
        events=tuple(events),
        prototypes=prototype_tensor,
        correct_actions=correct_actions,
        initial_categories=tuple(initial_categories),
        novel_categories=tuple(novel_categories),
    )


class RewardMemoryAgent:
    def __init__(
        self,
        condition: str,
        config: ExperimentConfig,
        lifetime: Lifetime,
        *,
        model: ProjectedCosineAssent | DirectPairMLP | ProjectedBilinearNullAttention | None,
        training: bool,
        action_seed: int,
    ) -> None:
        self.condition = condition
        self.config = config
        self.model = model
        self.training = training
        self.action_generator = torch.Generator().manual_seed(action_seed)
        self.slots = [
            MemorySlot(
                family=category // 3,
                content=lifetime.prototypes[category].detach().clone(),
                action_values=F.one_hot(
                    lifetime.correct_actions[category],
                    num_classes=config.action_dim,
                ).to(torch.float32),
                last_used=-1,
                established=True,
                origin_content=lifetime.prototypes[category].detach().clone(),
            )
            for category in lifetime.initial_categories
        ]
        self.buffer = ProvisionalCandidateBank(
            config.buffer_capacity,
            config.standard_similarity_threshold,
        )
        self.pending: dict[int, PendingAction] = {}
        self.loss_terms: list[Tensor] = []
        self.metrics = AgentMetrics()
        self.next_promotion_id = 0

    @property
    def is_chevron(self) -> bool:
        return self.condition.startswith("chevron")

    def _content(self) -> Tensor:
        return torch.stack([slot.content for slot in self.slots])

    def _family_retrieval(self, family: int) -> Tensor:
        same = torch.tensor([slot.family == family for slot in self.slots])
        return same.to(torch.float32) / same.sum().clamp_min(1)

    def _distribution(self, observation: AgentObservation) -> ReadTrace:
        content = self._content()
        alpha = self._family_retrieval(observation.family)
        evidence = observation.evidence.unsqueeze(0)
        if self.condition == "content_attention_buffer":
            similarities = content @ observation.evidence
            masked = torch.where(alpha > 0, similarities / self.config.standard_temperature, torch.full_like(similarities, -1e9))
            slot_mass = torch.softmax(masked, dim=-1)
            best_similarity = similarities.masked_select(alpha > 0).max()
            q = torch.sigmoid(20.0 * (self.config.standard_similarity_threshold - best_similarity))
            slot_mass = slot_mass * (1.0 - q)
            assent = torch.ones_like(slot_mass) * (1.0 - q)
            write_assent = assent
            should_candidate = bool(q.detach() > self.config.null_threshold)
        elif self.condition == "direct_mlp_buffer":
            assert isinstance(self.model, DirectPairMLP)
            output = self.model(evidence, content.unsqueeze(0), retrieval_mass=alpha.unsqueeze(0))
            slot_mass = output.slot_mass.squeeze(0)
            q = output.null_mass.squeeze(0)
            assent = slot_mass / alpha.clamp_min(1e-8)
            write_assent = assent
            should_candidate = bool(q.detach() > slot_mass.max().detach())
        elif self.condition == "bilinear_buffer":
            assert isinstance(self.model, ProjectedBilinearNullAttention)
            output = self.model(
                evidence,
                content.unsqueeze(0),
                retrieval_mass=alpha.unsqueeze(0),
            )
            slot_mass = output.slot_mass.squeeze(0)
            q = output.null_mass.squeeze(0)
            assent = slot_mass / alpha.clamp_min(1e-8)
            write_assent = assent
            should_candidate = bool(
                q.detach() > self.config.null_threshold
                and slot_mass.max().detach() < self.config.admitted_threshold
            )
        else:
            assert isinstance(self.model, ProjectedCosineAssent)
            output = self.model(evidence, content.unsqueeze(0))
            assent = output.assent.squeeze(0)
            slot_mass = alpha * assent
            q = 1.0 - slot_mass.sum()
            if self.condition == "chevron_coupled_write":
                write_assent = assent
            else:
                write_assent = self.model.assent_with_margin(
                    evidence,
                    content.unsqueeze(0),
                    threshold_margin=self.config.write_threshold_margin,
                ).squeeze(0)
            should_candidate = bool(
                q.detach() > self.config.null_threshold
                and slot_mass.max().detach() < self.config.admitted_threshold
            )
        selected = int(slot_mass.argmax().detach())
        return ReadTrace(
            slot_mass=slot_mass,
            q=q,
            selected_slot=selected,
            should_candidate=should_candidate,
            read_assent=assent,
            write_assent=write_assent,
        )

    def act(self, observation: AgentObservation) -> tuple[int, ReadTrace]:
        trace = self._distribution(observation)
        values = torch.stack([slot.action_values for slot in self.slots])
        action_scores = torch.einsum("s,sa->a", trace.slot_mass, values)
        logits = self.config.policy_scale * action_scores
        probabilities = torch.softmax(logits, dim=0)
        if self.training:
            action = int(
                torch.multinomial(
                    probabilities.detach(),
                    num_samples=1,
                    generator=self.action_generator,
                )
            )
        elif trace.should_candidate:
            action = int(
                torch.randint(
                    self.config.action_dim,
                    (),
                    generator=self.action_generator,
                )
            )
        else:
            action = int(probabilities.argmax().detach())
        log_probabilities = torch.log(probabilities.clamp_min(1e-8))
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum()
        selected_action_support = torch.sum(
            trace.slot_mass * values[:, action].clamp(min=0.0, max=1.0)
        )
        pending = PendingAction(
            observation=observation,
            action=action,
            log_probabilities=log_probabilities,
            entropy=entropy,
            selected_action_support=selected_action_support,
            trace=trace,
        )
        if trace.slot_mass.sum().detach() >= 0.5:
            self.metrics.admitted_reads += 1
        selected = trace.selected_slot
        alpha = self._family_retrieval(observation.family)
        self.metrics.read_gate_sum += float((alpha[selected] * trace.read_assent[selected]).detach())
        self.metrics.write_gate_sum += float((alpha[selected] * trace.write_assent[selected]).detach())
        self.metrics.gate_observations += 1
        if trace.should_candidate:
            if self.condition == "chevron_immediate":
                pending.immediate_slot = self._allocate(
                    observation,
                    F.one_hot(
                        torch.tensor(action),
                        num_classes=self.config.action_dim,
                    ).to(torch.float32),
                    premature=True,
                )
            else:
                pending.candidate_id, evicted = self.buffer.add(observation)
                if evicted is not None:
                    self.metrics.buffer_evictions += 1
        self.pending[observation.event_id] = pending
        return action, trace

    def resolve_reward(self, event_id: int, reward: float) -> Resolution:
        pending = self.pending.pop(event_id)
        if self.training and self.model is not None:
            loss = (
                -reward * pending.log_probabilities[pending.action]
                - self.config.entropy_coefficient * pending.entropy
            )
            if self.config.retrospective_loss_weight > 0.0:
                consistency_loss = retrospective_consistency_loss(
                    pending.selected_action_support,
                    reward,
                )
                loss = loss + self.config.retrospective_loss_weight * consistency_loss
            self.loss_terms.append(loss)
        promotion_id: int | None = None
        if pending.immediate_slot is not None:
            if pending.immediate_slot < len(self.slots):
                self._update_slot(
                    pending.immediate_slot,
                    pending.observation,
                    pending.action,
                    reward,
                    1.0,
                )
            return Resolution(event_id, None)

        if pending.candidate_id is not None:
            candidate = self.buffer.resolve(
                pending.candidate_id,
                event_id,
                pending.action,
                reward,
                support=self.config.promotion_support,
            )
        else:
            candidate = None
        if candidate is not None:
            candidate_observation = AgentObservation(
                event_id=event_id,
                family=candidate.family,
                evidence=candidate.content,
            )
            matched = self._matching_slot(candidate_observation)
            winning_action = int(candidate.positive_action_counts.argmax())
            candidate_values = F.one_hot(
                torch.tensor(winning_action),
                num_classes=self.config.action_dim,
            ).to(torch.float32)
            if matched is None:
                index = self._allocate(
                    candidate_observation,
                    candidate_values,
                    premature=False,
                )
                promotion_id = self.slots[index].promotion_id
            else:
                self.metrics.false_candidates += 1
                self._update_slot(
                    matched,
                    candidate_observation,
                    winning_action,
                    1.0,
                    1.0,
                )
            return Resolution(event_id, promotion_id)

        if pending.trace.should_candidate or pending.candidate_id is not None:
            return Resolution(event_id, None)
        selected = pending.trace.selected_slot
        alpha = self._family_retrieval(pending.observation.family)
        write_gate = float((alpha[selected] * pending.trace.write_assent[selected]).detach())
        if write_gate > 0.0:
            self.metrics.write_permissions += int(write_gate >= 0.25)
            self._update_slot(
                selected,
                pending.observation,
                pending.action,
                reward,
                write_gate,
            )
        return Resolution(event_id, None)

    def _matching_slot(self, observation: AgentObservation) -> int | None:
        similarities = self._content() @ observation.evidence
        family_mask = torch.tensor([slot.family == observation.family for slot in self.slots])
        similarities = torch.where(family_mask, similarities, torch.full_like(similarities, -1.0))
        best = int(similarities.argmax())
        return best if float(similarities[best]) >= self.config.standard_similarity_threshold else None

    def _allocate(
        self,
        observation: AgentObservation,
        action_values: Tensor,
        *,
        premature: bool,
    ) -> int:
        promotion_id = self.next_promotion_id
        self.next_promotion_id += 1
        slot = MemorySlot(
            family=observation.family,
            content=observation.evidence.detach().clone(),
            action_values=action_values.detach().clone(),
            last_used=observation.event_id,
            established=False,
            promotion_id=promotion_id,
            origin_content=None,
        )
        if len(self.slots) < self.config.permanent_capacity:
            self.slots.append(slot)
            index = len(self.slots) - 1
        else:
            provisional = [index for index, existing in enumerate(self.slots) if not existing.established]
            pool = provisional or list(range(len(self.slots)))
            index = min(pool, key=lambda item: self.slots[item].last_used)
            if self.slots[index].established:
                self.metrics.established_overwrites += 1
            self.slots[index] = slot
        self.metrics.promotions += int(not premature)
        self.metrics.premature_writes += int(premature)
        self.metrics.permanent_writes += 1
        return index

    def _update_slot(
        self,
        index: int,
        observation: AgentObservation,
        action: int,
        reward: float,
        gate: float,
    ) -> None:
        slot = self.slots[index]
        gate = min(max(gate, 0.0), 1.0)
        value_rate = self.config.value_update_rate * gate
        content_rate = self.config.content_update_rate * gate
        slot.action_values[action] = (
            (1.0 - value_rate) * slot.action_values[action]
            + value_rate * reward
        )
        slot.content = _unit(
            (1.0 - content_rate) * slot.content
            + content_rate * observation.evidence.detach()
        )
        slot.last_used = observation.event_id
        self.metrics.permanent_writes += 1

    def training_loss(self) -> Tensor:
        if not self.loss_terms:
            raise RuntimeError("no reward-derived training terms")
        return torch.stack(self.loss_terms).mean()


def _probe(
    agent: RewardMemoryAgent,
    lifetime: Lifetime,
    categories: tuple[int, ...],
) -> float:
    correct = 0
    with torch.no_grad():
        for offset, category in enumerate(categories):
            observation = AgentObservation(
                event_id=agent.config.stream_steps + agent.config.outcome_delay + offset,
                family=category // 3,
                evidence=lifetime.prototypes[category],
            )
            trace = agent._distribution(observation)
            values = torch.stack([slot.action_values for slot in agent.slots])
            scores = torch.einsum("s,sa->a", trace.slot_mass, values)
            action = int(scores.argmax())
            correct += int(action == int(lifetime.correct_actions[category]))
    return correct / len(categories)


def _distinct_category_coverage(
    agent: RewardMemoryAgent,
    lifetime: Lifetime,
) -> float:
    """Maximum one-to-one category/slot matching above the similarity floor."""

    adjacency: dict[int, list[int]] = {}
    for category in range(len(lifetime.prototypes)):
        adjacency[category] = [
            slot_index
            for slot_index, slot in enumerate(agent.slots)
            if slot.family == category // 3
            and float(slot.content @ lifetime.prototypes[category])
            >= agent.config.standard_similarity_threshold
        ]

    matched_slot: dict[int, int] = {}

    def assign(category: int, visited: set[int]) -> bool:
        for slot_index in adjacency[category]:
            if slot_index in visited:
                continue
            visited.add(slot_index)
            previous = matched_slot.get(slot_index)
            if previous is None or assign(previous, visited):
                matched_slot[slot_index] = category
                return True
        return False

    matched = sum(assign(category, set()) for category in adjacency)
    return matched / len(lifetime.prototypes)


def run_lifetime(
    condition: str,
    config: ExperimentConfig,
    lifetime: Lifetime,
    *,
    model: ProjectedCosineAssent | DirectPairMLP | ProjectedBilinearNullAttention | None,
    training: bool,
    training_seed: int,
    lifetime_seed: int,
) -> tuple[LifetimeMetrics, Tensor | None]:
    agent = RewardMemoryAgent(
        condition,
        config,
        lifetime,
        model=model,
        training=training,
        action_seed=50_000_000 + 1000 * training_seed + lifetime_seed,
    )
    due: dict[int, list[tuple[int, float, int]]] = {}
    decisions: list[dict[str, Any]] = []
    resolved_novel: set[int] = set()
    promotion_category: dict[int, int] = {}
    useful_promotions: set[int] = set()

    for step, event in enumerate(lifetime.events):
        for event_id, reward, category in due.get(step, []):
            resolution = agent.resolve_reward(event_id, reward)
            if resolution.promotion_id is not None:
                promotion_category[resolution.promotion_id] = category
            if category in lifetime.novel_categories:
                resolved_novel.add(category)

        action, trace = agent.act(event.observation)
        correct = action == event.correct_action
        selected_promotion = agent.slots[trace.selected_slot].promotion_id
        if selected_promotion is not None and promotion_category.get(selected_promotion) == event.category and correct:
            useful_promotions.add(selected_promotion)
        reward = 1.0 if correct else -1.0
        due.setdefault(step + config.outcome_delay, []).append(
            (event.observation.event_id, reward, event.category)
        )
        decisions.append(
            {
                "step": step,
                "category": event.category,
                "novel": event.is_novel,
                "correct": correct,
                "reward": reward,
                "q": float(trace.q.detach()),
                "resolved": event.category in resolved_novel,
            }
        )

    for step in range(config.stream_steps, config.stream_steps + config.outcome_delay):
        for event_id, reward, category in due.get(step, []):
            resolution = agent.resolve_reward(event_id, reward)
            if resolution.promotion_id is not None:
                promotion_category[resolution.promotion_id] = category

    final = [item for item in decisions if item["step"] >= config.stream_steps - 200]
    old_final = [item["correct"] for item in final if not item["novel"]]
    new_final = [item["correct"] for item in final if item["novel"]]
    unresolved_q = [item["q"] for item in decisions if item["novel"] and not item["resolved"]]
    resolved_q = [item["q"] for item in decisions if item["novel"] and item["resolved"]]
    old_probe = _probe(agent, lifetime, lifetime.initial_categories)
    new_probe = _probe(agent, lifetime, lifetime.novel_categories)
    established_drifts = [
        float(torch.linalg.vector_norm(slot.content - slot.origin_content))
        for slot in agent.slots
        if slot.established and slot.origin_content is not None
    ]
    premature_rate = agent.metrics.premature_writes / config.stream_steps
    overwrite_rate = agent.metrics.established_overwrites / (8 * config.stream_steps)
    final_old = _mean([float(value) for value in old_final])
    final_new = _mean([float(value) for value in new_final])
    promotion_precision = (
        len(useful_promotions) / len(promotion_category)
        if promotion_category
        else float("nan")
    )
    metrics = LifetimeMetrics(
        condition=condition,
        training_seed=training_seed,
        lifetime_seed=lifetime_seed,
        return_per_decision=_mean([item["reward"] for item in decisions]),
        overall_accuracy=_mean([float(item["correct"]) for item in decisions]),
        final_old_accuracy=final_old,
        final_new_accuracy=final_new,
        old_probe_accuracy=old_probe,
        new_probe_accuracy=new_probe,
        category_coverage=_distinct_category_coverage(agent, lifetime),
        unresolved_q=_mean(unresolved_q),
        resolved_q=_mean(resolved_q),
        residual_calibration=_mean(unresolved_q) - _mean(resolved_q),
        premature_write_rate=premature_rate,
        established_overwrite_rate=overwrite_rate,
        established_drift=_mean(established_drifts),
        false_candidate_rate=agent.metrics.false_candidates / config.stream_steps,
        buffer_evictions=float(agent.metrics.buffer_evictions),
        promotions=float(agent.metrics.promotions),
        promotion_precision=promotion_precision,
        read_write_margin=(
            (agent.metrics.read_gate_sum - agent.metrics.write_gate_sum)
            / max(1, agent.metrics.gate_observations)
        ),
        retention_plasticity_score=final_old + final_new - 0.5 * overwrite_rate - 0.5 * premature_rate,
    )
    loss = agent.training_loss() if training and model is not None else None
    return metrics, loss


def _train_seed(
    config: ExperimentConfig,
    seed: int,
) -> tuple[ProjectedCosineAssent, DirectPairMLP, dict[str, list[float]]]:
    torch.manual_seed(10_000 + seed)
    chevron = ProjectedCosineAssent(
        config.content_dim,
        config.content_dim,
        config.comparison_dim,
        initial_threshold=0.25,
        initial_slope=8.0,
    )
    direct = DirectPairMLP(config.content_dim, config.content_dim, config.direct_hidden_dim)
    if sum(parameter.numel() for parameter in chevron.parameters()) != 314:
        raise RuntimeError("unexpected Chevron parameter count")
    if sum(parameter.numel() for parameter in direct.parameters()) != 314:
        raise RuntimeError("unexpected direct comparator parameter count")
    optimizers = {
        "chevron": torch.optim.AdamW(
            chevron.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        ),
        "direct": torch.optim.AdamW(
            direct.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        ),
    }
    losses = {"chevron": [], "direct": []}
    for lifetime_index in range(config.training_lifetimes):
        lifetime_seed = 1_000_000 + 10_000 * seed + lifetime_index
        lifetime = make_lifetime(config, lifetime_seed)
        for name, condition, model in (
            ("chevron", "chevron_buffer", chevron),
            ("direct", "direct_mlp_buffer", direct),
        ):
            _, loss = run_lifetime(
                condition,
                config,
                lifetime,
                model=model,
                training=True,
                training_seed=seed,
                lifetime_seed=lifetime_seed,
            )
            assert loss is not None
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            losses[name].append(float(loss.detach()))
    return chevron, direct, losses


def _aggregate(results: list[LifetimeMetrics]) -> dict[str, dict[str, dict[str, float]]]:
    fields = [
        name
        for name in LifetimeMetrics.__dataclass_fields__
        if name not in {"condition", "training_seed", "lifetime_seed"}
    ]
    output: dict[str, dict[str, dict[str, float]]] = {}
    for condition in CONDITIONS:
        rows = [row for row in results if row.condition == condition]
        output[condition] = {
            field: _mean_sd([float(getattr(row, field)) for row in rows])
            for field in fields
        }
    return output


def _paired(results: list[LifetimeMetrics]) -> dict[str, dict[str, float]]:
    by_key = {(row.condition, row.training_seed, row.lifetime_seed): row for row in results}
    keys = sorted({(row.training_seed, row.lifetime_seed) for row in results})
    comparisons = (
        ("chevron_buffer", "content_attention_buffer"),
        ("chevron_buffer", "direct_mlp_buffer"),
        ("chevron_buffer", "chevron_immediate"),
        ("chevron_buffer", "chevron_coupled_write"),
    )
    output: dict[str, dict[str, float]] = {}
    for left, right in comparisons:
        label = f"{left}_minus_{right}"
        output[label] = {}
        for metric in ("return_per_decision", "final_old_accuracy", "final_new_accuracy"):
            values = [
                float(getattr(by_key[(left, *key)], metric) - getattr(by_key[(right, *key)], metric))
                for key in keys
            ]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            se = sd / math.sqrt(len(values))
            output[label][f"{metric}_mean"] = mean
            output[label][f"{metric}_sd"] = sd
            output[label][f"{metric}_wins"] = sum(value > 0 for value in values)
            output[label][f"{metric}_approx_95ci_low"] = mean - 1.96 * se
            output[label][f"{metric}_approx_95ci_high"] = mean + 1.96 * se
    return output


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    all_results: list[LifetimeMetrics] = []
    training_records: list[dict[str, Any]] = []
    for training_seed in range(config.seed_offset, config.seed_offset + config.training_seeds):
        chevron, direct, losses = _train_seed(config, training_seed)
        training_records.append(
            {
                "seed": training_seed,
                "chevron_initial_loss": losses["chevron"][0],
                "chevron_final_loss": losses["chevron"][-1],
                "direct_initial_loss": losses["direct"][0],
                "direct_final_loss": losses["direct"][-1],
                "chevron_threshold": float(chevron.threshold.detach()),
                "chevron_slope": float(chevron.slope.detach()),
            }
        )
        frozen_models = {
            "content_attention_buffer": None,
            "direct_mlp_buffer": copy.deepcopy(direct).eval(),
            "chevron_buffer": copy.deepcopy(chevron).eval(),
            "chevron_immediate": copy.deepcopy(chevron).eval(),
            "chevron_coupled_write": copy.deepcopy(chevron).eval(),
        }
        for evaluation_index in range(config.evaluation_lifetimes):
            lifetime_seed = 20_000_000 + 10_000 * training_seed + evaluation_index
            lifetime = make_lifetime(config, lifetime_seed)
            for condition in CONDITIONS:
                metrics, _ = run_lifetime(
                    condition,
                    config,
                    lifetime,
                    model=frozen_models[condition],
                    training=False,
                    training_seed=training_seed,
                    lifetime_seed=lifetime_seed,
                )
                all_results.append(metrics)
    return {
        "experiment": "004_reward_memory",
        "config": asdict(config),
        "training": training_records,
        "aggregate": _aggregate(all_results),
        "paired": _paired(all_results),
        "individual": [asdict(result) for result in all_results],
    }


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = result["aggregate"]
    config = result["config"]
    lines = [
        f"# Experiment 004: {label} reward-derived memory",
        "",
        f"- Training seeds: {config['seed_offset']}–{config['seed_offset'] + config['training_seeds'] - 1}",
        f"- Training lifetimes per seed: {config['training_lifetimes']}",
        f"- Fresh evaluation lifetimes per seed: {config['evaluation_lifetimes']}",
        f"- Reward delay: {config['outcome_delay']}",
        f"- Provisional buffer capacity: {config['buffer_capacity']}",
        "- Learned comparator parameters: 314 each",
        "- Learning signal: delayed scalar reward only",
        "",
        "| Method | Return/decision | Final old | Final new | Old probe | New probe | q calibration | N drift | Premature writes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = aggregate[condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['final_old_accuracy'])} | {_pm(metrics['final_new_accuracy'])} | "
            f"{_pm(metrics['old_probe_accuracy'])} | {_pm(metrics['new_probe_accuracy'])} | "
            f"{_pm(metrics['residual_calibration'])} | {_pm(metrics['established_drift'])} | "
            f"{_pm(metrics['premature_write_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
            "This is a delayed contextual-bandit RL experiment. It does not yet establish",
            "spatial game performance, PPO trainability, or a persistent core self.",
            "",
        ]
    )
    stem = f"experiment_004_{label}"
    (output_dir / f"{stem}_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / f"{stem}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("development", "confirmation"), default="development")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument("--training-lifetimes", type=int, default=None)
    parser.add_argument("--evaluation-lifetimes", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    defaults = ExperimentConfig()
    if args.label == "confirmation":
        config = ExperimentConfig(
            training_seeds=10 if args.seeds is None else args.seeds,
            seed_offset=300 if args.seed_offset is None else args.seed_offset,
            training_lifetimes=defaults.training_lifetimes if args.training_lifetimes is None else args.training_lifetimes,
            evaluation_lifetimes=20 if args.evaluation_lifetimes is None else args.evaluation_lifetimes,
        )
        if config.seed_offset < 300:
            raise ValueError("confirmation seed offset must be at least 300")
    else:
        config = ExperimentConfig(
            training_seeds=defaults.training_seeds if args.seeds is None else args.seeds,
            seed_offset=defaults.seed_offset if args.seed_offset is None else args.seed_offset,
            training_lifetimes=defaults.training_lifetimes if args.training_lifetimes is None else args.training_lifetimes,
            evaluation_lifetimes=defaults.evaluation_lifetimes if args.evaluation_lifetimes is None else args.evaluation_lifetimes,
        )
    if args.buffer_capacity is not None:
        config = replace(config, buffer_capacity=args.buffer_capacity)
    result = run_experiment(config)
    write_report(result, args.output_dir, args.label)
    print(json.dumps(_json_safe({"training": result["training"], "aggregate": result["aggregate"], "paired": result["paired"]}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
