"""Experiment 010: delayed outcome evidence for protected policy revision.

The observation contains a broad address and noisy identity evidence only.
Policy mismatch is discovered retrospectively from delayed, stochastic reward;
the hidden correct action and reward-noise flag are audit data, not agent input.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from experiments.experiment_004_reward_memory import (
    _json_safe,
    _mean,
    _mean_sd,
    geometric_cosine_assent,
)


CONDITIONS = (
    "direct_update",
    "retrospective_protected",
    "retrospective_fast_veto",
    "retrospective_immediate_write",
)

DISPLAY_NAMES = {
    "direct_update": "Direct value adaptation",
    "retrospective_protected": "Protected retrospective Chevron",
    "retrospective_fast_veto": "Fast-veto Chevron",
    "retrospective_immediate_write": "Immediate-write Chevron",
}


@dataclass(frozen=True)
class ExperimentConfig:
    groups: int = 4
    action_dim: int = 4
    identity_dim: int = 12
    stream_steps: int = 800
    shift_step: int = 200
    retention_step: int = 600
    outcome_delay: int = 3
    novel_probability: float = 0.30
    reversal_probability: float = 0.35
    reward_flip_probability: float = 0.10
    identity_noise: float = 0.15
    hard_identity_noise: float = 0.32
    hard_noise_probability: float = 0.10
    novel_anchor_cosine: float = 0.55
    permanent_capacity: int = 12
    buffer_capacity: int = 8
    identity_promotion_support: int = 2
    policy_veto_support: int = 2
    policy_promotion_support: int = 2
    similarity_threshold: float = 0.62
    gate_slope: float = 40.0
    identity_null_threshold: float = 0.80
    admitted_threshold: float = 0.25
    surprise_prediction_threshold: float = 0.60
    direct_update_rate: float = 0.35
    incumbent_value: float = 0.95
    unknown_action_value: float = 0.50


@dataclass(frozen=True)
class RetrospectiveObservation:
    event_id: int
    family: int
    identity: Tensor


@dataclass(frozen=True)
class RetrospectiveEvent:
    observation: RetrospectiveObservation
    category: int
    correct_action: int
    kind: str
    reward_flipped: bool


@dataclass(frozen=True)
class RetrospectiveLifetime:
    events: tuple[RetrospectiveEvent, ...]
    identity_prototypes: Tensor
    initial_actions: Tensor
    current_actions: Tensor
    stable_categories: tuple[int, ...]
    reversed_categories: tuple[int, ...]
    novel_categories: tuple[int, ...]


@dataclass
class RetrospectiveSlot:
    memory_id: int
    family: int
    identity: Tensor
    action_values: Tensor
    last_used: int
    established: bool
    origin_category: int | None


@dataclass
class Candidate:
    candidate_id: int
    kind: str
    family: int
    target_memory_id: int | None
    identity: Tensor
    observations: int
    pending_events: set[int]
    positive_actions: Tensor
    last_seen: int
    incumbent_action: int | None = None
    supported_failures: int = 0
    search_attempts: int = 0


@dataclass(frozen=True)
class RetrospectiveTrace:
    identity_mass: Tensor
    q_identity: Tensor
    q_policy: Tensor
    selected_slot: int
    target_memory_id: int | None
    candidate_kind: str | None
    policy_search_active: bool


@dataclass
class Pending:
    observation: RetrospectiveObservation
    action: int
    trace: RetrospectiveTrace
    predicted_success: float
    identity_candidate_id: int | None


@dataclass(frozen=True)
class Resolution:
    kind: str | None
    target_origin_category: int | None = None
    action: int | None = None


@dataclass
class AgentCounters:
    new_promotions: int = 0
    revision_promotions: int = 0
    false_stable_revisions: int = 0
    under_supported_writes: int = 0
    established_overwrites: int = 0
    duplicate_allocations: int = 0
    identity_reconciliations: int = 0
    buffer_evictions: int = 0
    policy_suspicions: int = 0
    policy_dismissals: int = 0


@dataclass(frozen=True)
class ExperimentMetrics:
    condition: str
    seed: int
    return_per_decision: float
    clean_accuracy: float
    retention_accuracy: float
    late_stable_accuracy: float
    late_reversed_accuracy: float
    late_novel_accuracy: float
    stable_probe_accuracy: float
    reversed_probe_accuracy: float
    novel_probe_accuracy: float
    identity_residual_calibration: float
    policy_residual_calibration: float
    stable_policy_alarm_rate: float
    reversed_policy_alarm_rate: float
    new_promotions: float
    revision_promotions: float
    unique_revision_categories: float
    false_stable_revisions: float
    mean_reversal_detection_occurrences: float
    under_supported_writes: float
    established_overwrites: float
    duplicate_allocations: float
    identity_reconciliations: float
    buffer_evictions: float
    policy_suspicions: float
    policy_dismissals: float


def _unit(value: Tensor) -> Tensor:
    return F.normalize(value, dim=-1)


def _action_values(config: ExperimentConfig, incumbent: int) -> Tensor:
    values = torch.full((config.action_dim,), config.unknown_action_value)
    values[incumbent] = config.incumbent_value
    return values


def make_lifetime(config: ExperimentConfig, seed: int) -> RetrospectiveLifetime:
    if not (0 < config.shift_step < config.retention_step < config.stream_steps):
        raise ValueError("phase boundaries must satisfy 0 < shift < retention < stream")
    if config.novel_probability + config.reversal_probability > 1.0:
        raise ValueError("phase-two context probabilities must sum to at most one")
    generator = torch.Generator().manual_seed(seed)
    identities: list[Tensor] = []
    initial_actions = torch.empty(config.groups * 3, dtype=torch.long)
    current_actions = torch.empty(config.groups * 3, dtype=torch.long)
    stable: list[int] = []
    reversed_categories: list[int] = []
    novel: list[int] = []

    for family in range(config.groups):
        first = _unit(torch.randn(config.identity_dim, generator=generator))
        second = _unit(torch.randn(config.identity_dim, generator=generator))
        while float(first @ second) > 0.35:
            second = _unit(torch.randn(config.identity_dim, generator=generator))
        direction = torch.randn(config.identity_dim, generator=generator)
        orthogonal = _unit(direction - (direction @ first) * first)
        third = _unit(
            config.novel_anchor_cosine * first
            + math.sqrt(1.0 - config.novel_anchor_cosine**2) * orthogonal
        )
        identities.extend((first, second, third))
        order = torch.randperm(config.action_dim, generator=generator)
        initial_actions[3 * family] = order[0]
        initial_actions[3 * family + 1] = order[1]
        initial_actions[3 * family + 2] = order[2]
        current_actions[3 * family] = order[0]
        current_actions[3 * family + 1] = order[3]
        current_actions[3 * family + 2] = order[2]
        stable.append(3 * family)
        reversed_categories.append(3 * family + 1)
        novel.append(3 * family + 2)

    identity_tensor = torch.stack(identities)
    events: list[RetrospectiveEvent] = []
    for step in range(config.stream_steps):
        family = int(torch.randint(config.groups, (), generator=generator))
        if step < config.shift_step:
            offset = int(torch.randint(2, (), generator=generator))
            category = 3 * family + offset
            kind = "initial"
            action = int(initial_actions[category])
        elif step < config.retention_step:
            draw = float(torch.rand((), generator=generator))
            if draw < config.novel_probability:
                category = 3 * family + 2
                kind = "novel"
            elif draw < config.novel_probability + config.reversal_probability:
                category = 3 * family + 1
                kind = "reversed"
            else:
                category = 3 * family
                kind = "stable"
            action = int(current_actions[category])
        else:
            category = 3 * family
            kind = "retention"
            action = int(current_actions[category])

        noise = (
            config.hard_identity_noise
            if float(torch.rand((), generator=generator)) < config.hard_noise_probability
            else config.identity_noise
        )
        identity = _unit(
            identity_tensor[category]
            + noise * torch.randn(config.identity_dim, generator=generator)
        )
        events.append(
            RetrospectiveEvent(
                observation=RetrospectiveObservation(step, family, identity),
                category=category,
                correct_action=action,
                kind=kind,
                reward_flipped=(
                    float(torch.rand((), generator=generator))
                    < config.reward_flip_probability
                ),
            )
        )

    return RetrospectiveLifetime(
        events=tuple(events),
        identity_prototypes=identity_tensor,
        initial_actions=initial_actions,
        current_actions=current_actions,
        stable_categories=tuple(stable),
        reversed_categories=tuple(reversed_categories),
        novel_categories=tuple(novel),
    )


class CandidateBank:
    """One shared capacity pool with typed consolidation destinations."""

    def __init__(self, config: ExperimentConfig) -> None:
        if config.buffer_capacity <= 0:
            raise ValueError("candidate bank capacity must be positive")
        self.config = config
        self.capacity = config.buffer_capacity
        self.entries: list[Candidate] = []
        self.next_id = 0

    def _insert(self, candidate: Candidate) -> Candidate | None:
        evicted = None
        if len(self.entries) >= self.capacity:
            index = min(range(len(self.entries)), key=lambda i: self.entries[i].last_seen)
            evicted = self.entries.pop(index)
        self.entries.append(candidate)
        return evicted

    def _next_candidate_id(self) -> int:
        candidate_id = self.next_id
        self.next_id += 1
        return candidate_id

    def get(self, candidate_id: int) -> Candidate | None:
        return next(
            (entry for entry in self.entries if entry.candidate_id == candidate_id),
            None,
        )

    def remove(self, candidate: Candidate) -> None:
        if candidate in self.entries:
            self.entries.remove(candidate)

    def find_policy(self, target_memory_id: int) -> Candidate | None:
        return next(
            (
                entry
                for entry in self.entries
                if entry.kind == "policy_revision"
                and entry.target_memory_id == target_memory_id
            ),
            None,
        )

    def add_identity(
        self, observation: RetrospectiveObservation
    ) -> tuple[int, Candidate | None]:
        matches = [
            float(entry.identity @ observation.identity)
            if entry.kind == "new_identity" and entry.family == observation.family
            else -1.0
            for entry in self.entries
        ]
        if matches and max(matches) >= self.config.similarity_threshold:
            entry = self.entries[int(torch.tensor(matches).argmax())]
            count = entry.observations + 1
            entry.identity = _unit(
                (entry.identity * entry.observations + observation.identity) / count
            )
            entry.observations = count
            entry.pending_events.add(observation.event_id)
            entry.last_seen = observation.event_id
            return entry.candidate_id, None

        candidate = Candidate(
            candidate_id=self._next_candidate_id(),
            kind="new_identity",
            family=observation.family,
            target_memory_id=None,
            identity=observation.identity.detach().clone(),
            observations=1,
            pending_events={observation.event_id},
            positive_actions=torch.zeros(self.config.action_dim, dtype=torch.long),
            last_seen=observation.event_id,
        )
        return candidate.candidate_id, self._insert(candidate)

    def resolve_identity(
        self,
        candidate_id: int,
        event_id: int,
        action: int,
        observed_success: float,
    ) -> Candidate | None:
        entry = self.get(candidate_id)
        if (
            entry is None
            or entry.kind != "new_identity"
            or event_id not in entry.pending_events
        ):
            return None
        entry.pending_events.remove(event_id)
        if observed_success > 0.0:
            entry.positive_actions[action] += 1
        if int(entry.positive_actions.max()) >= self.config.identity_promotion_support:
            self.remove(entry)
            return entry
        return None

    def record_policy_failure(
        self,
        observation: RetrospectiveObservation,
        target_memory_id: int,
        incumbent_action: int,
    ) -> tuple[Candidate, Candidate | None, bool]:
        entry = self.find_policy(target_memory_id)
        if entry is not None:
            count = entry.observations + 1
            entry.identity = _unit(
                (entry.identity * entry.observations + observation.identity) / count
            )
            entry.observations = count
            entry.supported_failures += 1
            entry.last_seen = observation.event_id
            return entry, None, False

        candidate = Candidate(
            candidate_id=self._next_candidate_id(),
            kind="policy_revision",
            family=observation.family,
            target_memory_id=target_memory_id,
            identity=observation.identity.detach().clone(),
            observations=1,
            pending_events=set(),
            positive_actions=torch.zeros(self.config.action_dim, dtype=torch.long),
            last_seen=observation.event_id,
            incumbent_action=incumbent_action,
            supported_failures=1,
        )
        evicted = self._insert(candidate)
        return candidate, evicted, True


class RetrospectiveAgent:
    def __init__(
        self,
        condition: str,
        config: ExperimentConfig,
        lifetime: RetrospectiveLifetime,
        seed: int,
        *,
        revalidate_identity_promotion: bool = False,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        self.condition = condition
        self.config = config
        self.lifetime = lifetime
        self.generator = torch.Generator().manual_seed(seed)
        self.revalidate_identity_promotion = revalidate_identity_promotion
        self.next_memory_id = 0
        self.slots: list[RetrospectiveSlot] = []
        for category in tuple(lifetime.stable_categories) + tuple(
            lifetime.reversed_categories
        ):
            incumbent = int(lifetime.initial_actions[category])
            self.slots.append(
                RetrospectiveSlot(
                    memory_id=self.next_memory_id,
                    family=category // 3,
                    identity=lifetime.identity_prototypes[category].clone(),
                    action_values=_action_values(config, incumbent),
                    last_used=-1,
                    established=True,
                    origin_category=category,
                )
            )
            self.next_memory_id += 1
        self.bank = CandidateBank(config)
        self.pending: dict[int, Pending] = {}
        self.counters = AgentCounters()
        self.revised_origins: set[int] = set()

    @property
    def veto_support(self) -> int:
        if self.condition == "retrospective_protected":
            return self.config.policy_veto_support
        return 1

    @property
    def revision_support(self) -> int:
        if self.condition == "retrospective_immediate_write":
            return 1
        return self.config.policy_promotion_support

    def _alpha(self, family: int) -> Tensor:
        mask = torch.tensor([slot.family == family for slot in self.slots])
        return mask.to(torch.float32) / mask.sum().clamp_min(1)

    def _slot_index(self, memory_id: int) -> int | None:
        return next(
            (i for i, slot in enumerate(self.slots) if slot.memory_id == memory_id),
            None,
        )

    def distribution(
        self, observation: RetrospectiveObservation
    ) -> RetrospectiveTrace:
        alpha = self._alpha(observation.family)
        identities = torch.stack([slot.identity for slot in self.slots])
        identity_assent = geometric_cosine_assent(
            observation.identity,
            identities,
            similarity_threshold=self.config.similarity_threshold,
            slope=self.config.gate_slope,
        )
        identity_mass = alpha * identity_assent
        q_identity = 1.0 - identity_mass.sum()
        selected_slot = int(identity_mass.argmax())
        target_memory_id = self.slots[selected_slot].memory_id
        candidate_kind = None
        if (
            float(q_identity) > self.config.identity_null_threshold
            and float(identity_mass.max()) < self.config.admitted_threshold
        ):
            candidate_kind = "new_identity"
            target_memory_id = None

        policy_candidate = (
            self.bank.find_policy(target_memory_id)
            if target_memory_id is not None and self.condition != "direct_update"
            else None
        )
        q_policy = (
            min(1.0, policy_candidate.supported_failures / self.veto_support)
            if policy_candidate is not None
            else 0.0
        )
        search_active = (
            policy_candidate is not None
            and policy_candidate.supported_failures >= self.veto_support
        )
        return RetrospectiveTrace(
            identity_mass=identity_mass,
            q_identity=q_identity,
            q_policy=torch.tensor(q_policy, dtype=identity_mass.dtype),
            selected_slot=selected_slot,
            target_memory_id=target_memory_id,
            candidate_kind=candidate_kind,
            policy_search_active=search_active,
        )

    def _search_action(self, candidate: Candidate) -> int:
        if candidate.incumbent_action is None:
            raise ValueError("policy candidate requires an incumbent action")
        alternatives = [
            action
            for action in range(self.config.action_dim)
            if action != candidate.incumbent_action
        ]
        order = alternatives + [candidate.incumbent_action]
        action = order[candidate.search_attempts % len(order)]
        candidate.search_attempts += 1
        return action

    def act(
        self, observation: RetrospectiveObservation
    ) -> tuple[int, RetrospectiveTrace]:
        trace = self.distribution(observation)
        identity_candidate_id = None
        predicted_success = self.config.unknown_action_value
        if trace.candidate_kind == "new_identity":
            action = int(
                torch.randint(self.config.action_dim, (), generator=self.generator)
            )
            identity_candidate_id, evicted = self.bank.add_identity(observation)
            self.counters.buffer_evictions += int(evicted is not None)
        else:
            if trace.target_memory_id is None:
                raise RuntimeError("recognised identity must target a memory")
            index = self._slot_index(trace.target_memory_id)
            if index is None:
                raise RuntimeError("target memory disappeared")
            policy_candidate = self.bank.find_policy(trace.target_memory_id)
            if trace.policy_search_active and policy_candidate is not None:
                action = self._search_action(policy_candidate)
            else:
                values = torch.stack([slot.action_values for slot in self.slots])
                scores = torch.einsum("s,sa->a", trace.identity_mass, values)
                action = int(scores.argmax())
            predicted_success = float(self.slots[index].action_values[action])

        self.pending[observation.event_id] = Pending(
            observation=observation,
            action=action,
            trace=trace,
            predicted_success=predicted_success,
            identity_candidate_id=identity_candidate_id,
        )
        return action, trace

    def _identity_match(self, observation: RetrospectiveObservation) -> bool:
        similarities = torch.tensor(
            [
                float(slot.identity @ observation.identity)
                if slot.family == observation.family
                else -1.0
                for slot in self.slots
            ]
        )
        return float(similarities.max()) >= self.config.similarity_threshold

    def _allocate(self, candidate: Candidate, action: int) -> bool:
        duplicate = self._identity_match(
            RetrospectiveObservation(
                candidate.last_seen,
                candidate.family,
                candidate.identity,
            )
        )
        if duplicate and self.revalidate_identity_promotion:
            self.counters.identity_reconciliations += 1
            return False
        slot = RetrospectiveSlot(
            memory_id=self.next_memory_id,
            family=candidate.family,
            identity=candidate.identity.detach().clone(),
            action_values=_action_values(self.config, action),
            last_used=candidate.last_seen,
            established=False,
            origin_category=None,
        )
        self.next_memory_id += 1
        if len(self.slots) < self.config.permanent_capacity:
            self.slots.append(slot)
        else:
            provisional = [i for i, item in enumerate(self.slots) if not item.established]
            pool = provisional or list(range(len(self.slots)))
            index = min(pool, key=lambda i: self.slots[i].last_used)
            self.counters.established_overwrites += int(self.slots[index].established)
            self.slots[index] = slot
        self.counters.new_promotions += int(not duplicate)
        self.counters.duplicate_allocations += int(duplicate)
        return True

    def _revise(
        self,
        target_memory_id: int,
        action: int,
        positive_support: int,
    ) -> Resolution:
        index = self._slot_index(target_memory_id)
        if index is None:
            return Resolution(None)
        slot = self.slots[index]
        slot.action_values = _action_values(self.config, action)
        slot.last_used += 1
        self.counters.revision_promotions += 1
        self.counters.under_supported_writes += int(
            positive_support < self.config.policy_promotion_support
        )
        if slot.origin_category in self.lifetime.stable_categories:
            self.counters.false_stable_revisions += 1
        if slot.origin_category is not None:
            self.revised_origins.add(slot.origin_category)
        return Resolution("policy_revision", slot.origin_category, action)

    def _direct_update(
        self, target_memory_id: int, action: int, observed_success: float
    ) -> None:
        index = self._slot_index(target_memory_id)
        if index is None:
            return
        slot = self.slots[index]
        rate = self.config.direct_update_rate
        slot.action_values[action] = (
            (1.0 - rate) * slot.action_values[action]
            + rate * observed_success
        )
        slot.last_used += 1

    def resolve_reward(self, event_id: int, reward: float) -> Resolution:
        pending = self.pending.pop(event_id)
        observed_success = float(reward > 0.0)
        if pending.identity_candidate_id is not None:
            candidate = self.bank.resolve_identity(
                pending.identity_candidate_id,
                event_id,
                pending.action,
                observed_success,
            )
            if candidate is not None:
                action = int(candidate.positive_actions.argmax())
                allocated = self._allocate(candidate, action)
                return Resolution(
                    "new_identity" if allocated else "identity_reconciled",
                    None,
                    action,
                )
            return Resolution(None)

        target_memory_id = pending.trace.target_memory_id
        if target_memory_id is None:
            return Resolution(None)
        if self.condition == "direct_update":
            self._direct_update(
                target_memory_id,
                pending.action,
                observed_success,
            )
            return Resolution(None)

        policy_candidate = self.bank.find_policy(target_memory_id)
        unexpected_failure = (
            observed_success == 0.0
            and pending.predicted_success
            >= self.config.surprise_prediction_threshold
        )
        if policy_candidate is None:
            if unexpected_failure:
                _, evicted, created = self.bank.record_policy_failure(
                    pending.observation,
                    target_memory_id,
                    pending.action,
                )
                self.counters.buffer_evictions += int(evicted is not None)
                self.counters.policy_suspicions += int(created)
            return Resolution(None)

        policy_candidate.last_seen = pending.observation.event_id
        incumbent = policy_candidate.incumbent_action
        if pending.action == incumbent:
            if observed_success > 0.0:
                self.bank.remove(policy_candidate)
                self.counters.policy_dismissals += 1
            elif unexpected_failure:
                policy_candidate.supported_failures += 1
            return Resolution(None)

        if (
            policy_candidate.supported_failures >= self.veto_support
            and observed_success > 0.0
        ):
            policy_candidate.positive_actions[pending.action] += 1
            support = int(policy_candidate.positive_actions[pending.action])
            if support >= self.revision_support:
                self.bank.remove(policy_candidate)
                return self._revise(target_memory_id, pending.action, support)
        return Resolution(None)

    def probe_action(self, observation: RetrospectiveObservation) -> int:
        trace = self.distribution(observation)
        values = torch.stack([slot.action_values for slot in self.slots])
        return int(torch.einsum("s,sa->a", trace.identity_mass, values).argmax())


def _probe(
    agent: RetrospectiveAgent,
    lifetime: RetrospectiveLifetime,
    categories: tuple[int, ...],
) -> float:
    correct = 0
    for offset, category in enumerate(categories):
        observation = RetrospectiveObservation(
            agent.config.stream_steps + offset,
            category // 3,
            lifetime.identity_prototypes[category],
        )
        correct += int(
            agent.probe_action(observation)
            == int(lifetime.current_actions[category])
        )
    return correct / len(categories)


def run_lifetime(
    condition: str,
    config: ExperimentConfig,
    lifetime: RetrospectiveLifetime,
    seed: int,
    *,
    revalidate_identity_promotion: bool = False,
) -> ExperimentMetrics:
    agent = RetrospectiveAgent(
        condition,
        config,
        lifetime,
        105_000_000 + seed,
        revalidate_identity_promotion=revalidate_identity_promotion,
    )
    due: dict[int, list[tuple[int, float]]] = {}
    decisions: list[dict[str, Any]] = []
    resolved_novel: set[int] = set()
    resolved_reversed: set[int] = set()
    reversed_occurrences = {category: 0 for category in lifetime.reversed_categories}
    first_revision_occurrence: dict[int, int] = {}

    for step, event in enumerate(lifetime.events):
        for event_id, reward in due.get(step, []):
            resolution = agent.resolve_reward(event_id, reward)
            source = lifetime.events[event_id]
            if resolution.kind == "new_identity" and source.kind == "novel":
                resolved_novel.add(source.category)
            if (
                resolution.kind == "policy_revision"
                and resolution.target_origin_category in lifetime.reversed_categories
            ):
                category = int(resolution.target_origin_category)
                resolved_reversed.add(category)
                first_revision_occurrence.setdefault(
                    category,
                    reversed_occurrences[category],
                )

        if event.kind == "reversed":
            reversed_occurrences[event.category] += 1
        action, trace = agent.act(event.observation)
        correct = action == event.correct_action
        base_reward = 1.0 if correct else -1.0
        reward = -base_reward if event.reward_flipped else base_reward
        due.setdefault(step + config.outcome_delay, []).append(
            (event.observation.event_id, reward)
        )
        decisions.append(
            {
                "step": step,
                "kind": event.kind,
                "category": event.category,
                "correct": correct,
                "reward": reward,
                "q_identity": float(trace.q_identity),
                "q_policy": float(trace.q_policy),
                "identity_resolved": event.category in resolved_novel,
                "policy_resolved": event.category in resolved_reversed,
            }
        )

    for step in range(config.stream_steps, config.stream_steps + config.outcome_delay):
        for event_id, reward in due.get(step, []):
            resolution = agent.resolve_reward(event_id, reward)
            if (
                resolution.kind == "policy_revision"
                and resolution.target_origin_category in lifetime.reversed_categories
            ):
                category = int(resolution.target_origin_category)
                first_revision_occurrence.setdefault(
                    category,
                    reversed_occurrences[category],
                )

    late_phase_two = [
        row
        for row in decisions
        if config.retention_step - 200 <= row["step"] < config.retention_step
    ]
    retention = [row for row in decisions if row["step"] >= config.retention_step]
    by_kind = {
        kind: [float(row["correct"]) for row in late_phase_two if row["kind"] == kind]
        for kind in ("stable", "reversed", "novel")
    }
    unresolved_identity = [
        row["q_identity"]
        for row in decisions
        if row["kind"] == "novel" and not row["identity_resolved"]
    ]
    resolved_identity = [
        row["q_identity"]
        for row in decisions
        if row["kind"] == "novel" and row["identity_resolved"]
    ]
    unresolved_policy = [
        row["q_policy"]
        for row in decisions
        if row["kind"] == "reversed" and not row["policy_resolved"]
    ]
    stable_policy = [
        row["q_policy"]
        for row in decisions
        if row["kind"] in {"stable", "retention"}
    ]
    reversed_policy = [
        row["q_policy"] for row in decisions if row["kind"] == "reversed"
    ]
    detection_occurrences = [
        float(
            first_revision_occurrence.get(
                category,
                reversed_occurrences[category] + 1,
            )
        )
        for category in lifetime.reversed_categories
    ]
    return ExperimentMetrics(
        condition=condition,
        seed=seed,
        return_per_decision=_mean([row["reward"] for row in decisions]),
        clean_accuracy=_mean([float(row["correct"]) for row in decisions]),
        retention_accuracy=_mean([float(row["correct"]) for row in retention]),
        late_stable_accuracy=_mean(by_kind["stable"]),
        late_reversed_accuracy=_mean(by_kind["reversed"]),
        late_novel_accuracy=_mean(by_kind["novel"]),
        stable_probe_accuracy=_probe(agent, lifetime, lifetime.stable_categories),
        reversed_probe_accuracy=_probe(agent, lifetime, lifetime.reversed_categories),
        novel_probe_accuracy=_probe(agent, lifetime, lifetime.novel_categories),
        identity_residual_calibration=(
            _mean(unresolved_identity) - _mean(resolved_identity)
        ),
        policy_residual_calibration=(
            _mean(unresolved_policy) - _mean(stable_policy)
        ),
        stable_policy_alarm_rate=_mean(
            [float(value > 0.0) for value in stable_policy]
        ),
        reversed_policy_alarm_rate=_mean(
            [float(value > 0.0) for value in reversed_policy]
        ),
        new_promotions=float(agent.counters.new_promotions),
        revision_promotions=float(agent.counters.revision_promotions),
        unique_revision_categories=float(
            len(agent.revised_origins.intersection(lifetime.reversed_categories))
        ),
        false_stable_revisions=float(agent.counters.false_stable_revisions),
        mean_reversal_detection_occurrences=_mean(detection_occurrences),
        under_supported_writes=float(agent.counters.under_supported_writes),
        established_overwrites=float(agent.counters.established_overwrites),
        duplicate_allocations=float(agent.counters.duplicate_allocations),
        identity_reconciliations=float(agent.counters.identity_reconciliations),
        buffer_evictions=float(agent.counters.buffer_evictions),
        policy_suspicions=float(agent.counters.policy_suspicions),
        policy_dismissals=float(agent.counters.policy_dismissals),
    )


def _aggregate(
    rows: list[ExperimentMetrics],
) -> dict[str, dict[str, dict[str, float]]]:
    fields = [
        field
        for field in ExperimentMetrics.__dataclass_fields__
        if field not in {"condition", "seed"}
    ]
    return {
        condition: {
            field: _mean_sd(
                [getattr(row, field) for row in rows if row.condition == condition]
            )
            for field in fields
        }
        for condition in CONDITIONS
    }


def _paired(
    rows: list[ExperimentMetrics], first: str, second: str
) -> dict[str, float]:
    by_key = {(row.condition, row.seed): row for row in rows}
    seeds = sorted({row.seed for row in rows})
    output: dict[str, float] = {}
    for field in (
        "return_per_decision",
        "clean_accuracy",
        "retention_accuracy",
        "reversed_probe_accuracy",
        "novel_probe_accuracy",
        "false_stable_revisions",
        "mean_reversal_detection_occurrences",
    ):
        values = [
            getattr(by_key[(first, seed)], field)
            - getattr(by_key[(second, seed)], field)
            for seed in seeds
        ]
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values)
        half = 1.96 * sd / math.sqrt(len(values))
        output[f"{field}_mean"] = mean
        output[f"{field}_population_sd"] = sd
        output[f"{field}_approx_95ci_low"] = mean - half
        output[f"{field}_approx_95ci_high"] = mean + half
        output[f"{field}_wins"] = sum(value > 0.0 for value in values)
    return output


def _development_gate(result: dict[str, Any]) -> dict[str, bool]:
    protected = result["aggregate"]["retrospective_protected"]
    versus_direct = result["paired"][
        "retrospective_protected_minus_direct_update"
    ]
    versus_immediate = result["paired"][
        "retrospective_protected_minus_retrospective_immediate_write"
    ]
    return {
        "retention_accuracy_at_least_0.90": protected["retention_accuracy"]["mean"]
        >= 0.90,
        "reversed_probe_at_least_0.75": protected["reversed_probe_accuracy"]["mean"]
        >= 0.75,
        "novel_probe_at_least_0.75": protected["novel_probe_accuracy"]["mean"]
        >= 0.75,
        "new_promotions_at_least_3": protected["new_promotions"]["mean"] >= 3.0,
        "unique_revisions_at_least_3": protected["unique_revision_categories"]["mean"]
        >= 3.0,
        "policy_calibration_at_least_0.10": protected[
            "policy_residual_calibration"
        ]["mean"]
        >= 0.10,
        "false_stable_revisions_at_most_0.25": protected[
            "false_stable_revisions"
        ]["mean"]
        <= 0.25,
        "no_under_supported_writes": protected["under_supported_writes"]["mean"]
        == 0.0,
        "no_established_overwrites": protected["established_overwrites"]["mean"]
        == 0.0,
        "no_duplicate_allocations": protected["duplicate_allocations"]["mean"]
        == 0.0,
        "return_noninferior_to_direct": versus_direct[
            "return_per_decision_approx_95ci_low"
        ]
        > -0.08,
        "clean_accuracy_noninferior_to_direct": versus_direct[
            "clean_accuracy_approx_95ci_low"
        ]
        > -0.08,
        "retention_noninferior_to_direct": versus_direct[
            "retention_accuracy_approx_95ci_low"
        ]
        > -0.03,
        "fewer_false_revisions_than_immediate": versus_immediate[
            "false_stable_revisions_approx_95ci_high"
        ]
        < 0.0,
    }


def run_development(
    config: ExperimentConfig,
    *,
    seeds: int = 20,
    seed_offset: int = 100_000_000,
) -> dict[str, Any]:
    if seed_offset < 100_000_000:
        raise ValueError("development seeds must start at 100,000,000 or later")
    rows: list[ExperimentMetrics] = []
    for seed in range(seed_offset, seed_offset + seeds):
        lifetime = make_lifetime(config, seed)
        for condition in CONDITIONS:
            rows.append(run_lifetime(condition, config, lifetime, seed))
    comparisons = (
        ("retrospective_protected", "direct_update"),
        ("retrospective_protected", "retrospective_fast_veto"),
        ("retrospective_protected", "retrospective_immediate_write"),
    )
    result: dict[str, Any] = {
        "experiment": "010_retrospective_policy",
        "status": "development",
        "config": asdict(config),
        "seeds": seeds,
        "seed_offset": seed_offset,
        "aggregate": _aggregate(rows),
        "paired": {
            f"{first}_minus_{second}": _paired(rows, first, second)
            for first, second in comparisons
        },
        "individual": [asdict(row) for row in rows],
    }
    result["development_gate"] = _development_gate(result)
    result["confirmation_triggered"] = all(result["development_gate"].values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 010: retrospective-policy development",
        "",
        f"- Seeds: {result['seed_offset']}–{result['seed_offset'] + result['seeds'] - 1}",
        "- Policy signature in observation: **none**",
        "",
        "| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | False revisions | Detection occurrences |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['clean_accuracy'])} | {_pm(metrics['retention_accuracy'])} | "
            f"{_pm(metrics['reversed_probe_accuracy'])} | "
            f"{_pm(metrics['novel_probe_accuracy'])} | "
            f"{_pm(metrics['false_stable_revisions'])} | "
            f"{_pm(metrics['mean_reversal_detection_occurrences'])} |"
        )
    lines.extend(["", "## Frozen development gate", ""])
    for name, passed in result["development_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Confirmation triggered: **{result['confirmation_triggered']}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    (output_dir / "experiment_010_development_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "experiment_010_development_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=100_000_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
    )
    args = parser.parse_args()
    result = run_development(
        ExperimentConfig(),
        seeds=args.seeds,
        seed_offset=args.seed_offset,
    )
    write_report(result, args.output_dir)
    print(
        json.dumps(
            _json_safe(
                {
                    "aggregate": result["aggregate"],
                    "paired": result["paired"],
                    "development_gate": result["development_gate"],
                    "confirmation_triggered": result["confirmation_triggered"],
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
