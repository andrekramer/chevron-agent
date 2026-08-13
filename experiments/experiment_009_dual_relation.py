"""Experiment 009: separate memory identity from policy compatibility."""

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
    "dual_buffer",
    "collapsed_buffer",
    "identity_only_buffer",
    "dual_immediate",
)

DISPLAY_NAMES = {
    "dual_buffer": "Dual relation + buffer",
    "collapsed_buffer": "Collapsed relation + buffer",
    "identity_only_buffer": "Identity only + buffer",
    "dual_immediate": "Dual relation + immediate revision",
}


@dataclass(frozen=True)
class ExperimentConfig:
    groups: int = 4
    action_dim: int = 4
    identity_dim: int = 12
    stream_steps: int = 600
    shift_step: int = 200
    outcome_delay: int = 3
    novel_probability: float = 0.30
    reversal_probability: float = 0.35
    identity_noise: float = 0.15
    hard_identity_noise: float = 0.32
    hard_noise_probability: float = 0.10
    policy_noise: float = 0.05
    novel_anchor_cosine: float = 0.55
    permanent_capacity: int = 12
    buffer_capacity: int = 4
    promotion_support: int = 2
    similarity_threshold: float = 0.62
    gate_slope: float = 40.0
    identity_null_threshold: float = 0.80
    policy_null_threshold: float = 0.80
    admitted_threshold: float = 0.25
    value_update_rate: float = 0.35


@dataclass(frozen=True)
class DualObservation:
    event_id: int
    family: int
    identity: Tensor
    policy: Tensor


@dataclass(frozen=True)
class DualEvent:
    observation: DualObservation
    category: int
    correct_action: int
    kind: str


@dataclass(frozen=True)
class DualLifetime:
    events: tuple[DualEvent, ...]
    identity_prototypes: Tensor
    initial_actions: Tensor
    current_actions: Tensor
    stable_categories: tuple[int, ...]
    reversed_categories: tuple[int, ...]
    novel_categories: tuple[int, ...]


@dataclass
class DualSlot:
    memory_id: int
    family: int
    identity: Tensor
    policy: Tensor
    action_values: Tensor
    last_used: int
    established: bool
    origin_identity: Tensor | None


@dataclass
class Candidate:
    candidate_id: int
    kind: str
    family: int
    target_memory_id: int | None
    identity: Tensor
    policy: Tensor
    observations: int
    pending_events: set[int]
    positive_actions: Tensor
    last_seen: int


@dataclass(frozen=True)
class DualTrace:
    identity_mass: Tensor
    policy_mass: Tensor
    q_identity: Tensor
    q_policy: Tensor
    selected_slot: int
    candidate_kind: str | None
    target_memory_id: int | None


@dataclass
class Pending:
    observation: DualObservation
    action: int
    trace: DualTrace
    candidate_id: int | None
    candidate_route: str | None


@dataclass(frozen=True)
class Resolution:
    kind: str | None


@dataclass
class AgentCounters:
    new_promotions: int = 0
    revision_promotions: int = 0
    duplicate_allocations: int = 0
    premature_writes: int = 0
    established_overwrites: int = 0
    buffer_evictions: int = 0


@dataclass(frozen=True)
class ExperimentMetrics:
    condition: str
    seed: int
    return_per_decision: float
    overall_accuracy: float
    final_stable_accuracy: float
    final_reversed_accuracy: float
    final_novel_accuracy: float
    stable_probe_accuracy: float
    reversed_probe_accuracy: float
    novel_probe_accuracy: float
    identity_residual_calibration: float
    policy_residual_calibration: float
    new_promotions: float
    revision_promotions: float
    duplicate_allocations: float
    premature_write_rate: float
    established_overwrite_rate: float
    buffer_evictions: float


def _unit(value: Tensor) -> Tensor:
    return F.normalize(value, dim=-1)


def policy_signature(action: int, action_dim: int) -> Tensor:
    value = F.one_hot(torch.tensor(action), num_classes=action_dim).to(torch.float32)
    return _unit(value - value.mean())


def make_lifetime(config: ExperimentConfig, seed: int) -> DualLifetime:
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
    events: list[DualEvent] = []
    for step in range(config.stream_steps):
        after_shift = step >= config.shift_step
        draw = float(torch.rand((), generator=generator))
        family = int(torch.randint(config.groups, (), generator=generator))
        if after_shift and draw < config.novel_probability:
            category = 3 * family + 2
            kind = "novel"
        elif after_shift and draw < config.novel_probability + config.reversal_probability:
            category = 3 * family + 1
            kind = "reversed"
        else:
            if after_shift:
                category = 3 * family
                kind = "stable"
            else:
                offset = int(torch.randint(2, (), generator=generator))
                category = 3 * family + offset
                kind = "stable"
        action = int(current_actions[category] if after_shift else initial_actions[category])
        noise = (
            config.hard_identity_noise
            if float(torch.rand((), generator=generator)) < config.hard_noise_probability
            else config.identity_noise
        )
        identity = _unit(
            identity_tensor[category]
            + noise * torch.randn(config.identity_dim, generator=generator)
        )
        base_policy = policy_signature(action, config.action_dim)
        policy = _unit(
            base_policy
            + config.policy_noise * torch.randn(config.action_dim, generator=generator)
        )
        events.append(
            DualEvent(
                observation=DualObservation(step, family, identity, policy),
                category=category,
                correct_action=action,
                kind=kind,
            )
        )
    return DualLifetime(
        events=tuple(events),
        identity_prototypes=identity_tensor,
        initial_actions=initial_actions,
        current_actions=current_actions,
        stable_categories=tuple(stable),
        reversed_categories=tuple(reversed_categories),
        novel_categories=tuple(novel),
    )


class CandidateBuffer:
    def __init__(self, config: ExperimentConfig, capacity: int | None = None) -> None:
        self.config = config
        self.capacity = config.buffer_capacity if capacity is None else capacity
        if self.capacity <= 0:
            raise ValueError("candidate buffer capacity must be positive")
        self.entries: list[Candidate] = []
        self.next_id = 0

    def add(
        self,
        kind: str,
        observation: DualObservation,
        target_memory_id: int | None,
    ) -> tuple[int, bool]:
        matches: list[float] = []
        for entry in self.entries:
            compatible_route = (
                entry.kind == kind
                and entry.family == observation.family
                and entry.target_memory_id == target_memory_id
            )
            matches.append(
                float(entry.identity @ observation.identity)
                if compatible_route
                else -1.0
            )
        if matches and max(matches) >= self.config.similarity_threshold:
            entry = self.entries[int(torch.tensor(matches).argmax())]
            count = entry.observations + 1
            entry.identity = _unit(
                (entry.identity * entry.observations + observation.identity) / count
            )
            entry.policy = _unit(
                (entry.policy * entry.observations + observation.policy) / count
            )
            entry.observations = count
            entry.pending_events.add(observation.event_id)
            entry.last_seen = observation.event_id
            return entry.candidate_id, False

        evicted = False
        if len(self.entries) >= self.capacity:
            index = min(range(len(self.entries)), key=lambda i: self.entries[i].last_seen)
            self.entries.pop(index)
            evicted = True
        candidate_id = self.next_id
        self.next_id += 1
        self.entries.append(
            Candidate(
                candidate_id=candidate_id,
                kind=kind,
                family=observation.family,
                target_memory_id=target_memory_id,
                identity=observation.identity.detach().clone(),
                policy=observation.policy.detach().clone(),
                observations=1,
                pending_events={observation.event_id},
                positive_actions=torch.zeros(self.config.action_dim, dtype=torch.long),
                last_seen=observation.event_id,
            )
        )
        return candidate_id, evicted

    def resolve(self, candidate_id: int, event_id: int, action: int, reward: float) -> Candidate | None:
        entry = next((item for item in self.entries if item.candidate_id == candidate_id), None)
        if entry is None or event_id not in entry.pending_events:
            return None
        entry.pending_events.remove(event_id)
        if reward > 0.0:
            entry.positive_actions[action] += 1
        if int(entry.positive_actions.max()) >= self.config.promotion_support:
            self.entries.remove(entry)
            return entry
        return None


class DualRelationAgent:
    def __init__(
        self,
        condition: str,
        config: ExperimentConfig,
        lifetime: DualLifetime,
        seed: int,
        *,
        buffer_layout: str = "shared",
        shared_capacity: int | None = None,
        identity_capacity: int | None = None,
        policy_capacity: int | None = None,
    ) -> None:
        self.condition = condition
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        self.next_memory_id = 0
        self.slots: list[DualSlot] = []
        for category in tuple(lifetime.stable_categories) + tuple(lifetime.reversed_categories):
            action = int(lifetime.initial_actions[category])
            self.slots.append(
                DualSlot(
                    memory_id=self.next_memory_id,
                    family=category // 3,
                    identity=lifetime.identity_prototypes[category].clone(),
                    policy=policy_signature(action, config.action_dim),
                    action_values=F.one_hot(
                        torch.tensor(action), num_classes=config.action_dim
                    ).to(torch.float32),
                    last_used=-1,
                    established=True,
                    origin_identity=lifetime.identity_prototypes[category].clone(),
                )
            )
            self.next_memory_id += 1
        if buffer_layout == "shared":
            self.buffers = {
                "shared": CandidateBuffer(config, shared_capacity),
            }
        elif buffer_layout == "split":
            self.buffers = {
                "identity": CandidateBuffer(config, identity_capacity),
                "policy": CandidateBuffer(config, policy_capacity),
            }
        else:
            raise ValueError("buffer_layout must be 'shared' or 'split'")
        self.buffer_layout = buffer_layout
        self.pending: dict[int, Pending] = {}
        self.counters = AgentCounters()

    def _candidate_route(self, candidate_kind: str) -> str:
        if self.buffer_layout == "shared":
            return "shared"
        return "policy" if candidate_kind == "policy_revision" else "identity"

    def _alpha(self, family: int) -> Tensor:
        mask = torch.tensor([slot.family == family for slot in self.slots])
        return mask.to(torch.float32) / mask.sum().clamp_min(1)

    def distribution(self, observation: DualObservation) -> DualTrace:
        alpha = self._alpha(observation.family)
        identities = torch.stack([slot.identity for slot in self.slots])
        policies = torch.stack([slot.policy for slot in self.slots])
        identity_assent = geometric_cosine_assent(
            observation.identity,
            identities,
            similarity_threshold=self.config.similarity_threshold,
            slope=self.config.gate_slope,
        )
        identity_mass = alpha * identity_assent
        q_identity = 1.0 - identity_mass.sum()
        policy_assent = geometric_cosine_assent(
            observation.policy,
            policies,
            similarity_threshold=self.config.similarity_threshold,
            slope=self.config.gate_slope,
        )
        policy_mass = identity_mass * policy_assent
        admitted_identity = identity_mass.sum()
        q_policy = (
            (identity_mass * (1.0 - policy_assent)).sum()
            / admitted_identity.clamp_min(1e-8)
        )
        selected = int(identity_mass.argmax())
        candidate_kind: str | None = None
        target_memory_id: int | None = None

        if self.condition == "collapsed_buffer":
            q_collapsed = 1.0 - policy_mass.sum()
            if (
                float(q_collapsed) > self.config.identity_null_threshold
                and float(policy_mass.max()) < self.config.admitted_threshold
            ):
                candidate_kind = "collapsed_identity"
        else:
            if (
                float(q_identity) > self.config.identity_null_threshold
                and float(identity_mass.max()) < self.config.admitted_threshold
            ):
                candidate_kind = "new_identity"
            elif (
                self.condition != "identity_only_buffer"
                and float(identity_mass.max()) >= self.config.admitted_threshold
                and float(q_policy) > self.config.policy_null_threshold
            ):
                candidate_kind = "policy_revision"
                target_memory_id = self.slots[selected].memory_id

        if self.condition == "identity_only_buffer":
            policy_mass = identity_mass
            q_policy = torch.zeros_like(q_policy)
        return DualTrace(
            identity_mass=identity_mass,
            policy_mass=policy_mass,
            q_identity=q_identity,
            q_policy=q_policy,
            selected_slot=selected,
            candidate_kind=candidate_kind,
            target_memory_id=target_memory_id,
        )

    def act(self, observation: DualObservation) -> tuple[int, DualTrace]:
        trace = self.distribution(observation)
        values = torch.stack([slot.action_values for slot in self.slots])
        scores = torch.einsum("s,sa->a", trace.policy_mass, values)
        if trace.candidate_kind is not None:
            action = int(torch.randint(self.config.action_dim, (), generator=self.generator))
        else:
            action = int(scores.argmax())

        candidate_id: int | None = None
        candidate_route: str | None = None
        if trace.candidate_kind is not None:
            if self.condition == "dual_immediate":
                if trace.candidate_kind == "new_identity":
                    self._allocate(
                        observation,
                        action,
                        duplicate=False,
                        premature=True,
                    )
                else:
                    self._revise(
                        trace.target_memory_id,
                        observation,
                        action,
                        premature=True,
                    )
            else:
                candidate_route = self._candidate_route(trace.candidate_kind)
                candidate_id, evicted = self.buffers[candidate_route].add(
                    trace.candidate_kind,
                    observation,
                    trace.target_memory_id,
                )
                self.counters.buffer_evictions += int(evicted)
        self.pending[observation.event_id] = Pending(
            observation=observation,
            action=action,
            trace=trace,
            candidate_id=candidate_id,
            candidate_route=candidate_route,
        )
        return action, trace

    def resolve_reward(self, event_id: int, reward: float) -> Resolution:
        pending = self.pending.pop(event_id)
        candidate = (
            self.buffers[pending.candidate_route].resolve(
                pending.candidate_id, event_id, pending.action, reward
            )
            if pending.candidate_id is not None and pending.candidate_route is not None
            else None
        )
        if candidate is not None:
            action = int(candidate.positive_actions.argmax())
            observation = DualObservation(
                event_id,
                candidate.family,
                candidate.identity,
                candidate.policy,
            )
            if candidate.kind == "policy_revision":
                self._revise(
                    candidate.target_memory_id,
                    observation,
                    action,
                    premature=False,
                )
                return Resolution("policy_revision")
            duplicate = candidate.kind == "collapsed_identity" and self._identity_match(observation)
            self._allocate(
                observation,
                action,
                duplicate=duplicate,
                premature=False,
            )
            return Resolution("duplicate" if duplicate else "new_identity")

        if pending.candidate_id is not None or pending.trace.candidate_kind is not None:
            return Resolution(None)
        selected = pending.trace.selected_slot
        self._update_values(selected, pending.action, reward)
        return Resolution(None)

    def _identity_match(self, observation: DualObservation) -> bool:
        similarities = torch.tensor(
            [
                float(slot.identity @ observation.identity)
                if slot.family == observation.family
                else -1.0
                for slot in self.slots
            ]
        )
        return float(similarities.max()) >= self.config.similarity_threshold

    def _allocate(self, observation: DualObservation, action: int, *, duplicate: bool, premature: bool) -> None:
        slot = DualSlot(
            memory_id=self.next_memory_id,
            family=observation.family,
            identity=observation.identity.detach().clone(),
            policy=observation.policy.detach().clone(),
            action_values=F.one_hot(
                torch.tensor(action), num_classes=self.config.action_dim
            ).to(torch.float32),
            last_used=observation.event_id,
            established=False,
            origin_identity=None,
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
        self.counters.new_promotions += int(not premature and not duplicate)
        self.counters.duplicate_allocations += int(duplicate)
        self.counters.premature_writes += int(premature)

    def _revise(
        self,
        memory_id: int | None,
        observation: DualObservation,
        action: int,
        *,
        premature: bool,
    ) -> None:
        index = next(
            (i for i, slot in enumerate(self.slots) if slot.memory_id == memory_id),
            None,
        )
        if index is None:
            return
        slot = self.slots[index]
        slot.policy = observation.policy.detach().clone()
        slot.action_values = F.one_hot(
            torch.tensor(action), num_classes=self.config.action_dim
        ).to(torch.float32)
        slot.last_used = observation.event_id
        self.counters.revision_promotions += int(not premature)
        self.counters.premature_writes += int(premature)

    def _update_values(self, index: int, action: int, reward: float) -> None:
        slot = self.slots[index]
        rate = self.config.value_update_rate
        slot.action_values[action] = (
            (1.0 - rate) * slot.action_values[action] + rate * reward
        )
        slot.last_used += 1

    def probe_action(self, observation: DualObservation) -> int:
        trace = self.distribution(observation)
        values = torch.stack([slot.action_values for slot in self.slots])
        return int(torch.einsum("s,sa->a", trace.policy_mass, values).argmax())


def _probe(agent: DualRelationAgent, lifetime: DualLifetime, categories: tuple[int, ...]) -> float:
    correct = 0
    for offset, category in enumerate(categories):
        action = int(lifetime.current_actions[category])
        observation = DualObservation(
            agent.config.stream_steps + offset,
            category // 3,
            lifetime.identity_prototypes[category],
            policy_signature(action, agent.config.action_dim),
        )
        correct += int(agent.probe_action(observation) == action)
    return correct / len(categories)


def run_lifetime(
    condition: str,
    config: ExperimentConfig,
    lifetime: DualLifetime,
    seed: int,
    *,
    buffer_layout: str = "shared",
    shared_capacity: int | None = None,
    identity_capacity: int | None = None,
    policy_capacity: int | None = None,
) -> ExperimentMetrics:
    agent = DualRelationAgent(
        condition,
        config,
        lifetime,
        95_000_000 + seed,
        buffer_layout=buffer_layout,
        shared_capacity=shared_capacity,
        identity_capacity=identity_capacity,
        policy_capacity=policy_capacity,
    )
    due: dict[int, list[tuple[int, float, str]]] = {}
    decisions: list[dict[str, Any]] = []
    resolved_novel: set[int] = set()
    resolved_reversed: set[int] = set()

    for step, event in enumerate(lifetime.events):
        for event_id, reward, event_kind in due.get(step, []):
            resolution = agent.resolve_reward(event_id, reward)
            source = lifetime.events[event_id]
            if resolution.kind == "new_identity" and source.kind == "novel":
                resolved_novel.add(source.category)
            if resolution.kind == "policy_revision" and source.kind == "reversed":
                resolved_reversed.add(source.category)

        action, trace = agent.act(event.observation)
        correct = action == event.correct_action
        reward = 1.0 if correct else -1.0
        due.setdefault(step + config.outcome_delay, []).append(
            (event.observation.event_id, reward, event.kind)
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
        for event_id, reward, _ in due.get(step, []):
            agent.resolve_reward(event_id, reward)

    final = [row for row in decisions if row["step"] >= config.stream_steps - 200]
    by_kind = {
        kind: [float(row["correct"]) for row in final if row["kind"] == kind]
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
    resolved_policy = [
        row["q_policy"]
        for row in decisions
        if row["kind"] == "reversed" and row["policy_resolved"]
    ]
    return ExperimentMetrics(
        condition=condition,
        seed=seed,
        return_per_decision=_mean([row["reward"] for row in decisions]),
        overall_accuracy=_mean([float(row["correct"]) for row in decisions]),
        final_stable_accuracy=_mean(by_kind["stable"]),
        final_reversed_accuracy=_mean(by_kind["reversed"]),
        final_novel_accuracy=_mean(by_kind["novel"]),
        stable_probe_accuracy=_probe(agent, lifetime, lifetime.stable_categories),
        reversed_probe_accuracy=_probe(agent, lifetime, lifetime.reversed_categories),
        novel_probe_accuracy=_probe(agent, lifetime, lifetime.novel_categories),
        identity_residual_calibration=_mean(unresolved_identity) - _mean(resolved_identity),
        policy_residual_calibration=_mean(unresolved_policy) - _mean(resolved_policy),
        new_promotions=float(agent.counters.new_promotions),
        revision_promotions=float(agent.counters.revision_promotions),
        duplicate_allocations=float(agent.counters.duplicate_allocations),
        premature_write_rate=agent.counters.premature_writes / config.stream_steps,
        established_overwrite_rate=agent.counters.established_overwrites / config.stream_steps,
        buffer_evictions=float(agent.counters.buffer_evictions),
    )


def _aggregate(rows: list[ExperimentMetrics]) -> dict[str, dict[str, dict[str, float]]]:
    fields = [
        name
        for name in ExperimentMetrics.__dataclass_fields__
        if name not in {"condition", "seed"}
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
        "final_stable_accuracy",
        "final_reversed_accuracy",
        "final_novel_accuracy",
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
        output[f"{field}_approx_95ci_low"] = mean - half
        output[f"{field}_approx_95ci_high"] = mean + half
        output[f"{field}_wins"] = sum(value > 0.0 for value in values)
    return output


def _gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["dual_buffer"]
    versus_collapsed = result["paired"]["dual_buffer_minus_collapsed_buffer"]
    versus_identity = result["paired"]["dual_buffer_minus_identity_only_buffer"]
    versus_immediate = result["paired"]["dual_buffer_minus_dual_immediate"]
    return {
        "stable_accuracy_at_least_0.95": metrics["final_stable_accuracy"]["mean"] >= 0.95,
        "reversed_accuracy_at_least_0.75": metrics["final_reversed_accuracy"]["mean"] >= 0.75,
        "novel_accuracy_at_least_0.75": metrics["final_novel_accuracy"]["mean"] >= 0.75,
        "identity_calibration_at_least_0.15": metrics["identity_residual_calibration"]["mean"] >= 0.15,
        "policy_calibration_at_least_0.15": metrics["policy_residual_calibration"]["mean"] >= 0.15,
        "new_promotions_at_least_3": metrics["new_promotions"]["mean"] >= 3.0,
        "revision_promotions_at_least_3": metrics["revision_promotions"]["mean"] >= 3.0,
        "no_premature_writes": metrics["premature_write_rate"]["mean"] == 0.0,
        "no_established_overwrites": metrics["established_overwrite_rate"]["mean"] == 0.0,
        "no_duplicate_allocations": metrics["duplicate_allocations"]["mean"] == 0.0,
        "return_better_than_collapsed": versus_collapsed["return_per_decision_approx_95ci_low"] > 0.0,
        "return_better_than_identity_only": versus_identity["return_per_decision_approx_95ci_low"] > 0.0,
        "return_better_than_immediate": versus_immediate["return_per_decision_approx_95ci_low"] > 0.0,
        "novel_better_than_collapsed": versus_collapsed["final_novel_accuracy_approx_95ci_low"] > 0.0,
    }


def run_experiment(config: ExperimentConfig, *, seeds: int, seed_offset: int) -> dict[str, Any]:
    rows: list[ExperimentMetrics] = []
    for seed in range(seed_offset, seed_offset + seeds):
        lifetime = make_lifetime(config, seed)
        for condition in CONDITIONS:
            rows.append(run_lifetime(condition, config, lifetime, seed))
    comparisons = (
        ("dual_buffer", "collapsed_buffer"),
        ("dual_buffer", "identity_only_buffer"),
        ("dual_buffer", "dual_immediate"),
    )
    result: dict[str, Any] = {
        "experiment": "009_dual_relation",
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
    gate = _gate(result)
    result["gate"] = gate
    result["confirmation_triggered"] = all(gate.values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Experiment 009: {label} dual-relation assent",
        "",
        f"- Seeds: {result['seed_offset']}–{result['seed_offset'] + result['seeds'] - 1}",
        "",
        "| Condition | Return | Stable | Reversed | Novel | q identity | q policy | New promotions | Revisions | Duplicates | Premature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['final_stable_accuracy'])} | {_pm(metrics['final_reversed_accuracy'])} | "
            f"{_pm(metrics['final_novel_accuracy'])} | {_pm(metrics['identity_residual_calibration'])} | "
            f"{_pm(metrics['policy_residual_calibration'])} | {_pm(metrics['new_promotions'])} | "
            f"{_pm(metrics['revision_promotions'])} | {_pm(metrics['duplicate_allocations'])} | "
            f"{_pm(metrics['premature_write_rate'])} |"
        )
    lines.extend(["", "## Frozen gate", ""])
    for name, passed in result["gate"].items():
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
    stem = f"experiment_009_{label}"
    (output_dir / f"{stem}_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / f"{stem}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("development", "confirmation"), default="development")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    if args.label == "confirmation":
        seeds = 100 if args.seeds is None else args.seeds
        seed_offset = 91_000_000 if args.seed_offset is None else args.seed_offset
        if seed_offset < 91_000_000:
            raise ValueError("confirmation seeds must start at 91,000,000 or later")
    else:
        seeds = 20 if args.seeds is None else args.seeds
        seed_offset = 90_000_000 if args.seed_offset is None else args.seed_offset
    result = run_experiment(ExperimentConfig(), seeds=seeds, seed_offset=seed_offset)
    write_report(result, args.output_dir, args.label)
    print(json.dumps(_json_safe({
        "aggregate": result["aggregate"],
        "paired": result["paired"],
        "gate": result["gate"],
        "confirmation_triggered": result["confirmation_triggered"],
    }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
