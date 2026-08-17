"""Experiment 012a: protect self-created mature slots from allocation eviction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

import torch
from torch import Tensor

from experiments.experiment_004_reward_memory import _json_safe, _mean, _mean_sd
from experiments.experiment_006_predictive_geometry import FixedNonlinearSensor
from experiments.experiment_010_retrospective_policy import (
    Candidate,
    Resolution,
    RetrospectiveAgent,
    RetrospectiveLifetime,
    RetrospectiveObservation,
    make_lifetime,
)
from experiments.experiment_011_persistent_identity import (
    ExperimentConfig as BaseConfig,
    RepresentationDiagnostics,
    representation_diagnostics,
    train_identity_encoder,
    transform_lifetime,
)
from experiments.experiment_012_empty_memory import (
    BootstrapMetrics,
    ColdStartAgent,
    _probe,
)


CONDITIONS = (
    "learned_preloaded_protected",
    "learned_cold_baseline",
    "learned_cold_mature",
    "learned_cold_immediate_protection",
    "learned_cold_mature_direct",
)

DISPLAY_NAMES = {
    "learned_preloaded_protected": "Learned preloaded protected",
    "learned_cold_baseline": "Learned cold baseline",
    "learned_cold_mature": "Learned cold mature",
    "learned_cold_immediate_protection": "Learned cold immediately protected",
    "learned_cold_mature_direct": "Learned cold mature direct",
}


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    maturity_support: int = 4


@dataclass(frozen=True)
class MaturityMetrics(BootstrapMetrics):
    postshift_return: float
    mature_core_slots_at_shift: float
    mature_slots_evicted: float
    immature_slots_evicted: float
    allocation_deferrals: float


class MatureColdStartAgent(ColdStartAgent):
    """Cold-start agent with use-derived or immediate slot maturity."""

    def __init__(
        self,
        condition: str,
        config: ExperimentConfig,
        lifetime: RetrospectiveLifetime,
        seed: int,
        *,
        immediate_protection: bool = False,
    ) -> None:
        if config.maturity_support <= 0:
            raise ValueError("maturity support must be positive")
        super().__init__(condition, config, lifetime, seed)
        self.immediate_protection = immediate_protection
        self.successful_uses: dict[int, int] = {}
        self.mature_memory_ids: set[int] = set()
        self.mature_slots_evicted = 0
        self.immature_slots_evicted = 0
        self.allocation_deferrals = 0
        self.last_allocation_deferred = False

    def _slot_by_id(self, memory_id: int) -> Any | None:
        return next(
            (slot for slot in self.slots if slot.memory_id == memory_id), None
        )

    def _make_mature(self, memory_id: int) -> None:
        slot = self._slot_by_id(memory_id)
        if slot is None:
            return
        self.mature_memory_ids.add(memory_id)
        slot.established = True

    def _record_successful_use(self, memory_id: int) -> None:
        if memory_id in self.mature_memory_ids:
            return
        support = self.successful_uses.get(memory_id, 0) + 1
        self.successful_uses[memory_id] = support
        if support >= self.config.maturity_support:
            self._make_mature(memory_id)

    def _defer_candidate(self, candidate: Candidate) -> None:
        evicted = self.bank._insert(candidate)
        self.counters.buffer_evictions += int(evicted is not None)
        self.allocation_deferrals += 1
        self.last_allocation_deferred = True

    def _allocate(self, candidate: Candidate, action: int) -> bool:
        self.last_allocation_deferred = False
        candidate_observation = RetrospectiveObservation(
            candidate.last_seen,
            candidate.family,
            candidate.identity,
        )
        if self._identity_match(candidate_observation):
            return super()._allocate(candidate, action)
        if len(self.slots) >= self.config.permanent_capacity:
            eligible = [
                slot
                for slot in self.slots
                if slot.memory_id not in self.mature_memory_ids
            ]
            if not eligible:
                self._defer_candidate(candidate)
                return False

        before_ids = {slot.memory_id for slot in self.slots}
        before_mature = set(self.mature_memory_ids)
        allocated = super()._allocate(candidate, action)
        if not allocated:
            return False

        after_ids = {slot.memory_id for slot in self.slots}
        evicted_ids = before_ids - after_ids
        self.mature_slots_evicted += len(evicted_ids.intersection(before_mature))
        self.immature_slots_evicted += len(evicted_ids - before_mature)
        for memory_id in evicted_ids:
            self.successful_uses.pop(memory_id, None)
            self.mature_memory_ids.discard(memory_id)

        new_ids = after_ids - before_ids
        if not new_ids:
            new_ids = {max(after_ids)}
        new_id = max(new_ids)
        self.successful_uses[new_id] = 0
        if self.immediate_protection:
            self._make_mature(new_id)
        return True

    def resolve_reward(self, event_id: int, reward: float) -> Resolution:
        pending = self.pending.get(event_id)
        target_memory_id = (
            pending.trace.target_memory_id if pending is not None else None
        )
        admitted = (
            float(pending.trace.identity_mass.max())
            if pending is not None and pending.trace.identity_mass.numel()
            else 0.0
        )
        resolution = super().resolve_reward(event_id, reward)
        if (
            target_memory_id is not None
            and reward > 0.0
            and admitted >= self.config.admitted_threshold
            and self._slot_by_id(target_memory_id) is not None
        ):
            self._record_successful_use(target_memory_id)
        if resolution.kind == "identity_reconciled" and self.last_allocation_deferred:
            return Resolution("allocation_deferred", None, resolution.action)
        return resolution


def run_maturity_lifetime(
    output_condition: str,
    config: ExperimentConfig,
    lifetime: RetrospectiveLifetime,
    seed: int,
    *,
    cold_start: bool,
    direct_update: bool = False,
    maturity: bool = False,
    immediate_protection: bool = False,
) -> MaturityMetrics:
    agent_condition = "direct_update" if direct_update else "retrospective_protected"
    agent_seed = 117_000_000 + seed
    if not cold_start:
        agent: RetrospectiveAgent = RetrospectiveAgent(
            agent_condition,
            config,
            lifetime,
            agent_seed,
            revalidate_identity_promotion=True,
        )
    elif maturity:
        agent = MatureColdStartAgent(
            agent_condition,
            config,
            lifetime,
            agent_seed,
            immediate_protection=immediate_protection,
        )
    else:
        agent = ColdStartAgent(agent_condition, config, lifetime, agent_seed)

    due: dict[int, list[tuple[int, float]]] = {}
    decisions: list[dict[str, Any]] = []
    core_categories = tuple(lifetime.stable_categories) + tuple(
        lifetime.reversed_categories
    )
    promoted_core: set[int] = set()
    promoted_novel: set[int] = set()
    revised: set[int] = set()
    core_completion_step = config.shift_step + 1
    core_probe_at_shift = 0.0
    core_memory_ids: set[int] = set()
    mature_core_slots_at_shift = 0

    def record_resolution(resolution: Resolution, step: int) -> None:
        nonlocal core_completion_step
        origin = resolution.target_origin_category
        if resolution.kind == "new_identity" and origin is not None:
            if origin in core_categories and step < config.shift_step:
                promoted_core.add(origin)
                if len(promoted_core) == len(core_categories):
                    core_completion_step = min(core_completion_step, step)
            if origin in lifetime.novel_categories:
                promoted_novel.add(origin)
        if resolution.kind == "policy_revision" and origin is not None:
            if origin in lifetime.reversed_categories:
                revised.add(origin)

    for step, event in enumerate(lifetime.events):
        for event_id, reward in due.get(step, []):
            record_resolution(agent.resolve_reward(event_id, reward), step)

        if step == config.shift_step:
            core_probe_at_shift = _probe(
                agent,
                lifetime,
                core_categories,
                config.shift_step + config.stream_steps,
                lifetime.initial_actions,
            )
            core_memory_ids = {
                slot.memory_id
                for slot in agent.slots
                if slot.origin_category in core_categories
            }
            mature_ids = getattr(agent, "mature_memory_ids", set())
            mature_core_slots_at_shift = len(core_memory_ids.intersection(mature_ids))

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
                "correct": correct,
                "reward": reward,
                "q_identity": float(trace.q_identity),
                "q_policy": float(trace.q_policy),
                "identity_resolved": event.category in promoted_novel,
                "policy_resolved": event.category in revised,
            }
        )

    for step in range(config.stream_steps, config.stream_steps + config.outcome_delay):
        for event_id, reward in due.get(step, []):
            record_resolution(agent.resolve_reward(event_id, reward), step)

    early = [row for row in decisions if row["step"] < 50]
    late_core = [
        row
        for row in decisions
        if config.shift_step - 50 <= row["step"] < config.shift_step
    ]
    postshift = [row for row in decisions if row["step"] >= config.shift_step]
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
    final_ids = {slot.memory_id for slot in agent.slots}
    return MaturityMetrics(
        condition=output_condition,
        seed=seed,
        return_per_decision=_mean([row["reward"] for row in decisions]),
        clean_accuracy=_mean([float(row["correct"]) for row in decisions]),
        early_core_accuracy=_mean([float(row["correct"]) for row in early]),
        late_core_accuracy=_mean([float(row["correct"]) for row in late_core]),
        core_probe_at_shift=core_probe_at_shift,
        core_promotions_before_shift=float(len(promoted_core)),
        core_completion_step=float(core_completion_step),
        retention_accuracy=_mean([float(row["correct"]) for row in retention]),
        late_stable_accuracy=_mean(by_kind["stable"]),
        late_reversed_accuracy=_mean(by_kind["reversed"]),
        late_novel_accuracy=_mean(by_kind["novel"]),
        stable_probe_accuracy=_probe(
            agent,
            lifetime,
            lifetime.stable_categories,
            config.stream_steps + 20,
            lifetime.current_actions,
        ),
        reversed_probe_accuracy=_probe(
            agent,
            lifetime,
            lifetime.reversed_categories,
            config.stream_steps + 30,
            lifetime.current_actions,
        ),
        novel_probe_accuracy=_probe(
            agent,
            lifetime,
            lifetime.novel_categories,
            config.stream_steps + 40,
            lifetime.current_actions,
        ),
        identity_residual_calibration=_mean(unresolved_identity)
        - _mean(resolved_identity),
        policy_residual_calibration=_mean(unresolved_policy) - _mean(stable_policy),
        postshift_novel_promotions=float(len(promoted_novel)),
        unique_revision_categories=float(len(revised)),
        false_stable_revisions=float(agent.counters.false_stable_revisions),
        under_supported_writes=float(agent.counters.under_supported_writes),
        established_overwrites=float(agent.counters.established_overwrites),
        duplicate_allocations=float(agent.counters.duplicate_allocations),
        identity_reconciliations=float(agent.counters.identity_reconciliations),
        core_slots_lost=float(len(core_memory_ids - final_ids)),
        buffer_evictions=float(agent.counters.buffer_evictions),
        postshift_return=_mean([row["reward"] for row in postshift]),
        mature_core_slots_at_shift=float(mature_core_slots_at_shift),
        mature_slots_evicted=float(getattr(agent, "mature_slots_evicted", 0)),
        immature_slots_evicted=float(getattr(agent, "immature_slots_evicted", 0)),
        allocation_deferrals=float(getattr(agent, "allocation_deferrals", 0)),
    )


def _aggregate(
    rows: list[MaturityMetrics],
) -> dict[str, dict[str, dict[str, float]]]:
    fields = [
        field
        for field in MaturityMetrics.__dataclass_fields__
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
    rows: list[MaturityMetrics], first: str, second: str
) -> dict[str, float]:
    by_key = {(row.condition, row.seed): row for row in rows}
    seeds = sorted({row.seed for row in rows})
    output: dict[str, float] = {}
    for field in (
        "return_per_decision",
        "postshift_return",
        "retention_accuracy",
        "core_probe_at_shift",
        "reversed_probe_accuracy",
        "novel_probe_accuracy",
        "core_slots_lost",
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


def _aggregate_representation(
    rows: list[RepresentationDiagnostics],
) -> dict[str, dict[str, float]]:
    return {
        field: _mean_sd([getattr(row, field) for row in rows])
        for field in RepresentationDiagnostics.__dataclass_fields__
    }


def _gate(result: dict[str, Any]) -> dict[str, bool]:
    mature = result["aggregate"]["learned_cold_mature"]
    versus_baseline = result["paired"][
        "learned_cold_mature_minus_learned_cold_baseline"
    ]
    versus_immediate = result["paired"][
        "learned_cold_mature_minus_learned_cold_immediate_protection"
    ]
    versus_preloaded = result["paired"][
        "learned_cold_mature_minus_learned_preloaded_protected"
    ]
    versus_direct = result["paired"][
        "learned_cold_mature_minus_learned_cold_mature_direct"
    ]
    return {
        "core_promotions_at_least_7.5": mature["core_promotions_before_shift"]["mean"] >= 7.5,
        "mature_core_slots_at_least_7": mature["mature_core_slots_at_shift"]["mean"] >= 7.0,
        "core_probe_at_least_0.75": mature["core_probe_at_shift"]["mean"] >= 0.75,
        "retention_at_least_0.85": mature["retention_accuracy"]["mean"] >= 0.85,
        "reversed_probe_at_least_0.70": mature["reversed_probe_accuracy"]["mean"] >= 0.70,
        "novel_probe_at_least_0.70": mature["novel_probe_accuracy"]["mean"] >= 0.70,
        "postshift_novel_promotions_at_least_3": mature["postshift_novel_promotions"]["mean"] >= 3.0,
        "unique_revisions_at_least_3": mature["unique_revision_categories"]["mean"] >= 3.0,
        "identity_calibration_at_least_0.10": mature["identity_residual_calibration"]["mean"] >= 0.10,
        "policy_calibration_at_least_0.10": mature["policy_residual_calibration"]["mean"] >= 0.10,
        "no_mature_slot_evictions": mature["mature_slots_evicted"]["mean"] == 0.0,
        "no_duplicate_allocations": mature["duplicate_allocations"]["mean"] == 0.0,
        "no_established_overwrites": mature["established_overwrites"]["mean"] == 0.0,
        "no_under_supported_writes": mature["under_supported_writes"]["mean"] == 0.0,
        "no_core_slots_lost": mature["core_slots_lost"]["mean"] == 0.0,
        "postshift_return_noninferior_to_baseline": versus_baseline["postshift_return_approx_95ci_low"] > -0.05,
        "retention_noninferior_to_baseline": versus_baseline["retention_accuracy_approx_95ci_low"] > -0.05,
        "novel_probe_noninferior_to_baseline": versus_baseline["novel_probe_accuracy_approx_95ci_low"] > -0.05,
        "postshift_return_noninferior_to_immediate": versus_immediate["postshift_return_approx_95ci_low"] > -0.05,
        "novel_probe_noninferior_to_immediate": versus_immediate["novel_probe_accuracy_approx_95ci_low"] > -0.05,
        "postshift_return_noninferior_to_preloaded": versus_preloaded["postshift_return_approx_95ci_low"] > -0.08,
        "retention_noninferior_to_preloaded": versus_preloaded["retention_accuracy_approx_95ci_low"] > -0.05,
        "retention_noninferior_to_direct": versus_direct["retention_accuracy_approx_95ci_low"] > -0.05,
        "novel_probe_better_than_direct": versus_direct["novel_probe_accuracy_approx_95ci_low"] > 0.0,
        "total_return_cost_vs_direct_within_0.15": versus_direct["return_per_decision_approx_95ci_low"] > -0.15,
    }


def run_study(
    config: ExperimentConfig,
    *,
    training_seeds: tuple[int, ...],
    lifetimes_per_encoder: int,
    lifetime_seed_offset: int,
    status: str,
) -> dict[str, Any]:
    sensor = FixedNonlinearSensor(
        config.identity_dim, config.sensor_hidden_dim, config.sensor_seed
    ).eval()
    rows: list[MaturityMetrics] = []
    diagnostics: list[RepresentationDiagnostics] = []
    training: list[dict[str, float | int]] = []

    for encoder_index, training_seed in enumerate(training_seeds):
        encoder, losses = train_identity_encoder(
            config, sensor, training_seed, "pairwise"
        )
        diagnostics.append(
            representation_diagnostics(config, sensor, encoder, training_seed)
        )
        training.append(
            {
                "training_seed": training_seed,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "last_50_loss": statistics.fmean(losses[-50:]),
            }
        )
        transform: Callable[[Tensor], Tensor] = (
            lambda value, encoder=encoder: encoder(sensor(value))
        )
        for lifetime_index in range(lifetimes_per_encoder):
            seed = lifetime_seed_offset + encoder_index * 1_000 + lifetime_index
            latent = make_lifetime(config, seed)
            learned = transform_lifetime(latent, transform)
            specifications = (
                ("learned_preloaded_protected", False, False, False, False),
                ("learned_cold_baseline", True, False, False, False),
                ("learned_cold_mature", True, False, True, False),
                ("learned_cold_immediate_protection", True, False, True, True),
                ("learned_cold_mature_direct", True, True, True, False),
            )
            for output, cold, direct, maturity, immediate in specifications:
                rows.append(
                    run_maturity_lifetime(
                        output,
                        config,
                        learned,
                        seed,
                        cold_start=cold,
                        direct_update=direct,
                        maturity=maturity,
                        immediate_protection=immediate,
                    )
                )

    comparisons = (
        ("learned_cold_mature", "learned_cold_baseline"),
        ("learned_cold_mature", "learned_cold_immediate_protection"),
        ("learned_cold_mature", "learned_preloaded_protected"),
        ("learned_cold_mature", "learned_cold_mature_direct"),
    )
    result: dict[str, Any] = {
        "experiment": "012a_slot_maturity",
        "status": status,
        "config": asdict(config),
        "training_seeds": list(training_seeds),
        "lifetimes_per_encoder": lifetimes_per_encoder,
        "lifetime_seed_offset": lifetime_seed_offset,
        "aggregate": _aggregate(rows),
        "representation": _aggregate_representation(diagnostics),
        "training": training,
        "paired": {
            f"{first}_minus_{second}": _paired(rows, first, second)
            for first, second in comparisons
        },
        "individual": [asdict(row) for row in rows],
    }
    result["frozen_gate"] = _gate(result)
    result["passed"] = all(result["frozen_gate"].values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = result["status"]
    lines = [
        f"# Experiment 012a: slot maturity {status}",
        "",
        f"- Encoder seeds: {result['training_seeds']}",
        f"- Lifetimes per encoder: {result['lifetimes_per_encoder']}",
        f"- Successful uses required for maturity: {result['config']['maturity_support']}",
        "",
        "| Condition | Return | Post-shift return | Shift probe | Mature core | Retention | Novel probe | Core lost | Deferrals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['postshift_return'])} | "
            f"{_pm(metrics['core_probe_at_shift'])} | "
            f"{_pm(metrics['mature_core_slots_at_shift'])} | "
            f"{_pm(metrics['retention_accuracy'])} | "
            f"{_pm(metrics['novel_probe_accuracy'])} | "
            f"{_pm(metrics['core_slots_lost'])} | "
            f"{_pm(metrics['allocation_deferrals'])} |"
        )
    lines.extend(["", "## Frozen gate", ""])
    for name, passed in result["frozen_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Overall {status} result: **{'PASS' if result['passed'] else 'FAIL'}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    stem = f"experiment_012a_{status}"
    (output_dir / f"{stem}_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / f"{stem}_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    if args.mode == "development":
        result = run_study(
            ExperimentConfig(),
            training_seeds=(1320, 1321),
            lifetimes_per_encoder=10,
            lifetime_seed_offset=115_000_000,
            status="development",
        )
    else:
        result = run_study(
            ExperimentConfig(),
            training_seeds=tuple(range(1330, 1340)),
            lifetimes_per_encoder=20,
            lifetime_seed_offset=116_000_000,
            status="confirmation",
        )
    write_report(result, args.output_dir)
    print(
        json.dumps(
            _json_safe(
                {
                    "aggregate": result["aggregate"],
                    "representation": result["representation"],
                    "paired": result["paired"],
                    "frozen_gate": result["frozen_gate"],
                    "passed": result["passed"],
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
