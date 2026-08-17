"""Experiment 012: construct the protected memory core from empty N."""

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
    ExperimentMetrics,
    Resolution,
    RetrospectiveAgent,
    RetrospectiveLifetime,
    RetrospectiveObservation,
    RetrospectiveTrace,
    make_lifetime,
)
from experiments.experiment_011_persistent_identity import (
    ExperimentConfig,
    RepresentationDiagnostics,
    representation_diagnostics,
    train_identity_encoder,
    transform_lifetime,
)


CONDITIONS = (
    "oracle_preloaded_protected",
    "learned_preloaded_protected",
    "oracle_cold_protected",
    "raw_sensor_cold_protected",
    "learned_cold_protected",
    "learned_cold_direct",
)

DISPLAY_NAMES = {
    "oracle_preloaded_protected": "Oracle preloaded protected",
    "learned_preloaded_protected": "Learned preloaded protected",
    "oracle_cold_protected": "Oracle cold protected",
    "raw_sensor_cold_protected": "Raw-sensor cold protected",
    "learned_cold_protected": "Learned cold protected",
    "learned_cold_direct": "Learned cold direct adaptation",
}


@dataclass(frozen=True)
class BootstrapMetrics:
    condition: str
    seed: int
    return_per_decision: float
    clean_accuracy: float
    early_core_accuracy: float
    late_core_accuracy: float
    core_probe_at_shift: float
    core_promotions_before_shift: float
    core_completion_step: float
    retention_accuracy: float
    late_stable_accuracy: float
    late_reversed_accuracy: float
    late_novel_accuracy: float
    stable_probe_accuracy: float
    reversed_probe_accuracy: float
    novel_probe_accuracy: float
    identity_residual_calibration: float
    policy_residual_calibration: float
    postshift_novel_promotions: float
    unique_revision_categories: float
    false_stable_revisions: float
    under_supported_writes: float
    established_overwrites: float
    duplicate_allocations: float
    identity_reconciliations: float
    core_slots_lost: float
    buffer_evictions: float


class ColdStartAgent(RetrospectiveAgent):
    """The confirmed agent with no permanent slots installed at construction."""

    def __init__(
        self,
        condition: str,
        config: ExperimentConfig,
        lifetime: RetrospectiveLifetime,
        seed: int,
    ) -> None:
        super().__init__(
            condition,
            config,
            lifetime,
            seed,
            revalidate_identity_promotion=True,
        )
        self.slots.clear()
        self.next_memory_id = 0
        self.last_allocation_origin: int | None = None

    def distribution(
        self, observation: RetrospectiveObservation
    ) -> RetrospectiveTrace:
        if self.slots:
            return super().distribution(observation)
        empty = torch.empty(0, dtype=observation.identity.dtype)
        return RetrospectiveTrace(
            identity_mass=empty,
            q_identity=torch.tensor(1.0, dtype=observation.identity.dtype),
            q_policy=torch.tensor(0.0, dtype=observation.identity.dtype),
            selected_slot=-1,
            target_memory_id=None,
            candidate_kind="new_identity",
            policy_search_active=False,
        )

    def _allocate(self, candidate: Candidate, action: int) -> bool:
        self.last_allocation_origin = None
        allocated = super()._allocate(candidate, action)
        if not allocated:
            return False
        origin = self.lifetime.events[candidate.last_seen].category
        new_slot = max(self.slots, key=lambda slot: slot.memory_id)
        new_slot.origin_category = origin
        self.last_allocation_origin = origin
        return True

    def _identity_match(self, observation: RetrospectiveObservation) -> bool:
        if not self.slots:
            return False
        return super()._identity_match(observation)

    def resolve_reward(self, event_id: int, reward: float) -> Resolution:
        resolution = super().resolve_reward(event_id, reward)
        if resolution.kind == "new_identity":
            return Resolution(
                kind=resolution.kind,
                target_origin_category=self.last_allocation_origin,
                action=resolution.action,
            )
        return resolution

    def probe_action(self, observation: RetrospectiveObservation) -> int:
        if not self.slots:
            return 0
        return super().probe_action(observation)


def _probe(
    agent: RetrospectiveAgent,
    lifetime: RetrospectiveLifetime,
    categories: tuple[int, ...],
    event_offset: int,
    actions: Tensor,
) -> float:
    correct = 0
    for offset, category in enumerate(categories):
        observation = RetrospectiveObservation(
            event_offset + offset,
            category // 3,
            lifetime.identity_prototypes[category],
        )
        correct += int(
            agent.probe_action(observation)
            == int(actions[category])
        )
    return correct / len(categories)


def run_bootstrap_lifetime(
    output_condition: str,
    agent_condition: str,
    config: ExperimentConfig,
    lifetime: RetrospectiveLifetime,
    seed: int,
    *,
    cold_start: bool,
) -> BootstrapMetrics:
    agent_class = ColdStartAgent if cold_start else RetrospectiveAgent
    if cold_start:
        agent = agent_class(
            agent_condition, config, lifetime, 115_000_000 + seed
        )
    else:
        agent = agent_class(
            agent_condition,
            config,
            lifetime,
            115_000_000 + seed,
            revalidate_identity_promotion=True,
        )
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
    return BootstrapMetrics(
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
    )


def _aggregate(
    rows: list[BootstrapMetrics],
) -> dict[str, dict[str, dict[str, float]]]:
    fields = [
        field
        for field in BootstrapMetrics.__dataclass_fields__
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
    rows: list[BootstrapMetrics], first: str, second: str
) -> dict[str, float]:
    by_key = {(row.condition, row.seed): row for row in rows}
    seeds = sorted({row.seed for row in rows})
    output: dict[str, float] = {}
    for field in (
        "return_per_decision",
        "clean_accuracy",
        "core_probe_at_shift",
        "retention_accuracy",
        "reversed_probe_accuracy",
        "novel_probe_accuracy",
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
    cold = result["aggregate"]["learned_cold_protected"]
    versus_raw = result["paired"][
        "learned_cold_protected_minus_raw_sensor_cold_protected"
    ]
    versus_oracle = result["paired"][
        "learned_cold_protected_minus_oracle_cold_protected"
    ]
    versus_preloaded = result["paired"][
        "learned_cold_protected_minus_learned_preloaded_protected"
    ]
    versus_direct = result["paired"][
        "learned_cold_protected_minus_learned_cold_direct"
    ]
    return {
        "core_promotions_at_least_7.5": cold["core_promotions_before_shift"]["mean"] >= 7.5,
        "late_core_accuracy_at_least_0.70": cold["late_core_accuracy"]["mean"] >= 0.70,
        "core_probe_at_least_0.75": cold["core_probe_at_shift"]["mean"] >= 0.75,
        "retention_accuracy_at_least_0.85": cold["retention_accuracy"]["mean"] >= 0.85,
        "reversed_probe_at_least_0.70": cold["reversed_probe_accuracy"]["mean"] >= 0.70,
        "novel_probe_at_least_0.70": cold["novel_probe_accuracy"]["mean"] >= 0.70,
        "postshift_novel_promotions_at_least_3": cold["postshift_novel_promotions"]["mean"] >= 3.0,
        "unique_revisions_at_least_3": cold["unique_revision_categories"]["mean"] >= 3.0,
        "identity_calibration_at_least_0.10": cold["identity_residual_calibration"]["mean"] >= 0.10,
        "policy_calibration_at_least_0.10": cold["policy_residual_calibration"]["mean"] >= 0.10,
        "false_stable_revisions_at_most_0.50": cold["false_stable_revisions"]["mean"] <= 0.50,
        "no_duplicate_allocations": cold["duplicate_allocations"]["mean"] == 0.0,
        "no_established_overwrites": cold["established_overwrites"]["mean"] == 0.0,
        "no_under_supported_writes": cold["under_supported_writes"]["mean"] == 0.0,
        "return_better_than_raw_cold": versus_raw["return_per_decision_approx_95ci_low"] > 0.0,
        "core_probe_better_than_raw_cold": versus_raw["core_probe_at_shift_approx_95ci_low"] > 0.0,
        "return_noninferior_to_oracle_cold": versus_oracle["return_per_decision_approx_95ci_low"] > -0.08,
        "clean_accuracy_noninferior_to_oracle_cold": versus_oracle["clean_accuracy_approx_95ci_low"] > -0.08,
        "core_probe_noninferior_to_oracle_cold": versus_oracle["core_probe_at_shift_approx_95ci_low"] > -0.10,
        "return_noninferior_to_learned_preloaded": versus_preloaded["return_per_decision_approx_95ci_low"] > -0.15,
        "retention_noninferior_to_learned_preloaded": versus_preloaded["retention_accuracy_approx_95ci_low"] > -0.08,
        "return_noninferior_to_direct": versus_direct["return_per_decision_approx_95ci_low"] > -0.08,
        "clean_accuracy_noninferior_to_direct": versus_direct["clean_accuracy_approx_95ci_low"] > -0.08,
        "retention_noninferior_to_direct": versus_direct["retention_accuracy_approx_95ci_low"] > -0.05,
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
    rows: list[BootstrapMetrics] = []
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
        transforms: dict[str, Callable[[Tensor], Tensor]] = {
            "raw": sensor,
            "learned": lambda value, encoder=encoder: encoder(sensor(value)),
        }
        for lifetime_index in range(lifetimes_per_encoder):
            seed = lifetime_seed_offset + encoder_index * 1_000 + lifetime_index
            latent = make_lifetime(config, seed)
            raw = transform_lifetime(latent, transforms["raw"])
            learned = transform_lifetime(latent, transforms["learned"])
            specifications = (
                ("oracle_preloaded_protected", "retrospective_protected", latent, False),
                ("learned_preloaded_protected", "retrospective_protected", learned, False),
                ("oracle_cold_protected", "retrospective_protected", latent, True),
                ("raw_sensor_cold_protected", "retrospective_protected", raw, True),
                ("learned_cold_protected", "retrospective_protected", learned, True),
                ("learned_cold_direct", "direct_update", learned, True),
            )
            for output, agent_condition, lifetime, cold_start in specifications:
                rows.append(
                    run_bootstrap_lifetime(
                        output,
                        agent_condition,
                        config,
                        lifetime,
                        seed,
                        cold_start=cold_start,
                    )
                )

    comparisons = (
        ("learned_cold_protected", "raw_sensor_cold_protected"),
        ("learned_cold_protected", "oracle_cold_protected"),
        ("learned_cold_protected", "learned_preloaded_protected"),
        ("learned_cold_protected", "learned_cold_direct"),
    )
    result: dict[str, Any] = {
        "experiment": "012_empty_memory_bootstrap",
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
        f"# Experiment 012: empty-memory bootstrap {status}",
        "",
        f"- Encoder seeds: {result['training_seeds']}",
        f"- RL lifetimes per encoder: {result['lifetimes_per_encoder']}",
        "- Main condition begins with **zero permanent memory slots**",
        "",
        "| Condition | Return | Late core | Shift probe | Core IDs | Retention | Reversed probe | Novel probe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['late_core_accuracy'])} | "
            f"{_pm(metrics['core_probe_at_shift'])} | "
            f"{_pm(metrics['core_promotions_before_shift'])} | "
            f"{_pm(metrics['retention_accuracy'])} | "
            f"{_pm(metrics['reversed_probe_accuracy'])} | "
            f"{_pm(metrics['novel_probe_accuracy'])} |"
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
    stem = f"experiment_012_{status}"
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
            training_seeds=(1300, 1301),
            lifetimes_per_encoder=10,
            lifetime_seed_offset=113_000_000,
            status="development",
        )
    else:
        result = run_study(
            ExperimentConfig(),
            training_seeds=tuple(range(1310, 1320)),
            lifetimes_per_encoder=20,
            lifetime_seed_offset=114_000_000,
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
