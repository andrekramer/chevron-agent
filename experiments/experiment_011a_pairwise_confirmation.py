"""Fresh-seed confirmation of Experiment 011's pairwise identity learner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

from torch import Tensor

from experiments.experiment_004_reward_memory import _json_safe, _mean_sd
from experiments.experiment_006_predictive_geometry import FixedNonlinearSensor
from experiments.experiment_010_retrospective_policy import (
    ExperimentMetrics,
    make_lifetime,
    run_lifetime,
)
from experiments.experiment_011_persistent_identity import (
    ExperimentConfig,
    RepresentationDiagnostics,
    representation_diagnostics,
    train_identity_encoder,
    transform_lifetime,
)


CONDITIONS = (
    "oracle_protected",
    "raw_sensor_protected",
    "pairwise_temporal_protected",
    "pairwise_temporal_direct",
)

DISPLAY_NAMES = {
    "oracle_protected": "Oracle protected Chevron",
    "raw_sensor_protected": "Raw-sensor protected Chevron",
    "pairwise_temporal_protected": "Pairwise-temporal protected Chevron",
    "pairwise_temporal_direct": "Pairwise-temporal direct adaptation",
}


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


def _confirmation_gate(result: dict[str, Any]) -> dict[str, bool]:
    learned = result["aggregate"]["pairwise_temporal_protected"]
    representation = result["representation"]
    versus_raw = result["paired"][
        "pairwise_temporal_protected_minus_raw_sensor_protected"
    ]
    versus_oracle = result["paired"][
        "pairwise_temporal_protected_minus_oracle_protected"
    ]
    versus_direct = result["paired"][
        "pairwise_temporal_protected_minus_pairwise_temporal_direct"
    ]
    return {
        "retention_accuracy_at_least_0.90": learned["retention_accuracy"]["mean"] >= 0.90,
        "reversed_probe_at_least_0.75": learned["reversed_probe_accuracy"]["mean"] >= 0.75,
        "novel_probe_at_least_0.75": learned["novel_probe_accuracy"]["mean"] >= 0.75,
        "new_promotions_at_least_3": learned["new_promotions"]["mean"] >= 3.0,
        "unique_revisions_at_least_3": learned["unique_revision_categories"]["mean"] >= 3.0,
        "identity_calibration_at_least_0.10": learned["identity_residual_calibration"]["mean"] >= 0.10,
        "policy_calibration_at_least_0.10": learned["policy_residual_calibration"]["mean"] >= 0.10,
        "false_stable_revisions_at_most_0.25": learned["false_stable_revisions"]["mean"] <= 0.25,
        "no_duplicate_allocations": learned["duplicate_allocations"]["mean"] == 0.0,
        "no_established_overwrites": learned["established_overwrites"]["mean"] == 0.0,
        "no_under_supported_writes": learned["under_supported_writes"]["mean"] == 0.0,
        "same_identity_admission_at_least_0.90": representation["same_identity_admission"]["mean"] >= 0.90,
        "confusable_change_rejection_at_least_0.80": representation["confusable_change_rejection"]["mean"] >= 0.80,
        "balanced_identity_accuracy_at_least_0.85": representation["balanced_identity_accuracy"]["mean"] >= 0.85,
        "return_better_than_raw_sensor": versus_raw["return_per_decision_approx_95ci_low"] > 0.0,
        "novel_probe_better_than_raw_sensor": versus_raw["novel_probe_accuracy_approx_95ci_low"] > 0.0,
        "return_noninferior_to_oracle": versus_oracle["return_per_decision_approx_95ci_low"] > -0.08,
        "clean_accuracy_noninferior_to_oracle": versus_oracle["clean_accuracy_approx_95ci_low"] > -0.08,
        "retention_noninferior_to_oracle": versus_oracle["retention_accuracy_approx_95ci_low"] > -0.05,
        "return_noninferior_to_direct": versus_direct["return_per_decision_approx_95ci_low"] > -0.08,
        "clean_accuracy_noninferior_to_direct": versus_direct["clean_accuracy_approx_95ci_low"] > -0.08,
        "retention_noninferior_to_direct": versus_direct["retention_accuracy_approx_95ci_low"] > -0.03,
    }


def run_confirmation(
    config: ExperimentConfig,
    *,
    training_seeds: tuple[int, ...] = tuple(range(1200, 1210)),
    lifetimes_per_encoder: int = 20,
    lifetime_seed_offset: int = 112_000_000,
) -> dict[str, Any]:
    if not training_seeds or min(training_seeds) < 1200:
        raise ValueError("confirmation encoder seeds must start at 1200 or later")
    if lifetime_seed_offset < 112_000_000:
        raise ValueError("confirmation lifetime seeds must start at 112,000,000")
    if lifetimes_per_encoder <= 0:
        raise ValueError("lifetimes per encoder must be positive")

    sensor = FixedNonlinearSensor(
        config.identity_dim, config.sensor_hidden_dim, config.sensor_seed
    ).eval()
    rows: list[ExperimentMetrics] = []
    diagnostic_rows: list[RepresentationDiagnostics] = []
    training: list[dict[str, float | int]] = []

    for encoder_index, training_seed in enumerate(training_seeds):
        encoder, losses = train_identity_encoder(
            config, sensor, training_seed, "pairwise"
        )
        diagnostic_rows.append(
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
            latent_lifetime = make_lifetime(config, seed)
            raw_lifetime = transform_lifetime(latent_lifetime, transforms["raw"])
            learned_lifetime = transform_lifetime(
                latent_lifetime, transforms["learned"]
            )
            specifications = (
                ("oracle_protected", "retrospective_protected", latent_lifetime),
                ("raw_sensor_protected", "retrospective_protected", raw_lifetime),
                ("pairwise_temporal_protected", "retrospective_protected", learned_lifetime),
                ("pairwise_temporal_direct", "direct_update", learned_lifetime),
            )
            for output_condition, agent_condition, lifetime in specifications:
                metric = run_lifetime(
                    agent_condition,
                    config,
                    lifetime,
                    seed,
                    revalidate_identity_promotion=True,
                )
                rows.append(replace(metric, condition=output_condition))

    comparisons = (
        ("pairwise_temporal_protected", "raw_sensor_protected"),
        ("pairwise_temporal_protected", "oracle_protected"),
        ("pairwise_temporal_protected", "pairwise_temporal_direct"),
    )
    result: dict[str, Any] = {
        "experiment": "011a_pairwise_identity_confirmation",
        "status": "fresh_seed_confirmation",
        "config": asdict(config),
        "training_seeds": list(training_seeds),
        "lifetimes_per_encoder": lifetimes_per_encoder,
        "lifetime_seed_offset": lifetime_seed_offset,
        "aggregate": _aggregate(rows),
        "representation": _aggregate_representation(diagnostic_rows),
        "training": training,
        "paired": {
            f"{first}_minus_{second}": _paired(rows, first, second)
            for first, second in comparisons
        },
        "individual": [asdict(row) for row in rows],
    }
    result["confirmation_gate"] = _confirmation_gate(result)
    result["confirmation_passed"] = all(result["confirmation_gate"].values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 011a: pairwise-identity confirmation",
        "",
        f"- Encoder seeds: {result['training_seeds']}",
        f"- Paired RL lifetimes: {len(result['individual']) // len(CONDITIONS)}",
        "- Representation learner: **frozen pairwise temporal contrastive**",
        "- Downstream identity threshold: **fixed at cosine 0.62**",
        "",
        "| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | Identity calibration | False revisions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['clean_accuracy'])} | {_pm(metrics['retention_accuracy'])} | "
            f"{_pm(metrics['reversed_probe_accuracy'])} | "
            f"{_pm(metrics['novel_probe_accuracy'])} | "
            f"{_pm(metrics['identity_residual_calibration'])} | "
            f"{_pm(metrics['false_stable_revisions'])} |"
        )
    representation = result["representation"]
    lines.extend(
        [
            "",
            "## Learned representation",
            "",
            f"- Same-identity admission: {_pm(representation['same_identity_admission'])}",
            f"- Confusable-change rejection: {_pm(representation['confusable_change_rejection'])}",
            f"- Balanced identity accuracy: {_pm(representation['balanced_identity_accuracy'])}",
            f"- Latent cosine correlation: {_pm(representation['latent_cosine_correlation'])}",
            "",
            "## Frozen confirmation gate",
            "",
        ]
    )
    for name, passed in result["confirmation_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Confirmation passed: **{result['confirmation_passed']}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    (output_dir / "experiment_011a_confirmation_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "experiment_011a_confirmation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    result = run_confirmation(ExperimentConfig())
    write_report(result, args.output_dir)
    print(
        json.dumps(
            _json_safe(
                {
                    "aggregate": result["aggregate"],
                    "representation": result["representation"],
                    "paired": result["paired"],
                    "confirmation_gate": result["confirmation_gate"],
                    "confirmation_passed": result["confirmation_passed"],
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
