"""Experiment 009b: fresh-seed confirmation of shared candidate capacity."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
from typing import Any

from experiments.experiment_004_reward_memory import _json_safe, _mean_sd
from experiments.experiment_009_dual_relation import (
    ExperimentConfig,
    ExperimentMetrics,
    make_lifetime,
    run_lifetime,
)


CONDITIONS = (
    "dual_shared_4",
    "dual_shared_8",
    "identity_only_shared_4",
)

DISPLAY_NAMES = {
    "dual_shared_4": "Dual shared 4",
    "dual_shared_8": "Dual shared 8",
    "identity_only_shared_4": "Identity only shared 4",
}


def _run_condition(
    label: str,
    config: ExperimentConfig,
    lifetime: Any,
    seed: int,
) -> ExperimentMetrics:
    if label == "dual_shared_4":
        metric = run_lifetime(
            "dual_buffer", config, lifetime, seed, shared_capacity=4
        )
    elif label == "dual_shared_8":
        metric = run_lifetime(
            "dual_buffer", config, lifetime, seed, shared_capacity=8
        )
    elif label == "identity_only_shared_4":
        metric = run_lifetime(
            "identity_only_buffer", config, lifetime, seed, shared_capacity=4
        )
    else:
        raise ValueError(f"unknown confirmation condition {label}")
    return replace(metric, condition=label)


def _aggregate(rows: list[ExperimentMetrics]) -> dict[str, dict[str, dict[str, float]]]:
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
        "final_stable_accuracy",
        "final_reversed_accuracy",
        "final_novel_accuracy",
        "buffer_evictions",
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


def _capacity_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["dual_shared_8"]
    comparison = result["paired"]["dual_shared_8_minus_dual_shared_4"]
    shared_4_evictions = result["aggregate"]["dual_shared_4"]["buffer_evictions"]["mean"]
    shared_8_evictions = metrics["buffer_evictions"]["mean"]
    return {
        "stable_accuracy_at_least_0.95": metrics["final_stable_accuracy"]["mean"] >= 0.95,
        "reversed_accuracy_at_least_0.80": metrics["final_reversed_accuracy"]["mean"] >= 0.80,
        "novel_accuracy_at_least_0.80": metrics["final_novel_accuracy"]["mean"] >= 0.80,
        "identity_calibration_at_least_0.15": metrics["identity_residual_calibration"]["mean"] >= 0.15,
        "policy_calibration_at_least_0.15": metrics["policy_residual_calibration"]["mean"] >= 0.15,
        "new_promotions_at_least_3": metrics["new_promotions"]["mean"] >= 3.0,
        "revision_promotions_at_least_3": metrics["revision_promotions"]["mean"] >= 3.0,
        "no_premature_writes": metrics["premature_write_rate"]["mean"] == 0.0,
        "no_established_overwrites": metrics["established_overwrite_rate"]["mean"] == 0.0,
        "no_duplicate_allocations": metrics["duplicate_allocations"]["mean"] == 0.0,
        "return_better_than_shared_4": comparison[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "reversed_better_than_shared_4": comparison[
            "final_reversed_accuracy_approx_95ci_low"
        ] > 0.0,
        "novel_better_than_shared_4": comparison[
            "final_novel_accuracy_approx_95ci_low"
        ] > 0.0,
        "evictions_reduced_by_50_percent": shared_8_evictions <= 0.50 * shared_4_evictions,
    }


def _control_gate(result: dict[str, Any]) -> dict[str, bool]:
    comparison = result["paired"][
        "dual_shared_8_minus_identity_only_shared_4"
    ]
    return {
        "return_noninferior_to_identity_only": comparison[
            "return_per_decision_approx_95ci_low"
        ] > -0.05,
        "stable_noninferior_to_identity_only": comparison[
            "final_stable_accuracy_approx_95ci_low"
        ] > -0.05,
        "reversed_noninferior_to_identity_only": comparison[
            "final_reversed_accuracy_approx_95ci_low"
        ] > -0.05,
        "novel_noninferior_to_identity_only": comparison[
            "final_novel_accuracy_approx_95ci_low"
        ] > -0.05,
    }


def run_confirmation(
    config: ExperimentConfig,
    *,
    seeds: int = 100,
    seed_offset: int = 92_000_000,
) -> dict[str, Any]:
    if seed_offset < 92_000_000:
        raise ValueError("confirmation seeds must start at 92,000,000 or later")
    rows: list[ExperimentMetrics] = []
    for seed in range(seed_offset, seed_offset + seeds):
        lifetime = make_lifetime(config, seed)
        for condition in CONDITIONS:
            rows.append(_run_condition(condition, config, lifetime, seed))
    comparisons = (
        ("dual_shared_8", "dual_shared_4"),
        ("dual_shared_8", "identity_only_shared_4"),
    )
    result: dict[str, Any] = {
        "experiment": "009b_capacity_confirmation",
        "status": "fresh_seed_confirmation",
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
    result["capacity_gate"] = _capacity_gate(result)
    result["control_gate"] = _control_gate(result)
    result["capacity_confirmed"] = all(result["capacity_gate"].values())
    result["competitive_with_identity_only"] = all(
        result["control_gate"].values()
    )
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 009b: shared-capacity confirmation",
        "",
        f"- Fresh seeds: {result['seed_offset']}–{result['seed_offset'] + result['seeds'] - 1}",
        "",
        "| Condition | Return | Stable | Reversed | Novel | New promotions | Revisions | Evictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['final_stable_accuracy'])} | {_pm(metrics['final_reversed_accuracy'])} | "
            f"{_pm(metrics['final_novel_accuracy'])} | {_pm(metrics['new_promotions'])} | "
            f"{_pm(metrics['revision_promotions'])} | {_pm(metrics['buffer_evictions'])} |"
        )
    lines.extend(["", "## Capacity gate", ""])
    for name, passed in result["capacity_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(["", "## Identity-only non-inferiority gate", ""])
    for name, passed in result["control_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Capacity confirmed: **{result['capacity_confirmed']}**",
            f"Competitive with identity-only: **{result['competitive_with_identity_only']}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    (output_dir / "experiment_009b_confirmation_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "experiment_009b_confirmation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=92_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    result = run_confirmation(
        ExperimentConfig(), seeds=args.seeds, seed_offset=args.seed_offset
    )
    write_report(result, args.output_dir)
    print(json.dumps(_json_safe({
        "aggregate": result["aggregate"],
        "paired": result["paired"],
        "capacity_gate": result["capacity_gate"],
        "control_gate": result["control_gate"],
        "capacity_confirmed": result["capacity_confirmed"],
        "competitive_with_identity_only": result["competitive_with_identity_only"],
    }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
