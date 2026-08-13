"""Experiment 009a: distinguish provisional routing from total capacity."""

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
    "dual_split_2_2",
    "dual_shared_8",
    "dual_split_4_4",
    "identity_only_shared_4",
)

DISPLAY_NAMES = {
    "dual_shared_4": "Dual shared 4",
    "dual_split_2_2": "Dual split 2+2",
    "dual_shared_8": "Dual shared 8",
    "dual_split_4_4": "Dual split 4+4",
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
    elif label == "dual_split_2_2":
        metric = run_lifetime(
            "dual_buffer",
            config,
            lifetime,
            seed,
            buffer_layout="split",
            identity_capacity=2,
            policy_capacity=2,
        )
    elif label == "dual_shared_8":
        metric = run_lifetime(
            "dual_buffer", config, lifetime, seed, shared_capacity=8
        )
    elif label == "dual_split_4_4":
        metric = run_lifetime(
            "dual_buffer",
            config,
            lifetime,
            seed,
            buffer_layout="split",
            identity_capacity=4,
            policy_capacity=4,
        )
    elif label == "identity_only_shared_4":
        metric = run_lifetime(
            "identity_only_buffer", config, lifetime, seed, shared_capacity=4
        )
    else:
        raise ValueError(f"unknown condition {label}")
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
        output[f"{field}_approx_95ci_low"] = mean - half
        output[f"{field}_approx_95ci_high"] = mean + half
        output[f"{field}_wins"] = sum(value > 0.0 for value in values)
    return output


def _primary_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["dual_split_4_4"]
    versus_shared = result["paired"]["dual_split_4_4_minus_dual_shared_8"]
    shared_evictions = result["aggregate"]["dual_shared_8"]["buffer_evictions"]["mean"]
    split_evictions = metrics["buffer_evictions"]["mean"]
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
        "return_better_than_shared_8": versus_shared[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "novel_better_than_shared_8": versus_shared[
            "final_novel_accuracy_approx_95ci_low"
        ] > 0.0,
        "evictions_reduced_by_25_percent": split_evictions <= 0.75 * shared_evictions,
    }


def _secondary_diagnostics(result: dict[str, Any]) -> dict[str, bool]:
    capacity = result["paired"]["dual_shared_8_minus_dual_shared_4"]
    equal_budget = result["paired"]["dual_split_2_2_minus_dual_shared_4"]
    return {
        "shared_8_return_better_than_shared_4": capacity[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "shared_8_novel_better_than_shared_4": capacity[
            "final_novel_accuracy_approx_95ci_low"
        ] > 0.0,
        "split_2_2_return_noninferior_to_shared_4": equal_budget[
            "return_per_decision_approx_95ci_low"
        ] > -0.05,
        "split_2_2_novel_noninferior_to_shared_4": equal_budget[
            "final_novel_accuracy_approx_95ci_low"
        ] > -0.05,
    }


def run_experiment(
    config: ExperimentConfig, *, seeds: int, seed_offset: int
) -> dict[str, Any]:
    rows: list[ExperimentMetrics] = []
    for seed in range(seed_offset, seed_offset + seeds):
        lifetime = make_lifetime(config, seed)
        for condition in CONDITIONS:
            rows.append(_run_condition(condition, config, lifetime, seed))
    comparisons = (
        ("dual_split_4_4", "dual_shared_8"),
        ("dual_shared_8", "dual_shared_4"),
        ("dual_split_2_2", "dual_shared_4"),
        ("dual_split_4_4", "identity_only_shared_4"),
    )
    result: dict[str, Any] = {
        "experiment": "009a_split_queues",
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
    result["primary_gate"] = _primary_gate(result)
    result["secondary_diagnostics"] = _secondary_diagnostics(result)
    result["confirmation_triggered"] = all(result["primary_gate"].values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Experiment 009a: {label} split-queue diagnostic",
        "",
        f"- Seeds: {result['seed_offset']}–{result['seed_offset'] + result['seeds'] - 1}",
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
    lines.extend(["", "## Primary routing gate", ""])
    for name, passed in result["primary_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(["", "## Secondary diagnostics", ""])
    for name, passed in result["secondary_diagnostics"].items():
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
    stem = f"experiment_009a_{label}"
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
        seed_offset = 92_000_000 if args.seed_offset is None else args.seed_offset
        if seed_offset < 92_000_000:
            raise ValueError("confirmation seeds must start at 92,000,000 or later")
    else:
        seeds = 20 if args.seeds is None else args.seeds
        seed_offset = 90_000_000 if args.seed_offset is None else args.seed_offset
    result = run_experiment(ExperimentConfig(), seeds=seeds, seed_offset=seed_offset)
    write_report(result, args.output_dir, args.label)
    print(json.dumps(_json_safe({
        "aggregate": result["aggregate"],
        "paired": result["paired"],
        "primary_gate": result["primary_gate"],
        "secondary_diagnostics": result["secondary_diagnostics"],
        "confirmation_triggered": result["confirmation_triggered"],
    }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
