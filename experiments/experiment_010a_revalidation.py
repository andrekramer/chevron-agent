"""Experiment 010a: fresh-seed audit of promotion-time identity revalidation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
from typing import Any

from experiments.experiment_004_reward_memory import _json_safe, _mean_sd
from experiments.experiment_010_retrospective_policy import (
    ExperimentConfig,
    ExperimentMetrics,
    make_lifetime,
    run_lifetime,
)


CONDITIONS = (
    "protected_original",
    "protected_revalidated",
)

DISPLAY_NAMES = {
    "protected_original": "Original protected Chevron",
    "protected_revalidated": "Revalidated protected Chevron",
}


def _run_condition(
    label: str,
    config: ExperimentConfig,
    lifetime: Any,
    seed: int,
) -> ExperimentMetrics:
    if label not in CONDITIONS:
        raise ValueError(f"unknown revalidation condition {label}")
    metric = run_lifetime(
        "retrospective_protected",
        config,
        lifetime,
        seed,
        revalidate_identity_promotion=(label == "protected_revalidated"),
    )
    return replace(metric, condition=label)


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


def _paired(rows: list[ExperimentMetrics]) -> dict[str, float]:
    by_key = {(row.condition, row.seed): row for row in rows}
    seeds = sorted({row.seed for row in rows})
    output: dict[str, float] = {}
    for field in (
        "return_per_decision",
        "clean_accuracy",
        "retention_accuracy",
        "reversed_probe_accuracy",
        "novel_probe_accuracy",
        "new_promotions",
        "duplicate_allocations",
        "identity_reconciliations",
    ):
        values = [
            getattr(by_key[("protected_revalidated", seed)], field)
            - getattr(by_key[("protected_original", seed)], field)
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


def _audit_gate(result: dict[str, Any]) -> dict[str, bool]:
    original = result["aggregate"]["protected_original"]
    revalidated = result["aggregate"]["protected_revalidated"]
    paired = result["paired"]["protected_revalidated_minus_original"]
    return {
        "zero_duplicate_allocations": revalidated["duplicate_allocations"]["mean"]
        == 0.0,
        "zero_established_overwrites": revalidated["established_overwrites"]["mean"]
        == 0.0,
        "zero_under_supported_writes": revalidated["under_supported_writes"]["mean"]
        == 0.0,
        "retention_at_least_0.90": revalidated["retention_accuracy"]["mean"]
        >= 0.90,
        "reversed_probe_at_least_0.75": revalidated["reversed_probe_accuracy"]["mean"]
        >= 0.75,
        "novel_probe_at_least_0.75": revalidated["novel_probe_accuracy"]["mean"]
        >= 0.75,
        "new_promotions_at_least_3": revalidated["new_promotions"]["mean"] >= 3.0,
        "unique_revisions_at_least_3": revalidated["unique_revision_categories"]["mean"]
        >= 3.0,
        "return_noninferior": paired["return_per_decision_approx_95ci_low"]
        > -0.01,
        "clean_accuracy_noninferior": paired["clean_accuracy_approx_95ci_low"]
        > -0.01,
        "retention_noninferior": paired["retention_accuracy_approx_95ci_low"]
        > -0.01,
        "reversed_probe_noninferior": paired[
            "reversed_probe_accuracy_approx_95ci_low"
        ]
        > -0.01,
        "novel_probe_noninferior": paired["novel_probe_accuracy_approx_95ci_low"]
        > -0.01,
        "reconciliations_cover_original_duplicates": revalidated[
            "identity_reconciliations"
        ]["mean"]
        >= original["duplicate_allocations"]["mean"],
    }


def run_audit(
    config: ExperimentConfig,
    *,
    seeds: int = 100,
    seed_offset: int = 102_000_000,
) -> dict[str, Any]:
    if seed_offset < 102_000_000:
        raise ValueError("audit seeds must start at 102,000,000 or later")
    rows: list[ExperimentMetrics] = []
    for seed in range(seed_offset, seed_offset + seeds):
        lifetime = make_lifetime(config, seed)
        for condition in CONDITIONS:
            rows.append(_run_condition(condition, config, lifetime, seed))
    result: dict[str, Any] = {
        "experiment": "010a_preconsolidation_revalidation",
        "status": "fresh_seed_correction_audit",
        "config": asdict(config),
        "seeds": seeds,
        "seed_offset": seed_offset,
        "aggregate": _aggregate(rows),
        "paired": {
            "protected_revalidated_minus_original": _paired(rows),
        },
        "individual": [asdict(row) for row in rows],
    }
    result["audit_gate"] = _audit_gate(result)
    result["correction_passed"] = all(result["audit_gate"].values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 010a: pre-consolidation identity revalidation",
        "",
        f"- Fresh seeds: {result['seed_offset']}–{result['seed_offset'] + result['seeds'] - 1}",
        "- Changed mechanism: promotion-time identity revalidation only",
        "",
        "| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | Duplicates | Reconciliations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['clean_accuracy'])} | {_pm(metrics['retention_accuracy'])} | "
            f"{_pm(metrics['reversed_probe_accuracy'])} | "
            f"{_pm(metrics['novel_probe_accuracy'])} | "
            f"{_pm(metrics['duplicate_allocations'])} | "
            f"{_pm(metrics['identity_reconciliations'])} |"
        )
    lines.extend(["", "## Frozen audit gate", ""])
    for name, passed in result["audit_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Correction passed: **{result['correction_passed']}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    (output_dir / "experiment_010a_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "experiment_010a_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=102_000_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
    )
    args = parser.parse_args()
    result = run_audit(
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
                    "audit_gate": result["audit_gate"],
                    "correction_passed": result["correction_passed"],
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
