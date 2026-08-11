"""Experiment 005a: parameter-free geometric Chevron gate isolation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

from experiments.experiment_004_reward_memory import (
    ExperimentConfig,
    LifetimeMetrics,
    _json_safe,
    _mean_sd,
    make_lifetime,
    run_lifetime,
)


CONDITIONS = (
    "content_attention_buffer",
    "chevron_retrospective_buffer",
    "geometric_chevron_buffer",
    "geometric_chevron_immediate",
    "geometric_chevron_coupled_write",
)

CONFIRMATION_CONDITIONS = (
    "content_attention_buffer",
    "geometric_chevron_buffer",
    "geometric_chevron_immediate",
    "geometric_chevron_coupled_write",
)

DISPLAY_NAMES = {
    "content_attention_buffer": "Content attention + buffer",
    "chevron_retrospective_buffer": "Learned Chevron + retrospective + buffer",
    "geometric_chevron_buffer": "Geometric Chevron + buffer",
    "geometric_chevron_immediate": "Geometric Chevron + immediate",
    "geometric_chevron_coupled_write": "Geometric Chevron + coupled write",
}


def _metric_from_json(row: dict[str, Any]) -> LifetimeMetrics:
    values: dict[str, Any] = {}
    for name in LifetimeMetrics.__dataclass_fields__:
        value = row[name]
        values[name] = float("nan") if value is None else value
    return LifetimeMetrics(**values)


def _aggregate(
    results: list[LifetimeMetrics],
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, dict[str, dict[str, float]]]:
    metric_fields = [
        name
        for name in LifetimeMetrics.__dataclass_fields__
        if name not in {"condition", "training_seed", "lifetime_seed"}
    ]
    return {
        condition: {
            field: _mean_sd(
                [
                    float(getattr(row, field))
                    for row in results
                    if row.condition == condition
                ]
            )
            for field in metric_fields
        }
        for condition in conditions
    }


def _paired(
    results: list[LifetimeMetrics],
    comparisons: tuple[tuple[str, str], ...] | None = None,
) -> dict[str, dict[str, float]]:
    by_key = {
        (row.condition, row.training_seed, row.lifetime_seed): row
        for row in results
    }
    keys = sorted({(row.training_seed, row.lifetime_seed) for row in results})
    if comparisons is None:
        comparisons = (
            ("geometric_chevron_buffer", "content_attention_buffer"),
            ("geometric_chevron_buffer", "chevron_retrospective_buffer"),
            ("geometric_chevron_buffer", "geometric_chevron_immediate"),
            ("geometric_chevron_buffer", "geometric_chevron_coupled_write"),
        )
    output: dict[str, dict[str, float]] = {}
    for left, right in comparisons:
        label = f"{left}_minus_{right}"
        output[label] = {}
        for metric in (
            "return_per_decision",
            "final_old_accuracy",
            "final_new_accuracy",
            "residual_calibration",
        ):
            values = [
                float(
                    getattr(by_key[(left, *key)], metric)
                    - getattr(by_key[(right, *key)], metric)
                )
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


def _diagnostic_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["geometric_chevron_buffer"]
    comparison = result["paired"][
        "geometric_chevron_buffer_minus_content_attention_buffer"
    ]
    return {
        "old_accuracy_at_least_0.95": metrics["final_old_accuracy"]["mean"] >= 0.95,
        "new_accuracy_at_least_0.75": metrics["final_new_accuracy"]["mean"] >= 0.75,
        "new_probe_at_least_0.75": metrics["new_probe_accuracy"]["mean"] >= 0.75,
        "q_calibration_at_least_0.15": metrics["residual_calibration"]["mean"] >= 0.15,
        "promotions_at_least_3": metrics["promotions"]["mean"] >= 3.0,
        "no_premature_writes": metrics["premature_write_rate"]["mean"] == 0.0,
        "positive_read_write_margin": metrics["read_write_margin"]["mean"] > 0.0,
        "return_within_0.05_of_content": comparison["return_per_decision_mean"] >= -0.05,
    }


def run_experiment(source_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("experiment") != "005_retrospective_assent":
        raise ValueError("source must be Experiment 005 results")
    config = ExperimentConfig(**source["config"])
    retained_conditions = {
        "content_attention_buffer",
        "chevron_retrospective_buffer",
    }
    results = [
        _metric_from_json(row)
        for row in source["individual"]
        if row["condition"] in retained_conditions
    ]
    keys = sorted(
        {
            (row.training_seed, row.lifetime_seed)
            for row in results
            if row.condition == "content_attention_buffer"
        }
    )
    for training_seed, lifetime_seed in keys:
        lifetime = make_lifetime(config, lifetime_seed)
        for condition in (
            "geometric_chevron_buffer",
            "geometric_chevron_immediate",
            "geometric_chevron_coupled_write",
        ):
            metrics, _ = run_lifetime(
                condition,
                config,
                lifetime,
                model=None,
                training=False,
                training_seed=training_seed,
                lifetime_seed=lifetime_seed,
            )
            results.append(metrics)

    result: dict[str, Any] = {
        "experiment": "005a_geometric_gate",
        "source_results": str(source_path),
        "config": asdict(config),
        "geometric_gate": {
            "similarity_threshold": config.standard_similarity_threshold,
            "mismatch_threshold": 0.5
            * (1.0 - config.standard_similarity_threshold),
            "slope": 40.0,
            "write_threshold_margin": config.write_threshold_margin,
        },
        "aggregate": _aggregate(results),
        "paired": _paired(results),
        "individual": [asdict(row) for row in results],
    }
    diagnostic = _diagnostic_gate(result)
    result["diagnostic_gate"] = diagnostic
    result["geometric_gate_viable"] = all(diagnostic.values())
    return result


def _confirmation_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["geometric_chevron_buffer"]
    versus_content = result["paired"][
        "geometric_chevron_buffer_minus_content_attention_buffer"
    ]
    versus_immediate = result["paired"][
        "geometric_chevron_buffer_minus_geometric_chevron_immediate"
    ]
    return {
        "old_accuracy_at_least_0.95": metrics["final_old_accuracy"]["mean"] >= 0.95,
        "new_accuracy_at_least_0.75": metrics["final_new_accuracy"]["mean"] >= 0.75,
        "new_probe_at_least_0.75": metrics["new_probe_accuracy"]["mean"] >= 0.75,
        "q_calibration_at_least_0.15": metrics["residual_calibration"]["mean"] >= 0.15,
        "promotions_at_least_3": metrics["promotions"]["mean"] >= 3.0,
        "no_premature_writes": metrics["premature_write_rate"]["mean"] == 0.0,
        "positive_read_write_margin": metrics["read_write_margin"]["mean"] > 0.0,
        "buffer_return_better_than_immediate": versus_immediate[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "buffer_novel_better_than_immediate": versus_immediate[
            "final_new_accuracy_approx_95ci_low"
        ] > 0.0,
        "return_noninferior_to_content": versus_content[
            "return_per_decision_approx_95ci_low"
        ] > -0.05,
        "novel_noninferior_to_content": versus_content[
            "final_new_accuracy_approx_95ci_low"
        ] > -0.05,
    }


def run_confirmation(
    source_path: Path,
    *,
    lifetime_seed_offset: int,
    lifetimes: int,
) -> dict[str, Any]:
    if lifetimes <= 1:
        raise ValueError("confirmation requires at least two lifetimes")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = ExperimentConfig(**source["config"])
    results: list[LifetimeMetrics] = []
    runner_conditions = (
        "content_attention_buffer",
        "geometric_chevron_buffer",
        "geometric_chevron_immediate",
        "geometric_chevron_coupled_write",
    )
    for index in range(lifetimes):
        lifetime_seed = lifetime_seed_offset + index
        lifetime = make_lifetime(config, lifetime_seed)
        for condition in runner_conditions:
            metrics, _ = run_lifetime(
                condition,
                config,
                lifetime,
                model=None,
                training=False,
                training_seed=-1,
                lifetime_seed=lifetime_seed,
            )
            results.append(metrics)
    comparisons = (
        ("geometric_chevron_buffer", "content_attention_buffer"),
        ("geometric_chevron_buffer", "geometric_chevron_immediate"),
        ("geometric_chevron_buffer", "geometric_chevron_coupled_write"),
    )
    result: dict[str, Any] = {
        "experiment": "005a_geometric_gate_confirmation",
        "source_results": str(source_path),
        "lifetime_seed_offset": lifetime_seed_offset,
        "lifetimes": lifetimes,
        "config": asdict(config),
        "aggregate": _aggregate(results, CONFIRMATION_CONDITIONS),
        "paired": _paired(results, comparisons),
        "individual": [asdict(row) for row in results],
    }
    gate = _confirmation_gate(result)
    result["confirmation_gate"] = gate
    result["confirmed"] = all(gate.values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = result["aggregate"]
    gate = result["geometric_gate"]
    lines = [
        "# Experiment 005a: geometric gate isolation",
        "",
        "- Evaluation: exact 20 Experiment 005 development lifetimes",
        "- Additional training: none",
        f"- Cosine similarity threshold: {gate['similarity_threshold']}",
        f"- Half-cosine mismatch threshold: {gate['mismatch_threshold']}",
        f"- Gate slope: {gate['slope']}",
        f"- Write threshold margin: {gate['write_threshold_margin']}",
        "",
        "| Method | Return | Final old | Final new | Old probe | New probe | q calibration | Promotions | N drift | Premature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = aggregate[condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['final_old_accuracy'])} | {_pm(metrics['final_new_accuracy'])} | "
            f"{_pm(metrics['old_probe_accuracy'])} | {_pm(metrics['new_probe_accuracy'])} | "
            f"{_pm(metrics['residual_calibration'])} | {_pm(metrics['promotions'])} | "
            f"{_pm(metrics['established_drift'])} | {_pm(metrics['premature_write_rate'])} |"
        )
    lines.extend(["", "## Frozen diagnostic gate", ""])
    for name, passed in result["diagnostic_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Geometric gate viable: **{result['geometric_gate_viable']}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    (output_dir / "experiment_005a_geometric_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "experiment_005a_geometric_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_confirmation_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = result["aggregate"]
    lines = [
        "# Experiment 005a: fresh-seed geometric confirmation",
        "",
        f"- Lifetime seeds: {result['lifetime_seed_offset']}–{result['lifetime_seed_offset'] + result['lifetimes'] - 1}",
        f"- Fresh lifetimes: {result['lifetimes']}",
        "- Additional training: none",
        "- Formula and thresholds: frozen from the development diagnostic",
        "",
        "| Method | Return | Final old | Final new | New probe | q calibration | Promotions | N drift | Premature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONFIRMATION_CONDITIONS:
        metrics = aggregate[condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['final_old_accuracy'])} | {_pm(metrics['final_new_accuracy'])} | "
            f"{_pm(metrics['new_probe_accuracy'])} | {_pm(metrics['residual_calibration'])} | "
            f"{_pm(metrics['promotions'])} | {_pm(metrics['established_drift'])} | "
            f"{_pm(metrics['premature_write_rate'])} |"
        )
    lines.extend(["", "## Frozen confirmation gate", ""])
    for name, passed in result["confirmation_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Confirmed: **{result['confirmed']}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    (output_dir / "experiment_005a_confirmation_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "experiment_005a_confirmation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/results/experiment_005_development_results.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results")
    )
    parser.add_argument("--fresh-confirmation", action="store_true")
    parser.add_argument("--confirmation-lifetimes", type=int, default=100)
    parser.add_argument("--confirmation-seed-offset", type=int, default=60_000_000)
    args = parser.parse_args()
    if args.fresh_confirmation:
        result = run_confirmation(
            args.source,
            lifetime_seed_offset=args.confirmation_seed_offset,
            lifetimes=args.confirmation_lifetimes,
        )
        write_confirmation_report(result, args.output_dir)
        summary = {
            "aggregate": result["aggregate"],
            "confirmation_gate": result["confirmation_gate"],
            "confirmed": result["confirmed"],
        }
    else:
        result = run_experiment(args.source)
        write_report(result, args.output_dir)
        summary = {
            "aggregate": result["aggregate"],
            "diagnostic_gate": result["diagnostic_gate"],
            "geometric_gate_viable": result["geometric_gate_viable"],
        }
    print(
        json.dumps(
            _json_safe(summary),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
