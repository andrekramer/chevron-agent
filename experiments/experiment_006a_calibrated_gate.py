"""Experiment 006a: label-free calibration of temporal Chevron geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

from experiments.experiment_004_reward_memory import (
    LifetimeMetrics,
    _json_safe,
    _mean_sd,
    make_lifetime,
    run_lifetime,
)
from experiments.experiment_005a_geometric_gate import _aggregate, _paired
from experiments.experiment_006_predictive_geometry import (
    ExperimentConfig,
    FixedNonlinearSensor,
    _transform_lifetime,
    calibrate_gate_from_temporal_pairs,
    train_encoder,
)


CONDITIONS = (
    "oracle_geometric_chevron",
    "inherited_temporal_geometric_chevron",
    "calibrated_temporal_geometric_chevron",
    "calibrated_temporal_content_attention",
    "calibrated_temporal_immediate",
)

DISPLAY_NAMES = {
    "oracle_geometric_chevron": "Oracle geometric Chevron",
    "inherited_temporal_geometric_chevron": "Temporal Chevron, inherited gate",
    "calibrated_temporal_geometric_chevron": "Temporal Chevron, calibrated gate",
    "calibrated_temporal_content_attention": "Temporal content attention, calibrated",
    "calibrated_temporal_immediate": "Temporal Chevron, calibrated immediate",
}


def _calibration_aggregate(
    records: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        field: _mean_sd([record[field] for record in records])
        for field in records[0]
        if field != "seed"
    }


def _development_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["calibrated_temporal_geometric_chevron"]
    inherited = result["paired"][
        "calibrated_temporal_geometric_chevron_minus_inherited_temporal_geometric_chevron"
    ]
    immediate = result["paired"][
        "calibrated_temporal_geometric_chevron_minus_calibrated_temporal_immediate"
    ]
    oracle = result["paired"][
        "calibrated_temporal_geometric_chevron_minus_oracle_geometric_chevron"
    ]
    content = result["paired"][
        "calibrated_temporal_geometric_chevron_minus_calibrated_temporal_content_attention"
    ]
    return {
        "old_accuracy_at_least_0.95": metrics["final_old_accuracy"]["mean"] >= 0.95,
        "new_accuracy_at_least_0.75": metrics["final_new_accuracy"]["mean"] >= 0.75,
        "new_probe_at_least_0.75": metrics["new_probe_accuracy"]["mean"] >= 0.75,
        "q_calibration_at_least_0.15": metrics["residual_calibration"]["mean"] >= 0.15,
        "promotions_at_least_3": metrics["promotions"]["mean"] >= 3.0,
        "return_better_than_inherited": inherited[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "return_better_than_immediate": immediate[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "novel_better_than_immediate": immediate[
            "final_new_accuracy_approx_95ci_low"
        ] > 0.0,
        "return_noninferior_to_oracle": oracle[
            "return_per_decision_approx_95ci_low"
        ] > -0.05,
        "novel_noninferior_to_oracle": oracle[
            "final_new_accuracy_approx_95ci_low"
        ] > -0.05,
        "return_noninferior_to_content": content[
            "return_per_decision_approx_95ci_low"
        ] > -0.05,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    sensor = FixedNonlinearSensor(
        config.content_dim,
        config.sensor_hidden_dim,
        config.sensor_seed,
    ).eval()
    results: list[LifetimeMetrics] = []
    training_records: list[dict[str, float]] = []
    calibration_records: list[dict[str, float]] = []
    evaluation_base = 71_000_000 if config.seed_offset >= 700 else 41_000_000

    for seed in range(config.seed_offset, config.seed_offset + config.training_seeds):
        encoder, _, losses = train_encoder(config, sensor, seed)
        calibration = calibrate_gate_from_temporal_pairs(
            config,
            sensor,
            encoder,
            seed,
        )
        calibrated_config = replace(
            config,
            standard_similarity_threshold=calibration.similarity_threshold,
            geometric_slope=calibration.mismatch_slope,
        )
        training_records.append(
            {
                "seed": float(seed),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
            }
        )
        calibration_records.append(
            {"seed": float(seed), **asdict(calibration)}
        )
        for evaluation_index in range(config.evaluation_lifetimes):
            lifetime_seed = evaluation_base + 10_000 * seed + evaluation_index
            latent = make_lifetime(config, lifetime_seed)
            encoded = _transform_lifetime(
                latent,
                lambda value: encoder(sensor(value)),
            )
            runs = (
                (
                    "oracle_geometric_chevron",
                    "geometric_chevron_buffer",
                    latent,
                    config,
                ),
                (
                    "inherited_temporal_geometric_chevron",
                    "geometric_chevron_buffer",
                    encoded,
                    config,
                ),
                (
                    "calibrated_temporal_geometric_chevron",
                    "geometric_chevron_buffer",
                    encoded,
                    calibrated_config,
                ),
                (
                    "calibrated_temporal_content_attention",
                    "content_attention_buffer",
                    encoded,
                    calibrated_config,
                ),
                (
                    "calibrated_temporal_immediate",
                    "geometric_chevron_immediate",
                    encoded,
                    calibrated_config,
                ),
            )
            for label, condition, lifetime, run_config in runs:
                metrics, _ = run_lifetime(
                    condition,
                    run_config,
                    lifetime,
                    model=None,
                    training=False,
                    training_seed=seed,
                    lifetime_seed=lifetime_seed,
                )
                results.append(replace(metrics, condition=label))

    comparisons = (
        (
            "calibrated_temporal_geometric_chevron",
            "inherited_temporal_geometric_chevron",
        ),
        (
            "calibrated_temporal_geometric_chevron",
            "calibrated_temporal_immediate",
        ),
        (
            "calibrated_temporal_geometric_chevron",
            "oracle_geometric_chevron",
        ),
        (
            "calibrated_temporal_geometric_chevron",
            "calibrated_temporal_content_attention",
        ),
    )
    result: dict[str, Any] = {
        "experiment": "006a_calibrated_gate",
        "config": asdict(config),
        "training": training_records,
        "calibration": calibration_records,
        "calibration_aggregate": _calibration_aggregate(calibration_records),
        "aggregate": _aggregate(results, CONDITIONS),
        "paired": _paired(results, comparisons),
        "individual": [asdict(row) for row in results],
    }
    gate = _development_gate(result)
    result["development_gate"] = gate
    result["confirmation_triggered"] = all(gate.values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = result["config"]
    aggregate = result["aggregate"]
    calibration = result["calibration_aggregate"]
    lines = [
        f"# Experiment 006a: {label} self-calibrated gate",
        "",
        f"- Encoder seeds: {config['seed_offset']}–{config['seed_offset'] + config['training_seeds'] - 1}",
        f"- Evaluation lifetimes per seed: {config['evaluation_lifetimes']}",
        f"- Calibrated similarity threshold: {_pm(calibration['similarity_threshold'])}",
        f"- Calibrated mismatch slope: {_pm(calibration['mismatch_slope'])}",
        "- Calibration labels: none",
        "",
        "| Method | Return | Final old | Final new | New probe | q calibration | Promotions | N drift | Premature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = aggregate[condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['final_old_accuracy'])} | {_pm(metrics['final_new_accuracy'])} | "
            f"{_pm(metrics['new_probe_accuracy'])} | {_pm(metrics['residual_calibration'])} | "
            f"{_pm(metrics['promotions'])} | {_pm(metrics['established_drift'])} | "
            f"{_pm(metrics['premature_write_rate'])} |"
        )
    lines.extend(["", "## Frozen confirmation gate", ""])
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
    stem = f"experiment_006a_{label}"
    (output_dir / f"{stem}_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / f"{stem}_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument("--pretraining-steps", type=int, default=None)
    parser.add_argument("--evaluation-lifetimes", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results")
    )
    args = parser.parse_args()
    defaults = ExperimentConfig()
    if args.label == "confirmation":
        config = ExperimentConfig(
            training_seeds=10 if args.seeds is None else args.seeds,
            seed_offset=700 if args.seed_offset is None else args.seed_offset,
            pretraining_steps=(
                defaults.pretraining_steps
                if args.pretraining_steps is None
                else args.pretraining_steps
            ),
            evaluation_lifetimes=(
                20
                if args.evaluation_lifetimes is None
                else args.evaluation_lifetimes
            ),
        )
        if config.seed_offset < 700:
            raise ValueError("confirmation seed offset must be at least 700")
    else:
        config = ExperimentConfig(
            training_seeds=(
                defaults.training_seeds if args.seeds is None else args.seeds
            ),
            seed_offset=(
                defaults.seed_offset if args.seed_offset is None else args.seed_offset
            ),
            pretraining_steps=(
                defaults.pretraining_steps
                if args.pretraining_steps is None
                else args.pretraining_steps
            ),
            evaluation_lifetimes=(
                defaults.evaluation_lifetimes
                if args.evaluation_lifetimes is None
                else args.evaluation_lifetimes
            ),
        )
    result = run_experiment(config)
    write_report(result, args.output_dir, args.label)
    print(
        json.dumps(
            _json_safe(
                {
                    "training": result["training"],
                    "calibration_aggregate": result["calibration_aggregate"],
                    "aggregate": result["aggregate"],
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
