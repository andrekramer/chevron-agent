"""Experiment 005: delayed reward with retrospective assent credit.

The environment and protected online-memory rules are inherited from
Experiment 004.  The new signal asks whether admitted memory support predicted
the delayed outcome of the action actually selected.  No latent context,
correct-action, compatibility, or target-slot labels enter training.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch

from chevron_agent import (
    ProjectedBilinearNullAttention,
    ProjectedCosineAssent,
)
from experiments.experiment_004_reward_memory import (
    ExperimentConfig as BaseConfig,
    LifetimeMetrics,
    _json_safe,
    _mean_sd,
    make_lifetime,
    run_lifetime,
)


CONDITIONS = (
    "content_attention_buffer",
    "bilinear_retrospective_buffer",
    "chevron_retrospective_buffer",
    "chevron_policy_only_buffer",
    "chevron_retrospective_immediate",
    "chevron_retrospective_coupled_write",
)

DISPLAY_NAMES = {
    "content_attention_buffer": "Content attention + buffer",
    "bilinear_retrospective_buffer": "Bilinear null attention + retrospective",
    "chevron_retrospective_buffer": "Chevron + retrospective + buffer",
    "chevron_policy_only_buffer": "Chevron + policy only + buffer",
    "chevron_retrospective_immediate": "Chevron + retrospective + immediate",
    "chevron_retrospective_coupled_write": "Chevron + retrospective + coupled write",
}


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    buffer_capacity: int = 4
    retrospective_loss_weight: float = 1.0


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _train_seed(
    config: ExperimentConfig,
    seed: int,
) -> tuple[
    ProjectedCosineAssent,
    ProjectedCosineAssent,
    ProjectedBilinearNullAttention,
    dict[str, list[float]],
]:
    torch.manual_seed(30_000 + seed)
    retrospective = ProjectedCosineAssent(
        config.content_dim,
        config.content_dim,
        config.comparison_dim,
        initial_threshold=0.25,
        initial_slope=8.0,
    )
    policy_only = copy.deepcopy(retrospective)
    bilinear = ProjectedBilinearNullAttention(
        config.content_dim,
        config.content_dim,
        config.comparison_dim,
    )
    counts = {
        _parameter_count(retrospective),
        _parameter_count(policy_only),
        _parameter_count(bilinear),
    }
    if counts != {314}:
        raise RuntimeError(f"expected matched 314-parameter models, got {counts}")

    models: dict[str, torch.nn.Module] = {
        "retrospective": retrospective,
        "policy_only": policy_only,
        "bilinear": bilinear,
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        for name, model in models.items()
    }
    losses = {name: [] for name in models}
    policy_config = replace(config, retrospective_loss_weight=0.0)

    for lifetime_index in range(config.training_lifetimes):
        lifetime_seed = 3_000_000 + 10_000 * seed + lifetime_index
        lifetime = make_lifetime(config, lifetime_seed)
        runs = (
            ("retrospective", "chevron_buffer", retrospective, config),
            ("policy_only", "chevron_buffer", policy_only, policy_config),
            ("bilinear", "bilinear_buffer", bilinear, config),
        )
        for name, condition, model, run_config in runs:
            _, loss = run_lifetime(
                condition,
                run_config,
                lifetime,
                model=model,
                training=True,
                training_seed=seed,
                lifetime_seed=lifetime_seed,
            )
            assert loss is not None
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            losses[name].append(float(loss.detach()))
    return retrospective, policy_only, bilinear, losses


def _aggregate(
    results: list[LifetimeMetrics],
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
        for condition in CONDITIONS
    }


def _paired(results: list[LifetimeMetrics]) -> dict[str, dict[str, float]]:
    by_key = {
        (row.condition, row.training_seed, row.lifetime_seed): row
        for row in results
    }
    keys = sorted({(row.training_seed, row.lifetime_seed) for row in results})
    comparisons = (
        ("chevron_retrospective_buffer", "content_attention_buffer"),
        ("chevron_retrospective_buffer", "bilinear_retrospective_buffer"),
        ("chevron_retrospective_buffer", "chevron_policy_only_buffer"),
        ("chevron_retrospective_buffer", "chevron_retrospective_immediate"),
        (
            "chevron_retrospective_buffer",
            "chevron_retrospective_coupled_write",
        ),
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


def _development_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["chevron_retrospective_buffer"]
    paired = result["paired"][
        "chevron_retrospective_buffer_minus_chevron_policy_only_buffer"
    ]
    return {
        "old_accuracy_at_least_0.90": metrics["final_old_accuracy"]["mean"] >= 0.90,
        "new_accuracy_at_least_0.75": metrics["final_new_accuracy"]["mean"] >= 0.75,
        "new_probe_at_least_0.75": metrics["new_probe_accuracy"]["mean"] >= 0.75,
        "promotions_at_least_3": metrics["promotions"]["mean"] >= 3.0,
        "q_calibration_at_least_0.15": metrics["residual_calibration"]["mean"] >= 0.15,
        "policy_only_return_ci_low_above_0": paired[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "no_premature_writes": metrics["premature_write_rate"]["mean"] == 0.0,
        "positive_read_write_margin": metrics["read_write_margin"]["mean"] > 0.0,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    results: list[LifetimeMetrics] = []
    training_records: list[dict[str, Any]] = []
    for seed in range(config.seed_offset, config.seed_offset + config.training_seeds):
        retrospective, policy_only, bilinear, losses = _train_seed(config, seed)
        training_records.append(
            {
                "seed": seed,
                "retrospective_initial_loss": losses["retrospective"][0],
                "retrospective_final_loss": losses["retrospective"][-1],
                "policy_only_initial_loss": losses["policy_only"][0],
                "policy_only_final_loss": losses["policy_only"][-1],
                "bilinear_initial_loss": losses["bilinear"][0],
                "bilinear_final_loss": losses["bilinear"][-1],
                "retrospective_threshold": float(retrospective.threshold.detach()),
                "retrospective_slope": float(retrospective.slope.detach()),
                "bilinear_temperature": float(bilinear.temperature.detach()),
            }
        )
        frozen = {
            "bilinear_retrospective_buffer": copy.deepcopy(bilinear).eval(),
            "chevron_retrospective_buffer": copy.deepcopy(retrospective).eval(),
            "chevron_policy_only_buffer": copy.deepcopy(policy_only).eval(),
            "chevron_retrospective_immediate": copy.deepcopy(retrospective).eval(),
            "chevron_retrospective_coupled_write": copy.deepcopy(retrospective).eval(),
        }
        runner_conditions = {
            "content_attention_buffer": "content_attention_buffer",
            "bilinear_retrospective_buffer": "bilinear_buffer",
            "chevron_retrospective_buffer": "chevron_buffer",
            "chevron_policy_only_buffer": "chevron_buffer",
            "chevron_retrospective_immediate": "chevron_immediate",
            "chevron_retrospective_coupled_write": "chevron_coupled_write",
        }
        for evaluation_index in range(config.evaluation_lifetimes):
            lifetime_seed = 30_000_000 + 10_000 * seed + evaluation_index
            lifetime = make_lifetime(config, lifetime_seed)
            for label in CONDITIONS:
                model = None if label == "content_attention_buffer" else frozen[label]
                metrics, _ = run_lifetime(
                    runner_conditions[label],
                    config,
                    lifetime,
                    model=model,
                    training=False,
                    training_seed=seed,
                    lifetime_seed=lifetime_seed,
                )
                results.append(replace(metrics, condition=label))

    result: dict[str, Any] = {
        "experiment": "005_retrospective_assent",
        "config": asdict(config),
        "training": training_records,
        "aggregate": _aggregate(results),
        "paired": _paired(results),
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
    lines = [
        f"# Experiment 005: {label} retrospective assent",
        "",
        f"- Training seeds: {config['seed_offset']}–{config['seed_offset'] + config['training_seeds'] - 1}",
        f"- Training lifetimes per seed: {config['training_lifetimes']}",
        f"- Evaluation lifetimes per seed: {config['evaluation_lifetimes']}",
        f"- Provisional buffer capacity: {config['buffer_capacity']}",
        f"- Retrospective loss weight: {config['retrospective_loss_weight']}",
        "- Learned model parameters: 314 each",
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
    lines.extend(
        [
            "",
            "## Frozen confirmation gate",
            "",
        ]
    )
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
            "This remains a delayed contextual-bandit experiment. It does not establish",
            "spatial-game performance or a persistent agent self.",
            "",
        ]
    )
    stem = f"experiment_005_{label}"
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
    parser.add_argument("--training-lifetimes", type=int, default=None)
    parser.add_argument("--evaluation-lifetimes", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results")
    )
    args = parser.parse_args()
    defaults = ExperimentConfig()
    if args.label == "confirmation":
        config = ExperimentConfig(
            training_seeds=10 if args.seeds is None else args.seeds,
            seed_offset=400 if args.seed_offset is None else args.seed_offset,
            training_lifetimes=(
                defaults.training_lifetimes
                if args.training_lifetimes is None
                else args.training_lifetimes
            ),
            evaluation_lifetimes=(
                20
                if args.evaluation_lifetimes is None
                else args.evaluation_lifetimes
            ),
        )
        if config.seed_offset < 400:
            raise ValueError("confirmation seed offset must be at least 400")
    else:
        config = ExperimentConfig(
            training_seeds=(
                defaults.training_seeds if args.seeds is None else args.seeds
            ),
            seed_offset=(
                defaults.seed_offset if args.seed_offset is None else args.seed_offset
            ),
            training_lifetimes=(
                defaults.training_lifetimes
                if args.training_lifetimes is None
                else args.training_lifetimes
            ),
            evaluation_lifetimes=(
                defaults.evaluation_lifetimes
                if args.evaluation_lifetimes is None
                else args.evaluation_lifetimes
            ),
        )
    result = run_experiment(config)
    write_report(result, args.output_dir, args.label)
    summary = {
        "training": result["training"],
        "aggregate": result["aggregate"],
        "development_gate": result["development_gate"],
        "confirmation_triggered": result["confirmation_triggered"],
    }
    print(json.dumps(_json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
