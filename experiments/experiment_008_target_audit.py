"""Post-development audit of Experiment 008's ideal consequence target."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
from typing import Any, Callable

import torch
from torch import Tensor

from experiments.experiment_004_reward_memory import (
    LifetimeMetrics,
    _json_safe,
    run_lifetime,
)
from experiments.experiment_005a_geometric_gate import _aggregate, _paired
from experiments.experiment_006_predictive_geometry import _transform_lifetime
from experiments.experiment_008_consequence_geometry import (
    ExperimentConfig,
    FixedAffordanceWorld,
    make_affordance_lifetime,
)


CONDITIONS = ("oracle_latent_chevron", "oracle_consequence_chevron")


def _signature_transform(world: FixedAffordanceWorld) -> Callable[[Tensor], Tensor]:
    def transform(value: Tensor) -> Tensor:
        if value.ndim == 1:
            return world.consequence_signature(value.unsqueeze(0)).squeeze(0)
        return world.consequence_signature(value)

    return transform


def run_target_audit(config: ExperimentConfig) -> dict[str, Any]:
    world = FixedAffordanceWorld(
        config.action_dim,
        config.content_dim,
        affordance_seed=config.affordance_seed,
        dynamics_seed=config.dynamics_seed,
        discount=config.consequence_discount,
        reward_scale=config.reward_scale,
    ).eval()
    transform = _signature_transform(world)
    results: list[LifetimeMetrics] = []
    competing_similarities: list[float] = []
    matching_similarities: list[float] = []
    threshold = config.standard_similarity_threshold

    for seed in range(config.seed_offset, config.seed_offset + config.training_seeds):
        for evaluation_index in range(config.evaluation_lifetimes):
            lifetime_seed = 52_000_000 + 10_000 * seed + evaluation_index
            latent = make_affordance_lifetime(config, world, lifetime_seed)
            consequence = _transform_lifetime(latent, transform)
            for label, lifetime in (
                ("oracle_latent_chevron", latent),
                ("oracle_consequence_chevron", consequence),
            ):
                metrics, _ = run_lifetime(
                    "geometric_chevron_buffer",
                    config,
                    lifetime,
                    model=None,
                    training=False,
                    training_seed=seed,
                    lifetime_seed=lifetime_seed,
                )
                results.append(replace(metrics, condition=label))

            prototype_signature = world.consequence_signature(latent.prototypes)
            for family in range(config.groups):
                indices = (3 * family, 3 * family + 1, 3 * family + 2)
                for first, second in ((0, 1), (0, 2), (1, 2)):
                    competing_similarities.append(
                        float(
                            prototype_signature[indices[first]]
                            @ prototype_signature[indices[second]]
                        )
                    )
            for event in latent.events:
                event_signature = transform(event.observation.evidence)
                matching_similarities.append(
                    float(event_signature @ prototype_signature[event.category])
                )

    competing_above = statistics.fmean(
        float(value >= threshold) for value in competing_similarities
    )
    matching_below = statistics.fmean(
        float(value < threshold) for value in matching_similarities
    )
    return {
        "experiment": "008_consequence_target_audit",
        "status": "post_development_causal_diagnostic",
        "config": asdict(config),
        "aggregate": _aggregate(results, CONDITIONS),
        "paired": _paired(
            results,
            (("oracle_consequence_chevron", "oracle_latent_chevron"),),
        ),
        "target_geometry": {
            "assent_similarity_threshold": threshold,
            "competing_within_family_mean_similarity": statistics.fmean(
                competing_similarities
            ),
            "competing_within_family_max_similarity": max(
                competing_similarities
            ),
            "competing_within_family_fraction_above_threshold": competing_above,
            "matching_event_mean_similarity": statistics.fmean(
                matching_similarities
            ),
            "matching_event_fraction_below_threshold": matching_below,
        },
        "individual": [asdict(row) for row in results],
    }


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_audit(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = result["aggregate"]
    geometry = result["target_geometry"]
    lines = [
        "# Experiment 008: ideal consequence-target audit",
        "",
        "This is a post-development causal diagnostic, not a reopened confirmation gate.",
        "It asks whether the exact target geometry would support the frozen Chevron mechanism.",
        "",
        "| Geometry | Return | Final old | Final new | New probe | q calibration | Promotions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, name in (
        ("oracle_latent_chevron", "Oracle latent"),
        ("oracle_consequence_chevron", "Oracle consequence signature"),
    ):
        metrics = aggregate[condition]
        lines.append(
            f"| {name} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['final_old_accuracy'])} | {_pm(metrics['final_new_accuracy'])} | "
            f"{_pm(metrics['new_probe_accuracy'])} | {_pm(metrics['residual_calibration'])} | "
            f"{_pm(metrics['promotions'])} |"
        )
    lines.extend(
        [
            "",
            "## Geometry audit",
            "",
            f"- Competing within-family mean similarity: {geometry['competing_within_family_mean_similarity']:.3f}",
            f"- Competing within-family maximum similarity: {geometry['competing_within_family_max_similarity']:.3f}",
            f"- Competing contexts above the 0.62 assent boundary: {100.0 * geometry['competing_within_family_fraction_above_threshold']:.1f}%",
            f"- Matching noisy-event mean similarity: {geometry['matching_event_mean_similarity']:.3f}",
            f"- Matching noisy events below the assent boundary: {100.0 * geometry['matching_event_fraction_below_threshold']:.1f}%",
            "",
            "The exact consequence signature is therefore not a sufficient memory-identity geometry.",
            "It sometimes treats distinct contexts as mutually assenting and sometimes moves a noisy",
            "observation too far from its own retained prototype.",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    (output_dir / "experiment_008_target_audit_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "experiment_008_target_audit_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    result = run_target_audit(ExperimentConfig())
    write_audit(result, args.output_dir)
    print(json.dumps(_json_safe(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
