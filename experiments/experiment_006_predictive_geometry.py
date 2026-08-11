"""Experiment 006: learn Chevron comparison geometry from temporal pairs."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import statistics
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.experiment_004_reward_memory import (
    AgentObservation,
    AuditEvent,
    ExperimentConfig as BaseConfig,
    Lifetime,
    LifetimeMetrics,
    _json_safe,
    _mean_sd,
    make_lifetime,
    run_lifetime,
)
from experiments.experiment_005a_geometric_gate import _aggregate, _paired


CONDITIONS = (
    "oracle_geometric_chevron",
    "raw_sensor_geometric_chevron",
    "random_encoder_geometric_chevron",
    "temporal_geometric_chevron",
    "temporal_content_attention",
)

DISPLAY_NAMES = {
    "oracle_geometric_chevron": "Oracle geometric Chevron",
    "raw_sensor_geometric_chevron": "Raw-sensor geometric Chevron",
    "random_encoder_geometric_chevron": "Random-encoder geometric Chevron",
    "temporal_geometric_chevron": "Temporal-contrastive geometric Chevron",
    "temporal_content_attention": "Temporal-contrastive content attention",
}


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    training_seeds: int = 2
    seed_offset: int = 0
    evaluation_lifetimes: int = 10
    buffer_capacity: int = 4
    sensor_hidden_dim: int = 32
    sensor_seed: int = 606
    encoder_hidden_dim: int = 32
    pretraining_steps: int = 500
    pretraining_batch_size: int = 256
    temporal_view_noise: float = 0.15
    contrastive_temperature: float = 0.10
    representation_evaluation_size: int = 4096


class FixedNonlinearSensor(nn.Module):
    """A deterministic information-preserving but cosine-distorting sensor."""

    def __init__(self, dimension: int, hidden_dim: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "first_weight",
            torch.randn(hidden_dim, dimension, generator=generator)
            / dimension**0.5,
        )
        self.register_buffer(
            "first_bias",
            0.35 * torch.randn(hidden_dim, generator=generator),
        )
        self.register_buffer(
            "second_weight",
            torch.randn(dimension, hidden_dim, generator=generator)
            / hidden_dim**0.5,
        )
        self.register_buffer(
            "second_bias",
            0.15 * torch.randn(dimension, generator=generator),
        )

    def forward(self, latent: Tensor) -> Tensor:
        hidden = torch.tanh(
            2.0 * F.linear(latent, self.first_weight, self.first_bias)
        )
        observed = F.linear(hidden, self.second_weight, self.second_bias)
        return F.normalize(observed, dim=-1)


class TemporalContrastiveEncoder(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimension),
        )

    def forward(self, observation: Tensor) -> Tensor:
        return F.normalize(self.network(observation), dim=-1)


@dataclass(frozen=True)
class GateCalibration:
    similarity_threshold: float
    mismatch_slope: float
    positive_tenth_percentile: float
    negative_ninety_fifth_percentile: float


def temporal_contrastive_loss(
    first: Tensor,
    second: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired embeddings must have the same [batch, dim] shape")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = first @ second.T / temperature
    target = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (
        F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)
    )


def _unit(vector: Tensor) -> Tensor:
    return F.normalize(vector, dim=-1)


def _sample_temporal_views(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    generator: torch.Generator,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    base = _unit(
        torch.randn(batch_size, config.content_dim, generator=generator)
    )
    first = _unit(
        base
        + config.temporal_view_noise
        * torch.randn(base.shape, generator=generator)
    )
    second = _unit(
        base
        + config.temporal_view_noise
        * torch.randn(base.shape, generator=generator)
    )
    with torch.no_grad():
        return sensor(first), sensor(second)


def train_encoder(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    seed: int,
) -> tuple[TemporalContrastiveEncoder, TemporalContrastiveEncoder, list[float]]:
    torch.manual_seed(60_000 + seed)
    encoder = TemporalContrastiveEncoder(
        config.content_dim,
        config.encoder_hidden_dim,
    )
    random_encoder = copy.deepcopy(encoder).eval()
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(61_000 + seed)
    losses: list[float] = []
    encoder.train()
    for _ in range(config.pretraining_steps):
        first, second = _sample_temporal_views(
            config,
            sensor,
            generator,
            config.pretraining_batch_size,
        )
        loss = temporal_contrastive_loss(
            encoder(first),
            encoder(second),
            temperature=config.contrastive_temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return encoder.eval(), random_encoder, losses


def _transform_lifetime(
    lifetime: Lifetime,
    transform: Callable[[Tensor], Tensor],
) -> Lifetime:
    with torch.no_grad():
        prototypes = transform(lifetime.prototypes).detach()
        events = tuple(
            AuditEvent(
                observation=AgentObservation(
                    event_id=event.observation.event_id,
                    family=event.observation.family,
                    evidence=transform(event.observation.evidence).detach(),
                ),
                category=event.category,
                correct_action=event.correct_action,
                is_novel=event.is_novel,
            )
            for event in lifetime.events
        )
    return Lifetime(
        events=events,
        prototypes=prototypes,
        correct_actions=lifetime.correct_actions,
        initial_categories=lifetime.initial_categories,
        novel_categories=lifetime.novel_categories,
    )


def representation_diagnostics(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    encoder: TemporalContrastiveEncoder,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(62_000 + seed)
    size = config.representation_evaluation_size
    first_raw, second_raw = _sample_temporal_views(
        config,
        sensor,
        generator,
        size,
    )
    with torch.no_grad():
        first_encoded = encoder(first_raw)
        second_encoded = encoder(second_raw)
    positive = torch.sum(first_encoded * second_encoded, dim=-1)
    negative = torch.sum(first_encoded * second_encoded.roll(1, dims=0), dim=-1)

    latent_first = _unit(
        torch.randn(size, config.content_dim, generator=generator)
    )
    latent_second = _unit(
        torch.randn(size, config.content_dim, generator=generator)
    )
    with torch.no_grad():
        raw_first = sensor(latent_first)
        raw_second = sensor(latent_second)
        encoded_first = encoder(raw_first)
        encoded_second = encoder(raw_second)
    latent_cosine = torch.sum(latent_first * latent_second, dim=-1)
    raw_cosine = torch.sum(raw_first * raw_second, dim=-1)
    encoded_cosine = torch.sum(encoded_first * encoded_second, dim=-1)
    raw_correlation = float(torch.corrcoef(torch.stack((latent_cosine, raw_cosine)))[0, 1])
    encoded_correlation = float(
        torch.corrcoef(torch.stack((latent_cosine, encoded_cosine)))[0, 1]
    )
    return {
        "temporal_positive_cosine": float(positive.mean()),
        "temporal_negative_cosine": float(negative.mean()),
        "temporal_cosine_gap": float(positive.mean() - negative.mean()),
        "raw_latent_cosine_correlation": raw_correlation,
        "encoded_latent_cosine_correlation": encoded_correlation,
    }


def calibrate_gate_from_temporal_pairs(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    encoder: TemporalContrastiveEncoder,
    seed: int,
) -> GateCalibration:
    """Calibrate a monotone gate using no task or memory labels."""

    generator = torch.Generator().manual_seed(63_000 + seed)
    first_raw, second_raw = _sample_temporal_views(
        config,
        sensor,
        generator,
        config.representation_evaluation_size,
    )
    with torch.no_grad():
        first = encoder(first_raw)
        second = encoder(second_raw)
    positive = torch.sum(first * second, dim=-1)
    negative = torch.sum(first * second.roll(1, dims=0), dim=-1)
    positive_floor = float(torch.quantile(positive, 0.10))
    negative_ceiling = float(torch.quantile(negative, 0.95))
    separation = max(positive_floor - negative_ceiling, 0.02)
    threshold = 0.5 * (positive_floor + negative_ceiling)
    logit_ninety_percent = torch.log(torch.tensor(9.0)).item()
    mismatch_slope = min(
        max(4.0 * logit_ninety_percent / separation, 20.0),
        120.0,
    )
    return GateCalibration(
        similarity_threshold=threshold,
        mismatch_slope=mismatch_slope,
        positive_tenth_percentile=positive_floor,
        negative_ninety_fifth_percentile=negative_ceiling,
    )


def _representation_aggregate(
    records: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        field: _mean_sd([record[field] for record in records])
        for field in records[0]
    }


def _development_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["temporal_geometric_chevron"]
    representation = result["representation_aggregate"]
    versus_raw = result["paired"][
        "temporal_geometric_chevron_minus_raw_sensor_geometric_chevron"
    ]
    versus_random = result["paired"][
        "temporal_geometric_chevron_minus_random_encoder_geometric_chevron"
    ]
    versus_oracle = result["paired"][
        "temporal_geometric_chevron_minus_oracle_geometric_chevron"
    ]
    versus_content = result["paired"][
        "temporal_geometric_chevron_minus_temporal_content_attention"
    ]
    return {
        "old_accuracy_at_least_0.95": metrics["final_old_accuracy"]["mean"] >= 0.95,
        "new_accuracy_at_least_0.75": metrics["final_new_accuracy"]["mean"] >= 0.75,
        "new_probe_at_least_0.75": metrics["new_probe_accuracy"]["mean"] >= 0.75,
        "q_calibration_at_least_0.15": metrics["residual_calibration"]["mean"] >= 0.15,
        "promotions_at_least_3": metrics["promotions"]["mean"] >= 3.0,
        "return_better_than_raw": versus_raw[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "return_better_than_random": versus_random[
            "return_per_decision_approx_95ci_low"
        ] > 0.0,
        "return_noninferior_to_oracle": versus_oracle[
            "return_per_decision_approx_95ci_low"
        ] > -0.05,
        "novel_noninferior_to_oracle": versus_oracle[
            "final_new_accuracy_approx_95ci_low"
        ] > -0.05,
        "return_noninferior_to_content": versus_content[
            "return_per_decision_approx_95ci_low"
        ] > -0.05,
        "temporal_cosine_gap_at_least_0.30": representation[
            "temporal_cosine_gap"
        ]["mean"] >= 0.30,
        "latent_cosine_correlation_at_least_0.60": representation[
            "encoded_latent_cosine_correlation"
        ]["mean"] >= 0.60,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    sensor = FixedNonlinearSensor(
        config.content_dim,
        config.sensor_hidden_dim,
        config.sensor_seed,
    ).eval()
    results: list[LifetimeMetrics] = []
    training_records: list[dict[str, float]] = []
    representation_records: list[dict[str, float]] = []
    for seed in range(config.seed_offset, config.seed_offset + config.training_seeds):
        encoder, random_encoder, losses = train_encoder(config, sensor, seed)
        diagnostic = representation_diagnostics(config, sensor, encoder, seed)
        diagnostic["seed"] = float(seed)
        representation_records.append(diagnostic)
        training_records.append(
            {
                "seed": float(seed),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "parameter_count": float(
                    sum(parameter.numel() for parameter in encoder.parameters())
                ),
            }
        )
        for evaluation_index in range(config.evaluation_lifetimes):
            lifetime_seed = 40_000_000 + 10_000 * seed + evaluation_index
            latent = make_lifetime(config, lifetime_seed)
            raw = _transform_lifetime(latent, sensor)
            random_encoded = _transform_lifetime(
                latent,
                lambda value: random_encoder(sensor(value)),
            )
            temporal_encoded = _transform_lifetime(
                latent,
                lambda value: encoder(sensor(value)),
            )
            runs = (
                ("oracle_geometric_chevron", "geometric_chevron_buffer", latent),
                ("raw_sensor_geometric_chevron", "geometric_chevron_buffer", raw),
                (
                    "random_encoder_geometric_chevron",
                    "geometric_chevron_buffer",
                    random_encoded,
                ),
                (
                    "temporal_geometric_chevron",
                    "geometric_chevron_buffer",
                    temporal_encoded,
                ),
                (
                    "temporal_content_attention",
                    "content_attention_buffer",
                    temporal_encoded,
                ),
            )
            for label, condition, lifetime in runs:
                metrics, _ = run_lifetime(
                    condition,
                    config,
                    lifetime,
                    model=None,
                    training=False,
                    training_seed=seed,
                    lifetime_seed=lifetime_seed,
                )
                results.append(replace(metrics, condition=label))

    comparisons = (
        ("temporal_geometric_chevron", "raw_sensor_geometric_chevron"),
        ("temporal_geometric_chevron", "random_encoder_geometric_chevron"),
        ("temporal_geometric_chevron", "oracle_geometric_chevron"),
        ("temporal_geometric_chevron", "temporal_content_attention"),
    )
    result: dict[str, Any] = {
        "experiment": "006_predictive_geometry",
        "config": asdict(config),
        "training": training_records,
        "representation": representation_records,
        "representation_aggregate": _representation_aggregate(
            [
                {key: value for key, value in record.items() if key != "seed"}
                for record in representation_records
            ]
        ),
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
    representation = result["representation_aggregate"]
    lines = [
        f"# Experiment 006: {label} temporal-contrastive geometry",
        "",
        f"- Encoder seeds: {config['seed_offset']}–{config['seed_offset'] + config['training_seeds'] - 1}",
        f"- Contrastive steps per seed: {config['pretraining_steps']}",
        f"- Evaluation lifetimes per seed: {config['evaluation_lifetimes']}",
        f"- Encoder parameters: {int(result['training'][0]['parameter_count'])}",
        "- Downstream Chevron parameters: zero",
        "",
        "| Method | Return | Final old | Final new | New probe | q calibration | Promotions | N drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = aggregate[condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['final_old_accuracy'])} | {_pm(metrics['final_new_accuracy'])} | "
            f"{_pm(metrics['new_probe_accuracy'])} | {_pm(metrics['residual_calibration'])} | "
            f"{_pm(metrics['promotions'])} | {_pm(metrics['established_drift'])} |"
        )
    lines.extend(
        [
            "",
            "## Representation diagnostics",
            "",
            f"- Temporal positive cosine: {_pm(representation['temporal_positive_cosine'])}",
            f"- Temporal negative cosine: {_pm(representation['temporal_negative_cosine'])}",
            f"- Temporal cosine gap: {_pm(representation['temporal_cosine_gap'])}",
            f"- Raw/latent cosine correlation: {_pm(representation['raw_latent_cosine_correlation'])}",
            f"- Encoded/latent cosine correlation: {_pm(representation['encoded_latent_cosine_correlation'])}",
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
        ]
    )
    stem = f"experiment_006_{label}"
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
                    "representation_aggregate": result[
                        "representation_aggregate"
                    ],
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
