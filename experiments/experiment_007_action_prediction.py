"""Experiment 007: learn Chevron geometry by predicting action transitions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.experiment_004_reward_memory import (
    LifetimeMetrics,
    _json_safe,
    _mean_sd,
    make_lifetime,
    run_lifetime,
)
from experiments.experiment_005a_geometric_gate import _aggregate, _paired
from experiments.experiment_006_predictive_geometry import (
    ExperimentConfig as BaseConfig,
    FixedNonlinearSensor,
    TemporalContrastiveEncoder,
    _transform_lifetime,
    _unit,
    train_encoder,
)


CONDITIONS = (
    "oracle_geometric_chevron",
    "raw_sensor_geometric_chevron",
    "temporal_geometric_chevron",
    "action_predictive_geometric_chevron",
    "action_predictive_content_attention",
)

DISPLAY_NAMES = {
    "oracle_geometric_chevron": "Oracle geometric Chevron",
    "raw_sensor_geometric_chevron": "Raw-sensor geometric Chevron",
    "temporal_geometric_chevron": "Temporal-contrastive geometric Chevron",
    "action_predictive_geometric_chevron": "Action-predictive geometric Chevron",
    "action_predictive_content_attention": "Action-predictive content attention",
}


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    dynamics_seed: int = 607
    transition_noise: float = 0.05


class FixedLatentDynamics(nn.Module):
    """Four fixed near-identity orthogonal latent transition operators."""

    def __init__(self, actions: int, dimension: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        transforms = []
        for _ in range(actions):
            random = torch.randn(dimension, dimension, generator=generator)
            skew = random - random.T
            transforms.append(torch.matrix_exp(0.20 * skew))
        self.register_buffer("transforms", torch.stack(transforms))

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        if latent.ndim != 2 or action.shape != (latent.shape[0],):
            raise ValueError("latent must be [batch, dim] and action [batch]")
        selected = self.transforms[action]
        return torch.einsum("bij,bj->bi", selected, latent)


class ActionConditionedPredictor(nn.Module):
    def __init__(self, actions: int, dimension: int) -> None:
        super().__init__()
        identity = torch.eye(dimension).expand(actions, -1, -1).clone()
        self.weight = nn.Parameter(identity)
        self.bias = nn.Parameter(torch.zeros(actions, dimension))

    def forward(self, embedding: Tensor, action: Tensor) -> Tensor:
        selected_weight = self.weight[action]
        selected_bias = self.bias[action]
        prediction = torch.einsum("bij,bj->bi", selected_weight, embedding)
        return F.normalize(prediction + selected_bias, dim=-1)


def transition_prediction_loss(
    predicted_next: Tensor,
    observed_next: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if predicted_next.shape != observed_next.shape or predicted_next.ndim != 2:
        raise ValueError("next embeddings must have the same [batch, dim] shape")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = predicted_next @ observed_next.T / temperature
    target = torch.arange(predicted_next.shape[0], device=predicted_next.device)
    return F.cross_entropy(logits, target)


def _sample_transitions(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    dynamics: FixedLatentDynamics,
    generator: torch.Generator,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    latent = _unit(
        torch.randn(batch_size, config.content_dim, generator=generator)
    )
    action = torch.randint(
        config.action_dim,
        (batch_size,),
        generator=generator,
    )
    next_latent = _unit(
        dynamics(latent, action)
        + config.transition_noise
        * torch.randn(latent.shape, generator=generator)
    )
    current_view = _unit(
        latent
        + config.temporal_view_noise
        * torch.randn(latent.shape, generator=generator)
    )
    next_view = _unit(
        next_latent
        + config.temporal_view_noise
        * torch.randn(next_latent.shape, generator=generator)
    )
    with torch.no_grad():
        return sensor(current_view), action, sensor(next_view), latent, next_latent


def train_action_encoder(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    dynamics: FixedLatentDynamics,
    seed: int,
) -> tuple[TemporalContrastiveEncoder, ActionConditionedPredictor, list[float]]:
    torch.manual_seed(60_000 + seed)
    encoder = TemporalContrastiveEncoder(
        config.content_dim,
        config.encoder_hidden_dim,
    )
    predictor = ActionConditionedPredictor(config.action_dim, config.content_dim)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(64_000 + seed)
    losses: list[float] = []
    encoder.train()
    predictor.train()
    for _ in range(config.pretraining_steps):
        current, action, observed_next, _, _ = _sample_transitions(
            config,
            sensor,
            dynamics,
            generator,
            config.pretraining_batch_size,
        )
        predicted_next = predictor(encoder(current), action)
        target_next = encoder(observed_next)
        loss = transition_prediction_loss(
            predicted_next,
            target_next,
            temperature=config.contrastive_temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()),
            1.0,
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    return encoder.eval(), predictor.eval(), losses


def transition_diagnostics(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    dynamics: FixedLatentDynamics,
    encoder: TemporalContrastiveEncoder,
    predictor: ActionConditionedPredictor,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(65_000 + seed)
    current, action, observed_next, latent, next_latent = _sample_transitions(
        config,
        sensor,
        dynamics,
        generator,
        config.representation_evaluation_size,
    )
    with torch.no_grad():
        encoded_current = encoder(current)
        encoded_next = encoder(observed_next)
        predicted_next = predictor(encoded_current, action)
        raw_current = sensor(latent)
        raw_next = sensor(next_latent)
        clean_encoded_current = encoder(raw_current)
        clean_encoded_next = encoder(raw_next)
    positive = torch.sum(predicted_next * encoded_next, dim=-1)
    negative = torch.sum(predicted_next * encoded_next.roll(1, dims=0), dim=-1)
    latent_cosine = torch.sum(latent * next_latent, dim=-1)
    raw_cosine = torch.sum(raw_current * raw_next, dim=-1)
    encoded_cosine = torch.sum(
        clean_encoded_current * clean_encoded_next,
        dim=-1,
    )
    raw_correlation = float(
        torch.corrcoef(torch.stack((latent_cosine, raw_cosine)))[0, 1]
    )
    encoded_correlation = float(
        torch.corrcoef(torch.stack((latent_cosine, encoded_cosine)))[0, 1]
    )
    return {
        "predicted_next_cosine": float(positive.mean()),
        "permuted_next_cosine": float(negative.mean()),
        "transition_cosine_gap": float(positive.mean() - negative.mean()),
        "raw_transition_cosine_correlation": raw_correlation,
        "encoded_transition_cosine_correlation": encoded_correlation,
    }


def _diagnostic_aggregate(
    records: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        field: _mean_sd([record[field] for record in records])
        for field in records[0]
        if field != "seed"
    }


def _development_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["aggregate"]["action_predictive_geometric_chevron"]
    diagnostics = result["diagnostic_aggregate"]
    versus_temporal = result["paired"][
        "action_predictive_geometric_chevron_minus_temporal_geometric_chevron"
    ]
    versus_oracle = result["paired"][
        "action_predictive_geometric_chevron_minus_oracle_geometric_chevron"
    ]
    versus_content = result["paired"][
        "action_predictive_geometric_chevron_minus_action_predictive_content_attention"
    ]
    return {
        "old_accuracy_at_least_0.95": metrics["final_old_accuracy"]["mean"] >= 0.95,
        "new_accuracy_at_least_0.75": metrics["final_new_accuracy"]["mean"] >= 0.75,
        "new_probe_at_least_0.75": metrics["new_probe_accuracy"]["mean"] >= 0.75,
        "q_calibration_at_least_0.15": metrics["residual_calibration"]["mean"] >= 0.15,
        "promotions_at_least_3": metrics["promotions"]["mean"] >= 3.0,
        "return_better_than_temporal": versus_temporal[
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
        "transition_cosine_gap_at_least_0.30": diagnostics[
            "transition_cosine_gap"
        ]["mean"] >= 0.30,
        "transition_correlation_at_least_0.80": diagnostics[
            "encoded_transition_cosine_correlation"
        ]["mean"] >= 0.80,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    sensor = FixedNonlinearSensor(
        config.content_dim,
        config.sensor_hidden_dim,
        config.sensor_seed,
    ).eval()
    dynamics = FixedLatentDynamics(
        config.action_dim,
        config.content_dim,
        config.dynamics_seed,
    ).eval()
    results: list[LifetimeMetrics] = []
    training_records: list[dict[str, float]] = []
    diagnostic_records: list[dict[str, float]] = []
    evaluation_base = 72_000_000 if config.seed_offset >= 800 else 42_000_000

    for seed in range(config.seed_offset, config.seed_offset + config.training_seeds):
        action_encoder, predictor, action_losses = train_action_encoder(
            config,
            sensor,
            dynamics,
            seed,
        )
        temporal_encoder, _, temporal_losses = train_encoder(config, sensor, seed)
        diagnostics = transition_diagnostics(
            config,
            sensor,
            dynamics,
            action_encoder,
            predictor,
            seed,
        )
        diagnostics["seed"] = float(seed)
        diagnostic_records.append(diagnostics)
        training_records.append(
            {
                "seed": float(seed),
                "action_initial_loss": action_losses[0],
                "action_final_loss": action_losses[-1],
                "temporal_initial_loss": temporal_losses[0],
                "temporal_final_loss": temporal_losses[-1],
                "encoder_parameter_count": float(
                    sum(parameter.numel() for parameter in action_encoder.parameters())
                ),
                "training_predictor_parameter_count": float(
                    sum(parameter.numel() for parameter in predictor.parameters())
                ),
            }
        )
        for evaluation_index in range(config.evaluation_lifetimes):
            lifetime_seed = evaluation_base + 10_000 * seed + evaluation_index
            latent = make_lifetime(config, lifetime_seed)
            raw = _transform_lifetime(latent, sensor)
            temporal = _transform_lifetime(
                latent,
                lambda value: temporal_encoder(sensor(value)),
            )
            action_predictive = _transform_lifetime(
                latent,
                lambda value: action_encoder(sensor(value)),
            )
            runs = (
                ("oracle_geometric_chevron", "geometric_chevron_buffer", latent),
                ("raw_sensor_geometric_chevron", "geometric_chevron_buffer", raw),
                (
                    "temporal_geometric_chevron",
                    "geometric_chevron_buffer",
                    temporal,
                ),
                (
                    "action_predictive_geometric_chevron",
                    "geometric_chevron_buffer",
                    action_predictive,
                ),
                (
                    "action_predictive_content_attention",
                    "content_attention_buffer",
                    action_predictive,
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
        (
            "action_predictive_geometric_chevron",
            "temporal_geometric_chevron",
        ),
        (
            "action_predictive_geometric_chevron",
            "oracle_geometric_chevron",
        ),
        (
            "action_predictive_geometric_chevron",
            "action_predictive_content_attention",
        ),
        (
            "action_predictive_geometric_chevron",
            "raw_sensor_geometric_chevron",
        ),
    )
    result: dict[str, Any] = {
        "experiment": "007_action_prediction",
        "config": asdict(config),
        "training": training_records,
        "diagnostics": diagnostic_records,
        "diagnostic_aggregate": _diagnostic_aggregate(diagnostic_records),
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
    diagnostics = result["diagnostic_aggregate"]
    lines = [
        f"# Experiment 007: {label} action-conditioned prediction",
        "",
        f"- Encoder seeds: {config['seed_offset']}–{config['seed_offset'] + config['training_seeds'] - 1}",
        f"- Training steps per objective: {config['pretraining_steps']}",
        f"- Evaluation lifetimes per seed: {config['evaluation_lifetimes']}",
        f"- Downstream encoder parameters: {int(result['training'][0]['encoder_parameter_count'])}",
        f"- Discarded predictor parameters: {int(result['training'][0]['training_predictor_parameter_count'])}",
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
            "## Transition diagnostics",
            "",
            f"- Predicted-next cosine: {_pm(diagnostics['predicted_next_cosine'])}",
            f"- Permuted-next cosine: {_pm(diagnostics['permuted_next_cosine'])}",
            f"- Transition cosine gap: {_pm(diagnostics['transition_cosine_gap'])}",
            f"- Raw transition-cosine correlation: {_pm(diagnostics['raw_transition_cosine_correlation'])}",
            f"- Encoded transition-cosine correlation: {_pm(diagnostics['encoded_transition_cosine_correlation'])}",
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
    stem = f"experiment_007_{label}"
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
            seed_offset=800 if args.seed_offset is None else args.seed_offset,
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
        if config.seed_offset < 800:
            raise ValueError("confirmation seed offset must be at least 800")
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
                    "diagnostic_aggregate": result["diagnostic_aggregate"],
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
