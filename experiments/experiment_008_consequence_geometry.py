"""Experiment 008: learn Chevron geometry from dense action consequences."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.experiment_004_reward_memory import (
    AgentObservation,
    AuditEvent,
    Lifetime,
    LifetimeMetrics,
    _json_safe,
    _mean_sd,
    run_lifetime,
)
from experiments.experiment_005a_geometric_gate import _aggregate, _paired
from experiments.experiment_006_predictive_geometry import (
    FixedNonlinearSensor,
    TemporalContrastiveEncoder,
    _transform_lifetime,
    _unit,
    train_encoder,
)
from experiments.experiment_007_action_prediction import (
    ExperimentConfig as BaseConfig,
    FixedLatentDynamics,
    train_action_encoder,
)


CONDITIONS = (
    "oracle_geometric_chevron",
    "raw_sensor_geometric_chevron",
    "temporal_geometric_chevron",
    "action_predictive_geometric_chevron",
    "consequence_geometric_chevron",
    "consequence_content_attention",
)

DISPLAY_NAMES = {
    "oracle_geometric_chevron": "Oracle geometric Chevron",
    "raw_sensor_geometric_chevron": "Raw-sensor geometric Chevron",
    "temporal_geometric_chevron": "Temporal-contrastive geometric Chevron",
    "action_predictive_geometric_chevron": "Action-predictive geometric Chevron",
    "consequence_geometric_chevron": "Consequence-metric geometric Chevron",
    "consequence_content_attention": "Consequence-metric content attention",
}


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    affordance_seed: int = 808
    consequence_discount: float = 0.8
    reward_scale: float = 2.5


class FixedAffordanceWorld(nn.Module):
    """Fixed reward and transition consequences with no trainable parameters."""

    def __init__(
        self,
        actions: int,
        dimension: int,
        *,
        affordance_seed: int,
        dynamics_seed: int,
        discount: float,
        reward_scale: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if reward_scale <= 0.0:
            raise ValueError("reward scale must be positive")
        generator = torch.Generator().manual_seed(affordance_seed)
        reward_weight = F.normalize(
            torch.randn(actions, dimension, generator=generator), dim=-1
        )
        reward_bias = 0.10 * torch.randn(actions, generator=generator)
        self.register_buffer("reward_weight", reward_weight)
        self.register_buffer("reward_bias", reward_bias)
        self.dynamics = FixedLatentDynamics(actions, dimension, dynamics_seed)
        self.actions = actions
        self.discount = discount
        self.reward_scale = reward_scale

    def immediate_rewards(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2:
            raise ValueError("latent must be [batch, dim]")
        return torch.tanh(
            self.reward_scale
            * F.linear(latent, self.reward_weight, self.reward_bias)
        )

    def consequence_values(self, latent: Tensor) -> Tensor:
        rewards = self.immediate_rewards(latent)
        batch, dimension = latent.shape
        expanded = latent[:, None, :].expand(batch, self.actions, dimension)
        actions = torch.arange(self.actions, device=latent.device)
        actions = actions[None, :].expand(batch, self.actions)
        next_latent = self.dynamics(
            expanded.reshape(-1, dimension), actions.reshape(-1)
        ).reshape(batch, self.actions, dimension)
        next_rewards = self.immediate_rewards(
            next_latent.reshape(-1, dimension)
        ).reshape(batch, self.actions, self.actions)
        continuation = next_rewards.max(dim=-1).values
        return rewards + self.discount * continuation

    def consequence_signature(self, latent: Tensor) -> Tensor:
        values = self.consequence_values(latent)
        centred = values - values.mean(dim=-1, keepdim=True)
        return F.normalize(centred, dim=-1)

    def optimal_action(self, latent: Tensor) -> Tensor:
        return self.consequence_values(latent).argmax(dim=-1)


def consequence_metric_loss(
    first_embedding: Tensor,
    second_embedding: Tensor,
    consequence_signature: Tensor,
) -> Tensor:
    if (
        first_embedding.shape != second_embedding.shape
        or first_embedding.ndim != 2
        or consequence_signature.ndim != 2
        or consequence_signature.shape[0] != first_embedding.shape[0]
    ):
        raise ValueError("embeddings and signatures must share a batch dimension")
    represented = first_embedding @ second_embedding.T
    target = consequence_signature @ consequence_signature.T
    return F.mse_loss(represented, target)


def _sample_consequence_views(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    world: FixedAffordanceWorld,
    generator: torch.Generator,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    latent = _unit(
        torch.randn(batch_size, config.content_dim, generator=generator)
    )
    first = _unit(
        latent
        + config.temporal_view_noise
        * torch.randn(latent.shape, generator=generator)
    )
    second = _unit(
        latent
        + config.temporal_view_noise
        * torch.randn(latent.shape, generator=generator)
    )
    with torch.no_grad():
        return (
            sensor(first),
            sensor(second),
            world.consequence_signature(latent),
            latent,
        )


def train_consequence_encoder(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    world: FixedAffordanceWorld,
    seed: int,
) -> tuple[TemporalContrastiveEncoder, list[float]]:
    torch.manual_seed(60_000 + seed)
    encoder = TemporalContrastiveEncoder(
        config.content_dim, config.encoder_hidden_dim
    )
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(66_000 + seed)
    losses: list[float] = []
    encoder.train()
    for _ in range(config.pretraining_steps):
        first, second, signature, _ = _sample_consequence_views(
            config,
            sensor,
            world,
            generator,
            config.pretraining_batch_size,
        )
        loss = consequence_metric_loss(
            encoder(first), encoder(second), signature
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return encoder.eval(), losses


def _sample_distinct_action_state(
    config: ExperimentConfig,
    world: FixedAffordanceWorld,
    generator: torch.Generator,
    *,
    excluded_actions: set[int],
    reference: Tensor | None = None,
) -> Tensor:
    for _ in range(20_000):
        if reference is None:
            candidate = _unit(
                torch.randn(config.content_dim, generator=generator)
            )
        else:
            direction = torch.randn(config.content_dim, generator=generator)
            orthogonal = _unit(direction - (direction @ reference) * reference)
            candidate = _unit(
                config.novel_anchor_cosine * reference
                + math.sqrt(1.0 - config.novel_anchor_cosine**2) * orthogonal
            )
        action = int(world.optimal_action(candidate.unsqueeze(0))[0])
        if action not in excluded_actions:
            return candidate
    raise RuntimeError("could not sample a state with a distinct affordance")


def make_affordance_lifetime(
    config: ExperimentConfig,
    world: FixedAffordanceWorld,
    seed: int,
) -> Lifetime:
    generator = torch.Generator().manual_seed(seed)
    prototypes: list[Tensor] = []
    initial_categories: list[int] = []
    novel_categories: list[int] = []

    for family in range(config.groups):
        first = _sample_distinct_action_state(
            config, world, generator, excluded_actions=set()
        )
        first_action = int(world.optimal_action(first.unsqueeze(0))[0])
        for _ in range(20_000):
            second = _sample_distinct_action_state(
                config,
                world,
                generator,
                excluded_actions={first_action},
            )
            if float(first @ second) <= 0.35:
                break
        else:
            raise RuntimeError("could not sample a separated second context")
        second_action = int(world.optimal_action(second.unsqueeze(0))[0])
        novel = _sample_distinct_action_state(
            config,
            world,
            generator,
            excluded_actions={first_action, second_action},
            reference=first,
        )
        prototypes.extend((first, second, novel))
        initial_categories.extend((3 * family, 3 * family + 1))
        novel_categories.append(3 * family + 2)

    prototype_tensor = torch.stack(prototypes)
    correct_actions = world.optimal_action(prototype_tensor).to(torch.long)
    events: list[AuditEvent] = []
    novel_set = set(novel_categories)
    for step in range(config.stream_steps):
        use_novel = (
            step >= config.shift_step
            and float(torch.rand((), generator=generator))
            < config.novel_probability
        )
        pool = novel_categories if use_novel else initial_categories
        category = pool[int(torch.randint(len(pool), (), generator=generator))]
        noise = (
            config.hard_noise
            if float(torch.rand((), generator=generator))
            < config.hard_noise_probability
            else config.evidence_noise
        )
        evidence = _unit(
            prototype_tensor[category]
            + noise * torch.randn(config.content_dim, generator=generator)
        )
        events.append(
            AuditEvent(
                observation=AgentObservation(
                    event_id=step,
                    family=category // 3,
                    evidence=evidence,
                ),
                category=category,
                correct_action=int(correct_actions[category]),
                is_novel=category in novel_set,
            )
        )
    return Lifetime(
        events=tuple(events),
        prototypes=prototype_tensor,
        correct_actions=correct_actions,
        initial_categories=tuple(initial_categories),
        novel_categories=tuple(novel_categories),
    )


def consequence_diagnostics(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    world: FixedAffordanceWorld,
    encoder: TemporalContrastiveEncoder,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(67_000 + seed)
    size = config.representation_evaluation_size
    first_latent = _unit(
        torch.randn(size, config.content_dim, generator=generator)
    )
    second_latent = _unit(
        torch.randn(size, config.content_dim, generator=generator)
    )
    with torch.no_grad():
        first_signature = world.consequence_signature(first_latent)
        second_signature = world.consequence_signature(second_latent)
        target = torch.sum(first_signature * second_signature, dim=-1)
        first_raw = sensor(first_latent)
        second_raw = sensor(second_latent)
        raw = torch.sum(first_raw * second_raw, dim=-1)
        encoded = torch.sum(
            encoder(first_raw) * encoder(second_raw), dim=-1
        )
    raw_correlation = float(
        torch.corrcoef(torch.stack((target, raw)))[0, 1]
    )
    encoded_correlation = float(
        torch.corrcoef(torch.stack((target, encoded)))[0, 1]
    )
    return {
        "raw_consequence_cosine_correlation": raw_correlation,
        "encoded_consequence_cosine_correlation": encoded_correlation,
        "consequence_correlation_gain": encoded_correlation - raw_correlation,
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
    metrics = result["aggregate"]["consequence_geometric_chevron"]
    diagnostics = result["diagnostic_aggregate"]
    versus_temporal = result["paired"][
        "consequence_geometric_chevron_minus_temporal_geometric_chevron"
    ]
    versus_action = result["paired"][
        "consequence_geometric_chevron_minus_action_predictive_geometric_chevron"
    ]
    versus_oracle = result["paired"][
        "consequence_geometric_chevron_minus_oracle_geometric_chevron"
    ]
    versus_content = result["paired"][
        "consequence_geometric_chevron_minus_consequence_content_attention"
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
        "return_better_than_action_predictive": versus_action[
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
        "consequence_correlation_at_least_0.85": diagnostics[
            "encoded_consequence_cosine_correlation"
        ]["mean"] >= 0.85,
        "consequence_correlation_gain_at_least_0.20": diagnostics[
            "consequence_correlation_gain"
        ]["mean"] >= 0.20,
    }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    sensor = FixedNonlinearSensor(
        config.content_dim, config.sensor_hidden_dim, config.sensor_seed
    ).eval()
    dynamics = FixedLatentDynamics(
        config.action_dim, config.content_dim, config.dynamics_seed
    ).eval()
    world = FixedAffordanceWorld(
        config.action_dim,
        config.content_dim,
        affordance_seed=config.affordance_seed,
        dynamics_seed=config.dynamics_seed,
        discount=config.consequence_discount,
        reward_scale=config.reward_scale,
    ).eval()
    results: list[LifetimeMetrics] = []
    training_records: list[dict[str, float]] = []
    diagnostic_records: list[dict[str, float]] = []
    evaluation_base = 82_000_000 if config.seed_offset >= 900 else 52_000_000

    for seed in range(config.seed_offset, config.seed_offset + config.training_seeds):
        consequence_encoder, consequence_losses = train_consequence_encoder(
            config, sensor, world, seed
        )
        temporal_encoder, _, temporal_losses = train_encoder(config, sensor, seed)
        action_encoder, _, action_losses = train_action_encoder(
            config, sensor, dynamics, seed
        )
        diagnostics = consequence_diagnostics(
            config, sensor, world, consequence_encoder, seed
        )
        diagnostics["seed"] = float(seed)
        diagnostic_records.append(diagnostics)
        training_records.append(
            {
                "seed": float(seed),
                "consequence_initial_loss": consequence_losses[0],
                "consequence_final_loss": consequence_losses[-1],
                "temporal_initial_loss": temporal_losses[0],
                "temporal_final_loss": temporal_losses[-1],
                "action_initial_loss": action_losses[0],
                "action_final_loss": action_losses[-1],
                "encoder_parameter_count": float(
                    sum(p.numel() for p in consequence_encoder.parameters())
                ),
            }
        )
        for evaluation_index in range(config.evaluation_lifetimes):
            lifetime_seed = evaluation_base + 10_000 * seed + evaluation_index
            latent = make_affordance_lifetime(config, world, lifetime_seed)
            raw = _transform_lifetime(latent, sensor)
            temporal = _transform_lifetime(
                latent, lambda value: temporal_encoder(sensor(value))
            )
            action_predictive = _transform_lifetime(
                latent, lambda value: action_encoder(sensor(value))
            )
            consequence = _transform_lifetime(
                latent, lambda value: consequence_encoder(sensor(value))
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
                    "consequence_geometric_chevron",
                    "geometric_chevron_buffer",
                    consequence,
                ),
                (
                    "consequence_content_attention",
                    "content_attention_buffer",
                    consequence,
                ),
            )
            for label, condition, transformed in runs:
                metrics, _ = run_lifetime(
                    condition,
                    config,
                    transformed,
                    model=None,
                    training=False,
                    training_seed=seed,
                    lifetime_seed=lifetime_seed,
                )
                results.append(replace(metrics, condition=label))

    comparisons = (
        ("consequence_geometric_chevron", "temporal_geometric_chevron"),
        (
            "consequence_geometric_chevron",
            "action_predictive_geometric_chevron",
        ),
        ("consequence_geometric_chevron", "oracle_geometric_chevron"),
        ("consequence_geometric_chevron", "consequence_content_attention"),
        ("consequence_geometric_chevron", "raw_sensor_geometric_chevron"),
    )
    result: dict[str, Any] = {
        "experiment": "008_consequence_geometry",
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
        f"# Experiment 008: {label} consequence geometry",
        "",
        f"- Encoder seeds: {config['seed_offset']}–{config['seed_offset'] + config['training_seeds'] - 1}",
        f"- Training steps per objective: {config['pretraining_steps']}",
        f"- Evaluation lifetimes per seed: {config['evaluation_lifetimes']}",
        f"- Downstream encoder parameters: {int(result['training'][0]['encoder_parameter_count'])}",
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
            "## Consequence diagnostics",
            "",
            f"- Raw consequence-cosine correlation: {_pm(diagnostics['raw_consequence_cosine_correlation'])}",
            f"- Encoded consequence-cosine correlation: {_pm(diagnostics['encoded_consequence_cosine_correlation'])}",
            f"- Correlation gain: {_pm(diagnostics['consequence_correlation_gain'])}",
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
    stem = f"experiment_008_{label}"
    (output_dir / f"{stem}_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / f"{stem}_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
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
            seed_offset=900 if args.seed_offset is None else args.seed_offset,
            pretraining_steps=(
                defaults.pretraining_steps
                if args.pretraining_steps is None
                else args.pretraining_steps
            ),
            evaluation_lifetimes=(
                20 if args.evaluation_lifetimes is None else args.evaluation_lifetimes
            ),
        )
        if config.seed_offset < 900:
            raise ValueError("confirmation seed offset must be at least 900")
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
