"""Experiment 011: learn identity assent from temporal persistence.

The representation learner sees only short persistence windows from a frozen
nonlinear sensor. It never receives task identities, actions, rewards, or
policies. The downstream agent is the protected/revalidated Experiment 010a
mechanism with its identity and policy thresholds left unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.experiment_004_reward_memory import _json_safe, _mean_sd
from experiments.experiment_006_predictive_geometry import (
    FixedNonlinearSensor,
    temporal_contrastive_loss,
)
from experiments.experiment_010_retrospective_policy import (
    ExperimentConfig as BaseConfig,
    ExperimentMetrics,
    RetrospectiveEvent,
    RetrospectiveLifetime,
    RetrospectiveObservation,
    make_lifetime,
    run_lifetime,
)


CONDITIONS = (
    "oracle_protected",
    "raw_sensor_protected",
    "pairwise_temporal_protected",
    "multiview_temporal_protected",
    "hard_persistence_protected",
    "hard_persistence_direct",
)

DISPLAY_NAMES = {
    "oracle_protected": "Oracle protected Chevron",
    "raw_sensor_protected": "Raw-sensor protected Chevron",
    "pairwise_temporal_protected": "Pairwise-temporal protected Chevron",
    "multiview_temporal_protected": "Multi-view-temporal protected Chevron",
    "hard_persistence_protected": "Hard-persistence protected Chevron",
    "hard_persistence_direct": "Hard-persistence direct adaptation",
}

OBJECTIVES = ("pairwise", "multiview", "hard_persistence")


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    sensor_hidden_dim: int = 32
    sensor_seed: int = 606
    encoder_hidden_dim: int = 48
    pretraining_steps: int = 750
    pretraining_observations: int = 256
    persistence_views: int = 4
    temporal_view_noise: float = 0.15
    hard_negative_cosine: float = 0.55
    contrastive_temperature: float = 0.10
    learning_rate: float = 0.003
    weight_decay: float = 0.00001
    representation_evaluation_size: int = 4096


@dataclass(frozen=True)
class RepresentationDiagnostics:
    same_identity_admission: float
    confusable_change_rejection: float
    balanced_identity_accuracy: float
    same_identity_similarity: float
    confusable_change_similarity: float
    persistence_gap: float
    latent_cosine_correlation: float


class ResidualIdentityEncoder(nn.Module):
    """A small correction to sensor geometry, initially equal to identity."""

    def __init__(self, dimension: int, hidden_dim: int) -> None:
        super().__init__()
        self.correction = nn.Sequential(
            nn.Linear(dimension, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dimension),
        )
        final = self.correction[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("residual encoder must end in a linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, observation: Tensor) -> Tensor:
        return F.normalize(observation + self.correction(observation), dim=-1)


def persistence_contrastive_loss(
    embeddings: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Multi-positive contrastive loss; the middle axis is a persistence window."""

    if embeddings.ndim != 3:
        raise ValueError("embeddings must have [windows, views, dimension] shape")
    windows, views, dimension = embeddings.shape
    if windows < 2 or views < 2 or dimension < 1:
        raise ValueError("at least two windows and two views are required")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    flat = F.normalize(embeddings.reshape(windows * views, dimension), dim=-1)
    logits = flat @ flat.T / temperature
    group = torch.arange(windows, device=flat.device).repeat_interleave(views)
    self_mask = torch.eye(windows * views, dtype=torch.bool, device=flat.device)
    positive_mask = group[:, None].eq(group[None, :]) & ~self_mask
    denominator_mask = ~self_mask
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logsum = torch.logsumexp(
        logits.masked_fill(~positive_mask, negative_infinity), dim=1
    )
    total_logsum = torch.logsumexp(
        logits.masked_fill(~denominator_mask, negative_infinity), dim=1
    )
    return (total_logsum - positive_logsum).mean()


def _unit(value: Tensor) -> Tensor:
    return F.normalize(value, dim=-1)


def _random_bases(
    count: int,
    dimension: int,
    generator: torch.Generator,
) -> Tensor:
    return _unit(torch.randn(count, dimension, generator=generator))


def _confusable_pairs(
    count: int,
    dimension: int,
    cosine: float,
    generator: torch.Generator,
) -> Tensor:
    if count % 2 != 0:
        raise ValueError("confusable windows must have an even count")
    anchors = _random_bases(count // 2, dimension, generator)
    direction = torch.randn(count // 2, dimension, generator=generator)
    orthogonal = _unit(
        direction - (direction * anchors).sum(dim=-1, keepdim=True) * anchors
    )
    neighbours = _unit(
        cosine * anchors + math.sqrt(1.0 - cosine**2) * orthogonal
    )
    return torch.stack((anchors, neighbours), dim=1).reshape(count, dimension)


def sample_persistence_windows(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    generator: torch.Generator,
    *,
    hard_negatives: bool,
) -> Tensor:
    if config.pretraining_observations % config.persistence_views != 0:
        raise ValueError("pretraining observations must divide into complete windows")
    windows = config.pretraining_observations // config.persistence_views
    bases = (
        _confusable_pairs(
            windows,
            config.identity_dim,
            config.hard_negative_cosine,
            generator,
        )
        if hard_negatives
        else _random_bases(windows, config.identity_dim, generator)
    )
    noise = torch.randn(
        windows,
        config.persistence_views,
        config.identity_dim,
        generator=generator,
    )
    latent_views = _unit(
        bases[:, None, :] + config.temporal_view_noise * noise
    )
    with torch.no_grad():
        observed = sensor(latent_views.reshape(-1, config.identity_dim))
    return observed.reshape(windows, config.persistence_views, config.identity_dim)


def _sample_pairwise_views(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    pairs = config.pretraining_observations // 2
    bases = _random_bases(pairs, config.identity_dim, generator)
    first = _unit(
        bases
        + config.temporal_view_noise
        * torch.randn(bases.shape, generator=generator)
    )
    second = _unit(
        bases
        + config.temporal_view_noise
        * torch.randn(bases.shape, generator=generator)
    )
    with torch.no_grad():
        return sensor(first), sensor(second)


def train_identity_encoder(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    seed: int,
    objective: str,
) -> tuple[ResidualIdentityEncoder, list[float]]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown representation objective {objective}")
    torch.manual_seed(111_000 + 10 * seed + OBJECTIVES.index(objective))
    encoder = ResidualIdentityEncoder(config.identity_dim, config.encoder_hidden_dim)
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(
        112_000 + 10 * seed + OBJECTIVES.index(objective)
    )
    losses: list[float] = []
    encoder.train()
    for _ in range(config.pretraining_steps):
        if objective == "pairwise":
            first, second = _sample_pairwise_views(config, sensor, generator)
            loss = temporal_contrastive_loss(
                encoder(first),
                encoder(second),
                temperature=config.contrastive_temperature,
            )
        else:
            windows = sample_persistence_windows(
                config,
                sensor,
                generator,
                hard_negatives=objective == "hard_persistence",
            )
            encoded = encoder(windows.reshape(-1, config.identity_dim)).reshape(
                windows.shape
            )
            loss = persistence_contrastive_loss(
                encoded,
                temperature=config.contrastive_temperature,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return encoder.eval(), losses


def _encode(
    sensor: FixedNonlinearSensor,
    encoder: nn.Module | None,
    latent: Tensor,
) -> Tensor:
    observed = sensor(latent)
    return observed if encoder is None else encoder(observed)


def representation_diagnostics(
    config: ExperimentConfig,
    sensor: FixedNonlinearSensor,
    encoder: nn.Module | None,
    seed: int,
) -> RepresentationDiagnostics:
    generator = torch.Generator().manual_seed(113_000 + seed)
    size = config.representation_evaluation_size
    anchors = _random_bases(size, config.identity_dim, generator)
    direction = torch.randn(size, config.identity_dim, generator=generator)
    orthogonal = _unit(
        direction - (direction * anchors).sum(dim=-1, keepdim=True) * anchors
    )
    neighbours = _unit(
        config.hard_negative_cosine * anchors
        + math.sqrt(1.0 - config.hard_negative_cosine**2) * orthogonal
    )
    positive_views = _unit(
        anchors
        + config.temporal_view_noise
        * torch.randn(anchors.shape, generator=generator)
    )
    negative_views = _unit(
        neighbours
        + config.temporal_view_noise
        * torch.randn(neighbours.shape, generator=generator)
    )
    with torch.no_grad():
        clean_encoded = _encode(sensor, encoder, anchors)
        positive_encoded = _encode(sensor, encoder, positive_views)
        negative_encoded = _encode(sensor, encoder, negative_views)
        positive_similarity = (clean_encoded * positive_encoded).sum(dim=-1)
        negative_similarity = (clean_encoded * negative_encoded).sum(dim=-1)

        left = _random_bases(size, config.identity_dim, generator)
        right = _random_bases(size, config.identity_dim, generator)
        latent_similarity = (left * right).sum(dim=-1)
        encoded_similarity = (
            _encode(sensor, encoder, left) * _encode(sensor, encoder, right)
        ).sum(dim=-1)
        latent_centered = latent_similarity - latent_similarity.mean()
        encoded_centered = encoded_similarity - encoded_similarity.mean()
        correlation = (latent_centered * encoded_centered).mean() / (
            latent_centered.square().mean().sqrt()
            * encoded_centered.square().mean().sqrt()
        ).clamp_min(1e-8)

    admission = float((positive_similarity >= config.similarity_threshold).float().mean())
    rejection = float((negative_similarity < config.similarity_threshold).float().mean())
    return RepresentationDiagnostics(
        same_identity_admission=admission,
        confusable_change_rejection=rejection,
        balanced_identity_accuracy=0.5 * (admission + rejection),
        same_identity_similarity=float(positive_similarity.mean()),
        confusable_change_similarity=float(negative_similarity.mean()),
        persistence_gap=float(positive_similarity.mean() - negative_similarity.mean()),
        latent_cosine_correlation=float(correlation),
    )


def transform_lifetime(
    lifetime: RetrospectiveLifetime,
    transform: Callable[[Tensor], Tensor],
) -> RetrospectiveLifetime:
    with torch.no_grad():
        prototypes = transform(lifetime.identity_prototypes).detach()
        events = tuple(
            RetrospectiveEvent(
                observation=RetrospectiveObservation(
                    event.observation.event_id,
                    event.observation.family,
                    transform(event.observation.identity.unsqueeze(0))[0].detach(),
                ),
                category=event.category,
                correct_action=event.correct_action,
                kind=event.kind,
                reward_flipped=event.reward_flipped,
            )
            for event in lifetime.events
        )
    return RetrospectiveLifetime(
        events=events,
        identity_prototypes=prototypes,
        initial_actions=lifetime.initial_actions.clone(),
        current_actions=lifetime.current_actions.clone(),
        stable_categories=lifetime.stable_categories,
        reversed_categories=lifetime.reversed_categories,
        novel_categories=lifetime.novel_categories,
    )


def _aggregate_metrics(
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


def _paired(
    rows: list[ExperimentMetrics], first: str, second: str
) -> dict[str, float]:
    by_key = {(row.condition, row.seed): row for row in rows}
    seeds = sorted({row.seed for row in rows})
    output: dict[str, float] = {}
    fields = (
        "return_per_decision",
        "clean_accuracy",
        "retention_accuracy",
        "reversed_probe_accuracy",
        "novel_probe_accuracy",
    )
    for field in fields:
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


def _aggregate_diagnostics(
    rows: dict[str, list[RepresentationDiagnostics]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        objective: {
            field: _mean_sd([getattr(row, field) for row in values])
            for field in RepresentationDiagnostics.__dataclass_fields__
        }
        for objective, values in rows.items()
    }


def _gate(result: dict[str, Any]) -> dict[str, bool]:
    hard = result["aggregate"]["hard_persistence_protected"]
    representation = result["representation"]["hard_persistence"]
    versus_raw = result["paired"]["hard_persistence_protected_minus_raw_sensor_protected"]
    versus_pairwise = result["paired"][
        "hard_persistence_protected_minus_pairwise_temporal_protected"
    ]
    versus_multiview = result["paired"][
        "hard_persistence_protected_minus_multiview_temporal_protected"
    ]
    versus_oracle = result["paired"]["hard_persistence_protected_minus_oracle_protected"]
    versus_direct = result["paired"]["hard_persistence_protected_minus_hard_persistence_direct"]
    return {
        "retention_accuracy_at_least_0.90": hard["retention_accuracy"]["mean"] >= 0.90,
        "reversed_probe_at_least_0.75": hard["reversed_probe_accuracy"]["mean"] >= 0.75,
        "novel_probe_at_least_0.75": hard["novel_probe_accuracy"]["mean"] >= 0.75,
        "new_promotions_at_least_3": hard["new_promotions"]["mean"] >= 3.0,
        "unique_revisions_at_least_3": hard["unique_revision_categories"]["mean"] >= 3.0,
        "identity_calibration_at_least_0.10": hard["identity_residual_calibration"]["mean"] >= 0.10,
        "policy_calibration_at_least_0.10": hard["policy_residual_calibration"]["mean"] >= 0.10,
        "no_duplicate_allocations": hard["duplicate_allocations"]["mean"] == 0.0,
        "no_established_overwrites": hard["established_overwrites"]["mean"] == 0.0,
        "no_under_supported_writes": hard["under_supported_writes"]["mean"] == 0.0,
        "same_identity_admission_at_least_0.90": representation["same_identity_admission"]["mean"] >= 0.90,
        "confusable_change_rejection_at_least_0.80": representation["confusable_change_rejection"]["mean"] >= 0.80,
        "balanced_identity_accuracy_at_least_0.85": representation["balanced_identity_accuracy"]["mean"] >= 0.85,
        "return_better_than_raw_sensor": versus_raw["return_per_decision_approx_95ci_low"] > 0.0,
        "return_better_than_pairwise_temporal": versus_pairwise["return_per_decision_approx_95ci_low"] > 0.0,
        "return_noninferior_to_multiview": versus_multiview["return_per_decision_approx_95ci_low"] > -0.02,
        "return_noninferior_to_oracle": versus_oracle["return_per_decision_approx_95ci_low"] > -0.08,
        "clean_accuracy_noninferior_to_oracle": versus_oracle["clean_accuracy_approx_95ci_low"] > -0.08,
        "retention_noninferior_to_oracle": versus_oracle["retention_accuracy_approx_95ci_low"] > -0.05,
        "return_noninferior_to_direct": versus_direct["return_per_decision_approx_95ci_low"] > -0.08,
        "clean_accuracy_noninferior_to_direct": versus_direct["clean_accuracy_approx_95ci_low"] > -0.08,
        "retention_noninferior_to_direct": versus_direct["retention_accuracy_approx_95ci_low"] > -0.03,
    }


def run_study(
    config: ExperimentConfig,
    *,
    training_seeds: tuple[int, ...],
    lifetimes_per_encoder: int,
    lifetime_seed_offset: int,
    status: str,
) -> dict[str, Any]:
    sensor = FixedNonlinearSensor(
        config.identity_dim, config.sensor_hidden_dim, config.sensor_seed
    ).eval()
    rows: list[ExperimentMetrics] = []
    diagnostic_rows: dict[str, list[RepresentationDiagnostics]] = {
        "raw_sensor": [],
        **{objective: [] for objective in OBJECTIVES},
    }
    training: list[dict[str, Any]] = []

    for encoder_index, training_seed in enumerate(training_seeds):
        encoders: dict[str, ResidualIdentityEncoder] = {}
        for objective in OBJECTIVES:
            encoder, losses = train_identity_encoder(
                config, sensor, training_seed, objective
            )
            encoders[objective] = encoder
            diagnostic_rows[objective].append(
                representation_diagnostics(
                    config, sensor, encoder, 10 * training_seed + OBJECTIVES.index(objective)
                )
            )
            training.append(
                {
                    "training_seed": training_seed,
                    "objective": objective,
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                    "last_50_loss": statistics.fmean(losses[-50:]),
                }
            )
        diagnostic_rows["raw_sensor"].append(
            representation_diagnostics(config, sensor, None, 10 * training_seed + 9)
        )

        transforms: dict[str, Callable[[Tensor], Tensor]] = {
            "raw": sensor,
            **{
                objective: (
                    lambda value, encoder=encoder: encoder(sensor(value))
                )
                for objective, encoder in encoders.items()
            },
        }
        for lifetime_index in range(lifetimes_per_encoder):
            seed = lifetime_seed_offset + encoder_index * 1_000 + lifetime_index
            latent_lifetime = make_lifetime(config, seed)
            transformed = {
                name: transform_lifetime(latent_lifetime, transform)
                for name, transform in transforms.items()
            }
            specifications = (
                ("oracle_protected", "retrospective_protected", latent_lifetime),
                ("raw_sensor_protected", "retrospective_protected", transformed["raw"]),
                ("pairwise_temporal_protected", "retrospective_protected", transformed["pairwise"]),
                ("multiview_temporal_protected", "retrospective_protected", transformed["multiview"]),
                ("hard_persistence_protected", "retrospective_protected", transformed["hard_persistence"]),
                ("hard_persistence_direct", "direct_update", transformed["hard_persistence"]),
            )
            for output_condition, agent_condition, lifetime in specifications:
                metrics = run_lifetime(
                    agent_condition,
                    config,
                    lifetime,
                    seed,
                    revalidate_identity_promotion=True,
                )
                rows.append(replace(metrics, condition=output_condition))

    comparisons = (
        ("hard_persistence_protected", "raw_sensor_protected"),
        ("hard_persistence_protected", "pairwise_temporal_protected"),
        ("hard_persistence_protected", "multiview_temporal_protected"),
        ("hard_persistence_protected", "oracle_protected"),
        ("hard_persistence_protected", "hard_persistence_direct"),
    )
    result: dict[str, Any] = {
        "experiment": "011_persistent_identity",
        "status": status,
        "config": asdict(config),
        "training_seeds": list(training_seeds),
        "lifetimes_per_encoder": lifetimes_per_encoder,
        "lifetime_seed_offset": lifetime_seed_offset,
        "aggregate": _aggregate_metrics(rows),
        "representation": _aggregate_diagnostics(diagnostic_rows),
        "training": training,
        "paired": {
            f"{first}_minus_{second}": _paired(rows, first, second)
            for first, second in comparisons
        },
        "individual": [asdict(row) for row in rows],
    }
    result["frozen_gate"] = _gate(result)
    result["passed"] = all(result["frozen_gate"].values())
    return result


def _pm(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.3f} +/- {metric['population_sd']:.3f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = result["status"]
    lines = [
        f"# Experiment 011: persistence-derived identity {status}",
        "",
        f"- Training seeds: {result['training_seeds']}",
        f"- RL lifetimes per encoder: {result['lifetimes_per_encoder']}",
        "- Downstream gate threshold: **fixed at cosine 0.62**",
        "- Policy mechanism: **frozen protected retrospective Chevron**",
        "",
        "| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | Identity calibration | New IDs | Revisions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['clean_accuracy'])} | {_pm(metrics['retention_accuracy'])} | "
            f"{_pm(metrics['reversed_probe_accuracy'])} | "
            f"{_pm(metrics['novel_probe_accuracy'])} | "
            f"{_pm(metrics['identity_residual_calibration'])} | "
            f"{_pm(metrics['new_promotions'])} | "
            f"{_pm(metrics['unique_revision_categories'])} |"
        )
    lines.extend(
        [
            "",
            "## Representation diagnostics",
            "",
            "| Representation | Same admitted | Confusable rejected | Balanced accuracy | Gap | Latent correlation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for objective, metrics in result["representation"].items():
        lines.append(
            f"| {objective} | {_pm(metrics['same_identity_admission'])} | "
            f"{_pm(metrics['confusable_change_rejection'])} | "
            f"{_pm(metrics['balanced_identity_accuracy'])} | "
            f"{_pm(metrics['persistence_gap'])} | "
            f"{_pm(metrics['latent_cosine_correlation'])} |"
        )
    lines.extend(["", "## Frozen gate", ""])
    for name, passed in result["frozen_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Overall {status} result: **{'PASS' if result['passed'] else 'FAIL'}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    stem = f"experiment_011_{status}"
    (output_dir / f"{stem}_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / f"{stem}_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    if args.mode == "development":
        result = run_study(
            ExperimentConfig(),
            training_seeds=(0, 1),
            lifetimes_per_encoder=10,
            lifetime_seed_offset=110_000_000,
            status="development",
        )
    else:
        result = run_study(
            ExperimentConfig(),
            training_seeds=tuple(range(1100, 1110)),
            lifetimes_per_encoder=20,
            lifetime_seed_offset=111_000_000,
            status="confirmation",
        )
    write_report(result, args.output_dir)
    print(
        json.dumps(
            _json_safe(
                {
                    "aggregate": result["aggregate"],
                    "representation": result["representation"],
                    "frozen_gate": result["frozen_gate"],
                    "passed": result["passed"],
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
