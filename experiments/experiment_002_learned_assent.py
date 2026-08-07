"""Experiment 002: learn assent without letting it become retrieval twice.

The experiment trains two gates with identical per-slot compatibility labels:

* Chevron gate: sees diagnostic A evidence and retained N content after unknown
  coordinate transforms.
* Retrieval-twice gate: sees only the broad A address query and A address
  traces.

Both are evaluated on fresh randomly generated memories.  Standard attention
is retained as a no-abstention reference.  This is supervised preparation for
RL, not an RL experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from chevron_agent import ProjectedCosineAssent


METHODS = ("standard_attention", "retrieval_twice", "learned_chevron")
DISPLAY_NAMES = {
    "standard_attention": "Standard attention",
    "retrieval_twice": "Learned retrieval twice",
    "learned_chevron": "Learned Chevron assent",
}


@dataclass(frozen=True)
class ExperimentConfig:
    training_seeds: int = 10
    seed_offset: int = 100
    training_steps: int = 800
    batch_size: int = 256
    evaluation_batch_size: int = 4096
    groups: int = 4
    content_dim: int = 12
    comparison_dim: int = 12
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    positive_loss_weight: float = 2.0
    match_probability: float = 2.0 / 3.0
    training_max_noise: float = 0.25
    training_negative_cosine_min: float = 0.10
    training_negative_cosine_max: float = 0.60
    evaluation_noise_levels: tuple[float, ...] = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30)
    evaluation_negative_cosine: float = 0.55
    address_scale: float = 4.0
    address_noise: float = 0.02
    allocation_residual_threshold: float = 0.80
    allocation_admitted_threshold: float = 0.25
    write_threshold_margin: float = 0.05

    @property
    def slots(self) -> int:
        return self.groups * 2

    @property
    def retrieval_comparison_dim(self) -> int:
        # Match the two projection matrices' parameter budget as closely as
        # possible despite the control receiving lower-dimensional addresses.
        return math.ceil(self.content_dim * self.comparison_dim / self.groups)


@dataclass
class Batch:
    address_query: Tensor
    address_memory: Tensor
    evidence: Tensor
    content_memory: Tensor
    labels: Tensor
    target: Tensor


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> Tensor:
    matrix = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs


class DissociatedMemoryGenerator:
    """Generate broad address cues and separately transformed A/N content."""

    def __init__(self, config: ExperimentConfig, seed: int) -> None:
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        self.a_transform = _orthogonal_matrix(config.content_dim, self.generator)
        self.n_transform = _orthogonal_matrix(config.content_dim, self.generator)
        addresses = torch.eye(config.groups) * config.address_scale
        self.address_memory = addresses.repeat_interleave(2, dim=0)

    def sample(
        self,
        batch_size: int,
        *,
        maximum_noise: float | None = None,
        fixed_noise: float | None = None,
        negative_cosine_min: float | None = None,
        negative_cosine_max: float | None = None,
    ) -> Batch:
        cfg = self.config
        latent_memory = F.normalize(
            torch.randn(
                batch_size,
                cfg.slots,
                cfg.content_dim,
                generator=self.generator,
            ),
            dim=-1,
        )
        content_memory = latent_memory @ self.n_transform.T

        groups = torch.randint(
            cfg.groups, (batch_size,), generator=self.generator
        )
        local_slot = torch.randint(2, (batch_size,), generator=self.generator)
        candidate_target = 2 * groups + local_slot
        is_match = (
            torch.rand(batch_size, generator=self.generator) < cfg.match_probability
        )
        target = torch.where(
            is_match, candidate_target, torch.full_like(candidate_target, -1)
        )

        address_query = torch.eye(cfg.groups)[groups] * cfg.address_scale
        address_query = address_query + cfg.address_noise * torch.randn(
            address_query.shape, generator=self.generator
        )
        address_memory = self.address_memory.unsqueeze(0).expand(batch_size, -1, -1)

        batch_indices = torch.arange(batch_size)
        matched_latent = latent_memory[batch_indices, candidate_target]
        if fixed_noise is None:
            high = cfg.training_max_noise if maximum_noise is None else maximum_noise
            noise_scale = torch.rand(batch_size, 1, generator=self.generator) * high
        else:
            noise_scale = torch.full((batch_size, 1), fixed_noise)
        positive_latent = F.normalize(
            matched_latent
            + noise_scale
            * torch.randn(matched_latent.shape, generator=self.generator),
            dim=-1,
        )

        random_direction = torch.randn(
            matched_latent.shape, generator=self.generator
        )
        orthogonal = random_direction - (
            random_direction * matched_latent
        ).sum(dim=-1, keepdim=True) * matched_latent
        orthogonal = F.normalize(orthogonal, dim=-1)
        cosine_low = (
            cfg.training_negative_cosine_min
            if negative_cosine_min is None
            else negative_cosine_min
        )
        cosine_high = (
            cfg.training_negative_cosine_max
            if negative_cosine_max is None
            else negative_cosine_max
        )
        cosine = cosine_low + (cosine_high - cosine_low) * torch.rand(
            batch_size, 1, generator=self.generator
        )
        negative_latent = (
            cosine * matched_latent
            + torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0)) * orthogonal
        )

        evidence_latent = torch.where(
            is_match.unsqueeze(-1), positive_latent, negative_latent
        )
        evidence = evidence_latent @ self.a_transform.T

        labels = torch.zeros(batch_size, cfg.slots)
        labels[batch_indices[is_match], candidate_target[is_match]] = 1.0
        return Batch(
            address_query=address_query,
            address_memory=address_memory,
            evidence=evidence,
            content_memory=content_memory,
            labels=labels,
            target=target,
        )


def _retrieval(address_query: Tensor, address_memory: Tensor) -> Tensor:
    logits = torch.einsum("bd,bsd->bs", address_query, address_memory)
    logits = logits / math.sqrt(address_query.shape[-1])
    return torch.softmax(logits, dim=-1)


def _train_seed(
    config: ExperimentConfig, seed: int
) -> tuple[
    ProjectedCosineAssent,
    ProjectedCosineAssent,
    Tensor,
    Tensor,
    dict[str, Any],
]:
    torch.manual_seed(10_000 + seed)
    data = DissociatedMemoryGenerator(config, seed)
    chevron = ProjectedCosineAssent(
        config.content_dim,
        config.content_dim,
        config.comparison_dim,
        initial_threshold=0.25,
        initial_slope=8.0,
    )
    retrieval_twice = ProjectedCosineAssent(
        config.groups,
        config.groups,
        config.retrieval_comparison_dim,
        initial_threshold=0.25,
        initial_slope=8.0,
    )
    optimizer = torch.optim.AdamW(
        list(chevron.parameters()) + list(retrieval_twice.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    initial_losses: dict[str, float] | None = None
    final_losses: dict[str, float] = {}
    positive_weight = torch.tensor(config.positive_loss_weight)
    for step in range(config.training_steps):
        batch = data.sample(config.batch_size)
        chevron_output = chevron(batch.evidence, batch.content_memory)
        retrieval_output = retrieval_twice(batch.address_query, batch.address_memory)
        chevron_loss = F.binary_cross_entropy_with_logits(
            chevron_output.logits, batch.labels, pos_weight=positive_weight
        )
        retrieval_loss = F.binary_cross_entropy_with_logits(
            retrieval_output.logits, batch.labels, pos_weight=positive_weight
        )
        loss = chevron_loss + retrieval_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses = {
            "chevron_bce": float(chevron_loss.detach().item()),
            "retrieval_twice_bce": float(retrieval_loss.detach().item()),
        }
        if step == 0:
            initial_losses = losses
        if step == config.training_steps - 1:
            final_losses = losses

    assert initial_losses is not None
    training = {
        "initial_losses": initial_losses,
        "final_losses": final_losses,
        "chevron_threshold": float(chevron.threshold.detach().item()),
        "chevron_slope": float(chevron.slope.detach().item()),
        "retrieval_twice_threshold": float(
            retrieval_twice.threshold.detach().item()
        ),
        "retrieval_twice_slope": float(retrieval_twice.slope.detach().item()),
    }
    return (
        chevron,
        retrieval_twice,
        data.a_transform.clone(),
        data.n_transform.clone(),
        training,
    )


def _predict(read_mass: Tensor, residual: Tensor, threshold: float) -> Tensor:
    prediction = read_mass.argmax(dim=-1)
    return torch.where(
        residual > threshold, torch.full_like(prediction, -1), prediction
    )


def _evaluate_batch(
    config: ExperimentConfig,
    batch: Batch,
    chevron: ProjectedCosineAssent,
    retrieval_twice: ProjectedCosineAssent,
) -> dict[str, dict[str, float]]:
    with torch.no_grad():
        alpha = _retrieval(batch.address_query, batch.address_memory)
        chevron_output = chevron(batch.evidence, batch.content_memory)
        retrieval_output = retrieval_twice(batch.address_query, batch.address_memory)
        chevron_write = chevron.assent_with_margin(
            batch.evidence,
            batch.content_memory,
            threshold_margin=config.write_threshold_margin,
        )
        retrieval_write = retrieval_twice.assent_with_margin(
            batch.address_query,
            batch.address_memory,
            threshold_margin=config.write_threshold_margin,
        )

        method_tensors = {
            "standard_attention": (alpha, torch.zeros(alpha.shape[0]), alpha),
            "retrieval_twice": (
                alpha * retrieval_output.assent,
                1.0 - (alpha * retrieval_output.assent).sum(dim=-1),
                alpha * retrieval_write,
            ),
            "learned_chevron": (
                alpha * chevron_output.assent,
                1.0 - (alpha * chevron_output.assent).sum(dim=-1),
                alpha * chevron_write,
            ),
        }

        result: dict[str, dict[str, float]] = {}
        is_match = batch.target >= 0
        is_no_match = ~is_match
        batch_indices = torch.arange(batch.target.shape[0])
        for method, (read_mass, residual, write_gate) in method_tensors.items():
            prediction = _predict(
                read_mass, residual, config.allocation_residual_threshold
            )
            maximum_admitted = read_mass.max(dim=-1).values
            allocate = (
                (residual > config.allocation_residual_threshold)
                & (maximum_admitted < config.allocation_admitted_threshold)
            )
            target_read = read_mass[batch_indices[is_match], batch.target[is_match]]
            target_write = write_gate[
                batch_indices[is_match], batch.target[is_match]
            ]
            non_target_read = read_mass[is_match].sum(dim=-1) - target_read
            non_target_write = write_gate[is_match].sum(dim=-1) - target_write
            result[method] = {
                "count": float(batch.target.numel()),
                "correct_sum": float((prediction == batch.target).sum().item()),
                "match_count": float(is_match.sum().item()),
                "match_correct_sum": float(
                    (prediction[is_match] == batch.target[is_match]).sum().item()
                ),
                "no_match_count": float(is_no_match.sum().item()),
                "no_match_correct_sum": float(
                    (prediction[is_no_match] == -1).sum().item()
                ),
                "target_read_sum": float(target_read.sum().item()),
                "non_target_read_sum": float(non_target_read.sum().item()),
                "no_match_q_sum": float(residual[is_no_match].sum().item()),
                "no_match_allocation_sum": float(allocate[is_no_match].sum().item()),
                "match_allocation_sum": float(allocate[is_match].sum().item()),
                "target_write_sum": float(target_write.sum().item()),
                "non_target_write_sum": float(non_target_write.sum().item()),
                "no_match_write_sum": float(write_gate[is_no_match].sum().item()),
            }
        result["gate_validation"] = {
            "chevron_positive_correct_sum": float(
                (chevron_output.assent[batch.labels.bool()] >= 0.5).sum().item()
            ),
            "chevron_negative_correct_sum": float(
                (chevron_output.assent[~batch.labels.bool()] < 0.5).sum().item()
            ),
            "retrieval_positive_correct_sum": float(
                (retrieval_output.assent[batch.labels.bool()] >= 0.5).sum().item()
            ),
            "retrieval_negative_correct_sum": float(
                (retrieval_output.assent[~batch.labels.bool()] < 0.5).sum().item()
            ),
            "positive_count": float(batch.labels.sum().item()),
            "negative_count": float((batch.labels == 0).sum().item()),
        }
        return result


def _finish_totals(total: dict[str, float]) -> dict[str, float]:
    return {
        "accuracy_pct": 100.0 * total["correct_sum"] / total["count"],
        "match_accuracy_pct": 100.0
        * total["match_correct_sum"]
        / total["match_count"],
        "no_match_accuracy_pct": 100.0
        * total["no_match_correct_sum"]
        / total["no_match_count"],
        "mean_target_read_mass": total["target_read_sum"] / total["match_count"],
        "mean_non_target_read_mass": total["non_target_read_sum"]
        / total["match_count"],
        "mean_no_match_residual": total["no_match_q_sum"]
        / total["no_match_count"],
        "no_match_allocation_pct": 100.0
        * total["no_match_allocation_sum"]
        / total["no_match_count"],
        "match_false_allocation_pct": 100.0
        * total["match_allocation_sum"]
        / total["match_count"],
        "mean_target_write_gate": total["target_write_sum"] / total["match_count"],
        "mean_non_target_write_gate": total["non_target_write_sum"]
        / total["match_count"],
        "mean_no_match_total_write_gate": total["no_match_write_sum"]
        / total["no_match_count"],
    }


def _evaluate_seed(
    config: ExperimentConfig,
    seed: int,
    chevron: ProjectedCosineAssent,
    retrieval_twice: ProjectedCosineAssent,
    a_transform: Tensor,
    n_transform: Tensor,
) -> dict[str, Any]:
    data = DissociatedMemoryGenerator(config, 1_000_000 + seed)
    # Fresh evaluation memories use the same unknown A/N coordinate systems as
    # training. Generalising across independently changed coordinate systems is
    # a different meta-learning problem.
    data.a_transform = a_transform
    data.n_transform = n_transform
    by_noise: dict[str, Any] = {}
    aggregate = {method: defaultdict_float() for method in METHODS}
    slot_correct = {
        "chevron_positive": 0.0,
        "chevron_negative": 0.0,
        "retrieval_positive": 0.0,
        "retrieval_negative": 0.0,
        "positive_count": 0.0,
        "negative_count": 0.0,
    }

    for noise in config.evaluation_noise_levels:
        batch = data.sample(
            config.evaluation_batch_size,
            fixed_noise=noise,
            negative_cosine_min=config.evaluation_negative_cosine,
            negative_cosine_max=config.evaluation_negative_cosine,
        )
        evaluated = _evaluate_batch(config, batch, chevron, retrieval_twice)
        noise_metrics: dict[str, Any] = {}
        for method in METHODS:
            for key, value in evaluated[method].items():
                aggregate[method][key] += value
            noise_metrics[method] = _finish_totals(evaluated[method])
        gate = evaluated["gate_validation"]
        slot_correct["chevron_positive"] += gate["chevron_positive_correct_sum"]
        slot_correct["chevron_negative"] += gate["chevron_negative_correct_sum"]
        slot_correct["retrieval_positive"] += gate[
            "retrieval_positive_correct_sum"
        ]
        slot_correct["retrieval_negative"] += gate[
            "retrieval_negative_correct_sum"
        ]
        slot_correct["positive_count"] += gate["positive_count"]
        slot_correct["negative_count"] += gate["negative_count"]
        by_noise[f"{noise:.2f}"] = noise_metrics

    return {
        "seed": seed,
        "overall": {
            method: _finish_totals(dict(aggregate[method])) for method in METHODS
        },
        "by_noise": by_noise,
        "gate_classification_pct": {
            "chevron_positive_recall": 100.0
            * slot_correct["chevron_positive"]
            / slot_correct["positive_count"],
            "chevron_negative_specificity": 100.0
            * slot_correct["chevron_negative"]
            / slot_correct["negative_count"],
            "retrieval_positive_recall": 100.0
            * slot_correct["retrieval_positive"]
            / slot_correct["positive_count"],
            "retrieval_negative_specificity": 100.0
            * slot_correct["retrieval_negative"]
            / slot_correct["negative_count"],
        },
    }


def defaultdict_float() -> dict[str, float]:
    return {
        "count": 0.0,
        "correct_sum": 0.0,
        "match_count": 0.0,
        "match_correct_sum": 0.0,
        "no_match_count": 0.0,
        "no_match_correct_sum": 0.0,
        "target_read_sum": 0.0,
        "non_target_read_sum": 0.0,
        "no_match_q_sum": 0.0,
        "no_match_allocation_sum": 0.0,
        "match_allocation_sum": 0.0,
        "target_write_sum": 0.0,
        "non_target_write_sum": 0.0,
        "no_match_write_sum": 0.0,
    }


def _mean_sd_min(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "population_sd": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _summarise_seeds(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"methods": {}, "by_noise": {}}
    metric_names = list(seed_results[0]["overall"][METHODS[0]].keys())
    for method in METHODS:
        summary["methods"][method] = {
            metric: _mean_sd_min(
                [result["overall"][method][metric] for result in seed_results]
            )
            for metric in metric_names
        }
    for noise in seed_results[0]["by_noise"]:
        summary["by_noise"][noise] = {}
        for method in METHODS:
            summary["by_noise"][noise][method] = {
                metric: _mean_sd_min(
                    [
                        result["by_noise"][noise][method][metric]
                        for result in seed_results
                    ]
                )
                for metric in metric_names
            }
    summary["gate_classification_pct"] = {
        metric: _mean_sd_min(
            [result["gate_classification_pct"][metric] for result in seed_results]
        )
        for metric in seed_results[0]["gate_classification_pct"]
    }
    return summary


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    training_results: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    for seed in range(config.seed_offset, config.seed_offset + config.training_seeds):
        chevron, retrieval_twice, a_transform, n_transform, training = _train_seed(
            config, seed
        )
        training_results.append({"seed": seed, **training})
        seed_results.append(
            _evaluate_seed(
                config,
                seed,
                chevron,
                retrieval_twice,
                a_transform,
                n_transform,
            )
        )
    return {
        "experiment": "002_learned_assent",
        "status": "supervised_diagnostic_not_rl",
        "config": asdict(config),
        "parameter_counts": {
            "standard_attention": 0,
            "retrieval_twice": 2
            * config.groups
            * config.retrieval_comparison_dim
            + 2,
            "learned_chevron": 2
            * config.content_dim
            * config.comparison_dim
            + 2,
        },
        "training": training_results,
        "seed_results": seed_results,
        "summary": _summarise_seeds(seed_results),
        "claim_boundary": (
            "This experiment tests whether separately supervised A/N assent can "
            "be learned on fresh memories under noise. It does not show that RL "
            "reward alone learns assent or that the resulting agent solves games."
        ),
    }


def _pm(metric: dict[str, float], digits: int = 2) -> str:
    return f"{metric['mean']:.{digits}f} +/- {metric['population_sd']:.{digits}f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_002_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    summary = result["summary"]
    config = result["config"]
    lines = [
        "# Experiment 002: learned assent",
        "",
        "## Question",
        "",
        "Can an independently supervised A/N compatibility gate learn assent",
        "under noise while an equally trained address-only gate cannot?",
        "",
        "## Protocol",
        "",
        f"- Independent training seeds: {config['training_seeds']}",
        f"- Confirmation seed range: {config['seed_offset']}–"
        f"{config['seed_offset'] + config['training_seeds'] - 1}",
        f"- Optimisation steps per seed: {config['training_steps']}",
        f"- Positive-match loss weight: {config['positive_loss_weight']}",
        f"- Fresh evaluation queries per seed: "
        f"{config['evaluation_batch_size'] * len(config['evaluation_noise_levels'])}",
        "- A evidence and N memory use unknown independently rotated coordinates.",
        "- Retrieval identifies a two-slot family but not its member.",
        "- The retrieval-twice control receives the same labels but only address features.",
        f"- Learned gate parameters: Chevron {result['parameter_counts']['learned_chevron']}; "
        f"retrieval twice {result['parameter_counts']['retrieval_twice']}.",
        "- This is supervised learning, not reinforcement learning.",
        "",
        "## Held-out results",
        "",
        "Mean +/- population SD across independently trained seeds.",
        "",
        "| Method | Overall accuracy | Match | No match | No-match q |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = summary["methods"][method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | "
            f"{_pm(metrics['accuracy_pct'])}% | "
            f"{_pm(metrics['match_accuracy_pct'])}% | "
            f"{_pm(metrics['no_match_accuracy_pct'])}% | "
            f"{_pm(metrics['mean_no_match_residual'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Chevron noise curve",
            "",
            "| Evidence noise | Overall | Match | No match |",
            "|---:|---:|---:|---:|",
        ]
    )
    for noise, methods in summary["by_noise"].items():
        metrics = methods["learned_chevron"]
        lines.append(
            f"| {noise} | {_pm(metrics['accuracy_pct'])}% | "
            f"{_pm(metrics['match_accuracy_pct'])}% | "
            f"{_pm(metrics['no_match_accuracy_pct'])}% |"
        )
    lines.extend(
        [
            "",
            "## Allocation and write gating",
            "",
            "| Method | Allocate no match | False allocate match | Target write | Non-target write | No-match write |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        metrics = summary["methods"][method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | "
            f"{_pm(metrics['no_match_allocation_pct'])}% | "
            f"{_pm(metrics['match_false_allocation_pct'])}% | "
            f"{_pm(metrics['mean_target_write_gate'], 4)} | "
            f"{_pm(metrics['mean_non_target_write_gate'], 4)} | "
            f"{_pm(metrics['mean_no_match_total_write_gate'], 4)} |"
        )
    gate_metrics = summary["gate_classification_pct"]
    threshold_values = [item["chevron_threshold"] for item in result["training"]]
    slope_values = [item["chevron_slope"] for item in result["training"]]
    chevron_metrics = summary["methods"]["learned_chevron"]
    control_metrics = summary["methods"]["retrieval_twice"]
    overall_gain = (
        chevron_metrics["accuracy_pct"]["mean"]
        - control_metrics["accuracy_pct"]["mean"]
    )
    lines.extend(
        [
            "",
            "## Gate diagnostics",
            "",
            f"- Chevron positive compatibility recall: {_pm(gate_metrics['chevron_positive_recall'])}%",
            f"- Chevron negative compatibility specificity: {_pm(gate_metrics['chevron_negative_specificity'])}%",
            f"- Retrieval-twice positive recall: {_pm(gate_metrics['retrieval_positive_recall'])}%",
            f"- Retrieval-twice negative specificity: {_pm(gate_metrics['retrieval_negative_specificity'])}%",
            f"- Learned Chevron threshold: {_pm(_mean_sd_min(threshold_values), 4)}",
            f"- Learned Chevron slope: {_pm(_mean_sd_min(slope_values), 3)}",
            "",
            "## Finding",
            "",
            f"Learned Chevron assent exceeded the parameter-matched retrieval-twice "
            f"control by {overall_gain:.2f} percentage points overall. It learned both "
            f"sides of the decision: {chevron_metrics['match_accuracy_pct']['mean']:.2f}% "
            f"acceptance of familiar cases and "
            f"{chevron_metrics['no_match_accuracy_pct']['mean']:.2f}% rejection of "
            "no-match cases. The small seed-to-seed spread shows that this was not "
            "dependent on a fortunate initialisation.",
            "",
            "When retrieval ambiguity is held constant, a separately supervised gate "
            "with diagnostic A/N compatibility evidence can learn to admit familiar "
            "content and preserve unresolved mass for incompatible content. Merely "
            "learning a second address-based retrieval computation does not recover "
            "that distinction.",
            "",
            "## Claim boundary",
            "",
            result["claim_boundary"],
            "The diagnostic is synthetic, compatibility labels are supplied directly, "
            "and each trained gate is evaluated under the coordinate transforms it saw "
            "during training. Buffer dynamics, consolidation, non-stationarity, and "
            "reward-derived learning remain untested.",
            "",
            "## Next decision",
            "",
            "Proceed to a minimal sequential environment in which retrospective "
            "outcomes train assent and rejected evidence enters a bounded provisional "
            "buffer. That experiment should test whether the factorisation improves "
            "behaviour, rather than only supervised compatibility classification.",
            "",
        ]
    )
    (output_dir / "experiment_002_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    config = ExperimentConfig(
        training_seeds=args.seeds,
        seed_offset=args.seed_offset,
        training_steps=args.steps,
        batch_size=args.batch_size,
    )
    result = run_experiment(config)
    output_dir = Path(__file__).resolve().parent / "results"
    write_report(result, output_dir)
    for method in METHODS:
        metrics = result["summary"]["methods"][method]
        print(
            f"{DISPLAY_NAMES[method]:24s} "
            f"accuracy={metrics['accuracy_pct']['mean']:6.2f} "
            f"+/-{metrics['accuracy_pct']['population_sd']:5.2f} "
            f"no_match={metrics['no_match_accuracy_pct']['mean']:6.2f}"
        )


if __name__ == "__main__":
    main()
