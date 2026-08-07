"""Experiment 001: causal separation of retrieval and assent.

The address cue identifies a two-slot memory family but cannot distinguish the
members.  Diagnostic evidence either matches one retained N content or matches
neither.  This creates the smallest setting in which retrieval and assent must
be different computations.

The experiment is intentionally not learned and is not an RL result.  It asks
whether the refined equations have the declared causal behavior before a policy
or optimizer can obscure failures.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from chevron_agent import (
    ChevronAttentionConfig,
    apply_convex_write,
    chevron_attention,
    normalized_cosine_mismatch,
)


METHODS = ("standard_attention", "retrieval_twice", "chevron")
DISPLAY_NAMES = {
    "standard_attention": "Standard attention",
    "retrieval_twice": "Retrieval twice",
    "chevron": "Chevron A/N assent",
}


def _normal(vector: Tensor, generator: torch.Generator) -> Tensor:
    return torch.randn(vector.shape, generator=generator, dtype=vector.dtype)


def _novel_evidence(
    pair_content: Tensor,
    content_dim: int,
    generator: torch.Generator,
    maximum_cosine: float = 0.20,
) -> Tensor:
    for _ in range(1_000):
        candidate = F.normalize(
            torch.randn(content_dim, generator=generator), dim=-1
        )
        if torch.max(pair_content @ candidate).item() < maximum_cosine:
            return candidate
    raise RuntimeError("could not sample evidence sufficiently distinct from the pair")


def _retrieval_twice(
    alpha: Tensor,
    address_query: Tensor,
    address_memory: Tensor,
    config: ChevronAttentionConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    address_mismatch = normalized_cosine_mismatch(
        address_query.unsqueeze(0), address_memory.unsqueeze(0), eps=config.eps
    )
    read_assent = torch.sigmoid(
        config.read_slope * (config.read_threshold - address_mismatch)
    )
    write_assent = torch.sigmoid(
        config.write_slope * (config.write_threshold - address_mismatch)
    )
    read_mass = alpha * read_assent
    residual = 1.0 - read_mass.sum(dim=-1)
    write_gate = alpha * write_assent
    return read_mass, residual, write_gate


def _method_views(
    address_query: Tensor,
    evidence: Tensor,
    address_memory: Tensor,
    content_memory: Tensor,
    config: ChevronAttentionConfig,
) -> tuple[dict[str, dict[str, Tensor]], Any]:
    chevron = chevron_attention(
        address_query,
        evidence,
        address_memory,
        content_memory,
        config=config,
    )
    alpha = chevron.alpha
    retrieval_read, retrieval_q, retrieval_write = _retrieval_twice(
        alpha, address_query, address_memory, config
    )
    views = {
        "standard_attention": {
            "read_mass": alpha,
            "q": torch.zeros_like(chevron.total_residual),
            "write_gate": alpha,
        },
        "retrieval_twice": {
            "read_mass": retrieval_read,
            "q": retrieval_q,
            "write_gate": retrieval_write,
        },
        "chevron": {
            "read_mass": chevron.read_mass,
            "q": chevron.total_residual,
            "write_gate": chevron.write_gate,
        },
    }
    for view in views.values():
        maximum_admitted = view["read_mass"].max(dim=-1).values
        view["allocate"] = (
            (view["q"] > config.allocation_residual_threshold)
            & (maximum_admitted < config.allocation_admitted_threshold)
        )
    return views, chevron


def _mean(records: list[dict[str, Any]], field: str) -> float:
    return sum(float(record[field]) for record in records) / len(records)


def _percent(records: list[dict[str, Any]], field: str) -> float:
    return 100.0 * _mean(records, field)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in METHODS:
        method_records = [record for record in records if record["method"] == method]
        match = [record for record in method_records if record["kind"] == "match"]
        no_match = [record for record in method_records if record["kind"] == "no_match"]
        by_noise: dict[str, Any] = {}
        for noise in sorted({float(record["noise"]) for record in method_records}):
            subset = [record for record in method_records if float(record["noise"]) == noise]
            by_noise[f"{noise:.2f}"] = {
                "accuracy_pct": _percent(subset, "correct"),
                "match_accuracy_pct": _percent(
                    [record for record in subset if record["kind"] == "match"], "correct"
                ),
                "no_match_accuracy_pct": _percent(
                    [record for record in subset if record["kind"] == "no_match"], "correct"
                ),
            }
        result[method] = {
            "queries": len(method_records),
            "overall_accuracy_pct": _percent(method_records, "correct"),
            "match_accuracy_pct": _percent(match, "correct"),
            "no_match_accuracy_pct": _percent(no_match, "correct"),
            "mean_match_target_read_mass": _mean(match, "target_read_mass"),
            "mean_match_non_target_read_mass": _mean(match, "non_target_read_mass"),
            "mean_no_match_residual": _mean(no_match, "q"),
            "no_match_provisional_allocation_pct": _percent(no_match, "allocate"),
            "match_false_allocation_pct": _percent(match, "allocate"),
            "mean_no_match_memory_drift_l2": _mean(no_match, "memory_drift_l2"),
            "mean_match_target_write_gate": _mean(match, "target_write_gate"),
            "mean_match_non_target_write_gate": _mean(match, "non_target_write_gate"),
            "by_evidence_noise": by_noise,
        }
    return result


def run_experiment(
    *,
    seeds: int = 50,
    groups: int = 8,
    content_dim: int = 16,
    evidence_noise_levels: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10),
    address_noise: float = 0.02,
) -> dict[str, Any]:
    config = ChevronAttentionConfig()
    records: list[dict[str, Any]] = []
    intervention_stats: dict[str, list[float]] = defaultdict(list)

    for seed in range(seeds):
        generator = torch.Generator().manual_seed(seed)
        address_memory = torch.eye(groups).repeat_interleave(2, dim=0) * 4.0
        content_memory = F.normalize(
            torch.randn(groups * 2, content_dim, generator=generator), dim=-1
        )

        for noise in evidence_noise_levels:
            for group in range(groups):
                pair = (2 * group, 2 * group + 1)
                base_address = torch.eye(groups)[group] * 4.0
                address_query = base_address + address_noise * _normal(base_address, generator)

                evidences = [
                    F.normalize(
                        content_memory[target]
                        + noise * _normal(content_memory[target], generator),
                        dim=-1,
                    )
                    for target in pair
                ]
                evidences.append(
                    _novel_evidence(content_memory[list(pair)], content_dim, generator)
                )

                chevron_outputs = []
                retrieval_twice_assents = []
                for query_index, evidence in enumerate(evidences):
                    target = pair[query_index] if query_index < 2 else -1
                    kind = "match" if target >= 0 else "no_match"
                    views, chevron = _method_views(
                        address_query,
                        evidence,
                        address_memory,
                        content_memory,
                        config,
                    )
                    chevron_outputs.append(chevron)
                    address_mismatch = normalized_cosine_mismatch(
                        address_query.unsqueeze(0),
                        address_memory.unsqueeze(0),
                        eps=config.eps,
                    )
                    retrieval_twice_assents.append(
                        torch.sigmoid(
                            config.read_slope
                            * (config.read_threshold - address_mismatch)
                        )
                    )

                    for method, view in views.items():
                        read_mass = view["read_mass"].squeeze(0)
                        q = float(view["q"].item())
                        prediction = (
                            -1
                            if q > config.allocation_residual_threshold
                            else int(read_mass.argmax().item())
                        )
                        write_gate = view["write_gate"].squeeze(0)
                        _, updated_content = apply_convex_write(
                            address_memory,
                            content_memory,
                            address_query,
                            evidence,
                            write_gate,
                            eta_address=0.20,
                            eta_content=0.10,
                        )
                        drift = torch.linalg.vector_norm(
                            updated_content - content_memory, dim=-1
                        ).sum()

                        target_read = float(read_mass[target].item()) if target >= 0 else 0.0
                        non_target_read = (
                            float((read_mass.sum() - read_mass[target]).item())
                            if target >= 0
                            else float(read_mass.sum().item())
                        )
                        target_write = (
                            float(write_gate[target].item()) if target >= 0 else 0.0
                        )
                        non_target_write = (
                            float((write_gate.sum() - write_gate[target]).item())
                            if target >= 0
                            else float(write_gate.sum().item())
                        )
                        records.append(
                            {
                                "method": method,
                                "seed": seed,
                                "noise": noise,
                                "group": group,
                                "kind": kind,
                                "target": target,
                                "prediction": prediction,
                                "correct": float(prediction == target),
                                "target_read_mass": target_read,
                                "non_target_read_mass": non_target_read,
                                "q": q,
                                "allocate": float(view["allocate"].item()),
                                "memory_drift_l2": float(drift.item()),
                                "target_write_gate": target_write,
                                "non_target_write_gate": non_target_write,
                            }
                        )

                base_alpha = chevron_outputs[0].alpha
                intervention_stats["alpha_max_change"].append(
                    max(
                        float((output.alpha - base_alpha).abs().max().item())
                        for output in chevron_outputs[1:]
                    )
                )
                intervention_stats["chevron_assent_switch_margin"].extend(
                    [
                        float(
                            (
                                chevron_outputs[0].read_assent[0, pair[0]]
                                - chevron_outputs[0].read_assent[0, pair[1]]
                            ).item()
                        ),
                        float(
                            (
                                chevron_outputs[1].read_assent[0, pair[1]]
                                - chevron_outputs[1].read_assent[0, pair[0]]
                            ).item()
                        ),
                    ]
                )
                base_retrieval_assent = retrieval_twice_assents[0]
                intervention_stats["retrieval_twice_assent_max_change"].append(
                    max(
                        float((assent - base_retrieval_assent).abs().max().item())
                        for assent in retrieval_twice_assents[1:]
                    )
                )

    summary = _aggregate(records)
    return {
        "experiment": "001_causal_factorisation",
        "status": "diagnostic_not_rl",
        "protocol": {
            "seeds": seeds,
            "groups_per_seed": groups,
            "slots_per_group": 2,
            "content_dimension": content_dim,
            "evidence_noise_levels": list(evidence_noise_levels),
            "address_noise": address_noise,
            "queries_per_method": len(records) // len(METHODS),
            "read_threshold": config.read_threshold,
            "write_threshold": config.write_threshold,
            "read_slope": config.read_slope,
            "write_slope": config.write_slope,
            "allocation_residual_threshold": config.allocation_residual_threshold,
            "allocation_admitted_threshold": config.allocation_admitted_threshold,
        },
        "causal_interventions": {
            key: {
                "mean": sum(values) / len(values),
                "maximum": max(values),
                "minimum": min(values),
            }
            for key, values in intervention_stats.items()
        },
        "methods": summary,
        "claim_boundary": (
            "This engineered diagnostic tests the declared equations and causal "
            "separation. It does not show that an RL optimizer will learn the "
            "factorisation from reward, or that Chevron improves game performance."
        ),
    }


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "experiment_001_results.json"
    markdown_path = output_dir / "experiment_001_report.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Experiment 001: causal factorisation",
        "",
        "## Question",
        "",
        "Can retrieval remain fixed while independent A/N compatibility changes",
        "read assent, residual mass, allocation, and write permission?",
        "",
        "## Protocol",
        "",
        f"- Seeds: {result['protocol']['seeds']}",
        f"- Queries per method: {result['protocol']['queries_per_method']}",
        "- Each address retrieves a two-slot family and cannot distinguish its members.",
        "- Diagnostic evidence matches the first member, the second member, or neither.",
        "- No parameters are trained; this is a causal mathematical diagnostic, not RL.",
        "",
        "## Results",
        "",
        "| Method | Overall accuracy | Match | No match | No-match q | No-match N drift |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = result["methods"][method]
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES[method],
                    _fmt(metrics["overall_accuracy_pct"]) + "%",
                    _fmt(metrics["match_accuracy_pct"]) + "%",
                    _fmt(metrics["no_match_accuracy_pct"]) + "%",
                    _fmt(metrics["mean_no_match_residual"], 4),
                    _fmt(metrics["mean_no_match_memory_drift_l2"], 6),
                ]
            )
            + " |"
        )

    interventions = result["causal_interventions"]
    lines.extend(
        [
            "",
            "## Causal checks",
            "",
            "- Maximum change in retrieval alpha when only diagnostic evidence changed: "
            + _fmt(interventions["alpha_max_change"]["maximum"], 8),
            "- Mean Chevron assent switch margin: "
            + _fmt(interventions["chevron_assent_switch_margin"]["mean"], 6),
            "- Maximum change in retrieval-twice assent under the same intervention: "
            + _fmt(
                interventions["retrieval_twice_assent_max_change"]["maximum"], 8
            ),
        ]
    )

    stress = result.get("stress_test")
    if stress is not None:
        lines.extend(
            [
                "",
                "## Evidence-noise stress test",
                "",
                f"Additional seeds: {stress['protocol']['seeds']}",
                "",
                "| Evidence noise | Chevron overall | Chevron match | Chevron no match |",
                "|---:|---:|---:|---:|",
            ]
        )
        chevron_noise = stress["methods"]["chevron"]["by_evidence_noise"]
        for noise, metrics in chevron_noise.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        noise,
                        _fmt(metrics["accuracy_pct"]) + "%",
                        _fmt(metrics["match_accuracy_pct"]) + "%",
                        _fmt(metrics["no_match_accuracy_pct"]) + "%",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Allocation and write selectivity",
            "",
            "| Method | Allocate on no match | False allocation on match | Target write | Non-target write |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        metrics = result["methods"][method]
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES[method],
                    _fmt(metrics["no_match_provisional_allocation_pct"]) + "%",
                    _fmt(metrics["match_false_allocation_pct"]) + "%",
                    _fmt(metrics["mean_match_target_write_gate"], 6),
                    _fmt(metrics["mean_match_non_target_write_gate"], 6),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            result["claim_boundary"],
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    result = run_experiment()
    stress = run_experiment(
        seeds=20,
        evidence_noise_levels=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
    )
    result["stress_test"] = {
        "protocol": stress["protocol"],
        "methods": stress["methods"],
    }
    output_dir = Path(__file__).resolve().parent / "results"
    write_report(result, output_dir)
    for method in METHODS:
        metrics = result["methods"][method]
        print(
            f"{DISPLAY_NAMES[method]:22s} "
            f"accuracy={metrics['overall_accuracy_pct']:6.2f}% "
            f"no_match={metrics['no_match_accuracy_pct']:6.2f}% "
            f"q={metrics['mean_no_match_residual']:.4f} "
            f"drift={metrics['mean_no_match_memory_drift_l2']:.6f}"
        )


if __name__ == "__main__":
    main()
