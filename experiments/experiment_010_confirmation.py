"""Fresh-seed confirmation for Experiment 010 retrospective policy assent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.experiment_004_reward_memory import _json_safe
from experiments.experiment_010_retrospective_policy import (
    CONDITIONS,
    DISPLAY_NAMES,
    ExperimentConfig,
    _development_gate,
    _pm,
    run_development,
)


def run_confirmation(
    config: ExperimentConfig,
    *,
    seeds: int = 100,
    seed_offset: int = 101_000_000,
) -> dict[str, Any]:
    if seed_offset < 101_000_000:
        raise ValueError("confirmation seeds must start at 101,000,000 or later")
    result = run_development(
        config,
        seeds=seeds,
        seed_offset=seed_offset,
    )
    result["status"] = "fresh_seed_confirmation"
    result["confirmation_gate"] = _development_gate(result)
    result["confirmation_passed"] = all(result["confirmation_gate"].values())
    result.pop("development_gate", None)
    result.pop("confirmation_triggered", None)
    return result


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 010: retrospective-policy confirmation",
        "",
        f"- Fresh seeds: {result['seed_offset']}–{result['seed_offset'] + result['seeds'] - 1}",
        "- Policy signature in observation: **none**",
        "",
        "| Condition | Return | Clean accuracy | Retention | Reversed probe | Novel probe | False revisions | Detection occurrences |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics = result["aggregate"][condition]
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | {_pm(metrics['return_per_decision'])} | "
            f"{_pm(metrics['clean_accuracy'])} | {_pm(metrics['retention_accuracy'])} | "
            f"{_pm(metrics['reversed_probe_accuracy'])} | "
            f"{_pm(metrics['novel_probe_accuracy'])} | "
            f"{_pm(metrics['false_stable_revisions'])} | "
            f"{_pm(metrics['mean_reversal_detection_occurrences'])} |"
        )
    lines.extend(["", "## Frozen confirmation gate", ""])
    for name, passed in result["confirmation_gate"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            f"Confirmation passed: **{result['confirmation_passed']}**",
            "",
            "## Paired diagnostics",
            "",
            "```json",
            json.dumps(_json_safe(result["paired"]), indent=2, allow_nan=False),
            "```",
            "",
        ]
    )
    (output_dir / "experiment_010_confirmation_results.json").write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "experiment_010_confirmation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=101_000_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
    )
    args = parser.parse_args()
    result = run_confirmation(
        ExperimentConfig(),
        seeds=args.seeds,
        seed_offset=args.seed_offset,
    )
    write_report(result, args.output_dir)
    print(
        json.dumps(
            _json_safe(
                {
                    "aggregate": result["aggregate"],
                    "paired": result["paired"],
                    "confirmation_gate": result["confirmation_gate"],
                    "confirmation_passed": result["confirmation_passed"],
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
