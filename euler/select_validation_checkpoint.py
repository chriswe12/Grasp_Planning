#!/usr/bin/env python3
"""Rank periodic held-out validation reports and record the selected checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EPOCH_PATTERN = re.compile(r"epoch_(\d+)$")


def _weighted_metrics(summary: dict[str, object]) -> dict[str, float]:
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("Validation summary does not contain condition metrics.")
    attempts = 0
    successes = 0
    termination_counts: dict[str, int] = {}
    for raw_metrics in conditions.values():
        if not isinstance(raw_metrics, dict):
            raise ValueError("Condition metrics must be objects.")
        condition_attempts = int(raw_metrics.get("attempts", 0))
        attempts += condition_attempts
        successes += int(raw_metrics.get("successes", 0))
        raw_terminations = raw_metrics.get("terminations", {})
        if not isinstance(raw_terminations, dict):
            raise ValueError("Condition termination counts must be an object.")
        for name, count in raw_terminations.items():
            termination_counts[str(name)] = termination_counts.get(str(name), 0) + int(count)
    if attempts <= 0:
        raise ValueError("Validation summary contains no attempts.")

    def rate(name: str) -> float:
        return termination_counts.get(name, 0) / attempts

    metrics = {
        "attempts": float(attempts),
        "success_rate": successes / attempts,
        "collision_rate": rate("unsafe_collision"),
        "premature_completion_rate": rate("premature_completion"),
        "divergence_rate": rate("diverged"),
        "timeout_rate": rate("timeout") + rate("horizon"),
    }
    metrics["selection_score"] = (
        metrics["success_rate"]
        - metrics["collision_rate"]
        - 0.50 * metrics["premature_completion_rate"]
        - 0.25 * metrics["divergence_rate"]
        - 0.10 * metrics["timeout_rate"]
    )
    return metrics


def rank_validation_reports(root: Path) -> list[dict[str, object]]:
    """Return valid validation summaries ordered from best to worst."""

    records: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("epoch_*/summary.json")):
        match = EPOCH_PATTERN.fullmatch(summary_path.parent.name)
        if match is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("catalog_split") != "validation":
            raise ValueError(f"Refusing non-validation report: {summary_path}")
        reported_checkpoint = Path(str(summary.get("checkpoint", "")))
        # Evaluation runs inside a container and therefore records a
        # /workspace/grasping_rl path. Prefer the equivalent host checkpoint
        # beside this report so best_checkpoint.txt is directly executable.
        local_checkpoint = root.parent.parent / "nn" / reported_checkpoint.name
        checkpoint = local_checkpoint.resolve() if local_checkpoint.is_file() else reported_checkpoint
        metrics = _weighted_metrics(summary)
        records.append(
            {
                "epoch": int(match.group(1)),
                "checkpoint": str(checkpoint),
                "summary": str(summary_path.resolve()),
                **metrics,
            }
        )
    records.sort(
        key=lambda record: (
            float(record["selection_score"]),
            float(record["success_rate"]),
            -float(record["collision_rate"]),
            int(record["epoch"]),
        ),
        reverse=True,
    )
    return records


def write_selection(root: Path, records: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_rule": ("success - collision - 0.50*premature - 0.25*divergence - 0.10*(timeout+horizon)"),
        "best": records[0] if records else None,
        "ranked_checkpoints": records,
    }
    (root / "checkpoint_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Periodic validation checkpoint selection",
        "",
        "Score = success - collision - 0.50 premature - 0.25 divergence - 0.10 (timeout + horizon).",
        "",
        "| Rank | Epoch | Score | Success | Collision | Premature | Diverged | Timeout |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(records, start=1):
        lines.append(
            f"| {rank} | {record['epoch']} | {float(record['selection_score']):.4f} "
            f"| {100.0 * float(record['success_rate']):.1f}% "
            f"| {100.0 * float(record['collision_rate']):.1f}% "
            f"| {100.0 * float(record['premature_completion_rate']):.1f}% "
            f"| {100.0 * float(record['divergence_rate']):.1f}% "
            f"| {100.0 * float(record['timeout_rate']):.1f}% |"
        )
    (root / "checkpoint_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if records:
        (root / "best_checkpoint.txt").write_text(str(records[0]["checkpoint"]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_root", type=Path)
    args = parser.parse_args()
    root = args.validation_root.expanduser().resolve()
    records = rank_validation_reports(root)
    write_selection(root, records)
    if records:
        best = records[0]
        print(
            f"[VALIDATION] best epoch={best['epoch']} "
            f"score={float(best['selection_score']):.4f} "
            f"success={100.0 * float(best['success_rate']):.1f}%"
        )
    else:
        print(f"[VALIDATION] no reports found below {root}")


if __name__ == "__main__":
    main()
