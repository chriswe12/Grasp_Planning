#!/usr/bin/env python3
"""Decide whether a long distributed PPO probe has stable GPU memory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0.0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator


def analyze_memory_logs(
    paths: list[Path],
    *,
    warmup_epochs: int,
    min_samples: int,
    max_growth_mib_per_epoch: float,
    min_free_mib: float,
) -> dict[str, object]:
    ranks: list[dict[str, object]] = []
    overall = "PASS"
    for path in sorted(paths):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            ranks.append({"path": str(path), "status": "INSUFFICIENT_SAMPLES", "samples": 0})
            overall = "INSUFFICIENT_SAMPLES" if overall == "PASS" else overall
            continue
        first_epoch = int(float(rows[0]["epoch"]))
        analyzed = [row for row in rows if int(float(row["epoch"])) >= first_epoch + warmup_epochs]
        if len(analyzed) < min_samples:
            ranks.append(
                {
                    "path": str(path),
                    "status": "INSUFFICIENT_SAMPLES",
                    "samples": len(analyzed),
                    "required_samples": min_samples,
                }
            )
            overall = "INSUFFICIENT_SAMPLES" if overall == "PASS" else overall
            continue
        epochs = [float(row["epoch"]) for row in analyzed]
        device_used = [float(row["device_used_mib"]) for row in analyzed]
        reserved = [float(row["reserved_mib"]) for row in analyzed]
        free = [float(row["device_free_mib"]) for row in analyzed]
        device_slope = _linear_slope(epochs, device_used)
        reserved_slope = _linear_slope(epochs, reserved)
        rank_status = "PASS"
        reasons: list[str] = []
        if device_slope > max_growth_mib_per_epoch:
            rank_status = "FAIL"
            reasons.append(
                f"device memory grows {device_slope:.3f} MiB/epoch "
                f"(limit {max_growth_mib_per_epoch:.3f})"
            )
        if min(free) < min_free_mib:
            rank_status = "FAIL"
            reasons.append(f"minimum free VRAM {min(free):.1f} MiB is below {min_free_mib:.1f} MiB")
        if rank_status == "FAIL":
            overall = "FAIL"
        ranks.append(
            {
                "path": str(path),
                "status": rank_status,
                "samples": len(analyzed),
                "epoch_start": int(epochs[0]),
                "epoch_end": int(epochs[-1]),
                "device_used_slope_mib_per_epoch": device_slope,
                "reserved_slope_mib_per_epoch": reserved_slope,
                "projected_device_growth_over_3000_epochs_mib": max(0.0, device_slope) * 3000.0,
                "minimum_free_mib": min(free),
                "final_free_mib": free[-1],
                "reasons": reasons,
            }
        )
    if not paths:
        overall = "INSUFFICIENT_SAMPLES"
    return {
        "status": overall,
        "criteria": {
            "warmup_epochs": warmup_epochs,
            "min_samples": min_samples,
            "max_growth_mib_per_epoch": max_growth_mib_per_epoch,
            "min_free_mib": min_free_mib,
        },
        "ranks": ranks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="RL-Games experiment directory containing gpu_memory_rank_*.csv")
    parser.add_argument("--warmup-epochs", type=int, default=25)
    parser.add_argument("--min-samples", type=int, default=150)
    parser.add_argument("--max-growth-mib-per-epoch", type=float, default=0.50)
    parser.add_argument("--min-free-mib", type=float, default=1024.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    payload = analyze_memory_logs(
        list(args.run_dir.glob("gpu_memory_rank_*.csv")),
        warmup_epochs=args.warmup_epochs,
        min_samples=args.min_samples,
        max_growth_mib_per_epoch=args.max_growth_mib_per_epoch,
        min_free_mib=args.min_free_mib,
    )
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 2 if payload["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
