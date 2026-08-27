#!/usr/bin/env python3
"""Validate first-curriculum NPZ episodes and report recovery statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REQUIRED_TIME_SERIES = (
    "rgb_live",
    "depth_live",
    "object_mask",
    "joint_positions",
    "nominal_twist",
    "expert_twist",
    "expert_residual_twist",
    "pose_error",
    "trajectory_progress",
)
REQUIRED_GOALS = ("rgb_goal", "depth_goal", "goal_object_mask")


def validate_episode(npz_path: Path) -> dict[str, object]:
    metadata_path = npz_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing metadata beside {npz_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as episode:
        missing = [name for name in (*REQUIRED_TIME_SERIES, *REQUIRED_GOALS) if name not in episode]
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {missing}")
        step_count = int(episode["rgb_live"].shape[0])
        if any(int(episode[name].shape[0]) != step_count for name in REQUIRED_TIME_SERIES):
            raise ValueError(f"{npz_path} has inconsistent time-series lengths.")
        for name in ("depth_live", "joint_positions", "nominal_twist", "expert_twist", "pose_error"):
            if not np.all(np.isfinite(episode[name])):
                raise ValueError(f"{npz_path} contains non-finite values in {name}.")
        if not np.allclose(
            episode["expert_twist"],
            episode["nominal_twist"] + episode["expert_residual_twist"],
            atol=1.0e-5,
        ):
            raise ValueError(f"{npz_path} has inconsistent residual action labels.")
        for name in ("object_mask", "goal_object_mask"):
            if not np.all(np.isin(episode[name], (0, 1))):
                raise ValueError(f"{npz_path} contains a non-binary target mask in {name}.")
    return {
        "episode": int(metadata["episode_index"]),
        "split": str(metadata["split"]),
        "success": bool(metadata["success"]),
        "steps": step_count,
        "final_position_error_m": float(metadata["final_position_error_m"]),
        "final_rotation_error_deg": float(metadata["final_rotation_error_deg"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    episodes = [validate_episode(path) for path in sorted(args.dataset_dir.glob("episode_*.npz"))]
    if not episodes:
        raise SystemExit(f"No episode NPZ files found under {args.dataset_dir}.")
    summary = {
        "episode_count": len(episodes),
        "success_count": sum(item["success"] for item in episodes),
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "mean_final_position_error_m": float(np.mean([item["final_position_error_m"] for item in episodes])),
        "mean_final_rotation_error_deg": float(np.mean([item["final_rotation_error_deg"] for item in episodes])),
        "episodes": episodes,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
