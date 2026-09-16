#!/usr/bin/env python3
"""Record matched single-IK and multi-IK Isaac pickups at several locations."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    ("center", 0.50, 0.00),
    ("left", 0.48, 0.12),
    ("right", 0.48, -0.12),
    ("far_center", 0.68, 0.00),
    ("far_left", 0.60, 0.30),
    ("far_right", 0.60, -0.30),
    ("wide_left", 0.42, 0.34),
    ("wide_right", 0.42, -0.34),
)
ISAAC_PYTHON = Path("/media/pdz/Elements1/IsaacLab/_isaac_sim/python.sh")


def _case_config(
    base: dict[str, object],
    *,
    case_name: str,
    x: float,
    y: float,
    strategy: str,
    output_dir: Path,
) -> dict[str, object]:
    payload = deepcopy(base)
    pose = dict(payload["execution_world_pose"])
    position = list(pose["position_world"])
    position[0:2] = [float(x), float(y)]
    pose["position_world"] = position
    payload["execution_world_pose"] = pose

    artifacts = dict(payload.get("artifacts", {}))
    for key, suffix in (
        ("stage1_json", "stage1.json"),
        ("stage1_html", "stage1.html"),
        ("stage2_json", "stage2.json"),
        ("stage2_html", "stage2.html"),
        ("part_frame_html", "part_frame.html"),
    ):
        artifacts[key] = str(output_dir / case_name / strategy / suffix)
    payload["artifacts"] = artifacts

    mujoco = dict(payload.get("mujoco_execution", {}))
    mujoco["enabled"] = False
    payload["mujoco_execution"] = mujoco
    isaac = dict(payload.get("isaac_execution", {}))
    isaac["enabled"] = True
    isaac["headless"] = True
    isaac["attempt_artifact"] = str(output_dir / case_name / strategy / "attempt.json")
    isaac["record_video"] = str(output_dir / case_name / strategy / "raw.mp4")
    isaac["video_fps"] = 30.0
    isaac["video_width"] = 640
    isaac["video_height"] = 480
    isaac["video_camera_eye"] = [1.45, -1.05, 0.9]
    isaac["video_camera_target"] = [0.42, 0.0, 0.25]
    if strategy == "single_ik":
        isaac["moveit_ik_candidate_count"] = 1
        isaac["moveit_ik_beam_width"] = 1
        isaac["moveit_ik_joint_weights"] = []
    else:
        isaac["moveit_ik_candidate_count"] = 8
        isaac["moveit_ik_beam_width"] = 3
        isaac["moveit_ik_seed_perturbation_rad"] = 0.7
        isaac["moveit_ik_dedup_tolerance_rad"] = 0.05
        isaac["moveit_ik_joint_weights"] = [0.5, 2.0, 2.0, 2.5, 1.5, 1.5, 0.5]
    payload["isaac_execution"] = isaac
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grasp_pipeline_sim.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multi_ik_video_comparison"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    pair_outputs: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="multi-ik-video-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for case_name, x, y in DEFAULT_CASES:
            for strategy in ("single_ik", "multi_ik"):
                raw_video = output_dir / case_name / strategy / "raw.mp4"
                if args.skip_existing and raw_video.exists():
                    continue
                config = _case_config(
                    base,
                    case_name=case_name,
                    x=x,
                    y=y,
                    strategy=strategy,
                    output_dir=output_dir,
                )
                config_path = temporary_root / f"{case_name}_{strategy}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                subprocess.run(
                    [
                        str(REPO_ROOT / "run_pipeline.sh"),
                        "--mode",
                        "sim",
                        "--config",
                        str(config_path),
                        "--backend",
                        "isaac",
                        "--headless",
                    ],
                    check=True,
                    cwd=REPO_ROOT,
                )
            pair_output = output_dir / f"{case_name}_side_by_side.mp4"
            subprocess.run(
                [
                    str(ISAAC_PYTHON),
                    str(REPO_ROOT / "scripts/compose_multi_ik_comparison_videos.py"),
                    "--single",
                    str(output_dir / case_name / "single_ik" / "raw.mp4"),
                    "--multi",
                    str(output_dir / case_name / "multi_ik" / "raw.mp4"),
                    "--case-label",
                    f"{case_name}: x={x:.2f}, y={y:.2f}",
                    "--output",
                    str(pair_output),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            pair_outputs.append(pair_output)
    subprocess.run(
        [
            str(ISAAC_PYTHON),
            str(REPO_ROOT / "scripts/compose_multi_ik_comparison_videos.py"),
            "--concatenate",
            *(str(path) for path in pair_outputs),
            "--output",
            str(output_dir / "all_locations_side_by_side.mp4"),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
