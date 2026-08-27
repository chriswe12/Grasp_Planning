#!/usr/bin/env python3
"""Render nominal and perturbed object-part poses through the training reset path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target-index", type=int, default=0)
parser.add_argument("--reset-progress", type=float, default=0.50)
parser.add_argument("--seed", type=int, default=29)
parser.add_argument("--settle-steps", type=int, default=3)
parser.add_argument(
    "--case",
    choices=("nominal", "translation_only", "yaw_only", "translation_and_yaw"),
    default="nominal",
    help="Render one isolated case. The comparison sheet is assembled after all four cases exist.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/object_pose_randomization_check"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaac_rl.tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK_ID = "Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0"
CASES = (
    ("nominal", False, False, 0.0, 0.0),
    ("translation_only", True, False, 1.0, 0.0),
    ("yaw_only", False, True, 0.0, 1.0),
    ("translation_and_yaw", True, True, 0.60, 1.0),
)
CASE_BY_NAME = {case[0]: case for case in CASES}


def _rgb_uint8(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _policy_rgb(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _rotation_angle_deg(left: torch.Tensor, right: torch.Tensor) -> float:
    relative = left.transpose(0, 1) @ right
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).item())


def _capture_case(task, case_name: str) -> dict[str, object]:
    with torch.inference_mode():
        for _ in range(int(args.settle_steps)):
            task.scene.write_data_to_sim()
            task.sim.step()
            task.scene.update(task.sim.get_physics_dt())
        task.sim.forward()
        task.sim.render()
        task.sim.render()
        task.wrist_camera.update(task.sim.get_physics_dt(), force_recompute=True)
        task.debug_camera.update(task.sim.get_physics_dt(), force_recompute=True)
        policy_visual = task._camera_observation(randomize_live=False)[0]

    target_index = int(task.target_index[0].item())
    part_index = int(task.target_part_indices[target_index].item())
    active_part = task.parts[part_index]
    object_pose = active_part.data.root_pose_w[0].detach()
    env_origin = task.scene.env_origins[0]
    object_position = object_pose[:3] - env_origin
    object_quaternion = object_pose[3:7]
    goal_position = task.goal_tcp_position[0].detach() - env_origin
    goal_quaternion = task.goal_tcp_quaternion[0].detach()
    nominal_object_position = task.object_positions_catalog[target_index]
    nominal_object_quaternion = task.object_quaternions_catalog[target_index]
    nominal_goal_position = task.goal_tcp_positions_catalog[target_index]
    nominal_goal_quaternion = task.goal_tcp_quaternions_catalog[target_index]

    rotation_object = matrix_from_quat(object_quaternion.unsqueeze(0))[0]
    rotation_goal = matrix_from_quat(goal_quaternion.unsqueeze(0))[0]
    rotation_nominal_object = matrix_from_quat(nominal_object_quaternion.unsqueeze(0))[0]
    rotation_nominal_goal = matrix_from_quat(nominal_goal_quaternion.unsqueeze(0))[0]
    relative_position = rotation_object.transpose(0, 1) @ (goal_position - object_position)
    nominal_relative_position = rotation_nominal_object.transpose(0, 1) @ (
        nominal_goal_position - nominal_object_position
    )
    relative_rotation = rotation_object.transpose(0, 1) @ rotation_goal
    nominal_relative_rotation = rotation_nominal_object.transpose(0, 1) @ rotation_nominal_goal
    relative_position_error_mm = float(
        torch.linalg.norm(relative_position - nominal_relative_position).item() * 1000.0
    )
    relative_rotation_error_deg = _rotation_angle_deg(relative_rotation, nominal_relative_rotation)

    return {
        "case": case_name,
        "target_index": target_index,
        "target_id": task.target_ids[target_index],
        "part_id": task.part_names[part_index],
        "translation_xy_mm": (
            task.reset_position_offset[0, :2].detach().cpu().numpy() * 1000.0
        ).tolist(),
        "translation_magnitude_mm": float(
            torch.linalg.norm(task.reset_position_offset[0]).item() * 1000.0
        ),
        "object_yaw_deg": float(torch.rad2deg(task.reset_object_yaw_offset[0]).item()),
        "requested_object_yaw_deg": float(
            torch.rad2deg(task.reset_object_yaw_requested[0]).item()
        ),
        "safe_object_yaw_cap_deg": float(
            torch.rad2deg(task.reset_object_yaw_safe_cap[0]).item()
        ),
        "initial_position_error_mm": float(task.initial_position_error[0].item() * 1000.0),
        "initial_rotation_error_deg": float(torch.rad2deg(task.initial_rotation_error[0]).item()),
        "part_relative_target_position_error_mm": relative_position_error_mm,
        "part_relative_target_rotation_error_deg": relative_rotation_error_deg,
        "collision": bool(task._gripper_collision()[0].item()),
        "overview": _rgb_uint8(task.debug_camera.data.output["rgb"][0]),
        "live_rgb": _policy_rgb(policy_visual[..., :3]),
        "goal_rgb": _policy_rgb(policy_visual[..., 4:7]),
    }


def _write_sheet(captures: list[dict[str, object]], output: Path) -> None:
    canvas = Image.new("RGB", (1500, 1270), (13, 16, 21))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "STABLE ACTUAL PART vs NOMINAL POSE ESTIMATE", fill=(246, 248, 251))
    draw.text(
        (18, 34),
        "Actual part varies only in support-plane XY/yaw; Z and roll/pitch stay stable. Target follows part; goal stays fixed.",
        fill=(170, 181, 195),
    )
    draw.text((20, 62), "EXTERNAL SCENE", fill=(225, 231, 238))
    draw.text((790, 62), "LIVE POLICY RGB", fill=(225, 231, 238))
    draw.text((1130, 62), "CANONICAL GOAL RGB", fill=(225, 231, 238))

    for row, capture in enumerate(captures):
        y = 88 + row * 292
        overview = Image.fromarray(capture["overview"]).resize((720, 270), Image.Resampling.LANCZOS)
        live = Image.fromarray(capture["live_rgb"]).resize((320, 180), Image.Resampling.NEAREST)
        goal = Image.fromarray(capture["goal_rgb"]).resize((320, 180), Image.Resampling.NEAREST)
        canvas.paste(overview, (20, y))
        canvas.paste(live, (790, y))
        canvas.paste(goal, (1130, y))
        xy = capture["translation_xy_mm"]
        collision_text = "COLLISION" if capture["collision"] else "collision-free"
        draw.text(
            (790, y + 190),
            (
                f"{capture['case']}  XY=({xy[0]:+.2f}, {xy[1]:+.2f}) mm  "
                f"yaw={capture['object_yaw_deg']:+.2f} deg  {collision_text}"
            ),
            fill=(245, 198, 105) if capture["collision"] else (119, 225, 159),
        )
        draw.text(
            (790, y + 214),
            (
                f"initial error={capture['initial_position_error_mm']:.2f} mm / "
                f"{capture['initial_rotation_error_deg']:.2f} deg"
            ),
            fill=(188, 198, 212),
        )
        draw.text(
            (790, y + 238),
            (
                "part-relative target invariant error="
                f"{capture['part_relative_target_position_error_mm']:.5f} mm / "
                f"{capture['part_relative_target_rotation_error_deg']:.5f} deg"
            ),
            fill=(127, 202, 255),
        )
    canvas.save(output)


def _assemble_existing_cases(output_dir: Path) -> bool:
    captures: list[dict[str, object]] = []
    canonical_goal: np.ndarray | None = None
    for case_name, *_ in CASES:
        case_dir = output_dir / case_name
        metadata_path = case_dir / "metadata.json"
        paths = {
            "overview": case_dir / "scene_overview.png",
            "live_rgb": case_dir / "policy_live_rgb_native.png",
            "goal_rgb": case_dir / "policy_goal_rgb_native.png",
        }
        if not metadata_path.is_file() or any(not path.is_file() for path in paths.values()):
            return False
        capture = json.loads(metadata_path.read_text(encoding="utf-8"))
        capture.update({key: np.asarray(Image.open(path).convert("RGB")) for key, path in paths.items()})
        if canonical_goal is None:
            canonical_goal = capture["goal_rgb"]
        elif not np.array_equal(canonical_goal, capture["goal_rgb"]):
            raise RuntimeError("Canonical goal RGB changed across object-pose perturbations.")
        captures.append(capture)

    serializable = [
        {key: value for key, value in capture.items() if key not in {"overview", "live_rgb", "goal_rgb"}}
        for capture in captures
    ]
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task": TASK_ID,
                "seed": int(args.seed),
                "reset_progress": float(args.reset_progress),
                "cases": serializable,
                "canonical_goal_rgb_identical": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_sheet(captures, output_dir / "object_pose_randomization_check.png")
    return True


def main() -> None:
    if args.target_index < 0:
        raise ValueError("--target-index must be non-negative.")
    if not 0.0 <= args.reset_progress <= 1.0:
        raise ValueError("--reset-progress must lie in [0, 1].")
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    cfg.seed = int(args.seed)
    cfg.scene.replicate_physics = False
    cfg.debug_camera_enabled = True
    cfg.catalog_split = "all"
    cfg.fixed_target_index = int(args.target_index)
    cfg.training_curriculum_enabled = False
    cfg.training_reset_mixture_enabled = False
    cfg.variable_reset_timeouts_enabled = False
    cfg.failure_replay_fraction = 0.0
    cfg.completion_positive_reset_fraction = 0.0
    cfg.reset_progress_min = float(args.reset_progress)
    cfg.reset_progress_max = float(args.reset_progress)
    cfg.reset_rotation_randomization_enabled = True
    cfg.reset_rotation_fraction_min = 0.0
    cfg.reset_rotation_fraction_max = 0.0
    case_name, translate, yaw, translation_fraction, yaw_fraction = CASE_BY_NAME[args.case]
    cfg.reset_position_randomization_enabled = translate
    cfg.reset_object_yaw_randomization_enabled = yaw
    cfg.reset_position_fraction_min = translation_fraction
    cfg.reset_position_fraction_max = translation_fraction
    cfg.reset_object_yaw_fraction_min = yaw_fraction
    cfg.reset_object_yaw_fraction_max = yaw_fraction
    cfg.live_observation_randomization_enabled = False
    cfg.scene_appearance_randomization_enabled = False
    cfg.scene_tslot_geometry_randomization_enabled = False

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env = gym.make(TASK_ID, cfg=cfg)
        task = env.unwrapped
        env.reset()
        capture = _capture_case(task, case_name)
        if capture["collision"]:
            raise RuntimeError(f"Case {case_name} initialized in collision.")
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        serializable = {
            key: value for key, value in capture.items() if key not in {"overview", "live_rgb", "goal_rgb"}
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(serializable, indent=2) + "\n",
            encoding="utf-8",
        )
        Image.fromarray(capture["overview"]).save(case_dir / "scene_overview.png")
        Image.fromarray(capture["live_rgb"]).save(case_dir / "policy_live_rgb_native.png")
        Image.fromarray(capture["goal_rgb"]).save(case_dir / "policy_goal_rgb_native.png")
        Image.fromarray(capture["live_rgb"]).resize(
            (512, 288), Image.Resampling.NEAREST
        ).save(case_dir / "policy_live_rgb.png")
        Image.fromarray(capture["goal_rgb"]).resize(
            (512, 288), Image.Resampling.NEAREST
        ).save(case_dir / "policy_goal_rgb.png")
        assembled = _assemble_existing_cases(output_dir)
        print(
            f"[DONE] rendered collision-free case {case_name} to {case_dir}; "
            f"comparison_sheet_ready={assembled}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
