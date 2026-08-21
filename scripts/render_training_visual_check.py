#!/usr/bin/env python3
"""Render one exact named training profile and its policy RGB-D input."""

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
parser.add_argument("--seed", type=int, default=19)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument(
    "--sim2real-profile",
    choices=("combined_sim2real", "combined_clutter", "combined_depth_robust"),
    default="combined_sim2real",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/training_visual_check"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaac_rl.tasks  # noqa: E402, F401
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from grasp_planning.isaac_visual_materials import VISUAL_SERVO_MATERIAL_PROFILE  # noqa: E402
from grasp_planning.isaac_visual_scene import VISUAL_SERVO_SCENE_PROFILE  # noqa: E402
from grasp_planning.rl.sim2real_profiles import apply_sim2real_profile  # noqa: E402
from grasp_planning.visual_servo_workspace import VISUAL_SERVO_TSLOT_PROFILE  # noqa: E402

TASK_ID = "Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0"


def _rgb_uint8(value: torch.Tensor) -> np.ndarray:
    array = value[..., :3].detach().cpu().numpy()
    if array.dtype == np.uint8:
        return array
    if float(np.max(array)) <= 1.5:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _policy_rgb(value: np.ndarray) -> np.ndarray:
    return np.round(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def _depth_color(normalized_depth: np.ndarray) -> np.ndarray:
    value = np.clip(normalized_depth, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    result = np.stack((red, green, blue), axis=-1)
    result[value >= 1.0] = 0.0
    return np.round(result * 255.0).astype(np.uint8)


def _save_scaled(array: np.ndarray, path: Path, *, scale: int, nearest: bool = False) -> None:
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    image = Image.fromarray(array)
    image.resize((image.width * scale, image.height * scale), resampling).save(path)


def _make_sheet(
    overview: np.ndarray,
    wrist_raw: np.ndarray,
    live_rgb: np.ndarray,
    live_depth: np.ndarray,
    goal_rgb: np.ndarray,
    goal_depth: np.ndarray,
    output: Path,
    *,
    profile: str,
    clutter_count: int,
) -> None:
    canvas = Image.new("RGB", (1280, 1050), (16, 19, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), f"ACTUAL TRAINING VISUAL CHECK - {profile}", fill=(245, 247, 250))

    scene = Image.fromarray(overview).resize((960, 540), Image.Resampling.LANCZOS)
    canvas.paste(scene, (20, 42))
    draw.text((1000, 58), "External check camera", fill=(230, 235, 241))
    draw.text((1000, 82), "Not given to policy", fill=(165, 176, 190))
    draw.text((1000, 112), f"Clutter prims: {clutter_count}", fill=(115, 220, 150))
    draw.text((1000, 136), "Collision: flat plane", fill=(165, 176, 190))

    raw = Image.fromarray(wrist_raw).resize((384, 216), Image.Resampling.LANCZOS)
    canvas.paste(raw, (20, 620))
    draw.text((20, 594), "Raw wrist render before policy preprocessing", fill=(230, 235, 241))

    panels = (
        ("Policy live RGB (128 x 72)", live_rgb),
        ("Policy live normalized depth", live_depth),
        ("Policy goal RGB (128 x 72)", goal_rgb),
        ("Policy goal normalized depth", goal_depth),
    )
    for index, (label, array) in enumerate(panels):
        column = index % 2
        row = index // 2
        x = 440 + column * 410
        y = 620 + row * 205
        draw.text((x, y - 26), label, fill=(230, 235, 241))
        panel = Image.fromarray(array).resize((384, 216), Image.Resampling.NEAREST)
        panel.thumbnail((384, 170), Image.Resampling.NEAREST)
        canvas.paste(panel, (x, y))
    canvas.save(output)


def main() -> None:
    if args.target_index < 0:
        raise ValueError("--target-index must be non-negative.")
    if args.settle_steps < 1:
        raise ValueError("--settle-steps must be positive.")

    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    profile = apply_sim2real_profile(cfg, args.sim2real_profile)
    cfg.seed = int(args.seed)
    cfg.scene.replicate_physics = False
    cfg.debug_camera_enabled = True
    cfg.catalog_split = "all"
    cfg.fixed_target_index = int(args.target_index)
    cfg.training_curriculum_enabled = False
    cfg.training_reset_mixture_enabled = False
    cfg.variable_reset_timeouts_enabled = False
    cfg.completion_positive_reset_fraction = 1.0
    cfg.reset_ready_exact_fraction = 1.0
    cfg.failure_replay_fraction = 0.0
    cfg.reset_rotation_randomization_enabled = False
    cfg.reset_position_randomization_enabled = False
    cfg.require_rotation_reset_data = False
    # Use the most common training layout so the corrected 5 mm slots can be
    # judged directly. Appearance and D405 sensor randomization remain active.
    cfg.scene_tslot_geometry_randomization_enabled = False

    env = None
    try:
        env = gym.make(TASK_ID, cfg=cfg)
        env.reset()
        task = env.unwrapped
        with torch.inference_mode():
            for _ in range(args.settle_steps):
                task.scene.write_data_to_sim()
                task.sim.step()
                task.scene.update(task.sim.get_physics_dt())
                task.wrist_camera.update(task.sim.get_physics_dt(), force_recompute=True)
                task.debug_camera.update(task.sim.get_physics_dt(), force_recompute=True)

            policy_visual = task._camera_observation(randomize_live=True)[0].detach().cpu().numpy()

        stage = omni.usd.get_context().get_stage()
        clutter_prims = tuple(
            str(prim.GetPath())
            for prim in stage.Traverse()
            if "/VisualClutter/Object_" in str(prim.GetPath())
        )
        if args.sim2real_profile == "combined_clutter" and not clutter_prims:
            raise RuntimeError("combined_clutter did not author any visual clutter prims.")
        if args.sim2real_profile != "combined_clutter" and clutter_prims:
            raise RuntimeError(
                f"Profile {args.sim2real_profile} unexpectedly contains clutter/distractor prims: {clutter_prims}"
            )

        overview = _rgb_uint8(task.debug_camera.data.output["rgb"])[0]
        wrist_raw = _rgb_uint8(task.wrist_camera.data.output["rgb"])[0]
        live_rgb = _policy_rgb(policy_visual[..., :3])
        live_depth = _depth_color(policy_visual[..., 3])
        goal_rgb = _policy_rgb(policy_visual[..., 4:7])
        goal_depth = _depth_color(policy_visual[..., 7])

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overview).save(output_dir / "scene_overview.png")
        _save_scaled(wrist_raw, output_dir / "wrist_rgb_raw.png", scale=4)
        _save_scaled(live_rgb, output_dir / "policy_live_rgb.png", scale=4, nearest=True)
        _save_scaled(live_depth, output_dir / "policy_live_depth.png", scale=4, nearest=True)
        _save_scaled(goal_rgb, output_dir / "policy_goal_rgb.png", scale=4, nearest=True)
        _save_scaled(goal_depth, output_dir / "policy_goal_depth.png", scale=4, nearest=True)
        _make_sheet(
            overview,
            wrist_raw,
            live_rgb,
            live_depth,
            goal_rgb,
            goal_depth,
            output_dir / "training_visual_check.png",
            profile=profile.identifier,
            clutter_count=len(clutter_prims),
        )
        metadata = {
            "task": TASK_ID,
            "target_index": int(args.target_index),
            "target_id": task.target_ids[int(task.target_index[0])],
            "seed": int(args.seed),
            "sim2real_profile": profile.identifier,
            "material_profile": VISUAL_SERVO_MATERIAL_PROFILE,
            "scene_profile": VISUAL_SERVO_SCENE_PROFILE,
            "tslot_profile": VISUAL_SERVO_TSLOT_PROFILE,
            "policy_visual_shape_hwc": list(policy_visual.shape),
            "tslot_layout": "nominal",
            "clutter_profile": task.clutter_visual_bindings["profile"],
            "clutter_prim_count": len(task.clutter_visual_bindings["prim_paths"]),
            "clutter_active_environment_count": task.clutter_visual_bindings["active_environment_count"],
            "collision_surface": task.tslot_visual_bindings["collision_surface"],
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(
            f"[DONE] target={metadata['target_id']} clutter={metadata['clutter_prim_count']} "
            f"policy_shape={tuple(policy_visual.shape)} output={output_dir}",
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
