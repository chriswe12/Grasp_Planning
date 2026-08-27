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
    "--reset-progress",
    type=float,
    default=None,
    help="Optional fixed approach-path progress in [0, 1]; default renders the exact successful pose.",
)
parser.add_argument(
    "--sim2real-profile",
    choices=(
        "combined_sim2real",
        "combined_clutter",
        "combined_busy_background",
        "combined_depth_robust",
    ),
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
    background_count: int,
    people_count: int,
    worker_reach_count: int,
    background_styles: dict[str, int],
) -> None:
    canvas = Image.new("RGB", (1280, 1050), (16, 19, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), f"ACTUAL TRAINING VISUAL CHECK - {profile}", fill=(245, 247, 250))

    scene = Image.fromarray(overview).resize((960, 540), Image.Resampling.LANCZOS)
    canvas.paste(scene, (20, 42))
    draw.text((1000, 58), "External check camera", fill=(230, 235, 241))
    draw.text((1000, 82), "Not given to policy", fill=(165, 176, 190))
    draw.text((1000, 112), f"Clutter prims: {clutter_count}", fill=(115, 220, 150))
    draw.text((1000, 136), f"Background prims: {background_count}", fill=(115, 220, 150))
    draw.text((1000, 160), f"Standing people: {people_count}", fill=(230, 190, 115))
    draw.text((1000, 184), f"Table-edge coworkers: {worker_reach_count}", fill=(230, 190, 115))
    draw.text((1000, 208), f"Styles: {background_styles}", fill=(165, 176, 190))
    draw.text((1000, 232), "Collision: flat plane", fill=(165, 176, 190))

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
    if args.reset_progress is not None and not 0.0 <= args.reset_progress <= 1.0:
        raise ValueError("--reset-progress must lie in [0, 1].")

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
    if args.reset_progress is None:
        cfg.completion_positive_reset_fraction = 1.0
        cfg.reset_ready_exact_fraction = 1.0
    else:
        cfg.completion_positive_reset_fraction = 0.0
        cfg.reset_ready_exact_fraction = 0.0
        cfg.reset_progress_min = float(args.reset_progress)
        cfg.reset_progress_max = float(args.reset_progress)
    cfg.failure_replay_fraction = 0.0
    cfg.reset_rotation_randomization_enabled = False
    cfg.reset_position_randomization_enabled = False
    cfg.reset_object_yaw_randomization_enabled = False
    cfg.require_rotation_reset_data = False
    # Use the most common training layout so the corrected 5 mm slots can be
    # judged directly. Appearance and D405 sensor randomization remain active.
    cfg.scene_tslot_geometry_randomization_enabled = False

    env = None
    try:
        env = gym.make(TASK_ID, cfg=cfg)
        env.reset()
        task = env.unwrapped
        print(
            "[MATERIALS] "
            f"source={task.visual_material_bindings['robot_material_source']} "
            f"finger_geometry={len(task.visual_material_bindings['finger_geometry'])} "
            f"contact_pads={len(task.visual_material_bindings['contact_pads'])}",
            flush=True,
        )
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
        busy_background_prims = tuple(
            str(prim.GetPath())
            for prim in stage.Traverse()
            if "/BusyBackground/Slot_" in str(prim.GetPath())
        )
        clutter_profiles = {"combined_clutter", "combined_busy_background"}
        if args.sim2real_profile in clutter_profiles and not clutter_prims:
            raise RuntimeError(f"{args.sim2real_profile} did not author any visual clutter prims.")
        if args.sim2real_profile not in clutter_profiles and clutter_prims:
            raise RuntimeError(
                f"Profile {args.sim2real_profile} unexpectedly contains clutter/distractor prims: {clutter_prims}"
            )
        if args.sim2real_profile == "combined_busy_background" and not busy_background_prims:
            raise RuntimeError("combined_busy_background did not author any busy-background prims.")
        if args.sim2real_profile != "combined_busy_background" and busy_background_prims:
            raise RuntimeError(
                f"Profile {args.sim2real_profile} unexpectedly contains busy-background prims: "
                f"{busy_background_prims}"
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
            background_count=len(task.busy_background_visual_bindings["prim_paths"]),
            people_count=int(task.busy_background_visual_bindings["people_count"]),
            worker_reach_count=int(
                task.busy_background_visual_bindings["worker_reach_count"]
            ),
            background_styles=dict(task.busy_background_visual_bindings["style_counts"]),
        )
        metadata = {
            "task": TASK_ID,
            "target_index": int(args.target_index),
            "target_id": task.target_ids[int(args.target_index)],
            "seed": int(args.seed),
            "requested_reset_progress": args.reset_progress,
            "realized_reset_progress": float(task.reset_progress[0]),
            "sim2real_profile": profile.identifier,
            "material_profile": VISUAL_SERVO_MATERIAL_PROFILE,
            "scene_profile": VISUAL_SERVO_SCENE_PROFILE,
            "tslot_profile": VISUAL_SERVO_TSLOT_PROFILE,
            "policy_visual_shape_hwc": list(policy_visual.shape),
            "tslot_layout": "nominal",
            "clutter_profile": task.clutter_visual_bindings["profile"],
            "clutter_prim_count": len(task.clutter_visual_bindings["prim_paths"]),
            "clutter_active_environment_count": task.clutter_visual_bindings["active_environment_count"],
            "busy_background_profile": task.busy_background_visual_bindings["profile"],
            "busy_background_prim_count": len(task.busy_background_visual_bindings["prim_paths"]),
            "busy_background_active_environment_count": task.busy_background_visual_bindings[
                "active_environment_count"
            ],
            "busy_background_people_count": task.busy_background_visual_bindings["people_count"],
            "busy_background_worker_reach_count": task.busy_background_visual_bindings[
                "worker_reach_count"
            ],
            "busy_background_style_counts": task.busy_background_visual_bindings["style_counts"],
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
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()
