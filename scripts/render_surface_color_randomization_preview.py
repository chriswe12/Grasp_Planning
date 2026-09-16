#!/usr/bin/env python3
"""Render true Isaac goal/live color pairs with T-slot surface markings."""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target-index", type=int, default=0)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/surface_color_randomization_preview"),
)
parser.add_argument("--settle-steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaac_rl.tasks  # noqa: E402, F401
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Gf, UsdShade  # noqa: E402

from grasp_planning.isaac_visual_materials import (  # noqa: E402
    PDZ_GRIPPER_APPEARANCE_VARIANT_SET,
    VISUAL_SERVO_MATERIAL_PROFILE,
    VISUAL_SERVO_PART_PALETTE,
    get_gripper_appearance_variant,
)
from grasp_planning.rl.goal_catalog_profiles import MUJOCO_GOAL_RENDERER_PROFILE  # noqa: E402
from grasp_planning.visual_servo_surface_markings import SURFACE_MARKING_PROFILE  # noqa: E402

VARIANTS = (
    ("clean_match", "clean surface; exact color", 2, 2, 0.82, 0.00),
    ("writing_scratch_match", "writing/scratch; exact color", 3, 3, 0.70, 0.04),
    ("tape_writing_similar", "tape/writing/scratch; similar colors", 9, 20, 0.58, 0.08),
    ("dirt_scratch_different", "dirt/scratch; different colors", 19, 5, 0.88, 0.00),
    ("dirt_scratch_mixed", "dirt/scratch; different colors", 23, 18, 0.42, 0.14),
    ("tape_writing_semigloss", "translucent tape/writing; semi-gloss", 0, 15, 0.30, 0.18),
)
# These are the real variants authored inside the instanced robot USD, not
# disconnected preview shaders. The first row remains exactly canonical.
GRIPPER_VARIANTS = tuple(
    get_gripper_appearance_variant(name)
    for name in (
        "canonical",
        "dark_glossy_cool",
        "bright_matte_warm",
        "dark_matte_warm",
        "bright_glossy_cool",
        "charcoal_satin_neutral",
    )
)
GOAL_ENV_INDEX = 1
LIVE_ENV_INDICES = (6, 0, 2, 3, 4, 5)
PREVIEW_ENV_COUNT = 8


def _write_preview_catalog(source_path: Path, source_target_index: int) -> Path:
    with np.load(source_path, allow_pickle=False) as source:
        target_count = len(source["target_ids"])
        if not 0 <= source_target_index < target_count:
            raise ValueError(f"--target-index must lie in [0, {target_count - 1}].")
        payload = {
            name: (
                value[source_target_index : source_target_index + 1].copy()
                if value.ndim > 0 and value.shape[0] == target_count
                else value.copy()
            )
            for name in source.files
            for value in (source[name],)
        }
    # This diagnostic renders both panels live in Isaac and only consumes the
    # selected catalog pose. Keep the temporary one-target catalog compatible
    # when the checked-in RGB-D was captured with an older renderer profile.
    payload["goal_renderer_profile"] = np.asarray(MUJOCO_GOAL_RENDERER_PROFILE)
    descriptor = tempfile.NamedTemporaryFile(prefix="surface-color-preview-", suffix=".npz", delete=False)
    descriptor.close()
    output = Path(descriptor.name)
    np.savez_compressed(output, **payload)
    return output


def _set_shader(stage, shader_path: str, palette_index: int, roughness: float, metallic: float) -> None:
    shader = stage.GetPrimAtPath(shader_path)
    color = VISUAL_SERVO_PART_PALETTE[palette_index].color
    for name, value in (
        ("inputs:diffuseColor", Gf.Vec3f(*color)),
        ("inputs:roughness", float(roughness)),
        ("inputs:metallic", float(metallic)),
    ):
        if not shader.GetAttribute(name).Set(value):
            raise RuntimeError(f"Could not set {shader_path}.{name}")


def _set_gripper_materials(stage, variant_roots: tuple[str, ...], variant) -> None:
    if not variant_roots:
        raise RuntimeError("PDZ robot USD has no authored gripper appearance variant roots.")
    for path in variant_roots:
        prim = stage.GetPrimAtPath(path)
        variant_set = prim.GetVariantSets().GetVariantSet(PDZ_GRIPPER_APPEARANCE_VARIANT_SET)
        if not variant_set.SetVariantSelection(variant.name):
            raise RuntimeError(f"Could not select {variant.name} on {path}")


def _capture(task, steps: int) -> np.ndarray:
    with torch.inference_mode():
        for _ in range(steps):
            task.scene.write_data_to_sim()
            task.sim.step()
            task.scene.update(task.sim.get_physics_dt())
            task.wrist_camera.update(task.sim.get_physics_dt(), force_recompute=True)
    value = task.wrist_camera.data.output["rgb"][..., :3].detach().cpu().numpy()
    if value.dtype != np.uint8:
        value = np.clip(value * (255.0 if float(value.max()) <= 1.5 else 1.0), 0.0, 255.0).astype(np.uint8)
    return value


def _report_gripper_material_bindings(stage, bindings: dict[str, object]) -> None:
    print(
        "[MATERIALS] "
        f"source={bindings['robot_material_source']} "
        f"finger_geometry={len(bindings['finger_geometry'])} "
        f"editable_fingers={len(bindings['editable_finger_geometry'])} "
        f"contact_pads={len(bindings['contact_pads'])} "
        f"editable_pads={len(bindings['editable_contact_pads'])}",
        flush=True,
    )
    for kind in ("finger_geometry", "contact_pads"):
        for path in bindings[kind][:2]:
            prim = stage.GetPrimAtPath(path)
            material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            print(
                f"[MATERIALS] kind={kind} prim={path} proxy={prim.IsInstanceProxy()} "
                f"material={material.GetPath() if material else '<none>'}",
                flush=True,
            )


def _write_sheet(goal: np.ndarray, live: np.ndarray, output: Path) -> None:
    scale = 3
    width, height = goal.shape[2] * scale, goal.shape[1] * scale
    label_h = 74
    sheet = Image.new("RGB", (3 * width, 2 * (2 * height + label_h)), (16, 19, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (name, description, goal_index, live_index, roughness, metallic) in enumerate(VARIANTS):
        column, row = index % 3, index // 3
        x, y = column * width, row * (2 * height + label_h)
        draw.text((x + 8, y + 6), name.replace("_", " ").upper(), fill=(245, 247, 250))
        draw.text((x + 8, y + 27), description, fill=(174, 184, 198))
        draw.text(
            (x + 8, y + 43),
            f"goal={goal_index} live={live_index} rough={roughness:.2f} metal={metallic:.2f}",
            fill=(126, 142, 160),
        )
        gripper = GRIPPER_VARIANTS[index]
        draw.text(
            (x + 8, y + 58),
            f"gripper={gripper.name} finger_r={gripper.finger_roughness:.2f} "
            f"pad_r={gripper.pad_roughness:.2f}",
            fill=(126, 142, 160),
        )
        goal_panel = Image.fromarray(goal[index]).resize((width, height), Image.Resampling.LANCZOS)
        live_panel = Image.fromarray(live[index]).resize((width, height), Image.Resampling.LANCZOS)
        sheet.paste(goal_panel, (x, y + label_h))
        sheet.paste(live_panel, (x, y + label_h + height))
        draw.text((x + 8, y + label_h + 6), "SYNTHETIC GOAL", fill=(235, 235, 235))
        draw.text((x + 8, y + label_h + height + 6), "RANDOMIZED LIVE", fill=(235, 235, 235))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _write_gripper_sheet(frames: np.ndarray, output: Path) -> None:
    scale = 3
    width, height = frames.shape[2] * scale, frames.shape[1] * scale
    label_h = 42
    sheet = Image.new("RGB", (3 * width, 2 * (height + label_h)), (16, 19, 24))
    draw = ImageDraw.Draw(sheet)
    canonical = frames[0].astype(np.int16)
    for index, (variant, frame) in enumerate(zip(GRIPPER_VARIANTS, frames, strict=True)):
        column, row = index % 3, index // 3
        x, y = column * width, row * (height + label_h)
        mean_abs_delta = float(np.abs(frame.astype(np.int16) - canonical).mean())
        draw.text((x + 8, y + 6), variant.name.upper(), fill=(245, 247, 250))
        draw.text(
            (x + 8, y + 23),
            f"finger_r={variant.finger_roughness:.2f} pad_r={variant.pad_roughness:.2f} "
            f"frame_MAE={mean_abs_delta:.3f}",
            fill=(150, 165, 183),
        )
        panel = Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
        sheet.paste(panel, (x, y + label_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    task_id = "Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0"
    cfg = parse_env_cfg(task_id, device=args.device, num_envs=PREVIEW_ENV_COUNT)
    preview_catalog = _write_preview_catalog(Path(cfg.goal_catalog_data_path), args.target_index)
    cfg.seed = 0
    cfg.goal_catalog_data_path = str(preview_catalog)
    cfg.catalog_split = "all"
    cfg.fixed_target_index = 0
    cfg.scene.replicate_physics = False
    cfg.training_curriculum_enabled = False
    cfg.training_reset_mixture_enabled = False
    cfg.completion_positive_reset_fraction = 1.0
    cfg.reset_ready_exact_fraction = 1.0
    cfg.live_observation_randomization_enabled = False
    cfg.scene_appearance_randomization_enabled = False
    cfg.scene_tslot_geometry_randomization_enabled = False
    cfg.scene_surface_markings_enabled = True
    cfg.scene_surface_markings_environment_fraction = 1.0
    cfg.scene_surface_markings_clean_fraction = 0.22
    cfg.scene_surface_markings_min_count = 2
    cfg.scene_surface_markings_max_count = 6
    cfg.reset_rotation_randomization_enabled = False
    cfg.reset_position_randomization_enabled = False
    cfg.reset_object_yaw_randomization_enabled = False
    cfg.reset_collision_safe_sampling_enabled = False
    cfg.require_rotation_reset_data = False

    env = None
    try:
        env = gym.make(task_id, cfg=cfg)
        env.reset()
        task = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        shader_paths = task.visual_material_bindings["part_shaders_by_env"]
        gripper_roots_by_env = task.visual_material_bindings[
            "gripper_appearance_variant_roots_by_env"
        ]
        _report_gripper_material_bindings(stage, task.visual_material_bindings)
        _set_shader(stage, shader_paths[GOAL_ENV_INDEX], 2, 0.82, 0.0)
        gripper_frames = []
        for gripper_variant in GRIPPER_VARIANTS:
            _set_gripper_materials(
                stage,
                gripper_roots_by_env[GOAL_ENV_INDEX],
                gripper_variant,
            )
            gripper_frames.append(_capture(task, max(2, args.settle_steps // 2))[GOAL_ENV_INDEX])
        gripper_frames_array = np.stack(gripper_frames)

        goal_frames = []
        _set_gripper_materials(stage, gripper_roots_by_env[GOAL_ENV_INDEX], GRIPPER_VARIANTS[0])
        for variant in VARIANTS:
            _set_shader(stage, shader_paths[GOAL_ENV_INDEX], variant[2], 0.82, 0.0)
            goal_frames.append(_capture(task, max(2, args.settle_steps // 2))[GOAL_ENV_INDEX])
        goal = np.stack(goal_frames)

        for env_index, variant in zip(LIVE_ENV_INDICES, VARIANTS, strict=True):
            _set_shader(stage, shader_paths[env_index], variant[3], variant[4], variant[5])
        live_frames = []
        for env_index, gripper_variant in zip(LIVE_ENV_INDICES, GRIPPER_VARIANTS, strict=True):
            _set_gripper_materials(stage, gripper_roots_by_env[env_index], gripper_variant)
            live_frames.append(_capture(task, max(2, args.settle_steps // 2))[env_index])
        live = np.stack(live_frames)

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_sheet(goal, live, output_dir / "goal_live_surface_color_sheet.png")
        _write_gripper_sheet(
            gripper_frames_array,
            output_dir / "gripper_material_variants_controlled_sheet.png",
        )
        for index, (name, *_rest) in enumerate(VARIANTS):
            Image.fromarray(goal[index]).save(output_dir / f"{name}_goal.png")
            Image.fromarray(live[index]).save(output_dir / f"{name}_live.png")
        print(
            f"[DONE] material={VISUAL_SERVO_MATERIAL_PROFILE} markings={SURFACE_MARKING_PROFILE} "
            f"output={output_dir}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()
        preview_catalog.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # SimulationApp.close() can tear down Python before the default
        # excepthook flushes, so emit preview failures before renderer shutdown.
        traceback.print_exc()
        raise
    finally:
        app.close()
