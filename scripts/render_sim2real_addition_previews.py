#!/usr/bin/env python3
"""Render controlled wrist-camera previews for proposed sim-to-real additions.

This is deliberately preview-only: it does not change the training profiles.
Five cloned environments share the same target and exact successful robot pose:
canonical, render-only clutter, semi-gloss plastic, metallic target, and a
projected coarse surface texture.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
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
    default=Path("artifacts/sim2real_addition_previews"),
)
parser.add_argument("--settle-steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaac_rl.tasks  # noqa: E402, F401
import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from grasp_planning.isaac_visual_materials import VISUAL_SERVO_MATERIAL_PROFILE  # noqa: E402
from grasp_planning.isaac_visual_scene import VISUAL_SERVO_SCENE_PROFILE  # noqa: E402
from grasp_planning.visual_servo_workspace import VISUAL_SERVO_TSLOT_PROFILE  # noqa: E402

VARIANTS = (
    ("baseline", "Canonical matte plastic"),
    ("clutter", "Render-only coherent clutter"),
    ("semigloss_plastic", "Semi-gloss plastic"),
    ("metallic_part", "Metallic target"),
    ("surface_texture", "Coarse FDM-like surface bands"),
)


def _set_shader(
    stage,
    shader_path: str,
    *,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
) -> None:
    shader = stage.GetPrimAtPath(shader_path)
    if not shader.IsValid():
        raise RuntimeError(f"Missing preview shader: {shader_path}")
    values = {
        "inputs:diffuseColor": Gf.Vec3f(*color),
        "inputs:roughness": float(roughness),
        "inputs:metallic": float(metallic),
    }
    for name, value in values.items():
        attribute = shader.GetAttribute(name)
        if not attribute.IsValid() or not attribute.Set(value):
            raise RuntimeError(f"Could not set {shader_path}.{name}")


def _make_material(
    path: str,
    *,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float = 0.0,
) -> str:
    material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        roughness=roughness,
        metallic=metallic,
    )
    material.func(path, material)
    return path


def _spawn_clutter(stage) -> None:
    """Add visually coherent props to env 1 without collision schemas."""

    materials = (
        _make_material(
            "/World/Looks/PreviewClutterBlue",
            color=(0.04, 0.16, 0.34),
            roughness=0.42,
        ),
        _make_material(
            "/World/Looks/PreviewClutterOrange",
            color=(0.58, 0.16, 0.025),
            roughness=0.58,
        ),
        _make_material(
            "/World/Looks/PreviewClutterSteel",
            color=(0.34, 0.39, 0.44),
            roughness=0.24,
            metallic=0.85,
        ),
    )
    root = "/World/envs/env_1/PreviewClutter"

    cube = UsdGeom.Cube.Define(stage, f"{root}/BlueBlock")
    cube.CreateSizeAttr(1.0)
    cube_xform = UsdGeom.Xformable(cube)
    cube_xform.AddTranslateOp().Set(Gf.Vec3d(0.365, 0.118, 0.017))
    cube_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 24.0))
    cube_xform.AddScaleOp().Set(Gf.Vec3d(0.026, 0.020, 0.034))
    sim_utils.bind_visual_material(str(cube.GetPath()), materials[0], stage=stage)

    cylinder = UsdGeom.Cylinder.Define(stage, f"{root}/OrangeCap")
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateRadiusAttr(0.014)
    cylinder.CreateHeightAttr(0.045)
    cylinder_xform = UsdGeom.Xformable(cylinder)
    cylinder_xform.AddTranslateOp().Set(Gf.Vec3d(0.486, 0.105, 0.0225))
    sim_utils.bind_visual_material(str(cylinder.GetPath()), materials[1], stage=stage)

    bar = UsdGeom.Cube.Define(stage, f"{root}/SteelBar")
    bar.CreateSizeAttr(1.0)
    bar_xform = UsdGeom.Xformable(bar)
    bar_xform.AddTranslateOp().Set(Gf.Vec3d(0.430, -0.020, 0.009))
    bar_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, -18.0))
    bar_xform.AddScaleOp().Set(Gf.Vec3d(0.070, 0.014, 0.018))
    sim_utils.bind_visual_material(str(bar.GetPath()), materials[2], stage=stage)


def _apply_surface_texture_preview(rgb: np.ndarray) -> np.ndarray:
    """Overlay visible FDM-like bands only on the central brown target pixels.

    This preview illustrates the image-scale effect. A training implementation
    should use a true projected material rather than modifying observations.
    """

    result = rgb.copy()
    red = result[..., 0].astype(np.int16)
    green = result[..., 1].astype(np.int16)
    blue = result[..., 2].astype(np.int16)
    target = (red - green > 10) & (green - blue > 8) & (blue > 50)
    height, width = target.shape
    x = np.arange(width)[None, :]
    y = np.arange(height)[:, None]
    central = (x > 0.39 * width) & (x < 0.62 * width) & (y < 0.72 * height)
    bands = ((y + x // 8) % 9) < 2
    mask = target & central & bands
    result[mask] = np.clip(result[mask].astype(np.float32) * 0.58, 0.0, 255.0).astype(np.uint8)
    return result


def _write_preview_catalog(source_path: Path, source_target_index: int) -> Path:
    """Create a one-target catalog for rendering across a visual-profile transition.

    The saved goal image is not shown or evaluated by this script; only its
    target pose and reset trajectory are used. Production training continues
    to reject stale visual-profile metadata until the full catalog is rendered
    again.
    """

    with np.load(source_path, allow_pickle=False) as source:
        target_count = len(source["target_ids"])
        if not 0 <= source_target_index < target_count:
            raise ValueError(
                f"--target-index={source_target_index} is outside the {target_count}-target catalog."
            )
        payload = {
            name: (
                value[source_target_index : source_target_index + 1].copy()
                if value.ndim > 0 and value.shape[0] == target_count
                else value.copy()
            )
            for name in source.files
            for value in (source[name],)
        }
    payload["visual_material_profile"] = np.asarray(VISUAL_SERVO_MATERIAL_PROFILE)
    payload["visual_scene_profile"] = np.asarray(VISUAL_SERVO_SCENE_PROFILE)
    payload["visual_tslot_profile"] = np.asarray(VISUAL_SERVO_TSLOT_PROFILE)
    payload["capture_validation_passed"] = np.ones(1, dtype=np.bool_)
    payload["isaac_goal_rgbd_captured"] = np.ones(1, dtype=np.bool_)
    descriptor = tempfile.NamedTemporaryFile(prefix="workspace-preview-", suffix=".npz", delete=False)
    descriptor.close()
    output = Path(descriptor.name)
    np.savez_compressed(output, **payload)
    return output


def _rgb_uint8(value: torch.Tensor) -> np.ndarray:
    array = value[..., :3].detach().cpu().numpy()
    if array.dtype == np.uint8:
        return array
    if float(np.max(array)) <= 1.5:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _depth_color(depth_m: np.ndarray) -> np.ndarray:
    minimum, maximum = 0.04, 0.50
    normalized = np.clip((depth_m - minimum) / (maximum - minimum), 0.0, 1.0)
    invalid = (~np.isfinite(depth_m)) | (depth_m <= minimum) | (depth_m >= maximum)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    image = np.stack((red, green, blue), axis=-1)
    image[invalid] = 0.0
    return np.round(255.0 * image).astype(np.uint8)


def _comparison_sheet(images: list[np.ndarray], output: Path, title: str) -> None:
    scale = 3
    tile_width, tile_height = images[0].shape[1] * scale, images[0].shape[0] * scale
    label_height = 54
    columns = 3
    rows = 2
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        (17, 20, 25),
    )
    draw = ImageDraw.Draw(sheet)
    for index, ((name, description), image) in enumerate(zip(VARIANTS, images, strict=True)):
        column, row = index % columns, index // columns
        x = column * tile_width
        y = row * (tile_height + label_height)
        draw.text((x + 10, y + 7), name.replace("_", " ").upper(), fill=(245, 247, 250))
        draw.text((x + 10, y + 28), description, fill=(170, 180, 194))
        panel = Image.fromarray(image).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        sheet.paste(panel, (x, y + label_height))
    draw.text((2 * tile_width + 12, tile_height + label_height + 18), title, fill=(130, 142, 158))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    if args.target_index < 0:
        raise ValueError("--target-index must be non-negative.")
    if args.settle_steps < 1:
        raise ValueError("--settle-steps must be positive.")

    task_id = "Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0"
    cfg = parse_env_cfg(task_id, device=args.device, num_envs=len(VARIANTS))
    preview_catalog = _write_preview_catalog(Path(cfg.goal_catalog_data_path), args.target_index)
    cfg.seed = 11
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
    cfg.scene_tslot_surface_enabled = True
    cfg.scene_tslot_geometry_randomization_enabled = False
    cfg.live_observation_delay_max_steps = 0
    cfg.live_observation_repeat_probability = 0.0
    cfg.motion_action_delay_max_steps = 0
    cfg.motion_action_two_step_probability = 0.0
    cfg.failure_replay_fraction = 0.0
    cfg.reset_rotation_randomization_enabled = False
    cfg.reset_position_randomization_enabled = False
    cfg.require_rotation_reset_data = False

    env = None
    try:
        env = gym.make(task_id, cfg=cfg)
        env.reset()
        task = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        shader_paths = task.visual_material_bindings["part_shaders_by_env"]

        _spawn_clutter(stage)
        _set_shader(
            stage,
            shader_paths[2],
            color=(0.36, 0.11, 0.045),
            roughness=0.30,
            metallic=0.0,
        )
        _set_shader(
            stage,
            shader_paths[3],
            color=(0.42, 0.46, 0.52),
            roughness=0.18,
            metallic=0.92,
        )
        print("[PREVIEW] authored clutter and material variants", flush=True)

        with torch.inference_mode():
            for _ in range(args.settle_steps):
                task.scene.write_data_to_sim()
                task.sim.step()
                task.scene.update(task.sim.get_physics_dt())
                task.wrist_camera.update(task.sim.get_physics_dt(), force_recompute=True)
        print("[PREVIEW] captured settled wrist-camera tensors", flush=True)

        rgb = _rgb_uint8(task.wrist_camera.data.output["rgb"])
        rgb[4] = _apply_surface_texture_preview(rgb[4])
        depth = task.wrist_camera.data.output["distance_to_image_plane"].detach().cpu().numpy()
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if rgb.shape[0] != len(VARIANTS) or depth.shape[0] != len(VARIANTS):
            raise RuntimeError(f"Unexpected camera batches: rgb={rgb.shape}, depth={depth.shape}")

        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb_images: list[np.ndarray] = []
        depth_images: list[np.ndarray] = []
        for index, (name, _) in enumerate(VARIANTS):
            rgb_images.append(rgb[index])
            depth_images.append(_depth_color(depth[index]))
            Image.fromarray(rgb[index]).resize((rgb.shape[2] * 4, rgb.shape[1] * 4), Image.Resampling.LANCZOS).save(
                output_dir / f"{name}.png"
            )
            Image.fromarray(depth_images[-1]).resize(
                (rgb.shape[2] * 4, rgb.shape[1] * 4), Image.Resampling.NEAREST
            ).save(output_dir / f"{name}_depth.png")

        _comparison_sheet(rgb_images, output_dir / "comparison_rgb.png", "Wrist-camera RGB")
        _comparison_sheet(depth_images, output_dir / "comparison_depth.png", "Metric depth visualization")
        print(
            f"[DONE] target={task.target_ids[int(task.target_index[0])]} output={output_dir}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()
        preview_catalog.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
