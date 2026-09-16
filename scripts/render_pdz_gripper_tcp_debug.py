#!/usr/bin/env python3
"""Render the generated PDZ gripper with its reconstructed TCP marker."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/pdz_gripper_tcp_debug"),
)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--settle-steps", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

from grasp_planning.envs.fr3_part_env import (  # noqa: E402
    make_fr3_part_scene_cfg,
    make_robot_overview_camera_cfg,
)
from grasp_planning.isaac_visual_materials import apply_visual_servo_materials  # noqa: E402
from grasp_planning.isaac_visual_scene import make_visual_servo_render_cfg  # noqa: E402

ROBOT_USD = REPO_ROOT / "assets/usd/kuka_iiwa7_pdz_gripper/kuka_iiwa7_pdz_gripper.usd"
PART_USD = REPO_ROOT / "isaac_rl/data/plumbers_block/usd/part_0_bundle_local.usd"
TCP_OFFSET_BASE_M = (0.0, 0.0, 0.1355)
TCP_YAW_BASE_RAD = -0.5 * math.pi


def _color(prim, rgb: tuple[float, float, float]) -> None:
    primvars = UsdGeom.PrimvarsAPI(prim)
    display = primvars.CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.constant,
    )
    display.Set([Gf.Vec3f(*rgb)])


def _sphere(stage, path: str, position: np.ndarray, radius: float, rgb: tuple[float, float, float]) -> None:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    _color(sphere.GetPrim(), rgb)


def _add_tcp_marker(
    *,
    position_w: torch.Tensor,
) -> None:
    print("[TCP-DEBUG] authoring marker root", flush=True)
    stage = omni.usd.get_context().get_stage()
    position = position_w.detach().cpu().numpy()
    root = UsdGeom.Xform.Define(stage, "/World/PDZTCPDebug")
    root.GetPrim().SetMetadata(
        "documentation",
        "Runtime PDZ TCP reconstructed from pdz_gripper_base_link: xyz=(0,0,0.1355), yaw=-90deg."
    )
    print("[TCP-DEBUG] authoring TCP sphere", flush=True)
    _sphere(stage, "/World/PDZTCPDebug/TCP", position, 0.008, (1.0, 0.85, 0.0))

    print("[TCP-DEBUG] marker authored", flush=True)


def _rgb_uint8(value: torch.Tensor) -> np.ndarray:
    array = value[..., :3].detach().cpu().numpy()
    if array.dtype == np.uint8:
        return array
    if float(np.max(array)) <= 1.5:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _render_view(camera, scene, sim, *, eye: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
    for _ in range(5):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt(), force_recompute=True)
    return _rgb_uint8(camera.data.output["rgb"])[0]


def _compose(front: np.ndarray, angle: np.ndarray, output: Path) -> None:
    panel_width = 720
    panel_height = 540
    canvas = Image.new("RGB", (panel_width * 2, panel_height + 100), (17, 20, 25))
    front_image = Image.fromarray(front).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    angle_image = Image.fromarray(angle).resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    canvas.paste(front_image, (0, 60))
    canvas.paste(angle_image, (panel_width, 60))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), "PDZ GRIPPER TCP - exact runtime transform", fill=(245, 247, 250))
    draw.text((20, 38), "Center of yellow sphere = exact TCP", fill=(255, 218, 70))
    draw.text((20, 608), "Pad view: looking along TCP -Z", fill=(225, 230, 238))
    draw.text((panel_width + 20, 608), "Oblique view", fill=(225, 230, 238))
    canvas.save(output)


def main() -> None:
    if args.width < 64 or args.height < 64:
        raise ValueError("--width and --height must be at least 64 pixels.")
    if args.settle_steps < 1:
        raise ValueError("--settle-steps must be positive.")
    for path in (ROBOT_USD, PART_USD):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("[TCP-DEBUG] configuring scene", flush=True)
    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=str(ROBOT_USD),
        part_usd_path=str(PART_USD),
        part_position=(2.0, 2.0, 0.1),
        part_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    scene_cfg.num_envs = 1
    scene_cfg.overview_camera = make_robot_overview_camera_cfg(
        width=args.width,
        height=args.height,
    )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device=args.device,
            render=make_visual_servo_render_cfg(),
        )
    )
    print("[TCP-DEBUG] spawning robot and camera", flush=True)
    scene = InteractiveScene(scene_cfg)
    print("[TCP-DEBUG] resetting simulation", flush=True)
    sim.reset()
    scene.reset()
    print("[TCP-DEBUG] applying visual materials", flush=True)
    apply_visual_servo_materials()

    print("[TCP-DEBUG] settling articulation", flush=True)
    for _ in range(args.settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    try:
        base_index = list(robot.body_names).index("pdz_gripper_base_link")
    except ValueError as exc:
        raise RuntimeError(f"PDZ gripper base not found in bodies: {robot.body_names}") from exc
    base_pose_w = robot.data.body_pose_w[0, base_index]
    base_position_w = base_pose_w[:3]
    base_quaternion_wxyz_w = base_pose_w[3:7]
    local_offset = torch.tensor(TCP_OFFSET_BASE_M, device=sim.device, dtype=torch.float32)
    local_yaw = torch.tensor(
        [math.cos(0.5 * TCP_YAW_BASE_RAD), 0.0, 0.0, math.sin(0.5 * TCP_YAW_BASE_RAD)],
        device=sim.device,
        dtype=torch.float32,
    )
    tcp_position_w = base_position_w + quat_apply(
        base_quaternion_wxyz_w.unsqueeze(0), local_offset.unsqueeze(0)
    )[0]
    tcp_quaternion_wxyz_w = quat_mul(
        base_quaternion_wxyz_w.unsqueeze(0), local_yaw.unsqueeze(0)
    )[0]
    print(
        f"[TCP-DEBUG] computed world TCP xyz={tcp_position_w.detach().cpu().tolist()}",
        flush=True,
    )
    _add_tcp_marker(
        position_w=tcp_position_w,
    )

    camera = scene["overview_camera"]
    front_offset_tcp = torch.tensor((0.0, 0.0, 0.20), device=sim.device)
    angle_offset_tcp = torch.tensor((0.12, -0.14, 0.16), device=sim.device)
    front_eye = tcp_position_w + quat_apply(
        tcp_quaternion_wxyz_w.unsqueeze(0), front_offset_tcp.unsqueeze(0)
    )[0]
    angle_eye = tcp_position_w + quat_apply(
        tcp_quaternion_wxyz_w.unsqueeze(0), angle_offset_tcp.unsqueeze(0)
    )[0]
    print("[TCP-DEBUG] rendering front view", flush=True)
    front = _render_view(camera, scene, sim, eye=front_eye, target=tcp_position_w)
    print("[TCP-DEBUG] rendering oblique view", flush=True)
    angle = _render_view(camera, scene, sim, eye=angle_eye, target=tcp_position_w)

    output_dir = args.output_dir.expanduser().resolve()
    print(f"[TCP-DEBUG] writing output under {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(front).save(output_dir / "tcp_front.png")
    Image.fromarray(angle).save(output_dir / "tcp_oblique.png")
    _compose(front, angle, output_dir / "tcp_views.png")
    np.savez(
        output_dir / "tcp_pose.npz",
        tcp_position_w=tcp_position_w.detach().cpu().numpy(),
        tcp_orientation_wxyz_w=tcp_quaternion_wxyz_w.detach().cpu().numpy(),
        tcp_offset_base_m=np.asarray(TCP_OFFSET_BASE_M, dtype=np.float32),
        tcp_yaw_base_rad=np.asarray(TCP_YAW_BASE_RAD, dtype=np.float32),
    )
    print(f"[TCP-DEBUG] robot USD: {ROBOT_USD}", flush=True)
    print("[TCP-DEBUG] body: pdz_gripper_base_link", flush=True)
    print(f"[TCP-DEBUG] local TCP: xyz={TCP_OFFSET_BASE_M} yaw={math.degrees(TCP_YAW_BASE_RAD):.1f} deg", flush=True)
    print(f"[TCP-DEBUG] wrote: {output_dir / 'tcp_views.png'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close(wait_for_replicator=False)
