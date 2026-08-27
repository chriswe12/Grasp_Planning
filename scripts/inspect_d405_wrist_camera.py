#!/usr/bin/env python3
"""Open one KUKA scene in Isaac and inspect the calibrated D405 mount."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-seconds", type=float, default=0.0, help="0 keeps the GUI open until closed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.d405_wrist_camera import D405WristCameraConfig  # noqa: E402
from grasp_planning.envs.fr3_part_env import make_fr3_part_scene_cfg  # noqa: E402
from grasp_planning.isaac_visual_materials import apply_visual_servo_materials  # noqa: E402
from grasp_planning.isaac_visual_scene import make_visual_servo_render_cfg  # noqa: E402


def _set_debug_color(geometry_prim, color_rgb: tuple[float, float, float]) -> None:
    primvars = UsdGeom.PrimvarsAPI(geometry_prim.GetPrim())
    color = primvars.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant)
    color.Set([Gf.Vec3f(*color_rgb)])


def _add_camera_debug_geometry(camera_prim_path: str) -> None:
    """Show the D405 body and exact optical origin without adding physics."""

    stage = omni.usd.get_context().get_stage()
    housing = UsdGeom.Cube.Define(stage, f"{camera_prim_path}/DebugHousing")
    housing.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(housing)
    # Isaac's camera internal OpenGL convention views along -Z; put the body
    # behind the optical origin so it does not obscure the wrist observation.
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0115))
    xform.AddScaleOp().Set(Gf.Vec3d(0.042, 0.042, 0.023))
    _set_debug_color(housing, (1.0, 0.25, 0.02))

    # Sphere center is exactly (0, 0, 0) in the camera optical frame.  Its
    # center therefore marks the precise pinhole/sensor origin used to render.
    optical_center = UsdGeom.Sphere.Define(stage, f"{camera_prim_path}/OpticalCenterMarker")
    optical_center.CreateRadiusAttr(0.006)
    _set_debug_color(optical_center, (0.0, 1.0, 1.0))


def main() -> None:
    if args_cli.run_seconds < 0.0:
        raise ValueError("--run-seconds must be non-negative.")
    print("[D405-INSPECT] preparing scene configuration", flush=True)
    robot_usd = REPO_ROOT / "assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"
    part_usd = REPO_ROOT / "isaac_rl/data/plumbers_block/usd/part_0_bundle_local.usd"
    camera = D405WristCameraConfig(enabled=True, include_privileged_mask=False)
    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=str(robot_usd),
        part_usd_path=str(part_usd),
        part_position=(0.45, 0.0, 0.03),
        part_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    scene_cfg.num_envs = 1
    print("[D405-INSPECT] creating simulation context", flush=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, render=make_visual_servo_render_cfg())
    )
    sim.set_camera_view([1.25, -1.1, 0.95], [0.4, 0.0, 0.32])
    print("[D405-INSPECT] spawning scene", flush=True)
    scene = InteractiveScene(scene_cfg)
    print("[D405-INSPECT] waiting for robot USD", flush=True)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    print("[D405-INSPECT] attaching D405 to loaded link7", flush=True)
    from grasp_planning.envs.fr3_part_env import make_d405_wrist_camera_cfg

    wrist_camera = Camera(
        cfg=make_d405_wrist_camera_cfg(
            parent_prim_path="/World/envs/env_.*/Robot/link7",
            wrist_camera=camera,
        )
    )
    print("[D405-INSPECT] resetting simulation", flush=True)
    sim.reset()
    scene.reset()
    print("[D405-INSPECT] adding D405 housing and optical-center marker", flush=True)
    _add_camera_debug_geometry("/World/envs/env_0/Robot/link7/D405LeftCamera")
    print("[D405-INSPECT] applying materials", flush=True)
    apply_visual_servo_materials()
    print(
        "[INFO] D405 inspector ready. Orange=body; cyan sphere center=exact optical sensor origin. "
        "Select either marker in the Stage tree and press F.",
        flush=True,
    )
    elapsed_s = 0.0
    while simulation_app.is_running() and (args_cli.run_seconds <= 0.0 or elapsed_s < args_cli.run_seconds):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        wrist_camera.update(sim.get_physics_dt())
        elapsed_s += sim.get_physics_dt()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit can otherwise close before Python displays an initialization
        # error, which looks like an inspector window that simply disappears.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
