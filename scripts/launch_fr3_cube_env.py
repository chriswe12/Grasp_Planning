"""Launch an Isaac Lab scene with a ground plane, FR3 robot, and graspable cube."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Launch an FR3 + cube grasp environment in Isaac Lab.")
parser.add_argument(
    "--fr3-usd",
    type=str,
    default="",
    help="Optional override for the Franka Research 3 USD asset path or Omniverse URL.",
)
parser.add_argument(
    "--run-seconds",
    type=float,
    default=0.0,
    help="Optional wall-clock duration to keep the simulation alive. Use 0 for until interrupted.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from grasp_planning.envs import make_fr3_cube_scene_cfg  # noqa: E402


ROBOT_BASE_POSITION = (0.0, 0.0, 0.0)
ROBOT_BASE_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)

# Keep the cube pose local to the scene launcher for now. Controller integration can
# replace this constant with externally provided pose data later.
CUBE_POSITION = (0.45, 0.0, 0.025)
CUBE_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)


def resolve_fr3_usd_path() -> str:
    """Return the user-supplied FR3 asset path or the built-in Isaac asset URL."""

    if args_cli.fr3_usd:
        return args_cli.fr3_usd

    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in FR3 asset.")
    return assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"


def build_scene() -> tuple[sim_utils.SimulationContext, InteractiveScene]:
    """Create the simulator and populate the FR3 cube scene."""

    print("[INFO]: Creating simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    sim.set_camera_view([1.6, -1.2, 1.0], [0.35, 0.0, 0.3])
    print("[INFO]: Resolving FR3 asset path...", flush=True)
    fr3_usd_path = resolve_fr3_usd_path()
    print(f"[INFO]: FR3 asset path: {fr3_usd_path}", flush=True)

    scene_cfg = make_fr3_cube_scene_cfg(
        fr3_asset_path=fr3_usd_path,
        cube_position=CUBE_POSITION,
        cube_orientation_xyzw=CUBE_ORIENTATION_XYZW,
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
    )
    print("[INFO]: Creating interactive scene...", flush=True)
    scene = InteractiveScene(scene_cfg)
    print("[INFO]: Waiting for stage assets to finish loading...", flush=True)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Simulator reset finished.", flush=True)
    print("[INFO]: Resetting scene buffers...", flush=True)
    scene.reset()
    print("[INFO]: Scene ready.", flush=True)
    return sim, scene


def run() -> None:
    """Step the simulation until the Isaac app is closed."""

    sim, scene = build_scene()
    physics_dt = sim.get_physics_dt()
    elapsed_s = 0.0
    print("[INFO]: Setup complete...")

    while args_cli.run_seconds <= 0.0 or elapsed_s < args_cli.run_seconds:
        try:
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.step()
                continue
            scene.write_data_to_sim()
            sim.step()
            scene.update(physics_dt)
            elapsed_s += physics_dt
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
