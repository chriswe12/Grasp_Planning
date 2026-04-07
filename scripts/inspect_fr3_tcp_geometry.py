"""Inspect FR3 hand/TCP/finger geometry from the spawned Isaac asset."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect FR3 TCP geometry from the Isaac asset.")
parser.add_argument(
    "--fr3-usd",
    type=str,
    default="",
    help="Optional override for the Franka Research 3 USD asset path or Omniverse URL.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

from grasp_planning.envs import make_fr3_cube_scene_cfg  # noqa: E402
from grasp_planning.scene_defaults import (  # noqa: E402
    CUBE_ORIENTATION_XYZW,
    CUBE_POSITION,
    ROBOT_BASE_ORIENTATION_XYZW,
    ROBOT_BASE_POSITION,
)


def resolve_fr3_usd_path() -> str:
    if args_cli.fr3_usd:
        return args_cli.fr3_usd
    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in FR3 asset.")
    return assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"


def quat_to_rotmat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    two = torch.tensor(2.0, dtype=quat_wxyz.dtype, device=quat_wxyz.device)
    return torch.stack(
        (
            torch.stack((1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), dim=-1),
            torch.stack((two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)), dim=-1),
            torch.stack((two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    scene_cfg = make_fr3_cube_scene_cfg(
        fr3_asset_path=resolve_fr3_usd_path(),
        cube_position=CUBE_POSITION,
        cube_orientation_xyzw=CUBE_ORIENTATION_XYZW,
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
    )
    scene = InteractiveScene(scene_cfg)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    name_to_idx = {name: idx for idx, name in enumerate(robot.body_names)}
    hand_idx = name_to_idx["fr3_hand"]
    tcp_idx = name_to_idx["fr3_hand_tcp"]
    left_idx = name_to_idx["fr3_leftfinger"]
    right_idx = name_to_idx["fr3_rightfinger"]

    hand_pose = robot.data.body_pose_w[0, hand_idx]
    tcp_pose = robot.data.body_pose_w[0, tcp_idx]
    left_pose = robot.data.body_pose_w[0, left_idx]
    right_pose = robot.data.body_pose_w[0, right_idx]

    hand_pos = hand_pose[:3]
    hand_quat = hand_pose[3:7]
    tcp_pos = tcp_pose[:3]
    left_pos = left_pose[:3]
    right_pos = right_pose[:3]
    finger_mid = 0.5 * (left_pos + right_pos)

    hand_rot = quat_to_rotmat_wxyz(hand_quat)
    tcp_rel_in_hand = hand_rot.transpose(-1, -2) @ (tcp_pos - hand_pos)
    finger_mid_rel_in_tcp = hand_rot.transpose(-1, -2) @ (finger_mid - tcp_pos)

    print(f"hand_to_tcp_in_hand_frame={tuple(float(v) for v in tcp_rel_in_hand.tolist())}", flush=True)
    print(f"finger_midpoint_in_tcp_frame={tuple(float(v) for v in finger_mid_rel_in_tcp.tolist())}", flush=True)
    print("expected_tcp_to_grasp_center_offset=(0.0, 0.0, -0.045)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[ERROR]: Unhandled exception in inspect_fr3_tcp_geometry.py", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        simulation_app.close()
