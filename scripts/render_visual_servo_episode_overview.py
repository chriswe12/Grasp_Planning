#!/usr/bin/env python3
"""Replay a saved curriculum episode and render a whole-robot perspective camera."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("episode_npz", type=Path)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output NPZ sidecar. Defaults to <episode>_overview.npz.",
)
parser.add_argument(
    "--robot-usd",
    type=Path,
    default=Path("assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"),
)
parser.add_argument(
    "--part-usd",
    type=Path,
    default=Path(
        "artifacts/isaac_bundle_assets/pipeline_stage2_ground_feasible_bundle_local.usd"
    ),
)
parser.add_argument("--camera-width", type=int, default=640)
parser.add_argument("--camera-height", type=int, default=480)
parser.add_argument("--eye", type=float, nargs=3, default=(1.6, -1.2, 1.0))
parser.add_argument("--target", type=float, nargs=3, default=(0.35, 0.0, 0.3))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.envs.fr3_part_env import (  # noqa: E402
    make_fr3_part_scene_cfg,
    make_robot_overview_camera_cfg,
)
from grasp_planning.isaac_visual_materials import apply_visual_servo_materials  # noqa: E402
from grasp_planning.isaac_visual_scene import make_visual_servo_render_cfg  # noqa: E402


def _parallel_env_origin(metadata: dict[str, object]) -> np.ndarray:
    """Reconstruct InteractiveScene's centered grid origin for a saved environment."""

    env_count = int(metadata.get("parallel_env_count", 1))
    env_index = int(metadata.get("parallel_env_index", 0))
    if env_count <= 1:
        return np.zeros(3, dtype=np.float32)
    rows = int(np.ceil(env_count / int(np.sqrt(env_count))))
    columns = int(np.ceil(env_count / rows))
    row = env_index // columns
    column = env_index % columns
    spacing = 2.5
    return np.array(
        [
            (rows - 1) * spacing * 0.5 - row * spacing,
            column * spacing - (columns - 1) * spacing * 0.5,
            0.0,
        ],
        dtype=np.float32,
    )


def main() -> None:
    episode_path = args_cli.episode_npz.resolve()
    metadata_path = episode_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing episode metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(episode_path) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    required = {
        "joint_positions",
        "object_position_w",
        "object_orientation_xyzw_w",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(
            f"{episode_path} lacks {missing}; use data from the batched curriculum collector."
        )

    robot_usd = args_cli.robot_usd.resolve()
    part_usd = args_cli.part_usd.resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(robot_usd)
    if not part_usd.is_file():
        raise FileNotFoundError(part_usd)

    env_origin = _parallel_env_origin(metadata)
    object_positions = arrays["object_position_w"].astype(np.float32) - env_origin
    object_quaternions_xyzw = arrays["object_orientation_xyzw_w"].astype(np.float32)
    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=str(robot_usd),
        part_usd_path=str(part_usd),
        part_position=tuple(float(value) for value in object_positions[0]),
        part_orientation_xyzw=tuple(float(value) for value in object_quaternions_xyzw[0]),
    )
    scene_cfg.num_envs = 1
    scene_cfg.overview_camera = make_robot_overview_camera_cfg(
        width=args_cli.camera_width,
        height=args_cli.camera_height,
    )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            render=make_visual_servo_render_cfg(),
        )
    )
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    apply_visual_servo_materials()

    robot = scene["robot"]
    part = scene["part"]
    camera = scene["overview_camera"]
    arm_joint_ids, _ = robot.find_joints(r"joint[1-7]")
    if len(arm_joint_ids) != 7:
        raise RuntimeError(f"Expected 7 KUKA arm joints, found {len(arm_joint_ids)}.")
    eye = torch.tensor([args_cli.eye], dtype=torch.float32, device=sim.device)
    target = torch.tensor([args_cli.target], dtype=torch.float32, device=sim.device)
    camera.set_world_poses_from_view(eye, target)

    zero_arm_velocity = torch.zeros((1, 7), dtype=torch.float32, device=sim.device)
    zero_part_velocity = torch.zeros((1, 6), dtype=torch.float32, device=sim.device)
    overview_frames: list[np.ndarray] = []
    for step_index, joint_positions in enumerate(arrays["joint_positions"]):
        q_arm = torch.tensor(
            joint_positions[None, :], dtype=torch.float32, device=sim.device
        )
        robot.write_joint_state_to_sim(
            q_arm,
            zero_arm_velocity,
            joint_ids=arm_joint_ids,
        )
        robot.set_joint_position_target(q_arm, joint_ids=arm_joint_ids)
        quaternion_xyzw = object_quaternions_xyzw[step_index]
        object_pose_wxyz = torch.tensor(
            [
                [
                    *object_positions[step_index],
                    quaternion_xyzw[3],
                    quaternion_xyzw[0],
                    quaternion_xyzw[1],
                    quaternion_xyzw[2],
                ]
            ],
            dtype=torch.float32,
            device=sim.device,
        )
        part.write_root_pose_to_sim(object_pose_wxyz)
        part.write_root_velocity_to_sim(zero_part_velocity)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        overview_frames.append(np.asarray(rgb, dtype=np.uint8))
        if step_index == 0 or (step_index + 1) % 30 == 0:
            print(
                f"[OVERVIEW] rendered {step_index + 1}/{len(arrays['joint_positions'])}",
                flush=True,
            )

    output = (
        args_cli.output.resolve()
        if args_cli.output is not None
        else episode_path.with_name(f"{episode_path.stem}_overview.npz")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        overview_rgb=np.stack(overview_frames, axis=0),
        eye=np.asarray(args_cli.eye, dtype=np.float32),
        target=np.asarray(args_cli.target, dtype=np.float32),
        source_episode=np.asarray(str(episode_path)),
    )
    print(
        f"Wrote {len(overview_frames)} synchronized overview frames to {output}.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
