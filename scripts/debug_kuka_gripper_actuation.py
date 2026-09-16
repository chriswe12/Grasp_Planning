"""Minimal IsaacLab diagnostic for the generated KUKA/Y gripper finger joints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--robot-usd",
    type=Path,
    default=Path("assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"),
)
parser.add_argument(
    "--part-usd",
    type=Path,
    default=Path("artifacts/isaac_bundle_assets/plumbers_block0_kuka_pipeline_stage2_ground_feasible_bundle_local.usd"),
)
parser.add_argument("--steps-per-target", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from grasp_planning.envs import make_fr3_part_scene_cfg  # noqa: E402
from grasp_planning.start_poses import gripper_joint_target_from_width, is_gripper_command_joint_name  # noqa: E402


def _resolve(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return str(resolved)


def _hand_state(robot, joint_ids: list[int]) -> list[float]:
    return [float(value) for value in robot.data.joint_pos[:, joint_ids][0].detach().cpu().tolist()]


def _hand_limits(robot, joint_ids: list[int]) -> list[list[float]] | None:
    limits = getattr(robot.data, "joint_pos_limits", None)
    if limits is None:
        limits = getattr(robot.data, "joint_limits", None)
    if limits is None:
        return None
    return [[float(v) for v in pair] for pair in limits[:, joint_ids, :][0].detach().cpu().tolist()]


def _body_y(robot, body_name: str) -> float | None:
    if body_name not in robot.body_names:
        return None
    body_id = robot.body_names.index(body_name)
    return float(robot.data.body_pose_w[0, body_id, 1].item())


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=_resolve(args.robot_usd),
        part_usd_path=_resolve(args.part_usd),
        part_position=(2.0, 2.0, 0.2),
        part_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    scene = InteractiveScene(scene_cfg)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    hand_joint_names = [name for name in ("left_finger_joint", "right_finger_joint") if name in name_to_idx]
    if len(hand_joint_names) != 2:
        raise RuntimeError(f"Could not resolve both finger joints. joint_names={robot.joint_names}")
    hand_joint_ids = [name_to_idx[name] for name in hand_joint_names]
    command_joint_names = [name for name in hand_joint_names if is_gripper_command_joint_name(name)]
    command_joint_ids = [name_to_idx[name] for name in command_joint_names]

    print(f"robot_usd={_resolve(args.robot_usd)}", flush=True)
    print(f"joint_names={robot.joint_names}", flush=True)
    print(f"hand_joint_names={hand_joint_names}", flush=True)
    print(f"hand_joint_ids={hand_joint_ids}", flush=True)
    print(f"command_joint_names={command_joint_names}", flush=True)
    print(f"command_joint_ids={command_joint_ids}", flush=True)
    print(f"hand_joint_limits={_hand_limits(robot, hand_joint_ids)}", flush=True)
    print(f"initial_hand_q={_hand_state(robot, hand_joint_ids)}", flush=True)
    print(
        "initial_body_y="
        f"base={_body_y(robot, 'gripper_base_link')} "
        f"left={_body_y(robot, 'left_finger_link')} "
        f"right={_body_y(robot, 'right_finger_link')}",
        flush=True,
    )

    for width in (0.084, 0.05932855919589185, 0.001):
        target = torch.zeros((1, len(command_joint_ids)), dtype=torch.float32, device=robot.device)
        for index, name in enumerate(command_joint_names):
            target[0, index] = gripper_joint_target_from_width(name, width)
        print(f"target_width={width:.6f} target_q={[float(v) for v in target[0].tolist()]}", flush=True)
        for step in range(1, max(1, int(args.steps_per_target)) + 1):
            robot.set_joint_position_target(target, joint_ids=command_joint_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            if step in {1, 10, 50, 100, int(args.steps_per_target)}:
                print(
                    f"  step={step} hand_q={_hand_state(robot, hand_joint_ids)} "
                    f"body_y=left:{_body_y(robot, 'left_finger_link')} right:{_body_y(robot, 'right_finger_link')}",
                    flush=True,
                )

    simulation_app.close()


if __name__ == "__main__":
    main()
