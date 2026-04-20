"""Compare built-in and imported Franka USD assets on direct joint-target realization.

The intent is to isolate articulation realization behavior from higher-level controller logic:

1. Spawn a robot asset in a simple FR3 + far-away cube scene.
2. Drive to the repository's nominal start pose.
3. Solve a target grasp pose with the existing IK stack.
4. Hold the solved joint target directly.
5. Measure final joint error and TCP pose error.

This can be run against the built-in Isaac FR3 asset and any USDs listed in a
manifest produced by ``prepare_franka_asset_variants.py``.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--asset-manifest",
    type=Path,
    required=True,
    help="Manifest produced by prepare_franka_asset_variants.py.",
)
parser.add_argument(
    "--include-builtin",
    action="store_true",
    help="Include the built-in Isaac FR3 USD in the comparison.",
)
parser.add_argument(
    "--source-label",
    action="append",
    default=[],
    help="Optional source_label filter from the manifest. May be passed multiple times.",
)
parser.add_argument(
    "--robot-type",
    action="append",
    default=[],
    help="Optional robot_type filter from the manifest. May be passed multiple times.",
)
parser.add_argument(
    "--target-tcp-position",
    type=float,
    nargs=3,
    action="append",
    default=[],
    metavar=("X", "Y", "Z"),
    help="Explicit TCP target in world coordinates. May be repeated. Defaults to a top-down height sweep over the cube.",
)
parser.add_argument(
    "--target-tcp-orientation",
    type=float,
    nargs=4,
    default=(0.0, 1.0, 0.0, 0.0),
    metavar=("X", "Y", "Z", "W"),
    help="Shared TCP target orientation in xyzw order.",
)
parser.add_argument(
    "--hold-seconds",
    type=float,
    default=2.5,
    help="Seconds to hold the solved joint target before measuring final realization error.",
)
parser.add_argument(
    "--ik-max-iters",
    type=int,
    default=220,
    help="Maximum IK iterations per target.",
)
parser.add_argument(
    "--tcp-to-grasp-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, -0.045),
    metavar=("X", "Y", "Z"),
    help="Override the fixed TCP-to-grasp-center offset used for pose conversion.",
)
parser.add_argument(
    "--output-json",
    type=Path,
    default=Path("artifacts/franka_asset_realization_comparison.json"),
    help="Output JSON report.",
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
from grasp_planning.envs.fr3_cube_env import DEFAULT_ARM_START_JOINT_POS, DEFAULT_HAND_START_JOINT_POS  # noqa: E402
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.planning.goal_ik import GoalIKSolver  # noqa: E402
from grasp_planning.planning.types import PoseCommand  # noqa: E402
from grasp_planning.scene_defaults import ROBOT_BASE_ORIENTATION_XYZW, ROBOT_BASE_POSITION  # noqa: E402


@dataclass
class TargetResult:
    target_tcp_position_w: tuple[float, float, float]
    target_tcp_orientation_xyzw: tuple[float, float, float, float]
    solved: bool
    ik_goal_q: list[float] | None
    final_q: list[float] | None
    joint_error: list[float] | None
    max_joint_error: float | None
    actual_tcp_position_w: tuple[float, float, float] | None
    actual_tcp_orientation_xyzw: tuple[float, float, float, float] | None
    tcp_position_error_xyz: tuple[float, float, float] | None
    tcp_orientation_error_xyz: tuple[float, float, float] | None
    tcp_position_error_norm: float | None
    tcp_orientation_error_norm: float | None
    status: str


def resolve_builtin_fr3_usd_path() -> str:
    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in FR3 asset.")
    return assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"


def default_target_tcp_positions() -> list[tuple[float, float, float]]:
    cube_center = (-0.45, 0.0, 0.025)
    cube_half_extent_z = 0.025
    cube_top_z = cube_center[2] + cube_half_extent_z
    return [(cube_center[0], cube_center[1], cube_top_z + offset) for offset in (0.10, 0.06, 0.03, 0.02, 0.01)]


def load_asset_entries() -> list[dict]:
    manifest = json.loads(args_cli.asset_manifest.read_text(encoding="utf-8"))
    source_filter = set(args_cli.source_label)
    robot_filter = set(args_cli.robot_type)
    entries = []
    if args_cli.include_builtin:
        entries.append(
            {
                "source_label": "builtin_isaac",
                "robot_type": "fr3",
                "usd_path": resolve_builtin_fr3_usd_path(),
                "source_kind": "builtin_isaac",
                "source_tag": None,
            }
        )
    for variant in manifest["variants"]:
        usd_path = variant.get("usd_path")
        if not usd_path:
            continue
        if source_filter and variant["source_label"] not in source_filter:
            continue
        if robot_filter and variant["robot_type"] not in robot_filter:
            continue
        entries.append(variant)
    if not entries:
        raise RuntimeError("No asset entries selected for comparison.")
    return entries


def _remap_joint_name(base_name: str, robot_type: str) -> str:
    if robot_type == "fr3":
        return base_name
    return base_name.replace("fr3_", f"{robot_type}_", 1)


def _remap_joint_targets(robot_type: str) -> dict[str, float]:
    joint_targets = {_remap_joint_name(name, robot_type): value for name, value in DEFAULT_ARM_START_JOINT_POS.items()}
    joint_targets.update(
        {_remap_joint_name(name, robot_type): value for name, value in DEFAULT_HAND_START_JOINT_POS.items()}
    )
    return joint_targets


def build_scene(asset_path: str, robot_type: str) -> tuple[sim_utils.SimulationContext, InteractiveScene]:
    omni.usd.get_context().new_stage()
    simulation_app.update()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    sim.set_camera_view([1.6, -1.2, 1.0], [0.35, 0.0, 0.3])
    scene_cfg = make_fr3_cube_scene_cfg(
        fr3_asset_path=asset_path,
        cube_position=(2.0, 2.0, 2.0),
        cube_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
    )
    scene_cfg.robot.init_state.joint_pos = _remap_joint_targets(robot_type)
    scene_cfg.robot.actuators["fr3_arm"].joint_names_expr = [rf"{robot_type}_joint[1-7]"]
    scene_cfg.robot.actuators["fr3_hand"].joint_names_expr = [rf"{robot_type}_finger_joint.*"]
    scene = InteractiveScene(scene_cfg)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    sim.reset()
    scene.reset()
    return sim, scene


def drive_robot_to_start_pose(sim, scene, robot_type: str) -> None:
    robot = scene["robot"]
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_joint_names = [_remap_joint_name(name, robot_type) for name in DEFAULT_ARM_START_JOINT_POS.keys()]
    arm_joint_ids = [joint_name_to_idx[name] for name in arm_joint_names]
    arm_targets = torch.tensor(
        [[DEFAULT_ARM_START_JOINT_POS[name.replace(f"{robot_type}_", "fr3_", 1)] for name in arm_joint_names]],
        dtype=torch.float32,
        device=robot.device,
    )
    hand_joint_names = tuple(name for name in robot.joint_names if name.startswith(f"{robot_type}_finger_joint"))
    hand_target = float(DEFAULT_HAND_START_JOINT_POS["fr3_finger_joint.*"])
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]
    physics_dt = sim.get_physics_dt()

    for _ in range(max(1, int(1.5 / physics_dt))):
        robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
        if hand_joint_names:
            hand_targets = torch.full(
                (1, len(hand_joint_ids)),
                hand_target,
                dtype=torch.float32,
                device=robot.device,
            )
            robot.set_joint_position_target(hand_targets, joint_ids=hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)


def destroy_scene(sim, scene) -> None:
    try:
        del scene
        del sim
        omni.usd.get_context().close_stage()
        omni.usd.get_context().new_stage()
        simulation_app.update()
    finally:
        pass


def _target_result_from_failure(
    target_tcp_position_w: tuple[float, float, float],
    target_tcp_orientation_xyzw: tuple[float, float, float, float],
    status: str,
) -> TargetResult:
    return TargetResult(
        target_tcp_position_w=target_tcp_position_w,
        target_tcp_orientation_xyzw=target_tcp_orientation_xyzw,
        solved=False,
        ik_goal_q=None,
        final_q=None,
        joint_error=None,
        max_joint_error=None,
        actual_tcp_position_w=None,
        actual_tcp_orientation_xyzw=None,
        tcp_position_error_xyz=None,
        tcp_orientation_error_xyz=None,
        tcp_position_error_norm=None,
        tcp_orientation_error_norm=None,
        status=status,
    )


def evaluate_asset(asset_entry: dict) -> dict:
    sim, scene = build_scene(asset_entry["usd_path"], asset_entry["robot_type"])
    try:
        robot = scene["robot"]
        physics_dt = sim.get_physics_dt()
        for _ in range(max(1, int(0.1 / physics_dt))):
            scene.write_data_to_sim()
            sim.step()
            scene.update(physics_dt)
        drive_robot_to_start_pose(sim, scene, asset_entry["robot_type"])

        FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET = tuple(float(v) for v in args_cli.tcp_to_grasp_offset)
        context = FR3MotionContext(robot=robot, scene=scene, sim=sim, fixed_gripper_width=0.04)
        solver = GoalIKSolver(context)
        hold_steps = max(1, int(args_cli.hold_seconds / physics_dt))
        target_tcp_orientation_xyzw = tuple(float(v) for v in args_cli.target_tcp_orientation)
        target_positions = args_cli.target_tcp_position or default_target_tcp_positions()

        results: list[TargetResult] = []
        for target_tcp_position_w in target_positions:
            target_tcp_position_w = tuple(float(v) for v in target_tcp_position_w)
            grasp_position_w, grasp_orientation_xyzw = FR3MotionContext.tcp_pose_to_grasp_pose(
                target_tcp_position_w,
                target_tcp_orientation_xyzw,
            )
            q_start = context.get_arm_q().clone()
            goal_q = solver.solve(
                PoseCommand(position_w=grasp_position_w, orientation_xyzw=grasp_orientation_xyzw),
                max_iters=args_cli.ik_max_iters,
                restore_start_state=True,
            )
            context.hold_position(q_start, steps=10)
            if goal_q is None:
                results.append(
                    _target_result_from_failure(
                        target_tcp_position_w=target_tcp_position_w,
                        target_tcp_orientation_xyzw=target_tcp_orientation_xyzw,
                        status="ik_failed",
                    )
                )
                continue

            context.hold_position(goal_q, steps=hold_steps)
            final_q = context.get_arm_q()
            joint_error = (final_q - goal_q)[0]
            target_position_tensor = torch.tensor([grasp_position_w], dtype=torch.float32, device=context.device)
            target_orientation_tensor = torch.tensor(
                [grasp_orientation_xyzw], dtype=torch.float32, device=context.device
            )
            pos_error, rot_error = context.compute_pose_error(target_position_tensor, target_orientation_tensor)
            tcp_pos_w, tcp_quat_w = context.get_tcp_pose_w()
            actual_tcp_position_w = tuple(float(v) for v in tcp_pos_w[0].tolist())
            actual_tcp_orientation_xyzw = (
                float(tcp_quat_w[0, 1].item()),
                float(tcp_quat_w[0, 2].item()),
                float(tcp_quat_w[0, 3].item()),
                float(tcp_quat_w[0, 0].item()),
            )
            results.append(
                TargetResult(
                    target_tcp_position_w=target_tcp_position_w,
                    target_tcp_orientation_xyzw=target_tcp_orientation_xyzw,
                    solved=True,
                    ik_goal_q=[float(v) for v in goal_q[0].tolist()],
                    final_q=[float(v) for v in final_q[0].tolist()],
                    joint_error=[float(v) for v in joint_error.tolist()],
                    max_joint_error=float(torch.max(torch.abs(joint_error)).item()),
                    actual_tcp_position_w=actual_tcp_position_w,
                    actual_tcp_orientation_xyzw=actual_tcp_orientation_xyzw,
                    tcp_position_error_xyz=tuple(float(v) for v in pos_error[0].tolist()),
                    tcp_orientation_error_xyz=tuple(float(v) for v in rot_error[0].tolist()),
                    tcp_position_error_norm=float(torch.linalg.norm(pos_error).item()),
                    tcp_orientation_error_norm=float(torch.linalg.norm(rot_error).item()),
                    status="ok",
                )
            )
            context.hold_position(q_start, steps=10)

        return {
            "asset": {
                "source_label": asset_entry["source_label"],
                "robot_type": asset_entry["robot_type"],
                "source_kind": asset_entry.get("source_kind"),
                "source_tag": asset_entry.get("source_tag"),
                "usd_path": asset_entry["usd_path"],
            },
            "ee_body_name": context.ee_body_name,
            "arm_joint_names": list(context.arm_joint_names),
            "results": [asdict(result) for result in results],
        }
    finally:
        destroy_scene(sim, scene)


def main() -> None:
    comparison = {
        "assets": [],
    }
    for entry in load_asset_entries():
        print(
            f"[INFO] Evaluating asset source_label={entry['source_label']} robot_type={entry['robot_type']}",
            flush=True,
        )
        comparison["assets"].append(evaluate_asset(entry))

    args_cli.output_json.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote comparison report to {args_cli.output_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
