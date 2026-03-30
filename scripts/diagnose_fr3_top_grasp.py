"""Systematic diagnosis for FR3 top-grasp target realizability in Isaac Sim."""

from __future__ import annotations

import argparse
import math
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose whether FR3 top-grasp failures come from target pose or tracking.")
parser.add_argument("--fr3-usd", type=str, default="", help="Optional override for the FR3 USD asset path.")
parser.add_argument("--grasp-face", type=str, default="pos_z", choices=("pos_z", "pos_x", "neg_x", "pos_y", "neg_y", "neg_z"))
parser.add_argument("--pregrasp-offset", type=float, default=0.20)
parser.add_argument("--baselines-only", action="store_true", help="Only run the baseline pregrasp/grasp comparisons.")
parser.add_argument(
    "--tcp-to-grasp-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, -0.045),
    metavar=("X", "Y", "Z"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
import torch  # noqa: E402

from grasp_planning import CubeFaceGraspGenerator, FR3AdmittanceController  # noqa: E402
from grasp_planning.envs import make_fr3_cube_scene_cfg  # noqa: E402
from grasp_planning.envs.fr3_cube_env import DEFAULT_ARM_START_JOINT_POS, DEFAULT_CUBE_CFG, DEFAULT_HAND_START_JOINT_POS  # noqa: E402
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.planning.types import PoseCommand  # noqa: E402
from grasp_planning.scene_defaults import CUBE_ORIENTATION_XYZW, CUBE_POSITION, ROBOT_BASE_ORIENTATION_XYZW, ROBOT_BASE_POSITION  # noqa: E402


def resolve_fr3_usd_path() -> str:
    if args_cli.fr3_usd:
        return args_cli.fr3_usd
    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in FR3 asset.")
    return assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaFR3/fr3.usd"


def quat_mul_xyzw(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def axis_angle_quat_xyzw(axis: tuple[float, float, float], degrees: float) -> tuple[float, float, float, float]:
    angle = math.radians(float(degrees))
    half = 0.5 * angle
    s = math.sin(half)
    ax, ay, az = axis
    return (ax * s, ay * s, az * s, math.cos(half))


def build_scene() -> tuple[sim_utils.SimulationContext, InteractiveScene]:
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
    return sim, scene


def settle_home(sim, scene) -> None:
    robot = scene["robot"]
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_joint_names = tuple(DEFAULT_ARM_START_JOINT_POS.keys())
    arm_joint_ids = [joint_name_to_idx[name] for name in arm_joint_names]
    arm_targets = torch.tensor(
        [[DEFAULT_ARM_START_JOINT_POS[name] for name in arm_joint_names]],
        dtype=torch.float32,
        device=robot.device,
    )
    hand_joint_names = tuple(name for name in robot.joint_names if name.startswith("fr3_finger_joint"))
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]
    hand_target = float(DEFAULT_HAND_START_JOINT_POS["fr3_finger_joint.*"])
    physics_dt = sim.get_physics_dt()
    for _ in range(max(1, int(1.5 / physics_dt))):
        robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
        if hand_joint_ids:
            hand_targets = torch.full((1, len(hand_joint_ids)), hand_target, dtype=torch.float32, device=robot.device)
            robot.set_joint_position_target(hand_targets, joint_ids=hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)


def build_selected_grasp():
    label_from_face = {
        "pos_x": "+x",
        "neg_x": "-x",
        "pos_y": "+y",
        "neg_y": "-y",
        "pos_z": "+z",
        "neg_z": "-z",
    }
    requested_label = label_from_face[args_cli.grasp_face]
    generator = CubeFaceGraspGenerator(cube_size=DEFAULT_CUBE_CFG.size, pregrasp_offset=args_cli.pregrasp_offset)
    for grasp in generator.generate(
        cube_position_w=CUBE_POSITION,
        cube_orientation_xyzw=CUBE_ORIENTATION_XYZW,
        robot_base_position_w=ROBOT_BASE_POSITION,
    ):
        if grasp.label == requested_label:
            return grasp
    raise RuntimeError(f"No grasp candidate for label {requested_label}.")


def get_tcp_pose(context: FR3MotionContext) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pos_w, quat_w = context.get_tcp_pose_w()
    return (
        tuple(float(v) for v in pos_w[0].tolist()),
        (
            float(quat_w[0, 1].item()),
            float(quat_w[0, 2].item()),
            float(quat_w[0, 3].item()),
            float(quat_w[0, 0].item()),
        ),
    )


def offline_solve_pose(context: FR3MotionContext, pose: PoseCommand) -> dict[str, object]:
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

    q_start = context.get_arm_q()
    q_current = q_start.clone()
    zero_vel = torch.zeros_like(q_current)
    lower, upper = context.get_joint_limits()
    cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(cfg=cfg, num_envs=1, device=context.device)
    target_position = torch.tensor([pose.position_w], dtype=torch.float32, device=context.device)
    target_orientation = torch.tensor([pose.orientation_xyzw], dtype=torch.float32, device=context.device)
    best = {
        "q": q_start.clone(),
        "pos": float("inf"),
        "rot": float("inf"),
        "actual_tcp_position": None,
        "actual_tcp_orientation": None,
    }
    try:
        for _ in range(220):
            context.robot.write_joint_state_to_sim(q_current, zero_vel, joint_ids=context.arm_joint_ids)
            context.scene.update(0.0)
            q_des = context.command_pose_via_differential_ik(ik_controller, pose)
            if context.joint_limits_are_usable(lower, upper):
                q_des = torch.max(torch.min(q_des, upper), lower)
            context.robot.write_joint_state_to_sim(q_des, zero_vel, joint_ids=context.arm_joint_ids)
            context.scene.update(0.0)
            pos_error, rot_error = context.compute_pose_error(target_position, target_orientation)
            pos_norm = float(torch.linalg.norm(pos_error).item())
            rot_norm = float(torch.linalg.norm(rot_error).item())
            q_current = q_des.clone()
            if pos_norm + rot_norm < best["pos"] + best["rot"]:
                actual_pos, actual_quat = get_tcp_pose(context)
                best = {
                    "q": q_current.clone(),
                    "pos": pos_norm,
                    "rot": rot_norm,
                    "actual_tcp_position": actual_pos,
                    "actual_tcp_orientation": actual_quat,
                }
    finally:
        context.robot.write_joint_state_to_sim(q_start, zero_vel, joint_ids=context.arm_joint_ids)
        context.scene.update(0.0)
    return best


def run_admittance_pose(robot, scene, sim, pose: PoseCommand) -> dict[str, object]:
    controller = FR3AdmittanceController(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=float(DEFAULT_HAND_START_JOINT_POS["fr3_finger_joint.*"]),
    )
    result = controller.move_to_pose(position_w=pose.position_w, orientation_xyzw=pose.orientation_xyzw)
    actual_pos, actual_quat = controller.get_current_tcp_pose()
    target_pos = torch.tensor([pose.position_w], dtype=torch.float32, device=robot.device)
    target_quat = torch.tensor([pose.orientation_xyzw], dtype=torch.float32, device=robot.device)
    pos_err, rot_err = controller._context.compute_pose_error(target_pos, target_quat)  # diagnostic only
    return {
        "success": result.success,
        "status": result.status,
        "message": result.message,
        "pos": float(torch.linalg.norm(pos_err).item()),
        "rot": float(torch.linalg.norm(rot_err).item()),
        "actual_tcp_position": actual_pos,
        "actual_tcp_orientation": actual_quat,
    }


def track_exact_q(context: FR3MotionContext, pose: PoseCommand, q_target: torch.Tensor, steps: int = 600) -> dict[str, object]:
    target_position = torch.tensor([pose.position_w], dtype=torch.float32, device=context.device)
    target_orientation = torch.tensor([pose.orientation_xyzw], dtype=torch.float32, device=context.device)
    context.hold_position(q_target, steps=steps)
    actual_pos, actual_quat = get_tcp_pose(context)
    actual_q = context.get_arm_q()
    joint_error = actual_q - q_target
    pos_err, rot_err = context.compute_pose_error(target_position, target_orientation)
    return {
        "pos": float(torch.linalg.norm(pos_err).item()),
        "rot": float(torch.linalg.norm(rot_err).item()),
        "actual_tcp_position": actual_pos,
        "actual_tcp_orientation": actual_quat,
        "joint_error": tuple(float(v) for v in joint_error[0].tolist()),
        "max_joint_error": float(torch.max(torch.abs(joint_error)).item()),
    }


def track_exact_q_with_gains(
    context: FR3MotionContext,
    pose: PoseCommand,
    q_target: torch.Tensor,
    *,
    stiffness: float,
    damping: float,
    steps: int = 600,
) -> dict[str, object]:
    num_joints = int(context.arm_joint_ids.numel())
    stiffness_tensor = torch.full((1, num_joints), float(stiffness), dtype=torch.float32, device=context.device)
    damping_tensor = torch.full((1, num_joints), float(damping), dtype=torch.float32, device=context.device)
    context.robot.write_joint_stiffness_to_sim(stiffness_tensor, joint_ids=context.arm_joint_ids)
    context.robot.write_joint_damping_to_sim(damping_tensor, joint_ids=context.arm_joint_ids)
    return track_exact_q(context, pose, q_target, steps=steps)


def summarize_case(
    label: str,
    pose: PoseCommand,
    offline: dict[str, object],
    adm: dict[str, object] | None = None,
    tracked: dict[str, object] | None = None,
) -> None:
    print(
        f"{label}: target_pos={pose.position_w} target_quat={pose.orientation_xyzw} "
        f"offline_pos_err={offline['pos']:.4f} offline_rot_err={offline['rot']:.4f} "
        f"offline_actual={offline['actual_tcp_position']}",
        flush=True,
    )
    if adm is not None:
        print(
            f"  admittance: success={adm['success']} status={adm['status']} "
            f"pos_err={adm['pos']:.4f} rot_err={adm['rot']:.4f} actual={adm['actual_tcp_position']}",
            flush=True,
        )
    if tracked is not None:
        print(
            f"  exact_q_tracking: pos_err={tracked['pos']:.4f} rot_err={tracked['rot']:.4f} "
            f"actual={tracked['actual_tcp_position']} max_joint_error={tracked['max_joint_error']:.4f} "
            f"joint_error={tracked['joint_error']}",
            flush=True,
        )


def main() -> None:
    FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET = tuple(float(v) for v in args_cli.tcp_to_grasp_offset)
    sim, scene = build_scene()
    settle_home(sim, scene)
    robot = scene["robot"]
    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=float(DEFAULT_HAND_START_JOINT_POS["fr3_finger_joint.*"]),
    )
    grasp = build_selected_grasp()
    pre_tcp_pos, pre_tcp_quat = FR3MotionContext.grasp_pose_to_tcp_pose(grasp.pregrasp_position_w, grasp.orientation_xyzw)
    grasp_tcp_pos, grasp_tcp_quat = FR3MotionContext.grasp_pose_to_tcp_pose(grasp.position_w, grasp.orientation_xyzw)

    print(f"selected_grasp label={grasp.label} grasp_position={grasp.position_w} pregrasp_position={grasp.pregrasp_position_w}", flush=True)
    print(f"pregrasp_tcp={pre_tcp_pos} quat={pre_tcp_quat}", flush=True)
    print(f"grasp_tcp={grasp_tcp_pos} quat={grasp_tcp_quat}", flush=True)

    base_pre = PoseCommand(position_w=grasp.pregrasp_position_w, orientation_xyzw=grasp.orientation_xyzw)
    base_grasp = PoseCommand(position_w=grasp.position_w, orientation_xyzw=grasp.orientation_xyzw)

    # Baseline: offline IK and admittance at pregrasp and grasp.
    settle_home(sim, scene)
    pre_offline = offline_solve_pose(context, base_pre)
    settle_home(sim, scene)
    pre_adm = run_admittance_pose(robot, scene, sim, base_pre)
    settle_home(sim, scene)
    pre_tracked = track_exact_q(context, base_pre, pre_offline["q"])
    settle_home(sim, scene)
    pre_stiff_tracked = track_exact_q_with_gains(context, base_pre, pre_offline["q"], stiffness=2000.0, damping=200.0)
    summarize_case("baseline_pregrasp", base_pre, pre_offline, pre_adm, pre_tracked)
    print(
        f"  exact_q_tracking_high_gains: pos_err={pre_stiff_tracked['pos']:.4f} rot_err={pre_stiff_tracked['rot']:.4f} "
        f"actual={pre_stiff_tracked['actual_tcp_position']} max_joint_error={pre_stiff_tracked['max_joint_error']:.4f} "
        f"joint_error={pre_stiff_tracked['joint_error']}",
        flush=True,
    )

    settle_home(sim, scene)
    grasp_offline = offline_solve_pose(context, base_grasp)
    settle_home(sim, scene)
    grasp_adm = run_admittance_pose(robot, scene, sim, base_grasp)
    settle_home(sim, scene)
    grasp_tracked = track_exact_q(context, base_grasp, grasp_offline["q"])
    summarize_case("baseline_grasp", base_grasp, grasp_offline, grasp_adm, grasp_tracked)

    if args_cli.baselines_only:
        return

    # Position sweep around the final grasp target in x/z.
    best_position_case: tuple[str, dict[str, object], PoseCommand] | None = None
    for x_offset in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02):
        for z_offset in (0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
            pose = PoseCommand(
                position_w=(
                    grasp.position_w[0] + x_offset,
                    grasp.position_w[1],
                    grasp.position_w[2] + z_offset,
                ),
                orientation_xyzw=grasp.orientation_xyzw,
            )
            settle_home(sim, scene)
            offline = offline_solve_pose(context, pose)
            label = f"pos_sweep dx={x_offset:+.3f} dz={z_offset:+.3f}"
            summarize_case(label, pose, offline)
            if best_position_case is None or (offline["pos"] + offline["rot"]) < (best_position_case[1]["pos"] + best_position_case[1]["rot"]):
                best_position_case = (label, offline, pose)

    # Orientation sweep at the nominal final position.
    best_orientation_case: tuple[str, dict[str, object], PoseCommand] | None = None
    for roll_deg in (-10.0, -5.0, 0.0, 5.0, 10.0):
        for pitch_deg in (-10.0, -5.0, 0.0, 5.0, 10.0):
            roll_q = axis_angle_quat_xyzw((1.0, 0.0, 0.0), roll_deg)
            pitch_q = axis_angle_quat_xyzw((0.0, 1.0, 0.0), pitch_deg)
            quat = quat_mul_xyzw(quat_mul_xyzw(grasp.orientation_xyzw, roll_q), pitch_q)
            pose = PoseCommand(position_w=grasp.position_w, orientation_xyzw=quat)
            settle_home(sim, scene)
            offline = offline_solve_pose(context, pose)
            label = f"ori_sweep roll={roll_deg:+.1f} pitch={pitch_deg:+.1f}"
            summarize_case(label, pose, offline)
            if best_orientation_case is None or (offline["pos"] + offline["rot"]) < (best_orientation_case[1]["pos"] + best_orientation_case[1]["rot"]):
                best_orientation_case = (label, offline, pose)

    if best_position_case is not None:
        print(
            "best_position_case "
            f"{best_position_case[0]} target={best_position_case[2].position_w} "
            f"offline_pos_err={best_position_case[1]['pos']:.4f} offline_rot_err={best_position_case[1]['rot']:.4f} "
            f"actual={best_position_case[1]['actual_tcp_position']}",
            flush=True,
        )
    if best_orientation_case is not None:
        print(
            "best_orientation_case "
            f"{best_orientation_case[0]} target_quat={best_orientation_case[2].orientation_xyzw} "
            f"offline_pos_err={best_orientation_case[1]['pos']:.4f} offline_rot_err={best_orientation_case[1]['rot']:.4f} "
            f"actual={best_orientation_case[1]['actual_tcp_position']}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[ERROR]: Unhandled exception in diagnose_fr3_top_grasp.py", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        simulation_app.close()
