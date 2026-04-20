#!/usr/bin/env python3
"""Run one saved Fabrica grasp in MuJoCo using MoveIt2 for arm path planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning import (
    accepted_grasps,
    build_pickup_pose_world,
    evaluate_saved_grasps_against_pickup_pose,
    load_grasp_bundle,
    sample_pickup_placement_spec,
    saved_grasp_to_world_grasp,
    score_grasps,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.moveit import MoveItHeadlessFr3Server, MoveItJointPlanner, MoveItPlannerConfig
from grasp_planning.mujoco import (
    MujocoAttemptResult,
    MujocoExecutionConfig,
    MujocoPickupRuntime,
    MujocoTrajectoryPoint,
    build_bundle_local_mesh,
    load_robot_config,
    write_temporary_triangle_mesh_stl,
)


def _parse_vec2(raw: str) -> tuple[float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 2:
        raise ValueError(f"Expected exactly 2 comma-separated values, got '{raw}'.")
    return values


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated float, got '{raw}'.")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated token, got '{raw}'.")
    return values


def _resolve_placement_spec_from_bundle(args_cli, bundle, *, require_bundle_pose: bool = False):
    from grasp_planning.grasping.fabrica_grasp_debug import PickupPlacementSpec

    metadata = dict(bundle.metadata)
    support_face = str(metadata.get("pickup_support_face", "")).strip()
    yaw_deg = metadata.get("pickup_yaw_deg")
    xy_world = metadata.get("pickup_xy_world", (0.0, 0.0))

    if support_face and yaw_deg is not None:
        base_spec = PickupPlacementSpec(
            support_face=support_face,
            yaw_deg=float(yaw_deg),
            xy_world=tuple(float(v) for v in xy_world),
        )
    elif require_bundle_pose:
        raise RuntimeError(
            "Bundle pickup metadata is incomplete. Pass --support-face, --yaw-deg, and optionally --xy-world "
            "or rerun stage 2 so the bundle stores the pickup pose."
        )
    else:
        rng = np.random.default_rng(args_cli.seed)
        base_spec = sample_pickup_placement_spec(
            rng=rng,
            allowed_support_faces=_parse_str_tuple(args_cli.allowed_support_faces),
            allowed_yaw_deg=_parse_float_tuple(args_cli.allowed_yaw_deg),
            xy_min_world=_parse_vec2(args_cli.xy_min_world),
            xy_max_world=_parse_vec2(args_cli.xy_max_world),
        )

    return PickupPlacementSpec(
        support_face=args_cli.support_face or base_spec.support_face,
        yaw_deg=float(args_cli.yaw_deg) if args_cli.yaw_deg is not None else float(base_spec.yaw_deg),
        xy_world=_parse_vec2(args_cli.xy_world) if args_cli.xy_world else tuple(float(v) for v in base_spec.xy_world),
    )


def _bundle_execution_pose_world(bundle) -> ObjectWorldPose | None:
    metadata = dict(bundle.metadata)
    raw_pose = metadata.get("execution_world_pose")
    if not isinstance(raw_pose, dict):
        return None
    position_world = raw_pose.get("position_world")
    orientation_xyzw_world = raw_pose.get("orientation_xyzw_world")
    if not isinstance(position_world, (list, tuple)) or not isinstance(orientation_xyzw_world, (list, tuple)):
        return None
    if len(position_world) != 3 or len(orientation_xyzw_world) != 4:
        return None
    return ObjectWorldPose(
        position_world=tuple(float(v) for v in position_world),
        orientation_xyzw_world=tuple(float(v) for v in orientation_xyzw_world),
    )


def _explicit_pose_spec(object_pose_world: ObjectWorldPose):
    return type(
        "PlacementSpec",
        (),
        {
            "support_face": "explicit_pose",
            "yaw_deg": 0.0,
            "xy_world": tuple(float(v) for v in object_pose_world.position_world[:2]),
        },
    )()


def _to_mujoco_points(plan) -> list[MujocoTrajectoryPoint]:
    return [
        MujocoTrajectoryPoint(positions=tuple(point), time_from_start_s=time_s)
        for point, time_s in zip(plan.points, plan.time_from_start_s)
    ]


def _write_attempt_artifact(
    *,
    output_path: Path,
    placement_spec,
    object_pose_world,
    selected_grasp,
    result: MujocoAttemptResult,
    planning_times: dict[str, float],
    move_group_log: str | None,
) -> None:
    payload = {
        "placement": {
            "support_face": placement_spec.support_face,
            "yaw_deg": placement_spec.yaw_deg,
            "xy_world": list(placement_spec.xy_world),
            "object_position_world": list(object_pose_world.position_world),
            "object_orientation_xyzw_world": list(object_pose_world.orientation_xyzw_world),
        },
        "selected_grasp": {
            "grasp_id": selected_grasp.grasp_id,
            "position_w": list(selected_grasp.position_w),
            "orientation_xyzw": list(selected_grasp.orientation_xyzw),
            "pregrasp_position_w": list(selected_grasp.pregrasp_position_w),
            "gripper_width": selected_grasp.gripper_width,
            "jaw_width": selected_grasp.jaw_width,
        },
        "planning_times_s": planning_times,
        "move_group_log": move_group_log,
        "result": {
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "pregrasp_reached": result.pregrasp_reached,
            "grasp_reached": result.grasp_reached,
            "initial_object_position_world": list(result.initial_object_position_world),
            "final_object_position_world": list(result.final_object_position_world),
            "lift_height_m": result.lift_height_m,
            "target_lift_height_m": result.target_lift_height_m,
            "position_error_m": result.position_error_m,
            "orientation_error_rad": result.orientation_error_rad,
            "generated_scene_xml": result.generated_scene_xml,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _failure_result(
    *,
    status: str,
    message: str,
    initial_object_position_world: np.ndarray,
    final_object_position_world: np.ndarray,
    generated_scene_xml: str | None,
    pregrasp_reached: bool,
    grasp_reached: bool,
    target_lift_height_m: float,
    position_error_m: float | None = None,
    orientation_error_rad: float | None = None,
) -> MujocoAttemptResult:
    return MujocoAttemptResult(
        success=False,
        status=status,
        message=message,
        pregrasp_reached=pregrasp_reached,
        grasp_reached=grasp_reached,
        initial_object_position_world=tuple(float(v) for v in initial_object_position_world),
        final_object_position_world=tuple(float(v) for v in final_object_position_world),
        lift_height_m=0.0,
        target_lift_height_m=float(target_lift_height_m),
        position_error_m=position_error_m,
        orientation_error_rad=orientation_error_rad,
        generated_scene_xml=generated_scene_xml,
    )


def _fmt_vec(values) -> str:
    return "[" + ", ".join(f"{float(v): .6f}" for v in values) + "]"


def _log_pose_state(*, label: str, position_w, orientation_xyzw) -> None:
    print(
        f"[DEBUG]: {label} position_w={_fmt_vec(position_w)} orientation_xyzw={_fmt_vec(orientation_xyzw)}",
        flush=True,
    )


def _log_ik_target(*, label: str, runtime, target_position_w, target_orientation_xyzw) -> None:
    current_position_w, current_orientation_xyzw = runtime.site_pose()
    pos_error = np.asarray(target_position_w, dtype=float) - np.asarray(current_position_w, dtype=float)
    print(
        f"[DEBUG]: {label} site_position_w={_fmt_vec(current_position_w)} "
        f"target_position_w={_fmt_vec(target_position_w)} pos_error_w={_fmt_vec(pos_error)}",
        flush=True,
    )
    print(
        f"[DEBUG]: {label} site_orientation_xyzw={_fmt_vec(current_orientation_xyzw)} "
        f"target_orientation_xyzw={_fmt_vec(target_orientation_xyzw)}",
        flush=True,
    )


def _log_gripper_state(*, label: str, runtime) -> None:
    state = runtime.gripper_state_summary()
    print(
        f"[DEBUG]: {label} gripper_opening={state['opening']:.6f} "
        f"max_abs_velocity={state['max_abs_velocity']:.6f} "
        f"qpos={_fmt_vec(state['qpos'])} qvel={_fmt_vec(state['qvel'])}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True, help="Input grasp bundle, typically from stage 2.")
    parser.add_argument("--robot-config", type=Path, required=True, help="MuJoCo robot binding JSON.")
    parser.add_argument("--grasp-id", type=str, default="", help="Optional explicit grasp id to execute.")
    parser.add_argument("--support-face", type=str, default="", help="Optional explicit support face.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Optional explicit pickup yaw in degrees.")
    parser.add_argument("--xy-world", type=str, default="", help="Optional explicit world XY as x,y.")
    parser.add_argument("--allowed-support-faces", type=str, default="pos_x,neg_x,pos_y,neg_y,neg_z")
    parser.add_argument("--allowed-yaw-deg", type=str, default="0,90,180,270")
    parser.add_argument("--xy-min-world", type=str, default="-0.45,-0.05")
    parser.add_argument("--xy-max-world", type=str, default="-0.35,0.05")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pregrasp-offset", type=float, default=0.20)
    parser.add_argument("--gripper-width-clearance", type=float, default=0.01)
    parser.add_argument("--contact-gap-m", type=float, default=0.002)
    parser.add_argument("--object-mass-kg", type=float, default=0.15)
    parser.add_argument("--object-scale", type=float, default=1.0)
    parser.add_argument("--ik-max-iters", type=int, default=200)
    parser.add_argument("--ik-damping", type=float, default=1.0e-3)
    parser.add_argument("--ik-step-size", type=float, default=0.7)
    parser.add_argument("--ik-position-tolerance-m", type=float, default=0.003)
    parser.add_argument("--ik-orientation-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--close-steps", type=int, default=240)
    parser.add_argument("--hold-steps", type=int, default=240)
    parser.add_argument("--lift-height-m", type=float, default=0.12)
    parser.add_argument("--success-height-margin-m", type=float, default=0.05)
    parser.add_argument("--moveit-group", type=str, default="fr3_arm")
    parser.add_argument("--moveit-pipeline-id", type=str, default="move_group")
    parser.add_argument("--moveit-planner-id", type=str, default="")
    parser.add_argument("--moveit-planning-time", type=float, default=5.0)
    parser.add_argument("--moveit-velocity-scale", type=float, default=0.03)
    parser.add_argument("--moveit-acceleration-scale", type=float, default=0.03)
    parser.add_argument(
        "--trajectory-time-scale",
        type=float,
        default=10.0,
        help="Extra slowdown factor applied when executing planned arm trajectories in MuJoCo.",
    )
    parser.add_argument(
        "--trajectory-min-segment-seconds",
        type=float,
        default=0.20,
        help="Minimum duration for each executed arm trajectory segment in MuJoCo.",
    )
    parser.add_argument("--goal-joint-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--viewer", action="store_true", help="Open the MuJoCo passive viewer during execution.")
    parser.add_argument("--viewer-left-ui", action="store_true", help="Show the left MuJoCo viewer UI.")
    parser.add_argument("--viewer-right-ui", action="store_true", help="Show the right MuJoCo viewer UI.")
    parser.add_argument(
        "--viewer-no-realtime",
        action="store_true",
        help="Do not slow the simulation down to approximate real time in the viewer.",
    )
    parser.add_argument(
        "--viewer-hold-seconds",
        type=float,
        default=8.0,
        help="How long to keep the viewer open after the run finishes when --viewer is set.",
    )
    parser.add_argument(
        "--viewer-block-at-end",
        action="store_true",
        help="Keep the viewer open until you close it manually.",
    )
    parser.add_argument(
        "--skip-ground-recheck",
        action="store_true",
        help="Trust the input bundle as already ground-feasible and skip host-side pickup re-evaluation.",
    )
    parser.add_argument("--skip-start-move-group", action="store_true")
    parser.add_argument(
        "--debug-log-poses",
        action="store_true",
        help="Print detailed object/site/target pose logs around MuJoCo IK and planning.",
    )
    parser.add_argument("--move-group-log", type=Path, default=Path("artifacts/move_group_headless.log"))
    parser.add_argument(
        "--attempt-artifact",
        type=Path,
        default=Path("artifacts/mujoco_pick_attempt_moveit.json"),
    )
    parser.add_argument("--keep-generated-scene", action="store_true")
    args_cli = parser.parse_args()

    bundle = load_grasp_bundle(args_cli.input_json)
    mesh_local = build_bundle_local_mesh(bundle)
    object_mesh_path = write_temporary_triangle_mesh_stl(mesh_local, prefix=f"{args_cli.input_json.stem}_bundle_local_")
    bundle_execution_pose_world = _bundle_execution_pose_world(bundle)
    has_pickup_override = bool(args_cli.support_face or args_cli.yaw_deg is not None or args_cli.xy_world)
    if bundle_execution_pose_world is not None and not has_pickup_override:
        object_pose_world = bundle_execution_pose_world
        placement_spec = _explicit_pose_spec(object_pose_world)
    else:
        placement_spec = _resolve_placement_spec_from_bundle(
            args_cli,
            bundle,
            require_bundle_pose=bool(args_cli.skip_ground_recheck),
        )
        object_pose_world = build_pickup_pose_world(
            mesh_local,
            support_face=placement_spec.support_face,
            yaw_deg=placement_spec.yaw_deg,
            xy_world=placement_spec.xy_world,
        )
    if args_cli.skip_ground_recheck:
        feasible = score_grasps(bundle.candidates, mesh_local=mesh_local)
        if not feasible:
            raise RuntimeError("The input bundle does not contain any grasps.")
        if args_cli.grasp_id:
            selected_grasp = next((grasp for grasp in feasible if grasp.grasp_id == args_cli.grasp_id), None)
            if selected_grasp is None:
                raise RuntimeError(f"Requested grasp id '{args_cli.grasp_id}' is not present in the trusted bundle.")
        else:
            selected_grasp = feasible[0]
    else:
        statuses = evaluate_saved_grasps_against_pickup_pose(
            bundle.candidates,
            object_pose_world=object_pose_world,
            contact_gap_m=args_cli.contact_gap_m,
        )
        feasible = accepted_grasps(statuses)
        if not feasible:
            raise RuntimeError("No ground-feasible grasps remain for the requested pickup pose.")
        feasible = score_grasps(feasible, mesh_local=mesh_local)
        if args_cli.grasp_id:
            selected_grasp = next((grasp for grasp in feasible if grasp.grasp_id == args_cli.grasp_id), None)
            if selected_grasp is None:
                raise RuntimeError(
                    f"Requested grasp id '{args_cli.grasp_id}' is not ground-feasible for this pickup pose."
                )
        else:
            selected_grasp = feasible[0]

    selected_world_grasp = saved_grasp_to_world_grasp(
        selected_grasp,
        object_pose_world,
        pregrasp_offset=args_cli.pregrasp_offset,
        gripper_width_clearance=args_cli.gripper_width_clearance,
        frame_convention="mesh_grasp",
    )
    robot_cfg = load_robot_config(args_cli.robot_config)
    execution_cfg = MujocoExecutionConfig(
        ik_max_iters=args_cli.ik_max_iters,
        ik_damping=args_cli.ik_damping,
        ik_step_size=args_cli.ik_step_size,
        ik_position_tolerance_m=args_cli.ik_position_tolerance_m,
        ik_orientation_tolerance_rad=args_cli.ik_orientation_tolerance_rad,
        close_steps=args_cli.close_steps,
        hold_steps=args_cli.hold_steps,
        trajectory_time_scale=args_cli.trajectory_time_scale,
        trajectory_min_segment_duration_s=args_cli.trajectory_min_segment_seconds,
        object_mass_kg=args_cli.object_mass_kg,
        object_scale=float(args_cli.object_scale),
        lift_height_m=args_cli.lift_height_m,
        success_height_margin_m=args_cli.success_height_margin_m,
    )
    planner_cfg = MoveItPlannerConfig(
        group_name=args_cli.moveit_group,
        pipeline_id=args_cli.moveit_pipeline_id,
        planner_id=args_cli.moveit_planner_id,
        allowed_planning_time=args_cli.moveit_planning_time,
        max_velocity_scaling_factor=args_cli.moveit_velocity_scale,
        max_acceleration_scaling_factor=args_cli.moveit_acceleration_scale,
        goal_tolerance=args_cli.goal_joint_tolerance,
        arm_joint_names=tuple(robot_cfg.arm_joint_names),
    )
    planning_times: dict[str, float] = {}

    server_cm = None
    if not args_cli.skip_start_move_group:
        server_cm = MoveItHeadlessFr3Server(planner_cfg, log_path=args_cli.move_group_log)

    try:
        with server_cm if server_cm is not None else _NullContext():
            planner = MoveItJointPlanner()
            try:
                if not planner.wait_until_ready(timeout_sec=30.0):
                    raise RuntimeError("MoveIt `/plan_kinematic_path` service did not become ready.")

                runtime = MujocoPickupRuntime(
                    robot_cfg=robot_cfg,
                    execution_cfg=execution_cfg,
                    object_mesh_path=object_mesh_path,
                    object_pose_world=object_pose_world,
                    keep_generated_scene=args_cli.keep_generated_scene,
                )
                try:
                    viewer_cm = _NullContext()
                    if args_cli.viewer:
                        import mujoco.viewer  # type: ignore

                        viewer_cm = mujoco.viewer.launch_passive(
                            runtime.model,
                            runtime.data,
                            show_left_ui=args_cli.viewer_left_ui,
                            show_right_ui=args_cli.viewer_right_ui,
                        )

                    with viewer_cm as viewer:
                        if args_cli.viewer:
                            runtime.attach_viewer(viewer, realtime=not args_cli.viewer_no_realtime)

                        runtime.settle_home()
                        initial_object_position_world = runtime.object_position_world()
                        if args_cli.debug_log_poses:
                            _log_pose_state(
                                label="object_pose_world",
                                position_w=object_pose_world.position_world,
                                orientation_xyzw=object_pose_world.orientation_xyzw_world,
                            )
                            _log_pose_state(
                                label="selected_grasp",
                                position_w=selected_world_grasp.position_w,
                                orientation_xyzw=selected_world_grasp.orientation_xyzw,
                            )
                            _log_pose_state(
                                label="selected_pregrasp",
                                position_w=selected_world_grasp.pregrasp_position_w,
                                orientation_xyzw=selected_world_grasp.orientation_xyzw,
                            )
                            _log_gripper_state(label="home_state", runtime=runtime)
                            _log_ik_target(
                                label="pregrasp_target",
                                runtime=runtime,
                                target_position_w=selected_world_grasp.pregrasp_position_w,
                                target_orientation_xyzw=selected_world_grasp.orientation_xyzw,
                            )

                        pregrasp_q, pregrasp_pos_err, pregrasp_rot_err = runtime.solve_ik(
                            selected_world_grasp.pregrasp_position_w,
                            selected_world_grasp.orientation_xyzw,
                            gripper_ctrl=robot_cfg.open_gripper_ctrl,
                        )
                        if pregrasp_q is None:
                            if args_cli.debug_log_poses:
                                _log_ik_target(
                                    label="pregrasp_failure",
                                    runtime=runtime,
                                    target_position_w=selected_world_grasp.pregrasp_position_w,
                                    target_orientation_xyzw=selected_world_grasp.orientation_xyzw,
                                )
                            result = _failure_result(
                                status="pregrasp_ik_failed",
                                message=(
                                    "MuJoCo IK failed for pregrasp before MoveIt planning. "
                                    f"position_error={pregrasp_pos_err:.4f} orientation_error={pregrasp_rot_err:.4f}"
                                ),
                                initial_object_position_world=initial_object_position_world,
                                final_object_position_world=runtime.object_position_world(),
                                generated_scene_xml=runtime.generated_scene_xml_path
                                if args_cli.keep_generated_scene
                                else None,
                                pregrasp_reached=False,
                                grasp_reached=False,
                                target_lift_height_m=execution_cfg.success_height_margin_m,
                                position_error_m=pregrasp_pos_err,
                                orientation_error_rad=pregrasp_rot_err,
                            )
                        else:
                            pregrasp_plan = planner.plan_joint_path(
                                start_positions=runtime.get_arm_qpos(),
                                goal_positions=pregrasp_q,
                                cfg=planner_cfg,
                            )
                            planning_times["pregrasp"] = pregrasp_plan.planning_time_s
                            if not runtime.execute_arm_trajectory(
                                joint_names=pregrasp_plan.joint_names,
                                points=_to_mujoco_points(pregrasp_plan),
                                gripper_ctrl=robot_cfg.open_gripper_ctrl,
                            ):
                                result = _failure_result(
                                    status="pregrasp_execution_failed",
                                    message="MoveIt pregrasp plan executed but did not settle in MuJoCo.",
                                    initial_object_position_world=initial_object_position_world,
                                    final_object_position_world=runtime.object_position_world(),
                                    generated_scene_xml=runtime.generated_scene_xml_path
                                    if args_cli.keep_generated_scene
                                    else None,
                                    pregrasp_reached=False,
                                    grasp_reached=False,
                                    target_lift_height_m=execution_cfg.success_height_margin_m,
                                )
                            else:
                                if args_cli.debug_log_poses:
                                    _log_ik_target(
                                        label="grasp_target",
                                        runtime=runtime,
                                        target_position_w=selected_world_grasp.position_w,
                                        target_orientation_xyzw=selected_world_grasp.orientation_xyzw,
                                    )
                                grasp_q, grasp_pos_err, grasp_rot_err = runtime.solve_ik(
                                    selected_world_grasp.position_w,
                                    selected_world_grasp.orientation_xyzw,
                                    gripper_ctrl=robot_cfg.open_gripper_ctrl,
                                )
                                if grasp_q is None:
                                    if args_cli.debug_log_poses:
                                        _log_ik_target(
                                            label="grasp_failure",
                                            runtime=runtime,
                                            target_position_w=selected_world_grasp.position_w,
                                            target_orientation_xyzw=selected_world_grasp.orientation_xyzw,
                                        )
                                    result = _failure_result(
                                        status="grasp_ik_failed",
                                        message=(
                                            "MuJoCo IK failed for grasp before MoveIt planning. "
                                            f"position_error={grasp_pos_err:.4f} orientation_error={grasp_rot_err:.4f}"
                                        ),
                                        initial_object_position_world=initial_object_position_world,
                                        final_object_position_world=runtime.object_position_world(),
                                        generated_scene_xml=runtime.generated_scene_xml_path
                                        if args_cli.keep_generated_scene
                                        else None,
                                        pregrasp_reached=True,
                                        grasp_reached=False,
                                        target_lift_height_m=execution_cfg.success_height_margin_m,
                                        position_error_m=grasp_pos_err,
                                        orientation_error_rad=grasp_rot_err,
                                    )
                                else:
                                    grasp_plan = planner.plan_joint_path(
                                        start_positions=runtime.get_arm_qpos(),
                                        goal_positions=grasp_q,
                                        cfg=planner_cfg,
                                    )
                                    planning_times["grasp"] = grasp_plan.planning_time_s
                                    if not runtime.execute_arm_trajectory(
                                        joint_names=grasp_plan.joint_names,
                                        points=_to_mujoco_points(grasp_plan),
                                        gripper_ctrl=robot_cfg.open_gripper_ctrl,
                                    ):
                                        result = _failure_result(
                                            status="grasp_execution_failed",
                                            message="MoveIt grasp plan executed but did not settle in MuJoCo.",
                                            initial_object_position_world=initial_object_position_world,
                                            final_object_position_world=runtime.object_position_world(),
                                            generated_scene_xml=runtime.generated_scene_xml_path
                                            if args_cli.keep_generated_scene
                                            else None,
                                            pregrasp_reached=True,
                                            grasp_reached=False,
                                            target_lift_height_m=execution_cfg.success_height_margin_m,
                                        )
                                    else:
                                        close_summary = runtime.close_gripper()
                                        if args_cli.debug_log_poses:
                                            _log_gripper_state(label="post_close_state", runtime=runtime)
                                            print(
                                                f"[DEBUG]: post_close_settled={close_summary['settled']} "
                                                f"stable_steps={close_summary['stable_steps']}",
                                                flush=True,
                                            )
                                        lift_target = (
                                            float(selected_world_grasp.position_w[0]),
                                            float(selected_world_grasp.position_w[1]),
                                            float(selected_world_grasp.position_w[2] + execution_cfg.lift_height_m),
                                        )
                                        if args_cli.debug_log_poses:
                                            _log_ik_target(
                                                label="lift_target",
                                                runtime=runtime,
                                                target_position_w=lift_target,
                                                target_orientation_xyzw=selected_world_grasp.orientation_xyzw,
                                            )
                                        lift_q, lift_pos_err, lift_rot_err = runtime.solve_ik(
                                            lift_target,
                                            selected_world_grasp.orientation_xyzw,
                                            gripper_ctrl=robot_cfg.closed_gripper_ctrl,
                                        )
                                        if lift_q is None:
                                            if args_cli.debug_log_poses:
                                                _log_ik_target(
                                                    label="lift_failure",
                                                    runtime=runtime,
                                                    target_position_w=lift_target,
                                                    target_orientation_xyzw=selected_world_grasp.orientation_xyzw,
                                                )
                                            result = _failure_result(
                                                status="lift_ik_failed",
                                                message=(
                                                    "MuJoCo IK failed for lift before MoveIt planning. "
                                                    f"position_error={lift_pos_err:.4f} orientation_error={lift_rot_err:.4f}"
                                                ),
                                                initial_object_position_world=initial_object_position_world,
                                                final_object_position_world=runtime.object_position_world(),
                                                generated_scene_xml=runtime.generated_scene_xml_path
                                                if args_cli.keep_generated_scene
                                                else None,
                                                pregrasp_reached=True,
                                                grasp_reached=True,
                                                target_lift_height_m=execution_cfg.success_height_margin_m,
                                                position_error_m=lift_pos_err,
                                                orientation_error_rad=lift_rot_err,
                                            )
                                        else:
                                            lift_plan = planner.plan_joint_path(
                                                start_positions=runtime.get_arm_qpos(),
                                                goal_positions=lift_q,
                                                cfg=planner_cfg,
                                            )
                                            planning_times["lift"] = lift_plan.planning_time_s
                                            runtime.execute_arm_trajectory(
                                                joint_names=lift_plan.joint_names,
                                                points=_to_mujoco_points(lift_plan),
                                                gripper_ctrl=robot_cfg.closed_gripper_ctrl,
                                            )
                                            runtime.hold_closed()
                                            final_object_position_world = runtime.object_position_world()
                                            lift_height_m = float(
                                                final_object_position_world[2] - initial_object_position_world[2]
                                            )
                                            target_lift_height_m = float(execution_cfg.success_height_margin_m)
                                            success = lift_height_m >= target_lift_height_m
                                            result = MujocoAttemptResult(
                                                success=success,
                                                status="ok" if success else "lift_failed",
                                                message=(
                                                    f"Object lifted by {lift_height_m:.4f} m (required {target_lift_height_m:.4f} m)."
                                                    if success
                                                    else f"Object only lifted by {lift_height_m:.4f} m (required {target_lift_height_m:.4f} m)."
                                                ),
                                                pregrasp_reached=True,
                                                grasp_reached=True,
                                                initial_object_position_world=tuple(
                                                    float(v) for v in initial_object_position_world
                                                ),
                                                final_object_position_world=tuple(
                                                    float(v) for v in final_object_position_world
                                                ),
                                                lift_height_m=lift_height_m,
                                                target_lift_height_m=target_lift_height_m,
                                                generated_scene_xml=runtime.generated_scene_xml_path
                                                if args_cli.keep_generated_scene
                                                else None,
                                            )
                        if args_cli.viewer:
                            runtime.hold_viewer_open(
                                seconds=None if args_cli.viewer_block_at_end else args_cli.viewer_hold_seconds
                            )
                finally:
                    runtime.close()
            finally:
                planner.close()
    finally:
        if not args_cli.keep_generated_scene:
            try:
                object_mesh_path.unlink()
            except FileNotFoundError:
                pass

    _write_attempt_artifact(
        output_path=args_cli.attempt_artifact,
        placement_spec=placement_spec,
        object_pose_world=object_pose_world,
        selected_grasp=selected_world_grasp,
        result=result,
        planning_times=planning_times,
        move_group_log=None if args_cli.skip_start_move_group else str(args_cli.move_group_log),
    )
    print(
        f"[INFO]: MoveIt+MuJoCo grasp attempt finished success={result.success} status={result.status} "
        f"lift_height_m={result.lift_height_m:.4f} message={result.message}",
        flush=True,
    )
    print(f"[INFO]: Wrote attempt artifact to {args_cli.attempt_artifact}", flush=True)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()
