"""Run one saved Fabrica grasp in MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping

import numpy as np
import yaml

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
from grasp_planning.mujoco import (
    MujocoExecutionConfig,
    MujocoRegraspAttemptResult,
    build_bundle_local_mesh,
    load_robot_config,
    run_regrasp_plan_in_mujoco,
    run_world_grasp_in_mujoco,
    write_temporary_triangle_mesh_stl,
)
from grasp_planning.pipeline.regrasp_fallback import load_mujoco_regrasp_plan
from grasp_planning.ros2.moveit_pose_commander import MoveItPoseCommander, MoveItPoseCommanderConfig, PoseTarget, rclpy
from grasp_planning.ros2.moveit_world_grasp import pose_target_from_world, world_grasp_pose_targets


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


def _tuple_floats(values: object, *, expected_len: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list/tuple of floats, got {values!r}.")
    parsed = tuple(float(value) for value in values)
    if len(parsed) != expected_len:
        raise ValueError(f"{field_name} must contain {expected_len} values, got {len(parsed)}.")
    return parsed


def _float_tuple(values: object, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list/tuple of floats, got {values!r}.")
    parsed = tuple(float(value) for value in values)
    if not parsed:
        raise ValueError(f"{field_name} must not be empty.")
    return parsed


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in '{path}'.")
    return payload


def _load_simulation_defaults(path: Path | None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "pregrasp_offset": 0.20,
        "gripper_width_clearance": 0.01,
        "contact_gap_m": 0.002,
        "robot_cfg_updates": {},
        "execution_cfg_kwargs": {},
    }
    if path is None:
        return defaults

    payload = _load_yaml(path)
    grasp = dict(payload.get("grasp", {}))
    scene = dict(payload.get("scene", {}))
    object_raw = dict(scene.get("object", {}))
    ground_raw = dict(scene.get("ground", {}))
    robot_raw = dict(payload.get("robot", {}))
    gripper_raw = dict(payload.get("gripper", {}))
    defaults["pregrasp_offset"] = float(grasp.get("pregrasp_offset_m", defaults["pregrasp_offset"]))
    defaults["gripper_width_clearance"] = float(
        grasp.get("gripper_width_clearance_m", defaults["gripper_width_clearance"])
    )
    defaults["contact_gap_m"] = float(grasp.get("contact_gap_m", defaults["contact_gap_m"]))

    robot_cfg_updates: dict[str, object] = {}
    if "timestep_s" in robot_raw:
        robot_cfg_updates["timestep"] = float(robot_raw["timestep_s"])
    if "control_substeps" in robot_raw:
        robot_cfg_updates["control_substeps"] = int(robot_raw["control_substeps"])
    if "open_ctrl" in gripper_raw:
        robot_cfg_updates["open_gripper_ctrl"] = _float_tuple(gripper_raw["open_ctrl"], field_name="gripper.open_ctrl")
    if "closed_ctrl" in gripper_raw:
        robot_cfg_updates["closed_gripper_ctrl"] = _float_tuple(
            gripper_raw["closed_ctrl"], field_name="gripper.closed_ctrl"
        )
    defaults["robot_cfg_updates"] = robot_cfg_updates

    execution_cfg_kwargs: dict[str, object] = {}
    if "mass_kg" in object_raw:
        execution_cfg_kwargs["object_mass_kg"] = float(object_raw["mass_kg"])
    if "scale" in object_raw:
        execution_cfg_kwargs["object_scale"] = float(object_raw["scale"])
    if "friction" in object_raw:
        execution_cfg_kwargs["object_friction"] = _tuple_floats(
            object_raw["friction"], expected_len=3, field_name="scene.object.friction"
        )
    if "condim" in object_raw:
        execution_cfg_kwargs["object_condim"] = int(object_raw["condim"])
    if "solref" in object_raw:
        execution_cfg_kwargs["object_solref"] = _tuple_floats(
            object_raw["solref"], expected_len=2, field_name="scene.object.solref"
        )
    if "solimp" in object_raw:
        execution_cfg_kwargs["object_solimp"] = _tuple_floats(
            object_raw["solimp"], expected_len=3, field_name="scene.object.solimp"
        )
    if "margin_m" in object_raw:
        execution_cfg_kwargs["object_margin"] = float(object_raw["margin_m"])
    if "gap_m" in object_raw:
        execution_cfg_kwargs["object_gap"] = float(object_raw["gap_m"])
    if "friction" in ground_raw:
        execution_cfg_kwargs["ground_friction"] = _tuple_floats(
            ground_raw["friction"], expected_len=3, field_name="scene.ground.friction"
        )
    if "settle_steps" in robot_raw:
        execution_cfg_kwargs["settle_steps"] = int(robot_raw["settle_steps"])
    if "ik_max_iters" in robot_raw:
        execution_cfg_kwargs["ik_max_iters"] = int(robot_raw["ik_max_iters"])
    if "ik_damping" in robot_raw:
        execution_cfg_kwargs["ik_damping"] = float(robot_raw["ik_damping"])
    if "ik_step_size" in robot_raw:
        execution_cfg_kwargs["ik_step_size"] = float(robot_raw["ik_step_size"])
    if "ik_position_tolerance_m" in robot_raw:
        execution_cfg_kwargs["ik_position_tolerance_m"] = float(robot_raw["ik_position_tolerance_m"])
    if "ik_orientation_tolerance_rad" in robot_raw:
        execution_cfg_kwargs["ik_orientation_tolerance_rad"] = float(robot_raw["ik_orientation_tolerance_rad"])
    if "trajectory_waypoints" in robot_raw:
        execution_cfg_kwargs["trajectory_waypoints"] = int(robot_raw["trajectory_waypoints"])
    if "waypoint_settle_steps" in robot_raw:
        execution_cfg_kwargs["waypoint_settle_steps"] = int(robot_raw["waypoint_settle_steps"])
    if "speed_scale" in robot_raw:
        execution_cfg_kwargs["arm_speed_scale"] = float(robot_raw["speed_scale"])
    if "lift_height_m" in robot_raw:
        execution_cfg_kwargs["lift_height_m"] = float(robot_raw["lift_height_m"])
    if "regrasp_transport_clearance_m" in robot_raw:
        execution_cfg_kwargs["regrasp_transport_clearance_m"] = float(robot_raw["regrasp_transport_clearance_m"])
    if "hold_steps" in robot_raw:
        execution_cfg_kwargs["hold_steps"] = int(robot_raw["hold_steps"])
    if "success_height_margin_m" in robot_raw:
        execution_cfg_kwargs["success_height_margin_m"] = float(robot_raw["success_height_margin_m"])
    if "close_steps" in gripper_raw:
        execution_cfg_kwargs["close_steps"] = int(gripper_raw["close_steps"])
    if "settle_position_delta_m" in gripper_raw:
        execution_cfg_kwargs["gripper_settle_position_delta_m"] = float(gripper_raw["settle_position_delta_m"])
    if "settle_velocity_mps" in gripper_raw:
        execution_cfg_kwargs["gripper_settle_velocity_mps"] = float(gripper_raw["settle_velocity_mps"])
    if "settle_consecutive_steps" in gripper_raw:
        execution_cfg_kwargs["gripper_settle_consecutive_steps"] = int(gripper_raw["settle_consecutive_steps"])
    defaults["execution_cfg_kwargs"] = execution_cfg_kwargs
    return defaults


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


def _resolve_placement_spec_from_bundle(args_cli, bundle):
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


def _resolve_object_pose_world_from_bundle(args_cli, bundle, mesh_local):
    bundle_execution_pose_world = _bundle_execution_pose_world(bundle)
    has_pickup_override = bool(args_cli.support_face or args_cli.yaw_deg is not None or args_cli.xy_world)
    if bundle_execution_pose_world is not None and not has_pickup_override:
        object_pose_world = bundle_execution_pose_world
        return _explicit_pose_spec(object_pose_world), object_pose_world

    placement_spec = _resolve_placement_spec_from_bundle(args_cli, bundle)
    object_pose_world = build_pickup_pose_world(
        mesh_local,
        support_face=placement_spec.support_face,
        yaw_deg=placement_spec.yaw_deg,
        xy_world=placement_spec.xy_world,
    )
    return placement_spec, object_pose_world


def _attempt_result_payload(*, selected_grasp, result) -> dict[str, object]:
    return {
        "selected_grasp": {
            "grasp_id": selected_grasp.grasp_id,
            "position_w": list(selected_grasp.position_w),
            "orientation_xyzw": list(selected_grasp.orientation_xyzw),
            "pregrasp_position_w": list(selected_grasp.pregrasp_position_w),
            "gripper_width": selected_grasp.gripper_width,
            "jaw_width": selected_grasp.jaw_width,
        },
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


def _write_attempt_artifact(
    *,
    output_path: Path,
    placement_spec,
    object_pose_world,
    selected_grasp,
    result,
    attempts: list[dict[str, object]] | None = None,
) -> None:
    attempt_payload = _attempt_result_payload(selected_grasp=selected_grasp, result=result)
    payload = {
        "placement": {
            "support_face": placement_spec.support_face,
            "yaw_deg": placement_spec.yaw_deg,
            "xy_world": list(placement_spec.xy_world),
            "object_position_world": list(object_pose_world.position_world),
            "object_orientation_xyzw_world": list(object_pose_world.orientation_xyzw_world),
        },
        **attempt_payload,
    }
    if attempts is not None:
        payload["attempts"] = attempts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_regrasp_attempt_artifact(
    *,
    output_path: Path,
    input_json: Path,
    regrasp_plan_json: Path,
    plan,
    transfer_initial_world_grasp,
    transfer_staging_world_grasp,
    final_world_grasp,
    result,
    attempts: list[dict[str, object]] | None = None,
    planned_candidates: list[dict[str, object]] | None = None,
) -> None:
    payload = {
        "mode": "mujoco_regrasp_fallback",
        "input_json": str(input_json),
        "regrasp_plan_json": str(regrasp_plan_json),
        "initial_object_pose_world": {
            "position_world": list(plan.initial_object_pose_world.position_world),
            "orientation_xyzw_world": list(plan.initial_object_pose_world.orientation_xyzw_world),
        },
        "staging_object_pose_world": {
            "position_world": list(plan.staging_object_pose_world.position_world),
            "orientation_xyzw_world": list(plan.staging_object_pose_world.orientation_xyzw_world),
        },
        "support_facet": asdict(plan.support_facet),
        "transfer_grasp_id": plan.transfer_grasp.grasp_id,
        "final_grasp_id": plan.final_grasp.grasp_id,
        "transfer_initial_world_grasp": asdict(transfer_initial_world_grasp),
        "transfer_staging_world_grasp": asdict(transfer_staging_world_grasp),
        "final_world_grasp": asdict(final_world_grasp),
        "result": _object_payload(result),
        "plan_metadata": dict(plan.metadata),
    }
    if attempts is not None:
        payload["attempts"] = attempts
    if planned_candidates is not None:
        payload["planned_candidates"] = planned_candidates
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _object_payload(value) -> dict[str, object]:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return dict(value)


def _ordered_feasible_grasps(*, feasible: list, requested_grasp_id: str) -> list:
    if requested_grasp_id:
        selected = next((grasp for grasp in feasible if grasp.grasp_id == requested_grasp_id), None)
        if selected is None:
            raise RuntimeError(
                f"Requested grasp id '{requested_grasp_id}' is not ground-feasible for this pickup pose."
            )
        return [selected]
    return list(feasible)


def _ordered_regrasp_candidates(primary, candidates: tuple) -> list:
    ordered = [primary]
    seen = {str(primary.grasp_id)}
    for candidate in candidates:
        if str(candidate.grasp_id) in seen:
            continue
        ordered.append(candidate)
        seen.add(str(candidate.grasp_id))
    return ordered


def _active_regrasp_plan_for_attempt(*, plan, placement_option, transfer_candidate, final_candidate):
    updates = {
        "staging_object_pose_world": placement_option.staging_object_pose_world,
        "support_facet": placement_option.support_facet,
        "transfer_grasp": transfer_candidate,
        "final_grasp": final_candidate,
        "transfer_grasp_candidates": placement_option.transfer_grasp_candidates,
        "final_grasp_candidates": placement_option.final_grasp_candidates,
    }
    return replace(plan, **updates) if is_dataclass(plan) else SimpleNamespace(**(dict(vars(plan)) | updates))


def _regrasp_failure_depends_on_transfer(status_or_message: str) -> bool:
    text = str(status_or_message)
    return any(
        token in text
        for token in (
            "transfer_pregrasp",
            "transfer_grasp",
            "transfer_lift",
            "transfer_transport_lift",
            "transfer_transport_rotate",
            "staging_transport",
            "staging_preplace",
            "placement",
            "staging_retreat",
        )
    )


def _home_arm_joint_positions(robot_cfg) -> tuple[float, ...]:
    missing = [name for name in robot_cfg.arm_joint_names if name not in robot_cfg.home_joint_positions]
    if missing:
        raise RuntimeError(
            f"MoveIt-backed MuJoCo execution requires home_joint_positions for all arm joints; missing={missing}."
        )
    return tuple(float(robot_cfg.home_joint_positions[name]) for name in robot_cfg.arm_joint_names)


def _trajectory_waypoints_for_joints(trajectory, *, joint_names: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    joint_trajectory = trajectory.joint_trajectory
    source_joint_names = tuple(str(name) for name in joint_trajectory.joint_names)
    name_to_index = {name: index for index, name in enumerate(source_joint_names)}
    missing = [name for name in joint_names if name not in name_to_index]
    if missing:
        raise RuntimeError(f"MoveIt trajectory is missing MuJoCo arm joints: {missing}.")
    ordered_indices = [name_to_index[name] for name in joint_names]
    waypoints = tuple(
        tuple(float(point.positions[index]) for index in ordered_indices) for point in tuple(joint_trajectory.points)
    )
    if not waypoints:
        raise RuntimeError("MoveIt returned a trajectory with no points.")
    return waypoints


def _moveit_config_from_args(args_cli, *, joint_names: tuple[str, ...]) -> MoveItPoseCommanderConfig:
    return MoveItPoseCommanderConfig(
        planning_group=str(args_cli.moveit_planning_group),
        pose_link=str(args_cli.moveit_pose_link),
        joint_names=joint_names,
        planner_id=str(args_cli.moveit_planner_id),
        wait_for_moveit_timeout_s=float(args_cli.moveit_wait_for_moveit_timeout_s),
        ik_timeout_s=float(args_cli.moveit_ik_timeout_s),
        fk_timeout_s=float(args_cli.moveit_ik_timeout_s),
        planning_time_s=float(args_cli.moveit_planning_time_s),
        num_planning_attempts=int(args_cli.moveit_num_planning_attempts),
        velocity_scale=float(args_cli.moveit_velocity_scale),
        acceleration_scale=float(args_cli.moveit_acceleration_scale),
        execute_timeout_s=float(args_cli.moveit_execute_timeout_s),
        post_execute_sleep_s=0.0,
        avoid_collisions=not bool(args_cli.moveit_allow_collisions),
    )


def _plan_moveit_target_sequence(
    args_cli,
    *,
    robot_cfg,
    targets: Mapping[str, PoseTarget],
    labels: tuple[str, ...],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace first.")
    initialized_here = False
    commander = None
    try:
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        moveit_config = _moveit_config_from_args(args_cli, joint_names=tuple(robot_cfg.arm_joint_names))
        commander = MoveItPoseCommander(moveit_config, node_name="mujoco_moveit_trajectory_planner")
        commander.wait_for_moveit(require_execute=False)
        start_joint_positions = _home_arm_joint_positions(robot_cfg)
        planned: dict[str, tuple[tuple[float, ...], ...]] = {}
        for label in labels:
            trajectory, message = commander.plan_to_pose(
                targets[label],
                label=f"mujoco_{label}",
                start_joint_positions=start_joint_positions,
            )
            if trajectory is None:
                raise RuntimeError(f"MoveIt failed to plan {label}: {message}")
            waypoints = _trajectory_waypoints_for_joints(trajectory, joint_names=tuple(robot_cfg.arm_joint_names))
            planned[label] = waypoints
            start_joint_positions = waypoints[-1]
        return planned
    finally:
        if commander is not None:
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def _plan_moveit_joint_trajectories(
    args_cli,
    *,
    robot_cfg,
    execution_cfg,
    world_grasp,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    targets = world_grasp_pose_targets(
        world_grasp,
        frame_id=str(args_cli.moveit_frame_id),
        lift_height_m=float(execution_cfg.lift_height_m),
    )
    return _plan_moveit_target_sequence(
        args_cli,
        robot_cfg=robot_cfg,
        targets=targets,
        labels=("pregrasp", "grasp", "lift"),
    )


def _regrasp_pose_targets(
    *,
    transfer_initial_world_grasp,
    transfer_staging_world_grasp,
    final_world_grasp,
    frame_id: str,
    lift_height_m: float,
    transport_clearance_m: float,
) -> dict[str, PoseTarget]:
    transfer_initial_orientation = tuple(float(v) for v in transfer_initial_world_grasp.orientation_xyzw)
    transfer_staging_orientation = tuple(float(v) for v in transfer_staging_world_grasp.orientation_xyzw)
    final_orientation = tuple(float(v) for v in final_world_grasp.orientation_xyzw)
    transport_z = max(
        float(transfer_initial_world_grasp.position_w[2]),
        float(transfer_staging_world_grasp.position_w[2]),
    ) + max(float(lift_height_m), float(transport_clearance_m))
    return {
        "transfer_pregrasp": pose_target_from_world(
            position_xyz=tuple(float(v) for v in transfer_initial_world_grasp.pregrasp_position_w),
            orientation_xyzw=transfer_initial_orientation,
            frame_id=frame_id,
        ),
        "transfer_grasp": pose_target_from_world(
            position_xyz=tuple(float(v) for v in transfer_initial_world_grasp.position_w),
            orientation_xyzw=transfer_initial_orientation,
            frame_id=frame_id,
        ),
        "transfer_lift": pose_target_from_world(
            position_xyz=(
                float(transfer_initial_world_grasp.position_w[0]),
                float(transfer_initial_world_grasp.position_w[1]),
                float(transfer_initial_world_grasp.position_w[2] + lift_height_m),
            ),
            orientation_xyzw=transfer_initial_orientation,
            frame_id=frame_id,
        ),
        "transfer_transport_lift": pose_target_from_world(
            position_xyz=(
                float(transfer_initial_world_grasp.position_w[0]),
                float(transfer_initial_world_grasp.position_w[1]),
                float(transport_z),
            ),
            orientation_xyzw=transfer_initial_orientation,
            frame_id=frame_id,
        ),
        "transfer_transport_rotate": pose_target_from_world(
            position_xyz=(
                float(transfer_initial_world_grasp.position_w[0]),
                float(transfer_initial_world_grasp.position_w[1]),
                float(transport_z),
            ),
            orientation_xyzw=transfer_staging_orientation,
            frame_id=frame_id,
        ),
        "staging_transport": pose_target_from_world(
            position_xyz=(
                float(transfer_staging_world_grasp.position_w[0]),
                float(transfer_staging_world_grasp.position_w[1]),
                float(transport_z),
            ),
            orientation_xyzw=transfer_staging_orientation,
            frame_id=frame_id,
        ),
        "staging_preplace": pose_target_from_world(
            position_xyz=(
                float(transfer_staging_world_grasp.position_w[0]),
                float(transfer_staging_world_grasp.position_w[1]),
                float(transfer_staging_world_grasp.position_w[2] + lift_height_m),
            ),
            orientation_xyzw=transfer_staging_orientation,
            frame_id=frame_id,
        ),
        "placement": pose_target_from_world(
            position_xyz=tuple(float(v) for v in transfer_staging_world_grasp.position_w),
            orientation_xyzw=transfer_staging_orientation,
            frame_id=frame_id,
        ),
        "staging_retreat": pose_target_from_world(
            position_xyz=tuple(float(v) for v in transfer_staging_world_grasp.pregrasp_position_w),
            orientation_xyzw=transfer_staging_orientation,
            frame_id=frame_id,
        ),
        "final_pregrasp": pose_target_from_world(
            position_xyz=tuple(float(v) for v in final_world_grasp.pregrasp_position_w),
            orientation_xyzw=final_orientation,
            frame_id=frame_id,
        ),
        "final_grasp": pose_target_from_world(
            position_xyz=tuple(float(v) for v in final_world_grasp.position_w),
            orientation_xyzw=final_orientation,
            frame_id=frame_id,
        ),
        "final_lift": pose_target_from_world(
            position_xyz=(
                float(final_world_grasp.position_w[0]),
                float(final_world_grasp.position_w[1]),
                float(final_world_grasp.position_w[2] + lift_height_m),
            ),
            orientation_xyzw=final_orientation,
            frame_id=frame_id,
        ),
    }


def _plan_moveit_regrasp_joint_trajectories(
    args_cli,
    *,
    robot_cfg,
    execution_cfg,
    transfer_initial_world_grasp,
    transfer_staging_world_grasp,
    final_world_grasp,
) -> dict[str, tuple[tuple[float, ...], ...]]:
    targets = _regrasp_pose_targets(
        transfer_initial_world_grasp=transfer_initial_world_grasp,
        transfer_staging_world_grasp=transfer_staging_world_grasp,
        final_world_grasp=final_world_grasp,
        frame_id=str(args_cli.moveit_frame_id),
        lift_height_m=float(execution_cfg.lift_height_m),
        transport_clearance_m=float(execution_cfg.regrasp_transport_clearance_m),
    )
    return _plan_moveit_target_sequence(
        args_cli,
        robot_cfg=robot_cfg,
        targets=targets,
        labels=(
            "transfer_pregrasp",
            "transfer_grasp",
            "transfer_lift",
            "transfer_transport_lift",
            "transfer_transport_rotate",
            "staging_transport",
            "staging_preplace",
            "placement",
            "staging_retreat",
            "final_pregrasp",
            "final_grasp",
            "final_lift",
        ),
    )


def _moveit_joint_trajectory_diagnostics(
    trajectories: Mapping[str, tuple[tuple[float, ...], ...]],
) -> tuple[dict[str, object], ...]:
    diagnostics: list[dict[str, object]] = []
    for label, waypoints in trajectories.items():
        diagnostic: dict[str, object] = {
            "label": str(label),
            "point_count": int(len(waypoints)),
        }
        if not waypoints:
            diagnostic["joint_path_length_rad"] = 0.0
            diagnostic["max_joint_step_rad"] = 0.0
            diagnostics.append(diagnostic)
            continue
        q_matrix = np.asarray(waypoints, dtype=float)
        if q_matrix.ndim == 1:
            q_matrix = q_matrix.reshape(1, -1)
        q_deltas = np.diff(q_matrix, axis=0)
        if q_deltas.size:
            diagnostic["joint_path_length_rad"] = float(np.sum(np.linalg.norm(q_deltas, axis=1)))
            diagnostic["max_joint_step_rad"] = float(np.max(np.abs(q_deltas)))
        else:
            diagnostic["joint_path_length_rad"] = 0.0
            diagnostic["max_joint_step_rad"] = 0.0
        diagnostic["start_joint_positions"] = [float(v) for v in q_matrix[0]]
        diagnostic["end_joint_positions"] = [float(v) for v in q_matrix[-1]]
        diagnostics.append(diagnostic)
    return tuple(diagnostics)


def _moveit_joint_trajectory_summary(diagnostics: Iterable[Mapping[str, object]]) -> dict[str, object]:
    items = tuple(diagnostics)
    return {
        "segment_count": len(items),
        "point_count": int(sum(int(item.get("point_count", 0)) for item in items)),
        "joint_path_length_rad": float(sum(float(item.get("joint_path_length_rad", 0.0)) for item in items)),
        "max_joint_step_rad": float(max((float(item.get("max_joint_step_rad", 0.0)) for item in items), default=0.0)),
    }


def _moveit_regrasp_trajectory_cost(
    diagnostics: Iterable[Mapping[str, object]],
    *,
    placement_score: float = 0.0,
) -> float:
    carried_labels = {
        "transfer_lift",
        "transfer_transport_lift",
        "transfer_transport_rotate",
        "staging_transport",
        "staging_preplace",
        "placement",
    }
    cost = 0.0
    max_joint_step_rad = 0.0
    point_count = 0
    for diagnostic in diagnostics:
        label = str(diagnostic.get("label", ""))
        joint_path_length_rad = float(diagnostic.get("joint_path_length_rad", 0.0))
        max_joint_step_rad = max(max_joint_step_rad, float(diagnostic.get("max_joint_step_rad", 0.0)))
        point_count += int(diagnostic.get("point_count", 0))
        weight = 1.7 if label in carried_labels else 1.0
        cost += weight * joint_path_length_rad
    cost += 2.5 * max_joint_step_rad
    cost += 0.002 * float(point_count)
    cost -= 0.05 * float(placement_score)
    return float(cost)


def _print_moveit_joint_trajectory_diagnostics(
    trajectories: Mapping[str, tuple[tuple[float, ...], ...]],
    *,
    prefix: str,
) -> None:
    for diagnostic in _moveit_joint_trajectory_diagnostics(trajectories):
        print(
            f"[INFO]: {prefix} {diagnostic['label']}: "
            f"points={diagnostic['point_count']} "
            f"joint_path={float(diagnostic['joint_path_length_rad']):.3f} rad "
            f"max_step={float(diagnostic['max_joint_step_rad']):.3f} rad",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True, help="Input grasp bundle, typically from stage 2.")
    parser.add_argument(
        "--regrasp-plan-json",
        type=Path,
        default=None,
        help="Optional MuJoCo regrasp fallback plan artifact. When set, execute transfer-place-final-pick.",
    )
    parser.add_argument("--robot-config", type=Path, required=True, help="MuJoCo robot binding JSON.")
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=None,
        help="Optional YAML file with shared MuJoCo scene, robot, and gripper settings.",
    )
    parser.add_argument("--grasp-id", type=str, default="", help="Optional explicit grasp id to execute.")
    parser.add_argument("--support-face", type=str, default="", help="Optional explicit support face.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Optional explicit pickup yaw in degrees.")
    parser.add_argument("--xy-world", type=str, default="", help="Optional explicit world XY as x,y.")
    parser.add_argument(
        "--allowed-support-faces",
        type=str,
        default="pos_x,neg_x,pos_y,neg_y,neg_z",
        help="Comma-separated support faces used by the random sampler.",
    )
    parser.add_argument(
        "--allowed-yaw-deg",
        type=str,
        default="0,90,180,270",
        help="Comma-separated yaw values used by the random sampler.",
    )
    parser.add_argument("--xy-min-world", type=str, default="-0.45,-0.05", help="Random placement XY lower bound.")
    parser.add_argument("--xy-max-world", type=str, default="-0.35,0.05", help="Random placement XY upper bound.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--controller",
        choices=("native", "moveit"),
        default="native",
        help="Arm trajectory source: native MuJoCo IK or MoveIt-planned joint trajectories.",
    )
    parser.add_argument("--pregrasp-offset", type=float, default=None, help="Optional pregrasp offset in meters.")
    parser.add_argument(
        "--gripper-width-clearance",
        type=float,
        default=None,
        help="Optional clearance added to the saved grasp jaw width for the open approach width.",
    )
    parser.add_argument(
        "--contact-gap-m",
        type=float,
        default=None,
        help="Optional detailed Franka finger contact gap used during the ground recheck.",
    )
    parser.add_argument("--object-mass-kg", type=float, default=None, help="Optional target object mass in kg.")
    parser.add_argument("--object-scale", type=float, default=None, help="Optional uniform object mesh scale.")
    parser.add_argument("--lift-height-m", type=float, default=None, help="Optional vertical lift height after grasp.")
    parser.add_argument(
        "--success-height-margin-m",
        type=float,
        default=None,
        help="Optional required object Z gain for the attempt to count as success.",
    )
    parser.add_argument(
        "--attempt-artifact",
        type=Path,
        default=Path("artifacts/mujoco_pick_attempt.json"),
        help="Optional JSON artifact for the attempt.",
    )
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
        "--keep-generated-scene",
        action="store_true",
        help="Keep the generated MuJoCo scene XML for inspection.",
    )
    parser.add_argument("--moveit-frame-id", type=str, default="base", help="MoveIt planning frame.")
    parser.add_argument("--moveit-planning-group", type=str, default="fr3_arm", help="MoveIt planning group.")
    parser.add_argument("--moveit-pose-link", type=str, default="fr3_hand_tcp", help="MoveIt pose link.")
    parser.add_argument("--moveit-planner-id", type=str, default="", help="Optional MoveIt planner id.")
    parser.add_argument("--moveit-wait-for-moveit-timeout-s", type=float, default=15.0)
    parser.add_argument("--moveit-ik-timeout-s", type=float, default=2.0)
    parser.add_argument("--moveit-planning-time-s", type=float, default=5.0)
    parser.add_argument("--moveit-num-planning-attempts", type=int, default=5)
    parser.add_argument("--moveit-velocity-scale", type=float, default=0.05)
    parser.add_argument("--moveit-acceleration-scale", type=float, default=0.05)
    parser.add_argument("--moveit-execute-timeout-s", type=float, default=120.0)
    parser.add_argument("--moveit-allow-collisions", action="store_true")
    parser.add_argument(
        "--regrasp-moveit-max-candidate-plans",
        type=int,
        default=36,
        help="Maximum MoveIt regrasp candidate plans to score before execution; <=0 means unlimited.",
    )
    parser.add_argument(
        "--regrasp-moveit-transfer-candidates-per-placement",
        type=int,
        default=3,
        help="Maximum transfer grasps considered per placement option during MoveIt regrasp selection.",
    )
    parser.add_argument(
        "--regrasp-moveit-final-candidates-per-placement",
        type=int,
        default=3,
        help="Maximum final grasps considered per placement option during MoveIt regrasp selection.",
    )
    args_cli = parser.parse_args()
    simulation_defaults = _load_simulation_defaults(args_cli.simulation_config)
    pregrasp_offset = (
        float(args_cli.pregrasp_offset)
        if args_cli.pregrasp_offset is not None
        else float(simulation_defaults["pregrasp_offset"])
    )
    gripper_width_clearance = (
        float(args_cli.gripper_width_clearance)
        if args_cli.gripper_width_clearance is not None
        else float(simulation_defaults["gripper_width_clearance"])
    )
    contact_gap_m = (
        float(args_cli.contact_gap_m)
        if args_cli.contact_gap_m is not None
        else float(simulation_defaults["contact_gap_m"])
    )

    bundle = load_grasp_bundle(args_cli.input_json)
    mesh_local = build_bundle_local_mesh(bundle)
    object_mesh_path = write_temporary_triangle_mesh_stl(mesh_local, prefix=f"{args_cli.input_json.stem}_bundle_local_")
    robot_cfg = load_robot_config(args_cli.robot_config)
    robot_cfg_updates = dict(simulation_defaults["robot_cfg_updates"])
    if robot_cfg_updates:
        robot_cfg = replace(robot_cfg, **robot_cfg_updates)
    execution_cfg_kwargs = dict(simulation_defaults["execution_cfg_kwargs"])
    if args_cli.object_mass_kg is not None:
        execution_cfg_kwargs["object_mass_kg"] = float(args_cli.object_mass_kg)
    if args_cli.object_scale is not None:
        execution_cfg_kwargs["object_scale"] = float(args_cli.object_scale)
    if args_cli.lift_height_m is not None:
        execution_cfg_kwargs["lift_height_m"] = float(args_cli.lift_height_m)
    if args_cli.success_height_margin_m is not None:
        execution_cfg_kwargs["success_height_margin_m"] = float(args_cli.success_height_margin_m)
    execution_cfg = MujocoExecutionConfig(**execution_cfg_kwargs)
    try:
        if args_cli.regrasp_plan_json is not None:
            plan = load_mujoco_regrasp_plan(args_cli.regrasp_plan_json)
            print(
                "[INFO]: MuJoCo regrasp fallback "
                f"transfer={plan.transfer_grasp.grasp_id} final={plan.final_grasp.grasp_id}",
                flush=True,
            )
            placement_options = tuple(getattr(plan, "placement_options", ()) or ())
            if not placement_options:
                placement_options = (
                    SimpleNamespace(
                        staging_object_pose_world=plan.staging_object_pose_world,
                        support_facet=getattr(plan, "support_facet", None),
                        transfer_grasp=plan.transfer_grasp,
                        final_grasp=plan.final_grasp,
                        transfer_grasp_candidates=tuple(getattr(plan, "transfer_grasp_candidates", ()) or ()),
                        final_grasp_candidates=tuple(getattr(plan, "final_grasp_candidates", ()) or ()),
                        metadata={},
                    ),
                )
            max_attempts = 0
            for placement_option in placement_options:
                transfer_candidates_for_option = _ordered_regrasp_candidates(
                    placement_option.transfer_grasp,
                    tuple(getattr(placement_option, "transfer_grasp_candidates", ()) or ()),
                )
                final_candidates_for_option = _ordered_regrasp_candidates(
                    placement_option.final_grasp,
                    tuple(getattr(placement_option, "final_grasp_candidates", ()) or ()),
                )
                max_attempts += len(transfer_candidates_for_option) * len(final_candidates_for_option)
            attempts: list[dict[str, object]] = []
            regrasp_result = None
            transfer_initial_world_grasp = None
            transfer_staging_world_grasp = None
            final_world_grasp = None
            active_plan = plan
            attempt_index = 0
            planned_candidate_records: list[dict[str, object]] = []
            if args_cli.controller == "moveit":
                placement_infos = []
                transfer_limit = max(1, int(args_cli.regrasp_moveit_transfer_candidates_per_placement))
                final_limit = max(1, int(args_cli.regrasp_moveit_final_candidates_per_placement))
                max_candidate_plans = int(args_cli.regrasp_moveit_max_candidate_plans)
                for placement_option_index, placement_option in enumerate(placement_options, start=1):
                    option_plan = _active_regrasp_plan_for_attempt(
                        plan=plan,
                        placement_option=placement_option,
                        transfer_candidate=placement_option.transfer_grasp,
                        final_candidate=placement_option.final_grasp,
                    )
                    staging_xy = (
                        float(placement_option.staging_object_pose_world.position_world[0]),
                        float(placement_option.staging_object_pose_world.position_world[1]),
                    )
                    transfer_candidates = _ordered_regrasp_candidates(
                        placement_option.transfer_grasp,
                        placement_option.transfer_grasp_candidates,
                    )[:transfer_limit]
                    final_candidates = _ordered_regrasp_candidates(
                        placement_option.final_grasp,
                        placement_option.final_grasp_candidates,
                    )[:final_limit]
                    placement_infos.append(
                        SimpleNamespace(
                            placement_option_index=placement_option_index,
                            placement_option=placement_option,
                            active_plan=option_plan,
                            staging_xy=staging_xy,
                            transfer_candidates=transfer_candidates,
                            final_candidates=final_candidates,
                        )
                    )

                candidate_specs = []
                max_transfer_rank = max((len(info.transfer_candidates) for info in placement_infos), default=0)
                max_final_rank = max((len(info.final_candidates) for info in placement_infos), default=0)
                rank_pairs = sorted(
                    (
                        (transfer_rank, final_rank)
                        for transfer_rank in range(max_transfer_rank)
                        for final_rank in range(max_final_rank)
                    ),
                    key=lambda pair: (pair[0] + pair[1], pair[1], pair[0]),
                )
                limit_reached = False
                for transfer_rank, final_rank in rank_pairs:
                    for info in placement_infos:
                        if transfer_rank >= len(info.transfer_candidates) or final_rank >= len(info.final_candidates):
                            continue
                        candidate_specs.append(
                            SimpleNamespace(
                                info=info,
                                transfer_rank=transfer_rank,
                                final_rank=final_rank,
                                transfer_candidate=info.transfer_candidates[transfer_rank],
                                final_candidate=info.final_candidates[final_rank],
                            )
                        )
                        if max_candidate_plans > 0 and len(candidate_specs) >= max_candidate_plans:
                            limit_reached = True
                            break
                    if limit_reached:
                        break

                print(
                    f"[INFO]: Planning and scoring {len(candidate_specs)} MoveIt regrasp candidate(s) "
                    f"across {len(placement_infos)} placement option(s).",
                    flush=True,
                )
                planned_moveit_candidates = []
                transfer_planning_failures: set[tuple[int, str]] = set()
                for spec in candidate_specs:
                    info = spec.info
                    transfer_candidate = spec.transfer_candidate
                    final_candidate = spec.final_candidate
                    transfer_key = (int(info.placement_option_index), str(transfer_candidate.grasp_id))
                    if transfer_key in transfer_planning_failures:
                        continue
                    attempt_index += 1
                    active_plan = _active_regrasp_plan_for_attempt(
                        plan=plan,
                        placement_option=info.placement_option,
                        transfer_candidate=transfer_candidate,
                        final_candidate=final_candidate,
                    )
                    transfer_initial_world_grasp = saved_grasp_to_world_grasp(
                        transfer_candidate,
                        plan.initial_object_pose_world,
                        pregrasp_offset=pregrasp_offset,
                        gripper_width_clearance=gripper_width_clearance,
                    )
                    transfer_staging_world_grasp = saved_grasp_to_world_grasp(
                        transfer_candidate,
                        info.placement_option.staging_object_pose_world,
                        pregrasp_offset=pregrasp_offset,
                        gripper_width_clearance=gripper_width_clearance,
                    )
                    final_world_grasp = saved_grasp_to_world_grasp(
                        final_candidate,
                        info.placement_option.staging_object_pose_world,
                        pregrasp_offset=pregrasp_offset,
                        gripper_width_clearance=gripper_width_clearance,
                    )
                    print(
                        f"[INFO]: Planning regrasp candidate {attempt_index}/{len(candidate_specs)} "
                        f"placement={info.placement_option_index} "
                        f"xy=({info.staging_xy[0]:.3f}, {info.staging_xy[1]:.3f}) "
                        f"transfer={transfer_candidate.grasp_id} final={final_candidate.grasp_id}",
                        flush=True,
                    )
                    try:
                        moveit_joint_trajectories = _plan_moveit_regrasp_joint_trajectories(
                            args_cli,
                            robot_cfg=robot_cfg,
                            execution_cfg=execution_cfg,
                            transfer_initial_world_grasp=transfer_initial_world_grasp,
                            transfer_staging_world_grasp=transfer_staging_world_grasp,
                            final_world_grasp=final_world_grasp,
                        )
                    except RuntimeError as exc:
                        message = str(exc)
                        attempts.append(
                            {
                                "attempt_index": attempt_index,
                                "placement_option_index": info.placement_option_index,
                                "staging_xy_world": list(info.staging_xy),
                                "transfer_grasp_id": transfer_candidate.grasp_id,
                                "final_grasp_id": final_candidate.grasp_id,
                                "success": False,
                                "status": "moveit_planning_failed",
                                "message": message,
                            }
                        )
                        print(
                            f"[WARN]: Regrasp candidate {attempt_index}/{len(candidate_specs)} planning failed: "
                            f"{message}",
                            flush=True,
                        )
                        if _regrasp_failure_depends_on_transfer(message):
                            transfer_planning_failures.add(transfer_key)
                        continue

                    diagnostics = _moveit_joint_trajectory_diagnostics(moveit_joint_trajectories)
                    summary = _moveit_joint_trajectory_summary(diagnostics)
                    placement_score = float(info.placement_option.metadata.get("placement_score", 0.0))
                    path_cost = _moveit_regrasp_trajectory_cost(diagnostics, placement_score=placement_score)
                    record = {
                        "planning_index": attempt_index,
                        "placement_option_index": info.placement_option_index,
                        "staging_xy_world": list(info.staging_xy),
                        "transfer_grasp_id": transfer_candidate.grasp_id,
                        "final_grasp_id": final_candidate.grasp_id,
                        "path_cost": float(path_cost),
                        "placement_score": placement_score,
                        "trajectory_summary": summary,
                        "trajectory_diagnostics": list(diagnostics),
                    }
                    planned_candidate_records.append(record)
                    planned_moveit_candidates.append(
                        SimpleNamespace(
                            planning_index=attempt_index,
                            placement_option_index=info.placement_option_index,
                            staging_xy=info.staging_xy,
                            placement_option=info.placement_option,
                            active_plan=active_plan,
                            transfer_candidate=transfer_candidate,
                            final_candidate=final_candidate,
                            transfer_initial_world_grasp=transfer_initial_world_grasp,
                            transfer_staging_world_grasp=transfer_staging_world_grasp,
                            final_world_grasp=final_world_grasp,
                            moveit_joint_trajectories=moveit_joint_trajectories,
                            path_cost=path_cost,
                            trajectory_summary=summary,
                            record=record,
                        )
                    )
                    print(
                        f"[INFO]: Planned regrasp candidate {attempt_index}/{len(candidate_specs)} "
                        f"cost={path_cost:.3f} joint_path={float(summary['joint_path_length_rad']):.3f} rad "
                        f"max_step={float(summary['max_joint_step_rad']):.3f} rad "
                        f"points={int(summary['point_count'])}",
                        flush=True,
                    )

                planned_moveit_candidates.sort(key=lambda candidate: float(candidate.path_cost))
                if planned_moveit_candidates:
                    best = planned_moveit_candidates[0]
                    print(
                        f"[INFO]: Selected cheapest MoveIt regrasp candidate "
                        f"planning_index={best.planning_index} placement={best.placement_option_index} "
                        f"xy=({best.staging_xy[0]:.3f}, {best.staging_xy[1]:.3f}) "
                        f"cost={float(best.path_cost):.3f}",
                        flush=True,
                    )

                for execution_rank, candidate in enumerate(planned_moveit_candidates, start=1):
                    candidate.record["execution_rank"] = execution_rank
                    candidate.record["execution_attempted"] = True
                    active_plan = _active_regrasp_plan_for_attempt(
                        plan=plan,
                        placement_option=candidate.placement_option,
                        transfer_candidate=candidate.transfer_candidate,
                        final_candidate=candidate.final_candidate,
                    )
                    transfer_initial_world_grasp = candidate.transfer_initial_world_grasp
                    transfer_staging_world_grasp = candidate.transfer_staging_world_grasp
                    final_world_grasp = candidate.final_world_grasp
                    print(
                        f"[INFO]: Executing ranked MoveIt regrasp candidate "
                        f"{execution_rank}/{len(planned_moveit_candidates)} "
                        f"planning_index={candidate.planning_index} cost={float(candidate.path_cost):.3f}",
                        flush=True,
                    )
                    regrasp_result = run_regrasp_plan_in_mujoco(
                        robot_cfg=robot_cfg,
                        execution_cfg=execution_cfg,
                        object_mesh_path=object_mesh_path,
                        initial_object_pose_world=plan.initial_object_pose_world,
                        transfer_initial_grasp=transfer_initial_world_grasp,
                        transfer_staging_grasp=transfer_staging_world_grasp,
                        final_grasp=final_world_grasp,
                        staging_object_pose_world=candidate.placement_option.staging_object_pose_world,
                        final_grasp_candidate=candidate.final_candidate,
                        pregrasp_offset=pregrasp_offset,
                        gripper_width_clearance=gripper_width_clearance,
                        moveit_joint_trajectories=candidate.moveit_joint_trajectories,
                        keep_generated_scene=args_cli.keep_generated_scene,
                        show_viewer=args_cli.viewer,
                        viewer_left_ui=args_cli.viewer_left_ui,
                        viewer_right_ui=args_cli.viewer_right_ui,
                        viewer_realtime=not args_cli.viewer_no_realtime,
                        viewer_hold_seconds=args_cli.viewer_hold_seconds,
                        viewer_block_at_end=args_cli.viewer_block_at_end,
                    )
                    attempts.append(
                        {
                            "attempt_index": candidate.planning_index,
                            "execution_rank": execution_rank,
                            "placement_option_index": candidate.placement_option_index,
                            "staging_xy_world": list(candidate.staging_xy),
                            "transfer_grasp_id": candidate.transfer_candidate.grasp_id,
                            "final_grasp_id": candidate.final_candidate.grasp_id,
                            "path_cost": float(candidate.path_cost),
                            "trajectory_summary": candidate.trajectory_summary,
                            "result": _object_payload(regrasp_result),
                        }
                    )
                    print(
                        f"[INFO]: Ranked MoveIt regrasp candidate {execution_rank}/{len(planned_moveit_candidates)} "
                        f"finished success={regrasp_result.success} status={regrasp_result.status} "
                        f"message={regrasp_result.message}",
                        flush=True,
                    )
                    if regrasp_result.success:
                        _write_regrasp_attempt_artifact(
                            output_path=args_cli.attempt_artifact,
                            input_json=args_cli.input_json,
                            regrasp_plan_json=args_cli.regrasp_plan_json,
                            plan=active_plan,
                            transfer_initial_world_grasp=transfer_initial_world_grasp,
                            transfer_staging_world_grasp=transfer_staging_world_grasp,
                            final_world_grasp=final_world_grasp,
                            result=regrasp_result,
                            attempts=attempts,
                            planned_candidates=planned_candidate_records,
                        )
                        print(f"[INFO]: Wrote attempt artifact to {args_cli.attempt_artifact}", flush=True)
                        return

                if regrasp_result is None:
                    if (
                        transfer_initial_world_grasp is None
                        or transfer_staging_world_grasp is None
                        or final_world_grasp is None
                    ):
                        transfer_initial_world_grasp = saved_grasp_to_world_grasp(
                            plan.transfer_grasp,
                            plan.initial_object_pose_world,
                            pregrasp_offset=pregrasp_offset,
                            gripper_width_clearance=gripper_width_clearance,
                        )
                        transfer_staging_world_grasp = saved_grasp_to_world_grasp(
                            plan.transfer_grasp,
                            plan.staging_object_pose_world,
                            pregrasp_offset=pregrasp_offset,
                            gripper_width_clearance=gripper_width_clearance,
                        )
                        final_world_grasp = saved_grasp_to_world_grasp(
                            plan.final_grasp,
                            plan.staging_object_pose_world,
                            pregrasp_offset=pregrasp_offset,
                            gripper_width_clearance=gripper_width_clearance,
                        )
                    regrasp_result = MujocoRegraspAttemptResult(
                        success=False,
                        status="moveit_planning_failed",
                        message=(
                            "No MoveIt plan was found for any capped regrasp "
                            "placement/transfer/final candidate combination."
                        ),
                        transfer_pregrasp_reached=False,
                        transfer_grasp_reached=False,
                        transfer_lift_reached=False,
                        placement_reached=False,
                        final_pregrasp_reached=False,
                        final_grasp_reached=False,
                        initial_object_position_world=tuple(
                            float(v) for v in plan.initial_object_pose_world.position_world
                        ),
                        staged_object_position_world=tuple(
                            float(v) for v in active_plan.staging_object_pose_world.position_world
                        ),
                        final_object_position_world=tuple(
                            float(v) for v in plan.initial_object_pose_world.position_world
                        ),
                        final_lift_height_m=0.0,
                        target_lift_height_m=float(execution_cfg.success_height_margin_m),
                    )
                _write_regrasp_attempt_artifact(
                    output_path=args_cli.attempt_artifact,
                    input_json=args_cli.input_json,
                    regrasp_plan_json=args_cli.regrasp_plan_json,
                    plan=active_plan,
                    transfer_initial_world_grasp=transfer_initial_world_grasp,
                    transfer_staging_world_grasp=transfer_staging_world_grasp,
                    final_world_grasp=final_world_grasp,
                    result=regrasp_result,
                    attempts=attempts,
                    planned_candidates=planned_candidate_records,
                )
                print(
                    f"[INFO]: MuJoCo regrasp fallback exhausted {len(attempts)} attempt(s); "
                    f"planned_candidates={len(planned_candidate_records)} "
                    f"last_status={regrasp_result.status} message={regrasp_result.message}",
                    flush=True,
                )
                print(f"[INFO]: Wrote attempt artifact to {args_cli.attempt_artifact}", flush=True)
                raise RuntimeError(f"MuJoCo regrasp fallback failed after {len(attempts)} attempt(s).")
            for placement_option_index, placement_option in enumerate(placement_options, start=1):
                active_plan = _active_regrasp_plan_for_attempt(
                    plan=plan,
                    placement_option=placement_option,
                    transfer_candidate=placement_option.transfer_grasp,
                    final_candidate=placement_option.final_grasp,
                )
                staging_xy = (
                    float(placement_option.staging_object_pose_world.position_world[0]),
                    float(placement_option.staging_object_pose_world.position_world[1]),
                )
                transfer_candidates = _ordered_regrasp_candidates(
                    placement_option.transfer_grasp,
                    placement_option.transfer_grasp_candidates,
                )
                final_candidates = _ordered_regrasp_candidates(
                    placement_option.final_grasp,
                    placement_option.final_grasp_candidates,
                )
                print(
                    f"[INFO]: Trying regrasp placement option {placement_option_index}/{len(placement_options)} "
                    f"xy=({staging_xy[0]:.3f}, {staging_xy[1]:.3f}) "
                    f"score={float(placement_option.metadata.get('placement_score', 0.0)):.3f}",
                    flush=True,
                )
                for transfer_candidate in transfer_candidates:
                    for final_candidate in final_candidates:
                        attempt_index += 1
                        active_plan = _active_regrasp_plan_for_attempt(
                            plan=plan,
                            placement_option=placement_option,
                            transfer_candidate=transfer_candidate,
                            final_candidate=final_candidate,
                        )
                        transfer_initial_world_grasp = saved_grasp_to_world_grasp(
                            transfer_candidate,
                            plan.initial_object_pose_world,
                            pregrasp_offset=pregrasp_offset,
                            gripper_width_clearance=gripper_width_clearance,
                        )
                        transfer_staging_world_grasp = saved_grasp_to_world_grasp(
                            transfer_candidate,
                            placement_option.staging_object_pose_world,
                            pregrasp_offset=pregrasp_offset,
                            gripper_width_clearance=gripper_width_clearance,
                        )
                        final_world_grasp = saved_grasp_to_world_grasp(
                            final_candidate,
                            placement_option.staging_object_pose_world,
                            pregrasp_offset=pregrasp_offset,
                            gripper_width_clearance=gripper_width_clearance,
                        )
                        print(
                            f"[INFO]: MuJoCo regrasp attempt {attempt_index}/{max_attempts} "
                            f"placement={placement_option_index} "
                            f"transfer={transfer_candidate.grasp_id} final={final_candidate.grasp_id}",
                            flush=True,
                        )
                        moveit_joint_trajectories = None
                        if args_cli.controller == "moveit":
                            print("[INFO]: Planning MuJoCo regrasp fallback with MoveIt.", flush=True)
                            try:
                                moveit_joint_trajectories = _plan_moveit_regrasp_joint_trajectories(
                                    args_cli,
                                    robot_cfg=robot_cfg,
                                    execution_cfg=execution_cfg,
                                    transfer_initial_world_grasp=transfer_initial_world_grasp,
                                    transfer_staging_world_grasp=transfer_staging_world_grasp,
                                    final_world_grasp=final_world_grasp,
                                )
                                _print_moveit_joint_trajectory_diagnostics(
                                    moveit_joint_trajectories,
                                    prefix="MoveIt regrasp trajectory",
                                )
                            except RuntimeError as exc:
                                message = str(exc)
                                attempts.append(
                                    {
                                        "attempt_index": attempt_index,
                                        "placement_option_index": placement_option_index,
                                        "staging_xy_world": list(staging_xy),
                                        "transfer_grasp_id": transfer_candidate.grasp_id,
                                        "final_grasp_id": final_candidate.grasp_id,
                                        "success": False,
                                        "status": "moveit_planning_failed",
                                        "message": message,
                                    }
                                )
                                print(
                                    f"[WARN]: Regrasp attempt {attempt_index}/{max_attempts} planning failed: "
                                    f"{message}",
                                    flush=True,
                                )
                                if _regrasp_failure_depends_on_transfer(message):
                                    break
                                continue
                        regrasp_result = run_regrasp_plan_in_mujoco(
                            robot_cfg=robot_cfg,
                            execution_cfg=execution_cfg,
                            object_mesh_path=object_mesh_path,
                            initial_object_pose_world=plan.initial_object_pose_world,
                            transfer_initial_grasp=transfer_initial_world_grasp,
                            transfer_staging_grasp=transfer_staging_world_grasp,
                            final_grasp=final_world_grasp,
                            staging_object_pose_world=placement_option.staging_object_pose_world,
                            final_grasp_candidate=final_candidate,
                            pregrasp_offset=pregrasp_offset,
                            gripper_width_clearance=gripper_width_clearance,
                            moveit_joint_trajectories=moveit_joint_trajectories,
                            keep_generated_scene=args_cli.keep_generated_scene,
                            show_viewer=args_cli.viewer,
                            viewer_left_ui=args_cli.viewer_left_ui,
                            viewer_right_ui=args_cli.viewer_right_ui,
                            viewer_realtime=not args_cli.viewer_no_realtime,
                            viewer_hold_seconds=args_cli.viewer_hold_seconds,
                            viewer_block_at_end=args_cli.viewer_block_at_end,
                        )
                        attempts.append(
                            {
                                "attempt_index": attempt_index,
                                "placement_option_index": placement_option_index,
                                "staging_xy_world": list(staging_xy),
                                "transfer_grasp_id": transfer_candidate.grasp_id,
                                "final_grasp_id": final_candidate.grasp_id,
                                "result": _object_payload(regrasp_result),
                            }
                        )
                        print(
                            f"[INFO]: MuJoCo regrasp attempt {attempt_index}/{max_attempts} finished "
                            f"success={regrasp_result.success} status={regrasp_result.status} "
                            f"message={regrasp_result.message}",
                            flush=True,
                        )
                        if regrasp_result.success:
                            _write_regrasp_attempt_artifact(
                                output_path=args_cli.attempt_artifact,
                                input_json=args_cli.input_json,
                                regrasp_plan_json=args_cli.regrasp_plan_json,
                                plan=active_plan,
                                transfer_initial_world_grasp=transfer_initial_world_grasp,
                                transfer_staging_world_grasp=transfer_staging_world_grasp,
                                final_world_grasp=final_world_grasp,
                                result=regrasp_result,
                                attempts=attempts,
                            )
                            print(f"[INFO]: Wrote attempt artifact to {args_cli.attempt_artifact}", flush=True)
                            return
                        if _regrasp_failure_depends_on_transfer(regrasp_result.status):
                            break
            if regrasp_result is None:
                if (
                    transfer_initial_world_grasp is None
                    or transfer_staging_world_grasp is None
                    or final_world_grasp is None
                ):
                    transfer_initial_world_grasp = saved_grasp_to_world_grasp(
                        plan.transfer_grasp,
                        plan.initial_object_pose_world,
                        pregrasp_offset=pregrasp_offset,
                        gripper_width_clearance=gripper_width_clearance,
                    )
                    transfer_staging_world_grasp = saved_grasp_to_world_grasp(
                        plan.transfer_grasp,
                        plan.staging_object_pose_world,
                        pregrasp_offset=pregrasp_offset,
                        gripper_width_clearance=gripper_width_clearance,
                    )
                    final_world_grasp = saved_grasp_to_world_grasp(
                        plan.final_grasp,
                        plan.staging_object_pose_world,
                        pregrasp_offset=pregrasp_offset,
                        gripper_width_clearance=gripper_width_clearance,
                    )
                regrasp_result = MujocoRegraspAttemptResult(
                    success=False,
                    status="moveit_planning_failed",
                    message="No MoveIt plan was found for any regrasp placement/transfer/final candidate combination.",
                    transfer_pregrasp_reached=False,
                    transfer_grasp_reached=False,
                    transfer_lift_reached=False,
                    placement_reached=False,
                    final_pregrasp_reached=False,
                    final_grasp_reached=False,
                    initial_object_position_world=tuple(float(v) for v in plan.initial_object_pose_world.position_world),
                    staged_object_position_world=tuple(
                        float(v) for v in active_plan.staging_object_pose_world.position_world
                    ),
                    final_object_position_world=tuple(float(v) for v in plan.initial_object_pose_world.position_world),
                    final_lift_height_m=0.0,
                    target_lift_height_m=float(execution_cfg.success_height_margin_m),
                )
            _write_regrasp_attempt_artifact(
                output_path=args_cli.attempt_artifact,
                input_json=args_cli.input_json,
                regrasp_plan_json=args_cli.regrasp_plan_json,
                plan=active_plan,
                transfer_initial_world_grasp=transfer_initial_world_grasp,
                transfer_staging_world_grasp=transfer_staging_world_grasp,
                final_world_grasp=final_world_grasp,
                result=regrasp_result,
                attempts=attempts,
            )
            print(
                f"[INFO]: MuJoCo regrasp fallback exhausted {len(attempts)} attempt(s); "
                f"last_status={regrasp_result.status} message={regrasp_result.message}",
                flush=True,
            )
            print(f"[INFO]: Wrote attempt artifact to {args_cli.attempt_artifact}", flush=True)
            raise RuntimeError(f"MuJoCo regrasp fallback failed after {len(attempts)} attempt(s).")

        placement_spec, object_pose_world = _resolve_object_pose_world_from_bundle(args_cli, bundle, mesh_local)
        statuses = evaluate_saved_grasps_against_pickup_pose(
            bundle.candidates,
            object_pose_world=object_pose_world,
            contact_gap_m=contact_gap_m,
        )
        feasible = accepted_grasps(statuses)
        if not feasible:
            raise RuntimeError("No ground-feasible grasps remain for the requested pickup pose.")

        feasible = score_grasps(feasible, mesh_local=mesh_local)
        selected_grasps = _ordered_feasible_grasps(feasible=feasible, requested_grasp_id=args_cli.grasp_id)
        if not selected_grasps:
            raise RuntimeError("No feasible grasp could be selected after scoring.")

        attempts: list[dict[str, object]] = []
        selected_world_grasp = None
        result = None
        for attempt_index, selected_grasp in enumerate(selected_grasps, start=1):
            selected_world_grasp = saved_grasp_to_world_grasp(
                selected_grasp,
                object_pose_world,
                pregrasp_offset=pregrasp_offset,
                gripper_width_clearance=gripper_width_clearance,
            )
            print(
                f"[INFO]: MuJoCo attempt {attempt_index}/{len(selected_grasps)} grasp_id={selected_grasp.grasp_id}",
                flush=True,
            )
            moveit_joint_trajectories = None
            if args_cli.controller == "moveit":
                print("[INFO]: Planning MuJoCo attempt with MoveIt.", flush=True)
                moveit_joint_trajectories = _plan_moveit_joint_trajectories(
                    args_cli,
                    robot_cfg=robot_cfg,
                    execution_cfg=execution_cfg,
                    world_grasp=selected_world_grasp,
                )
                _print_moveit_joint_trajectory_diagnostics(
                    moveit_joint_trajectories,
                    prefix="MoveIt pickup trajectory",
                )
            result = run_world_grasp_in_mujoco(
                robot_cfg=robot_cfg,
                execution_cfg=execution_cfg,
                object_mesh_path=object_mesh_path,
                object_pose_world=object_pose_world,
                world_grasp=selected_world_grasp,
                moveit_joint_trajectories=moveit_joint_trajectories,
                keep_generated_scene=args_cli.keep_generated_scene,
                show_viewer=args_cli.viewer,
                viewer_left_ui=args_cli.viewer_left_ui,
                viewer_right_ui=args_cli.viewer_right_ui,
                viewer_realtime=not args_cli.viewer_no_realtime,
                viewer_hold_seconds=args_cli.viewer_hold_seconds,
                viewer_block_at_end=args_cli.viewer_block_at_end,
            )
            attempt_payload = _attempt_result_payload(selected_grasp=selected_world_grasp, result=result)
            attempt_payload["attempt_index"] = attempt_index
            attempts.append(attempt_payload)
            print(
                f"[INFO]: MuJoCo attempt {attempt_index}/{len(selected_grasps)} finished "
                f"grasp_id={selected_grasp.grasp_id} success={result.success} status={result.status} "
                f"lift_height_m={result.lift_height_m:.4f} message={result.message}",
                flush=True,
            )
            if result.success:
                break
    finally:
        if not args_cli.keep_generated_scene:
            try:
                object_mesh_path.unlink()
            except FileNotFoundError:
                pass
    if selected_world_grasp is None or result is None:
        raise RuntimeError("No MuJoCo grasp attempts were executed.")
    _write_attempt_artifact(
        output_path=args_cli.attempt_artifact,
        placement_spec=placement_spec,
        object_pose_world=object_pose_world,
        selected_grasp=selected_world_grasp,
        result=result,
        attempts=attempts,
    )
    print(
        f"[INFO]: MuJoCo grasp attempt finished success={result.success} status={result.status} "
        f"grasp_id={selected_world_grasp.grasp_id} attempts={len(attempts)}/{len(selected_grasps)} "
        f"lift_height_m={result.lift_height_m:.4f} message={result.message}",
        flush=True,
    )
    print(f"[INFO]: Wrote attempt artifact to {args_cli.attempt_artifact}", flush=True)
    if not result.success:
        raise RuntimeError(f"All {len(attempts)} MuJoCo grasp attempt(s) failed; last status={result.status}.")


if __name__ == "__main__":
    main()
