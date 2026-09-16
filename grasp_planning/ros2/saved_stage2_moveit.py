"""MoveIt preplanning for unified Isaac execution of a saved stage-2 bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle
from grasp_planning.grasping.grasp_transforms import saved_grasp_to_world_grasp
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.ros2.moveit_pose_commander import (
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    rclpy,
)
from grasp_planning.ros2.moveit_world_grasp import world_grasp_pose_targets
from grasp_planning.start_poses import DEFAULT_ARM_START_JOINT_VALUES, DEFAULT_MOVEIT_ARM_JOINT_NAMES


@dataclass(frozen=True)
class SavedStage2MoveItConfig:
    frame_id: str = "base"
    target_position_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tcp_to_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    planning_group: str = "fr3_arm"
    pose_link: str = "fr3_hand_tcp"
    namespace: str = ""
    joint_names: tuple[str, ...] = DEFAULT_MOVEIT_ARM_JOINT_NAMES
    start_joint_positions: tuple[float, ...] = DEFAULT_ARM_START_JOINT_VALUES
    pipeline_id: str = ""
    planner_id: str = ""
    wait_for_moveit_timeout_s: float = 15.0
    ik_timeout_s: float = 2.0
    planning_time_s: float = 5.0
    num_planning_attempts: int = 5
    velocity_scale: float = 0.05
    acceleration_scale: float = 0.05
    lift_height_m: float = 0.08
    pregrasp_offset_m: float = 0.20
    gripper_width_clearance_m: float = 0.01
    pregrasp_only: bool = False
    allow_collisions: bool = False


def _csv_floats(value: str, *, length: int, label: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if len(parsed) != length:
        raise ValueError(f"{label} must contain exactly {length} comma-separated values.")
    return parsed


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def config_from_backend_command(command: Sequence[str]) -> SavedStage2MoveItConfig:
    """Read only MoveIt-related options from the resolved Isaac backend command."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--moveit-frame-id", default="base")
    parser.add_argument("--moveit-target-position-signs", default="1,1,1")
    parser.add_argument("--tcp-to-grasp-offset", nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--moveit-planning-group", default="fr3_arm")
    parser.add_argument("--moveit-pose-link", default="fr3_hand_tcp")
    parser.add_argument("--moveit-namespace", default="")
    parser.add_argument("--moveit-joint-names", default=",".join(DEFAULT_MOVEIT_ARM_JOINT_NAMES))
    parser.add_argument(
        "--moveit-start-joint-positions",
        default=",".join(str(value) for value in DEFAULT_ARM_START_JOINT_VALUES),
    )
    parser.add_argument("--moveit-pipeline-id", default="")
    parser.add_argument("--moveit-planner-id", default="")
    parser.add_argument("--moveit-wait-for-moveit-timeout-s", type=float, default=15.0)
    parser.add_argument("--moveit-ik-timeout-s", type=float, default=2.0)
    parser.add_argument("--moveit-planning-time-s", type=float, default=5.0)
    parser.add_argument("--moveit-num-planning-attempts", type=int, default=5)
    parser.add_argument("--moveit-velocity-scale", type=float, default=0.05)
    parser.add_argument("--moveit-acceleration-scale", type=float, default=0.05)
    parser.add_argument("--moveit-lift-height-m", type=float, default=0.08)
    parser.add_argument("--pregrasp-offset", type=float, default=0.20)
    parser.add_argument("--gripper-width-clearance", type=float, default=0.01)
    parser.add_argument("--pregrasp-only", action="store_true")
    parser.add_argument("--moveit-allow-collisions", action="store_true")
    args, _unknown = parser.parse_known_args(list(command))

    joint_names = _csv_strings(args.moveit_joint_names)
    start_positions = _csv_floats(
        args.moveit_start_joint_positions,
        length=len(joint_names),
        label="--moveit-start-joint-positions",
    )
    signs = _csv_floats(
        args.moveit_target_position_signs,
        length=3,
        label="--moveit-target-position-signs",
    )
    return SavedStage2MoveItConfig(
        frame_id=str(args.moveit_frame_id),
        target_position_signs=(signs[0], signs[1], signs[2]),
        tcp_to_grasp_offset=tuple(float(value) for value in args.tcp_to_grasp_offset),
        planning_group=str(args.moveit_planning_group),
        pose_link=str(args.moveit_pose_link),
        namespace=str(args.moveit_namespace),
        joint_names=joint_names,
        start_joint_positions=start_positions,
        pipeline_id=str(args.moveit_pipeline_id),
        planner_id=str(args.moveit_planner_id),
        wait_for_moveit_timeout_s=float(args.moveit_wait_for_moveit_timeout_s),
        ik_timeout_s=float(args.moveit_ik_timeout_s),
        planning_time_s=float(args.moveit_planning_time_s),
        num_planning_attempts=int(args.moveit_num_planning_attempts),
        velocity_scale=float(args.moveit_velocity_scale),
        acceleration_scale=float(args.moveit_acceleration_scale),
        lift_height_m=float(args.moveit_lift_height_m),
        pregrasp_offset_m=float(args.pregrasp_offset),
        gripper_width_clearance_m=float(args.gripper_width_clearance),
        pregrasp_only=bool(args.pregrasp_only),
        allow_collisions=bool(args.moveit_allow_collisions),
    )


def _object_pose(bundle, *, bundle_path: Path) -> ObjectWorldPose:
    raw_pose = bundle.metadata.get("execution_world_pose")
    if not isinstance(raw_pose, dict):
        raise RuntimeError(f"Stage-2 bundle '{bundle_path}' is missing metadata.execution_world_pose.")
    position = raw_pose.get("position_world")
    orientation = raw_pose.get("orientation_xyzw_world")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise RuntimeError(f"Stage-2 bundle '{bundle_path}' has an invalid execution world position.")
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
        raise RuntimeError(f"Stage-2 bundle '{bundle_path}' has an invalid execution world orientation.")
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in position),
        orientation_xyzw_world=tuple(float(value) for value in orientation),
    )


def _waypoints(trajectory, *, joint_names: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    joint_trajectory = trajectory.joint_trajectory
    source_names = tuple(str(name) for name in joint_trajectory.joint_names)
    source_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in joint_names if name not in source_index]
    if missing:
        raise RuntimeError(f"MoveIt trajectory is missing arm joints: {missing}.")
    result = tuple(
        tuple(float(point.positions[source_index[name]]) for name in joint_names)
        for point in joint_trajectory.points
    )
    if not result:
        raise RuntimeError("MoveIt returned a trajectory with no points.")
    return result


def preplan_saved_stage2_for_isaac(
    *,
    stage2_bundle: Path,
    grasp_id: str,
    output_path: Path,
    config: SavedStage2MoveItConfig,
) -> Path:
    """Plan collision-aware pregrasp/grasp/lift waypoints before Isaac starts."""

    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable. Source setup_robot_env.sh first.")
    bundle = load_grasp_bundle(stage2_bundle)
    selected = next((candidate for candidate in bundle.candidates if candidate.grasp_id == grasp_id), None)
    if selected is None:
        raise RuntimeError(f"Requested Isaac grasp id '{grasp_id}' is not present in {stage2_bundle}.")
    world_grasp = saved_grasp_to_world_grasp(
        selected,
        _object_pose(bundle, bundle_path=stage2_bundle),
        pregrasp_offset=float(config.pregrasp_offset_m),
        gripper_width_clearance=float(config.gripper_width_clearance_m),
    )
    if world_grasp.pregrasp_position_w[2] <= 0.05:
        raise RuntimeError(
            f"Requested Isaac grasp id '{grasp_id}' has unsafe pregrasp height "
            f"{world_grasp.pregrasp_position_w[2]:.4f} m."
        )
    targets = world_grasp_pose_targets(
        world_grasp,
        frame_id=config.frame_id,
        lift_height_m=float(config.lift_height_m),
        position_signs=config.target_position_signs,
        tcp_to_grasp_offset=config.tcp_to_grasp_offset,
    )
    labels = ("pregrasp",) if config.pregrasp_only else ("pregrasp", "grasp", "lift")
    moveit_config = MoveItPoseCommanderConfig(
        planning_group=config.planning_group,
        pose_link=config.pose_link,
        joint_names=config.joint_names,
        moveit_namespace=config.namespace,
        pipeline_id=config.pipeline_id,
        planner_id=config.planner_id,
        wait_for_moveit_timeout_s=config.wait_for_moveit_timeout_s,
        ik_timeout_s=config.ik_timeout_s,
        fk_timeout_s=config.ik_timeout_s,
        planning_time_s=config.planning_time_s,
        num_planning_attempts=config.num_planning_attempts,
        velocity_scale=config.velocity_scale,
        acceleration_scale=config.acceleration_scale,
        post_execute_sleep_s=0.0,
        avoid_collisions=not config.allow_collisions,
    )
    initialized_here = False
    commander = None
    try:
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        commander = MoveItPoseCommander(moveit_config, node_name="unified_saved_stage2_moveit")
        commander.wait_for_moveit(require_execute=False)
        current_start = config.start_joint_positions
        planned: dict[str, tuple[tuple[float, ...], ...]] = {}
        for label in labels:
            print(f"[PIPELINE] MoveIt preplanning saved-stage2 Isaac {label}.", flush=True)
            trajectory, message = commander.plan_to_pose(
                targets[label],
                label=f"saved_stage2_isaac_{label}",
                start_joint_positions=current_start,
            )
            if trajectory is None:
                raise RuntimeError(f"MoveIt failed to preplan Isaac {label}: {message}")
            planned[label] = _waypoints(trajectory, joint_names=config.joint_names)
            current_start = planned[label][-1]
    finally:
        if commander is not None:
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "selected_grasp_id": selected.grasp_id,
                "joint_names": list(config.joint_names),
                "start_joint_positions": list(config.start_joint_positions),
                "trajectories": {
                    label: [list(waypoint) for waypoint in waypoints]
                    for label, waypoints in planned.items()
                },
                "moveit": {
                    "frame_id": config.frame_id,
                    "target_position_signs": list(config.target_position_signs),
                    "tcp_to_grasp_offset": list(config.tcp_to_grasp_offset),
                    "planning_group": config.planning_group,
                    "pose_link": config.pose_link,
                    "namespace": config.namespace,
                    "pipeline_id": config.pipeline_id,
                    "planner_id": config.planner_id,
                    "lift_height_m": config.lift_height_m,
                    "allow_collisions": config.allow_collisions,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PIPELINE] Wrote saved-stage2 Isaac MoveIt plan to {output_path}.", flush=True)
    return output_path


__all__ = [
    "SavedStage2MoveItConfig",
    "config_from_backend_command",
    "preplan_saved_stage2_for_isaac",
]
