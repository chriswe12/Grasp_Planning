"""Execute a saved stage-2 grasp bundle on the real robot through MoveIt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from grasp_planning import load_grasp_bundle, saved_grasp_to_world_grasp
from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.ros2.franka_gripper_client import FrankaGripperClient
from grasp_planning.ros2.moveit_pose_commander import MoveItPoseCommander, MoveItPoseCommanderConfig, PoseTarget, rclpy


@dataclass(frozen=True)
class RealExecutionResult:
    success: bool
    status: str
    message: str
    grasp_id: str
    pregrasp_reached: bool
    grasp_reached: bool
    lift_reached: bool
    attempt_artifact_path: Path


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


def _select_bundle_grasp(bundle, *, grasp_id: str):
    if not bundle.candidates:
        raise RuntimeError("The stage-2 bundle contains no feasible grasps to execute.")
    if grasp_id:
        selected = next((candidate for candidate in bundle.candidates if candidate.grasp_id == grasp_id), None)
        if selected is None:
            raise RuntimeError(f"Requested grasp id '{grasp_id}' is not present in the stage-2 bundle.")
        return selected
    return bundle.candidates[0]


def _pose_target_from_world(
    *,
    position_xyz: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    frame_id: str,
) -> PoseTarget:
    return PoseTarget.from_quaternion(
        x=position_xyz[0],
        y=position_xyz[1],
        z=position_xyz[2],
        quaternion_xyzw=orientation_xyzw,
        frame_id=frame_id,
    )


def _target_dict(
    world_grasp: WorldFrameGraspCandidate, *, frame_id: str, lift_height_m: float
) -> dict[str, PoseTarget]:
    orientation_xyzw = tuple(float(v) for v in world_grasp.orientation_xyzw)
    return {
        "pregrasp": _pose_target_from_world(
            position_xyz=tuple(float(v) for v in world_grasp.pregrasp_position_w),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
        "grasp": _pose_target_from_world(
            position_xyz=tuple(float(v) for v in world_grasp.position_w),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
        "lift": _pose_target_from_world(
            position_xyz=(
                float(world_grasp.position_w[0]),
                float(world_grasp.position_w[1]),
                float(world_grasp.position_w[2] + float(lift_height_m)),
            ),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
    }


def _confirmation_text(*, input_json: Path, config, world_grasp: WorldFrameGraspCandidate) -> str:
    return (
        "Real execution requested.\n"
        f"  stage2_bundle: {input_json}\n"
        f"  grasp_id:      {world_grasp.grasp_id}\n"
        f"  frame_id:      {config.frame_id}\n"
        f"  stop_after:    {config.stop_after}\n"
        f"  pregrasp_xyz:  {tuple(round(v, 4) for v in world_grasp.pregrasp_position_w)}\n"
        f"  grasp_xyz:     {tuple(round(v, 4) for v in world_grasp.position_w)}\n"
        "Type 'yes' to continue: "
    )


def _confirm_or_abort(*, input_json: Path, config, world_grasp: WorldFrameGraspCandidate) -> bool:
    if not bool(config.require_confirmation):
        return True
    reply = input(_confirmation_text(input_json=input_json, config=config, world_grasp=world_grasp))
    return reply.strip().lower() in {"y", "yes"}


def _write_attempt_artifact(
    *,
    output_path: Path,
    input_json: Path,
    object_pose_world: ObjectWorldPose,
    world_grasp: WorldFrameGraspCandidate,
    config,
    result: RealExecutionResult,
    steps: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_stage2_json": str(input_json),
        "config": {
            "frame_id": config.frame_id,
            "stop_after": config.stop_after,
            "pregrasp_offset_m": float(config.pregrasp_offset_m),
            "gripper_width_clearance_m": float(config.gripper_width_clearance_m),
            "lift_height_m": float(config.lift_height_m),
            "gripper_enabled": bool(config.gripper_enabled),
        },
        "object_pose_world": {
            "position_world": list(object_pose_world.position_world),
            "orientation_xyzw_world": list(object_pose_world.orientation_xyzw_world),
        },
        "selected_grasp": {
            "grasp_id": world_grasp.grasp_id,
            "position_w": list(world_grasp.position_w),
            "orientation_xyzw": list(world_grasp.orientation_xyzw),
            "pregrasp_position_w": list(world_grasp.pregrasp_position_w),
            "gripper_width": float(world_grasp.gripper_width),
            "jaw_width": float(world_grasp.jaw_width),
        },
        "steps": steps,
        "result": {
            "success": bool(result.success),
            "status": result.status,
            "message": result.message,
            "pregrasp_reached": bool(result.pregrasp_reached),
            "grasp_reached": bool(result.grasp_reached),
            "lift_reached": bool(result.lift_reached),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stop_after_success_result(
    *,
    config,
    grasp_id: str,
    attempt_artifact_path: Path,
    pregrasp_reached: bool,
    grasp_reached: bool,
    lift_reached: bool,
) -> RealExecutionResult:
    if config.stop_after == "pregrasp":
        status = "stopped_at_pregrasp"
        message = "Reached pregrasp and stopped by configuration."
    elif config.stop_after == "grasp":
        status = "stopped_at_grasp"
        message = "Reached grasp pose and stopped by configuration."
    elif config.stop_after == "lift":
        status = "stopped_at_lift"
        message = "Reached lift pose and stopped by configuration."
    else:
        status = "completed"
        message = "Completed the configured real-execution sequence."
    return RealExecutionResult(
        success=True,
        status=status,
        message=message,
        grasp_id=grasp_id,
        pregrasp_reached=pregrasp_reached,
        grasp_reached=grasp_reached,
        lift_reached=lift_reached,
        attempt_artifact_path=attempt_artifact_path,
    )


def _execute_selected_world_grasp(
    *,
    commander,
    gripper,
    world_grasp: WorldFrameGraspCandidate,
    config,
    attempt_artifact_path: Path,
) -> tuple[RealExecutionResult, list[dict[str, object]]]:
    targets = _target_dict(world_grasp, frame_id=config.frame_id, lift_height_m=config.lift_height_m)
    pregrasp_reached = False
    grasp_reached = False
    lift_reached = False
    steps: list[dict[str, object]] = []

    def _record_step(name: str, *, ok: bool, message: str, target: PoseTarget | None = None) -> None:
        payload: dict[str, object] = {"name": name, "ok": bool(ok), "message": message}
        if target is not None:
            payload["target_pose"] = {
                "frame_id": target.frame_id,
                "position_xyz": list(target.position_xyz),
                "orientation_xyzw": list(target.orientation_xyzw),
            }
        steps.append(payload)

    if gripper is not None:
        ok, message = gripper.open(width=config.gripper_open_width)
        _record_step("open_gripper", ok=ok, message=message)
        if not ok:
            return (
                RealExecutionResult(
                    success=False,
                    status="open_gripper_failed",
                    message=message,
                    grasp_id=world_grasp.grasp_id,
                    pregrasp_reached=pregrasp_reached,
                    grasp_reached=grasp_reached,
                    lift_reached=lift_reached,
                    attempt_artifact_path=attempt_artifact_path,
                ),
                steps,
            )

    ok, message = commander.move_to_pose(targets["pregrasp"], label="pregrasp", execute=True)
    _record_step("pregrasp", ok=ok, message=message, target=targets["pregrasp"])
    if not ok:
        return (
            RealExecutionResult(
                success=False,
                status="pregrasp_failed",
                message=message,
                grasp_id=world_grasp.grasp_id,
                pregrasp_reached=False,
                grasp_reached=False,
                lift_reached=False,
                attempt_artifact_path=attempt_artifact_path,
            ),
            steps,
        )
    pregrasp_reached = True
    if config.stop_after == "pregrasp":
        return (
            _stop_after_success_result(
                config=config,
                grasp_id=world_grasp.grasp_id,
                attempt_artifact_path=attempt_artifact_path,
                pregrasp_reached=pregrasp_reached,
                grasp_reached=grasp_reached,
                lift_reached=lift_reached,
            ),
            steps,
        )

    ok, message = commander.move_to_pose(targets["grasp"], label="grasp", execute=True)
    _record_step("grasp", ok=ok, message=message, target=targets["grasp"])
    if not ok:
        return (
            RealExecutionResult(
                success=False,
                status="grasp_failed",
                message=message,
                grasp_id=world_grasp.grasp_id,
                pregrasp_reached=pregrasp_reached,
                grasp_reached=False,
                lift_reached=False,
                attempt_artifact_path=attempt_artifact_path,
            ),
            steps,
        )
    grasp_reached = True
    if config.stop_after == "grasp":
        return (
            _stop_after_success_result(
                config=config,
                grasp_id=world_grasp.grasp_id,
                attempt_artifact_path=attempt_artifact_path,
                pregrasp_reached=pregrasp_reached,
                grasp_reached=grasp_reached,
                lift_reached=lift_reached,
            ),
            steps,
        )

    if gripper is not None:
        ok, message = gripper.close(width=world_grasp.jaw_width)
        _record_step("close_gripper", ok=ok, message=message)
        if not ok:
            return (
                RealExecutionResult(
                    success=False,
                    status="close_gripper_failed",
                    message=message,
                    grasp_id=world_grasp.grasp_id,
                    pregrasp_reached=pregrasp_reached,
                    grasp_reached=grasp_reached,
                    lift_reached=lift_reached,
                    attempt_artifact_path=attempt_artifact_path,
                ),
                steps,
            )
    else:
        _record_step("close_gripper", ok=True, message="Skipped because gripper_enabled=false.")

    ok, message = commander.move_to_pose(targets["lift"], label="lift", execute=True)
    _record_step("lift", ok=ok, message=message, target=targets["lift"])
    if not ok:
        return (
            RealExecutionResult(
                success=False,
                status="lift_failed",
                message=message,
                grasp_id=world_grasp.grasp_id,
                pregrasp_reached=pregrasp_reached,
                grasp_reached=grasp_reached,
                lift_reached=False,
                attempt_artifact_path=attempt_artifact_path,
            ),
            steps,
        )
    lift_reached = True
    return (
        _stop_after_success_result(
            config=config,
            grasp_id=world_grasp.grasp_id,
            attempt_artifact_path=attempt_artifact_path,
            pregrasp_reached=pregrasp_reached,
            grasp_reached=grasp_reached,
            lift_reached=lift_reached,
        ),
        steps,
    )


def execute_real_grasp_from_bundle(*, input_json: Path, config) -> RealExecutionResult:
    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace first.")
    bundle = load_grasp_bundle(input_json)
    object_pose_world = _bundle_execution_pose_world(bundle)
    if object_pose_world is None:
        raise RuntimeError("The stage-2 bundle does not contain execution_world_pose metadata.")

    selected_grasp = _select_bundle_grasp(bundle, grasp_id=str(config.grasp_id))
    world_grasp = saved_grasp_to_world_grasp(
        selected_grasp,
        object_pose_world,
        pregrasp_offset=float(config.pregrasp_offset_m),
        gripper_width_clearance=float(config.gripper_width_clearance_m),
    )

    attempt_artifact_path = Path(str(config.attempt_artifact))
    if not _confirm_or_abort(input_json=input_json, config=config, world_grasp=world_grasp):
        result = RealExecutionResult(
            success=False,
            status="aborted",
            message="Execution aborted by user confirmation prompt.",
            grasp_id=world_grasp.grasp_id,
            pregrasp_reached=False,
            grasp_reached=False,
            lift_reached=False,
            attempt_artifact_path=attempt_artifact_path,
        )
        _write_attempt_artifact(
            output_path=attempt_artifact_path,
            input_json=input_json,
            object_pose_world=object_pose_world,
            world_grasp=world_grasp,
            config=config,
            result=result,
            steps=[],
        )
        return result

    moveit_config = MoveItPoseCommanderConfig(
        planning_group=str(config.planning_group),
        pose_link=str(config.pose_link),
        wait_for_moveit_timeout_s=float(config.wait_for_moveit_timeout_s),
        ik_timeout_s=float(config.ik_timeout_s),
        fk_timeout_s=float(config.ik_timeout_s),
        planning_time_s=float(config.planning_time_s),
        num_planning_attempts=int(config.num_planning_attempts),
        velocity_scale=float(config.velocity_scale),
        acceleration_scale=float(config.acceleration_scale),
        execute_timeout_s=float(config.execute_timeout_s),
        post_execute_sleep_s=float(config.post_execute_sleep_s),
        avoid_collisions=not bool(config.allow_collisions),
    )

    commander = None
    gripper = None
    initialized_here = False
    try:
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True

        commander = MoveItPoseCommander(moveit_config)
        commander.wait_for_moveit()

        if bool(config.gripper_enabled):
            gripper = FrankaGripperClient(
                commander,
                grasp_action_name=str(config.gripper_grasp_action),
                move_action_name=str(config.gripper_move_action),
                timeout_s=float(config.gripper_timeout_s),
                grasp_speed=float(config.gripper_grasp_speed),
                grasp_force=float(config.gripper_grasp_force),
                epsilon_inner=float(config.gripper_epsilon_inner),
                epsilon_outer=float(config.gripper_epsilon_outer),
                grasp_settle_time_s=float(config.grasp_settle_time_s),
            )
            gripper.wait_for_server(timeout_s=float(config.wait_for_moveit_timeout_s))

        result, steps = _execute_selected_world_grasp(
            commander=commander,
            gripper=gripper,
            world_grasp=world_grasp,
            config=config,
            attempt_artifact_path=attempt_artifact_path,
        )
        _write_attempt_artifact(
            output_path=attempt_artifact_path,
            input_json=input_json,
            object_pose_world=object_pose_world,
            world_grasp=world_grasp,
            config=config,
            result=result,
            steps=steps,
        )
        return result
    finally:
        if commander is not None:
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()
