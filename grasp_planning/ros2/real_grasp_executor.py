"""Execute a saved stage-2 grasp bundle on the real robot through MoveIt."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

from grasp_planning import load_grasp_bundle, saved_grasp_to_world_grasp
from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.ros2.franka_gripper_client import FrankaGripperClient
from grasp_planning.ros2.gripper_command_client import GripperCommandClient
from grasp_planning.ros2.moveit_pose_commander import MoveItPoseCommander, MoveItPoseCommanderConfig, PoseTarget, rclpy
from grasp_planning.ros2.moveit_world_grasp import world_grasp_pose_targets
from grasp_planning.ros2.normalized_position_gripper_client import (
    NormalizedPositionGripperClient,
)
from grasp_planning.ros2.trigger_service_gripper_client import TriggerServiceGripperClient
from grasp_planning.start_poses import kuka_moveit_gripper_driver_position_from_width

GRIPPER_CLIENT_FRANKA = "franka"
GRIPPER_CLIENT_GRIPPER_COMMAND = "gripper_command"
GRIPPER_CLIENT_TRIGGER_SERVICE = "trigger_service"
GRIPPER_CLIENT_NORMALIZED_POSITION = "normalized_position"


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


def _confirmation_text(*, input_json: Path, config, world_grasp: WorldFrameGraspCandidate) -> str:
    return (
        "Real execution requested.\n"
        f"  stage2_bundle: {input_json}\n"
        f"  grasp_id:      {world_grasp.grasp_id}\n"
        f"  frame_id:      {config.frame_id}\n"
        f"  stop_after:    {config.stop_after}\n"
        f"  approach:      {config.grasp_approach_controller}\n"
        f"  policy_config: {config.visual_servo_config or 'disabled'}\n"
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
            "grasp_approach_controller": str(config.grasp_approach_controller),
            "visual_servo_config": str(config.visual_servo_config),
            "pregrasp_offset_m": float(config.pregrasp_offset_m),
            "gripper_width_clearance_m": float(config.gripper_width_clearance_m),
            "lift_height_m": float(config.lift_height_m),
            "gripper_enabled": bool(config.gripper_enabled),
            "gripper_client": str(config.gripper_client),
            "gripper_command_action": str(config.gripper_command_action),
            "gripper_command_position_mode": str(config.gripper_command_position_mode),
            "gripper_trigger_open_service": str(config.gripper_trigger_open_service),
            "gripper_trigger_close_service": str(config.gripper_trigger_close_service),
            "gripper_trigger_stop_service": str(config.gripper_trigger_stop_service),
            "gripper_position_command_topic": str(config.gripper_position_command_topic),
            "gripper_position_feedback_topic": str(config.gripper_position_feedback_topic),
            "gripper_position_feedback_tolerance": float(
                config.gripper_position_feedback_tolerance
            ),
            "moveit_gripper_joint_name": str(config.moveit_gripper_joint_name),
            "gripper_closed_width": float(config.gripper_closed_width),
            "gripper_open_width": float(config.gripper_open_width),
            "planning_scene_obstacles": list(config.planning_scene_obstacles),
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
        message = "Completed grasp phase and stopped before lift by configuration."
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


def _normalize_gripper_client(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "": GRIPPER_CLIENT_FRANKA,
        "franka": GRIPPER_CLIENT_FRANKA,
        "franka_hand": GRIPPER_CLIENT_FRANKA,
        "fr3": GRIPPER_CLIENT_FRANKA,
        "gripper_command": GRIPPER_CLIENT_GRIPPER_COMMAND,
        "control_msgs": GRIPPER_CLIENT_GRIPPER_COMMAND,
        "control_msgs_gripper_command": GRIPPER_CLIENT_GRIPPER_COMMAND,
        "generic": GRIPPER_CLIENT_GRIPPER_COMMAND,
        "generic_gripper_command": GRIPPER_CLIENT_GRIPPER_COMMAND,
        "trigger": GRIPPER_CLIENT_TRIGGER_SERVICE,
        "trigger_service": GRIPPER_CLIENT_TRIGGER_SERVICE,
        "std_srvs": GRIPPER_CLIENT_TRIGGER_SERVICE,
        "normalized": GRIPPER_CLIENT_NORMALIZED_POSITION,
        "normalized_position": GRIPPER_CLIENT_NORMALIZED_POSITION,
        "closure_fraction": GRIPPER_CLIENT_NORMALIZED_POSITION,
        "servo_gripper": GRIPPER_CLIENT_NORMALIZED_POSITION,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported real_execution.gripper_client '{name}'. "
            "Expected one of: franka, gripper_command, trigger_service, normalized_position."
        ) from exc


def _make_gripper_client(*, commander, config):
    client_name = _normalize_gripper_client(str(config.gripper_client))
    if client_name == GRIPPER_CLIENT_FRANKA:
        return FrankaGripperClient(
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
    if client_name == GRIPPER_CLIENT_GRIPPER_COMMAND:
        return GripperCommandClient(
            commander,
            action_name=str(config.gripper_command_action),
            timeout_s=float(config.gripper_timeout_s),
            max_effort=float(config.gripper_command_max_effort),
            position_mode=str(config.gripper_command_position_mode),
            grasp_settle_time_s=float(config.grasp_settle_time_s),
        )
    if client_name == GRIPPER_CLIENT_TRIGGER_SERVICE:
        return TriggerServiceGripperClient(
            commander,
            open_service_name=str(config.gripper_trigger_open_service),
            close_service_name=str(config.gripper_trigger_close_service),
            stop_service_name=str(config.gripper_trigger_stop_service),
            timeout_s=float(config.gripper_timeout_s),
            grasp_settle_time_s=float(config.grasp_settle_time_s),
        )
    if client_name == GRIPPER_CLIENT_NORMALIZED_POSITION:
        return NormalizedPositionGripperClient(
            commander,
            position_command_topic=str(config.gripper_position_command_topic),
            position_feedback_topic=str(config.gripper_position_feedback_topic),
            open_service_name=str(config.gripper_trigger_open_service),
            close_service_name=str(config.gripper_trigger_close_service),
            stop_service_name=str(config.gripper_trigger_stop_service),
            timeout_s=float(config.gripper_timeout_s),
            feedback_tolerance=float(config.gripper_position_feedback_tolerance),
            grasp_settle_time_s=float(config.grasp_settle_time_s),
            closed_width_m=float(config.gripper_closed_width),
            open_width_m=float(config.gripper_open_width),
        )
    raise AssertionError(f"Unhandled gripper client '{client_name}'.")


def _best_effort_stop_gripper(gripper, *, reason: str) -> None:
    stop = getattr(gripper, "stop", None)
    if not callable(stop):
        return
    try:
        ok, message = stop()
    except Exception as exc:
        print(f"[WARN]: Gripper emergency stop after {reason} failed: {exc!r}", flush=True)
        return
    print(
        f"[INFO]: Gripper emergency stop after {reason}: success={bool(ok)} message={message}",
        flush=True,
    )


def _execute_selected_world_grasp(
    *,
    commander,
    gripper,
    world_grasp: WorldFrameGraspCandidate,
    config,
    attempt_artifact_path: Path,
    visual_servo_runner=None,
    pregrasp_trajectory=None,
) -> tuple[RealExecutionResult, list[dict[str, object]]]:
    targets = world_grasp_pose_targets(world_grasp, frame_id=config.frame_id, lift_height_m=config.lift_height_m)
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
        ok, message = gripper.open(width=world_grasp.gripper_width)
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

    if pregrasp_trajectory is None:
        ok, message = commander.move_to_pose(targets["pregrasp"], label="pregrasp", execute=True)
    else:
        ok, message = commander.execute_trajectory(pregrasp_trajectory, label="pregrasp")
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

    policy_result = None
    if str(config.grasp_approach_controller) == "d405_policy":
        if visual_servo_runner is None:
            message = "D405 policy approach was selected without a visual-servo runner."
            _record_step("d405_policy_approach", ok=False, message=message)
            return (
                RealExecutionResult(
                    success=False,
                    status="visual_servo_unavailable",
                    message=message,
                    grasp_id=world_grasp.grasp_id,
                    pregrasp_reached=pregrasp_reached,
                    grasp_reached=False,
                    lift_reached=False,
                    attempt_artifact_path=attempt_artifact_path,
                ),
                steps,
            )
        policy_result = visual_servo_runner()
        _record_step(
            "d405_policy_approach",
            ok=bool(policy_result.completed),
            message=str(policy_result.message),
        )
        steps[-1].update(
            {
                "state": str(policy_result.state),
                "goal_id": str(policy_result.goal_id),
                "motion_applied": bool(policy_result.motion_applied),
                "allow_gripper_close": bool(policy_result.allow_gripper_close),
                "policy_step_count": int(policy_result.step_count),
                "run_directory": str(policy_result.run_directory),
            }
        )
        if not bool(policy_result.completed):
            return (
                RealExecutionResult(
                    success=False,
                    status="visual_servo_failed",
                    message=str(policy_result.message),
                    grasp_id=world_grasp.grasp_id,
                    pregrasp_reached=pregrasp_reached,
                    grasp_reached=False,
                    lift_reached=False,
                    attempt_artifact_path=attempt_artifact_path,
                ),
                steps,
            )
        if not bool(policy_result.motion_applied):
            result = RealExecutionResult(
                success=True,
                status="visual_servo_dry_run_completed",
                message="Policy completion gate passed in dry-run; no robot motion or gripper close was executed.",
                grasp_id=world_grasp.grasp_id,
                pregrasp_reached=pregrasp_reached,
                grasp_reached=False,
                lift_reached=False,
                attempt_artifact_path=attempt_artifact_path,
            )
            _record_step("close_gripper", ok=True, message="Skipped because the policy command sink was dry-run.")
            return result, steps
        grasp_reached = True
    else:
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
    if gripper is not None:
        if policy_result is not None and not bool(policy_result.allow_gripper_close):
            message = "Policy completed, but deployment configuration does not approve gripper closure."
            _record_step("close_gripper", ok=False, message=message)
            return (
                RealExecutionResult(
                    success=False,
                    status="gripper_close_not_approved",
                    message=message,
                    grasp_id=world_grasp.grasp_id,
                    pregrasp_reached=pregrasp_reached,
                    grasp_reached=grasp_reached,
                    lift_reached=False,
                    attempt_artifact_path=attempt_artifact_path,
                ),
                steps,
            )
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

    moveit_gripper_state = _configured_moveit_gripper_state(
        config=config,
        width_m=world_grasp.jaw_width,
    )
    if moveit_gripper_state:
        ok, message = commander.apply_planning_scene_robot_state(moveit_gripper_state)
        _record_step("apply_closed_gripper_moveit_state", ok=ok, message=message)
        if not ok:
            return (
                RealExecutionResult(
                    success=False,
                    status="closed_gripper_moveit_state_failed",
                    message=message,
                    grasp_id=world_grasp.grasp_id,
                    pregrasp_reached=pregrasp_reached,
                    grasp_reached=grasp_reached,
                    lift_reached=False,
                    attempt_artifact_path=attempt_artifact_path,
                ),
                steps,
            )

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


def _planning_scene_failed_result(
    *,
    grasp_id: str,
    message: str,
    attempt_artifact_path: Path,
) -> RealExecutionResult:
    return RealExecutionResult(
        success=False,
        status="planning_scene_failed",
        message=message,
        grasp_id=grasp_id,
        pregrasp_reached=False,
        grasp_reached=False,
        lift_reached=False,
        attempt_artifact_path=attempt_artifact_path,
    )


def _configured_moveit_gripper_state(*, config, width_m: float) -> dict[str, float]:
    """Return the explicit single-KUKA passive driver state, when configured."""

    joint_name = str(getattr(config, "moveit_gripper_joint_name", "")).strip()
    if not joint_name:
        return {}
    return {
        joint_name: kuka_moveit_gripper_driver_position_from_width(float(width_m)),
    }


def _real_execution_candidate_queue(bundle, *, config) -> tuple[object, ...]:
    """Return ordinary stage-2 candidates in their live score order."""

    minimum_closed_width = float(config.gripper_closed_width)
    maximum_open_width = float(config.gripper_open_width)
    if (
        not math.isfinite(minimum_closed_width)
        or minimum_closed_width < 0.0
        or not math.isfinite(maximum_open_width)
        or maximum_open_width <= minimum_closed_width
    ):
        raise ValueError(
            "real_execution gripper widths must satisfy "
            "0 <= gripper_closed_width < gripper_open_width."
        )
    if str(config.grasp_approach_controller) != "d405_policy":
        selected = _select_bundle_grasp(bundle, grasp_id=str(config.grasp_id))
        jaw_width = float(selected.jaw_width)
        if not math.isfinite(jaw_width) or not (
            minimum_closed_width - 1.0e-9
            <= jaw_width
            <= maximum_open_width + 1.0e-9
        ):
            raise RuntimeError(
                f"Selected grasp '{selected.grasp_id}' does not fit the physical gripper: "
                f"jaw_width={jaw_width:.6f} m must lie in "
                f"{minimum_closed_width:.6f}--{maximum_open_width:.6f} m."
            )
        return (selected,)
    if not bundle.candidates:
        raise RuntimeError("The stage-2 bundle contains no feasible grasps to execute.")
    candidates = tuple(
        candidate
        for candidate in bundle.candidates
        if math.isfinite(float(candidate.jaw_width))
        and float(candidate.jaw_width) >= minimum_closed_width - 1.0e-9
        and float(candidate.jaw_width) <= maximum_open_width + 1.0e-9
    )
    rejected_count = len(bundle.candidates) - len(candidates)
    if rejected_count:
        print(
            "[REAL-PREPLAN] "
            f"Skipped {rejected_count}/{len(bundle.candidates)} stage-2 grasps because jaw width "
            f"falls outside the configured {minimum_closed_width:.6f}--"
            f"{maximum_open_width:.6f} m physical stroke; "
            f"{len(candidates)} remain.",
            flush=True,
        )
    if not candidates:
        raise RuntimeError(
            f"None of the {len(bundle.candidates)} live stage-2 grasps fits the physical gripper: "
            f"jaw_width must lie in {minimum_closed_width:.6f}--"
            f"{maximum_open_width:.6f} m."
        )
    return candidates


def _policy_approach_width(*, jaw_width_m: float, config) -> float:
    """Clamp candidate clearance to the configured physical gripper stroke."""

    clearance_total = float(config.gripper_width_clearance_m)
    if not math.isfinite(clearance_total) or clearance_total < 0.0:
        raise ValueError("real_execution.gripper_width_clearance_m must be finite and non-negative.")
    minimum_width = float(config.gripper_closed_width)
    maximum_width = float(config.gripper_open_width)
    return max(
        minimum_width,
        min(maximum_width, float(jaw_width_m) + clearance_total),
    )


def execute_real_grasp_from_bundle(
    *,
    input_json: Path,
    config,
    pregrasp_selected_callback=None,
) -> RealExecutionResult:
    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace first.")
    bundle = load_grasp_bundle(input_json)
    object_pose_world = _bundle_execution_pose_world(bundle)
    if object_pose_world is None:
        raise RuntimeError("The stage-2 bundle does not contain execution_world_pose metadata.")

    execution_candidates = _real_execution_candidate_queue(bundle, config=config)
    selected_grasp = execution_candidates[0]
    config = replace(config, grasp_id=str(selected_grasp.grasp_id))
    world_grasp = saved_grasp_to_world_grasp(
        selected_grasp,
        object_pose_world,
        pregrasp_offset=float(config.pregrasp_offset_m),
        gripper_width_clearance=float(config.gripper_width_clearance_m),
    )
    if str(config.grasp_approach_controller) == "d405_policy":
        world_grasp = replace(
            world_grasp,
            gripper_width=_policy_approach_width(jaw_width_m=selected_grasp.jaw_width, config=config),
        )
    expected_part_id = Path(str(bundle.target_mesh_path)).stem
    attempt_artifact_path = Path(str(config.attempt_artifact))
    steps: list[dict[str, object]] = []
    planning_scene_obstacles = tuple(config.planning_scene_obstacles)

    moveit_config = MoveItPoseCommanderConfig(
        planning_group=str(config.planning_group),
        pose_link=str(config.pose_link),
        joint_names=tuple(config.joint_names) or MoveItPoseCommanderConfig().joint_names,
        moveit_namespace=str(config.moveit_namespace),
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

        open_moveit_gripper_state = _configured_moveit_gripper_state(
            config=config,
            width_m=float(config.gripper_open_width),
        )
        if open_moveit_gripper_state:
            ok, message = commander.apply_planning_scene_robot_state(open_moveit_gripper_state)
            steps.append(
                {
                    "name": "apply_open_gripper_moveit_state",
                    "ok": bool(ok),
                    "message": message,
                    "joint_state": dict(open_moveit_gripper_state),
                }
            )
            if not ok:
                result = _planning_scene_failed_result(
                    grasp_id=world_grasp.grasp_id,
                    message=message,
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

        if planning_scene_obstacles:
            ok, message = commander.apply_planning_scene_obstacles(
                planning_scene_obstacles,
                default_frame_id=str(config.frame_id),
            )
            steps.append(
                {
                    "name": "apply_planning_scene",
                    "ok": bool(ok),
                    "message": message,
                    "obstacle_count": len(planning_scene_obstacles),
                }
            )
            if not ok:
                result = _planning_scene_failed_result(
                    grasp_id=world_grasp.grasp_id,
                    message=message,
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

        if bool(config.gripper_enabled):
            gripper = _make_gripper_client(commander=commander, config=config)
            gripper.wait_for_server(timeout_s=float(config.wait_for_moveit_timeout_s))

        pregrasp_trajectory = None
        goal_joint_positions = None
        pregrasp_plan_failures: list[str] = []
        selected_candidate_rank = 0
        for candidate_rank, candidate in enumerate(execution_candidates, start=1):
            candidate_world_grasp = saved_grasp_to_world_grasp(
                candidate,
                object_pose_world,
                pregrasp_offset=float(config.pregrasp_offset_m),
                gripper_width_clearance=float(config.gripper_width_clearance_m),
            )
            if str(config.grasp_approach_controller) == "d405_policy":
                candidate_world_grasp = replace(
                    candidate_world_grasp,
                    gripper_width=_policy_approach_width(jaw_width_m=candidate.jaw_width, config=config),
                )
            pregrasp_target = world_grasp_pose_targets(
                candidate_world_grasp,
                frame_id=str(config.frame_id),
                lift_height_m=float(config.lift_height_m),
            )["pregrasp"]
            candidate_trajectory, pregrasp_plan_message = commander.plan_to_pose(
                pregrasp_target,
                label=f"pregrasp_{candidate.grasp_id}",
            )
            candidate_ok = candidate_trajectory is not None
            candidate_goal_joints = None
            goal_ik_message = "not requested"
            if candidate_trajectory is not None and str(config.grasp_approach_controller) == "d405_policy":
                trajectory_names = tuple(candidate_trajectory.joint_trajectory.joint_names)
                trajectory_points = tuple(candidate_trajectory.joint_trajectory.points)
                if not trajectory_points:
                    goal_ik_message = "pregrasp trajectory has no points"
                else:
                    final_positions = tuple(float(value) for value in trajectory_points[-1].positions)
                    final_by_name = dict(zip(trajectory_names, final_positions))
                    try:
                        pregrasp_seed = tuple(final_by_name[name] for name in moveit_config.joint_names)
                    except KeyError as exc:
                        goal_ik_message = f"pregrasp trajectory is missing joint {exc.args[0]}"
                    else:
                        grasp_target = world_grasp_pose_targets(
                            candidate_world_grasp,
                            frame_id=str(config.frame_id),
                            lift_height_m=float(config.lift_height_m),
                        )["grasp"]
                        candidate_goal_joints, goal_ik_message = commander.compute_ik(
                            grasp_target,
                            seed_joint_positions=pregrasp_seed,
                            avoid_collisions=True,
                        )
                candidate_ok = candidate_goal_joints is not None
            steps.append(
                {
                    "name": "preplan_pregrasp_candidate",
                    "ok": candidate_ok,
                    "message": pregrasp_plan_message,
                    "candidate_rank": candidate_rank,
                    "grasp_id": str(candidate.grasp_id),
                    "live_score": float(candidate.score or 0.0),
                    "goal_ik_message": goal_ik_message,
                    "goal_joint_positions": candidate_goal_joints,
                    "target_pose": {
                        "frame_id": pregrasp_target.frame_id,
                        "position_xyz": list(pregrasp_target.position_xyz),
                        "orientation_xyzw": list(pregrasp_target.orientation_xyzw),
                    },
                }
            )
            print(
                "[REAL-PREPLAN] "
                f"rank={candidate_rank}/{len(execution_candidates)} grasp={candidate.grasp_id} "
                f"success={candidate_ok} pregrasp={pregrasp_plan_message} goal_ik={goal_ik_message}",
                flush=True,
            )
            if candidate_ok:
                selected_grasp = candidate
                world_grasp = candidate_world_grasp
                config = replace(config, grasp_id=str(candidate.grasp_id))
                pregrasp_trajectory = candidate_trajectory
                goal_joint_positions = candidate_goal_joints
                selected_candidate_rank = candidate_rank
                break
            failure = pregrasp_plan_message
            if candidate_trajectory is not None:
                failure = f"goal IK failed: {goal_ik_message}"
            pregrasp_plan_failures.append(f"{candidate.grasp_id}: {failure}")
        if pregrasp_trajectory is None:
            failure_summary = "; ".join(pregrasp_plan_failures[:8])
            if len(pregrasp_plan_failures) > 8:
                failure_summary += f"; and {len(pregrasp_plan_failures) - 8} more"
            result = RealExecutionResult(
                success=False,
                status="pregrasp_planning_failed",
                message=(
                    f"All {len(execution_candidates)} live stage-2 candidates failed collision-aware "
                    f"MoveIt pregrasp planning or grasp IK: {failure_summary}"
                ),
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
                steps=steps,
            )
            return result

        visual_servo_preparation = None
        visual_servo_entrypoint = None
        if str(config.grasp_approach_controller) == "d405_policy":
            from grasp_planning.rl.d405_goal_renderer import render_d405_goal_for_grasp
            from grasp_planning.ros2.d405_visual_servo import (
                prepare_d405_policy_visual_servo,
                run_d405_policy_visual_servo,
            )

            if goal_joint_positions is None:
                raise RuntimeError("MoveIt did not provide grasp IK joints for runtime goal rendering.")
            goal_observation_path = attempt_artifact_path.with_name(
                f"policy_goal_{world_grasp.grasp_id}.npz"
            )
            rendered_goal = render_d405_goal_for_grasp(
                config_path=Path(str(config.visual_servo_config)),
                stage2_bundle_path=input_json,
                grasp_id=world_grasp.grasp_id,
                part_id=expected_part_id,
                goal_joint_positions=goal_joint_positions,
                goal_tcp_position=world_grasp.position_w,
                goal_tcp_orientation_xyzw=world_grasp.orientation_xyzw,
                approach_width_m=world_grasp.gripper_width,
                maximum_approach_width_m=float(config.gripper_open_width),
                output_path=goal_observation_path,
            )
            steps.append(
                {
                    "name": "render_policy_goal_observation",
                    "ok": True,
                    "grasp_id": world_grasp.grasp_id,
                    "goal_id": rendered_goal.goal_id,
                    "path": str(rendered_goal.path),
                    "sha256": rendered_goal.sha256,
                    "approach_width_m": float(world_grasp.gripper_width),
                    "maximum_approach_width_m": float(config.gripper_open_width),
                }
            )
            if pregrasp_selected_callback is not None:
                pregrasp_selected_callback(
                    selected_grasp=selected_grasp,
                    config=config,
                    candidate_rank=selected_candidate_rank,
                    goal_observation_path=rendered_goal.path,
                )
            visual_servo_preparation = prepare_d405_policy_visual_servo(
                config_path=Path(str(config.visual_servo_config)),
                expected_grasp_id=world_grasp.grasp_id,
                expected_part_id=expected_part_id,
                goal_observation_path_override=rendered_goal.path,
            )
            visual_servo_entrypoint = run_d405_policy_visual_servo

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
                steps=steps,
            )
            return result

        visual_servo_runner = None
        if str(config.grasp_approach_controller) == "d405_policy":
            assert visual_servo_entrypoint is not None
            assert visual_servo_preparation is not None

            def visual_servo_runner():
                return visual_servo_entrypoint(
                    config_path=Path(str(config.visual_servo_config)),
                    expected_grasp_id=world_grasp.grasp_id,
                    expected_part_id=expected_part_id,
                    allow_real_motion=True,
                    preparation=visual_servo_preparation,
                )

        try:
            result, execution_steps = _execute_selected_world_grasp(
                commander=commander,
                gripper=gripper,
                world_grasp=world_grasp,
                config=config,
                attempt_artifact_path=attempt_artifact_path,
                visual_servo_runner=visual_servo_runner,
                pregrasp_trajectory=pregrasp_trajectory,
            )
        except KeyboardInterrupt:
            _best_effort_stop_gripper(gripper, reason="keyboard interrupt")
            raise
        except Exception as exc:
            if not bool(getattr(exc, "gripper_stop_attempted", False)):
                _best_effort_stop_gripper(gripper, reason="execution exception")
            raise
        steps.extend(execution_steps)
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
