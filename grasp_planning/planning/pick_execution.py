"""Shared FR3 pick-execution helpers for debug grasp attempts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

from grasp_planning.planning.fr3_motion_context import FR3MotionContext
from grasp_planning.start_poses import (
    DEFAULT_ARM_START_JOINT_POS,
    DEFAULT_HAND_OPEN_WIDTH,
    DEFAULT_KUKA_ARM_START_JOINT_POS,
    gripper_joint_target_from_width,
    gripper_max_open_width,
    is_gripper_command_joint_name,
    is_gripper_joint_name,
)

from .trajectory_executor import TrajectoryExecutor
from .types import JointTrajectory


@dataclass(frozen=True)
class PickExecutionResult:
    success: bool
    status: str
    message: str
    object_lift_height_m: float | None = None
    target_lift_height_m: float | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


StepCallback = Callable[[], None]
GRIPPER_CLOSE_SETTLE_DURATION_S = 0.5


def drive_robot_to_start_pose(
    sim,
    scene,
    *,
    hand_open_width: float | None = None,
    step_callback: StepCallback | None = None,
) -> None:
    """Actively settle the FR3 into a safe home pose before planning."""

    robot = scene["robot"]
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_start_positions = {**DEFAULT_ARM_START_JOINT_POS, **DEFAULT_KUKA_ARM_START_JOINT_POS}
    arm_joint_names = tuple(name for name in arm_start_positions if name in joint_name_to_idx)
    arm_joint_ids = [joint_name_to_idx[name] for name in arm_joint_names]
    arm_targets = torch.tensor(
        [[arm_start_positions[name] for name in arm_joint_names]], dtype=torch.float32, device=robot.device
    )
    hand_joint_names = tuple(name for name in robot.joint_names if is_gripper_command_joint_name(name))
    physics_dt = sim.get_physics_dt()
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]

    for _ in range(max(1, int(1.5 / physics_dt))):
        robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
        if hand_joint_names:
            hand_targets = torch.full(
                (1, len(hand_joint_ids)),
                0.0,
                dtype=torch.float32,
                device=robot.device,
            )
            for index, name in enumerate(hand_joint_names):
                hand_target = gripper_max_open_width(name) if hand_open_width is None else float(hand_open_width)
                hand_targets[0, index] = gripper_joint_target_from_width(name, hand_target)
            robot.set_joint_position_target(hand_targets, joint_ids=hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)
        if step_callback is not None:
            step_callback()


def _command_gripper_width(
    *,
    sim,
    scene,
    robot,
    width: float,
    duration_s: float,
    max_duration_s: float | None = None,
    hold_context: FR3MotionContext | None = None,
    hold_arm_waypoint: torch.Tensor | None = None,
    position_tolerance: float = 0.001,
    contact_position_tolerance: float = 0.003,
    stall_delta_tolerance: float = 1.0e-5,
    min_contact_motion_m: float = 0.001,
    settle_duration_s: float = 0.25,
    force_joint_state: bool = False,
    step_callback: StepCallback | None = None,
) -> dict[str, object]:
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    hand_joint_names = tuple(name for name in robot.joint_names if is_gripper_command_joint_name(name))
    if not hand_joint_names:
        return {"gripper_close_status": "no_hand_joints", "gripper_close_steps": 0}
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]
    physics_dt = sim.get_physics_dt()
    min_steps = max(1, int(duration_s / physics_dt))
    max_steps = max(1, int((duration_s if max_duration_s is None else max_duration_s) / physics_dt))
    max_steps = max(max_steps, min_steps)
    settle_steps_required = max(1, int(settle_duration_s / physics_dt))
    hand_targets = torch.full(
        (1, len(hand_joint_ids)),
        0.0,
        dtype=torch.float32,
        device=robot.device,
    )
    for index, name in enumerate(hand_joint_names):
        hand_targets[0, index] = gripper_joint_target_from_width(name, width)
    last_hand_q = None
    stable_steps = 0
    close_status = "duration_elapsed"
    final_error = None
    final_delta = None
    max_motion_since_start = 0.0
    steps_run = 0
    saw_hand_state = False
    initial_hand_q = _hand_joint_positions(robot=robot, hand_joint_ids=hand_joint_ids)
    final_hand_q = initial_hand_q.clone() if initial_hand_q is not None else None
    for step_idx in range(1, max_steps + 1):
        if hold_context is not None and hold_arm_waypoint is not None:
            hold_context.command_arm(hold_arm_waypoint)
        robot.set_joint_position_target(hand_targets, joint_ids=hand_joint_ids)
        if force_joint_state and initial_hand_q is not None and hasattr(robot, "write_joint_state_to_sim"):
            alpha = min(1.0, float(step_idx) / float(min_steps))
            hand_q_cmd = ((1.0 - alpha) * initial_hand_q + alpha * hand_targets).clone()
            hand_qd_cmd = torch.zeros_like(hand_q_cmd)
            robot.write_joint_state_to_sim(hand_q_cmd, hand_qd_cmd, joint_ids=hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)
        if step_callback is not None:
            step_callback()
        steps_run = step_idx
        hand_q = _hand_joint_positions(robot=robot, hand_joint_ids=hand_joint_ids)
        if hand_q is None:
            continue
        final_hand_q = hand_q.clone()
        saw_hand_state = True
        final_error = float(torch.max(torch.abs(hand_q - hand_targets)).item())
        if initial_hand_q is None:
            initial_hand_q = hand_q.clone()
        max_motion_since_start = max(
            max_motion_since_start,
            float(torch.max(torch.abs(hand_q - initial_hand_q)).item()),
        )
        if last_hand_q is not None:
            final_delta = float(torch.max(torch.abs(hand_q - last_hand_q)).item())
        target_reached = final_error <= float(position_tolerance)
        contact_stalled = (
            final_delta is not None
            and final_delta <= float(stall_delta_tolerance)
            and max_motion_since_start >= float(min_contact_motion_m)
        )
        if step_idx >= min_steps and (target_reached or contact_stalled):
            stable_steps += 1
            close_status = "target_reached" if target_reached else "contact_stalled"
        else:
            stable_steps = 0
        last_hand_q = hand_q
        if stable_steps >= settle_steps_required:
            break
    else:
        if (
            saw_hand_state
            and final_delta is not None
            and final_delta <= float(stall_delta_tolerance)
            and max_motion_since_start >= float(min_contact_motion_m)
        ):
            close_status = "contact_stalled"
        else:
            close_status = "max_duration_elapsed" if saw_hand_state else "max_duration_elapsed_no_hand_state"
    diagnostics: dict[str, object] = {
        "gripper_close_status": close_status,
        "gripper_close_steps": int(steps_run),
        "gripper_close_duration_s": float(steps_run * physics_dt),
        "gripper_close_target_width_m": float(width),
        "gripper_close_joint_names": list(hand_joint_names),
        "gripper_close_target_joint_positions": [float(value) for value in hand_targets[0].tolist()],
        "gripper_close_saw_hand_state": bool(saw_hand_state),
        "gripper_close_max_motion_since_start_m": float(max_motion_since_start),
        "gripper_close_forced_joint_state": bool(force_joint_state),
        "gripper_close_position_tolerance_m": float(position_tolerance),
        "gripper_close_contact_position_tolerance_m": float(contact_position_tolerance),
        "gripper_close_stall_delta_tolerance_m": float(stall_delta_tolerance),
        "gripper_close_min_contact_motion_m": float(min_contact_motion_m),
    }
    if initial_hand_q is not None:
        diagnostics["gripper_close_initial_joint_positions"] = [float(value) for value in initial_hand_q[0].tolist()]
    if final_hand_q is not None:
        diagnostics["gripper_close_final_joint_positions"] = [float(value) for value in final_hand_q[0].tolist()]
    if final_error is not None:
        diagnostics["gripper_close_final_max_position_error"] = final_error
    if final_delta is not None:
        diagnostics["gripper_close_final_max_step_delta"] = final_delta
    return diagnostics


def _hand_joint_positions(*, robot, hand_joint_ids: list[int]) -> torch.Tensor | None:
    data = getattr(robot, "data", None)
    joint_pos = getattr(data, "joint_pos", None)
    if joint_pos is None:
        return None
    try:
        return joint_pos[:, hand_joint_ids].clone().to(dtype=torch.float32)
    except (IndexError, TypeError, AttributeError):
        return None


def _object_root_z(object_asset) -> float | None:
    if object_asset is None:
        return None
    try:
        value = object_asset.data.root_link_pose_w[0, 2]
    except (AttributeError, IndexError, TypeError):
        return None
    if hasattr(value, "item"):
        value = value.item()
    z = float(value)
    return z if math.isfinite(z) else None


def _finite_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _nominal_max_open_gripper_width(hand_joint_names: tuple[str, ...]) -> float:
    gripper_joint_names = tuple(name for name in hand_joint_names if is_gripper_joint_name(name))
    if not gripper_joint_names:
        return float(DEFAULT_HAND_OPEN_WIDTH)
    return max(float(gripper_max_open_width(name)) for name in gripper_joint_names)


def _kuka_contact_stall_matches_grasp_width(
    diagnostics: dict[str, object],
    selected_gripper_width_m: float | None,
) -> bool | None:
    if selected_gripper_width_m is None:
        return None

    joint_names = diagnostics.get("gripper_close_joint_names")
    if not isinstance(joint_names, list) or "left_finger_joint" not in joint_names:
        return None

    final_positions = diagnostics.get("gripper_close_final_joint_positions")
    if not isinstance(final_positions, list) or len(final_positions) != len(joint_names):
        diagnostics["gripper_close_contact_stall_accept_reason"] = "missing final KUKA finger positions"
        return False

    final_step_delta = diagnostics.get("gripper_close_final_max_step_delta")
    if final_step_delta is not None and float(final_step_delta) > 1.0e-4:
        diagnostics["gripper_close_contact_stall_accept_reason"] = (
            f"finger still moving at {float(final_step_delta):.6f} m/step"
        )
        return False

    left_index = joint_names.index("left_finger_joint")
    final_close = abs(float(final_positions[left_index]))
    expected_close = abs(gripper_joint_target_from_width("left_finger_joint", float(selected_gripper_width_m)))
    tolerance = 0.003
    diagnostics["gripper_close_contact_stall_max_abs_joint_position"] = float(final_close)
    diagnostics["gripper_close_contact_stall_expected_min_joint_position"] = float(expected_close)
    diagnostics["gripper_close_contact_stall_expected_tolerance_m"] = float(tolerance)
    accepted = final_close + tolerance >= expected_close
    diagnostics["gripper_close_contact_stall_accepted"] = bool(accepted)
    if not accepted:
        diagnostics["gripper_close_contact_stall_accept_reason"] = (
            f"finger only closed to {final_close:.4f} m, expected at least {expected_close:.4f} m "
            f"for selected grasp width {float(selected_gripper_width_m):.4f} m"
        )
    return accepted


def _validate_object_lift(
    *,
    object_asset,
    initial_object_z: float | None,
    success_height_margin_m: float,
    observed_object_max_z: float | None = None,
    extra_diagnostics: Mapping[str, object] | None = None,
) -> PickExecutionResult | None:
    if object_asset is None:
        return None
    final_object_z = _object_root_z(object_asset)
    observed_object_max_z_was_provided = observed_object_max_z is not None
    initial_object_z = _finite_float_or_none(initial_object_z)
    final_object_z = _finite_float_or_none(final_object_z)
    observed_object_max_z = _finite_float_or_none(observed_object_max_z)
    if (
        initial_object_z is None
        or final_object_z is None
        or (observed_object_max_z_was_provided and observed_object_max_z is None)
    ):
        return PickExecutionResult(
            False,
            "object_pose_unavailable",
            "Could not read a finite Isaac object pose to validate pickup lift.",
            target_lift_height_m=float(success_height_margin_m),
        )

    peak_object_z = final_object_z
    if observed_object_max_z is not None:
        peak_object_z = max(peak_object_z, observed_object_max_z)
    object_lift_height_m = float(peak_object_z - initial_object_z)
    final_object_lift_height_m = float(final_object_z - initial_object_z)
    target_lift_height_m = float(success_height_margin_m)
    diagnostics: dict[str, object] = {
        "initial_object_z_m": float(initial_object_z),
        "final_object_z_m": float(final_object_z),
        "peak_object_z_m": float(peak_object_z),
        "final_object_lift_height_m": final_object_lift_height_m,
    }
    if extra_diagnostics:
        diagnostics.update(dict(extra_diagnostics))
    lift_message = (
        f"Isaac pickup lifted object by {object_lift_height_m:.4f} m during lift "
        f"(final after validation {final_object_lift_height_m:.4f} m)"
    )
    if object_lift_height_m < target_lift_height_m:
        return PickExecutionResult(
            False,
            "object_lift_failed",
            f"{lift_message} (required {target_lift_height_m:.4f} m).",
            object_lift_height_m=object_lift_height_m,
            target_lift_height_m=target_lift_height_m,
            diagnostics=diagnostics,
        )
    return PickExecutionResult(
        True,
        "ok",
        f"{lift_message}.",
        object_lift_height_m=object_lift_height_m,
        target_lift_height_m=target_lift_height_m,
        diagnostics=diagnostics,
    )


def _joint_trajectory_from_moveit_waypoints(
    *,
    context: FR3MotionContext,
    waypoints: tuple[tuple[float, ...], ...],
    label: str,
) -> JointTrajectory:
    if not waypoints:
        raise ValueError(f"MoveIt trajectory '{label}' has no waypoints.")
    expected = int(context.arm_joint_ids.numel())
    tensors = []
    for waypoint in waypoints:
        if len(waypoint) != expected:
            raise ValueError(f"MoveIt trajectory '{label}' expected {expected} joints, got {len(waypoint)}.")
        tensors.append(torch.tensor([waypoint], dtype=torch.float32, device=context.device))
    return JointTrajectory(waypoints=tensors, dt=context.physics_dt)


def _execute_moveit_waypoint_segment(
    *,
    context: FR3MotionContext,
    executor: TrajectoryExecutor,
    moveit_joint_trajectories: Mapping[str, tuple[tuple[float, ...], ...]],
    label: str,
) -> tuple[bool, str]:
    try:
        trajectory = _joint_trajectory_from_moveit_waypoints(
            context=context,
            waypoints=tuple(moveit_joint_trajectories.get(label, ())),
            label=label,
        )
    except ValueError as exc:
        return False, str(exc)
    ok, detail = executor.execute(trajectory)
    return bool(ok), detail


def execute_moveit_joint_trajectory_sequence(
    *,
    sim,
    scene,
    robot,
    moveit_joint_trajectories: Mapping[str, tuple[tuple[float, ...], ...]],
    labels: tuple[str, ...],
    fixed_gripper_width: float,
    max_joint_speed_rad_s: float = 0.35,
    step_callback: StepCallback | None = None,
) -> PickExecutionResult:
    """Execute arbitrary MoveIt segments continuously without resetting the arm."""
    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=float(fixed_gripper_width),
    )
    executor = TrajectoryExecutor(
        context,
        max_joint_speed_rad_s=float(max_joint_speed_rad_s),
        step_callback=step_callback,
    )
    diagnostics: dict[str, object] = {"labels": list(labels), "completed_labels": []}
    for label in labels:
        ok, detail = _execute_moveit_waypoint_segment(
            context=context,
            executor=executor,
            moveit_joint_trajectories=moveit_joint_trajectories,
            label=label,
        )
        if not ok:
            return PickExecutionResult(
                False,
                "moveit_sequence_failed",
                f"MoveIt sequence segment '{label}' failed: {detail}",
                diagnostics=diagnostics,
            )
        diagnostics["completed_labels"].append(label)
    return PickExecutionResult(
        True,
        "ok",
        f"Executed {len(labels)} continuous MoveIt trajectory segments.",
        diagnostics=diagnostics,
    )


def _moveit_waypoint_tensor(
    *,
    context: FR3MotionContext,
    moveit_joint_trajectories: Mapping[str, tuple[tuple[float, ...], ...]],
    label: str,
    index: int = -1,
) -> torch.Tensor:
    trajectory = _joint_trajectory_from_moveit_waypoints(
        context=context,
        waypoints=tuple(moveit_joint_trajectories.get(label, ())),
        label=label,
    )
    return trajectory.waypoints[index].clone()


def _hold_arm_waypoint(
    *,
    context: FR3MotionContext,
    waypoint: torch.Tensor,
    duration_s: float,
    step_callback: StepCallback | None,
) -> None:
    steps = max(0, int(float(duration_s) / context.physics_dt))
    for _ in range(steps):
        context.command_arm(waypoint)
        context.command_fixed_gripper()
        context.scene.write_data_to_sim()
        context.sim.step()
        context.scene.update(context.physics_dt)
        if step_callback is not None:
            step_callback()


def _hold_arm_waypoint_until_settled(
    *,
    context: FR3MotionContext,
    waypoint: torch.Tensor,
    duration_s: float,
    tolerance_rad: float,
    step_callback: StepCallback | None,
) -> dict[str, object]:
    if not hasattr(context, "command_arm") or not hasattr(context, "command_fixed_gripper"):
        return {"grasp_preclose_hold_supported": False}
    steps = max(1, int(float(duration_s) / context.physics_dt))
    last_error = None
    settled = False
    for step in range(1, steps + 1):
        context.command_arm(waypoint)
        context.command_fixed_gripper()
        context.scene.write_data_to_sim()
        context.sim.step()
        context.scene.update(context.physics_dt)
        if step_callback is not None:
            step_callback()
        if hasattr(context, "get_arm_q"):
            error = torch.max(torch.abs(context.get_arm_q() - waypoint))
            last_error = float(error.item())
            if last_error <= float(tolerance_rad):
                settled = True
                break
    return {
        "grasp_preclose_hold_supported": True,
        "grasp_preclose_hold_steps": int(step),
        "grasp_preclose_hold_duration_s": float(step * context.physics_dt),
        "grasp_preclose_hold_settled": bool(settled),
        "grasp_preclose_hold_final_error_rad": None if last_error is None else float(last_error),
    }


def execute_pick_from_moveit_joint_trajectories(
    *,
    sim,
    scene,
    robot,
    object_asset=None,
    moveit_joint_trajectories: Mapping[str, tuple[tuple[float, ...], ...]],
    open_gripper_width: float,
    closed_gripper_width: float,
    pregrasp_only: bool,
    success_height_margin_m: float = 0.05,
    max_joint_speed_rad_s: float = 0.35,
    grasp_settle_time_s: float = 0.0,
    gripper_close_duration_s: float = 1.2,
    gripper_close_max_duration_s: float = 8.0,
    postclose_hold_s: float = 0.0,
    selected_gripper_width_m: float | None = None,
    step_callback: StepCallback | None = None,
) -> PickExecutionResult:
    """Execute MoveIt-planned direct-pick joint waypoints inside Isaac."""

    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=float(open_gripper_width),
    )
    print(
        "[INFO]: Isaac motion context "
        f"fixed_base={getattr(robot, 'is_fixed_base', None)} "
        f"arm_joints={list(getattr(context, 'arm_joint_names', ()))} "
        f"hand_joints={list(getattr(context, 'hand_joint_names', ()))}",
        flush=True,
    )
    initial_object_z = _object_root_z(object_asset)
    observed_lift_object_max_z = None
    capture_lift_object_z = False

    def _capture_lift_object_z() -> None:
        nonlocal observed_lift_object_max_z
        object_z = _object_root_z(object_asset)
        if object_z is None:
            return
        if observed_lift_object_max_z is None:
            observed_lift_object_max_z = object_z
        else:
            observed_lift_object_max_z = max(float(observed_lift_object_max_z), float(object_z))

    def _step_callback() -> None:
        if capture_lift_object_z:
            _capture_lift_object_z()
        if step_callback is not None:
            step_callback()

    nominal_max_open_width = _nominal_max_open_gripper_width(tuple(getattr(context, "hand_joint_names", ())))
    moveit_diagnostics = {
        "open_gripper_width_m": float(open_gripper_width),
        "closed_gripper_width_m": float(closed_gripper_width),
        "nominal_max_open_gripper_width_m": float(nominal_max_open_width),
        "open_gripper_width_exceeds_nominal_limit": float(open_gripper_width) > nominal_max_open_width + 1.0e-6,
        "max_joint_speed_rad_s": float(max_joint_speed_rad_s),
        "grasp_settle_time_s": float(grasp_settle_time_s),
        "gripper_close_duration_s_requested": float(gripper_close_duration_s),
        "gripper_close_max_duration_s_requested": float(gripper_close_max_duration_s),
        "postclose_hold_s": float(postclose_hold_s),
        "selected_gripper_width_m": None if selected_gripper_width_m is None else float(selected_gripper_width_m),
    }
    executor_kwargs = {
        "max_joint_speed_rad_s": float(max_joint_speed_rad_s),
        "step_callback": _step_callback,
    }
    executor = TrajectoryExecutor(context, **executor_kwargs)
    if hasattr(context, "get_arm_q") and hasattr(context, "reset_joint_state"):
        first_pregrasp_waypoint = _moveit_waypoint_tensor(
            context=context,
            moveit_joint_trajectories=moveit_joint_trajectories,
            label="pregrasp",
            index=0,
        )
        current_q = context.get_arm_q()
        initial_start_error = float(torch.max(torch.abs(current_q - first_pregrasp_waypoint)).item())
        print(
            "[INFO]: Aligning Isaac arm state to MoveIt first waypoint "
            f"initial_max_joint_error={initial_start_error:.4f}.",
            flush=True,
        )
        context.reset_joint_state(first_pregrasp_waypoint, steps=5)
        reset_start_error = float(torch.max(torch.abs(context.get_arm_q() - first_pregrasp_waypoint)).item())
        print(
            f"[INFO]: Isaac arm state after reset max_joint_error={reset_start_error:.4f}.",
            flush=True,
        )
        moveit_diagnostics["initial_start_error_rad"] = initial_start_error
        moveit_diagnostics["reset_start_error_rad"] = reset_start_error

    ok, detail = _execute_moveit_waypoint_segment(
        context=context,
        executor=executor,
        moveit_joint_trajectories=moveit_joint_trajectories,
        label="pregrasp",
    )
    if not ok:
        return PickExecutionResult(
            False,
            "moveit_pregrasp_failed",
            f"MoveIt pregrasp execution failed: {detail}",
            diagnostics=moveit_diagnostics,
        )
    if pregrasp_only:
        return PickExecutionResult(True, "ok", "MoveIt pregrasp trajectory executed.", diagnostics=moveit_diagnostics)

    ok, detail = _execute_moveit_waypoint_segment(
        context=context,
        executor=executor,
        moveit_joint_trajectories=moveit_joint_trajectories,
        label="grasp",
    )
    moveit_diagnostics["grasp_waypoint_settled"] = bool(ok)
    if not ok:
        moveit_diagnostics["grasp_waypoint_settle_detail"] = str(detail)
        print(
            "[WARN]: MoveIt grasp waypoint did not fully settle before close; "
            f"continuing with gripper close at current pose: {detail}",
            flush=True,
        )
    grasp_waypoint = _moveit_waypoint_tensor(
        context=context,
        moveit_joint_trajectories=moveit_joint_trajectories,
        label="grasp",
    )
    preclose_hold_s = float(grasp_settle_time_s)
    if not ok:
        preclose_hold_s = max(preclose_hold_s, 2.0)
    if preclose_hold_s > 0.0:
        preclose_hold_diagnostics = _hold_arm_waypoint_until_settled(
            context=context,
            waypoint=grasp_waypoint,
            duration_s=preclose_hold_s,
            tolerance_rad=0.025,
            step_callback=_step_callback,
        )
        moveit_diagnostics.update(preclose_hold_diagnostics)
        print(
            "[INFO]: Isaac grasp pre-close hold complete "
            f"settled={preclose_hold_diagnostics.get('grasp_preclose_hold_settled', 'n/a')} "
            f"duration_s={float(preclose_hold_diagnostics.get('grasp_preclose_hold_duration_s', 0.0)):.3f} "
            f"final_error={preclose_hold_diagnostics.get('grasp_preclose_hold_final_error_rad', 'n/a')}.",
            flush=True,
        )

    gripper_close_diagnostics = _command_gripper_width(
        sim=sim,
        scene=scene,
        robot=robot,
        width=float(closed_gripper_width),
        duration_s=float(gripper_close_duration_s),
        max_duration_s=float(gripper_close_max_duration_s),
        hold_context=context,
        hold_arm_waypoint=grasp_waypoint,
        settle_duration_s=GRIPPER_CLOSE_SETTLE_DURATION_S,
        min_contact_motion_m=max(
            0.001, min(0.003, 0.125 * abs(float(open_gripper_width) - float(closed_gripper_width)))
        ),
        force_joint_state=False,
        step_callback=_step_callback,
    )
    if isinstance(gripper_close_diagnostics, Mapping):
        moveit_diagnostics.update(dict(gripper_close_diagnostics))
    print(
        "[INFO]: Isaac gripper close complete "
        f"status={moveit_diagnostics.get('gripper_close_status', 'unknown')} "
        f"duration_s={float(moveit_diagnostics.get('gripper_close_duration_s', 0.0)):.3f} "
        f"final_error={moveit_diagnostics.get('gripper_close_final_max_position_error', 'n/a')}.",
        flush=True,
    )
    close_status = str(moveit_diagnostics.get("gripper_close_status", "unknown"))
    close_is_acceptable = close_status in {"target_reached", "no_hand_joints"}
    if close_status == "contact_stalled":
        matched_grasp_width = _kuka_contact_stall_matches_grasp_width(moveit_diagnostics, selected_gripper_width_m)
        close_is_acceptable = True if matched_grasp_width is None else bool(matched_grasp_width)
    elif close_status == "max_duration_elapsed":
        close_is_acceptable = bool(
            _kuka_contact_stall_matches_grasp_width(moveit_diagnostics, selected_gripper_width_m)
        )
    if not close_is_acceptable:
        return PickExecutionResult(
            False,
            "gripper_close_failed",
            "Isaac gripper did not reach the closed target or a plausible selected-grasp contact before lift: "
            f"status={close_status}, reason={moveit_diagnostics.get('gripper_close_contact_stall_accept_reason', 'n/a')}.",
            diagnostics=moveit_diagnostics,
        )
    context.fixed_gripper_width = float(closed_gripper_width)
    if float(postclose_hold_s) > 0.0:
        print(f"[INFO]: Holding closed Isaac gripper for {float(postclose_hold_s):.2f}s before lift.", flush=True)
        _hold_arm_waypoint(
            context=context,
            waypoint=grasp_waypoint,
            duration_s=float(postclose_hold_s),
            step_callback=_step_callback,
        )
    capture_lift_object_z = True
    try:
        ok, detail = _execute_moveit_waypoint_segment(
            context=context,
            executor=executor,
            moveit_joint_trajectories=moveit_joint_trajectories,
            label="lift",
        )
    finally:
        capture_lift_object_z = False
    if not ok:
        return PickExecutionResult(
            False,
            "moveit_lift_failed",
            f"MoveIt lift execution failed: {detail}",
            diagnostics=moveit_diagnostics,
        )
    lift_result = _validate_object_lift(
        object_asset=object_asset,
        initial_object_z=initial_object_z,
        success_height_margin_m=success_height_margin_m,
        observed_object_max_z=observed_lift_object_max_z,
        extra_diagnostics=moveit_diagnostics,
    )
    if lift_result is not None:
        return lift_result
    return PickExecutionResult(
        True,
        "ok",
        "MoveIt direct-pick trajectories executed in Isaac.",
        diagnostics=moveit_diagnostics,
    )
