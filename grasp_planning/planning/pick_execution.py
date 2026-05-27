"""Shared FR3 pick-execution helpers for debug grasp attempts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

from grasp_planning.planning.fr3_motion_context import FR3MotionContext
from grasp_planning.start_poses import DEFAULT_ARM_START_JOINT_POS, DEFAULT_HAND_OPEN_WIDTH

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


def drive_robot_to_start_pose(sim, scene, *, step_callback: StepCallback | None = None) -> None:
    """Actively settle the FR3 into a safe home pose before planning."""

    robot = scene["robot"]
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_joint_names = tuple(DEFAULT_ARM_START_JOINT_POS.keys())
    arm_joint_ids = [joint_name_to_idx[name] for name in arm_joint_names]
    arm_targets = torch.tensor(
        [[DEFAULT_ARM_START_JOINT_POS[name] for name in arm_joint_names]], dtype=torch.float32, device=robot.device
    )
    hand_joint_names = tuple(
        name for name in robot.joint_names if name.startswith(("panda_finger_joint", "fr3_finger_joint"))
    )
    hand_target = float(DEFAULT_HAND_OPEN_WIDTH)
    physics_dt = sim.get_physics_dt()
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]

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
        if step_callback is not None:
            step_callback()


def _command_gripper_width(
    *,
    sim,
    scene,
    robot,
    width: float,
    duration_s: float,
    step_callback: StepCallback | None = None,
) -> None:
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot.joint_names)}
    hand_joint_names = tuple(
        name for name in robot.joint_names if name.startswith(("panda_finger_joint", "fr3_finger_joint"))
    )
    if not hand_joint_names:
        return
    hand_joint_ids = [joint_name_to_idx[name] for name in hand_joint_names]
    physics_dt = sim.get_physics_dt()
    steps = max(1, int(duration_s / physics_dt))
    hand_targets = torch.full(
        (1, len(hand_joint_ids)),
        float(width),
        dtype=torch.float32,
        device=robot.device,
    )
    for _ in range(steps):
        robot.set_joint_position_target(hand_targets, joint_ids=hand_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)
        if step_callback is not None:
            step_callback()


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
    step_callback: StepCallback | None = None,
) -> PickExecutionResult:
    """Execute MoveIt-planned direct-pick joint waypoints inside Isaac."""

    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=float(open_gripper_width),
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

    moveit_diagnostics = {
        "open_gripper_width_m": float(open_gripper_width),
        "closed_gripper_width_m": float(closed_gripper_width),
        "nominal_max_open_gripper_width_m": float(DEFAULT_HAND_OPEN_WIDTH),
        "open_gripper_width_exceeds_nominal_limit": float(open_gripper_width) > float(DEFAULT_HAND_OPEN_WIDTH) + 1.0e-6,
        "max_joint_speed_rad_s": float(max_joint_speed_rad_s),
        "grasp_settle_time_s": float(grasp_settle_time_s),
    }
    executor_kwargs = {
        "max_joint_speed_rad_s": float(max_joint_speed_rad_s),
        "step_callback": _step_callback,
    }
    executor = TrajectoryExecutor(context, **executor_kwargs)

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
    if not ok:
        return PickExecutionResult(
            False,
            "moveit_grasp_failed",
            f"MoveIt grasp execution failed: {detail}",
            diagnostics=moveit_diagnostics,
        )
    if grasp_settle_time_s > 0.0:
        grasp_waypoint = _moveit_waypoint_tensor(
            context=context,
            moveit_joint_trajectories=moveit_joint_trajectories,
            label="grasp",
        )
        _hold_arm_waypoint(
            context=context,
            waypoint=grasp_waypoint,
            duration_s=float(grasp_settle_time_s),
            step_callback=_step_callback,
        )

    _command_gripper_width(
        sim=sim,
        scene=scene,
        robot=robot,
        width=float(closed_gripper_width),
        duration_s=1.2,
        step_callback=_step_callback,
    )
    context.fixed_gripper_width = float(closed_gripper_width)
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
    return PickExecutionResult(True, "ok", "MoveIt direct-pick trajectories executed in Isaac.")
