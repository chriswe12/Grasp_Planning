"""Trajectory execution for planned FR3 arm paths."""

from __future__ import annotations

from typing import Callable

import torch

from .fr3_motion_context import FR3MotionContext
from .types import JointTrajectory


class TrajectoryExecutor:
    """Execute a discrete arm trajectory as one streamed joint reference."""

    _QUINTIC_SMOOTHSTEP_MAX_SLOPE = 1.875

    def __init__(
        self,
        context: FR3MotionContext,
        waypoint_tolerance_rad: float = 0.025,
        velocity_tolerance_rad_s: float = 0.05,
        max_steps_per_waypoint: int | None = None,
        max_joint_speed_rad_s: float = 0.35,
        final_settle_steps: int = 60,
        step_callback: Callable[[], None] | None = None,
    ) -> None:
        self._context = context
        self._waypoint_tolerance_rad = float(waypoint_tolerance_rad)
        self._velocity_tolerance_rad_s = float(velocity_tolerance_rad_s)
        self._max_steps_per_waypoint = None if max_steps_per_waypoint is None else int(max_steps_per_waypoint)
        self._max_joint_speed_rad_s = float(max_joint_speed_rad_s)
        self._final_settle_steps = int(final_settle_steps)
        self._step_callback = step_callback
        if self._max_joint_speed_rad_s <= 0.0:
            raise ValueError("max_joint_speed_rad_s must be positive.")
        if self._velocity_tolerance_rad_s < 0.0:
            raise ValueError("velocity_tolerance_rad_s must be non-negative.")
        if self._max_steps_per_waypoint is not None and self._max_steps_per_waypoint < 1:
            raise ValueError("max_steps_per_waypoint must be at least 1 when set.")
        if self._final_settle_steps < 1:
            raise ValueError("final_settle_steps must be at least 1.")

    def _command_arm(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        self._context.command_arm(q)
        command_velocity = getattr(self._context, "command_arm_velocity", None)
        if command_velocity is not None:
            command_velocity(qd)

    def _arm_speed(self) -> float | None:
        get_arm_qd = getattr(self._context, "get_arm_qd", None)
        if get_arm_qd is None:
            return None
        return float(torch.max(torch.abs(get_arm_qd())).item())

    def _max_error_detail(self, actual: torch.Tensor, command: torch.Tensor, target: torch.Tensor) -> str:
        target_error = torch.abs(actual - target)
        command_error = torch.abs(actual - command)
        flat_target_error = target_error.reshape(-1)
        max_index = int(torch.argmax(flat_target_error).item())
        joint_names = tuple(getattr(self._context, "arm_joint_names", ()))
        joint_name = joint_names[max_index] if max_index < len(joint_names) else f"joint[{max_index}]"
        actual_value = float(actual.reshape(-1)[max_index].item())
        command_value = float(command.reshape(-1)[max_index].item())
        target_value = float(target.reshape(-1)[max_index].item())
        command_error_value = float(command_error.reshape(-1)[max_index].item())
        return (
            f"max_joint={joint_name} actual={actual_value:.4f} "
            f"command={command_value:.4f} target={target_value:.4f} "
            f"command_error={command_error_value:.4f}"
        )

    def execute(self, trajectory: JointTrajectory) -> tuple[bool, str]:
        if not trajectory.waypoints:
            return True, "ok"

        # MoveIt points describe one collision-checked polyline. Parameterize
        # that whole path once so intermediate points do not become artificial
        # stop-and-settle commands. The max-norm segment length keeps every
        # commanded joint velocity within the configured scalar speed limit.
        points = [self._context.get_arm_q()]
        for waypoint in trajectory.waypoints:
            candidate = waypoint.clone()
            if float(torch.max(torch.abs(candidate - points[-1])).item()) > 1.0e-9:
                points.append(candidate)

        if len(points) > 1:
            deltas = [end - start for start, end in zip(points[:-1], points[1:], strict=True)]
            lengths = [float(torch.max(torch.abs(delta)).item()) for delta in deltas]
            total_length = float(sum(lengths))
            dt = max(float(trajectory.dt), 1.0e-6)
            required_steps = max(
                1,
                int(
                    torch.ceil(
                        torch.tensor(
                            self._QUINTIC_SMOOTHSTEP_MAX_SLOPE * total_length / (self._max_joint_speed_rad_s * dt)
                        )
                    ).item()
                ),
            )
            if self._max_steps_per_waypoint is not None:
                maximum_steps = self._max_steps_per_waypoint * len(deltas)
                if required_steps > maximum_steps:
                    return (
                        False,
                        f"trajectory needs {required_steps} stream steps to respect the joint-speed limit, "
                        f"exceeding {maximum_steps} for {len(deltas)} path segments",
                    )
            duration_s = max(float(required_steps) * dt, dt)
            segment_index = 0
            segment_start_distance = 0.0
            for step_idx in range(1, required_steps + 1):
                alpha = float(step_idx) / float(required_steps)
                smooth_alpha = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
                smooth_slope = 30.0 * alpha**2 * (1.0 - alpha) ** 2
                distance = smooth_alpha * total_length
                while segment_index < len(lengths) - 1 and distance > segment_start_distance + lengths[segment_index]:
                    segment_start_distance += lengths[segment_index]
                    segment_index += 1
                segment_length = lengths[segment_index]
                local_alpha = min(
                    1.0,
                    max(
                        0.0,
                        (distance - segment_start_distance) / max(segment_length, 1.0e-12),
                    ),
                )
                q_cmd = (points[segment_index] + local_alpha * deltas[segment_index]).clone()
                path_speed = smooth_slope * total_length / duration_s
                qd_cmd = (path_speed * deltas[segment_index] / max(segment_length, 1.0e-12)).clone()
                self._command_arm(q_cmd, qd_cmd)
                self._context.command_fixed_gripper()
                self._context.scene.write_data_to_sim()
                self._context.sim.step()
                self._context.scene.update(self._context.physics_dt)
                if self._step_callback is not None:
                    self._step_callback()
                if step_idx == 1 or step_idx % 60 == 0:
                    actual = self._context.get_arm_q()
                    final_waypoint = trajectory.waypoints[-1]
                    error = torch.max(torch.abs(actual - final_waypoint))
                    print(
                        "[INFO]: Executor streaming continuous path "
                        f"segment={segment_index + 1}/{len(deltas)} "
                        f"step={step_idx}/{required_steps} path_progress={smooth_alpha:.2f} "
                        f"final_target_error={float(error.item()):.4f} "
                        f"{self._max_error_detail(actual, q_cmd, final_waypoint)}",
                        flush=True,
                    )
        final_waypoint = trajectory.waypoints[-1]
        zero_velocity = torch.zeros_like(final_waypoint)
        last_error = None
        last_speed = None
        for settle_step in range(1, self._final_settle_steps + 1):
            self._command_arm(final_waypoint, zero_velocity)
            self._context.command_fixed_gripper()
            self._context.scene.write_data_to_sim()
            self._context.sim.step()
            self._context.scene.update(self._context.physics_dt)
            if self._step_callback is not None:
                self._step_callback()
            actual = self._context.get_arm_q()
            error = torch.max(torch.abs(actual - final_waypoint))
            last_error = float(error.item())
            last_speed = self._arm_speed()
            if settle_step == 1 or settle_step == self._final_settle_steps:
                print(
                    "[INFO]: Executor final settle "
                    f"step={settle_step}/{self._final_settle_steps} max_joint_error={last_error:.4f} "
                    f"max_joint_speed={last_speed if last_speed is not None else 'n/a'} "
                    f"{self._max_error_detail(actual, final_waypoint, final_waypoint)}",
                    flush=True,
                )
            velocity_settled = last_speed is None or last_speed <= self._velocity_tolerance_rad_s
            if last_error <= self._waypoint_tolerance_rad and velocity_settled:
                return True, "ok"
        speed_detail = "n/a" if last_speed is None else f"{last_speed:.4f}"
        return (
            False,
            f"final waypoint did not settle; last_max_joint_error={last_error:.4f} last_max_joint_speed={speed_detail}",
        )
