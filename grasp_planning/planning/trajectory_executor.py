"""Trajectory execution for planned FR3 arm paths."""

from __future__ import annotations

import torch

from .fr3_motion_context import FR3MotionContext
from .types import JointTrajectory


class TrajectoryExecutor:
    """Execute a discrete arm trajectory as one streamed joint reference."""

    def __init__(
        self,
        context: FR3MotionContext,
        waypoint_tolerance_rad: float = 0.025,
        max_steps_per_waypoint: int = 300,
        max_joint_speed_rad_s: float = 0.35,
        final_settle_steps: int = 60,
    ) -> None:
        self._context = context
        self._waypoint_tolerance_rad = float(waypoint_tolerance_rad)
        self._max_steps_per_waypoint = int(max_steps_per_waypoint)
        self._max_joint_speed_rad_s = float(max_joint_speed_rad_s)
        self._final_settle_steps = int(final_settle_steps)
        if self._max_joint_speed_rad_s <= 0.0:
            raise ValueError("max_joint_speed_rad_s must be positive.")
        if self._final_settle_steps < 1:
            raise ValueError("final_settle_steps must be at least 1.")

    def execute(self, trajectory: JointTrajectory) -> tuple[bool, str]:
        if not trajectory.waypoints:
            return True, "ok"

        q_ref = self._context.get_arm_q()
        total_stream_steps = 0
        for waypoint_index, waypoint in enumerate(trajectory.waypoints, start=1):
            start_ref = q_ref.clone()
            max_delta = float(torch.max(torch.abs(waypoint - start_ref)).item())
            max_delta_per_step = self._max_joint_speed_rad_s * max(float(trajectory.dt), 1.0e-6)
            segment_steps = max(1, int(torch.ceil(torch.tensor(max_delta / max_delta_per_step)).item()))
            segment_steps = min(segment_steps, self._max_steps_per_waypoint)
            for step_idx in range(1, segment_steps + 1):
                alpha = float(step_idx) / float(segment_steps)
                smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                q_cmd = ((1.0 - smooth_alpha) * start_ref + smooth_alpha * waypoint).clone()
                self._context.command_arm(q_cmd)
                self._context.command_fixed_gripper()
                self._context.scene.write_data_to_sim()
                self._context.sim.step()
                self._context.scene.update(self._context.physics_dt)
                total_stream_steps += 1
                if total_stream_steps == 1 or total_stream_steps % 60 == 0:
                    error = torch.max(torch.abs(self._context.get_arm_q() - waypoint))
                    print(
                        "[INFO]: Executor streaming "
                        f"waypoint={waypoint_index}/{len(trajectory.waypoints)} "
                        f"step={total_stream_steps} segment_progress={alpha:.2f} "
                        f"target_error={float(error.item()):.4f}",
                        flush=True,
                    )
            q_ref = waypoint.clone()

        final_waypoint = trajectory.waypoints[-1]
        last_error = None
        for settle_step in range(1, self._final_settle_steps + 1):
            self._context.command_arm(final_waypoint)
            self._context.command_fixed_gripper()
            self._context.scene.write_data_to_sim()
            self._context.sim.step()
            self._context.scene.update(self._context.physics_dt)
            error = torch.max(torch.abs(self._context.get_arm_q() - final_waypoint))
            last_error = float(error.item())
            if settle_step == 1 or settle_step == self._final_settle_steps:
                print(
                    "[INFO]: Executor final settle "
                    f"step={settle_step}/{self._final_settle_steps} max_joint_error={last_error:.4f}",
                    flush=True,
                )
            if last_error <= self._waypoint_tolerance_rad:
                return True, "ok"
        return False, f"final waypoint did not settle; last_max_joint_error={last_error:.4f}"
