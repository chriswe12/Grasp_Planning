"""Trajectory execution for planned FR3 arm paths."""

from __future__ import annotations

import torch

from .fr3_motion_context import FR3MotionContext
from .types import JointTrajectory


class TrajectoryExecutor:
    """Execute a discrete arm trajectory waypoint by waypoint."""

    def __init__(
        self,
        context: FR3MotionContext,
        waypoint_tolerance_rad: float = 0.01,
        max_steps_per_waypoint: int = 300,
    ) -> None:
        self._context = context
        self._waypoint_tolerance_rad = float(waypoint_tolerance_rad)
        self._max_steps_per_waypoint = int(max_steps_per_waypoint)

    def execute(self, trajectory: JointTrajectory) -> tuple[bool, str]:
        for waypoint_index, waypoint in enumerate(trajectory.waypoints, start=1):
            ok, detail = self._drive_to_waypoint(waypoint, waypoint_index, len(trajectory.waypoints))
            if not ok:
                return False, detail
        return True, "ok"

    def _drive_to_waypoint(self, waypoint: torch.Tensor, waypoint_index: int, waypoint_count: int) -> tuple[bool, str]:
        last_error = None
        for step_idx in range(1, self._max_steps_per_waypoint + 1):
            self._context.command_arm(waypoint)
            self._context.command_fixed_gripper()
            self._context.scene.write_data_to_sim()
            self._context.sim.step()
            self._context.scene.update(self._context.physics_dt)
            error = torch.max(torch.abs(self._context.get_arm_q() - waypoint))
            last_error = float(error.item())
            if step_idx == 1 or step_idx % 30 == 0:
                print(
                    "[INFO]: Executor waypoint "
                    f"{waypoint_index}/{waypoint_count} step={step_idx} max_joint_error={last_error:.4f}",
                    flush=True,
                )
            if last_error <= self._waypoint_tolerance_rad:
                return True, f"waypoint {waypoint_index}/{waypoint_count} converged"
        return False, (
            f"waypoint {waypoint_index}/{waypoint_count} did not converge within "
            f"{self._max_steps_per_waypoint} steps; last_max_joint_error={last_error:.4f}"
        )
