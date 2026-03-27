"""Simple joint-space path planning."""

from __future__ import annotations

import math

import torch

from .collision_checker import CollisionChecker
from .types import JointTrajectory


class JointPathPlanner:
    """Plan a collision-checked joint interpolation."""

    def __init__(
        self,
        collision_checker: CollisionChecker,
        max_joint_step_rad: float = 0.06,
    ) -> None:
        self._collision_checker = collision_checker
        self._max_joint_step_rad = float(max_joint_step_rad)

    def plan(self, q_start: torch.Tensor, q_goal: torch.Tensor, dt: float) -> tuple[JointTrajectory | None, str]:
        max_delta = float(torch.max(torch.abs(q_goal - q_start)).item())
        num_segments = max(1, int(math.ceil(max((max_delta - 1.0e-6), 0.0) / self._max_joint_step_rad)))
        valid, reason = self._collision_checker.is_edge_valid(q_start, q_goal, num_checks=max(4, num_segments))
        if not valid:
            return None, reason

        waypoints = []
        for step_idx in range(1, num_segments + 1):
            alpha = float(step_idx) / float(num_segments)
            q_interp = ((1.0 - alpha) * q_start + alpha * q_goal).clone()
            waypoints.append(q_interp)
        return JointTrajectory(waypoints=waypoints, dt=dt), "ok"
