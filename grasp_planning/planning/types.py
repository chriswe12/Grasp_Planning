"""Shared motion-planning datatypes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PoseCommand:
    """Target end-effector pose in the world frame."""

    position_w: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class JointTrajectory:
    """Discrete joint-space trajectory."""

    waypoints: list[torch.Tensor]
    dt: float


@dataclass(frozen=True)
class PlanResult:
    """Outcome of a move-to-pose request."""

    success: bool
    status: str
    message: str
    trajectory: JointTrajectory | None = None
    goal_q: torch.Tensor | None = None
