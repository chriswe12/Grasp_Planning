"""Shared scene defaults used by launcher and local debug tools."""

from __future__ import annotations

ROBOT_BASE_POSITION = (0.0, 0.0, 0.0)
# Isaac Lab initial-state rotations use wxyz order. This rotates the robot 180 deg from the previous visual pose.
ROBOT_BASE_ORIENTATION_XYZW = (1.0, 0.0, 0.0, 0.0)
CUBE_POSITION = (0.45, 0.0, 0.025)
CUBE_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
