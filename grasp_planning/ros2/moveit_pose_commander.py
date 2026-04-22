"""Compatibility wrapper around the ROS2 workspace package implementation."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_PACKAGE_ROOT = _REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros"
if str(_WORKSPACE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_PACKAGE_ROOT))

from robot_integration_ros.moveit_pose_commander import (  # noqa: E402
    DEFAULT_FR3_MOVEIT_RPY,
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    PoseTarget,
    normalize_quaternion_xyzw,
    quaternion_from_rpy,
    rclpy,
)

__all__ = [
    "DEFAULT_FR3_MOVEIT_RPY",
    "MoveItPoseCommander",
    "MoveItPoseCommanderConfig",
    "PoseTarget",
    "normalize_quaternion_xyzw",
    "quaternion_from_rpy",
    "rclpy",
]
