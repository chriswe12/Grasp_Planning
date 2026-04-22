#!/usr/bin/env python3
"""Compatibility shim for the ROS2 workspace EE mover package."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PACKAGE_ROOT = REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros"
if str(WORKSPACE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_PACKAGE_ROOT))

from robot_integration_ros.move_real_robot_ee import (  # noqa: E402
    build_argument_parser,
    commander_config_from_args,
    main,
    pose_target_from_args,
)

__all__ = [
    "build_argument_parser",
    "commander_config_from_args",
    "main",
    "pose_target_from_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
