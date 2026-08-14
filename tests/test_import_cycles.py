"""Regression tests for import ordering around cartesian_waypoint_ik.py.

`grasp_planning/ros2/__init__.py` eagerly imports `real_grasp_executor`,
which needs `cartesian_waypoint_ik.resolve_ik`. `cartesian_waypoint_ik.py`
in turn imports `PoseTarget` from `grasp_planning.ros2.moveit_pose_commander`,
which forces that same `grasp_planning.ros2` package `__init__` to run. If
`real_grasp_executor.py` imports `cartesian_waypoint_ik` at module level,
importing `grasp_planning.pipeline.cartesian_waypoint_ik` *first*, before
anything has triggered `grasp_planning.ros2.__init__`, re-enters that
still-initializing module and fails with an `ImportError` about a partially
initialized module. Whichever module happened to be imported first in a
given process hid this for a while: only some import orders trigger it, so
a regression here would not necessarily show up just because the rest of
the suite passes in one process. These tests each start a fresh
interpreter to pin the failing order down.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_import(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cartesian_waypoint_ik_imports_first_in_a_fresh_interpreter() -> None:
    result = _run_import("from grasp_planning.pipeline.cartesian_waypoint_ik import IK_STRATEGIES, resolve_ik")
    assert result.returncode == 0, result.stderr


def test_ros2_package_imports_first_in_a_fresh_interpreter() -> None:
    result = _run_import("from grasp_planning.ros2 import real_grasp_executor")
    assert result.returncode == 0, result.stderr


def test_dual_real_grasp_executor_imports_first_in_a_fresh_interpreter() -> None:
    result = _run_import("from grasp_planning.ros2.dual_real_grasp_executor import DualRealExecutionConfig")
    assert result.returncode == 0, result.stderr


def test_isolated_arm_preflight_imports_first_in_a_fresh_interpreter() -> None:
    result = _run_import(
        "from grasp_planning.pipeline.dual_robot_isolated_arm_preflight import compare_candidate_arm_isolation"
    )
    assert result.returncode == 0, result.stderr
