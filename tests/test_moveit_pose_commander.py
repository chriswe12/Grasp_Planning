from __future__ import annotations

import math
from argparse import Namespace

import numpy as np

from grasp_planning.ros2.moveit_pose_commander import (
    DEFAULT_FR3_MOVEIT_RPY,
    PoseTarget,
    normalize_quaternion_xyzw,
    quaternion_from_rpy,
)
from scripts.move_real_robot_ee import commander_config_from_args, pose_target_from_args


def test_normalize_quaternion_xyzw_returns_unit_quaternion() -> None:
    quaternion = normalize_quaternion_xyzw((0.0, 0.0, 0.0, 2.0))
    np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0), atol=1.0e-9)


def test_quaternion_from_rpy_identity() -> None:
    quaternion = quaternion_from_rpy(0.0, 0.0, 0.0)
    np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0), atol=1.0e-9)


def test_pose_target_from_rpy_normalizes_orientation() -> None:
    target = PoseTarget.from_rpy(x=0.1, y=0.2, z=0.3, roll=math.pi, pitch=0.0, yaw=math.pi / 2.0, frame_id="base")
    norm = math.sqrt(sum(component * component for component in target.orientation_xyzw))
    assert math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9)


def test_pose_target_from_args_uses_default_thesis_orientation() -> None:
    args = Namespace(
        x=0.30,
        y=0.01,
        z=0.40,
        frame_id="base",
        keep_current_orientation=False,
        roll=None,
        pitch=None,
        yaw=None,
        qx=None,
        qy=None,
        qz=None,
        qw=None,
    )

    target = pose_target_from_args(args)

    expected = PoseTarget.from_rpy(
        x=0.30,
        y=0.01,
        z=0.40,
        roll=DEFAULT_FR3_MOVEIT_RPY[0],
        pitch=DEFAULT_FR3_MOVEIT_RPY[1],
        yaw=DEFAULT_FR3_MOVEIT_RPY[2],
        frame_id="base",
    )
    np.testing.assert_allclose(target.orientation_xyzw, expected.orientation_xyzw, atol=1.0e-9)


def test_pose_target_from_args_prefers_quaternion_when_provided() -> None:
    args = Namespace(
        x=0.10,
        y=-0.20,
        z=0.30,
        frame_id="map",
        keep_current_orientation=False,
        roll=0.1,
        pitch=0.2,
        yaw=0.3,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=2.0,
    )

    target = pose_target_from_args(args)

    np.testing.assert_allclose(target.orientation_xyzw, (0.0, 0.0, 0.0, 1.0), atol=1.0e-9)
    assert target.frame_id == "map"


def test_pose_target_from_args_rejects_partial_quaternion() -> None:
    args = Namespace(
        x=0.10,
        y=-0.20,
        z=0.30,
        frame_id="base",
        keep_current_orientation=False,
        roll=None,
        pitch=None,
        yaw=None,
        qx=0.0,
        qy=None,
        qz=0.0,
        qw=1.0,
    )

    try:
        pose_target_from_args(args)
    except ValueError as exc:
        assert "provide all of --qx --qy --qz --qw" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected pose_target_from_args to reject partial quaternion inputs.")


def test_pose_target_from_args_uses_current_orientation_when_requested() -> None:
    args = Namespace(
        x=0.40,
        y=0.05,
        z=0.25,
        frame_id="base",
        keep_current_orientation=True,
        roll=None,
        pitch=None,
        yaw=None,
        qx=None,
        qy=None,
        qz=None,
        qw=None,
    )

    target = pose_target_from_args(args, current_orientation_xyzw=(0.0, 0.0, 0.70710678, 0.70710678))

    np.testing.assert_allclose(target.orientation_xyzw, (0.0, 0.0, 0.70710678, 0.70710678), atol=1.0e-8)


def test_pose_target_from_args_rejects_keep_current_orientation_with_explicit_orientation() -> None:
    args = Namespace(
        x=0.40,
        y=0.05,
        z=0.25,
        frame_id="base",
        keep_current_orientation=True,
        roll=None,
        pitch=None,
        yaw=0.5,
        qx=None,
        qy=None,
        qz=None,
        qw=None,
    )

    try:
        pose_target_from_args(args, current_orientation_xyzw=(0.0, 0.0, 0.0, 1.0))
    except ValueError as exc:
        assert "--keep-current-orientation cannot be combined" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected pose_target_from_args to reject mixed explicit and current-orientation inputs.")


def test_commander_config_from_args_uses_slow_defaults() -> None:
    args = Namespace(
        planning_group="fr3_arm",
        pose_link="fr3_hand_tcp",
        planner_id="",
        wait_for_moveit_timeout_s=15.0,
        ik_timeout_s=2.0,
        planning_time_s=5.0,
        num_planning_attempts=5,
        velocity_scale=0.05,
        acceleration_scale=0.05,
        execute_timeout_s=120.0,
        post_execute_sleep_s=0.5,
        allow_collisions=False,
    )

    config = commander_config_from_args(args)

    assert math.isclose(config.velocity_scale, 0.05)
    assert math.isclose(config.acceleration_scale, 0.05)
