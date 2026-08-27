from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def bridge(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend("ros2_ws/src/robot_integration_ros")
    return importlib.import_module("robot_integration_ros.gripper_joint_state_bridge")


def test_pdz_physical_feedback_maps_to_prefixed_dual_driver(bridge) -> None:
    open_state = bridge.gripper_driver_state_from_closure_fraction(
        layout="dual",
        side="left",
        gripper_model="pdz_gripper",
        closure_fraction=0.0,
    )
    closed_state = bridge.gripper_driver_state_from_closure_fraction(
        layout="dual",
        side="right",
        gripper_model="pdz_gripper",
        closure_fraction=1.0,
    )

    assert open_state == ("lbr_one_pdz_gripper_left_finger_joint", pytest.approx(0.031))
    assert closed_state == ("lbr_two_pdz_gripper_left_finger_joint", 0.0)


def test_feedback_fallback_uses_full_modeled_open_collision_envelope(bridge) -> None:
    assert bridge.modeled_open_driver_state(
        layout="single",
        side="left",
        gripper_model="pdz_gripper",
    ) == ("pdz_gripper_left_finger_joint", 0.032)
    assert bridge.modeled_open_driver_state(
        layout="dual",
        side="right",
        gripper_model="y_gripper",
    ) == ("lbr_two_left_finger_joint", 0.0)


def test_stale_feedback_remains_latched_after_motion_finishes(bridge) -> None:
    node = object.__new__(bridge.GripperJointStateBridge)
    node._layout = "dual"
    node._gripper_model = "pdz_gripper"
    node._physical_sides = {"left"}
    node._feedback = {"left": (0.5, 0.0)}
    node._feedback_stale_warning_s = 1.0
    node._warn_throttled = lambda *_args: None

    assert node._state_for_side("left", now=100.0) == (
        "lbr_one_pdz_gripper_left_finger_joint",
        pytest.approx(0.01425),
    )
