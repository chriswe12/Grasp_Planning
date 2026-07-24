from __future__ import annotations

from pathlib import Path
from unittest import mock

from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.pipeline import RealExecutionConfig
from grasp_planning.ros2 import real_grasp_executor


class _FakeCommander:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def move_to_pose(self, target, *, label: str, execute: bool) -> tuple[bool, str]:
        self.calls.append((label, target.frame_id))
        return True, f"{label} ok"


class _FakeGripper:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def open(self, *, width: float) -> tuple[bool, str]:
        self.calls.append(("open", width))
        return True, "open ok"

    def close(self, *, width: float) -> tuple[bool, str]:
        self.calls.append(("close", width))
        return True, "close ok"


class _FakeGripperCommandClient:
    def __init__(
        self,
        node,
        *,
        action_name: str,
        timeout_s: float,
        max_effort: float,
        position_mode: str,
        grasp_settle_time_s: float,
    ) -> None:
        self.node = node
        self.action_name = action_name
        self.timeout_s = timeout_s
        self.max_effort = max_effort
        self.position_mode = position_mode
        self.grasp_settle_time_s = grasp_settle_time_s


class _FakeTriggerServiceGripperClient:
    def __init__(
        self,
        node,
        *,
        open_service_name: str,
        close_service_name: str,
        stop_service_name: str,
        timeout_s: float,
        grasp_settle_time_s: float,
    ) -> None:
        self.node = node
        self.open_service_name = open_service_name
        self.close_service_name = close_service_name
        self.stop_service_name = stop_service_name
        self.timeout_s = timeout_s
        self.grasp_settle_time_s = grasp_settle_time_s


def _world_grasp() -> WorldFrameGraspCandidate:
    return WorldFrameGraspCandidate(
        grasp_id="g0001",
        position_w=(0.4, 0.0, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        normal_w=(0.0, 0.0, 1.0),
        pregrasp_offset=0.1,
        pregrasp_position_w=(0.4, 0.0, 0.1),
        gripper_width=0.03,
        jaw_width=0.02,
        roll_angle_rad=0.0,
        contact_point_a_w=(0.39, 0.0, 0.2),
        contact_point_b_w=(0.41, 0.0, 0.2),
    )


def test_execute_selected_world_grasp_stops_at_pregrasp() -> None:
    commander = _FakeCommander()
    config = RealExecutionConfig(enabled=True, stop_after="pregrasp", frame_id="base", gripper_enabled=False)

    result, steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=None,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("artifacts/test_attempt.json"),
    )

    assert result.success is True
    assert result.status == "stopped_at_pregrasp"
    assert result.pregrasp_reached is True
    assert result.grasp_reached is False
    assert commander.calls == [("pregrasp", "base")]
    assert [step["name"] for step in steps] == ["pregrasp"]


def test_execute_selected_world_grasp_stops_after_closing_at_grasp() -> None:
    commander = _FakeCommander()
    gripper = _FakeGripper()
    config = RealExecutionConfig(enabled=True, stop_after="grasp", frame_id="base", gripper_enabled=True)

    result, steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=gripper,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("artifacts/test_attempt.json"),
    )

    assert result.success is True
    assert result.status == "stopped_at_grasp"
    assert result.pregrasp_reached is True
    assert result.grasp_reached is True
    assert result.lift_reached is False
    assert commander.calls == [("pregrasp", "base"), ("grasp", "base")]
    assert gripper.calls == [("open", 0.08), ("close", 0.02)]
    assert [step["name"] for step in steps] == ["open_gripper", "pregrasp", "grasp", "close_gripper"]


def test_execute_selected_world_grasp_runs_full_sequence_with_gripper() -> None:
    commander = _FakeCommander()
    gripper = _FakeGripper()
    config = RealExecutionConfig(enabled=True, stop_after="full", frame_id="base", gripper_enabled=True)

    result, steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=gripper,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("artifacts/test_attempt.json"),
    )

    assert result.success is True
    assert result.status == "completed"
    assert result.pregrasp_reached is True
    assert result.grasp_reached is True
    assert result.lift_reached is True
    assert commander.calls == [("pregrasp", "base"), ("grasp", "base"), ("lift", "base")]
    assert gripper.calls == [("open", 0.08), ("close", 0.02)]
    assert [step["name"] for step in steps] == ["open_gripper", "pregrasp", "grasp", "close_gripper", "lift"]


def test_make_gripper_client_can_select_generic_gripper_command_client() -> None:
    commander = _FakeCommander()
    config = RealExecutionConfig(
        gripper_client="control_msgs",
        gripper_command_action="/hand/gripper_cmd",
        gripper_command_position_mode="kuka_y_gripper",
        gripper_command_max_effort=12.5,
        gripper_timeout_s=3.0,
        grasp_settle_time_s=0.2,
    )

    with mock.patch.object(real_grasp_executor, "GripperCommandClient", _FakeGripperCommandClient):
        gripper = real_grasp_executor._make_gripper_client(commander=commander, config=config)

    assert isinstance(gripper, _FakeGripperCommandClient)
    assert gripper.node is commander
    assert gripper.action_name == "/hand/gripper_cmd"
    assert gripper.timeout_s == 3.0
    assert gripper.max_effort == 12.5
    assert gripper.position_mode == "kuka_y_gripper"
    assert gripper.grasp_settle_time_s == 0.2


def test_make_gripper_client_can_select_trigger_service_client() -> None:
    commander = _FakeCommander()
    config = RealExecutionConfig(
        gripper_client="servo_gripper",
        gripper_trigger_open_service="/hand/open",
        gripper_trigger_close_service="/hand/close",
        gripper_trigger_stop_service="/hand/stop",
        gripper_timeout_s=12.0,
        grasp_settle_time_s=0.3,
    )

    with mock.patch.object(
        real_grasp_executor,
        "TriggerServiceGripperClient",
        _FakeTriggerServiceGripperClient,
    ):
        gripper = real_grasp_executor._make_gripper_client(commander=commander, config=config)

    assert isinstance(gripper, _FakeTriggerServiceGripperClient)
    assert gripper.node is commander
    assert gripper.open_service_name == "/hand/open"
    assert gripper.close_service_name == "/hand/close"
    assert gripper.stop_service_name == "/hand/stop"
    assert gripper.timeout_s == 12.0
    assert gripper.grasp_settle_time_s == 0.3


def test_best_effort_stop_gripper_calls_supported_client() -> None:
    gripper = mock.Mock()
    gripper.stop.return_value = (True, "stopped")

    real_grasp_executor._best_effort_stop_gripper(gripper, reason="test")

    gripper.stop.assert_called_once_with()


def test_normalize_gripper_client_rejects_unsupported_client() -> None:
    try:
        real_grasp_executor._normalize_gripper_client("unsupported")
    except ValueError as exc:
        assert "Unsupported real_execution.gripper_client" in str(exc)
    else:
        raise AssertionError("Expected unsupported gripper client to fail.")
