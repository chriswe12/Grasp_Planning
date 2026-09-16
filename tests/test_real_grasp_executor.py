from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from grasp_planning import SavedGraspCandidate
from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.pipeline import RealExecutionConfig
from grasp_planning.ros2 import real_grasp_executor


class _FakeCommander:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.executed: list[tuple[str, object]] = []
        self.scene_states: list[dict[str, float]] = []

    def move_to_pose(self, target, *, label: str, execute: bool) -> tuple[bool, str]:
        self.calls.append((label, target.frame_id))
        return True, f"{label} ok"

    def execute_trajectory(self, trajectory, *, label: str) -> tuple[bool, str]:
        self.executed.append((label, trajectory))
        return True, f"{label} preplanned trajectory ok"

    def apply_planning_scene_robot_state(self, state) -> tuple[bool, str]:
        self.scene_states.append(dict(state))
        return True, "state ok"


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


class _FakeNormalizedPositionGripperClient:
    def __init__(
        self,
        node,
        *,
        position_command_topic: str,
        position_feedback_topic: str,
        open_service_name: str,
        close_service_name: str,
        stop_service_name: str,
        timeout_s: float,
        feedback_tolerance: float,
        grasp_settle_time_s: float,
        closed_width_m: float,
        open_width_m: float,
    ) -> None:
        self.node = node
        self.position_command_topic = position_command_topic
        self.position_feedback_topic = position_feedback_topic
        self.open_service_name = open_service_name
        self.close_service_name = close_service_name
        self.stop_service_name = stop_service_name
        self.timeout_s = timeout_s
        self.feedback_tolerance = feedback_tolerance
        self.grasp_settle_time_s = grasp_settle_time_s
        self.closed_width_m = closed_width_m
        self.open_width_m = open_width_m


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


def _policy_saved_grasp(
    grasp_id: str,
    *,
    score: float,
    refined: bool = False,
    jaw_width: float = 0.04,
) -> SavedGraspCandidate:
    source_position = (0.01, 0.02, 0.03)
    actual_position = (0.011, 0.02, 0.03) if refined else source_position
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=actual_position,
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.0, 0.0, 0.0),
        contact_point_b_obj=(0.04, 0.0, 0.0),
        contact_normal_a_obj=(-1.0, 0.0, 0.0),
        contact_normal_b_obj=(1.0, 0.0, 0.0),
        jaw_width=jaw_width,
        roll_angle_rad=0.0,
        score=score,
        metadata={"stage2_contact_refined": refined},
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


def test_execute_selected_world_grasp_uses_exact_preplanned_pregrasp_trajectory() -> None:
    commander = _FakeCommander()
    config = RealExecutionConfig(enabled=True, stop_after="pregrasp", frame_id="base", gripper_enabled=False)
    trajectory = object()

    result, steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=None,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("artifacts/test_attempt.json"),
        pregrasp_trajectory=trajectory,
    )

    assert result.success is True
    assert commander.calls == []
    assert commander.executed == [("pregrasp", trajectory)]
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
    assert gripper.calls == [("open", 0.03), ("close", 0.02)]
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
    assert gripper.calls == [("open", 0.03), ("close", 0.02)]
    assert [step["name"] for step in steps] == ["open_gripper", "pregrasp", "grasp", "close_gripper", "lift"]


def test_single_kuka_gripper_state_is_closed_before_lift_planning() -> None:
    commander = _FakeCommander()
    gripper = _FakeGripper()
    config = RealExecutionConfig(
        enabled=True,
        stop_after="full",
        frame_id="base",
        gripper_enabled=True,
        moveit_gripper_joint_name="left_finger_joint",
    )

    result, steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=gripper,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("artifacts/test_attempt.json"),
    )

    assert result.success is True
    assert commander.scene_states == [{"left_finger_joint": 0.032}]
    assert [step["name"] for step in steps] == [
        "open_gripper",
        "pregrasp",
        "grasp",
        "close_gripper",
        "apply_closed_gripper_moveit_state",
        "lift",
    ]


def test_single_kuka_open_state_matches_74mm_physical_aperture() -> None:
    config = RealExecutionConfig(moveit_gripper_joint_name="left_finger_joint")

    state = real_grasp_executor._configured_moveit_gripper_state(config=config, width_m=0.074)

    assert set(state) == {"left_finger_joint"}
    assert state["left_finger_joint"] == pytest.approx(0.005)


def test_policy_execution_queue_skips_grasps_outside_the_physical_stroke() -> None:
    too_wide = _policy_saved_grasp("g0000", score=0.95, refined=True, jaw_width=0.090)
    too_narrow = _policy_saved_grasp("g0003", score=0.92, jaw_width=0.006)
    first = _policy_saved_grasp("g0001", score=0.90, jaw_width=0.070)
    second = _policy_saved_grasp("g0002", score=0.80)
    bundle = SimpleNamespace(candidates=(too_wide, too_narrow, first, second))
    config = RealExecutionConfig(
        grasp_approach_controller="d405_policy",
        gripper_closed_width=0.007,
        gripper_open_width=0.074,
        gripper_width_clearance_m=0.01,
    )

    queue = real_grasp_executor._real_execution_candidate_queue(bundle, config=config)

    assert [candidate.grasp_id for candidate in queue] == ["g0001", "g0002"]


def test_policy_execution_queue_fails_clearly_when_no_grasp_fits_physical_gripper() -> None:
    bundle = SimpleNamespace(
        candidates=(
            _policy_saved_grasp("g0000", score=0.95, jaw_width=0.090),
            _policy_saved_grasp("g0001", score=0.90, jaw_width=0.075),
        )
    )
    config = RealExecutionConfig(
        grasp_approach_controller="d405_policy",
        gripper_closed_width=0.007,
        gripper_open_width=0.074,
        gripper_width_clearance_m=0.01,
    )

    with pytest.raises(RuntimeError, match="fits the physical gripper"):
        real_grasp_executor._real_execution_candidate_queue(bundle, config=config)


def test_moveit_pose_execution_rejects_selected_grasp_outside_physical_stroke() -> None:
    bundle = SimpleNamespace(
        candidates=(_policy_saved_grasp("g0001", score=0.9, jaw_width=0.080),)
    )
    config = RealExecutionConfig(
        grasp_approach_controller="moveit_pose",
        gripper_closed_width=0.007,
        gripper_open_width=0.074,
    )

    with pytest.raises(RuntimeError, match="does not fit the physical gripper"):
        real_grasp_executor._real_execution_candidate_queue(bundle, config=config)


def test_policy_approach_width_clamps_to_74mm_physical_aperture() -> None:
    config = RealExecutionConfig(
        grasp_approach_controller="d405_policy",
        gripper_closed_width=0.007,
        gripper_open_width=0.074,
        gripper_width_clearance_m=0.01,
    )

    width = real_grasp_executor._policy_approach_width(jaw_width_m=0.07003736962775151, config=config)

    assert width == 0.074


def test_d405_policy_replaces_only_pregrasp_to_grasp_and_then_closes() -> None:
    commander = _FakeCommander()
    gripper = _FakeGripper()
    config = RealExecutionConfig(
        enabled=True,
        stop_after="grasp",
        frame_id="base",
        gripper_enabled=True,
        grasp_approach_controller="d405_policy",
        visual_servo_config="configs/visual_servo_real_d405.yaml",
    )
    policy_result = SimpleNamespace(
        completed=True,
        state="COMPLETED_HOLD",
        message="learned completion gate satisfied",
        goal_id="runtime__part_0__g0001",
        motion_applied=True,
        allow_gripper_close=True,
        step_count=12,
        run_directory=Path("artifacts/policy-run"),
    )

    result, steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=gripper,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("attempt.json"),
        visual_servo_runner=lambda: policy_result,
    )

    assert result.success
    assert result.grasp_reached
    assert commander.calls == [("pregrasp", "base")]
    assert gripper.calls == [("open", 0.03), ("close", 0.02)]
    assert [step["name"] for step in steps] == [
        "open_gripper",
        "pregrasp",
        "d405_policy_approach",
        "close_gripper",
    ]


def test_d405_policy_dry_run_never_closes_gripper() -> None:
    commander = _FakeCommander()
    gripper = _FakeGripper()
    config = RealExecutionConfig(
        enabled=True,
        stop_after="grasp",
        frame_id="base",
        gripper_enabled=True,
        grasp_approach_controller="d405_policy",
        visual_servo_config="configs/visual_servo_real_d405.yaml",
    )
    policy_result = SimpleNamespace(
        completed=True,
        state="COMPLETED_HOLD",
        message="dry run",
        goal_id="runtime__part_0__g0001",
        motion_applied=False,
        allow_gripper_close=False,
        step_count=4,
        run_directory=Path("artifacts/policy-run"),
    )

    result, _steps = real_grasp_executor._execute_selected_world_grasp(
        commander=commander,
        gripper=gripper,
        world_grasp=_world_grasp(),
        config=config,
        attempt_artifact_path=Path("attempt.json"),
        visual_servo_runner=lambda: policy_result,
    )

    assert result.success
    assert result.status == "visual_servo_dry_run_completed"
    assert not result.grasp_reached
    assert gripper.calls == [("open", 0.03)]


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
        gripper_client="trigger_service",
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


def test_make_gripper_client_can_select_new_normalized_position_gripper() -> None:
    commander = _FakeCommander()
    config = RealExecutionConfig(
        gripper_client="servo_gripper",
        gripper_trigger_open_service="/gripper_controller/open",
        gripper_trigger_close_service="/gripper_controller/close",
        gripper_trigger_stop_service="/gripper_controller/stop",
        gripper_position_command_topic="/gripper_controller/position_command",
        gripper_position_feedback_topic="/gripper_controller/position",
        gripper_position_feedback_tolerance=0.03,
        gripper_closed_width=0.007,
        gripper_open_width=0.074,
        gripper_timeout_s=12.0,
        grasp_settle_time_s=0.3,
    )

    with mock.patch.object(
        real_grasp_executor,
        "NormalizedPositionGripperClient",
        _FakeNormalizedPositionGripperClient,
    ):
        gripper = real_grasp_executor._make_gripper_client(commander=commander, config=config)

    assert isinstance(gripper, _FakeNormalizedPositionGripperClient)
    assert gripper.position_command_topic == "/gripper_controller/position_command"
    assert gripper.position_feedback_topic == "/gripper_controller/position"
    assert gripper.open_service_name == "/gripper_controller/open"
    assert gripper.close_service_name == "/gripper_controller/close"
    assert gripper.stop_service_name == "/gripper_controller/stop"
    assert gripper.closed_width_m == 0.007
    assert gripper.open_width_m == 0.074


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
