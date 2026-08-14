from __future__ import annotations

from grasp_planning.ros2.mock_gripper_client import MockGripperClient


class _FakeCommander:
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def apply_robot_state_joint_positions(self, joint_positions):
        payload = dict(joint_positions)
        self.calls.append(payload)
        return True, f"applied {payload}"


def test_wait_for_server_does_not_block_or_raise() -> None:
    client = MockGripperClient(_FakeCommander(), finger_joint_name="lbr_one_left_finger_joint", grasp_settle_time_s=0.0)

    client.wait_for_server(timeout_s=0.001)  # must return immediately, no service to wait for


def test_open_sets_finger_to_half_the_configured_open_width() -> None:
    commander = _FakeCommander()
    client = MockGripperClient(
        commander,
        finger_joint_name="lbr_one_left_finger_joint",
        grasp_settle_time_s=0.0,
        open_width_m=0.08,
    )

    ok, message = client.open(width=0.5)  # width is ignored for "open", like the real Trigger client

    assert ok is True
    assert "Mock gripper open" in message
    assert commander.calls == [{"lbr_one_left_finger_joint": 0.04}]


def test_close_sets_finger_to_half_the_requested_jaw_width_and_settles(monkeypatch) -> None:
    commander = _FakeCommander()
    sleeps: list[float] = []
    monkeypatch.setattr(
        "grasp_planning.ros2.mock_gripper_client.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    client = MockGripperClient(
        commander,
        finger_joint_name="lbr_two_left_finger_joint",
        grasp_settle_time_s=0.5,
    )

    ok, message = client.close(width=0.036)

    assert ok is True
    assert "Mock gripper close" in message
    assert commander.calls == [{"lbr_two_left_finger_joint": 0.018}]
    assert sleeps == [0.5]


def test_close_clamps_negative_width_to_zero() -> None:
    commander = _FakeCommander()
    client = MockGripperClient(commander, finger_joint_name="lbr_one_left_finger_joint", grasp_settle_time_s=0.0)

    client.close(width=-0.01)

    assert commander.calls == [{"lbr_one_left_finger_joint": 0.0}]


def test_close_does_not_settle_when_the_commander_reports_failure() -> None:
    class _FailingCommander(_FakeCommander):
        def apply_robot_state_joint_positions(self, joint_positions):
            super().apply_robot_state_joint_positions(joint_positions)
            return False, "planning scene rejected the update"

    commander = _FailingCommander()
    client = MockGripperClient(commander, finger_joint_name="lbr_one_left_finger_joint", grasp_settle_time_s=100.0)

    ok, message = client.close(width=0.04)

    assert ok is False
    assert "planning scene rejected the update" in message


def test_stop_is_a_no_op_success() -> None:
    commander = _FakeCommander()
    client = MockGripperClient(commander, finger_joint_name="lbr_one_left_finger_joint", grasp_settle_time_s=0.0)

    ok, message = client.stop()

    assert ok is True
    assert commander.calls == []
    assert "no-op" in message
