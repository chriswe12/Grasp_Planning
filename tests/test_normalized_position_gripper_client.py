from __future__ import annotations

from types import SimpleNamespace

from grasp_planning.ros2 import normalized_position_gripper_client as gripper_module
from grasp_planning.ros2.normalized_position_gripper_client import (
    NormalizedPositionGripperClient,
)
from grasp_planning.start_poses import (
    kuka_gripper_approach_width,
    kuka_gripper_normalized_position_from_width,
)


class _Float64:
    def __init__(self) -> None:
        self.data = 0.0


class _Trigger:
    class Request:
        pass


class _Future:
    def __init__(self, response) -> None:
        self._response = response

    def done(self) -> bool:
        return True

    def exception(self):
        return None

    def result(self):
        return self._response


class _Client:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def wait_for_service(self, *, timeout_sec: float) -> bool:
        del timeout_sec
        return True

    def call_async(self, request):
        del request
        self.calls += 1
        return _Future(SimpleNamespace(success=True, message=self.message))


class _Publisher:
    def __init__(self) -> None:
        self.values: list[float] = []

    def publish(self, message) -> None:
        self.values.append(float(message.data))


class _Node:
    def __init__(self) -> None:
        self.publisher = _Publisher()
        self.clients: dict[str, _Client] = {}
        self.feedback_callback = None

    def create_publisher(self, message_type, topic: str, qos: int):
        del message_type, topic, qos
        return self.publisher

    def create_subscription(self, message_type, topic: str, callback, qos: int):
        del message_type, topic, qos
        self.feedback_callback = callback
        return object()

    def create_client(self, service_type, service_name: str):
        del service_type
        client = _Client(service_name)
        self.clients[service_name] = client
        return client


def _client(monkeypatch) -> tuple[NormalizedPositionGripperClient, _Node]:
    monkeypatch.setattr(gripper_module, "Float64", _Float64)
    monkeypatch.setattr(gripper_module, "Trigger", _Trigger)
    monkeypatch.setattr(
        gripper_module,
        "rclpy",
        SimpleNamespace(
            spin_until_future_complete=lambda node, future, timeout_sec: None,
            spin_once=lambda node, timeout_sec: None,
        ),
    )
    node = _Node()
    return (
        NormalizedPositionGripperClient(
            node,
            position_command_topic="/robot/gripper/position_command",
            position_feedback_topic="/robot/gripper/position",
            open_service_name="/robot/gripper/open",
            stop_service_name="/robot/gripper/stop",
            timeout_s=0.01,
            feedback_tolerance=0.02,
            grasp_settle_time_s=0.0,
        ),
        node,
    )


def test_initializes_open_zero_once_and_skips_unchanged_commands(monkeypatch) -> None:
    client, node = _client(monkeypatch)

    assert client.initialize_open() == (True, "/robot/gripper/open")
    assert node.clients["/robot/gripper/open"].calls == 1
    assert client.command_position(0.0, wait_for_feedback=False)[0] is True
    assert node.publisher.values == []

    assert client.command_position(1.5, wait_for_feedback=False)[0] is True
    assert client.command_position(1.0, wait_for_feedback=False)[0] is True
    assert node.publisher.values == [1.0]


def test_candidate_width_maps_to_normalized_partially_closed_position(monkeypatch) -> None:
    client, node = _client(monkeypatch)
    jaw_width_m = 0.040
    approach_width_m = kuka_gripper_approach_width(jaw_width_m)

    assert client.command_width(approach_width_m, wait_for_feedback=False)[0] is True
    assert client.command_width(jaw_width_m, wait_for_feedback=False)[0] is True

    assert node.publisher.values == [
        kuka_gripper_normalized_position_from_width(approach_width_m),
        kuka_gripper_normalized_position_from_width(jaw_width_m),
    ]
    assert 0.0 < node.publisher.values[0] < node.publisher.values[1] < 1.0
