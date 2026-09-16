from __future__ import annotations

from types import SimpleNamespace

import pytest

from grasp_planning.ros2 import trigger_service_gripper_client


class _Trigger:
    class Request:
        pass


class _Future:
    def __init__(self, response=None, *, done: bool = True, exception=None) -> None:
        self._response = response
        self._done = done
        self._exception = exception

    def done(self) -> bool:
        return self._done

    def exception(self):
        return self._exception

    def result(self):
        return self._response


class _ServiceClient:
    def __init__(self, futures: list[_Future], *, available: bool = True) -> None:
        self._futures = list(futures)
        self.available = available
        self.wait_timeouts: list[float] = []
        self.requests: list[object] = []

    def wait_for_service(self, *, timeout_sec: float) -> bool:
        self.wait_timeouts.append(timeout_sec)
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        return self._futures.pop(0)


class _Node:
    def __init__(self, clients: dict[str, _ServiceClient]) -> None:
        self.clients = clients
        self.created: list[tuple[object, str]] = []

    def create_client(self, service_type, service_name: str):
        self.created.append((service_type, service_name))
        return self.clients[service_name]


def _response(success: bool, message: str):
    return SimpleNamespace(success=success, message=message)


def _make_client(monkeypatch, *, open_futures=None, close_futures=None, stop_futures=None):
    clients = {
        "/gripper_controller/open": _ServiceClient(open_futures or [_Future(_response(True, "opened"))]),
        "/gripper_controller/close": _ServiceClient(close_futures or [_Future(_response(True, "closed"))]),
        "/gripper_controller/stop": _ServiceClient(stop_futures or [_Future(_response(True, "stopped"))]),
    }
    node = _Node(clients)
    monkeypatch.setattr(trigger_service_gripper_client, "Trigger", _Trigger)
    monkeypatch.setattr(
        trigger_service_gripper_client,
        "rclpy",
        SimpleNamespace(spin_until_future_complete=lambda node, future, timeout_sec: None),
    )
    client = trigger_service_gripper_client.TriggerServiceGripperClient(
        node,
        open_service_name="/gripper_controller/open",
        close_service_name="/gripper_controller/close",
        stop_service_name="/gripper_controller/stop",
        timeout_s=12.0,
        grasp_settle_time_s=0.0,
    )
    return client, clients


def test_trigger_service_gripper_calls_open_close_and_stop(monkeypatch) -> None:
    client, clients = _make_client(monkeypatch)

    client.wait_for_server(timeout_s=3.0)
    assert client.open(width=0.08) == (True, "opened")
    assert client.close(width=0.02) == (True, "closed")
    assert client.stop() == (True, "stopped")

    assert all(service.wait_timeouts == [3.0] for service in clients.values())
    assert all(len(service.requests) == 1 for service in clients.values())
    assert all(isinstance(service.requests[0], _Trigger.Request) for service in clients.values())


def test_trigger_service_gripper_stops_after_command_timeout(monkeypatch) -> None:
    client, clients = _make_client(
        monkeypatch,
        open_futures=[_Future(done=False)],
        stop_futures=[_Future(_response(True, "motor stopped"))],
    )

    with pytest.raises(TimeoutError, match="emergency stop result: success=True, message=motor stopped"):
        client.open(width=0.08)

    assert len(clients["/gripper_controller/stop"].requests) == 1


def test_trigger_service_gripper_requires_emergency_stop_service(monkeypatch) -> None:
    client, clients = _make_client(monkeypatch)
    clients["/gripper_controller/stop"].available = False

    with pytest.raises(RuntimeError, match="stop service.*unavailable"):
        client.wait_for_server(timeout_s=2.0)
