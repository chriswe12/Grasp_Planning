"""Dry-run and MoveIt Servo Cartesian command adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

try:  # pragma: no cover - exercised only in a sourced ROS2 environment
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Int8
    from std_srvs.srv import Trigger
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    TwistStamped = None
    qos_profile_sensor_data = None
    Int8 = None
    Trigger = None

MOVEIT_SERVO_STATUS = {
    -1: "invalid",
    0: "no_warning",
    1: "decelerate_approaching_singularity",
    2: "halt_singularity",
    3: "decelerate_collision",
    4: "halt_collision",
    5: "halt_joint_bound",
    6: "decelerate_leaving_singularity",
}
MOVEIT_SERVO_HALT_CODES = frozenset({-1, 2, 4, 5})


@dataclass(frozen=True)
class CommandSinkHealth:
    healthy: bool
    consumer_exists: bool
    status_code: int | None
    status_text: str
    status_age_s: float | None


class DryRunCommandSink:
    is_real = False

    def __init__(self) -> None:
        self.active = False
        self.commands: list[dict[str, object]] = []

    def activate(self, *, timeout_s: float) -> None:
        del timeout_s
        self.active = True

    def preflight(self, *, timeout_s: float) -> None:
        del timeout_s

    def send_twist(
        self,
        values: Sequence[float],
        *,
        frame_id: str,
        stamp_s: float,
    ) -> bool:
        command = tuple(float(value) for value in values)
        if len(command) != 6:
            raise ValueError("A Cartesian twist must contain six values.")
        self.commands.append({"stamp_s": float(stamp_s), "frame_id": str(frame_id), "twist": command})
        return self.active

    def hold(self, *, frame_id: str, stamp_s: float) -> None:
        self.send_twist((0.0,) * 6, frame_id=frame_id, stamp_s=stamp_s)

    def health(self, *, now_s: float) -> CommandSinkHealth:
        del now_s
        return CommandSinkHealth(
            healthy=self.active,
            consumer_exists=True,
            status_code=0,
            status_text="dry_run",
            status_age_s=0.0,
        )

    def deactivate(self, *, timeout_s: float) -> None:
        del timeout_s
        self.active = False


class MoveItServoCommandSink:
    """Publish speed-unit twists to an existing collision-checking Servo node."""

    is_real = True

    def __init__(
        self,
        node,
        *,
        twist_topic: str,
        status_topic: str,
        start_service: str,
        stop_service: str,
    ) -> None:
        if any(value is None for value in (rclpy, TwistStamped, Int8, Trigger)):
            raise RuntimeError("ROS2 geometry/std/std_srvs dependencies are required for MoveIt Servo.")
        self._node = node
        self._publisher = node.create_publisher(TwistStamped, str(twist_topic), qos_profile_sensor_data)
        self._status_subscription = node.create_subscription(
            Int8,
            str(status_topic),
            self._on_status,
            qos_profile_sensor_data,
        )
        self._start_client = node.create_client(Trigger, str(start_service))
        self._stop_client = node.create_client(Trigger, str(stop_service))
        self._status_code: int | None = None
        self._status_receipt_s: float | None = None
        self._last_frame_id = ""
        self.active = False

    def _on_status(self, message) -> None:
        self._status_code = int(message.data)
        self._status_receipt_s = float(self._node.get_clock().now().nanoseconds) * 1.0e-9

    def activate(self, *, timeout_s: float) -> None:
        if not self._start_client.wait_for_service(timeout_sec=float(timeout_s)):
            raise RuntimeError("MoveIt Servo start service is unavailable.")
        future = self._start_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=float(timeout_s))
        if not future.done():
            raise TimeoutError("MoveIt Servo start service timed out.")
        if future.exception() is not None:
            raise RuntimeError(f"MoveIt Servo start service raised {future.exception()!r}.")
        response = future.result()
        if response is None or not bool(response.success):
            message = "no response" if response is None else str(response.message)
            raise RuntimeError(f"MoveIt Servo refused activation: {message}")
        self.active = True

    def preflight(self, *, timeout_s: float) -> None:
        """Verify the Servo graph without starting Servo or publishing motion."""

        timeout = float(timeout_s)
        if not self._start_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("MoveIt Servo start service is unavailable.")
        if not self._stop_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("MoveIt Servo stop service is unavailable.")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._publisher.get_subscription_count() >= 1:
                return
            rclpy.spin_once(self._node, timeout_sec=0.02)
        raise RuntimeError("MoveIt Servo twist consumer is unavailable.")

    def wait_until_healthy(self, *, timeout_s: float, frame_id: str) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            now_s = float(self._node.get_clock().now().nanoseconds) * 1.0e-9
            self.hold(frame_id=frame_id, stamp_s=now_s)
            rclpy.spin_once(self._node, timeout_sec=0.02)
            health = self.health(now_s=now_s)
            if health.healthy and health.consumer_exists and health.status_code is not None:
                return
        raise TimeoutError("MoveIt Servo did not publish a healthy status before the startup timeout.")

    def send_twist(
        self,
        values: Sequence[float],
        *,
        frame_id: str,
        stamp_s: float,
    ) -> bool:
        command = tuple(float(value) for value in values)
        if len(command) != 6:
            raise ValueError("A Cartesian twist must contain six values.")
        if not self.active or self._publisher.get_subscription_count() < 1:
            return False
        if not str(frame_id).strip():
            return False
        message = TwistStamped()
        nanoseconds = int(round(float(stamp_s) * 1.0e9))
        message.header.stamp.sec = nanoseconds // 1_000_000_000
        message.header.stamp.nanosec = nanoseconds % 1_000_000_000
        message.header.frame_id = str(frame_id)
        self._last_frame_id = str(frame_id)
        (
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.linear.z,
            message.twist.angular.x,
            message.twist.angular.y,
            message.twist.angular.z,
        ) = command
        self._publisher.publish(message)
        return True

    def hold(self, *, frame_id: str, stamp_s: float) -> None:
        for _ in range(4):
            self.send_twist((0.0,) * 6, frame_id=frame_id, stamp_s=stamp_s)

    def health(self, *, now_s: float) -> CommandSinkHealth:
        consumer_exists = self._publisher.get_subscription_count() >= 1
        age = None if self._status_receipt_s is None else max(0.0, float(now_s) - self._status_receipt_s)
        status_text = "unavailable" if self._status_code is None else MOVEIT_SERVO_STATUS.get(
            self._status_code,
            f"unknown_{self._status_code}",
        )
        healthy = self.active and consumer_exists and self._status_code not in MOVEIT_SERVO_HALT_CODES
        return CommandSinkHealth(
            healthy=healthy,
            consumer_exists=consumer_exists,
            status_code=self._status_code,
            status_text=status_text,
            status_age_s=age,
        )

    def deactivate(self, *, timeout_s: float) -> None:
        now_s = float(self._node.get_clock().now().nanoseconds) * 1.0e-9
        if self._last_frame_id:
            self.hold(frame_id=self._last_frame_id, stamp_s=now_s)
        if self._stop_client.wait_for_service(timeout_sec=min(float(timeout_s), 1.0)):
            future = self._stop_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=float(timeout_s))
        self.active = False


__all__ = [
    "CommandSinkHealth",
    "DryRunCommandSink",
    "MOVEIT_SERVO_HALT_CODES",
    "MOVEIT_SERVO_STATUS",
    "MoveItServoCommandSink",
]
