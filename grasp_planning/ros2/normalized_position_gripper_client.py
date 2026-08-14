"""Normalized position control for the dual KUKA grippers."""

from __future__ import annotations

import time

from grasp_planning.start_poses import kuka_gripper_normalized_position_from_width

try:
    import rclpy
    from std_msgs.msg import Float64
    from std_srvs.srv import Trigger
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    Float64 = None
    Trigger = None


class NormalizedPositionGripperClient:
    """Home once, then command a clamped 0=open to 1=closed position."""

    def __init__(
        self,
        node,
        *,
        position_command_topic: str,
        position_feedback_topic: str,
        open_service_name: str,
        stop_service_name: str,
        timeout_s: float,
        feedback_tolerance: float = 0.02,
        grasp_settle_time_s: float = 0.5,
    ) -> None:
        if rclpy is None or Float64 is None or Trigger is None:
            raise RuntimeError(
                "std_msgs/Float64 and std_srvs/Trigger dependencies are unavailable. "
                "Source the ROS2 environment before enabling normalized gripper control."
            )
        names = {
            "position_command": str(position_command_topic).strip(),
            "position_feedback": str(position_feedback_topic).strip(),
            "open": str(open_service_name).strip(),
            "stop": str(stop_service_name).strip(),
        }
        missing = [label for label, name in names.items() if not name]
        if missing:
            raise ValueError(f"Normalized gripper interfaces must be non-empty: missing {missing}.")
        if float(feedback_tolerance) < 0.0:
            raise ValueError("feedback_tolerance must be non-negative.")

        self._node = node
        self._names = names
        self._timeout_s = float(timeout_s)
        self._feedback_tolerance = float(feedback_tolerance)
        self._grasp_settle_time_s = float(grasp_settle_time_s)
        self._publisher = node.create_publisher(Float64, names["position_command"], 1)
        self._subscription = node.create_subscription(
            Float64,
            names["position_feedback"],
            self._feedback_callback,
            1,
        )
        self._open_client = node.create_client(Trigger, names["open"])
        self._stop_client = node.create_client(Trigger, names["stop"])
        self._feedback_position: float | None = None
        self._last_requested_position: float | None = None

    @property
    def feedback_position(self) -> float | None:
        return self._feedback_position

    def wait_for_server(self, *, timeout_s: float) -> None:
        for label, client in (("open", self._open_client), ("stop", self._stop_client)):
            if not client.wait_for_service(timeout_sec=float(timeout_s)):
                raise RuntimeError(
                    f"Normalized gripper {label} service '{self._names[label]}' is unavailable."
                )

    def initialize_open(self) -> tuple[bool, str]:
        """Establish the controller's persistent multi-turn zero once."""

        ok, message = self._call_trigger("open", self._open_client)
        if ok:
            self._last_requested_position = 0.0
        return ok, message

    def command_width(
        self,
        width_m: float,
        *,
        wait_for_feedback: bool,
        settle_after_command: bool = False,
    ) -> tuple[bool, str]:
        normalized = kuka_gripper_normalized_position_from_width(width_m)
        ok, message = self.command_position(
            normalized,
            wait_for_feedback=wait_for_feedback,
        )
        if ok and settle_after_command:
            time.sleep(max(self._grasp_settle_time_s, 0.0))
        return ok, f"width={float(width_m):.4f} m normalized={normalized:.4f}: {message}"

    def command_position(
        self,
        position: float,
        *,
        wait_for_feedback: bool,
    ) -> tuple[bool, str]:
        normalized = max(0.0, min(1.0, float(position)))
        changed = (
            self._last_requested_position is None
            or abs(normalized - self._last_requested_position) > 1.0e-9
        )
        if changed:
            message = Float64()
            message.data = normalized
            self._publisher.publish(message)
            self._last_requested_position = normalized
        if wait_for_feedback:
            ok, feedback_message = self._wait_for_position(normalized)
            if not ok:
                stop_detail = self._best_effort_stop()
                return False, f"{feedback_message}; emergency stop result: {stop_detail}"
        else:
            feedback_message = "feedback wait skipped"
        action = "published" if changed else "unchanged; publish skipped"
        return True, f"position={normalized:.4f} {action}; {feedback_message}"

    def stop(self) -> tuple[bool, str]:
        return self._call_trigger("stop", self._stop_client)

    def _feedback_callback(self, message) -> None:
        self._feedback_position = max(0.0, min(1.0, float(message.data)))

    def _wait_for_position(self, target: float) -> tuple[bool, str]:
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            if (
                self._feedback_position is not None
                and abs(float(self._feedback_position) - float(target)) <= self._feedback_tolerance
            ):
                return True, f"feedback reached {self._feedback_position:.4f}"
            rclpy.spin_once(self._node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
        feedback = "none" if self._feedback_position is None else f"{self._feedback_position:.4f}"
        return (
            False,
            f"position feedback '{self._names['position_feedback']}' timed out after "
            f"{self._timeout_s:.1f}s (target={target:.4f}, feedback={feedback})",
        )

    def _call_trigger(self, label: str, client) -> tuple[bool, str]:
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._timeout_s)
        if not future.done():
            raise TimeoutError(
                f"Normalized gripper {label} service '{self._names[label]}' timed out after "
                f"{self._timeout_s:.1f}s"
            )
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"Normalized gripper {label} service raised {exception!r}")
        response = future.result()
        if response is None:
            return False, f"Normalized gripper {label} service returned no response."
        success = bool(getattr(response, "success", False))
        message = str(getattr(response, "message", "")).strip()
        if not message:
            message = f"Normalized gripper {label} service returned success={success}."
        return success, message

    def _best_effort_stop(self) -> str:
        try:
            ok, message = self.stop()
        except Exception as exc:
            return f"failed with {exc!r}"
        return f"success={ok}, message={message}"


__all__ = ["NormalizedPositionGripperClient"]
