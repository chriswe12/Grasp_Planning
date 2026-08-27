"""Normalized position and endpoint control for persistent servo grippers."""

from __future__ import annotations

import time

from grasp_planning.start_poses import (
    KUKA_Y_GRIPPER_SOURCE_CLOSED_WIDTH_M,
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
)

try:
    import rclpy
    from std_msgs.msg import Float64
    from std_srvs.srv import Trigger
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    Float64 = None
    Trigger = None


class NormalizedPositionGripperClient:
    """Command a calibrated jaw range where 0=open and 1=closed."""

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
        feedback_tolerance: float = 0.02,
        grasp_settle_time_s: float = 0.5,
        closed_width_m: float = KUKA_Y_GRIPPER_SOURCE_CLOSED_WIDTH_M,
        open_width_m: float = KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
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
            "close": str(close_service_name).strip(),
            "stop": str(stop_service_name).strip(),
        }
        missing = [label for label, name in names.items() if not name]
        if missing:
            raise ValueError(f"Normalized gripper interfaces must be non-empty: missing {missing}.")
        if float(feedback_tolerance) < 0.0:
            raise ValueError("feedback_tolerance must be non-negative.")
        if float(closed_width_m) < 0.0 or float(open_width_m) <= float(closed_width_m):
            raise ValueError("open_width_m must be greater than non-negative closed_width_m.")

        self._node = node
        self._names = names
        self._timeout_s = float(timeout_s)
        self._feedback_tolerance = float(feedback_tolerance)
        self._grasp_settle_time_s = float(grasp_settle_time_s)
        self._closed_width_m = float(closed_width_m)
        self._open_width_m = float(open_width_m)
        self._publisher = node.create_publisher(Float64, names["position_command"], 1)
        self._subscription = node.create_subscription(
            Float64,
            names["position_feedback"],
            self._feedback_callback,
            1,
        )
        self._open_client = node.create_client(Trigger, names["open"])
        self._close_client = node.create_client(Trigger, names["close"])
        self._stop_client = node.create_client(Trigger, names["stop"])
        self._feedback_position: float | None = None
        self._last_requested_position: float | None = None

    @property
    def feedback_position(self) -> float | None:
        return self._feedback_position

    def wait_for_server(self, *, timeout_s: float) -> None:
        for label, client in (
            ("open", self._open_client),
            ("close", self._close_client),
            ("stop", self._stop_client),
        ):
            if not client.wait_for_service(timeout_sec=float(timeout_s)):
                raise RuntimeError(
                    f"Normalized gripper {label} service '{self._names[label]}' is unavailable."
                )

    def initialize_open(self) -> tuple[bool, str]:
        """Open an already calibrated persistent controller."""

        ok, message = self._command_endpoint("open", self._open_client)
        if ok:
            self._last_requested_position = 0.0
        return ok, message

    def open(self, *, width: float) -> tuple[bool, str]:
        """Fully open before approach using the acknowledged endpoint service.

        The persistent gripper exposes arbitrary position commands, but position
        feedback is optional.  Hardware pickup must not fail merely because the
        optional feedback topic is absent, so the approach uses the service that
        reports command completion and leaves width commands available through
        :meth:`command_width` for callers that explicitly need them.
        """

        requested_width = self._clamp_width(width)
        ok, message = self.initialize_open()
        return (
            ok,
            f"open endpoint={self._open_width_m:.4f} m "
            f"planned_approach={requested_width:.4f} m: {message}",
        )

    def close(self, *, width: float) -> tuple[bool, str]:
        """Close toward the calibrated endpoint, allowing motor contact/stall."""

        requested_width = self._clamp_width(width)
        ok, message = self._command_endpoint("close", self._close_client)
        if ok:
            self._last_requested_position = 1.0
            time.sleep(max(self._grasp_settle_time_s, 0.0))
        return (
            ok,
            f"close endpoint={self._closed_width_m:.4f} m "
            f"planned_contact={requested_width:.4f} m: {message}",
        )

    def command_width(
        self,
        width_m: float,
        *,
        wait_for_feedback: bool,
        settle_after_command: bool = False,
    ) -> tuple[bool, str]:
        commanded_width = self._clamp_width(width_m)
        normalized = self._closure_fraction_from_width(commanded_width)
        ok, message = self.command_position(
            normalized,
            wait_for_feedback=wait_for_feedback,
        )
        if ok and settle_after_command:
            time.sleep(max(self._grasp_settle_time_s, 0.0))
        return (
            ok,
            f"requested_width={float(width_m):.4f} m commanded_width={commanded_width:.4f} m "
            f"closure_fraction={normalized:.4f}: {message}",
        )

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

    def _clamp_width(self, width_m: float) -> float:
        return max(self._closed_width_m, min(self._open_width_m, float(width_m)))

    def _closure_fraction_from_width(self, width_m: float) -> float:
        span = self._open_width_m - self._closed_width_m
        return (self._open_width_m - self._clamp_width(width_m)) / span

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

    def _command_endpoint(self, label: str, client) -> tuple[bool, str]:
        try:
            ok, message = self._call_trigger(label, client)
        except TimeoutError as exc:
            stop_detail = self._best_effort_stop()
            error = TimeoutError(f"{exc}; emergency stop result: {stop_detail}")
            error.gripper_stop_attempted = True
            raise error from None
        if ok:
            return True, message
        stop_detail = self._best_effort_stop()
        return False, f"{message}; emergency stop result: {stop_detail}"

    def _best_effort_stop(self) -> str:
        try:
            ok, message = self.stop()
        except Exception as exc:
            return f"failed with {exc!r}"
        return f"success={ok}, message={message}"


__all__ = ["NormalizedPositionGripperClient"]
