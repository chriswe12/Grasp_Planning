"""Blocking std_srvs/Trigger gripper wrapper for real-robot execution."""

from __future__ import annotations

import time

try:
    import rclpy
    from std_srvs.srv import Trigger
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    Trigger = None


class TriggerServiceGripperClient:
    """Control an endpoint-driven gripper through blocking Trigger services."""

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
        if rclpy is None or Trigger is None:
            raise RuntimeError(
                "std_srvs/Trigger dependencies are unavailable. Source the ROS2 Humble environment "
                "before enabling the Trigger-service gripper client."
            )
        service_names = {
            "open": str(open_service_name).strip(),
            "close": str(close_service_name).strip(),
            "stop": str(stop_service_name).strip(),
        }
        missing = [label for label, name in service_names.items() if not name]
        if missing:
            raise ValueError(f"Trigger gripper service names must be non-empty: missing {missing}.")

        self._node = node
        self._timeout_s = float(timeout_s)
        self._grasp_settle_time_s = float(grasp_settle_time_s)
        self._service_names = service_names
        self._clients = {
            label: node.create_client(Trigger, service_name) for label, service_name in service_names.items()
        }

    def wait_for_server(self, *, timeout_s: float) -> None:
        for label in ("open", "close", "stop"):
            if not self._clients[label].wait_for_service(timeout_sec=float(timeout_s)):
                raise RuntimeError(f"Trigger gripper {label} service '{self._service_names[label]}' is unavailable.")

    def open(self, *, width: float) -> tuple[bool, str]:
        del width
        return self._command("open")

    def close(self, *, width: float) -> tuple[bool, str]:
        del width
        ok, message = self._command("close")
        if ok:
            time.sleep(max(self._grasp_settle_time_s, 0.0))
        return ok, message

    def stop(self) -> tuple[bool, str]:
        return self._call("stop")

    def _command(self, label: str) -> tuple[bool, str]:
        try:
            ok, message = self._call(label)
        except TimeoutError as exc:
            stop_detail = self._best_effort_stop()
            error = TimeoutError(f"{exc}; emergency stop result: {stop_detail}")
            error.gripper_stop_attempted = True
            raise error from None
        if ok:
            return True, message
        stop_detail = self._best_effort_stop()
        return False, f"{message}; emergency stop result: {stop_detail}"

    def _call(self, label: str) -> tuple[bool, str]:
        future = self._clients[label].call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._timeout_s)
        if not future.done():
            raise TimeoutError(
                f"Trigger gripper {label} service '{self._service_names[label]}' timed out after {self._timeout_s:.1f}s"
            )
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"Trigger gripper {label} service raised {exception!r}")
        response = future.result()
        if response is None:
            return False, f"Trigger gripper {label} service returned no response."
        success = bool(getattr(response, "success", False))
        message = str(getattr(response, "message", "")).strip()
        if not message:
            message = f"Trigger gripper {label} service returned success={success}."
        return success, message

    def _best_effort_stop(self) -> str:
        try:
            ok, message = self.stop()
        except Exception as exc:
            return f"failed with {exc!r}"
        return f"success={ok}, message={message}"


__all__ = ["TriggerServiceGripperClient"]
