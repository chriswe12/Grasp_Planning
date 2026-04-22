"""Optional Franka gripper action wrapper for real-robot execution."""

from __future__ import annotations

import time

try:
    import rclpy
    from franka_msgs.action import Grasp
    from franka_msgs.action import Move as GripperMove
    from rclpy.action import ActionClient
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    Grasp = None
    GripperMove = None
    ActionClient = None


class FrankaGripperClient:
    """Small synchronous gripper wrapper sharing the MoveIt node event loop."""

    def __init__(
        self,
        node,
        *,
        grasp_action_name: str,
        move_action_name: str,
        timeout_s: float,
        grasp_speed: float,
        grasp_force: float,
        epsilon_inner: float,
        epsilon_outer: float,
        grasp_settle_time_s: float,
    ) -> None:
        if rclpy is None or Grasp is None or GripperMove is None or ActionClient is None:
            raise RuntimeError(
                "Franka gripper dependencies are unavailable. Source the ROS2 / Franka workspace before enabling gripper execution."
            )

        self._node = node
        self._timeout_s = float(timeout_s)
        self._grasp_speed = float(grasp_speed)
        self._grasp_force = float(grasp_force)
        self._epsilon_inner = float(epsilon_inner)
        self._epsilon_outer = float(epsilon_outer)
        self._grasp_settle_time_s = float(grasp_settle_time_s)
        self._grasp_client = ActionClient(node, Grasp, str(grasp_action_name))
        self._move_client = ActionClient(node, GripperMove, str(move_action_name))

    def wait_for_server(self, *, timeout_s: float) -> None:
        if not self._move_client.wait_for_server(timeout_sec=float(timeout_s)):
            raise RuntimeError("Franka gripper move action is unavailable.")
        if not self._grasp_client.wait_for_server(timeout_sec=float(timeout_s)):
            raise RuntimeError("Franka gripper grasp action is unavailable.")

    def open(self, *, width: float) -> tuple[bool, str]:
        goal = GripperMove.Goal()
        goal.width = float(width)
        goal.speed = float(self._grasp_speed)

        send_future = self._move_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, label="gripper open goal", timeout_s=5.0)
        if goal_handle is None or not goal_handle.accepted:
            return False, "Gripper open goal was rejected."

        result_future = goal_handle.get_result_async()
        result_wrapper = self._wait_for_future(result_future, label="gripper open result", timeout_s=self._timeout_s)
        result = result_wrapper.result
        if not bool(getattr(result, "success", False)):
            return False, "Franka gripper failed to open."
        return True, f"Opened gripper to width={width:.4f}."

    def close(self, *, width: float) -> tuple[bool, str]:
        goal = Grasp.Goal()
        goal.width = float(width)
        goal.speed = float(self._grasp_speed)
        goal.force = float(self._grasp_force)
        goal.epsilon.inner = float(self._epsilon_inner)
        goal.epsilon.outer = float(self._epsilon_outer)

        send_future = self._grasp_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, label="gripper close goal", timeout_s=5.0)
        if goal_handle is None or not goal_handle.accepted:
            return False, "Gripper close goal was rejected."

        result_future = goal_handle.get_result_async()
        result_wrapper = self._wait_for_future(result_future, label="gripper close result", timeout_s=self._timeout_s)
        result = result_wrapper.result
        time.sleep(max(self._grasp_settle_time_s, 0.0))

        if bool(getattr(result, "success", False)):
            return True, f"Closed gripper toward width={width:.4f}."
        error_text = str(getattr(result, "error", "unknown"))
        return True, f"Closed gripper with warning: {error_text}"

    def _wait_for_future(self, future, *, label: str, timeout_s: float):
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=float(timeout_s))
        if not future.done():
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"{label} raised {exception!r}")
        return future.result()
