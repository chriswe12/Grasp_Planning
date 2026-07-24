"""Generic control_msgs GripperCommand action wrapper for real execution."""

from __future__ import annotations

import time

from grasp_planning.start_poses import gripper_joint_target_from_width

try:
    import rclpy
    from control_msgs.action import GripperCommand
    from rclpy.action import ActionClient
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    GripperCommand = None
    ActionClient = None


class GripperCommandClient:
    """Synchronous wrapper around control_msgs/action/GripperCommand."""

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
        if rclpy is None or GripperCommand is None or ActionClient is None:
            raise RuntimeError(
                "control_msgs GripperCommand dependencies are unavailable. "
                "Source the ROS2 gripper-control workspace before enabling this gripper client."
            )
        if not str(action_name).strip():
            raise ValueError("real_execution.gripper_command_action must be set for gripper_client='gripper_command'.")

        self._node = node
        self._timeout_s = float(timeout_s)
        self._max_effort = float(max_effort)
        self._position_mode = _normalize_position_mode(position_mode)
        self._grasp_settle_time_s = float(grasp_settle_time_s)
        self._client = ActionClient(node, GripperCommand, str(action_name))

    def wait_for_server(self, *, timeout_s: float) -> None:
        if not self._client.wait_for_server(timeout_sec=float(timeout_s)):
            raise RuntimeError("GripperCommand action is unavailable.")

    def open(self, *, width: float) -> tuple[bool, str]:
        position = self._position_from_width(width)
        return self._send_position(position=position, label="open", accept_stalled=False)

    def close(self, *, width: float) -> tuple[bool, str]:
        position = self._position_from_width(width)
        ok, message = self._send_position(position=position, label="close", accept_stalled=True)
        time.sleep(max(self._grasp_settle_time_s, 0.0))
        return ok, message

    def _position_from_width(self, width: float) -> float:
        if self._position_mode == "width":
            return max(float(width), 0.0)
        if self._position_mode == "kuka_y_finger_joint":
            return float(gripper_joint_target_from_width("left_finger_joint", float(width)))
        raise AssertionError(f"Unhandled gripper position mode '{self._position_mode}'.")

    def _send_position(self, *, position: float, label: str, accept_stalled: bool) -> tuple[bool, str]:
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(self._max_effort)

        send_future = self._client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_future, label=f"gripper {label} goal", timeout_s=5.0)
        if goal_handle is None or not goal_handle.accepted:
            return False, f"GripperCommand {label} goal was rejected."

        result_future = goal_handle.get_result_async()
        result_wrapper = self._wait_for_future(
            result_future,
            label=f"gripper {label} result",
            timeout_s=self._timeout_s,
        )
        result = result_wrapper.result
        reached_goal = getattr(result, "reached_goal", None)
        stalled = bool(getattr(result, "stalled", False))
        if reached_goal is False and not (accept_stalled and stalled):
            return False, f"GripperCommand {label} did not reach position={position:.4f}."
        if accept_stalled and stalled and reached_goal is False:
            return True, f"GripperCommand {label} stalled after contact at position={position:.4f}."
        return True, f"GripperCommand {label} reached position={position:.4f}."

    def _wait_for_future(self, future, *, label: str, timeout_s: float):
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=float(timeout_s))
        if not future.done():
            raise TimeoutError(f"{label} timed out after {timeout_s:.1f}s")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"{label} raised {exception!r}")
        return future.result()


def _normalize_position_mode(position_mode: str) -> str:
    normalized = str(position_mode).strip().lower().replace("-", "_")
    aliases = {
        "": "width",
        "jaw_width": "width",
        "opening_width": "width",
        "width": "width",
        "kuka": "kuka_y_finger_joint",
        "kuka_y": "kuka_y_finger_joint",
        "kuka_y_gripper": "kuka_y_finger_joint",
        "kuka_y_finger": "kuka_y_finger_joint",
        "kuka_y_finger_joint": "kuka_y_finger_joint",
        "lbr": "kuka_y_finger_joint",
        "lbr_iiwa7_y_gripper": "kuka_y_finger_joint",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported gripper_command_position_mode '{position_mode}'. Expected one of: width, kuka_y_finger_joint."
        ) from exc
