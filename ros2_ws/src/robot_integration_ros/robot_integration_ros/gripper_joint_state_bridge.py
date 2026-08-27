"""Mirror persistent servo-gripper feedback into MoveIt's robot state."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

_SIDES = ("left", "right")
_DUAL_ROBOTS = {"left": "lbr_one", "right": "lbr_two"}
_PHYSICAL_CLOSED_WIDTH_M = 0.007
_PHYSICAL_OPEN_WIDTH_M = 0.074
_Y_GRIPPER_OPEN_WIDTH_M = 0.084
_Y_GRIPPER_TRAVEL_M = 0.040
_PDZ_GRIPPER_CLOSED_WIDTH_M = 0.012
_PDZ_GRIPPER_TRAVEL_M = 0.032
_PDZ_GRIPPER_OPEN_WIDTH_M = _PDZ_GRIPPER_CLOSED_WIDTH_M + 2.0 * _PDZ_GRIPPER_TRAVEL_M


def gripper_driver_state_from_width(
    *,
    layout: str,
    side: str,
    gripper_model: str,
    width_m: float,
) -> tuple[str, float]:
    """Return the passive MoveIt driver joint and coordinate for one jaw width."""

    if layout not in {"single", "dual"}:
        raise ValueError("layout must be 'single' or 'dual'.")
    if side not in _SIDES:
        raise ValueError("side must be 'left' or 'right'.")
    if gripper_model not in {"pdz_gripper", "y_gripper"}:
        raise ValueError("gripper_model must be 'pdz_gripper' or 'y_gripper'.")
    if not math.isfinite(float(width_m)):
        raise ValueError("width_m must be finite.")

    source_joint = (
        "pdz_gripper_left_finger_joint"
        if gripper_model == "pdz_gripper"
        else "left_finger_joint"
    )
    prefix = f"{_DUAL_ROBOTS[side]}_" if layout == "dual" else ""
    if gripper_model == "pdz_gripper":
        position = max(
            0.0,
            min(
                _PDZ_GRIPPER_TRAVEL_M,
                0.5 * (float(width_m) - _PDZ_GRIPPER_CLOSED_WIDTH_M),
            ),
        )
    else:
        position = max(
            0.0,
            min(
                _Y_GRIPPER_TRAVEL_M,
                0.5 * (_Y_GRIPPER_OPEN_WIDTH_M - float(width_m)),
            ),
        )
    return f"{prefix}{source_joint}", position


def gripper_driver_state_from_closure_fraction(
    *,
    layout: str,
    side: str,
    gripper_model: str,
    closure_fraction: float,
) -> tuple[str, float]:
    """Map physical feedback (0=open, 1=closed) into the selected MoveIt model."""

    if not math.isfinite(float(closure_fraction)):
        raise ValueError("closure_fraction must be finite.")
    normalized = max(0.0, min(1.0, float(closure_fraction)))
    width_m = _PHYSICAL_OPEN_WIDTH_M - normalized * (
        _PHYSICAL_OPEN_WIDTH_M - _PHYSICAL_CLOSED_WIDTH_M
    )
    return gripper_driver_state_from_width(
        layout=layout,
        side=side,
        gripper_model=gripper_model,
        width_m=width_m,
    )


def modeled_open_driver_state(
    *,
    layout: str,
    side: str,
    gripper_model: str,
) -> tuple[str, float]:
    """Return the fully-open model state used as a conservative feedback fallback."""

    open_width_m = (
        _PDZ_GRIPPER_OPEN_WIDTH_M
        if gripper_model == "pdz_gripper"
        else _Y_GRIPPER_OPEN_WIDTH_M
    )
    return gripper_driver_state_from_width(
        layout=layout,
        side=side,
        gripper_model=gripper_model,
        width_m=open_width_m,
    )


class GripperJointStateBridge(Node):
    """Publish current passive finger joints beside the arm joint broadcaster."""

    def __init__(self) -> None:
        super().__init__("gripper_joint_state_bridge")
        self.declare_parameter("layout", "dual")
        self.declare_parameter("gripper_model", "pdz_gripper")
        self.declare_parameter("physical_sides", "")
        self.declare_parameter("single_side", "left")
        self.declare_parameter("left_feedback_topic", "/left/gripper_controller/position")
        self.declare_parameter("right_feedback_topic", "/right/gripper_controller/position")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("feedback_stale_warning_s", 1.0)

        self._layout = str(self.get_parameter("layout").value).strip()
        self._gripper_model = str(self.get_parameter("gripper_model").value).strip()
        self._single_side = str(self.get_parameter("single_side").value).strip()
        physical_sides = {
            value.strip()
            for value in str(self.get_parameter("physical_sides").value).split(",")
            if value.strip()
        }
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._feedback_stale_warning_s = float(
            self.get_parameter("feedback_stale_warning_s").value
        )

        if self._layout not in {"single", "dual"}:
            raise ValueError("layout must be 'single' or 'dual'.")
        if self._gripper_model not in {"pdz_gripper", "y_gripper"}:
            raise ValueError("gripper_model must be 'pdz_gripper' or 'y_gripper'.")
        if self._single_side not in _SIDES:
            raise ValueError("single_side must be 'left' or 'right'.")
        if not physical_sides.issubset(_SIDES):
            raise ValueError("physical_sides may contain only left and right.")
        if self._layout == "single" and physical_sides - {self._single_side}:
            raise ValueError("single layout may only read feedback for single_side.")
        if not math.isfinite(publish_rate_hz) or publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive and finite.")
        if (
            not math.isfinite(self._feedback_stale_warning_s)
            or self._feedback_stale_warning_s <= 0.0
        ):
            raise ValueError("feedback_stale_warning_s must be positive and finite.")

        self._output_sides = _SIDES if self._layout == "dual" else (self._single_side,)
        self._physical_sides = physical_sides
        self._feedback: dict[str, tuple[float, float]] = {}
        self._last_warning_at: dict[str, float] = {}
        self._publisher = self.create_publisher(JointState, "joint_states", 10)
        self._subscriptions = []
        topics = {
            "left": str(self.get_parameter("left_feedback_topic").value).strip(),
            "right": str(self.get_parameter("right_feedback_topic").value).strip(),
        }
        for side in sorted(self._physical_sides):
            if not topics[side]:
                raise ValueError(f"{side}_feedback_topic must not be empty for a physical gripper.")
            self._subscriptions.append(
                self.create_subscription(
                    Float64,
                    topics[side],
                    lambda message, selected_side=side: self._feedback_callback(
                        selected_side,
                        message,
                    ),
                    10,
                )
            )
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish)
        active = ",".join(sorted(self._physical_sides)) or "none (mock)"
        self.get_logger().info(
            f"publishing {self._layout} {self._gripper_model} state from physical sides: {active}"
        )

    def _feedback_callback(self, side: str, message: Float64) -> None:
        position = float(message.data)
        if not math.isfinite(position):
            self._warn_throttled(side, f"ignoring non-finite {side} gripper feedback")
            return
        self._feedback[side] = (max(0.0, min(1.0, position)), time.monotonic())

    def _warn_throttled(self, side: str, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_at.get(side, -math.inf) >= 5.0:
            self.get_logger().warning(message)
            self._last_warning_at[side] = now

    def _state_for_side(self, side: str, now: float) -> tuple[str, float]:
        feedback = self._feedback.get(side)
        if side in self._physical_sides and feedback is not None:
            position, received_at = feedback
            age_s = now - received_at
            if age_s > self._feedback_stale_warning_s:
                self._warn_throttled(
                    side,
                    f"{side} gripper feedback has not updated for {age_s:.2f}s; "
                    "continuing to publish the last measured position",
                )
            return gripper_driver_state_from_closure_fraction(
                layout=self._layout,
                side=side,
                gripper_model=self._gripper_model,
                closure_fraction=position,
            )
        elif side in self._physical_sides:
            self._warn_throttled(
                side,
                f"waiting for {side} gripper position feedback; "
                "publishing the fully-open collision envelope",
            )
        return modeled_open_driver_state(
            layout=self._layout,
            side=side,
            gripper_model=self._gripper_model,
        )

    def _publish(self) -> None:
        now = time.monotonic()
        states = [self._state_for_side(side, now) for side in self._output_sides]
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = [name for name, _position in states]
        message.position = [position for _name, position in states]
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GripperJointStateBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


__all__ = [
    "GripperJointStateBridge",
    "gripper_driver_state_from_closure_fraction",
    "gripper_driver_state_from_width",
    "main",
    "modeled_open_driver_state",
]
