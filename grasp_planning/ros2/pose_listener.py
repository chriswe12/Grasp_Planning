"""ROS2 helpers to wait for DebugFrame-based object poses."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from grasp_planning.grasping.world_constraints import ObjectWorldPose

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    Node = object
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None

try:
    from fp_debug_msgs.msg import DebugFrame
except Exception:  # pragma: no cover - optional dependency path
    DebugFrame = None


@dataclass(frozen=True)
class DebugFrameTopicConfig:
    topic_name: str
    message_type: str
    object_id: str
    timeout_s: float


def _pose_to_object_world_pose(pose_msg: Any) -> ObjectWorldPose:
    return ObjectWorldPose(
        position_world=(
            float(pose_msg.position.x),
            float(pose_msg.position.y),
            float(pose_msg.position.z),
        ),
        orientation_xyzw_world=(
            float(pose_msg.orientation.x),
            float(pose_msg.orientation.y),
            float(pose_msg.orientation.z),
            float(pose_msg.orientation.w),
        ),
    )


def extract_execution_pose_from_debug_frame(debug_frame_msg: Any, *, object_id: str) -> ObjectWorldPose | None:
    """Select the highest-score pose_base entry for the requested object."""

    best_pose: ObjectWorldPose | None = None
    best_score = float("-inf")
    for item in tuple(getattr(debug_frame_msg, "pose_items", ())):
        if str(getattr(item, "object_id", "")) != str(object_id):
            continue
        pose_base = getattr(item, "pose_base", None)
        if pose_base is None:
            continue
        try:
            pose = _pose_to_object_world_pose(pose_base)
        except Exception:
            continue
        score = float(getattr(item, "score", 0.0))
        if best_pose is None or score > best_score:
            best_pose = pose
            best_score = score
    return best_pose


def _subscription_qos(depth: int = 10):
    if QoSProfile is None or ReliabilityPolicy is None or HistoryPolicy is None:
        return depth
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


class _DebugFramePoseListener(Node):
    def __init__(self, config: DebugFrameTopicConfig) -> None:
        super().__init__("grasp_planning_debug_frame_listener")
        self._object_id = str(config.object_id)
        self._latest_pose: ObjectWorldPose | None = None
        if str(config.message_type) != "fp_debug_msgs/msg/DebugFrame":
            raise ValueError(f"Unsupported debug frame message type '{config.message_type}'.")
        self.create_subscription(DebugFrame, config.topic_name, self._on_debug_frame, _subscription_qos())

    @property
    def latest_pose(self) -> ObjectWorldPose | None:
        return self._latest_pose

    def _on_debug_frame(self, msg: DebugFrame) -> None:
        pose = extract_execution_pose_from_debug_frame(msg, object_id=self._object_id)
        if pose is not None:
            self._latest_pose = pose

    def publish_status(self, text: str) -> None:
        self.get_logger().info(text)


def wait_for_debug_frame_pose_message(
    *,
    topic_name: str,
    message_type: str,
    object_id: str,
    timeout_s: float,
) -> ObjectWorldPose:
    if rclpy is None or DebugFrame is None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Source ROS2 and the repo overlay before using DebugFrame "
            "subscribers. For example: source /opt/ros/<distro>/setup.bash; "
            "cd ros2_ws && colcon build --packages-select fp_debug_msgs --symlink-install; "
            "source install/setup.bash."
        )
    if not str(object_id):
        raise ValueError("object_id must be non-empty when subscribing to fp_debug_msgs/msg/DebugFrame.")

    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True

    node = _DebugFramePoseListener(
        DebugFrameTopicConfig(
            topic_name=topic_name,
            message_type=message_type,
            object_id=str(object_id),
            timeout_s=float(timeout_s),
        )
    )
    try:
        deadline = time.monotonic() + float(timeout_s)
        node.publish_status(
            f"Waiting for object pose on '{topic_name}' ({message_type}) for object_id='{object_id}'..."
        )
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest_pose is not None:
                node.publish_status("Received object pose.")
                return node.latest_pose
        raise TimeoutError(
            f"Timed out after {timeout_s:.1f}s waiting for object pose on '{topic_name}' for object_id='{object_id}'."
        )
    finally:
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()
