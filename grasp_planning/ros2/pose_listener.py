"""ROS2 helpers to wait for fused DebugPoseItem object poses."""

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
    from fp_debug_msgs.msg import DebugPoseItem
except Exception:  # pragma: no cover - optional dependency path
    DebugPoseItem = None


@dataclass(frozen=True)
class DebugPoseItemTopicConfig:
    topic_name: str
    message_type: str
    assembly_name: str
    part_id: int
    timeout_s: float


def _pose_to_object_world_pose(pose_msg: Any) -> ObjectWorldPose:
    # DebugPoseItem uses PoseStamped. Accept a bare Pose as well so this helper
    # remains useful with simple test doubles and other pose-producing callers.
    pose = getattr(pose_msg, "pose", pose_msg)
    return ObjectWorldPose(
        position_world=(
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ),
        orientation_xyzw_world=(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ),
    )


def extract_execution_pose_from_debug_pose_item(
    pose_item_msg: Any,
    *,
    assembly_name: str,
    part_id: int,
) -> ObjectWorldPose | None:
    """Return pose_base when one fused pose item matches the requested Fabrica part."""

    if str(getattr(pose_item_msg, "assembly_name", "")) != str(assembly_name):
        return None
    if int(getattr(pose_item_msg, "part_id", -1)) != int(part_id):
        return None
    pose_base = getattr(pose_item_msg, "pose_base", None)
    if pose_base is None:
        return None
    try:
        return _pose_to_object_world_pose(pose_base)
    except Exception:
        return None


def _subscription_qos(depth: int = 10):
    if QoSProfile is None or ReliabilityPolicy is None or HistoryPolicy is None:
        return depth
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )


class _DebugPoseItemListener(Node):
    def __init__(self, config: DebugPoseItemTopicConfig) -> None:
        super().__init__("grasp_planning_debug_pose_item_listener")
        self._assembly_name = str(config.assembly_name)
        self._part_id = int(config.part_id)
        self._latest_pose: ObjectWorldPose | None = None
        if str(config.message_type) != "fp_debug_msgs/msg/DebugPoseItem":
            raise ValueError(f"Unsupported fused pose message type '{config.message_type}'.")
        self.create_subscription(DebugPoseItem, config.topic_name, self._on_debug_pose_item, _subscription_qos())

    @property
    def latest_pose(self) -> ObjectWorldPose | None:
        return self._latest_pose

    def _on_debug_pose_item(self, msg: DebugPoseItem) -> None:
        pose = extract_execution_pose_from_debug_pose_item(
            msg,
            assembly_name=self._assembly_name,
            part_id=self._part_id,
        )
        if pose is not None:
            self._latest_pose = pose

    def publish_status(self, text: str) -> None:
        self.get_logger().info(text)


def wait_for_debug_pose_item_message(
    *,
    topic_name: str,
    message_type: str,
    assembly_name: str,
    part_id: int,
    timeout_s: float,
) -> ObjectWorldPose:
    if rclpy is None or DebugPoseItem is None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Source ROS2 and the repo overlay before using DebugPoseItem "
            "subscribers. For example: source /opt/ros/<distro>/setup.bash; "
            "cd ros2_ws && colcon build --packages-select fp_debug_msgs --symlink-install; "
            "source install/setup.bash."
        )
    if not str(assembly_name):
        raise ValueError("assembly_name must be non-empty when subscribing to fp_debug_msgs/msg/DebugPoseItem.")
    if int(part_id) < 0:
        raise ValueError("part_id must be non-negative when subscribing to fp_debug_msgs/msg/DebugPoseItem.")

    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True

    node = _DebugPoseItemListener(
        DebugPoseItemTopicConfig(
            topic_name=topic_name,
            message_type=message_type,
            assembly_name=str(assembly_name),
            part_id=int(part_id),
            timeout_s=float(timeout_s),
        )
    )
    try:
        deadline = time.monotonic() + float(timeout_s)
        part_key = f"{assembly_name}/{part_id}"
        node.publish_status(f"Waiting for object pose on '{topic_name}' ({message_type}) for '{part_key}'...")
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest_pose is not None:
                node.publish_status("Received object pose.")
                return node.latest_pose
        raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for object pose on '{topic_name}' for '{part_key}'.")
    finally:
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()
