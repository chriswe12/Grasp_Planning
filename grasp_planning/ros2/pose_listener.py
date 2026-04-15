"""ROS2 helper to wait for an object pose update."""

from __future__ import annotations

import time
from dataclasses import dataclass

from grasp_planning.grasping.world_constraints import ObjectWorldPose

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseStamped
    from rclpy.node import Node
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    Pose = None
    PoseStamped = None
    Node = object


@dataclass(frozen=True)
class PoseTopicConfig:
    topic_name: str
    message_type: str
    timeout_s: float


class _ObjectPoseListener(Node):
    def __init__(self, config: PoseTopicConfig) -> None:
        super().__init__("grasp_planning_pose_listener")
        self._message_type = str(config.message_type)
        self._latest_pose: ObjectWorldPose | None = None
        if self._message_type == "geometry_msgs/msg/Pose":
            self.create_subscription(Pose, config.topic_name, self._on_pose, 10)
        elif self._message_type == "geometry_msgs/msg/PoseStamped":
            self.create_subscription(PoseStamped, config.topic_name, self._on_pose_stamped, 10)
        else:
            raise ValueError(f"Unsupported pose message type '{config.message_type}'.")

    @property
    def latest_pose(self) -> ObjectWorldPose | None:
        return self._latest_pose

    def _on_pose(self, msg: Pose) -> None:
        self._latest_pose = ObjectWorldPose(
            position_world=(float(msg.position.x), float(msg.position.y), float(msg.position.z)),
            orientation_xyzw_world=(
                float(msg.orientation.x),
                float(msg.orientation.y),
                float(msg.orientation.z),
                float(msg.orientation.w),
            ),
        )

    def _on_pose_stamped(self, msg: PoseStamped) -> None:
        self._on_pose(msg.pose)

    def publish_status(self, text: str) -> None:
        self.get_logger().info(text)


def wait_for_object_pose_message(
    *,
    topic_name: str,
    message_type: str,
    timeout_s: float,
) -> ObjectWorldPose:
    if rclpy is None or Pose is None or PoseStamped is None:
        raise RuntimeError("ROS2 dependencies are unavailable. Install rclpy and geometry_msgs to use --mode real.")

    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True

    node = _ObjectPoseListener(
        PoseTopicConfig(topic_name=topic_name, message_type=message_type, timeout_s=float(timeout_s))
    )
    try:
        deadline = time.monotonic() + float(timeout_s)
        node.publish_status(f"Waiting for object pose on '{topic_name}' ({message_type})...")
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest_pose is not None:
                node.publish_status("Received object pose.")
                return node.latest_pose
        raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for object pose on '{topic_name}'.")
    finally:
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()
