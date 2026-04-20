"""ROS2 helpers to wait for object-frame updates from external topics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from grasp_planning.grasping.world_constraints import ObjectWorldPose

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseStamped, Vector3Stamped
    from rclpy.node import Node
except Exception:  # pragma: no cover - optional dependency path
    rclpy = None
    Pose = None
    PoseStamped = None
    Vector3Stamped = None
    Node = object

try:
    from fp_debug_msgs.msg import DebugFrame
except Exception:  # pragma: no cover - optional dependency path
    DebugFrame = None


@dataclass(frozen=True)
class PoseTopicConfig:
    topic_name: str
    message_type: str
    timeout_s: float


@dataclass(frozen=True)
class DebugFrameTopicConfig:
    topic_name: str
    message_type: str
    object_id: str
    timeout_s: float


@dataclass(frozen=True)
class RealFramePair:
    source_frame_pose_obj_world: ObjectWorldPose
    execution_pose_world: ObjectWorldPose


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


def centroid_offset_to_source_frame_pose(offset_msg: Any) -> ObjectWorldPose:
    """Convert an obj-to-local centroid offset into the stored local-frame pose."""

    vector = getattr(offset_msg, "vector", offset_msg)
    return ObjectWorldPose(
        position_world=(float(vector.x), float(vector.y), float(vector.z)),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
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


def _stamp_key_from_msg(msg: Any) -> tuple[int, int] | None:
    stamp = getattr(msg, "stamp", None)
    if stamp is None:
        header = getattr(msg, "header", None)
        stamp = None if header is None else getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return (int(sec), int(nanosec))


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
        self._latest_pose = _pose_to_object_world_pose(msg)

    def _on_pose_stamped(self, msg: PoseStamped) -> None:
        self._on_pose(msg.pose)

    def publish_status(self, text: str) -> None:
        self.get_logger().info(text)


class _Vector3StampedListener(Node):
    def __init__(self, config: PoseTopicConfig) -> None:
        super().__init__("grasp_planning_vector_listener")
        self._latest_pose: ObjectWorldPose | None = None
        if str(config.message_type) != "geometry_msgs/msg/Vector3Stamped":
            raise ValueError(f"Unsupported vector message type '{config.message_type}'.")
        self.create_subscription(Vector3Stamped, config.topic_name, self._on_vector3_stamped, 10)

    @property
    def latest_pose(self) -> ObjectWorldPose | None:
        return self._latest_pose

    def _on_vector3_stamped(self, msg: Vector3Stamped) -> None:
        self._latest_pose = centroid_offset_to_source_frame_pose(msg)

    def publish_status(self, text: str) -> None:
        self.get_logger().info(text)


class _DebugFramePoseListener(Node):
    def __init__(self, config: DebugFrameTopicConfig) -> None:
        super().__init__("grasp_planning_debug_frame_listener")
        self._object_id = str(config.object_id)
        self._latest_pose: ObjectWorldPose | None = None
        if str(config.message_type) != "fp_debug_msgs/msg/DebugFrame":
            raise ValueError(f"Unsupported debug frame message type '{config.message_type}'.")
        self.create_subscription(DebugFrame, config.topic_name, self._on_debug_frame, 10)

    @property
    def latest_pose(self) -> ObjectWorldPose | None:
        return self._latest_pose

    def _on_debug_frame(self, msg: DebugFrame) -> None:
        pose = extract_execution_pose_from_debug_frame(msg, object_id=self._object_id)
        if pose is not None:
            self._latest_pose = pose

    def publish_status(self, text: str) -> None:
        self.get_logger().info(text)


class _RealFramePairListener(Node):
    _MAX_STAMP_CACHE_ENTRIES = 16

    def __init__(self, *, source_config: PoseTopicConfig, execution_config: DebugFrameTopicConfig) -> None:
        super().__init__("grasp_planning_real_frame_pair_listener")
        self._object_id = str(execution_config.object_id)
        if str(source_config.message_type) != "geometry_msgs/msg/Vector3Stamped":
            raise ValueError(f"Unsupported vector message type '{source_config.message_type}'.")
        if str(execution_config.message_type) != "fp_debug_msgs/msg/DebugFrame":
            raise ValueError(f"Unsupported debug frame message type '{execution_config.message_type}'.")
        if not self._object_id:
            raise ValueError("object_id must be non-empty when subscribing to fp_debug_msgs/msg/DebugFrame.")

        self._latest_source_pose: ObjectWorldPose | None = None
        self._latest_execution_pose: ObjectWorldPose | None = None
        self._latest_source_stamp: tuple[int, int] | None = None
        self._latest_execution_stamp: tuple[int, int] | None = None
        self._source_by_stamp: dict[tuple[int, int], ObjectWorldPose] = {}
        self._execution_by_stamp: dict[tuple[int, int], ObjectWorldPose] = {}

        self.create_subscription(Vector3Stamped, source_config.topic_name, self._on_vector3_stamped, 10)
        self.create_subscription(DebugFrame, execution_config.topic_name, self._on_debug_frame, 10)

    @property
    def latest_pair(self) -> RealFramePair | None:
        shared_stamps = tuple(set(self._source_by_stamp) & set(self._execution_by_stamp))
        if shared_stamps:
            matched_stamp = max(shared_stamps)
            return RealFramePair(
                source_frame_pose_obj_world=self._source_by_stamp[matched_stamp],
                execution_pose_world=self._execution_by_stamp[matched_stamp],
            )
        if self._latest_source_pose is None or self._latest_execution_pose is None:
            return None
        if self._latest_source_stamp is None or self._latest_execution_stamp is None:
            return RealFramePair(
                source_frame_pose_obj_world=self._latest_source_pose,
                execution_pose_world=self._latest_execution_pose,
            )
        return None

    def _trim_cache(self, cache: dict[tuple[int, int], ObjectWorldPose]) -> None:
        while len(cache) > self._MAX_STAMP_CACHE_ENTRIES:
            oldest_stamp = next(iter(cache))
            del cache[oldest_stamp]

    def _on_vector3_stamped(self, msg: Vector3Stamped) -> None:
        pose = centroid_offset_to_source_frame_pose(msg)
        stamp = _stamp_key_from_msg(msg)
        self._latest_source_pose = pose
        self._latest_source_stamp = stamp
        if stamp is not None:
            self._source_by_stamp[stamp] = pose
            self._trim_cache(self._source_by_stamp)

    def _on_debug_frame(self, msg: DebugFrame) -> None:
        pose = extract_execution_pose_from_debug_frame(msg, object_id=self._object_id)
        if pose is None:
            return
        stamp = _stamp_key_from_msg(msg)
        self._latest_execution_pose = pose
        self._latest_execution_stamp = stamp
        if stamp is not None:
            self._execution_by_stamp[stamp] = pose
            self._trim_cache(self._execution_by_stamp)

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


def wait_for_centroid_offset_message(
    *,
    topic_name: str,
    message_type: str,
    timeout_s: float,
) -> ObjectWorldPose:
    if rclpy is None or Vector3Stamped is None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Install rclpy and geometry_msgs to use centroid-offset subscribers."
        )

    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True

    node = _Vector3StampedListener(
        PoseTopicConfig(topic_name=topic_name, message_type=message_type, timeout_s=float(timeout_s))
    )
    try:
        deadline = time.monotonic() + float(timeout_s)
        node.publish_status(f"Waiting for centroid offset on '{topic_name}' ({message_type})...")
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest_pose is not None:
                node.publish_status("Received centroid offset.")
                return node.latest_pose
        raise TimeoutError(f"Timed out after {timeout_s:.1f}s waiting for centroid offset on '{topic_name}'.")
    finally:
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()


def wait_for_debug_frame_pose_message(
    *,
    topic_name: str,
    message_type: str,
    object_id: str,
    timeout_s: float,
) -> ObjectWorldPose:
    if rclpy is None or DebugFrame is None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Install rclpy and fp_debug_msgs to use DebugFrame subscribers."
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
            f"Waiting for execution pose on '{topic_name}' ({message_type}) for object_id='{object_id}'..."
        )
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest_pose is not None:
                node.publish_status("Received execution pose.")
                return node.latest_pose
        raise TimeoutError(
            f"Timed out after {timeout_s:.1f}s waiting for execution pose on '{topic_name}' for object_id='{object_id}'."
        )
    finally:
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()


def wait_for_real_frame_pair_messages(
    *,
    source_topic_name: str,
    source_message_type: str,
    execution_topic_name: str,
    execution_message_type: str,
    object_id: str,
    timeout_s: float,
) -> RealFramePair:
    if rclpy is None or Vector3Stamped is None or DebugFrame is None:
        raise RuntimeError(
            "ROS2 dependencies are unavailable. Install rclpy, geometry_msgs, and fp_debug_msgs to use dual subscribers."
        )

    initialized_here = False
    if not rclpy.ok():
        rclpy.init()
        initialized_here = True

    node = _RealFramePairListener(
        source_config=PoseTopicConfig(
            topic_name=source_topic_name,
            message_type=source_message_type,
            timeout_s=float(timeout_s),
        ),
        execution_config=DebugFrameTopicConfig(
            topic_name=execution_topic_name,
            message_type=execution_message_type,
            object_id=str(object_id),
            timeout_s=float(timeout_s),
        ),
    )
    try:
        deadline = time.monotonic() + float(timeout_s)
        node.publish_status(
            "Waiting for source-frame offset on "
            f"'{source_topic_name}' and execution pose on '{execution_topic_name}' for object_id='{object_id}'..."
        )
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            pair = node.latest_pair
            if pair is not None:
                node.publish_status("Received matched real-world frame pair.")
                return pair
        raise TimeoutError(
            f"Timed out after {timeout_s:.1f}s waiting for real-world frame pair from "
            f"'{source_topic_name}' and '{execution_topic_name}' for object_id='{object_id}'."
        )
    finally:
        node.destroy_node()
        if initialized_here:
            rclpy.shutdown()
