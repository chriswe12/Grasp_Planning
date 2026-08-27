"""Synchronized RealSense D405 RGB-D intake without a cv_bridge dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:  # pragma: no cover - exercised only in a sourced ROS2 environment
    import message_filters
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
except Exception:  # pragma: no cover - optional dependency path
    message_filters = None
    qos_profile_sensor_data = None
    Image = None


def ros_stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def image_message_to_rgb8(message) -> np.ndarray:
    encoding = str(message.encoding).strip().lower()
    if encoding != "rgb8":
        raise ValueError(f"D405 color encoding must be rgb8, got '{message.encoding}'.")
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if height <= 0 or width <= 0 or step < width * 3:
        raise ValueError(f"Malformed RGB image dimensions height={height} width={width} step={step}.")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError(f"RGB image payload has {raw.size} bytes; expected {height * step}.")
    return raw.reshape(height, step)[:, : width * 3].reshape(height, width, 3).copy()


def image_message_to_depth_z16(message) -> np.ndarray:
    encoding = str(message.encoding).strip().lower()
    if encoding != "16uc1":
        raise ValueError(f"D405 aligned depth encoding must be 16UC1, got '{message.encoding}'.")
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if height <= 0 or width <= 0 or step < width * 2 or step % 2:
        raise ValueError(f"Malformed depth image dimensions height={height} width={width} step={step}.")
    byte_order = ">" if bool(message.is_bigendian) else "<"
    raw = np.frombuffer(message.data, dtype=np.dtype(f"{byte_order}u2"))
    row_values = step // 2
    if raw.size != height * row_values:
        raise ValueError(f"Depth image payload has {raw.size} values; expected {height * row_values}.")
    return raw.reshape(height, row_values)[:, :width].astype(np.uint16, copy=True)


@dataclass(frozen=True)
class SynchronizedD405Frame:
    rgb_uint8: np.ndarray
    depth_z16: np.ndarray
    color_stamp_s: float
    depth_stamp_s: float
    camera_frame_id: str


class D405RgbdSubscriber:
    """Queue-one approximate synchronizer for rectified color/aligned depth."""

    def __init__(
        self,
        node,
        *,
        color_topic: str,
        depth_topic: str,
        maximum_skew_s: float,
        callback: Callable[[SynchronizedD405Frame], None],
    ) -> None:
        if message_filters is None or Image is None:
            raise RuntimeError("ROS2 message_filters and sensor_msgs are required for D405 deployment.")
        if maximum_skew_s <= 0.0:
            raise ValueError("maximum_skew_s must be positive.")
        self._callback = callback
        self._maximum_skew_s = float(maximum_skew_s)
        self._last_stamp_s = -float("inf")
        self.color_subscriber = message_filters.Subscriber(
            node,
            Image,
            str(color_topic),
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_subscriber = message_filters.Subscriber(
            node,
            Image,
            str(depth_topic),
            qos_profile=qos_profile_sensor_data,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.color_subscriber, self.depth_subscriber),
            queue_size=2,
            slop=self._maximum_skew_s,
            allow_headerless=False,
        )
        self.synchronizer.registerCallback(self._on_messages)

    def _on_messages(self, color_message, depth_message) -> None:
        color_stamp_s = ros_stamp_seconds(color_message.header.stamp)
        depth_stamp_s = ros_stamp_seconds(depth_message.header.stamp)
        if abs(color_stamp_s - depth_stamp_s) > self._maximum_skew_s:
            raise ValueError("Synchronized D405 pair exceeds the configured timestamp skew.")
        newest_stamp = max(color_stamp_s, depth_stamp_s)
        if newest_stamp <= self._last_stamp_s:
            return
        rgb = image_message_to_rgb8(color_message)
        depth = image_message_to_depth_z16(depth_message)
        if rgb.shape[:2] != depth.shape:
            raise ValueError("Aligned D405 depth dimensions do not match the color image.")
        color_frame = str(color_message.header.frame_id).strip()
        depth_frame = str(depth_message.header.frame_id).strip()
        if not color_frame or not depth_frame:
            raise ValueError("D405 image messages must carry non-empty frame IDs.")
        self._last_stamp_s = newest_stamp
        self._callback(
            SynchronizedD405Frame(
                rgb_uint8=rgb,
                depth_z16=depth,
                color_stamp_s=color_stamp_s,
                depth_stamp_s=depth_stamp_s,
                camera_frame_id=color_frame,
            )
        )


__all__ = [
    "D405RgbdSubscriber",
    "SynchronizedD405Frame",
    "image_message_to_depth_z16",
    "image_message_to_rgb8",
    "ros_stamp_seconds",
]
