"""Synchronized RealSense D405 RGB-D intake without a cv_bridge dependency."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:  # pragma: no cover - exercised only in a sourced ROS2 environment
    import message_filters
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
except Exception:  # pragma: no cover - optional dependency path
    message_filters = None
    qos_profile_sensor_data = None
    CompressedImage = None

try:
    from PIL import Image as PillowImage
except Exception:  # pragma: no cover - optional compressed-transport dependency
    PillowImage = None


def ros_stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def compressed_color_message_to_rgb8(message) -> np.ndarray:
    if PillowImage is None:
        raise RuntimeError("Pillow is required for compressed D405 color transport.")
    image_format = str(message.format).strip().lower()
    if "jpeg" not in image_format and "jpg" not in image_format and "png" not in image_format:
        raise ValueError(f"Unsupported compressed D405 color format '{message.format}'.")
    with PillowImage.open(io.BytesIO(bytes(message.data))) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Decoded D405 color image has invalid shape {rgb.shape}.")
    return rgb


def compressed_depth_message_to_z16(message) -> np.ndarray:
    if PillowImage is None:
        raise RuntimeError("Pillow is required for compressed D405 depth transport.")
    image_format = str(message.format).strip().lower()
    if "16uc1" not in image_format or "compresseddepth" not in image_format:
        raise ValueError(
            "D405 compressed depth must use the lossless 16UC1 compressedDepth transport; "
            f"got '{message.format}'."
        )
    payload = bytes(message.data)
    png_signature = b"\x89PNG\r\n\x1a\n"
    png_offset = payload.find(png_signature)
    if png_offset < 0:
        raise ValueError("D405 compressedDepth payload does not contain a PNG image.")
    with PillowImage.open(io.BytesIO(payload[png_offset:])) as image:
        depth = np.asarray(image).copy()
    if depth.ndim != 2:
        raise ValueError(f"Decoded D405 depth image has invalid shape {depth.shape}.")
    if depth.dtype != np.uint16:
        if not np.issubdtype(depth.dtype, np.integer):
            raise ValueError(f"Decoded D405 depth image has invalid dtype {depth.dtype}.")
        if depth.size and (int(depth.min()) < 0 or int(depth.max()) > np.iinfo(np.uint16).max):
            raise ValueError("Decoded D405 depth values do not fit uint16.")
        depth = depth.astype(np.uint16)
    return depth


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
        callback_group=None,
    ) -> None:
        if message_filters is None or CompressedImage is None:
            raise RuntimeError("ROS2 message_filters and sensor_msgs are required for RGB-D deployment.")
        if maximum_skew_s <= 0.0:
            raise ValueError("maximum_skew_s must be positive.")
        if PillowImage is None:
            raise RuntimeError("Compressed RGB-D transport requires Pillow.")
        self._callback = callback
        self._maximum_skew_s = float(maximum_skew_s)
        self._last_stamp_s = -float("inf")
        self.color_subscriber = message_filters.Subscriber(
            node,
            CompressedImage,
            str(color_topic),
            qos_profile=qos_profile_sensor_data,
            callback_group=callback_group,
        )
        self.depth_subscriber = message_filters.Subscriber(
            node,
            CompressedImage,
            str(depth_topic),
            qos_profile=qos_profile_sensor_data,
            callback_group=callback_group,
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
        rgb = compressed_color_message_to_rgb8(color_message)
        depth = compressed_depth_message_to_z16(depth_message)
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
    "compressed_color_message_to_rgb8",
    "compressed_depth_message_to_z16",
    "ros_stamp_seconds",
]
