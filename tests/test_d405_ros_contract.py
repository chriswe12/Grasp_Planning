from __future__ import annotations

import io
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from grasp_planning.ros2 import d405_rgbd_subscriber
from grasp_planning.ros2.d405_rgbd_subscriber import (
    D405RgbdSubscriber,
    compressed_color_message_to_rgb8,
    compressed_depth_message_to_z16,
    image_message_to_depth_z16,
    image_message_to_rgb8,
)


def test_rgbd_subscriptions_use_the_dedicated_control_callback_group(monkeypatch) -> None:
    color_subscription = object()
    depth_subscription = object()
    subscriber_type = mock.Mock(side_effect=[color_subscription, depth_subscription])
    synchronizer = mock.Mock()
    filters = SimpleNamespace(
        Subscriber=subscriber_type,
        ApproximateTimeSynchronizer=mock.Mock(return_value=synchronizer),
    )
    callback_group = object()
    node = object()
    monkeypatch.setattr(d405_rgbd_subscriber, "message_filters", filters)
    monkeypatch.setattr(d405_rgbd_subscriber, "Image", object())
    monkeypatch.setattr(d405_rgbd_subscriber, "qos_profile_sensor_data", "sensor-qos")

    D405RgbdSubscriber(
        node,
        color_topic="/color",
        depth_topic="/depth",
        image_transport="raw",
        maximum_skew_s=0.01,
        callback=mock.Mock(),
        callback_group=callback_group,
    )

    assert subscriber_type.call_args_list == [
        mock.call(
            node,
            mock.ANY,
            "/color",
            qos_profile="sensor-qos",
            callback_group=callback_group,
        ),
        mock.call(
            node,
            mock.ANY,
            "/depth",
            qos_profile="sensor-qos",
            callback_group=callback_group,
        ),
    ]
    filters.ApproximateTimeSynchronizer.assert_called_once_with(
        (color_subscription, depth_subscription),
        queue_size=2,
        slop=0.01,
        allow_headerless=False,
    )


def test_rgb_message_preserves_rgb_channel_order_and_row_padding() -> None:
    row = bytes([10, 20, 30, 40, 50, 60, 0, 0])
    message = SimpleNamespace(encoding="rgb8", height=1, width=2, step=8, data=row)

    rgb = image_message_to_rgb8(message)

    assert rgb.dtype == np.uint8
    assert rgb.tolist() == [[[10, 20, 30], [40, 50, 60]]]


def test_depth_message_decodes_little_endian_16uc1_with_padding() -> None:
    values = np.asarray([250, 500, 999], dtype="<u2")
    message = SimpleNamespace(
        encoding="16UC1",
        height=1,
        width=2,
        step=6,
        is_bigendian=False,
        data=values.tobytes(),
    )

    depth = image_message_to_depth_z16(message)

    assert depth.dtype == np.uint16
    assert depth.tolist() == [[250, 500]]


def test_depth_message_rejects_float_encoding() -> None:
    message = SimpleNamespace(
        encoding="32FC1",
        height=1,
        width=1,
        step=4,
        is_bigendian=False,
        data=b"\0" * 4,
    )

    try:
        image_message_to_depth_z16(message)
    except ValueError as exc:
        assert "16UC1" in str(exc)
    else:
        raise AssertionError("Expected the wrong depth encoding to be rejected.")


def test_compressed_color_message_decodes_jpeg_to_rgb8() -> None:
    source = np.zeros((8, 10, 3), dtype=np.uint8)
    source[..., 0] = 240
    payload = io.BytesIO()
    Image.fromarray(source, mode="RGB").save(payload, format="JPEG", quality=95)
    message = SimpleNamespace(format="rgb8; jpeg compressed bgr8", data=payload.getvalue())

    rgb = compressed_color_message_to_rgb8(message)

    assert rgb.shape == (8, 10, 3)
    assert rgb.dtype == np.uint8
    assert float(rgb[..., 0].mean()) > 220.0
    assert float(rgb[..., 1].mean()) < 20.0
    assert float(rgb[..., 2].mean()) < 20.0


def test_compressed_depth_message_decodes_lossless_uint16_png() -> None:
    source = np.asarray([[0, 250, 1000], [65535, 42, 7]], dtype=np.uint16)
    payload = io.BytesIO()
    Image.fromarray(source).save(payload, format="PNG")
    message = SimpleNamespace(
        format="16UC1; compressedDepth",
        data=b"\0" * 12 + payload.getvalue(),
    )

    depth = compressed_depth_message_to_z16(message)

    assert depth.dtype == np.uint16
    assert np.array_equal(depth, source)
