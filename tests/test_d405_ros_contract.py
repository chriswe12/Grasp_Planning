from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grasp_planning.ros2.d405_rgbd_subscriber import (
    image_message_to_depth_z16,
    image_message_to_rgb8,
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
