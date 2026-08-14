from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
    VISUAL_SERVO_RENDER_HEIGHT,
    VISUAL_SERVO_RENDER_WIDTH,
    D405WristCameraConfig,
    camera_pose_in_link7,
    nominal_focal_lengths_from_fov,
)


def test_visual_servo_render_profile_keeps_native_intrinsic_reference() -> None:
    config = D405WristCameraConfig()

    assert (config.width, config.height) == (848, 480)
    assert (VISUAL_SERVO_RENDER_WIDTH, VISUAL_SERVO_RENDER_HEIGHT) == (256, 144)
    assert (VISUAL_SERVO_OBSERVATION_WIDTH, VISUAL_SERVO_OBSERVATION_HEIGHT) == (128, 72)
    assert "native_848x480_intrinsics" in D405_VISUAL_SERVO_CAMERA_PROFILE
    assert "area_128x72" in D405_VISUAL_SERVO_OBSERVATION_PROFILE


def _quat_wxyz_to_matrix(quaternion: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_nominal_d405_intrinsics_match_published_fov_placeholder() -> None:
    fx, fy = nominal_focal_lengths_from_fov(
        width=848,
        height=480,
        horizontal_fov_deg=84.0,
        vertical_fov_deg=58.0,
    )
    assert fx == pytest.approx(470.900, abs=1.0e-3)
    assert fy == pytest.approx(432.971, abs=1.0e-3)


def test_cad_flange_to_camera_pose_is_used_directly_in_link7() -> None:
    config = D405WristCameraConfig()
    position_link7, quaternion_link7_camera = camera_pose_in_link7(config)
    rotation_camera_in_link7 = _quat_wxyz_to_matrix(quaternion_link7_camera)
    rotation_camera_in_ee = np.asarray(config.rotation_camera_in_calibration_parent).reshape(3, 3)

    np.testing.assert_allclose(rotation_camera_in_link7, rotation_camera_in_ee, atol=1.0e-7)
    np.testing.assert_allclose(
        rotation_camera_in_link7,
        (
            (0.0, -np.sqrt(3.0) / 2.0, -0.5),
            (1.0, 0.0, 0.0),
            (0.0, -0.5, np.sqrt(3.0) / 2.0),
        ),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        position_link7,
        (0.055667, 0.009, 0.070776),
        atol=1.0e-9,
    )


def test_intrinsic_matrix_is_row_major() -> None:
    config = D405WristCameraConfig(fx=1.0, fy=2.0, cx=3.0, cy=4.0)
    assert config.intrinsic_matrix_row_major == [1.0, 0.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0, 1.0]


def test_camera_parent_is_link7() -> None:
    assert D405WristCameraConfig().parent_prim_path == "link7"


def test_isaac_runner_has_no_detached_world_recording_camera() -> None:
    runner = (Path(__file__).resolve().parents[1] / "scripts/run_fabrica_grasp_in_isaac.py").read_text(
        encoding="utf-8"
    )
    assert "/World/ExecutionBenchmarkVideo/CameraSensor" not in runner
    assert "Attached D405 wrist camera under" in runner
    assert "if wrist_camera is not None and args_cli.record_video is not None:" in runner
    assert "grasp_observation_callback=_capture_visual_servo_goal" in runner
