from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grasp_planning.d405_wrist_camera import (
    D405_LEGACY_HAND_EYE_MOUNT_PROFILE,
    D405_PDZ_NAMED_FRAME_MOUNT_PROFILE,
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
    VISUAL_SERVO_RENDER_HEIGHT,
    VISUAL_SERVO_RENDER_WIDTH,
    D405WristCameraConfig,
    camera_pose_in_link7,
    camera_rotation_in_link7,
)


def test_visual_servo_render_profile_keeps_native_intrinsic_reference() -> None:
    config = D405WristCameraConfig()

    assert (config.width, config.height) == (848, 480)
    assert (VISUAL_SERVO_RENDER_WIDTH, VISUAL_SERVO_RENDER_HEIGHT) == (256, 144)
    assert (VISUAL_SERVO_OBSERVATION_WIDTH, VISUAL_SERVO_OBSERVATION_HEIGHT) == (128, 72)
    assert "color_848x480_intrinsics_260322275185" in D405_VISUAL_SERVO_CAMERA_PROFILE
    assert "pdz_named_frame_link7_mount35mm" in D405_VISUAL_SERVO_CAMERA_PROFILE
    assert config.mount_profile == D405_PDZ_NAMED_FRAME_MOUNT_PROFILE
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


def test_deployed_d405_color_intrinsics_and_distortion_are_preserved() -> None:
    config = D405WristCameraConfig()
    assert config.fx == pytest.approx(436.3104248046875)
    assert config.fy == pytest.approx(435.6492614746094)
    assert config.cx == pytest.approx(418.62664794921875)
    assert config.cy == pytest.approx(236.5121307373047)
    assert config.distortion_model == "plumb_bob"
    assert config.distortion_coefficients == pytest.approx(
        (-0.05201759934425354, 0.05433472618460655, 0.0002693705027922988, 0.0008704775245860219, -0.017724450677633286)
    )
    assert config.stereo_baseline_m == pytest.approx(0.018)
    assert config.reliable_depth_range_m == pytest.approx((0.07, 0.50))
    assert config.depth_unit_m == pytest.approx(0.0001)


def test_pdz_named_frame_camera_pose_matches_audited_sensor_origin() -> None:
    config = D405WristCameraConfig()
    position_link7, quaternion_link7_camera = camera_pose_in_link7(config)
    rotation_camera_in_link7 = _quat_wxyz_to_matrix(quaternion_link7_camera)
    np.testing.assert_allclose(
        rotation_camera_in_link7,
        (
            (-1.0, 0.0, 0.0),
            (0.0, -np.sqrt(3.0) / 2.0, 0.5),
            (0.0, 0.5, np.sqrt(3.0) / 2.0),
        ),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        position_link7,
        (0.009, -0.050560254038, 0.097927071163),
        atol=1.0e-9,
    )
    np.testing.assert_allclose(rotation_camera_in_link7, camera_rotation_in_link7(config), atol=1.0e-9)


def test_legacy_hand_eye_mount_remains_explicitly_selectable() -> None:
    config = D405WristCameraConfig(mount_profile=D405_LEGACY_HAND_EYE_MOUNT_PROFILE)
    position_link7, _ = camera_pose_in_link7(config)
    np.testing.assert_allclose(
        position_link7,
        (0.0556666756801677, 0.008999999999999992, 0.10577648943349074),
        atol=1.0e-9,
    )


def test_intrinsic_matrix_is_row_major() -> None:
    config = D405WristCameraConfig(fx=1.0, fy=2.0, cx=3.0, cy=4.0)
    assert config.intrinsic_matrix_row_major == [1.0, 0.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0, 1.0]


def test_camera_parent_is_link7() -> None:
    assert D405WristCameraConfig().parent_prim_path == "link7"


def test_policy_goal_rendering_is_separate_from_isaac_execution() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/run_fabrica_grasp_in_isaac.py").read_text(encoding="utf-8")
    renderer = (root / "scripts/render_d405_policy_goal.py").read_text(encoding="utf-8")
    executor = (root / "grasp_planning/ros2/real_grasp_executor.py").read_text(encoding="utf-8")

    assert "visual-servo-goal-image" not in runner
    assert "_capture_visual_servo_goal" not in runner
    assert "renderer-backend" in renderer
    assert "render_d405_goal_for_grasp" in executor


def test_rl_action_frame_uses_the_same_corrected_camera_rotation_as_rendering() -> None:
    environment = (
        Path(__file__).resolve().parents[1] / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/isaac_rl_env.py"
    ).read_text(encoding="utf-8")

    assert "camera_rotation_in_link7(self.camera_config)" in environment
    assert "D405WristCameraConfig().rotation_camera_in_calibration_parent" not in environment
