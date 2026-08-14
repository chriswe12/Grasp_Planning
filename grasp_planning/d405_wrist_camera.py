"""Calibrated RealSense D405 wrist-camera configuration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VISUAL_SERVO_RENDER_WIDTH = 256
VISUAL_SERVO_RENDER_HEIGHT = 144
VISUAL_SERVO_OBSERVATION_WIDTH = 128
VISUAL_SERVO_OBSERVATION_HEIGHT = 72
D405_VISUAL_SERVO_CAMERA_PROFILE = (
    "d405_native_848x480_intrinsics_render_256x144_cad_flange_v2"
)
D405_VISUAL_SERVO_OBSERVATION_PROFILE = "rgbd_render_256x144_area_128x72_v2"


@dataclass(frozen=True)
class D405WristCameraConfig:
    """Deployable camera geometry plus simulation observation settings."""

    enabled: bool = False
    parent_prim_path: str = "link7"
    width: int = 848
    height: int = 480
    fx: float = 470.900
    fy: float = 432.971
    cx: float = 423.5
    cy: float = 239.5
    clipping_range_m: tuple[float, float] = (0.04, 2.0)
    update_period_s: float = 0.0
    include_privileged_mask: bool = True
    # The CAD transform is expressed directly in the robot flange/link7 frame;
    # no legacy lbr_link_ee translation is composed on top of it.
    calibration_parent_position_in_link7_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # T_flange_cam: camera frame expressed in the flange/link7 frame.  The
    # supplied CAD values use rounded 0.866 entries; sqrt(3)/2 preserves the
    # intended proper 30-degree rotation exactly.
    rotation_camera_in_calibration_parent: tuple[float, ...] = (
        0.0,
        -0.8660254037844386,
        -0.5,
        1.0,
        0.0,
        0.0,
        0.0,
        -0.5,
        0.8660254037844386,
    )
    translation_camera_in_calibration_parent_m: tuple[float, float, float] = (
        0.055667,
        0.009,
        0.070776,
    )

    @property
    def intrinsic_matrix_row_major(self) -> list[float]:
        return [
            float(self.fx),
            0.0,
            float(self.cx),
            0.0,
            float(self.fy),
            float(self.cy),
            0.0,
            0.0,
            1.0,
        ]


def nominal_focal_lengths_from_fov(
    *, width: int, height: int, horizontal_fov_deg: float, vertical_fov_deg: float
) -> tuple[float, float]:
    """Return centered-pinhole focal lengths in pixels from nominal field of view."""

    fx = 0.5 * float(width) / math.tan(0.5 * math.radians(float(horizontal_fov_deg)))
    fy = 0.5 * float(height) / math.tan(0.5 * math.radians(float(vertical_fov_deg)))
    return fx, fy


def camera_pose_in_link7(config: D405WristCameraConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the camera pose expressed in Isaac's flange/link7 frame."""

    rotation_camera_in_calibration_parent = np.asarray(
        config.rotation_camera_in_calibration_parent, dtype=float
    ).reshape(3, 3)
    if not np.allclose(
        rotation_camera_in_calibration_parent.T @ rotation_camera_in_calibration_parent,
        np.eye(3),
        atol=1.0e-5,
    ):
        raise ValueError("D405 EE-to-camera rotation must be orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation_camera_in_calibration_parent)), 1.0, abs_tol=1.0e-5):
        raise ValueError("D405 EE-to-camera rotation must have determinant +1.")
    translation_camera_in_calibration_parent = np.asarray(
        config.translation_camera_in_calibration_parent_m, dtype=float
    )
    translation_camera_in_link7 = (
        np.asarray(config.calibration_parent_position_in_link7_m, dtype=float)
        + translation_camera_in_calibration_parent
    )
    quaternion_wxyz = _rotation_matrix_to_quaternion_wxyz(rotation_camera_in_calibration_parent)
    return tuple(float(value) for value in translation_camera_in_link7), quaternion_wxyz


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized WXYZ quaternion."""

    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)
