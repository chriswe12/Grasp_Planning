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
    "d405_color_848x480_intrinsics_260322275185_render_256x144_hand_eye_link7_ee35mm_z180_v7"
)
D405_VISUAL_SERVO_OBSERVATION_PROFILE = "rgbd_render_256x144_valid_area_128x72_d405_range_v3"


@dataclass(frozen=True)
class D405WristCameraConfig:
    """Deployable camera geometry plus simulation observation settings."""

    enabled: bool = False
    parent_prim_path: str = "link7"
    width: int = 848
    height: int = 480
    # CameraInfo from realsense_1 color optical frame, 848x480.  These are
    # the deployed K values, not the previous FOV-derived placeholder.
    fx: float = 436.3104248046875
    fy: float = 435.6492614746094
    cx: float = 418.62664794921875
    cy: float = 236.5121307373047
    distortion_model: str = "plumb_bob"
    distortion_coefficients: tuple[float, float, float, float, float] = (
        -0.05201759934425354,
        0.05433472618460655,
        0.0002693705027922988,
        0.0008704775245860219,
        -0.017724450677633286,
    )
    clipping_range_m: tuple[float, float] = (0.04, 2.0)
    # Provisional documented D405/D400 defaults.  The D405 product range is
    # 7--50 cm and its stereo baseline is 18 mm.  RealSense recommends 100 um
    # depth units at close range; replace this with the queried device setting
    # once a real capture profile is available.
    stereo_baseline_m: float = 0.018
    reliable_depth_range_m: tuple[float, float] = (0.07, 0.50)
    depth_unit_m: float = 0.0001
    update_period_s: float = 0.0
    include_privileged_mask: bool = True
    # Hand-eye calibration for RealSense serial 260322275185 maps the camera
    # optical frame into MoveIt's lbr_link_ee.  MoveIt defines lbr_link_ee as
    # a fixed child of link7 at +35 mm local Z, with identity rotation.
    calibration_parent_position_in_link7_m: tuple[float, float, float] = (0.0, 0.0, 0.035)
    # T_link7_cam, with p_link7 = R @ p_camera_optical + t.  R is flattened
    # row-major exactly as in the supplied calibration YAML.
    rotation_camera_in_calibration_parent: tuple[float, ...] = (
        0.002321764157194206,
        -0.8654223294953782,
        0.5010377241505786,
        0.9994952228900846,
        -0.013866922434525808,
        -0.028583349735374547,
        0.03168452037033634,
        0.5008511755731314,
        0.8649532883895603,
    )
    translation_camera_in_calibration_parent_m: tuple[float, float, float] = (
        0.0556666756801677,
        0.008999999999999992,
        0.07077648943349073,
    )
    # The calibration-parent axes in the deployed MoveIt mount are rotated
    # 180 degrees about the flange/tool Z axis relative to the generated Isaac
    # robot USD. The translation above is the visually adjusted final position
    # in lbr_link_ee, so this correction applies only to the optical orientation.
    mount_correction_about_parent_z_deg: float = 180.0

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

    translation_camera_in_calibration_parent = np.asarray(
        config.translation_camera_in_calibration_parent_m, dtype=float
    )
    rotation_camera_in_link7 = camera_rotation_in_link7(config)
    translation_camera_in_link7 = (
        np.asarray(config.calibration_parent_position_in_link7_m, dtype=float)
        + translation_camera_in_calibration_parent
    )
    quaternion_wxyz = _rotation_matrix_to_quaternion_wxyz(rotation_camera_in_link7)
    return tuple(float(value) for value in translation_camera_in_link7), quaternion_wxyz


def camera_rotation_in_link7(config: D405WristCameraConfig) -> np.ndarray:
    """Return the corrected camera-to-link7 rotation used by render and control."""

    calibrated_rotation = np.asarray(config.rotation_camera_in_calibration_parent, dtype=float).reshape(3, 3)
    if not np.allclose(calibrated_rotation.T @ calibrated_rotation, np.eye(3), atol=1.0e-5):
        raise ValueError("D405 EE-to-camera rotation must be orthonormal.")
    if not math.isclose(float(np.linalg.det(calibrated_rotation)), 1.0, abs_tol=1.0e-5):
        raise ValueError("D405 EE-to-camera rotation must have determinant +1.")
    angle_rad = math.radians(float(config.mount_correction_about_parent_z_deg))
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    correction = np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=float,
    )
    return correction @ calibrated_rotation


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
