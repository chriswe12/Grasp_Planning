"""Shared D405 RGB-D preprocessing for simulation and deployment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from grasp_planning.d405_wrist_camera import (
    VISUAL_SERVO_OBSERVATION_HEIGHT,
    VISUAL_SERVO_OBSERVATION_WIDTH,
    D405WristCameraConfig,
)


@dataclass(frozen=True)
class D405ObservationPreprocessCfg:
    """Policy-shape and validity rules shared by live and goal RGB-D."""

    output_height: int = VISUAL_SERVO_OBSERVATION_HEIGHT
    output_width: int = VISUAL_SERVO_OBSERVATION_WIDTH
    valid_depth_min_m: float = 0.07
    valid_depth_max_m: float = 0.50
    normalization_min_m: float = 0.04
    normalization_max_m: float = 0.50
    minimum_valid_area_fraction: float = 0.25

    @classmethod
    def from_camera(cls, camera: D405WristCameraConfig) -> "D405ObservationPreprocessCfg":
        return cls(
            valid_depth_min_m=float(camera.reliable_depth_range_m[0]),
            valid_depth_max_m=float(camera.reliable_depth_range_m[1]),
        )

    def validate(self) -> None:
        if self.output_height < 1 or self.output_width < 1:
            raise ValueError("D405 policy image dimensions must be positive.")
        if not 0.0 < self.valid_depth_min_m < self.valid_depth_max_m:
            raise ValueError("D405 valid depth range must be positive and ordered.")
        if not 0.0 <= self.normalization_min_m < self.normalization_max_m:
            raise ValueError("D405 normalization range must be ordered.")
        if not 0.0 <= self.minimum_valid_area_fraction <= 1.0:
            raise ValueError("minimum_valid_area_fraction must lie in [0, 1].")


def resize_aligned_rgbd_torch(
    rgb: torch.Tensor,
    depth_m: torch.Tensor,
    *,
    cfg: D405ObservationPreprocessCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resize aligned RGB-D while preventing invalid depth from contaminating valid pixels.

    Args:
        rgb: Float RGB in ``[N,H,W,3]`` and range ``[0,1]``.
        depth_m: Metric depth in ``[N,H,W]`` or ``[N,H,W,1]``. Zero and
            non-finite values are treated as invalid, matching RealSense Z16.

    Returns:
        Resized RGB, metric depth with invalid pixels filled by the configured
        maximum, and a boolean validity tensor.
    """

    cfg.validate()
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB [N,H,W,3], got {tuple(rgb.shape)}.")
    if depth_m.ndim == 3:
        depth_m = depth_m.unsqueeze(-1)
    if depth_m.ndim != 4 or depth_m.shape[-1] != 1:
        raise ValueError(f"Expected depth [N,H,W,1], got {tuple(depth_m.shape)}.")
    if rgb.shape[:3] != depth_m.shape[:3]:
        raise ValueError("Aligned D405 RGB and depth shapes must match.")

    output_size = (int(cfg.output_height), int(cfg.output_width))
    rgb_resized = F.interpolate(
        rgb.permute(0, 3, 1, 2),
        size=output_size,
        mode="area",
    ).permute(0, 2, 3, 1)

    # The upper endpoint is the policy's invalid/far sentinel. Goal capture
    # also converts non-finite RTX depth to exactly 0.50 m.
    valid = (
        torch.isfinite(depth_m) & (depth_m >= float(cfg.valid_depth_min_m)) & (depth_m < float(cfg.valid_depth_max_m))
    )
    safe_depth = torch.where(valid, depth_m, torch.zeros_like(depth_m))
    depth_sum = F.interpolate(
        safe_depth.permute(0, 3, 1, 2),
        size=output_size,
        mode="area",
    ).permute(0, 2, 3, 1)
    valid_fraction = F.interpolate(
        valid.float().permute(0, 3, 1, 2),
        size=output_size,
        mode="area",
    ).permute(0, 2, 3, 1)
    resized_valid = valid_fraction >= float(cfg.minimum_valid_area_fraction)
    depth_resized = depth_sum / valid_fraction.clamp_min(1.0e-6)
    depth_resized = torch.where(
        resized_valid,
        depth_resized,
        depth_resized.new_full((), float(cfg.valid_depth_max_m)),
    ).clamp(float(cfg.valid_depth_min_m), float(cfg.valid_depth_max_m))
    return rgb_resized.clamp(0.0, 1.0), depth_resized, resized_valid


def normalize_depth_torch(depth_m: torch.Tensor, *, cfg: D405ObservationPreprocessCfg) -> torch.Tensor:
    """Normalize metric depth to the policy's stable ``[0,1]`` interval."""

    scale = float(cfg.normalization_max_m - cfg.normalization_min_m)
    return depth_m.sub(float(cfg.normalization_min_m)).div(scale).clamp(0.0, 1.0)


def pack_policy_rgbd_torch(
    rgb: torch.Tensor,
    depth_m: torch.Tensor,
    *,
    cfg: D405ObservationPreprocessCfg,
) -> torch.Tensor:
    """Pack preprocessed RGB and metric depth into the existing four-channel contract."""

    return torch.cat((rgb.clamp(0.0, 1.0), normalize_depth_torch(depth_m, cfg=cfg)), dim=-1)


def preprocess_aligned_rgbd_torch(
    rgb: torch.Tensor,
    depth_m: torch.Tensor,
    *,
    cfg: D405ObservationPreprocessCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize and pack already aligned, rectified RGB-D for actor input."""

    rgb_resized, depth_resized, valid = resize_aligned_rgbd_torch(rgb, depth_m, cfg=cfg)
    return pack_policy_rgbd_torch(rgb_resized, depth_resized, cfg=cfg), valid


def rectify_aligned_rgbd_numpy(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    *,
    camera: D405WristCameraConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify real color-aligned D405 arrays to the pinhole convention used by Isaac.

    Depth must already be aligned into the color optical frame by librealsense
    or the ROS wrapper. RGB uses linear interpolation; metric depth uses nearest
    interpolation so invalid and discontinuous measurements are not blended.
    """

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - deployment dependency
        raise RuntimeError("OpenCV is required to rectify real D405 frames.") from error

    if rgb.shape[:2] != depth_m.shape[:2]:
        raise ValueError("Real D405 depth must be aligned to the color frame before rectification.")
    height, width = rgb.shape[:2]
    intrinsic = np.asarray(camera.intrinsic_matrix_row_major, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera.distortion_coefficients, dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsic,
        distortion,
        np.eye(3, dtype=np.float64),
        intrinsic,
        (width, height),
        cv2.CV_32FC1,
    )
    rectified_rgb = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    rectified_depth = cv2.remap(
        depth_m.astype(np.float32, copy=False),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return rectified_rgb, rectified_depth


__all__ = [
    "D405ObservationPreprocessCfg",
    "normalize_depth_torch",
    "pack_policy_rgbd_torch",
    "preprocess_aligned_rgbd_torch",
    "rectify_aligned_rgbd_numpy",
    "resize_aligned_rgbd_torch",
]
