"""ZED Mini profile and aligned RGB-D preprocessing, independent of RealSense."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_ZED_PROFILE = Path(__file__).resolve().parents[2] / "configs/franka_zed_mini.json"


def load_zed_profile(path=DEFAULT_ZED_PROFILE) -> dict:
    profile = json.loads(Path(path).read_text())
    if profile.get("model") != "ZED Mini" or profile.get("schema_version") != 1:
        raise ValueError("Expected ZED Mini profile schema 1")
    if profile.get("image_stream") != "left_rectified" or profile.get("convention") != "ros":
        raise ValueError("Training expects rectified LEFT RGB and optical-frame aligned metric depth")
    if profile.get("parent_link") != "panda_hand":
        raise ValueError("This task expects the supplied mount relative to panda_hand")
    if not 0 < profile["depth_min_m"] < profile["depth_max_m"]:
        raise ValueError("Invalid depth range")
    for key in ("source_width", "source_height", "render_width", "render_height", "fx", "fy"):
        if not np.isfinite(profile[key]) or profile[key] <= 0:
            raise ValueError(f"Invalid camera value {key}")
    if (profile["observation_width"], profile["observation_height"]) != (128, 72):
        raise ValueError("The shared visual policy expects 128x72 observations")
    quat = np.asarray(profile["quaternion_wxyz"])
    if quat.shape != (4,) or not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
        raise ValueError("Camera quaternion must be unit WXYZ")
    if not np.isfinite([profile["cx"], profile["cy"]]).all():
        raise ValueError("Invalid camera principal point")
    if not 0.04 < float(profile["stereo_baseline_m"]) < 0.09:
        raise ValueError("Baseline is inconsistent with the ZED Mini; check the camera identity")
    for key, shape in (("position_m", (3,)), ("tcp_offset_in_hand_m", (3,))):
        value = np.asarray(profile[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}")
    return profile


def profile_id(profile: dict) -> str:
    return "zed_mini_" + hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:16]


def scaled_intrinsics(profile: dict, width: int, height: int) -> list[float]:
    sx, sy = width / profile["source_width"], height / profile["source_height"]
    return [profile["fx"] * sx, 0.0, profile["cx"] * sx, 0.0, profile["fy"] * sy, profile["cy"] * sy, 0.0, 0.0, 1.0]


def pack_zed_rgbd(rgb: torch.Tensor, depth: torch.Tensor, profile: dict):
    """Area resize with invalid-depth exclusion; invalid/far is normalized 1.

    Input is rectified LEFT RGB (uint8 or float [0,1]) and aligned optical-Z
    depth in metres. This does not run stereo matching or rectify raw images.
    """
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise ValueError("Expected batched NHWC RGB")
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)
    if depth.shape[:3] != rgb.shape[:3] or depth.shape[-1] != 1:
        raise ValueError("Depth must be aligned with the LEFT RGB image")
    color = rgb[..., :3].float() / (255.0 if rgb.dtype == torch.uint8 else 1.0)
    lo, hi = profile["depth_min_m"], profile["depth_max_m"]
    valid = torch.isfinite(depth) & (depth >= lo) & (depth < hi)
    size = (profile["observation_height"], profile["observation_width"])

    def resize(x):
        return F.interpolate(x.permute(0, 3, 1, 2), size=size, mode="area").permute(0, 2, 3, 1)

    coverage = resize(valid.float())
    metric = resize(torch.where(valid, depth, 0.0)) / coverage.clamp_min(1e-6)
    metric = torch.where(coverage >= 0.25, metric, hi)
    return torch.cat((resize(color).clamp(0, 1), ((metric - lo) / (hi - lo)).clamp(0, 1)), -1), coverage >= 0.25


def reproject_intrinsics(rgb, depth, source_matrix, target_matrix):
    """Resample the same optical rays to calibrated intrinsics (no pose change).

    Isaac renders centered square pixels. This explicitly handles calibrated
    principal points and unequal focal lengths. Optical-Z depth is unchanged;
    nearest sampling avoids blending foreground and background depths.
    """
    n, h, w = rgb.shape[:3]
    if depth.ndim == 3:
        depth = depth[..., None]
    src = torch.as_tensor(source_matrix, device=rgb.device, dtype=torch.float32).reshape(-1, 3, 3)
    dst = torch.as_tensor(target_matrix, device=rgb.device, dtype=torch.float32).reshape(-1, 3, 3)
    y, x = torch.meshgrid(torch.arange(h, device=rgb.device), torch.arange(w, device=rgb.device), indexing="ij")
    u = (x[None] - dst[:, 0, 2, None, None]) / dst[:, 0, 0, None, None]
    v = (y[None] - dst[:, 1, 2, None, None]) / dst[:, 1, 1, None, None]
    u = u * src[:, 0, 0, None, None] + src[:, 0, 2, None, None]
    v = v * src[:, 1, 1, None, None] + src[:, 1, 2, None, None]
    grid = torch.stack((2 * (u + 0.5) / w - 1, 2 * (v + 0.5) / h - 1), -1).expand(n, -1, -1, -1)
    color = rgb[..., :3].float() / (255.0 if rgb.dtype == torch.uint8 else 1.0)
    color = F.grid_sample(color.permute(0, 3, 1, 2), grid, align_corners=False).permute(0, 2, 3, 1)
    metric = F.grid_sample(depth.permute(0, 3, 1, 2), grid, mode="nearest", align_corners=False).permute(0, 2, 3, 1)
    return color, metric


def offset_jacobian(jacobian: torch.Tensor, offset_w: torch.Tensor) -> torch.Tensor:
    """Translate a world-frame body Jacobian to a rigidly attached TCP."""
    result = jacobian.clone()
    angular = jacobian[:, 3:, :].transpose(1, 2)
    result[:, :3, :] += torch.cross(angular, offset_w[:, None, :].expand_as(angular), dim=-1).transpose(1, 2)
    return result


def damped_joint_velocity(jacobian: torch.Tensor, twist: torch.Tensor, damping=0.05):
    identity = torch.eye(6, dtype=jacobian.dtype, device=jacobian.device).expand(jacobian.shape[0], -1, -1)
    return (
        jacobian.transpose(1, 2)
        @ torch.linalg.solve(jacobian @ jacobian.transpose(1, 2) + damping**2 * identity, twist[..., None])
    ).squeeze(-1)
