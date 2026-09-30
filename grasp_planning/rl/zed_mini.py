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
    return validate_zed_profile(json.loads(Path(path).read_text()))


def validate_zed_profile(profile: dict) -> dict:
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
    if (profile["observation_width"], profile["observation_height"]) not in ((128, 72), (256, 144), (384, 216)):
        raise ValueError("Unsupported native policy resolution; use 128x72, 256x144 or 384x216")
    if (
        profile["render_width"] < profile["observation_width"]
        or profile["render_height"] < profile["observation_height"]
    ):
        raise ValueError("Render resolution must cover native observation detail")
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


def resolve_zed_profile(contract: dict, path=None) -> dict:
    """Resolve self-contained new catalogs while retaining the legacy profile.

    Explicit overrides must match the catalog hash; camera changes require
    regenerating reference images rather than relabeling old observations.
    """
    embedded = contract.get("camera_profile_data")
    profile = (
        load_zed_profile(path)
        if path
        else (validate_zed_profile(embedded) if embedded is not None else load_zed_profile())
    )
    if contract.get("camera_profile", profile_id(profile)) != profile_id(profile):
        raise ValueError("Camera profile differs from training catalog")
    if embedded is not None and profile_id(embedded) != profile_id(profile):
        raise ValueError("Embedded camera profile differs from explicit override")
    return profile


def scaled_intrinsics(profile: dict, width: int, height: int) -> list[float]:
    sx, sy = width / profile["source_width"], height / profile["source_height"]
    return [profile["fx"] * sx, 0.0, profile["cx"] * sx, 0.0, profile["fy"] * sy, profile["cy"] * sy, 0.0, 0.0, 1.0]


def pack_zed_rgbd(rgb: torch.Tensor, depth: torch.Tensor, profile: dict, *, legacy_batch_layout=False):
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
    # grid_sample -> NHWC can carry a singleton-channel stride of H*W.
    # CUDA adaptive-area pooling then misreads later batch elements. Reset that
    # arbitrary singleton stride explicitly; contiguous() alone is a no-op here.
    if not legacy_batch_layout:
        depth = depth.squeeze(-1).unsqueeze(-1)
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


def optical_depth_from_radial(radial: torch.Tensor, intrinsics: torch.Tensor):
    """Convert per-ray distance to optical Z using each tile's actual intrinsics.

    This explicit optical-Z conversion is independently checked against MuJoCo.
    """
    if radial.ndim != 4 or radial.shape[-1] != 1:
        raise ValueError("Expected NHW1 radial depth")
    n, h, w, _ = radial.shape
    k = torch.as_tensor(intrinsics, device=radial.device, dtype=radial.dtype).reshape(-1, 3, 3)
    if k.shape[0] not in (1, n) or not torch.all(k[:, [0, 1], [0, 1]] > 0):
        raise ValueError("Invalid per-camera intrinsics")
    y, x = torch.meshgrid(torch.arange(h, device=radial.device), torch.arange(w, device=radial.device), indexing="ij")
    u = (x[None] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None]
    v = (y[None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None]
    return radial / torch.sqrt(1 + u * u + v * v)[..., None]


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
