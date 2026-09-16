"""Vectorized live-only RGB-D domain randomization for visual-servo training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LiveObservationRandomizationCfg:
    """Ranges for episode-stable appearance and per-frame sensor noise."""

    enabled: bool = True
    exposure_stops: tuple[float, float] = (-0.30, 0.30)
    contrast: tuple[float, float] = (0.85, 1.15)
    gamma: tuple[float, float] = (0.90, 1.10)
    white_balance_gain: tuple[float, float] = (0.93, 1.07)
    vignette_strength: tuple[float, float] = (0.0, 0.14)
    rgb_noise_std: tuple[float, float] = (0.0, 0.015)
    blur_probability: float = 0.12
    blur_mix: tuple[float, float] = (0.25, 0.60)
    depth_scale: tuple[float, float] = (0.99, 1.01)
    depth_bias_m: tuple[float, float] = (-0.002, 0.002)
    depth_noise_std_m: tuple[float, float] = (0.0, 0.0002)
    # Provisional D405 stereo model. Official D405 baseline is 18 mm; D400
    # disparity has 1/32-pixel subpixel resolution. These conservative noise
    # ranges are replaceable once device-specific plane captures are analyzed.
    correlated_depth_enabled: bool = True
    stereo_focal_length_px: float = 436.3104248046875
    stereo_baseline_m: float = 0.018
    disparity_bias_px: tuple[float, float] = (-0.04, 0.04)
    disparity_independent_noise_std_px: tuple[float, float] = (0.01, 0.03)
    disparity_spatial_noise_std_px: tuple[float, float] = (0.02, 0.07)
    disparity_temporal_noise_std_px: tuple[float, float] = (0.01, 0.04)
    disparity_temporal_correlation: tuple[float, float] = (0.60, 0.92)
    spatial_field_shape: tuple[int, int] = (9, 16)
    stereo_edge_mismatch_probability: float = 0.12
    stereo_edge_horizontal_radius_px: int = 2
    depth_quantization_m: float = 0.0001
    depth_dropout_probability: tuple[float, float] = (0.0, 0.004)
    depth_edge_dropout_probability: tuple[float, float] = (0.0, 0.035)
    depth_edge_threshold_m: float = 0.008
    calibration_warp_enabled: bool = True
    calibration_shift_x_px: tuple[float, float] = (-1.5, 1.5)
    calibration_shift_y_px: tuple[float, float] = (-1.0, 1.0)
    calibration_scale: tuple[float, float] = (0.99, 1.01)
    calibration_roll_deg: tuple[float, float] = (-1.0, 1.0)
    clean_episode_fraction: float = 0.0
    rgb_patch_occlusion_probability: float = 0.06
    depth_patch_dropout_probability: float = 0.04
    patch_area_fraction: tuple[float, float] = (0.005, 0.03)
    depth_min_m: float = 0.07
    depth_max_m: float = 0.50

    def validate(self) -> None:
        for name in (
            "exposure_stops",
            "contrast",
            "gamma",
            "white_balance_gain",
            "vignette_strength",
            "rgb_noise_std",
            "blur_mix",
            "depth_scale",
            "depth_bias_m",
            "depth_noise_std_m",
            "disparity_bias_px",
            "disparity_independent_noise_std_px",
            "disparity_spatial_noise_std_px",
            "disparity_temporal_noise_std_px",
            "disparity_temporal_correlation",
            "depth_dropout_probability",
            "depth_edge_dropout_probability",
            "calibration_shift_x_px",
            "calibration_shift_y_px",
            "calibration_scale",
            "calibration_roll_deg",
        ):
            lower, upper = getattr(self, name)
            if lower > upper:
                raise ValueError(f"{name} must be ordered, got {(lower, upper)}.")
        if not 0.0 <= self.blur_probability <= 1.0:
            raise ValueError("blur_probability must be in [0, 1].")
        for name in (
            "rgb_patch_occlusion_probability",
            "depth_patch_dropout_probability",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if not 0.0 <= self.patch_area_fraction[0] <= self.patch_area_fraction[1] <= 0.25:
            raise ValueError("patch_area_fraction must be ordered inside [0, 0.25].")
        if self.depth_quantization_m < 0.0:
            raise ValueError("depth_quantization_m cannot be negative.")
        if self.stereo_focal_length_px <= 0.0 or self.stereo_baseline_m <= 0.0:
            raise ValueError("Stereo focal length and baseline must be positive.")
        if any(value < 0.0 or value >= 1.0 for value in self.disparity_temporal_correlation):
            raise ValueError("Temporal disparity correlation must lie in [0, 1).")
        if len(self.spatial_field_shape) != 2 or min(self.spatial_field_shape) < 2:
            raise ValueError("spatial_field_shape must contain two dimensions >= 2.")
        if not 0.0 <= self.stereo_edge_mismatch_probability <= 1.0:
            raise ValueError("stereo_edge_mismatch_probability must lie in [0, 1].")
        if not 0.0 <= self.clean_episode_fraction <= 1.0:
            raise ValueError("clean_episode_fraction must lie in [0, 1].")
        if self.stereo_edge_horizontal_radius_px < 0:
            raise ValueError("stereo_edge_horizontal_radius_px cannot be negative.")
        if self.depth_edge_threshold_m <= 0.0:
            raise ValueError("depth_edge_threshold_m must be positive.")
        if self.depth_min_m >= self.depth_max_m:
            raise ValueError("depth_min_m must be smaller than depth_max_m.")


class LiveObservationRandomizer:
    """Apply per-environment live RGB-D variation while leaving goals untouched.

    Exposure, color response, vignetting, blur, and depth calibration are
    sampled once per episode. Pixel noise and missing depth are sampled on each
    observation to approximate frame-to-frame sensor variation.
    """

    def __init__(
        self,
        cfg: LiveObservationRandomizationCfg,
        *,
        num_envs: int,
        device: torch.device | str,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive.")

        self.exposure_gain = torch.ones((self.num_envs, 1, 1, 1), device=self.device)
        self.contrast = torch.ones_like(self.exposure_gain)
        self.gamma = torch.ones_like(self.exposure_gain)
        self.white_balance_gain = torch.ones((self.num_envs, 1, 1, 3), device=self.device)
        self.vignette_strength = torch.zeros_like(self.exposure_gain)
        self.rgb_noise_std = torch.zeros_like(self.exposure_gain)
        self.blur_mix = torch.zeros_like(self.exposure_gain)
        self.depth_scale = torch.ones_like(self.exposure_gain)
        self.depth_bias_m = torch.zeros_like(self.exposure_gain)
        self.depth_noise_std_m = torch.zeros_like(self.exposure_gain)
        self.disparity_bias_px = torch.zeros_like(self.exposure_gain)
        self.disparity_independent_noise_std_px = torch.zeros_like(self.exposure_gain)
        self.disparity_spatial_noise_std_px = torch.zeros_like(self.exposure_gain)
        self.disparity_temporal_noise_std_px = torch.zeros_like(self.exposure_gain)
        self.disparity_temporal_correlation = torch.zeros_like(self.exposure_gain)
        self.depth_dropout_probability = torch.zeros_like(self.exposure_gain)
        self.depth_edge_dropout_probability = torch.zeros_like(self.exposure_gain)
        self.calibration_shift_x_px = torch.zeros((self.num_envs,), device=self.device)
        self.calibration_shift_y_px = torch.zeros_like(self.calibration_shift_x_px)
        self.calibration_scale = torch.ones_like(self.calibration_shift_x_px)
        self.calibration_roll_rad = torch.zeros_like(self.calibration_shift_x_px)
        field_height, field_width = cfg.spatial_field_shape
        self.disparity_fixed_field = torch.zeros((self.num_envs, 1, field_height, field_width), device=self.device)
        self.disparity_temporal_field = torch.zeros_like(self.disparity_fixed_field)
        self.randomization_strength = torch.ones_like(self.exposure_gain)
        self.rgb_patch_enabled = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.depth_patch_enabled = torch.zeros_like(self.rgb_patch_enabled)
        self.patch_center_x = torch.zeros((self.num_envs,), device=self.device)
        self.patch_center_y = torch.zeros_like(self.patch_center_x)
        self.patch_area_fraction = torch.zeros_like(self.patch_center_x)
        self.patch_aspect_log = torch.zeros_like(self.patch_center_x)
        self._vignette_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.last_disparity_error_abs_mean_px = torch.zeros((), device=self.device)
        self.last_depth_invalid_fraction = torch.zeros((), device=self.device)
        self.sample(torch.arange(self.num_envs, device=self.device))

    def _uniform(self, env_ids: torch.Tensor, value_range: tuple[float, float], *, channels: int = 1) -> torch.Tensor:
        shape = (len(env_ids), 1, 1, channels)
        lower, upper = value_range
        if lower == upper:
            return torch.full(shape, float(lower), device=self.device)
        return torch.empty(shape, device=self.device).uniform_(float(lower), float(upper))

    def sample(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        *,
        strength: float | torch.Tensor = 1.0,
    ) -> None:
        """Sample new episode-level parameters for the selected environments."""

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        strength_tensor = torch.as_tensor(strength, dtype=torch.float32, device=self.device)
        if strength_tensor.ndim == 0:
            strength_tensor = strength_tensor.expand(len(env_ids))
        if strength_tensor.shape != (len(env_ids),) or torch.any((strength_tensor < 0.0) | (strength_tensor > 1.0)):
            raise ValueError("Randomization strength must lie in [0, 1] per environment.")
        if self.cfg.enabled and self.cfg.clean_episode_fraction > 0.0:
            clean = torch.rand(len(env_ids), device=self.device) < float(self.cfg.clean_episode_fraction)
            strength_tensor = torch.where(clean, torch.zeros_like(strength_tensor), strength_tensor)
        self.randomization_strength[env_ids] = strength_tensor.reshape(-1, 1, 1, 1)
        if not self.cfg.enabled:
            self.exposure_gain[env_ids] = 1.0
            self.contrast[env_ids] = 1.0
            self.gamma[env_ids] = 1.0
            self.white_balance_gain[env_ids] = 1.0
            self.vignette_strength[env_ids] = 0.0
            self.rgb_noise_std[env_ids] = 0.0
            self.blur_mix[env_ids] = 0.0
            self.depth_scale[env_ids] = 1.0
            self.depth_bias_m[env_ids] = 0.0
            self.depth_noise_std_m[env_ids] = 0.0
            self.disparity_bias_px[env_ids] = 0.0
            self.disparity_independent_noise_std_px[env_ids] = 0.0
            self.disparity_spatial_noise_std_px[env_ids] = 0.0
            self.disparity_temporal_noise_std_px[env_ids] = 0.0
            self.disparity_temporal_correlation[env_ids] = 0.0
            self.disparity_fixed_field[env_ids] = 0.0
            self.disparity_temporal_field[env_ids] = 0.0
            self.calibration_shift_x_px[env_ids] = 0.0
            self.calibration_shift_y_px[env_ids] = 0.0
            self.calibration_scale[env_ids] = 1.0
            self.calibration_roll_rad[env_ids] = 0.0
            self.depth_dropout_probability[env_ids] = 0.0
            self.depth_edge_dropout_probability[env_ids] = 0.0
            self.randomization_strength[env_ids] = 0.0
            self.rgb_patch_enabled[env_ids] = False
            self.depth_patch_enabled[env_ids] = False
            return

        exposure_stops = self._uniform(env_ids, self.cfg.exposure_stops)
        self.exposure_gain[env_ids] = torch.pow(2.0, exposure_stops)
        self.contrast[env_ids] = self._uniform(env_ids, self.cfg.contrast)
        self.gamma[env_ids] = self._uniform(env_ids, self.cfg.gamma)
        white_balance = self._uniform(env_ids, self.cfg.white_balance_gain, channels=3)
        self.white_balance_gain[env_ids] = white_balance / white_balance.mean(dim=-1, keepdim=True)
        self.vignette_strength[env_ids] = self._uniform(env_ids, self.cfg.vignette_strength)
        self.rgb_noise_std[env_ids] = self._uniform(env_ids, self.cfg.rgb_noise_std)
        blur_enabled = torch.rand((len(env_ids), 1, 1, 1), device=self.device) < self.cfg.blur_probability
        self.blur_mix[env_ids] = self._uniform(env_ids, self.cfg.blur_mix) * blur_enabled
        self.depth_scale[env_ids] = self._uniform(env_ids, self.cfg.depth_scale)
        self.depth_bias_m[env_ids] = self._uniform(env_ids, self.cfg.depth_bias_m)
        self.depth_noise_std_m[env_ids] = self._uniform(env_ids, self.cfg.depth_noise_std_m)
        self.disparity_bias_px[env_ids] = self._uniform(env_ids, self.cfg.disparity_bias_px)
        self.disparity_independent_noise_std_px[env_ids] = self._uniform(
            env_ids, self.cfg.disparity_independent_noise_std_px
        )
        self.disparity_spatial_noise_std_px[env_ids] = self._uniform(env_ids, self.cfg.disparity_spatial_noise_std_px)
        self.disparity_temporal_noise_std_px[env_ids] = self._uniform(env_ids, self.cfg.disparity_temporal_noise_std_px)
        self.disparity_temporal_correlation[env_ids] = self._uniform(env_ids, self.cfg.disparity_temporal_correlation)
        self.disparity_fixed_field[env_ids] = torch.randn_like(self.disparity_fixed_field[env_ids])
        self.disparity_temporal_field[env_ids] = torch.randn_like(self.disparity_temporal_field[env_ids])
        self.depth_dropout_probability[env_ids] = self._uniform(env_ids, self.cfg.depth_dropout_probability)
        self.depth_edge_dropout_probability[env_ids] = self._uniform(env_ids, self.cfg.depth_edge_dropout_probability)
        if self.cfg.calibration_warp_enabled:
            self.calibration_shift_x_px[env_ids] = self._uniform(env_ids, self.cfg.calibration_shift_x_px).flatten()
            self.calibration_shift_y_px[env_ids] = self._uniform(env_ids, self.cfg.calibration_shift_y_px).flatten()
            self.calibration_scale[env_ids] = self._uniform(env_ids, self.cfg.calibration_scale).flatten()
            roll_deg = self._uniform(env_ids, self.cfg.calibration_roll_deg).flatten()
            self.calibration_roll_rad[env_ids] = torch.deg2rad(roll_deg)
        else:
            self.calibration_shift_x_px[env_ids] = 0.0
            self.calibration_shift_y_px[env_ids] = 0.0
            self.calibration_scale[env_ids] = 1.0
            self.calibration_roll_rad[env_ids] = 0.0
        strength_flat = strength_tensor
        self.rgb_patch_enabled[env_ids] = torch.rand(len(env_ids), device=self.device) < (
            self.cfg.rgb_patch_occlusion_probability * strength_flat
        )
        self.depth_patch_enabled[env_ids] = torch.rand(len(env_ids), device=self.device) < (
            self.cfg.depth_patch_dropout_probability * strength_flat
        )
        self.patch_center_x[env_ids] = torch.rand(len(env_ids), device=self.device)
        self.patch_center_y[env_ids] = torch.rand(len(env_ids), device=self.device)
        lower, upper = self.cfg.patch_area_fraction
        self.patch_area_fraction[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(lower, upper)
        self.patch_aspect_log[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-0.7, 0.7)

    def _vignette(self, height: int, width: int) -> torch.Tensor:
        key = (height, width)
        if key not in self._vignette_cache:
            y = torch.linspace(-1.0, 1.0, height, device=self.device)
            x = torch.linspace(-1.0, 1.0, width, device=self.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            radius_squared = (xx.square() + yy.square()).clamp(0.0, 1.0)
            self._vignette_cache[key] = radius_squared.reshape(1, height, width, 1)
        return self._vignette_cache[key]

    @staticmethod
    def _depth_edges(depth_m: torch.Tensor, threshold_m: float) -> torch.Tensor:
        horizontal = F.pad(
            (depth_m[:, :, 1:, :] - depth_m[:, :, :-1, :]).abs(),
            (0, 0, 0, 1),
        )
        vertical = F.pad(
            (depth_m[:, 1:, :, :] - depth_m[:, :-1, :, :]).abs(),
            (0, 0, 0, 0, 0, 1),
        )
        return ((horizontal + vertical) / float(threshold_m)).clamp(0.0, 1.0)

    def _warp_live_rgbd(self, rgb: torch.Tensor, depth_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one small episode-stable calibration warp to both modalities."""

        if not self.cfg.calibration_warp_enabled:
            return rgb, depth_m
        height, width = rgb.shape[1:3]
        cosine = torch.cos(self.calibration_roll_rad) / self.calibration_scale
        sine = torch.sin(self.calibration_roll_rad) / self.calibration_scale
        theta = torch.zeros((self.num_envs, 2, 3), device=self.device, dtype=rgb.dtype)
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        theta[:, 0, 2] = 2.0 * self.calibration_shift_x_px / max(width - 1, 1)
        theta[:, 1, 2] = 2.0 * self.calibration_shift_y_px / max(height - 1, 1)
        grid = F.affine_grid(
            theta,
            size=(self.num_envs, 1, height, width),
            align_corners=False,
        )
        warped_rgb = F.grid_sample(
            rgb.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        warped_depth = F.grid_sample(
            depth_m.permute(0, 3, 1, 2),
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        return warped_rgb, warped_depth

    def _apply_correlated_disparity_error(self, depth_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply episode, spatial, temporal, and pixel stereo errors in disparity."""

        if not self.cfg.correlated_depth_enabled:
            return depth_m, torch.zeros_like(depth_m)
        height, width = depth_m.shape[1:3]
        rho = self.disparity_temporal_correlation
        innovation_scale = torch.sqrt((1.0 - rho.square()).clamp_min(0.0))
        self.disparity_temporal_field.mul_(rho).add_(torch.randn_like(self.disparity_temporal_field) * innovation_scale)
        fixed_field = F.interpolate(
            self.disparity_fixed_field,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        temporal_field = F.interpolate(
            self.disparity_temporal_field,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        disparity_error = (
            self.disparity_bias_px
            + fixed_field * self.disparity_spatial_noise_std_px
            + temporal_field * self.disparity_temporal_noise_std_px
            + torch.randn_like(depth_m) * self.disparity_independent_noise_std_px
        )
        focal_baseline = float(self.cfg.stereo_focal_length_px * self.cfg.stereo_baseline_m)
        disparity = focal_baseline / depth_m.clamp_min(float(self.cfg.depth_min_m))
        minimum_disparity = focal_baseline / float(self.cfg.depth_max_m)
        maximum_disparity = focal_baseline / float(self.cfg.depth_min_m)
        noisy_disparity = (disparity + disparity_error).clamp(minimum_disparity, maximum_disparity)
        return focal_baseline / noisy_disparity, disparity_error

    def _apply_stereo_edge_mismatch(self, original_depth: torch.Tensor, randomized_depth: torch.Tensor) -> torch.Tensor:
        """Mix foreground, background, and invalid values around stereo boundaries."""

        probability = float(self.cfg.stereo_edge_mismatch_probability)
        if probability <= 0.0:
            return randomized_depth
        left = torch.cat((original_depth[:, :, :1], original_depth[:, :, :-1]), dim=2)
        right = torch.cat((original_depth[:, :, 1:], original_depth[:, :, -1:]), dim=2)
        edge_strength = (
            torch.maximum(
                (original_depth - left).abs(),
                (original_depth - right).abs(),
            )
            .div(float(self.cfg.depth_edge_threshold_m))
            .clamp(0.0, 1.0)
        )
        radius = int(self.cfg.stereo_edge_horizontal_radius_px)
        if radius > 0:
            edge_strength = F.max_pool2d(
                edge_strength.permute(0, 3, 1, 2),
                kernel_size=(1, 2 * radius + 1),
                stride=1,
                padding=(0, radius),
            ).permute(0, 2, 3, 1)
        mismatch = torch.rand_like(edge_strength) < edge_strength * probability
        near_depth = torch.minimum(original_depth, torch.minimum(left, right))
        far_depth = torch.maximum(original_depth, torch.maximum(left, right))
        selection = torch.rand_like(edge_strength)
        replacement = torch.where(
            selection < 0.50,
            randomized_depth.new_full((), float(self.cfg.depth_max_m)),
            torch.where(selection < 0.75, near_depth, far_depth),
        )
        return torch.where(mismatch, replacement, randomized_depth)

    def apply(self, rgb: torch.Tensor, depth_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomize live tensors shaped ``[env, height, width, channels]``."""

        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB [N,H,W,3], got {tuple(rgb.shape)}.")
        if depth_m.ndim != 4 or depth_m.shape[-1] != 1:
            raise ValueError(f"Expected depth [N,H,W,1], got {tuple(depth_m.shape)}.")
        if rgb.shape[:3] != depth_m.shape[:3] or rgb.shape[0] != self.num_envs:
            raise ValueError("RGB/depth shapes must agree and contain every environment.")
        if not self.cfg.enabled:
            return rgb, depth_m

        original_rgb = rgb
        original_depth = depth_m
        warped_rgb, warped_depth = self._warp_live_rgbd(rgb, depth_m)
        rgb_mean = warped_rgb.mean(dim=(1, 2), keepdim=True)
        randomized_rgb = (warped_rgb - rgb_mean) * self.contrast + rgb_mean
        randomized_rgb = randomized_rgb * self.exposure_gain * self.white_balance_gain
        randomized_rgb = randomized_rgb.clamp(0.0, 1.0).pow(self.gamma)
        randomized_rgb = randomized_rgb * (
            1.0 - self.vignette_strength * self._vignette(randomized_rgb.shape[1], randomized_rgb.shape[2])
        )
        rgb_channels_first = randomized_rgb.permute(0, 3, 1, 2)
        blurred = F.avg_pool2d(
            F.pad(rgb_channels_first, (1, 1, 1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        ).permute(0, 2, 3, 1)
        randomized_rgb = torch.lerp(randomized_rgb, blurred, self.blur_mix)
        randomized_rgb = (randomized_rgb + torch.randn_like(randomized_rgb) * self.rgb_noise_std).clamp(0.0, 1.0)

        edge_strength = self._depth_edges(warped_depth, self.cfg.depth_edge_threshold_m)
        randomized_depth, disparity_error = self._apply_correlated_disparity_error(warped_depth)
        randomized_depth = randomized_depth * self.depth_scale + self.depth_bias_m
        depth_noise_scale = self.depth_noise_std_m * (1.0 + edge_strength)
        randomized_depth = randomized_depth + torch.randn_like(randomized_depth) * depth_noise_scale
        randomized_depth = self._apply_stereo_edge_mismatch(warped_depth, randomized_depth)
        if self.cfg.depth_quantization_m > 0.0:
            quantum = float(self.cfg.depth_quantization_m)
            randomized_depth = torch.round(randomized_depth / quantum) * quantum
        dropout_probability = (
            self.depth_dropout_probability + self.depth_edge_dropout_probability * edge_strength
        ).clamp(0.0, 1.0)
        missing = torch.rand_like(randomized_depth) < dropout_probability
        randomized_depth = torch.where(
            missing,
            randomized_depth.new_full((), self.cfg.depth_max_m),
            randomized_depth,
        ).clamp(self.cfg.depth_min_m, self.cfg.depth_max_m)
        outside_reliable_range = (
            (warped_depth < float(self.cfg.depth_min_m))
            | (warped_depth >= float(self.cfg.depth_max_m))
            | ~torch.isfinite(warped_depth)
        )
        randomized_depth = torch.where(
            outside_reliable_range,
            randomized_depth.new_full((), float(self.cfg.depth_max_m)),
            randomized_depth,
        )
        height, width = rgb.shape[1:3]
        aspect = torch.exp(self.patch_aspect_log)
        patch_height = torch.round(height * torch.sqrt(self.patch_area_fraction / aspect)).long()
        patch_width = torch.round(width * torch.sqrt(self.patch_area_fraction * aspect)).long()
        patch_height.clamp_(1, height)
        patch_width.clamp_(1, width)
        center_y = torch.round(self.patch_center_y * (height - 1)).long()
        center_x = torch.round(self.patch_center_x * (width - 1)).long()
        y0 = (center_y - patch_height // 2).clamp_min(0)
        x0 = (center_x - patch_width // 2).clamp_min(0)
        y0 = torch.minimum(y0, height - patch_height)
        x0 = torch.minimum(x0, width - patch_width)
        y_coordinates = torch.arange(height, device=self.device).view(1, height, 1, 1)
        x_coordinates = torch.arange(width, device=self.device).view(1, 1, width, 1)
        patch_mask = (
            (y_coordinates >= y0.view(-1, 1, 1, 1))
            & (y_coordinates < (y0 + patch_height).view(-1, 1, 1, 1))
            & (x_coordinates >= x0.view(-1, 1, 1, 1))
            & (x_coordinates < (x0 + patch_width).view(-1, 1, 1, 1))
        )
        randomized_rgb = torch.where(
            patch_mask & self.rgb_patch_enabled.view(-1, 1, 1, 1),
            rgb_mean,
            randomized_rgb,
        )
        randomized_depth = torch.where(
            patch_mask & self.depth_patch_enabled.view(-1, 1, 1, 1),
            randomized_depth.new_full((), self.cfg.depth_max_m),
            randomized_depth,
        )

        strength = self.randomization_strength
        self.last_disparity_error_abs_mean_px = disparity_error.abs().mean().detach()
        self.last_depth_invalid_fraction = (randomized_depth >= float(self.cfg.depth_max_m)).float().mean().detach()
        return (
            torch.lerp(original_rgb, randomized_rgb, strength),
            torch.lerp(original_depth, randomized_depth, strength),
        )


__all__ = ["LiveObservationRandomizationCfg", "LiveObservationRandomizer"]
