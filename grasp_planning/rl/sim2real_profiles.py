"""Named, reproducible sim-to-real randomization profiles.

The values in the documented D405 profiles are deliberately provisional. They
come from Intel's published D405/D400 geometry and conservative engineering
bounds, not from measurements of the two project cameras.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SIM2REAL_PROFILE_VERSION = "d405_documented_provisional_v6_15hz"
SIM2REAL_PROFILE_NAMES = (
    "nominal",
    "sensor_only",
    "camera_uncertainty",
    "timing_control",
    "appearance",
    "combined_sim2real",
    "combined_clutter",
    "combined_busy_background",
    "combined_depth_robust",
    "stress_test",
)


@dataclass(frozen=True)
class Sim2RealProfile:
    """One immutable set of environment-configuration overrides."""

    name: str
    overrides: Mapping[str, Any]
    description: str

    @property
    def identifier(self) -> str:
        return f"{SIM2REAL_PROFILE_VERSION}:{self.name}"


_SENSOR_IDENTITY: dict[str, Any] = {
    "live_rgb_exposure_stops": (0.0, 0.0),
    "live_rgb_contrast": (1.0, 1.0),
    "live_rgb_gamma": (1.0, 1.0),
    "live_rgb_white_balance_gain": (1.0, 1.0),
    "live_rgb_vignette_strength": (0.0, 0.0),
    "live_rgb_noise_std": (0.0, 0.0),
    "live_rgb_blur_probability": 0.0,
    "live_rgb_blur_mix": (0.0, 0.0),
    "live_depth_scale": (1.0, 1.0),
    "live_depth_bias_m": (0.0, 0.0),
    "live_depth_noise_std_m": (0.0, 0.0),
    "live_correlated_depth_enabled": False,
    "live_disparity_bias_px": (0.0, 0.0),
    "live_disparity_independent_noise_std_px": (0.0, 0.0),
    "live_disparity_spatial_noise_std_px": (0.0, 0.0),
    "live_disparity_temporal_noise_std_px": (0.0, 0.0),
    "live_stereo_edge_mismatch_probability": 0.0,
    "live_depth_dropout_probability": (0.0, 0.0),
    "live_depth_edge_dropout_probability": (0.0, 0.0),
    "live_rgb_patch_occlusion_probability": 0.0,
    "live_depth_patch_dropout_probability": 0.0,
    "live_calibration_warp_enabled": False,
    "live_clean_episode_fraction": 0.0,
}

_TIMING_IDENTITY: dict[str, Any] = {
    "live_observation_delay_max_steps": 0,
    "live_observation_repeat_probability": 0.0,
    "motion_action_delay_max_steps": 0,
    "motion_action_two_step_probability": 0.0,
    "motion_response_scale": (1.0, 1.0),
    "motion_response_alpha": (1.0, 1.0),
    "motion_bias": (0.0, 0.0),
    "physics_joint_stiffness_scale": (1.0, 1.0),
    "physics_joint_damping_scale": (1.0, 1.0),
}

_DOCUMENTED_SENSOR: dict[str, Any] = {
    "live_rgb_exposure_stops": (-0.30, 0.30),
    "live_rgb_contrast": (0.85, 1.15),
    "live_rgb_gamma": (0.90, 1.10),
    "live_rgb_white_balance_gain": (0.93, 1.07),
    "live_rgb_vignette_strength": (0.0, 0.14),
    "live_rgb_noise_std": (0.0, 0.015),
    "live_rgb_blur_probability": 0.12,
    "live_rgb_blur_mix": (0.25, 0.60),
    "live_depth_scale": (0.99, 1.01),
    "live_depth_bias_m": (-0.002, 0.002),
    "live_depth_noise_std_m": (0.0, 0.0002),
    "live_correlated_depth_enabled": True,
    "live_disparity_bias_px": (-0.04, 0.04),
    "live_disparity_independent_noise_std_px": (0.01, 0.03),
    "live_disparity_spatial_noise_std_px": (0.02, 0.07),
    "live_disparity_temporal_noise_std_px": (0.01, 0.04),
    "live_disparity_temporal_correlation": (0.60, 0.92),
    "live_stereo_edge_mismatch_probability": 0.12,
    "live_depth_dropout_probability": (0.0, 0.004),
    "live_depth_edge_dropout_probability": (0.0, 0.035),
    "live_rgb_patch_occlusion_probability": 0.06,
    "live_depth_patch_dropout_probability": 0.04,
    "live_clean_episode_fraction": 0.15,
}

_CAMERA_UNCERTAINTY: dict[str, Any] = {
    "live_calibration_warp_enabled": True,
    "live_calibration_shift_x_px": (-1.5, 1.5),
    "live_calibration_shift_y_px": (-1.0, 1.0),
    "live_calibration_scale": (0.99, 1.01),
    "live_calibration_roll_deg": (-1.0, 1.0),
}

_DOCUMENTED_TIMING: dict[str, Any] = {
    "live_observation_delay_max_steps": 1,
    "live_observation_repeat_probability": 0.02,
    "motion_action_delay_max_steps": 1,
    "motion_action_two_step_probability": 0.0,
    "motion_response_scale": (0.88, 1.12),
    "motion_response_alpha": (0.91, 1.0),
    "motion_bias": (-0.015, 0.015),
    "physics_joint_stiffness_scale": (0.90, 1.10),
    "physics_joint_damping_scale": (0.90, 1.10),
}

_DOCUMENTED_APPEARANCE: dict[str, Any] = {
    "scene_tslot_nominal_fraction": 0.60,
    "scene_tslot_phase_fraction": 0.20,
    "scene_part_color_scale": (0.90, 1.10),
    "scene_part_saturation_scale": (0.90, 1.10),
    "scene_part_hue_shift_deg": (-5.0, 5.0),
    "scene_part_roughness": (0.65, 0.90),
    "scene_tslot_color_scale": (0.88, 1.12),
    "scene_tslot_saturation_scale": (0.90, 1.10),
    "scene_tslot_hue_shift_deg": (-5.0, 5.0),
    "scene_tslot_roughness_delta": (-0.08, 0.08),
}

_CLUTTER_DISABLED: dict[str, Any] = {
    "scene_clutter_enabled": False,
    "scene_clutter_environment_fraction": 0.0,
    "scene_clutter_min_objects": 1,
    "scene_clutter_max_objects": 3,
    "scene_busy_background_enabled": False,
    "scene_busy_background_environment_fraction": 0.0,
    "scene_busy_background_min_people": 2,
    "scene_busy_background_max_people": 4,
}

_PERIPHERAL_CLUTTER: dict[str, Any] = {
    "scene_clutter_enabled": True,
    "scene_clutter_environment_fraction": 0.60,
    "scene_clutter_min_objects": 1,
    "scene_clutter_max_objects": 3,
    "scene_busy_background_enabled": False,
    "scene_busy_background_environment_fraction": 0.0,
    "scene_busy_background_min_people": 2,
    "scene_busy_background_max_people": 4,
}

_BUSY_BACKGROUND: dict[str, Any] = {
    "scene_clutter_enabled": True,
    "scene_clutter_environment_fraction": 0.80,
    "scene_clutter_min_objects": 2,
    "scene_clutter_max_objects": 3,
    "scene_busy_background_enabled": True,
    "scene_busy_background_environment_fraction": 0.70,
    "scene_busy_background_min_people": 4,
    "scene_busy_background_max_people": 4,
}


def _merged(*values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        result.update(value)
    return result


# A deliberately stronger, still structured bracket around the provisional
# D405 model. It preserves spatial/temporal correlation and stereo-edge
# failure instead of replacing the depth image with independent white noise.
_DEPTH_ROBUST_SENSOR: dict[str, Any] = _merged(
    _DOCUMENTED_SENSOR,
    {
        "live_depth_scale": (0.985, 1.015),
        "live_depth_bias_m": (-0.0035, 0.0035),
        "live_depth_noise_std_m": (0.0, 0.0004),
        "live_disparity_bias_px": (-0.08, 0.08),
        "live_disparity_independent_noise_std_px": (0.02, 0.05),
        "live_disparity_spatial_noise_std_px": (0.04, 0.12),
        "live_disparity_temporal_noise_std_px": (0.02, 0.07),
        "live_stereo_edge_mismatch_probability": 0.22,
        "live_depth_dropout_probability": (0.0, 0.008),
        "live_depth_edge_dropout_probability": (0.0, 0.07),
        "live_depth_patch_dropout_probability": 0.08,
        "live_patch_area_fraction": (0.005, 0.04),
    },
)


_PROFILES: dict[str, Sim2RealProfile] = {
    "nominal": Sim2RealProfile(
        "nominal",
        _merged(
            _SENSOR_IDENTITY,
            _TIMING_IDENTITY,
            {
                "live_observation_randomization_enabled": False,
                "scene_appearance_randomization_enabled": False,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": False,
            },
            _CLUTTER_DISABLED,
        ),
        "Canonical renderer, current frame, and nominal controller response.",
    ),
    "sensor_only": Sim2RealProfile(
        "sensor_only",
        _merged(
            _DOCUMENTED_SENSOR,
            {"live_calibration_warp_enabled": False},
            _TIMING_IDENTITY,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": False,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": False,
            },
            _CLUTTER_DISABLED,
        ),
        "D405 RGB/disparity artifacts without calibration or timing uncertainty.",
    ),
    "camera_uncertainty": Sim2RealProfile(
        "camera_uncertainty",
        _merged(
            _SENSOR_IDENTITY,
            _CAMERA_UNCERTAINTY,
            _TIMING_IDENTITY,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": False,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": False,
            },
            _CLUTTER_DISABLED,
        ),
        "Coupled RGB-D image warp approximating bounded intrinsic and hand-eye error.",
    ),
    "timing_control": Sim2RealProfile(
        "timing_control",
        _merged(
            _SENSOR_IDENTITY,
            _DOCUMENTED_TIMING,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": False,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": False,
            },
            _CLUTTER_DISABLED,
        ),
        "Frame latency/repeats and bounded actuator response mismatch.",
    ),
    "appearance": Sim2RealProfile(
        "appearance",
        _merged(
            _SENSOR_IDENTITY,
            _TIMING_IDENTITY,
            _DOCUMENTED_APPEARANCE,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": True,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": True,
            },
            _CLUTTER_DISABLED,
        ),
        "Physical light, shadow, material, background, and part appearance variation.",
    ),
    "combined_sim2real": Sim2RealProfile(
        "combined_sim2real",
        _merged(
            _DOCUMENTED_SENSOR,
            _CAMERA_UNCERTAINTY,
            _DOCUMENTED_TIMING,
            _DOCUMENTED_APPEARANCE,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": True,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": True,
            },
            _CLUTTER_DISABLED,
        ),
        "Provisional D405 sensor, calibration, timing, control, and appearance profile.",
    ),
    "combined_clutter": Sim2RealProfile(
        "combined_clutter",
        _merged(
            _DOCUMENTED_SENSOR,
            _CAMERA_UNCERTAINTY,
            _DOCUMENTED_TIMING,
            _DOCUMENTED_APPEARANCE,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": True,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": True,
            },
            _PERIPHERAL_CLUTTER,
        ),
        "Combined profile plus render/depth-only peripheral clutter in 60% of environments.",
    ),
    "combined_busy_background": Sim2RealProfile(
        "combined_busy_background",
        _merged(
            _DOCUMENTED_SENSOR,
            _CAMERA_UNCERTAINTY,
            _DOCUMENTED_TIMING,
            _DOCUMENTED_APPEARANCE,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": True,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": True,
            },
            _BUSY_BACKGROUND,
        ),
        (
            "Combined profile plus a clean mixture of tall, highly varied four-sided procedural "
            "office/factory walls, balanced perimeter storage, panels, cables, people, and "
            "table-edge coworker reaches."
        ),
    ),
    "combined_depth_robust": Sim2RealProfile(
        "combined_depth_robust",
        _merged(
            _DEPTH_ROBUST_SENSOR,
            _CAMERA_UNCERTAINTY,
            _DOCUMENTED_TIMING,
            _DOCUMENTED_APPEARANCE,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": True,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": True,
            },
            _CLUTTER_DISABLED,
        ),
        "Combined profile with a stronger structured D405 depth-error bracket and no clutter.",
    ),
    "stress_test": Sim2RealProfile(
        "stress_test",
        _merged(
            _DOCUMENTED_SENSOR,
            _CAMERA_UNCERTAINTY,
            _DOCUMENTED_TIMING,
            _DOCUMENTED_APPEARANCE,
            {
                "live_observation_randomization_enabled": True,
                "scene_appearance_randomization_enabled": True,
                "scene_tslot_surface_enabled": True,
                "scene_tslot_geometry_randomization_enabled": True,
                "scene_tslot_nominal_fraction": 0.40,
                "scene_tslot_phase_fraction": 0.25,
                "live_disparity_bias_px": (-0.08, 0.08),
                "live_disparity_spatial_noise_std_px": (0.04, 0.12),
                "live_disparity_temporal_noise_std_px": (0.02, 0.07),
                "live_stereo_edge_mismatch_probability": 0.22,
                "live_calibration_shift_x_px": (-3.0, 3.0),
                "live_calibration_shift_y_px": (-2.0, 2.0),
                "live_calibration_scale": (0.98, 1.02),
                "live_calibration_roll_deg": (-2.0, 2.0),
                "live_observation_repeat_probability": 0.05,
                "motion_response_scale": (0.80, 1.20),
                "physics_joint_stiffness_scale": (0.80, 1.20),
                "physics_joint_damping_scale": (0.80, 1.20),
                "live_clean_episode_fraction": 0.0,
            },
            _CLUTTER_DISABLED,
        ),
        "Intentionally broader holdout; use for evaluation, not baseline training.",
    ),
}


def get_sim2real_profile(name: str) -> Sim2RealProfile:
    """Return a named profile or fail with the complete supported list."""

    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown sim-to-real profile {name!r}; expected one of {SIM2REAL_PROFILE_NAMES}.") from exc


def apply_sim2real_profile(cfg: Any, name: str) -> Sim2RealProfile:
    """Apply one profile to an Isaac environment config and record its ID."""

    profile = get_sim2real_profile(name)
    missing = sorted(key for key in profile.overrides if not hasattr(cfg, key))
    if missing:
        raise AttributeError(f"Environment config cannot apply profile {name!r}; missing fields: {missing}.")
    for key, value in profile.overrides.items():
        setattr(cfg, key, value)
    cfg.sim2real_randomization_profile = profile.identifier
    return profile
