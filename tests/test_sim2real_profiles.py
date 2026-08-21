from __future__ import annotations

from types import SimpleNamespace

import pytest

from grasp_planning.rl.sim2real_profiles import (
    SIM2REAL_PROFILE_NAMES,
    apply_sim2real_profile,
    get_sim2real_profile,
)


def _config_with_all_fields() -> SimpleNamespace:
    fields = {key: None for name in SIM2REAL_PROFILE_NAMES for key in get_sim2real_profile(name).overrides}
    fields["sim2real_randomization_profile"] = "unset"
    return SimpleNamespace(**fields)


def test_combined_profile_uses_small_tslot_with_sensor_and_appearance_randomization() -> None:
    cfg = _config_with_all_fields()
    profile = apply_sim2real_profile(cfg, "combined_sim2real")

    assert cfg.sim2real_randomization_profile == profile.identifier
    assert cfg.live_observation_randomization_enabled
    assert cfg.live_correlated_depth_enabled
    assert cfg.live_calibration_warp_enabled
    assert cfg.scene_appearance_randomization_enabled
    assert cfg.scene_tslot_surface_enabled
    assert cfg.scene_tslot_geometry_randomization_enabled
    assert cfg.scene_tslot_nominal_fraction == pytest.approx(0.60)
    assert cfg.scene_tslot_phase_fraction == pytest.approx(0.20)
    assert cfg.live_observation_delay_max_steps == 2
    assert cfg.motion_action_delay_max_steps == 2
    assert not cfg.scene_clutter_enabled
    assert cfg.scene_clutter_environment_fraction == 0.0


def test_clutter_profile_changes_only_clutter_fields_from_combined() -> None:
    combined_cfg = _config_with_all_fields()
    clutter_cfg = _config_with_all_fields()
    apply_sim2real_profile(combined_cfg, "combined_sim2real")
    apply_sim2real_profile(clutter_cfg, "combined_clutter")

    differing = {
        key
        for key in vars(combined_cfg)
        if getattr(combined_cfg, key) != getattr(clutter_cfg, key)
    }
    assert differing == {
        "sim2real_randomization_profile",
        "scene_clutter_enabled",
        "scene_clutter_environment_fraction",
    }
    assert clutter_cfg.scene_clutter_enabled
    assert clutter_cfg.scene_clutter_environment_fraction == pytest.approx(0.60)


def test_depth_robust_profile_strengthens_depth_only_from_combined() -> None:
    combined_cfg = _config_with_all_fields()
    depth_cfg = _config_with_all_fields()
    apply_sim2real_profile(combined_cfg, "combined_sim2real")
    apply_sim2real_profile(depth_cfg, "combined_depth_robust")

    differing = {
        key
        for key in vars(combined_cfg)
        if getattr(combined_cfg, key) != getattr(depth_cfg, key)
    }
    assert differing == {
        "sim2real_randomization_profile",
        "live_depth_scale",
        "live_depth_bias_m",
        "live_depth_noise_std_m",
        "live_disparity_bias_px",
        "live_disparity_independent_noise_std_px",
        "live_disparity_spatial_noise_std_px",
        "live_disparity_temporal_noise_std_px",
        "live_stereo_edge_mismatch_probability",
        "live_depth_dropout_probability",
        "live_depth_edge_dropout_probability",
        "live_depth_patch_dropout_probability",
        "live_patch_area_fraction",
    }
    assert depth_cfg.live_depth_bias_m == pytest.approx((-0.0035, 0.0035))
    assert depth_cfg.live_disparity_spatial_noise_std_px == pytest.approx((0.04, 0.12))
    assert depth_cfg.live_depth_edge_dropout_probability == pytest.approx((0.0, 0.07))
    assert not depth_cfg.scene_clutter_enabled


def test_nominal_profile_removes_sensor_and_timing_variation() -> None:
    cfg = _config_with_all_fields()
    apply_sim2real_profile(cfg, "nominal")

    assert not cfg.live_observation_randomization_enabled
    assert not cfg.live_correlated_depth_enabled
    assert not cfg.live_calibration_warp_enabled
    assert not cfg.scene_appearance_randomization_enabled
    assert cfg.scene_tslot_surface_enabled
    assert not cfg.scene_tslot_geometry_randomization_enabled
    assert cfg.live_observation_delay_max_steps == 0
    assert cfg.motion_response_scale == (1.0, 1.0)


def test_stress_profile_explicitly_enables_small_tslot_surface() -> None:
    cfg = _config_with_all_fields()
    apply_sim2real_profile(cfg, "stress_test")

    assert cfg.scene_tslot_surface_enabled
    assert cfg.scene_tslot_geometry_randomization_enabled


def test_unknown_profile_fails_with_supported_names() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        get_sim2real_profile("unknown")


def test_profile_rejects_incompatible_environment_config() -> None:
    with pytest.raises(AttributeError, match="missing fields"):
        apply_sim2real_profile(SimpleNamespace(), "combined_sim2real")
