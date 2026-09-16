from __future__ import annotations

import torch

from grasp_planning.rl.live_observation_randomization import (
    LiveObservationRandomizationCfg,
    LiveObservationRandomizer,
)


def test_disabled_randomization_is_an_exact_noop() -> None:
    randomizer = LiveObservationRandomizer(
        LiveObservationRandomizationCfg(enabled=False),
        num_envs=2,
        device="cpu",
    )
    rgb = torch.rand((2, 8, 12, 3))
    depth = torch.rand((2, 8, 12, 1)) * 0.20 + 0.10

    randomized_rgb, randomized_depth = randomizer.apply(rgb, depth)

    assert torch.equal(randomized_rgb, rgb)
    assert torch.equal(randomized_depth, depth)


def test_fixed_photometric_and_depth_calibration_apply_only_to_live_input() -> None:
    cfg = LiveObservationRandomizationCfg(
        exposure_stops=(1.0, 1.0),
        contrast=(1.0, 1.0),
        gamma=(1.0, 1.0),
        white_balance_gain=(1.0, 1.0),
        vignette_strength=(0.0, 0.0),
        rgb_noise_std=(0.0, 0.0),
        blur_probability=0.0,
        depth_scale=(1.10, 1.10),
        depth_bias_m=(0.01, 0.01),
        depth_noise_std_m=(0.0, 0.0),
        correlated_depth_enabled=False,
        depth_quantization_m=0.0,
        depth_dropout_probability=(0.0, 0.0),
        depth_edge_dropout_probability=(0.0, 0.0),
        rgb_patch_occlusion_probability=0.0,
        depth_patch_dropout_probability=0.0,
        calibration_warp_enabled=False,
    )
    randomizer = LiveObservationRandomizer(cfg, num_envs=1, device="cpu")
    live_rgb = torch.full((1, 6, 8, 3), 0.20)
    live_depth = torch.full((1, 6, 8, 1), 0.20)
    goal_rgb = torch.full((1, 6, 8, 3), 0.35)
    goal_depth = torch.full((1, 6, 8, 1), 0.15)

    randomized_rgb, randomized_depth = randomizer.apply(live_rgb, live_depth)

    torch.testing.assert_close(randomized_rgb, torch.full_like(live_rgb, 0.40))
    torch.testing.assert_close(randomized_depth, torch.full_like(live_depth, 0.23))
    # The deployment contract calls the randomizer only for live RGB-D.
    assert torch.equal(goal_rgb, torch.full_like(goal_rgb, 0.35))
    assert torch.equal(goal_depth, torch.full_like(goal_depth, 0.15))


def test_randomized_rgbd_stays_finite_and_sensor_bounded() -> None:
    torch.manual_seed(11)
    randomizer = LiveObservationRandomizer(
        LiveObservationRandomizationCfg(),
        num_envs=3,
        device="cpu",
    )
    rgb = torch.rand((3, 16, 24, 3))
    depth = torch.linspace(0.05, 0.49, 16 * 24).reshape(1, 16, 24, 1).repeat(3, 1, 1, 1)

    randomized_rgb, randomized_depth = randomizer.apply(rgb, depth)

    assert randomized_rgb.shape == rgb.shape
    assert randomized_depth.shape == depth.shape
    assert torch.isfinite(randomized_rgb).all()
    assert torch.isfinite(randomized_depth).all()
    assert randomized_rgb.min() >= 0.0
    assert randomized_rgb.max() <= 1.0
    assert randomized_depth.min() >= randomizer.cfg.depth_min_m
    assert randomized_depth.max() <= randomizer.cfg.depth_max_m


def test_resampling_one_environment_preserves_other_episode_parameters() -> None:
    torch.manual_seed(7)
    randomizer = LiveObservationRandomizer(
        LiveObservationRandomizationCfg(),
        num_envs=2,
        device="cpu",
    )
    preserved_exposure = randomizer.exposure_gain[1].clone()
    preserved_depth_scale = randomizer.depth_scale[1].clone()

    randomizer.sample(torch.tensor([0]))

    assert torch.equal(randomizer.exposure_gain[1], preserved_exposure)
    assert torch.equal(randomizer.depth_scale[1], preserved_depth_scale)


def test_zero_curriculum_strength_is_an_exact_noop() -> None:
    randomizer = LiveObservationRandomizer(
        LiveObservationRandomizationCfg(
            rgb_patch_occlusion_probability=1.0,
            depth_patch_dropout_probability=1.0,
        ),
        num_envs=1,
        device="cpu",
    )
    randomizer.sample(torch.tensor([0]), strength=0.0)
    rgb = torch.rand((1, 12, 16, 3))
    depth = torch.rand((1, 12, 16, 1)) * 0.20 + 0.10

    randomized_rgb, randomized_depth = randomizer.apply(rgb, depth)

    assert torch.equal(randomized_rgb, rgb)
    assert torch.equal(randomized_depth, depth)


def test_live_patch_occlusion_never_changes_goal_input_contract() -> None:
    cfg = LiveObservationRandomizationCfg(
        exposure_stops=(0.0, 0.0),
        contrast=(1.0, 1.0),
        gamma=(1.0, 1.0),
        white_balance_gain=(1.0, 1.0),
        vignette_strength=(0.0, 0.0),
        rgb_noise_std=(0.0, 0.0),
        blur_probability=0.0,
        depth_scale=(1.0, 1.0),
        depth_bias_m=(0.0, 0.0),
        depth_noise_std_m=(0.0, 0.0),
        correlated_depth_enabled=False,
        depth_quantization_m=0.0,
        depth_dropout_probability=(0.0, 0.0),
        depth_edge_dropout_probability=(0.0, 0.0),
        rgb_patch_occlusion_probability=1.0,
        depth_patch_dropout_probability=1.0,
        patch_area_fraction=(0.04, 0.04),
        calibration_warp_enabled=False,
    )
    randomizer = LiveObservationRandomizer(cfg, num_envs=1, device="cpu")
    randomizer.patch_center_x[:] = 0.5
    randomizer.patch_center_y[:] = 0.5
    randomizer.patch_aspect_log[:] = 0.0
    live_rgb = torch.linspace(0.0, 1.0, 12 * 16 * 3).reshape(1, 12, 16, 3)
    live_depth = torch.full((1, 12, 16, 1), 0.20)
    goal_rgb = torch.rand((1, 12, 16, 3))
    goal_depth = torch.rand((1, 12, 16, 1))
    original_goal_rgb = goal_rgb.clone()
    original_goal_depth = goal_depth.clone()

    randomized_rgb, randomized_depth = randomizer.apply(live_rgb, live_depth)

    assert not torch.equal(randomized_rgb, live_rgb)
    assert not torch.equal(randomized_depth, live_depth)
    assert (randomized_depth == cfg.depth_max_m).any()
    assert torch.equal(goal_rgb, original_goal_rgb)
    assert torch.equal(goal_depth, original_goal_depth)


def test_disparity_error_produces_larger_metric_error_at_longer_range() -> None:
    cfg = LiveObservationRandomizationCfg(
        exposure_stops=(0.0, 0.0),
        contrast=(1.0, 1.0),
        gamma=(1.0, 1.0),
        white_balance_gain=(1.0, 1.0),
        vignette_strength=(0.0, 0.0),
        rgb_noise_std=(0.0, 0.0),
        blur_probability=0.0,
        depth_scale=(1.0, 1.0),
        depth_bias_m=(0.0, 0.0),
        depth_noise_std_m=(0.0, 0.0),
        disparity_bias_px=(0.05, 0.05),
        disparity_independent_noise_std_px=(0.0, 0.0),
        disparity_spatial_noise_std_px=(0.0, 0.0),
        disparity_temporal_noise_std_px=(0.0, 0.0),
        disparity_temporal_correlation=(0.0, 0.0),
        stereo_edge_mismatch_probability=0.0,
        depth_quantization_m=0.0,
        depth_dropout_probability=(0.0, 0.0),
        depth_edge_dropout_probability=(0.0, 0.0),
        rgb_patch_occlusion_probability=0.0,
        depth_patch_dropout_probability=0.0,
        calibration_warp_enabled=False,
    )
    randomizer = LiveObservationRandomizer(cfg, num_envs=1, device="cpu")
    rgb = torch.full((1, 2, 2, 3), 0.5)
    depth = torch.tensor([[[[0.10], [0.10]], [[0.40], [0.40]]]])

    _, noisy_depth = randomizer.apply(rgb, depth)

    near_error = (noisy_depth[:, 0] - depth[:, 0]).abs().mean()
    far_error = (noisy_depth[:, 1] - depth[:, 1]).abs().mean()
    assert far_error > near_error * 8.0


def test_depth_below_d405_reliable_range_becomes_invalid() -> None:
    cfg = LiveObservationRandomizationCfg(
        correlated_depth_enabled=False,
        calibration_warp_enabled=False,
        depth_scale=(1.0, 1.0),
        depth_bias_m=(0.0, 0.0),
        depth_noise_std_m=(0.0, 0.0),
        depth_quantization_m=0.0,
        stereo_edge_mismatch_probability=0.0,
        depth_dropout_probability=(0.0, 0.0),
        depth_edge_dropout_probability=(0.0, 0.0),
        rgb_patch_occlusion_probability=0.0,
        depth_patch_dropout_probability=0.0,
    )
    randomizer = LiveObservationRandomizer(cfg, num_envs=1, device="cpu")
    rgb = torch.full((1, 2, 2, 3), 0.5)
    depth = torch.tensor([[[[0.05], [0.07]], [[0.20], [0.50]]]])

    _, noisy_depth = randomizer.apply(rgb, depth)

    assert noisy_depth[0, 0, 0, 0] == cfg.depth_max_m
    assert noisy_depth[0, 0, 1, 0] == cfg.depth_min_m
