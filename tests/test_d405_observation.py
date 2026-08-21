from __future__ import annotations

import torch

from grasp_planning.rl.d405_observation import (
    D405ObservationPreprocessCfg,
    normalize_depth_torch,
    resize_aligned_rgbd_torch,
)


def test_invalid_depth_does_not_bias_valid_area_average() -> None:
    cfg = D405ObservationPreprocessCfg(
        output_height=1,
        output_width=1,
        minimum_valid_area_fraction=0.25,
    )
    rgb = torch.ones((1, 2, 2, 3))
    depth = torch.tensor([[[[0.20], [0.0]], [[0.20], [0.0]]]])

    _, resized_depth, valid = resize_aligned_rgbd_torch(rgb, depth, cfg=cfg)

    assert valid.item()
    torch.testing.assert_close(resized_depth, torch.full_like(resized_depth, 0.20))


def test_max_range_sentinel_does_not_bias_valid_area_average() -> None:
    cfg = D405ObservationPreprocessCfg(output_height=1, output_width=1)
    rgb = torch.ones((1, 2, 2, 3))
    depth = torch.tensor([[[[0.20], [0.50]], [[0.20], [0.50]]]])

    _, resized_depth, valid = resize_aligned_rgbd_torch(rgb, depth, cfg=cfg)

    assert valid.item()
    torch.testing.assert_close(resized_depth, torch.full_like(resized_depth, 0.20))


def test_insufficient_valid_area_uses_invalid_fill() -> None:
    cfg = D405ObservationPreprocessCfg(
        output_height=1,
        output_width=1,
        minimum_valid_area_fraction=0.50,
    )
    rgb = torch.ones((1, 2, 2, 3))
    depth = torch.tensor([[[[0.20], [0.0]], [[0.0], [0.0]]]])

    _, resized_depth, valid = resize_aligned_rgbd_torch(rgb, depth, cfg=cfg)

    assert not valid.item()
    assert resized_depth.item() == cfg.valid_depth_max_m


def test_depth_normalization_preserves_configured_endpoints() -> None:
    cfg = D405ObservationPreprocessCfg()
    depth = torch.tensor([cfg.normalization_min_m, cfg.normalization_max_m]).reshape(1, 1, 2, 1)

    normalized = normalize_depth_torch(depth, cfg=cfg)

    torch.testing.assert_close(normalized, torch.tensor([0.0, 1.0]).reshape(1, 1, 2, 1))
