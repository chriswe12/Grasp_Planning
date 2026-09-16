import json

import pytest
import torch

from grasp_planning.rl.zed_mini import (
    load_zed_profile,
    offset_jacobian,
    pack_zed_rgbd,
    reproject_intrinsics,
)


def test_invalid_depth_does_not_pollute_valid_surface():
    profile = dict(load_zed_profile(), observation_height=1, observation_width=1)
    rgb = torch.zeros((1, 2, 2, 3), dtype=torch.uint8)
    depth = torch.tensor([[[[0.2], [float("inf")]], [[0.0], [float("nan")]]]])
    packed, valid = pack_zed_rgbd(rgb, depth, profile)
    assert valid.item()
    assert packed[0, 0, 0, 3].item() == pytest.approx(1 / 9)
    depth.fill_(float("nan"))
    packed, valid = pack_zed_rgbd(rgb, depth, profile)
    assert not valid.item()
    assert torch.isfinite(packed).all() and packed[0, 0, 0, 3] == 1


def test_tcp_offset_angular_velocity_sign():
    jac = torch.zeros((1, 6, 1))
    jac[0, 5, 0] = 1  # z rotation around body origin
    moved = offset_jacobian(jac, torch.tensor([[2.0, 0.0, 0.0]]))
    torch.testing.assert_close(moved[0, :3, 0], torch.tensor([0.0, 2.0, 0.0]))


def test_intrinsics_principal_point_moves_image_and_depth_together():
    rgb = torch.zeros((1, 4, 6, 3))
    rgb[0, 2, 2] = 1
    depth = torch.zeros((1, 4, 6, 1))
    depth[0, 2, 2] = 0.3
    src = torch.tensor([[4.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
    dst = src.clone()
    dst[0, 2] += 1
    color, metric = reproject_intrinsics(rgb, depth, src, dst)
    torch.testing.assert_close(color[0, 2, 3], torch.ones(3))
    assert metric[0, 2, 3, 0].item() == pytest.approx(0.3)
    assert metric[0, :, 0].count_nonzero() == 0


def test_rejects_other_zed_baseline(tmp_path):
    p = dict(load_zed_profile(), stereo_baseline_m=0.12008)
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="camera identity"):
        load_zed_profile(path)
