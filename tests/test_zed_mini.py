import json
from pathlib import Path

import pytest
import torch

from grasp_planning.rl.zed_mini import (
    load_zed_profile,
    offset_jacobian,
    pack_zed_rgbd,
    profile_id,
    reproject_intrinsics,
    resolve_zed_profile,
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


def test_catalog_camera_is_self_contained_and_legacy_still_resolves(tmp_path):
    old = load_zed_profile()
    measured = load_zed_profile(Path(__file__).resolve().parents[1] / "configs/franka_zed_mini_sn13829658.json")
    assert resolve_zed_profile({"camera_profile": profile_id(old)}) == old
    contract = {"camera_profile": profile_id(measured), "camera_profile_data": measured}
    assert resolve_zed_profile(contract) == measured
    override = tmp_path / "wrong_camera.json"
    override.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="differs"):
        resolve_zed_profile(contract, override)
    with pytest.raises(ValueError, match="differs"):
        resolve_zed_profile({**contract, "camera_profile_data": dict(measured, fx=428.4)})


def test_measured_mini_preserves_mount_and_covers_recorded_field_of_view():
    import math

    old = load_zed_profile()
    p = load_zed_profile(Path(__file__).resolve().parents[1] / "configs/franka_zed_mini_sn13829658.json")
    assert p["serial_number"] == 13829658
    for key in ("position_m", "quaternion_wxyz", "tcp_offset_in_hand_m"):
        assert p[key] == old[key]
    hfov = math.degrees(math.atan(p["cx"] / p["fx"]) + math.atan((p["source_width"] - p["cx"]) / p["fx"]))
    vfov = math.degrees(math.atan(p["cy"] / p["fy"]) + math.atan((p["source_height"] - p["cy"]) / p["fy"]))
    assert hfov == pytest.approx(89.6, abs=0.1)
    assert vfov == pytest.approx(58.1, abs=0.1)


def test_radial_depth_conversion_uses_each_camera_intrinsics():
    from grasp_planning.rl.zed_mini import optical_depth_from_radial

    k = torch.tensor(
        [[[100.0, 0.0, 2.0], [0.0, 110.0, 1.0], [0.0, 0.0, 1.0]], [[60.0, 0.0, 1.0], [0.0, 70.0, 2.0], [0.0, 0.0, 1.0]]]
    )
    y, x = torch.meshgrid(torch.arange(4), torch.arange(5), indexing="ij")
    u = (x[None] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None]
    v = (y[None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None]
    # A fronto-parallel surface with the same 0.3 m Z at every pixel.
    radial = 0.3 * torch.sqrt(1 + u * u + v * v)[..., None]
    z = optical_depth_from_radial(radial, k)
    assert torch.allclose(z, torch.full_like(z, 0.3), atol=1e-7)
    radial[0, 0, 0] = torch.inf
    assert torch.isinf(optical_depth_from_radial(radial, k)[0, 0, 0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reprojected_depth_batches_do_not_leak_into_each_other(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA runtime required to reproduce the adaptive-pooling layout bug")
    profile = load_zed_profile()
    rgb = torch.zeros((3, 144, 256, 3), device=device)
    values = torch.tensor([0.2, 0.5, 0.8], device=device)
    depth = values[:, None, None, None].expand(-1, 144, 256, 1).clone()
    k = torch.tensor([[163.0, 0, 128], [0, 163.0, 72], [0, 0, 1]], device=device)
    color, metric = reproject_intrinsics(rgb, depth, k, k)
    assert metric.stride(-1) != 1  # The actual grid_sample -> NHWC layout.
    packed, _ = pack_zed_rgbd(color, metric, profile)
    expected = (values - 0.1) / 0.9
    torch.testing.assert_close(packed[:, 20, 10, 3], expected)
    torch.testing.assert_close(packed[:, 50, 90, 3], expected)
