from copy import deepcopy

import numpy as np
import pytest
import torch

from grasp_planning.rl.franka_placement import (
    PLACEMENT_PROFILE,
    placement_transfer_allowed,
    sample_placement_deltas,
    transform_placement,
    validate_placement_poses,
)


def test_world_xy_distribution_does_not_depend_on_catalog_location():
    anchors_a = np.tile([0.43, -0.035], (1000, 1))
    anchors_b = np.tile([0.48, 0.035], (1000, 1))
    a = sample_placement_deltas(np.random.default_rng(12), anchors_a)
    b = sample_placement_deltas(np.random.default_rng(12), anchors_b)
    assert np.allclose(a[:, :2] + anchors_a, b[:, :2] + anchors_b)
    assert np.array_equal(a[:, 2], b[:, 2])
    assert np.all((a[:, 0] + anchors_a[:, 0] >= 0.33) & (a[:, 0] + anchors_a[:, 0] <= 0.58))
    assert np.all(np.abs(a[:, 1] + anchors_a[:, 1]) <= 0.185)
    assert np.all(np.abs(a[:, 2]) <= np.pi / 2)


def test_canonical_contract_transfer_is_explicit_and_only_placement_changes():
    source = {"camera": {"fx": 100.0}, "scene": "pencil", "policy_hz": 15}
    target = deepcopy(source)
    target["placement_randomization"] = deepcopy(PLACEMENT_PROFILE)
    assert placement_transfer_allowed(source, target)
    assert not placement_transfer_allowed(source, source)
    target["camera"]["fx"] += 1
    assert not placement_transfer_allowed(source, target)
    target["camera"]["fx"] -= 1
    target["placement_randomization"]["goal_image"] = "follows_live_object"
    assert not placement_transfer_allowed(source, target)
    assert "placement_randomization" not in source


def test_world_placement_changes_absolute_pose_but_preserves_relative_geometry():
    obj = torch.tensor([[0.43, -0.035, 0.1, 1, 0, 0, 0]])
    tcp = obj.clone()
    tcp[:, :3] += torch.tensor([0.10, 0.0, 0.02])
    delta = torch.tensor([[0.05, -0.08, np.pi / 2]])
    moved_obj = transform_placement(obj, obj[:, :3], delta)
    moved_tcp = transform_placement(tcp, obj[:, :3], delta)
    assert torch.allclose(moved_obj[:, :3], torch.tensor([[0.48, -0.115, 0.1]]))
    assert torch.allclose(moved_tcp[:, :3] - moved_obj[:, :3], torch.tensor([[0.0, 0.10, 0.02]]), atol=1e-7)
    assert torch.allclose(moved_obj[:, 3:], torch.tensor([[2**-0.5, 0, 0, 2**-0.5]]))
    recovered = transform_placement(moved_tcp, moved_obj[:, :3], -delta)
    assert torch.allclose(recovered, tcp, atol=1e-7)
    assert torch.equal(obj, torch.tensor([[0.43, -0.035, 0.1, 1, 0, 0, 0]]))


def test_only_selectable_poses_must_have_normalized_quaternions():
    value = np.zeros((2, 3, 7), dtype=np.float32)
    valid = np.zeros((2, 3), dtype=bool)
    valid[0, 1] = True
    with pytest.raises(ValueError, match="normalized"):
        validate_placement_poses(value, valid)
    value[0, 1, 3] = 1
    validate_placement_poses(value, valid)
    value[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="poses"):
        validate_placement_poses(value, valid)
