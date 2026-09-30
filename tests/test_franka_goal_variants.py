import copy

import numpy as np
import pytest
import torch

from grasp_planning.rl.franka_goal_variants import (
    GOAL_VARIANTS_PROFILE,
    GoalVariantBank,
    validate_variants,
    variant_digest,
)


def fixture_bank():
    images = np.zeros((2, 4, 72, 128, 4), dtype=np.float16)
    for t in range(2):
        for v in range(4):
            images[t, v] = 0.1 * (v + 1) + 0.01 * t
    data = {"target_ids": np.array(["a", "b"]), "goal_rgbd_variants": images}
    profile = {**copy.deepcopy(GOAL_VARIANTS_PROFILE), "images_sha256": variant_digest(images)}
    return data, profile


def test_goal_target_and_variant_selection_preserves_canonical():
    data, profile = fixture_bank()
    bank = GoalVariantBank(data, np.array([1, 0]), profile, "cpu")
    canonical = torch.full((2, 72, 128, 4), 0.9)
    targets = torch.tensor([1, 0, 1])
    for v in range(5):
        selected, ids = bank.sample(targets, canonical, v)
        assert ids.tolist() == [v] * 3
        if v == 0:
            assert torch.equal(selected, canonical[targets])
        else:
            expected = data["goal_rgbd_variants"][np.array([0, 1, 0]), v - 1].astype(np.float32)
            np.testing.assert_array_equal(selected.numpy(), expected)
    assert torch.all(canonical == 0.9)


def test_balanced_backend_sampling_and_bank_integrity():
    data, profile = fixture_bank()
    bank = GoalVariantBank(data, np.arange(2), profile, "cpu")
    # Check multinomial weights without allocating large image batches.
    torch.manual_seed(42)
    draws = torch.multinomial(bank.weights, 10000, replacement=True)
    assert 0.48 < float((draws >= 3).float().mean()) < 0.52
    data["goal_rgbd_variants"][0, 0, 0, 0, 0] = 0.8
    with pytest.raises(ValueError, match="digest"):
        validate_variants(data, profile)
