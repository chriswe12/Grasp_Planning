"""Native image migration keeps camera hashes and mixed banks consistent."""

import copy
from pathlib import Path

import numpy as np
import pytest

from grasp_planning.rl.franka_goal_variants import GOAL_VARIANTS_PROFILE, validate_variants, variant_digest
from grasp_planning.rl.zed_mini import load_zed_profile, profile_id, resolve_zed_profile, validate_zed_profile

ROOT = Path(__file__).resolve().parents[1]


def test_native_profile_is_distinct_and_cannot_override_old_catalog():
    old = load_zed_profile(ROOT / "configs/franka_zed_mini_sn13829658.json")
    new = load_zed_profile(ROOT / "configs/franka_zed_mini_sn13829658_384.json")
    assert profile_id(old) != profile_id(new)
    assert all(old[k] == new[k] for k in ["fx", "fy", "cx", "cy", "position_m", "quaternion_wxyz"])
    with pytest.raises(ValueError, match="differs"):
        resolve_zed_profile({"camera_profile": profile_id(old), "camera_profile_data": new})
    new["render_width"] = 128
    with pytest.raises(ValueError, match="Render resolution"):
        validate_zed_profile(new)


def test_native_bank_rejects_upscaled_metadata_with_old_pixels():
    images = np.zeros((1, 4, 216, 384, 4), dtype=np.float16)
    data = dict(target_ids=["a"], goal_rgbd=np.zeros((1, 216, 384, 4), np.float16), goal_rgbd_variants=images)
    profile = {**copy.deepcopy(GOAL_VARIANTS_PROFILE), "images_sha256": variant_digest(images)}
    validate_variants(data, profile)
    data["goal_rgbd"] = np.zeros((1, 72, 128, 4), np.float16)
    with pytest.raises(ValueError, match="shape"):
        validate_variants(data, profile)
