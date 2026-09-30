import copy

import pytest

from grasp_planning.rl.franka_performance import (
    FAST_PROFILE,
    optimized_contract,
    resume_epoch_for_frames,
    validate_performance_profile,
)


def test_upgrade_preserves_robot_pose_and_color_contract():
    source = dict(
        lab_scene={"asset_sha256": "untouched"},
        appearance_randomization={"palette": {"blue": [0.1, 0.2, 0.4]}},
        robot_usd="panda",
        pose_reset_profile={"version": "validated"},
        camera_profile="zed",
    )
    before = copy.deepcopy(source)
    result = optimized_contract(source)
    assert source == before
    assert {k: v for k, v in result.items() if k != "performance_profile"} == source
    result["appearance_randomization"]["palette"]["blue"][0] = 0
    assert source == before


def test_profile_rejects_unrecorded_color_and_pose_changes():
    profile = copy.deepcopy(FAST_PROFILE)
    profile["protected_surfaces"] = ["table"]
    with pytest.raises(ValueError):
        validate_performance_profile(profile)
    profile = copy.deepcopy(FAST_PROFILE)
    profile["reset_render_count"] = 1
    with pytest.raises(ValueError):
        validate_performance_profile(profile)


def test_rebatching_preserves_training_and_curriculum_fraction():
    frames = 16_384_000
    for total in (32, 64, 128, 160, 256, 512, 640, 1024):
        epoch = resume_epoch_for_frames(frames, total)
        final_epoch = resume_epoch_for_frames(163_840_000, total)
        assert epoch / final_epoch == pytest.approx(0.1)
        assert epoch * 64 * total == frames
        assert (frames // total) * total / 512 == frames / 512
    with pytest.raises(ValueError):
        resume_epoch_for_frames(frames, 63)
