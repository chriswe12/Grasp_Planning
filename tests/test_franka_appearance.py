import copy
from pathlib import Path

import pytest

from grasp_planning.rl.franka_appearance import load_profile, sample_appearance, validate_profile

PROFILE = Path(__file__).resolve().parents[1] / "configs/franka_pencil_randomization.json"


def test_reproducibility_diversity_and_bounds():
    profile = load_profile(PROFILE)
    before = copy.deepcopy(profile)
    samples = [sample_appearance(profile, seed) for seed in range(1000)]
    assert samples == [sample_appearance(profile, seed) for seed in range(1000)]
    assert profile == before
    assert {s["part_color_name"] for s in samples} == set(profile["palette"])
    assert 0.2 < sum(s["part_color_name"] == "blue" for s in samples) / len(samples) < 0.3
    assert 0.73 < sum(s["wear_visible"] for s in samples) / len(samples) < 0.87
    for s in samples:
        for key in ("part_roughness", "table_roughness", "light_intensity", "light_radius_m", "light_temperature_k"):
            assert profile[key][0] <= s[key] <= profile[key][1]
        assert all(0 <= c <= 1 for c in s["part_color"])
        assert len(s["prop_colors"]) == 5


def test_rejects_invalid_profile():
    profile = load_profile(PROFILE)
    profile["light_intensity"] = [100.0, float("nan")]
    with pytest.raises(ValueError):
        validate_profile(profile)
    profile = load_profile(PROFILE)
    profile["goal_appearance"] = "silently_recolor_goal"
    with pytest.raises(ValueError):
        validate_profile(profile)
