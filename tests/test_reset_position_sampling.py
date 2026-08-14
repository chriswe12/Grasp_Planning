from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/reset_position_sampling.py"
)
SPEC = importlib.util.spec_from_file_location("reset_position_sampling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reset_position_sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reset_position_sampling)
position_offset_profile = reset_position_sampling.position_offset_profile
sample_collision_safe_xy_offsets = reset_position_sampling.sample_collision_safe_xy_offsets
sample_collision_safe_xy_offsets_from_profile = reset_position_sampling.sample_collision_safe_xy_offsets_from_profile


def test_position_offset_profile_tapers_from_far_to_near() -> None:
    progress = torch.tensor([0.0, 0.5, 1.0])

    result = position_offset_profile(
        progress,
        far_offset_m=0.010,
        near_offset_m=0.003,
        exponent=1.0,
    )

    assert result.tolist() == pytest.approx([0.010, 0.0065, 0.003])


def test_position_offsets_never_consume_validated_clearance() -> None:
    progress = torch.tensor([0.0, 0.5, 1.0, 0.0])
    clearance = torch.tensor([0.050, 0.006, 0.00105, 0.050])
    positive = torch.tensor([False, False, False, True])

    offsets, requested, safe_cap = sample_collision_safe_xy_offsets(
        progress,
        clearance,
        positive,
        minimum_collision_clearance_m=0.001,
        clearance_guard_m=0.0001,
        far_offset_m=0.010,
        near_offset_m=0.003,
        exponent=1.0,
        fraction_min=1.0,
        fraction_max=1.0,
        magnitude_unit_samples=torch.ones(4),
        direction_unit_samples=torch.tensor([0.0, 0.25, 0.5, 0.75]),
    )

    magnitudes = torch.linalg.norm(offsets, dim=-1)
    assert magnitudes.tolist() == pytest.approx([0.010, 0.0049, 0.0, 0.0])
    assert requested.tolist() == pytest.approx([0.010, 0.0065, 0.003, 0.0])
    assert safe_cap.tolist() == pytest.approx([0.0489, 0.0049, 0.0, 0.0489])
    assert torch.all(magnitudes <= safe_cap + 1.0e-9)
    assert torch.allclose(offsets[:, 2], torch.zeros(4))
    assert offsets[0, 0].item() == pytest.approx(0.010)
    assert offsets[1, 1].item() == pytest.approx(0.0049)
    assert abs(offsets[2, 0].item()) < 1.0e-9
    assert math.isclose(float(torch.linalg.norm(offsets[3])), 0.0)


def test_position_offset_configuration_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="near position-reset offset"):
        position_offset_profile(
            torch.tensor([0.5]),
            far_offset_m=0.003,
            near_offset_m=0.010,
            exponent=1.0,
        )


def test_continuous_ready_offsets_include_zero_and_respect_clearance() -> None:
    profile = torch.full((4,), 0.0035)
    clearance = torch.tensor([0.0200, 0.0200, 0.0030, 0.0200])
    exact = torch.tensor([True, False, False, False])

    offsets, requested, safe_cap = sample_collision_safe_xy_offsets_from_profile(
        profile,
        clearance,
        exact,
        minimum_collision_clearance_m=0.001,
        clearance_guard_m=0.0001,
        magnitude_unit_samples=torch.tensor([1.0, 0.0, 1.0, 0.5]),
        direction_unit_samples=torch.tensor([0.0, 0.25, 0.50, 0.75]),
    )

    magnitudes = torch.linalg.norm(offsets, dim=-1)
    assert requested.tolist() == pytest.approx([0.0, 0.0, 0.0035, 0.00175])
    assert magnitudes.tolist() == pytest.approx([0.0, 0.0, 0.0019, 0.00175])
    assert torch.all(magnitudes <= safe_cap + 1.0e-9)
