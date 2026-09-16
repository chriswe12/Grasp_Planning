from __future__ import annotations

from collections import Counter

import pytest

from grasp_planning.isaac_visual_materials import (
    VISUAL_SERVO_CANONICAL_PART_INDEX,
    VISUAL_SERVO_FINGER_COLOR,
    VISUAL_SERVO_PART_PALETTE,
    sample_weighted_part_palette_index,
)
from grasp_planning.visual_servo_workspace import (
    VISUAL_SERVO_TSLOT_ASSET,
    VISUAL_SERVO_TSLOT_BACKGROUNDS,
    VISUAL_SERVO_TSLOT_PITCH_M,
    VISUAL_SERVO_TSLOT_SCALE,
    sample_surface_appearance,
    sample_tslot_layout_variants,
    sample_weighted_tslot_background_index,
    validate_render_only_tslot_asset,
)


def test_tracked_tslot_asset_is_render_only_and_not_cache_local() -> None:
    validate_render_only_tslot_asset()

    assert VISUAL_SERVO_TSLOT_ASSET.is_file()
    assert ".cache" not in VISUAL_SERVO_TSLOT_ASSET.parts
    source = VISUAL_SERVO_TSLOT_ASSET.read_text(encoding="utf-8")
    assert "CollisionAPI" not in source
    assert "double3 xformOp:translate = (0, 0, -0.003)" in source
    assert "double3 xformOp:scale = (0.0205, 0.60, 0.003)" in source
    assert VISUAL_SERVO_TSLOT_PITCH_M == pytest.approx(0.0255)
    assert VISUAL_SERVO_TSLOT_SCALE == pytest.approx((1.0, 1.0, 1.0))


def test_layout_sampler_samples_all_tslot_directions_uniformly_and_bounded() -> None:
    variants = sample_tslot_layout_variants(256, enabled=True, seed=17)
    counts = Counter(variant.name for variant in variants)

    assert counts == {"rotated": 256}
    for variant in variants:
        assert abs(variant.phase_m) <= 0.5 * VISUAL_SERVO_TSLOT_PITCH_M
        assert -90.0 <= variant.rotation_deg <= 90.0
    histogram = [0] * 6
    for variant in variants:
        bin_index = min(5, int((variant.rotation_deg + 90.0) / 30.0))
        histogram[bin_index] += 1
    assert all(25 <= bin_count <= 60 for bin_count in histogram)


def test_disabled_layout_randomization_is_fully_nominal() -> None:
    variants = sample_tslot_layout_variants(8, enabled=False, seed=17)

    assert all(variant.name == "nominal" for variant in variants)
    assert all(variant.phase_m == 0.0 and variant.rotation_deg == 0.0 for variant in variants)


def test_part_palette_is_muted_and_all_colors_are_equally_likely() -> None:
    assert len(VISUAL_SERVO_PART_PALETTE) == 24
    assert VISUAL_SERVO_CANONICAL_PART_INDEX == 0
    assert VISUAL_SERVO_PART_PALETTE[0].name == "soft_brown"
    assert {entry.weight for entry in VISUAL_SERVO_PART_PALETTE} == {1.0}
    assert all(0.0 <= channel <= 0.5 for entry in VISUAL_SERVO_PART_PALETTE for channel in entry.color)
    assert sample_weighted_part_palette_index(0.0) == VISUAL_SERVO_CANONICAL_PART_INDEX
    assert sample_weighted_part_palette_index(1.0) == len(VISUAL_SERVO_PART_PALETTE) - 1
    assert VISUAL_SERVO_FINGER_COLOR == pytest.approx((0.025, 0.030, 0.035))


def test_background_sampler_keeps_neutral_dominant() -> None:
    assert sample_weighted_tslot_background_index(0.0) == 0
    assert sample_weighted_tslot_background_index(0.749) == 0
    assert sample_weighted_tslot_background_index(0.80) == 1
    assert sample_weighted_tslot_background_index(0.95) == 2


def test_continuous_surface_variation_preserves_clean_sample_and_stays_bounded() -> None:
    background = VISUAL_SERVO_TSLOT_BACKGROUNDS[0]
    clean = sample_surface_appearance(
        background.color,
        background.roughness,
        (0.0, 1.0, 0.0, 1.0),
        strength=0.0,
        color_scale=(0.88, 1.12),
        saturation_scale=(0.90, 1.10),
        hue_shift_deg=(-5.0, 5.0),
        roughness=(0.17, 0.33),
    )
    varied = sample_surface_appearance(
        background.color,
        background.roughness,
        (1.0, 1.0, 1.0, 1.0),
        strength=1.0,
        color_scale=(0.88, 1.12),
        saturation_scale=(0.90, 1.10),
        hue_shift_deg=(-5.0, 5.0),
        roughness=(0.17, 0.33),
    )

    assert clean.color == pytest.approx(background.color)
    assert clean.roughness == pytest.approx(background.roughness)
    assert clean.metallic == pytest.approx(0.0)
    assert varied.color != pytest.approx(background.color)
    assert varied.roughness == pytest.approx(0.33)
    assert all(0.0 <= channel <= 1.0 for channel in varied.color)


@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_weighted_samplers_reject_values_outside_unit_interval(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sample_weighted_part_palette_index(value)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sample_weighted_tslot_background_index(value)
