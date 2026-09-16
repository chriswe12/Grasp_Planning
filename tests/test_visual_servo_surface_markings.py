from __future__ import annotations

import math

import pytest

from grasp_planning.visual_servo_surface_markings import sample_surface_markings


def test_markings_are_deterministic_diverse_and_outside_target_corridor() -> None:
    first = sample_surface_markings(
        200,
        enabled=True,
        seed=41,
        environment_fraction=1.0,
        clean_fraction=0.20,
    )
    second = sample_surface_markings(
        200,
        enabled=True,
        seed=41,
        environment_fraction=1.0,
        clean_fraction=0.20,
    )

    assert first == second
    assert 25 <= sum(not markings for markings in first) <= 55
    kinds = {marking.kind for markings in first for marking in markings}
    materials = {marking.material for markings in first for marking in markings}
    assert kinds == {"tape", "writing", "dirt", "scratch"}
    assert {"tape_white", "tape_gray", "tape_black", "tape_blue", "tape_translucent"} <= materials
    dirt_count = sum(
        marking.kind == "dirt" for markings in first for marking in markings
    )
    active_count = sum(bool(markings) for markings in first)
    assert dirt_count >= 1.20 * active_count
    assert all(
        math.hypot(marking.x_m, marking.y_m) >= 0.075
        for markings in first
        for marking in markings
    )
    assert sum(
        math.hypot(marking.x_m, marking.y_m) <= 0.18
        for markings in first
        for marking in markings
    ) >= 0.60 * sum(len(markings) for markings in first)


def test_marking_sampler_supports_completely_clean_scenes_and_validates_ranges() -> None:
    assert sample_surface_markings(4, enabled=False, seed=1) == ((), (), (), ())
    assert sample_surface_markings(
        4, enabled=True, seed=1, environment_fraction=0.0
    ) == ((), (), (), ())
    with pytest.raises(ValueError, match="environment_fraction"):
        sample_surface_markings(1, enabled=True, seed=1, environment_fraction=1.1)
