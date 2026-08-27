from __future__ import annotations

import pytest

from grasp_planning.visual_servo_clutter import sample_visual_clutter_layouts


def test_clutter_sampler_preserves_clean_environments_and_is_reproducible() -> None:
    first = sample_visual_clutter_layouts(
        256,
        enabled=True,
        seed=17,
        environment_fraction=0.60,
        min_objects=1,
        max_objects=3,
    )
    second = sample_visual_clutter_layouts(
        256,
        enabled=True,
        seed=17,
        environment_fraction=0.60,
        min_objects=1,
        max_objects=3,
    )

    assert first == second
    assert sum(bool(layout) for layout in first) == 154
    assert sum(not layout for layout in first) == 102
    assert all(1 <= len(layout) <= 3 for layout in first if layout)


def test_clutter_primitives_remain_peripheral_and_above_the_flat_plane() -> None:
    layouts = sample_visual_clutter_layouts(
        64,
        enabled=True,
        seed=23,
        environment_fraction=1.0,
        min_objects=3,
        max_objects=3,
    )

    for layout in layouts:
        for item in layout:
            x, y, z = item.position
            assert 0.34 <= x <= 0.50
            assert -0.04 <= y <= 0.14
            assert z > 0.0
            assert z - 0.5 * item.size[2] == pytest.approx(0.001)
            # The nominal target is near (0.425, 0.060); anchors stay outside
            # its central approach corridor.
            assert abs(x - 0.425) >= 0.035 or abs(y - 0.060) >= 0.070


def test_disabled_clutter_has_no_objects() -> None:
    layouts = sample_visual_clutter_layouts(8, enabled=False, seed=3)

    assert layouts == ((),) * 8
