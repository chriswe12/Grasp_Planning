from __future__ import annotations

import pytest

from grasp_planning.visual_servo_busy_background import sample_busy_background_layouts


def test_busy_background_sampler_is_reproducible_and_preserves_clean_environments() -> None:
    first = sample_busy_background_layouts(
        256,
        enabled=True,
        seed=31,
        environment_fraction=0.70,
        min_people=2,
        max_people=4,
    )
    second = sample_busy_background_layouts(
        256,
        enabled=True,
        seed=31,
        environment_fraction=0.70,
        min_people=2,
        max_people=4,
    )

    assert first == second
    assert sum(bool(layout.primitives) for layout in first) == 179
    assert sum(not layout.primitives for layout in first) == 77
    assert {layout.style for layout in first if layout.primitives} == {"office", "factory", "mixed"}
    assert all(2 <= layout.people_count <= 4 for layout in first if layout.primitives)
    assert all(
        layout.worker_reach_count == layout.people_count for layout in first if layout.primitives
    )


def test_busy_background_has_unique_slots_and_stays_behind_the_task_corridor() -> None:
    layouts = sample_busy_background_layouts(
        32,
        enabled=True,
        seed=41,
        environment_fraction=1.0,
        min_people=4,
        max_people=4,
    )

    for layout in layouts:
        slots = [primitive.slot for primitive in layout.primitives]
        assert len(slots) == len(set(slots))
        assert layout.people_count == 4
        heads = [primitive for primitive in layout.primitives if primitive.role == "person_head"]
        cables = [primitive for primitive in layout.primitives if primitive.role.endswith("_gap_cable")]
        walls = [primitive for primitive in layout.primitives if primitive.role.endswith("_wall")]
        panels = [primitive for primitive in layout.primitives if primitive.role.endswith("_wall_panel")]
        props = [primitive for primitive in layout.primitives if primitive.role.endswith("_gap_prop")]
        hands = [primitive for primitive in layout.primitives if primitive.role.endswith("_worker_hand")]
        assert len(heads) == 4
        assert len(cables) == 8
        assert len(walls) == 4
        assert len(panels) == 4
        assert len(props) == 8
        assert len(hands) == layout.worker_reach_count
        assert all(head.position[2] >= 0.335 for head in heads)
        assert all(cable.position[2] >= 0.006 for cable in cables)

        by_role = {primitive.role: primitive for primitive in walls}
        assert set(by_role) == {"rear_wall", "back_wall", "negative_y_wall", "positive_y_wall"}
        # T-slot bounds are x=[0.10, 0.75], y=[-0.25, 0.35]. Every wall is
        # outside those edges, and all wall bottoms overlap below the visual
        # T-slot backing at z=-0.011 so no black seam remains.
        assert by_role["rear_wall"].position[0] - 0.5 * by_role["rear_wall"].size[0] > 0.75
        assert by_role["back_wall"].position[0] + 0.5 * by_role["back_wall"].size[0] < 0.10
        assert (
            by_role["negative_y_wall"].position[1]
            + 0.5 * by_role["negative_y_wall"].size[1]
            < -0.25
        )
        assert (
            by_role["positive_y_wall"].position[1]
            - 0.5 * by_role["positive_y_wall"].size[1]
            > 0.35
        )
        assert all(wall.position[2] - 0.5 * wall.size[2] <= -0.05 for wall in walls)
        assert all(wall.position[2] + 0.5 * wall.size[2] >= 0.45 for wall in walls)

        for side in ("rear", "back", "negative_y", "positive_y"):
            side_props = [primitive for primitive in props if primitive.role == f"{side}_gap_prop"]
            side_cables = [
                primitive for primitive in cables if primitive.role == f"{side}_gap_cable"
            ]
            side_hands = [
                primitive for primitive in hands if primitive.role == f"{side}_worker_hand"
            ]
            assert len(side_props) == 2
            assert len(side_cables) == 2
            assert len(side_hands) == 1

        assert all(0.75 < prop.position[0] < 0.90 for prop in props if prop.role == "rear_gap_prop")
        assert all(-0.18 < prop.position[0] < 0.10 for prop in props if prop.role == "back_gap_prop")
        assert all(
            -0.40 < prop.position[1] < -0.25
            for prop in props
            if prop.role == "negative_y_gap_prop"
        )
        assert all(
            0.35 < prop.position[1] < 0.50
            for prop in props
            if prop.role == "positive_y_gap_prop"
        )


def test_busy_background_varies_materials_geometry_people_and_reaches_across_environments() -> None:
    layouts = sample_busy_background_layouts(
        96,
        enabled=True,
        seed=97,
        environment_fraction=1.0,
        min_people=4,
        max_people=4,
    )

    wall_material_signatures = set()
    panel_geometry_signatures = set()
    prop_materials = set()
    prop_geometry_signatures = set()
    torso_materials = set()
    head_heights = set()
    reach_endpoints = set()
    for layout in layouts:
        walls = [primitive for primitive in layout.primitives if primitive.role.endswith("_wall")]
        panels = [
            primitive for primitive in layout.primitives if primitive.role.endswith("_wall_panel")
        ]
        props = [primitive for primitive in layout.primitives if primitive.role.endswith("_gap_prop")]
        torsos = [
            primitive for primitive in layout.primitives if primitive.role.endswith("_person_torso")
        ]
        heads = [primitive for primitive in layout.primitives if primitive.role == "person_head"]
        hands = [
            primitive for primitive in layout.primitives if primitive.role.endswith("_worker_hand")
        ]
        wall_material_signatures.add(tuple(primitive.material_index for primitive in walls))
        panel_geometry_signatures.add(
            tuple(
                (*[round(value, 3) for value in primitive.position], *[round(value, 3) for value in primitive.size])
                for primitive in panels
            )
        )
        prop_materials.update(primitive.material_index for primitive in props)
        prop_geometry_signatures.add(
            tuple(round(value, 3) for primitive in props for value in primitive.size)
        )
        torso_materials.update(primitive.material_index for primitive in torsos)
        head_heights.update(round(primitive.position[2], 3) for primitive in heads)
        reach_endpoints.update(
            tuple(round(value, 3) for value in primitive.position) for primitive in hands
        )

    assert len(wall_material_signatures) >= 12
    assert len(panel_geometry_signatures) >= 80
    assert len(prop_materials) >= 12
    assert len(prop_geometry_signatures) >= 80
    assert len(torso_materials) >= 8
    assert len(head_heights) >= 40
    assert len(reach_endpoints) >= 100


def test_disabled_busy_background_is_clean() -> None:
    layouts = sample_busy_background_layouts(8, enabled=False, seed=5)

    assert all(layout.style == "clean" for layout in layouts)
    assert all(layout.people_count == 0 for layout in layouts)
    assert all(layout.worker_reach_count == 0 for layout in layouts)
    assert all(not layout.primitives for layout in layouts)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    ((-1, 2), (3, 2), (0, 5)),
)
def test_busy_background_rejects_invalid_people_ranges(minimum: int, maximum: int) -> None:
    with pytest.raises(ValueError, match="min_people"):
        sample_busy_background_layouts(
            2,
            enabled=True,
            seed=1,
            min_people=minimum,
            max_people=maximum,
        )
