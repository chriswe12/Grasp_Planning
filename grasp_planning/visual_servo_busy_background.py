"""Procedural render-only office/factory backgrounds for visual-servo training."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

VISUAL_SERVO_BUSY_BACKGROUND_PROFILE = "procedural_office_factory_people_cables_v5"


@dataclass(frozen=True)
class BusyBackgroundPrimitive:
    """One fixed-slot primitive expressed in an environment's local world frame."""

    slot: int
    role: str
    shape: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    material_index: int


@dataclass(frozen=True)
class BusyBackgroundLayout:
    """One deterministic, per-environment background composition."""

    style: str
    people_count: int
    worker_reach_count: int
    primitives: tuple[BusyBackgroundPrimitive, ...]


_MATERIALS: tuple[tuple[str, tuple[float, float, float], float, float], ...] = (
    ("office_wall", (0.48, 0.53, 0.57), 0.82, 0.0),
    ("office_wall_warm", (0.64, 0.57, 0.47), 0.86, 0.0),
    ("office_wall_cool", (0.39, 0.49, 0.59), 0.84, 0.0),
    ("office_wall_cream", (0.73, 0.69, 0.59), 0.88, 0.0),
    ("factory_wall", (0.16, 0.20, 0.24), 0.72, 0.12),
    ("factory_wall_blue", (0.10, 0.22, 0.31), 0.70, 0.16),
    ("factory_wall_green", (0.14, 0.27, 0.22), 0.76, 0.10),
    ("factory_wall_concrete", (0.36, 0.35, 0.32), 0.91, 0.0),
    ("factory_wall_rust", (0.37, 0.18, 0.10), 0.83, 0.06),
    ("steel", (0.30, 0.34, 0.38), 0.30, 0.78),
    ("safety_yellow", (0.92, 0.58, 0.025), 0.58, 0.05),
    ("bin_blue", (0.035, 0.18, 0.43), 0.65, 0.0),
    ("bin_red", (0.58, 0.045, 0.035), 0.62, 0.0),
    ("cabinet", (0.22, 0.26, 0.29), 0.48, 0.35),
    ("monitor", (0.012, 0.016, 0.022), 0.34, 0.0),
    ("screen", (0.035, 0.31, 0.36), 0.24, 0.12),
    ("cable_black", (0.012, 0.014, 0.017), 0.78, 0.0),
    ("cable_red", (0.46, 0.025, 0.020), 0.70, 0.0),
    ("cable_yellow", (0.73, 0.42, 0.015), 0.68, 0.0),
    ("skin_light", (0.72, 0.48, 0.34), 0.78, 0.0),
    ("skin_medium", (0.46, 0.27, 0.16), 0.80, 0.0),
    ("skin_dark", (0.20, 0.105, 0.060), 0.82, 0.0),
    ("shirt_blue", (0.04, 0.16, 0.38), 0.72, 0.0),
    ("shirt_green", (0.035, 0.28, 0.16), 0.74, 0.0),
    ("shirt_orange", (0.70, 0.22, 0.025), 0.70, 0.0),
    ("shirt_grey", (0.28, 0.31, 0.35), 0.78, 0.0),
    ("shirt_red", (0.52, 0.035, 0.055), 0.74, 0.0),
    ("shirt_purple", (0.31, 0.08, 0.43), 0.76, 0.0),
    ("shirt_cyan", (0.02, 0.39, 0.48), 0.70, 0.0),
    ("shirt_tan", (0.49, 0.36, 0.22), 0.82, 0.0),
    ("high_visibility", (0.79, 0.79, 0.025), 0.66, 0.0),
    ("paper", (0.76, 0.78, 0.74), 0.90, 0.0),
    ("cardboard", (0.48, 0.31, 0.16), 0.91, 0.0),
    ("plastic_green", (0.025, 0.39, 0.16), 0.63, 0.0),
    ("plastic_orange", (0.82, 0.27, 0.015), 0.61, 0.0),
    ("plastic_white", (0.70, 0.72, 0.73), 0.66, 0.0),
)

_MAT = {name: index for index, (name, *_rest) in enumerate(_MATERIALS)}
_STYLES = ("office", "factory", "mixed")
_SIDES = ("rear", "back", "negative_y", "positive_y")
_PERSON_ANCHORS_BY_SIDE: dict[str, tuple[tuple[float, float], ...]] = {
    "rear": ((0.825, 0.030), (0.825, 0.180)),
    # The back wall sits behind the robot base; keep its person away from the
    # base center at (0, 0).
    "back": ((-0.140, -0.300), (-0.140, 0.300)),
    "negative_y": ((0.330, -0.325), (0.540, -0.325)),
    "positive_y": ((0.330, 0.425), (0.540, 0.425)),
}
_MAX_PEOPLE = 4
_WALL_SLOT_START = 0
_WALL_SLOT_COUNT = 4
_PANEL_SLOT_START = _WALL_SLOT_START + _WALL_SLOT_COUNT
_PANEL_SLOT_COUNT = 4
_GAP_PROP_SLOT_START = _PANEL_SLOT_START + _PANEL_SLOT_COUNT
_GAP_PROP_SLOT_COUNT = 8
_CABLE_SLOT_START = _GAP_PROP_SLOT_START + _GAP_PROP_SLOT_COUNT
_CABLE_SLOT_COUNT = 8
_PERSON_SLOT_START = _CABLE_SLOT_START + _CABLE_SLOT_COUNT
_PERSON_SLOT_COUNT = 4
_WORKER_REACH_SLOT_START = _PERSON_SLOT_START + _MAX_PEOPLE * _PERSON_SLOT_COUNT
_MAX_WORKER_REACHES = 4
_WORKER_REACH_SLOT_COUNT = 2
_TOTAL_SLOTS = _WORKER_REACH_SLOT_START + _MAX_WORKER_REACHES * _WORKER_REACH_SLOT_COUNT

# Every environment must use the same primitive type at a given slot because
# Isaac's cloned environments inherit one authored env_0 namespace.
_SLOT_SHAPES: tuple[str, ...] = (
    *("box" for _ in range(_WALL_SLOT_COUNT)),
    *("box" for _ in range(_PANEL_SLOT_COUNT)),
    *("box" for _ in range(_GAP_PROP_SLOT_COUNT)),
    *("cylinder" for _ in range(_CABLE_SLOT_COUNT)),
    *(shape for _ in range(_MAX_PEOPLE) for shape in ("sphere", "box", "cylinder", "cylinder")),
    *(shape for _ in range(_MAX_WORKER_REACHES) for shape in ("cylinder", "sphere")),
)

if len(_SLOT_SHAPES) != _TOTAL_SLOTS:  # pragma: no cover - source contract
    raise AssertionError("Busy-background slot schema is inconsistent.")


def _identity() -> tuple[float, float, float, float]:
    return (1.0, 0.0, 0.0, 0.0)


def _quat_from_z_to_vector(
    vector: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Return a unit quaternion rotating the local +Z axis onto ``vector``."""

    x, y, z = vector
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("Cannot orient a cylinder along a zero-length vector.")
    x, y, z = x / norm, y / norm, z / norm
    if z < -1.0 + 1.0e-9:
        return (0.0, 1.0, 0.0, 0.0)
    w = math.sqrt(max(0.0, 0.5 * (1.0 + z)))
    scale = 0.5 / w if w > 1.0e-12 else 0.0
    return (w, -y * scale, x * scale, 0.0)


def _box(
    slot: int,
    role: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    material: str,
) -> BusyBackgroundPrimitive:
    return BusyBackgroundPrimitive(slot, role, "box", position, size, _identity(), _MAT[material])


def _box_yaw(
    slot: int,
    role: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
    material: str,
) -> BusyBackgroundPrimitive:
    half_yaw = 0.5 * math.radians(yaw_deg)
    return BusyBackgroundPrimitive(
        slot,
        role,
        "box",
        position,
        size,
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        _MAT[material],
    )


def _vertical_cylinder(
    slot: int,
    role: str,
    position: tuple[float, float, float],
    diameter: float,
    height: float,
    material: str,
) -> BusyBackgroundPrimitive:
    return BusyBackgroundPrimitive(
        slot,
        role,
        "cylinder",
        position,
        (diameter, diameter, height),
        _identity(),
        _MAT[material],
    )


def _sphere(
    slot: int,
    role: str,
    position: tuple[float, float, float],
    diameter: float,
    material: str,
) -> BusyBackgroundPrimitive:
    return BusyBackgroundPrimitive(
        slot,
        role,
        "sphere",
        position,
        (diameter, diameter, diameter),
        _identity(),
        _MAT[material],
    )


def _segment(
    slot: int,
    role: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    diameter: float,
    material: str,
) -> BusyBackgroundPrimitive:
    vector = tuple(finish - begin for begin, finish in zip(start, end, strict=True))
    length = math.sqrt(sum(component * component for component in vector))
    midpoint = tuple(0.5 * (begin + finish) for begin, finish in zip(start, end, strict=True))
    return BusyBackgroundPrimitive(
        slot,
        role,
        "cylinder",
        midpoint,
        (diameter, diameter, length),
        _quat_from_z_to_vector(vector),
        _MAT[material],
    )


def _sample_active_layout(
    rng: random.Random,
    *,
    min_people: int,
    max_people: int,
) -> BusyBackgroundLayout:
    style = rng.choice(_STYLES)
    primitives: list[BusyBackgroundPrimitive] = []
    # The T-slot surface spans x=[0.10, 0.75], y=[-0.25, 0.35]. Walls sit
    # outside that footprint, leaving visible staging strips for props. Their
    # bottoms extend to z=-0.06, below the T-slot backing at z=-0.011, so no
    # black horizon seam can appear between the table and a wall. The top is
    # z=0.46, 12 cm higher than v4, so room dressing remains behind tall people
    # and oblique wrist-camera views.
    office_walls = ("office_wall", "office_wall_warm", "office_wall_cool", "office_wall_cream")
    factory_walls = (
        "factory_wall",
        "factory_wall_blue",
        "factory_wall_green",
        "factory_wall_concrete",
        "factory_wall_rust",
    )
    wall_palette = office_walls if style == "office" else factory_walls
    if style == "mixed":
        wall_materials = tuple(
            rng.choice(office_walls if side_index % 2 else factory_walls)
            for side_index in range(len(_SIDES))
        )
    else:
        primary_wall = rng.choice(wall_palette)
        wall_materials = tuple(
            primary_wall if rng.random() < 0.70 else rng.choice(wall_palette)
            for _ in _SIDES
        )
    wall_specs = {
        "rear": ((0.910, 0.050, 0.200), (0.020, 0.940, 0.520)),
        "back": ((-0.190, 0.050, 0.200), (0.020, 0.940, 0.520)),
        "negative_y": ((0.360, -0.410, 0.200), (1.100, 0.020, 0.520)),
        "positive_y": ((0.360, 0.510, 0.200), (1.100, 0.020, 0.520)),
    }
    office_panels = ("screen", "monitor", "paper", "plastic_white", "shirt_blue")
    factory_panels = (
        "safety_yellow",
        "bin_red",
        "bin_blue",
        "high_visibility",
        "plastic_orange",
    )
    for side_index, side in enumerate(_SIDES):
        wall_position, wall_size = wall_specs[side]
        panel_height = rng.uniform(0.070, 0.155)
        panel_center_z = rng.uniform(0.185, 0.350)
        if side in {"rear", "back"}:
            panel_position = (
                0.899 if side == "rear" else -0.179,
                rng.uniform(-0.255, 0.345),
                panel_center_z,
            )
            panel_size = (0.006, rng.uniform(0.090, 0.240), panel_height)
        else:
            panel_position = (
                rng.uniform(0.145, 0.675),
                -0.399 if side == "negative_y" else 0.499,
                panel_center_z,
            )
            panel_size = (rng.uniform(0.090, 0.240), 0.006, panel_height)
        panel_palette = (
            office_panels
            if style == "office"
            else factory_panels
            if style == "factory"
            else office_panels + factory_panels
        )
        primitives.extend(
            (
                _box(
                    _WALL_SLOT_START + side_index,
                    f"{side}_wall",
                    wall_position,
                    wall_size,
                    wall_materials[side_index],
                ),
                _box(
                    _PANEL_SLOT_START + side_index,
                    f"{side}_wall_panel",
                    panel_position,
                    panel_size,
                    rng.choice(panel_palette),
                ),
            )
        )

    # Every side receives exactly two similarly sized props in the strip
    # between wall and T-slot. Shape/material details vary, but density does
    # not, preventing one direction from becoming systematically busier.
    gap_prop_anchors = {
        "rear": ((0.825, -0.215), (0.825, 0.270)),
        "back": ((-0.120, -0.205), (-0.120, 0.105)),
        "negative_y": ((0.175, -0.325), (0.675, -0.325)),
        "positive_y": ((0.175, 0.425), (0.675, 0.425)),
    }
    office_props = (
        "cabinet",
        "monitor",
        "paper",
        "cardboard",
        "plastic_white",
        "shirt_blue",
        "shirt_green",
    )
    factory_props = (
        "steel",
        "safety_yellow",
        "bin_blue",
        "bin_red",
        "cardboard",
        "plastic_green",
        "plastic_orange",
        "high_visibility",
    )
    prop_materials = (
        office_props if style == "office" else factory_props if style == "factory" else office_props + factory_props
    )
    for side_index, side in enumerate(_SIDES):
        for prop_index, (anchor_x, anchor_y) in enumerate(gap_prop_anchors[side]):
            size = (
                rng.uniform(0.028, 0.082),
                rng.uniform(0.026, 0.078),
                rng.uniform(0.025, 0.135),
            )
            primitives.append(
                _box_yaw(
                    _GAP_PROP_SLOT_START + side_index * 2 + prop_index,
                    f"{side}_gap_prop",
                    (
                        anchor_x + rng.uniform(-0.012, 0.012),
                        anchor_y + rng.uniform(-0.012, 0.012),
                        0.5 * size[2],
                    ),
                    size,
                    rng.uniform(-35.0, 35.0),
                    rng.choice(prop_materials),
                )
            )

    # Two connected cable segments per side supply the same fine depth/RGB
    # structure around the complete perimeter.
    cable_paths = {
        "rear": ((0.790, -0.120, 0.006), (0.850, -0.085, 0.006), (0.795, -0.040, 0.006)),
        "back": ((-0.155, -0.120, 0.006), (-0.085, -0.075, 0.006), (-0.145, -0.025, 0.006)),
        "negative_y": ((0.500, -0.370, 0.006), (0.560, -0.315, 0.006), (0.625, -0.365, 0.006)),
        "positive_y": ((0.500, 0.470, 0.006), (0.560, 0.415, 0.006), (0.625, 0.465, 0.006)),
    }
    cable_materials = ("cable_black", "cable_red", "cable_yellow")
    for side_index, side in enumerate(_SIDES):
        points = tuple(
            (
                x + rng.uniform(-0.016, 0.016),
                y + rng.uniform(-0.016, 0.016),
                z + rng.uniform(0.0, 0.004),
            )
            for x, y, z in cable_paths[side]
        )
        material = rng.choice(cable_materials)
        for segment_index in range(2):
            primitives.append(
                _segment(
                    _CABLE_SLOT_START + side_index * 2 + segment_index,
                    f"{side}_gap_cable",
                    points[segment_index],
                    points[segment_index + 1],
                    rng.uniform(0.0035, 0.0080),
                    material,
                )
            )

    people_count = rng.randint(min_people, max_people)
    selected_sides = list(_SIDES) if people_count == len(_SIDES) else rng.sample(_SIDES, people_count)
    skin_materials = ("skin_light", "skin_medium", "skin_dark")
    shirt_materials = (
        "shirt_blue",
        "shirt_green",
        "shirt_orange",
        "shirt_grey",
        "shirt_red",
        "shirt_purple",
        "shirt_cyan",
        "shirt_tan",
        "high_visibility",
    )
    person_positions: dict[str, tuple[float, float]] = {}
    for person_index, side in enumerate(selected_sides):
        anchor_x, anchor_y = rng.choice(_PERSON_ANCHORS_BY_SIDE[side])
        slot = _PERSON_SLOT_START + person_index * _PERSON_SLOT_COUNT
        person_x = anchor_x + rng.uniform(-0.018, 0.018)
        person_y = anchor_y + rng.uniform(-0.018, 0.018)
        person_positions[side] = (person_x, person_y)
        height_scale = rng.uniform(0.82, 1.18)
        head_z = rng.uniform(0.350, 0.415)
        torso_z = head_z - 0.105 * height_scale
        shoulder_z = torso_z + 0.030 * height_scale
        skin = rng.choice(skin_materials)
        shirt = rng.choice(shirt_materials)
        torso_width = rng.uniform(0.052, 0.074) * height_scale
        torso_depth = rng.uniform(0.068, 0.094) * height_scale
        torso_height = rng.uniform(0.125, 0.165) * height_scale
        primitives.extend(
            (
                _sphere(slot, "person_head", (person_x, person_y, head_z), 0.058 * height_scale, skin),
                _box_yaw(
                    slot + 1,
                    f"{side}_person_torso",
                    (person_x, person_y, torso_z),
                    (torso_width, torso_depth, torso_height),
                    rng.uniform(-18.0, 18.0),
                    shirt,
                ),
                _segment(
                    slot + 2,
                    f"{side}_person_arm",
                    (person_x, person_y - 0.033 * height_scale, shoulder_z),
                    (
                        person_x + rng.uniform(-0.040, 0.040),
                        person_y - rng.uniform(0.055, 0.095) * height_scale,
                        torso_z + rng.uniform(-0.055, 0.025),
                    ),
                    rng.uniform(0.017, 0.025) * height_scale,
                    shirt,
                ),
                _segment(
                    slot + 3,
                    f"{side}_person_arm",
                    (person_x, person_y + 0.033 * height_scale, shoulder_z),
                    (
                        person_x + rng.uniform(-0.040, 0.040),
                        person_y + rng.uniform(0.055, 0.095) * height_scale,
                        torso_z + rng.uniform(-0.055, 0.025),
                    ),
                    rng.uniform(0.017, 0.025) * height_scale,
                    shirt,
                ),
            )
        )

    # Give every occupied side one matching table-edge reach. Endpoints stay
    # near the T-slot perimeter and away from the central grasp corridor.
    reach_endpoints = {
        "rear": (0.700, 0.030, 0.024),
        "back": (0.140, 0.220, 0.024),
        "negative_y": (0.330, -0.200, 0.024),
        "positive_y": (0.540, 0.300, 0.024),
    }
    for reach_index, side in enumerate(selected_sides):
        slot = _WORKER_REACH_SLOT_START + reach_index * _WORKER_REACH_SLOT_COUNT
        sleeve = rng.choice(shirt_materials)
        hand = rng.choice(
            skin_materials if style == "office" else ("bin_blue", "shirt_grey")
        )
        outer_x, outer_y = person_positions[side]
        inner_x, inner_y, inner_z = reach_endpoints[side]
        if side in {"rear", "back"}:
            inner_x += rng.uniform(-0.012, 0.012)
            inner_y += rng.uniform(-0.060, 0.060)
        else:
            inner_x += rng.uniform(-0.060, 0.060)
            inner_y += rng.uniform(-0.012, 0.012)
        inner_z += rng.uniform(-0.006, 0.012)
        primitives.extend(
            (
                _segment(
                    slot,
                    f"{side}_worker_sleeve",
                    (outer_x, outer_y, rng.uniform(0.022, 0.045)),
                    (inner_x, inner_y, inner_z),
                    rng.uniform(0.032, 0.052),
                    sleeve,
                ),
                _sphere(
                    slot + 1,
                    f"{side}_worker_hand",
                    (inner_x, inner_y, inner_z),
                    rng.uniform(0.034, 0.052),
                    hand,
                ),
            )
        )

    slots = [primitive.slot for primitive in primitives]
    if len(slots) != len(set(slots)):  # pragma: no cover - source contract
        raise AssertionError(f"Busy-background layout reused slots: {slots}")
    return BusyBackgroundLayout(
        style=style,
        people_count=people_count,
        worker_reach_count=len(selected_sides),
        primitives=tuple(primitives),
    )


def sample_busy_background_layouts(
    num_envs: int,
    *,
    enabled: bool,
    seed: int,
    environment_fraction: float = 0.70,
    min_people: int = 2,
    max_people: int = 4,
) -> tuple[BusyBackgroundLayout, ...]:
    """Sample deterministic busy scenes while preserving a clean subset."""

    count = int(num_envs)
    if count < 1:
        raise ValueError("num_envs must be positive.")
    if not 0.0 <= environment_fraction <= 1.0:
        raise ValueError("environment_fraction must lie in [0, 1].")
    if min_people < 0 or max_people < min_people or max_people > _MAX_PEOPLE:
        raise ValueError(f"Expected 0 <= min_people <= max_people <= {_MAX_PEOPLE}.")
    clean = BusyBackgroundLayout(style="clean", people_count=0, worker_reach_count=0, primitives=())
    if not enabled or environment_fraction == 0.0:
        return tuple(clean for _ in range(count))

    rng = random.Random(int(seed))
    active_count = round(count * environment_fraction)
    active_indices = set(rng.sample(range(count), active_count))
    return tuple(
        _sample_active_layout(rng, min_people=min_people, max_people=max_people)
        if env_index in active_indices
        else clean
        for env_index in range(count)
    )


def spawn_visual_servo_busy_background(
    num_envs: int,
    *,
    enabled: bool,
    seed: int,
    environment_fraction: float = 0.70,
    min_people: int = 2,
    max_people: int = 4,
) -> dict[str, Any]:
    """Spawn clone-safe background geometry without collision or rigid bodies."""

    layouts = sample_busy_background_layouts(
        num_envs,
        enabled=enabled,
        seed=seed,
        environment_fraction=environment_fraction,
        min_people=min_people,
        max_people=max_people,
    )
    if not any(layout.primitives for layout in layouts):
        return {
            "profile": VISUAL_SERVO_BUSY_BACKGROUND_PROFILE,
            "layouts": layouts,
            "prim_paths": (),
            "active_environment_count": 0,
            "people_count": 0,
            "worker_reach_count": 0,
            "style_counts": {},
            "collision_surface": "/World/GroundPlane",
        }

    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sim.utils.transforms import standardize_xform_ops
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    material_paths: list[str] = []
    for name, color, roughness, metallic in _MATERIALS:
        path = f"/World/Looks/VisualServoBusyBackground_{name}"
        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness, metallic=metallic)
        cfg.func(path, cfg)
        material_paths.append(path)

    for slot, shape in enumerate(_SLOT_SHAPES):
        prim_path = f"/World/envs/env_0/BusyBackground/Slot_{slot:02d}"
        if shape == "box":
            geometry = UsdGeom.Cube.Define(stage, prim_path)
            geometry.CreateSizeAttr(1.0)
        elif shape == "sphere":
            geometry = UsdGeom.Sphere.Define(stage, prim_path)
            geometry.CreateRadiusAttr(0.5)
        elif shape == "cylinder":
            geometry = UsdGeom.Cylinder.Define(stage, prim_path)
            geometry.CreateAxisAttr("Z")
            geometry.CreateRadiusAttr(0.5)
            geometry.CreateHeightAttr(1.0)
        else:  # pragma: no cover - source contract
            raise AssertionError(f"Unsupported busy-background slot shape: {shape}")

    prim_paths: list[str] = []
    for env_index, layout in enumerate(layouts):
        specs = {primitive.slot: primitive for primitive in layout.primitives}
        for slot, expected_shape in enumerate(_SLOT_SHAPES):
            prim_path = f"/World/envs/env_{env_index}/BusyBackground/Slot_{slot:02d}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Expected inherited busy-background prim: {prim_path}")
            imageable = UsdGeom.Imageable(prim)
            spec = specs.get(slot)
            if spec is None:
                imageable.MakeInvisible()
                continue
            if spec.shape != expected_shape:
                raise RuntimeError(
                    f"Busy-background slot {slot} expects {expected_shape}, got {spec.shape}."
                )
            imageable.MakeVisible()
            standardize_xform_ops(prim, spec.position, spec.orientation_wxyz, spec.size)
            sim_utils.bind_visual_material(prim_path, material_paths[spec.material_index], stage=stage)
            prim_paths.append(prim_path)

    style_counts = Counter(layout.style for layout in layouts if layout.primitives)
    return {
        "profile": VISUAL_SERVO_BUSY_BACKGROUND_PROFILE,
        "layouts": layouts,
        "prim_paths": tuple(prim_paths),
        "active_environment_count": sum(bool(layout.primitives) for layout in layouts),
        "people_count": sum(layout.people_count for layout in layouts),
        "worker_reach_count": sum(layout.worker_reach_count for layout in layouts),
        "style_counts": dict(style_counts),
        "collision_surface": "/World/GroundPlane",
    }


__all__ = [
    "BusyBackgroundLayout",
    "BusyBackgroundPrimitive",
    "VISUAL_SERVO_BUSY_BACKGROUND_PROFILE",
    "sample_busy_background_layouts",
    "spawn_visual_servo_busy_background",
]
