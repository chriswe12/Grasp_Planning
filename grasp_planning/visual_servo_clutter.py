"""Deterministic render-only peripheral clutter for visual-servo training."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

VISUAL_SERVO_CLUTTER_PROFILE = "peripheral_render_depth_primitives_flat_collision_v1"


@dataclass(frozen=True)
class VisualClutterObject:
    """One visual primitive expressed in an environment's local world frame."""

    shape: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_deg: float
    material_index: int


_CLUTTER_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.360, 0.122),
    (0.490, 0.112),
    (0.432, -0.025),
)
_CLUTTER_MATERIALS: tuple[tuple[str, tuple[float, float, float], float, float], ...] = (
    ("muted_blue", (0.04, 0.16, 0.34), 0.42, 0.0),
    ("muted_orange", (0.58, 0.16, 0.025), 0.58, 0.0),
    ("dark_green", (0.06, 0.23, 0.13), 0.62, 0.0),
    ("neutral_plastic", (0.30, 0.32, 0.34), 0.68, 0.0),
    ("steel", (0.34, 0.39, 0.44), 0.24, 0.85),
)


def sample_visual_clutter_layouts(
    num_envs: int,
    *,
    enabled: bool,
    seed: int,
    environment_fraction: float = 0.60,
    min_objects: int = 1,
    max_objects: int = 3,
) -> tuple[tuple[VisualClutterObject, ...], ...]:
    """Sample bounded peripheral clutter while preserving clean environments."""

    count = int(num_envs)
    if count < 1:
        raise ValueError("num_envs must be positive.")
    if not 0.0 <= environment_fraction <= 1.0:
        raise ValueError("environment_fraction must lie in [0, 1].")
    if min_objects < 0 or max_objects < min_objects or max_objects > len(_CLUTTER_ANCHORS):
        raise ValueError(
            f"Expected 0 <= min_objects <= max_objects <= {len(_CLUTTER_ANCHORS)}."
        )
    if not enabled or environment_fraction == 0.0 or max_objects == 0:
        return tuple(() for _ in range(count))

    rng = random.Random(int(seed))
    active_count = round(count * environment_fraction)
    active_indices = set(rng.sample(range(count), active_count))
    layouts: list[tuple[VisualClutterObject, ...]] = []
    for env_index in range(count):
        if env_index not in active_indices:
            layouts.append(())
            continue
        object_count = rng.randint(min_objects, max_objects)
        anchors = rng.sample(_CLUTTER_ANCHORS, object_count)
        objects: list[VisualClutterObject] = []
        for object_index, (anchor_x, anchor_y) in enumerate(anchors):
            # Fixed slot types are important for Isaac's inherited cloned-env
            # namespace: every environment overrides one shared prim layout.
            shape = ("box", "cylinder", "box")[object_index]
            if shape == "box":
                size = (
                    rng.uniform(0.022, 0.050),
                    rng.uniform(0.014, 0.030),
                    rng.uniform(0.012, 0.036),
                )
            else:
                diameter = rng.uniform(0.020, 0.032)
                size = (diameter, diameter, rng.uniform(0.018, 0.045))
            position = (
                anchor_x + rng.uniform(-0.010, 0.010),
                anchor_y + rng.uniform(-0.008, 0.008),
                0.5 * size[2] + 0.001,
            )
            objects.append(
                VisualClutterObject(
                    shape=shape,
                    position=position,
                    size=size,
                    yaw_deg=rng.uniform(-45.0, 45.0),
                    material_index=(env_index + object_index + rng.randrange(len(_CLUTTER_MATERIALS)))
                    % len(_CLUTTER_MATERIALS),
                )
            )
        layouts.append(tuple(objects))
    return tuple(layouts)


def spawn_visual_servo_clutter(
    num_envs: int,
    *,
    enabled: bool,
    seed: int,
    environment_fraction: float = 0.60,
    min_objects: int = 1,
    max_objects: int = 3,
) -> dict[str, Any]:
    """Spawn static visual/depth clutter without collision or rigid-body schemas."""

    layouts = sample_visual_clutter_layouts(
        num_envs,
        enabled=enabled,
        seed=seed,
        environment_fraction=environment_fraction,
        min_objects=min_objects,
        max_objects=max_objects,
    )
    if not any(layouts):
        return {
            "profile": VISUAL_SERVO_CLUTTER_PROFILE,
            "layouts": layouts,
            "prim_paths": (),
            "active_environment_count": 0,
            "collision_surface": "/World/GroundPlane",
        }

    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sim.utils.transforms import standardize_xform_ops
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    material_paths: list[str] = []
    for name, color, roughness, metallic in _CLUTTER_MATERIALS:
        path = f"/World/Looks/VisualServoClutter_{name}"
        cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=roughness,
            metallic=metallic,
        )
        cfg.func(path, cfg)
        material_paths.append(path)

    # Author one fixed set of slots below env_0. Isaac's cloned environments
    # inherit these children, after which every clone gets local transform and
    # visibility overrides. This avoids adding duplicate xform ops to inherited
    # prims and lets clean environments explicitly hide every slot.
    for object_index in range(max_objects):
        prim_path = f"/World/envs/env_0/VisualClutter/Object_{object_index:02d}"
        shape = ("box", "cylinder", "box")[object_index]
        if shape == "box":
            geometry = UsdGeom.Cube.Define(stage, prim_path)
            geometry.CreateSizeAttr(1.0)
        else:
            geometry = UsdGeom.Cylinder.Define(stage, prim_path)
            geometry.CreateAxisAttr("Z")
            geometry.CreateRadiusAttr(0.5)
            geometry.CreateHeightAttr(1.0)

    prim_paths: list[str] = []
    for env_index, objects in enumerate(layouts):
        specs = {object_index: spec for object_index, spec in enumerate(objects)}
        for object_index in range(max_objects):
            prim_path = f"/World/envs/env_{env_index}/VisualClutter/Object_{object_index:02d}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Expected inherited visual clutter prim: {prim_path}")
            imageable = UsdGeom.Imageable(prim)
            spec = specs.get(object_index)
            if spec is None:
                imageable.MakeInvisible()
                if str(imageable.ComputeVisibility()) != "invisible":
                    raise RuntimeError(f"Failed to hide clean-environment clutter prim: {prim_path}")
                continue
            imageable.MakeVisible()
            if str(imageable.ComputeVisibility()) == "invisible":
                raise RuntimeError(f"Failed to show active visual clutter prim: {prim_path}")
            half_yaw = math.radians(spec.yaw_deg) * 0.5
            orientation = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
            standardize_xform_ops(prim, spec.position, orientation, spec.size)
            sim_utils.bind_visual_material(
                prim_path,
                material_paths[spec.material_index],
                stage=stage,
            )
            prim_paths.append(prim_path)

    return {
        "profile": VISUAL_SERVO_CLUTTER_PROFILE,
        "layouts": layouts,
        "prim_paths": tuple(prim_paths),
        "active_environment_count": sum(bool(layout) for layout in layouts),
        "collision_surface": "/World/GroundPlane",
    }


__all__ = [
    "VISUAL_SERVO_CLUTTER_PROFILE",
    "VisualClutterObject",
    "sample_visual_clutter_layouts",
    "spawn_visual_servo_clutter",
]
