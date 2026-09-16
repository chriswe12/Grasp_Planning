"""Render-only tape, writing, dirt, and scratch variation for T-slot scenes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

SURFACE_MARKING_PROFILE = "tslot_tape_writing_dirt_scratches_render_only_v2_dirt"
_COLORS = {
    "tape_white": ((0.82, 0.83, 0.80), 0.72, 0.0, 1.0),
    "tape_gray": ((0.31, 0.33, 0.35), 0.62, 0.0, 1.0),
    "tape_black": ((0.018, 0.020, 0.022), 0.70, 0.0, 1.0),
    "tape_blue": ((0.025, 0.12, 0.32), 0.56, 0.0, 1.0),
    "tape_translucent": ((0.62, 0.57, 0.40), 0.42, 0.0, 0.34),
    "ink": ((0.012, 0.014, 0.016), 0.88, 0.0, 1.0),
    "dirt": ((0.075, 0.055, 0.035), 0.96, 0.0, 0.68),
    "scratch": ((0.76, 0.78, 0.79), 0.24, 0.72, 0.85),
}


@dataclass(frozen=True)
class SurfaceMarkingSpec:
    """One flat visual primitive in the local T-slot frame."""

    kind: str
    x_m: float
    y_m: float
    length_m: float
    width_m: float
    yaw_deg: float
    material: str
    height_m: float


def _outside_target_corridor(x_m: float, y_m: float, clearance_radius_m: float) -> bool:
    return math.hypot(x_m, y_m) >= clearance_radius_m


def sample_surface_markings(
    num_envs: int,
    *,
    enabled: bool,
    seed: int,
    environment_fraction: float = 0.75,
    clean_fraction: float = 0.20,
    min_markings: int = 2,
    max_markings: int = 6,
    target_clearance_radius_m: float = 0.075,
) -> tuple[tuple[SurfaceMarkingSpec, ...], ...]:
    """Sample deterministic markings, preserving explicit completely clean scenes."""

    count = int(num_envs)
    if count < 1:
        raise ValueError("num_envs must be positive.")
    if not 0.0 <= environment_fraction <= 1.0 or not 0.0 <= clean_fraction <= 1.0:
        raise ValueError("environment_fraction and clean_fraction must lie in [0, 1].")
    if min_markings < 1 or max_markings < min_markings:
        raise ValueError("Marking counts must satisfy 1 <= min_markings <= max_markings.")
    if target_clearance_radius_m <= 0.0:
        raise ValueError("target_clearance_radius_m must be positive.")
    if not enabled:
        return tuple(() for _ in range(count))

    rng = random.Random(int(seed))
    result: list[tuple[SurfaceMarkingSpec, ...]] = []
    materials = tuple(_COLORS)
    kind_materials = {
        "tape": materials[:5],
        "writing": ("ink",),
        "dirt": ("dirt",),
        "scratch": ("scratch",),
    }
    # Dirt is intentionally the dominant table-wear mode. At the default
    # count this produces about 1.5 dirt patches per active environment,
    # versus about 0.6 in v1, while retaining tape, writing, and scratches.
    kind_weights = (0.28, 0.16, 0.42, 0.14)
    kinds = tuple(kind_materials)
    for _env_index in range(count):
        active = rng.random() < environment_fraction and rng.random() >= clean_fraction
        if not active:
            result.append(())
            continue
        markings: list[SurfaceMarkingSpec] = []
        for _ in range(rng.randint(min_markings, max_markings)):
            for _attempt in range(64):
                if rng.random() < 0.72:
                    # Most marks lie just outside the target footprint, where
                    # the close wrist view can catch only a partial label or
                    # strip. The remaining samples cover the wider tabletop.
                    angle = rng.uniform(-math.pi, math.pi)
                    outer_radius_m = min(0.18, target_clearance_radius_m + 0.09)
                    radius_m = math.sqrt(
                        rng.uniform(
                            target_clearance_radius_m**2,
                            outer_radius_m**2,
                        )
                    )
                    x_m = radius_m * math.cos(angle)
                    y_m = radius_m * math.sin(angle)
                else:
                    x_m = rng.uniform(-0.29, 0.29)
                    y_m = rng.uniform(-0.26, 0.26)
                if _outside_target_corridor(x_m, y_m, target_clearance_radius_m):
                    break
            else:
                continue
            kind = rng.choices(kinds, weights=kind_weights, k=1)[0]
            material = rng.choice(kind_materials[kind])
            if kind == "tape":
                length_m, width_m, height_m = (
                    rng.uniform(0.045, 0.16),
                    rng.uniform(0.010, 0.032),
                    rng.uniform(0.00012, 0.00035),
                )
            elif kind == "writing":
                # Short strokes combine into partial handwritten labels across
                # neighboring samples without embedding readable fixed text.
                length_m, width_m, height_m = (
                    rng.uniform(0.014, 0.050),
                    rng.uniform(0.0010, 0.0025),
                    0.00006,
                )
            elif kind == "dirt":
                length_m, width_m, height_m = (
                    rng.uniform(0.025, 0.090),
                    rng.uniform(0.015, 0.060),
                    0.00004,
                )
            else:
                length_m, width_m, height_m = (
                    rng.uniform(0.025, 0.11),
                    rng.uniform(0.00035, 0.0012),
                    0.00003,
                )
            markings.append(
                SurfaceMarkingSpec(
                    kind=kind,
                    x_m=x_m,
                    y_m=y_m,
                    length_m=length_m,
                    width_m=width_m,
                    yaw_deg=rng.uniform(-180.0, 180.0),
                    material=material,
                    height_m=height_m,
                )
            )
        result.append(tuple(markings))
    return tuple(result)


def spawn_visual_servo_surface_markings(
    num_envs: int,
    *,
    enabled: bool,
    seed: int,
    tslot_variants: Sequence[Any],
    environment_fraction: float = 0.75,
    clean_fraction: float = 0.20,
    min_markings: int = 2,
    max_markings: int = 6,
    target_clearance_radius_m: float = 0.075,
) -> dict[str, Any]:
    """Author markings as visual-only USD primitives aligned with each T-slot.

    The marking roots are siblings of the referenced T-slot asset. USD does not
    permit reliably authoring children beneath an asset-reference instance, so
    the sampled local pose is composed with the T-slot phase and rotation here.
    """

    samples = sample_surface_markings(
        num_envs,
        enabled=enabled,
        seed=seed,
        environment_fraction=environment_fraction,
        clean_fraction=clean_fraction,
        min_markings=min_markings,
        max_markings=max_markings,
        target_clearance_radius_m=target_clearance_radius_m,
    )
    if not enabled:
        return {"profile": SURFACE_MARKING_PROFILE, "samples": samples, "prim_count": 0}
    if len(tslot_variants) != int(num_envs):
        raise ValueError("tslot_variants must contain one layout per environment.")

    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sim.utils.transforms import standardize_xform_ops
    from pxr import UsdGeom

    from grasp_planning.visual_servo_workspace import VISUAL_SERVO_TSLOT_CENTER

    stage = omni.usd.get_context().get_stage()
    material_paths: dict[str, str] = {}
    for name, (color, roughness, metallic, opacity) in _COLORS.items():
        path = f"/World/Looks/SurfaceMarking_{name}"
        cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=roughness,
            metallic=metallic,
            opacity=opacity,
        )
        cfg.func(path, cfg)
        material_paths[name] = path

    # Children authored below env_0 are propagated to the existing cloned
    # environments. Use fixed box/dirt slots and edit inherited transforms and
    # visibility, matching the established visual-clutter authoring pattern.
    source_root = "/World/envs/env_0/SurfaceMarkings"
    UsdGeom.Xform.Define(stage, source_root)
    for marking_index in range(max_markings):
        box = UsdGeom.Cube.Define(stage, f"{source_root}/Slot_{marking_index:02d}_Box")
        box.CreateSizeAttr(1.0)
        dirt = UsdGeom.Cylinder.Define(stage, f"{source_root}/Slot_{marking_index:02d}_Dirt")
        dirt.CreateAxisAttr("Z")
        dirt.CreateRadiusAttr(0.5)
        dirt.CreateHeightAttr(1.0)

    prim_count = 0
    for env_index, markings in enumerate(samples):
        variant = tslot_variants[env_index]
        angle_rad = math.radians(float(variant.rotation_deg))
        cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
        phase_x = float(variant.phase_m) * cosine
        phase_y = float(variant.phase_m) * sine
        root = f"/World/envs/env_{env_index}/SurfaceMarkings"
        for marking_index in range(max_markings):
            box_path = f"{root}/Slot_{marking_index:02d}_Box"
            dirt_path = f"{root}/Slot_{marking_index:02d}_Dirt"
            box_prim = stage.GetPrimAtPath(box_path)
            dirt_prim = stage.GetPrimAtPath(dirt_path)
            if not box_prim.IsValid() or not dirt_prim.IsValid():
                raise RuntimeError(f"Expected inherited surface-marking slots under {root}.")
            box_imageable = UsdGeom.Imageable(box_prim)
            dirt_imageable = UsdGeom.Imageable(dirt_prim)
            if marking_index >= len(markings):
                box_imageable.MakeInvisible()
                dirt_imageable.MakeInvisible()
                continue
            marking = markings[marking_index]
            if marking.kind == "dirt":
                prim, path = dirt_prim, dirt_path
                dirt_imageable.MakeVisible()
                box_imageable.MakeInvisible()
            else:
                prim, path = box_prim, box_path
                box_imageable.MakeVisible()
                dirt_imageable.MakeInvisible()
            world_x = (
                VISUAL_SERVO_TSLOT_CENTER[0]
                + phase_x
                + cosine * marking.x_m
                - sine * marking.y_m
            )
            world_y = (
                VISUAL_SERVO_TSLOT_CENTER[1]
                + phase_y
                + sine * marking.x_m
                + cosine * marking.y_m
            )
            yaw_rad = math.radians(float(variant.rotation_deg) + marking.yaw_deg)
            standardize_xform_ops(
                prim,
                (
                    world_x,
                    world_y,
                    VISUAL_SERVO_TSLOT_CENTER[2] + 0.0004 + 0.5 * marking.height_m,
                ),
                (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)),
                (marking.length_m, marking.width_m, marking.height_m),
            )
            sim_utils.bind_visual_material(path, material_paths[marking.material], stage=stage)
            prim_count += 1
    return {
        "profile": SURFACE_MARKING_PROFILE,
        "samples": samples,
        "prim_count": prim_count,
        "active_environment_count": sum(bool(sample) for sample in samples),
        "collision_surface": "/World/GroundPlane",
    }


__all__ = [
    "SURFACE_MARKING_PROFILE",
    "SurfaceMarkingSpec",
    "sample_surface_markings",
    "spawn_visual_servo_surface_markings",
]
