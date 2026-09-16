"""Shared Isaac visual materials for wrist-camera grasp alignment."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Any, Sequence

VISUAL_SERVO_MATERIAL_PROFILE = (
    "muted_fdm_palette_pdz_black_whitepads_leafbindings_small_tslot_v7"
)

_FINGER_LINK_NAMES = frozenset(
    {
        "left_finger_link",
        "right_finger_link",
        "pdz_gripper_left_finger_link",
        "pdz_gripper_right_finger_link",
    }
)
_CONTACT_PAD_PATH_TOKENS = ("pad_8mm", "tpu_pad")
_VISUAL_GEOMETRY_TYPE_NAMES = frozenset(
    {"Capsule", "Cone", "Cube", "Cylinder", "Mesh", "Sphere"}
)


@dataclass(frozen=True)
class VisualServoPartMaterial:
    """One deliberately muted, printable part appearance."""

    name: str
    color: tuple[float, float, float]
    weight: float


def _rgb8(red: int, green: int, blue: int) -> tuple[float, float, float]:
    return red / 255.0, green / 255.0, blue / 255.0


# All printable part colors are equally likely during training.  The first
# entry remains a deterministic fallback for tools that require one material,
# but it has no privileged sampling probability.
VISUAL_SERVO_PART_PALETTE: tuple[VisualServoPartMaterial, ...] = (
    VisualServoPartMaterial("soft_brown", _rgb8(95, 72, 58), 1.0),
    VisualServoPartMaterial("soft_clay", _rgb8(100, 64, 62), 1.0),
    VisualServoPartMaterial("soft_sage", _rgb8(64, 88, 68), 1.0),
    VisualServoPartMaterial("soft_slate", _rgb8(65, 74, 94), 1.0),
    VisualServoPartMaterial("soft_bluegray", _rgb8(60, 82, 84), 1.0),
    VisualServoPartMaterial("soft_tan", _rgb8(105, 86, 55), 1.0),
    VisualServoPartMaterial("soft_burgundy", _rgb8(78, 48, 58), 1.0),
    VisualServoPartMaterial("soft_rust", _rgb8(98, 64, 46), 1.0),
    VisualServoPartMaterial("soft_olive", _rgb8(78, 82, 55), 1.0),
    VisualServoPartMaterial("soft_moss", _rgb8(54, 76, 58), 1.0),
    VisualServoPartMaterial("soft_denim", _rgb8(52, 66, 88), 1.0),
    VisualServoPartMaterial("soft_mauve", _rgb8(82, 64, 78), 1.0),
    VisualServoPartMaterial("soft_charcoal", _rgb8(52, 56, 58), 1.0),
    VisualServoPartMaterial("soft_cream", _rgb8(100, 94, 80), 1.0),
    VisualServoPartMaterial("soft_darkblue", _rgb8(40, 52, 72), 1.0),
    VisualServoPartMaterial("soft_dustyrose", _rgb8(92, 67, 70), 1.0),
    VisualServoPartMaterial("soft_yellow", _rgb8(108, 98, 55), 1.0),
    VisualServoPartMaterial("soft_purple", _rgb8(78, 65, 88), 1.0),
    VisualServoPartMaterial("soft_red", _rgb8(104, 61, 59), 1.0),
    VisualServoPartMaterial("soft_blue", _rgb8(58, 72, 96), 1.0),
    VisualServoPartMaterial("soft_green", _rgb8(58, 86, 64), 1.0),
    VisualServoPartMaterial("soft_orange", _rgb8(110, 78, 52), 1.0),
    VisualServoPartMaterial("soft_black", _rgb8(45, 47, 49), 1.0),
    VisualServoPartMaterial("soft_white", _rgb8(108, 104, 96), 1.0),
)
VISUAL_SERVO_CANONICAL_PART_INDEX = 0
VISUAL_SERVO_PART_COLOR = VISUAL_SERVO_PART_PALETTE[VISUAL_SERVO_CANONICAL_PART_INDEX].color
VISUAL_SERVO_PART_ROUGHNESS = 0.80
VISUAL_SERVO_FINGER_COLOR = (0.025, 0.030, 0.035)
VISUAL_SERVO_FINGER_ROUGHNESS = 0.48
VISUAL_SERVO_CONTACT_PAD_COLOR = (0.95, 0.96, 0.97)
VISUAL_SERVO_CONTACT_PAD_ROUGHNESS = 0.72
VISUAL_SERVO_WORK_SURFACE_COLOR = (0.075, 0.085, 0.10)
VISUAL_SERVO_WORK_SURFACE_ROUGHNESS = 0.90
PDZ_GRIPPER_APPEARANCE_VARIANT_SET = "pdzAppearance"


@dataclass(frozen=True)
class VisualServoGripperAppearanceVariant:
    """One render-material pair supported by the instanced PDZ robot USD."""

    name: str
    finger_color: tuple[float, float, float]
    finger_roughness: float
    pad_color: tuple[float, float, float]
    pad_roughness: float


def _clamped_scaled_color(
    color: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    return tuple(min(1.0, max(0.0, channel * scale)) for channel in color)  # type: ignore[return-value]


def _finger_variant_color(scale: float, hue_shift_deg: float) -> tuple[float, float, float]:
    hue, saturation, value = colorsys.rgb_to_hsv(*VISUAL_SERVO_FINGER_COLOR)
    shifted = colorsys.hsv_to_rgb((hue + hue_shift_deg / 360.0) % 1.0, saturation, value)
    return _clamped_scaled_color(shifted, scale)


def _pad_variant_color(scale: float, temperature_shift: float) -> tuple[float, float, float]:
    gains = (1.0 + temperature_shift, 1.0, 1.0 - temperature_shift)
    shifted = tuple(
        min(1.0, max(0.0, channel * gain))
        for channel, gain in zip(VISUAL_SERVO_CONTACT_PAD_COLOR, gains, strict=True)
    )
    return _clamped_scaled_color(shifted, scale)


def _gripper_variant(
    name: str,
    *,
    finger_scale: float,
    finger_hue_deg: float,
    finger_roughness: float,
    pad_scale: float,
    pad_temperature: float,
    pad_roughness: float,
) -> VisualServoGripperAppearanceVariant:
    return VisualServoGripperAppearanceVariant(
        name=name,
        finger_color=_finger_variant_color(finger_scale, finger_hue_deg),
        finger_roughness=finger_roughness,
        pad_color=_pad_variant_color(pad_scale, pad_temperature),
        pad_roughness=pad_roughness,
    )


# USD instances cannot accept opinions on their descendant mesh proxies. These
# variants are authored inside the source visual scopes so runtime selection on
# the instance root changes the real rendered shaders without de-instancing 256
# robots. Non-canonical variants remain recognizably black/white.
VISUAL_SERVO_GRIPPER_APPEARANCE_VARIANTS: tuple[VisualServoGripperAppearanceVariant, ...] = (
    VisualServoGripperAppearanceVariant(
        name="canonical",
        finger_color=VISUAL_SERVO_FINGER_COLOR,
        finger_roughness=VISUAL_SERVO_FINGER_ROUGHNESS,
        pad_color=VISUAL_SERVO_CONTACT_PAD_COLOR,
        pad_roughness=VISUAL_SERVO_CONTACT_PAD_ROUGHNESS,
    ),
    _gripper_variant(
        "dark_glossy_cool",
        finger_scale=0.50,
        finger_hue_deg=-6.0,
        finger_roughness=0.30,
        pad_scale=0.88,
        pad_temperature=-0.03,
        pad_roughness=0.55,
    ),
    _gripper_variant(
        "dark_matte_warm",
        finger_scale=0.50,
        finger_hue_deg=6.0,
        finger_roughness=0.70,
        pad_scale=0.88,
        pad_temperature=0.03,
        pad_roughness=0.88,
    ),
    _gripper_variant(
        "mid_glossy_warm",
        finger_scale=1.00,
        finger_hue_deg=-3.0,
        finger_roughness=0.30,
        pad_scale=0.97,
        pad_temperature=0.03,
        pad_roughness=0.55,
    ),
    _gripper_variant(
        "mid_matte_cool",
        finger_scale=1.00,
        finger_hue_deg=3.0,
        finger_roughness=0.70,
        pad_scale=0.97,
        pad_temperature=-0.03,
        pad_roughness=0.88,
    ),
    _gripper_variant(
        "bright_glossy_cool",
        finger_scale=2.00,
        finger_hue_deg=-6.0,
        finger_roughness=0.30,
        pad_scale=1.06,
        pad_temperature=-0.03,
        pad_roughness=0.55,
    ),
    _gripper_variant(
        "bright_matte_warm",
        finger_scale=2.00,
        finger_hue_deg=6.0,
        finger_roughness=0.70,
        pad_scale=1.06,
        pad_temperature=0.03,
        pad_roughness=0.88,
    ),
    _gripper_variant(
        "charcoal_satin_neutral",
        finger_scale=1.50,
        finger_hue_deg=0.0,
        finger_roughness=0.48,
        pad_scale=0.92,
        pad_temperature=0.0,
        pad_roughness=0.72,
    ),
    _gripper_variant(
        "dark_satin_neutral",
        finger_scale=0.65,
        finger_hue_deg=0.0,
        finger_roughness=0.48,
        pad_scale=1.02,
        pad_temperature=0.0,
        pad_roughness=0.72,
    ),
)


def get_gripper_appearance_variant(name: str) -> VisualServoGripperAppearanceVariant:
    for variant in VISUAL_SERVO_GRIPPER_APPEARANCE_VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"Unknown PDZ gripper appearance variant: {name}")


def nearest_gripper_appearance_variant(
    *,
    finger_color: tuple[float, float, float],
    finger_roughness: float,
    pad_color: tuple[float, float, float],
    pad_roughness: float,
) -> VisualServoGripperAppearanceVariant:
    """Quantize a continuous sample to a material pair authored in the USD."""

    def distance(variant: VisualServoGripperAppearanceVariant) -> float:
        finger_color_error = sum(
            ((actual - requested) / 0.07) ** 2
            for actual, requested in zip(variant.finger_color, finger_color, strict=True)
        )
        pad_color_error = sum(
            ((actual - requested) / 0.20) ** 2
            for actual, requested in zip(variant.pad_color, pad_color, strict=True)
        )
        return (
            finger_color_error
            + ((variant.finger_roughness - finger_roughness) / 0.40) ** 2
            + pad_color_error
            + ((variant.pad_roughness - pad_roughness) / 0.33) ** 2
        )

    return min(VISUAL_SERVO_GRIPPER_APPEARANCE_VARIANTS[1:], key=distance)


def sample_weighted_part_palette_index(unit_value: float) -> int:
    """Map a unit sample to the weighted part palette deterministically."""

    value = float(unit_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("unit_value must lie in [0, 1].")
    total = sum(entry.weight for entry in VISUAL_SERVO_PART_PALETTE)
    threshold = min(value, 1.0 - 1.0e-12) * total
    cumulative = 0.0
    for index, entry in enumerate(VISUAL_SERVO_PART_PALETTE):
        cumulative += entry.weight
        if threshold < cumulative:
            return index
    return len(VISUAL_SERVO_PART_PALETTE) - 1


def sample_weighted_part_palette_indices(unit_values: Sequence[float]) -> tuple[int, ...]:
    return tuple(sample_weighted_part_palette_index(value) for value in unit_values)


def classify_robot_finger_geometry_material(
    prim_path: str,
    prim_type_name: str,
) -> str | None:
    """Classify one robot geometry prim for an exact leaf material binding.

    Binding black at a finger-link ancestor with ``strongerThanDescendants``
    also overrides the white TPU-pad binding below it.  Classifying and binding
    the concrete geometry prims avoids that USD material-strength conflict.
    """

    path = str(prim_path)
    if "/Robot/" not in path or str(prim_type_name) not in _VISUAL_GEOMETRY_TYPE_NAMES:
        return None
    components = {component.lower() for component in path.split("/") if component}
    if not components.intersection(_FINGER_LINK_NAMES):
        return None
    lowered_path = path.lower()
    if any(token in lowered_path for token in _CONTACT_PAD_PATH_TOKENS):
        return "white_contact_pad"
    return "black_pla"


def author_pdz_gripper_material_variants(stage) -> dict[str, tuple[str, ...]]:
    """Author selectable finger/pad material pairs into importer visual instances.

    This operates on the generated robot base USD, not on a cloned training
    stage. Each left/right ``visuals`` scope is temporarily de-instanced while
    variant opinions are written, then restored to an instanceable scope.
    """

    from pxr import Gf, Sdf, Usd, UsdShade

    def set_material_inputs(
        material: UsdShade.Material,
        *,
        color: tuple[float, float, float],
        roughness: float,
    ) -> None:
        source = None
        for render_context in ("mdl", "", "universal"):
            candidate = material.ComputeSurfaceSource(render_context)
            if candidate and candidate[0]:
                source = candidate
                break
        if source is None:
            raise RuntimeError(f"Material {material.GetPath()} has no supported surface shader.")
        shader = UsdShade.Shader(source[0])
        color_written = False
        for name in ("diffuseColor", "diffuse_color_constant", "diffuse_color"):
            shader_input = shader.GetInput(name)
            if shader_input:
                shader_input.Set(Gf.Vec3f(*color))
                color_written = True
        if not color_written:
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*color)
            )
        roughness_written = False
        for name in ("roughness", "reflection_roughness_constant"):
            shader_input = shader.GetInput(name)
            if shader_input:
                shader_input.Set(float(roughness))
                roughness_written = True
        if not roughness_written:
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
                float(roughness)
            )

    visual_root_paths = tuple(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsInstance()
        and prim.GetName() == "visuals"
        and any(
            token in str(prim.GetPath()).lower()
            for token in ("pdz_gripper_left_finger_link", "pdz_gripper_right_finger_link")
        )
    )
    if len(visual_root_paths) != 2:
        raise RuntimeError(
            "Expected exactly two instanceable PDZ finger visual scopes in the base USD, "
            f"found {visual_root_paths}."
        )
    for path in visual_root_paths:
        stage.GetPrimAtPath(path).SetInstanceable(False)

    observed: dict[str, set[str]] = {"finger": set(), "pad": set()}
    try:
        for root_path in visual_root_paths:
            root = stage.GetPrimAtPath(root_path)
            material_paths: dict[str, set[str]] = {"finger": set(), "pad": set()}
            for prim in Usd.PrimRange(root):
                if prim.GetTypeName() != "Mesh":
                    continue
                path = str(prim.GetPath())
                material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if not material:
                    raise RuntimeError(f"PDZ visual geometry has no bound material: {path}")
                kind = "pad" if "tpu_pad" in path.lower() or "pad_8mm" in path.lower() else "finger"
                material_paths[kind].add(str(material.GetPath()))
                observed[kind].add(str(material.GetPath()))
            if not material_paths["finger"] or not material_paths["pad"]:
                raise RuntimeError(
                    f"PDZ visual scope {root_path} does not contain both finger and pad materials: "
                    f"{material_paths}"
                )

            variant_set = root.GetVariantSets().AddVariantSet(PDZ_GRIPPER_APPEARANCE_VARIANT_SET)
            for variant in VISUAL_SERVO_GRIPPER_APPEARANCE_VARIANTS:
                variant_set.AddVariant(variant.name)
                variant_set.SetVariantSelection(variant.name)
                with variant_set.GetVariantEditContext():
                    for material_path in material_paths["finger"]:
                        set_material_inputs(
                            UsdShade.Material(stage.GetPrimAtPath(material_path)),
                            color=variant.finger_color,
                            roughness=variant.finger_roughness,
                        )
                    for material_path in material_paths["pad"]:
                        set_material_inputs(
                            UsdShade.Material(stage.GetPrimAtPath(material_path)),
                            color=variant.pad_color,
                            roughness=variant.pad_roughness,
                        )
            variant_set.SetVariantSelection("canonical")
    finally:
        for path in visual_root_paths:
            stage.GetPrimAtPath(path).SetInstanceable(True)

    return {
        "visual_roots": visual_root_paths,
        "finger_materials": tuple(sorted(observed["finger"])),
        "pad_materials": tuple(sorted(observed["pad"])),
        "variants": tuple(variant.name for variant in VISUAL_SERVO_GRIPPER_APPEARANCE_VARIANTS),
    }


def apply_visual_servo_materials() -> dict[str, Any]:
    """Bind the same high-contrast materials in execution, RL, and goal capture.

    Isaac imports deliberately stay inside the function.  This module is also
    importable by non-Isaac tooling that only needs the profile identifier.
    """

    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    material_specs = {
        "brown_pla": sim_utils.PreviewSurfaceCfg(
            diffuse_color=VISUAL_SERVO_PART_COLOR,
            roughness=VISUAL_SERVO_PART_ROUGHNESS,
            metallic=0.0,
        ),
        "black_pla": sim_utils.PreviewSurfaceCfg(
            # The legacy key is retained because the physical appearance
            # randomizer and older tooling consume it.
            diffuse_color=VISUAL_SERVO_FINGER_COLOR,
            roughness=VISUAL_SERVO_FINGER_ROUGHNESS,
            metallic=0.0,
        ),
        "white_contact_pad": sim_utils.PreviewSurfaceCfg(
            diffuse_color=VISUAL_SERVO_CONTACT_PAD_COLOR,
            roughness=VISUAL_SERVO_CONTACT_PAD_ROUGHNESS,
            metallic=0.0,
        ),
        "work_surface": sim_utils.PreviewSurfaceCfg(
            diffuse_color=VISUAL_SERVO_WORK_SURFACE_COLOR,
            roughness=VISUAL_SERVO_WORK_SURFACE_ROUGHNESS,
            metallic=0.0,
        ),
    }
    material_paths: dict[str, str] = {}
    for name, cfg in material_specs.items():
        path = f"/World/Looks/{name}"
        cfg.func(path, cfg)
        material_paths[name] = path

    part_paths: list[str] = []
    part_paths_by_env: dict[int, list[str]] = {}
    finger_paths: list[str] = []
    finger_geometry_paths: list[str] = []
    contact_pad_geometry_paths: list[str] = []
    editable_finger_geometry_paths: list[str] = []
    editable_contact_pad_geometry_paths: list[str] = []
    gripper_appearance_variant_roots: list[str] = []
    gripper_appearance_variant_roots_by_env: dict[int, list[str]] = {}
    # URDF-imported ``visuals`` scopes remain internally instanced even when
    # the robot asset itself is not instanceable. Include their instance
    # proxies for classification/validation, while only authoring bindings on
    # editable concrete prims. The regenerated PDZ USD carries the same
    # black/white contract inside those read-only prototypes.
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        name = prim.GetName()
        if (
            "/Robot/" in path
            and name == "visuals"
            and prim.IsInstance()
            and PDZ_GRIPPER_APPEARANCE_VARIANT_SET in prim.GetVariantSets().GetNames()
            and any(token in path.lower() for token in _FINGER_LINK_NAMES)
        ):
            gripper_appearance_variant_roots.append(path)
            try:
                env_index = int(path.split("/envs/env_", 1)[1].split("/", 1)[0])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Cannot parse gripper environment index: {path}") from exc
            gripper_appearance_variant_roots_by_env.setdefault(env_index, []).append(path)
        if "/envs/env_" in path and (name == "Part" or name.startswith("Part_")):
            part_paths.append(path)
            try:
                env_index = int(path.split("/envs/env_", 1)[1].split("/", 1)[0])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Cannot parse environment index from part path: {path}") from exc
            part_paths_by_env.setdefault(env_index, []).append(path)
        elif "/Robot/" in path and name in _FINGER_LINK_NAMES:
            finger_paths.append(path)
        material_name = classify_robot_finger_geometry_material(path, prim.GetTypeName())
        if material_name == "white_contact_pad":
            contact_pad_geometry_paths.append(path)
            if not prim.IsInstanceProxy():
                editable_contact_pad_geometry_paths.append(path)
        elif material_name == "black_pla":
            finger_geometry_paths.append(path)
            if not prim.IsInstanceProxy():
                editable_finger_geometry_paths.append(path)

    if not part_paths:
        raise RuntimeError("Expected at least one RL/execution target-part prim for material binding.")
    if not finger_paths:
        raise RuntimeError("Expected loaded left/right gripper finger prims for material binding.")
    if not finger_geometry_paths:
        raise RuntimeError("Expected concrete left/right finger geometry prims for material binding.")
    if not contact_pad_geometry_paths:
        raise RuntimeError("Expected concrete left/right TPU contact-pad geometry prims for material binding.")

    # Bind once, then update only the per-environment shader values at reset.
    # Rebinding up to five rigid objects in every completed environment would
    # otherwise become a material USD bottleneck at 256 parallel environments.
    part_materials_by_env: dict[int, str] = {}
    part_shaders_by_env: dict[int, str] = {}
    for env_index, env_part_paths in sorted(part_paths_by_env.items()):
        material_path = f"/World/Looks/part_live_env_{env_index}"
        part_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=VISUAL_SERVO_PART_COLOR,
            roughness=VISUAL_SERVO_PART_ROUGHNESS,
            metallic=0.0,
        )
        part_cfg.func(material_path, part_cfg)
        material_paths[f"part_live_env_{env_index}"] = material_path
        part_materials_by_env[env_index] = material_path
        part_shaders_by_env[env_index] = f"{material_path}/Shader"
        for part_path in env_part_paths:
            sim_utils.bind_visual_material(
                part_path,
                material_path,
                stage=stage,
                stronger_than_descendants=True,
            )
    for finger_path in editable_finger_geometry_paths:
        sim_utils.bind_visual_material(
            finger_path,
            material_paths["black_pla"],
            stage=stage,
            stronger_than_descendants=True,
        )
    for contact_pad_path in editable_contact_pad_geometry_paths:
        sim_utils.bind_visual_material(
            contact_pad_path,
            material_paths["white_contact_pad"],
            stage=stage,
            stronger_than_descendants=True,
        )
    ground_path = "/World/GroundPlane"
    if stage.GetPrimAtPath(ground_path).IsValid():
        sim_utils.bind_visual_material(
            ground_path,
            material_paths["work_surface"],
            stage=stage,
            stronger_than_descendants=True,
        )

    return {
        "profile": VISUAL_SERVO_MATERIAL_PROFILE,
        "parts": tuple(part_paths),
        "parts_by_env": {
            env_index: tuple(paths) for env_index, paths in sorted(part_paths_by_env.items())
        },
        "fingers": tuple(finger_paths),
        "finger_geometry": tuple(finger_geometry_paths),
        "contact_pads": tuple(contact_pad_geometry_paths),
        "editable_finger_geometry": tuple(editable_finger_geometry_paths),
        "editable_contact_pads": tuple(editable_contact_pad_geometry_paths),
        "gripper_appearance_variant_roots": tuple(gripper_appearance_variant_roots),
        "gripper_appearance_variant_roots_by_env": {
            env_index: tuple(paths)
            for env_index, paths in sorted(gripper_appearance_variant_roots_by_env.items())
        },
        "gripper_appearance_variants": tuple(
            variant.name for variant in VISUAL_SERVO_GRIPPER_APPEARANCE_VARIANTS
        ),
        "robot_material_source": (
            "authored_pdz_usd_material_variants"
            if gripper_appearance_variant_roots
            else (
                "runtime_leaf_bindings"
                if editable_contact_pad_geometry_paths
                else "authored_pdz_usd_fixed_instance_materials"
            )
        ),
        "materials": material_paths,
        "part_materials_by_env": part_materials_by_env,
        "part_shaders_by_env": part_shaders_by_env,
    }


__all__ = [
    "PDZ_GRIPPER_APPEARANCE_VARIANT_SET",
    "VISUAL_SERVO_FINGER_COLOR",
    "VISUAL_SERVO_FINGER_ROUGHNESS",
    "VISUAL_SERVO_CONTACT_PAD_COLOR",
    "VISUAL_SERVO_CONTACT_PAD_ROUGHNESS",
    "VISUAL_SERVO_CANONICAL_PART_INDEX",
    "VISUAL_SERVO_MATERIAL_PROFILE",
    "VISUAL_SERVO_PART_PALETTE",
    "VISUAL_SERVO_PART_COLOR",
    "VISUAL_SERVO_PART_ROUGHNESS",
    "VISUAL_SERVO_WORK_SURFACE_COLOR",
    "VISUAL_SERVO_WORK_SURFACE_ROUGHNESS",
    "VISUAL_SERVO_GRIPPER_APPEARANCE_VARIANTS",
    "VisualServoGripperAppearanceVariant",
    "VisualServoPartMaterial",
    "apply_visual_servo_materials",
    "author_pdz_gripper_material_variants",
    "classify_robot_finger_geometry_material",
    "get_gripper_appearance_variant",
    "nearest_gripper_appearance_variant",
    "sample_weighted_part_palette_index",
    "sample_weighted_part_palette_indices",
]
