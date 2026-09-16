"""Shared Isaac visual materials for wrist-camera grasp alignment."""

from __future__ import annotations

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


# The canonical brown remains deliberately dominant. The other colors prevent
# the policy from treating hue as an object or completion cue, while the very
# dark/light extremes are kept rare because they are harder for RGB-D sensing.
VISUAL_SERVO_PART_PALETTE: tuple[VisualServoPartMaterial, ...] = (
    VisualServoPartMaterial("soft_brown", _rgb8(95, 72, 58), 0.40),
    VisualServoPartMaterial("soft_clay", _rgb8(100, 64, 62), 0.035),
    VisualServoPartMaterial("soft_sage", _rgb8(64, 88, 68), 0.035),
    VisualServoPartMaterial("soft_slate", _rgb8(65, 74, 94), 0.035),
    VisualServoPartMaterial("soft_bluegray", _rgb8(60, 82, 84), 0.035),
    VisualServoPartMaterial("soft_tan", _rgb8(105, 86, 55), 0.035),
    VisualServoPartMaterial("soft_burgundy", _rgb8(78, 48, 58), 0.025),
    VisualServoPartMaterial("soft_rust", _rgb8(98, 64, 46), 0.025),
    VisualServoPartMaterial("soft_olive", _rgb8(78, 82, 55), 0.025),
    VisualServoPartMaterial("soft_moss", _rgb8(54, 76, 58), 0.025),
    VisualServoPartMaterial("soft_denim", _rgb8(52, 66, 88), 0.025),
    VisualServoPartMaterial("soft_mauve", _rgb8(82, 64, 78), 0.020),
    VisualServoPartMaterial("soft_charcoal", _rgb8(52, 56, 58), 0.020),
    VisualServoPartMaterial("soft_cream", _rgb8(100, 94, 80), 0.020),
    VisualServoPartMaterial("soft_darkblue", _rgb8(40, 52, 72), 0.020),
    VisualServoPartMaterial("soft_dustyrose", _rgb8(92, 67, 70), 0.020),
    VisualServoPartMaterial("soft_yellow", _rgb8(108, 98, 55), 0.020),
    VisualServoPartMaterial("soft_purple", _rgb8(78, 65, 88), 0.020),
    VisualServoPartMaterial("soft_red", _rgb8(104, 61, 59), 0.020),
    VisualServoPartMaterial("soft_blue", _rgb8(58, 72, 96), 0.020),
    VisualServoPartMaterial("soft_green", _rgb8(58, 86, 64), 0.020),
    VisualServoPartMaterial("soft_orange", _rgb8(110, 78, 52), 0.020),
    VisualServoPartMaterial("soft_black", _rgb8(45, 47, 49), 0.010),
    VisualServoPartMaterial("soft_white", _rgb8(108, 104, 96), 0.010),
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
    # URDF-imported ``visuals`` scopes remain internally instanced even when
    # the robot asset itself is not instanceable. Include their instance
    # proxies for classification/validation, while only authoring bindings on
    # editable concrete prims. The regenerated PDZ USD carries the same
    # black/white contract inside those read-only prototypes.
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        name = prim.GetName()
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
        "robot_material_source": (
            "runtime_leaf_bindings"
            if editable_contact_pad_geometry_paths
            else "authored_pdz_usd_instance_materials"
        ),
        "materials": material_paths,
        "part_materials_by_env": part_materials_by_env,
        "part_shaders_by_env": part_shaders_by_env,
    }


__all__ = [
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
    "VisualServoPartMaterial",
    "apply_visual_servo_materials",
    "classify_robot_finger_geometry_material",
    "sample_weighted_part_palette_index",
    "sample_weighted_part_palette_indices",
]
