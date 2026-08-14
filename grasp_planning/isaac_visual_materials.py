"""Shared Isaac visual materials for wrist-camera grasp alignment."""

from __future__ import annotations

from typing import Any

VISUAL_SERVO_MATERIAL_PROFILE = "matte_brown_part_readable_black_fingers_v2"
VISUAL_SERVO_PART_COLOR = (0.24, 0.075, 0.025)
VISUAL_SERVO_PART_ROUGHNESS = 0.80
VISUAL_SERVO_FINGER_COLOR = (0.035, 0.042, 0.050)
VISUAL_SERVO_FINGER_ROUGHNESS = 0.80
VISUAL_SERVO_WORK_SURFACE_COLOR = (0.075, 0.085, 0.10)
VISUAL_SERVO_WORK_SURFACE_ROUGHNESS = 0.90


def apply_visual_servo_materials() -> dict[str, Any]:
    """Bind the same high-contrast materials in execution, RL, and goal capture.

    Isaac imports deliberately stay inside the function.  This module is also
    importable by non-Isaac tooling that only needs the profile identifier.
    """

    import isaaclab.sim as sim_utils
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    material_specs = {
        "brown_pla": sim_utils.PreviewSurfaceCfg(
            diffuse_color=VISUAL_SERVO_PART_COLOR,
            roughness=VISUAL_SERVO_PART_ROUGHNESS,
            metallic=0.0,
        ),
        "black_pla": sim_utils.PreviewSurfaceCfg(
            # Dark enough to resemble black printed fingers, but not so close
            # to zero that their silhouette and bevels collapse in RGB.
            diffuse_color=VISUAL_SERVO_FINGER_COLOR,
            roughness=VISUAL_SERVO_FINGER_ROUGHNESS,
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
    finger_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName()
        if "/envs/env_" in path and (name == "Part" or name.startswith("Part_")):
            part_paths.append(path)
        elif "/Robot/" in path and name in {"left_finger_link", "right_finger_link"}:
            finger_paths.append(path)

    if not part_paths:
        raise RuntimeError("Expected at least one RL/execution target-part prim for material binding.")
    if not finger_paths:
        raise RuntimeError("Expected loaded left/right gripper finger prims for material binding.")

    for part_path in part_paths:
        sim_utils.bind_visual_material(
            part_path,
            material_paths["brown_pla"],
            stage=stage,
            stronger_than_descendants=True,
        )
    for finger_path in finger_paths:
        sim_utils.bind_visual_material(
            finger_path,
            material_paths["black_pla"],
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
        "fingers": tuple(finger_paths),
        "materials": material_paths,
    }


__all__ = [
    "VISUAL_SERVO_FINGER_COLOR",
    "VISUAL_SERVO_FINGER_ROUGHNESS",
    "VISUAL_SERVO_MATERIAL_PROFILE",
    "VISUAL_SERVO_PART_COLOR",
    "VISUAL_SERVO_PART_ROUGHNESS",
    "VISUAL_SERVO_WORK_SURFACE_COLOR",
    "VISUAL_SERVO_WORK_SURFACE_ROUGHNESS",
    "apply_visual_servo_materials",
]
