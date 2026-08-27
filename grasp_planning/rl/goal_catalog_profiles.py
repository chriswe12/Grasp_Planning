"""Shared provenance and material contract for synthetic goal images."""

from __future__ import annotations

from dataclasses import dataclass

from grasp_planning.isaac_visual_materials import (
    VISUAL_SERVO_CONTACT_PAD_COLOR,
    VISUAL_SERVO_CONTACT_PAD_ROUGHNESS,
    VISUAL_SERVO_FINGER_COLOR,
    VISUAL_SERVO_FINGER_ROUGHNESS,
    VISUAL_SERVO_PART_COLOR,
    VISUAL_SERVO_PART_ROUGHNESS,
)
from grasp_planning.visual_servo_workspace import (
    VISUAL_SERVO_TSLOT_ALUMINUM_COLOR,
    VISUAL_SERVO_TSLOT_ALUMINUM_METALLIC,
    VISUAL_SERVO_TSLOT_ALUMINUM_ROUGHNESS,
    VISUAL_SERVO_TSLOT_SLOT_COLOR,
    VISUAL_SERVO_TSLOT_SLOT_METALLIC,
    VISUAL_SERVO_TSLOT_SLOT_ROUGHNESS,
)

MUJOCO_GOAL_RENDERER_BACKEND = "filament"
MUJOCO_GOAL_RENDERER_PROFILE = (
    "mujoco_filament_pdz_d405_ibl_matte_visual_mesh_tslot_parity_v4"
)

# Filament and Isaac/Omniverse do not produce the same perceptual roughness for
# identical USD Preview Surface numbers.  These are deliberately backend
# calibrated: the base colors remain canonical while the higher roughness and
# reduced T-slot contrast make the MuJoCo goal images match the matte Isaac
# reference instead of looking wet or mirror-polished.
MUJOCO_FILAMENT_PART_ROUGHNESS = 0.96
MUJOCO_FILAMENT_FINGER_ROUGHNESS = 0.82
MUJOCO_FILAMENT_PAD_ROUGHNESS = 0.90
MUJOCO_FILAMENT_TSLOT_ALUMINUM_METALLIC = 0.72
MUJOCO_FILAMENT_TSLOT_ALUMINUM_ROUGHNESS = 0.58
MUJOCO_FILAMENT_TSLOT_SLOT_COLOR = (0.12, 0.135, 0.15)
MUJOCO_FILAMENT_TSLOT_SLOT_METALLIC = 0.25
MUJOCO_FILAMENT_TSLOT_SLOT_ROUGHNESS = 0.72


@dataclass(frozen=True)
class GoalFilamentMaterial:
    color: tuple[float, float, float]
    metallic: float
    roughness: float
    emission: float = 0.0


GOAL_FILAMENT_MATERIALS: dict[str, GoalFilamentMaterial] = {
    "part_canonical": GoalFilamentMaterial(
        VISUAL_SERVO_PART_COLOR,
        0.0,
        MUJOCO_FILAMENT_PART_ROUGHNESS,
    ),
    "pdz_finger_black": GoalFilamentMaterial(
        VISUAL_SERVO_FINGER_COLOR,
        0.0,
        MUJOCO_FILAMENT_FINGER_ROUGHNESS,
    ),
    "pdz_contact_white": GoalFilamentMaterial(
        VISUAL_SERVO_CONTACT_PAD_COLOR,
        0.0,
        MUJOCO_FILAMENT_PAD_ROUGHNESS,
    ),
    "tslot_aluminum": GoalFilamentMaterial(
        VISUAL_SERVO_TSLOT_ALUMINUM_COLOR,
        MUJOCO_FILAMENT_TSLOT_ALUMINUM_METALLIC,
        MUJOCO_FILAMENT_TSLOT_ALUMINUM_ROUGHNESS,
    ),
    "tslot_slot": GoalFilamentMaterial(
        MUJOCO_FILAMENT_TSLOT_SLOT_COLOR,
        MUJOCO_FILAMENT_TSLOT_SLOT_METALLIC,
        MUJOCO_FILAMENT_TSLOT_SLOT_ROUGHNESS,
    ),
}

__all__ = [
    "GOAL_FILAMENT_MATERIALS",
    "GoalFilamentMaterial",
    "MUJOCO_GOAL_RENDERER_BACKEND",
    "MUJOCO_GOAL_RENDERER_PROFILE",
    "MUJOCO_FILAMENT_FINGER_ROUGHNESS",
    "MUJOCO_FILAMENT_PAD_ROUGHNESS",
    "MUJOCO_FILAMENT_PART_ROUGHNESS",
    "MUJOCO_FILAMENT_TSLOT_ALUMINUM_METALLIC",
    "MUJOCO_FILAMENT_TSLOT_ALUMINUM_ROUGHNESS",
    "MUJOCO_FILAMENT_TSLOT_SLOT_COLOR",
    "MUJOCO_FILAMENT_TSLOT_SLOT_METALLIC",
    "MUJOCO_FILAMENT_TSLOT_SLOT_ROUGHNESS",
]
