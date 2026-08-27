"""MuJoCo Filament provenance and materials for runtime D405 goals."""

from __future__ import annotations

from dataclasses import dataclass

from grasp_planning.isaac_visual_materials import (
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
    "mujoco_filament_y_gripper_d405_shared_materials_fill6_v1"
)


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
        VISUAL_SERVO_PART_ROUGHNESS,
    ),
    "finger_canonical": GoalFilamentMaterial(
        VISUAL_SERVO_FINGER_COLOR,
        0.0,
        VISUAL_SERVO_FINGER_ROUGHNESS,
    ),
    "tslot_aluminum": GoalFilamentMaterial(
        VISUAL_SERVO_TSLOT_ALUMINUM_COLOR,
        VISUAL_SERVO_TSLOT_ALUMINUM_METALLIC,
        VISUAL_SERVO_TSLOT_ALUMINUM_ROUGHNESS,
    ),
    "tslot_slot": GoalFilamentMaterial(
        VISUAL_SERVO_TSLOT_SLOT_COLOR,
        VISUAL_SERVO_TSLOT_SLOT_METALLIC,
        VISUAL_SERVO_TSLOT_SLOT_ROUGHNESS,
    ),
}


__all__ = [
    "GOAL_FILAMENT_MATERIALS",
    "GoalFilamentMaterial",
    "MUJOCO_GOAL_RENDERER_BACKEND",
    "MUJOCO_GOAL_RENDERER_PROFILE",
]
