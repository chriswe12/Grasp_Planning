"""Shared robot start poses used by planning and simulator execution."""

from __future__ import annotations

import math

DEFAULT_ARM_START_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
}
# MoveIt uses the physical lbr-stack joint coordinates. Isaac USD expresses a
# negative URDF axis as a positive USD axis with an inverted coordinate, so
# only A4 requires a sign change at the backend boundary.
KUKA_MOVEIT_TO_ISAAC_JOINT_SIGNS = (1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0)
KUKA_MOVEIT_ARM_START_JOINT_VALUES = (
    0.0,
    0.5,
    0.0,
    -1.3962634015954636,
    0.0,
    1.1,
    0.0,
)
DEFAULT_KUKA_ARM_START_JOINT_POS = {
    f"joint{index}": float(sign * value)
    for index, (sign, value) in enumerate(
        zip(KUKA_MOVEIT_TO_ISAAC_JOINT_SIGNS, KUKA_MOVEIT_ARM_START_JOINT_VALUES, strict=True),
        start=1,
    )
}
DEFAULT_HAND_OPEN_WIDTH = 0.04
KUKA_Y_GRIPPER_TRAVEL_M = 0.04
KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M = 0.084
KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M = 0.005
KUKA_Y_GRIPPER_APPROACH_CLEARANCE_TOTAL_M = (
    2.0 * KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M
)
KUKA_Y_GRIPPER_APPROACH_PROFILE = "jaw_width_plus_10mm_v1"
DEFAULT_HAND_START_JOINT_POS = {
    "panda_finger_joint.*": 0.04,
    "left_finger_joint": 0.0,
    "right_finger_joint": 0.0,
}
DEFAULT_MOVEIT_ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
DEFAULT_ARM_START_JOINT_VALUES = tuple(DEFAULT_ARM_START_JOINT_POS[f"panda_joint{index}"] for index in range(1, 8))


def gripper_joint_target_from_width(joint_name: str, width_m: float) -> float:
    """Map a requested opening width to the loaded robot's finger joint target."""

    if str(joint_name) == "left_finger_joint":
        close_distance = 0.5 * (KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - float(width_m))
        return max(0.0, min(KUKA_Y_GRIPPER_TRAVEL_M, close_distance))
    if str(joint_name) == "right_finger_joint":
        close_distance = 0.5 * (KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - float(width_m))
        return -max(0.0, min(KUKA_Y_GRIPPER_TRAVEL_M, close_distance))
    return float(width_m)


def kuka_y_gripper_approach_width_from_jaw_width(jaw_width_m: float) -> float:
    """Return the collision-free approach aperture for one selected grasp.

    The approach keeps 5 mm of clearance between each fingertip and its final
    contact location.  Reject unachievable grasps instead of silently clamping
    them to the physical opening limit, because clamping would violate the
    visual-training contract encoded by :data:`KUKA_Y_GRIPPER_APPROACH_PROFILE`.
    """

    jaw_width = float(jaw_width_m)
    if not math.isfinite(jaw_width) or jaw_width < 0.0:
        raise ValueError(f"KUKA Y-gripper jaw width must be finite and non-negative, got {jaw_width_m}.")
    approach_width = jaw_width + KUKA_Y_GRIPPER_APPROACH_CLEARANCE_TOTAL_M
    if approach_width > KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M + 1.0e-9:
        raise ValueError(
            "KUKA Y-gripper approach aperture exceeds the physical opening: "
            f"jaw={jaw_width:.6f} m, approach={approach_width:.6f} m, "
            f"maximum={KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M:.6f} m."
        )
    return approach_width


def gripper_max_open_width(joint_name: str) -> float:
    """Return the width command that fully opens the named gripper joint family."""

    if str(joint_name) in {"left_finger_joint", "right_finger_joint"}:
        return KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M
    return DEFAULT_HAND_OPEN_WIDTH


def is_gripper_joint_name(joint_name: str) -> bool:
    """Return whether a joint belongs to a supported gripper."""

    name = str(joint_name)
    return name.startswith(("panda_finger_joint", "fr3_finger_joint")) or name in {
        "left_finger_joint",
        "right_finger_joint",
    }


def is_gripper_command_joint_name(joint_name: str) -> bool:
    """Return whether a gripper joint should receive direct position commands."""

    name = str(joint_name)
    if name == "right_finger_joint":
        return False
    return is_gripper_joint_name(name)


def kuka_moveit_to_isaac_joint_positions(values: tuple[float, ...]) -> tuple[float, ...]:
    """Convert LBR MoveIt joint coordinates to this branch's generated USD coordinates."""

    if len(values) != len(KUKA_MOVEIT_TO_ISAAC_JOINT_SIGNS):
        raise ValueError(f"Expected 7 KUKA joint values, got {len(values)}.")
    return tuple(
        float(sign) * float(value) for sign, value in zip(KUKA_MOVEIT_TO_ISAAC_JOINT_SIGNS, values, strict=True)
    )


def kuka_isaac_to_moveit_joint_positions(values: tuple[float, ...]) -> tuple[float, ...]:
    """Convert generated USD joint coordinates back to LBR MoveIt coordinates."""

    return kuka_moveit_to_isaac_joint_positions(values)


__all__ = [
    "DEFAULT_ARM_START_JOINT_POS",
    "DEFAULT_ARM_START_JOINT_VALUES",
    "DEFAULT_HAND_OPEN_WIDTH",
    "DEFAULT_HAND_START_JOINT_POS",
    "DEFAULT_KUKA_ARM_START_JOINT_POS",
    "KUKA_MOVEIT_ARM_START_JOINT_VALUES",
    "KUKA_MOVEIT_TO_ISAAC_JOINT_SIGNS",
    "DEFAULT_MOVEIT_ARM_JOINT_NAMES",
    "KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M",
    "KUKA_Y_GRIPPER_APPROACH_CLEARANCE_TOTAL_M",
    "KUKA_Y_GRIPPER_APPROACH_PROFILE",
    "KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M",
    "KUKA_Y_GRIPPER_TRAVEL_M",
    "gripper_joint_target_from_width",
    "gripper_max_open_width",
    "is_gripper_command_joint_name",
    "is_gripper_joint_name",
    "kuka_y_gripper_approach_width_from_jaw_width",
    "kuka_isaac_to_moveit_joint_positions",
    "kuka_moveit_to_isaac_joint_positions",
]
