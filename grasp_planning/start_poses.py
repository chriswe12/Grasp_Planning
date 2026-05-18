"""Shared robot start poses used by planning and simulator execution."""

from __future__ import annotations

DEFAULT_ARM_START_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
}
DEFAULT_HAND_START_JOINT_POS = {"panda_finger_joint.*": 0.04}
DEFAULT_HAND_OPEN_WIDTH = 0.04
DEFAULT_MOVEIT_ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
DEFAULT_ARM_START_JOINT_VALUES = tuple(DEFAULT_ARM_START_JOINT_POS[f"panda_joint{index}"] for index in range(1, 8))


__all__ = [
    "DEFAULT_ARM_START_JOINT_POS",
    "DEFAULT_ARM_START_JOINT_VALUES",
    "DEFAULT_HAND_OPEN_WIDTH",
    "DEFAULT_HAND_START_JOINT_POS",
    "DEFAULT_MOVEIT_ARM_JOINT_NAMES",
]
