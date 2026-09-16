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
KUKA_Y_GRIPPER_SOURCE_CLOSED_WIDTH_M = (
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - 2.0 * KUKA_Y_GRIPPER_TRAVEL_M
)
KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M = 0.005
KUKA_Y_GRIPPER_APPROACH_CLEARANCE_TOTAL_M = (
    2.0 * KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M
)
KUKA_Y_GRIPPER_APPROACH_PROFILE = "jaw_width_plus_10mm_v1"
PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M = 0.005
PDZ_GRIPPER_APPROACH_CLEARANCE_TOTAL_M = 2.0 * PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M
PDZ_GRIPPER_APPROACH_PROFILE = "pdz_jaw_width_plus_10mm_v1"
PDZ_GRIPPER_CLOSED_WIDTH_M = 0.012
PDZ_GRIPPER_TRAVEL_M = 0.032
PDZ_GRIPPER_OPEN_WIDTH_M = PDZ_GRIPPER_CLOSED_WIDTH_M + 2.0 * PDZ_GRIPPER_TRAVEL_M
VISUAL_SERVO_GRIPPER_PROFILE = "kuka_iiwa7_pdz_gripper_slim8_v1"
DEFAULT_HAND_START_JOINT_POS = {
    "panda_finger_joint.*": 0.04,
    "left_finger_joint": 0.0,
    "right_finger_joint": 0.0,
    "pdz_gripper_left_finger_joint": PDZ_GRIPPER_TRAVEL_M,
    "pdz_gripper_right_finger_joint": PDZ_GRIPPER_TRAVEL_M,
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
    if str(joint_name) in {
        "pdz_gripper_left_finger_joint",
        "pdz_gripper_right_finger_joint",
    }:
        open_distance = 0.5 * (float(width_m) - PDZ_GRIPPER_CLOSED_WIDTH_M)
        return max(0.0, min(PDZ_GRIPPER_TRAVEL_M, open_distance))
    return float(width_m)


def kuka_y_gripper_approach_width_from_jaw_width(jaw_width_m: float) -> float:
    """Return the trained Y-gripper approach aperture without silent clamping."""

    jaw_width = float(jaw_width_m)
    if not math.isfinite(jaw_width) or jaw_width < 0.0:
        raise ValueError(
            f"KUKA Y-gripper jaw width must be finite and non-negative, got {jaw_width_m}."
        )
    approach_width = jaw_width + KUKA_Y_GRIPPER_APPROACH_CLEARANCE_TOTAL_M
    if approach_width > KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M + 1.0e-9:
        raise ValueError(
            "KUKA Y-gripper approach aperture exceeds the physical opening: "
            f"jaw={jaw_width:.6f} m, approach={approach_width:.6f} m, "
            f"maximum={KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M:.6f} m."
        )
    return approach_width


def pdz_gripper_approach_width_from_jaw_width(jaw_width_m: float) -> float:
    """Return the trained PDZ approach aperture without silent clamping."""

    jaw_width = float(jaw_width_m)
    if not math.isfinite(jaw_width) or jaw_width < PDZ_GRIPPER_CLOSED_WIDTH_M:
        raise ValueError(
            f"PDZ jaw width must be finite and at least {PDZ_GRIPPER_CLOSED_WIDTH_M:.3f} m, "
            f"got {jaw_width_m}."
        )
    approach_width = jaw_width + PDZ_GRIPPER_APPROACH_CLEARANCE_TOTAL_M
    if approach_width > PDZ_GRIPPER_OPEN_WIDTH_M + 1.0e-9:
        raise ValueError(
            "PDZ approach aperture exceeds the physical opening: "
            f"jaw={jaw_width:.6f} m, approach={approach_width:.6f} m, "
            f"maximum={PDZ_GRIPPER_OPEN_WIDTH_M:.6f} m."
        )
    return approach_width


def kuka_gripper_clamp_width(width_m: float) -> float:
    """Clamp a physical jaw width to the modeled KUKA Y-gripper range."""

    return max(
        KUKA_Y_GRIPPER_SOURCE_CLOSED_WIDTH_M,
        min(KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M, float(width_m)),
    )


def gripper_clamp_width(width_m: float, *, gripper_model: str = "kuka_y_gripper") -> float:
    """Clamp a jaw width to the selected KUKA end-effector's modeled range."""

    if str(gripper_model) == "pdz_gripper":
        return max(PDZ_GRIPPER_CLOSED_WIDTH_M, min(PDZ_GRIPPER_OPEN_WIDTH_M, float(width_m)))
    return kuka_gripper_clamp_width(width_m)


def kuka_gripper_approach_width(
    jaw_width_m: float,
    *,
    clearance_per_finger_m: float = KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
) -> float:
    """Return the partially closed opening used before contact."""

    if float(clearance_per_finger_m) < 0.0:
        raise ValueError("clearance_per_finger_m must be non-negative.")
    return kuka_gripper_clamp_width(float(jaw_width_m) + 2.0 * float(clearance_per_finger_m))


def gripper_approach_width(
    jaw_width_m: float,
    *,
    gripper_model: str = "kuka_y_gripper",
    clearance_per_finger_m: float | None = None,
) -> float:
    """Return a collision-model-aware partially open approach width."""

    clearance = (
        PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M
        if clearance_per_finger_m is None and str(gripper_model) == "pdz_gripper"
        else KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M
        if clearance_per_finger_m is None
        else float(clearance_per_finger_m)
    )
    if clearance < 0.0:
        raise ValueError("clearance_per_finger_m must be non-negative.")
    return gripper_clamp_width(
        float(jaw_width_m) + 2.0 * clearance,
        gripper_model=gripper_model,
    )


def kuka_gripper_normalized_position_from_width(width_m: float) -> float:
    """Map physical jaw width to controller position: 0=open, 1=closed."""

    width = kuka_gripper_clamp_width(width_m)
    span = KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - KUKA_Y_GRIPPER_SOURCE_CLOSED_WIDTH_M
    return max(0.0, min(1.0, (KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - width) / span))


def kuka_gripper_width_from_normalized_position(position: float) -> float:
    """Map normalized controller position to physical jaw width."""

    normalized = max(0.0, min(1.0, float(position)))
    span = KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - KUKA_Y_GRIPPER_SOURCE_CLOSED_WIDTH_M
    return KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - normalized * span


def kuka_moveit_gripper_driver_position_from_width(width_m: float) -> float:
    """Return the MoveIt prismatic driver-joint position for a jaw width."""

    return max(
        0.0,
        min(
            KUKA_Y_GRIPPER_TRAVEL_M,
            0.5 * (KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - kuka_gripper_clamp_width(width_m)),
        ),
    )


def kuka_moveit_gripper_state(
    robot_name: str,
    width_m: float,
    *,
    gripper_model: str = "kuka_y_gripper",
) -> dict[str, float]:
    """Return the passive driver joint needed to place one MoveIt gripper."""

    robot = str(robot_name).strip()
    if robot not in {"lbr_one", "lbr_two"}:
        raise ValueError(f"Unsupported KUKA robot name {robot_name!r}.")
    if str(gripper_model) == "pdz_gripper":
        driver_name = f"{robot}_pdz_gripper_left_finger_joint"
        return {driver_name: gripper_joint_target_from_width("pdz_gripper_left_finger_joint", width_m)}
    return {f"{robot}_left_finger_joint": kuka_moveit_gripper_driver_position_from_width(width_m)}


def gripper_width_from_joint_position(joint_name: str, joint_position_m: float) -> float:
    """Recover a physical jaw width from one supported driver coordinate."""

    name = str(joint_name)
    position = abs(float(joint_position_m))
    if "pdz_gripper_left_finger_joint" in name:
        return max(
            PDZ_GRIPPER_CLOSED_WIDTH_M,
            min(PDZ_GRIPPER_OPEN_WIDTH_M, PDZ_GRIPPER_CLOSED_WIDTH_M + 2.0 * position),
        )
    if name.endswith("left_finger_joint"):
        return max(
            KUKA_Y_GRIPPER_SOURCE_CLOSED_WIDTH_M,
            min(KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M, KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M - 2.0 * position),
        )
    return float(joint_position_m)


def gripper_max_open_width(joint_name: str) -> float:
    """Return the width command that fully opens the named gripper joint family."""

    if str(joint_name) in {"left_finger_joint", "right_finger_joint"}:
        return KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M
    if str(joint_name) in {
        "pdz_gripper_left_finger_joint",
        "pdz_gripper_right_finger_joint",
    }:
        return PDZ_GRIPPER_OPEN_WIDTH_M
    return DEFAULT_HAND_OPEN_WIDTH


def is_gripper_joint_name(joint_name: str) -> bool:
    """Return whether a joint belongs to a supported gripper."""

    name = str(joint_name)
    return name.startswith(("panda_finger_joint", "fr3_finger_joint")) or name in {
        "left_finger_joint",
        "right_finger_joint",
        "pdz_gripper_left_finger_joint",
        "pdz_gripper_right_finger_joint",
    }


def is_gripper_command_joint_name(joint_name: str) -> bool:
    """Return whether a gripper joint should receive direct position commands."""

    name = str(joint_name)
    # The established Y-gripper USD physically couples its right follower.
    # Isaac's imported PDZ USD exposes the URDF mimic joint as an independent
    # DOF under load, so both PDZ fingers must receive the same width target.
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
    "KUKA_Y_GRIPPER_SOURCE_CLOSED_WIDTH_M",
    "KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M",
    "KUKA_Y_GRIPPER_TRAVEL_M",
    "PDZ_GRIPPER_CLOSED_WIDTH_M",
    "PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M",
    "PDZ_GRIPPER_APPROACH_CLEARANCE_TOTAL_M",
    "PDZ_GRIPPER_APPROACH_PROFILE",
    "PDZ_GRIPPER_OPEN_WIDTH_M",
    "PDZ_GRIPPER_TRAVEL_M",
    "VISUAL_SERVO_GRIPPER_PROFILE",
    "gripper_joint_target_from_width",
    "gripper_approach_width",
    "gripper_clamp_width",
    "gripper_max_open_width",
    "gripper_width_from_joint_position",
    "is_gripper_command_joint_name",
    "is_gripper_joint_name",
    "kuka_gripper_approach_width",
    "kuka_gripper_clamp_width",
    "kuka_gripper_normalized_position_from_width",
    "kuka_gripper_width_from_normalized_position",
    "kuka_isaac_to_moveit_joint_positions",
    "kuka_moveit_gripper_driver_position_from_width",
    "kuka_moveit_gripper_state",
    "kuka_moveit_to_isaac_joint_positions",
    "kuka_y_gripper_approach_width_from_jaw_width",
    "pdz_gripper_approach_width_from_jaw_width",
]
