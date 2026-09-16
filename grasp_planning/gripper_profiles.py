"""Physical gripper profiles shared by real-robot controller clients."""

from __future__ import annotations

SERVO_GRIPPER_CLOSED_WIDTH_M = 0.007
SERVO_GRIPPER_OPEN_WIDTH_M = 0.074


def servo_gripper_clamp_width(width_m: float) -> float:
    """Clamp a requested jaw width to the calibrated 7--74 mm stroke."""

    return max(
        SERVO_GRIPPER_CLOSED_WIDTH_M,
        min(SERVO_GRIPPER_OPEN_WIDTH_M, float(width_m)),
    )


def servo_gripper_closure_fraction_from_width(width_m: float) -> float:
    """Map jaw width to the controller contract: 0=open and 1=closed."""

    width = servo_gripper_clamp_width(width_m)
    span = SERVO_GRIPPER_OPEN_WIDTH_M - SERVO_GRIPPER_CLOSED_WIDTH_M
    return (SERVO_GRIPPER_OPEN_WIDTH_M - width) / span


def servo_gripper_width_from_closure_fraction(position: float) -> float:
    """Map a controller closure fraction back to calibrated jaw width."""

    normalized = max(0.0, min(1.0, float(position)))
    span = SERVO_GRIPPER_OPEN_WIDTH_M - SERVO_GRIPPER_CLOSED_WIDTH_M
    return SERVO_GRIPPER_OPEN_WIDTH_M - normalized * span


__all__ = [
    "SERVO_GRIPPER_CLOSED_WIDTH_M",
    "SERVO_GRIPPER_OPEN_WIDTH_M",
    "servo_gripper_clamp_width",
    "servo_gripper_closure_fraction_from_width",
    "servo_gripper_width_from_closure_fraction",
]
