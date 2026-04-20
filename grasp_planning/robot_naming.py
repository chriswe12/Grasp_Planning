"""Helpers for FR3-family joint and asset naming across imported variants."""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_FRANKA_PREFIXES = ("fr3v2_1", "fr3v2", "fr3")


def infer_robot_name_prefix_from_asset_path(asset_path: str) -> str:
    """Infer the Franka joint-name prefix from a USD path or Omniverse URL."""

    normalized = str(asset_path).lower()
    basename = Path(normalized).name
    for prefix in SUPPORTED_FRANKA_PREFIXES:
        if prefix in basename or f"/{prefix}/" in normalized or f"_{prefix}_" in normalized:
            return prefix
    return "fr3"


def infer_robot_name_prefix_from_joint_names(joint_names: list[str] | tuple[str, ...]) -> str:
    """Infer the Franka joint-name prefix from articulation joint names."""

    for name in joint_names:
        match = re.fullmatch(r"(.+)_joint1", name)
        if match is not None:
            return match.group(1)
    raise RuntimeError(
        f"Could not infer the robot joint-name prefix from the articulation. Available joints: {', '.join(joint_names)}"
    )


def remap_joint_name(base_name: str, prefix: str) -> str:
    """Map a base FR3 joint name or regex to the requested Franka prefix."""

    if prefix == "fr3":
        return base_name
    return base_name.replace("fr3_", f"{prefix}_", 1)


def remap_joint_targets(joint_targets: dict[str, float], prefix: str) -> dict[str, float]:
    """Map FR3-base joint targets to a specific Franka variant prefix."""

    return {remap_joint_name(name, prefix): value for name, value in joint_targets.items()}


def arm_joint_names_for_prefix(prefix: str) -> tuple[str, ...]:
    """Return the canonical 7 arm joint names for a Franka variant."""

    return tuple(remap_joint_name(f"fr3_joint{joint_idx}", prefix) for joint_idx in range(1, 8))


def finger_joint_names_for_prefix(prefix: str) -> tuple[str, str]:
    """Return the canonical finger joint names for a Franka variant."""

    return (
        remap_joint_name("fr3_finger_joint1", prefix),
        remap_joint_name("fr3_finger_joint2", prefix),
    )


def hand_joint_regex_for_prefix(prefix: str) -> str:
    """Return the hand-joint regex for a Franka variant."""

    return remap_joint_name("fr3_finger_joint.*", prefix)
