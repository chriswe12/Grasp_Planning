"""Postconditions for Isaac-rendered visual-servo goal catalogs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_RENDER_HEIGHT,
    VISUAL_SERVO_RENDER_WIDTH,
)
from grasp_planning.isaac_visual_materials import VISUAL_SERVO_MATERIAL_PROFILE
from grasp_planning.isaac_visual_scene import VISUAL_SERVO_SCENE_PROFILE
from grasp_planning.start_poses import KUKA_Y_GRIPPER_APPROACH_PROFILE

CatalogFileSignature = tuple[int, int, int]


def catalog_file_signature(path: str | Path) -> CatalogFileSignature | None:
    """Return enough file identity to detect whether an atomic capture replaced it."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return None
    stat = resolved.stat()
    return int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size)


def _scalar_string(arrays: Any, name: str) -> str:
    if name not in arrays:
        raise RuntimeError(f"Captured catalog is missing required profile '{name}'.")
    value = np.asarray(arrays[name])
    if value.ndim != 0:
        raise RuntimeError(
            f"Captured catalog profile '{name}' must be scalar, got {value.shape}."
        )
    return str(value.item())


def validate_fresh_goal_catalog_capture(
    catalog_path: str | Path,
    paths_asset_path: str | Path,
    *,
    previous_signature: CatalogFileSignature | None,
) -> int:
    """Reject stale, incomplete, or visually incompatible capture output."""

    catalog = Path(catalog_path).expanduser().resolve()
    paths_asset = Path(paths_asset_path).expanduser().resolve()
    current_signature = catalog_file_signature(catalog)
    if current_signature is None:
        raise RuntimeError(
            f"Isaac capture returned without creating the goal catalog: {catalog}."
        )
    if previous_signature is not None and current_signature == previous_signature:
        raise RuntimeError(
            "Isaac capture returned without replacing the existing goal catalog. "
            "The renderer likely stopped before Python capture code ran (for example, "
            "because no CUDA/Vulkan device was available); the stale catalog was left "
            f"untouched at {catalog}."
        )
    if not paths_asset.is_file():
        raise FileNotFoundError(paths_asset)

    with np.load(paths_asset, allow_pickle=False) as source:
        expected_target_ids = np.asarray(source["target_ids"]).astype(str)
    with np.load(catalog, allow_pickle=False) as source:
        target_ids = (
            np.asarray(source["target_ids"]).astype(str)
            if "target_ids" in source
            else np.asarray([])
        )
        if not np.array_equal(target_ids, expected_target_ids):
            raise RuntimeError(
                "Captured catalog target_ids do not exactly match the validated path asset."
            )
        target_count = int(target_ids.size)
        if target_count < 1:
            raise RuntimeError("Captured catalog contains no targets.")

        expected_shapes = {
            "goal_rgb": (
                target_count,
                VISUAL_SERVO_RENDER_HEIGHT,
                VISUAL_SERVO_RENDER_WIDTH,
                3,
            ),
            "goal_depth": (
                target_count,
                VISUAL_SERVO_RENDER_HEIGHT,
                VISUAL_SERVO_RENDER_WIDTH,
            ),
            "moveit_plan_validated": (target_count,),
            "isaac_goal_rgbd_captured": (target_count,),
        }
        for name, expected_shape in expected_shapes.items():
            value = source[name] if name in source else None
            if value is None or value.shape != expected_shape:
                actual_shape = None if value is None else value.shape
                raise RuntimeError(
                    f"Captured catalog array '{name}' must have shape {expected_shape}, "
                    f"got {actual_shape}."
                )
        for name in ("moveit_plan_validated", "isaac_goal_rgbd_captured"):
            value = source[name]
            if value.dtype != np.bool_ or not bool(value.all()):
                raise RuntimeError(
                    f"Captured catalog array '{name}' must be complete and boolean."
                )

        expected_profiles = {
            "approach_gripper_profile": KUKA_Y_GRIPPER_APPROACH_PROFILE,
            "visual_material_profile": VISUAL_SERVO_MATERIAL_PROFILE,
            "visual_scene_profile": VISUAL_SERVO_SCENE_PROFILE,
            "goal_camera_profile": D405_VISUAL_SERVO_CAMERA_PROFILE,
            "goal_observation_profile": D405_VISUAL_SERVO_OBSERVATION_PROFILE,
        }
        for name, expected in expected_profiles.items():
            actual = _scalar_string(source, name)
            if actual != expected:
                raise RuntimeError(
                    f"Captured catalog profile '{name}' is '{actual}', "
                    f"expected '{expected}'."
                )
    return target_count


__all__ = [
    "CatalogFileSignature",
    "catalog_file_signature",
    "validate_fresh_goal_catalog_capture",
]
