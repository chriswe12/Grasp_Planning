from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_RENDER_HEIGHT,
    VISUAL_SERVO_RENDER_WIDTH,
)
from grasp_planning.isaac_visual_materials import VISUAL_SERVO_MATERIAL_PROFILE
from grasp_planning.isaac_visual_scene import VISUAL_SERVO_SCENE_PROFILE
from grasp_planning.rl.catalog_capture_validation import (
    catalog_file_signature,
    validate_fresh_goal_catalog_capture,
)
from grasp_planning.rl.goal_catalog_profiles import MUJOCO_GOAL_RENDERER_PROFILE
from grasp_planning.start_poses import (
    PDZ_GRIPPER_APPROACH_PROFILE,
    VISUAL_SERVO_GRIPPER_PROFILE,
)
from grasp_planning.visual_servo_workspace import VISUAL_SERVO_TSLOT_PROFILE


def _write_paths(path: Path) -> None:
    np.savez_compressed(path, target_ids=np.asarray(["target_a", "target_b"]))


def _write_catalog(path: Path, *, scene_profile: str = VISUAL_SERVO_SCENE_PROFILE) -> None:
    target_count = 2
    np.savez_compressed(
        path,
        target_ids=np.asarray(["target_a", "target_b"]),
        goal_rgb=np.zeros(
            (
                target_count,
                VISUAL_SERVO_RENDER_HEIGHT,
                VISUAL_SERVO_RENDER_WIDTH,
                3,
            ),
            dtype=np.uint8,
        ),
        goal_depth=np.ones(
            (
                target_count,
                VISUAL_SERVO_RENDER_HEIGHT,
                VISUAL_SERVO_RENDER_WIDTH,
            ),
            dtype=np.float32,
        ),
        moveit_plan_validated=np.ones(target_count, dtype=np.bool_),
        isaac_goal_rgbd_captured=np.ones(target_count, dtype=np.bool_),
        robot_profile=np.asarray(VISUAL_SERVO_GRIPPER_PROFILE),
        approach_gripper_profile=np.asarray(PDZ_GRIPPER_APPROACH_PROFILE),
        goal_renderer_profile=np.asarray(MUJOCO_GOAL_RENDERER_PROFILE),
        visual_material_profile=np.asarray(VISUAL_SERVO_MATERIAL_PROFILE),
        visual_scene_profile=np.asarray(scene_profile),
        visual_tslot_profile=np.asarray(VISUAL_SERVO_TSLOT_PROFILE),
        goal_camera_profile=np.asarray(D405_VISUAL_SERVO_CAMERA_PROFILE),
        goal_observation_profile=np.asarray(D405_VISUAL_SERVO_OBSERVATION_PROFILE),
    )


def test_rejects_unchanged_catalog_after_capture_command(tmp_path: Path) -> None:
    paths = tmp_path / "paths.npz"
    catalog = tmp_path / "catalog.npz"
    _write_paths(paths)
    _write_catalog(catalog)
    previous_signature = catalog_file_signature(catalog)

    with pytest.raises(RuntimeError, match="without replacing"):
        validate_fresh_goal_catalog_capture(
            catalog,
            paths,
            previous_signature=previous_signature,
        )


def test_accepts_fresh_complete_catalog_with_active_profiles(tmp_path: Path) -> None:
    paths = tmp_path / "paths.npz"
    catalog = tmp_path / "catalog.npz"
    _write_paths(paths)
    _write_catalog(catalog)

    assert (
        validate_fresh_goal_catalog_capture(
            catalog,
            paths,
            previous_signature=None,
        )
        == 2
    )


def test_rejects_fresh_catalog_with_stale_visual_profile(tmp_path: Path) -> None:
    paths = tmp_path / "paths.npz"
    catalog = tmp_path / "catalog.npz"
    _write_paths(paths)
    _write_catalog(catalog, scene_profile="old_scene")

    with pytest.raises(RuntimeError, match="visual_scene_profile"):
        validate_fresh_goal_catalog_capture(
            catalog,
            paths,
            previous_signature=None,
        )
