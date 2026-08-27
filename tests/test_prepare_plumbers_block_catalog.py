from __future__ import annotations

import importlib.util
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
from grasp_planning.rl.goal_catalog_profiles import MUJOCO_GOAL_RENDERER_PROFILE
from grasp_planning.start_poses import (
    PDZ_GRIPPER_APPROACH_PROFILE,
    VISUAL_SERVO_GRIPPER_PROFILE,
)
from grasp_planning.visual_servo_workspace import VISUAL_SERVO_TSLOT_PROFILE

SCRIPT = Path(__file__).resolve().parents[1] / "isaac_rl/scripts/prepare_plumbers_block_catalog.py"
SPEC = importlib.util.spec_from_file_location("prepare_plumbers_block_catalog", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_finalize_failed_isaac_capture = MODULE._finalize_failed_isaac_capture
_write_target_subset_paths_asset = MODULE._write_target_subset_paths_asset


def _write_source_paths(path: Path) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(2, dtype=np.int64),
        target_ids=np.asarray(["a", "b", "c", "d"]),
        reset_joint_trajectories=np.arange(4 * 2 * 7).reshape(4, 2, 7),
        moveit_plan_validated=np.ones(4, dtype=np.bool_),
        orientation_names=np.asarray(["o0", "o1"]),
        reset_path_progress=np.asarray([0.0, 1.0]),
    )


def test_target_subset_paths_follow_rotation_reset_order(tmp_path: Path) -> None:
    source = tmp_path / "paths.npz"
    output = tmp_path / "capture_paths.npz"
    _write_source_paths(source)

    count = _write_target_subset_paths_asset(
        source,
        output,
        np.asarray(["c", "a"]),
    )

    assert count == 2
    with np.load(output, allow_pickle=False) as data:
        assert data["target_ids"].tolist() == ["c", "a"]
        np.testing.assert_array_equal(
            data["reset_joint_trajectories"],
            np.concatenate(
                (
                    np.arange(4 * 2 * 7).reshape(4, 2, 7)[2:3],
                    np.arange(4 * 2 * 7).reshape(4, 2, 7)[0:1],
                )
            ),
        )
        assert data["moveit_plan_validated"].tolist() == [True, True]
        assert data["orientation_names"].tolist() == ["o0", "o1"]
        np.testing.assert_array_equal(
            data["reset_path_progress"], np.asarray([0.0, 1.0])
        )


def test_target_subset_paths_reject_unknown_rotation_target(tmp_path: Path) -> None:
    source = tmp_path / "paths.npz"
    output = tmp_path / "capture_paths.npz"
    _write_source_paths(source)

    with pytest.raises(ValueError, match="absent from the path asset"):
        _write_target_subset_paths_asset(
            source,
            output,
            np.asarray(["missing"]),
        )


def _write_failed_capture_fixture(data_root: Path) -> None:
    target_ids = np.asarray(["a", "b", "c"])
    np.savez_compressed(
        data_root / "paths.npz",
        schema_version=np.asarray(2, dtype=np.int64),
        target_ids=target_ids,
        reset_joint_trajectories=np.zeros((3, 2, 7), dtype=np.float32),
        moveit_plan_validated=np.ones(3, dtype=np.bool_),
    )
    np.savez_compressed(
        data_root / "rotation_resets.npz",
        schema_version=np.asarray(2, dtype=np.int64),
        target_ids=target_ids,
        rotation_axes_w=np.ones((3, 2, 3), dtype=np.float32),
        rotation_angle_profile_rad=np.asarray([0.2, 0.0], dtype=np.float32),
        rotation_joint_trajectories=np.zeros((3, 2, 2, 7), dtype=np.float32),
    )
    passed = np.asarray([True, False, True], dtype=np.bool_)
    np.savez_compressed(
        data_root / "goal_catalog_failed_validation.npz",
        schema_version=np.asarray(2, dtype=np.int64),
        target_ids=target_ids,
        part_ids=np.asarray(["0", "1", "2"]),
        goal_rgb=np.zeros(
            (3, VISUAL_SERVO_RENDER_HEIGHT, VISUAL_SERVO_RENDER_WIDTH, 3),
            dtype=np.uint8,
        ),
        goal_depth=np.ones(
            (3, VISUAL_SERVO_RENDER_HEIGHT, VISUAL_SERVO_RENDER_WIDTH),
            dtype=np.float32,
        ),
        moveit_plan_validated=np.ones(3, dtype=np.bool_),
        capture_validation_passed=passed,
        isaac_goal_rgbd_captured=passed,
        robot_profile=np.asarray(VISUAL_SERVO_GRIPPER_PROFILE),
        goal_renderer_profile=np.asarray(MUJOCO_GOAL_RENDERER_PROFILE),
        approach_gripper_profile=np.asarray(PDZ_GRIPPER_APPROACH_PROFILE),
        visual_material_profile=np.asarray(VISUAL_SERVO_MATERIAL_PROFILE),
        visual_scene_profile=np.asarray(VISUAL_SERVO_SCENE_PROFILE),
        visual_tslot_profile=np.asarray(VISUAL_SERVO_TSLOT_PROFILE),
        goal_camera_profile=np.asarray(D405_VISUAL_SERVO_CAMERA_PROFILE),
        goal_observation_profile=np.asarray(D405_VISUAL_SERVO_OBSERVATION_PROFILE),
    )
    np.savez_compressed(
        data_root / "goal_catalog.npz",
        target_ids=np.asarray(["old"]),
    )


def test_finalize_promotes_passing_goals_and_matching_rotation_rows(
    tmp_path: Path,
) -> None:
    _write_failed_capture_fixture(tmp_path)

    assert _finalize_failed_isaac_capture(tmp_path) == 2

    with np.load(tmp_path / "goal_catalog.npz", allow_pickle=False) as catalog:
        assert catalog["target_ids"].tolist() == ["a", "c"]
        assert catalog["part_ids"].tolist() == ["0", "2"]
        assert bool(catalog["capture_validation_passed"].all())
        assert bool(catalog["isaac_goal_rgbd_captured"].all())
    with np.load(tmp_path / "rotation_resets.npz", allow_pickle=False) as resets:
        assert resets["target_ids"].tolist() == ["a", "c"]
        assert resets["rotation_joint_trajectories"].shape == (2, 2, 2, 7)
    with np.load(
        tmp_path / "goal_catalog_failed_validation.npz", allow_pickle=False
    ) as diagnostic:
        assert len(diagnostic["target_ids"]) == 3


def test_finalize_is_idempotent_after_assets_are_aligned(tmp_path: Path) -> None:
    _write_failed_capture_fixture(tmp_path)
    _finalize_failed_isaac_capture(tmp_path)

    assert _finalize_failed_isaac_capture(tmp_path) == 2
