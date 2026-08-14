from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from grasp_planning.start_poses import (
    KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
    KUKA_Y_GRIPPER_APPROACH_PROFILE,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaac_rl/source/isaac_rl/isaac_rl/tasks/direct/isaac_rl/multigrasp_catalog.py"
)
SPEC = importlib.util.spec_from_file_location("multigrasp_catalog_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
multigrasp_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multigrasp_catalog)
CATALOG_SCHEMA_VERSION = multigrasp_catalog.CATALOG_SCHEMA_VERSION
ROTATION_COLLISION_VALIDATION_PROFILE = (
    multigrasp_catalog.ROTATION_COLLISION_VALIDATION_PROFILE
)
ROTATION_RESET_SCHEMA_VERSION = multigrasp_catalog.ROTATION_RESET_SCHEMA_VERSION
load_multigrasp_catalog = multigrasp_catalog.load_multigrasp_catalog
load_multigrasp_rotation_resets = multigrasp_catalog.load_multigrasp_rotation_resets


def _catalog_payload() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(CATALOG_SCHEMA_VERSION, dtype=np.int64),
        "target_ids": np.asarray(["target_a"]),
        "orientation_names": np.asarray(["orientation_a"]),
        "orientation_ids": np.asarray(["orientation_a"]),
        "orientation_indices": np.asarray([0], dtype=np.int64),
        "grasp_ids": np.asarray(["grasp_a"]),
        "goal_rgb": np.zeros((1, 2, 3, 3), dtype=np.uint8),
        "goal_depth": np.ones((1, 2, 3), dtype=np.float32),
        "object_positions_w": np.zeros((1, 3), dtype=np.float32),
        "object_orientations_xyzw_w": np.asarray(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        "goal_grasp_positions_w": np.zeros((1, 3), dtype=np.float32),
        "goal_grasp_orientations_xyzw_w": np.asarray(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        "goal_tcp_positions_w": np.zeros((1, 3), dtype=np.float32),
        "goal_tcp_orientations_xyzw_w": np.asarray(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        ),
        "reset_joint_trajectories": np.zeros((1, 2, 7), dtype=np.float32),
        "reset_path_progress": np.asarray([0.0, 1.0], dtype=np.float32),
        "moveit_plan_validated": np.ones(1, dtype=np.bool_),
        "isaac_goal_rgbd_captured": np.ones(1, dtype=np.bool_),
        "approach_gripper_profile": np.asarray(KUKA_Y_GRIPPER_APPROACH_PROFILE),
        "approach_clearance_per_finger_m": np.asarray(
            KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M, dtype=np.float32
        ),
        "grasp_jaw_widths_m": np.asarray([0.042], dtype=np.float32),
        "approach_gripper_widths_m": np.asarray([0.052], dtype=np.float32),
        "assembly_name": np.asarray("assembly"),
        "part_names": np.asarray(["0"]),
        "part_usd_paths": np.asarray(["part.usd"]),
        "part_ids": np.asarray(["0"]),
        "part_indices": np.asarray([0], dtype=np.int64),
        "local_orientation_ids": np.asarray(["orientation_a"]),
        "split_names": np.asarray(["train", "validation", "test"]),
        "split_ids": np.asarray(["train"]),
        "split_indices": np.asarray([0], dtype=np.int64),
    }


def test_schema_three_catalog_requires_jaw_plus_ten_mm(tmp_path: Path) -> None:
    path = tmp_path / "catalog.npz"
    payload = _catalog_payload()
    np.savez_compressed(path, **payload)

    loaded = load_multigrasp_catalog(path)
    assert loaded["approach_gripper_widths_m"].tolist() == pytest.approx([0.052])

    payload["approach_gripper_widths_m"] = np.asarray([0.084], dtype=np.float32)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="final jaw width plus 10 mm"):
        load_multigrasp_catalog(path)


def _rotation_payload() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(ROTATION_RESET_SCHEMA_VERSION, dtype=np.int64),
        "axis_selection_method": np.asarray("fibonacci_farthest_point_v1"),
        "target_ids": np.asarray(["target_a"]),
        "rotation_axes_w": np.asarray(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32
        ),
        "rotation_joint_trajectories": np.zeros((1, 2, 2, 7), dtype=np.float32),
        "rotation_angle_profile_rad": np.asarray([0.2, 0.1], dtype=np.float32),
        "ik_position_residual_m": np.zeros((1, 2, 2), dtype=np.float32),
        "ik_rotation_residual_rad": np.zeros((1, 2, 2), dtype=np.float32),
        "collision_validation_profile": np.asarray(
            ROTATION_COLLISION_VALIDATION_PROFILE
        ),
        "minimum_collision_clearance_m": np.asarray(0.001, dtype=np.float32),
        "collision_validated": np.ones((1, 2, 2), dtype=np.bool_),
        "collision_clearance_m": np.full((1, 2, 2), 0.002, dtype=np.float32),
        "nominal_collision_validated": np.ones((1, 2), dtype=np.bool_),
        "nominal_collision_clearance_m": np.full((1, 2), 0.002, dtype=np.float32),
        "approach_gripper_widths_m": np.asarray([0.052], dtype=np.float32),
    }


def test_rotation_asset_requires_every_reset_to_be_collision_validated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rotation.npz"
    payload = _rotation_payload()
    np.savez_compressed(path, **payload)
    load_multigrasp_rotation_resets(
        path,
        expected_target_ids=("target_a",),
        expected_waypoint_count=2,
        expected_approach_gripper_widths_m=np.asarray([0.052]),
    )

    payload["collision_validated"][0, 1, 1] = False
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="Every authored rotation-reset state"):
        load_multigrasp_rotation_resets(
            path,
            expected_target_ids=("target_a",),
            expected_waypoint_count=2,
            expected_approach_gripper_widths_m=np.asarray([0.052]),
        )
