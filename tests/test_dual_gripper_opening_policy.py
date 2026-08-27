from __future__ import annotations

import math
from unittest import mock

import numpy as np

from grasp_planning.grasping import fabrica_grasp_debug
from grasp_planning.grasping.collision import (
    KUKA_Y_GRIPPER_COLLISION_GEOMETRY_VERSION,
    make_gripper_collision_model,
    make_gripper_collision_models,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    evaluate_grasps_against_ground,
    transform_primitive_to_world,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.start_poses import (
    KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
    kuka_gripper_approach_width,
    kuka_gripper_normalized_position_from_width,
    kuka_moveit_gripper_state,
)


def test_collision_policy_builds_contact_and_five_mm_per_finger_models() -> None:
    models = make_gripper_collision_models(
        "kuka_y_gripper",
        approach_gap_m=KUKA_Y_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
    )

    assert [model.contact_gap_m for model in models] == [0.0, 0.005]
    assert "dual_opening" in KUKA_Y_GRIPPER_COLLISION_GEOMETRY_VERSION


def test_hardware_and_moveit_use_the_same_candidate_specific_openings() -> None:
    jaw_width_m = 0.040
    approach_width_m = kuka_gripper_approach_width(jaw_width_m)
    approach_normalized = kuka_gripper_normalized_position_from_width(approach_width_m)
    contact_normalized = kuka_gripper_normalized_position_from_width(jaw_width_m)
    approach_moveit = kuka_moveit_gripper_state("lbr_one", approach_width_m)
    contact_moveit = kuka_moveit_gripper_state("lbr_one", jaw_width_m)

    assert math.isclose(approach_width_m, 0.050)
    assert math.isclose(approach_moveit["lbr_one_left_finger_joint"], 0.017)
    assert math.isclose(contact_moveit["lbr_one_left_finger_joint"], 0.022)
    assert approach_normalized < contact_normalized


def test_ground_filter_requires_both_contact_and_approach_openings() -> None:
    candidate = SavedGraspCandidate(
        grasp_id="dual_opening",
        grasp_position_obj=(0.0, 0.0, 0.1),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.0, -0.02, 0.1),
        contact_point_b_obj=(0.0, 0.02, 0.1),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.04,
        roll_angle_rad=0.0,
        score=1.0,
        score_components={},
    )

    class _Evaluator:
        def __init__(self, model) -> None:
            self.model = model

        def is_grasp_above_plane(self, *args, **kwargs) -> bool:
            del args, kwargs
            return self.model == "contact"

    with (
        mock.patch.object(
            fabrica_grasp_debug,
            "make_gripper_collision_models",
            return_value=("contact", "approach"),
        ),
        mock.patch.object(
            fabrica_grasp_debug,
            "WorldCollisionConstraintEvaluator",
            _Evaluator,
        ),
    ):
        statuses = evaluate_grasps_against_ground(
            [candidate],
            object_pose_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.0),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
            contact_gap_m=0.005,
            gripper_collision_model="kuka_y_gripper",
            contact_lateral_offsets_m=(0.0,),
            contact_approach_offsets_m=(0.0,),
        )

    assert statuses[0].status == "rejected"
    assert statuses[0].reason == "ground_collision"


def test_fast_kuka_floor_support_matches_transformed_collision_meshes() -> None:
    model = make_gripper_collision_models(
        "kuka_y_gripper",
        approach_gap_m=0.005,
    )[1]
    grasp_rotation = np.array(
        [
            [0.8660254, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [-0.5, 0.0, 0.8660254],
        ],
        dtype=float,
    )
    pose = ObjectWorldPose(
        position_world=(0.1, -0.2, 0.03),
        orientation_xyzw_world=(0.0, 0.0, 0.3826834324, 0.9238795325),
    )
    contact_a = np.array([0.02, -0.02, 0.04])
    contact_b = np.array([0.02, 0.02, 0.04])
    center = np.array([0.02, 0.0, 0.04])

    fast_minimum = model.minimum_world_z_for_grasp(
        grasp_rotmat_obj=grasp_rotation,
        contact_point_a_obj=contact_a,
        contact_point_b_obj=contact_b,
        grasp_center_obj=center,
        rotation_world_from_object=pose.rotation_world_from_object,
        translation_world_from_object=pose.translation_world,
    )
    primitives = model.primitives_for_grasp(
        grasp_rotmat=grasp_rotation,
        contact_point_a=contact_a,
        contact_point_b=contact_b,
        grasp_center=center,
    )
    transformed = tuple(transform_primitive_to_world(primitive, pose) for primitive in primitives)
    mesh_minimum = min(float(np.min(primitive.vertices_obj[:, 2])) for primitive in transformed)

    assert math.isclose(fast_minimum, mesh_minimum, abs_tol=1.0e-10)
    fast_bounds = model.world_component_aabb_bounds_for_grasp(
        grasp_rotmat_obj=grasp_rotation,
        contact_point_a_obj=contact_a,
        contact_point_b_obj=contact_b,
        grasp_center_obj=center,
        rotation_world_from_object=pose.rotation_world_from_object,
        translation_world_from_object=pose.translation_world,
    )
    assert [item[0] for item in fast_bounds] == [primitive.name for primitive in transformed]
    for (_, minimum, maximum), primitive in zip(fast_bounds, transformed, strict=True):
        np.testing.assert_allclose(minimum, primitive.vertices_obj.min(axis=0), atol=1.0e-10)
        np.testing.assert_allclose(maximum, primitive.vertices_obj.max(axis=0), atol=1.0e-10)


def test_mesh_gripper_contact_offsets_shift_an_explicit_saved_grasp_center() -> None:
    grasp_rotation = np.eye(3, dtype=float)
    contact_a = np.array([0.0, -0.02, 0.04])
    contact_b = np.array([0.0, 0.02, 0.04])
    center = np.array([0.0, 0.0, 0.04])
    lateral_offset_m = 0.01
    approach_offset_m = -0.005
    expected_translation = np.array([-0.01, 0.0, 0.005])

    for model_name in ("kuka_y_gripper", "pdz_gripper"):
        nominal_model = make_gripper_collision_model(model_name, contact_gap_m=0.0)
        offset_model = make_gripper_collision_model(
            model_name,
            contact_gap_m=0.0,
            contact_patch_lateral_offset_m=lateral_offset_m,
            contact_patch_approach_offset_m=approach_offset_m,
        )
        nominal_primitives = nominal_model.primitives_for_grasp(
            grasp_rotmat=grasp_rotation,
            contact_point_a=contact_a,
            contact_point_b=contact_b,
            grasp_center=center,
        )
        offset_primitives = offset_model.primitives_for_grasp(
            grasp_rotmat=grasp_rotation,
            contact_point_a=contact_a,
            contact_point_b=contact_b,
            grasp_center=center,
        )

        assert [item.name for item in nominal_primitives] == [item.name for item in offset_primitives]
        for nominal, offset in zip(nominal_primitives, offset_primitives, strict=True):
            np.testing.assert_allclose(
                offset.vertices_obj - nominal.vertices_obj,
                np.broadcast_to(expected_translation, offset.vertices_obj.shape),
                atol=1.0e-12,
            )


def test_fast_pdz_floor_support_matches_transformed_collision_hulls() -> None:
    model = make_gripper_collision_model(
        "pdz_gripper",
        contact_gap_m=0.005,
        contact_patch_lateral_offset_m=0.015,
        contact_patch_approach_offset_m=0.005,
    )
    grasp_rotation = np.array(
        [
            [0.8660254, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [-0.5, 0.0, 0.8660254],
        ],
        dtype=float,
    )
    pose = ObjectWorldPose(
        position_world=(0.1, -0.2, 0.03),
        orientation_xyzw_world=(0.0, 0.0, 0.3826834324, 0.9238795325),
    )
    contact_a = np.array([0.02, -0.02, 0.04])
    contact_b = np.array([0.02, 0.02, 0.04])
    center = np.array([0.02, 0.0, 0.04])

    fast_minimum = model.minimum_world_z_for_grasp(
        grasp_rotmat_obj=grasp_rotation,
        contact_point_a_obj=contact_a,
        contact_point_b_obj=contact_b,
        grasp_center_obj=center,
        rotation_world_from_object=pose.rotation_world_from_object,
        translation_world_from_object=pose.translation_world,
    )
    primitives = model.primitives_for_grasp(
        grasp_rotmat=grasp_rotation,
        contact_point_a=contact_a,
        contact_point_b=contact_b,
        grasp_center=center,
    )
    transformed = tuple(transform_primitive_to_world(primitive, pose) for primitive in primitives)
    mesh_minimum = min(float(np.min(primitive.vertices_obj[:, 2])) for primitive in transformed)

    assert math.isclose(fast_minimum, mesh_minimum, abs_tol=1.0e-10)
    fast_bounds = model.world_component_aabb_bounds_for_grasp(
        grasp_rotmat_obj=grasp_rotation,
        contact_point_a_obj=contact_a,
        contact_point_b_obj=contact_b,
        grasp_center_obj=center,
        rotation_world_from_object=pose.rotation_world_from_object,
        translation_world_from_object=pose.translation_world,
    )
    assert [item[0] for item in fast_bounds] == [primitive.name for primitive in transformed]
    for (_, minimum, maximum), primitive in zip(fast_bounds, transformed, strict=True):
        np.testing.assert_allclose(minimum, primitive.vertices_obj.min(axis=0), atol=1.0e-10)
        np.testing.assert_allclose(maximum, primitive.vertices_obj.max(axis=0), atol=1.0e-10)
