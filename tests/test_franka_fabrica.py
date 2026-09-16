from dataclasses import replace

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspCandidate, quat_to_rotmat_xyzw
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.rl.franka_fabrica import PLANNER_HAND_TO_CONTACT_M, task_pose


def candidate():
    return SavedGraspCandidate(
        grasp_id="g1",
        grasp_position_obj=(0.01, 0.02, 0.03),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.01, 0.0, 0.03),
        contact_point_b_obj=(0.01, 0.04, 0.03),
        contact_normal_a_obj=(0.0, -1.0, 0.0),
        contact_normal_b_obj=(0.0, 1.0, 0.0),
        jaw_width=0.04,
        roll_angle_rad=0.0,
    )


def test_pad_offset_and_tcp_correction_applied_in_rotated_frame():
    c = replace(candidate(), contact_patch_lateral_offset_m=0.002, contact_patch_approach_offset_m=-0.003)
    pose = ObjectWorldPose(position_world=(0.4, 0.1, 0.06), orientation_xyzw_world=(0.0, 0.0, 2**-0.5, 2**-0.5))
    goal, axis = task_pose(c, pose)
    r = pose.rotation_world_from_object
    expected = pose.translation_world + r @ np.array([0.008, 0.02, 0.033 + 0.1034 - PLANNER_HAND_TO_CONTACT_M])
    np.testing.assert_allclose(goal[:3], expected, atol=1e-12)
    np.testing.assert_allclose(quat_to_rotmat_xyzw((*goal[4:], goal[3])), r, atol=1e-12)
    np.testing.assert_allclose(axis, r[:, 2], atol=1e-12)


def test_tcp_change_preserves_hand_origin():
    c = candidate()
    obj = ObjectWorldPose(position_world=(0.0, 0.0, 0.0), orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0))
    first, _ = task_pose(c, obj, (0.0, 0.0, 0.1034))
    second, _ = task_pose(c, obj, (0.0, 0.0, 0.12))
    np.testing.assert_allclose(first[:3] - [0, 0, 0.1034], second[:3] - [0, 0, 0.12], atol=1e-12)
