from __future__ import annotations

import unittest

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspCandidate, rotmat_to_quat_xyzw
from grasp_planning.grasping.grasp_transforms import saved_grasp_to_world_grasp
from grasp_planning.grasping.world_constraints import ObjectWorldPose


class SavedGraspTransformTests(unittest.TestCase):
    def _candidate(self) -> SavedGraspCandidate:
        return SavedGraspCandidate(
            grasp_id="g0001",
            grasp_position_obj=(0.0, 0.0, 0.0),
            grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
            contact_point_a_obj=(0.0, -0.02, 0.0),
            contact_point_b_obj=(0.0, 0.02, 0.0),
            contact_normal_a_obj=(0.0, 1.0, 0.0),
            contact_normal_b_obj=(0.0, -1.0, 0.0),
            jaw_width=0.04,
            roll_angle_rad=0.0,
        )

    def test_saved_grasp_to_world_grasp_identity_pose(self) -> None:
        world_grasp = saved_grasp_to_world_grasp(
            self._candidate(),
            ObjectWorldPose(position_world=(0.0, 0.0, 0.0), orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0)),
            pregrasp_offset=0.2,
            gripper_width_clearance=0.01,
        )

        np.testing.assert_allclose(world_grasp.position_w, (0.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(world_grasp.normal_w, (0.0, 0.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(world_grasp.pregrasp_position_w, (0.0, 0.0, -0.2), atol=1e-6)
        self.assertAlmostEqual(world_grasp.gripper_width, 0.05)

    def test_saved_grasp_to_world_grasp_applies_translation(self) -> None:
        world_grasp = saved_grasp_to_world_grasp(
            self._candidate(),
            ObjectWorldPose(position_world=(0.3, -0.1, 0.2), orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0)),
            pregrasp_offset=0.1,
            gripper_width_clearance=0.0,
        )

        np.testing.assert_allclose(world_grasp.position_w, (0.3, -0.1, 0.2), atol=1e-6)
        np.testing.assert_allclose(world_grasp.contact_point_a_w, (0.3, -0.12, 0.2), atol=1e-6)
        np.testing.assert_allclose(world_grasp.contact_point_b_w, (0.3, -0.08, 0.2), atol=1e-6)

    def test_saved_grasp_to_world_grasp_applies_rotation(self) -> None:
        yaw_90 = rotmat_to_quat_xyzw(
            np.array(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        )
        world_grasp = saved_grasp_to_world_grasp(
            self._candidate(),
            ObjectWorldPose(position_world=(0.0, 0.0, 0.0), orientation_xyzw_world=yaw_90),
            pregrasp_offset=0.1,
            gripper_width_clearance=0.0,
        )

        np.testing.assert_allclose(world_grasp.normal_w, (0.0, 0.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(world_grasp.contact_point_a_w, (0.02, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(world_grasp.contact_point_b_w, (-0.02, 0.0, 0.0), atol=1e-6)

    def test_pregrasp_position_uses_transformed_approach_axis(self) -> None:
        rot_x_neg_90 = rotmat_to_quat_xyzw(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=float,
            )
        )
        world_grasp = saved_grasp_to_world_grasp(
            self._candidate(),
            ObjectWorldPose(position_world=(0.0, 0.0, 0.0), orientation_xyzw_world=rot_x_neg_90),
            pregrasp_offset=0.1,
            gripper_width_clearance=0.0,
        )

        np.testing.assert_allclose(world_grasp.normal_w, (0.0, 1.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(world_grasp.pregrasp_position_w, (0.0, -0.1, 0.0), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
