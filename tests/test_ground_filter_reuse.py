from __future__ import annotations

import unittest

from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    accepted_grasps,
    evaluate_saved_grasps_against_pickup_pose,
    trimesh,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose


@unittest.skipIf(trimesh is None, "trimesh is required to evaluate saved grasps against the ground")
class GroundFilterReuseTests(unittest.TestCase):
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

    def test_evaluate_saved_grasps_against_pickup_pose_returns_statuses(self) -> None:
        statuses = evaluate_saved_grasps_against_pickup_pose(
            [self._candidate()],
            object_pose_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.1), orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0)
            ),
            contact_gap_m=0.002,
        )

        self.assertEqual(len(statuses), 1)
        self.assertIn(statuses[0].status, {"accepted", "rejected"})

    def test_accepted_grasps_returns_only_accepted_entries(self) -> None:
        statuses = evaluate_saved_grasps_against_pickup_pose(
            [self._candidate()],
            object_pose_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.2), orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0)
            ),
            contact_gap_m=0.002,
        )

        accepted = accepted_grasps(statuses)
        self.assertLessEqual(len(accepted), len(statuses))
        for grasp in accepted:
            self.assertEqual(grasp.grasp_id, "g0001")

    def test_pickup_pose_evaluation_does_not_require_hardcoded_spec(self) -> None:
        statuses = evaluate_saved_grasps_against_pickup_pose(
            [self._candidate()],
            object_pose_world=ObjectWorldPose(
                position_world=(0.15, -0.05, 0.25), orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0)
            ),
            contact_gap_m=0.002,
        )

        self.assertEqual(len(statuses), 1)


if __name__ == "__main__":
    unittest.main()
