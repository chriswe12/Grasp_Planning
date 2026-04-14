from __future__ import annotations

import unittest

from grasp_planning.grasping.fabrica_grasp_debug import (
    CandidateStatus,
    SavedGraspCandidate,
    select_first_feasible_grasp,
)


def _candidate(grasp_id: str) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=(0.0, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.0, -0.02, 0.0),
        contact_point_b_obj=(0.0, 0.02, 0.0),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.04,
        roll_angle_rad=0.0,
    )


class FirstFeasibleSelectionTests(unittest.TestCase):
    def test_first_feasible_grasp_selection_picks_first(self) -> None:
        statuses = [
            CandidateStatus(grasp=_candidate("g0001"), status="rejected", reason="ground_collision"),
            CandidateStatus(grasp=_candidate("g0002"), status="accepted", reason="clear_of_ground"),
            CandidateStatus(grasp=_candidate("g0003"), status="accepted", reason="clear_of_ground"),
        ]

        selected = select_first_feasible_grasp(statuses)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.grasp_id, "g0002")

    def test_first_feasible_grasp_selection_handles_empty_list(self) -> None:
        self.assertIsNone(select_first_feasible_grasp([]))


if __name__ == "__main__":
    unittest.main()
