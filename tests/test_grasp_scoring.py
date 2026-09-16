from __future__ import annotations

import inspect
import unittest

import numpy as np

from grasp_planning.grasping import SavedGraspCandidate, TriangleMesh, score_grasps


def _make_box_mesh(size_xyz: tuple[float, float, float]) -> TriangleMesh:
    half_x, half_y, half_z = (0.5 * float(value) for value in size_xyz)
    vertices = np.array(
        [
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, half_y, -half_z],
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [half_x, half_y, half_z],
            [-half_x, half_y, half_z],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices_obj=vertices, faces=faces)


def _side_face_grasp(grasp_id: str, *, x: float = 0.0, z: float = 0.0) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=(x, 0.0, z),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(x, -0.02, z),
        contact_point_b_obj=(x, 0.02, z),
        contact_normal_a_obj=(0.0, -1.0, 0.0),
        contact_normal_b_obj=(0.0, 1.0, 0.0),
        jaw_width=0.04,
        roll_angle_rad=0.0,
    )


class GraspScoringTests(unittest.TestCase):
    def test_legacy_support_tuning_kwargs_are_not_public_api(self) -> None:
        parameters = inspect.signature(score_grasps).parameters

        self.assertNotIn("support_target", parameters)
        self.assertNotIn("contact_radius_m", parameters)

    def test_contact_support_scores_pad_footprint_not_nearby_vertices(self) -> None:
        mesh = _make_box_mesh((0.10, 0.04, 0.10))

        scored = {
            candidate.grasp_id: candidate
            for candidate in score_grasps(
                [
                    _side_face_grasp("center", x=0.0),
                    _side_face_grasp("near_edge", x=0.048),
                ],
                mesh_local=mesh,
            )
        }

        center_components = scored["center"].score_components or {}
        edge_components = scored["near_edge"].score_components or {}

        self.assertAlmostEqual(center_components["contact_support"], 1.0)
        self.assertGreater(center_components["pad_support_fraction_left"], 0.99)
        self.assertGreater(center_components["pad_support_fraction_right"], 0.99)
        self.assertLess(edge_components["contact_support"], center_components["contact_support"])
        self.assertLess(edge_components["pad_support_fraction_left"], center_components["pad_support_fraction_left"])
        self.assertLess(edge_components["pad_support_fraction_right"], center_components["pad_support_fraction_right"])


if __name__ == "__main__":
    unittest.main()
