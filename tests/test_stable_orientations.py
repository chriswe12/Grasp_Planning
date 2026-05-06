from __future__ import annotations

import unittest

import numpy as np

from grasp_planning.grasping import TriangleMesh
from grasp_planning.pipeline.stable_orientations import (
    StableOrientationConfig,
    enumerate_stable_orientations,
    stable_orientation_result_payload,
)


def _make_box_mesh(size_xyz: tuple[float, float, float]) -> TriangleMesh:
    sx, sy, sz = (0.5 * float(value) for value in size_xyz)
    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
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


def _make_open_tetra_mesh() -> TriangleMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.0, 0.06, 0.0],
            [0.0, 0.0, 0.06],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices_obj=vertices, faces=faces)


class StableOrientationTests(unittest.TestCase):
    def test_cube_has_six_robust_stable_orientations(self) -> None:
        mesh = _make_box_mesh((0.05, 0.05, 0.05))

        result = enumerate_stable_orientations(
            mesh,
            StableOrientationConfig(
                robust_tilt_deg=5.0,
                min_support_area_m2=0.0,
                min_support_area_fraction=0.0,
            ),
        )

        self.assertEqual(len(result.orientations), 6)
        self.assertEqual(result.raw_facet_count, 6)
        self.assertEqual(result.com_method, "volume")
        self.assertEqual([orientation.orientation_id for orientation in result.orientations], [f"orientation_{i:03d}" for i in range(6)])
        for orientation in result.orientations:
            self.assertGreaterEqual(orientation.max_stable_tilt_deg, 44.0)
            vertices_world = orientation.object_pose_world.transform_points_to_world(mesh.vertices_obj)
            self.assertAlmostEqual(float(vertices_world[:, 2].min()), 0.0, places=7)

    def test_rectangular_box_reports_expected_margin_and_height(self) -> None:
        mesh = _make_box_mesh((0.10, 0.05, 0.02))

        result = enumerate_stable_orientations(
            mesh,
            StableOrientationConfig(
                robust_tilt_deg=5.0,
                min_support_area_m2=0.0,
                min_support_area_fraction=0.0,
            ),
        )

        self.assertEqual(len(result.orientations), 6)
        largest = result.orientations[0]
        self.assertAlmostEqual(largest.area_m2, 0.005, places=7)
        self.assertAlmostEqual(largest.com_height_m, 0.01, places=7)
        self.assertAlmostEqual(largest.stability_margin_m, 0.025, places=7)
        self.assertGreater(largest.max_stable_tilt_deg, 68.0)

    def test_tiny_facets_are_filtered_by_area_threshold(self) -> None:
        mesh = _make_box_mesh((0.05, 0.05, 0.05))

        result = enumerate_stable_orientations(
            mesh,
            StableOrientationConfig(
                robust_tilt_deg=5.0,
                min_support_area_m2=1.0,
                min_support_area_fraction=0.0,
            ),
        )

        self.assertEqual(len(result.orientations), 0)
        self.assertTrue(result.rejected_candidates)
        self.assertTrue(
            all(candidate.rejection_reason == "support_area_too_small" for candidate in result.rejected_candidates)
        )

    def test_non_volume_mesh_uses_surface_centroid_fallback(self) -> None:
        mesh = _make_open_tetra_mesh()

        result = enumerate_stable_orientations(
            mesh,
            StableOrientationConfig(
                robust_tilt_deg=5.0,
                min_support_area_m2=0.0,
                min_support_area_fraction=0.0,
            ),
        )

        self.assertEqual(result.com_method, "surface_centroid_fallback")
        self.assertTrue(result.orientations)
        self.assertTrue(all(orientation.com_method == "surface_centroid_fallback" for orientation in result.orientations))

    def test_robust_tilt_threshold_rejects_marginal_supports(self) -> None:
        mesh = _make_box_mesh((0.05, 0.05, 0.05))

        result = enumerate_stable_orientations(
            mesh,
            StableOrientationConfig(
                robust_tilt_deg=50.0,
                min_support_area_m2=0.0,
                min_support_area_fraction=0.0,
            ),
        )

        self.assertEqual(len(result.orientations), 0)
        self.assertTrue(result.rejected_candidates)
        self.assertTrue(
            all(candidate.rejection_reason == "tilt_margin_too_small" for candidate in result.rejected_candidates)
        )

    def test_result_payload_contains_orientation_and_rejection_details(self) -> None:
        mesh = _make_box_mesh((0.05, 0.05, 0.05))
        result = enumerate_stable_orientations(mesh, StableOrientationConfig(min_support_area_m2=1.0))

        payload = stable_orientation_result_payload(result)

        self.assertEqual(payload["stable_orientation_count"], 0)
        self.assertEqual(payload["rejected_candidate_count"], 6)
        self.assertIn("rejected_candidates", payload)


if __name__ == "__main__":
    unittest.main()
