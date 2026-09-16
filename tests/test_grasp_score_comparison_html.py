from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_planning.grasping import SavedGraspCandidate, TriangleMesh
from scripts.write_grasp_score_comparison_html import (
    _ordered_records,
    build_score_comparison_payload,
    write_score_comparison_html,
)


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


def _side_face_grasp(grasp_id: str, *, x: float) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=(x, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(x, -0.02, 0.0),
        contact_point_b_obj=(x, 0.02, 0.0),
        contact_normal_a_obj=(0.0, -1.0, 0.0),
        contact_normal_b_obj=(0.0, 1.0, 0.0),
        jaw_width=0.04,
        roll_angle_rad=0.0,
    )


class GraspScoreComparisonHtmlTests(unittest.TestCase):
    def test_payload_contains_old_new_scores_and_html_writes(self) -> None:
        payload = build_score_comparison_payload(
            mesh_local=_make_box_mesh((0.10, 0.04, 0.10)),
            candidates=[
                _side_face_grasp("center", x=0.0),
                _side_face_grasp("near_edge", x=0.048),
            ],
            source_label="unit-test",
            max_display=10,
            sort_by="score_loss",
        )

        self.assertEqual(payload["candidate_count"], 2)
        records = {record["grasp_id"]: record for record in payload["candidates"]}  # type: ignore[index]
        self.assertIn("old_components", records["center"])
        self.assertIn("new_components", records["center"])
        self.assertIn("franka_hand_vertices_obj", records["center"])
        self.assertIn("franka_left_contact_grid_obj", records["center"])
        self.assertLess(records["near_edge"]["new_contact_support"], records["center"]["new_contact_support"])

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "comparison.html"
            write_score_comparison_html(output, payload)
            html = output.read_text(encoding="utf-8")

        self.assertIn("Grasp Score Comparison", html)
        self.assertIn("old_top", html)
        self.assertIn("Solid Mesh", html)
        self.assertIn("Accepted Only", html)
        self.assertIn("Gripper mesh", html)
        self.assertIn("Most score lost", html)
        self.assertNotIn("New #1, old #1", html)

    def test_score_loss_order_puts_most_negative_delta_first(self) -> None:
        records = [
            {"grasp_id": "a", "score_delta": 0.2},
            {"grasp_id": "b", "score_delta": -0.4},
            {"grasp_id": "c", "score_delta": -0.1},
        ]

        ordered = _ordered_records(records, "score_loss")

        self.assertEqual([record["grasp_id"] for record in ordered], ["b", "c", "a"])


if __name__ == "__main__":
    unittest.main()
