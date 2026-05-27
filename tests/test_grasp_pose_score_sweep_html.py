from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_planning.grasping import SavedGraspCandidate, TriangleMesh
from grasp_planning.grasping.fabrica_grasp_debug import rotmat_to_quat_xyzw
from grasp_planning.pipeline import PlanningConfig
from scripts.write_grasp_pose_score_sweep_html import (
    build_floor_pose_records,
    build_pose_score_sweep_payload,
    write_pose_score_sweep_html,
)


def _box_mesh(size_xyz: tuple[float, float, float]) -> TriangleMesh:
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


def _orientation_with_approach_axis(axis_obj: tuple[float, float, float]) -> tuple[float, float, float, float]:
    approach = np.asarray(axis_obj, dtype=float)
    approach /= np.linalg.norm(approach)
    x_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    y_axis = np.cross(approach, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return rotmat_to_quat_xyzw(np.column_stack((x_axis, y_axis, approach)))


def _x_side_candidate(grasp_id: str, approach_axis: tuple[float, float, float]) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=(0.0, 0.0, 0.0),
        grasp_orientation_xyzw_obj=_orientation_with_approach_axis(approach_axis),
        contact_point_a_obj=(-0.02, 0.0, 0.0),
        contact_point_b_obj=(0.02, 0.0, 0.0),
        contact_normal_a_obj=(-1.0, 0.0, 0.0),
        contact_normal_b_obj=(1.0, 0.0, 0.0),
        jaw_width=0.04,
        roll_angle_rad=0.0,
    )


class GraspPoseScoreSweepHtmlTests(unittest.TestCase):
    def test_floor_pose_records_place_mesh_on_floor(self) -> None:
        mesh = _box_mesh((0.04, 0.06, 0.08))

        records = build_floor_pose_records(
            mesh_local=mesh,
            support_faces=("neg_z",),
            yaw_deg_values=(0.0, 90.0),
            x_values=(0.25,),
            y_values=(0.0,),
        )

        self.assertEqual(len(records), 2)
        for record in records:
            vertices_world = record.object_pose_world.transform_points_to_world(mesh.vertices_obj)
            self.assertAlmostEqual(float(vertices_world[:, 2].min()), 0.0, places=9)

    def test_payload_contains_pose_scores_and_html_controls(self) -> None:
        mesh = _box_mesh((0.04, 0.04, 0.04))
        candidates = [
            _x_side_candidate("near_side", (1.0, 0.0, 0.0)),
            _x_side_candidate("far_side", (-1.0, 0.0, 0.0)),
        ]
        poses = build_floor_pose_records(
            mesh_local=mesh,
            support_faces=("neg_z",),
            yaw_deg_values=(0.0,),
            x_values=(0.25, 0.70),
            y_values=(0.0,),
        )

        payload = build_pose_score_sweep_payload(
            mesh_local=mesh,
            candidates=candidates,
            source_label="unit-test",
            planning=PlanningConfig(
                floor_clearance_margin_m=-1.0,
                top_grasp_score_weight=0.0,
                reachability_proxy_score_weight=1.0,
                reachability_proxy_hand_offset_m=0.10,
            ),
            pose_records=poses,
            max_candidates=10,
            max_display=10,
            top_per_pose=2,
        )

        self.assertEqual(payload["pose_count"], 2)
        self.assertEqual(payload["display_count"], 2)
        first = payload["candidates"][0]  # type: ignore[index]
        self.assertEqual(len(first["pose_scores"]), 2)
        self.assertIn("object_score", first)
        self.assertIn("franka_hand_vertices_obj", first)
        self.assertIn("reachability_proxy", first["pose_scores"][0])

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pose_sweep.html"
            write_pose_score_sweep_html(output, payload)
            html = output.read_text(encoding="utf-8")

        self.assertIn("Grasp Runtime Pose Score Sweep", html)
        self.assertIn("Prev Pose", html)
        self.assertIn("xSlider", html)
        self.assertIn("objectYawSlider", html)
        self.assertIn("handRollSlider", html)
        self.assertIn("Live scoring sliders", html)
        self.assertIn("score trace", html)
        self.assertIn("Robot base", html)
        self.assertIn("reachability_proxy", html)


if __name__ == "__main__":
    unittest.main()
