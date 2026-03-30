from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grasp_planning.grasping import (
    AntipodalGraspGeneratorConfig,
    AntipodalMeshGraspGenerator,
    TriangleMesh,
    export_grasp_candidates_json,
)


def _make_cube_mesh(side_length: float) -> TriangleMesh:
    half = 0.5 * float(side_length)
    vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
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


def _make_cylinder_mesh(radius: float, height: float, radial_segments: int) -> TriangleMesh:
    if radial_segments < 3:
        raise ValueError("radial_segments must be at least 3.")
    half_height = 0.5 * float(height)
    angles = np.linspace(0.0, 2.0 * np.pi, num=radial_segments, endpoint=False)

    vertices: list[list[float]] = []
    for z in (-half_height, half_height):
        for angle in angles:
            vertices.append([float(radius * np.cos(angle)), float(radius * np.sin(angle)), z])
    bottom_center_index = len(vertices)
    vertices.append([0.0, 0.0, -half_height])
    top_center_index = len(vertices)
    vertices.append([0.0, 0.0, half_height])

    faces: list[list[int]] = []
    for idx in range(radial_segments):
        next_idx = (idx + 1) % radial_segments
        bottom_a = idx
        bottom_b = next_idx
        top_a = radial_segments + idx
        top_b = radial_segments + next_idx
        faces.append([bottom_a, bottom_b, top_b])
        faces.append([bottom_a, top_b, top_a])
        faces.append([bottom_center_index, bottom_b, bottom_a])
        faces.append([top_center_index, top_a, top_b])

    return TriangleMesh(vertices_obj=np.array(vertices, dtype=float), faces=np.array(faces, dtype=np.int64))


class AntipodalMeshGraspGeneratorTests(unittest.TestCase):
    def test_default_config_generates_candidates_for_cube_and_cylinder(self) -> None:
        cube_candidates = AntipodalMeshGraspGenerator().generate(_make_cube_mesh(side_length=0.05))
        cylinder_candidates = AntipodalMeshGraspGenerator().generate(
            _make_cylinder_mesh(radius=0.02, height=0.05, radial_segments=24)
        )

        self.assertTrue(cube_candidates)
        self.assertTrue(cylinder_candidates)

    def test_cube_generation_returns_object_frame_antipodal_grasps(self) -> None:
        cube_mesh = _make_cube_mesh(side_length=0.05)
        generator = AntipodalMeshGraspGenerator(
            AntipodalGraspGeneratorConfig(
                num_surface_samples=192,
                min_jaw_width=0.03,
                max_jaw_width=0.06,
                antipodal_cosine_threshold=0.98,
                finger_depth=0.01,
                finger_length=0.012,
                finger_thickness=0.012,
                contact_patch_radius=0.006,
                collision_sample_count=256,
                rng_seed=7,
            )
        )

        candidates = generator.generate(cube_mesh)

        self.assertTrue(candidates)
        for candidate in candidates:
            point_a = np.asarray(candidate.contact_point_a_obj, dtype=float)
            point_b = np.asarray(candidate.contact_point_b_obj, dtype=float)
            center = np.asarray(candidate.grasp_position_obj, dtype=float)
            normal_a = np.asarray(candidate.contact_normal_a_obj, dtype=float)
            normal_b = np.asarray(candidate.contact_normal_b_obj, dtype=float)
            closing_axis = (point_b - point_a) / np.linalg.norm(point_b - point_a)
            quat = np.asarray(candidate.grasp_orientation_xyzw_obj, dtype=float)

            np.testing.assert_allclose(center, 0.5 * (point_a + point_b), atol=1e-6)
            self.assertGreaterEqual(candidate.jaw_width, 0.03)
            self.assertLessEqual(candidate.jaw_width, 0.06)
            self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=6)
            self.assertGreaterEqual(float(np.dot(-normal_a, closing_axis)), 0.98)
            self.assertGreaterEqual(float(np.dot(normal_b, closing_axis)), 0.98)
            self.assertLessEqual(float(np.dot(normal_a, normal_b)), -0.98)

    def test_cube_roll_enumeration_preserves_contacts_and_width(self) -> None:
        cube_mesh = _make_cube_mesh(side_length=0.05)
        base_config = AntipodalGraspGeneratorConfig(
            num_surface_samples=128,
            min_jaw_width=0.03,
            max_jaw_width=0.06,
            antipodal_cosine_threshold=0.98,
            finger_depth=0.01,
            finger_length=0.012,
            finger_thickness=0.012,
            contact_patch_radius=0.006,
            collision_sample_count=128,
            rng_seed=3,
        )
        rolled_config = AntipodalGraspGeneratorConfig(
            **{
                **base_config.__dict__,
                "roll_angles_rad": (0.0, 0.5 * np.pi),
            }
        )

        base_candidates = AntipodalMeshGraspGenerator(base_config).generate(cube_mesh)
        rolled_candidates = AntipodalMeshGraspGenerator(rolled_config).generate(cube_mesh)

        self.assertTrue(base_candidates)
        self.assertEqual(len(rolled_candidates), 2 * len(base_candidates))

        base_groups = {
            tuple(np.round(np.array([*candidate.contact_point_a_obj, *candidate.contact_point_b_obj, candidate.jaw_width]), 6))
            for candidate in base_candidates
        }
        rolled_groups = {
            tuple(np.round(np.array([*candidate.contact_point_a_obj, *candidate.contact_point_b_obj, candidate.jaw_width]), 6))
            for candidate in rolled_candidates
        }
        self.assertSetEqual(base_groups, rolled_groups)

    def test_cylinder_generation_finds_side_grasps(self) -> None:
        cylinder_mesh = _make_cylinder_mesh(radius=0.02, height=0.05, radial_segments=24)
        generator = AntipodalMeshGraspGenerator(
            AntipodalGraspGeneratorConfig(
                num_surface_samples=256,
                min_jaw_width=0.03,
                max_jaw_width=0.045,
                antipodal_cosine_threshold=0.93,
                finger_depth=0.008,
                finger_length=0.012,
                finger_thickness=0.01,
                contact_patch_radius=0.006,
                collision_sample_count=256,
                rng_seed=11,
            )
        )

        candidates = generator.generate(cylinder_mesh)

        self.assertTrue(candidates)
        side_candidates = [
            candidate
            for candidate in candidates
            if abs(candidate.contact_point_a_obj[2]) < 0.02 and abs(candidate.contact_point_b_obj[2]) < 0.02
        ]
        self.assertTrue(side_candidates)
        for candidate in side_candidates:
            self.assertGreaterEqual(candidate.jaw_width, 0.03)
            self.assertLessEqual(candidate.jaw_width, 0.045)
            center = np.asarray(candidate.grasp_position_obj, dtype=float)
            self.assertLess(abs(float(center[2])), 0.02)

    def test_large_fingers_can_reject_cube_grasps(self) -> None:
        cube_mesh = _make_cube_mesh(side_length=0.05)
        generator = AntipodalMeshGraspGenerator(
            AntipodalGraspGeneratorConfig(
                num_surface_samples=192,
                min_jaw_width=0.03,
                max_jaw_width=0.06,
                antipodal_cosine_threshold=0.98,
                finger_depth=0.01,
                finger_length=0.08,
                finger_thickness=0.08,
                contact_patch_radius=0.004,
                collision_sample_count=512,
                rng_seed=7,
            )
        )

        candidates = generator.generate(cube_mesh)

        self.assertEqual(candidates, [])

    def test_export_writes_expected_fields(self) -> None:
        cube_mesh = _make_cube_mesh(side_length=0.05)
        generator = AntipodalMeshGraspGenerator(
            AntipodalGraspGeneratorConfig(
                num_surface_samples=64,
                min_jaw_width=0.03,
                max_jaw_width=0.06,
                antipodal_cosine_threshold=0.98,
                finger_depth=0.01,
                finger_length=0.012,
                finger_thickness=0.012,
                contact_patch_radius=0.006,
                collision_sample_count=128,
                rng_seed=19,
            )
        )
        candidates = generator.generate(cube_mesh)
        self.assertTrue(candidates)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "grasps.json"
            export_grasp_candidates_json(candidates[:2], output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(payload), 2)
        first = payload[0]
        self.assertEqual(sorted(first.keys()), ["contact_normals_obj", "contact_points_obj", "grasp_pose_obj", "jaw_width", "roll_angle_rad"])
        self.assertEqual(sorted(first["grasp_pose_obj"].keys()), ["orientation_xyzw", "position"])
        self.assertEqual(len(first["contact_points_obj"]), 2)
        self.assertEqual(len(first["contact_normals_obj"]), 2)


if __name__ == "__main__":
    unittest.main()
