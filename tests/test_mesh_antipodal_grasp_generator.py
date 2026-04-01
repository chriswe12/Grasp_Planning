from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grasp_planning.grasping import (
    AntipodalGraspGeneratorConfig,
    AntipodalMeshGraspGenerator,
    FingerBoxGripperCollisionModel,
    FingerBoxWithHandMeshCollisionModel,
    FrankaHandFingerCollisionModel,
    GraspCollisionEvaluator,
    TriangleMesh,
    export_grasp_candidates_json,
    finger_boxes_from_grasp,
)
from grasp_planning.grasping.collision import CollisionManager, trimesh
from scripts.debug_mesh_antipodal_grasps import _config_from_sources, parser


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


@unittest.skipIf(trimesh is None or CollisionManager is None, "trimesh/FCL collision backend is required")
class AntipodalMeshGraspGeneratorTests(unittest.TestCase):
    def test_collision_scene_is_prepared_once_per_mesh_generation(self) -> None:
        class _Scene:
            def intersects_box(self, primitive) -> bool:
                return False

            def intersects_mesh(self, primitive) -> bool:
                return False

        class _Backend:
            backend_name = "test_backend"

            def __init__(self) -> None:
                self.build_count = 0

            def build_scene(self, mesh) -> _Scene:
                self.build_count += 1
                return _Scene()

        cube_mesh = _make_cube_mesh(side_length=0.05)
        backend = _Backend()
        generator = AntipodalMeshGraspGenerator(
            AntipodalGraspGeneratorConfig(
                num_surface_samples=64,
                min_jaw_width=0.03,
                max_jaw_width=0.06,
                antipodal_cosine_threshold=0.98,
                rng_seed=5,
            )
        )
        generator._collision_evaluator = GraspCollisionEvaluator(  # type: ignore[attr-defined]
            FingerBoxGripperCollisionModel(
                finger_extent_lateral=generator._config.finger_extent_lateral,  # type: ignore[attr-defined]
                finger_extent_closing=generator._config.finger_extent_closing,  # type: ignore[attr-defined]
                finger_extent_approach=generator._config.finger_extent_approach,  # type: ignore[attr-defined]
                finger_clearance=generator._config.finger_clearance,  # type: ignore[attr-defined]
            ),
            backend=backend,
        )

        generator.generate(cube_mesh)

        self.assertEqual(backend.build_count, 1)

    def test_config_accepts_legacy_finger_dimension_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "legacy.yaml"
            config_path.write_text(
                """
geometry:
  type: cube
generator:
  finger_depth: 0.021
  finger_length: 0.022
  finger_thickness: 0.023
""".strip(),
                encoding="utf-8",
            )

            args = parser.parse_args(["--config", str(config_path)])
            resolved = _config_from_sources(args)

        self.assertAlmostEqual(resolved.finger_extent_lateral, 0.021)
        self.assertAlmostEqual(resolved.finger_extent_closing, 0.022)
        self.assertAlmostEqual(resolved.finger_extent_approach, 0.023)

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
                finger_extent_lateral=0.01,
                finger_extent_closing=0.012,
                finger_extent_approach=0.012,
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
            finger_extent_lateral=0.01,
            finger_extent_closing=0.012,
            finger_extent_approach=0.012,
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
        self.assertGreaterEqual(len(rolled_candidates), len(base_candidates))

        def group_key(candidate: object) -> tuple[float, ...]:
            return tuple(
                np.round(np.array([*candidate.contact_point_a_obj, *candidate.contact_point_b_obj, candidate.jaw_width]), 6)
            )

        base_groups = {group_key(candidate) for candidate in base_candidates}
        rolled_group_counts: dict[tuple[float, ...], int] = {}
        for candidate in rolled_candidates:
            key = group_key(candidate)
            rolled_group_counts[key] = rolled_group_counts.get(key, 0) + 1

        self.assertTrue(base_groups.issubset(set(rolled_group_counts)))
        for count in rolled_group_counts.values():
            self.assertIn(count, {1, 2})

    def test_cylinder_generation_finds_side_grasps(self) -> None:
        cylinder_mesh = _make_cylinder_mesh(radius=0.02, height=0.05, radial_segments=24)
        generator = AntipodalMeshGraspGenerator(
            AntipodalGraspGeneratorConfig(
                num_surface_samples=256,
                min_jaw_width=0.03,
                max_jaw_width=0.045,
                antipodal_cosine_threshold=0.93,
                finger_extent_lateral=0.008,
                finger_extent_closing=0.012,
                finger_extent_approach=0.01,
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
                finger_extent_lateral=0.08,
                finger_extent_closing=0.012,
                finger_extent_approach=0.04,
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
                finger_extent_lateral=0.01,
                finger_extent_closing=0.012,
                finger_extent_approach=0.012,
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

    def test_finger_boxes_follow_the_grasp_pose_frame(self) -> None:
        point_a = np.array([0.0, -0.02, 0.08], dtype=float)
        point_b = np.array([0.0, 0.02, 0.08], dtype=float)
        rotation = np.eye(3, dtype=float)

        box_a, box_b = finger_boxes_from_grasp(
            grasp_rotmat=rotation,
            contact_point_a=point_a,
            contact_point_b=point_b,
            finger_extent_lateral=0.01,
            finger_extent_closing=0.02,
            finger_extent_approach=0.004,
            finger_clearance=0.002,
        )

        expected_offset = np.array([0.0, 0.011, 0.0], dtype=float)
        np.testing.assert_allclose(box_a[0], point_a - expected_offset, atol=1e-8)
        np.testing.assert_allclose(box_b[0], point_b + expected_offset, atol=1e-8)
        np.testing.assert_allclose(box_a[1], rotation, atol=1e-8)
        np.testing.assert_allclose(box_b[1], rotation, atol=1e-8)

    def test_clearance_is_checked_per_roll_pose(self) -> None:
        mesh = TriangleMesh(
            vertices_obj=np.array(
                [
                    [0.004, -0.031, -0.001],
                    [0.004, -0.029, -0.001],
                    [0.004, -0.029, 0.001],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        evaluator = GraspCollisionEvaluator(
            FingerBoxGripperCollisionModel(
                finger_extent_lateral=0.01,
                finger_extent_closing=0.02,
                finger_extent_approach=0.004,
                finger_clearance=0.002,
            )
        )
        point_a = np.array([0.0, -0.02, 0.0], dtype=float)
        point_b = np.array([0.0, 0.02, 0.0], dtype=float)
        identity = np.eye(3, dtype=float)
        ninety_deg_roll = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        self.assertFalse(
            evaluator.is_grasp_collision_free(
                scene=evaluator.build_scene(mesh),
                grasp_rotmat=identity,
                contact_point_a=point_a,
                contact_point_b=point_b,
            )
        )
        self.assertTrue(
            evaluator.is_grasp_collision_free(
                scene=evaluator.build_scene(mesh),
                grasp_rotmat=ninety_deg_roll,
                contact_point_a=point_a,
                contact_point_b=point_b,
            )
        )

    def test_hand_mesh_collision_rejects_grasp(self) -> None:
        mesh = _make_cube_mesh(side_length=0.08)
        evaluator = GraspCollisionEvaluator(
            FingerBoxWithHandMeshCollisionModel(
                finger_extent_lateral=0.01,
                finger_extent_closing=0.02,
                finger_extent_approach=0.004,
                finger_clearance=0.002,
            )
        )
        point_a = np.array([0.0, -0.02, 0.0], dtype=float)
        point_b = np.array([0.0, 0.02, 0.0], dtype=float)

        self.assertFalse(
            evaluator.is_grasp_collision_free(
                scene=evaluator.build_scene(mesh),
                grasp_rotmat=np.eye(3, dtype=float),
                contact_point_a=point_a,
                contact_point_b=point_b,
            )
        )


if __name__ == "__main__":
    unittest.main()
