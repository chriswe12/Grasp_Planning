from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import quat_to_rotmat_xyzw
from grasp_planning.grasping.gpd_grasp_generator import (
    ExternalGpdGraspGenerator,
    GpdGraspGeneratorConfig,
    sample_mesh_surface,
    write_ascii_pcd,
    write_normals_csv,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh


def _make_contact_test_mesh() -> TriangleMesh:
    vertices = np.array(
        [
            [-0.02, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.0, 0.0, 0.02],
        ],
        dtype=float,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    return TriangleMesh(vertices_obj=vertices, faces=faces)


class GpdPointCloudArtifactTests(unittest.TestCase):
    def test_sampling_and_ascii_artifacts_preserve_point_normal_counts(self) -> None:
        mesh = _make_contact_test_mesh()
        samples = sample_mesh_surface(mesh, num_samples=7, rng_seed=13)

        with tempfile.TemporaryDirectory() as temp_dir:
            pcd_path = Path(temp_dir) / "cloud.pcd"
            normals_path = Path(temp_dir) / "normals.csv"
            write_ascii_pcd(pcd_path, samples)
            write_normals_csv(normals_path, samples)

            pcd_text = pcd_path.read_text(encoding="utf-8")
            normals_text = normals_path.read_text(encoding="utf-8")

        self.assertEqual(len(samples), 7)
        self.assertIn("FIELDS x y z", pcd_text)
        self.assertIn("POINTS 7", pcd_text)
        self.assertEqual(len([line for line in normals_text.splitlines() if line.strip()]), 7)


class ExternalGpdGraspGeneratorTests(unittest.TestCase):
    def test_external_json_grasps_are_converted_to_pipeline_frame(self) -> None:
        mesh = _make_contact_test_mesh()
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "fake_gpd.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import json, sys",
                        "out = sys.argv[sys.argv.index('--out') + 1]",
                        "payload = {'grasps': [{",
                        "  'position': [0.0, 0.0, 0.0],",
                        "  'approach': [0.0, 0.0, 1.0],",
                        "  'binormal': [1.0, 0.0, 0.0],",
                        "  'axis': [0.0, 1.0, 0.0],",
                        "  'width': 0.04,",
                        "  'score': 0.91,",
                        "}]}",
                        "open(out, 'w', encoding='utf-8').write(json.dumps(payload))",
                    ]
                ),
                encoding="utf-8",
            )
            generator = ExternalGpdGraspGenerator(
                GpdGraspGeneratorConfig(
                    command_template=f"{sys.executable} {script_path} --pcd {{pcd}} --normals {{normals}} --out {{output_json}}",
                    num_pointcloud_samples=9,
                    check_target_collision=False,
                )
            )

            candidates = generator.generate(mesh)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        rotmat = quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)
        np.testing.assert_allclose(rotmat[:, 1], np.array([1.0, 0.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(rotmat[:, 2], np.array([0.0, 0.0, 1.0]), atol=1.0e-6)
        np.testing.assert_allclose(candidate.contact_point_a_obj, np.array([-0.02, 0.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(candidate.contact_point_b_obj, np.array([0.02, 0.0, 0.0]), atol=1.0e-6)
        self.assertAlmostEqual(candidate.jaw_width, 0.04)
        self.assertEqual(len(generator.last_surface_samples), 9)

    def test_json_stdout_is_supported_when_no_output_file_is_written(self) -> None:
        mesh = _make_contact_test_mesh()
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "fake_gpd_stdout.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import json",
                        "print(json.dumps({'grasps': [{",
                        "  'position': [0.0, 0.0, 0.0],",
                        "  'approach': [0.0, 0.0, 1.0],",
                        "  'binormal': [1.0, 0.0, 0.0],",
                        "  'axis': [0.0, 1.0, 0.0],",
                        "  'width': 0.04,",
                        "}]}))",
                    ]
                ),
                encoding="utf-8",
            )
            generator = ExternalGpdGraspGenerator(
                GpdGraspGeneratorConfig(
                    command_template=f"{sys.executable} {script_path}",
                    num_pointcloud_samples=4,
                    check_target_collision=False,
                )
            )

            candidates = generator.generate(mesh)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].jaw_width, 0.04)

    def test_payload_with_existing_bundle_candidate_shape_is_supported(self) -> None:
        mesh = _make_contact_test_mesh()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "gpd_grasps.json"
            script_path = Path(temp_dir) / "write_bundle_shape.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import json, sys",
                        f"out = {str(output_path)!r}",
                        "payload = {'candidates': [{",
                        "  'grasp_pose_obj': {'position': [0.0, 0.0, 0.0], 'orientation_xyzw': [0.0, 0.0, 0.0, 1.0]},",
                        "  'contact_points_obj': [[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]],",
                        "  'contact_normals_obj': [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],",
                        "  'jaw_width': 0.04,",
                        "}]}",
                        "open(out, 'w', encoding='utf-8').write(json.dumps(payload))",
                    ]
                ),
                encoding="utf-8",
            )
            generator = ExternalGpdGraspGenerator(
                GpdGraspGeneratorConfig(
                    command_template=f"{sys.executable} {script_path}",
                    output_json=str(output_path),
                    artifact_dir=temp_dir,
                    check_target_collision=False,
                )
            )

            candidates = generator.generate(mesh)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].contact_point_a_obj, (-0.02, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
