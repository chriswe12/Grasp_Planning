from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from scripts import precompute_fabrica_symmetries as symmetries


class FabricaSymmetryPrecomputeTests(unittest.TestCase):
    def test_detects_box_half_turns_and_rejects_wrong_quarter_turn(self) -> None:
        mesh = trimesh.creation.box(extents=(0.04, 0.10, 0.16))
        mesh.apply_translation((0.03, -0.02, 0.05))
        config = symmetries.DetectionConfig(
            mesh_scale=1.0,
            tolerance_m=0.001,
            sample_count=3_000,
            visual_sample_count=64,
            orders=(2, 4),
            max_candidate_axes=18,
            max_face_axes=4,
            near_miss_count=4,
        )

        result = symmetries.detect_mesh_symmetries(
            mesh,
            assembly="test_assembly",
            part_id="0",
            mesh_path="obj/fabrica/test_assembly/0.obj",
            config=config,
        )

        accepted = result["symmetries"]
        self.assertGreaterEqual(len(accepted), 4)
        self.assertEqual(accepted[0]["name"], "identity")
        self.assertEqual(accepted[0]["description"], "Identity")
        self.assertTrue(
            any("180° about +X" in record["description"] for record in accepted if record["type"] == "finite_rotation")
        )
        for axis in np.eye(3):
            self.assertTrue(
                any(
                    record["order"] == 2
                    and abs(float(np.dot(np.asarray(record["axis_obj"], dtype=float), axis))) > 0.999
                    for record in accepted
                    if record["type"] == "finite_rotation"
                ),
                f"missing 180 degree symmetry around axis {axis.tolist()}",
            )
        self.assertFalse(
            any(
                record["order"] == 4
                and record["step"] in (1, 3)
                and abs(float(np.dot(np.asarray(record.get("axis_obj", [0.0, 0.0, 0.0]), dtype=float), [0, 0, 1])))
                > 0.999
                for record in accepted
            )
        )
        self.assertEqual(result["pose_equivalence"], "T_world_object_equivalent = T_world_object @ matrix_obj")
        self.assertNotEqual(accepted[1]["translation_obj_m"], [0.0, 0.0, 0.0])
        self.assertEqual(accepted[1]["validation"]["validation_mode"], "mesh_connectivity")

    def test_mesh_connectivity_validation_rejects_vertex_only_match_with_wrong_faces(self) -> None:
        vertices = np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        candidate = symmetries.CandidateTransform(
            name="quarter_turn",
            source="test",
            matrix=symmetries._transform_about_center(
                np.asarray([0.0, 0.0, 1.0], dtype=float),
                np.asarray([0.0, 0.0, 0.0], dtype=float),
                np.pi / 2.0,
            ),
            axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
            center=np.asarray([0.0, 0.0, 0.0], dtype=float),
            angle_deg=90.0,
            order=4,
            step=1,
        )
        context = symmetries._build_mesh_connectivity_context(
            vertices,
            faces,
            tolerance_m=1.0e-6,
            max_probe_vertices=16,
        )

        validation = symmetries._evaluate_mesh_transform(
            candidate,
            mesh_context=context,
            tolerance_m=1.0e-6,
        )

        self.assertFalse(validation["accepted"])
        self.assertEqual(validation["mesh_reject_reason"], "face_connectivity")
        self.assertGreater(validation["mesh_missing_faces"], 0)
        self.assertEqual(validation["mesh_unique_vertex_matches"], 4)

    def test_geometric_validation_can_skip_vertex_prefilter(self) -> None:
        mesh = trimesh.creation.box(extents=(0.04, 0.10, 0.16))
        config = symmetries.DetectionConfig(
            mesh_scale=1.0,
            tolerance_m=0.001,
            validation_mode="geometric",
            skip_geometric_vertex_prefilter=True,
            sample_count=500,
            visual_sample_count=16,
            orders=(2,),
            max_candidate_axes=3,
            max_face_axes=0,
            near_miss_count=2,
        )

        result = symmetries.detect_mesh_symmetries(
            mesh,
            assembly="test_assembly",
            part_id="0",
            mesh_path="obj/fabrica/test_assembly/0.obj",
            config=config,
        )

        self.assertTrue(result["candidate_summary"]["skip_geometric_vertex_prefilter"])
        self.assertTrue(
            any(
                record["validation"]["validation_mode"] == "geometric_surface_no_vertex_prefilter"
                for record in result["symmetries"]
            )
        )

    def test_asset_payload_omits_visual_mesh_but_keeps_validation_summary(self) -> None:
        mesh = trimesh.creation.box(extents=(0.04, 0.10, 0.16))
        config = symmetries.DetectionConfig(
            mesh_scale=1.0,
            tolerance_m=0.001,
            sample_count=500,
            visual_sample_count=16,
            max_visual_faces=4,
            max_visual_edges=5,
            orders=(2,),
            max_candidate_axes=3,
            max_face_axes=0,
            near_miss_count=2,
        )
        result = symmetries.detect_mesh_symmetries(
            mesh,
            assembly="test_assembly",
            part_id="0",
            mesh_path="obj/fabrica/test_assembly/0.obj",
            config=config,
        )

        payload = symmetries._asset_payload("test_assembly", [result], config)
        part_payload = payload["parts"]["0"]

        self.assertIn("symmetries", part_payload)
        self.assertIn("candidate_summary", part_payload)
        self.assertNotIn("visual_points_obj", part_payload)
        self.assertNotIn("visual_mesh_vertices_obj", part_payload)
        self.assertNotIn("visual_mesh_faces", part_payload)
        self.assertIn("visual_mesh_vertices_obj", result)
        self.assertIn("visual_mesh_faces", result)
        self.assertLessEqual(len(result["visual_mesh_faces"]), 4)
        self.assertLessEqual(len(result["visual_mesh_edges"]), 5)
        self.assertEqual(result["visual_mesh_original_faces"], len(mesh.faces))
        self.assertEqual(result["candidate_summary"]["visual_faces"], len(result["visual_mesh_faces"]))
        self.assertEqual(result["candidate_summary"]["visual_edges"], len(result["visual_mesh_edges"]))
        self.assertEqual(payload["frame"], "object")

    def test_writes_visual_report_html_and_json_data_is_inspectable(self) -> None:
        mesh = trimesh.creation.box(extents=(0.04, 0.04, 0.08))
        config = symmetries.DetectionConfig(
            mesh_scale=1.0,
            tolerance_m=0.001,
            sample_count=500,
            visual_sample_count=16,
            orders=(2,),
            max_candidate_axes=3,
            max_face_axes=0,
            near_miss_count=2,
        )
        result = symmetries.detect_mesh_symmetries(
            mesh,
            assembly="test_assembly",
            part_id="0",
            mesh_path="obj/fabrica/test_assembly/0.obj",
            config=config,
        )
        report = {
            "schema_version": 1,
            "generated_by": "scripts/precompute_fabrica_symmetries.py",
            "asset_root": "assets/obj/fabrica",
            "config": {"mesh_scale": 1.0},
            "assemblies": ["test_assembly"],
            "parts": [result],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_html = Path(temp_dir) / "symmetries.html"
            symmetries.write_symmetry_report_html(output_html, report)
            html = output_html.read_text(encoding="utf-8")

        self.assertIn("Fabrica Symmetry Inspection", html)
        self.assertIn("Near Misses", html)
        self.assertIn('<canvas id="scene"', html)
        self.assertIn("progressSlider", html)
        self.assertIn("drawArrowLine", html)
        self.assertIn("drawMesh", html)
        self.assertIn("requestAnimationFrame", html)
        self.assertIn("visual faces:", html)
        self.assertIn("transform.description", html)
        self.assertIn(json.dumps(report, separators=(",", ":"))[:80], html)


if __name__ == "__main__":
    unittest.main()
