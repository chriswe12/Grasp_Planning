from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_planning.pipeline.assembly_sequence import (
    REPO_ROOT,
    compile_assembly_sequence,
    write_assembly_sequence_json,
)
from grasp_planning.pipeline.assembly_sequence_debug_html import write_assembly_sequence_html
from scripts import build_assembly_sequence


def _transform_for_vector(vector: tuple[float, float, float]) -> list[list[float]]:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = -np.asarray(vector, dtype=float)
    return transform.tolist()


def _write_obj(path: Path, *, min_z: float) -> None:
    path.write_text(
        "\n".join(
            [
                f"v 0 0 {min_z}",
                f"v 1 0 {min_z}",
                f"v 0 1 {min_z}",
                f"v 0 0 {min_z + 1.0}",
                "f 1 2 3",
                "f 1 4 2",
                "f 2 4 3",
                "f 3 4 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_synthetic_assets(root: Path) -> Path:
    assembly_dir = root / "synthetic"
    assembly_dir.mkdir(parents=True)
    order = ["2", "0", "3"]
    (assembly_dir / "precedence_plan.json").write_text(
        json.dumps(
            {
                "assembly": "synthetic",
                "parts": ["0", "2", "3"],
                "forward_assembly_orders": [order, ["2", "3", "0"]],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    vectors = {
        "2": (-0.03, 0.0, 0.0),
        "0": (0.0, 0.04, 0.0),
        "3": (0.0, 0.0, -0.02),
    }
    (assembly_dir / "pre_insertion_poses.json").write_text(
        json.dumps(
            {
                "assembly": "synthetic",
                "parts": {
                    part_id: {
                        "role": "moving_part",
                        "final_to_pre_insertion_transform_m": _transform_for_vector(vector),
                        "pre_to_final_insertion_vector_m": list(vector),
                        "pre_to_final_insertion_distance_m": float(np.linalg.norm(vector)),
                        "disassembly_path_waypoints": 12,
                    }
                    for part_id, vector in vectors.items()
                },
            }
        ),
        encoding="utf-8",
    )
    _write_obj(assembly_dir / "2.obj", min_z=0.0)
    _write_obj(assembly_dir / "0.obj", min_z=0.2)
    _write_obj(assembly_dir / "3.obj", min_z=1.0)
    return assembly_dir


class AssemblySequenceTests(unittest.TestCase):
    def test_explicit_late_base_override_marks_availability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assembly_dir = _write_synthetic_assets(root)
            sequence = compile_assembly_sequence(
                assembly_dir,
                base_part_id="0",
                mesh_scale=1.0,
                repo_root=root,
            )

        self.assertEqual(sequence.selected_order, ("2", "0", "3"))
        self.assertEqual(sequence.base_part_source, "explicit_override")
        self.assertEqual(sequence.base_part_order_index, 1)
        self.assertEqual(sequence.first_holder_step_index, 2)
        self.assertEqual(sequence.table_contact_part_ids, ("2",))
        self.assertEqual(sequence.steps[0].assembled_part_ids_before, ())
        self.assertEqual(sequence.steps[0].assembled_part_ids_after, ("2",))
        self.assertEqual(sequence.steps[0].base_part_status, "not_present")
        self.assertFalse(sequence.steps[0].holder_base_available)
        self.assertEqual(sequence.steps[1].base_part_status, "incoming")
        self.assertFalse(sequence.steps[1].holder_base_available)
        self.assertEqual(sequence.steps[2].base_part_status, "assembled")
        self.assertTrue(sequence.steps[2].holder_base_available)
        self.assertIn("Base-only holder planning starts at step 2", sequence.warnings[0])

    def test_defaults_base_to_first_part_of_selected_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assembly_dir = _write_synthetic_assets(root)
            sequence = compile_assembly_sequence(
                assembly_dir,
                mesh_scale=1.0,
                repo_root=root,
            )

        self.assertEqual(sequence.selected_order, ("2", "0", "3"))
        self.assertEqual(sequence.base_part_id, "2")
        self.assertEqual(sequence.base_part_source, "selected_order[0]")
        self.assertEqual(sequence.base_part_order_index, 0)
        self.assertEqual(sequence.first_holder_step_index, 1)
        self.assertEqual(sequence.steps[0].base_part_status, "incoming")
        self.assertFalse(sequence.steps[0].holder_base_available)
        self.assertEqual(sequence.steps[1].base_part_status, "assembled")
        self.assertTrue(sequence.steps[1].holder_base_available)

    def test_rejects_rotational_pre_insertion_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assembly_dir = _write_synthetic_assets(root)
            poses_path = assembly_dir / "pre_insertion_poses.json"
            payload = json.loads(poses_path.read_text(encoding="utf-8"))
            payload["parts"]["2"]["final_to_pre_insertion_transform_m"] = [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            poses_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "translation-only"):
                compile_assembly_sequence(assembly_dir, base_part_id="0", mesh_scale=1.0, repo_root=root)

    def test_rejects_inconsistent_insertion_vector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assembly_dir = _write_synthetic_assets(root)
            poses_path = assembly_dir / "pre_insertion_poses.json"
            payload = json.loads(poses_path.read_text(encoding="utf-8"))
            payload["parts"]["2"]["pre_to_final_insertion_vector_m"] = [0.03, 0.0, 0.0]
            poses_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inconsistent final-to-pre"):
                compile_assembly_sequence(assembly_dir, base_part_id="0", mesh_scale=1.0, repo_root=root)

    def test_writes_json_and_interactive_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assembly_dir = _write_synthetic_assets(root)
            sequence = compile_assembly_sequence(
                assembly_dir,
                mesh_scale=1.0,
                repo_root=root,
            )
            output_json = root / "out" / "assembly_sequence.json"
            output_html = root / "out" / "assembly_sequence.html"
            write_assembly_sequence_json(sequence, output_json)
            write_assembly_sequence_html(sequence, output_html, max_edges_per_part=8)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            html = output_html.read_text(encoding="utf-8")

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["selected_order_source"], "forward_assembly_orders[0]")
        self.assertEqual(payload["base_part_id"], "2")
        self.assertEqual(payload["base_part_source"], "selected_order[0]")
        self.assertEqual(payload["steps"][2]["assembled_part_ids_before"], ["2", "0"])
        self.assertIn('id="stepSlider"', html)
        self.assertIn('id="progressSlider"', html)
        self.assertIn('id="showFuture"', html)
        self.assertIn("requestAnimationFrame", html)
        self.assertIn("holder_base_available", html)
        self.assertIn('"selected_order":["2","0","3"]', html)

    def test_plumbers_block_defaults_base_to_part_two_in_real_assets_and_cli(self) -> None:
        assembly_dir = REPO_ROOT / "assets" / "obj" / "fabrica" / "plumbers_block"
        sequence = compile_assembly_sequence(
            assembly_dir,
            mesh_scale=0.01,
            repo_root=REPO_ROOT,
        )

        self.assertEqual(sequence.selected_order, ("2", "0", "3", "1", "4"))
        self.assertEqual(sequence.base_part_id, "2")
        self.assertEqual(sequence.base_part_source, "selected_order[0]")
        self.assertEqual(sequence.base_part_order_index, 0)
        self.assertEqual(sequence.first_holder_step_index, 1)
        self.assertEqual(sequence.table_contact_part_ids, ("2",))
        self.assertAlmostEqual(sequence.parts_by_id["2"].bounds_min_assembly_m[2], 0.0)
        self.assertTrue(sequence.parts_by_id["2"].touches_table)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "artifacts"
            exit_code = build_assembly_sequence.main(
                [
                    "--assembly",
                    "plumbers_block",
                    "--output-dir",
                    str(output_dir),
                    "--max-edges-per-part",
                    "20",
                ]
            )
            payload = json.loads((output_dir / "assembly_sequence.json").read_text(encoding="utf-8"))
            html = (output_dir / "assembly_sequence.html").read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["base_part_id"], "2")
        self.assertEqual(payload["base_part_source"], "selected_order[0]")
        self.assertEqual(payload["table"]["contact_part_ids"], ["2"])
        self.assertIn("plumbers_block", html)
        self.assertIn('"selected_order":["2","0","3","1","4"]', html)


if __name__ == "__main__":
    unittest.main()
