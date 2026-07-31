from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspBundle,
    SavedGraspCandidate,
    load_grasp_bundle,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline import (
    PlanningConfig,
    Stage1Result,
    compile_assembly_sequence,
    generate_holder_grasp_library,
    holder_grasp_library,
    write_holder_grasp_library_artifacts,
)
from grasp_planning.pipeline.assembly_sequence import REPO_ROOT
from scripts import build_holder_grasp_library


def _candidate(grasp_id: str = "g0007") -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=(0.0, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.0, -0.02, 0.0),
        contact_point_b_obj=(0.0, 0.02, 0.0),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.04,
        roll_angle_rad=0.25,
        contact_patch_lateral_offset_m=-0.002,
        contact_patch_approach_offset_m=0.003,
        score=0.75,
        score_components={"contact_support": 0.8, "antipodal": 0.7},
        metadata={"existing": "preserved"},
    )


def _mesh() -> TriangleMesh:
    return TriangleMesh(
        vertices_obj=np.asarray(
            [
                [-0.03, -0.02, -0.01],
                [0.03, -0.02, -0.01],
                [0.0, 0.02, -0.01],
                [0.0, 0.0, 0.03],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], dtype=np.int64),
    )


def _plumbers_sequence():
    return compile_assembly_sequence(
        REPO_ROOT / "assets" / "obj" / "fabrica" / "plumbers_block",
        mesh_scale=0.01,
        repo_root=REPO_ROOT,
    )


def _stage1_fixture() -> Stage1Result:
    candidate = _candidate()
    pose = ObjectWorldPose(
        position_world=(0.01, 0.02, 0.03),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )
    return Stage1Result(
        bundle=SavedGraspBundle(
            target_mesh_path="obj/fabrica/plumbers_block/2.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=pose.position_world,
            source_frame_orientation_xyzw_obj_world=pose.orientation_xyzw_world,
            candidates=(candidate,),
            metadata={
                "collision_backend": "synthetic",
                "stage1_cache_key": "cache-key",
                "stage1_collision_checks_skipped": True,
            },
        ),
        target_mesh_local=_mesh(),
        target_pose_in_obj_world=pose,
        obstacle_mesh_world=None,
        collision_backend_name="synthetic",
        raw_candidate_count=1,
        raw_candidates=(candidate,),
    )


class HolderGraspLibraryTests(unittest.TestCase):
    def test_reuses_stage1_without_state_filter_and_preserves_candidate_data(self) -> None:
        sequence = _plumbers_sequence()
        planning = PlanningConfig(
            gripper_collision_model="kuka_y_gripper",
            skip_stage1_collision_checks=False,
        )
        fixture = _stage1_fixture()

        with mock.patch.object(holder_grasp_library, "generate_stage1_result", return_value=fixture) as generate:
            result = generate_holder_grasp_library(sequence=sequence, planning=planning)

        call = generate.call_args.kwargs
        self.assertEqual(call["geometry"].target_mesh_path, str(sequence.parts_by_id["2"].resolved_mesh_path))
        self.assertEqual(call["geometry"].assembly_obstacle_paths, ())
        self.assertTrue(call["planning"].skip_stage1_collision_checks)
        self.assertFalse(planning.skip_stage1_collision_checks)

        saved = result.bundle.candidates[0]
        self.assertEqual(saved.grasp_id, "h0007")
        self.assertEqual(saved.contact_point_a_obj, fixture.bundle.candidates[0].contact_point_a_obj)
        self.assertEqual(saved.contact_normal_b_obj, fixture.bundle.candidates[0].contact_normal_b_obj)
        self.assertEqual(saved.jaw_width, 0.04)
        self.assertEqual(saved.roll_angle_rad, 0.25)
        self.assertEqual(saved.contact_patch_lateral_offset_m, -0.002)
        self.assertEqual(saved.contact_patch_approach_offset_m, 0.003)
        self.assertEqual(saved.score_components, {"contact_support": 0.8, "antipodal": 0.7})
        self.assertEqual(saved.metadata["existing"], "preserved")
        self.assertEqual(saved.metadata["source_grasp_id"], "g0007")
        self.assertEqual(result.raw_candidates[0].grasp_id, "h0007")

        metadata = result.bundle.metadata
        self.assertEqual(metadata["artifact_kind"], "holder_base_candidate_library")
        self.assertEqual(metadata["assembly"], "plumbers_block")
        self.assertEqual(metadata["base_part_id"], "2")
        self.assertEqual(metadata["base_part_source"], "selected_order[0]")
        self.assertEqual(metadata["selected_assembly_order"], ["2", "0", "3", "1", "4"])
        self.assertEqual(metadata["stage1_cache_key"], "cache-key")
        self.assertFalse(metadata["state_filter_applied"])
        self.assertFalse(metadata["table_filter_applied"])
        self.assertFalse(metadata["incoming_part_sweep_filter_applied"])

    def test_requires_kuka_y_gripper(self) -> None:
        with self.assertRaisesRegex(ValueError, "kuka_y_gripper"):
            generate_holder_grasp_library(
                sequence=_plumbers_sequence(),
                planning=PlanningConfig(gripper_collision_model="franka_hand"),
            )

    def test_artifacts_round_trip_and_embed_interactive_kuka_viewer(self) -> None:
        sequence = _plumbers_sequence()
        planning = PlanningConfig(gripper_collision_model="kuka_y_gripper")
        with mock.patch.object(
            holder_grasp_library,
            "generate_stage1_result",
            return_value=_stage1_fixture(),
        ):
            result = generate_holder_grasp_library(sequence=sequence, planning=planning)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_json = Path(temp_dir) / "holder_base_candidates.json"
            output_html = Path(temp_dir) / "holder_base_candidates.html"
            write_holder_grasp_library_artifacts(
                result,
                sequence=sequence,
                planning=planning,
                output_json=output_json,
                output_html=output_html,
            )
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            loaded = load_grasp_bundle(output_json)
            html = output_html.read_text(encoding="utf-8")

        self.assertEqual(payload["metadata"]["planning_stage"], "dual_robot_stage_1")
        self.assertEqual(payload["candidates"][0]["grasp_id"], "h0007")
        self.assertEqual(payload["candidates"][0]["contact_patch_offset_local"], [-0.002, 0.003])
        self.assertEqual(loaded.candidates, result.bundle.candidates)
        self.assertIn("Fabrica Base Holder Candidate Library", html)
        self.assertIn("kuka_y_gripper", html)
        self.assertIn("contact_support", html)
        self.assertIn('id="candidateList"', html)
        self.assertIn('id="scoreMin"', html)
        self.assertIn('"tcp_to_grasp_center_m"', html)
        self.assertIn('"table_plane_local"', html)
        self.assertIn("drawTable", html)
        self.assertIn("componentWorld", html)
        self.assertIn("base_selection:    selected_order[0]", html)
        self.assertIn("state_filter:      not applied (Stage 2)", html)

    def test_cli_loads_config_and_writes_named_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "dual.yaml"
            output_dir = root / "output"
            config.write_text(
                "\n".join(
                    [
                        "assembly:",
                        '  name: "plumbers_block"',
                        f'  asset_root: "{REPO_ROOT / "assets" / "obj" / "fabrica"}"',
                        "  mesh_scale: 0.01",
                        "artifacts:",
                        f'  output_root: "{root / "unused"}"',
                        "planning:",
                        "  stage1_cache_enabled: false",
                        "  num_surface_samples: 32",
                        '  gripper_collision_model: "kuka_y_gripper"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                build_holder_grasp_library,
                "generate_holder_grasp_library",
                return_value=_stage1_fixture(),
            ) as generate:
                exit_code = build_holder_grasp_library.main(
                    [
                        "--config",
                        str(config),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            payload = json.loads((output_dir / "holder_base_candidates.json").read_text(encoding="utf-8"))
            html_exists = (output_dir / "holder_base_candidates.html").is_file()

        self.assertEqual(exit_code, 0)
        self.assertEqual(generate.call_args.kwargs["sequence"].base_part_id, "2")
        self.assertEqual(generate.call_args.kwargs["sequence"].base_part_source, "selected_order[0]")
        self.assertEqual(generate.call_args.kwargs["planning"].num_surface_samples, 32)
        self.assertEqual(generate.call_args.kwargs["planning"].gripper_collision_model, "kuka_y_gripper")
        self.assertEqual(payload["target"]["mesh_path"], "obj/fabrica/plumbers_block/2.obj")
        self.assertTrue(html_exists)


if __name__ == "__main__":
    unittest.main()
