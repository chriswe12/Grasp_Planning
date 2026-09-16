from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from grasp_planning.grasping.collision import (
    MeshCollisionPrimitive,
    trimesh_fcl_backend_available,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspBundle,
    SavedGraspCandidate,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline import (
    HolderFeasibilityConfig,
    PlanningConfig,
    Stage1Result,
    compile_assembly_sequence,
    evaluate_holder_state_feasibility,
    holder_state_feasibility,
    write_holder_state_debug_artifacts,
    write_holder_state_feasibility_json,
)


def _transform(translation: tuple[float, float, float]) -> list[list[float]]:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix.tolist()


def _write_box_obj(
    path: Path,
    *,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> None:
    center_array = np.asarray(center, dtype=float)
    half = np.asarray(half_extents, dtype=float)
    vertices = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    vertices = center_array[None, :] + vertices * half[None, :]
    faces = (
        (1, 3, 2),
        (1, 4, 3),
        (5, 6, 7),
        (5, 7, 8),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 4, 8),
        (3, 8, 7),
        (4, 1, 5),
        (4, 5, 8),
    )
    lines = [*(f"v {x} {y} {z}" for x, y, z in vertices), *(f"f {a} {b} {c}" for a, b, c in faces), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_assembly(root: Path) -> Path:
    assembly = root / "stage2_fixture"
    assembly.mkdir()
    (assembly / "precedence_plan.json").write_text(
        json.dumps(
            {
                "assembly": "stage2_fixture",
                "forward_assembly_orders": [["2", "0", "1"]],
            }
        ),
        encoding="utf-8",
    )
    part_translations = {
        "2": (0.0, 0.0, 0.0),
        "0": (-0.6, 0.0, 0.0),
        "1": (0.0, 0.1, 0.0),
    }
    (assembly / "pre_insertion_poses.json").write_text(
        json.dumps(
            {
                "assembly": "stage2_fixture",
                "parts": {
                    part_id: {
                        "role": "moving_part",
                        "final_to_pre_insertion_transform_m": _transform(translation),
                        "pre_to_final_insertion_vector_m": list(-np.asarray(translation, dtype=float)),
                        "pre_to_final_insertion_distance_m": float(np.linalg.norm(translation)),
                    }
                    for part_id, translation in part_translations.items()
                },
            }
        ),
        encoding="utf-8",
    )
    _write_box_obj(
        assembly / "2.obj",
        center=(0.0, -0.5, 0.025),
        half_extents=(0.08, 0.08, 0.025),
    )
    _write_box_obj(
        assembly / "0.obj",
        center=(0.3, 0.0, 0.2),
        half_extents=(0.025, 0.025, 0.025),
    )
    _write_box_obj(
        assembly / "1.obj",
        center=(0.0, 0.5, 0.2),
        half_extents=(0.025, 0.025, 0.025),
    )
    return assembly


def _candidate(
    grasp_id: str,
    position: tuple[float, float, float],
    *,
    vertical_approach: bool = False,
    score: float,
) -> SavedGraspCandidate:
    orientation = (0.0, 0.0, 0.0, 1.0) if vertical_approach else (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=position,
        grasp_orientation_xyzw_obj=orientation,
        contact_point_a_obj=(position[0], position[1] - 0.005, position[2]),
        contact_point_b_obj=(position[0], position[1] + 0.005, position[2]),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.01,
        roll_angle_rad=0.0,
        score=score,
        score_components={"contact_support": score},
        metadata={"candidate_role": "assembly_holder", "base_part_id": "2"},
    )


def _holder_library(candidates: tuple[SavedGraspCandidate, ...]) -> Stage1Result:
    mesh = TriangleMesh(
        vertices_obj=np.asarray([[-0.01, -0.01, 0.0], [0.01, -0.01, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.02]]),
        faces=np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
    )
    pose = ObjectWorldPose(
        position_world=(0.0, 0.0, 0.0),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )
    return Stage1Result(
        bundle=SavedGraspBundle(
            target_mesh_path="stage2_fixture/2.obj",
            mesh_scale=1.0,
            source_frame_origin_obj_world=pose.position_world,
            source_frame_orientation_xyzw_obj_world=pose.orientation_xyzw_world,
            candidates=candidates,
            metadata={"base_part_id": "2", "stage1_cache_key": "fixture-cache"},
        ),
        target_mesh_local=mesh,
        target_pose_in_obj_world=pose,
        obstacle_mesh_world=None,
        collision_backend_name="fixture",
        raw_candidate_count=len(candidates),
        raw_candidates=candidates,
    )


class _SmallCubeGripper:
    _vertices = (
        np.asarray(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        * 0.01
    )
    _faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )

    def primitives_for_grasp(
        self,
        *,
        grasp_rotmat,
        contact_point_a,
        contact_point_b,
        grasp_center=None,
    ):
        del grasp_rotmat, contact_point_a, contact_point_b
        center = np.asarray(grasp_center, dtype=float)
        return (
            MeshCollisionPrimitive(
                name="small_cube_gripper",
                vertices_obj=self._vertices + center[None, :],
                faces=self._faces,
            ),
        )


class HolderOpeningOwnershipTests(unittest.TestCase):
    def test_intended_contact_state_is_not_queried_against_its_own_base(self) -> None:
        candidate = _candidate("h_contact", (0.0, 0.0, 0.2), score=1.0)
        contact_meshes = ("contact_base", "contact_left", "contact_right")
        approach_meshes = ("approach_base", "approach_left", "approach_right")
        queried_meshes: list[tuple[object, ...]] = []

        def query(_manager, meshes):
            queried_meshes.append(tuple(meshes))
            return holder_state_feasibility._CollisionQuery(False, (), 0.1)

        with (
            mock.patch.object(
                holder_state_feasibility,
                "_candidate_opening_meshes_assembly",
                return_value=(contact_meshes, approach_meshes),
            ),
            mock.patch.object(
                holder_state_feasibility,
                "_translated_meshes",
                return_value=("pregrasp",),
            ),
            mock.patch.object(
                holder_state_feasibility,
                "_swept_meshes",
                return_value=("approach_sweep",),
            ),
            mock.patch.object(
                holder_state_feasibility,
                "_minimum_table_clearance",
                return_value=0.1,
            ),
            mock.patch.object(holder_state_feasibility, "_query_manager", side_effect=query),
        ):
            prepared = holder_state_feasibility._prepare_holder_candidate(
                candidate,
                table_z_m=0.0,
                source_pose_assembly=ObjectWorldPose(
                    position_world=(0.0, 0.0, 0.0),
                    orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
                ),
                base_manager=object(),
                planning=PlanningConfig(gripper_collision_model="kuka_y_gripper"),
                config=HolderFeasibilityConfig(),
                model_cache={},
            )

        self.assertIsNone(prepared.static_failure)
        self.assertEqual(queried_meshes[0], approach_meshes)
        self.assertNotIn("contact_left", queried_meshes[0])
        self.assertEqual(prepared.pregrasp_meshes, ("pregrasp",))
        self.assertEqual(prepared.approach_swept_meshes, ("approach_sweep",))


@unittest.skipUnless(trimesh_fcl_backend_available(), "python-fcl is unavailable")
class HolderStateFeasibilityTests(unittest.TestCase):
    def _evaluate(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        assembly_dir = _write_assembly(root)
        sequence = compile_assembly_sequence(
            assembly_dir,
            mesh_scale=1.0,
            repo_root=root,
        )
        candidates = (
            _candidate("h_mid", (0.0, 0.0, 0.2), score=1.0),
            _candidate("h_table", (-0.4, 0.0, 0.005), score=0.9),
            _candidate("h_margin", (-0.4, 0.1, 0.0115), score=0.8),
            _candidate("h_clear", (-0.5, 0.0, 0.2), score=0.7),
            _candidate("h_approach", (0.3, 0.0, 0.3), vertical_approach=True, score=0.6),
            _candidate("h_static", (0.3, 0.0, 0.2), score=0.5),
        )
        with mock.patch.object(
            holder_state_feasibility,
            "make_gripper_collision_model",
            return_value=_SmallCubeGripper(),
        ):
            result = evaluate_holder_state_feasibility(
                sequence=sequence,
                holder_library=_holder_library(candidates),
                planning=PlanningConfig(gripper_collision_model="kuka_y_gripper"),
                config=HolderFeasibilityConfig(
                    pregrasp_offset_m=0.2,
                    table_clearance_margin_m=0.002,
                    geometry_clearance_margin_m=0.0,
                    incoming_path_samples=25,
                ),
            )
        return root, sequence, result

    def test_separates_static_table_state_and_incoming_sweep_reasons(self) -> None:
        _, _, result = self._evaluate()
        state0, state1, state2 = result.states
        results1 = {entry.grasp_id: entry for entry in state1.candidate_results}
        results2 = {entry.grasp_id: entry for entry in state2.candidate_results}

        self.assertEqual(state0.reason_counts, {"base_not_available": 6})
        self.assertEqual(results1["h_mid"].reason, "incoming_part_sweep_collision")
        self.assertEqual(results1["h_table"].reason, "table_collision")
        self.assertEqual(results1["h_margin"].reason, "clearance_margin_failed")
        self.assertEqual(results1["h_clear"].reason, "accepted")
        self.assertEqual(results2["h_static"].reason, "assembled_part_collision")
        self.assertEqual(results2["h_approach"].reason, "holder_approach_sweep_collision")

    def test_incoming_sweep_detects_midpoint_when_endpoints_are_clear(self) -> None:
        _, _, result = self._evaluate()
        state = result.states[1]
        midpoint = next(entry for entry in state.candidate_results if entry.grasp_id == "h_mid")

        progress = midpoint.details["first_failing_insertion_progress"]
        self.assertIsNotNone(progress)
        self.assertGreater(progress, 0.0)
        self.assertLess(progress, 1.0)
        motion = state.to_payload()["obstacle_motion_specs"][-1]
        self.assertEqual(motion["motion"], "linear_insertion")
        self.assertEqual(motion["translation_start_m"], [-0.6, 0.0, 0.0])
        self.assertEqual(motion["translation_end_m"], [0.0, 0.0, 0.0])

    def test_artifacts_preserve_reasons_matrix_and_step_views(self) -> None:
        root, sequence, result = self._evaluate()
        output_dir = root / "artifacts"
        output_json = output_dir / "holder_state_feasibility.json"
        write_holder_state_feasibility_json(result, output_json)
        matrix_path, state_paths = write_holder_state_debug_artifacts(
            result,
            sequence,
            output_dir,
        )
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        matrix_html = matrix_path.read_text(encoding="utf-8")

        self.assertEqual(payload["kind"], "holder_state_feasibility")
        self.assertEqual(payload["base_part_id"], "2")
        self.assertEqual(payload["states"][1]["incoming_part_id"], "0")
        self.assertEqual(len(payload["candidates"]), 6)
        self.assertEqual(len(state_paths), 3)
        self.assertTrue(all(path.is_file() for path in state_paths))
        self.assertIn('id="matrix"', matrix_html)
        self.assertIn("incoming_part_sweep_collision", matrix_html)
        self.assertIn("holder_approach_sweep_collision", matrix_html)
        self.assertIn("table_clearance_margin_m", matrix_html)
        self.assertIn("componentWorld", matrix_html)

    def test_rejects_holder_library_for_another_base(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        sequence = compile_assembly_sequence(
            _write_assembly(root),
            mesh_scale=1.0,
            repo_root=root,
        )
        library = _holder_library((_candidate("h1", (0.0, 0.0, 0.2), score=1.0),))
        library.bundle.metadata["base_part_id"] = "0"

        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_holder_state_feasibility(
                sequence=sequence,
                holder_library=library,
                planning=PlanningConfig(gripper_collision_model="kuka_y_gripper"),
                config=HolderFeasibilityConfig(),
            )


if __name__ == "__main__":
    unittest.main()
