from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import grasp_planning.pipeline.fabrica_pipeline as fabrica_pipeline
import grasp_planning.pipeline.regrasp_debug_html as regrasp_debug_html
import grasp_planning.pipeline.regrasp_fallback as regrasp_fallback
from grasp_planning.grasping.fabrica_grasp_debug import CandidateStatus, SavedGraspBundle, SavedGraspCandidate
from grasp_planning.grasping.mesh_antipodal_grasp_generator import (
    ObjectFrameGraspCandidate,
    SurfaceSample,
    TriangleMesh,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline import GeometryConfig, PlanningConfig, generate_stage1_result
from scripts import run_fabrica_grasp_in_mujoco, run_grasp_pipeline


class RunGraspPipelineModeTests(unittest.TestCase):
    def test_normalize_mode_maps_aliases(self) -> None:
        self.assertEqual(run_grasp_pipeline._normalize_mode("simulation"), "sim")
        self.assertEqual(run_grasp_pipeline._normalize_mode("perception_in_the_loop"), "pitl")
        self.assertEqual(run_grasp_pipeline._normalize_mode("perception-in-the-loop"), "pitl")
        self.assertEqual(run_grasp_pipeline._normalize_mode("real"), "real")

    def test_run_mujoco_execution_uses_stage2_bundle(self) -> None:
        cfg = run_grasp_pipeline.MujocoPipelineConfig(
            enabled=True,
            robot_config="configs/mujoco_fr3_with_hand.json",
            attempt_artifact="artifacts/test_pick_attempt.json",
        )

        with mock.patch.object(run_grasp_pipeline.subprocess, "run") as subprocess_run:
            run_grasp_pipeline._run_mujoco_execution(
                cfg,
                input_json=Path("artifacts/pipeline_stage2_ground_feasible.json"),
                headless=False,
            )

        subprocess_run.assert_called_once()
        command = subprocess_run.call_args.args[0]
        self.assertIn("scripts/run_fabrica_grasp_in_mujoco.py", command)
        self.assertIn("artifacts/pipeline_stage2_ground_feasible.json", command)
        self.assertIn("configs/mujoco_fr3_with_hand.json", command)
        self.assertIn("--viewer", command)
        self.assertEqual(subprocess_run.call_args.kwargs["cwd"], run_grasp_pipeline.REPO_ROOT)
        self.assertTrue(subprocess_run.call_args.kwargs["check"])

    def test_run_mujoco_execution_headless_suppresses_viewer(self) -> None:
        cfg = run_grasp_pipeline.MujocoPipelineConfig(
            enabled=True,
            robot_config="configs/mujoco_fr3_with_hand.json",
        )

        with mock.patch.object(run_grasp_pipeline.subprocess, "run") as subprocess_run:
            run_grasp_pipeline._run_mujoco_execution(
                cfg,
                input_json=Path("artifacts/pipeline_stage2_ground_feasible.json"),
                headless=True,
            )

        command = subprocess_run.call_args.args[0]
        self.assertNotIn("--viewer", command)

    def test_run_mujoco_execution_passes_shared_simulation_config(self) -> None:
        cfg = run_grasp_pipeline.MujocoPipelineConfig(
            enabled=True,
            robot_config="configs/mujoco_fr3_with_hand.json",
            simulation_config="configs/mujoco_simulation.yaml",
        )

        with mock.patch.object(run_grasp_pipeline.subprocess, "run") as subprocess_run:
            run_grasp_pipeline._run_mujoco_execution(
                cfg,
                input_json=Path("artifacts/pipeline_stage2_ground_feasible.json"),
                headless=False,
            )

        command = subprocess_run.call_args.args[0]
        self.assertIn("--simulation-config", command)
        self.assertIn("configs/mujoco_simulation.yaml", command)

    def test_run_mujoco_execution_passes_moveit_controller_config(self) -> None:
        cfg = run_grasp_pipeline.MujocoPipelineConfig(
            enabled=True,
            robot_config="configs/mujoco_fr3_with_hand.json",
            controller="moveit",
            moveit_frame_id="base",
            moveit_planner_id="RRTConnectkConfigDefault",
            moveit_allow_collisions=True,
            regrasp_moveit_max_candidate_plans=17,
            regrasp_moveit_transfer_candidates_per_placement=2,
            regrasp_moveit_final_candidates_per_placement=4,
        )

        with mock.patch.object(run_grasp_pipeline.subprocess, "run") as subprocess_run:
            run_grasp_pipeline._run_mujoco_execution(
                cfg,
                input_json=Path("artifacts/pipeline_stage2_ground_feasible.json"),
                headless=True,
            )

        command = subprocess_run.call_args.args[0]
        self.assertIn("--controller", command)
        self.assertIn("moveit", command)
        self.assertIn("--moveit-frame-id", command)
        self.assertIn("base", command)
        self.assertIn("--moveit-planner-id", command)
        self.assertIn("RRTConnectkConfigDefault", command)
        self.assertIn("--moveit-allow-collisions", command)
        self.assertIn("--regrasp-moveit-max-candidate-plans", command)
        self.assertIn("17", command)
        self.assertIn("--regrasp-moveit-transfer-candidates-per-placement", command)
        self.assertIn("2", command)
        self.assertIn("--regrasp-moveit-final-candidates-per-placement", command)
        self.assertIn("4", command)

    def test_run_mujoco_execution_passes_regrasp_plan_and_preserves_moveit_controller(self) -> None:
        cfg = run_grasp_pipeline.MujocoPipelineConfig(
            enabled=True,
            robot_config="configs/mujoco_fr3_with_hand.json",
            controller="moveit",
            moveit_allow_collisions=True,
        )

        with mock.patch.object(run_grasp_pipeline.subprocess, "run") as subprocess_run:
            run_grasp_pipeline._run_mujoco_execution(
                cfg,
                input_json=Path("artifacts/pipeline_stage2_ground_feasible.json"),
                headless=True,
                regrasp_plan_json=Path("artifacts/regrasp_plan.json"),
            )

        command = subprocess_run.call_args.args[0]
        self.assertIn("--regrasp-plan-json", command)
        self.assertIn("artifacts/regrasp_plan.json", command)
        controller_index = command.index("--controller") + 1
        self.assertEqual(command[controller_index], "moveit")
        self.assertIn("--moveit-allow-collisions", command)

    def test_run_isaac_execution_uses_stage2_bundle(self) -> None:
        cfg = run_grasp_pipeline.IsaacPipelineConfig(
            enabled=True,
            python_executable="/isaac-sim/python.sh",
            part_usd="artifacts/beam.usd",
            attempt_artifact="artifacts/test_isaac_attempt.json",
            headless=True,
        )

        with mock.patch.object(run_grasp_pipeline.subprocess, "run") as subprocess_run:
            run_grasp_pipeline._run_isaac_execution(
                cfg,
                input_json=Path("artifacts/pipeline_stage2_ground_feasible.json"),
                headless=False,
            )

        subprocess_run.assert_called_once()
        command = subprocess_run.call_args.args[0]
        self.assertEqual(command[0], "/isaac-sim/python.sh")
        self.assertIn("scripts/run_fabrica_grasp_in_isaac.py", command)
        self.assertIn("artifacts/pipeline_stage2_ground_feasible.json", command)
        self.assertIn("artifacts/beam.usd", command)
        self.assertIn("--headless", command)
        self.assertEqual(subprocess_run.call_args.kwargs["cwd"], run_grasp_pipeline.REPO_ROOT)
        self.assertTrue(subprocess_run.call_args.kwargs["check"])

    def test_run_isaac_execution_allows_generated_bundle_local_usd(self) -> None:
        cfg = run_grasp_pipeline.IsaacPipelineConfig(enabled=True)

        with mock.patch.object(run_grasp_pipeline.subprocess, "run") as subprocess_run:
            run_grasp_pipeline._run_isaac_execution(
                cfg,
                input_json=Path("artifacts/pipeline_stage2_ground_feasible.json"),
                headless=True,
            )

        command = subprocess_run.call_args.args[0]
        self.assertNotIn("--part-usd", command)
        self.assertIn("--headless", command)

    def test_backend_override_selects_one_execution_backend(self) -> None:
        mujoco_cfg = run_grasp_pipeline.MujocoPipelineConfig(enabled=True)
        isaac_cfg = run_grasp_pipeline.IsaacPipelineConfig(enabled=True)

        mujoco_selected, isaac_selected = run_grasp_pipeline._execution_backend_configs(
            mujoco_execution=mujoco_cfg,
            isaac_execution=isaac_cfg,
            backend="isaac",
        )
        self.assertFalse(mujoco_selected.enabled)
        self.assertTrue(isaac_selected.enabled)

        mujoco_selected, isaac_selected = run_grasp_pipeline._execution_backend_configs(
            mujoco_execution=mujoco_cfg,
            isaac_execution=isaac_cfg,
            backend="mujoco",
        )
        self.assertTrue(mujoco_selected.enabled)
        self.assertFalse(isaac_selected.enabled)

    def test_artifacts_derives_part_frame_html_when_omitted(self) -> None:
        artifacts = run_grasp_pipeline._artifacts(
            {
                "artifacts": {
                    "stage1_json": "artifacts/test_stage1.json",
                    "stage1_html": "artifacts/test_stage1.html",
                    "stage2_json": "artifacts/test_stage2.json",
                    "stage2_html": "artifacts/test_stage2.html",
                }
            }
        )

        self.assertEqual(artifacts["part_frame_html"], Path("artifacts/test_stage2_part_frame.html"))

    def test_mujoco_execution_config_parses_regrasp_html_artifact(self) -> None:
        config = run_grasp_pipeline._mujoco_execution_config(
            {
                "mujoco_execution": {
                    "regrasp_html_artifact": "artifacts/regrasp_debug.html",
                    "regrasp_staging_xy_offsets_m": [[0.0, 0.0], [0.1, -0.1]],
                    "regrasp_max_placement_options": 4,
                    "regrasp_moveit_max_candidate_plans": 12,
                    "regrasp_moveit_transfer_candidates_per_placement": 2,
                    "regrasp_moveit_final_candidates_per_placement": 5,
                }
            }
        )

        self.assertEqual(config.regrasp_html_artifact, "artifacts/regrasp_debug.html")
        self.assertEqual(config.regrasp_staging_xy_offsets_m, ((0.0, 0.0), (0.1, -0.1)))
        self.assertEqual(config.regrasp_max_placement_options, 4)
        self.assertEqual(config.regrasp_moveit_max_candidate_plans, 12)
        self.assertEqual(config.regrasp_moveit_transfer_candidates_per_placement, 2)
        self.assertEqual(config.regrasp_moveit_final_candidates_per_placement, 5)

    def test_settle_object_pose_on_floor_shifts_mesh_bottom_to_z_zero(self) -> None:
        mesh_local = SimpleNamespace(vertices_obj=np.array([[-0.1, 0.0, -0.02], [0.1, 0.0, 0.04]], dtype=float))
        pose = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.1),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )

        settled = run_grasp_pipeline._settle_object_pose_on_floor(pose, mesh_local)

        self.assertIsNotNone(settled)
        assert settled is not None
        self.assertEqual(settled.orientation_xyzw_world, pose.orientation_xyzw_world)
        self.assertAlmostEqual(settled.position_world[2], 0.02)

    def test_planning_config_parses_floor_clearance_margin(self) -> None:
        config = run_grasp_pipeline._planning_config({"planning": {"floor_clearance_margin_m": 0.012}})

        self.assertAlmostEqual(config.floor_clearance_margin_m, 0.012)

    def test_planning_config_parses_skip_stage1_collision_checks(self) -> None:
        config = run_grasp_pipeline._planning_config({"planning": {"skip_stage1_collision_checks": True}})

        self.assertTrue(config.skip_stage1_collision_checks)

    def test_planning_config_parses_stage1_cache_settings(self) -> None:
        config = run_grasp_pipeline._planning_config(
            {"planning": {"stage1_cache_enabled": False, "stage1_cache_dir": "artifacts/custom_cache"}}
        )

        self.assertFalse(config.stage1_cache_enabled)
        self.assertEqual(config.stage1_cache_dir, "artifacts/custom_cache")

    def test_planning_config_parses_top_grasp_score_weight(self) -> None:
        config = run_grasp_pipeline._planning_config({"planning": {"top_grasp_score_weight": 0.8}})

        self.assertAlmostEqual(config.top_grasp_score_weight, 0.8)

    def test_planning_config_parses_regrasp_transfer_top_grasp_score_weight(self) -> None:
        config = run_grasp_pipeline._planning_config(
            {"planning": {"regrasp_transfer_top_grasp_score_weight": 0.9}}
        )

        self.assertAlmostEqual(config.regrasp_transfer_top_grasp_score_weight, 0.9)

    def test_planning_config_expands_roll_angle_step_degrees(self) -> None:
        config = run_grasp_pipeline._planning_config({"planning": {"roll_angle_step_deg": 90.0}})

        self.assertEqual(config.roll_angles_rad, (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi))

    def test_planning_config_roll_angle_step_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            run_grasp_pipeline._planning_config({"planning": {"roll_angle_step_deg": 0.0}})


def _saved_candidate(grasp_id: str, orientation_xyzw: tuple[float, float, float, float]) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=(0.0, 0.0, 0.0),
        grasp_orientation_xyzw_obj=orientation_xyzw,
        contact_point_a_obj=(-0.01, 0.0, 0.0),
        contact_point_b_obj=(0.01, 0.0, 0.0),
        contact_normal_a_obj=(1.0, 0.0, 0.0),
        contact_normal_b_obj=(-1.0, 0.0, 0.0),
        jaw_width=0.02,
        roll_angle_rad=0.0,
    )


class Stage2WorldTopApproachScoringTests(unittest.TestCase):
    def test_stage2_scoring_prefers_top_down_world_approach(self) -> None:
        bottom_up = _saved_candidate("bottom_up", (0.0, 0.0, 0.0, 1.0))
        top_down = _saved_candidate("top_down", (1.0, 0.0, 0.0, 0.0))

        def fake_score_grasps(grasps: list[SavedGraspCandidate], *, mesh_local: object) -> list[SavedGraspCandidate]:
            return [
                replace(bottom_up, score=1.0, score_components={"score": 1.0}),
                replace(top_down, score=0.5, score_components={"score": 0.5}),
            ]

        with mock.patch.object(fabrica_pipeline, "score_grasps", side_effect=fake_score_grasps):
            scored = fabrica_pipeline._score_grasps_for_world_top_approach(
                [bottom_up, top_down],
                mesh_local=object(),
                object_pose_world=ObjectWorldPose(
                    position_world=(0.0, 0.0, 0.0),
                    orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
                ),
                top_grasp_score_weight=0.8,
            )

        self.assertEqual(scored[0].grasp_id, "top_down")
        self.assertAlmostEqual(scored[0].score_components["top_down_approach"], 1.0)
        self.assertAlmostEqual(scored[0].score_components["world_approach_z"], -1.0)


class Stage1CollisionSkipTests(unittest.TestCase):
    def test_generate_stage1_can_skip_assembly_collision_filter(self) -> None:
        mesh = TriangleMesh(
            vertices_obj=np.array(
                [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
                dtype=float,
            ),
            faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
        )
        raw_candidate = ObjectFrameGraspCandidate(
            grasp_position_obj=(0.01, 0.01, 0.0),
            grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
            contact_point_a_obj=(0.0, 0.01, 0.0),
            contact_point_b_obj=(0.02, 0.01, 0.0),
            contact_normal_a_obj=(1.0, 0.0, 0.0),
            contact_normal_b_obj=(-1.0, 0.0, 0.0),
            jaw_width=0.02,
            roll_angle_rad=0.0,
        )

        class FakeGenerator:
            collision_backend_name = "unit-test"

            def __init__(self, config: object) -> None:
                self.config = config

            def generate(self, mesh_local: object) -> list[ObjectFrameGraspCandidate]:
                return [raw_candidate]

        with (
            mock.patch("grasp_planning.pipeline.fabrica_pipeline.load_asset_mesh", return_value=mesh),
            mock.patch("grasp_planning.pipeline.fabrica_pipeline.AntipodalMeshGraspGenerator", FakeGenerator),
            mock.patch("grasp_planning.pipeline.fabrica_pipeline.load_assembly_obstacle_mesh") as load_obstacles,
            mock.patch("grasp_planning.pipeline.fabrica_pipeline.filter_grasps_against_assembly") as filter_assembly,
        ):
            result = generate_stage1_result(
                geometry=GeometryConfig(
                    target_mesh_path="obj/fabrica/beam/2.obj",
                    mesh_scale=0.01,
                    assembly_glob="obj/fabrica/beam/*.obj",
                ),
                planning=PlanningConfig(skip_stage1_collision_checks=True, stage1_cache_enabled=False),
            )

        load_obstacles.assert_not_called()
        filter_assembly.assert_not_called()
        self.assertEqual(result.raw_candidate_count, 1)
        self.assertEqual(len(result.raw_candidates), 1)
        self.assertEqual(len(result.bundle.candidates), 1)
        self.assertTrue(result.bundle.metadata["stage1_collision_checks_skipped"])

    def test_stage1_cache_key_records_asset_relative_assembly_files(self) -> None:
        records = fabrica_pipeline._assembly_cache_records(
            GeometryConfig(
                target_mesh_path="obj/fabrica/beam/2.obj",
                mesh_scale=0.01,
                assembly_glob="obj/fabrica/beam/*.obj",
            ),
            PlanningConfig(stage1_cache_enabled=True),
        )

        paths = [str(record["path"]) for record in records]
        self.assertIn("obj/fabrica/beam/0.obj", paths)
        self.assertIn("obj/fabrica/beam/1.obj", paths)
        self.assertNotIn("obj/fabrica/beam/2.obj", paths)

    def test_generate_stage1_reuses_cached_grasps_and_surface_samples(self) -> None:
        mesh = TriangleMesh(
            vertices_obj=np.array(
                [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
                dtype=float,
            ),
            faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
        )
        raw_candidate = ObjectFrameGraspCandidate(
            grasp_position_obj=(0.01, 0.01, 0.0),
            grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
            contact_point_a_obj=(0.0, 0.01, 0.0),
            contact_point_b_obj=(0.02, 0.01, 0.0),
            contact_normal_a_obj=(1.0, 0.0, 0.0),
            contact_normal_b_obj=(-1.0, 0.0, 0.0),
            jaw_width=0.02,
            roll_angle_rad=0.0,
        )
        surface_sample = SurfaceSample(
            point_obj=(0.01, 0.02, 0.0),
            normal_obj=(0.0, 0.0, 1.0),
            face_index=0,
        )

        class FakeGenerator:
            collision_backend_name = "unit-test"

            def __init__(self, config: object) -> None:
                self.config = config
                self.last_surface_samples = (surface_sample,)

            def generate(self, mesh_local: object) -> list[ObjectFrameGraspCandidate]:
                return [raw_candidate]

        geometry = GeometryConfig(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=0.01,
            assembly_glob="obj/fabrica/beam/*.obj",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            planning = PlanningConfig(
                skip_stage1_collision_checks=True,
                stage1_cache_enabled=True,
                stage1_cache_dir=temp_dir,
            )
            with (
                mock.patch("grasp_planning.pipeline.fabrica_pipeline.load_asset_mesh", return_value=mesh),
                mock.patch("grasp_planning.pipeline.fabrica_pipeline.AntipodalMeshGraspGenerator", FakeGenerator),
            ):
                first = generate_stage1_result(geometry=geometry, planning=planning)
            self.assertFalse(first.bundle.metadata["stage1_cache_hit"])
            self.assertEqual(len(first.surface_samples), 1)

            with (
                mock.patch("grasp_planning.pipeline.fabrica_pipeline.load_asset_mesh", return_value=mesh),
                mock.patch(
                    "grasp_planning.pipeline.fabrica_pipeline.AntipodalMeshGraspGenerator",
                    side_effect=AssertionError("generator should not run on cache hit"),
                ),
            ):
                second = generate_stage1_result(geometry=geometry, planning=planning)

        self.assertTrue(second.bundle.metadata["stage1_cache_hit"])
        self.assertEqual(second.raw_candidate_count, 1)
        self.assertEqual(len(second.raw_candidates), 1)
        self.assertEqual(len(second.bundle.candidates), 1)
        self.assertEqual(second.surface_samples, (surface_sample,))


class MujocoRegraspFallbackPlanningTests(unittest.TestCase):
    def _candidate(
        self,
        grasp_id: str,
        *,
        z: float = 0.0,
        orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> SavedGraspCandidate:
        return SavedGraspCandidate(
            grasp_id=grasp_id,
            grasp_position_obj=(0.0, 0.0, z),
            grasp_orientation_xyzw_obj=orientation_xyzw,
            contact_point_a_obj=(0.0, -0.01, z),
            contact_point_b_obj=(0.0, 0.01, z),
            contact_normal_a_obj=(0.0, 1.0, 0.0),
            contact_normal_b_obj=(0.0, -1.0, 0.0),
            jaw_width=0.02,
            roll_angle_rad=0.0,
        )

    def _cube_mesh(self) -> TriangleMesh:
        half = 0.02
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

    def test_regrasp_fallback_can_be_forced_even_with_direct_feasible_grasps(self) -> None:
        transfer = self._candidate("transfer")
        final = self._candidate("final")
        bundle = SavedGraspBundle(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=(0.0, 0.0, 0.0),
            source_frame_orientation_xyzw_obj_world=(0.0, 0.0, 0.0, 1.0),
            candidates=(final,),
            metadata={},
        )
        stage1 = fabrica_pipeline.Stage1Result(
            bundle=bundle,
            target_mesh_local=self._cube_mesh(),
            target_pose_in_obj_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.0),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
            obstacle_mesh_world=None,
            collision_backend_name="unit-test",
            raw_candidate_count=2,
            raw_candidates=(transfer, final),
        )
        direct_stage2 = SimpleNamespace(
            accepted=(final,),
            pickup_pose_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.02),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        def accept_all(grasps, **_kwargs):
            return [CandidateStatus(grasp=grasp, status="accepted", reason="unit_test") for grasp in grasps]

        with mock.patch.object(regrasp_fallback, "evaluate_saved_grasps_against_pickup_pose", side_effect=accept_all):
            plan = regrasp_fallback.plan_mujoco_regrasp_fallback(
                stage1=stage1,
                direct_stage2=direct_stage2,
                planning=PlanningConfig(),
                force=True,
                max_orientations=4,
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.metadata["reason"], "forced")
        self.assertNotEqual(plan.staging_object_pose_world.orientation_xyzw_world, direct_stage2.pickup_pose_world.orientation_xyzw_world)
        self.assertEqual(plan.final_grasp.grasp_id, "final")
        self.assertIn(plan.transfer_grasp.grasp_id, {"transfer", "final"})
        self.assertGreaterEqual(len(plan.transfer_grasp_candidates), 1)
        self.assertGreaterEqual(len(plan.final_grasp_candidates), 1)
        self.assertEqual(plan.transfer_grasp_candidates[0].grasp_id, plan.transfer_grasp.grasp_id)
        self.assertEqual(plan.final_grasp_candidates[0].grasp_id, plan.final_grasp.grasp_id)
        self.assertGreaterEqual(len(plan.placement_options), 1)
        self.assertEqual(plan.placement_options[0].transfer_grasp.grasp_id, plan.transfer_grasp.grasp_id)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "regrasp_plan.json"
            regrasp_fallback.write_mujoco_regrasp_plan(
                plan,
                output_path,
                input_stage2_json=Path("artifacts/test_stage2.json"),
            )
            loaded = regrasp_fallback.load_mujoco_regrasp_plan(output_path)

        self.assertEqual(loaded.transfer_grasp.grasp_id, plan.transfer_grasp.grasp_id)
        self.assertEqual(loaded.final_grasp.grasp_id, "final")
        self.assertEqual(loaded.metadata["reason"], "forced")
        self.assertEqual(loaded.transfer_grasp_candidates[0].grasp_id, plan.transfer_grasp.grasp_id)
        self.assertEqual(loaded.final_grasp_candidates[0].grasp_id, "final")
        self.assertGreaterEqual(len(loaded.placement_options), 1)

    def test_regrasp_transfer_scoring_strongly_prefers_initial_top_down_grasp(self) -> None:
        high_object_score = self._candidate("high_object_score")
        top_down_transfer = self._candidate("top_down_transfer", orientation_xyzw=(1.0, 0.0, 0.0, 0.0))
        final = self._candidate("final")
        bundle = SavedGraspBundle(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=(0.0, 0.0, 0.0),
            source_frame_orientation_xyzw_obj_world=(0.0, 0.0, 0.0, 1.0),
            candidates=(final,),
            metadata={},
        )
        stage1 = fabrica_pipeline.Stage1Result(
            bundle=bundle,
            target_mesh_local=self._cube_mesh(),
            target_pose_in_obj_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.0),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
            obstacle_mesh_world=None,
            collision_backend_name="unit-test",
            raw_candidate_count=3,
            raw_candidates=(high_object_score, top_down_transfer, final),
        )
        direct_stage2 = SimpleNamespace(
            accepted=(final,),
            pickup_pose_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.02),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        def accept_all(grasps, **_kwargs):
            return [CandidateStatus(grasp=grasp, status="accepted", reason="unit_test") for grasp in grasps]

        def fake_score_grasps(grasps: list[SavedGraspCandidate], *, mesh_local: object) -> list[SavedGraspCandidate]:
            scores = {"high_object_score": 1.0, "top_down_transfer": 0.2, "final": 0.7}
            return sorted(
                [
                    replace(grasp, score=scores[grasp.grasp_id], score_components={"score": scores[grasp.grasp_id]})
                    for grasp in grasps
                ],
                key=lambda grasp: (float(grasp.score), grasp.grasp_id),
                reverse=True,
            )

        with (
            mock.patch.object(regrasp_fallback, "evaluate_saved_grasps_against_pickup_pose", side_effect=accept_all),
            mock.patch.object(fabrica_pipeline, "score_grasps", side_effect=fake_score_grasps),
        ):
            plan = regrasp_fallback.plan_mujoco_regrasp_fallback(
                stage1=stage1,
                direct_stage2=direct_stage2,
                planning=PlanningConfig(
                    top_grasp_score_weight=0.1,
                    regrasp_transfer_top_grasp_score_weight=0.9,
                ),
                force=True,
                max_orientations=4,
            )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.transfer_grasp.grasp_id, "top_down_transfer")
        self.assertAlmostEqual(plan.transfer_grasp.score_components["top_down_approach"], 1.0)
        self.assertAlmostEqual(plan.transfer_grasp.score_components["top_grasp_score_weight"], 0.9)
        self.assertAlmostEqual(plan.metadata["transfer_top_grasp_score_weight"], 0.9)
        self.assertAlmostEqual(plan.metadata["final_top_grasp_score_weight"], 0.1)

    def test_force_regrasp_does_not_fall_back_to_direct_pose(self) -> None:
        transfer = self._candidate("transfer")
        final = self._candidate("final")
        bundle = SavedGraspBundle(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=(0.0, 0.0, 0.0),
            source_frame_orientation_xyzw_obj_world=(0.0, 0.0, 0.0, 1.0),
            candidates=(final,),
            metadata={},
        )
        direct_pose = ObjectWorldPose(
            position_world=(0.0, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        stage1 = fabrica_pipeline.Stage1Result(
            bundle=bundle,
            target_mesh_local=self._cube_mesh(),
            target_pose_in_obj_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.0),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
            obstacle_mesh_world=None,
            collision_backend_name="unit-test",
            raw_candidate_count=2,
            raw_candidates=(transfer, final),
        )
        direct_stage2 = SimpleNamespace(accepted=(final,), pickup_pose_world=direct_pose)

        def accept_all(grasps, **_kwargs):
            return [CandidateStatus(grasp=grasp, status="accepted", reason="unit_test") for grasp in grasps]

        with (
            mock.patch.object(regrasp_fallback, "_convex_hull_facets", return_value=[]),
            mock.patch.object(regrasp_fallback, "evaluate_saved_grasps_against_pickup_pose", side_effect=accept_all),
        ):
            plan = regrasp_fallback.plan_mujoco_regrasp_fallback(
                stage1=stage1,
                direct_stage2=direct_stage2,
                planning=PlanningConfig(),
                force=True,
            )

        self.assertIsNone(plan)

    def test_run_sim_force_regrasp_writes_plan_and_passes_it_to_mujoco(self) -> None:
        payload = {
            "geometry": {"target_mesh_path": "obj/fabrica/beam/2.obj", "mesh_scale": 0.01},
            "planning": {},
            "pickup_pose": {"support_face": "neg_z", "yaw_deg": 0.0, "xy_world": [0.5, 0.0]},
            "artifacts": {
                "stage1_json": "artifacts/test_stage1.json",
                "stage1_html": "artifacts/test_stage1.html",
                "stage2_json": "artifacts/test_stage2.json",
                "stage2_html": "artifacts/test_stage2.html",
            },
            "mujoco_execution": {
                "enabled": True,
                "robot_config": "configs/mujoco_fr3_with_hand.json",
                "force_regrasp_fallback": True,
                "regrasp_plan_artifact": "artifacts/forced_regrasp_plan.json",
            },
            "isaac_execution": {"enabled": False},
        }
        stage1 = SimpleNamespace(
            bundle=SimpleNamespace(candidates=("assembly",)),
            raw_candidate_count=1,
            target_mesh_local=SimpleNamespace(vertices_obj=np.array([[0.0, 0.0, 0.0]], dtype=float)),
        )
        stage2 = SimpleNamespace(
            accepted=("direct",),
            source_bundle=SimpleNamespace(candidates=("assembly",)),
            pickup_pose_world=ObjectWorldPose(
                position_world=(0.5, 0.0, 0.0),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        plan = SimpleNamespace(
            transfer_grasp=SimpleNamespace(grasp_id="transfer"),
            final_grasp=SimpleNamespace(grasp_id="final"),
        )

        with (
            mock.patch.object(run_grasp_pipeline, "generate_stage1_result", return_value=stage1),
            mock.patch.object(run_grasp_pipeline, "write_stage1_artifacts"),
            mock.patch.object(run_grasp_pipeline, "recheck_stage2_result", return_value=stage2),
            mock.patch.object(run_grasp_pipeline, "write_stage2_artifacts"),
            mock.patch.object(run_grasp_pipeline, "_write_part_frame_debug_artifact"),
            mock.patch.object(run_grasp_pipeline, "plan_mujoco_regrasp_fallback", return_value=plan) as plan_fallback,
            mock.patch.object(run_grasp_pipeline, "write_mujoco_regrasp_plan") as write_plan,
            mock.patch.object(run_grasp_pipeline, "write_mujoco_regrasp_debug_html") as write_html,
            mock.patch.object(run_grasp_pipeline, "_run_mujoco_execution") as run_mujoco,
            mock.patch.object(run_grasp_pipeline, "_run_isaac_execution"),
        ):
            run_grasp_pipeline.run_sim(payload, headless=True)

        plan_fallback.assert_called_once()
        self.assertTrue(plan_fallback.call_args.kwargs["force"])
        write_plan.assert_called_once()
        self.assertEqual(write_plan.call_args.args[1], Path("artifacts/forced_regrasp_plan.json"))
        write_html.assert_called_once()
        self.assertEqual(write_html.call_args.kwargs["output_html"], Path("artifacts/forced_regrasp_plan.html"))
        run_mujoco.assert_called_once()
        self.assertEqual(run_mujoco.call_args.kwargs["regrasp_plan_json"], Path("artifacts/forced_regrasp_plan.json"))

    def test_regrasp_plan_triggers_when_stage2_rejects_all(self) -> None:
        plan = SimpleNamespace(
            transfer_grasp=SimpleNamespace(grasp_id="transfer"),
            final_grasp=SimpleNamespace(grasp_id="final"),
        )
        mujoco_execution = run_grasp_pipeline.MujocoPipelineConfig(
            enabled=True,
            robot_config="configs/mujoco_fr3_with_hand.json",
            regrasp_plan_artifact="artifacts/empty_stage2_regrasp_plan.json",
        )

        with (
            mock.patch.object(run_grasp_pipeline, "plan_mujoco_regrasp_fallback", return_value=plan) as plan_fallback,
            mock.patch.object(run_grasp_pipeline, "write_mujoco_regrasp_plan") as write_plan,
            mock.patch.object(run_grasp_pipeline, "write_mujoco_regrasp_debug_html") as write_html,
        ):
            output_path = run_grasp_pipeline._maybe_write_mujoco_regrasp_plan(
                stage1=SimpleNamespace(),
                stage2=SimpleNamespace(accepted=()),
                planning=PlanningConfig(),
                mujoco_execution=mujoco_execution,
                input_stage2_json=Path("artifacts/test_stage2.json"),
            )

        self.assertEqual(output_path, Path("artifacts/empty_stage2_regrasp_plan.json"))
        self.assertFalse(plan_fallback.call_args.kwargs["force"])
        write_plan.assert_called_once()
        write_html.assert_called_once()

    def test_force_regrasp_writes_plan_even_when_backend_none_disables_mujoco(self) -> None:
        plan = SimpleNamespace(
            transfer_grasp=SimpleNamespace(grasp_id="transfer"),
            final_grasp=SimpleNamespace(grasp_id="final"),
        )
        mujoco_execution = run_grasp_pipeline.MujocoPipelineConfig(
            enabled=False,
            force_regrasp_fallback=True,
            regrasp_plan_artifact="artifacts/backend_none_regrasp_plan.json",
        )

        with (
            mock.patch.object(run_grasp_pipeline, "plan_mujoco_regrasp_fallback", return_value=plan) as plan_fallback,
            mock.patch.object(run_grasp_pipeline, "write_mujoco_regrasp_plan") as write_plan,
            mock.patch.object(run_grasp_pipeline, "write_mujoco_regrasp_debug_html") as write_html,
        ):
            output_path = run_grasp_pipeline._maybe_write_mujoco_regrasp_plan(
                stage1=SimpleNamespace(),
                stage2=SimpleNamespace(accepted=("direct",)),
                planning=PlanningConfig(),
                mujoco_execution=mujoco_execution,
                input_stage2_json=Path("artifacts/test_stage2.json"),
            )

        self.assertEqual(output_path, Path("artifacts/backend_none_regrasp_plan.json"))
        self.assertTrue(plan_fallback.call_args.kwargs["force"])
        write_plan.assert_called_once()
        write_html.assert_called_once()

    def test_force_regrasp_raises_when_no_different_surface_plan_exists(self) -> None:
        mujoco_execution = run_grasp_pipeline.MujocoPipelineConfig(
            enabled=True,
            robot_config="configs/mujoco_fr3_with_hand.json",
            force_regrasp_fallback=True,
        )

        with mock.patch.object(run_grasp_pipeline, "plan_mujoco_regrasp_fallback", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "different support surface"):
                run_grasp_pipeline._maybe_write_mujoco_regrasp_plan(
                    stage1=SimpleNamespace(),
                    stage2=SimpleNamespace(accepted=("direct",)),
                    planning=PlanningConfig(),
                    mujoco_execution=mujoco_execution,
                    input_stage2_json=Path("artifacts/test_stage2.json"),
                )

    def test_regrasp_debug_html_writes_side_by_side_grasp_view(self) -> None:
        transfer = self._candidate("transfer")
        final = self._candidate("final")
        mesh = self._cube_mesh()
        initial_pose = ObjectWorldPose(
            position_world=(0.0, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        staging_pose = ObjectWorldPose(
            position_world=(0.08, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )
        bundle = SavedGraspBundle(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=(0.0, 0.0, 0.0),
            source_frame_orientation_xyzw_obj_world=(0.0, 0.0, 0.0, 1.0),
            candidates=(final,),
            metadata={},
        )
        stage1 = fabrica_pipeline.Stage1Result(
            bundle=bundle,
            target_mesh_local=mesh,
            target_pose_in_obj_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.0),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
            obstacle_mesh_world=None,
            collision_backend_name="unit-test",
            raw_candidate_count=2,
            raw_candidates=(transfer, final),
        )
        plan = regrasp_fallback.MujocoRegraspFallbackPlan(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=(0.0, 0.0, 0.0),
            source_frame_orientation_xyzw_obj_world=(0.0, 0.0, 0.0, 1.0),
            initial_object_pose_world=initial_pose,
            staging_object_pose_world=staging_pose,
            support_facet=regrasp_fallback.HullSupportFacet(
                normal_obj=(0.0, 0.0, -1.0),
                area_m2=0.0016,
                vertex_indices=(0, 1, 2, 3),
                vertices_obj=tuple(tuple(float(v) for v in vertex) for vertex in mesh.vertices_obj[:4]),
                com_obj=(0.0, 0.0, 0.0),
                com_projection_obj=(0.0, 0.0, -0.02),
                stability_margin_m=0.01,
                yaw_deg=0.0,
            ),
            transfer_grasp=transfer,
            final_grasp=final,
            metadata={"reason": "unit_test"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_html = Path(temp_dir) / "regrasp.html"
            regrasp_debug_html.write_mujoco_regrasp_debug_html(
                plan=plan,
                stage1=stage1,
                planning=PlanningConfig(floor_clearance_margin_m=-0.01),
                output_html=output_html,
            )
            html = output_html.read_text(encoding="utf-8")

        self.assertIn("Fabrica MuJoCo Regrasp Debug", html)
        self.assertIn("Initial Pose", html)
        self.assertIn("Staging Pose", html)
        self.assertIn("Transfer grasps", html)
        self.assertIn("Final assembly grasps", html)
        self.assertIn("transfer", html)
        self.assertIn("final", html)


class MujocoBundleExecutionPoseTests(unittest.TestCase):
    def test_resolve_object_pose_world_from_bundle_prefers_exact_execution_pose(self) -> None:
        bundle = SavedGraspBundle(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=(0.0, 0.0, 0.0),
            source_frame_orientation_xyzw_obj_world=(0.0, 0.0, 0.0, 1.0),
            candidates=(),
            metadata={
                "execution_world_pose": {
                    "position_world": [0.1, -0.2, 0.3],
                    "orientation_xyzw_world": [0.0, 0.0, 0.70710678, 0.70710678],
                }
            },
        )
        args = SimpleNamespace(
            support_face="",
            yaw_deg=None,
            xy_world="",
            allowed_support_faces="pos_x,neg_x,pos_y,neg_y,neg_z",
            allowed_yaw_deg="0,90,180,270",
            xy_min_world="-0.45,-0.05",
            xy_max_world="-0.35,0.05",
            seed=0,
        )

        placement_spec, object_pose_world = run_fabrica_grasp_in_mujoco._resolve_object_pose_world_from_bundle(
            args,
            bundle,
            mesh_local=object(),
        )

        self.assertEqual(placement_spec.support_face, "explicit_pose")
        self.assertEqual(object_pose_world.position_world, (0.1, -0.2, 0.3))
        np.testing.assert_allclose(
            object_pose_world.orientation_xyzw_world,
            (0.0, 0.0, 0.70710678, 0.70710678),
            atol=1.0e-8,
        )

    def test_load_simulation_defaults_reads_shared_yaml(self) -> None:
        sim_cfg = run_fabrica_grasp_in_mujoco._load_simulation_defaults(Path("configs/mujoco_simulation.yaml"))

        self.assertEqual(sim_cfg["pregrasp_offset"], 0.05)
        self.assertEqual(sim_cfg["gripper_width_clearance"], 0.01)
        self.assertEqual(sim_cfg["contact_gap_m"], 0.002)
        self.assertEqual(sim_cfg["robot_cfg_updates"]["control_substeps"], 8)
        self.assertEqual(sim_cfg["execution_cfg_kwargs"]["object_mass_kg"], 0.15)
        self.assertEqual(sim_cfg["execution_cfg_kwargs"]["object_friction"], (7.5, 0.16, 0.03))
        self.assertEqual(sim_cfg["execution_cfg_kwargs"]["arm_speed_scale"], 5.0)
        self.assertEqual(sim_cfg["execution_cfg_kwargs"]["regrasp_transport_clearance_m"], 0.22)

    def test_trajectory_waypoints_for_joints_reorders_moveit_points(self) -> None:
        point_a = SimpleNamespace(positions=(1.0, 2.0, 3.0))
        point_b = SimpleNamespace(positions=(4.0, 5.0, 6.0))
        trajectory = SimpleNamespace(
            joint_trajectory=SimpleNamespace(
                joint_names=("fr3_joint2", "fr3_joint1", "fr3_joint3"),
                points=(point_a, point_b),
            )
        )

        waypoints = run_fabrica_grasp_in_mujoco._trajectory_waypoints_for_joints(
            trajectory,
            joint_names=("fr3_joint1", "fr3_joint2", "fr3_joint3"),
        )

        self.assertEqual(waypoints, ((2.0, 1.0, 3.0), (5.0, 4.0, 6.0)))

    def test_mujoco_script_retries_next_grasp_after_failed_attempt(self) -> None:
        object_pose_world = ObjectWorldPose(
            position_world=(0.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        placement_spec = SimpleNamespace(support_face="explicit_pose", yaw_deg=0.0, xy_world=(0.0, 0.0))
        candidates = [SimpleNamespace(grasp_id="g0001"), SimpleNamespace(grasp_id="g0002")]
        world_grasps = {
            "g0001": SimpleNamespace(
                grasp_id="g0001",
                position_w=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                pregrasp_position_w=(0.0, 0.0, 0.1),
                gripper_width=0.04,
                jaw_width=0.03,
            ),
            "g0002": SimpleNamespace(
                grasp_id="g0002",
                position_w=(0.0, 0.0, 0.0),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                pregrasp_position_w=(0.0, 0.0, 0.1),
                gripper_width=0.04,
                jaw_width=0.03,
            ),
        }
        failed = SimpleNamespace(
            success=False,
            status="pregrasp_failed",
            message="failed",
            pregrasp_reached=False,
            grasp_reached=False,
            initial_object_position_world=(0.0, 0.0, 0.0),
            final_object_position_world=(0.0, 0.0, 0.0),
            lift_height_m=0.0,
            target_lift_height_m=0.05,
            position_error_m=0.1,
            orientation_error_rad=0.2,
            generated_scene_xml=None,
        )
        succeeded = SimpleNamespace(
            success=True,
            status="ok",
            message="ok",
            pregrasp_reached=True,
            grasp_reached=True,
            initial_object_position_world=(0.0, 0.0, 0.0),
            final_object_position_world=(0.0, 0.0, 0.06),
            lift_height_m=0.06,
            target_lift_height_m=0.05,
            position_error_m=None,
            orientation_error_rad=None,
            generated_scene_xml=None,
        )

        argv = [
            "run_fabrica_grasp_in_mujoco.py",
            "--input-json",
            "artifacts/test_stage2.json",
            "--robot-config",
            "configs/mujoco_fr3_with_hand.json",
        ]
        with (
            mock.patch.object(run_fabrica_grasp_in_mujoco.sys, "argv", argv),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco, "load_grasp_bundle", return_value=SimpleNamespace(candidates=candidates)
            ),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "build_bundle_local_mesh", return_value=object()),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "write_temporary_triangle_mesh_stl",
                return_value=Path("artifacts/tmp_object.stl"),
            ),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "_resolve_object_pose_world_from_bundle",
                return_value=(placement_spec, object_pose_world),
            ),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "evaluate_saved_grasps_against_pickup_pose",
                return_value=[
                    SimpleNamespace(status="accepted", grasp=candidates[0]),
                    SimpleNamespace(status="accepted", grasp=candidates[1]),
                ],
            ),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "score_grasps", return_value=candidates),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_robot_config", return_value=object()),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "saved_grasp_to_world_grasp",
                side_effect=lambda grasp, *_args, **_kwargs: world_grasps[grasp.grasp_id],
            ),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "run_world_grasp_in_mujoco",
                side_effect=[failed, succeeded],
            ) as run_attempt,
            mock.patch.object(run_fabrica_grasp_in_mujoco, "_write_attempt_artifact") as write_artifact,
            mock.patch.object(Path, "unlink"),
        ):
            run_fabrica_grasp_in_mujoco.main()

        self.assertEqual(run_attempt.call_count, 2)
        self.assertEqual(run_attempt.call_args.kwargs["world_grasp"].grasp_id, "g0002")
        write_artifact.assert_called_once()
        self.assertEqual(len(write_artifact.call_args.kwargs["attempts"]), 2)
        self.assertEqual(write_artifact.call_args.kwargs["result"].status, "ok")

    def test_mujoco_script_executes_regrasp_plan_branch(self) -> None:
        initial_pose = ObjectWorldPose(
            position_world=(0.4, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        staging_pose = ObjectWorldPose(
            position_world=(0.5, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )
        transfer = SimpleNamespace(grasp_id="transfer")
        final = SimpleNamespace(grasp_id="final")
        plan = SimpleNamespace(
            transfer_grasp=transfer,
            final_grasp=final,
            initial_object_pose_world=initial_pose,
            staging_object_pose_world=staging_pose,
        )
        world_grasps = [
            SimpleNamespace(grasp_id="transfer_initial"),
            SimpleNamespace(grasp_id="transfer_staging"),
            SimpleNamespace(grasp_id="final_world"),
        ]
        result = SimpleNamespace(success=True, status="ok", message="regrasp ok")
        argv = [
            "run_fabrica_grasp_in_mujoco.py",
            "--input-json",
            "artifacts/test_stage2.json",
            "--regrasp-plan-json",
            "artifacts/regrasp_plan.json",
            "--robot-config",
            "configs/mujoco_fr3_with_hand.json",
        ]

        with (
            mock.patch.object(run_fabrica_grasp_in_mujoco.sys, "argv", argv),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_grasp_bundle", return_value=SimpleNamespace()),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "build_bundle_local_mesh", return_value=object()),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "write_temporary_triangle_mesh_stl",
                return_value=Path("artifacts/tmp_object.stl"),
            ),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_robot_config", return_value=object()),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_mujoco_regrasp_plan", return_value=plan),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "saved_grasp_to_world_grasp",
                side_effect=world_grasps,
            ) as to_world,
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "run_regrasp_plan_in_mujoco",
                return_value=result,
            ) as run_regrasp,
            mock.patch.object(run_fabrica_grasp_in_mujoco, "_write_regrasp_attempt_artifact") as write_artifact,
            mock.patch.object(run_fabrica_grasp_in_mujoco, "evaluate_saved_grasps_against_pickup_pose") as evaluate,
            mock.patch.object(Path, "unlink"),
        ):
            run_fabrica_grasp_in_mujoco.main()

        self.assertEqual(to_world.call_count, 3)
        run_regrasp.assert_called_once()
        self.assertEqual(run_regrasp.call_args.kwargs["initial_object_pose_world"], initial_pose)
        self.assertEqual(run_regrasp.call_args.kwargs["transfer_initial_grasp"].grasp_id, "transfer_initial")
        self.assertEqual(run_regrasp.call_args.kwargs["transfer_staging_grasp"].grasp_id, "transfer_staging")
        self.assertEqual(run_regrasp.call_args.kwargs["final_grasp"].grasp_id, "final_world")
        self.assertIsNone(run_regrasp.call_args.kwargs["moveit_joint_trajectories"])
        write_artifact.assert_called_once()
        evaluate.assert_not_called()

    def test_mujoco_script_executes_regrasp_plan_branch_with_moveit(self) -> None:
        initial_pose = ObjectWorldPose(
            position_world=(0.4, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        staging_pose = ObjectWorldPose(
            position_world=(0.5, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )
        transfer = SimpleNamespace(grasp_id="transfer")
        final = SimpleNamespace(grasp_id="final")
        plan = SimpleNamespace(
            transfer_grasp=transfer,
            final_grasp=final,
            initial_object_pose_world=initial_pose,
            staging_object_pose_world=staging_pose,
        )
        world_grasps = [
            SimpleNamespace(grasp_id="transfer_initial"),
            SimpleNamespace(grasp_id="transfer_staging"),
            SimpleNamespace(grasp_id="final_world"),
        ]
        moveit_trajectories = {"transfer_pregrasp": ((0.0,),), "final_lift": ((1.0,),)}
        result = SimpleNamespace(success=True, status="ok", message="regrasp ok")
        argv = [
            "run_fabrica_grasp_in_mujoco.py",
            "--input-json",
            "artifacts/test_stage2.json",
            "--regrasp-plan-json",
            "artifacts/regrasp_plan.json",
            "--robot-config",
            "configs/mujoco_fr3_with_hand.json",
            "--controller",
            "moveit",
        ]

        with (
            mock.patch.object(run_fabrica_grasp_in_mujoco.sys, "argv", argv),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_grasp_bundle", return_value=SimpleNamespace()),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "build_bundle_local_mesh", return_value=object()),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "write_temporary_triangle_mesh_stl",
                return_value=Path("artifacts/tmp_object.stl"),
            ),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_robot_config", return_value=object()),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_mujoco_regrasp_plan", return_value=plan),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "saved_grasp_to_world_grasp",
                side_effect=world_grasps,
            ),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "_plan_moveit_regrasp_joint_trajectories",
                return_value=moveit_trajectories,
            ) as plan_moveit,
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "run_regrasp_plan_in_mujoco",
                return_value=result,
            ) as run_regrasp,
            mock.patch.object(run_fabrica_grasp_in_mujoco, "_write_regrasp_attempt_artifact"),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "evaluate_saved_grasps_against_pickup_pose") as evaluate,
            mock.patch.object(Path, "unlink"),
        ):
            run_fabrica_grasp_in_mujoco.main()

        plan_moveit.assert_called_once()
        run_regrasp.assert_called_once()
        self.assertIs(run_regrasp.call_args.kwargs["moveit_joint_trajectories"], moveit_trajectories)
        evaluate.assert_not_called()

    def test_moveit_joint_trajectory_diagnostics_reports_joint_path_shape(self) -> None:
        trajectories = {
            "transfer_transport_rotate": (
                (0.0, 0.0, 0.0),
                (0.2, -0.1, 0.3),
                (0.5, -0.1, 0.1),
            ),
            "empty": (),
        }

        diagnostics = {
            item["label"]: item
            for item in run_fabrica_grasp_in_mujoco._moveit_joint_trajectory_diagnostics(trajectories)
        }

        rotate = diagnostics["transfer_transport_rotate"]
        expected_path_length = float(np.linalg.norm((0.2, -0.1, 0.3)) + np.linalg.norm((0.3, 0.0, -0.2)))
        self.assertEqual(rotate["point_count"], 3)
        self.assertAlmostEqual(rotate["joint_path_length_rad"], expected_path_length)
        self.assertAlmostEqual(rotate["max_joint_step_rad"], 0.3)
        self.assertEqual(rotate["start_joint_positions"], [0.0, 0.0, 0.0])
        self.assertEqual(rotate["end_joint_positions"], [0.5, -0.1, 0.1])

        empty = diagnostics["empty"]
        self.assertEqual(empty["point_count"], 0)
        self.assertEqual(empty["joint_path_length_rad"], 0.0)
        self.assertEqual(empty["max_joint_step_rad"], 0.0)

    def test_mujoco_script_executes_cheapest_planned_regrasp_candidate(self) -> None:
        initial_pose = ObjectWorldPose(
            position_world=(0.4, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        expensive_staging_pose = ObjectWorldPose(
            position_world=(0.5, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        cheap_staging_pose = ObjectWorldPose(
            position_world=(0.56, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        transfer = SimpleNamespace(grasp_id="transfer")
        final = SimpleNamespace(grasp_id="final")
        expensive_option = SimpleNamespace(
            staging_object_pose_world=expensive_staging_pose,
            support_facet=None,
            transfer_grasp=transfer,
            final_grasp=final,
            transfer_grasp_candidates=(transfer,),
            final_grasp_candidates=(final,),
            metadata={"placement_score": 0.0},
        )
        cheap_option = SimpleNamespace(
            staging_object_pose_world=cheap_staging_pose,
            support_facet=None,
            transfer_grasp=transfer,
            final_grasp=final,
            transfer_grasp_candidates=(transfer,),
            final_grasp_candidates=(final,),
            metadata={"placement_score": 0.0},
        )
        plan = SimpleNamespace(
            transfer_grasp=transfer,
            final_grasp=final,
            transfer_grasp_candidates=(transfer,),
            final_grasp_candidates=(final,),
            initial_object_pose_world=initial_pose,
            staging_object_pose_world=expensive_staging_pose,
            placement_options=(expensive_option, cheap_option),
        )

        def to_world(grasp, object_pose_world, **_kwargs):
            return SimpleNamespace(grasp_id=f"{grasp.grasp_id}_{object_pose_world.position_world[0]:.2f}")

        expensive_trajectory = {
            "transfer_lift": ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            "placement": ((2.0, 0.0), (2.5, 0.0)),
        }
        cheap_trajectory = {
            "transfer_lift": ((0.0, 0.0), (0.1, 0.0)),
            "placement": ((0.1, 0.0), (0.2, 0.0)),
        }
        result = SimpleNamespace(success=True, status="ok", message="regrasp ok")
        argv = [
            "run_fabrica_grasp_in_mujoco.py",
            "--input-json",
            "artifacts/test_stage2.json",
            "--regrasp-plan-json",
            "artifacts/regrasp_plan.json",
            "--robot-config",
            "configs/mujoco_fr3_with_hand.json",
            "--controller",
            "moveit",
        ]

        with (
            mock.patch.object(run_fabrica_grasp_in_mujoco.sys, "argv", argv),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_grasp_bundle", return_value=SimpleNamespace()),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "build_bundle_local_mesh", return_value=object()),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "write_temporary_triangle_mesh_stl",
                return_value=Path("artifacts/tmp_object.stl"),
            ),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_robot_config", return_value=object()),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_mujoco_regrasp_plan", return_value=plan),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "saved_grasp_to_world_grasp", side_effect=to_world),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "_plan_moveit_regrasp_joint_trajectories",
                side_effect=[expensive_trajectory, cheap_trajectory],
            ) as plan_moveit,
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "run_regrasp_plan_in_mujoco",
                return_value=result,
            ) as run_regrasp,
            mock.patch.object(run_fabrica_grasp_in_mujoco, "_write_regrasp_attempt_artifact") as write_artifact,
            mock.patch.object(Path, "unlink"),
        ):
            run_fabrica_grasp_in_mujoco.main()

        self.assertEqual(plan_moveit.call_count, 2)
        run_regrasp.assert_called_once()
        self.assertIs(run_regrasp.call_args.kwargs["staging_object_pose_world"], cheap_staging_pose)
        self.assertIs(run_regrasp.call_args.kwargs["moveit_joint_trajectories"], cheap_trajectory)
        planned_candidates = write_artifact.call_args.kwargs["planned_candidates"]
        self.assertEqual(len(planned_candidates), 2)
        self.assertEqual(planned_candidates[1]["execution_rank"], 1)
        self.assertLess(planned_candidates[1]["path_cost"], planned_candidates[0]["path_cost"])

    def test_regrasp_result_payload_includes_trajectory_diagnostics(self) -> None:
        result = run_fabrica_grasp_in_mujoco.MujocoRegraspAttemptResult(
            success=False,
            status="moveit_staging_transport_failed",
            message="failed",
            transfer_pregrasp_reached=True,
            transfer_grasp_reached=True,
            transfer_lift_reached=True,
            placement_reached=False,
            final_pregrasp_reached=False,
            final_grasp_reached=False,
            initial_object_position_world=(0.4, 0.0, 0.02),
            staged_object_position_world=(0.5, 0.0, 0.02),
            final_object_position_world=(0.4, 0.0, 0.02),
            final_lift_height_m=0.0,
            target_lift_height_m=0.05,
            trajectory_diagnostics=(
                {
                    "label": "staging_transport",
                    "joint_path_length_rad": 2.4,
                    "tcp_min_z_m": 0.18,
                    "observed_object_center_min_z_m": 0.01,
                },
            ),
        )

        payload = run_fabrica_grasp_in_mujoco._object_payload(result)

        self.assertEqual(payload["trajectory_diagnostics"][0]["label"], "staging_transport")
        self.assertEqual(payload["trajectory_diagnostics"][0]["joint_path_length_rad"], 2.4)

    def test_mujoco_script_retries_regrasp_candidates_after_moveit_planning_failure(self) -> None:
        initial_pose = ObjectWorldPose(
            position_world=(0.4, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        staging_pose = ObjectWorldPose(
            position_world=(0.5, 0.0, 0.02),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )
        bad_transfer = SimpleNamespace(grasp_id="bad_transfer")
        good_transfer = SimpleNamespace(grasp_id="good_transfer")
        final = SimpleNamespace(grasp_id="final")
        plan = SimpleNamespace(
            transfer_grasp=bad_transfer,
            final_grasp=final,
            transfer_grasp_candidates=(bad_transfer, good_transfer),
            final_grasp_candidates=(final,),
            initial_object_pose_world=initial_pose,
            staging_object_pose_world=staging_pose,
        )

        def to_world(grasp, object_pose_world, **_kwargs):
            pose_name = "initial" if object_pose_world is initial_pose else "staging"
            return SimpleNamespace(grasp_id=f"{grasp.grasp_id}_{pose_name}")

        moveit_trajectories = {"transfer_pregrasp": ((0.0,),), "final_lift": ((1.0,),)}
        result = SimpleNamespace(success=True, status="ok", message="regrasp ok")
        argv = [
            "run_fabrica_grasp_in_mujoco.py",
            "--input-json",
            "artifacts/test_stage2.json",
            "--regrasp-plan-json",
            "artifacts/regrasp_plan.json",
            "--robot-config",
            "configs/mujoco_fr3_with_hand.json",
            "--controller",
            "moveit",
        ]

        with (
            mock.patch.object(run_fabrica_grasp_in_mujoco.sys, "argv", argv),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_grasp_bundle", return_value=SimpleNamespace()),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "build_bundle_local_mesh", return_value=object()),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "write_temporary_triangle_mesh_stl",
                return_value=Path("artifacts/tmp_object.stl"),
            ),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_robot_config", return_value=object()),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "load_mujoco_regrasp_plan", return_value=plan),
            mock.patch.object(run_fabrica_grasp_in_mujoco, "saved_grasp_to_world_grasp", side_effect=to_world),
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "_plan_moveit_regrasp_joint_trajectories",
                side_effect=[RuntimeError("MoveIt failed to plan transfer_pregrasp: IK failed"), moveit_trajectories],
            ) as plan_moveit,
            mock.patch.object(
                run_fabrica_grasp_in_mujoco,
                "run_regrasp_plan_in_mujoco",
                return_value=result,
            ) as run_regrasp,
            mock.patch.object(run_fabrica_grasp_in_mujoco, "_write_regrasp_attempt_artifact") as write_artifact,
            mock.patch.object(Path, "unlink"),
        ):
            run_fabrica_grasp_in_mujoco.main()

        self.assertEqual(plan_moveit.call_count, 2)
        run_regrasp.assert_called_once()
        self.assertEqual(run_regrasp.call_args.kwargs["transfer_initial_grasp"].grasp_id, "good_transfer_initial")
        write_artifact.assert_called_once()
        attempts = write_artifact.call_args.kwargs["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["status"], "moveit_planning_failed")
        self.assertEqual(attempts[0]["transfer_grasp_id"], "bad_transfer")
        self.assertEqual(attempts[1]["transfer_grasp_id"], "good_transfer")
        self.assertEqual(write_artifact.call_args.kwargs["plan"].transfer_grasp.grasp_id, "good_transfer")
        self.assertEqual(write_artifact.call_args.kwargs["plan"].final_grasp.grasp_id, "final")


if __name__ == "__main__":
    unittest.main()
