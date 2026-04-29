from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import grasp_planning.pipeline.fabrica_pipeline as fabrica_pipeline
from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspBundle, SavedGraspCandidate
from grasp_planning.grasping.mesh_antipodal_grasp_generator import ObjectFrameGraspCandidate, TriangleMesh
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

    def test_planning_config_parses_top_grasp_score_weight(self) -> None:
        config = run_grasp_pipeline._planning_config({"planning": {"top_grasp_score_weight": 0.8}})

        self.assertAlmostEqual(config.top_grasp_score_weight, 0.8)

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
                planning=PlanningConfig(skip_stage1_collision_checks=True),
            )

        load_obstacles.assert_not_called()
        filter_assembly.assert_not_called()
        self.assertEqual(result.raw_candidate_count, 1)
        self.assertEqual(len(result.bundle.candidates), 1)
        self.assertTrue(result.bundle.metadata["stage1_collision_checks_skipped"])


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
        self.assertEqual(sim_cfg["execution_cfg_kwargs"]["arm_speed_scale"], 2.0)

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


if __name__ == "__main__":
    unittest.main()
