from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspBundle
from scripts import run_fabrica_grasp_in_mujoco, run_grasp_pipeline


class RunGraspPipelineModeTests(unittest.TestCase):
    def test_normalize_mode_maps_aliases(self) -> None:
        self.assertEqual(run_grasp_pipeline._normalize_mode("local"), "sim")
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

        self.assertEqual(sim_cfg["pregrasp_offset"], 0.20)
        self.assertEqual(sim_cfg["gripper_width_clearance"], 0.01)
        self.assertEqual(sim_cfg["contact_gap_m"], 0.002)
        self.assertEqual(sim_cfg["robot_cfg_updates"]["control_substeps"], 8)
        self.assertEqual(sim_cfg["execution_cfg_kwargs"]["object_mass_kg"], 0.15)
        self.assertEqual(sim_cfg["execution_cfg_kwargs"]["arm_speed_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
