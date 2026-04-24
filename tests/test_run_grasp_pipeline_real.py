from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from grasp_planning.grasping.world_constraints import ObjectWorldPose
from scripts import run_grasp_pipeline


class RunGraspPipelineRealTests(unittest.TestCase):
    def test_real_execution_config_parses_defaults_and_stop_after(self) -> None:
        config = run_grasp_pipeline._real_execution_config({"real_execution": {"enabled": True, "stop_after": "grasp"}})

        self.assertTrue(config.enabled)
        self.assertEqual(config.stop_after, "grasp")
        self.assertEqual(config.frame_id, "base")
        self.assertAlmostEqual(config.velocity_scale, 0.05)

    def test_real_execution_config_rejects_invalid_stop_after(self) -> None:
        with self.assertRaises(ValueError):
            run_grasp_pipeline._real_execution_config({"real_execution": {"stop_after": "unsupported"}})

    def test_resolve_object_pose_world_reads_single_debug_frame_topic(self) -> None:
        ros2 = run_grasp_pipeline.Ros2Config(
            debug_frame_topic="/perception/fp/debug_frame/{object_id}",
            frame_id="world",
            timeout_s=1.0,
            object_id="cooling_screw",
        )
        object_pose = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.2),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )

        with mock.patch.object(
            run_grasp_pipeline,
            "wait_for_debug_frame_pose_message",
            return_value=object_pose,
        ) as wait_pose:
            object_pose_world = run_grasp_pipeline._resolve_object_pose_world(ros2)

        self.assertEqual(object_pose_world, object_pose)
        wait_pose.assert_called_once_with(
            topic_name="/perception/fp/debug_frame/cooling_screw",
            message_type=run_grasp_pipeline.DEBUG_FRAME_MESSAGE_TYPE,
            object_id="cooling_screw",
            timeout_s=1.0,
        )

    def test_resolve_object_pose_world_requires_debug_frame_topic_and_object_id(self) -> None:
        with self.assertRaises(ValueError):
            run_grasp_pipeline._resolve_object_pose_world(
                run_grasp_pipeline.Ros2Config(debug_frame_topic="", object_id="cooling_screw")
            )

        with self.assertRaises(ValueError):
            run_grasp_pipeline._resolve_object_pose_world(
                run_grasp_pipeline.Ros2Config(
                    debug_frame_topic="/perception/fp/debug_frame/zed2i_2",
                    object_id="",
                )
            )

    def test_ros2_config_does_not_fallback_to_legacy_topic_keys(self) -> None:
        config = run_grasp_pipeline._ros2_config(
            {
                "ros2": {
                    "execution_frame_topic": "/legacy/execution",
                    "object_pose_topic": "/legacy/object_pose",
                    "object_id": "cooling_screw",
                }
            }
        )

        self.assertEqual(config.debug_frame_topic, "")
        self.assertEqual(config.object_id, "cooling_screw")

    def test_run_real_executes_stage2_bundle_when_real_execution_enabled(self) -> None:
        payload = {
            "geometry": {"target_mesh_path": "obj/fabrica/beam/2.obj", "mesh_scale": 0.01},
            "planning": {},
            "artifacts": {
                "stage1_json": "artifacts/test_stage1.json",
                "stage1_html": "artifacts/test_stage1.html",
                "stage2_json": "artifacts/test_stage2.json",
                "stage2_html": "artifacts/test_stage2.html",
            },
            "ros2": {},
            "real_execution": {"enabled": True, "require_confirmation": False, "stop_after": "pregrasp"},
        }
        object_pose_world = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.2),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        stage1 = SimpleNamespace(bundle=SimpleNamespace(candidates=("g0001",)), raw_candidate_count=1)
        stage2 = SimpleNamespace(source_bundle=SimpleNamespace(candidates=("g0001",)), accepted=("g0001",))
        execution_result = SimpleNamespace(
            success=True,
            status="stopped_at_pregrasp",
            grasp_id="g0001",
            message="ok",
            attempt_artifact_path=Path("artifacts/attempt.json"),
        )

        with (
            mock.patch.object(run_grasp_pipeline, "_resolve_object_pose_world", return_value=object_pose_world),
            mock.patch.object(run_grasp_pipeline, "generate_stage1_result", return_value=stage1) as generate_stage1,
            mock.patch.object(run_grasp_pipeline, "write_stage1_artifacts"),
            mock.patch.object(run_grasp_pipeline, "recheck_stage2_result", return_value=stage2),
            mock.patch.object(run_grasp_pipeline, "write_stage2_artifacts"),
            mock.patch.object(
                run_grasp_pipeline,
                "execute_real_grasp_from_bundle",
                return_value=execution_result,
            ) as execute_real,
        ):
            run_grasp_pipeline.run_real(payload)

        execute_real.assert_called_once()
        self.assertEqual(execute_real.call_args.kwargs["input_json"], Path("artifacts/test_stage2.json"))
        self.assertTrue(execute_real.call_args.kwargs["config"].enabled)
        self.assertNotIn("source_frame_pose_obj_world", generate_stage1.call_args.kwargs)

    def test_run_pitl_uses_mesh_defined_local_frame_and_debug_frame_world_pose(self) -> None:
        payload = {
            "geometry": {"target_mesh_path": "obj/fabrica/beam/2.obj", "mesh_scale": 0.01},
            "planning": {},
            "artifacts": {
                "stage1_json": "artifacts/test_stage1.json",
                "stage1_html": "artifacts/test_stage1.html",
                "stage2_json": "artifacts/test_stage2.json",
                "stage2_html": "artifacts/test_stage2.html",
            },
            "ros2": {},
            "mujoco_execution": {"enabled": False},
        }
        object_pose_world = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.2),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        stage1 = SimpleNamespace(bundle=SimpleNamespace(candidates=("g0001",)), raw_candidate_count=1)
        stage2 = SimpleNamespace(source_bundle=SimpleNamespace(candidates=("g0001",)), accepted=("g0001",))

        with (
            mock.patch.object(run_grasp_pipeline, "_resolve_object_pose_world", return_value=object_pose_world),
            mock.patch.object(run_grasp_pipeline, "generate_stage1_result", return_value=stage1) as generate_stage1,
            mock.patch.object(run_grasp_pipeline, "write_stage1_artifacts"),
            mock.patch.object(run_grasp_pipeline, "recheck_stage2_result", return_value=stage2) as recheck_stage2,
            mock.patch.object(run_grasp_pipeline, "write_stage2_artifacts"),
            mock.patch.object(run_grasp_pipeline, "_run_mujoco_execution"),
        ):
            run_grasp_pipeline.run_pitl(payload, headless=True)

        self.assertNotIn("source_frame_pose_obj_world", generate_stage1.call_args.kwargs)
        self.assertEqual(recheck_stage2.call_args.kwargs["object_pose_world"], object_pose_world)


if __name__ == "__main__":
    unittest.main()
