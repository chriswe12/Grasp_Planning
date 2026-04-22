from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

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

    def test_compose_object_world_poses_applies_translation(self) -> None:
        parent_pose_world = ObjectWorldPose(
            position_world=(1.0, 2.0, 3.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        child_pose_parent = ObjectWorldPose(
            position_world=(0.1, -0.2, 0.3),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )

        composed = run_grasp_pipeline._compose_object_world_poses(parent_pose_world, child_pose_parent)

        self.assertEqual(composed.position_world, (1.1, 1.8, 3.3))
        self.assertEqual(composed.orientation_xyzw_world, (0.0, 0.0, 0.0, 1.0))

    def test_compose_object_world_poses_rotates_child_translation(self) -> None:
        yaw_90 = (0.0, 0.0, 0.70710678, 0.70710678)
        parent_pose_world = ObjectWorldPose(
            position_world=(1.0, 0.0, 0.0),
            orientation_xyzw_world=yaw_90,
        )
        child_pose_parent = ObjectWorldPose(
            position_world=(0.2, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )

        composed = run_grasp_pipeline._compose_object_world_poses(parent_pose_world, child_pose_parent)

        np.testing.assert_allclose(composed.position_world, (1.0, 0.2, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(composed.orientation_xyzw_world, yaw_90, atol=1.0e-6)

    def test_resolve_real_world_frames_falls_back_to_legacy_pose_when_dual_topics_not_enabled(self) -> None:
        ros2 = run_grasp_pipeline.Ros2Config(
            object_pose_topic="/grasp_planning/object_pose",
            pose_message_type="geometry_msgs/msg/Pose",
            frame_id="world",
            timeout_s=1.0,
            object_id="",
            local_frame_offset_topic="",
            execution_frame_topic="",
        )
        legacy_pose = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.2),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )

        with (
            mock.patch.object(
                run_grasp_pipeline, "wait_for_object_pose_message", return_value=legacy_pose
            ) as wait_legacy,
            mock.patch.object(run_grasp_pipeline, "wait_for_real_frame_pair_messages") as wait_pair,
        ):
            source_frame_pose_obj_world, object_pose_world = run_grasp_pipeline._resolve_real_world_frames(ros2)

        self.assertIsNone(source_frame_pose_obj_world)
        self.assertEqual(object_pose_world, legacy_pose)
        wait_legacy.assert_called_once()
        wait_pair.assert_not_called()

    def test_resolve_real_world_frames_composes_execution_pose_with_source_frame_offset(self) -> None:
        ros2 = run_grasp_pipeline.Ros2Config(
            object_pose_topic="/grasp_planning/object_pose",
            pose_message_type="geometry_msgs/msg/Pose",
            frame_id="world",
            timeout_s=1.0,
            object_id="cooling_screw",
            local_frame_offset_topic="/perception/fp/mesh_centroid_offset/{object_id}",
            local_frame_offset_message_type="geometry_msgs/msg/Vector3Stamped",
            execution_frame_topic="/perception/fp/debug_frame/zed2i_2",
            execution_frame_message_type="fp_debug_msgs/msg/DebugFrame",
        )
        source_frame_pose_obj_world = ObjectWorldPose(
            position_world=(0.2, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        execution_pose_world = ObjectWorldPose(
            position_world=(1.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )

        frame_pair = mock.Mock(
            source_frame_pose_obj_world=source_frame_pose_obj_world,
            execution_pose_world=execution_pose_world,
        )

        with (
            mock.patch.object(
                run_grasp_pipeline,
                "wait_for_real_frame_pair_messages",
                return_value=frame_pair,
            ) as wait_pair,
            mock.patch.object(run_grasp_pipeline, "wait_for_object_pose_message") as wait_legacy,
        ):
            resolved_source_frame_pose_obj_world, object_pose_world = run_grasp_pipeline._resolve_real_world_frames(
                ros2
            )

        self.assertEqual(resolved_source_frame_pose_obj_world, source_frame_pose_obj_world)
        np.testing.assert_allclose(object_pose_world.position_world, (1.0, 0.2, 0.0), atol=1.0e-6)
        np.testing.assert_allclose(
            object_pose_world.orientation_xyzw_world, execution_pose_world.orientation_xyzw_world, atol=1.0e-6
        )
        wait_pair.assert_called_once_with(
            source_topic_name="/perception/fp/mesh_centroid_offset/cooling_screw",
            source_message_type="geometry_msgs/msg/Vector3Stamped",
            execution_topic_name="/perception/fp/debug_frame/zed2i_2",
            execution_message_type="fp_debug_msgs/msg/DebugFrame",
            object_id="cooling_screw",
            timeout_s=1.0,
        )
        wait_legacy.assert_not_called()

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
            mock.patch.object(run_grasp_pipeline, "_resolve_real_world_frames", return_value=(None, object_pose_world)),
            mock.patch.object(run_grasp_pipeline, "generate_stage1_result", return_value=stage1),
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


if __name__ == "__main__":
    unittest.main()
