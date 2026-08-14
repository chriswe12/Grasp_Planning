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
        self.assertEqual(config.ik_strategy, "direct")
        self.assertEqual(config.cartesian_waypoint_count, 10)

    def test_real_execution_config_parses_cartesian_waypoint_ik_strategy(self) -> None:
        config = run_grasp_pipeline._real_execution_config(
            {"real_execution": {"ik_strategy": "cartesian_waypoints", "cartesian_waypoint_count": 20}}
        )

        self.assertEqual(config.ik_strategy, "cartesian_waypoints")
        self.assertEqual(config.cartesian_waypoint_count, 20)

    def test_real_execution_config_parses_lbr_moveit_settings(self) -> None:
        config = run_grasp_pipeline._real_execution_config(
            {
                "real_execution": {
                    "planning_group": "arm",
                    "pose_link": "lbr_link_ee",
                    "moveit_namespace": "/lbr",
                    "joint_names": [
                        "lbr_A1",
                        "lbr_A2",
                        "lbr_A3",
                        "lbr_A4",
                        "lbr_A5",
                        "lbr_A6",
                        "lbr_A7",
                    ],
                    "frame_id": "lbr_link_0",
                    "gripper_client": "gripper_command",
                    "gripper_command_action": "/hand/gripper_cmd",
                    "gripper_command_position_mode": "kuka_y_finger_joint",
                    "gripper_command_max_effort": 12.0,
                }
            }
        )

        self.assertEqual(config.planning_group, "arm")
        self.assertEqual(config.pose_link, "lbr_link_ee")
        self.assertEqual(config.moveit_namespace, "/lbr")
        self.assertEqual(config.frame_id, "lbr_link_0")
        self.assertEqual(
            config.joint_names,
            ("lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4", "lbr_A5", "lbr_A6", "lbr_A7"),
        )
        self.assertEqual(config.gripper_client, "gripper_command")
        self.assertEqual(config.gripper_command_action, "/hand/gripper_cmd")
        self.assertEqual(config.gripper_command_position_mode, "kuka_y_finger_joint")
        self.assertAlmostEqual(config.gripper_command_max_effort, 12.0)

    def test_real_execution_config_parses_planning_scene_obstacles(self) -> None:
        config = run_grasp_pipeline._real_execution_config(
            {
                "real_execution": {
                    "planning_scene_obstacles": [
                        {
                            "id": "floor",
                            "type": "box",
                            "frame_id": "lbr_link_0",
                            "size_m": [2.0, 2.0, 0.02],
                            "xyz": [0.0, 0.0, -0.01],
                        }
                    ]
                }
            }
        )

        self.assertEqual(len(config.planning_scene_obstacles), 1)
        self.assertEqual(config.planning_scene_obstacles[0]["id"], "floor")
        self.assertEqual(config.planning_scene_obstacles[0]["frame_id"], "lbr_link_0")

    def test_real_execution_config_parses_trigger_gripper_services(self) -> None:
        config = run_grasp_pipeline._real_execution_config(
            {
                "real_execution": {
                    "gripper_client": "trigger_service",
                    "gripper_trigger_open_service": "/hand/open",
                    "gripper_trigger_close_service": "/hand/close",
                    "gripper_trigger_stop_service": "/hand/stop",
                }
            }
        )

        self.assertEqual(config.gripper_client, "trigger_service")
        self.assertEqual(config.gripper_trigger_open_service, "/hand/open")
        self.assertEqual(config.gripper_trigger_close_service, "/hand/close")
        self.assertEqual(config.gripper_trigger_stop_service, "/hand/stop")

    def test_real_execution_config_rejects_invalid_stop_after(self) -> None:
        with self.assertRaises(ValueError):
            run_grasp_pipeline._real_execution_config({"real_execution": {"stop_after": "unsupported"}})

    def test_resolve_object_pose_world_reads_fused_debug_pose_item_topic(self) -> None:
        ros2 = run_grasp_pipeline.Ros2Config(
            pose_base_topic="/perception/fp/pose_base/fused/assembly",
            frame_id="world",
            timeout_s=1.0,
            assembly_name="cooling_manifold",
            part_id=2,
        )
        object_pose = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.2),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )

        with mock.patch.object(
            run_grasp_pipeline,
            "wait_for_debug_pose_item_message",
            return_value=object_pose,
        ) as wait_pose:
            object_pose_world = run_grasp_pipeline._resolve_object_pose_world(ros2)

        self.assertEqual(object_pose_world, object_pose)
        wait_pose.assert_called_once_with(
            topic_name="/perception/fp/pose_base/fused/assembly",
            message_type=run_grasp_pipeline.DEBUG_POSE_ITEM_MESSAGE_TYPE,
            assembly_name="cooling_manifold",
            part_id=2,
            timeout_s=1.0,
        )

    def test_resolve_object_pose_world_applies_configured_world_position_offset(self) -> None:
        ros2 = run_grasp_pipeline.Ros2Config(
            pose_base_topic="/perception/fp/pose_base/fused/assembly",
            frame_id="world",
            timeout_s=1.0,
            assembly_name="plumbers_block",
            part_id=0,
            position_offset_m=(0.0, -0.840, 0.0),
        )
        perceived_pose = ObjectWorldPose(
            position_world=(0.5, 0.9, 0.04),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )

        with mock.patch.object(
            run_grasp_pipeline,
            "wait_for_debug_pose_item_message",
            return_value=perceived_pose,
        ):
            corrected_pose = run_grasp_pipeline._resolve_object_pose_world(ros2)

        np.testing.assert_allclose(corrected_pose.position_world, (0.5, 0.06, 0.04), atol=1.0e-12)
        self.assertEqual(corrected_pose.orientation_xyzw_world, perceived_pose.orientation_xyzw_world)

    def test_resolve_object_pose_world_requires_pose_base_topic_assembly_and_part(self) -> None:
        with self.assertRaises(ValueError):
            run_grasp_pipeline._resolve_object_pose_world(
                run_grasp_pipeline.Ros2Config(pose_base_topic="", assembly_name="cooling_manifold", part_id=2)
            )

        with self.assertRaises(ValueError):
            run_grasp_pipeline._resolve_object_pose_world(
                run_grasp_pipeline.Ros2Config(
                    pose_base_topic="/perception/fp/pose_base/fused/assembly",
                    assembly_name="",
                    part_id=2,
                )
            )

        with self.assertRaises(ValueError):
            run_grasp_pipeline._resolve_object_pose_world(
                run_grasp_pipeline.Ros2Config(
                    pose_base_topic="/perception/fp/pose_base/fused/assembly",
                    assembly_name="cooling_manifold",
                    part_id=None,
                )
            )

    def test_ros2_config_does_not_fallback_to_legacy_topic_keys(self) -> None:
        config = run_grasp_pipeline._ros2_config(
            {
                "ros2": {
                    "execution_frame_topic": "/legacy/execution",
                    "object_pose_topic": "/legacy/object_pose",
                    "assembly_name": "cooling_manifold",
                    "part_id": 2,
                    "position_offset_m": [0.0, -0.840, 0.0],
                }
            }
        )

        self.assertEqual(config.pose_base_topic, "")
        self.assertEqual(config.assembly_name, "cooling_manifold")
        self.assertEqual(config.part_id, 2)
        self.assertEqual(config.position_offset_m, (0.0, -0.840, 0.0))

    def test_ros2_config_requires_three_position_offset_values(self) -> None:
        with self.assertRaises(ValueError):
            run_grasp_pipeline._ros2_config({"ros2": {"position_offset_m": [0.0, -0.840]}})

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
        stage1 = SimpleNamespace(
            bundle=SimpleNamespace(candidates=("g0001",)),
            raw_candidate_count=1,
            target_mesh_local=SimpleNamespace(vertices_obj=np.array([[0.0, 0.0, -0.2]], dtype=float)),
        )
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
            mock.patch.object(run_grasp_pipeline, "_write_part_frame_debug_artifact"),
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

    def test_run_real_roll_sweep_keeps_live_axis_for_stage1_cache_augmentation(self) -> None:
        payload = {
            "geometry": {"target_mesh_path": "obj/fabrica/beam/2.obj", "mesh_scale": 0.01},
            "planning": {"roll_angle_step_deg": 15.0},
            "artifacts": {
                "stage1_json": "artifacts/test_stage1.json",
                "stage1_html": "artifacts/test_stage1.html",
                "stage2_json": "artifacts/test_stage2.json",
                "stage2_html": "artifacts/test_stage2.html",
            },
            "ros2": {},
            "real_execution": {"enabled": False},
        }
        object_pose_world = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.2),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )
        stage1 = SimpleNamespace(
            bundle=SimpleNamespace(candidates=("g0001",)),
            raw_candidate_count=1,
            target_mesh_local=SimpleNamespace(vertices_obj=np.array([[0.0, 0.0, -0.2]], dtype=float)),
        )
        stage2 = SimpleNamespace(source_bundle=SimpleNamespace(candidates=("g0001",)), accepted=("g0001",))

        with (
            mock.patch.object(run_grasp_pipeline, "_resolve_object_pose_world", return_value=object_pose_world),
            mock.patch.object(run_grasp_pipeline, "generate_stage1_result", return_value=stage1) as generate_stage1,
            mock.patch.object(run_grasp_pipeline, "write_stage1_artifacts"),
            mock.patch.object(run_grasp_pipeline, "recheck_stage2_result", return_value=stage2),
            mock.patch.object(run_grasp_pipeline, "write_stage2_artifacts"),
            mock.patch.object(run_grasp_pipeline, "_write_part_frame_debug_artifact"),
            mock.patch.object(run_grasp_pipeline, "execute_real_grasp_from_bundle"),
        ):
            run_grasp_pipeline.run_real(payload)

        self.assertNotEqual(generate_stage1.call_args.kwargs["upright_approach_axes_obj"], ())

    def test_run_real_skips_execution_when_real_execution_disabled(self) -> None:
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
            "real_execution": {"enabled": False},
        }
        object_pose_world = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.2),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        stage1 = SimpleNamespace(
            bundle=SimpleNamespace(candidates=("g0001",)),
            raw_candidate_count=1,
            target_mesh_local=SimpleNamespace(vertices_obj=np.array([[0.0, 0.0, -0.2]], dtype=float)),
        )
        stage2 = SimpleNamespace(source_bundle=SimpleNamespace(candidates=("g0001",)), accepted=("g0001",))

        with (
            mock.patch.object(run_grasp_pipeline, "_resolve_object_pose_world", return_value=object_pose_world),
            mock.patch.object(run_grasp_pipeline, "generate_stage1_result", return_value=stage1),
            mock.patch.object(run_grasp_pipeline, "write_stage1_artifacts"),
            mock.patch.object(run_grasp_pipeline, "recheck_stage2_result", return_value=stage2),
            mock.patch.object(run_grasp_pipeline, "write_stage2_artifacts"),
            mock.patch.object(run_grasp_pipeline, "_write_part_frame_debug_artifact"),
            mock.patch.object(run_grasp_pipeline, "execute_real_grasp_from_bundle") as execute_real,
        ):
            run_grasp_pipeline.run_real(payload)

        execute_real.assert_not_called()

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
        stage1 = SimpleNamespace(
            bundle=SimpleNamespace(candidates=("g0001",)),
            raw_candidate_count=1,
            target_mesh_local=SimpleNamespace(vertices_obj=np.array([[0.0, 0.0, -0.2]], dtype=float)),
        )
        stage2 = SimpleNamespace(source_bundle=SimpleNamespace(candidates=("g0001",)), accepted=("g0001",))

        with (
            mock.patch.object(run_grasp_pipeline, "_resolve_object_pose_world", return_value=object_pose_world),
            mock.patch.object(run_grasp_pipeline, "generate_stage1_result", return_value=stage1) as generate_stage1,
            mock.patch.object(run_grasp_pipeline, "write_stage1_artifacts"),
            mock.patch.object(run_grasp_pipeline, "recheck_stage2_result", return_value=stage2) as recheck_stage2,
            mock.patch.object(run_grasp_pipeline, "write_stage2_artifacts"),
            mock.patch.object(run_grasp_pipeline, "_write_part_frame_debug_artifact"),
            mock.patch.object(run_grasp_pipeline, "_run_mujoco_execution"),
        ):
            run_grasp_pipeline.run_pitl(payload, headless=True)

        self.assertNotIn("source_frame_pose_obj_world", generate_stage1.call_args.kwargs)
        self.assertEqual(recheck_stage2.call_args.kwargs["object_pose_world"], object_pose_world)


if __name__ == "__main__":
    unittest.main()
