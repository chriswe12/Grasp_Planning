from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from grasp_planning import SavedGraspBundle, SavedGraspCandidate
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from scripts import run_grasp_pipeline


class RunGraspPipelineRealTests(unittest.TestCase):
    @staticmethod
    def _candidate(grasp_id: str, x: float) -> SavedGraspCandidate:
        return SavedGraspCandidate(
            grasp_id=grasp_id,
            grasp_position_obj=(x, 0.0, 0.0),
            grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
            contact_point_a_obj=(x - 0.02, 0.0, 0.0),
            contact_point_b_obj=(x + 0.02, 0.0, 0.0),
            contact_normal_a_obj=(-1.0, 0.0, 0.0),
            contact_normal_b_obj=(1.0, 0.0, 0.0),
            jaw_width=0.04,
            roll_angle_rad=0.0,
        )

    @staticmethod
    def _bundle(*candidates: SavedGraspCandidate) -> SavedGraspBundle:
        return SavedGraspBundle(
            target_mesh_path="obj/fabrica/plumbers_block/0.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=(0.0, 0.0, 0.05),
            source_frame_orientation_xyzw_obj_world=(0.0, 0.0, 0.0, 1.0),
            candidates=tuple(candidates),
            metadata={},
        )

    def test_real_execution_config_parses_defaults_and_stop_after(self) -> None:
        config = run_grasp_pipeline._real_execution_config({"real_execution": {"enabled": True, "stop_after": "grasp"}})

        self.assertTrue(config.enabled)
        self.assertEqual(config.stop_after, "grasp")
        self.assertEqual(config.frame_id, "base")
        self.assertAlmostEqual(config.velocity_scale, 0.05)
        self.assertEqual(config.grasp_approach_controller, "moveit_pose")

    def test_execution_debug_html_uses_existing_browser_opener(self) -> None:
        from grasp_planning.pipeline import dual_robot_planning_debug

        path = Path("artifacts/live_part_frame.html")
        with mock.patch.object(
            dual_robot_planning_debug,
            "open_debug_html_in_browser",
            return_value="file:///artifacts/live_part_frame.html",
        ) as open_browser:
            run_grasp_pipeline._open_debug_html_if_requested(
                {"artifacts": {"open_debug_html": True}},
                path=path,
            )

        open_browser.assert_called_once_with(path)

    def test_policy_execution_debug_embeds_runtime_goal_rgb_and_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            goal_path = Path(temp_dir) / "runtime_goal.npz"
            np.savez_compressed(
                goal_path,
                goal_id=np.asarray("runtime__part_0__g0001"),
                goal_rgb=np.full((4, 6, 3), 127, dtype=np.uint8),
                goal_depth=np.full((4, 6), 0.25, dtype=np.float32),
            )
            images = run_grasp_pipeline._policy_goal_reference_images(
                goal_observation_path=goal_path,
            )

        self.assertEqual([image["title"] for image in images], ["Goal RGB render", "Goal policy depth"])
        self.assertTrue(all(image["data_url"].startswith("data:image/png;base64,") for image in images))
        self.assertIn("generated on demand", images[0]["caption"])
        self.assertIn("Valid area: 100.0%", images[1]["caption"])

    def test_real_execution_config_policy_uses_any_live_grasp(self) -> None:
        config = run_grasp_pipeline._real_execution_config(
            {
                "real_execution": {
                    "grasp_approach_controller": "d405_policy",
                    "visual_servo_config": "configs/visual_servo_real_d405.yaml",
                }
            }
        )

        self.assertEqual(config.grasp_approach_controller, "d405_policy")
        self.assertEqual(config.visual_servo_config, "configs/visual_servo_real_d405.yaml")
        self.assertEqual(config.grasp_id, "")
        self.assertFalse(hasattr(config, "policy_target_id"))
        self.assertFalse(hasattr(config, "policy_target_candidates"))

    def test_real_execution_config_requires_policy_config_for_d405_approach(self) -> None:
        with self.assertRaises(ValueError):
            run_grasp_pipeline._real_execution_config(
                {"real_execution": {"grasp_approach_controller": "d405_policy"}}
            )

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
                    "gripper_position_command_topic": "/hand/position_command",
                    "gripper_position_feedback_topic": "/hand/position",
                    "gripper_position_feedback_tolerance": 0.03,
                    "moveit_gripper_joint_name": "left_finger_joint",
                    "gripper_closed_width": 0.007,
                    "gripper_open_width": 0.064,
                }
            }
        )

        self.assertEqual(config.gripper_client, "trigger_service")
        self.assertEqual(config.gripper_trigger_open_service, "/hand/open")
        self.assertEqual(config.gripper_trigger_close_service, "/hand/close")
        self.assertEqual(config.gripper_trigger_stop_service, "/hand/stop")
        self.assertEqual(config.gripper_position_command_topic, "/hand/position_command")
        self.assertEqual(config.gripper_position_feedback_topic, "/hand/position")
        self.assertAlmostEqual(config.gripper_position_feedback_tolerance, 0.03)
        self.assertEqual(config.moveit_gripper_joint_name, "left_finger_joint")
        self.assertAlmostEqual(config.gripper_closed_width, 0.007)
        self.assertAlmostEqual(config.gripper_open_width, 0.064)

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

    def test_run_real_passes_all_stage2_grasps_to_executor_and_refreshes_runtime_goal(self) -> None:
        payload = {
            "geometry": {"target_mesh_path": "obj/fabrica/plumbers_block/0.obj", "mesh_scale": 0.01},
            "planning": {},
            "artifacts": {
                "stage1_json": "artifacts/test_stage1.json",
                "stage1_html": "artifacts/test_stage1.html",
                "stage2_json": "artifacts/test_stage2.json",
                "stage2_html": "artifacts/test_stage2.html",
            },
            "ros2": {},
            "real_execution": {
                "enabled": True,
                "grasp_approach_controller": "d405_policy",
                "visual_servo_config": "visual.yaml",
            },
        }
        object_pose_world = ObjectWorldPose(
            position_world=(0.4, -0.1, 0.2),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )
        stage1 = SimpleNamespace(
            bundle=SimpleNamespace(candidates=("g0001", "g0002")),
            raw_candidate_count=2,
            target_mesh_local=SimpleNamespace(vertices_obj=np.array([[0.0, 0.0, -0.2]], dtype=float)),
        )
        selected = run_grasp_pipeline.replace(self._candidate("g0001", -0.02), score=0.85)
        stage2 = SimpleNamespace(
            source_bundle=SimpleNamespace(candidates=("g0001", "g0002")),
            accepted=(selected,),
        )
        execution_result = SimpleNamespace(
            success=True,
            status="completed",
            grasp_id="g0001",
            message="ok",
            attempt_artifact_path=Path("artifacts/attempt.json"),
        )

        with (
            mock.patch.object(run_grasp_pipeline, "_resolve_object_pose_world", return_value=object_pose_world),
            mock.patch.object(run_grasp_pipeline, "generate_stage1_result", return_value=stage1),
            mock.patch.object(run_grasp_pipeline, "write_stage1_artifacts"),
            mock.patch.object(run_grasp_pipeline, "recheck_stage2_result", return_value=stage2),
            mock.patch.object(run_grasp_pipeline, "write_stage2_artifacts"),
            mock.patch.object(run_grasp_pipeline, "_write_part_frame_debug_artifact"),
            mock.patch.object(run_grasp_pipeline, "_write_policy_execution_debug_artifact"),
            mock.patch.object(
                run_grasp_pipeline,
                "execute_real_grasp_from_bundle",
                return_value=execution_result,
            ) as execute_real,
        ):
            run_grasp_pipeline.run_real(payload)

        selected_config = execute_real.call_args.kwargs["config"]
        self.assertEqual(selected_config.grasp_id, "")
        callback = execute_real.call_args.kwargs["pregrasp_selected_callback"]
        goal_path = Path("artifacts/policy_goal_g0001.npz")
        with mock.patch.object(
            run_grasp_pipeline,
            "_write_policy_execution_debug_artifact",
        ) as refreshed_debug:
            callback(
                selected_grasp=selected,
                config=selected_config,
                candidate_rank=1,
                goal_observation_path=goal_path,
            )
        self.assertEqual(refreshed_debug.call_args.kwargs["goal_observation_path"], goal_path)
        self.assertEqual(refreshed_debug.call_args.kwargs["candidate_rank"], 1)

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
