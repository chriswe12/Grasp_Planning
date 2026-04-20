from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from grasp_planning.grasping import ObjectWorldPose, TriangleMesh
from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspBundle
from grasp_planning.pipeline.fabrica_pipeline import (
    GeometryConfig,
    PlanningConfig,
    Stage1Result,
    _mesh_in_source_frame,
    recheck_stage2_result,
    write_stage1_artifacts,
)


class PipelineSourceFrameRotationTests(unittest.TestCase):
    def test_recheck_stage2_result_uses_bundle_source_frame_rotation_for_mesh_local(self) -> None:
        mesh_obj_world = TriangleMesh(
            vertices_obj=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        source_frame_pose = ObjectWorldPose(
            position_world=(1.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )
        bundle = SavedGraspBundle(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=1.0,
            source_frame_origin_obj_world=source_frame_pose.position_world,
            source_frame_orientation_xyzw_obj_world=source_frame_pose.orientation_xyzw_world,
            candidates=(),
            metadata={},
        )
        expected_mesh_local = _mesh_in_source_frame(mesh_obj_world, source_frame_pose)

        with (
            mock.patch(
                "grasp_planning.pipeline.fabrica_pipeline.load_asset_mesh",
                return_value=mesh_obj_world,
            ),
            mock.patch(
                "grasp_planning.pipeline.fabrica_pipeline.evaluate_saved_grasps_against_pickup_pose",
                return_value=[],
            ),
            mock.patch(
                "grasp_planning.pipeline.fabrica_pipeline.score_grasps",
                return_value=[],
            ),
        ):
            result = recheck_stage2_result(
                bundle=bundle,
                pickup_spec=None,
                planning=PlanningConfig(),
                object_pose_world=ObjectWorldPose(
                    position_world=(0.2, -0.1, 0.3),
                    orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
                ),
            )

        np.testing.assert_allclose(result.mesh_local.vertices_obj, expected_mesh_local.vertices_obj, atol=1.0e-6)

    def test_write_stage1_artifacts_rotates_obstacle_mesh_into_source_frame(self) -> None:
        source_frame_pose = ObjectWorldPose(
            position_world=(1.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.70710678, 0.70710678),
        )
        target_mesh_local = TriangleMesh(
            vertices_obj=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        obstacle_mesh_world = TriangleMesh(
            vertices_obj=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        bundle = SavedGraspBundle(
            target_mesh_path="obj/fabrica/beam/2.obj",
            mesh_scale=1.0,
            source_frame_origin_obj_world=source_frame_pose.position_world,
            source_frame_orientation_xyzw_obj_world=source_frame_pose.orientation_xyzw_world,
            candidates=(),
            metadata={},
        )
        stage1 = Stage1Result(
            bundle=bundle,
            target_mesh_local=target_mesh_local,
            target_pose_in_obj_world=source_frame_pose,
            obstacle_mesh_world=obstacle_mesh_world,
            collision_backend_name="unit-test",
            raw_candidate_count=0,
        )
        expected_obstacle_mesh_local = _mesh_in_source_frame(obstacle_mesh_world, source_frame_pose)

        with (
            mock.patch("grasp_planning.pipeline.fabrica_pipeline.save_grasp_bundle"),
            mock.patch("grasp_planning.pipeline.fabrica_pipeline.write_debug_html") as write_debug_html,
        ):
            write_stage1_artifacts(
                stage1,
                geometry=GeometryConfig(target_mesh_path="obj/fabrica/beam/2.obj"),
                planning=PlanningConfig(),
                output_json=Path("/tmp/stage1.json"),
                output_html=Path("/tmp/stage1.html"),
            )

        actual_obstacle_mesh_local = write_debug_html.call_args.kwargs["obstacle_mesh_local"]
        np.testing.assert_allclose(
            actual_obstacle_mesh_local.vertices_obj,
            expected_obstacle_mesh_local.vertices_obj,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
