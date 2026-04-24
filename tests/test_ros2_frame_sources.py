from __future__ import annotations

import unittest

import numpy as np

from grasp_planning.grasping import ObjectWorldPose, TriangleMesh
from grasp_planning.grasping.fabrica_grasp_debug import canonicalize_target_mesh
from grasp_planning.pipeline.fabrica_pipeline import _mesh_in_source_frame
from grasp_planning.ros2.pose_listener import extract_execution_pose_from_debug_frame


class _Point:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _Quaternion:
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class _Pose:
    def __init__(self, position: tuple[float, float, float], orientation: tuple[float, float, float, float]) -> None:
        self.position = _Point(*position)
        self.orientation = _Quaternion(*orientation)


class _PoseItem:
    def __init__(
        self,
        *,
        object_id: str,
        score: float,
        pose_base: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    ) -> None:
        self.object_id = object_id
        self.score = score
        self.pose_base = _Pose(*pose_base)


class _DebugFrame:
    def __init__(self, pose_items) -> None:
        self.pose_items = tuple(pose_items)


class Ros2FrameSourceTests(unittest.TestCase):
    def test_canonicalize_target_mesh_uses_vertex_average_origin(self) -> None:
        mesh_obj_world = TriangleMesh(
            vertices_obj=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
        )

        mesh_local, source_frame_pose = canonicalize_target_mesh(mesh_obj_world)

        self.assertEqual(source_frame_pose.position_world, (1.0, 0.5, 0.5))
        self.assertEqual(source_frame_pose.orientation_xyzw_world, (0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(
            mesh_local.vertices_obj,
            np.asarray(mesh_obj_world.vertices_obj, dtype=float) - np.array([[1.0, 0.5, 0.5]], dtype=float),
            atol=1.0e-6,
        )

    def test_extract_execution_pose_from_debug_frame_picks_highest_score_for_object(self) -> None:
        debug_frame = _DebugFrame(
            [
                _PoseItem(
                    object_id="cooling_screw",
                    score=0.4,
                    pose_base=((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)),
                ),
                _PoseItem(
                    object_id="cooling_screw",
                    score=0.9,
                    pose_base=((0.3, -0.1, 0.5), (0.0, 0.0, 0.70710678, 0.70710678)),
                ),
                _PoseItem(
                    object_id="pb_top",
                    score=1.0,
                    pose_base=((9.0, 9.0, 9.0), (0.0, 0.0, 0.0, 1.0)),
                ),
            ]
        )

        pose = extract_execution_pose_from_debug_frame(debug_frame, object_id="cooling_screw")

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose.position_world, (0.3, -0.1, 0.5))
        np.testing.assert_allclose(pose.orientation_xyzw_world, (0.0, 0.0, 0.70710678, 0.70710678), atol=1.0e-6)

    def test_extract_execution_pose_from_debug_frame_returns_none_when_missing_object(self) -> None:
        debug_frame = _DebugFrame(
            [_PoseItem(object_id="pb_top", score=0.9, pose_base=((0.3, -0.1, 0.5), (0.0, 0.0, 0.0, 1.0)))]
        )

        pose = extract_execution_pose_from_debug_frame(debug_frame, object_id="cooling_screw")

        self.assertIsNone(pose)

    def test_mesh_in_source_frame_applies_translation_offset(self) -> None:
        mesh_obj_world = TriangleMesh(
            vertices_obj=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        source_frame_pose = ObjectWorldPose(
            position_world=(1.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )

        mesh_local = _mesh_in_source_frame(mesh_obj_world, source_frame_pose)

        np.testing.assert_allclose(
            mesh_local.vertices_obj,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            atol=1.0e-6,
        )

    def test_mesh_in_source_frame_applies_rotation_offset(self) -> None:
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

        mesh_local = _mesh_in_source_frame(mesh_obj_world, source_frame_pose)

        np.testing.assert_allclose(
            mesh_local.vertices_obj,
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ],
                dtype=float,
            ),
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
