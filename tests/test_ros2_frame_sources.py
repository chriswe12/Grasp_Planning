from __future__ import annotations

import unittest

import numpy as np

from grasp_planning.grasping import ObjectWorldPose, TriangleMesh
from grasp_planning.grasping.fabrica_grasp_debug import canonicalize_target_mesh
from grasp_planning.pipeline.fabrica_pipeline import _mesh_in_source_frame
from grasp_planning.ros2.pose_listener import extract_execution_pose_from_debug_pose_item


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


class _PoseStamped:
    def __init__(self, pose: tuple[tuple[float, float, float], tuple[float, float, float, float]]) -> None:
        self.pose = _Pose(*pose)


class _PoseItem:
    def __init__(
        self,
        *,
        assembly_name: str,
        part_id: int,
        score: float,
        pose_base: tuple[tuple[float, float, float], tuple[float, float, float, float]],
    ) -> None:
        self.assembly_name = assembly_name
        self.part_id = part_id
        self.score = score
        self.pose_base = _PoseStamped(pose_base)


class Ros2FrameSourceTests(unittest.TestCase):
    def test_canonicalize_target_mesh_uses_trimesh_centroid_origin(self) -> None:
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

        expected_centroid = np.array([4.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
        np.testing.assert_allclose(source_frame_pose.position_world, expected_centroid, atol=1.0e-6)
        self.assertEqual(source_frame_pose.orientation_xyzw_world, (0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(
            mesh_local.vertices_obj,
            np.asarray(mesh_obj_world.vertices_obj, dtype=float) - expected_centroid,
            atol=1.0e-6,
        )

    def test_extract_execution_pose_from_debug_pose_item_matches_part(self) -> None:
        pose_item = _PoseItem(
            assembly_name="cooling_manifold",
            part_id=2,
            score=0.9,
            pose_base=((0.3, -0.1, 0.5), (0.0, 0.0, 0.70710678, 0.70710678)),
        )

        pose = extract_execution_pose_from_debug_pose_item(
            pose_item,
            assembly_name="cooling_manifold",
            part_id=2,
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose.position_world, (0.3, -0.1, 0.5))
        np.testing.assert_allclose(pose.orientation_xyzw_world, (0.0, 0.0, 0.70710678, 0.70710678), atol=1.0e-6)

    def test_extract_execution_pose_from_debug_pose_item_returns_none_when_part_does_not_match(self) -> None:
        pose_item = _PoseItem(
            assembly_name="plumbers_block",
            part_id=1,
            score=0.9,
            pose_base=((0.3, -0.1, 0.5), (0.0, 0.0, 0.0, 1.0)),
        )

        pose = extract_execution_pose_from_debug_pose_item(
            pose_item,
            assembly_name="cooling_manifold",
            part_id=2,
        )

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
