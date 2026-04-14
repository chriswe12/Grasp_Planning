from __future__ import annotations

import unittest

import numpy as np

from grasp_planning.grasping import TriangleMesh
from grasp_planning.grasping.fabrica_grasp_debug import (
    build_pickup_pose_world,
    sample_pickup_placement_spec,
)


def _make_cube_mesh(side_length: float) -> TriangleMesh:
    half = 0.5 * float(side_length)
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


class PickupPoseSamplingTests(unittest.TestCase):
    def test_build_pickup_pose_world_places_supported_mesh_on_floor(self) -> None:
        mesh_local = _make_cube_mesh(0.05)
        object_pose_world = build_pickup_pose_world(
            mesh_local,
            support_face="neg_z",
            yaw_deg=90.0,
            xy_world=(0.1, -0.2),
        )
        rot = object_pose_world.rotation_world_from_object
        vertices_world = np.asarray(mesh_local.vertices_obj, dtype=float) @ rot.T + object_pose_world.translation_world
        self.assertGreaterEqual(float(vertices_world[:, 2].min()), -1.0e-8)
        self.assertAlmostEqual(float(vertices_world[:, 2].min()), 0.0, places=6)

    def test_sample_pickup_placement_spec_respects_allowed_support_faces(self) -> None:
        rng = np.random.default_rng(7)
        spec = sample_pickup_placement_spec(
            rng=rng,
            allowed_support_faces=("pos_x", "neg_y"),
            allowed_yaw_deg=(0.0, 180.0),
            xy_min_world=(-0.4, -0.1),
            xy_max_world=(-0.3, 0.1),
        )
        self.assertIn(spec.support_face, {"pos_x", "neg_y"})

    def test_sample_pickup_placement_spec_respects_allowed_yaws(self) -> None:
        rng = np.random.default_rng(11)
        spec = sample_pickup_placement_spec(
            rng=rng,
            allowed_support_faces=("neg_z",),
            allowed_yaw_deg=(0.0, 90.0, 270.0),
            xy_min_world=(-0.4, -0.1),
            xy_max_world=(-0.3, 0.1),
        )
        self.assertIn(spec.yaw_deg, {0.0, 90.0, 270.0})

    def test_sample_pickup_placement_spec_respects_xy_bounds(self) -> None:
        rng = np.random.default_rng(13)
        spec = sample_pickup_placement_spec(
            rng=rng,
            allowed_support_faces=("neg_z",),
            allowed_yaw_deg=(0.0,),
            xy_min_world=(-0.4, -0.1),
            xy_max_world=(-0.3, 0.1),
        )
        self.assertGreaterEqual(spec.xy_world[0], -0.4)
        self.assertLessEqual(spec.xy_world[0], -0.3)
        self.assertGreaterEqual(spec.xy_world[1], -0.1)
        self.assertLessEqual(spec.xy_world[1], 0.1)


if __name__ == "__main__":
    unittest.main()
