from __future__ import annotations

import unittest

from grasp_planning.planning.collision_checker import CollisionChecker, _ColliderPrim


class CollisionCheckerClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = CollisionChecker.__new__(CollisionChecker)
        self.checker._robot_root_path = "/World/envs/env_0/Robot"
        self.checker._cube_root_path = "/World/envs/env_0/Cube"
        self.checker._find_rigid_body_ancestor = lambda prim_path: "/World/envs/env_0/Robot/link4"
        self.collider = _ColliderPrim(
            collider_path="/World/envs/env_0/Robot/link3/collider",
            rigid_body_path="/World/envs/env_0/Robot/link3",
        )

    def test_ignores_self_hit(self) -> None:
        reason = self.checker._classify_hit(
            self.collider,
            collision_path="/World/envs/env_0/Robot/link3/collider",
            rigid_body_path="/World/envs/env_0/Robot/link3",
        )
        self.assertIsNone(reason)

    def test_detects_robot_self_collision(self) -> None:
        reason = self.checker._classify_hit(
            self.collider,
            collision_path="/World/envs/env_0/Robot/link4/collider",
            rigid_body_path="/World/envs/env_0/Robot/link4",
        )
        self.assertEqual(reason, "self_collision")

    def test_detects_cube_collision(self) -> None:
        reason = self.checker._classify_hit(
            self.collider,
            collision_path="/World/envs/env_0/Cube/Collider",
            rigid_body_path="/World/envs/env_0/Cube",
        )
        self.assertEqual(reason, "cube_collision")

    def test_detects_ground_collision(self) -> None:
        reason = self.checker._classify_hit(
            self.collider,
            collision_path="/World/GroundPlane/Collider",
            rigid_body_path="",
        )
        self.assertEqual(reason, "plane_collision")


if __name__ == "__main__":
    unittest.main()
