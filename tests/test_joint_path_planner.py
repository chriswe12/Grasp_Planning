from __future__ import annotations

import unittest

import torch

from grasp_planning.planning.joint_path_planner import JointPathPlanner


class _AlwaysValidCollisionChecker:
    def __init__(self) -> None:
        self.calls = []

    def is_edge_valid(self, q_start: torch.Tensor, q_goal: torch.Tensor, num_checks: int = 20):
        self.calls.append((q_start.clone(), q_goal.clone(), num_checks))
        return True, "ok"


class _AlwaysInvalidCollisionChecker:
    def is_edge_valid(self, q_start: torch.Tensor, q_goal: torch.Tensor, num_checks: int = 20):
        return False, "cube_collision"


class JointPathPlannerTests(unittest.TestCase):
    def test_plan_returns_interpolated_waypoints(self) -> None:
        checker = _AlwaysValidCollisionChecker()
        planner = JointPathPlanner(checker, max_joint_step_rad=0.1)
        q_start = torch.zeros((1, 7), dtype=torch.float32)
        q_goal = torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        trajectory, reason = planner.plan(q_start, q_goal, dt=0.01)

        self.assertEqual(reason, "ok")
        self.assertIsNotNone(trajectory)
        assert trajectory is not None
        self.assertEqual(len(trajectory.waypoints), 2)
        torch.testing.assert_close(trajectory.waypoints[-1], q_goal)
        self.assertEqual(checker.calls[0][2], 4)

    def test_plan_returns_none_when_edge_is_invalid(self) -> None:
        planner = JointPathPlanner(_AlwaysInvalidCollisionChecker())
        q_start = torch.zeros((1, 7), dtype=torch.float32)
        q_goal = torch.ones((1, 7), dtype=torch.float32)

        trajectory, reason = planner.plan(q_start, q_goal, dt=0.01)

        self.assertIsNone(trajectory)
        self.assertEqual(reason, "cube_collision")


if __name__ == "__main__":
    unittest.main()
