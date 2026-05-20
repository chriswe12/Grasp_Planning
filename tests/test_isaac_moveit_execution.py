from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this CI environment")

from grasp_planning.planning import pick_execution


class _FakeArmJointIds:
    def numel(self) -> int:
        return 7


class _FakeContext:
    def __init__(self, **kwargs) -> None:
        self.fixed_gripper_width = float(kwargs["fixed_gripper_width"])
        self.arm_joint_ids = _FakeArmJointIds()
        self.device = "cpu"
        self.physics_dt = 0.01


class _FakeExecutor:
    executions: list[tuple[float, float]] = []

    def __init__(self, context) -> None:
        self._context = context

    def execute(self, trajectory) -> tuple[bool, str]:
        first_waypoint = float(trajectory.waypoints[0][0, 0].item())
        self.executions.append((first_waypoint, float(self._context.fixed_gripper_width)))
        return True, "ok"


class IsaacMoveItExecutionTests(unittest.TestCase):
    def test_moveit_pick_executes_pregrasp_grasp_close_and_lift(self) -> None:
        _FakeExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _FakeExecutor),
            mock.patch.object(pick_execution, "_command_gripper_width") as command_gripper_width,
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=0.04,
                closed_gripper_width=0.0,
                pregrasp_only=False,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(_FakeExecutor.executions), 3)
        for actual, expected in zip(_FakeExecutor.executions, [(0.1, 0.04), (0.2, 0.04), (0.3, 0.0)], strict=True):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])
        command_gripper_width.assert_called_once()
        self.assertEqual(command_gripper_width.call_args.kwargs["width"], 0.0)


if __name__ == "__main__":
    unittest.main()
