from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this CI environment")

from grasp_planning.planning import pick_execution
from grasp_planning.planning.fr3_motion_context import FR3MotionContext


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
    configs: list[dict[str, object]] = []

    def __init__(self, context, **kwargs) -> None:
        self._context = context
        self.configs.append(dict(kwargs))

    def execute(self, trajectory) -> tuple[bool, str]:
        first_waypoint = float(trajectory.waypoints[0][0, 0].item())
        self.executions.append((first_waypoint, float(self._context.fixed_gripper_width)))
        return True, "ok"


class _FakePlannerContext:
    def __init__(self, **kwargs) -> None:
        self.fixed_gripper_width = float(kwargs["fixed_gripper_width"])
        self.arm_joint_ids = _FakeArmJointIds()
        self.arm_joint_names = tuple(f"panda_joint{index}" for index in range(1, 8))
        self.device = "cpu"
        self.physics_dt = 0.01

    def get_tcp_pose_w(self):
        import torch

        return torch.tensor([[0.5, 0.0, 0.2]], dtype=torch.float32), torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )


class _FakePregraspController:
    calls: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []

    def __init__(self) -> None:
        self._context = _FakePlannerContext(fixed_gripper_width=0.04)
        self._executor = object()

    def move_to_pose(self, *, position_w, orientation_xyzw):
        self.calls.append((tuple(position_w), tuple(orientation_xyzw)))
        return SimpleNamespace(success=True, message="ok")


class IsaacMoveItExecutionTests(unittest.TestCase):
    def test_isaac_grasp_tcp_mapping_matches_grasp_frame(self) -> None:
        position_w = (0.4, -0.1, 0.2)
        orientation_xyzw = (0.1, 0.2, 0.3, 0.9273618495495703)

        tcp_position_w, tcp_orientation_xyzw = FR3MotionContext.grasp_pose_to_tcp_pose(position_w, orientation_xyzw)
        roundtrip_position_w, roundtrip_orientation_xyzw = FR3MotionContext.tcp_pose_to_grasp_pose(
            tcp_position_w,
            tcp_orientation_xyzw,
        )

        self.assertEqual(tcp_position_w, position_w)
        self.assertEqual(tcp_orientation_xyzw, orientation_xyzw)
        for actual, expected in zip(roundtrip_position_w, position_w, strict=True):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(roundtrip_orientation_xyzw, orientation_xyzw, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_planner_pregrasp_uses_joint_path_controller(self) -> None:
        _FakePregraspController.calls = []
        world_grasp = SimpleNamespace(
            pregrasp_position_w=(0.5, 0.0, 0.2),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        controller = _FakePregraspController()

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakePlannerContext),
            mock.patch.object(pick_execution, "_build_controller", return_value=controller),
        ):
            ok, message, tcp_position, tcp_orientation = pick_execution.move_to_pregrasp(
                sim=object(),
                scene=object(),
                robot=object(),
                object_asset=object(),
                world_grasp=world_grasp,
                controller_type="planner",
                fixed_gripper_width=0.04,
            )

        self.assertTrue(ok, message)
        self.assertEqual(_FakePregraspController.calls, [((0.5, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))])
        self.assertEqual(tcp_position, (0.5, 0.0, 0.20000000298023224))
        self.assertEqual(tcp_orientation, (0.0, 0.0, 0.0, 1.0))

    def test_planner_pick_uses_same_pregrasp_grasp_lift_pose_sequence_as_mujoco(self) -> None:
        _FakePregraspController.calls = []
        controllers = [_FakePregraspController(), _FakePregraspController(), _FakePregraspController()]
        world_grasp = SimpleNamespace(
            pregrasp_position_w=(0.5, 0.0, 0.08),
            position_w=(0.5, 0.0, 0.03),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            gripper_width=0.08,
        )

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakePlannerContext),
            mock.patch.object(pick_execution, "_build_controller", side_effect=controllers),
            mock.patch.object(pick_execution, "_command_gripper_width") as command_gripper_width,
            mock.patch.object(pick_execution, "_object_root_z", side_effect=(0.02, 0.09)),
        ):
            result = pick_execution.execute_pick_from_world_grasp(
                sim=object(),
                scene=object(),
                robot=object(),
                object_asset=object(),
                world_grasp=world_grasp,
                controller_type="planner",
                fixed_gripper_width=0.04,
                closed_gripper_width=0.0,
                pregrasp_only=False,
                lift_height_m=0.12,
                success_height_margin_m=0.05,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "ok")
        self.assertEqual(
            _FakePregraspController.calls,
            [
                ((0.5, 0.0, 0.08), (0.0, 0.0, 0.0, 1.0)),
                ((0.5, 0.0, 0.03), (0.0, 0.0, 0.0, 1.0)),
                ((0.5, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0)),
            ],
        )
        command_gripper_width.assert_called_once()
        self.assertEqual(command_gripper_width.call_args.kwargs["width"], 0.0)

    def test_moveit_pick_executes_pregrasp_grasp_close_and_lift(self) -> None:
        _FakeExecutor.executions = []
        _FakeExecutor.configs = []
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
        self.assertEqual(len(_FakeExecutor.configs), 1)
        self.assertEqual(_FakeExecutor.configs[0]["max_joint_speed_rad_s"], 0.35)
        self.assertIn("step_callback", _FakeExecutor.configs[0])
        self.assertEqual(len(_FakeExecutor.executions), 3)
        for actual, expected in zip(_FakeExecutor.executions, [(0.1, 0.04), (0.2, 0.04), (0.3, 0.0)], strict=True):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])
        command_gripper_width.assert_called_once()
        self.assertEqual(command_gripper_width.call_args.kwargs["width"], 0.0)

    def test_moveit_pick_validates_object_lift_when_asset_is_available(self) -> None:
        _FakeExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _FakeExecutor),
            mock.patch.object(pick_execution, "_command_gripper_width"),
            mock.patch.object(pick_execution, "_object_root_z", side_effect=(0.02, 0.08)),
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                object_asset=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=0.04,
                closed_gripper_width=0.0,
                pregrasp_only=False,
                success_height_margin_m=0.05,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.object_lift_height_m, 0.06)
        self.assertAlmostEqual(result.target_lift_height_m, 0.05)
        self.assertAlmostEqual(result.diagnostics["final_object_lift_height_m"], 0.06)

    def test_moveit_pick_fails_when_object_does_not_lift(self) -> None:
        _FakeExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _FakeExecutor),
            mock.patch.object(pick_execution, "_command_gripper_width"),
            mock.patch.object(pick_execution, "_object_root_z", side_effect=(0.02, 0.021)),
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                object_asset=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=0.04,
                closed_gripper_width=0.0,
                pregrasp_only=False,
                success_height_margin_m=0.05,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status, "object_lift_failed")
        self.assertAlmostEqual(result.object_lift_height_m, 0.001)
        self.assertAlmostEqual(result.target_lift_height_m, 0.05)

    def test_object_lift_validation_uses_peak_height_when_available(self) -> None:
        with mock.patch.object(pick_execution, "_object_root_z", return_value=0.03):
            result = pick_execution._validate_object_lift(
                object_asset=object(),
                initial_object_z=0.02,
                observed_object_max_z=0.09,
                success_height_margin_m=0.05,
            )

        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.object_lift_height_m, 0.07)
        self.assertAlmostEqual(result.diagnostics["final_object_lift_height_m"], 0.01)


if __name__ == "__main__":
    unittest.main()
