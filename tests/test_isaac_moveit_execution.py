from __future__ import annotations

import importlib.util
import unittest
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


class _PhaseCallbackExecutor:
    current_label = ""

    def __init__(self, context, **kwargs) -> None:
        self._step_callback = kwargs.get("step_callback")

    def execute(self, trajectory) -> tuple[bool, str]:
        first_waypoint = float(trajectory.waypoints[0][0, 0].item())
        label_by_waypoint = {
            0.1: "pregrasp",
            0.2: "grasp",
            0.3: "lift",
        }
        type(self).current_label = label_by_waypoint.get(round(first_waypoint, 1), "")
        try:
            if self._step_callback is not None:
                self._step_callback()
        finally:
            type(self).current_label = ""
        return True, "ok"


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

    def test_moveit_pick_ignores_pre_lift_object_z_peak_when_validating_lift(self) -> None:
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }
        non_lift_samples = iter((0.02, 0.021, 0.021))

        def object_root_z(_object_asset):
            if _PhaseCallbackExecutor.current_label == "pregrasp":
                return 0.09
            if _PhaseCallbackExecutor.current_label == "grasp":
                return 0.021
            if _PhaseCallbackExecutor.current_label == "lift":
                return 0.022
            return next(non_lift_samples)

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _PhaseCallbackExecutor),
            mock.patch.object(pick_execution, "_command_gripper_width"),
            mock.patch.object(pick_execution, "_object_root_z", side_effect=object_root_z),
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
        self.assertAlmostEqual(result.object_lift_height_m, 0.002)
        self.assertAlmostEqual(result.diagnostics["final_object_lift_height_m"], 0.001)

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

    def test_object_root_z_rejects_non_finite_pose_values(self) -> None:
        import torch

        for z_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(z_value=z_value):
                object_asset = type(
                    "ObjectAsset",
                    (),
                    {"data": type("Data", (), {"root_link_pose_w": torch.tensor([[0.0, 0.0, z_value]])})()},
                )()

                self.assertIsNone(pick_execution._object_root_z(object_asset))

    def test_object_lift_validation_rejects_non_finite_pose_values(self) -> None:
        cases = (
            ("initial", float("nan"), 0.08, None),
            ("final", 0.02, float("inf"), None),
            ("peak", 0.02, 0.08, float("inf")),
        )
        for _label, initial_z, final_z, observed_z in cases:
            with self.subTest(label=_label):
                with mock.patch.object(pick_execution, "_object_root_z", return_value=final_z):
                    result = pick_execution._validate_object_lift(
                        object_asset=object(),
                        initial_object_z=initial_z,
                        observed_object_max_z=observed_z,
                        success_height_margin_m=0.05,
                    )

                self.assertIsNotNone(result)
                self.assertFalse(result.success)
                self.assertEqual(result.status, "object_pose_unavailable")


if __name__ == "__main__":
    unittest.main()
