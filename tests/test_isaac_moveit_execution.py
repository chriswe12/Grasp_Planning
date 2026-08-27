from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is not installed in this CI environment")

from grasp_planning.planning import pick_execution
from grasp_planning.planning.fr3_motion_context import FR3MotionContext, critical_damping_from_stiffness_inertia
from grasp_planning.start_poses import (
    DEFAULT_KUKA_ARM_START_JOINT_POS,
    KUKA_MOVEIT_ARM_START_JOINT_VALUES,
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
    KUKA_Y_GRIPPER_TRAVEL_M,
    PDZ_GRIPPER_CLOSED_WIDTH_M,
    PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M,
    PDZ_GRIPPER_OPEN_WIDTH_M,
    PDZ_GRIPPER_TRAVEL_M,
    gripper_joint_target_from_width,
    gripper_approach_width,
    gripper_max_open_width,
    kuka_isaac_to_moveit_joint_positions,
    kuka_moveit_to_isaac_joint_positions,
    kuka_y_gripper_approach_width_from_jaw_width,
)


class _FakeArmJointIds:
    def numel(self) -> int:
        return 7


class _FakeContext:
    def __init__(self, **kwargs) -> None:
        self.fixed_gripper_width = float(kwargs["fixed_gripper_width"])
        self.arm_joint_ids = _FakeArmJointIds()
        self.device = "cpu"
        self.physics_dt = 0.01


class _FakeKukaContext(_FakeContext):
    hand_joint_names = ("left_finger_joint", "right_finger_joint")


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


class _GraspSettleFailingExecutor(_FakeExecutor):
    def execute(self, trajectory) -> tuple[bool, str]:
        first_waypoint = float(trajectory.waypoints[0][0, 0].item())
        self.executions.append((first_waypoint, float(self._context.fixed_gripper_width)))
        if round(first_waypoint, 1) == 0.2:
            return False, "final waypoint did not settle; last_max_joint_error=0.0532"
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
    def test_critical_damping_uses_stiffness_and_generalized_inertia(self) -> None:
        import torch

        damping = critical_damping_from_stiffness_inertia(
            torch.tensor([[4.0, 9.0]], dtype=torch.float32),
            torch.tensor([[1.0, 4.0]], dtype=torch.float32),
        )

        self.assertEqual(damping.tolist(), [[4.0, 12.0]])

    def test_dual_isaac_uses_plan_role_names_and_robot_base_positions(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_simple_dual_robot_sim_in_isaac.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("ROLE_ROBOT_NAMES[role]", source)
        self.assertIn("holder_robot_base_position=holder_robot_base_position", source)
        self.assertIn("inserter_robot_base_position=inserter_robot_base_position", source)
        env_source = (Path(__file__).resolve().parents[1] / "grasp_planning" / "envs" / "fr3_part_env.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("kuka_hand_effort_limit_sim: float = 40.0", env_source)
        self.assertIn("effort_limit_sim=float(kuka_hand_effort_limit_sim)", env_source)

    def test_kuka_gripper_width_maps_to_closing_joint_target(self) -> None:
        self.assertAlmostEqual(
            gripper_joint_target_from_width("left_finger_joint", KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M), 0.0
        )
        self.assertAlmostEqual(
            gripper_joint_target_from_width("right_finger_joint", KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M), 0.0
        )
        self.assertAlmostEqual(gripper_joint_target_from_width("left_finger_joint", 0.059), 0.0125)
        self.assertAlmostEqual(gripper_joint_target_from_width("right_finger_joint", 0.059), -0.0125)
        self.assertAlmostEqual(gripper_joint_target_from_width("left_finger_joint", 0.0), KUKA_Y_GRIPPER_TRAVEL_M)
        self.assertAlmostEqual(gripper_joint_target_from_width("right_finger_joint", 0.0), -KUKA_Y_GRIPPER_TRAVEL_M)
        self.assertAlmostEqual(gripper_joint_target_from_width("panda_finger_joint1", 0.035), 0.035)
        self.assertAlmostEqual(gripper_max_open_width("left_finger_joint"), KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M)
        self.assertAlmostEqual(gripper_max_open_width("panda_finger_joint1"), 0.04)
        self.assertAlmostEqual(
            kuka_y_gripper_approach_width_from_jaw_width(0.042889),
            0.052889,
        )
        with self.assertRaisesRegex(ValueError, "physical opening"):
            kuka_y_gripper_approach_width_from_jaw_width(0.075)

    def test_pdz_gripper_width_maps_to_outward_joint_target(self) -> None:
        self.assertAlmostEqual(
            gripper_joint_target_from_width("pdz_gripper_left_finger_joint", PDZ_GRIPPER_CLOSED_WIDTH_M),
            0.0,
        )
        self.assertAlmostEqual(
            gripper_joint_target_from_width("pdz_gripper_right_finger_joint", PDZ_GRIPPER_OPEN_WIDTH_M),
            PDZ_GRIPPER_TRAVEL_M,
        )
        self.assertAlmostEqual(gripper_joint_target_from_width("pdz_gripper_left_finger_joint", 0.040), 0.014)
        self.assertAlmostEqual(
            gripper_max_open_width("pdz_gripper_left_finger_joint"),
            PDZ_GRIPPER_OPEN_WIDTH_M,
        )
        self.assertAlmostEqual(PDZ_GRIPPER_APPROACH_CLEARANCE_PER_FINGER_M, 0.005)
        self.assertAlmostEqual(
            gripper_approach_width(0.040, gripper_model="pdz_gripper"),
            0.050,
        )

    def test_kuka_moveit_joint_positions_are_converted_to_generated_usd_coordinates(self) -> None:
        self.assertEqual(
            KUKA_MOVEIT_ARM_START_JOINT_VALUES,
            (0.0, 0.5, 0.0, -1.3962634015954636, 0.0, 1.1, 0.0),
        )
        self.assertEqual(
            kuka_moveit_to_isaac_joint_positions((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)),
            (1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 7.0),
        )
        self.assertEqual(
            kuka_isaac_to_moveit_joint_positions((1.0, 2.0, 3.0, -4.0, 5.0, 6.0, 7.0)),
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        )
        for index, moveit_value in enumerate(KUKA_MOVEIT_ARM_START_JOINT_VALUES, start=1):
            expected = kuka_moveit_to_isaac_joint_positions(KUKA_MOVEIT_ARM_START_JOINT_VALUES)[index - 1]
            self.assertAlmostEqual(DEFAULT_KUKA_ARM_START_JOINT_POS[f"joint{index}"], expected)
            if index == 4:
                self.assertAlmostEqual(DEFAULT_KUKA_ARM_START_JOINT_POS[f"joint{index}"], -moveit_value)

    def test_motion_context_resolves_kuka_tcp_and_joint_names(self) -> None:
        robot = SimpleNamespace(
            body_names=["base_link", "link7", "gripper_base_link", "gripper_tcp"],
            joint_names=[*(f"joint{i}" for i in range(1, 8)), "left_finger_joint", "right_finger_joint"],
            device="cpu",
            is_fixed_base=True,
        )

        context = FR3MotionContext(robot=robot, scene=object(), sim=object())

        self.assertEqual(context.ee_body_name, "gripper_tcp")
        self.assertEqual(context.arm_joint_names, tuple(f"joint{i}" for i in range(1, 8)))
        self.assertEqual(context.hand_joint_names, ("left_finger_joint", "right_finger_joint"))
        self.assertEqual(context.hand_command_joint_names, ("left_finger_joint",))

    def test_motion_context_resolves_pdz_tcp_and_driver_joint(self) -> None:
        robot = SimpleNamespace(
            body_names=["base_link", "link7", "pdz_gripper_base_link", "pdz_gripper_tcp"],
            joint_names=[
                *(f"joint{i}" for i in range(1, 8)),
                "pdz_gripper_left_finger_joint",
                "pdz_gripper_right_finger_joint",
            ],
            device="cpu",
            is_fixed_base=True,
        )

        context = FR3MotionContext(robot=robot, scene=object(), sim=object())

        self.assertEqual(context.ee_body_name, "pdz_gripper_tcp")
        self.assertEqual(
            context.hand_joint_names,
            ("pdz_gripper_left_finger_joint", "pdz_gripper_right_finger_joint"),
        )
        self.assertEqual(
            context.hand_command_joint_names,
            ("pdz_gripper_left_finger_joint", "pdz_gripper_right_finger_joint"),
        )

    def test_pdz_contact_stall_acceptance_uses_decreasing_driver_coordinate(self) -> None:
        accepted_diagnostics = {
            "gripper_close_joint_names": ["pdz_gripper_left_finger_joint"],
            "gripper_close_final_joint_positions": [0.015],
            "gripper_close_final_max_step_delta": 0.0,
        }
        rejected_diagnostics = {
            "gripper_close_joint_names": ["pdz_gripper_left_finger_joint"],
            "gripper_close_final_joint_positions": [0.028],
            "gripper_close_final_max_step_delta": 0.0,
        }

        self.assertTrue(pick_execution._kuka_contact_stall_matches_grasp_width(accepted_diagnostics, 0.040))
        self.assertFalse(pick_execution._kuka_contact_stall_matches_grasp_width(rejected_diagnostics, 0.040))
        self.assertEqual(
            accepted_diagnostics["gripper_close_contact_stall_driver_joint_name"],
            "pdz_gripper_left_finger_joint",
        )

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
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={"gripper_close_status": "target_reached"},
            ) as command_gripper_width,
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
        self.assertFalse(command_gripper_width.call_args.kwargs["force_joint_state"])

    def test_moveit_pick_applies_postclose_hold_once_without_changing_close_settle_threshold(self) -> None:
        _FakeExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _FakeExecutor),
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={"gripper_close_status": "target_reached"},
            ) as command_gripper_width,
            mock.patch.object(pick_execution, "_hold_arm_waypoint") as hold_arm_waypoint,
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=0.04,
                closed_gripper_width=0.0,
                pregrasp_only=False,
                postclose_hold_s=1.75,
            )

        self.assertTrue(result.success)
        command_gripper_width.assert_called_once()
        self.assertEqual(
            command_gripper_width.call_args.kwargs["settle_duration_s"],
            pick_execution.GRIPPER_CLOSE_SETTLE_DURATION_S,
        )
        hold_arm_waypoint.assert_called_once()
        self.assertEqual(hold_arm_waypoint.call_args.kwargs["duration_s"], 1.75)

    def test_fixed_kuka_pick_keeps_postclose_hold_out_of_close_settle_threshold(self) -> None:
        script_text = (Path(__file__).resolve().parents[1] / "scripts" / "run_fixed_kuka_pick_in_isaac.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("settle_duration_s=GRIPPER_CLOSE_SETTLE_DURATION_S", script_text)
        self.assertNotIn("settle_duration_s=max(0.5, float(args_cli.postclose_hold_s))", script_text)
        self.assertEqual(script_text.count("duration_s=float(args_cli.postclose_hold_s)"), 1)

    def test_fixed_kuka_pick_contact_stall_uses_jaw_width_without_clearance(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_fixed_kuka_pick_in_isaac.py"
        source = script_path.read_text(encoding="utf-8")
        parsed = ast.parse(source)
        function_node = next(
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name == "_contact_stall_matches_selected_grasp"
        )
        isolated_module = ast.Module(body=[function_node], type_ignores=[])
        ast.fix_missing_locations(isolated_module)
        namespace = {"gripper_joint_target_from_width": gripper_joint_target_from_width}
        exec(compile(isolated_module, str(script_path), "exec"), namespace)

        diagnostics = {
            "gripper_close_final_joint_positions": [0.005],
            "gripper_close_final_max_step_delta": 0.0,
        }
        world_grasp = SimpleNamespace(jaw_width=0.059, gripper_width=0.069)
        accepted = namespace["_contact_stall_matches_selected_grasp"](diagnostics, world_grasp)

        self.assertFalse(accepted)
        self.assertAlmostEqual(diagnostics["gripper_close_contact_stall_selected_jaw_width_m"], 0.059)
        self.assertIn("selected jaw width 0.0590 m", diagnostics["gripper_close_contact_stall_accept_reason"])

    def test_dual_isaac_accepts_only_bilateral_contacts_filtered_to_selected_object(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_simple_dual_robot_sim_in_isaac.py"
        source = script_path.read_text(encoding="utf-8")
        parsed = ast.parse(source)
        helper_node = next(
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name == "_filtered_bilateral_contact_matches_selected_object"
        )
        isolated_module = ast.Module(body=[helper_node], type_ignores=[])
        ast.fix_missing_locations(isolated_module)
        namespace = {"math": math}
        exec(compile(isolated_module, str(script_path), "exec"), namespace)
        matches = namespace["_filtered_bilateral_contact_matches_selected_object"]

        bilateral = {
            "left": {"available": True, "filtered_force_norm_n": 0.020},
            "right": {"available": True, "filtered_force_norm_n": 0.015},
        }
        one_sided = {
            **bilateral,
            "right": {"available": True, "filtered_force_norm_n": 0.005},
        }
        unavailable = {
            **bilateral,
            "right": {"available": False, "filtered_force_norm_n": 1.0},
        }

        self.assertTrue(matches(bilateral, minimum_force_n=0.01))
        self.assertFalse(matches(one_sided, minimum_force_n=0.01))
        self.assertFalse(matches(unavailable, minimum_force_n=0.01))

        close_node = next(
            node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "_close_gripper"
        )
        call_names = {
            node.func.id
            for node in ast.walk(close_node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_finger_contact_snapshot", call_names)
        self.assertIn("_filtered_bilateral_contact_matches_selected_object", call_names)

    def test_dual_isaac_unpins_incoming_part_before_loaded_transport(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_simple_dual_robot_sim_in_isaac.py"
        parsed = ast.parse(script_path.read_text(encoding="utf-8"))
        main_node = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        transport_call = next(
            node
            for node in ast.walk(main_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_execute_segments"
            and any(
                keyword.arg == "segments" and "transport_labels" in ast.unparse(keyword.value)
                for keyword in node.keywords
            )
        )
        callback = next(keyword.value for keyword in transport_call.keywords if keyword.arg == "step_callback")

        self.assertEqual(ast.unparse(callback), "_capture_step_callback")
        capture_node = next(
            node
            for node in main_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "_capture_step_callback"
        )
        self.assertNotIn("_pin_incoming_at_pickup", ast.unparse(capture_node))

    def test_dual_isaac_orientation_validation_keeps_same_base_symmetries(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_simple_dual_robot_sim_in_isaac.py"
        parsed = ast.parse(script_path.read_text(encoding="utf-8"))
        helper_names = {
            "_vec",
            "_pose",
            "_distance",
            "_quaternion_distance_rad",
            "_pose_matrix",
            "_finite_symmetry_powers",
            "_expected_incoming_preinsertion_poses",
        }
        helper_nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
        isolated_module = ast.Module(body=helper_nodes, type_ignores=[])
        ast.fix_missing_locations(isolated_module)
        from grasp_planning.grasping.fabrica_grasp_debug import (
            quat_to_rotmat_xyzw,
            rotmat_to_quat_xyzw,
        )

        namespace = {
            "math": math,
            "np": __import__("numpy"),
            "quat_to_rotmat_xyzw": quat_to_rotmat_xyzw,
            "rotmat_to_quat_xyzw": rotmat_to_quat_xyzw,
        }
        exec(compile(isolated_module, str(script_path), "exec"), namespace)

        def candidate(candidate_id: str, base_x: float, orientation) -> dict[str, object]:
            return {
                "execution_candidate_id": candidate_id,
                "objects": {
                    "subassembly": {
                        "source_pose_world": {
                            "position_world_m": [base_x, 0.0, 0.0],
                            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                        }
                    },
                    "incoming": {
                        "preinsertion_source_pose_world": {
                            "position_world_m": [0.5, 0.1, 0.1],
                            "orientation_xyzw_world": list(orientation),
                        }
                    },
                },
            }

        selected = candidate("selected", 0.5, (0.0, 0.0, 0.0, 1.0))
        same_base_symmetry = candidate("symmetric", 0.5, (0.0, 0.0, 1.0, 0.0))
        different_base = candidate("other_base", 0.7, (0.0, 1.0, 0.0, 0.0))
        selected["ranked_pair_candidates"] = [
            selected,
            same_base_symmetry,
            different_base,
        ]

        expected = namespace["_expected_incoming_preinsertion_poses"](selected)

        self.assertEqual(
            [pose["execution_candidate_id"] for pose in expected],
            ["selected", "symmetric"],
        )
        selected_only = candidate("selected_only", 0.5, (0.0, 0.0, 0.0, 1.0))
        selected_only["transition_symmetry"] = {
            "incoming_symmetry_source_m": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        }
        selected_equivalents = namespace["_expected_incoming_preinsertion_poses"](selected_only)
        self.assertEqual(len(selected_equivalents), 2)
        self.assertEqual(
            [pose["symmetry_power"] for pose in selected_equivalents],
            [0, 1],
        )
        quaternion_distance = namespace["_quaternion_distance_rad"]
        self.assertAlmostEqual(
            quaternion_distance((1.0, 0.0, 0.0, 0.0), (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))),
            math.pi / 2.0,
        )

        main_source = ast.unparse(
            next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        )
        self.assertIn("base_orientation_error", main_source)
        self.assertIn("incoming_preinsertion_orientation_error", main_source)

    def test_moveit_pick_allows_source_open_kuka_width_during_approach(self) -> None:
        _FakeExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeKukaContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _FakeExecutor),
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={"gripper_close_status": "target_reached"},
            ),
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
                closed_gripper_width=0.0,
                pregrasp_only=False,
            )

        self.assertTrue(result.success)
        for actual, expected in zip(
            _FakeExecutor.executions,
            [(0.1, KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M), (0.2, KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M), (0.3, 0.0)],
            strict=True,
        ):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])
        self.assertAlmostEqual(
            result.diagnostics["nominal_max_open_gripper_width_m"], KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M
        )
        self.assertFalse(result.diagnostics["open_gripper_width_exceeds_nominal_limit"])

    def test_gripper_close_waits_for_kuka_finger_convergence_before_lift(self) -> None:
        import torch

        class FakeSim:
            def get_physics_dt(self) -> float:
                return 0.1

            def step(self) -> None:
                pass

        class FakeRobot:
            joint_names = ["left_finger_joint", "right_finger_joint"]
            device = "cpu"

            def __init__(self) -> None:
                self.data = SimpleNamespace(joint_pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32))
                self.hand_target = torch.tensor([[0.0]], dtype=torch.float32)

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                if list(joint_ids) == [0]:
                    self.hand_target = target.detach().cpu().clone()

        class FakeScene:
            def __init__(self, robot: FakeRobot) -> None:
                self.robot = robot

            def write_data_to_sim(self) -> None:
                pass

            def update(self, physics_dt: float) -> None:
                delta = torch.clamp(
                    self.robot.hand_target - self.robot.data.joint_pos[:, [0]],
                    min=-0.01,
                    max=0.01,
                )
                self.robot.data.joint_pos[:, [0]] = self.robot.data.joint_pos[:, [0]] + delta
                self.robot.data.joint_pos[:, [1]] = -self.robot.data.joint_pos[:, [0]]

        class FakeHoldContext:
            def __init__(self) -> None:
                self.arm_commands = 0

            def command_arm(self, waypoint) -> None:
                self.arm_commands += 1

        robot = FakeRobot()
        hold_context = FakeHoldContext()

        diagnostics = pick_execution._command_gripper_width(
            sim=FakeSim(),
            scene=FakeScene(robot),
            robot=robot,
            width=0.0,
            duration_s=0.2,
            max_duration_s=1.0,
            settle_duration_s=0.2,
            hold_context=hold_context,
            hold_arm_waypoint=torch.tensor([[0.2]], dtype=torch.float32),
        )

        self.assertEqual(diagnostics["gripper_close_status"], "target_reached")
        self.assertGreater(diagnostics["gripper_close_steps"], 2)
        self.assertEqual(hold_context.arm_commands, diagnostics["gripper_close_steps"])
        for actual, expected in zip(
            diagnostics["gripper_close_target_joint_positions"],
            [KUKA_Y_GRIPPER_TRAVEL_M],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertLessEqual(diagnostics["gripper_close_final_max_position_error"], 0.001)

    def test_gripper_close_uses_gentle_ramp_and_latches_bilateral_contact(self) -> None:
        import torch

        class FakeSim:
            def get_physics_dt(self) -> float:
                return 0.1

            def step(self) -> None:
                pass

        class FakeRobot:
            joint_names = ["left_finger_joint", "right_finger_joint"]
            device = "cpu"

            def __init__(self) -> None:
                self.data = SimpleNamespace(joint_pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32))
                self.hand_target = torch.tensor([[0.0]], dtype=torch.float32)
                self.commanded_targets: list[float] = []

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                if list(joint_ids) == [0]:
                    self.hand_target = target.detach().cpu().clone()
                    self.commanded_targets.append(float(target[0, 0]))

        class FakeScene:
            def __init__(self, robot: FakeRobot) -> None:
                self.robot = robot

            def write_data_to_sim(self) -> None:
                pass

            def update(self, physics_dt: float) -> None:
                self.robot.data.joint_pos[:, [0]] = self.robot.hand_target
                self.robot.data.joint_pos[:, [1]] = -self.robot.hand_target

        robot = FakeRobot()
        contact_checks = 0

        def bilateral_contact() -> bool:
            nonlocal contact_checks
            contact_checks += 1
            return contact_checks >= 2

        diagnostics = pick_execution._command_gripper_width(
            sim=FakeSim(),
            scene=FakeScene(robot),
            robot=robot,
            width=0.0,
            duration_s=0.4,
            max_duration_s=1.0,
            settle_duration_s=0.1,
            stop_on_contact=bilateral_contact,
            contact_preload_m=0.0004,
            contact_hold_width_m=0.040,
        )

        self.assertEqual(diagnostics["gripper_close_target_profile"], "quintic_smoothstep")
        self.assertEqual(diagnostics["gripper_close_status"], "contact_latched")
        self.assertTrue(diagnostics["gripper_close_contact_latched"])
        self.assertEqual(diagnostics["gripper_close_contact_latched_step"], 2)
        self.assertLess(robot.commanded_targets[0], KUKA_Y_GRIPPER_TRAVEL_M)
        self.assertLess(robot.commanded_targets[-1], KUKA_Y_GRIPPER_TRAVEL_M)
        hold_target = diagnostics["gripper_close_contact_hold_joint_positions"][0]
        self.assertAlmostEqual(
            hold_target,
            gripper_joint_target_from_width("left_finger_joint", 0.040) + 0.0004,
            places=6,
        )
        self.assertAlmostEqual(robot.commanded_targets[-1], hold_target, places=6)

    def test_gripper_close_does_not_treat_stalled_fingers_as_closed(self) -> None:
        import torch

        class FakeSim:
            def get_physics_dt(self) -> float:
                return 0.1

            def step(self) -> None:
                pass

        class FakeRobot:
            joint_names = ["left_finger_joint", "right_finger_joint"]
            device = "cpu"

            def __init__(self) -> None:
                self.data = SimpleNamespace(joint_pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32))

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                pass

        class FakeScene:
            def write_data_to_sim(self) -> None:
                pass

            def update(self, physics_dt: float) -> None:
                pass

        diagnostics = pick_execution._command_gripper_width(
            sim=FakeSim(),
            scene=FakeScene(),
            robot=FakeRobot(),
            width=0.0,
            duration_s=0.2,
            max_duration_s=0.5,
            settle_duration_s=0.2,
        )

        self.assertEqual(diagnostics["gripper_close_status"], "max_duration_elapsed")
        self.assertEqual(diagnostics["gripper_close_steps"], 5)
        self.assertGreater(diagnostics["gripper_close_final_max_position_error"], 0.001)

    def test_gripper_close_accepts_near_target_contact_stall(self) -> None:
        import torch

        class FakeSim:
            def get_physics_dt(self) -> float:
                return 0.1

            def step(self) -> None:
                pass

        class FakeRobot:
            joint_names = ["left_finger_joint", "right_finger_joint"]
            device = "cpu"

            def __init__(self) -> None:
                self.data = SimpleNamespace(joint_pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32))
                self.hand_target = torch.tensor([[0.0]], dtype=torch.float32)

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                if list(joint_ids) == [0]:
                    self.hand_target = target.detach().cpu().clone()

        class FakeScene:
            def __init__(self, robot: FakeRobot) -> None:
                self.robot = robot

            def write_data_to_sim(self) -> None:
                pass

            def update(self, physics_dt: float) -> None:
                residual = torch.full_like(self.robot.hand_target, 0.0015)
                direction = torch.sign(self.robot.hand_target - self.robot.data.joint_pos[:, [0]])
                blocked_target = self.robot.hand_target - direction * residual
                delta = torch.clamp(blocked_target - self.robot.data.joint_pos[:, [0]], min=-0.01, max=0.01)
                self.robot.data.joint_pos[:, [0]] = self.robot.data.joint_pos[:, [0]] + delta
                self.robot.data.joint_pos[:, [1]] = -self.robot.data.joint_pos[:, [0]]

        robot = FakeRobot()
        diagnostics = pick_execution._command_gripper_width(
            sim=FakeSim(),
            scene=FakeScene(robot),
            robot=robot,
            width=0.0,
            duration_s=0.2,
            max_duration_s=1.0,
            settle_duration_s=0.2,
        )

        self.assertEqual(diagnostics["gripper_close_status"], "contact_stalled")
        self.assertGreater(diagnostics["gripper_close_max_motion_since_start_m"], 0.001)
        self.assertLessEqual(diagnostics["gripper_close_final_max_position_error"], 0.003)

    def test_gripper_close_accepts_physical_contact_stall_before_target(self) -> None:
        import torch

        class FakeSim:
            def get_physics_dt(self) -> float:
                return 0.1

            def step(self) -> None:
                pass

        class FakeRobot:
            joint_names = ["left_finger_joint", "right_finger_joint"]
            device = "cpu"

            def __init__(self) -> None:
                self.data = SimpleNamespace(joint_pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32))
                self.hand_target = torch.tensor([[0.0]], dtype=torch.float32)

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                if list(joint_ids) == [0]:
                    self.hand_target = target.detach().cpu().clone()

        class FakeScene:
            def __init__(self, robot: FakeRobot) -> None:
                self.robot = robot

            def write_data_to_sim(self) -> None:
                pass

            def update(self, physics_dt: float) -> None:
                blocked_q = torch.tensor([[0.010]], dtype=torch.float32)
                delta = torch.clamp(blocked_q - self.robot.data.joint_pos[:, [0]], min=-0.003, max=0.003)
                self.robot.data.joint_pos[:, [0]] = self.robot.data.joint_pos[:, [0]] + delta
                self.robot.data.joint_pos[:, [1]] = -self.robot.data.joint_pos[:, [0]]

        robot = FakeRobot()
        diagnostics = pick_execution._command_gripper_width(
            sim=FakeSim(),
            scene=FakeScene(robot),
            robot=robot,
            width=0.048,
            duration_s=0.2,
            max_duration_s=1.0,
            settle_duration_s=0.2,
        )

        self.assertEqual(diagnostics["gripper_close_status"], "contact_stalled")
        self.assertGreater(diagnostics["gripper_close_max_motion_since_start_m"], 0.001)
        self.assertGreater(diagnostics["gripper_close_final_max_position_error"], 0.003)

    def test_gripper_close_reports_stationary_max_duration_as_contact_stall(self) -> None:
        import torch

        class FakeSim:
            def get_physics_dt(self) -> float:
                return 0.1

            def step(self) -> None:
                pass

        class FakeRobot:
            joint_names = ["left_finger_joint", "right_finger_joint"]
            device = "cpu"

            def __init__(self) -> None:
                self.data = SimpleNamespace(joint_pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32))

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                pass

        class FakeScene:
            def __init__(self, robot: FakeRobot) -> None:
                self.robot = robot

            def write_data_to_sim(self) -> None:
                pass

            def update(self, physics_dt: float) -> None:
                blocked_q = torch.tensor([[0.013]], dtype=torch.float32)
                self.robot.data.joint_pos[:, [0]] = blocked_q
                self.robot.data.joint_pos[:, [1]] = -blocked_q

        robot = FakeRobot()
        diagnostics = pick_execution._command_gripper_width(
            sim=FakeSim(),
            scene=FakeScene(robot),
            robot=robot,
            width=0.0,
            duration_s=0.2,
            max_duration_s=0.3,
            settle_duration_s=1.0,
        )

        self.assertEqual(diagnostics["gripper_close_status"], "contact_stalled")
        self.assertGreater(diagnostics["gripper_close_max_motion_since_start_m"], 0.001)
        self.assertGreater(diagnostics["gripper_close_final_max_position_error"], 0.003)

    def test_forced_gripper_close_writes_kuka_fingers_to_closed_target(self) -> None:
        import torch

        class FakeSim:
            def get_physics_dt(self) -> float:
                return 0.1

            def step(self) -> None:
                pass

        class FakeRobot:
            joint_names = ["left_finger_joint", "right_finger_joint"]
            device = "cpu"

            def __init__(self) -> None:
                self.data = SimpleNamespace(joint_pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32))
                self.position_targets = []
                self.state_writes = []

            def set_joint_position_target(self, target, *, joint_ids) -> None:
                self.position_targets.append((target.detach().cpu().clone(), list(joint_ids)))

            def write_joint_state_to_sim(self, q, qd, *, joint_ids) -> None:
                self.state_writes.append((q.detach().cpu().clone(), qd.detach().cpu().clone(), list(joint_ids)))
                self.data.joint_pos[:, joint_ids] = q.detach().cpu()
                if list(joint_ids) == [0]:
                    self.data.joint_pos[:, [1]] = -self.data.joint_pos[:, [0]]

        class FakeScene:
            def write_data_to_sim(self) -> None:
                pass

            def update(self, physics_dt: float) -> None:
                pass

        robot = FakeRobot()
        diagnostics = pick_execution._command_gripper_width(
            sim=FakeSim(),
            scene=FakeScene(),
            robot=robot,
            width=0.0,
            duration_s=0.2,
            max_duration_s=1.0,
            settle_duration_s=0.2,
            force_joint_state=True,
        )

        self.assertEqual(diagnostics["gripper_close_status"], "target_reached")
        self.assertTrue(diagnostics["gripper_close_forced_joint_state"])
        self.assertGreaterEqual(len(robot.state_writes), 2)
        self.assertLessEqual(diagnostics["gripper_close_final_max_position_error"], 0.001)
        for actual, expected in zip(
            robot.data.joint_pos[0].tolist(),
            [KUKA_Y_GRIPPER_TRAVEL_M, -KUKA_Y_GRIPPER_TRAVEL_M],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_moveit_pick_stops_before_lift_when_gripper_close_fails(self) -> None:
        _FakeExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeKukaContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _FakeExecutor),
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={"gripper_close_status": "max_duration_elapsed"},
            ),
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
                closed_gripper_width=0.0,
                pregrasp_only=False,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status, "gripper_close_failed")
        self.assertEqual(len(_FakeExecutor.executions), 2)

    def test_isaac_runner_validates_kuka_contact_against_jaw_width_without_clearance(self) -> None:
        runner_path = Path(__file__).resolve().parents[1] / "scripts" / "run_fabrica_grasp_in_isaac.py"
        tree = ast.parse(runner_path.read_text(encoding="utf-8"))
        execution_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "execute_pick_from_moveit_joint_trajectories"
        ]

        self.assertEqual(len(execution_calls), 1)
        selected_width = next(
            keyword.value for keyword in execution_calls[0].keywords if keyword.arg == "selected_gripper_width_m"
        )
        self.assertEqual(ast.unparse(selected_width), "float(selected_world_grasp.jaw_width)")

        contact_diagnostics = {
            "gripper_close_joint_names": ["left_finger_joint"],
            "gripper_close_final_joint_positions": [0.010],
            "gripper_close_final_max_step_delta": 0.0,
        }
        clearance_expanded = dict(contact_diagnostics)
        actual_jaw = dict(contact_diagnostics)
        self.assertTrue(pick_execution._kuka_contact_stall_matches_grasp_width(clearance_expanded, 0.0593))
        self.assertFalse(pick_execution._kuka_contact_stall_matches_grasp_width(actual_jaw, 0.0493))

    def test_moveit_pick_accepts_kuka_contact_at_selected_grasp_width(self) -> None:
        _FakeExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeKukaContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _FakeExecutor),
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={
                    "gripper_close_status": "contact_stalled",
                    "gripper_close_joint_names": ["left_finger_joint"],
                    "gripper_close_final_joint_positions": [0.0132],
                    "gripper_close_final_max_step_delta": 0.0,
                },
            ),
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
                closed_gripper_width=0.0,
                selected_gripper_width_m=0.0593,
                pregrasp_only=False,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(_FakeExecutor.executions), 3)
        self.assertTrue(result.diagnostics["gripper_close_contact_stall_accepted"])

    def test_moveit_pick_rejects_kuka_contact_before_selected_grasp_width(self) -> None:
        _FakeExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeKukaContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _FakeExecutor),
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={
                    "gripper_close_status": "contact_stalled",
                    "gripper_close_joint_names": ["left_finger_joint"],
                    "gripper_close_final_joint_positions": [0.001],
                    "gripper_close_final_max_step_delta": 0.0,
                },
            ),
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
                closed_gripper_width=0.0,
                selected_gripper_width_m=0.0593,
                pregrasp_only=False,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status, "gripper_close_failed")
        self.assertEqual(len(_FakeExecutor.executions), 2)
        self.assertFalse(result.diagnostics["gripper_close_contact_stall_accepted"])

    def test_moveit_pick_closes_gripper_when_grasp_waypoint_does_not_settle(self) -> None:
        _GraspSettleFailingExecutor.executions = []
        trajectories = {
            "pregrasp": ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "grasp": ((0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
            "lift": ((0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        }

        with (
            mock.patch.object(pick_execution, "FR3MotionContext", _FakeKukaContext),
            mock.patch.object(pick_execution, "TrajectoryExecutor", _GraspSettleFailingExecutor),
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={"gripper_close_status": "target_reached"},
            ) as command_gripper_width,
        ):
            result = pick_execution.execute_pick_from_moveit_joint_trajectories(
                sim=object(),
                scene=object(),
                robot=object(),
                moveit_joint_trajectories=trajectories,
                open_gripper_width=KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
                closed_gripper_width=0.048,
                pregrasp_only=False,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "ok")
        self.assertFalse(result.diagnostics["grasp_waypoint_settled"])
        self.assertIn("final waypoint did not settle", result.diagnostics["grasp_waypoint_settle_detail"])
        command_gripper_width.assert_called_once()
        self.assertEqual(command_gripper_width.call_args.kwargs["width"], 0.048)
        expected = [
            (0.1, KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M),
            (0.2, KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M),
            (0.3, 0.048),
        ]
        for actual, expected_values in zip(_GraspSettleFailingExecutor.executions, expected, strict=True):
            self.assertAlmostEqual(actual[0], expected_values[0])
            self.assertAlmostEqual(actual[1], expected_values[1])

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
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={"gripper_close_status": "target_reached"},
            ),
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
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={"gripper_close_status": "target_reached"},
            ),
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
            mock.patch.object(
                pick_execution,
                "_command_gripper_width",
                return_value={"gripper_close_status": "target_reached"},
            ),
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
