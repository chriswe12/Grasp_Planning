from __future__ import annotations

import math
from argparse import Namespace
from types import SimpleNamespace
from unittest import mock

import numpy as np

from grasp_planning.ros2.moveit_pose_commander import (
    DEFAULT_FR3_MOVEIT_RPY,
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    PoseTarget,
    normalize_quaternion_xyzw,
    quaternion_from_rpy,
)
from scripts.move_real_robot_ee import commander_config_from_args, pose_target_from_args


def test_normalize_quaternion_xyzw_returns_unit_quaternion() -> None:
    quaternion = normalize_quaternion_xyzw((0.0, 0.0, 0.0, 2.0))
    np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0), atol=1.0e-9)


def test_quaternion_from_rpy_identity() -> None:
    quaternion = quaternion_from_rpy(0.0, 0.0, 0.0)
    np.testing.assert_allclose(quaternion, (0.0, 0.0, 0.0, 1.0), atol=1.0e-9)


def test_pose_target_from_rpy_normalizes_orientation() -> None:
    target = PoseTarget.from_rpy(x=0.1, y=0.2, z=0.3, roll=math.pi, pitch=0.0, yaw=math.pi / 2.0, frame_id="base")
    norm = math.sqrt(sum(component * component for component in target.orientation_xyzw))
    assert math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-9)


def test_pose_target_from_args_uses_default_thesis_orientation() -> None:
    args = Namespace(
        x=0.30,
        y=0.01,
        z=0.40,
        frame_id="base",
        keep_current_orientation=False,
        roll=None,
        pitch=None,
        yaw=None,
        qx=None,
        qy=None,
        qz=None,
        qw=None,
    )

    target = pose_target_from_args(args)

    expected = PoseTarget.from_rpy(
        x=0.30,
        y=0.01,
        z=0.40,
        roll=DEFAULT_FR3_MOVEIT_RPY[0],
        pitch=DEFAULT_FR3_MOVEIT_RPY[1],
        yaw=DEFAULT_FR3_MOVEIT_RPY[2],
        frame_id="base",
    )
    np.testing.assert_allclose(target.orientation_xyzw, expected.orientation_xyzw, atol=1.0e-9)


def test_pose_target_from_args_prefers_quaternion_when_provided() -> None:
    args = Namespace(
        x=0.10,
        y=-0.20,
        z=0.30,
        frame_id="map",
        keep_current_orientation=False,
        roll=0.1,
        pitch=0.2,
        yaw=0.3,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        qw=2.0,
    )

    target = pose_target_from_args(args)

    np.testing.assert_allclose(target.orientation_xyzw, (0.0, 0.0, 0.0, 1.0), atol=1.0e-9)
    assert target.frame_id == "map"


def test_pose_target_from_args_rejects_partial_quaternion() -> None:
    args = Namespace(
        x=0.10,
        y=-0.20,
        z=0.30,
        frame_id="base",
        keep_current_orientation=False,
        roll=None,
        pitch=None,
        yaw=None,
        qx=0.0,
        qy=None,
        qz=0.0,
        qw=1.0,
    )

    try:
        pose_target_from_args(args)
    except ValueError as exc:
        assert "provide all of --qx --qy --qz --qw" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected pose_target_from_args to reject partial quaternion inputs.")


def test_pose_target_from_args_uses_current_orientation_when_requested() -> None:
    args = Namespace(
        x=0.40,
        y=0.05,
        z=0.25,
        frame_id="base",
        keep_current_orientation=True,
        roll=None,
        pitch=None,
        yaw=None,
        qx=None,
        qy=None,
        qz=None,
        qw=None,
    )

    target = pose_target_from_args(args, current_orientation_xyzw=(0.0, 0.0, 0.70710678, 0.70710678))

    np.testing.assert_allclose(target.orientation_xyzw, (0.0, 0.0, 0.70710678, 0.70710678), atol=1.0e-8)


def test_pose_target_from_args_rejects_keep_current_orientation_with_explicit_orientation() -> None:
    args = Namespace(
        x=0.40,
        y=0.05,
        z=0.25,
        frame_id="base",
        keep_current_orientation=True,
        roll=None,
        pitch=None,
        yaw=0.5,
        qx=None,
        qy=None,
        qz=None,
        qw=None,
    )

    try:
        pose_target_from_args(args, current_orientation_xyzw=(0.0, 0.0, 0.0, 1.0))
    except ValueError as exc:
        assert "--keep-current-orientation cannot be combined" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected pose_target_from_args to reject mixed explicit and current-orientation inputs.")


def test_commander_config_from_args_uses_slow_defaults() -> None:
    args = Namespace(
        planning_group="fr3_arm",
        pose_link="fr3_hand_tcp",
        moveit_namespace="",
        joint_names="",
        planner_id="",
        wait_for_moveit_timeout_s=15.0,
        ik_timeout_s=2.0,
        planning_time_s=5.0,
        num_planning_attempts=5,
        velocity_scale=0.05,
        acceleration_scale=0.05,
        execute_timeout_s=120.0,
        post_execute_sleep_s=0.5,
        allow_collisions=False,
    )

    config = commander_config_from_args(args)

    assert config.pipeline_id == ""
    assert math.isclose(config.velocity_scale, 0.05)
    assert math.isclose(config.acceleration_scale, 0.05)


def test_moveit_pose_commander_config_prefixes_default_endpoints_with_namespace() -> None:
    config = MoveItPoseCommanderConfig(moveit_namespace="lbr")

    assert config.moveit_namespace == "/lbr"
    assert config.ik_service_name == "/lbr/compute_ik"
    assert config.planning_service_name == "/lbr/plan_kinematic_path"
    assert config.query_planner_interface_service_name == "/lbr/query_planner_interface"
    assert config.fk_service_name == "/lbr/compute_fk"
    assert config.apply_planning_scene_service_name == "/lbr/apply_planning_scene"
    assert config.execute_action_name == "/lbr/execute_trajectory"


def test_moveit_pose_commander_config_does_not_double_prefix_namespaced_endpoints() -> None:
    config = MoveItPoseCommanderConfig(
        moveit_namespace="/lbr",
        ik_service_name="/lbr/compute_ik",
        execute_action_name="lbr/execute_trajectory",
    )

    assert config.ik_service_name == "/lbr/compute_ik"
    assert config.execute_action_name == "/lbr/execute_trajectory"


def test_remove_planning_scene_obstacles_sends_remove_operations() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(
        moveit_namespace="/lbr_dual_arm",
    )
    commander._apply_planning_scene_client = mock.Mock()
    commander._apply_planning_scene_client.wait_for_service.return_value = True
    commander._apply_planning_scene_client.call_async.return_value = object()
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(success=True)

    with mock.patch.dict(
        MoveItPoseCommander.remove_planning_scene_obstacles.__globals__,
        {
            "CollisionObject": _FakeCollisionObject,
            "PlanningScene": _FakePlanningScene,
            "ApplyPlanningScene": _FakeApplyPlanningScene,
        },
    ):
        ok, message = commander.remove_planning_scene_obstacles(
            ("base_aabb", "incoming_aabb"),
            default_frame_id="base_link",
        )

    request = commander._apply_planning_scene_client.call_async.call_args.args[0]
    objects = request.scene.world.collision_objects
    assert ok is True
    assert message == "Removed 2 planning-scene obstacle(s)."
    assert [value.id for value in objects] == [
        "base_aabb",
        "incoming_aabb",
    ]
    assert all(value.operation == _FakeCollisionObject.REMOVE for value in objects)
    assert all(value.header.frame_id == "base_link" for value in objects)


def test_commander_config_from_args_accepts_lbr_moveit_settings() -> None:
    args = Namespace(
        planning_group="arm",
        pose_link="lbr_link_ee",
        moveit_namespace="/lbr",
        joint_names="lbr_A1,lbr_A2,lbr_A3,lbr_A4,lbr_A5,lbr_A6,lbr_A7",
        planner_id="",
        wait_for_moveit_timeout_s=15.0,
        ik_timeout_s=2.0,
        planning_time_s=5.0,
        num_planning_attempts=5,
        velocity_scale=0.05,
        acceleration_scale=0.05,
        execute_timeout_s=120.0,
        post_execute_sleep_s=0.5,
        allow_collisions=False,
    )

    config = commander_config_from_args(args)

    assert config.planning_group == "arm"
    assert config.pose_link == "lbr_link_ee"
    assert config.joint_names == ("lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4", "lbr_A5", "lbr_A6", "lbr_A7")
    assert config.ik_service_name == "/lbr/compute_ik"


def test_compute_ik_without_seed_uses_current_planning_scene_as_diff() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(
        planning_group="arm_one",
        pose_link="lbr_one_gripper_tcp",
        joint_names=("lbr_one_A1",),
    )
    commander._ik_client = mock.Mock()
    commander._ik_client.call_async.return_value = object()
    commander._pose_stamped = lambda target: mock.Mock()
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        solution=SimpleNamespace(
            joint_state=SimpleNamespace(name=["lbr_one_A1"], position=[0.25]),
        ),
    )

    with mock.patch.dict(
        MoveItPoseCommander.compute_ik.__globals__,
        {"GetPositionIK": _FakeGetPositionIK, "MoveItErrorCodes": _FakeMoveItErrorCodes},
    ):
        joints, message = commander.compute_ik(
            PoseTarget.from_quaternion(
                x=0.4,
                y=-0.2,
                z=0.5,
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            )
        )

    request = commander._ik_client.call_async.call_args.args[0]
    assert request.ik_request.robot_state.is_diff is True
    assert request.ik_request.robot_state.joint_state.name == []
    assert joints == [0.25]
    assert message == "ok"


def test_compute_ik_with_seed_uses_explicit_active_group_seed_state() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(
        planning_group="arm_two",
        pose_link="lbr_two_gripper_tcp",
        joint_names=("lbr_two_A1",),
    )
    commander._ik_client = mock.Mock()
    commander._ik_client.call_async.return_value = object()
    commander._pose_stamped = lambda target: mock.Mock()
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        solution=SimpleNamespace(
            joint_state=SimpleNamespace(name=["lbr_two_A1"], position=[-0.5]),
        ),
    )

    with mock.patch.dict(
        MoveItPoseCommander.compute_ik.__globals__,
        {"GetPositionIK": _FakeGetPositionIK, "MoveItErrorCodes": _FakeMoveItErrorCodes},
    ):
        joints, message = commander.compute_ik(
            PoseTarget.from_quaternion(
                x=0.4,
                y=0.2,
                z=0.5,
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
            seed_joint_positions=(-0.4,),
        )

    request = commander._ik_client.call_async.call_args.args[0]
    assert request.ik_request.robot_state.is_diff is False
    assert request.ik_request.robot_state.joint_state.name == ["lbr_two_A1"]
    assert request.ik_request.robot_state.joint_state.position == [-0.4]
    assert joints == [-0.5]
    assert message == "ok"


def test_compute_ik_with_complete_dual_robot_seed_merges_full_state_as_diff() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(
        planning_group="arm_two",
        pose_link="lbr_two_gripper_tcp",
        joint_names=("lbr_two_A1",),
    )
    commander._ik_client = mock.Mock()
    commander._ik_client.call_async.return_value = object()
    commander._pose_stamped = lambda target: mock.Mock()
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        solution=SimpleNamespace(
            joint_state=SimpleNamespace(name=["lbr_two_A1"], position=[-0.5]),
        ),
    )

    with mock.patch.dict(
        MoveItPoseCommander.compute_ik.__globals__,
        {"GetPositionIK": _FakeGetPositionIK, "MoveItErrorCodes": _FakeMoveItErrorCodes},
    ):
        joints, message = commander.compute_ik(
            PoseTarget.from_quaternion(
                x=0.4,
                y=0.2,
                z=0.5,
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
            seed_robot_state={"lbr_one_A1": 0.3, "lbr_two_A1": -0.4},
        )

    request = commander._ik_client.call_async.call_args.args[0]
    assert request.ik_request.robot_state.is_diff is True
    assert request.ik_request.robot_state.joint_state.name == ["lbr_one_A1", "lbr_two_A1"]
    assert request.ik_request.robot_state.joint_state.position == [0.3, -0.4]
    assert joints == [-0.5]
    assert message == "ok"


def test_compute_ik_can_disable_collision_rejection_per_request() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(
        planning_group="arm_one",
        pose_link="lbr_one_gripper_tcp",
        joint_names=("lbr_one_A1",),
        avoid_collisions=True,
    )
    commander._ik_client = mock.Mock()
    commander._ik_client.call_async.return_value = object()
    commander._pose_stamped = lambda target: mock.Mock()
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        solution=SimpleNamespace(joint_state=SimpleNamespace(name=["lbr_one_A1"], position=[0.2])),
    )

    with mock.patch.dict(
        MoveItPoseCommander.compute_ik.__globals__,
        {"GetPositionIK": _FakeGetPositionIK, "MoveItErrorCodes": _FakeMoveItErrorCodes},
    ):
        joints, message = commander.compute_ik(
            PoseTarget.from_quaternion(
                x=0.4,
                y=-0.2,
                z=0.5,
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
            seed_robot_state={"lbr_one_A1": 0.0, "lbr_two_A1": 0.1},
            avoid_collisions=False,
        )

    request = commander._ik_client.call_async.call_args.args[0]
    assert request.ik_request.avoid_collisions is False
    assert joints == [0.2]
    assert message == "ok"


def test_check_state_validity_returns_exact_contacts() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(
        planning_group="arm_one",
        joint_names=("lbr_one_A1",),
    )
    commander._state_validity_client = mock.Mock()
    commander._state_validity_client.wait_for_service.return_value = True
    commander._state_validity_client.call_async.return_value = object()
    contact = SimpleNamespace(
        contact_body_1="lbr_one_right_finger_link",
        body_type_1=0,
        contact_body_2="dual_sim_work_surface",
        body_type_2=1,
        depth=0.004,
        position=SimpleNamespace(x=0.5, y=-0.2, z=-0.03),
        normal=SimpleNamespace(x=0.0, y=0.0, z=1.0),
    )
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(
        valid=False,
        contacts=[contact],
        cost_sources=[],
        constraint_result=[],
    )

    with mock.patch.dict(
        MoveItPoseCommander.check_state_validity.__globals__,
        {"GetStateValidity": _FakeGetStateValidity},
    ):
        result, message = commander.check_state_validity(
            {"lbr_one_A1": 0.2, "lbr_two_A1": -0.1},
            group_name="",
        )

    request = commander._state_validity_client.call_async.call_args.args[0]
    assert request.robot_state.is_diff is True
    assert request.robot_state.joint_state.name == ["lbr_one_A1", "lbr_two_A1"]
    assert result is not None
    assert result["valid"] is False
    assert result["contacts"][0]["body_1"] == "lbr_one_right_finger_link"
    assert result["contacts"][0]["body_2"] == "dual_sim_work_surface"
    assert result["contacts"][0]["depth_m"] == 0.004
    assert message == "ok"


def test_validate_requested_pipeline_accepts_available_pipeline() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(pipeline_id="isaac_ros_cumotion")
    commander._planner_query_client = _FakePlannerQueryClient()
    commander._wait_for_future = lambda future, *, timeout_s, label: _FakePlannerQueryResponse(
        ["move_group", "isaac_ros_cumotion"]
    )

    with _patch_query_planner_interfaces(_FakeQueryPlannerInterfaces):
        commander._validate_requested_pipeline()


def test_validate_requested_pipeline_reports_available_pipeline_ids() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(pipeline_id="isaac_ros_cumotion")
    commander._planner_query_client = _FakePlannerQueryClient()
    commander._wait_for_future = lambda future, *, timeout_s, label: _FakePlannerQueryResponse(["move_group"])

    with _patch_query_planner_interfaces(_FakeQueryPlannerInterfaces):
        try:
            commander._validate_requested_pipeline()
        except RuntimeError as exc:
            assert "isaac_ros_cumotion" in str(exc)
            assert "move_group" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("Expected unavailable pipeline validation to raise.")


def test_validate_requested_pipeline_requires_query_service_type() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(pipeline_id="isaac_ros_cumotion")
    commander._planner_query_client = _FakePlannerQueryClient()
    commander._wait_for_future = lambda future, *, timeout_s, label: _FakePlannerQueryResponse(["move_group"])

    with _patch_query_planner_interfaces(None):
        try:
            commander._validate_requested_pipeline()
        except RuntimeError as exc:
            assert "planner-query service type is unavailable" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("Expected missing planner-query service type to raise.")


class _FakeQueryPlannerInterfaces:
    class Request:
        pass


def _patch_query_planner_interfaces(value):
    return mock.patch.dict(
        MoveItPoseCommander._planner_query_request.__globals__,
        {"QueryPlannerInterfaces": value},
    )


class _FakePlannerQueryClient:
    def call_async(self, request):
        return object()


class _FakePlannerQueryResponse:
    def __init__(self, pipeline_ids: list[str]) -> None:
        self.planner_interfaces = [_FakePlannerInterface(pipeline_id) for pipeline_id in pipeline_ids]


class _FakePlannerInterface:
    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id


class _FakeMoveItErrorCodes:
    SUCCESS = 1


class _FakeGetPositionIK:
    class Request:
        def __init__(self) -> None:
            self.ik_request = SimpleNamespace(
                group_name="",
                ik_link_name="",
                pose_stamped=None,
                avoid_collisions=True,
                robot_state=SimpleNamespace(
                    is_diff=False,
                    joint_state=SimpleNamespace(name=[], position=[]),
                ),
                timeout=SimpleNamespace(sec=0, nanosec=0),
            )


class _FakeGetStateValidity:
    class Request:
        def __init__(self) -> None:
            self.robot_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
            )
            self.group_name = ""


class _FakeCollisionObject:
    REMOVE = 1

    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.id = ""
        self.operation = -1


class _FakePlanningScene:
    def __init__(self) -> None:
        self.is_diff = False
        self.world = SimpleNamespace(collision_objects=[])


class _FakeApplyPlanningScene:
    class Request:
        def __init__(self) -> None:
            self.scene = None
