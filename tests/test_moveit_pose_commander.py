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
    assert config.cartesian_path_service_name == "/lbr/compute_cartesian_path"
    assert config.get_planning_scene_service_name == "/lbr/get_planning_scene"
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


def test_get_current_robot_state_reads_complete_moveit_scene_state() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(moveit_namespace="/lbr_dual_arm")
    commander._get_planning_scene_client = mock.Mock()
    commander._get_planning_scene_client.call_async.return_value = object()
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(
        scene=SimpleNamespace(
            robot_state=SimpleNamespace(
                joint_state=SimpleNamespace(
                    name=["lbr_one_A1", "lbr_two_A1"],
                    position=[0.25, -0.4],
                )
            )
        )
    )

    with mock.patch.dict(
        MoveItPoseCommander.get_current_robot_state.__globals__,
        {
            "GetPlanningScene": _FakeGetPlanningScene,
            "PlanningSceneComponents": SimpleNamespace(ROBOT_STATE=1),
        },
    ):
        state = commander.get_current_robot_state()

    request = commander._get_planning_scene_client.call_async.call_args.args[0]
    assert request.components.components == 1
    assert state == {"lbr_one_A1": 0.25, "lbr_two_A1": -0.4}


def test_plan_cartesian_to_pose_requires_full_path_and_scales_timing() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(
        planning_group="arm_one",
        pose_link="lbr_one_gripper_tcp",
        joint_names=("lbr_one_A1",),
        velocity_scale=0.05,
        acceleration_scale=0.05,
    )
    commander._cartesian_path_client = mock.Mock()
    commander._cartesian_path_client.call_async.return_value = object()
    trajectory = SimpleNamespace(
        joint_trajectory=SimpleNamespace(
            joint_names=["lbr_one_A1"],
            points=[
                SimpleNamespace(
                    positions=[0.0],
                    velocities=[0.0],
                    accelerations=[0.0],
                    time_from_start=SimpleNamespace(sec=0, nanosec=0),
                ),
                SimpleNamespace(
                    positions=[0.1],
                    velocities=[1.0],
                    accelerations=[1.0],
                    time_from_start=SimpleNamespace(sec=1, nanosec=0),
                ),
            ],
        )
    )
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(
        error_code=SimpleNamespace(val=1),
        fraction=1.0,
        solution=trajectory,
    )

    with mock.patch.dict(
        MoveItPoseCommander.plan_cartesian_to_pose.__globals__,
        {
            "GetCartesianPath": _FakeGetCartesianPath,
            "MoveItErrorCodes": _FakeMoveItErrorCodes,
            "Pose": _FakePose,
        },
    ):
        planned, message = commander.plan_cartesian_to_pose(
            PoseTarget.from_quaternion(
                x=0.4,
                y=-0.2,
                z=0.3,
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                frame_id="base_link",
            ),
            label="holder_grasp",
            start_robot_state={"lbr_one_A1": 0.0, "lbr_two_A1": -0.2},
            max_step_m=0.005,
            revolute_jump_threshold_rad=0.35,
        )

    request = commander._cartesian_path_client.call_async.call_args.args[0]
    assert planned is trajectory
    assert "fraction=1.000000" in message
    assert request.header.frame_id == "base_link"
    assert request.start_state.joint_state.name == ["lbr_one_A1", "lbr_two_A1"]
    assert request.max_step == 0.005
    assert request.revolute_jump_threshold == 0.35
    assert request.avoid_collisions is True
    final = trajectory.joint_trajectory.points[-1]
    assert final.time_from_start.sec == 20
    np.testing.assert_allclose(final.velocities, [0.05])
    np.testing.assert_allclose(final.accelerations, [0.0025])


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


def test_apply_planning_scene_retries_once_after_fastdds_response_timeout() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(
        moveit_namespace="/lbr",
        wait_for_moveit_timeout_s=15.0,
        planning_scene_apply_attempts=2,
        planning_scene_first_attempt_timeout_s=2.0,
    )
    commander._apply_planning_scene_client = mock.Mock()
    commander._apply_planning_scene_client.wait_for_service.return_value = True
    commander._apply_planning_scene_client.call_async.side_effect = [object(), object()]
    commander._wait_for_future = mock.Mock(
        side_effect=[
            TimeoutError("first response was not delivered"),
            SimpleNamespace(success=True),
        ]
    )
    logger = mock.Mock()
    commander.get_logger = mock.Mock(return_value=logger)

    with mock.patch.dict(
        MoveItPoseCommander.apply_planning_scene_obstacles.__globals__,
        {
            "CollisionObject": _FakeCollisionObject,
            "PlanningScene": _FakePlanningScene,
            "ApplyPlanningScene": _FakeApplyPlanningScene,
            "Pose": _FakePose,
            "SolidPrimitive": _FakeSolidPrimitive,
        },
    ):
        ok, message = commander.apply_planning_scene_obstacles(
            [
                {
                    "id": "floor",
                    "type": "box",
                    "size_m": [2.0, 2.0, 0.02],
                    "xyz": [0.0, 0.0, -0.01],
                }
            ],
            default_frame_id="lbr_link_0",
        )

    assert ok is True
    assert message == "Applied 1 planning-scene obstacle(s)."
    assert commander._apply_planning_scene_client.call_async.call_count == 2
    assert [call.kwargs["timeout_s"] for call in commander._wait_for_future.call_args_list] == [
        2.0,
        15.0,
    ]
    logger.warning.assert_called_once()


def test_apply_and_remove_attached_collision_box_uses_robot_state_diff() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(moveit_namespace="/lbr_dual_arm")
    commander._apply_planning_scene_client = mock.Mock()
    commander._apply_planning_scene_client.wait_for_service.return_value = True
    commander._apply_planning_scene_client.call_async.return_value = object()
    commander._wait_for_future = lambda future, *, timeout_s, label: SimpleNamespace(success=True)
    obstacle = {
        "id": "incoming_part",
        "type": "box",
        "link_name": "lbr_two_gripper_tcp",
        "frame_id": "lbr_two_gripper_tcp",
        "touch_links": ["lbr_two_left_finger_link", "lbr_two_right_finger_link"],
        "size_m": [0.04, 0.08, 0.05],
        "xyz": [0.0, 0.0, 0.03],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    message_types = {
        "AttachedCollisionObject": _FakeAttachedCollisionObject,
        "CollisionObject": _FakeCollisionObject,
        "PlanningScene": _FakePlanningScene,
        "ApplyPlanningScene": _FakeApplyPlanningScene,
        "Pose": _FakePose,
        "SolidPrimitive": _FakeSolidPrimitive,
    }

    with mock.patch.dict(
        MoveItPoseCommander.apply_planning_scene_attached_obstacles.__globals__,
        message_types,
    ):
        ok, message = commander.apply_planning_scene_attached_obstacles(
            [obstacle],
            default_frame_id="base_link",
        )

    apply_request = commander._apply_planning_scene_client.call_async.call_args.args[0]
    attached = apply_request.scene.robot_state.attached_collision_objects[0]
    assert ok is True
    assert message == "Attached 1 planning-scene obstacle(s)."
    assert apply_request.scene.robot_state.is_diff is True
    assert attached.link_name == "lbr_two_gripper_tcp"
    assert attached.object.id == "incoming_part"
    assert attached.object.operation == _FakeCollisionObject.ADD
    assert attached.touch_links == ["lbr_two_left_finger_link", "lbr_two_right_finger_link"]

    with mock.patch.dict(
        MoveItPoseCommander.remove_planning_scene_attached_obstacles.__globals__,
        message_types,
    ):
        ok, message = commander.remove_planning_scene_attached_obstacles(
            [obstacle],
            default_frame_id="base_link",
        )

    cleanup_requests = [
        call.args[0]
        for call in commander._apply_planning_scene_client.call_async.call_args_list[-2:]
    ]
    remove_request, world_remove_request = cleanup_requests
    removed = remove_request.scene.robot_state.attached_collision_objects[0]
    assert ok is True
    assert message == "Detached and removed 1 planning-scene obstacle(s) from the world."
    assert removed.object.id == "incoming_part"
    assert removed.object.operation == _FakeCollisionObject.REMOVE
    world_removed = world_remove_request.scene.world.collision_objects
    assert [value.id for value in world_removed] == ["incoming_part"]
    assert all(value.operation == _FakeCollisionObject.REMOVE for value in world_removed)


def test_attached_collision_cleanup_leaves_no_attached_or_world_copy() -> None:
    commander = object.__new__(MoveItPoseCommander)
    commander._config = MoveItPoseCommanderConfig(moveit_namespace="/lbr_dual_arm")
    commander._apply_planning_scene_client = _StatefulPlanningSceneClient(
        world_ids={"incoming_part"}
    )
    commander._wait_for_future = (
        lambda future, *, timeout_s, label: SimpleNamespace(success=True)
    )
    obstacle = {
        "id": "incoming_part",
        "type": "box",
        "link_name": "lbr_two_gripper_tcp",
        "frame_id": "lbr_two_gripper_tcp",
        "touch_links": ["lbr_two_left_finger_link", "lbr_two_right_finger_link"],
        "size_m": [0.04, 0.08, 0.05],
        "xyz": [0.0, 0.0, 0.03],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    message_types = {
        "AttachedCollisionObject": _FakeAttachedCollisionObject,
        "CollisionObject": _FakeCollisionObject,
        "PlanningScene": _FakePlanningScene,
        "ApplyPlanningScene": _FakeApplyPlanningScene,
        "Pose": _FakePose,
        "SolidPrimitive": _FakeSolidPrimitive,
    }

    with mock.patch.dict(
        MoveItPoseCommander.apply_planning_scene_attached_obstacles.__globals__,
        message_types,
    ):
        attach_ok, _message = commander.apply_planning_scene_attached_obstacles(
            [obstacle],
            default_frame_id="base_link",
        )
        assert attach_ok
        assert commander._apply_planning_scene_client.attached_ids == {"incoming_part"}
        assert commander._apply_planning_scene_client.world_ids == set()

        cleanup_ok, _message = commander.remove_planning_scene_attached_obstacles(
            [obstacle],
            default_frame_id="base_link",
        )

    assert cleanup_ok
    assert commander._apply_planning_scene_client.attached_ids == set()
    assert commander._apply_planning_scene_client.world_ids == set()


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


class _FakeGetPlanningScene:
    class Request:
        def __init__(self) -> None:
            self.components = SimpleNamespace(components=0)


class _FakeGetCartesianPath:
    class Request:
        def __init__(self) -> None:
            self.header = SimpleNamespace(frame_id="")
            self.start_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
            )
            self.group_name = ""
            self.link_name = ""
            self.waypoints = []
            self.max_step = 0.0
            self.jump_threshold = 0.0
            self.prismatic_jump_threshold = 0.0
            self.revolute_jump_threshold = 0.0
            self.avoid_collisions = False


class _FakeGetStateValidity:
    class Request:
        def __init__(self) -> None:
            self.robot_state = SimpleNamespace(
                is_diff=False,
                joint_state=SimpleNamespace(name=[], position=[]),
            )
            self.group_name = ""


class _FakeCollisionObject:
    ADD = 0
    REMOVE = 1

    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="")
        self.id = ""
        self.operation = -1
        self.primitives = []
        self.primitive_poses = []


class _FakeAttachedCollisionObject:
    def __init__(self) -> None:
        self.link_name = ""
        self.touch_links = []
        self.object = None


class _FakePose:
    def __init__(self) -> None:
        self.position = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)


class _FakeSolidPrimitive:
    BOX = 1

    def __init__(self) -> None:
        self.type = 0
        self.dimensions = []


class _FakePlanningScene:
    def __init__(self) -> None:
        self.is_diff = False
        self.world = SimpleNamespace(collision_objects=[])
        self.robot_state = SimpleNamespace(
            is_diff=False,
            attached_collision_objects=[],
            joint_state=SimpleNamespace(name=[], position=[]),
        )


class _FakeApplyPlanningScene:
    class Request:
        def __init__(self) -> None:
            self.scene = None


class _StatefulPlanningSceneClient:
    """Model MoveIt's detach-to-world behavior for cleanup regression tests."""

    def __init__(self, *, world_ids: set[str]) -> None:
        self.world_ids = set(world_ids)
        self.attached_ids: set[str] = set()
        self.requests = []

    def wait_for_service(self, *, timeout_sec: float) -> bool:
        del timeout_sec
        return True

    def call_async(self, request):
        self.requests.append(request)
        for attached in request.scene.robot_state.attached_collision_objects:
            obstacle_id = str(attached.object.id)
            if attached.object.operation == _FakeCollisionObject.ADD:
                self.world_ids.discard(obstacle_id)
                self.attached_ids.add(obstacle_id)
            elif attached.object.operation == _FakeCollisionObject.REMOVE:
                self.attached_ids.discard(obstacle_id)
                # This is the behavior that caused the benchmark leak: a
                # detach restores the collision object to the world.
                self.world_ids.add(obstacle_id)
        for collision_object in request.scene.world.collision_objects:
            if collision_object.operation == _FakeCollisionObject.REMOVE:
                self.world_ids.discard(str(collision_object.id))
        return object()
