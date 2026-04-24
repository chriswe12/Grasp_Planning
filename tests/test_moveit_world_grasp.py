from __future__ import annotations

from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.ros2.moveit_world_grasp import pose_target_from_world, world_grasp_pose_targets


def _world_grasp() -> WorldFrameGraspCandidate:
    return WorldFrameGraspCandidate(
        grasp_id="g0001",
        position_w=(0.4, 0.1, 0.2),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        normal_w=(0.0, 0.0, 1.0),
        pregrasp_offset=0.1,
        pregrasp_position_w=(0.4, 0.1, 0.1),
        gripper_width=0.03,
        jaw_width=0.02,
        roll_angle_rad=0.0,
        contact_point_a_w=(0.39, 0.1, 0.2),
        contact_point_b_w=(0.41, 0.1, 0.2),
    )


def test_world_grasp_pose_targets_builds_pregrasp_grasp_and_lift() -> None:
    targets = world_grasp_pose_targets(_world_grasp(), frame_id="base", lift_height_m=0.08)

    assert tuple(targets.keys()) == ("pregrasp", "grasp", "lift")
    assert targets["pregrasp"].frame_id == "base"
    assert targets["pregrasp"].position_xyz == (0.4, 0.1, 0.1)
    assert targets["grasp"].position_xyz == (0.4, 0.1, 0.2)
    assert targets["lift"].position_xyz == (0.4, 0.1, 0.28)


def test_pose_target_from_world_preserves_pose_and_frame() -> None:
    target = pose_target_from_world(
        position_xyz=(0.2, -0.1, 0.3),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id="map",
    )

    assert target.position_xyz == (0.2, -0.1, 0.3)
    assert target.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert target.frame_id == "map"
