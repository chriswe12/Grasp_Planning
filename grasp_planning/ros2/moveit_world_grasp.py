"""Shared pose-target helpers for world-frame grasp execution."""

from __future__ import annotations

from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate

from .moveit_pose_commander import PoseTarget


def pose_target_from_world(
    *,
    position_xyz: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    frame_id: str,
) -> PoseTarget:
    return PoseTarget.from_quaternion(
        x=position_xyz[0],
        y=position_xyz[1],
        z=position_xyz[2],
        quaternion_xyzw=orientation_xyzw,
        frame_id=frame_id,
    )


def world_grasp_pose_targets(
    world_grasp: WorldFrameGraspCandidate, *, frame_id: str, lift_height_m: float
) -> dict[str, PoseTarget]:
    orientation_xyzw = tuple(float(v) for v in world_grasp.orientation_xyzw)
    return {
        "pregrasp": pose_target_from_world(
            position_xyz=tuple(float(v) for v in world_grasp.pregrasp_position_w),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
        "grasp": pose_target_from_world(
            position_xyz=tuple(float(v) for v in world_grasp.position_w),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
        "lift": pose_target_from_world(
            position_xyz=(
                float(world_grasp.position_w[0]),
                float(world_grasp.position_w[1]),
                float(world_grasp.position_w[2] + float(lift_height_m)),
            ),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
    }


__all__ = [
    "pose_target_from_world",
    "world_grasp_pose_targets",
]
