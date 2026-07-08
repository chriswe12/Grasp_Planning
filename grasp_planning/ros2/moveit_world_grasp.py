"""Shared pose-target helpers for world-frame grasp execution."""

from __future__ import annotations

from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate

from .moveit_pose_commander import PoseTarget


def _quat_apply_xyzw(
    quat_xyzw: tuple[float, float, float, float], vector_xyz: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z, w = (float(value) for value in quat_xyzw)
    vx, vy, vz = (float(value) for value in vector_xyz)
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    uv = (
        y * vz - z * vy,
        z * vx - x * vz,
        x * vy - y * vx,
    )
    uuv = (
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    )
    return (
        vx + 2.0 * (w * uv[0] + uuv[0]),
        vy + 2.0 * (w * uv[1] + uuv[1]),
        vz + 2.0 * (w * uv[2] + uuv[2]),
    )


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


def _signed_position(
    position_xyz: tuple[float, float, float], position_signs: tuple[float, float, float]
) -> tuple[float, float, float]:
    if len(position_signs) != 3:
        raise ValueError(f"Expected 3 target position signs, got {len(position_signs)}.")
    return tuple(float(position_xyz[index]) * float(position_signs[index]) for index in range(3))


def _tcp_position_from_grasp_center(
    *,
    grasp_position_xyz: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    tcp_to_grasp_offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    if len(tcp_to_grasp_offset) != 3:
        raise ValueError(f"Expected 3 TCP-to-grasp offset values, got {len(tcp_to_grasp_offset)}.")
    offset = tuple(float(value) for value in tcp_to_grasp_offset)
    if offset == (0.0, 0.0, 0.0):
        return grasp_position_xyz
    rotated_offset = _quat_apply_xyzw(orientation_xyzw, offset)
    return tuple(float(grasp_position_xyz[index]) - rotated_offset[index] for index in range(3))


def world_grasp_pose_targets(
    world_grasp: WorldFrameGraspCandidate,
    *,
    frame_id: str,
    lift_height_m: float,
    position_signs: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tcp_to_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, PoseTarget]:
    orientation_xyzw = tuple(float(v) for v in world_grasp.orientation_xyzw)
    grasp_position = tuple(float(v) for v in world_grasp.position_w)
    pregrasp_tcp_position = _tcp_position_from_grasp_center(
        grasp_position_xyz=tuple(float(v) for v in world_grasp.pregrasp_position_w),
        orientation_xyzw=orientation_xyzw,
        tcp_to_grasp_offset=tcp_to_grasp_offset,
    )
    grasp_tcp_position = _tcp_position_from_grasp_center(
        grasp_position_xyz=grasp_position,
        orientation_xyzw=orientation_xyzw,
        tcp_to_grasp_offset=tcp_to_grasp_offset,
    )
    lift_tcp_position = _tcp_position_from_grasp_center(
        grasp_position_xyz=(grasp_position[0], grasp_position[1], grasp_position[2] + float(lift_height_m)),
        orientation_xyzw=orientation_xyzw,
        tcp_to_grasp_offset=tcp_to_grasp_offset,
    )
    return {
        "pregrasp": pose_target_from_world(
            position_xyz=_signed_position(pregrasp_tcp_position, position_signs),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
        "grasp": pose_target_from_world(
            position_xyz=_signed_position(grasp_tcp_position, position_signs),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
        "lift": pose_target_from_world(
            position_xyz=_signed_position(lift_tcp_position, position_signs),
            orientation_xyzw=orientation_xyzw,
            frame_id=frame_id,
        ),
    }


__all__ = [
    "pose_target_from_world",
    "world_grasp_pose_targets",
]
