"""Transform saved object-frame grasps into world-frame execution grasps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .fabrica_grasp_debug import SavedGraspCandidate, quat_to_rotmat_xyzw, rotmat_to_quat_xyzw
from .world_constraints import ObjectWorldPose

GraspFrameConvention = Literal["legacy_execution", "mesh_grasp"]


@dataclass(frozen=True)
class WorldFrameGraspCandidate:
    """Execution-ready grasp expressed in the world frame."""

    grasp_id: str
    position_w: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    normal_w: tuple[float, float, float]
    pregrasp_offset: float
    pregrasp_position_w: tuple[float, float, float]
    gripper_width: float
    jaw_width: float
    roll_angle_rad: float
    contact_point_a_w: tuple[float, float, float]
    contact_point_b_w: tuple[float, float, float]


def transform_point_obj_to_world(
    point_obj: tuple[float, float, float],
    object_pose_world: ObjectWorldPose,
) -> tuple[float, float, float]:
    point_world = object_pose_world.rotation_world_from_object @ np.asarray(point_obj, dtype=float)
    point_world = point_world + object_pose_world.translation_world
    return tuple(float(v) for v in point_world)


def transform_rotation_obj_to_world(
    orientation_xyzw_obj: tuple[float, float, float, float],
    object_pose_world: ObjectWorldPose,
) -> tuple[float, float, float, float]:
    rot_world = object_pose_world.rotation_world_from_object @ quat_to_rotmat_xyzw(orientation_xyzw_obj)
    return rotmat_to_quat_xyzw(rot_world)


def grasp_approach_axis_world(
    grasp_orientation_xyzw_world: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    rotmat = quat_to_rotmat_xyzw(grasp_orientation_xyzw_world)
    return tuple(float(v) for v in rotmat[:, 0].tolist())


def _mesh_grasp_rotmat_to_execution_rotmat(mesh_grasp_rotmat: np.ndarray) -> np.ndarray:
    """Convert mesh-grasp axes into the existing execution grasp convention.

    Mesh antipodal convention:
    - local x: lateral
    - local y: closing
    - local z: approach

    Execution convention used by the cube path:
    - local x: approach
    - local y: closing
    - local z: orthogonal axis completing the frame
    """

    lateral_axis = mesh_grasp_rotmat[:, 0]
    closing_axis = mesh_grasp_rotmat[:, 1]
    approach_axis = mesh_grasp_rotmat[:, 2]
    execution_rotmat = np.column_stack((approach_axis, closing_axis, -lateral_axis))
    return execution_rotmat


def _mesh_grasp_rotmat_to_target_rotmat(
    mesh_grasp_rotmat: np.ndarray,
    *,
    frame_convention: GraspFrameConvention,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the saved mesh-grasp frame into a target EE frame.

    `legacy_execution` preserves the historical Isaac/controller convention:
    - local x: approach
    - local y: closing
    - local z: -lateral

    `mesh_grasp` keeps the original saved grasp frame:
    - local x: lateral
    - local y: closing
    - local z: approach

    MuJoCo's Menagerie `gripper` site already follows the latter convention, so
    applying the Isaac remap there rotates the target frame incorrectly and
    pushes the pregrasp along the wrong axis.
    """

    if frame_convention == "legacy_execution":
        target_rotmat = _mesh_grasp_rotmat_to_execution_rotmat(mesh_grasp_rotmat)
        approach_axis_world = target_rotmat[:, 0]
        return target_rotmat, approach_axis_world
    if frame_convention == "mesh_grasp":
        target_rotmat = np.asarray(mesh_grasp_rotmat, dtype=float)
        approach_axis_world = target_rotmat[:, 2]
        return target_rotmat, approach_axis_world
    raise ValueError(f"Unsupported grasp frame convention '{frame_convention}'.")


def saved_grasp_to_world_grasp(
    grasp: SavedGraspCandidate,
    object_pose_world: ObjectWorldPose,
    *,
    pregrasp_offset: float,
    gripper_width_clearance: float,
    frame_convention: GraspFrameConvention = "legacy_execution",
) -> WorldFrameGraspCandidate:
    mesh_grasp_rotmat_obj = quat_to_rotmat_xyzw(grasp.grasp_orientation_xyzw_obj)
    mesh_grasp_rotmat_world = object_pose_world.rotation_world_from_object @ mesh_grasp_rotmat_obj
    target_rotmat_world, approach_axis_world = _mesh_grasp_rotmat_to_target_rotmat(
        mesh_grasp_rotmat_world,
        frame_convention=frame_convention,
    )
    orientation_xyzw_world = rotmat_to_quat_xyzw(target_rotmat_world)
    position_w = np.asarray(transform_point_obj_to_world(grasp.grasp_position_obj, object_pose_world), dtype=float)
    pregrasp_position_w = position_w - approach_axis_world * float(pregrasp_offset)
    return WorldFrameGraspCandidate(
        grasp_id=grasp.grasp_id,
        position_w=tuple(float(v) for v in position_w),
        orientation_xyzw=orientation_xyzw_world,
        normal_w=tuple(float(v) for v in approach_axis_world),
        pregrasp_offset=float(pregrasp_offset),
        pregrasp_position_w=tuple(float(v) for v in pregrasp_position_w),
        gripper_width=float(grasp.jaw_width + gripper_width_clearance),
        jaw_width=float(grasp.jaw_width),
        roll_angle_rad=float(grasp.roll_angle_rad),
        contact_point_a_w=transform_point_obj_to_world(grasp.contact_point_a_obj, object_pose_world),
        contact_point_b_w=transform_point_obj_to_world(grasp.contact_point_b_obj, object_pose_world),
    )
