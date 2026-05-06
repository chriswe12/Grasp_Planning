"""MuJoCo-only regrasp fallback planning helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    accepted_grasps,
    evaluate_saved_grasps_against_pickup_pose,
    mesh_area_weighted_triangle_centroid,
    rotmat_to_quat_xyzw,
    trimesh,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .fabrica_pipeline import (
    GroundRecheckResult,
    PlanningConfig,
    Stage1Result,
    _score_grasps_for_world_top_approach,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HullSupportFacet:
    normal_obj: tuple[float, float, float]
    area_m2: float
    vertex_indices: tuple[int, ...]
    vertices_obj: tuple[tuple[float, float, float], ...]
    com_obj: tuple[float, float, float]
    com_projection_obj: tuple[float, float, float]
    stability_margin_m: float
    yaw_deg: float


@dataclass(frozen=True)
class MujocoRegraspFallbackPlan:
    target_mesh_path: str
    mesh_scale: float
    source_frame_origin_obj_world: tuple[float, float, float]
    source_frame_orientation_xyzw_obj_world: tuple[float, float, float, float]
    initial_object_pose_world: ObjectWorldPose
    staging_object_pose_world: ObjectWorldPose
    support_facet: HullSupportFacet
    transfer_grasp: SavedGraspCandidate
    final_grasp: SavedGraspCandidate
    metadata: dict[str, object]
    transfer_grasp_candidates: tuple[SavedGraspCandidate, ...] = field(default_factory=tuple)
    final_grasp_candidates: tuple[SavedGraspCandidate, ...] = field(default_factory=tuple)
    placement_options: tuple["MujocoRegraspPlacementOption", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MujocoRegraspPlacementOption:
    staging_object_pose_world: ObjectWorldPose
    support_facet: HullSupportFacet
    transfer_grasp: SavedGraspCandidate
    final_grasp: SavedGraspCandidate
    transfer_grasp_candidates: tuple[SavedGraspCandidate, ...]
    final_grasp_candidates: tuple[SavedGraspCandidate, ...]
    metadata: dict[str, object] = field(default_factory=dict)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-10:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _rotation_aligning_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize(source)
    target = _normalize(target)
    dot = min(1.0, max(-1.0, float(np.dot(source, target))))
    if dot > 1.0 - 1.0e-10:
        return np.eye(3, dtype=float)
    if dot < -1.0 + 1.0e-10:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(axis, source))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _normalize(axis - float(np.dot(axis, source)) * source)
        return _axis_angle_to_rotmat(axis, math.pi)
    axis = np.cross(source, target)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _axis_angle_to_rotmat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = [float(v) for v in axis]
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def _yaw_rotmat(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _mesh_center_of_mass(mesh_local: object) -> np.ndarray:
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    faces = np.asarray(mesh_local.faces, dtype=np.int64)
    if trimesh is not None:
        try:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            if bool(getattr(mesh, "is_volume", False)):
                return np.asarray(mesh.center_mass, dtype=float)
        except Exception:
            pass
    return mesh_area_weighted_triangle_centroid(mesh_local)


@dataclass(frozen=True)
class _FacetGroup:
    normal: np.ndarray
    offset: float
    area: float
    vertex_indices: tuple[int, ...]


def _triangle_area(vertices: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])))


def _convex_hull_facets(mesh_local: object, *, coplanar_tolerance_m: float) -> list[_FacetGroup]:
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    if len(vertices) < 4:
        return []
    try:
        hull = ConvexHull(vertices)
    except QhullError:
        return []

    groups: list[dict[str, object]] = []
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        normal = _normalize(np.asarray(equation[:3], dtype=float))
        offset = float(equation[3])
        area = _triangle_area(vertices[np.asarray(simplex, dtype=np.int64)])
        if area <= 1.0e-12:
            continue
        matched: dict[str, object] | None = None
        for group in groups:
            group_normal = np.asarray(group["normal"], dtype=float)
            group_offset = float(group["offset"])
            if (
                float(np.dot(normal, group_normal)) >= 1.0 - 1.0e-6
                and abs(offset - group_offset) <= float(coplanar_tolerance_m)
            ):
                matched = group
                break
        if matched is None:
            groups.append(
                {
                    "normal": normal,
                    "offset": offset,
                    "area": area,
                    "indices": set(int(index) for index in simplex),
                }
            )
        else:
            matched["area"] = float(matched["area"]) + area
            matched["indices"].update(int(index) for index in simplex)  # type: ignore[union-attr]

    return [
        _FacetGroup(
            normal=np.asarray(group["normal"], dtype=float),
            offset=float(group["offset"]),
            area=float(group["area"]),
            vertex_indices=tuple(sorted(int(index) for index in group["indices"])),
        )
        for group in groups
    ]


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalize(normal)
    axis = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, normal))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
    basis_u = _normalize(axis - float(np.dot(axis, normal)) * normal)
    basis_v = _normalize(np.cross(normal, basis_u))
    return basis_u, basis_v


def _ordered_polygon_2d(points_2d: np.ndarray) -> np.ndarray:
    if len(points_2d) < 3:
        return points_2d
    hull = ConvexHull(points_2d)
    return points_2d[np.asarray(hull.vertices, dtype=np.int64)]


def _convex_polygon_margin(point_2d: np.ndarray, polygon_2d: np.ndarray) -> float:
    if len(polygon_2d) < 3:
        return float("-inf")
    signed_distances: list[float] = []
    for index, start in enumerate(polygon_2d):
        end = polygon_2d[(index + 1) % len(polygon_2d)]
        edge = end - start
        edge_len = float(np.linalg.norm(edge))
        if edge_len <= 1.0e-12:
            continue
        signed_distances.append(float(np.cross(edge, point_2d - start) / edge_len))
    if not signed_distances:
        return float("-inf")
    return min(signed_distances)


def _support_facet_stability(
    *,
    mesh_local: object,
    facet: _FacetGroup,
    yaw_deg: float,
) -> HullSupportFacet:
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    facet_vertices = vertices[np.asarray(facet.vertex_indices, dtype=np.int64)]
    com_obj = _mesh_center_of_mass(mesh_local)
    signed_distance = float(np.dot(facet.normal, com_obj) + facet.offset)
    com_projection_obj = com_obj - signed_distance * facet.normal
    basis_u, basis_v = _plane_basis(facet.normal)
    origin = facet_vertices[0]
    facet_points_2d = np.column_stack(
        ((facet_vertices - origin[None, :]) @ basis_u, (facet_vertices - origin[None, :]) @ basis_v)
    )
    polygon_2d = _ordered_polygon_2d(facet_points_2d)
    com_2d = np.asarray(
        [float(np.dot(com_projection_obj - origin, basis_u)), float(np.dot(com_projection_obj - origin, basis_v))],
        dtype=float,
    )
    margin = _convex_polygon_margin(com_2d, polygon_2d)
    return HullSupportFacet(
        normal_obj=tuple(float(v) for v in facet.normal),
        area_m2=float(facet.area),
        vertex_indices=facet.vertex_indices,
        vertices_obj=tuple(tuple(float(v) for v in vertex) for vertex in facet_vertices),
        com_obj=tuple(float(v) for v in com_obj),
        com_projection_obj=tuple(float(v) for v in com_projection_obj),
        stability_margin_m=float(margin),
        yaw_deg=float(yaw_deg),
    )


def _pose_for_support_facet(
    *,
    mesh_local: object,
    normal_obj: tuple[float, float, float],
    xy_world: tuple[float, float],
    yaw_deg: float,
) -> ObjectWorldPose:
    align = _rotation_aligning_vectors(np.asarray(normal_obj, dtype=float), np.array([0.0, 0.0, -1.0], dtype=float))
    rotation_world_from_obj = _yaw_rotmat(yaw_deg) @ align
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    rotated_vertices = vertices @ rotation_world_from_obj.T
    min_z = float(rotated_vertices[:, 2].min())
    translation = (float(xy_world[0]), float(xy_world[1]), float(-min_z))
    return ObjectWorldPose(
        position_world=translation,
        orientation_xyzw_world=rotmat_to_quat_xyzw(rotation_world_from_obj),
    )


def _staging_xy_candidates(
    *,
    base_xy_world: tuple[float, float],
    offsets_m: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    offsets = offsets_m or ((0.0, 0.0),)
    seen: set[tuple[float, float]] = set()
    candidates: list[tuple[float, float]] = []
    for dx_m, dy_m in offsets:
        xy = (round(float(base_xy_world[0]) + float(dx_m), 6), round(float(base_xy_world[1]) + float(dy_m), 6))
        if xy in seen:
            continue
        seen.add(xy)
        candidates.append(xy)
    return tuple(candidates)


def _placement_option_score(
    *,
    initial_xy_world: tuple[float, float],
    staging_xy_world: tuple[float, float],
    transfer_grasp: SavedGraspCandidate,
    final_grasp: SavedGraspCandidate,
    stability_margin_m: float,
) -> float:
    transfer_score = 0.0 if transfer_grasp.score is None else float(transfer_grasp.score)
    final_score = 0.0 if final_grasp.score is None else float(final_grasp.score)
    initial_xy = np.asarray(initial_xy_world, dtype=float)
    staging_xy = np.asarray(staging_xy_world, dtype=float)
    xy_distance_m = float(np.linalg.norm(staging_xy - initial_xy))
    workspace_radius_m = float(np.linalg.norm(staging_xy))
    comfortable_radius_penalty = abs(workspace_radius_m - 0.5)
    lateral_penalty = abs(float(staging_xy[1]))
    return (
        transfer_score
        + final_score
        + 0.5 * max(0.0, float(stability_margin_m))
        - 0.45 * xy_distance_m
        - 0.25 * comfortable_radius_penalty
        - 0.1 * lateral_penalty
    )


def _accepted_by_grasp_and_contact_offset(
    statuses: Iterable[object],
) -> dict[tuple[str, float, float], SavedGraspCandidate]:
    accepted: dict[tuple[str, float, float], SavedGraspCandidate] = {}
    for entry in statuses:
        if getattr(entry, "status") != "accepted":
            continue
        grasp = entry.grasp
        key = (
            str(grasp.grasp_id),
            round(float(grasp.contact_patch_lateral_offset_m), 9),
            round(float(grasp.contact_patch_approach_offset_m), 9),
        )
        accepted[key] = grasp
    return accepted


def _current_floor_support_facets(
    *,
    mesh_local: object,
    facets: Iterable[_FacetGroup],
    initial_pose: ObjectWorldPose,
    tolerance_m: float,
) -> set[tuple[int, ...]]:
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    vertices_world = initial_pose.transform_points_to_world(vertices)
    min_z = float(vertices_world[:, 2].min())
    tolerance = max(float(tolerance_m), 1.0e-6)
    current_support: set[tuple[int, ...]] = set()
    for facet in facets:
        if not facet.vertex_indices:
            continue
        facet_z = vertices_world[np.asarray(facet.vertex_indices, dtype=np.int64), 2]
        if float(np.max(np.abs(facet_z - min_z))) <= tolerance:
            current_support.add(tuple(facet.vertex_indices))
    return current_support


def _pose_payload(pose: ObjectWorldPose) -> dict[str, object]:
    return {
        "position_world": list(pose.position_world),
        "orientation_xyzw_world": list(pose.orientation_xyzw_world),
    }


def _pose_from_payload(payload: dict[str, object]) -> ObjectWorldPose:
    position = payload["position_world"]
    orientation = payload["orientation_xyzw_world"]
    if not isinstance(position, (list, tuple)) or not isinstance(orientation, (list, tuple)):
        raise ValueError("Pose payload must contain position_world and orientation_xyzw_world lists.")
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in position),
        orientation_xyzw_world=tuple(float(value) for value in orientation),
    )


def _candidate_payload(candidate: SavedGraspCandidate) -> dict[str, object]:
    return {
        "grasp_id": candidate.grasp_id,
        "grasp_pose_obj": {
            "position": list(candidate.grasp_position_obj),
            "orientation_xyzw": list(candidate.grasp_orientation_xyzw_obj),
        },
        "contact_points_obj": [list(candidate.contact_point_a_obj), list(candidate.contact_point_b_obj)],
        "contact_normals_obj": [list(candidate.contact_normal_a_obj), list(candidate.contact_normal_b_obj)],
        "jaw_width": candidate.jaw_width,
        "roll_angle_rad": candidate.roll_angle_rad,
        "contact_patch_offset_local": [
            candidate.contact_patch_lateral_offset_m,
            candidate.contact_patch_approach_offset_m,
        ],
        "score": candidate.score,
        "score_components": candidate.score_components,
    }


def _candidate_from_payload(payload: dict[str, object]) -> SavedGraspCandidate:
    contact_patch_offset = payload.get("contact_patch_offset_local", [0.0, 0.0])
    score_components = payload.get("score_components")
    return SavedGraspCandidate(
        grasp_id=str(payload["grasp_id"]),
        grasp_position_obj=tuple(float(v) for v in payload["grasp_pose_obj"]["position"]),  # type: ignore[index]
        grasp_orientation_xyzw_obj=tuple(
            float(v) for v in payload["grasp_pose_obj"]["orientation_xyzw"]  # type: ignore[index]
        ),
        contact_point_a_obj=tuple(float(v) for v in payload["contact_points_obj"][0]),  # type: ignore[index]
        contact_point_b_obj=tuple(float(v) for v in payload["contact_points_obj"][1]),  # type: ignore[index]
        contact_normal_a_obj=tuple(float(v) for v in payload["contact_normals_obj"][0]),  # type: ignore[index]
        contact_normal_b_obj=tuple(float(v) for v in payload["contact_normals_obj"][1]),  # type: ignore[index]
        jaw_width=float(payload["jaw_width"]),
        roll_angle_rad=float(payload["roll_angle_rad"]),
        contact_patch_lateral_offset_m=float(contact_patch_offset[0]),  # type: ignore[index]
        contact_patch_approach_offset_m=float(contact_patch_offset[1]),  # type: ignore[index]
        score=None if payload.get("score") is None else float(payload["score"]),
        score_components=None
        if score_components is None
        else {str(key): float(value) for key, value in dict(score_components).items()},
    )


def _facet_payload(facet: HullSupportFacet) -> dict[str, object]:
    return {
        "normal_obj": list(facet.normal_obj),
        "area_m2": facet.area_m2,
        "vertex_indices": list(facet.vertex_indices),
        "vertices_obj": [list(vertex) for vertex in facet.vertices_obj],
        "com_obj": list(facet.com_obj),
        "com_projection_obj": list(facet.com_projection_obj),
        "stability_margin_m": facet.stability_margin_m,
        "yaw_deg": facet.yaw_deg,
    }


def _facet_from_payload(payload: dict[str, object]) -> HullSupportFacet:
    return HullSupportFacet(
        normal_obj=tuple(float(v) for v in payload["normal_obj"]),  # type: ignore[arg-type]
        area_m2=float(payload["area_m2"]),
        vertex_indices=tuple(int(v) for v in payload["vertex_indices"]),  # type: ignore[arg-type]
        vertices_obj=tuple(tuple(float(v) for v in vertex) for vertex in payload["vertices_obj"]),  # type: ignore[arg-type]
        com_obj=tuple(float(v) for v in payload["com_obj"]),  # type: ignore[arg-type]
        com_projection_obj=tuple(float(v) for v in payload["com_projection_obj"]),  # type: ignore[arg-type]
        stability_margin_m=float(payload["stability_margin_m"]),
        yaw_deg=float(payload["yaw_deg"]),
    )


def _placement_option_payload(option: MujocoRegraspPlacementOption) -> dict[str, object]:
    return {
        "staging_object_pose_world": _pose_payload(option.staging_object_pose_world),
        "support_facet": _facet_payload(option.support_facet),
        "transfer_grasp": _candidate_payload(option.transfer_grasp),
        "final_grasp": _candidate_payload(option.final_grasp),
        "transfer_grasp_candidates": [_candidate_payload(candidate) for candidate in option.transfer_grasp_candidates],
        "final_grasp_candidates": [_candidate_payload(candidate) for candidate in option.final_grasp_candidates],
        "metadata": option.metadata,
    }


def _placement_option_from_payload(payload: dict[str, object]) -> MujocoRegraspPlacementOption:
    transfer_grasp = _candidate_from_payload(dict(payload["transfer_grasp"]))
    final_grasp = _candidate_from_payload(dict(payload["final_grasp"]))
    transfer_candidates_raw = payload.get("transfer_grasp_candidates")
    final_candidates_raw = payload.get("final_grasp_candidates")
    return MujocoRegraspPlacementOption(
        staging_object_pose_world=_pose_from_payload(payload["staging_object_pose_world"]),  # type: ignore[arg-type]
        support_facet=_facet_from_payload(payload["support_facet"]),  # type: ignore[arg-type]
        transfer_grasp=transfer_grasp,
        final_grasp=final_grasp,
        transfer_grasp_candidates=(
            tuple(_candidate_from_payload(dict(candidate)) for candidate in transfer_candidates_raw)
            if isinstance(transfer_candidates_raw, list)
            else (transfer_grasp,)
        ),
        final_grasp_candidates=(
            tuple(_candidate_from_payload(dict(candidate)) for candidate in final_candidates_raw)
            if isinstance(final_candidates_raw, list)
            else (final_grasp,)
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def plan_mujoco_regrasp_fallback(
    *,
    stage1: Stage1Result,
    direct_stage2: GroundRecheckResult,
    planning: PlanningConfig,
    force: bool = False,
    staging_xy_world: tuple[float, float] | None = None,
    staging_xy_offsets_m: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
    yaw_angles_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
    max_orientations: int = 24,
    max_placement_options: int = 6,
    min_facet_area_m2: float = 0.0,
    stability_margin_m: float = 0.0,
    coplanar_tolerance_m: float = 1.0e-6,
) -> MujocoRegraspFallbackPlan | None:
    """Find a one-placement MuJoCo regrasp plan after direct stage 2 fails."""

    if direct_stage2.accepted and not force:
        return None
    if not stage1.bundle.candidates:
        return None
    raw_candidates = tuple(stage1.raw_candidates) if stage1.raw_candidates else tuple(stage1.bundle.candidates)
    if not raw_candidates:
        return None
    yaw_angles = tuple(float(value) for value in yaw_angles_deg)
    if not yaw_angles:
        return None
    initial_pose = direct_stage2.pickup_pose_world
    staging_xy = (
        tuple(float(v) for v in staging_xy_world)
        if staging_xy_world is not None
        else (float(initial_pose.position_world[0]), float(initial_pose.position_world[1]))
    )
    staging_xy_candidates = _staging_xy_candidates(
        base_xy_world=staging_xy,
        offsets_m=staging_xy_offsets_m,
    )
    facets = sorted(
        (
            facet
            for facet in _convex_hull_facets(stage1.target_mesh_local, coplanar_tolerance_m=coplanar_tolerance_m)
            if float(facet.area) >= float(min_facet_area_m2)
        ),
        key=lambda facet: facet.area,
        reverse=True,
    )
    initial_support_facets = _current_floor_support_facets(
        mesh_local=stage1.target_mesh_local,
        facets=facets,
        initial_pose=initial_pose,
        tolerance_m=coplanar_tolerance_m,
    )
    transfer_initial_statuses = evaluate_saved_grasps_against_pickup_pose(
        raw_candidates,
        object_pose_world=initial_pose,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        floor_clearance_margin_m=planning.floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )
    initial_by_key = _accepted_by_grasp_and_contact_offset(transfer_initial_statuses)
    if not initial_by_key:
        return None

    checked_orientations = 0
    checked_placements = 0
    stable_orientations = 0
    skipped_initial_support_orientations = 0
    orientation_limit_reached = False
    max_options = max(1, int(max_placement_options))
    placement_options: list[MujocoRegraspPlacementOption] = []
    for facet in facets:
        if tuple(facet.vertex_indices) in initial_support_facets:
            skipped_initial_support_orientations += len(yaw_angles)
            continue
        for yaw_deg in yaw_angles:
            if max_orientations > 0 and checked_orientations >= max_orientations:
                orientation_limit_reached = True
                break
            checked_orientations += 1
            support = _support_facet_stability(
                mesh_local=stage1.target_mesh_local,
                facet=facet,
                yaw_deg=float(yaw_deg),
            )
            if support.stability_margin_m < float(stability_margin_m):
                continue
            stable_orientations += 1
            filter_pose = _pose_for_support_facet(
                mesh_local=stage1.target_mesh_local,
                normal_obj=support.normal_obj,
                xy_world=staging_xy,
                yaw_deg=float(yaw_deg),
            )
            final_statuses = evaluate_saved_grasps_against_pickup_pose(
                stage1.bundle.candidates,
                object_pose_world=filter_pose,
                contact_gap_m=planning.detailed_finger_contact_gap_m,
                floor_clearance_margin_m=planning.floor_clearance_margin_m,
                contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
                contact_approach_offsets_m=planning.contact_approach_offsets_m,
            )
            final_candidates = _score_grasps_for_world_top_approach(
                accepted_grasps(final_statuses),
                mesh_local=stage1.target_mesh_local,
                object_pose_world=filter_pose,
                top_grasp_score_weight=planning.top_grasp_score_weight,
            )
            if not final_candidates:
                continue

            transfer_staging_statuses = evaluate_saved_grasps_against_pickup_pose(
                raw_candidates,
                object_pose_world=filter_pose,
                contact_gap_m=planning.detailed_finger_contact_gap_m,
                floor_clearance_margin_m=planning.floor_clearance_margin_m,
                contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
                contact_approach_offsets_m=planning.contact_approach_offsets_m,
            )
            staging_by_key = _accepted_by_grasp_and_contact_offset(transfer_staging_statuses)
            common_keys = sorted(set(initial_by_key).intersection(staging_by_key))
            if not common_keys:
                continue
            transfer_candidates = _score_grasps_for_world_top_approach(
                [initial_by_key[key] for key in common_keys],
                mesh_local=stage1.target_mesh_local,
                object_pose_world=initial_pose,
                top_grasp_score_weight=planning.regrasp_transfer_top_grasp_score_weight,
            )
            if not transfer_candidates:
                continue
            for staging_xy_candidate in staging_xy_candidates:
                checked_placements += 1
                staging_pose = _pose_for_support_facet(
                    mesh_local=stage1.target_mesh_local,
                    normal_obj=support.normal_obj,
                    xy_world=staging_xy_candidate,
                    yaw_deg=float(yaw_deg),
                )
                placement_score = _placement_option_score(
                    initial_xy_world=(float(initial_pose.position_world[0]), float(initial_pose.position_world[1])),
                    staging_xy_world=staging_xy_candidate,
                    transfer_grasp=transfer_candidates[0],
                    final_grasp=final_candidates[0],
                    stability_margin_m=support.stability_margin_m,
                )
                placement_options.append(
                    MujocoRegraspPlacementOption(
                        staging_object_pose_world=staging_pose,
                        support_facet=support,
                        transfer_grasp=transfer_candidates[0],
                        final_grasp=final_candidates[0],
                        transfer_grasp_candidates=tuple(transfer_candidates),
                        final_grasp_candidates=tuple(final_candidates),
                        metadata={
                            "staging_xy_world": list(staging_xy_candidate),
                            "placement_score": float(placement_score),
                            "yaw_deg": float(yaw_deg),
                            "support_area_m2": float(support.area_m2),
                            "stability_margin_m": float(support.stability_margin_m),
                            "final_feasible_count_for_staging_pose": len(final_candidates),
                            "transfer_feasible_count_for_staging_pose": len(common_keys),
                        },
                    )
                )
                if len(placement_options) >= max_options:
                    orientation_limit_reached = True
                    break
            if orientation_limit_reached:
                break
        if orientation_limit_reached:
            break
    if not placement_options:
        return None
    placement_options = sorted(
        placement_options,
        key=lambda option: float(option.metadata.get("placement_score", 0.0)),
        reverse=True,
    )
    placement_options = placement_options[:max_options]
    selected = placement_options[0]
    return MujocoRegraspFallbackPlan(
        target_mesh_path=stage1.bundle.target_mesh_path,
        mesh_scale=stage1.bundle.mesh_scale,
        source_frame_origin_obj_world=stage1.bundle.source_frame_origin_obj_world,
        source_frame_orientation_xyzw_obj_world=stage1.bundle.source_frame_orientation_xyzw_obj_world,
        initial_object_pose_world=initial_pose,
        staging_object_pose_world=selected.staging_object_pose_world,
        support_facet=selected.support_facet,
        transfer_grasp=selected.transfer_grasp,
        final_grasp=selected.final_grasp,
        metadata={
            "reason": "forced" if force else "stage2_rejected_all",
            "direct_stage2_feasible_count": len(direct_stage2.accepted),
            "stage1_assembly_feasible_count": len(stage1.bundle.candidates),
            "raw_transfer_candidate_count": len(raw_candidates),
            "staging_xy_candidate_count": len(staging_xy_candidates),
            "checked_orientations": checked_orientations,
            "checked_staging_placements": checked_placements,
            "stable_orientations_before_selection": stable_orientations,
            "skipped_initial_support_orientations": skipped_initial_support_orientations,
            "placement_option_count": len(placement_options),
            "selected_staging_xy_world": list(selected.metadata.get("staging_xy_world", [])),
            "selected_placement_score": selected.metadata.get("placement_score", 0.0),
            "final_feasible_count_for_staging_pose": len(selected.final_grasp_candidates),
            "transfer_feasible_count_for_staging_pose": len(selected.transfer_grasp_candidates),
            "transfer_top_grasp_score_weight": planning.regrasp_transfer_top_grasp_score_weight,
            "final_top_grasp_score_weight": planning.top_grasp_score_weight,
        },
        transfer_grasp_candidates=selected.transfer_grasp_candidates,
        final_grasp_candidates=selected.final_grasp_candidates,
        placement_options=tuple(placement_options),
    )
    return None


def write_mujoco_regrasp_plan(
    plan: MujocoRegraspFallbackPlan,
    output_path: str | Path,
    *,
    input_stage2_json: str | Path,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "input_stage2_json": str(input_stage2_json),
        "target": {
            "mesh_path": plan.target_mesh_path,
            "mesh_scale": plan.mesh_scale,
            "source_frame_origin_obj_world": list(plan.source_frame_origin_obj_world),
            "source_frame_orientation_xyzw_obj_world": list(plan.source_frame_orientation_xyzw_obj_world),
        },
        "initial_object_pose_world": _pose_payload(plan.initial_object_pose_world),
        "staging_object_pose_world": _pose_payload(plan.staging_object_pose_world),
        "support_facet": _facet_payload(plan.support_facet),
        "transfer_grasp": _candidate_payload(plan.transfer_grasp),
        "final_grasp": _candidate_payload(plan.final_grasp),
        "transfer_grasp_candidates": [
            _candidate_payload(candidate)
            for candidate in (plan.transfer_grasp_candidates or (plan.transfer_grasp,))
        ],
        "final_grasp_candidates": [
            _candidate_payload(candidate)
            for candidate in (plan.final_grasp_candidates or (plan.final_grasp,))
        ],
        "placement_options": [
            _placement_option_payload(option)
            for option in (
                plan.placement_options
                or (
                    MujocoRegraspPlacementOption(
                        staging_object_pose_world=plan.staging_object_pose_world,
                        support_facet=plan.support_facet,
                        transfer_grasp=plan.transfer_grasp,
                        final_grasp=plan.final_grasp,
                        transfer_grasp_candidates=plan.transfer_grasp_candidates or (plan.transfer_grasp,),
                        final_grasp_candidates=plan.final_grasp_candidates or (plan.final_grasp,),
                        metadata={},
                    ),
                )
            )
        ],
        "metadata": plan.metadata,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_mujoco_regrasp_plan(path: str | Path) -> MujocoRegraspFallbackPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported MuJoCo regrasp plan schema_version={payload.get('schema_version')!r}.")
    target = payload["target"]
    transfer_grasp = _candidate_from_payload(payload["transfer_grasp"])
    final_grasp = _candidate_from_payload(payload["final_grasp"])
    transfer_candidates_raw = payload.get("transfer_grasp_candidates")
    final_candidates_raw = payload.get("final_grasp_candidates")
    transfer_grasp_candidates = (
        tuple(_candidate_from_payload(dict(candidate)) for candidate in transfer_candidates_raw)
        if isinstance(transfer_candidates_raw, list)
        else (transfer_grasp,)
    )
    final_grasp_candidates = (
        tuple(_candidate_from_payload(dict(candidate)) for candidate in final_candidates_raw)
        if isinstance(final_candidates_raw, list)
        else (final_grasp,)
    )
    placement_options_raw = payload.get("placement_options")
    placement_options = (
        tuple(_placement_option_from_payload(dict(option)) for option in placement_options_raw)
        if isinstance(placement_options_raw, list)
        else (
            MujocoRegraspPlacementOption(
                staging_object_pose_world=_pose_from_payload(payload["staging_object_pose_world"]),
                support_facet=_facet_from_payload(payload["support_facet"]),
                transfer_grasp=transfer_grasp,
                final_grasp=final_grasp,
                transfer_grasp_candidates=transfer_grasp_candidates,
                final_grasp_candidates=final_grasp_candidates,
                metadata={},
            ),
        )
    )
    return MujocoRegraspFallbackPlan(
        target_mesh_path=str(target["mesh_path"]),
        mesh_scale=float(target["mesh_scale"]),
        source_frame_origin_obj_world=tuple(float(v) for v in target["source_frame_origin_obj_world"]),
        source_frame_orientation_xyzw_obj_world=tuple(
            float(v) for v in target["source_frame_orientation_xyzw_obj_world"]
        ),
        initial_object_pose_world=_pose_from_payload(payload["initial_object_pose_world"]),
        staging_object_pose_world=_pose_from_payload(payload["staging_object_pose_world"]),
        support_facet=_facet_from_payload(payload["support_facet"]),
        transfer_grasp=transfer_grasp,
        final_grasp=final_grasp,
        metadata=dict(payload.get("metadata", {})),
        transfer_grasp_candidates=transfer_grasp_candidates,
        final_grasp_candidates=final_grasp_candidates,
        placement_options=placement_options,
    )
