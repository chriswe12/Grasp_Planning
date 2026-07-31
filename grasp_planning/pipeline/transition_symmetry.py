"""Finite grasp and partial-assembly symmetries for dual-arm transitions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
)
from grasp_planning.grasping.mesh_io import load_triangle_mesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .assembly_sequence import REPO_ROOT, AssemblySequence, AssemblySequenceStep

SCHEMA_VERSION = 1
_MATRIX_ATOL = 1.0e-9


def _identity_matrix() -> np.ndarray:
    return np.eye(4, dtype=float)


def _pose_matrix(pose: ObjectWorldPose) -> np.ndarray:
    matrix = _identity_matrix()
    matrix[:3, :3] = pose.rotation_world_from_object
    matrix[:3, 3] = pose.translation_world
    return matrix


def _matrix_pose(matrix: np.ndarray) -> ObjectWorldPose:
    value = np.asarray(matrix, dtype=float)
    return ObjectWorldPose(
        position_world=tuple(float(item) for item in value[:3, 3]),
        orientation_xyzw_world=rotmat_to_quat_xyzw(value[:3, :3]),
    )


def _matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=float)]


def _pose_payload(matrix: np.ndarray) -> dict[str, object]:
    pose = _matrix_pose(matrix)
    return {
        "position_assembly_m": list(pose.position_world),
        "orientation_xyzw_assembly": list(pose.orientation_xyzw_world),
        "matrix_assembly_m": _matrix_payload(matrix),
    }


def _safe_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in str(value).strip()
    )
    return cleaned[:80] or "symmetry"


def _is_identity(matrix: np.ndarray) -> bool:
    return bool(
        np.allclose(
            np.asarray(matrix, dtype=float),
            _identity_matrix(),
            atol=_MATRIX_ATOL,
        )
    )


def _validate_rigid_matrix(raw: object, *, context: str) -> np.ndarray:
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a numeric 4x4 matrix.") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{context} must be a finite numeric 4x4 matrix.")
    if not np.allclose(
        matrix[3],
        np.asarray((0.0, 0.0, 0.0, 1.0)),
        atol=_MATRIX_ATOL,
    ):
        raise ValueError(f"{context} has an invalid homogeneous bottom row.")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1.0e-7,
    ) or not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        abs_tol=1.0e-7,
    ):
        raise ValueError(f"{context} is not a proper rigid rotation.")
    return matrix


@dataclass(frozen=True)
class AssemblySymmetryRecord:
    """One finite asset symmetry scaled into Fabrica assembly coordinates."""

    part_id: str
    name: str
    description: str
    source: str
    angle_deg: float
    matrix_assembly_m: tuple[tuple[float, float, float, float], ...]
    asset_matrix_obj: tuple[tuple[float, float, float, float], ...]
    is_identity: bool

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.matrix_assembly_m, dtype=float)

    def to_payload(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "angle_deg": self.angle_deg,
            "is_identity": self.is_identity,
            "matrix_assembly_m": [list(row) for row in self.matrix_assembly_m],
            "asset_matrix_obj": [list(row) for row in self.asset_matrix_obj],
        }


@dataclass(frozen=True)
class TransitionSymmetryCandidate:
    """One equivalent destination frame and insertion corridor for a step."""

    transition_id: str
    partial_assembly_symmetry_name: str
    incoming_destination_symmetry_name: str
    incoming_equivalence_symmetry_name: str
    partial_assembly_transform_m: tuple[tuple[float, float, float, float], ...]
    incoming_symmetry_source_m: tuple[tuple[float, float, float, float], ...]
    incoming_destination_transform_assembly_m: tuple[tuple[float, float, float, float], ...]
    final_source_pose_assembly_m: tuple[tuple[float, float, float, float], ...]
    preinsertion_source_pose_assembly_m: tuple[tuple[float, float, float, float], ...]
    pre_to_final_translation_assembly_m: tuple[float, float, float]
    prefix_symmetry_matches: dict[str, str]
    maximum_transform_match_error_m: float
    is_identity: bool
    validation: dict[str, object] = field(default_factory=dict)

    @property
    def final_source_matrix(self) -> np.ndarray:
        return np.asarray(self.final_source_pose_assembly_m, dtype=float)

    @property
    def preinsertion_source_matrix(self) -> np.ndarray:
        return np.asarray(
            self.preinsertion_source_pose_assembly_m,
            dtype=float,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "partial_assembly_symmetry_name": (self.partial_assembly_symmetry_name),
            "incoming_destination_symmetry_name": (self.incoming_destination_symmetry_name),
            "incoming_equivalence_symmetry_name": (self.incoming_equivalence_symmetry_name),
            "is_identity": self.is_identity,
            "partial_assembly_transform_m": [list(row) for row in self.partial_assembly_transform_m],
            "incoming_symmetry_source_m": [list(row) for row in self.incoming_symmetry_source_m],
            "incoming_destination_transform_assembly_m": [
                list(row) for row in self.incoming_destination_transform_assembly_m
            ],
            "final_source_pose_assembly": _pose_payload(self.final_source_matrix),
            "preinsertion_source_pose_assembly": _pose_payload(self.preinsertion_source_matrix),
            "pre_to_final_translation_assembly_m": list(self.pre_to_final_translation_assembly_m),
            "prefix_symmetry_matches": dict(sorted(self.prefix_symmetry_matches.items())),
            "maximum_transform_match_error_m": (self.maximum_transform_match_error_m),
            "validation": dict(self.validation),
        }


def _identity_record(part_id: str) -> AssemblySymmetryRecord:
    identity = _identity_matrix()
    rows = tuple(tuple(float(value) for value in row) for row in identity)
    return AssemblySymmetryRecord(
        part_id=str(part_id),
        name="identity",
        description="Identity",
        source="synthetic_identity",
        angle_deg=0.0,
        matrix_assembly_m=rows,
        asset_matrix_obj=rows,
        is_identity=True,
    )


def _resolve_symmetry_asset_path(
    sequence: AssemblySequence,
    raw_path: str | Path | None,
) -> Path:
    if raw_path is None or not str(raw_path).strip():
        return (sequence.source_assembly_dir / "symmetries.json").resolve()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        if path.parts and path.parts[0] == "assets":
            path = REPO_ROOT / path
        else:
            path = sequence.source_assembly_dir / path
    return path.resolve()


def load_assembly_symmetry_records(
    sequence: AssemblySequence,
    *,
    symmetry_asset_path: str | Path | None = None,
    max_nonidentity: int = 0,
) -> tuple[dict[str, tuple[AssemblySymmetryRecord, ...]], dict[str, object]]:
    """Load finite symmetries and scale translations to the sequence scale."""

    path = _resolve_symmetry_asset_path(sequence, symmetry_asset_path)
    metadata: dict[str, object] = {
        "source_path": str(path),
        "load_status": "missing_file",
        "sequence_mesh_scale": float(sequence.mesh_scale),
    }
    identity_only = {part.part_id: (_identity_record(part.part_id),) for part in sequence.parts}
    if not path.is_file():
        return identity_only, metadata

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Symmetry asset '{path}' must contain an object.")
    asset_assembly = str(payload.get("assembly", sequence.assembly))
    if asset_assembly != sequence.assembly:
        raise ValueError(
            f"Symmetry asset assembly '{asset_assembly}' does not match sequence assembly '{sequence.assembly}'."
        )
    try:
        asset_scale = float(payload.get("mesh_scale", sequence.mesh_scale))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Symmetry asset '{path}' has an invalid mesh_scale.") from exc
    if not math.isfinite(asset_scale) or asset_scale <= 0.0:
        raise ValueError(f"Symmetry asset '{path}' mesh_scale must be > 0.")
    translation_scale = float(sequence.mesh_scale) / asset_scale
    raw_parts = payload.get("parts")
    if not isinstance(raw_parts, dict):
        raise ValueError(f"Symmetry asset '{path}' has no parts mapping.")

    records_by_part: dict[str, tuple[AssemblySymmetryRecord, ...]] = {}
    for part in sequence.parts:
        raw_part = raw_parts.get(part.part_id)
        raw_records = raw_part.get("symmetries", []) if isinstance(raw_part, dict) else []
        cleaned: list[AssemblySymmetryRecord] = []
        if isinstance(raw_records, list):
            for index, raw_record in enumerate(raw_records):
                if not isinstance(raw_record, dict):
                    continue
                asset_matrix = _validate_rigid_matrix(
                    raw_record.get("matrix_obj"),
                    context=(f"{path}: parts.{part.part_id}.symmetries[{index}]"),
                )
                matrix = np.array(asset_matrix, copy=True)
                matrix[:3, 3] *= translation_scale
                name = str(raw_record.get("name") or f"symmetry_{index}")
                cleaned.append(
                    AssemblySymmetryRecord(
                        part_id=part.part_id,
                        name=name,
                        description=str(raw_record.get("description") or name),
                        source=str(raw_record.get("source") or "unknown"),
                        angle_deg=float(raw_record.get("angle_deg", 0.0) or 0.0),
                        matrix_assembly_m=tuple(tuple(float(value) for value in row) for row in matrix),
                        asset_matrix_obj=tuple(tuple(float(value) for value in row) for row in asset_matrix),
                        is_identity=_is_identity(matrix),
                    )
                )
        identities = [record for record in cleaned if record.is_identity]
        nonidentities = [record for record in cleaned if not record.is_identity]
        if int(max_nonidentity) > 0:
            nonidentities = nonidentities[: int(max_nonidentity)]
        records_by_part[part.part_id] = tuple(
            [identities[0] if identities else _identity_record(part.part_id), *nonidentities]
        )

    metadata.update(
        {
            "load_status": "loaded",
            "asset_mesh_scale": asset_scale,
            "translation_scale": translation_scale,
            "part_transform_counts": {part_id: len(records) for part_id, records in sorted(records_by_part.items())},
        }
    )
    return records_by_part, metadata


def _normalize_vector(vector: np.ndarray) -> tuple[float, float, float]:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm > 1.0e-12:
        value = value / norm
    return tuple(float(item) for item in value)


def _candidate_geometry_key(candidate: SavedGraspCandidate) -> tuple[float, ...]:
    quaternion = np.asarray(candidate.grasp_orientation_xyzw_obj, dtype=float)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    values = (
        *candidate.grasp_position_obj,
        *quaternion.tolist(),
        *candidate.contact_point_a_obj,
        *candidate.contact_point_b_obj,
        *candidate.contact_normal_a_obj,
        *candidate.contact_normal_b_obj,
        float(candidate.jaw_width),
        float(candidate.contact_patch_lateral_offset_m),
        float(candidate.contact_patch_approach_offset_m),
    )
    return tuple(round(float(value), 7) for value in values)


def expand_grasp_candidates_by_symmetry(
    candidates: Iterable[SavedGraspCandidate],
    *,
    source_pose_assembly: ObjectWorldPose,
    symmetry_records: Iterable[AssemblySymmetryRecord],
) -> tuple[tuple[SavedGraspCandidate, ...], dict[str, object]]:
    """Expand object-frame grasps with asset symmetries in the source frame."""

    source_matrix = _pose_matrix(source_pose_assembly)
    assembly_from_source = source_matrix
    source_from_assembly = np.linalg.inv(assembly_from_source)
    records = tuple(symmetry_records)
    if not records:
        records = (_identity_record("unknown"),)
    output: list[SavedGraspCandidate] = []
    seen: set[tuple[float, ...]] = set()
    deduplicated = 0
    source_candidates = tuple(candidates)
    for candidate in source_candidates:
        for record in records:
            matrix_source = source_from_assembly @ record.matrix @ assembly_from_source
            rotation = matrix_source[:3, :3]
            translation = matrix_source[:3, 3]

            def point(raw: tuple[float, float, float]) -> tuple[float, float, float]:
                transformed = rotation @ np.asarray(raw, dtype=float) + translation
                return tuple(float(value) for value in transformed)

            def vector(raw: tuple[float, float, float]) -> tuple[float, float, float]:
                return _normalize_vector(rotation @ np.asarray(raw, dtype=float))

            metadata = {
                **(candidate.metadata or {}),
                "symmetry_pickup_parent_grasp_id": candidate.grasp_id,
                "symmetry_pickup_name": record.name,
                "symmetry_pickup_description": record.description,
                "symmetry_pickup_source": record.source,
                "symmetry_pickup_angle_deg": record.angle_deg,
                "symmetry_pickup_matrix_assembly_m": _matrix_payload(record.matrix),
                "symmetry_pickup_matrix_source": _matrix_payload(matrix_source),
                "symmetry_pickup_is_identity": record.is_identity,
            }
            if record.is_identity:
                expanded = replace(candidate, metadata=metadata)
            else:
                grasp_rotation = rotation @ quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj)
                expanded = SavedGraspCandidate(
                    grasp_id=(f"{candidate.grasp_id}__sym_{_safe_id(record.name)}"),
                    grasp_position_obj=point(candidate.grasp_position_obj),
                    grasp_orientation_xyzw_obj=rotmat_to_quat_xyzw(grasp_rotation),
                    contact_point_a_obj=point(candidate.contact_point_a_obj),
                    contact_point_b_obj=point(candidate.contact_point_b_obj),
                    contact_normal_a_obj=vector(candidate.contact_normal_a_obj),
                    contact_normal_b_obj=vector(candidate.contact_normal_b_obj),
                    jaw_width=candidate.jaw_width,
                    roll_angle_rad=candidate.roll_angle_rad,
                    contact_patch_lateral_offset_m=(candidate.contact_patch_lateral_offset_m),
                    contact_patch_approach_offset_m=(candidate.contact_patch_approach_offset_m),
                    score=candidate.score,
                    score_components=(None if candidate.score_components is None else dict(candidate.score_components)),
                    metadata=metadata,
                )
            key = _candidate_geometry_key(expanded)
            if key in seen:
                deduplicated += 1
                continue
            seen.add(key)
            output.append(expanded)
    return tuple(output), {
        "symmetry_pickup_source_candidate_count": len(source_candidates),
        "symmetry_pickup_transform_count": len(records),
        "symmetry_pickup_transform_names": [record.name for record in records],
        "symmetry_pickup_expanded_candidate_count": len(output),
        "symmetry_pickup_derived_candidate_count": max(
            0,
            len(output) - len(source_candidates),
        ),
        "symmetry_pickup_deduplicated_candidate_count": deduplicated,
    }


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    return np.asarray(points, dtype=float) @ value[:3, :3].T + value[:3, 3][None, :]


def _record_match(
    transform: np.ndarray,
    records: Iterable[AssemblySymmetryRecord],
    *,
    vertices_assembly: np.ndarray,
) -> tuple[AssemblySymmetryRecord | None, float]:
    transformed = _transform_points(transform, vertices_assembly)
    best: AssemblySymmetryRecord | None = None
    best_error = float("inf")
    for record in records:
        reference = _transform_points(record.matrix, vertices_assembly)
        error = float(np.max(np.linalg.norm(transformed - reference, axis=1)))
        if error < best_error:
            best = record
            best_error = error
    return best, best_error


def _yaw_preserving(matrix: np.ndarray) -> bool:
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    return bool(
        np.allclose(
            rotation @ np.asarray((0.0, 0.0, 1.0)),
            np.asarray((0.0, 0.0, 1.0)),
            atol=1.0e-7,
        )
    )


def compile_step_transition_symmetries(
    *,
    sequence: AssemblySequence,
    step: AssemblySequenceStep,
    incoming_source_pose_assembly: ObjectWorldPose,
    symmetry_asset_path: str | Path | None = None,
    geometry_tolerance_m: float = 0.001,
    max_partial_assembly_transforms: int = 0,
    max_incoming_transforms: int = 0,
    yaw_only: bool = True,
    enabled: bool = True,
) -> tuple[tuple[TransitionSymmetryCandidate, ...], dict[str, object]]:
    """Compile symmetry-equivalent pre-insertion targets for one step."""

    tolerance = float(geometry_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("geometry_tolerance_m must be finite and >= 0.")
    final_source = _pose_matrix(incoming_source_pose_assembly)
    final_to_pre = np.asarray(
        step.final_to_pre_insertion_transform_m,
        dtype=float,
    )
    canonical_pre = final_to_pre @ final_source
    if not enabled:
        translation = final_source[:3, 3] - canonical_pre[:3, 3]
        identity_candidate = TransitionSymmetryCandidate(
            transition_id="tr_identity__part_identity",
            partial_assembly_symmetry_name="identity",
            incoming_destination_symmetry_name="identity",
            incoming_equivalence_symmetry_name="identity",
            partial_assembly_transform_m=tuple(tuple(float(value) for value in row) for row in _identity_matrix()),
            incoming_symmetry_source_m=tuple(tuple(float(value) for value in row) for row in _identity_matrix()),
            incoming_destination_transform_assembly_m=tuple(
                tuple(float(value) for value in row) for row in _identity_matrix()
            ),
            final_source_pose_assembly_m=tuple(tuple(float(value) for value in row) for row in final_source),
            preinsertion_source_pose_assembly_m=tuple(tuple(float(value) for value in row) for row in canonical_pre),
            pre_to_final_translation_assembly_m=tuple(float(value) for value in translation),
            prefix_symmetry_matches={part_id: "identity" for part_id in step.assembled_part_ids_before},
            maximum_transform_match_error_m=0.0,
            is_identity=True,
            validation={
                "status": "identity_only_disabled",
                "geometry_tolerance_m": tolerance,
                "yaw_only": bool(yaw_only),
                "final_geometry_equivalence_checked": False,
                "assembled_prefix_equivalence_checked": False,
                "gripper_sweep_checked": True,
                "robot_path_checked": False,
            },
        )
        return (identity_candidate,), {
            "schema_version": SCHEMA_VERSION,
            "enabled": False,
            "source_path": None,
            "load_status": "disabled",
            "sequence_mesh_scale": float(sequence.mesh_scale),
            "geometry_tolerance_m": tolerance,
            "yaw_only": bool(yaw_only),
            "base_symmetry_candidate_count": 1,
            "incoming_symmetry_candidate_count": 1,
            "accepted_partial_assembly_symmetry_count": 1,
            "raw_transition_combination_count": 1,
            "transition_candidate_count": 1,
            "deduplicated_transition_combination_count": 0,
            "rejected_partial_assembly_symmetries": [],
        }

    records_by_part, load_metadata = load_assembly_symmetry_records(
        sequence,
        symmetry_asset_path=symmetry_asset_path,
        # Candidate limits must not truncate the symmetry records used to
        # prove prefix/incoming equivalence. Slice only the candidate sets
        # below; retain the complete finite set for matching.
        max_nonidentity=0,
    )
    base_records = list(records_by_part[sequence.base_part_id])
    if int(max_partial_assembly_transforms) > 0:
        base_records = [
            base_records[0],
            *base_records[1 : 1 + int(max_partial_assembly_transforms)],
        ]
    incoming_match_records = list(records_by_part[step.incoming_part_id])
    incoming_records = list(incoming_match_records)
    if int(max_incoming_transforms) > 0:
        incoming_records = [
            incoming_records[0],
            *incoming_records[1 : 1 + int(max_incoming_transforms)],
        ]

    vertices_by_part = {
        part_id: load_triangle_mesh(
            sequence.parts_by_id[part_id].resolved_mesh_path,
            scale=float(sequence.mesh_scale),
        ).vertices_obj
        for part_id in {
            *step.assembled_part_ids_before,
            step.incoming_part_id,
        }
    }
    source_from_final = np.linalg.inv(final_source)

    accepted_partial: list[tuple[AssemblySymmetryRecord, dict[str, str], str, float]] = []
    rejected: list[dict[str, object]] = []
    for partial in base_records:
        if yaw_only and not _yaw_preserving(partial.matrix):
            rejected.append(
                {
                    "partial_assembly_symmetry_name": partial.name,
                    "reason": "non_planar_symmetry",
                }
            )
            continue
        prefix_matches: dict[str, str] = {}
        maximum_error = 0.0
        failure: dict[str, object] | None = None
        for part_id in step.assembled_part_ids_before:
            match, error = _record_match(
                partial.matrix,
                records_by_part[part_id],
                vertices_assembly=vertices_by_part[part_id],
            )
            maximum_error = max(maximum_error, error)
            if match is None or error > tolerance:
                failure = {
                    "partial_assembly_symmetry_name": partial.name,
                    "reason": "assembled_prefix_not_equivalent",
                    "part_id": part_id,
                    "transform_match_error_m": error,
                }
                break
            prefix_matches[part_id] = match.name
        if failure is not None:
            rejected.append(failure)
            continue
        incoming_match, incoming_error = _record_match(
            partial.matrix,
            incoming_match_records,
            vertices_assembly=vertices_by_part[step.incoming_part_id],
        )
        maximum_error = max(maximum_error, incoming_error)
        if incoming_match is None or incoming_error > tolerance:
            rejected.append(
                {
                    "partial_assembly_symmetry_name": partial.name,
                    "reason": "incoming_final_not_equivalent",
                    "part_id": step.incoming_part_id,
                    "transform_match_error_m": incoming_error,
                }
            )
            continue
        accepted_partial.append(
            (
                partial,
                prefix_matches,
                incoming_match.name,
                maximum_error,
            )
        )

    candidates: list[TransitionSymmetryCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for partial, prefix_matches, incoming_match_name, match_error in accepted_partial:
        for destination in incoming_records:
            destination_source = source_from_final @ destination.matrix @ final_source
            final_candidate = partial.matrix @ destination.matrix @ final_source
            pre_candidate = partial.matrix @ destination.matrix @ canonical_pre
            key = tuple(
                round(float(value), 9)
                for value in np.concatenate((final_candidate.reshape(-1), pre_candidate.reshape(-1)))
            )
            if key in seen:
                continue
            seen.add(key)
            translation = final_candidate[:3, 3] - pre_candidate[:3, 3]
            transition_id = f"tr_{_safe_id(partial.name)}__part_{_safe_id(destination.name)}"
            candidates.append(
                TransitionSymmetryCandidate(
                    transition_id=transition_id,
                    partial_assembly_symmetry_name=partial.name,
                    incoming_destination_symmetry_name=destination.name,
                    incoming_equivalence_symmetry_name=(incoming_match_name),
                    partial_assembly_transform_m=tuple(tuple(float(value) for value in row) for row in partial.matrix),
                    incoming_symmetry_source_m=tuple(
                        tuple(float(value) for value in row) for row in destination_source
                    ),
                    incoming_destination_transform_assembly_m=tuple(
                        tuple(float(value) for value in row) for row in destination.matrix
                    ),
                    final_source_pose_assembly_m=tuple(tuple(float(value) for value in row) for row in final_candidate),
                    preinsertion_source_pose_assembly_m=tuple(
                        tuple(float(value) for value in row) for row in pre_candidate
                    ),
                    pre_to_final_translation_assembly_m=tuple(float(value) for value in translation),
                    prefix_symmetry_matches=dict(prefix_matches),
                    maximum_transform_match_error_m=float(match_error),
                    is_identity=(partial.is_identity and destination.is_identity),
                    validation={
                        "status": "accepted",
                        "geometry_tolerance_m": tolerance,
                        "yaw_only": bool(yaw_only),
                        "final_geometry_equivalence_checked": True,
                        "assembled_prefix_equivalence_checked": True,
                        "gripper_sweep_checked": False,
                        "robot_path_checked": False,
                    },
                )
            )

    candidates.sort(
        key=lambda candidate: (
            not candidate.is_identity,
            candidate.partial_assembly_symmetry_name,
            candidate.incoming_destination_symmetry_name,
            candidate.transition_id,
        )
    )
    if not candidates:
        raise RuntimeError(f"Step '{step.step_id}' produced no identity transition candidate.")
    metadata = {
        **load_metadata,
        "schema_version": SCHEMA_VERSION,
        "enabled": bool(enabled),
        "geometry_tolerance_m": tolerance,
        "yaw_only": bool(yaw_only),
        "base_symmetry_candidate_count": len(base_records),
        "incoming_symmetry_candidate_count": len(incoming_records),
        "incoming_symmetry_match_record_count": len(incoming_match_records),
        "accepted_partial_assembly_symmetry_count": len(accepted_partial),
        "raw_transition_combination_count": (len(accepted_partial) * len(incoming_records)),
        "transition_candidate_count": len(candidates),
        "deduplicated_transition_combination_count": (len(accepted_partial) * len(incoming_records) - len(candidates)),
        "rejected_partial_assembly_symmetries": rejected,
    }
    return tuple(candidates), metadata


__all__ = [
    "AssemblySymmetryRecord",
    "SCHEMA_VERSION",
    "TransitionSymmetryCandidate",
    "compile_step_transition_symmetries",
    "expand_grasp_candidates_by_symmetry",
    "load_assembly_symmetry_records",
]
