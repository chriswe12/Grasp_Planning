"""Resolve one dual-grasp pair into a simple holder/pickup simulation task."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from grasp_planning.grasping.collision import (
    BoxCollisionPrimitive,
    MeshCollisionPrimitive,
    make_gripper_collision_model,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspCandidate,
    evaluate_saved_grasps_against_pickup_pose,
    load_grasp_bundle,
    quat_to_rotmat_xyzw,
    rotmat_to_quat_xyzw,
    transform_primitive_to_world,
)
from grasp_planning.grasping.grasp_transforms import (
    WorldFrameGraspCandidate,
    saved_grasp_to_world_grasp,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.grasping.mesh_io import load_triangle_mesh, resolve_mesh_path
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .dual_robot_pair_scoring import (
    MovableFrame,
    ReachabilityProxyConfig,
    TaskTargetPose,
    pair_layout_score,
)

SIMPLE_SIM_SCHEMA_VERSION = 3
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = Path("artifacts/dual_grasp_planning")
DEFAULT_ASSEMBLY_NAME = "plumbers_block"
DEFAULT_INCOMING_PART_ID = "0"
DEFAULT_STEP_ID = "step_001_part_0"
DEFAULT_ARTIFACT_DIR = DEFAULT_ARTIFACT_ROOT / DEFAULT_ASSEMBLY_NAME
DEFAULT_ASSEMBLY_WORLD = MovableFrame((0.55, 0.0, 0.0), 0.0)
DEFAULT_HOLDER_BASE_WORLD = MovableFrame((0.0, -0.42, 0.0), 0.0)
DEFAULT_INSERTER_BASE_WORLD = MovableFrame((0.0, 0.42, 0.0), 0.0)
DEFAULT_PICKUP_SOURCE_WORLD_XY = (0.55, 0.28)
DEFAULT_PICKUP_ORIENTATION_RPY_DEG = (0.0, 0.0, 0.0)
DEFAULT_HOLDER_PREGRASP_OFFSET_M = 0.05
DEFAULT_INSERTER_PREGRASP_OFFSET_M = 0.10
DEFAULT_TRANSPORT_CLEARANCE_M = 0.08
DEFAULT_PICKUP_CONTACT_GAP_M = 0.002
DEFAULT_PICKUP_FLOOR_CLEARANCE_MARGIN_M = 0.001
DEFAULT_PICKUP_TOP_DOWN_SCORE_WEIGHT = 0.25
DEFAULT_TRANSITION_SCORE_WEIGHT = 0.35
DEFAULT_TRANSITION_TRANSLATION_SCALE_M = 0.50
DEFAULT_TRANSITION_TRANSLATION_WEIGHT = 0.50
DEFAULT_TRANSITION_ROTATION_WEIGHT = 0.50
DEFAULT_FLOOR_Z_WORLD_M = -0.030
DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT = 256
DEFAULT_PREGRASP_AABB_CORRIDOR_MARGIN_M = 0.002
_MIN_PREGRASP_AABB_PIECE_SIZE_M = 1.0e-5


@dataclass(frozen=True)
class PlanarRuntimeLayout:
    """Runtime placement inputs accepted by the table-supported slice."""

    assembly_world: MovableFrame
    pickup_source_world_xy: tuple[float, float]
    pickup_orientation_rpy_deg: tuple[float, float, float]
    perceived_part_aabbs: tuple["RuntimePartAabb", ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimePartAabb:
    """World bounds used for visualization or temporary pregrasp collision."""

    role: str
    minimum_world_m: tuple[float, float, float]
    maximum_world_m: tuple[float, float, float]


@dataclass(frozen=True)
class DualRobotStepSelection:
    """One executable selected-order step and its existing artifact directory."""

    artifact_dir: Path
    assembly: str
    base_part_id: str
    incoming_part_id: str
    step_id: str
    step_index: int
    assembled_part_ids_before: tuple[str, ...]


@dataclass(frozen=True)
class SimpleDualRobotSubassemblyPart:
    """One final-coordinate mesh included in the rigid assembled prefix."""

    part_id: str
    mesh_path: Path


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return payload


def _repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def resolve_dual_robot_step_selection(
    *,
    assembly: str | None = None,
    incoming_part_id: str | int | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    artifact_dir: str | Path | None = None,
    step_id: str | None = None,
) -> DualRobotStepSelection:
    """Resolve an assembly/incoming-part request to one selected-order step.

    Explicit ``artifact_dir`` and ``step_id`` remain supported for existing
    action-server and debugging callers. With no arguments this preserves the
    original plumbers-block first-step behavior.
    """

    requested_assembly = None if assembly is None else str(assembly)
    if artifact_dir is None:
        selected_assembly = requested_assembly or DEFAULT_ASSEMBLY_NAME
        root = _repo_path(Path(artifact_root) / selected_assembly)
    else:
        root = _repo_path(artifact_dir)

    sequence_path = root / "assembly_sequence.json"
    if not sequence_path.is_file():
        raise FileNotFoundError(f"Dual planning artifacts are missing the assembly sequence: {sequence_path}")
    sequence = _read_json(sequence_path)
    sequence_assembly = str(sequence.get("assembly", ""))
    if not sequence_assembly:
        raise ValueError(f"Assembly sequence has no assembly name: {sequence_path}")
    if requested_assembly is not None and requested_assembly != sequence_assembly:
        raise ValueError(
            f"Requested assembly '{requested_assembly}' does not match artifact assembly '{sequence_assembly}'."
        )

    raw_steps = sequence.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError(f"Assembly sequence has no steps list: {sequence_path}")
    executable_steps = [
        dict(raw_step)
        for raw_step in raw_steps
        if isinstance(raw_step, dict) and bool(raw_step.get("holder_base_available", False))
    ]
    if not executable_steps:
        raise ValueError(f"Assembly '{sequence_assembly}' has no holder-active insertion steps.")

    requested_step = None if step_id is None else str(step_id)
    requested_incoming = None if incoming_part_id is None else str(incoming_part_id)
    if requested_step is not None:
        matching = [step for step in executable_steps if str(step.get("step_id", "")) == requested_step]
    elif requested_incoming is not None:
        matching = [step for step in executable_steps if str(step.get("incoming_part_id", "")) == requested_incoming]
    else:
        first_holder_index = sequence.get("first_holder_step_index")
        matching = [step for step in executable_steps if int(step.get("step_index", -1)) == int(first_holder_index)]
        if not matching:
            matching = executable_steps[:1]

    if len(matching) != 1:
        requested = (
            f"step_id={requested_step!r}" if requested_step is not None else f"incoming_part_id={requested_incoming!r}"
        )
        raise ValueError(
            f"Could not resolve exactly one holder-active step for {requested} in assembly '{sequence_assembly}'."
        )
    selected = matching[0]
    selected_incoming = str(selected.get("incoming_part_id", ""))
    if requested_incoming is not None and selected_incoming != requested_incoming:
        raise ValueError(
            f"Step '{selected.get('step_id')}' inserts part "
            f"'{selected_incoming}', not requested part '{requested_incoming}'."
        )

    selected_step_id = str(selected.get("step_id", ""))
    required = (
        root / f"dual_grasp_pairs_{selected_step_id}.json",
        root / f"inserter_candidates_{selected_step_id}.json",
        root / "holder_base_candidates.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Dual planning artifacts are incomplete for '{sequence_assembly}' step '{selected_step_id}': {missing}"
        )

    return DualRobotStepSelection(
        artifact_dir=root,
        assembly=sequence_assembly,
        base_part_id=str(sequence.get("base_part_id", "")),
        incoming_part_id=selected_incoming,
        step_id=selected_step_id,
        step_index=int(selected.get("step_index", -1)),
        assembled_part_ids_before=tuple(str(value) for value in selected.get("assembled_part_ids_before", [])),
    )


def _yaw_rotation(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )


def _rpy_rotation(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


def _rotation_to_rpy_deg(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """Return XYZ roll/pitch/yaw for Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    matrix = np.asarray(rotation, dtype=float)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1.0e-8:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _assembly_pose(frame: MovableFrame) -> ObjectWorldPose:
    rotation = _yaw_rotation(frame.yaw_deg)
    return ObjectWorldPose(
        position_world=frame.position_world_m,
        orientation_xyzw_world=tuple(float(value) for value in rotmat_to_quat_xyzw(rotation)),
    )


def compose_source_pose_world(
    *,
    source_pose_assembly: ObjectWorldPose,
    assembly_world: MovableFrame,
    translation_assembly_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ObjectWorldPose:
    """Compose an artifact source frame and an optional assembly-frame translation."""

    assembly_pose = _assembly_pose(assembly_world)
    translated_source_position = source_pose_assembly.translation_world + np.asarray(
        translation_assembly_m, dtype=float
    )
    position_world = (
        assembly_pose.rotation_world_from_object @ translated_source_position + assembly_pose.translation_world
    )
    rotation_world = assembly_pose.rotation_world_from_object @ source_pose_assembly.rotation_world_from_object
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in position_world),
        orientation_xyzw_world=tuple(float(value) for value in rotmat_to_quat_xyzw(rotation_world)),
    )


def translated_source_pose_world(
    source_pose_world: ObjectWorldPose,
    *,
    position_world: tuple[float, float, float],
) -> ObjectWorldPose:
    return ObjectWorldPose(
        position_world=position_world,
        orientation_xyzw_world=source_pose_world.orientation_xyzw_world,
    )


def source_pose_resting_on_floor(
    *,
    mesh_assembly: TriangleMesh,
    source_pose_assembly: ObjectWorldPose,
    source_orientation_world: ObjectWorldPose,
    xy_world: tuple[float, float],
    floor_z_world_m: float = DEFAULT_FLOOR_Z_WORLD_M,
) -> ObjectWorldPose:
    """Place the source-local mesh with its lowest world-Z point on the floor."""

    vertices_source = (
        mesh_assembly.vertices_obj - source_pose_assembly.translation_world[None, :]
    ) @ source_pose_assembly.rotation_world_from_object
    vertices_world_offset = vertices_source @ source_orientation_world.rotation_world_from_object.T
    root_z_world = float(floor_z_world_m - np.min(vertices_world_offset[:, 2]))
    return ObjectWorldPose(
        position_world=(
            float(xy_world[0]),
            float(xy_world[1]),
            root_z_world,
        ),
        orientation_xyzw_world=(source_orientation_world.orientation_xyzw_world),
    )


def _source_pose_from_bundle(bundle) -> ObjectWorldPose:
    return ObjectWorldPose(
        position_world=bundle.source_frame_origin_obj_world,
        orientation_xyzw_world=(bundle.source_frame_orientation_xyzw_obj_world),
    )


def source_mesh_aabb_world(
    *,
    role: str,
    mesh_assembly: TriangleMesh,
    source_pose_assembly: ObjectWorldPose,
    source_pose_world: ObjectWorldPose,
) -> RuntimePartAabb:
    """Transform an assembly-frame mesh and return non-collision world bounds."""

    vertices_source = (
        mesh_assembly.vertices_obj - source_pose_assembly.translation_world[None, :]
    ) @ source_pose_assembly.rotation_world_from_object
    vertices_world = (
        vertices_source @ source_pose_world.rotation_world_from_object.T + source_pose_world.translation_world[None, :]
    )
    minimum = np.min(vertices_world, axis=0)
    maximum = np.max(vertices_world, axis=0)
    return RuntimePartAabb(
        role=str(role),
        minimum_world_m=tuple(float(value) for value in minimum),
        maximum_world_m=tuple(float(value) for value in maximum),
    )


def resolve_planar_runtime_layout(
    *,
    artifact_dir: str | Path,
    step_id: str,
    base_source_pose_world: ObjectWorldPose,
    incoming_source_pose_world: ObjectWorldPose,
    maximum_assembly_tilt_deg: float = 5.0,
) -> PlanarRuntimeLayout:
    """Derive assembly and pickup inputs from perceived source-frame poses."""

    root = Path(artifact_dir).expanduser().resolve()
    holder_bundle = load_grasp_bundle(root / "holder_base_candidates.json")
    inserter_bundle = load_grasp_bundle(root / f"inserter_candidates_{step_id}.json")
    base_source_pose_assembly = _source_pose_from_bundle(holder_bundle)
    pair_path = root / f"dual_grasp_pairs_{step_id}.json"
    if pair_path.is_file():
        _, base_source_pose_assembly, _ = _declared_holder_candidate_source(
            root=root,
            pair_payload=_read_json(pair_path),
            fallback_candidates=holder_bundle.candidates,
            fallback_source_pose_assembly=base_source_pose_assembly,
        )
    incoming_source_pose_assembly = _source_pose_from_bundle(inserter_bundle)
    perceived_part_aabbs = (
        source_mesh_aabb_world(
            role="base",
            mesh_assembly=load_triangle_mesh(
                resolve_mesh_path(holder_bundle.target_stl_path),
                scale=float(holder_bundle.stl_scale),
            ),
            source_pose_assembly=base_source_pose_assembly,
            source_pose_world=base_source_pose_world,
        ),
        source_mesh_aabb_world(
            role="incoming",
            mesh_assembly=load_triangle_mesh(
                resolve_mesh_path(inserter_bundle.target_stl_path),
                scale=float(inserter_bundle.stl_scale),
            ),
            source_pose_assembly=incoming_source_pose_assembly,
            source_pose_world=incoming_source_pose_world,
        ),
    )

    rotation_world_from_assembly = (
        base_source_pose_world.rotation_world_from_object @ base_source_pose_assembly.rotation_world_from_object.T
    )
    assembly_roll, assembly_pitch, assembly_yaw = _rotation_to_rpy_deg(rotation_world_from_assembly)
    maximum_tilt = float(maximum_assembly_tilt_deg)
    if maximum_tilt < 0.0:
        raise ValueError("maximum_assembly_tilt_deg must be non-negative.")
    runtime_warnings: list[str] = []
    if abs(assembly_roll) > maximum_tilt or abs(assembly_pitch) > maximum_tilt:
        runtime_warnings.append(
            "The perceived base implies a non-planar assembly frame: "
            f"roll={assembly_roll:.3f} deg pitch={assembly_pitch:.3f} deg; "
            f"warning threshold is {maximum_tilt:.3f} deg. "
            "Continuing with the yaw-only assembly layout."
        )

    assembly_translation_world = (
        base_source_pose_world.translation_world
        - rotation_world_from_assembly @ base_source_pose_assembly.translation_world
    )
    assembly_world = MovableFrame(
        tuple(float(value) for value in assembly_translation_world),
        float(assembly_yaw),
    )
    planar_rotation_world_from_assembly = _yaw_rotation(assembly_world.yaw_deg)
    incoming_final_rotation_world = (
        planar_rotation_world_from_assembly @ incoming_source_pose_assembly.rotation_world_from_object
    )
    pickup_delta_rotation_world = (
        incoming_source_pose_world.rotation_world_from_object @ incoming_final_rotation_world.T
    )
    return PlanarRuntimeLayout(
        assembly_world=assembly_world,
        pickup_source_world_xy=(
            float(incoming_source_pose_world.position_world[0]),
            float(incoming_source_pose_world.position_world[1]),
        ),
        pickup_orientation_rpy_deg=tuple(float(value) for value in _rotation_to_rpy_deg(pickup_delta_rotation_world)),
        perceived_part_aabbs=perceived_part_aabbs,
        warnings=tuple(runtime_warnings),
    )


def _candidate_by_id(
    candidates: Iterable[SavedGraspCandidate],
    grasp_id: str,
) -> SavedGraspCandidate:
    for candidate in candidates:
        if candidate.grasp_id == grasp_id:
            return candidate
    raise ValueError(f"Grasp candidate '{grasp_id}' is missing from its source artifact.")


def _saved_candidate_from_payload(
    raw_candidate: Mapping[str, object],
    *,
    expected_grasp_id: str | None = None,
) -> SavedGraspCandidate:
    """Deserialize one candidate embedded in a declared Stage-3 source."""

    grasp_id = str(raw_candidate["grasp_id"])
    if expected_grasp_id is not None and grasp_id != expected_grasp_id:
        raise ValueError(
            "Candidate mapping key does not match its payload grasp_id: "
            f"key={expected_grasp_id!r}, payload={grasp_id!r}."
        )
    grasp_pose = raw_candidate.get("grasp_pose_obj")
    contact_points = raw_candidate.get("contact_points_obj")
    contact_normals = raw_candidate.get("contact_normals_obj")
    if not isinstance(grasp_pose, Mapping):
        raise ValueError(f"Grasp candidate '{grasp_id}' has no grasp_pose_obj mapping.")
    if not isinstance(contact_points, list) or len(contact_points) != 2:
        raise ValueError(f"Grasp candidate '{grasp_id}' must contain two contact points.")
    if not isinstance(contact_normals, list) or len(contact_normals) != 2:
        raise ValueError(f"Grasp candidate '{grasp_id}' must contain two contact normals.")
    contact_patch_offset = raw_candidate.get("contact_patch_offset_local", (0.0, 0.0))
    if not isinstance(contact_patch_offset, (list, tuple)) or len(contact_patch_offset) != 2:
        raise ValueError(f"Grasp candidate '{grasp_id}' has an invalid contact_patch_offset_local.")
    score_components_raw = raw_candidate.get("score_components")
    metadata_raw = raw_candidate.get("metadata")
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=tuple(float(value) for value in grasp_pose["position"]),  # type: ignore[arg-type]
        grasp_orientation_xyzw_obj=tuple(
            float(value)
            for value in grasp_pose["orientation_xyzw"]  # type: ignore[arg-type]
        ),
        contact_point_a_obj=tuple(float(value) for value in contact_points[0]),  # type: ignore[arg-type]
        contact_point_b_obj=tuple(float(value) for value in contact_points[1]),  # type: ignore[arg-type]
        contact_normal_a_obj=tuple(float(value) for value in contact_normals[0]),  # type: ignore[arg-type]
        contact_normal_b_obj=tuple(float(value) for value in contact_normals[1]),  # type: ignore[arg-type]
        jaw_width=float(raw_candidate["jaw_width"]),
        roll_angle_rad=float(raw_candidate["roll_angle_rad"]),
        contact_patch_lateral_offset_m=float(contact_patch_offset[0]),
        contact_patch_approach_offset_m=float(contact_patch_offset[1]),
        score=None if raw_candidate.get("score") is None else float(raw_candidate["score"]),
        score_components=(
            None
            if score_components_raw is None
            else {str(key): float(value) for key, value in dict(score_components_raw).items()}  # type: ignore[arg-type]
        ),
        metadata=(dict(metadata_raw) or None if isinstance(metadata_raw, Mapping) else None),
    )


def _declared_holder_candidate_source(
    *,
    root: Path,
    pair_payload: Mapping[str, object],
    fallback_candidates: tuple[SavedGraspCandidate, ...],
    fallback_source_pose_assembly: ObjectWorldPose,
) -> tuple[tuple[SavedGraspCandidate, ...], ObjectWorldPose, dict[str, object]]:
    """Load holder poses from the source explicitly named by Stage 3.

    Sequential grasp IDs are meaningful only within one generated library. A
    pair artifact therefore must not resolve those IDs through an unrelated or
    stale ``holder_base_candidates.json`` file.
    """

    sources_raw = pair_payload.get("candidate_sources")
    sources = dict(sources_raw) if isinstance(sources_raw, Mapping) else {}
    holder_raw = sources.get("holder")
    if not isinstance(holder_raw, Mapping):
        return (
            fallback_candidates,
            fallback_source_pose_assembly,
            {
                "artifact": "holder_base_candidates.json",
                "candidate_collection": "candidates",
                "legacy_fallback": True,
                "candidate_count": len(fallback_candidates),
            },
        )

    artifact_raw = holder_raw.get("artifact")
    collection_name = str(holder_raw.get("candidate_collection", "candidates"))
    if not artifact_raw:
        raise ValueError("Stage-3 holder candidate source has no artifact path.")
    artifact_path = Path(str(artifact_raw)).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    artifact_path = artifact_path.resolve()
    if artifact_path != root and root not in artifact_path.parents:
        raise ValueError(f"Stage-3 holder candidate source escapes its artifact directory: {artifact_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Stage-3 holder candidate source does not exist: {artifact_path}")

    source_payload = _read_json(artifact_path)
    collection_raw = source_payload.get(collection_name)
    candidate_items: list[tuple[str | None, Mapping[str, object]]] = []
    if isinstance(collection_raw, Mapping):
        for candidate_id, raw_candidate in collection_raw.items():
            if not isinstance(raw_candidate, Mapping):
                raise ValueError(f"Candidate '{candidate_id}' in '{artifact_path}' is not a mapping.")
            candidate_items.append((str(candidate_id), raw_candidate))
    elif isinstance(collection_raw, list):
        for index, raw_candidate in enumerate(collection_raw):
            if not isinstance(raw_candidate, Mapping):
                raise ValueError(f"Candidate index {index} in '{artifact_path}' is not a mapping.")
            candidate_items.append((None, raw_candidate))
    else:
        raise ValueError(f"Stage-3 holder source '{artifact_path}' has no candidate collection '{collection_name}'.")
    candidates = tuple(
        _saved_candidate_from_payload(raw_candidate, expected_grasp_id=candidate_id)
        for candidate_id, raw_candidate in candidate_items
    )
    candidate_ids = [candidate.grasp_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"Stage-3 holder source '{artifact_path}' contains duplicate grasp IDs.")

    source_pose_raw = source_payload.get("source_frame_pose_assembly")
    if isinstance(source_pose_raw, Mapping):
        source_pose_assembly = ObjectWorldPose(
            position_world=tuple(float(value) for value in source_pose_raw["position"]),  # type: ignore[arg-type]
            orientation_xyzw_world=tuple(
                float(value)
                for value in source_pose_raw["orientation_xyzw"]  # type: ignore[arg-type]
            ),
        )
    else:
        source_pose_assembly = fallback_source_pose_assembly

    return (
        candidates,
        source_pose_assembly,
        {
            "artifact": artifact_path.name,
            "candidate_collection": collection_name,
            "legacy_fallback": False,
            "candidate_count": len(candidates),
            "source_holder_cache_key": source_payload.get("source_holder_cache_key"),
        },
    )


def _world_grasp_payload(
    grasp: WorldFrameGraspCandidate,
    *,
    candidate: SavedGraspCandidate | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "grasp_id": grasp.grasp_id,
        "position_world_m": list(grasp.position_w),
        "orientation_xyzw_world": list(grasp.orientation_xyzw),
        "pregrasp_position_world_m": list(grasp.pregrasp_position_w),
        "approach_axis_world": list(grasp.normal_w),
        "pregrasp_offset_m": float(grasp.pregrasp_offset),
        "jaw_width_m": float(grasp.jaw_width),
        "open_width_m": float(grasp.gripper_width),
    }
    if candidate is not None:
        metadata = dict(candidate.metadata or {})
        payload["part_to_tcp"] = {
            "position_part_m": list(candidate.grasp_position_obj),
            "orientation_xyzw_part": list(candidate.grasp_orientation_xyzw_obj),
        }
        payload["symmetry_provenance"] = {
            "parent_grasp_id": str(
                metadata.get(
                    "symmetry_pickup_parent_grasp_id",
                    metadata.get("source_grasp_id", candidate.grasp_id),
                )
            ),
            "pickup_symmetry_name": str(metadata.get("symmetry_pickup_name", "identity")),
            "pickup_symmetry_is_identity": bool(metadata.get("symmetry_pickup_is_identity", True)),
            "pickup_symmetry_matrix_source": metadata.get("symmetry_pickup_matrix_source"),
        }
    return payload


def _pose_payload(pose: ObjectWorldPose) -> dict[str, object]:
    return {
        "position_world_m": list(pose.position_world),
        "orientation_xyzw_world": list(pose.orientation_xyzw_world),
    }


def _pose_from_matrix_payload(raw: Mapping[str, object]) -> ObjectWorldPose:
    matrix = np.asarray(raw.get("matrix_assembly_m"), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("Transition source pose must contain a finite 4x4 matrix.")
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in matrix[:3, 3]),
        orientation_xyzw_world=rotmat_to_quat_xyzw(matrix[:3, :3]),
    )


def _legacy_transition_payload(
    *,
    source_pose_assembly: ObjectWorldPose,
    final_to_pre_translation_assembly_m: tuple[float, float, float],
) -> dict[str, object]:
    final_matrix = np.eye(4, dtype=float)
    final_matrix[:3, :3] = source_pose_assembly.rotation_world_from_object
    final_matrix[:3, 3] = source_pose_assembly.translation_world
    pre_matrix = np.array(final_matrix, copy=True)
    pre_matrix[:3, 3] += np.asarray(
        final_to_pre_translation_assembly_m,
        dtype=float,
    )

    def pose(matrix: np.ndarray) -> dict[str, object]:
        return {
            "matrix_assembly_m": matrix.tolist(),
            "position_assembly_m": matrix[:3, 3].tolist(),
            "orientation_xyzw_assembly": list(rotmat_to_quat_xyzw(matrix[:3, :3])),
        }

    return {
        "transition_id": "tr_identity__part_identity",
        "partial_assembly_symmetry_name": "identity",
        "incoming_destination_symmetry_name": "identity",
        "incoming_equivalence_symmetry_name": "identity",
        "is_identity": True,
        "final_source_pose_assembly": pose(final_matrix),
        "preinsertion_source_pose_assembly": pose(pre_matrix),
        "pre_to_final_translation_assembly_m": list(-np.asarray(final_to_pre_translation_assembly_m, dtype=float)),
        "validation": {
            "status": "legacy_identity",
            "gripper_sweep_checked": True,
            "robot_path_checked": False,
        },
    }


def _transition_payloads(
    pair_payload: Mapping[str, object],
    *,
    source_pose_assembly: ObjectWorldPose,
    final_to_pre_translation_assembly_m: tuple[float, float, float],
) -> tuple[dict[str, object], ...]:
    raw_section = pair_payload.get("transition_symmetry")
    if isinstance(raw_section, dict):
        raw_candidates = raw_section.get("candidates")
        if isinstance(raw_candidates, list) and raw_candidates:
            candidates = tuple(dict(candidate) for candidate in raw_candidates if isinstance(candidate, dict))
            if candidates:
                return candidates
    return (
        _legacy_transition_payload(
            source_pose_assembly=source_pose_assembly,
            final_to_pre_translation_assembly_m=(final_to_pre_translation_assembly_m),
        ),
    )


def _quaternion_distance_rad(
    first_xyzw: tuple[float, float, float, float],
    second_xyzw: tuple[float, float, float, float],
) -> float:
    first = np.asarray(first_xyzw, dtype=float)
    second = np.asarray(second_xyzw, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = abs(float(np.dot(first, second)))
    return float(2.0 * math.acos(np.clip(cosine, -1.0, 1.0)))


def _transition_motion_components(
    pickup_grasp: WorldFrameGraspCandidate,
    preinsertion_grasp: WorldFrameGraspCandidate,
    *,
    translation_scale_m: float,
    translation_weight: float,
    rotation_weight: float,
) -> dict[str, float]:
    translation = float(
        np.linalg.norm(
            np.asarray(preinsertion_grasp.position_w, dtype=float) - np.asarray(pickup_grasp.position_w, dtype=float)
        )
    )
    rotation = _quaternion_distance_rad(
        pickup_grasp.orientation_xyzw,
        preinsertion_grasp.orientation_xyzw,
    )
    translation_score = math.exp(-translation / max(float(translation_scale_m), 1.0e-9))
    rotation_score = max(0.0, 1.0 - rotation / math.pi)
    weight_sum = float(translation_weight) + float(rotation_weight)
    if weight_sum <= 0.0:
        raise ValueError("Transition translation/rotation weights must be positive.")
    score = (float(translation_weight) * translation_score + float(rotation_weight) * rotation_score) / weight_sum
    return {
        "translation_m": translation,
        "rotation_rad": rotation,
        "rotation_deg": math.degrees(rotation),
        "translation_score": translation_score,
        "rotation_score": rotation_score,
        "score": score,
    }


def source_local_subassembly_mesh(
    subassembly_payload: Mapping[str, object],
) -> TriangleMesh:
    """Combine final-coordinate prefix meshes in the holder-base source frame."""

    source_pose = dict(subassembly_payload["source_pose_assembly"])  # type: ignore[arg-type]
    source_position = np.asarray(
        source_pose["position_world_m"],
        dtype=float,
    )
    source_rotation = quat_to_rotmat_xyzw(
        tuple(
            float(value)
            for value in source_pose["orientation_xyzw_world"]  # type: ignore[index]
        )
    )
    raw_parts = subassembly_payload.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError("Subassembly payload must contain at least one part.")

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            raise ValueError("Each subassembly part must be a mapping.")
        mesh = load_triangle_mesh(
            str(raw_part["mesh_path"]),
            scale=float(raw_part["mesh_scale"]),
        )
        vertices_source = (mesh.vertices_obj - source_position[None, :]) @ source_rotation
        vertices.append(vertices_source)
        faces.append(mesh.faces + vertex_offset)
        vertex_offset += int(mesh.vertices_obj.shape[0])

    return TriangleMesh(
        vertices_obj=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
    )


@dataclass(frozen=True)
class SimpleDualRobotPairTask:
    artifact_dir: Path
    step_id: str
    step_index: int
    assembly: str
    base_part_id: str
    incoming_part_id: str
    pair_id: str
    transition_id: str
    execution_candidate_id: str
    pair_score: float
    selection_score: float
    transition_motion_score: float
    transition_motion_components: dict[str, float]
    pickup_top_down_score: float
    layout_proxy_score: float
    layout_proxy_components: dict[str, object]
    holder_reachability_proxy_score: float
    inserter_reachability_proxy_score: float
    holder_candidate: SavedGraspCandidate
    inserter_candidate: SavedGraspCandidate
    holder_source_pose_assembly: ObjectWorldPose
    incoming_source_pose_assembly: ObjectWorldPose
    holder_source_pose_world: ObjectWorldPose
    incoming_pickup_source_pose_world: ObjectWorldPose
    incoming_final_source_pose_world: ObjectWorldPose
    incoming_preinsertion_source_pose_world: ObjectWorldPose
    holder_world_grasp: WorldFrameGraspCandidate
    inserter_pickup_world_grasp: WorldFrameGraspCandidate
    inserter_preinsertion_world_grasp: WorldFrameGraspCandidate
    base_mesh_path: Path
    incoming_mesh_path: Path
    assembled_part_ids_before: tuple[str, ...]
    subassembly_parts: tuple[SimpleDualRobotSubassemblyPart, ...]
    mesh_scale: float
    assembly_world: MovableFrame
    final_to_preinsertion_translation_assembly_m: tuple[float, float, float]
    transport_clearance_m: float
    pickup_floor_z_world_m: float
    pickup_floor_check_reason: str
    pickup_floor_clearance_margin_m: float
    pickup_contact_gap_m: float
    pickup_gripper_collision_model: str
    transition_symmetry: dict[str, object]
    candidate_filter_diagnostics: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        pickup_grasp = np.asarray(
            self.inserter_pickup_world_grasp.position_w,
            dtype=float,
        )
        preinsertion_grasp = np.asarray(
            self.inserter_preinsertion_world_grasp.position_w,
            dtype=float,
        )
        lift_offset = np.asarray((0.0, 0.0, self.transport_clearance_m))
        return {
            "schema_version": SIMPLE_SIM_SCHEMA_VERSION,
            "kind": "dual_robot_simple_sim_task",
            "assembly": self.assembly,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "base_part_id": self.base_part_id,
            "incoming_part_id": self.incoming_part_id,
            "pair_id": self.pair_id,
            "transition_id": self.transition_id,
            "execution_candidate_id": self.execution_candidate_id,
            "pair_score": self.pair_score,
            "selection_score": self.selection_score,
            "transition_motion_score": self.transition_motion_score,
            "transition_motion_components": dict(self.transition_motion_components),
            "candidate_filter_diagnostics": dict(self.candidate_filter_diagnostics),
            "pickup_top_down_score": self.pickup_top_down_score,
            "layout_proxy_score": self.layout_proxy_score,
            "layout_proxy_components": dict(self.layout_proxy_components),
            "holder_reachability_proxy_score": (self.holder_reachability_proxy_score),
            "inserter_reachability_proxy_score": (self.inserter_reachability_proxy_score),
            "roles": {
                "holder": {
                    "robot": "lbr_one",
                    "planning_group": "arm_one",
                    "tcp_link": "lbr_one_gripper_tcp",
                },
                "inserter": {
                    "robot": "lbr_two",
                    "planning_group": "arm_two",
                    "tcp_link": "lbr_two_gripper_tcp",
                },
            },
            "layout": {
                "assembly_world": self.assembly_world.to_payload(),
                "holder_base_world_m": list(DEFAULT_HOLDER_BASE_WORLD.position_world_m),
                "inserter_base_world_m": list(DEFAULT_INSERTER_BASE_WORLD.position_world_m),
                "final_to_preinsertion_translation_assembly_m": list(self.final_to_preinsertion_translation_assembly_m),
                "pickup_floor_z_world_m": self.pickup_floor_z_world_m,
            },
            "transition_symmetry": dict(self.transition_symmetry),
            "objects": {
                "base": {
                    "part_id": self.base_part_id,
                    "mesh_path": str(self.base_mesh_path),
                    "mesh_scale": self.mesh_scale,
                    "source_pose_assembly": _pose_payload(self.holder_source_pose_assembly),
                    "source_pose_world": _pose_payload(self.holder_source_pose_world),
                },
                "subassembly": {
                    "base_part_id": self.base_part_id,
                    "part_ids": list(self.assembled_part_ids_before),
                    "mesh_scale": self.mesh_scale,
                    "source_pose_assembly": _pose_payload(self.holder_source_pose_assembly),
                    "source_pose_world": _pose_payload(self.holder_source_pose_world),
                    "parts": [
                        {
                            "part_id": part.part_id,
                            "mesh_path": str(part.mesh_path),
                            "mesh_scale": self.mesh_scale,
                        }
                        for part in self.subassembly_parts
                    ],
                    "physics": "single_rigid_compound",
                },
                "incoming": {
                    "part_id": self.incoming_part_id,
                    "mesh_path": str(self.incoming_mesh_path),
                    "mesh_scale": self.mesh_scale,
                    "source_pose_assembly": _pose_payload(self.incoming_source_pose_assembly),
                    "pickup_source_pose_world": _pose_payload(self.incoming_pickup_source_pose_world),
                    "final_source_pose_world": _pose_payload(self.incoming_final_source_pose_world),
                    "preinsertion_source_pose_world": _pose_payload(self.incoming_preinsertion_source_pose_world),
                },
            },
            "grasps": {
                "holder": _world_grasp_payload(
                    self.holder_world_grasp,
                    candidate=self.holder_candidate,
                ),
                "inserter_pickup": _world_grasp_payload(
                    self.inserter_pickup_world_grasp,
                    candidate=self.inserter_candidate,
                ),
                "inserter_preinsertion": _world_grasp_payload(
                    self.inserter_preinsertion_world_grasp,
                    candidate=self.inserter_candidate,
                ),
            },
            "collision_checks": {
                "offline_dual_pair": {
                    "artifact": str(self.artifact_dir / f"dual_grasp_pairs_{self.step_id}.json"),
                    "assembled_part_ids_before": list(self.assembled_part_ids_before),
                    "scope": (
                        "canonical end-effector geometry plus selected pair-conditioned transition corridor validation"
                    ),
                },
                "selected_transition": dict(
                    self.transition_symmetry.get(
                        "pair_collision_validation",
                        self.transition_symmetry.get("validation", {}),
                    )
                ),
                "inserter_pickup_floor": {
                    "status": "accepted",
                    "reason": self.pickup_floor_check_reason,
                    "gripper_collision_model": self.pickup_gripper_collision_model,
                    "floor_z_world_m": self.pickup_floor_z_world_m,
                    "floor_clearance_margin_m": self.pickup_floor_clearance_margin_m,
                    "contact_gap_m": self.pickup_contact_gap_m,
                    "contact_patch_lateral_offset_m": (self.inserter_candidate.contact_patch_lateral_offset_m),
                    "contact_patch_approach_offset_m": (self.inserter_candidate.contact_patch_approach_offset_m),
                },
            },
            "targets": {
                "holder_pregrasp": {
                    "position_world_m": list(self.holder_world_grasp.pregrasp_position_w),
                    "orientation_xyzw_world": list(self.holder_world_grasp.orientation_xyzw),
                },
                "holder_grasp": {
                    "position_world_m": list(self.holder_world_grasp.position_w),
                    "orientation_xyzw_world": list(self.holder_world_grasp.orientation_xyzw),
                },
                "inserter_pickup_pregrasp": {
                    "position_world_m": list(self.inserter_pickup_world_grasp.pregrasp_position_w),
                    "orientation_xyzw_world": list(self.inserter_pickup_world_grasp.orientation_xyzw),
                },
                "inserter_pickup_grasp": {
                    "position_world_m": list(pickup_grasp),
                    "orientation_xyzw_world": list(self.inserter_pickup_world_grasp.orientation_xyzw),
                },
                "inserter_pickup_lift": {
                    "position_world_m": list(pickup_grasp + lift_offset),
                    "orientation_xyzw_world": list(self.inserter_pickup_world_grasp.orientation_xyzw),
                },
                "inserter_above_preinsertion": {
                    "position_world_m": list(preinsertion_grasp + lift_offset),
                    "orientation_xyzw_world": list(self.inserter_preinsertion_world_grasp.orientation_xyzw),
                },
                "inserter_preinsertion": {
                    "position_world_m": list(preinsertion_grasp),
                    "orientation_xyzw_world": list(self.inserter_preinsertion_world_grasp.orientation_xyzw),
                },
            },
        }


def simple_dual_robot_pregrasp_aabb_obstacles(
    task: SimpleDualRobotPairTask,
    *,
    corridor_margin_m: float = DEFAULT_PREGRASP_AABB_CORRIDOR_MARGIN_M,
) -> dict[str, dict[str, object]]:
    """Build target-specific AABB pieces for collision-aware pregrasp transit.

    The object's full world AABB is retained except for the swept AABB of the
    selected detailed gripper model between pregrasp and grasp. This permits
    the intended approach without treating all empty space in one coarse AABB
    as occupied. The other object remains a full AABB for the holder.

    These pieces are intentionally not part of the grasp or transport scene:
    the selected grippers must contact their objects after reaching pregrasp.
    """

    margin = float(corridor_margin_m)
    if margin < 0.0:
        raise ValueError("corridor_margin_m must be non-negative.")

    task_payload = task.to_payload()
    objects = task_payload["objects"]
    assert isinstance(objects, dict)
    subassembly = objects["subassembly"]
    assert isinstance(subassembly, dict)
    subassembly_mesh_source = source_local_subassembly_mesh(subassembly)
    vertices_world = (
        subassembly_mesh_source.vertices_obj @ task.holder_source_pose_world.rotation_world_from_object.T
        + task.holder_source_pose_world.translation_world[None, :]
    )
    subassembly_bounds = RuntimePartAabb(
        role="subassembly",
        minimum_world_m=tuple(float(value) for value in np.min(vertices_world, axis=0)),
        maximum_world_m=tuple(float(value) for value in np.max(vertices_world, axis=0)),
    )
    incoming_bounds = source_mesh_aabb_world(
        role="incoming_pickup",
        mesh_assembly=load_triangle_mesh(
            task.incoming_mesh_path,
            scale=float(task.mesh_scale),
        ),
        source_pose_assembly=task.incoming_source_pose_assembly,
        source_pose_world=task.incoming_pickup_source_pose_world,
    )

    def _primitive_bounds(
        primitive: BoxCollisionPrimitive | MeshCollisionPrimitive,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(primitive, BoxCollisionPrimitive):
            return primitive.aabb_bounds_obj()
        vertices = np.asarray(primitive.vertices_obj, dtype=float)
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    def _selected_gripper_sweep_bounds(
        *,
        candidate: SavedGraspCandidate,
        source_pose_world: ObjectWorldPose,
        world_grasp: WorldFrameGraspCandidate,
        role: str,
    ) -> RuntimePartAabb:
        model = make_gripper_collision_model(
            task.pickup_gripper_collision_model,
            contact_gap_m=task.pickup_contact_gap_m,
            contact_patch_lateral_offset_m=(candidate.contact_patch_lateral_offset_m),
            contact_patch_approach_offset_m=(candidate.contact_patch_approach_offset_m),
        )
        candidate_obj = candidate.to_object_frame_candidate()
        primitives_obj = model.primitives_for_grasp(
            grasp_rotmat=quat_to_rotmat_xyzw(candidate.grasp_orientation_xyzw_obj),
            contact_point_a=np.asarray(
                candidate_obj.contact_point_a_obj,
                dtype=float,
            ),
            contact_point_b=np.asarray(
                candidate_obj.contact_point_b_obj,
                dtype=float,
            ),
            grasp_center=np.asarray(
                candidate_obj.grasp_position_obj,
                dtype=float,
            ),
        )
        pregrasp_translation = np.asarray(world_grasp.pregrasp_position_w, dtype=float) - np.asarray(
            world_grasp.position_w, dtype=float
        )
        minima: list[np.ndarray] = []
        maxima: list[np.ndarray] = []
        for primitive_obj in primitives_obj:
            primitive_world = transform_primitive_to_world(
                primitive_obj,
                source_pose_world,
            )
            minimum, maximum = _primitive_bounds(primitive_world)
            minima.extend((minimum, minimum + pregrasp_translation))
            maxima.extend((maximum, maximum + pregrasp_translation))
        if not minima:
            raise ValueError(f"The selected {role} gripper model produced no collision primitives.")
        corridor_minimum = np.min(np.stack(minima), axis=0) - margin
        corridor_maximum = np.max(np.stack(maxima), axis=0) + margin
        return RuntimePartAabb(
            role=role,
            minimum_world_m=tuple(float(value) for value in corridor_minimum),
            maximum_world_m=tuple(float(value) for value in corridor_maximum),
        )

    def _subtract_bounds(
        *,
        outer: RuntimePartAabb,
        cut: RuntimePartAabb,
    ) -> tuple[RuntimePartAabb, ...]:
        outer_minimum = np.asarray(outer.minimum_world_m, dtype=float)
        outer_maximum = np.asarray(outer.maximum_world_m, dtype=float)
        overlap_minimum = np.maximum(
            outer_minimum,
            np.asarray(cut.minimum_world_m, dtype=float),
        )
        overlap_maximum = np.minimum(
            outer_maximum,
            np.asarray(cut.maximum_world_m, dtype=float),
        )
        if np.any(overlap_maximum <= overlap_minimum):
            return (outer,)

        pieces: list[RuntimePartAabb] = []

        def _append(minimum: np.ndarray, maximum: np.ndarray) -> None:
            if np.any(maximum - minimum < _MIN_PREGRASP_AABB_PIECE_SIZE_M):
                return
            pieces.append(
                RuntimePartAabb(
                    role=outer.role,
                    minimum_world_m=tuple(float(value) for value in minimum),
                    maximum_world_m=tuple(float(value) for value in maximum),
                )
            )

        # Six non-overlapping slabs exactly cover outer minus the clipped cut.
        _append(
            outer_minimum,
            np.asarray(
                (
                    overlap_minimum[0],
                    outer_maximum[1],
                    outer_maximum[2],
                )
            ),
        )
        _append(
            np.asarray(
                (
                    overlap_maximum[0],
                    outer_minimum[1],
                    outer_minimum[2],
                )
            ),
            outer_maximum,
        )
        middle_x_minimum = np.asarray(
            (
                overlap_minimum[0],
                outer_minimum[1],
                outer_minimum[2],
            )
        )
        middle_x_maximum = np.asarray(
            (
                overlap_maximum[0],
                outer_maximum[1],
                outer_maximum[2],
            )
        )
        _append(
            middle_x_minimum,
            np.asarray(
                (
                    middle_x_maximum[0],
                    overlap_minimum[1],
                    middle_x_maximum[2],
                )
            ),
        )
        _append(
            np.asarray(
                (
                    middle_x_minimum[0],
                    overlap_maximum[1],
                    middle_x_minimum[2],
                )
            ),
            middle_x_maximum,
        )
        middle_xy_minimum = np.asarray(
            (
                overlap_minimum[0],
                overlap_minimum[1],
                outer_minimum[2],
            )
        )
        middle_xy_maximum = np.asarray(
            (
                overlap_maximum[0],
                overlap_maximum[1],
                outer_maximum[2],
            )
        )
        _append(
            middle_xy_minimum,
            np.asarray(
                (
                    middle_xy_maximum[0],
                    middle_xy_maximum[1],
                    overlap_minimum[2],
                )
            ),
        )
        _append(
            np.asarray(
                (
                    middle_xy_minimum[0],
                    middle_xy_minimum[1],
                    overlap_maximum[2],
                )
            ),
            middle_xy_maximum,
        )
        return tuple(pieces)

    def _obstacle(
        *,
        obstacle_id: str,
        bounds: RuntimePartAabb,
        active_target: str,
        source: str,
        carved_for_grasp_id: str | None = None,
        carved_sweep_bounds: RuntimePartAabb | None = None,
    ) -> dict[str, object]:
        minimum = np.asarray(bounds.minimum_world_m, dtype=float)
        maximum = np.asarray(bounds.maximum_world_m, dtype=float)
        size = maximum - minimum
        if np.any(size <= 0.0):
            raise ValueError(f"Cannot create AABB obstacle '{obstacle_id}' with non-positive size {size.tolist()}.")
        obstacle = {
            "id": obstacle_id,
            "type": "box",
            "frame_id": "base_link",
            "size_m": [float(value) for value in size],
            "xyz": [float(value) for value in 0.5 * (minimum + maximum)],
            "rpy": [0.0, 0.0, 0.0],
            "source": source,
            "role": bounds.role,
            "active_target": active_target,
            "carved_for_grasp_id": carved_for_grasp_id,
            "corridor_margin_m": margin,
        }
        if carved_sweep_bounds is not None:
            obstacle["carved_sweep_aabb"] = {
                "minimum_world_m": list(carved_sweep_bounds.minimum_world_m),
                "maximum_world_m": list(carved_sweep_bounds.maximum_world_m),
            }
        return obstacle

    holder_corridor = _selected_gripper_sweep_bounds(
        candidate=task.holder_candidate,
        source_pose_world=task.holder_source_pose_world,
        world_grasp=task.holder_world_grasp,
        role="holder_gripper_sweep",
    )
    inserter_corridor = _selected_gripper_sweep_bounds(
        candidate=task.inserter_candidate,
        source_pose_world=task.incoming_pickup_source_pose_world,
        world_grasp=task.inserter_pickup_world_grasp,
        role="inserter_gripper_sweep",
    )

    obstacles: dict[str, dict[str, object]] = {}

    def _add_pieces(
        *,
        key_prefix: str,
        obstacle_id_prefix: str,
        bounds: Iterable[RuntimePartAabb],
        active_target: str,
        source: str,
        carved_for_grasp_id: str | None,
        carved_sweep_bounds: RuntimePartAabb | None = None,
    ) -> None:
        for index, piece in enumerate(bounds):
            key = f"{key_prefix}_{index:02d}"
            obstacles[key] = _obstacle(
                obstacle_id=f"{obstacle_id_prefix}_{index:02d}",
                bounds=piece,
                active_target=active_target,
                source=source,
                carved_for_grasp_id=carved_for_grasp_id,
                carved_sweep_bounds=carved_sweep_bounds,
            )

    _add_pieces(
        key_prefix="holder_pregrasp_subassembly",
        obstacle_id_prefix="dual_holder_pregrasp_subassembly_aabb",
        bounds=_subtract_bounds(
            outer=subassembly_bounds,
            cut=holder_corridor,
        ),
        active_target="holder_pregrasp",
        source="world_aabb_minus_selected_gripper_sweep",
        carved_for_grasp_id=task.holder_candidate.grasp_id,
        carved_sweep_bounds=holder_corridor,
    )
    # The holder has no intended contact with the incoming pickup object.
    _add_pieces(
        key_prefix="holder_pregrasp_incoming_pickup",
        obstacle_id_prefix="dual_holder_pregrasp_incoming_pickup_aabb",
        bounds=(incoming_bounds,),
        active_target="holder_pregrasp",
        source="world_aabb",
        carved_for_grasp_id=None,
    )
    _add_pieces(
        key_prefix="inserter_pickup_pregrasp_incoming_pickup",
        obstacle_id_prefix=("dual_inserter_pickup_pregrasp_incoming_pickup_aabb"),
        bounds=_subtract_bounds(
            outer=incoming_bounds,
            cut=inserter_corridor,
        ),
        active_target="inserter_pickup_pregrasp",
        source="world_aabb_minus_selected_gripper_sweep",
        carved_for_grasp_id=task.inserter_candidate.grasp_id,
        carved_sweep_bounds=inserter_corridor,
    )
    return obstacles


def simple_dual_robot_pregrasp_aabb_schedule(
    obstacles: Mapping[str, Mapping[str, object]],
) -> dict[str, list[str]]:
    """Return the target-to-piece schedule embedded in MoveIt task artifacts."""

    schedule = {
        "holder_pregrasp": [],
        "inserter_pickup_pregrasp": [],
    }
    for key, obstacle in obstacles.items():
        active_target = str(obstacle.get("active_target", ""))
        if active_target in schedule:
            schedule[active_target].append(str(key))
    return schedule


def load_simple_dual_robot_pair_tasks(
    *,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    step_id: str = DEFAULT_STEP_ID,
    assembly_world: MovableFrame = DEFAULT_ASSEMBLY_WORLD,
    pickup_source_world_xy: tuple[float, float] = (DEFAULT_PICKUP_SOURCE_WORLD_XY),
    pickup_orientation_rpy_deg: tuple[float, float, float] = (DEFAULT_PICKUP_ORIENTATION_RPY_DEG),
    holder_pregrasp_offset_m: float = DEFAULT_HOLDER_PREGRASP_OFFSET_M,
    inserter_pregrasp_offset_m: float = (DEFAULT_INSERTER_PREGRASP_OFFSET_M),
    transport_clearance_m: float = DEFAULT_TRANSPORT_CLEARANCE_M,
    pickup_floor_z_world_m: float = DEFAULT_FLOOR_Z_WORLD_M,
    pickup_floor_clearance_margin_m: float = (DEFAULT_PICKUP_FLOOR_CLEARANCE_MARGIN_M),
    pickup_contact_gap_m: float = DEFAULT_PICKUP_CONTACT_GAP_M,
    pickup_top_down_score_weight: float = (DEFAULT_PICKUP_TOP_DOWN_SCORE_WEIGHT),
    transition_score_weight: float = DEFAULT_TRANSITION_SCORE_WEIGHT,
    transition_translation_scale_m: float = (DEFAULT_TRANSITION_TRANSLATION_SCALE_M),
    transition_translation_weight: float = (DEFAULT_TRANSITION_TRANSLATION_WEIGHT),
    transition_rotation_weight: float = DEFAULT_TRANSITION_ROTATION_WEIGHT,
    reachability_proxy_config: ReachabilityProxyConfig = (ReachabilityProxyConfig()),
    retained_only: bool = True,
    include_nonretained_identity_fallbacks: bool = False,
) -> tuple[SimpleDualRobotPairTask, ...]:
    """Resolve ranked Stage-3 pairs for the first holder/inserter simulation."""

    if retained_only and include_nonretained_identity_fallbacks:
        raise ValueError("retained_only and include_nonretained_identity_fallbacks are mutually exclusive.")

    root = Path(artifact_dir).expanduser().resolve()
    pair_path = root / f"dual_grasp_pairs_{step_id}.json"
    pair_payload = _read_json(pair_path)
    if str(pair_payload["step_id"]) != step_id:
        raise ValueError(f"Pair artifact step mismatch in '{pair_path}'.")
    accepted_offline_evaluations = [
        dict(raw_evaluation)
        for raw_evaluation in pair_payload.get("evaluations", [])  # type: ignore[arg-type]
        if isinstance(raw_evaluation, dict) and raw_evaluation.get("status") == "accepted"
    ]
    if not accepted_offline_evaluations:
        metadata = dict(pair_payload.get("metadata", {}))
        raise ValueError(
            f"No offline-compatible holder/inserter grasp pair exists in "
            f"'{pair_path}'. Stage-3 shortlisted "
            f"{metadata.get('holder_shortlist_count', 0)} holder and "
            f"{metadata.get('inserter_shortlist_count', 0)} inserter grasps; "
            f"reason counts={pair_payload.get('reason_counts', {})}."
        )

    holder_bundle = load_grasp_bundle(root / "holder_base_candidates.json")
    inserter_bundle = load_grasp_bundle(root / f"inserter_candidates_{step_id}.json")
    base_mesh_path = resolve_mesh_path(holder_bundle.target_stl_path)
    incoming_mesh_path = resolve_mesh_path(inserter_bundle.target_stl_path)
    holder_bundle_source_pose_assembly = _source_pose_from_bundle(holder_bundle)
    (
        holder_candidates,
        holder_source_pose_assembly,
        holder_candidate_source,
    ) = _declared_holder_candidate_source(
        root=root,
        pair_payload=pair_payload,
        fallback_candidates=holder_bundle.candidates,
        fallback_source_pose_assembly=holder_bundle_source_pose_assembly,
    )
    inserter_source_pose_assembly = _source_pose_from_bundle(inserter_bundle)
    holder_source_pose_world = compose_source_pose_world(
        source_pose_assembly=holder_source_pose_assembly,
        assembly_world=assembly_world,
    )
    incoming_final_source_pose_world = compose_source_pose_world(
        source_pose_assembly=inserter_source_pose_assembly,
        assembly_world=assembly_world,
    )
    final_to_pre = tuple(
        float(value)
        for value in dict(pair_payload["motion"])["insertion_translation_start_m"]  # type: ignore[index]
    )
    transition_payloads = _transition_payloads(
        pair_payload,
        source_pose_assembly=inserter_source_pose_assembly,
        final_to_pre_translation_assembly_m=final_to_pre,
    )
    pickup_rotation_world = (
        _rpy_rotation(*pickup_orientation_rpy_deg) @ incoming_final_source_pose_world.rotation_world_from_object
    )
    pickup_orientation_world = ObjectWorldPose(
        position_world=(0.0, 0.0, 0.0),
        orientation_xyzw_world=tuple(float(value) for value in rotmat_to_quat_xyzw(pickup_rotation_world)),
    )
    incoming_pickup_source_pose_world = source_pose_resting_on_floor(
        mesh_assembly=load_triangle_mesh(
            incoming_mesh_path,
            scale=float(inserter_bundle.stl_scale),
        ),
        source_pose_assembly=inserter_source_pose_assembly,
        source_orientation_world=pickup_orientation_world,
        xy_world=pickup_source_world_xy,
        floor_z_world_m=float(pickup_floor_z_world_m),
    )
    pickup_gripper_collision_model = str(
        inserter_bundle.metadata.get(
            "gripper_collision_model",
            inserter_bundle.metadata.get("gripper_model", "kuka_y_gripper"),
        )
    )
    contact_lateral_offsets_m = tuple(
        float(value)
        for value in inserter_bundle.metadata.get(
            "contact_lateral_offsets_m",
            (-0.002916666666666667, 0.0, 0.002916666666666667),
        )
    )
    contact_approach_offsets_m = tuple(
        float(value)
        for value in inserter_bundle.metadata.get(
            "contact_approach_offsets_m",
            (-0.0030833333333333333, 0.0, 0.0030833333333333333),
        )
    )
    pickup_floor_statuses = evaluate_saved_grasps_against_pickup_pose(
        inserter_bundle.candidates,
        object_pose_world=incoming_pickup_source_pose_world,
        contact_gap_m=float(pickup_contact_gap_m),
        gripper_collision_model=pickup_gripper_collision_model,
        floor_z_world_m=float(pickup_floor_z_world_m),
        floor_clearance_margin_m=float(pickup_floor_clearance_margin_m),
        contact_lateral_offsets_m=contact_lateral_offsets_m,
        contact_approach_offsets_m=contact_approach_offsets_m,
    )
    pickup_floor_accepted = {
        status.grasp.grasp_id: status for status in pickup_floor_statuses if status.status == "accepted"
    }

    retained_ids = set(
        str(value)
        for value in pair_payload["retained_pair_ids"]  # type: ignore[index]
    )
    raw_retained_execution_ids = pair_payload.get(
        "retained_execution_candidate_ids",
        (),
    )
    retained_execution_ids = (
        {str(value) for value in raw_retained_execution_ids} if isinstance(raw_retained_execution_ids, list) else set()
    )
    if not 0.0 <= float(pickup_top_down_score_weight) <= 1.0:
        raise ValueError("pickup_top_down_score_weight must be between 0 and 1.")
    if not 0.0 <= float(transition_score_weight) <= 1.0:
        raise ValueError("transition_score_weight must be between 0 and 1.")
    if float(transition_translation_scale_m) <= 0.0:
        raise ValueError("transition_translation_scale_m must be > 0.")
    if (
        float(transition_translation_weight) < 0.0
        or float(transition_rotation_weight) < 0.0
        or float(transition_translation_weight) + float(transition_rotation_weight) <= 0.0
    ):
        raise ValueError("Transition translation/rotation weights must be non-negative with a positive sum.")
    evaluations: list[dict[str, object]] = []
    for raw_evaluation in pair_payload["evaluations"]:  # type: ignore[index]
        evaluation = dict(raw_evaluation)
        inserter_grasp_id = str(evaluation.get("inserter_grasp_id"))
        if (
            evaluation.get("status") != "accepted"
            or inserter_grasp_id not in pickup_floor_accepted
            or (retained_only and str(evaluation.get("pair_id")) not in retained_ids)
        ):
            continue
        evaluations.append(evaluation)
    sequence_payload = _read_json(root / "assembly_sequence.json")
    raw_steps = sequence_payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("Assembly sequence is missing its steps list.")
    matching_steps = [
        dict(raw_step)
        for raw_step in raw_steps
        if isinstance(raw_step, dict) and str(raw_step.get("step_id", "")) == step_id
    ]
    if len(matching_steps) != 1:
        raise ValueError(f"Could not resolve exactly one sequence state for '{step_id}'.")
    sequence_step = matching_steps[0]
    assembled_part_ids_before = tuple(str(value) for value in sequence_step.get("assembled_part_ids_before", []))
    if not assembled_part_ids_before:
        raise ValueError(f"Step '{step_id}' has no assembled prefix for the holder.")
    if str(sequence_payload["base_part_id"]) not in assembled_part_ids_before:
        raise ValueError(
            f"Step '{step_id}' assembled prefix does not contain base part '{sequence_payload['base_part_id']}'."
        )
    raw_parts = sequence_payload.get("parts")
    if not isinstance(raw_parts, dict):
        raise ValueError("Assembly sequence is missing its parts mapping.")
    subassembly_parts = []
    for part_id in assembled_part_ids_before:
        raw_part = raw_parts.get(part_id)
        if not isinstance(raw_part, dict) or not raw_part.get("mesh_path"):
            raise ValueError(f"Assembly sequence is missing mesh metadata for part '{part_id}'.")
        mesh_path = _repo_path(str(raw_part["mesh_path"]))
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Assembled-prefix mesh does not exist: {mesh_path}")
        subassembly_parts.append(
            SimpleDualRobotSubassemblyPart(
                part_id=part_id,
                mesh_path=mesh_path,
            )
        )

    def target(
        name: str,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> TaskTargetPose:
        return TaskTargetPose(
            name=name,
            position_world_m=position,
            orientation_xyzw_world=orientation,
        )

    tasks: list[SimpleDualRobotPairTask] = []
    lift = np.asarray((0.0, 0.0, float(transport_clearance_m)))
    for evaluation in evaluations:
        holder_candidate = _candidate_by_id(
            holder_candidates,
            str(evaluation["holder_grasp_id"]),
        )
        pickup_floor_status = pickup_floor_accepted[str(evaluation["inserter_grasp_id"])]
        inserter_candidate = pickup_floor_status.grasp
        holder_world_grasp = saved_grasp_to_world_grasp(
            holder_candidate,
            holder_source_pose_world,
            pregrasp_offset=holder_pregrasp_offset_m,
            gripper_width_clearance=0.0,
        )
        pickup_world_grasp = saved_grasp_to_world_grasp(
            inserter_candidate,
            incoming_pickup_source_pose_world,
            pregrasp_offset=inserter_pregrasp_offset_m,
            gripper_width_clearance=0.0,
        )
        pickup_top_down_score = max(
            0.0,
            min(1.0, -float(pickup_world_grasp.normal_w[2])),
        )
        holder_targets = (
            target(
                "holder_pregrasp",
                holder_world_grasp.pregrasp_position_w,
                holder_world_grasp.orientation_xyzw,
            ),
            target(
                "holder_grasp",
                holder_world_grasp.position_w,
                holder_world_grasp.orientation_xyzw,
            ),
        )
        evaluation_details = evaluation.get("details")
        details = dict(evaluation_details) if isinstance(evaluation_details, dict) else {}
        raw_compatible_transition_ids = details.get("compatible_transition_ids")
        if isinstance(raw_compatible_transition_ids, list):
            compatible_transition_ids = {str(value) for value in raw_compatible_transition_ids}
            evaluation_transitions = tuple(
                transition
                for transition in transition_payloads
                if str(transition.get("transition_id", "")) in compatible_transition_ids
            )
        else:
            # Schema-1/legacy artifacts only contain the canonical corridor.
            evaluation_transitions = transition_payloads
        raw_transition_validation = details.get("transition_validation")
        transition_validation = dict(raw_transition_validation) if isinstance(raw_transition_validation, dict) else {}
        for transition in evaluation_transitions:
            transition_id = str(transition.get("transition_id", "tr_identity"))
            pair_id = str(evaluation["pair_id"])
            execution_candidate_id = f"{pair_id}__{transition_id}"
            execution_is_retained = (
                execution_candidate_id in retained_execution_ids if retained_execution_ids else pair_id in retained_ids
            )
            if (
                include_nonretained_identity_fallbacks
                and not execution_is_retained
                and not bool(transition.get("is_identity", False))
            ):
                # A transformed corridor may exceed the final execution cap
                # even though Stage 3 explicitly validated it for a retained
                # pair. Keep that useful fallback, but never extrapolate a
                # nonidentity transform onto an identity-only non-retained
                # pair or past a rejected/missing validation record.
                validation = dict(transition_validation.get(transition_id, {}))
                if pair_id not in retained_ids or validation.get("status") != "accepted":
                    continue
            if retained_only and retained_execution_ids and execution_candidate_id not in retained_execution_ids:
                continue
            raw_final_pose = transition.get("final_source_pose_assembly")
            raw_pre_pose = transition.get("preinsertion_source_pose_assembly")
            if not isinstance(raw_final_pose, dict) or not isinstance(
                raw_pre_pose,
                dict,
            ):
                raise ValueError("Transition candidate is missing final/pre-insertion source poses.")
            final_source_pose_assembly = _pose_from_matrix_payload(raw_final_pose)
            pre_source_pose_assembly = _pose_from_matrix_payload(raw_pre_pose)
            final_source_pose_world = compose_source_pose_world(
                source_pose_assembly=final_source_pose_assembly,
                assembly_world=assembly_world,
            )
            pre_source_pose_world = compose_source_pose_world(
                source_pose_assembly=pre_source_pose_assembly,
                assembly_world=assembly_world,
            )
            preinsertion_world_grasp = saved_grasp_to_world_grasp(
                inserter_candidate,
                pre_source_pose_world,
                pregrasp_offset=inserter_pregrasp_offset_m,
                gripper_width_clearance=0.0,
            )
            inserter_targets = (
                target(
                    "inserter_pickup_pregrasp",
                    pickup_world_grasp.pregrasp_position_w,
                    pickup_world_grasp.orientation_xyzw,
                ),
                target(
                    "inserter_pickup_grasp",
                    pickup_world_grasp.position_w,
                    pickup_world_grasp.orientation_xyzw,
                ),
                target(
                    "inserter_pickup_lift",
                    tuple(float(value) for value in np.asarray(pickup_world_grasp.position_w) + lift),
                    pickup_world_grasp.orientation_xyzw,
                ),
                target(
                    "inserter_above_preinsertion",
                    tuple(float(value) for value in np.asarray(preinsertion_world_grasp.position_w) + lift),
                    preinsertion_world_grasp.orientation_xyzw,
                ),
                target(
                    "inserter_preinsertion",
                    preinsertion_world_grasp.position_w,
                    preinsertion_world_grasp.orientation_xyzw,
                ),
            )
            pair_score = float(evaluation["score"])
            layout_proxy = pair_layout_score(
                offline_pair_score=pair_score,
                holder_targets=holder_targets,
                inserter_targets=inserter_targets,
                holder_grasp_target=holder_targets[-1],
                inserter_grasp_target=inserter_targets[1],
                inserter_transition_target=inserter_targets[-1],
                holder_robot_base_world=DEFAULT_HOLDER_BASE_WORLD,
                inserter_robot_base_world=DEFAULT_INSERTER_BASE_WORLD,
                config=reachability_proxy_config,
            )
            transition_motion = _transition_motion_components(
                pickup_world_grasp,
                preinsertion_world_grasp,
                translation_scale_m=transition_translation_scale_m,
                translation_weight=transition_translation_weight,
                rotation_weight=transition_rotation_weight,
            )
            pickup_layout_score = (1.0 - float(pickup_top_down_score_weight)) * float(layout_proxy["score"]) + float(
                pickup_top_down_score_weight
            ) * pickup_top_down_score
            selection_score = (1.0 - float(transition_score_weight)) * pickup_layout_score + float(
                transition_score_weight
            ) * float(transition_motion["score"])
            selected_transition = {
                **transition,
                "pair_collision_validation": dict(transition_validation.get(transition_id, {})),
            }
            final_to_pre_candidate = tuple(
                float(value)
                for value in (pre_source_pose_assembly.translation_world - final_source_pose_assembly.translation_world)
            )
            tasks.append(
                SimpleDualRobotPairTask(
                    artifact_dir=root,
                    step_id=step_id,
                    step_index=int(pair_payload["step_index"]),
                    assembly=str(sequence_payload["assembly"]),
                    base_part_id=str(sequence_payload["base_part_id"]),
                    incoming_part_id=str(pair_payload["incoming_part_id"]),
                    pair_id=pair_id,
                    transition_id=transition_id,
                    execution_candidate_id=execution_candidate_id,
                    pair_score=pair_score,
                    selection_score=float(selection_score),
                    transition_motion_score=float(transition_motion["score"]),
                    transition_motion_components=transition_motion,
                    pickup_top_down_score=pickup_top_down_score,
                    layout_proxy_score=float(layout_proxy["score"]),
                    layout_proxy_components={
                        "score_before_crossing_penalty": float(layout_proxy["score_before_crossing_penalty"]),
                        "crossing_penalty": float(layout_proxy["crossing_penalty"]),
                        "crossing_penalty_applied": float(layout_proxy["crossing_penalty_applied"]),
                        "pickup_segments_cross_xy": bool(layout_proxy["pickup_segments_cross_xy"]),
                        "transition_segments_cross_xy": bool(layout_proxy["transition_segments_cross_xy"]),
                    },
                    holder_reachability_proxy_score=float(dict(layout_proxy["holder"])["score"]),
                    inserter_reachability_proxy_score=float(dict(layout_proxy["inserter"])["score"]),
                    holder_candidate=holder_candidate,
                    inserter_candidate=inserter_candidate,
                    holder_source_pose_assembly=(holder_source_pose_assembly),
                    incoming_source_pose_assembly=(inserter_source_pose_assembly),
                    holder_source_pose_world=holder_source_pose_world,
                    incoming_pickup_source_pose_world=(incoming_pickup_source_pose_world),
                    incoming_final_source_pose_world=(final_source_pose_world),
                    incoming_preinsertion_source_pose_world=(pre_source_pose_world),
                    holder_world_grasp=holder_world_grasp,
                    inserter_pickup_world_grasp=pickup_world_grasp,
                    inserter_preinsertion_world_grasp=(preinsertion_world_grasp),
                    base_mesh_path=base_mesh_path,
                    incoming_mesh_path=incoming_mesh_path,
                    assembled_part_ids_before=assembled_part_ids_before,
                    subassembly_parts=tuple(subassembly_parts),
                    mesh_scale=float(holder_bundle.stl_scale),
                    assembly_world=assembly_world,
                    final_to_preinsertion_translation_assembly_m=(final_to_pre_candidate),
                    transport_clearance_m=float(transport_clearance_m),
                    pickup_floor_z_world_m=float(pickup_floor_z_world_m),
                    pickup_floor_check_reason=pickup_floor_status.reason,
                    pickup_floor_clearance_margin_m=float(pickup_floor_clearance_margin_m),
                    pickup_contact_gap_m=float(pickup_contact_gap_m),
                    pickup_gripper_collision_model=(pickup_gripper_collision_model),
                    transition_symmetry=selected_transition,
                )
            )

    def is_retained_execution(task: SimpleDualRobotPairTask) -> bool:
        if retained_execution_ids:
            return task.execution_candidate_id in retained_execution_ids
        return task.pair_id in retained_ids

    tasks.sort(
        key=lambda task: (
            bool(task.layout_proxy_components.get("transition_segments_cross_xy", False)),
            (0 if include_nonretained_identity_fallbacks and is_retained_execution(task) else 1),
            -task.selection_score,
            -task.pair_score,
            task.execution_candidate_id,
        )
    )
    if not tasks:
        rejection_counts = Counter(status.reason for status in pickup_floor_statuses if status.status != "accepted")
        reason_summary = ", ".join(f"{reason}={count}" for reason, count in sorted(rejection_counts.items()))
        accepted_pair_evaluations = accepted_offline_evaluations
        unique_pair_inserters = {
            str(dict(evaluation).get("inserter_grasp_id")) for evaluation in accepted_pair_evaluations
        }
        raise ValueError(
            "No compatible grasp pairs remain after checking the grounded "
            f"pickup pose in '{pair_path}'. "
            f"Pickup candidates accepted={len(pickup_floor_accepted)}/"
            f"{len(pickup_floor_statuses)}; rejection reasons: "
            f"{reason_summary or 'none'}. Offline-compatible pair "
            f"evaluations={len(accepted_pair_evaluations)} using "
            f"{len(unique_pair_inserters)} unique inserter grasps."
        )
    pickup_rejection_counts = Counter(status.reason for status in pickup_floor_statuses if status.status != "accepted")
    candidate_filter_diagnostics: dict[str, object] = {
        "pickup_grasps_checked": len(pickup_floor_statuses),
        "pickup_grasps_accepted": len(pickup_floor_accepted),
        "pickup_grasps_rejected": len(pickup_floor_statuses) - len(pickup_floor_accepted),
        "pickup_grasp_rejection_counts": dict(sorted(pickup_rejection_counts.items())),
        "offline_compatible_pairs": len(accepted_offline_evaluations),
        "stage3_retained_pairs": len(retained_ids),
        "stage3_retained_execution_candidates": (
            len(retained_execution_ids) if retained_execution_ids else len(retained_ids)
        ),
        "pose_feasible_retained_execution_candidates": sum(is_retained_execution(task) for task in tasks),
        "pose_feasible_identity_fallback_candidates": sum(
            not is_retained_execution(task) and bool(task.transition_symmetry.get("is_identity", False))
            for task in tasks
        ),
        "pose_feasible_validated_transition_fallback_candidates": sum(
            not is_retained_execution(task) and not bool(task.transition_symmetry.get("is_identity", False))
            for task in tasks
        ),
        "pose_feasible_pair_evaluations": len(evaluations),
        "pose_feasible_execution_candidates": len(tasks),
        "pose_feasible_unique_holder_grasps": len({task.holder_candidate.grasp_id for task in tasks}),
        "pose_feasible_unique_inserter_grasps": len({task.inserter_candidate.grasp_id for task in tasks}),
        "holder_candidate_source": holder_candidate_source,
    }
    tasks = [
        replace(
            task,
            candidate_filter_diagnostics=candidate_filter_diagnostics,
        )
        for task in tasks
    ]
    return tuple(tasks)


__all__ = [
    "DEFAULT_ARTIFACT_DIR",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_ASSEMBLY_NAME",
    "DEFAULT_ASSEMBLY_WORLD",
    "DEFAULT_HOLDER_PREGRASP_OFFSET_M",
    "DEFAULT_HOLDER_BASE_WORLD",
    "DEFAULT_INSERTER_BASE_WORLD",
    "DEFAULT_INSERTER_PREGRASP_OFFSET_M",
    "DEFAULT_INCOMING_PART_ID",
    "DEFAULT_PICKUP_SOURCE_WORLD_XY",
    "DEFAULT_PICKUP_CONTACT_GAP_M",
    "DEFAULT_PICKUP_FLOOR_CLEARANCE_MARGIN_M",
    "DEFAULT_PICKUP_TOP_DOWN_SCORE_WEIGHT",
    "DEFAULT_PREGRASP_AABB_CORRIDOR_MARGIN_M",
    "DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT",
    "DEFAULT_STEP_ID",
    "DEFAULT_TRANSPORT_CLEARANCE_M",
    "PlanarRuntimeLayout",
    "RuntimePartAabb",
    "DualRobotStepSelection",
    "SIMPLE_SIM_SCHEMA_VERSION",
    "SimpleDualRobotPairTask",
    "SimpleDualRobotSubassemblyPart",
    "compose_source_pose_world",
    "load_simple_dual_robot_pair_tasks",
    "resolve_dual_robot_step_selection",
    "resolve_planar_runtime_layout",
    "simple_dual_robot_pregrasp_aabb_obstacles",
    "simple_dual_robot_pregrasp_aabb_schedule",
    "source_local_subassembly_mesh",
    "source_pose_resting_on_floor",
    "source_mesh_aabb_world",
    "translated_source_pose_world",
]
