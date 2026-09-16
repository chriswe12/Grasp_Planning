"""Compile one selected Fabrica assembly order into explicit planning states."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grasp_planning.grasping.mesh_io import load_triangle_mesh

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = REPO_ROOT / "assets" / "obj" / "fabrica"
_TRANSFORM_ATOL = 1.0e-9
_VECTOR_ATOL_M = 1.0e-6


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required assembly asset does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level JSON object in '{path}'.")
    return payload


def _display_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _asset_record(path: Path, *, repo_root: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": _display_path(path, repo_root=repo_root),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _matrix4(raw: object, *, field_name: str) -> np.ndarray:
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a numeric 4x4 matrix.") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} must be a finite numeric 4x4 matrix.")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=_TRANSFORM_ATOL):
        raise ValueError(f"{field_name} must use the homogeneous bottom row [0, 0, 0, 1].")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_TRANSFORM_ATOL) or not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        abs_tol=_TRANSFORM_ATOL,
    ):
        raise ValueError(f"{field_name} contains an invalid rigid rotation.")
    return matrix


def _tuple3(raw: object, *, field_name: str) -> tuple[float, float, float]:
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain three numeric values.") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain three finite numeric values.")
    return tuple(float(value) for value in array)


@dataclass(frozen=True)
class AssemblyPartSpec:
    part_id: str
    mesh_path: str
    role: str
    bounds_min_assembly_m: tuple[float, float, float]
    bounds_max_assembly_m: tuple[float, float, float]
    table_clearance_m: float
    touches_table: bool
    vertex_count: int
    face_count: int
    asset_record: dict[str, object]
    resolved_mesh_path: Path = field(repr=False, compare=False)

    def to_payload(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "mesh_path": self.mesh_path,
            "role": self.role,
            "bounds_assembly_m": {
                "min": list(self.bounds_min_assembly_m),
                "max": list(self.bounds_max_assembly_m),
            },
            "table_clearance_m": self.table_clearance_m,
            "touches_table": self.touches_table,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "asset_record": self.asset_record,
        }


@dataclass(frozen=True)
class AssemblySequenceStep:
    step_id: str
    step_index: int
    incoming_part_id: str
    incoming_part_role: str
    assembled_part_ids_before: tuple[str, ...]
    assembled_part_ids_after: tuple[str, ...]
    base_part_status: str
    holder_base_available: bool
    final_to_pre_insertion_transform_m: tuple[tuple[float, float, float, float], ...]
    final_to_pre_insertion_translation_m: tuple[float, float, float]
    pre_to_final_insertion_vector_m: tuple[float, float, float] | None
    insertion_distance_m: float
    disassembly_path_waypoints: int | None

    def to_payload(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "step_index": self.step_index,
            "incoming_part_id": self.incoming_part_id,
            "incoming_part_role": self.incoming_part_role,
            "assembled_part_ids_before": list(self.assembled_part_ids_before),
            "assembled_part_ids_after": list(self.assembled_part_ids_after),
            "base_part_status": self.base_part_status,
            "holder_base_available": self.holder_base_available,
            "final_to_pre_insertion_transform_m": [list(row) for row in self.final_to_pre_insertion_transform_m],
            "final_to_pre_insertion_translation_m": list(self.final_to_pre_insertion_translation_m),
            "pre_to_final_insertion_vector_m": (
                None if self.pre_to_final_insertion_vector_m is None else list(self.pre_to_final_insertion_vector_m)
            ),
            "insertion_distance_m": self.insertion_distance_m,
            "disassembly_path_waypoints": self.disassembly_path_waypoints,
        }


@dataclass(frozen=True)
class AssemblySequence:
    assembly: str
    base_part_id: str
    base_part_source: str
    base_part_order_index: int
    first_holder_step_index: int | None
    selected_order: tuple[str, ...]
    mesh_scale: float
    table_z_assembly_m: float
    table_contact_tolerance_m: float
    table_contact_part_ids: tuple[str, ...]
    parts: tuple[AssemblyPartSpec, ...]
    steps: tuple[AssemblySequenceStep, ...]
    precedence_plan_record: dict[str, object]
    pre_insertion_poses_record: dict[str, object]
    warnings: tuple[str, ...]
    source_assembly_dir: Path = field(repr=False, compare=False)

    @property
    def parts_by_id(self) -> dict[str, AssemblyPartSpec]:
        return {part.part_id: part for part in self.parts}

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "fabrica_assembly_sequence",
            "generated_by": "scripts/build_assembly_sequence.py",
            "assembly": self.assembly,
            "base_part_id": self.base_part_id,
            "base_part_source": self.base_part_source,
            "base_part_order_index": self.base_part_order_index,
            "first_holder_step_index": self.first_holder_step_index,
            "selected_order_source": "forward_assembly_orders[0]",
            "selected_order": list(self.selected_order),
            "mesh_scale": self.mesh_scale,
            "frame_contract": {
                "assembly_frame": "Fabrica OBJ assembled coordinates scaled by mesh_scale",
                "part_final_pose": "OBJ vertices are already authored in final assembled coordinates",
                "pre_insertion_pose": "final assembled OBJ vertices transformed by final_to_pre_insertion_transform_m",
                "table_frame": "assembly asset frame",
            },
            "table": {
                "z_assembly_m": self.table_z_assembly_m,
                "contact_tolerance_m": self.table_contact_tolerance_m,
                "contact_part_ids": list(self.table_contact_part_ids),
            },
            "source_assets": {
                "precedence_plan": self.precedence_plan_record,
                "pre_insertion_poses": self.pre_insertion_poses_record,
            },
            "parts": {part.part_id: part.to_payload() for part in self.parts},
            "steps": [step.to_payload() for step in self.steps],
            "warnings": list(self.warnings),
        }


def compile_assembly_sequence(
    assembly_dir: str | Path,
    *,
    base_part_id: str | int | None = None,
    mesh_scale: float = 0.01,
    table_z_assembly_m: float = 0.0,
    table_contact_tolerance_m: float = 1.0e-6,
    repo_root: str | Path = REPO_ROOT,
) -> AssemblySequence:
    """Compile the first selected order, using its first part as the default base."""

    if mesh_scale <= 0.0:
        raise ValueError("mesh_scale must be > 0.")
    if table_contact_tolerance_m < 0.0:
        raise ValueError("table_contact_tolerance_m must be >= 0.")

    repo_root_path = Path(repo_root).expanduser().resolve()
    assembly_path = Path(assembly_dir).expanduser()
    if not assembly_path.is_absolute():
        assembly_path = (repo_root_path / assembly_path).resolve()
    else:
        assembly_path = assembly_path.resolve()
    if not assembly_path.is_dir():
        raise FileNotFoundError(f"Assembly directory does not exist: {assembly_path}")

    precedence_path = assembly_path / "precedence_plan.json"
    poses_path = assembly_path / "pre_insertion_poses.json"
    precedence = _read_mapping(precedence_path)
    pre_insertion = _read_mapping(poses_path)

    assembly_name = str(precedence.get("assembly", assembly_path.name))
    if assembly_name != assembly_path.name:
        raise ValueError(f"Precedence assembly name '{assembly_name}' does not match directory '{assembly_path.name}'.")
    poses_assembly = str(pre_insertion.get("assembly", assembly_name))
    if poses_assembly != assembly_name:
        raise ValueError(
            f"Pre-insertion assembly name '{poses_assembly}' does not match precedence assembly '{assembly_name}'."
        )

    raw_orders = precedence.get("forward_assembly_orders")
    if not isinstance(raw_orders, list) or not raw_orders or not isinstance(raw_orders[0], list):
        raise ValueError(f"'{precedence_path}' does not contain a non-empty forward_assembly_orders list.")
    selected_order = tuple(str(part_id) for part_id in raw_orders[0])
    if not selected_order:
        raise ValueError(f"The selected order in '{precedence_path}' is empty.")
    if len(set(selected_order)) != len(selected_order):
        raise ValueError(f"The selected order in '{precedence_path}' contains duplicate part IDs.")

    base_id = selected_order[0] if base_part_id is None else str(base_part_id)
    base_part_source = "selected_order[0]" if base_part_id is None else "explicit_override"
    if base_id not in selected_order:
        raise ValueError(f"Base part '{base_id}' is not in selected order {list(selected_order)}.")
    base_order_index = selected_order.index(base_id)

    raw_parts = pre_insertion.get("parts")
    if not isinstance(raw_parts, dict):
        raise ValueError(f"'{poses_path}' does not contain a parts mapping.")

    parts: list[AssemblyPartSpec] = []
    part_pose_payloads: dict[str, dict[str, Any]] = {}
    below_table_part_ids: list[str] = []
    for part_id in selected_order:
        mesh_path = assembly_path / f"{part_id}.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Selected part '{part_id}' has no mesh at '{mesh_path}'.")
        raw_part = raw_parts.get(part_id)
        if not isinstance(raw_part, dict):
            raise ValueError(f"Part '{part_id}' is missing from '{poses_path}'.")
        part_pose_payloads[part_id] = raw_part

        mesh = load_triangle_mesh(mesh_path, scale=float(mesh_scale))
        vertices = np.asarray(mesh.vertices_obj, dtype=float)
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        table_clearance_m = float(bounds_min[2] - float(table_z_assembly_m))
        touches_table = abs(table_clearance_m) <= float(table_contact_tolerance_m)
        if table_clearance_m < -float(table_contact_tolerance_m):
            below_table_part_ids.append(part_id)
        parts.append(
            AssemblyPartSpec(
                part_id=part_id,
                mesh_path=_display_path(mesh_path, repo_root=repo_root_path),
                role=str(raw_part.get("role", "")),
                bounds_min_assembly_m=tuple(float(value) for value in bounds_min),
                bounds_max_assembly_m=tuple(float(value) for value in bounds_max),
                table_clearance_m=table_clearance_m,
                touches_table=touches_table,
                vertex_count=len(vertices),
                face_count=len(mesh.faces),
                asset_record=_asset_record(mesh_path, repo_root=repo_root_path),
                resolved_mesh_path=mesh_path,
            )
        )

    steps: list[AssemblySequenceStep] = []
    for step_index, incoming_part_id in enumerate(selected_order):
        raw_part = part_pose_payloads[incoming_part_id]
        transform_field = f"parts.{incoming_part_id}.final_to_pre_insertion_transform_m"
        transform = _matrix4(raw_part.get("final_to_pre_insertion_transform_m"), field_name=transform_field)
        if not np.allclose(transform[:3, :3], np.eye(3), atol=_TRANSFORM_ATOL):
            raise ValueError(
                f"{transform_field} contains rotation; Stage 0 currently supports translation-only insertions."
            )
        translation = tuple(float(value) for value in transform[:3, 3])

        raw_vector = raw_part.get("pre_to_final_insertion_vector_m")
        if raw_vector is None:
            vector = None
            distance_m = 0.0
            if float(np.linalg.norm(transform[:3, 3])) > _VECTOR_ATOL_M:
                raise ValueError(
                    f"Part '{incoming_part_id}' has no pre_to_final_insertion_vector_m but its "
                    "final_to_pre_insertion_transform_m has non-zero translation."
                )
        else:
            vector = _tuple3(
                raw_vector,
                field_name=f"parts.{incoming_part_id}.pre_to_final_insertion_vector_m",
            )
            vector_array = np.asarray(vector, dtype=float)
            if not np.allclose(vector_array + transform[:3, 3], np.zeros(3), atol=_VECTOR_ATOL_M):
                raise ValueError(
                    f"Part '{incoming_part_id}' has inconsistent final-to-pre translation and "
                    "pre-to-final insertion vector."
                )
            distance_m = float(np.linalg.norm(vector_array))
            recorded_distance = raw_part.get("pre_to_final_insertion_distance_m")
            if recorded_distance is not None and not math.isclose(
                float(recorded_distance),
                distance_m,
                rel_tol=0.0,
                abs_tol=_VECTOR_ATOL_M,
            ):
                raise ValueError(
                    f"Part '{incoming_part_id}' records insertion distance {recorded_distance}, "
                    f"but its insertion vector has length {distance_m}."
                )

        before = selected_order[:step_index]
        after = selected_order[: step_index + 1]
        if base_id in before:
            base_status = "assembled"
        elif incoming_part_id == base_id:
            base_status = "incoming"
        else:
            base_status = "not_present"
        raw_waypoints = raw_part.get("disassembly_path_waypoints")
        steps.append(
            AssemblySequenceStep(
                step_id=f"step_{step_index:03d}_part_{incoming_part_id}",
                step_index=step_index,
                incoming_part_id=incoming_part_id,
                incoming_part_role=str(raw_part.get("role", "")),
                assembled_part_ids_before=before,
                assembled_part_ids_after=after,
                base_part_status=base_status,
                holder_base_available=base_id in before,
                final_to_pre_insertion_transform_m=tuple(tuple(float(value) for value in row) for row in transform),
                final_to_pre_insertion_translation_m=translation,
                pre_to_final_insertion_vector_m=vector,
                insertion_distance_m=distance_m,
                disassembly_path_waypoints=None if raw_waypoints is None else int(raw_waypoints),
            )
        )

    table_contact_parts = tuple(part.part_id for part in parts if part.touches_table)
    first_holder_step_index = base_order_index + 1 if base_order_index + 1 < len(selected_order) else None
    warnings: list[str] = []
    if base_order_index > 0:
        if first_holder_step_index is None:
            holder_detail = "No later insertion step can use a base-only holder grasp."
        else:
            holder_detail = f"Base-only holder planning starts at step {first_holder_step_index}."
        warnings.append(
            f"Base part override '{base_id}' enters at step {base_order_index}, after "
            f"{list(selected_order[:base_order_index])}. {holder_detail}"
        )
    base_part = next(part for part in parts if part.part_id == base_id)
    if not base_part.touches_table:
        warnings.append(
            f"Base part '{base_id}' has minimum assembly Z "
            f"{base_part.bounds_min_assembly_m[2]:.6f} m and does not directly touch table "
            f"Z {float(table_z_assembly_m):.6f} m within tolerance."
        )
    if below_table_part_ids:
        warnings.append(f"Parts {below_table_part_ids} extend below table Z {float(table_z_assembly_m):.6f} m.")
    if not table_contact_parts:
        warnings.append("No selected part touches the configured table plane within tolerance.")

    return AssemblySequence(
        assembly=assembly_name,
        base_part_id=base_id,
        base_part_source=base_part_source,
        base_part_order_index=base_order_index,
        first_holder_step_index=first_holder_step_index,
        selected_order=selected_order,
        mesh_scale=float(mesh_scale),
        table_z_assembly_m=float(table_z_assembly_m),
        table_contact_tolerance_m=float(table_contact_tolerance_m),
        table_contact_part_ids=table_contact_parts,
        parts=tuple(parts),
        steps=tuple(steps),
        precedence_plan_record=_asset_record(precedence_path, repo_root=repo_root_path),
        pre_insertion_poses_record=_asset_record(poses_path, repo_root=repo_root_path),
        warnings=tuple(warnings),
        source_assembly_dir=assembly_path,
    )


def write_assembly_sequence_json(sequence: AssemblySequence, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sequence.to_payload(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "AssemblyPartSpec",
    "AssemblySequence",
    "AssemblySequenceStep",
    "DEFAULT_ASSET_ROOT",
    "SCHEMA_VERSION",
    "compile_assembly_sequence",
    "write_assembly_sequence_json",
]
