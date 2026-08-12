#!/usr/bin/env python3
"""Compare full dual-arm planning with isolated per-arm pickup IK.

The B side independently tests each robot on the holder/base grasp and the
ground-supported incoming-part pickup. It deliberately ignores the passive
robot and separately placed object while retaining the active robot's joint
limits, self-collision, work-surface collision, exact TCP poses, sequential
targets, and candidate-specific gripper openings. This distinguishes
robot/grasp reachability from coordination failures without weakening the
production dual-arm planner.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    CandidateStatus,
    SavedGraspCandidate,
    evaluate_saved_grasps_against_pickup_pose,
    load_grasp_bundle,
    rotmat_to_quat_xyzw,
    world_point_to_object,
    write_debug_html,
)
from grasp_planning.grasping.grasp_transforms import (  # noqa: E402
    WorldFrameGraspCandidate,
    saved_grasp_to_world_grasp,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh  # noqa: E402
from grasp_planning.grasping.mesh_io import load_triangle_mesh, resolve_mesh_path  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline.dual_robot_pair_scoring import MovableFrame  # noqa: E402
from grasp_planning.pipeline.dual_robot_simple_sim import (  # noqa: E402
    DEFAULT_HOLDER_BASE_WORLD,
    DEFAULT_HOLDER_PREGRASP_OFFSET_M,
    DEFAULT_INSERTER_BASE_WORLD,
    DEFAULT_INSERTER_PREGRASP_OFFSET_M,
    DEFAULT_PICKUP_CONTACT_GAP_M,
    DEFAULT_PICKUP_FLOOR_CLEARANCE_MARGIN_M,
    DEFAULT_TRANSPORT_CLEARANCE_M,
    _declared_holder_candidate_source,
    _rpy_rotation,
    _source_pose_from_bundle,
    compose_source_pose_world,
    resolve_dual_robot_step_selection,
    source_pose_resting_on_floor,
)
from grasp_planning.ros2.moveit_pose_commander import (  # noqa: E402
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    PoseTarget,
    rclpy,
)
from grasp_planning.start_poses import (  # noqa: E402
    KUKA_MOVEIT_ARM_START_JOINT_VALUES,
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
    kuka_gripper_clamp_width,
    kuka_moveit_gripper_state,
)
from scripts.plan_simple_dual_robot_sim import (  # noqa: E402
    ARM_SPEC_BY_ROBOT,
    WORK_SURFACE_CENTER_XY_M,
    WORK_SURFACE_SIZE_M,
    _exact_ik_seed_candidates,
)

SCHEMA_VERSION = 1
ROBOT_NAMES = ("lbr_one", "lbr_two")


@dataclass(frozen=True)
class IncomingCaseCandidates:
    world_grasps: tuple[WorldFrameGraspCandidate, ...]
    object_candidates: tuple[SavedGraspCandidate, ...]
    mesh_local: TriangleMesh
    pickup_pose: ObjectWorldPose
    floor_candidates_checked: int
    gripper_collision_model: str


def _assembly_mesh_in_source_frame(
    mesh_assembly: TriangleMesh,
    source_pose_assembly: ObjectWorldPose,
) -> TriangleMesh:
    """Express an asset/assembly-frame mesh in the saved grasp source frame."""

    vertices_source = (
        np.asarray(mesh_assembly.vertices_obj, dtype=float)
        - source_pose_assembly.translation_world[None, :]
    ) @ source_pose_assembly.rotation_world_from_object
    return TriangleMesh(
        vertices_obj=vertices_source,
        faces=np.asarray(mesh_assembly.faces, dtype=np.int64),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path(
            "artifacts/dual_assembly_benchmark/"
            "plumbers_block_ik_after_gripper_fix_20260811/summary.json"
        ),
        help="Existing dual-arm benchmark summary providing the A cases/results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dual_assembly_benchmark/solo_pickup_ik_ab_20260812"),
    )
    parser.add_argument("--moveit-namespace", default="/lbr_dual_arm")
    parser.add_argument("--ik-timeout-s", type=float, default=0.35)
    parser.add_argument("--ik-candidates", type=int, default=7)
    parser.add_argument("--ik-beam-width", type=int, default=4)
    parser.add_argument("--seed-perturbation-rad", type=float, default=0.60)
    parser.add_argument("--approach-steps", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=0, help="0 checks every floor-valid Stage-3 grasp.")
    parser.add_argument("--limit-cases", type=int, default=0, help="0 runs every completed baseline case.")
    parser.add_argument(
        "--no-grasp-debug",
        action="store_true",
        help="Skip exhaustive per-grasp IK records and collision-style case HTML files.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _work_surface(*, floor_z: float) -> dict[str, object]:
    return {
        "id": "solo_pickup_work_surface",
        "type": "box",
        "frame_id": "base_link",
        "size_m": list(WORK_SURFACE_SIZE_M),
        "xyz": [
            float(WORK_SURFACE_CENTER_XY_M[0]),
            float(WORK_SURFACE_CENTER_XY_M[1]),
            float(floor_z) - 0.5 * float(WORK_SURFACE_SIZE_M[2]),
        ],
        "rpy": [0.0, 0.0, 0.0],
    }


def _body_robot(body: str) -> str:
    value = str(body)
    for robot in ROBOT_NAMES:
        if value.startswith(f"{robot}_"):
            return robot
    return ""


def blocking_solo_contacts(
    contacts: Iterable[Mapping[str, object]],
    *,
    active_robot: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split contacts into active-arm blockers and ignored passive-arm contacts."""

    passive_robot = "lbr_two" if active_robot == "lbr_one" else "lbr_one"
    blocking: list[dict[str, object]] = []
    ignored: list[dict[str, object]] = []
    for raw_contact in contacts:
        contact = dict(raw_contact)
        robots = {
            robot
            for robot in (
                _body_robot(str(contact.get("body_1", ""))),
                _body_robot(str(contact.get("body_2", ""))),
            )
            if robot
        }
        if active_robot not in robots:
            ignored.append(contact)
        elif passive_robot in robots:
            ignored.append(contact)
        else:
            blocking.append(contact)
    return blocking, ignored


def _pose_target(position: Iterable[float], orientation: Iterable[float]) -> PoseTarget:
    xyz = tuple(float(value) for value in position)
    quaternion = tuple(float(value) for value in orientation)
    return PoseTarget.from_quaternion(
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        quaternion_xyzw=quaternion,
        frame_id="base_link",
    )


def _pickup_targets(
    grasp,
    *,
    target_prefix: str,
    approach_steps: int,
    lift_m: float | None,
) -> tuple[tuple[str, PoseTarget, bool], ...]:
    pregrasp = np.asarray(grasp.pregrasp_position_w, dtype=float)
    contact = np.asarray(grasp.position_w, dtype=float)
    orientation = grasp.orientation_xyzw
    targets: list[tuple[str, PoseTarget, bool]] = [
        (f"{target_prefix}_pregrasp", _pose_target(pregrasp, orientation), False)
    ]
    for step in range(1, int(approach_steps) + 1):
        fraction = float(step) / float(approach_steps)
        position = pregrasp + fraction * (contact - pregrasp)
        targets.append(
            (
                f"{target_prefix}_grasp__approach_{step:02d}_of_{int(approach_steps):02d}",
                _pose_target(position, orientation),
                step == int(approach_steps),
            )
        )
    if lift_m is not None:
        targets.append(
            (
                f"{target_prefix}_lift",
                _pose_target(contact + np.asarray((0.0, 0.0, float(lift_m))), orientation),
                False,
            )
        )
    return tuple(targets)


def _complete_start_state() -> dict[str, float]:
    state: dict[str, float] = {}
    for robot in ROBOT_NAMES:
        state.update(
            (f"{robot}_A{index}", float(value))
            for index, value in enumerate(KUKA_MOVEIT_ARM_START_JOINT_VALUES, start=1)
        )
        state.update(kuka_moveit_gripper_state(robot, KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M))
    return state


def _valid_for_solo_arm(
    commander: MoveItPoseCommander,
    state: Mapping[str, float],
    *,
    active_robot: str,
    diagnostics: Counter,
) -> bool:
    validity, message = commander.check_state_validity(state, group_name="")
    diagnostics["state_validity_requests"] += 1
    if validity is None:
        diagnostics["state_validity_errors"] += 1
        diagnostics[f"state_validity_error:{message}"] += 1
        return False
    contacts = [dict(value) for value in validity.get("contacts", [])]
    blocking, ignored = blocking_solo_contacts(contacts, active_robot=active_robot)
    diagnostics["ignored_passive_contacts"] += len(ignored)
    diagnostics["blocking_contacts"] += len(blocking)
    for contact in blocking:
        bodies = sorted((str(contact.get("body_1", "")), str(contact.get("body_2", ""))))
        diagnostics[f"blocking_pair:{bodies[0]} <-> {bodies[1]}"] += 1
    if bool(validity.get("valid", False)):
        diagnostics["valid_states"] += 1
        return True
    if contacts and not blocking:
        diagnostics["valid_after_ignoring_passive_robot"] += 1
        return True
    diagnostics["invalid_states"] += 1
    if not contacts:
        diagnostics["invalid_without_contacts"] += 1
    return False


def _solve_candidate(
    commander: MoveItPoseCommander,
    *,
    active_robot: str,
    grasp,
    ik_candidates: int,
    beam_width: int,
    seed_perturbation_rad: float,
    approach_steps: int,
    target_prefix: str,
    lift_m: float | None,
) -> tuple[bool, str, dict[str, object]]:
    joint_names = tuple(str(value) for value in ARM_SPEC_BY_ROBOT[active_robot]["joint_names"])
    start_state = _complete_start_state()
    start_state.update(
        kuka_moveit_gripper_state(active_robot, kuka_gripper_clamp_width(grasp.gripper_width))
    )
    branches = [start_state]
    diagnostics: Counter = Counter()
    target_records: list[dict[str, object]] = []
    for target_name, target, closes_gripper in _pickup_targets(
        grasp,
        target_prefix=target_prefix,
        approach_steps=approach_steps,
        lift_m=lift_m,
    ):
        expanded: list[tuple[float, dict[str, float]]] = []
        for parent in branches:
            active_start = tuple(float(parent[name]) for name in joint_names)
            seeds = _exact_ik_seed_candidates(
                active_start,
                preferred_joint_positions=None,
                candidate_count=ik_candidates,
                perturbation_rad=seed_perturbation_rad,
            )
            solution_limit = int(beam_width) if len(branches) == 1 else 1
            distinct: list[tuple[float, ...]] = []
            for seed in seeds:
                seed_state = dict(parent)
                seed_state.update((name, float(value)) for name, value in zip(joint_names, seed))
                started = time.monotonic()
                joints, _message = commander.compute_ik(
                    target,
                    seed_robot_state=seed_state,
                    avoid_collisions=False,
                )
                diagnostics["ik_requests"] += 1
                diagnostics["ik_duration_us"] += int(1.0e6 * (time.monotonic() - started))
                if joints is None:
                    diagnostics["no_ik"] += 1
                    continue
                diagnostics["kinematic_solutions"] += 1
                solution = tuple(float(value) for value in joints)
                if any(np.max(np.abs(np.asarray(solution) - np.asarray(other))) < 0.05 for other in distinct):
                    diagnostics["duplicate_solutions"] += 1
                    continue
                candidate_state = dict(parent)
                candidate_state.update((name, value) for name, value in zip(joint_names, solution))
                if not _valid_for_solo_arm(
                    commander,
                    candidate_state,
                    active_robot=active_robot,
                    diagnostics=diagnostics,
                ):
                    continue
                if closes_gripper:
                    closed_state = dict(candidate_state)
                    closed_state.update(kuka_moveit_gripper_state(active_robot, grasp.jaw_width))
                    if not _valid_for_solo_arm(
                        commander,
                        closed_state,
                        active_robot=active_robot,
                        diagnostics=diagnostics,
                    ):
                        diagnostics["closed_state_failures"] += 1
                        continue
                    candidate_state = closed_state
                distinct.append(solution)
                delta = np.asarray(solution, dtype=float) - np.asarray(active_start, dtype=float)
                expanded.append((float(np.linalg.norm(delta)), candidate_state))
                if len(distinct) >= solution_limit:
                    break
        target_records.append(
            {
                "target": target_name,
                "input_branches": len(branches),
                "output_branches": len(expanded),
            }
        )
        if not expanded:
            return False, target_name, {
                **{key: int(value) for key, value in diagnostics.items()},
                "ik_duration_s": float(diagnostics["ik_duration_us"]) / 1.0e6,
                "targets": target_records,
            }
        branches = [state for _cost, state in sorted(expanded, key=lambda item: item[0])[: int(beam_width)]]
    return True, "", {
        **{key: int(value) for key, value in diagnostics.items()},
        "ik_duration_s": float(diagnostics["ik_duration_us"]) / 1.0e6,
        "targets": target_records,
    }


def _incoming_case_candidates(record: Mapping[str, object]) -> IncomingCaseCandidates:
    selection = resolve_dual_robot_step_selection(
        assembly=str(record["assembly"]),
        incoming_part_id=str(record["incoming_part_id"]),
        step_id=str(record["step_id"]),
    )
    bundle = load_grasp_bundle(selection.artifact_dir / f"inserter_candidates_{selection.step_id}.json")
    source_pose_assembly = _source_pose_from_bundle(bundle)
    assembly_world = MovableFrame(
        (
            float(record["assembly_x"]),
            float(record["assembly_y"]),
            float(record["assembly_z"]),
        ),
        float(record["assembly_yaw_deg"]),
    )
    final_source_pose_world = compose_source_pose_world(
        source_pose_assembly=source_pose_assembly,
        assembly_world=assembly_world,
    )
    pickup_rotation_world = (
        _rpy_rotation(
            float(record["pickup_roll_deg"]),
            float(record["pickup_pitch_deg"]),
            float(record["pickup_yaw_deg"]),
        )
        @ final_source_pose_world.rotation_world_from_object
    )
    pickup_orientation = ObjectWorldPose(
        position_world=(0.0, 0.0, 0.0),
        orientation_xyzw_world=tuple(float(value) for value in rotmat_to_quat_xyzw(pickup_rotation_world)),
    )
    mesh_path = resolve_mesh_path(bundle.target_stl_path)
    mesh_assembly = load_triangle_mesh(mesh_path, scale=float(bundle.stl_scale))
    pickup_pose = source_pose_resting_on_floor(
        mesh_assembly=mesh_assembly,
        source_pose_assembly=source_pose_assembly,
        source_orientation_world=pickup_orientation,
        xy_world=(float(record["pickup_x"]), float(record["pickup_y"])),
        floor_z_world_m=float(record["floor_z"]),
    )
    mesh_source = _assembly_mesh_in_source_frame(mesh_assembly, source_pose_assembly)
    metadata = dict(bundle.metadata)
    model_name = str(metadata.get("gripper_collision_model", metadata.get("gripper_model", "kuka_y_gripper")))
    statuses = evaluate_saved_grasps_against_pickup_pose(
        bundle.candidates,
        object_pose_world=pickup_pose,
        contact_gap_m=DEFAULT_PICKUP_CONTACT_GAP_M,
        gripper_collision_model=model_name,
        floor_z_world_m=float(record["floor_z"]),
        floor_clearance_margin_m=DEFAULT_PICKUP_FLOOR_CLEARANCE_MARGIN_M,
        contact_lateral_offsets_m=tuple(
            float(value)
            for value in metadata.get("contact_lateral_offsets_m", (-0.0029166667, 0.0, 0.0029166667))
        ),
        contact_approach_offsets_m=tuple(
            float(value)
            for value in metadata.get("contact_approach_offsets_m", (-0.0030833333, 0.0, 0.0030833333))
        ),
    )
    accepted = [status.grasp for status in statuses if status.status == "accepted"]
    world_grasps = [
        saved_grasp_to_world_grasp(
            candidate,
            pickup_pose,
            pregrasp_offset=DEFAULT_INSERTER_PREGRASP_OFFSET_M,
            gripper_width_clearance=2.0 * DEFAULT_PICKUP_CONTACT_GAP_M,
        )
        for candidate in accepted
    ]
    return IncomingCaseCandidates(
        world_grasps=tuple(world_grasps),
        object_candidates=tuple(accepted),
        mesh_local=mesh_source,
        pickup_pose=pickup_pose,
        floor_candidates_checked=len(statuses),
        gripper_collision_model=model_name,
    )


def _holder_step_candidates(record: Mapping[str, object]):
    selection = resolve_dual_robot_step_selection(
        assembly=str(record["assembly"]),
        incoming_part_id=str(record["incoming_part_id"]),
        step_id=str(record["step_id"]),
    )
    pair_payload = _read_json(selection.artifact_dir / f"dual_grasp_pairs_{selection.step_id}.json")
    holder_bundle = load_grasp_bundle(selection.artifact_dir / "holder_base_candidates.json")
    holder_candidates, holder_source_pose_assembly, _source = _declared_holder_candidate_source(
        root=selection.artifact_dir,
        pair_payload=pair_payload,
        fallback_candidates=holder_bundle.candidates,
        fallback_source_pose_assembly=_source_pose_from_bundle(holder_bundle),
    )
    accepted_ids = {
        str(value["grasp_id"])
        for value in pair_payload.get("holder_candidates", [])
        if isinstance(value, dict) and value.get("status") == "accepted"
    }
    assembly_world = MovableFrame(
        (
            float(record["assembly_x"]),
            float(record["assembly_y"]),
            float(record["assembly_z"]),
        ),
        float(record["assembly_yaw_deg"]),
    )
    holder_source_pose_world = compose_source_pose_world(
        source_pose_assembly=holder_source_pose_assembly,
        assembly_world=assembly_world,
    )
    world_grasps = [
        saved_grasp_to_world_grasp(
            candidate,
            holder_source_pose_world,
            pregrasp_offset=DEFAULT_HOLDER_PREGRASP_OFFSET_M,
            gripper_width_clearance=2.0 * DEFAULT_PICKUP_CONTACT_GAP_M,
        )
        for candidate in holder_candidates
        if candidate.grasp_id in accepted_ids
    ]
    return world_grasps, len(holder_candidates)


def _run_arm(
    commander: MoveItPoseCommander,
    *,
    active_robot: str,
    grasps,
    args: argparse.Namespace,
    target_prefix: str = "incoming_pickup",
    lift_m: float | None = DEFAULT_TRANSPORT_CLEARANCE_M,
    exhaustive: bool = False,
) -> dict[str, object]:
    aggregate: Counter = Counter()
    failures: Counter = Counter()
    checked = 0
    first_success_id = ""
    candidate_results: list[dict[str, object]] = []
    started = time.monotonic()
    candidates = grasps if int(args.candidate_limit) <= 0 else grasps[: int(args.candidate_limit)]
    for grasp in candidates:
        checked += 1
        ok, failed_target, diagnostics = _solve_candidate(
            commander,
            active_robot=active_robot,
            grasp=grasp,
            ik_candidates=int(args.ik_candidates),
            beam_width=int(args.ik_beam_width),
            seed_perturbation_rad=float(args.seed_perturbation_rad),
            approach_steps=int(args.approach_steps),
            target_prefix=target_prefix,
            lift_m=lift_m,
        )
        for key, value in diagnostics.items():
            if isinstance(value, (int, float)) and key not in {"ik_duration_s", "ik_duration_us"}:
                aggregate[key] += int(value)
        aggregate["ik_duration_us"] += int(1.0e6 * float(diagnostics.get("ik_duration_s", 0.0)))
        candidate_results.append(
            {
                "grasp_id": grasp.grasp_id,
                "success": bool(ok),
                "failed_target": failed_target,
                "ik_requests": int(diagnostics.get("ik_requests", 0)),
                "no_ik": int(diagnostics.get("no_ik", 0)),
                "kinematic_solutions": int(diagnostics.get("kinematic_solutions", 0)),
                "valid_states": int(diagnostics.get("valid_states", 0)),
                "invalid_states": int(diagnostics.get("invalid_states", 0)),
                "blocking_contacts": int(diagnostics.get("blocking_contacts", 0)),
                "targets": diagnostics.get("targets", []),
            }
        )
        if ok:
            if not first_success_id:
                first_success_id = grasp.grasp_id
            if not exhaustive:
                break
            continue
        failures[failed_target] += 1
    if first_success_id:
        return {
            "success": True,
            "grasp_id": first_success_id,
            "failed_target": "",
            "failure_target_counts": dict(failures),
            "floor_valid_candidates": len(grasps),
            "candidates_checked": checked,
            "candidate_success_count": sum(bool(value["success"]) for value in candidate_results),
            "candidate_results": candidate_results,
            "duration_s": time.monotonic() - started,
            "diagnostics": {
                **{key: int(value) for key, value in aggregate.items()},
                "ik_duration_s": float(aggregate["ik_duration_us"]) / 1.0e6,
            },
        }
    return {
        "success": False,
        "grasp_id": "",
        "failed_target": failures.most_common(1)[0][0] if failures else "no_floor_valid_grasp",
        "failure_target_counts": dict(failures),
        "floor_valid_candidates": len(grasps),
        "candidates_checked": checked,
        "candidate_success_count": 0,
        "candidate_results": candidate_results,
        "duration_s": time.monotonic() - started,
        "diagnostics": {
            **{key: int(value) for key, value in aggregate.items()},
            "ik_duration_s": float(aggregate["ik_duration_us"]) / 1.0e6,
        },
    }


def _ground_plane_overlay(
    case: IncomingCaseCandidates,
    *,
    floor_z: float,
) -> dict[str, object]:
    center_x, center_y = WORK_SURFACE_CENTER_XY_M
    size_x, size_y, _size_z = WORK_SURFACE_SIZE_M
    corners_world = np.asarray(
        [
            [center_x - 0.5 * size_x, center_y - 0.5 * size_y, floor_z],
            [center_x + 0.5 * size_x, center_y - 0.5 * size_y, floor_z],
            [center_x + 0.5 * size_x, center_y + 0.5 * size_y, floor_z],
            [center_x - 0.5 * size_x, center_y + 0.5 * size_y, floor_z],
        ],
        dtype=float,
    )
    return {
        "label": f"work surface z={float(floor_z):.3f} m",
        "corners_obj": [
            [float(value) for value in world_point_to_object(point, case.pickup_pose)]
            for point in corners_world
        ]
    }


def _world_scene_overlays(
    baseline_record: Mapping[str, object],
    *,
    floor_z: float,
) -> dict[str, object]:
    base_positions = {
        "lbr_one": np.asarray(DEFAULT_HOLDER_BASE_WORLD.position_world_m, dtype=float),
        "lbr_two": np.asarray(DEFAULT_INSERTER_BASE_WORLD.position_world_m, dtype=float),
    }
    colors = {"lbr_one": "#d97706", "lbr_two": "#7c3aed"}
    markers: list[dict[str, object]] = []
    lines: list[dict[str, object]] = []
    for robot, base in base_positions.items():
        base_display = np.asarray([base[0], base[1], float(floor_z)], dtype=float)
        shoulder = base_display + np.asarray([0.0, 0.0, 0.34], dtype=float)
        markers.extend(
            [
                {
                    "label": f"{robot} base",
                    "position": base_display.tolist(),
                    "color": colors[robot],
                    "radius": 8,
                },
                {
                    "label": f"{robot} shoulder",
                    "position": shoulder.tolist(),
                    "color": colors[robot],
                    "radius": 5,
                    "connect_to_pregrasp": True,
                },
            ]
        )
        lines.append(
            {
                "start": base_display.tolist(),
                "end": shoulder.tolist(),
                "color": colors[robot],
                "width": 3,
                "opacity": 0.75,
                "dash": "",
            }
        )
    markers.append(
        {
            "label": "assembly center",
            "position": [
                float(baseline_record.get("assembly_x", 0.55)),
                float(baseline_record.get("assembly_y", 0.0)),
                float(baseline_record.get("assembly_z", floor_z)),
            ],
            "color": "#0f766e",
            "radius": 6,
        }
    )
    return {
        "axes": {"origin": [0.0, 0.0, float(floor_z)], "length_m": 0.18},
        "markers": markers,
        "lines": lines,
    }


def _candidate_arm_label(result: Mapping[str, object] | None) -> str:
    if result is None:
        return "not_tested"
    if bool(result.get("success", False)):
        return (
            f"PASS(ik={int(result.get('ik_requests', 0))},"
            f"valid={int(result.get('valid_states', 0))})"
        )
    return (
        f"FAIL@{result.get('failed_target', 'unknown')}"
        f"(ik={int(result.get('ik_requests', 0))},"
        f"no_ik={int(result.get('no_ik', 0))},"
        f"invalid={int(result.get('invalid_states', 0))})"
    )


def _write_case_grasp_debug(
    output_html: Path,
    *,
    baseline_record: Mapping[str, object],
    case: IncomingCaseCandidates,
    solo: Mapping[str, Mapping[str, object]],
) -> None:
    per_arm = {
        arm: {
            str(value["grasp_id"]): value
            for value in arm_result.get("candidate_results", [])
            if isinstance(value, dict) and value.get("grasp_id")
        }
        for arm, arm_result in solo.items()
    }
    statuses: list[CandidateStatus] = []
    for candidate in case.object_candidates:
        one = per_arm.get("lbr_one", {}).get(candidate.grasp_id)
        two = per_arm.get("lbr_two", {}).get(candidate.grasp_id)
        reachable = any(bool(value and value.get("success", False)) for value in (one, two))
        statuses.append(
            CandidateStatus(
                grasp=candidate,
                status="accepted" if reachable else "rejected",
                reason=(
                    f"lbr_one={_candidate_arm_label(one)}; "
                    f"lbr_two={_candidate_arm_label(two)}"
                ),
            )
        )

    assigned_arm = str(baseline_record.get("inserter_arm", ""))
    assigned_result = solo.get(assigned_arm, {})
    metadata_lines = [
        f"case_id:           {baseline_record.get('case_id', '')}",
        f"assembly:          {baseline_record.get('assembly', '')}",
        f"incoming_part_id:  {baseline_record.get('incoming_part_id', '')}",
        f"step_id:           {baseline_record.get('step_id', '')}",
        f"placement:         {baseline_record.get('placement_id', '')}",
        f"orientation:       {baseline_record.get('orientation_id', '')}",
        (
            "pickup_xyz_m:      "
            f"({float(baseline_record.get('pickup_x', 0.0)):.4f}, "
            f"{float(baseline_record.get('pickup_y', 0.0)):.4f}, "
            f"{float(case.pickup_pose.position_world[2]):.4f})"
        ),
        (
            "pickup_rpy_deg:    "
            f"({float(baseline_record.get('pickup_roll_deg', 0.0)):.1f}, "
            f"{float(baseline_record.get('pickup_pitch_deg', 0.0)):.1f}, "
            f"{float(baseline_record.get('pickup_yaw_deg', 0.0)):.1f})"
        ),
        f"floor_z_m:         {float(baseline_record.get('floor_z', 0.0)):.4f}",
        f"assigned_inserter: {assigned_arm}",
        f"dual_result:       {'PASS' if baseline_record.get('success') else 'FAIL'}",
        f"assigned_solo:     {'PASS' if assigned_result.get('success') else 'FAIL'}",
        f"floor_valid:       {len(case.object_candidates)} of {case.floor_candidates_checked}",
        "color_semantics:   green=at least one isolated arm completes pickup; red=both fail",
        "scene_scope:       active-arm self/work-surface collision; passive arm and other part ignored",
        "mesh_frame:        saved grasp source frame",
        "display_frame:     base_link world frame; gripper is shown at contact pose",
        f"pregrasp_offset_m: {DEFAULT_INSERTER_PREGRASP_OFFSET_M:.3f}",
    ]
    write_debug_html(
        title=f"Pickup IK Grasps: {baseline_record.get('case_id', '')}",
        subtitle=(
            "Every floor-valid Stage-3 incoming grasp in its actual global pickup pose. "
            "Select a grasp to compare the contact gripper with its 10 cm pregrasp ghost, "
            "robot bases, shoulders, and workbench location."
        ),
        mesh_local=case.mesh_local,
        candidate_statuses=statuses,
        output_html=output_html,
        contact_gap_m=DEFAULT_PICKUP_CONTACT_GAP_M,
        ground_plane=_ground_plane_overlay(
            case,
            floor_z=float(baseline_record.get("floor_z", 0.0)),
        ),
        metadata_lines=metadata_lines,
        display_object_pose_world=case.pickup_pose,
        max_mesh_edges=8000,
        gripper_collision_model=case.gripper_collision_model,
        pregrasp_offset_m=DEFAULT_INSERTER_PREGRASP_OFFSET_M,
        pregrasp_width_clearance_m=2.0 * DEFAULT_PICKUP_CONTACT_GAP_M,
        scene_label="World Frame (base_link)",
        scene_overlays=_world_scene_overlays(
            baseline_record,
            floor_z=float(baseline_record.get("floor_z", 0.0)),
        ),
    )


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    complete = [record for record in records if record.get("status") == "complete"]
    matrix = Counter()
    assigned_solo_success = 0
    assigned_pickups_success = 0
    either_assignment_success = 0
    for record in complete:
        dual = bool(record.get("dual_success", False))
        isolated = bool(record.get("either_assignment_solo_success", False))
        matrix[f"dual_{'pass' if dual else 'fail'}__solo_{'pass' if isolated else 'fail'}"] += 1
        assigned_robot = str(record.get("baseline_inserter_arm", ""))
        robot_results = record.get("solo", {})
        if isinstance(robot_results, dict):
            assigned_result = robot_results.get(assigned_robot, {})
            if isinstance(assigned_result, dict) and bool(assigned_result.get("success", False)):
                assigned_solo_success += 1
        assigned_pickups_success += int(bool(record.get("assigned_roles_solo_success", False)))
        either_assignment_success += int(bool(record.get("either_assignment_solo_success", False)))
    return {
        "completed": len(complete),
        "dual_success": sum(bool(record.get("dual_success", False)) for record in complete),
        "assigned_inserter_solo_success": assigned_solo_success,
        "assigned_pickups_solo_success": assigned_pickups_success,
        "either_assignment_solo_success": either_assignment_success,
        "lbr_one_solo_success": sum(
            bool(dict(record.get("solo", {})).get("lbr_one", {}).get("success", False))
            for record in complete
        ),
        "lbr_two_solo_success": sum(
            bool(dict(record.get("solo", {})).get("lbr_two", {}).get("success", False))
            for record in complete
        ),
        "incoming_either_arm_solo_success": sum(
            bool(record.get("either_solo_success", False)) for record in complete
        ),
        "outcome_matrix": dict(matrix),
    }


def _write_html(path: Path, payload: Mapping[str, object]) -> None:
    records = [dict(value) for value in payload.get("records", []) if isinstance(value, dict)]
    aggregate = dict(payload.get("aggregate", {}))
    rows = []
    for record in records:
        solo = dict(record.get("solo", {}))
        holder_solo = dict(record.get("solo_holder", {}))
        one = dict(solo.get("lbr_one", {}))
        two = dict(solo.get("lbr_two", {}))
        holder_one = dict(holder_solo.get("lbr_one", {}))
        holder_two = dict(holder_solo.get("lbr_two", {}))
        status = str(record.get("status", "pending"))
        dual = "PASS" if record.get("dual_success") else "FAIL"
        grasp_debug = str(record.get("grasp_debug_html", ""))
        grasp_debug_link = (
            f"<a href='{html.escape(grasp_debug, quote=True)}'>world</a> · "
            f"<a href='{html.escape(grasp_debug, quote=True)}?view=focus'>focus</a>"
            if grasp_debug
            else ""
        )
        one_result = "PASS" if one.get("success") else str(one.get("failed_target", "FAIL"))
        two_result = "PASS" if two.get("success") else str(two.get("failed_target", "FAIL"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('case_id', '')))}</td>"
            f"<td>{grasp_debug_link}</td>"
            f"<td>{html.escape(status)}</td><td class='{dual.lower()}'>{dual}</td>"
            f"<td>{html.escape(one_result)}</td><td>{int(one.get('candidates_checked', 0))}</td>"
            f"<td>{html.escape(two_result)}</td><td>{int(two.get('candidates_checked', 0))}</td>"
            f"<td>{html.escape('PASS' if holder_one.get('success') else str(holder_one.get('failed_target', 'FAIL')))}</td>"
            f"<td>{html.escape('PASS' if holder_two.get('success') else str(holder_two.get('failed_target', 'FAIL')))}</td>"
            f"<td>{'PASS' if record.get('assigned_roles_solo_success') else 'FAIL'}</td>"
            f"<td>{'PASS' if record.get('either_assignment_solo_success') else 'FAIL'}</td>"
            f"<td>{html.escape(str(record.get('orientation_id', '')))}</td>"
            f"<td>{html.escape(str(record.get('placement_id', '')))}</td>"
            "</tr>"
        )
    viewer_records = [
        record
        for record in records
        if record.get("status") == "complete"
        and not record.get("dual_success")
        and record.get("grasp_debug_html")
    ]
    if not viewer_records:
        viewer_records = [
            record
            for record in records
            if record.get("status") == "complete" and record.get("grasp_debug_html")
        ]
    viewer_cases = [
        {
            "case_id": str(record.get("case_id", "")),
            "scene": str(record.get("grasp_debug_html", "")),
            "part": str(record.get("incoming_part_id", "")),
            "orientation": str(record.get("orientation_id", "")),
            "placement": str(record.get("placement_id", "")),
            "grasp_count": int(record.get("floor_valid_candidates", 0)),
        }
        for record in viewer_records
    ]
    viewer_json = json.dumps(viewer_cases).replace("</", "<\\/")
    initial_scene = "" if not viewer_cases else html.escape(str(viewer_cases[0]["scene"]), quote=True)
    viewer = (
        "<section class='pose-viewer'><div class='viewer-head'>"
        "<div><h2>Failed pose grasp viewer</h2>"
        "<p>Select one part pose, then use ←/→ or the grasp buttons to inspect every grasp in its world scene.</p></div>"
        "<div id='graspProgress' class='progress'>Loading first grasp…</div></div>"
        "<div class='viewer-toolbar'>"
        "<button id='prevPose' type='button'>Previous pose</button>"
        "<select id='poseSelector' aria-label='Failed part pose'></select>"
        "<button id='nextPose' type='button'>Next pose</button>"
        "<span class='toolbar-separator'></span>"
        "<button id='prevGrasp' type='button'>← Previous grasp</button>"
        "<button id='nextGrasp' type='button'>Next grasp →</button>"
        "<a id='openScene' class='button-link' target='_blank' rel='noopener'>Open separately</a>"
        "</div><div id='poseInfo' class='pose-info'></div>"
        f"<iframe id='graspViewer' title='Selected failed pose grasp world scene' src='{initial_scene}'></iframe>"
        "</section>"
    )
    viewer_script = (
        "<script>"
        f"const viewerCases={viewer_json};"
        "const poseSelector=document.getElementById('poseSelector');"
        "const graspViewer=document.getElementById('graspViewer');"
        "const poseInfo=document.getElementById('poseInfo');"
        "const graspProgress=document.getElementById('graspProgress');"
        "const openScene=document.getElementById('openScene');"
        "let poseIndex=0;"
        "viewerCases.forEach((item,index)=>{const option=document.createElement('option');"
        "option.value=String(index);option.textContent=`part ${item.part} | ${item.orientation} | ${item.placement} | ${item.grasp_count} grasps`;"
        "poseSelector.appendChild(option);});"
        "function loadPose(index){if(!viewerCases.length)return;"
        "poseIndex=(index+viewerCases.length)%viewerCases.length;const item=viewerCases[poseIndex];"
        "poseSelector.value=String(poseIndex);poseInfo.textContent=`${poseIndex+1} / ${viewerCases.length} · ${item.case_id}`;"
        "graspProgress.textContent='Loading first grasp…';graspViewer.src=item.scene;openScene.href=item.scene;}"
        "function stepGrasp(delta){if(graspViewer.contentWindow)graspViewer.contentWindow.postMessage({type:'fabrica-grasp-step',delta},'*');}"
        "poseSelector.addEventListener('change',()=>loadPose(Number(poseSelector.value)));"
        "document.getElementById('prevPose').addEventListener('click',()=>loadPose(poseIndex-1));"
        "document.getElementById('nextPose').addEventListener('click',()=>loadPose(poseIndex+1));"
        "document.getElementById('prevGrasp').addEventListener('click',()=>stepGrasp(-1));"
        "document.getElementById('nextGrasp').addEventListener('click',()=>stepGrasp(1));"
        "window.addEventListener('keydown',event=>{const tag=(event.target&&event.target.tagName)||'';"
        "if(['SELECT','INPUT','TEXTAREA'].includes(tag))return;"
        "if(event.key==='ArrowLeft'){event.preventDefault();stepGrasp(-1);}"
        "if(event.key==='ArrowRight'){event.preventDefault();stepGrasp(1);}});"
        "window.addEventListener('message',event=>{if(event.source!==graspViewer.contentWindow||!event.data||event.data.type!=='fabrica-grasp-selection')return;"
        "graspProgress.textContent=`Grasp ${event.data.index+1} / ${event.data.total} · ${event.data.graspId} · ${event.data.status}`;});"
        "if(viewerCases.length)loadPose(0);else{poseInfo.textContent='No completed pose galleries available.';graspProgress.textContent='';}"
        "</script>"
    )
    matrix = dict(aggregate.get("outcome_matrix", {}))
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>Solo Pickup IK A/B</title><style>"
        "body{font:14px system-ui;background:#111820;color:#e8eef5;margin:0;padding:24px}"
        "h1{margin-top:0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}"
        ".card{background:#1b2631;border:1px solid #344454;border-radius:10px;padding:14px}.n{font-size:28px;font-weight:700}"
        ".pose-viewer{margin:20px 0;background:#18222c;border:1px solid #344454;border-radius:14px;padding:14px}"
        ".viewer-head{display:flex;justify-content:space-between;gap:18px;align-items:end}.viewer-head h2{margin:0 0 5px}.viewer-head p{margin:0;color:#aebdca}"
        ".progress{font:13px ui-monospace,monospace;color:#9ac8ff;text-align:right;max-width:48%}.viewer-toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0 8px}"
        ".viewer-toolbar select{min-width:440px;flex:1;background:#111820;color:#e8eef5;border:1px solid #45586a;border-radius:8px;padding:9px}"
        ".viewer-toolbar button,.button-link{background:#24313d;color:#e8eef5;border:1px solid #45586a;border-radius:8px;padding:9px 12px;text-decoration:none;cursor:pointer}"
        ".viewer-toolbar button:hover,.button-link:hover{border-color:#78b7ef}.toolbar-separator{width:1px;height:30px;background:#45586a;margin:0 3px}"
        ".pose-info{font:13px ui-monospace,monospace;color:#aebdca;margin:8px 0}#graspViewer{display:block;width:100%;height:min(820px,82vh);border:0;border-radius:10px;background:#fff}"
        "table{width:100%;border-collapse:collapse;margin-top:20px;background:#18222c}th,td{padding:8px;border-bottom:1px solid #33414e;text-align:left}"
        "th{position:sticky;top:0;background:#24313d}.pass{color:#58dda0}.fail{color:#ff7b85}code{color:#9ac8ff}"
        "@media(max-width:800px){body{padding:12px}.viewer-head{display:block}.progress{text-align:left;max-width:none;margin-top:8px}.viewer-toolbar select{min-width:100%}#graspViewer{height:76vh}}"
        "</style></head><body><h1>Solo Pickup IK A/B</h1>"
        "<p>A = production dual-arm exact preflight. B = each iiwa independently tries the holder/base grasp and "
        "incoming-part pickup; the passive robot and other object are ignored, while active-arm self/floor collision "
        "remains enforced. Incoming pickup includes lift; the holder sequence ends at contact like production.</p>"
        + viewer
        + "<div class='cards'>"
        f"<div class='card'><div>completed</div><div class='n'>{int(aggregate.get('completed', 0))}</div></div>"
        f"<div class='card'><div>dual pass</div><div class='n'>{int(aggregate.get('dual_success', 0))}</div></div>"
        f"<div class='card'><div>assigned inserter alone</div><div class='n'>{int(aggregate.get('assigned_inserter_solo_success', 0))}</div></div>"
        f"<div class='card'><div>both assigned pickups alone</div><div class='n'>{int(aggregate.get('assigned_pickups_solo_success', 0))}</div></div>"
        f"<div class='card'><div>either arm assignment alone</div><div class='n'>{int(aggregate.get('either_assignment_solo_success', 0))}</div></div>"
        f"<div class='card'><div>dual fail → solo pass</div><div class='n'>{int(matrix.get('dual_fail__solo_pass', 0))}</div></div>"
        "</div><table><thead><tr><th>case</th><th>grasp scene</th><th>status</th><th>dual</th><th>lbr_one incoming</th><th>checked</th>"
        "<th>lbr_two incoming</th><th>checked</th><th>lbr_one holder</th><th>lbr_two holder</th>"
        "<th>assigned pickups</th><th>either assignment</th><th>orientation</th><th>placement</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + viewer_script
        + "</body></html>",
        encoding="utf-8",
    )


def _checkpoint(
    output_dir: Path,
    *,
    baseline_summary: Path,
    records: list[dict[str, object]],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "solo_pickup_ik_ab_benchmark",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_summary": str(baseline_summary),
        "scope": (
            "A is the existing full dual-arm result. B independently solves both holder/base and "
            "incoming-part pickup for each robot, ignoring passive-robot contacts while retaining "
            "active-arm self/work-surface collision."
        ),
        "aggregate": _aggregate(records),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_html(output_dir / "index.html", payload)


def main() -> int:
    args = _parse_args()
    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable; source ROS2 and ros2_ws first.")
    if args.ik_timeout_s <= 0.0 or args.ik_candidates < 1 or args.ik_beam_width < 1 or args.approach_steps < 1:
        raise ValueError("IK timeout, candidates, beam width, and approach steps must be positive.")
    baseline_path = args.baseline_summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    baseline = _read_json(baseline_path)
    baseline_records = [
        dict(value)
        for value in baseline.get("records", [])
        if isinstance(value, dict) and value.get("status") in {"success", "failed"}
    ]
    if int(args.limit_cases) > 0:
        baseline_records = baseline_records[: int(args.limit_cases)]
    existing: dict[str, dict[str, object]] = {}
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not args.no_resume:
        existing_payload = _read_json(summary_path)
        existing = {
            str(record["case_id"]): dict(record)
            for record in existing_payload.get("records", [])
            if isinstance(record, dict) and record.get("status") == "complete"
        }
    records = [existing.get(str(record["case_id"]), {"case_id": str(record["case_id"]), "status": "pending"}) for record in baseline_records]
    _checkpoint(output_dir, baseline_summary=baseline_path, records=records)

    rclpy.init()
    commanders: dict[str, MoveItPoseCommander] = {}
    holder_result_cache: dict[tuple[object, ...], dict[str, object]] = {}
    holder_candidate_cache: dict[tuple[object, ...], tuple[object, int]] = {}
    try:
        for robot in ROBOT_NAMES:
            spec = ARM_SPEC_BY_ROBOT[robot]
            commander = MoveItPoseCommander(
                MoveItPoseCommanderConfig(
                    planning_group=str(spec["planning_group"]),
                    pose_link=str(spec["pose_link"]),
                    joint_names=tuple(str(value) for value in spec["joint_names"]),
                    moveit_namespace=str(args.moveit_namespace),
                    ik_timeout_s=float(args.ik_timeout_s),
                    avoid_collisions=False,
                ),
                node_name=f"solo_pickup_ab_{robot}",
            )
            commander.wait_for_moveit(require_execute=False)
            commanders[robot] = commander
        floor_values = {round(float(record["floor_z"]), 9) for record in baseline_records}
        if len(floor_values) != 1:
            raise ValueError("The current A/B runner requires one shared floor_z across its cases.")
        floor_z = next(iter(floor_values))
        ok, message = commanders["lbr_one"].apply_planning_scene_obstacles(
            [_work_surface(floor_z=floor_z)],
            default_frame_id="base_link",
        )
        if not ok:
            raise RuntimeError(f"Could not apply solo pickup work surface: {message}")

        for index, baseline_record in enumerate(baseline_records):
            case_id = str(baseline_record["case_id"])
            if case_id in existing:
                print(f"[SOLO-IK-AB] {index + 1}/{len(baseline_records)} resume {case_id}", flush=True)
                continue
            print(f"[SOLO-IK-AB] {index + 1}/{len(baseline_records)} {case_id}", flush=True)
            case_started = time.monotonic()
            result: dict[str, object] = {
                "case_id": case_id,
                "status": "running",
                "dual_success": bool(baseline_record.get("success", False)),
                "baseline_inserter_arm": baseline_record.get("inserter_arm"),
                "baseline_holder_arm": baseline_record.get("holder_arm"),
                "assembly": baseline_record.get("assembly"),
                "incoming_part_id": baseline_record.get("incoming_part_id"),
                "step_id": baseline_record.get("step_id"),
                "placement_id": baseline_record.get("placement_id"),
                "orientation_id": baseline_record.get("orientation_id"),
                "pickup_x": baseline_record.get("pickup_x"),
                "pickup_y": baseline_record.get("pickup_y"),
                "pickup_roll_deg": baseline_record.get("pickup_roll_deg"),
                "pickup_pitch_deg": baseline_record.get("pickup_pitch_deg"),
                "pickup_yaw_deg": baseline_record.get("pickup_yaw_deg"),
            }
            records[index] = result
            _checkpoint(output_dir, baseline_summary=baseline_path, records=records)
            try:
                incoming_case = _incoming_case_candidates(baseline_record)
                solo = {
                    robot: _run_arm(
                        commanders[robot],
                        active_robot=robot,
                        grasps=incoming_case.world_grasps,
                        args=args,
                        exhaustive=not bool(args.no_grasp_debug),
                    )
                    for robot in ROBOT_NAMES
                }
                grasp_debug_relative = Path("cases") / case_id / "incoming_grasps.html"
                if not args.no_grasp_debug:
                    _write_case_grasp_debug(
                        output_dir / grasp_debug_relative,
                        baseline_record=baseline_record,
                        case=incoming_case,
                        solo=solo,
                    )
                holder_case_key = (
                    str(baseline_record["assembly"]),
                    str(baseline_record["step_id"]),
                    float(baseline_record["assembly_x"]),
                    float(baseline_record["assembly_y"]),
                    float(baseline_record["assembly_z"]),
                    float(baseline_record["assembly_yaw_deg"]),
                )
                holder_cached = holder_candidate_cache.get(holder_case_key)
                if holder_cached is None:
                    holder_grasps, holder_candidates_checked = _holder_step_candidates(baseline_record)
                    holder_candidate_cache[holder_case_key] = (holder_grasps, holder_candidates_checked)
                else:
                    holder_grasps, holder_candidates_checked = holder_cached
                solo_holder: dict[str, dict[str, object]] = {}
                for robot in ROBOT_NAMES:
                    holder_result_key = holder_case_key + (robot,)
                    holder_result = holder_result_cache.get(holder_result_key)
                    if holder_result is None:
                        holder_result = _run_arm(
                            commanders[robot],
                            active_robot=robot,
                            grasps=holder_grasps,
                            args=args,
                            target_prefix="holder",
                            lift_m=None,
                        )
                        holder_result_cache[holder_result_key] = holder_result
                        solo_holder[robot] = dict(holder_result, cache_hit=False)
                    else:
                        solo_holder[robot] = dict(
                            holder_result,
                            cache_hit=True,
                            source_duration_s=float(holder_result.get("duration_s", 0.0)),
                            duration_s=0.0,
                        )
                baseline_holder = str(baseline_record["holder_arm"])
                baseline_inserter = str(baseline_record["inserter_arm"])
                assigned_roles_solo_success = bool(
                    solo_holder[baseline_holder]["success"] and solo[baseline_inserter]["success"]
                )
                swapped_roles_solo_success = bool(
                    solo_holder[baseline_inserter]["success"] and solo[baseline_holder]["success"]
                )
                result.update(
                    {
                        "status": "complete",
                        "floor_candidates_checked": incoming_case.floor_candidates_checked,
                        "floor_valid_candidates": len(incoming_case.world_grasps),
                        "solo": solo,
                        "either_solo_success": any(bool(solo[robot]["success"]) for robot in ROBOT_NAMES),
                        "grasp_debug_html": "" if args.no_grasp_debug else grasp_debug_relative.as_posix(),
                        "holder_candidates_in_source": holder_candidates_checked,
                        "holder_unary_valid_candidates": len(holder_grasps),
                        "solo_holder": solo_holder,
                        "assigned_roles_solo_success": assigned_roles_solo_success,
                        "swapped_roles_solo_success": swapped_roles_solo_success,
                        "either_assignment_solo_success": bool(
                            assigned_roles_solo_success or swapped_roles_solo_success
                        ),
                        "duration_s": time.monotonic() - case_started,
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "status": "error",
                        "message": f"{type(exc).__name__}: {exc}",
                        "duration_s": time.monotonic() - case_started,
                    }
                )
            _checkpoint(output_dir, baseline_summary=baseline_path, records=records)
    except KeyboardInterrupt:
        _checkpoint(output_dir, baseline_summary=baseline_path, records=records)
        print(f"[SOLO-IK-AB] Interrupted; partial report saved at {output_dir / 'index.html'}", flush=True)
        return 130
    finally:
        for commander in commanders.values():
            commander.destroy_node()
        rclpy.shutdown()

    aggregate = _aggregate(records)
    print(
        f"[SOLO-IK-AB] complete={aggregate['completed']} dual={aggregate['dual_success']} "
        f"assigned_pickups={aggregate['assigned_pickups_solo_success']} "
        f"either_assignment={aggregate['either_assignment_solo_success']} "
        f"report={output_dir / 'index.html'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
