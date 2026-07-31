"""Guarded dual-KUKA execution of a saved holder/inserter task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

from grasp_planning.pipeline.dual_robot_simple_sim import (
    DEFAULT_FLOOR_Z_WORLD_M,
)
from grasp_planning.ros2.moveit_pose_commander import (
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    PoseTarget,
    rclpy,
)
from grasp_planning.ros2.trigger_service_gripper_client import (
    TriggerServiceGripperClient,
)
from grasp_planning.start_poses import KUKA_MOVEIT_ARM_START_JOINT_VALUES

ROLE_SPECS = {
    "holder": {
        "robot": "lbr_one",
        "planning_group": "arm_one",
        "pose_link": "lbr_one_gripper_tcp",
        "joint_names": tuple(f"lbr_one_A{index}" for index in range(1, 8)),
    },
    "inserter": {
        "robot": "lbr_two",
        "planning_group": "arm_two",
        "pose_link": "lbr_two_gripper_tcp",
        "joint_names": tuple(f"lbr_two_A{index}" for index in range(1, 8)),
    },
}
MOTION_SEQUENCE = (
    ("holder", "holder_pregrasp"),
    ("holder", "holder_grasp"),
    ("inserter", "inserter_pickup_pregrasp"),
    ("inserter", "inserter_pickup_grasp"),
    ("inserter", "inserter_pickup_lift"),
    ("inserter", "inserter_above_preinsertion"),
    ("inserter", "inserter_preinsertion"),
)
STOP_AFTER_CHOICES = tuple(target_name for _, target_name in MOTION_SEQUENCE)
GRIPPER_CLOSE_AFTER = {
    "holder_grasp": "holder",
    "inserter_pickup_grasp": "inserter",
}


def _motion_sequence_through(
    stop_after: str,
) -> tuple[tuple[str, str], ...]:
    if stop_after not in STOP_AFTER_CHOICES:
        raise ValueError(f"stop_after must be one of {STOP_AFTER_CHOICES}; got {stop_after!r}.")
    stop_index = STOP_AFTER_CHOICES.index(stop_after)
    return MOTION_SEQUENCE[: stop_index + 1]


@dataclass(frozen=True)
class DualRealExecutionConfig:
    moveit_namespace: str = "/lbr_dual_arm"
    frame_id: str = "base_link"
    wait_for_moveit_timeout_s: float = 20.0
    ik_timeout_s: float = 2.0
    planning_time_s: float = 8.0
    num_planning_attempts: int = 8
    velocity_scale: float = 0.05
    acceleration_scale: float = 0.05
    execute_timeout_s: float = 120.0
    post_execute_sleep_s: float = 0.5
    execute: bool = False
    require_confirmation: bool = True
    allow_objectless_planning: bool = False
    stop_after: str = "holder_pregrasp"
    grippers_enabled: bool = True
    gripper_timeout_s: float = 10.0
    grasp_settle_time_s: float = 0.5
    holder_gripper_open_service: str = "/lbr_one/gripper_controller/open"
    holder_gripper_close_service: str = "/lbr_one/gripper_controller/close"
    holder_gripper_stop_service: str = "/lbr_one/gripper_controller/stop"
    inserter_gripper_open_service: str = "/lbr_two/gripper_controller/open"
    inserter_gripper_close_service: str = "/lbr_two/gripper_controller/close"
    inserter_gripper_stop_service: str = "/lbr_two/gripper_controller/stop"


@dataclass(frozen=True)
class DualRealExecutionResult:
    success: bool
    status: str
    message: str
    pair_id: str
    last_completed_phase: str
    attempt_artifact_path: Path


def _validate_dual_plan_payload(
    payload: Mapping[str, object],
    *,
    context: str,
) -> None:
    if payload.get("kind") != "dual_robot_simple_sim_task":
        raise ValueError(f"{context} kind must be 'dual_robot_simple_sim_task'; got {payload.get('kind')!r}.")
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise ValueError(f"{context} is missing its targets object.")
    for _, target_name in MOTION_SEQUENCE:
        raw_target = targets.get(target_name)
        if not isinstance(raw_target, dict):
            raise ValueError(f"{context} is missing target '{target_name}'.")
        _target_from_payload(raw_target, frame_id="base_link")

    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise ValueError(f"{context} is missing its roles object.")
    for role, expected in ROLE_SPECS.items():
        actual = roles.get(role)
        if not isinstance(actual, dict):
            raise ValueError(f"{context} is missing role '{role}'.")
        for key in ("robot", "planning_group", "tcp_link"):
            expected_key = "pose_link" if key == "tcp_link" else key
            if str(actual.get(key, "")) != str(expected[expected_key]):
                raise ValueError(
                    f"{context} role '{role}' has {key}={actual.get(key)!r}; expected {expected[expected_key]!r}."
                )


def _ranked_candidate_plans(
    plan: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    raw_candidates = plan.get("ranked_pair_candidates")
    if raw_candidates is None:
        return (dict(plan),)
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("ranked_pair_candidates must be a non-empty list when present.")

    candidates: list[dict[str, object]] = []
    seen_candidate_ids: set[str] = set()
    previous_score = float("inf")
    for rank, raw_candidate in enumerate(raw_candidates, start=1):
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"Ranked real candidate {rank} must be a JSON object.")
        candidate = dict(raw_candidate)
        _validate_dual_plan_payload(
            candidate,
            context=f"Ranked real candidate {rank}",
        )
        pair_id = str(candidate.get("pair_id", ""))
        if not pair_id:
            raise ValueError(f"Ranked real candidate {rank} has no pair_id.")
        candidate_id = str(candidate.get("execution_candidate_id", pair_id))
        if not candidate_id:
            raise ValueError(f"Ranked real candidate {rank} has no execution identity.")
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"Ranked real execution candidate '{candidate_id}' is duplicated.")
        seen_candidate_ids.add(candidate_id)
        score = float(
            candidate.get(
                "selection_score",
                candidate.get("pair_score", 0.0),
            )
        )
        if score > previous_score + 1.0e-12:
            raise ValueError(
                f"ranked_pair_candidates are not in descending score order at rank {rank}: {score} > {previous_score}."
            )
        previous_score = score
        candidates.append(candidate)

    first_candidate_id = str(
        candidates[0].get(
            "execution_candidate_id",
            candidates[0].get("pair_id", ""),
        )
    )
    top_candidate_id = str(plan.get("execution_candidate_id", plan.get("pair_id", "")))
    if first_candidate_id != top_candidate_id:
        raise ValueError("The top-level real task must match ranked candidate 1.")
    return tuple(candidates)


def load_and_validate_dual_plan(plan_json: Path) -> dict[str, object]:
    payload = json.loads(plan_json.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dual plan JSON must contain an object.")
    _validate_dual_plan_payload(payload, context="Dual plan")
    _ranked_candidate_plans(payload)
    return payload


def _target_from_payload(raw: Mapping[str, object], *, frame_id: str) -> PoseTarget:
    position = raw.get("position_world_m")
    orientation = raw.get("orientation_xyzw_world")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("Target position_world_m must contain three values.")
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
        raise ValueError("Target orientation_xyzw_world must contain four values.")
    return PoseTarget.from_quaternion(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        quaternion_xyzw=tuple(float(value) for value in orientation),
        frame_id=frame_id,
    )


def _work_surface_obstacle(plan: Mapping[str, object]) -> dict[str, object]:
    moveit = plan.get("moveit")
    if isinstance(moveit, dict):
        configured = moveit.get("work_surface")
        if isinstance(configured, dict):
            obstacle = dict(configured)
            obstacle["id"] = "dual_real_work_surface"
            return obstacle
    layout = plan.get("layout")
    floor_z = DEFAULT_FLOOR_Z_WORLD_M
    if isinstance(layout, dict):
        floor_z = float(
            layout.get(
                "pickup_floor_z_world_m",
                DEFAULT_FLOOR_Z_WORLD_M,
            )
        )
    return {
        "id": "dual_real_work_surface",
        "type": "box",
        "frame_id": "base_link",
        "size_m": [1.20, 1.40, 0.05],
        "xyz": [0.75, 0.0, floor_z - 0.025],
        "rpy": [0.0, 0.0, 0.0],
    }


def _pregrasp_aabb_obstacles(
    plan: Mapping[str, object],
    *,
    target_name: str,
) -> list[dict[str, object]]:
    moveit = plan.get("moveit")
    if not isinstance(moveit, dict):
        return []
    raw_geometry = moveit.get("pregrasp_aabb_collision_geometry")
    if not isinstance(raw_geometry, dict):
        return []
    raw_obstacles = raw_geometry.get("obstacles")
    raw_schedule = raw_geometry.get("active_by_target")
    if not isinstance(raw_obstacles, dict) or not isinstance(raw_schedule, dict):
        return []
    active_keys = raw_schedule.get(target_name, ())
    if not isinstance(active_keys, (list, tuple)):
        raise ValueError(f"Pregrasp AABB schedule for '{target_name}' must be a list.")
    obstacles: list[dict[str, object]] = []
    for key in active_keys:
        obstacle = raw_obstacles.get(str(key))
        if not isinstance(obstacle, dict):
            raise ValueError(f"Pregrasp AABB obstacle '{key}' is missing from the plan.")
        obstacles.append(dict(obstacle))
    return obstacles


def _make_commander(
    *,
    role: str,
    config: DualRealExecutionConfig,
):
    spec = ROLE_SPECS[role]
    return MoveItPoseCommander(
        MoveItPoseCommanderConfig(
            planning_group=str(spec["planning_group"]),
            pose_link=str(spec["pose_link"]),
            joint_names=tuple(spec["joint_names"]),
            moveit_namespace=str(config.moveit_namespace),
            wait_for_moveit_timeout_s=float(config.wait_for_moveit_timeout_s),
            ik_timeout_s=float(config.ik_timeout_s),
            fk_timeout_s=float(config.ik_timeout_s),
            planning_time_s=float(config.planning_time_s),
            num_planning_attempts=int(config.num_planning_attempts),
            velocity_scale=float(config.velocity_scale),
            acceleration_scale=float(config.acceleration_scale),
            execute_timeout_s=float(config.execute_timeout_s),
            post_execute_sleep_s=float(config.post_execute_sleep_s),
            avoid_collisions=True,
        ),
        node_name=f"dual_real_{role}",
    )


def _make_gripper(
    *,
    role: str,
    commander,
    config: DualRealExecutionConfig,
):
    prefix = "holder" if role == "holder" else "inserter"
    return TriggerServiceGripperClient(
        commander,
        open_service_name=str(getattr(config, f"{prefix}_gripper_open_service")),
        close_service_name=str(getattr(config, f"{prefix}_gripper_close_service")),
        stop_service_name=str(getattr(config, f"{prefix}_gripper_stop_service")),
        timeout_s=float(config.gripper_timeout_s),
        grasp_settle_time_s=float(config.grasp_settle_time_s),
    )


def _stop_grippers(grippers: Mapping[str, object], *, reason: str) -> None:
    for role, gripper in grippers.items():
        stop = getattr(gripper, "stop", None)
        if not callable(stop):
            continue
        try:
            ok, message = stop()
            print(
                f"[DUAL-REAL] {role} gripper stop after {reason}: success={bool(ok)} message={message}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[WARN]: {role} gripper stop after {reason} failed: {exc!r}",
                flush=True,
            )


def _confirmation_text(
    *,
    plan_json: Path,
    plan: Mapping[str, object],
    config: DualRealExecutionConfig,
) -> str:
    grasps = plan.get("grasps")
    holder_id = ""
    inserter_id = ""
    if isinstance(grasps, dict):
        holder = grasps.get("holder")
        inserter = grasps.get("inserter_pickup")
        if isinstance(holder, dict):
            holder_id = str(holder.get("grasp_id", ""))
        if isinstance(inserter, dict):
            inserter_id = str(inserter.get("grasp_id", ""))
    return (
        "DUAL REAL-ROBOT EXECUTION REQUESTED\n"
        f"  plan:             {plan_json}\n"
        f"  pair:             {plan.get('pair_id', '')}\n"
        f"  holder:           lbr_one / {holder_id}\n"
        f"  inserter:         lbr_two / {inserter_id}\n"
        f"  stop_after:       {config.stop_after}\n"
        f"  velocity_scale:   {config.velocity_scale:.3f}\n"
        "  collision scene:  both robots + table + temporary pregrasp AABBs; "
        "exact Fabrica meshes omitted\n"
        "Verify the physical 840 mm base transform, clear the cell, keep both "
        "E-stops reachable, and type 'yes' to continue: "
    )


def _preflight_targets(
    *,
    plan: Mapping[str, object],
    commanders: Mapping[str, object],
    frame_id: str,
    record: Callable[..., None],
    role_filter: str | None = None,
    stop_on_failure: bool = False,
    candidate_rank: int | None = None,
    pair_id: str = "",
    stop_after: str = "inserter_preinsertion",
) -> bool:
    targets = plan["targets"]
    assert isinstance(targets, dict)
    success = True
    for role, target_name in _motion_sequence_through(stop_after):
        if role_filter is not None and role != role_filter:
            continue
        target = _target_from_payload(targets[target_name], frame_id=frame_id)
        joints, message = commanders[role].compute_ik(target)
        if joints is None:
            fallback_joints, fallback_message = commanders[role].compute_ik(
                target,
                seed_joint_positions=KUKA_MOVEIT_ARM_START_JOINT_VALUES,
            )
            if fallback_joints is not None:
                joints = fallback_joints
                message = f"live-state IK failed ({message}); start-seeded alternate IK succeeded"
            else:
                message = f"live-state IK failed ({message}); start-seeded alternate IK failed ({fallback_message})"
        ok = joints is not None
        record_args: dict[str, object] = {
            "name": f"preflight_{target_name}",
            "role": role,
            "ok": ok,
            "message": message,
            "target": target,
        }
        if candidate_rank is not None:
            record_args["candidate_rank"] = int(candidate_rank)
        if pair_id:
            record_args["pair_id"] = str(pair_id)
        record(
            **record_args,
        )
        if not ok:
            success = False
            if stop_on_failure:
                break
    return success


def _candidate_grasp_id(
    plan: Mapping[str, object],
    *,
    role: str,
) -> str:
    grasps = plan.get("grasps")
    if not isinstance(grasps, dict):
        raise ValueError("Ranked real candidate is missing grasps.")
    grasp_key = "holder" if role == "holder" else "inserter_pickup"
    grasp = grasps.get(grasp_key)
    if not isinstance(grasp, dict):
        raise ValueError(f"Ranked real candidate is missing grasp '{grasp_key}'.")
    grasp_id = str(grasp.get("grasp_id", ""))
    if not grasp_id:
        raise ValueError(f"Ranked real candidate grasp '{grasp_key}' has no grasp_id.")
    return grasp_id


def _role_target_signature(
    plan: Mapping[str, object],
    *,
    role: str,
    stop_after: str,
) -> tuple[float, ...]:
    """Return the exact target signature covered by a cached role preflight."""

    targets = plan.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("Ranked real candidate is missing targets.")
    values: list[float] = []
    for target_role, target_name in _motion_sequence_through(stop_after):
        if target_role != role:
            continue
        raw_target = targets.get(target_name)
        if not isinstance(raw_target, dict):
            raise ValueError(f"Ranked real candidate is missing target '{target_name}'.")
        position = raw_target.get("position_world_m")
        orientation = raw_target.get("orientation_xyzw_world")
        if not isinstance(position, (list, tuple)) or not isinstance(
            orientation,
            (list, tuple),
        ):
            raise ValueError(f"Target '{target_name}' has an invalid pose.")
        values.extend(round(float(value), 9) for value in position)
        values.extend(round(float(value), 9) for value in orientation)
    return tuple(values)


def _select_ranked_preflight_candidate(
    *,
    plan: Mapping[str, object],
    commanders: Mapping[str, object],
    frame_id: str,
    record: Callable[..., None],
    stop_after: str = "inserter_preinsertion",
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Select the first strict-score-order pair passing all target IK checks."""

    candidates = _ranked_candidate_plans(plan)
    role_cache: dict[
        tuple[str, str, tuple[float, ...]],
        tuple[bool, str],
    ] = {}
    candidate_records: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    selected_rank: int | None = None

    for rank, candidate in enumerate(candidates, start=1):
        pair_id = str(candidate["pair_id"])
        candidate_id = str(candidate.get("execution_candidate_id", pair_id))
        score = float(
            candidate.get(
                "selection_score",
                candidate.get("pair_score", 0.0),
            )
        )
        print(
            f"[DUAL-REAL] Preflight candidate {rank}/{len(candidates)} "
            f"candidate={candidate_id} pair={pair_id} "
            f"selection_score={score:.4f}",
            flush=True,
        )
        role_records: dict[str, object] = {}
        candidate_ok = True
        active_roles = tuple(dict.fromkeys(role for role, _ in _motion_sequence_through(stop_after)))
        for role in active_roles:
            grasp_id = _candidate_grasp_id(candidate, role=role)
            cache_key = (
                role,
                grasp_id,
                _role_target_signature(
                    candidate,
                    role=role,
                    stop_after=stop_after,
                ),
            )
            cache_hit = cache_key in role_cache
            if cache_hit:
                role_ok, failure = role_cache[cache_key]
                record(
                    name=f"preflight_{role}_cached",
                    role=role,
                    ok=role_ok,
                    message=(
                        f"Reused cached {'success' if role_ok else 'failure'} "
                        f"for grasp {grasp_id}" + (f": {failure}" if failure else ".")
                    ),
                    candidate_rank=rank,
                    pair_id=pair_id,
                )
            else:
                role_ok = _preflight_targets(
                    plan=candidate,
                    commanders=commanders,
                    frame_id=frame_id,
                    record=record,
                    role_filter=role,
                    stop_on_failure=True,
                    candidate_rank=rank,
                    pair_id=pair_id,
                    stop_after=stop_after,
                )
                failure = "" if role_ok else "target IK failed"
                role_cache[cache_key] = (role_ok, failure)
            role_records[role] = {
                "grasp_id": grasp_id,
                "cache_hit": cache_hit,
                "ok": role_ok,
                "failure": failure,
            }
            if not role_ok:
                candidate_ok = False
                break

        candidate_record = {
            "rank": rank,
            "pair_id": pair_id,
            "execution_candidate_id": candidate_id,
            "transition_id": str(candidate.get("transition_id", "")),
            "selection_score": score,
            "ok": candidate_ok,
            "roles": role_records,
        }
        candidate_records.append(candidate_record)
        record(
            name="candidate_preflight",
            role="shared",
            ok=candidate_ok,
            message=(
                f"Selected ranked pair {pair_id}."
                if candidate_ok
                else f"Rejected ranked pair {pair_id}; trying the next pair."
            ),
            candidate_rank=rank,
            pair_id=pair_id,
        )
        if candidate_ok:
            selected = candidate
            selected_rank = rank
            break

    summary = {
        "policy": "strict_score_order_collision_aware_ik_before_motion",
        "candidate_count": len(candidates),
        "candidates_checked": len(candidate_records),
        "selected_rank": selected_rank,
        "selected_pair_id": (str(selected.get("pair_id", "")) if selected is not None else ""),
        "selected_execution_candidate_id": (
            str(
                selected.get(
                    "execution_candidate_id",
                    selected.get("pair_id", ""),
                )
            )
            if selected is not None
            else ""
        ),
        "selected_transition_id": (str(selected.get("transition_id", "")) if selected is not None else ""),
        "records": candidate_records,
        "cached_role_grasp_count": len(role_cache),
        "stop_after": stop_after,
    }
    return selected, summary


def _execute_sequence(
    *,
    plan: Mapping[str, object],
    commanders: Mapping[str, object],
    grippers: Mapping[str, object],
    config: DualRealExecutionConfig,
    record: Callable[..., None],
) -> tuple[bool, str, str]:
    targets = plan["targets"]
    assert isinstance(targets, dict)
    last_completed = ""

    for role in ("holder", "inserter"):
        if role not in grippers:
            continue
        ok, message = grippers[role].open(width=0.06)
        record(name=f"open_{role}_gripper", role=role, ok=ok, message=message)
        if not ok:
            return False, f"{role}_gripper_open_failed", last_completed

    for role, target_name in MOTION_SEQUENCE:
        target = _target_from_payload(
            targets[target_name],
            frame_id=str(config.frame_id),
        )
        active_aabbs = _pregrasp_aabb_obstacles(
            plan,
            target_name=target_name,
        )
        if active_aabbs:
            ok, message = commanders["holder"].apply_planning_scene_obstacles(
                active_aabbs,
                default_frame_id=str(config.frame_id),
            )
            record(
                name=f"apply_{target_name}_aabbs",
                role="shared",
                ok=ok,
                message=message,
            )
            if not ok:
                return (
                    False,
                    f"{target_name}_aabb_apply_failed",
                    last_completed,
                )
        try:
            ok, message = commanders[role].move_to_pose(
                target,
                label=target_name,
                execute=True,
            )
            record(
                name=target_name,
                role=role,
                ok=ok,
                message=message,
                target=target,
            )
        finally:
            if active_aabbs:
                remove_ok, remove_message = commanders["holder"].remove_planning_scene_obstacles(
                    [str(obstacle["id"]) for obstacle in active_aabbs],
                    default_frame_id=str(config.frame_id),
                )
                record(
                    name=f"remove_{target_name}_aabbs",
                    role="shared",
                    ok=remove_ok,
                    message=remove_message,
                )
                if not remove_ok:
                    raise RuntimeError(f"Could not remove temporary AABBs after {target_name}: {remove_message}")
        if not ok:
            return False, f"{target_name}_failed", last_completed
        last_completed = target_name

        close_role = GRIPPER_CLOSE_AFTER.get(target_name)
        if close_role is not None and close_role in grippers:
            grasps = plan["grasps"]
            assert isinstance(grasps, dict)
            grasp_key = "holder" if close_role == "holder" else "inserter_pickup"
            raw_grasp = grasps[grasp_key]
            assert isinstance(raw_grasp, dict)
            width = float(raw_grasp["jaw_width_m"])
            ok, message = grippers[close_role].close(width=width)
            record(
                name=f"close_{close_role}_gripper",
                role=close_role,
                ok=ok,
                message=message,
            )
            if not ok:
                return False, f"{close_role}_gripper_close_failed", last_completed

        if target_name == config.stop_after:
            return True, f"stopped_at_{target_name}", last_completed

    return True, "completed", last_completed


def _write_attempt(
    *,
    output_path: Path,
    plan_json: Path,
    plan: Mapping[str, object],
    config: DualRealExecutionConfig,
    result: DualRealExecutionResult,
    steps: list[dict[str, object]],
    pair_selection: Mapping[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "kind": "dual_robot_real_execution_attempt",
        "input_plan_json": str(plan_json),
        "pair_id": str(result.pair_id),
        "execution_candidate_id": str(plan.get("execution_candidate_id", result.pair_id)),
        "transition_id": str(plan.get("transition_id", "")),
        "pair_selection": dict(pair_selection),
        "config": {
            **asdict(config),
            "require_confirmation": bool(config.require_confirmation),
        },
        "steps": steps,
        "result": {
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "last_completed_phase": result.last_completed_phase,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def execute_dual_real_plan(
    *,
    plan_json: Path,
    attempt_artifact_path: Path,
    config: DualRealExecutionConfig,
) -> DualRealExecutionResult:
    if config.stop_after not in STOP_AFTER_CHOICES:
        raise ValueError(f"stop_after must be one of {STOP_AFTER_CHOICES}; got {config.stop_after!r}.")
    if config.execute and not config.allow_objectless_planning:
        raise RuntimeError(
            "Hardware execution uses a table-and-robots MoveIt scene without "
            "Fabrica object meshes. Pass --allow-objectless-planning only after "
            "checking the real placement and approach paths."
        )
    if config.execute and not config.grippers_enabled and config.stop_after != "holder_pregrasp":
        raise RuntimeError("Execution without gripper control is limited to stop_after=holder_pregrasp.")
    if rclpy is None:
        raise RuntimeError(
            "ROS2 MoveIt dependencies are unavailable. Source ROS2, lbr-stack, "
            "and this repository's ros2_ws overlay first."
        )

    plan_json = plan_json.expanduser().resolve()
    output_path = attempt_artifact_path.expanduser().resolve()
    source_plan = load_and_validate_dual_plan(plan_json)
    execution_plan = source_plan
    pair_id = ""
    pair_selection: dict[str, object] = {
        "policy": "strict_score_order_collision_aware_ik_before_motion",
        "candidate_count": len(_ranked_candidate_plans(source_plan)),
        "candidates_checked": 0,
        "selected_rank": None,
        "selected_pair_id": "",
        "records": [],
        "cached_role_grasp_count": 0,
    }
    steps: list[dict[str, object]] = []
    commanders: dict[str, object] = {}
    grippers: dict[str, object] = {}
    initialized_here = False
    result = DualRealExecutionResult(
        success=False,
        status="not_started",
        message="Execution did not start.",
        pair_id=pair_id,
        last_completed_phase="",
        attempt_artifact_path=output_path,
    )

    def _record(
        *,
        name: str,
        role: str,
        ok: bool,
        message: str,
        target: PoseTarget | None = None,
        candidate_rank: int | None = None,
        pair_id: str = "",
    ) -> None:
        entry: dict[str, object] = {
            "name": name,
            "role": role,
            "ok": bool(ok),
            "message": str(message),
        }
        if candidate_rank is not None:
            entry["candidate_rank"] = int(candidate_rank)
        if pair_id:
            entry["pair_id"] = str(pair_id)
        if target is not None:
            entry["target_pose"] = {
                "frame_id": target.frame_id,
                "position_xyz": list(target.position_xyz),
                "orientation_xyzw": list(target.orientation_xyzw),
            }
        steps.append(entry)
        candidate_label = f" candidate={candidate_rank} pair={pair_id}" if candidate_rank is not None else ""
        print(
            f"[DUAL-REAL] {name}:{candidate_label} {'ok' if ok else 'failed'} {message}",
            flush=True,
        )

    try:
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True
        commanders = {role: _make_commander(role=role, config=config) for role in ROLE_SPECS}
        for commander in commanders.values():
            commander.wait_for_moveit(require_execute=bool(config.execute))

        obstacle = _work_surface_obstacle(source_plan)
        ok, message = commanders["holder"].apply_planning_scene_obstacles(
            [obstacle],
            default_frame_id=str(config.frame_id),
        )
        _record(
            name="apply_work_surface",
            role="shared",
            ok=ok,
            message=message,
        )
        if not ok:
            raise RuntimeError(message)

        selected_plan, pair_selection = _select_ranked_preflight_candidate(
            plan=source_plan,
            commanders=commanders,
            frame_id=str(config.frame_id),
            record=_record,
            stop_after=str(config.stop_after),
        )
        if selected_plan is None:
            result = DualRealExecutionResult(
                False,
                "ik_preflight_failed",
                (
                    "No ranked grasp pair passed collision-aware target IK "
                    f"preflight after checking "
                    f"{pair_selection['candidates_checked']} candidate(s); "
                    "no hardware motion was started."
                ),
                "",
                "",
                output_path,
            )
            return result
        execution_plan = selected_plan
        pair_id = str(execution_plan.get("pair_id", ""))
        if not config.execute:
            result = DualRealExecutionResult(
                True,
                "preflight_ok",
                (f"Ranked pair {pair_id} passed all target IK checks; hardware execution was not requested."),
                pair_id,
                "",
                output_path,
            )
            return result

        if config.require_confirmation:
            reply = input(
                _confirmation_text(
                    plan_json=plan_json,
                    plan=execution_plan,
                    config=config,
                )
            )
            if reply.strip().lower() not in {"y", "yes"}:
                result = DualRealExecutionResult(
                    False,
                    "aborted",
                    "Hardware execution was aborted at the confirmation prompt.",
                    pair_id,
                    "",
                    output_path,
                )
                return result

        if config.grippers_enabled:
            grippers = {
                role: _make_gripper(
                    role=role,
                    commander=commanders[role],
                    config=config,
                )
                for role in ROLE_SPECS
            }
            for role, gripper in grippers.items():
                gripper.wait_for_server(timeout_s=float(config.wait_for_moveit_timeout_s))
                _record(
                    name=f"wait_for_{role}_gripper",
                    role=role,
                    ok=True,
                    message="Open/close/stop Trigger services are available.",
                )

        success, status, last_completed = _execute_sequence(
            plan=execution_plan,
            commanders=commanders,
            grippers=grippers,
            config=config,
            record=_record,
        )
        message = (
            (f"Ranked pair {pair_id} stopped after {last_completed} as configured.")
            if success and status.startswith("stopped_at_")
            else (
                (f"Ranked pair {pair_id} completed the dual real holder/pickup/pre-insertion sequence.")
                if success
                else f"Dual real sequence failed with status {status}."
            )
        )
        result = DualRealExecutionResult(
            success,
            status,
            message,
            pair_id,
            last_completed,
            output_path,
        )
        if not success:
            _stop_grippers(grippers, reason=status)
        return result
    except Exception as exc:
        _stop_grippers(grippers, reason="exception")
        result = DualRealExecutionResult(
            False,
            "exception",
            str(exc),
            pair_id,
            "",
            output_path,
        )
        return result
    finally:
        _write_attempt(
            output_path=output_path,
            plan_json=plan_json,
            plan=execution_plan,
            config=config,
            result=result,
            steps=steps,
            pair_selection=pair_selection,
        )
        for commander in commanders.values():
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


__all__ = [
    "DualRealExecutionConfig",
    "DualRealExecutionResult",
    "MOTION_SEQUENCE",
    "STOP_AFTER_CHOICES",
    "execute_dual_real_plan",
    "load_and_validate_dual_plan",
]
