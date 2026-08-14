"""Guarded dual-KUKA execution of a saved holder/inserter task."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from grasp_planning.pipeline.cartesian_waypoint_ik import IK_STRATEGIES, resolve_ik
from grasp_planning.pipeline.dual_robot_simple_sim import (
    DEFAULT_FLOOR_Z_WORLD_M,
)
from grasp_planning.ros2.mock_gripper_client import MockGripperClient
from grasp_planning.ros2.moveit_pose_commander import (
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    PoseTarget,
    rclpy,
)
from grasp_planning.ros2.normalized_position_gripper_client import (
    NormalizedPositionGripperClient,
)
from grasp_planning.start_poses import (
    KUKA_MOVEIT_ARM_START_JOINT_VALUES,
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
    kuka_gripper_approach_width,
    kuka_moveit_gripper_state,
)

GRIPPER_CLIENTS = ("mock", "trigger_service")

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
    ik_strategy: str = "direct"
    cartesian_waypoint_count: int = 10
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
    # "mock" (default): no gripper hardware/service is required, matching
    # this repo's dual mock stack, which spawns no gripper controller at
    # all. Set to "trigger_service" for real hardware, where the separate
    # gripper-computer process (scripts/gripper_computer/start_dual_grippers.sh)
    # actually serves the *_gripper_open/close/stop_service endpoints below.
    gripper_client: str = "mock"
    gripper_timeout_s: float = 10.0
    grasp_settle_time_s: float = 0.5
    holder_gripper_open_service: str = "/lbr_one/gripper_controller/open"
    holder_gripper_close_service: str = "/lbr_one/gripper_controller/close"
    holder_gripper_stop_service: str = "/lbr_one/gripper_controller/stop"
    holder_gripper_position_command_topic: str = "/lbr_one/gripper_controller/position_command"
    holder_gripper_position_feedback_topic: str = "/lbr_one/gripper_controller/position"
    inserter_gripper_open_service: str = "/lbr_two/gripper_controller/open"
    inserter_gripper_close_service: str = "/lbr_two/gripper_controller/close"
    inserter_gripper_stop_service: str = "/lbr_two/gripper_controller/stop"
    inserter_gripper_position_command_topic: str = "/lbr_two/gripper_controller/position_command"
    inserter_gripper_position_feedback_topic: str = "/lbr_two/gripper_controller/position"
    gripper_position_feedback_tolerance: float = 0.02
    debug_gui: bool = False
    debug_gui_port: int = 0
    debug_gui_open_browser: bool = True


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
    raw_candidate_ranks = [
        raw_candidate.get("candidate_rank") if isinstance(raw_candidate, dict) else None
        for raw_candidate in raw_candidates
    ]
    uses_explicit_ranks = any(value is not None for value in raw_candidate_ranks)
    if uses_explicit_ranks and any(value is None for value in raw_candidate_ranks):
        raise ValueError("ranked_pair_candidates must either all declare candidate_rank or all omit it.")
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
        if uses_explicit_ranks:
            saved_rank = candidate.get("candidate_rank")
            if isinstance(saved_rank, bool) or not isinstance(saved_rank, int) or saved_rank != rank:
                raise ValueError(f"Ranked real candidate {rank} has candidate_rank={saved_rank!r}; expected {rank}.")
        elif score > previous_score + 1.0e-12:
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
    client = str(config.gripper_client)
    if client == "mock":
        return MockGripperClient(
            commander,
            finger_joint_name=f"{ROLE_SPECS[role]['robot']}_left_finger_joint",
            grasp_settle_time_s=float(config.grasp_settle_time_s),
        )
    if client != "trigger_service":
        raise ValueError(f"gripper_client must be one of {GRIPPER_CLIENTS}; got {client!r}.")
    prefix = "holder" if role == "holder" else "inserter"
    return NormalizedPositionGripperClient(
        commander,
        position_command_topic=str(getattr(config, f"{prefix}_gripper_position_command_topic")),
        position_feedback_topic=str(getattr(config, f"{prefix}_gripper_position_feedback_topic")),
        open_service_name=str(getattr(config, f"{prefix}_gripper_open_service")),
        stop_service_name=str(getattr(config, f"{prefix}_gripper_stop_service")),
        timeout_s=float(config.gripper_timeout_s),
        feedback_tolerance=float(config.gripper_position_feedback_tolerance),
        grasp_settle_time_s=float(config.grasp_settle_time_s),
    )


def _grasp_payload_for_role(plan: Mapping[str, object], role: str) -> Mapping[str, object]:
    grasps = plan.get("grasps")
    if not isinstance(grasps, dict):
        raise ValueError("Dual plan is missing grasps.")
    key = "holder" if role == "holder" else "inserter_pickup"
    grasp = grasps.get(key)
    if not isinstance(grasp, dict):
        raise ValueError(f"Dual plan is missing grasp '{key}'.")
    return grasp


def _gripper_width_for_role(plan: Mapping[str, object], role: str, *, contact: bool) -> float:
    grasp = _grasp_payload_for_role(plan, role)
    jaw_width = float(grasp["jaw_width_m"])
    return jaw_width if contact else kuka_gripper_approach_width(jaw_width)


def _moveit_gripper_state_for_plan(
    plan: Mapping[str, object],
    *,
    contact_roles: frozenset[str] = frozenset(),
) -> dict[str, float]:
    roles = plan.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Dual plan is missing roles.")
    state: dict[str, float] = {}
    for role in ("holder", "inserter"):
        role_payload = roles.get(role)
        if not isinstance(role_payload, dict):
            raise ValueError(f"Dual plan is missing role '{role}'.")
        state.update(
            kuka_moveit_gripper_state(
                str(role_payload["robot"]),
                _gripper_width_for_role(plan, role, contact=role in contact_roles),
            )
        )
    return state


def _apply_moveit_gripper_state(commander, state: Mapping[str, float]) -> tuple[bool, str]:
    apply_state = getattr(commander, "apply_planning_scene_robot_state", None)
    if not callable(apply_state):
        return True, "Planning-scene robot-state API is unavailable in this test adapter."
    return apply_state(state)


def _command_gripper_width(
    gripper,
    width_m: float,
    *,
    approach: bool,
) -> tuple[bool, str]:
    command_width = getattr(gripper, "command_width", None)
    if callable(command_width):
        return command_width(
            width_m,
            wait_for_feedback=bool(approach),
            settle_after_command=not bool(approach),
        )
    legacy = getattr(gripper, "open" if approach else "close")
    return legacy(width=width_m)


def _initialize_gripper_open(gripper) -> tuple[bool, str]:
    initialize_open = getattr(gripper, "initialize_open", None)
    if callable(initialize_open):
        return initialize_open()
    return gripper.open(width=KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M)


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
        f"  gripper_client:   {config.gripper_client}"
        + (
            "  <-- WILL NOT ACTUATE ANY REAL GRIPPER; grasps are simulated only\n"
            if config.gripper_client == "mock"
            else "\n"
        )
        + "  collision scene:  both robots + table + temporary pregrasp AABBs; "
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
    resolved_joint_targets: dict[str, tuple[float, ...]] | None = None,
    initial_contact_roles: frozenset[str] = frozenset(),
    ik_strategy: str = "direct",
    cartesian_waypoint_count: int = 10,
) -> bool:
    targets = plan["targets"]
    assert isinstance(targets, dict)
    success = True
    contact_roles = set(initial_contact_roles)
    for role, target_name in _motion_sequence_through(stop_after):
        if role_filter is not None and role != role_filter:
            continue
        finger_state = _moveit_gripper_state_for_plan(
            plan,
            contact_roles=frozenset(contact_roles),
        )
        state_ok, state_message = _apply_moveit_gripper_state(commanders["holder"], finger_state)
        record(
            name=f"preflight_{target_name}_gripper_state",
            role="shared",
            ok=state_ok,
            message=state_message,
        )
        if not state_ok:
            success = False
            if stop_on_failure:
                break
            continue
        target = _target_from_payload(targets[target_name], frame_id=frame_id)
        joints, message = resolve_ik(
            commanders[role],
            target,
            strategy=ik_strategy,
            num_waypoints=cartesian_waypoint_count,
        )
        if joints is None:
            if ik_strategy == "cartesian_waypoints":
                fallback_joints, fallback_message = resolve_ik(
                    commanders[role],
                    target,
                    strategy=ik_strategy,
                    seed_joint_positions=KUKA_MOVEIT_ARM_START_JOINT_VALUES,
                    num_waypoints=cartesian_waypoint_count,
                )
            else:
                fallback_seed_state = dict(finger_state)
                fallback_seed_state.update(
                    zip(ROLE_SPECS[role]["joint_names"], KUKA_MOVEIT_ARM_START_JOINT_VALUES)
                )
                try:
                    fallback_joints, fallback_message = commanders[role].compute_ik(
                        target,
                        seed_robot_state=fallback_seed_state,
                    )
                except TypeError as exc:
                    if "seed_robot_state" not in str(exc):
                        raise
                    # Compatibility for lightweight test adapters and older sourced
                    # ROS workspaces; production uses the complete state above.
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
        if joints is not None and resolved_joint_targets is not None:
            resolved_joint_targets[target_name] = tuple(float(value) for value in joints)
        close_role = GRIPPER_CLOSE_AFTER.get(target_name)
        if ok and close_role is not None:
            contact_roles.add(close_role)
            closed_finger_state = _moveit_gripper_state_for_plan(
                plan,
                contact_roles=frozenset(contact_roles),
            )
            candidate_state = dict(closed_finger_state)
            candidate_state.update(
                (str(name), float(value))
                for name, value in zip(ROLE_SPECS[role]["joint_names"], joints)
            )
            check_validity = getattr(commanders[role], "check_state_validity", None)
            if callable(check_validity):
                validity, validity_message = check_validity(candidate_state, group_name="")
                closed_ok = validity is not None and bool(validity["valid"])
            else:
                validity = {"valid": True, "contacts": []}
                validity_message = "State-validity API is unavailable in this test adapter."
                closed_ok = True
            if not closed_ok:
                contacts = [] if validity is None else validity.get("contacts", [])
                message = (
                    f"{message}; post-grasp closed state invalid: {validity_message}; contacts={contacts}"
                )
                ok = False
            else:
                closed_ok, closed_message = _apply_moveit_gripper_state(
                    commanders["holder"],
                    closed_finger_state,
                )
                if not closed_ok:
                    message = f"{message}; could not apply post-grasp finger state: {closed_message}"
                    ok = False
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
    update_debug: Callable[..., None] | None = None,
    ik_strategy: str = "direct",
    cartesian_waypoint_count: int = 10,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Select the first producer-ranked pair passing all target IK checks."""

    candidates = _ranked_candidate_plans(plan)
    role_cache: dict[
        tuple[str, str, tuple[float, ...]],
        tuple[bool, str, dict[str, tuple[float, ...]]],
    ] = {}
    candidate_records: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    selected_rank: int | None = None
    selected_joint_targets: dict[str, tuple[float, ...]] = {}

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
        if update_debug is not None:
            update_debug(
                candidate=candidate,
                attempt_index=rank,
                phase="ik_preflight",
                status="planning",
                message="Checking exact holder and inserter target IK before hardware motion.",
            )
        role_records: dict[str, object] = {}
        candidate_joint_targets: dict[str, tuple[float, ...]] = {}
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
                role_ok, failure, role_joint_targets = role_cache[cache_key]
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
                role_joint_targets = {}
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
                    resolved_joint_targets=role_joint_targets,
                    initial_contact_roles=(frozenset({"holder"}) if role == "inserter" else frozenset()),
                    ik_strategy=ik_strategy,
                    cartesian_waypoint_count=cartesian_waypoint_count,
                )
                failure = "" if role_ok else "target IK failed"
                role_cache[cache_key] = (
                    role_ok,
                    failure,
                    dict(role_joint_targets),
                )
            candidate_joint_targets.update(role_joint_targets)
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
        if update_debug is not None:
            update_debug(
                candidate=candidate,
                attempt_index=rank,
                phase="ik_preflight",
                status="succeeded" if candidate_ok else "failed",
                message=(
                    f"Selected ranked pair {pair_id}."
                    if candidate_ok
                    else f"Rejected ranked pair {pair_id}; trying the next pair."
                ),
            )
        if candidate_ok:
            selected = candidate
            selected_rank = rank
            selected_joint_targets = candidate_joint_targets
            break

    summary = {
        "policy": "producer_ranked_queue_collision_aware_ik_before_motion",
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
        "selected_joint_targets": {name: list(joints) for name, joints in selected_joint_targets.items()},
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
    preferred_joint_targets: Mapping[str, Sequence[float]] | None = None,
    candidate_rank: int | None = None,
    update_debug: Callable[..., None] | None = None,
) -> tuple[bool, str, str]:
    targets = plan["targets"]
    assert isinstance(targets, dict)
    last_completed = ""
    contact_roles: set[str] = set()

    for role in ("holder", "inserter"):
        if role not in grippers:
            continue
        debug_phase = "holder_pregrasp" if role == "holder" else "inserter_pickup_pregrasp"
        if update_debug is not None:
            update_debug(
                candidate=plan,
                attempt_index=candidate_rank,
                phase=debug_phase,
                status="planning",
                message=f"Commanding the candidate-specific {role} approach opening.",
            )
        approach_width = _gripper_width_for_role(plan, role, contact=False)
        ok, message = _command_gripper_width(
            grippers[role],
            approach_width,
            approach=True,
        )
        record(name=f"position_{role}_gripper_for_approach", role=role, ok=ok, message=message)
        if not ok:
            if update_debug is not None:
                update_debug(
                    candidate=plan,
                    attempt_index=candidate_rank,
                    phase=debug_phase,
                    status="failed",
                    message=message,
                )
            return False, f"{role}_gripper_approach_position_failed", last_completed

    approach_state = _moveit_gripper_state_for_plan(plan)
    state_ok, state_message = _apply_moveit_gripper_state(commanders["holder"], approach_state)
    record(
        name="apply_approach_gripper_state",
        role="shared",
        ok=state_ok,
        message=state_message,
    )
    if not state_ok:
        return False, "approach_gripper_moveit_state_failed", last_completed

    for role, target_name in MOTION_SEQUENCE:
        if update_debug is not None:
            update_debug(
                candidate=plan,
                attempt_index=candidate_rank,
                phase=target_name,
                status="planning",
                message=f"Planning and executing {role} target '{target_name}' on hardware.",
            )
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
            preferred_joints = None if preferred_joint_targets is None else preferred_joint_targets.get(target_name)
            if preferred_joints is None:
                ok, message = commanders[role].move_to_pose(
                    target,
                    label=target_name,
                    execute=True,
                )
            else:
                trajectory, plan_message = commanders[role].plan_to_joint_positions(
                    preferred_joints,
                    label=target_name,
                )
                if trajectory is None:
                    ok = False
                    message = f"{target_name}: planning to preflight IK target failed: {plan_message}"
                else:
                    ok, execute_message = commanders[role].execute_trajectory(
                        trajectory,
                        label=target_name,
                    )
                    message = f"preflight IK target planned ({plan_message}); {execute_message}"
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
            if update_debug is not None:
                update_debug(
                    candidate=plan,
                    attempt_index=candidate_rank,
                    phase=target_name,
                    status="failed",
                    message=message,
                )
            return False, f"{target_name}_failed", last_completed
        last_completed = target_name
        if update_debug is not None:
            update_debug(
                candidate=plan,
                attempt_index=candidate_rank,
                phase=target_name,
                status="succeeded",
                message=message,
            )

        close_role = GRIPPER_CLOSE_AFTER.get(target_name)
        if close_role is not None and close_role in grippers:
            width = _gripper_width_for_role(plan, close_role, contact=True)
            ok, message = _command_gripper_width(
                grippers[close_role],
                width,
                approach=False,
            )
            record(
                name=f"position_{close_role}_gripper_for_contact",
                role=close_role,
                ok=ok,
                message=message,
            )
            if not ok:
                if update_debug is not None:
                    update_debug(
                        candidate=plan,
                        attempt_index=candidate_rank,
                        phase=target_name,
                        status="failed",
                        message=message,
                    )
                return False, f"{close_role}_gripper_close_failed", last_completed
            contact_roles.add(close_role)
            closed_state = _moveit_gripper_state_for_plan(
                plan,
                contact_roles=frozenset(contact_roles),
            )
            state_ok, state_message = _apply_moveit_gripper_state(commanders["holder"], closed_state)
            record(
                name=f"apply_{close_role}_contact_gripper_state",
                role="shared",
                ok=state_ok,
                message=state_message,
            )
            if not state_ok:
                return False, f"{close_role}_gripper_moveit_state_failed", last_completed

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
    if config.ik_strategy not in IK_STRATEGIES:
        raise ValueError(f"ik_strategy must be one of {IK_STRATEGIES}; got {config.ik_strategy!r}.")
    if config.gripper_client not in GRIPPER_CLIENTS:
        raise ValueError(f"gripper_client must be one of {GRIPPER_CLIENTS}; got {config.gripper_client!r}.")
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
    ranked_candidates = _ranked_candidate_plans(source_plan)
    execution_plan = source_plan
    pair_id = ""
    pair_selection: dict[str, object] = {
        "policy": "producer_ranked_queue_collision_aware_ik_before_motion",
        "candidate_count": len(ranked_candidates),
        "candidates_checked": 0,
        "selected_rank": None,
        "selected_pair_id": "",
        "records": [],
        "cached_role_grasp_count": 0,
    }
    steps: list[dict[str, object]] = []
    commanders: dict[str, object] = {}
    grippers: dict[str, object] = {}
    debug_server = None
    debug_scene_candidate_id = ""
    candidate_counts = dict(ranked_candidates[0].get("candidate_filter_diagnostics", {}))
    candidate_counts.update(
        {
            "planner_queue_execution_candidates": len(ranked_candidates),
            "planner_queue_noncrossing_execution_candidates": sum(
                not bool(dict(candidate.get("layout_proxy_components", {})).get("transition_segments_cross_xy"))
                for candidate in ranked_candidates
            ),
            "planner_queue_crossed_execution_candidates": sum(
                bool(dict(candidate.get("layout_proxy_components", {})).get("transition_segments_cross_xy"))
                for candidate in ranked_candidates
            ),
            "planner_queue_unique_holder_grasps": len(
                {_candidate_grasp_id(candidate, role="holder") for candidate in ranked_candidates}
            ),
            "planner_queue_unique_inserter_grasps": len(
                {_candidate_grasp_id(candidate, role="inserter") for candidate in ranked_candidates}
            ),
            "exact_ik_pair_tasks_checked": 0,
            "exact_ik_holder_grasps_checked": 0,
            "exact_ik_inserter_grasps_checked": 0,
        }
    )
    initialized_here = False
    result = DualRealExecutionResult(
        success=False,
        status="not_started",
        message="Execution did not start.",
        pair_id=pair_id,
        last_completed_phase="",
        attempt_artifact_path=output_path,
    )

    if config.debug_gui:
        try:
            from grasp_planning.pipeline.dual_robot_planning_debug import (
                DualRobotPlanningDebugServer,
            )

            debug_server = DualRobotPlanningDebugServer(port=int(config.debug_gui_port))
            debug_url = debug_server.start(open_browser=bool(config.debug_gui_open_browser))
            print(f"[DUAL-REAL] Live planning debugger: {debug_url}", flush=True)
        except OSError as exc:
            print(f"[DUAL-REAL] Could not start live planning debugger: {exc}", flush=True)
            debug_server = None

    def _update_debug(
        *,
        candidate: Mapping[str, object],
        attempt_index: int | None,
        phase: str,
        status: str,
        message: str,
    ) -> None:
        nonlocal debug_scene_candidate_id
        if debug_server is None:
            return
        try:
            candidate_id = str(candidate.get("execution_candidate_id", candidate.get("pair_id", "")))
            scene_payload = None
            if candidate_id != debug_scene_candidate_id:
                from grasp_planning.pipeline.dual_robot_planning_debug import (
                    dual_robot_planning_scene_payload_from_plan,
                )

                scene_payload = dual_robot_planning_scene_payload_from_plan(dict(candidate))
                debug_scene_candidate_id = candidate_id
            debug_server.update(
                scene_payload=scene_payload,
                attempt_index=attempt_index,
                attempt_total=len(ranked_candidates),
                phase=phase,
                status=status,
                message=message,
                candidate_counts=candidate_counts,
            )
        except Exception as exc:  # pragma: no cover - display must not stop hardware handling
            print(f"[DUAL-REAL] Live debugger update failed: {exc}", flush=True)

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
            update_debug=_update_debug,
            ik_strategy=str(config.ik_strategy),
            cartesian_waypoint_count=int(config.cartesian_waypoint_count),
        )
        candidate_counts["exact_ik_pair_tasks_checked"] = int(pair_selection["candidates_checked"])
        candidate_counts["exact_ik_holder_grasps_checked"] = sum(
            1
            for candidate_record in pair_selection["records"]
            if isinstance(candidate_record, dict) and "holder" in dict(candidate_record.get("roles", {}))
        )
        candidate_counts["exact_ik_inserter_grasps_checked"] = sum(
            1
            for candidate_record in pair_selection["records"]
            if isinstance(candidate_record, dict) and "inserter" in dict(candidate_record.get("roles", {}))
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
                    message="Normalized position topics and open/stop Trigger services are configured.",
                )
                home_ok, home_message = _initialize_gripper_open(gripper)
                _record(
                    name=f"initialize_{role}_gripper_open_zero",
                    role=role,
                    ok=home_ok,
                    message=home_message,
                )
                if not home_ok:
                    raise RuntimeError(f"Could not establish the {role} gripper open zero: {home_message}")

        success, status, last_completed = _execute_sequence(
            plan=execution_plan,
            commanders=commanders,
            grippers=grippers,
            config=config,
            record=_record,
            preferred_joint_targets=(
                pair_selection.get("selected_joint_targets")
                if isinstance(pair_selection.get("selected_joint_targets"), dict)
                else None
            ),
            candidate_rank=(
                int(pair_selection["selected_rank"]) if pair_selection.get("selected_rank") is not None else None
            ),
            update_debug=_update_debug,
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
        if debug_server is not None:
            terminal_phase = result.last_completed_phase or (
                "ik_preflight" if result.status in {"preflight_ok", "ik_preflight_failed"} else "complete"
            )
            _update_debug(
                candidate=execution_plan,
                attempt_index=(
                    int(pair_selection["selected_rank"])
                    if pair_selection.get("selected_rank") is not None
                    else int(pair_selection.get("candidates_checked", 0))
                ),
                phase=terminal_phase,
                status="complete" if result.success else "failed",
                message=result.message,
            )
            time.sleep(0.35)
            debug_server.close()


__all__ = [
    "DualRealExecutionConfig",
    "DualRealExecutionResult",
    "GRIPPER_CLIENTS",
    "MOTION_SEQUENCE",
    "STOP_AFTER_CHOICES",
    "execute_dual_real_plan",
    "load_and_validate_dual_plan",
]
