"""Guarded dual-KUKA execution of a saved holder/inserter task."""

from __future__ import annotations

import json
import time
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from grasp_planning.gripper_profiles import (
    SERVO_GRIPPER_CLOSED_WIDTH_M,
    SERVO_GRIPPER_OPEN_WIDTH_M,
    servo_gripper_clamp_width,
)
from grasp_planning.pipeline.dual_robot_simple_sim import (
    DEFAULT_FLOOR_Z_WORLD_M,
)
from grasp_planning.ros2.kuka_ik_seeds import kuka_iiwa_ik_seed_candidates
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
    gripper_approach_width,
    kuka_moveit_gripper_state,
)

ARM_SPEC_BY_ROBOT = {
    robot: {
        "robot": robot,
        "planning_group": "arm_one" if robot == "lbr_one" else "arm_two",
        "joint_names": tuple(f"{robot}_A{index}" for index in range(1, 8)),
    }
    for robot in ("lbr_one", "lbr_two")
}
ROLE_SPECS = {
    "holder": ARM_SPEC_BY_ROBOT["lbr_one"],
    "inserter": ARM_SPEC_BY_ROBOT["lbr_two"],
}
GRIPPER_SIDE_BY_ROBOT = {"lbr_one": "left", "lbr_two": "right"}


def _gripper_model_for_plan(plan: Mapping[str, object]) -> str:
    raw_model = str(plan.get("gripper_model", "")).strip()
    if raw_model in {"pdz_gripper"}:
        return "pdz_gripper"
    if raw_model in {"", "y_gripper", "kuka_y_gripper"}:
        if not raw_model:
            roles = plan.get("roles")
            if isinstance(roles, dict) and any(
                "pdz_gripper_tcp" in str(value.get("tcp_link", ""))
                for value in roles.values()
                if isinstance(value, dict)
            ):
                return "pdz_gripper"
        return "kuka_y_gripper"
    raise ValueError(f"Dual plan has unsupported gripper_model {raw_model!r}.")


def _role_spec(plan: Mapping[str, object], role: str) -> Mapping[str, object]:
    roles = plan.get("roles")
    if not isinstance(roles, dict) or not isinstance(roles.get(role), dict):
        raise ValueError(f"Dual plan is missing role '{role}'.")
    robot = str(roles[role].get("robot", ""))
    if robot not in ARM_SPEC_BY_ROBOT:
        raise ValueError(f"Dual plan role '{role}' has unsupported robot {robot!r}.")
    spec = dict(ARM_SPEC_BY_ROBOT[robot])
    tcp_suffix = (
        "pdz_gripper_tcp"
        if _gripper_model_for_plan(plan) == "pdz_gripper"
        else "gripper_tcp"
    )
    spec["pose_link"] = f"{robot}_{tcp_suffix}"
    return spec
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
CARTESIAN_TARGETS = frozenset(
    {
        "holder_grasp",
        "inserter_pickup_grasp",
        "inserter_pickup_lift",
        "inserter_preinsertion",
    }
)


def _motion_sequence_through(
    stop_after: str,
) -> tuple[tuple[str, str], ...]:
    if stop_after not in STOP_AFTER_CHOICES:
        raise ValueError(f"stop_after must be one of {STOP_AFTER_CHOICES}; got {stop_after!r}.")
    stop_index = STOP_AFTER_CHOICES.index(stop_after)
    return MOTION_SEQUENCE[: stop_index + 1]


def _normalized_active_roles(active_roles: Sequence[str]) -> tuple[str, ...]:
    roles = tuple(dict.fromkeys(str(role) for role in active_roles))
    if not roles or any(role not in ROLE_SPECS for role in roles):
        raise ValueError("active_roles must contain holder and/or inserter.")
    return roles


def _motion_sequence_for(
    stop_after: str,
    active_roles: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    roles = set(_normalized_active_roles(active_roles))
    return tuple(
        (role, target_name)
        for role, target_name in _motion_sequence_through(stop_after)
        if role in roles
    )


@dataclass(frozen=True)
class DualRealExecutionConfig:
    moveit_namespace: str = "/lbr_dual_arm"
    frame_id: str = "base_link"
    wait_for_moveit_timeout_s: float = 20.0
    ik_timeout_s: float = 2.0
    ik_candidate_count: int = 7
    ik_beam_width: int = 4
    ik_seed_perturbation_rad: float = 0.60
    ik_solution_dedup_tolerance_rad: float = 0.05
    planning_time_s: float = 8.0
    num_planning_attempts: int = 8
    velocity_scale: float = 0.05
    acceleration_scale: float = 0.05
    cartesian_max_step_m: float = 0.005
    cartesian_revolute_jump_threshold_rad: float = 0.35
    execution_start_tolerance_rad: float = 0.05
    execute_timeout_s: float = 120.0
    post_execute_sleep_s: float = 0.5
    execute: bool = False
    require_confirmation: bool = True
    allow_objectless_planning: bool = False
    stop_after: str = "holder_pregrasp"
    active_roles: tuple[str, ...] = ("holder", "inserter")
    policy: str = ""
    left_camera: str = "realsense_1"
    right_camera: str = "realsense_2"
    grippers_enabled: bool = True
    allow_missing_grippers: bool = False
    gripper_timeout_s: float = 10.0
    grasp_settle_time_s: float = 0.5
    left_gripper_open_service: str = "/left/gripper_controller/open"
    left_gripper_close_service: str = "/left/gripper_controller/close"
    left_gripper_stop_service: str = "/left/gripper_controller/stop"
    left_gripper_position_command_topic: str = "/left/gripper_controller/position_command"
    left_gripper_position_feedback_topic: str = "/left/gripper_controller/position"
    right_gripper_open_service: str = "/right/gripper_controller/open"
    right_gripper_close_service: str = "/right/gripper_controller/close"
    right_gripper_stop_service: str = "/right/gripper_controller/stop"
    right_gripper_position_command_topic: str = "/right/gripper_controller/position_command"
    right_gripper_position_feedback_topic: str = "/right/gripper_controller/position"
    gripper_closed_width_m: float = SERVO_GRIPPER_CLOSED_WIDTH_M
    gripper_open_width_m: float = SERVO_GRIPPER_OPEN_WIDTH_M
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


@dataclass(frozen=True)
class PreplannedDualSequence:
    trajectories: Mapping[str, object]
    joint_targets: Mapping[str, tuple[float, ...]]
    segment_modes: Mapping[str, str]
    start_states: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class PreparedPolicyApproach:
    role: str
    grasp_id: str
    part_id: str
    config_path: Path
    goal_path: Path
    preparation: object


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
    assigned_robots = set()
    for role in ROLE_SPECS:
        expected = _role_spec(payload, role)
        actual = roles.get(role)
        if not isinstance(actual, dict):
            raise ValueError(f"{context} is missing role '{role}'.")
        for key in ("robot", "planning_group", "tcp_link"):
            expected_key = "pose_link" if key == "tcp_link" else key
            if str(actual.get(key, "")) != str(expected[expected_key]):
                raise ValueError(
                    f"{context} role '{role}' has {key}={actual.get(key)!r}; expected {expected[expected_key]!r}."
                )
        assigned_robots.add(str(actual["robot"]))
    if assigned_robots != set(ARM_SPEC_BY_ROBOT):
        raise ValueError("Dual plan must assign lbr_one and lbr_two exactly once.")


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


def _attached_collision_objects(
    plan: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    moveit = plan.get("moveit")
    if not isinstance(moveit, dict):
        return {}
    raw_geometry = moveit.get("attached_collision_geometry")
    if not isinstance(raw_geometry, dict):
        return {}
    raw_objects = raw_geometry.get("objects")
    if not isinstance(raw_objects, dict):
        return {}
    objects: dict[str, dict[str, object]] = {}
    for key, raw_object in raw_objects.items():
        if not isinstance(raw_object, dict):
            raise ValueError(f"Attached collision object '{key}' must be a mapping.")
        objects[str(key)] = dict(raw_object)
    return objects


def _apply_attached_collision_objects(
    commander,
    obstacles: Mapping[str, Mapping[str, object]],
    *,
    frame_id: str,
) -> tuple[bool, str]:
    apply_attached = getattr(commander, "apply_planning_scene_attached_obstacles", None)
    if not callable(apply_attached):
        return True, "Attached-collision API unavailable in lightweight test adapter."
    return apply_attached(
        [dict(value) for value in obstacles.values()],
        default_frame_id=frame_id,
    )


def _remove_attached_collision_objects(
    commander,
    obstacles: Mapping[str, Mapping[str, object]],
    *,
    frame_id: str,
) -> tuple[bool, str]:
    remove_attached = getattr(commander, "remove_planning_scene_attached_obstacles", None)
    if not callable(remove_attached):
        return True, "Attached-collision API unavailable in lightweight test adapter."
    return remove_attached(
        [dict(value) for value in obstacles.values()],
        default_frame_id=frame_id,
    )


def _has_loaded_part_collision_geometry(plan: Mapping[str, object]) -> bool:
    moveit = plan.get("moveit")
    if not isinstance(moveit, dict):
        return False
    phase_geometry = moveit.get("pregrasp_aabb_collision_geometry")
    attached_geometry = moveit.get("attached_collision_geometry")
    return (
        isinstance(phase_geometry, dict)
        and bool(phase_geometry.get("obstacles"))
        and isinstance(attached_geometry, dict)
        and bool(attached_geometry.get("objects"))
    )


def _make_commander(
    *,
    role: str,
    config: DualRealExecutionConfig,
    plan: Mapping[str, object],
):
    spec = _role_spec(plan, role)
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
    plan: Mapping[str, object],
    commander,
    config: DualRealExecutionConfig,
):
    prefix = _gripper_side_for_role(plan, role)
    return NormalizedPositionGripperClient(
        commander,
        position_command_topic=str(getattr(config, f"{prefix}_gripper_position_command_topic")),
        position_feedback_topic=str(getattr(config, f"{prefix}_gripper_position_feedback_topic")),
        open_service_name=str(getattr(config, f"{prefix}_gripper_open_service")),
        close_service_name=str(getattr(config, f"{prefix}_gripper_close_service")),
        stop_service_name=str(getattr(config, f"{prefix}_gripper_stop_service")),
        timeout_s=float(config.gripper_timeout_s),
        feedback_tolerance=float(config.gripper_position_feedback_tolerance),
        grasp_settle_time_s=float(config.grasp_settle_time_s),
        closed_width_m=float(config.gripper_closed_width_m),
        open_width_m=float(config.gripper_open_width_m),
    )


def _gripper_side_for_role(plan: Mapping[str, object], role: str) -> str:
    """Resolve a task role through its planned robot to the physical gripper side."""

    robot = str(_role_spec(plan, role)["robot"])
    return GRIPPER_SIDE_BY_ROBOT[robot]


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
    width = (
        jaw_width
        if contact
        else gripper_approach_width(
            jaw_width,
            gripper_model=_gripper_model_for_plan(plan),
        )
    )
    return servo_gripper_clamp_width(width)


def _gripper_fully_closed_width(_plan: Mapping[str, object], _role: str) -> float:
    del _plan, _role
    return SERVO_GRIPPER_CLOSED_WIDTH_M


def _moveit_gripper_state_for_plan(
    plan: Mapping[str, object],
    *,
    contact_roles: frozenset[str] = frozenset(),
) -> dict[str, float]:
    roles = plan.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Dual plan is missing roles.")
    state: dict[str, float] = {}
    gripper_model = _gripper_model_for_plan(plan)
    for role in ("holder", "inserter"):
        role_payload = roles.get(role)
        if not isinstance(role_payload, dict):
            raise ValueError(f"Dual plan is missing role '{role}'.")
        state.update(
            kuka_moveit_gripper_state(
                str(role_payload["robot"]),
                _gripper_width_for_role(plan, role, contact=role in contact_roles),
                gripper_model=gripper_model,
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
    if not approach:
        close = getattr(gripper, "close", None)
        if callable(close):
            return close(width=width_m)
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
    return gripper.open(width=SERVO_GRIPPER_OPEN_WIDTH_M)


def _activate_available_grippers(
    grippers: Mapping[str, object],
    *,
    timeout_s: float,
    record: Callable[..., None],
    allow_missing: bool = False,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Probe every role first, then open only controllers that are present."""

    available: dict[str, object] = {}
    skipped: list[str] = []
    for role, gripper in grippers.items():
        try:
            gripper.wait_for_server(timeout_s=float(timeout_s))
        except RuntimeError as exc:
            if "unavailable" not in str(exc).lower():
                raise
            if not allow_missing:
                raise RuntimeError(
                    f"Required {role} gripper endpoint is unavailable; no arm motion was started: {exc}"
                ) from exc
            skipped.append(role)
            record(
                name=f"skip_{role}_gripper_unavailable",
                role=role,
                ok=True,
                message=f"{exc} Continuing without hardware gripper commands for this role.",
            )
            continue
        available[role] = gripper
        record(
            name=f"wait_for_{role}_gripper",
            role=role,
            ok=True,
            message="Normalized position topics and open/stop Trigger services are configured.",
        )

    # Do not move either gripper until availability has been checked for both.
    for role, gripper in available.items():
        open_ok, open_message = _initialize_gripper_open(gripper)
        record(
            name=f"initialize_{role}_gripper_open",
            role=role,
            ok=open_ok,
            message=open_message,
        )
        if not open_ok:
            raise RuntimeError(f"Could not open the {role} gripper: {open_message}")
    return available, tuple(skipped)


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
    active_roles = _normalized_active_roles(config.active_roles)
    role_lines = []
    for role, grasp_id in (("holder", holder_id), ("inserter", inserter_id)):
        if role not in active_roles:
            continue
        role_lines.append(
            f"  {role + ':':18s}{_role_spec(plan, role)['robot']} / {grasp_id}\n"
        )
    policy_line = f"  policy:           {config.policy}\n" if config.policy else ""
    return (
        "DUAL REAL-ROBOT EXECUTION REQUESTED\n"
        f"  plan:             {plan_json}\n"
        f"  pair:             {plan.get('pair_id', '')}\n"
        f"  active_roles:     {','.join(active_roles)}\n"
        + "".join(role_lines)
        + policy_line
        + f"  stop_after:       {config.stop_after}\n"
        f"  velocity_scale:   {config.velocity_scale:.3f}\n"
        "  collision scene:  both robots + table + phase-aware part AABBs; "
        "incoming part attached during loaded transport\n"
        "Verify the physical 840 mm base transform, clear the cell, keep every "
        "active-arm E-stop reachable, and type 'yes' to continue: "
    )


def _source_artifact_path(plan: Mapping[str, object], *, role: str) -> Path:
    source_artifacts = plan.get("source_artifacts")
    if isinstance(source_artifacts, dict):
        key = "holder_stage2_bundle" if role == "holder" else "inserter_stage2_bundle"
        raw_path = str(source_artifacts.get(key, "")).strip()
        if raw_path:
            return Path(raw_path).expanduser().resolve()
    collision_checks = plan.get("collision_checks")
    if isinstance(collision_checks, dict):
        offline = collision_checks.get("offline_dual_pair")
        if isinstance(offline, dict) and str(offline.get("artifact", "")).strip():
            artifact_dir = Path(str(offline["artifact"])).expanduser().resolve().parent
            filename = (
                "holder_base_candidates.json"
                if role == "holder"
                else f"inserter_candidates_{plan['step_id']}.json"
            )
            return artifact_dir / filename
    raise ValueError(f"Dual task has no source stage-2 bundle for the {role} policy goal.")


def _local_pose_for_role(
    plan: Mapping[str, object],
    *,
    role: str,
    position_world: Sequence[float],
    orientation_xyzw_world: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    layout = plan.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("Dual task has no layout block for policy goal rendering.")
    base_key = "holder_base_world_m" if role == "holder" else "inserter_base_world_m"
    base_position = tuple(float(value) for value in layout[base_key])
    position = tuple(
        float(value) - float(offset)
        for value, offset in zip(position_world, base_position, strict=True)
    )
    orientation = tuple(float(value) for value in orientation_xyzw_world)
    if len(position) != 3 or len(orientation) != 4:
        raise ValueError("Policy rendering requires a 3D position and XYZW orientation.")
    return position, orientation  # type: ignore[return-value]


def _prepare_policy_approaches(
    *,
    plan: Mapping[str, object],
    config: DualRealExecutionConfig,
    preplanned_sequence: PreplannedDualSequence,
    attempt_artifact_path: Path,
    record: Callable[..., None],
) -> dict[str, PreparedPolicyApproach]:
    if not config.policy:
        return {}
    from grasp_planning.rl.d405_deployment_config import write_visual_servo_config
    from grasp_planning.rl.d405_goal_renderer import render_d405_goal_for_grasp
    from grasp_planning.rl.policy_registry import resolve_policy_reference
    from grasp_planning.ros2.d405_visual_servo import (
        preflight_d405_policy_visual_servo,
        prepare_d405_policy_visual_servo,
    )

    repo_root = Path(__file__).resolve().parents[2]
    policy_name, assets = resolve_policy_reference(
        config.policy,
        registry_path=repo_root / "configs/d405_policy_registry.yaml",
    )
    plan_gripper_model = _gripper_model_for_plan(plan)
    normalized_plan_model = "y_gripper" if plan_gripper_model == "kuka_y_gripper" else plan_gripper_model
    if str(assets["gripper_model"]) != normalized_plan_model:
        raise ValueError(
            f"Policy '{policy_name}' was trained for {assets['gripper_model']}, but the selected "
            f"dual task uses {normalized_plan_model}."
        )
    output_dir = attempt_artifact_path.expanduser().resolve().parent / "policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    grasps = plan.get("grasps")
    objects = plan.get("objects")
    targets = plan.get("targets")
    if not isinstance(grasps, dict) or not isinstance(objects, dict) or not isinstance(targets, dict):
        raise ValueError("Dual task is missing grasps, objects, or targets for policy deployment.")

    prepared: dict[str, PreparedPolicyApproach] = {}
    active_roles = _normalized_active_roles(config.active_roles)
    for role in active_roles:
        grasp_key = "holder" if role == "holder" else "inserter_pickup"
        target_name = "holder_grasp" if role == "holder" else "inserter_pickup_grasp"
        if target_name not in preplanned_sequence.joint_targets:
            continue
        grasp = grasps.get(grasp_key)
        target = targets.get(target_name)
        if not isinstance(grasp, dict) or not isinstance(target, dict):
            raise ValueError(f"Dual task has no {role} grasp target for policy deployment.")
        robot = str(_role_spec(plan, role)["robot"])
        camera_side = GRIPPER_SIDE_BY_ROBOT[robot]
        camera_name = config.left_camera if camera_side == "left" else config.right_camera
        config_path = output_dir / f"{role}_visual_servo.yaml"
        write_visual_servo_config(
            policy_name=policy_name,
            assets=assets,
            template_path=repo_root / "configs/visual_servo_real_d405.yaml",
            output_path=config_path,
            output_root=output_dir / f"{role}_runs",
            model_device="cuda:0",
            camera_name=camera_name,
            robot_name=robot,
        )
        goal_position, goal_orientation = _local_pose_for_role(
            plan,
            role=role,
            position_world=target["position_world_m"],
            orientation_xyzw_world=target["orientation_xyzw_world"],
        )
        if role == "holder":
            object_payload = objects.get("subassembly")
            object_pose_key = "source_pose_world"
            part_id = str(plan["base_part_id"])
        else:
            object_payload = objects.get("incoming")
            object_pose_key = "pickup_source_pose_world"
            part_id = str(plan["incoming_part_id"])
        if not isinstance(object_payload, dict) or not isinstance(object_payload.get(object_pose_key), dict):
            raise ValueError(f"Dual task has no {role} object pose for policy goal rendering.")
        object_pose = object_payload[object_pose_key]
        assert isinstance(object_pose, dict)
        object_position, object_orientation = _local_pose_for_role(
            plan,
            role=role,
            position_world=object_pose["position_world_m"],
            orientation_xyzw_world=object_pose["orientation_xyzw_world"],
        )
        grasp_id = str(grasp["grasp_id"])
        goal_path = output_dir / f"{role}_goal_{grasp_id}.npz"
        rendered = render_d405_goal_for_grasp(
            config_path=config_path,
            stage2_bundle_path=_source_artifact_path(plan, role=role),
            grasp_id=grasp_id,
            part_id=part_id,
            goal_joint_positions=preplanned_sequence.joint_targets[target_name],
            goal_tcp_position=goal_position,
            goal_tcp_orientation_xyzw=goal_orientation,
            approach_width_m=_gripper_width_for_role(plan, role, contact=False),
            maximum_approach_width_m=float(config.gripper_open_width_m),
            output_path=goal_path,
            object_position=object_position,
            object_orientation_xyzw=object_orientation,
        )
        preparation = prepare_d405_policy_visual_servo(
            config_path=config_path,
            expected_grasp_id=grasp_id,
            expected_part_id=part_id,
            goal_observation_path_override=rendered.path,
        )
        camera_frame = preflight_d405_policy_visual_servo(preparation)
        prepared[role] = PreparedPolicyApproach(
            role=role,
            grasp_id=grasp_id,
            part_id=part_id,
            config_path=config_path,
            goal_path=rendered.path,
            preparation=preparation,
        )
        record(
            name=f"prepare_{role}_policy",
            role=role,
            ok=True,
            message=(
                f"policy={policy_name} camera={camera_name} frame={camera_frame} goal={rendered.path} "
                f"sha256={rendered.sha256}"
            ),
        )
    if prepared:
        debug_html = _write_policy_goal_debug_html(
            approaches=prepared,
            output_path=output_dir / "policy_execution_debug.html",
        )
        print(f"[DUAL-REAL] Policy goal debug HTML: {debug_html.as_uri()}", flush=True)
        if config.debug_gui and config.debug_gui_open_browser:
            webbrowser.open(debug_html.as_uri())
    return prepared


def _write_policy_goal_debug_html(
    *,
    approaches: Mapping[str, PreparedPolicyApproach],
    output_path: Path,
) -> Path:
    """Embed every on-demand MuJoCo RGB-D goal in one execution handoff page."""

    import base64
    import html
    import io

    import numpy as np
    from PIL import Image

    def data_url(image: object) -> str:
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    cards: list[str] = []
    for role, approach in approaches.items():
        with np.load(approach.goal_path, allow_pickle=False) as rendered:
            rgb = np.asarray(rendered["goal_rgb"], dtype=np.uint8)
            depth = np.asarray(rendered["goal_depth"], dtype=np.float32)
        valid = np.isfinite(depth) & (depth >= 0.07) & (depth < 0.50)
        normalized = np.clip((depth - 0.07) / 0.43, 0.0, 1.0)
        depth_rgb = np.stack(
            (
                (255.0 * normalized),
                (255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))),
                (255.0 * (1.0 - normalized)),
            ),
            axis=-1,
        ).astype(np.uint8)
        depth_rgb[~valid] = (20, 24, 31)
        cards.append(
            "<section><h2>"
            + html.escape(role.title())
            + " policy approach</h2><p>grasp="
            + html.escape(approach.grasp_id)
            + " part="
            + html.escape(approach.part_id)
            + "</p><div class='images'><figure><img src='"
            + data_url(rgb)
            + "'><figcaption>On-demand MuJoCo goal RGB</figcaption></figure>"
            + "<figure><img src='"
            + data_url(depth_rgb)
            + "'><figcaption>Policy depth, valid fraction "
            + f"{float(valid.mean()):.1%}"
            + "</figcaption></figure></div></section>"
        )
    output_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Dual policy execution</title>"
        "<style>body{font:15px system-ui;background:#10141c;color:#edf2fa;margin:24px}"
        "section{background:#1a2230;border-radius:12px;padding:16px;margin:14px 0}"
        ".images{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0}"
        "img{width:512px;max-width:42vw;image-rendering:pixelated;border-radius:8px}"
        "figcaption{color:#aebdd0;margin-top:6px}</style>"
        "<h1>MoveIt-selected dual policy handoff</h1>"
        "<p>Each image was rendered on demand after connected collision-aware preflight. "
        "The policy replaces only pregrasp-to-grasp; gripper closure follows its completion gate.</p>"
        + "".join(cards),
        encoding="utf-8",
    )
    return output_path.resolve()


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
    active_roles: Sequence[str] = ("holder", "inserter"),
) -> bool:
    targets = plan["targets"]
    assert isinstance(targets, dict)
    success = True
    contact_roles = set(initial_contact_roles)
    attached_objects = _attached_collision_objects(plan)
    incoming_attached = False
    for role, target_name in _motion_sequence_for(stop_after, active_roles):
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
        active_aabbs = _pregrasp_aabb_obstacles(plan, target_name=target_name)
        if active_aabbs:
            scene_ok, scene_message = commanders["holder"].apply_planning_scene_obstacles(
                active_aabbs,
                default_frame_id=frame_id,
            )
            record(
                name=f"preflight_apply_{target_name}_part_aabbs",
                role="shared",
                ok=scene_ok,
                message=scene_message,
            )
            if not scene_ok:
                success = False
                if stop_on_failure:
                    break
                continue
        joints = None
        messages: list[str] = []
        for seed_index, seed in enumerate(
            kuka_iiwa_ik_seed_candidates(KUKA_MOVEIT_ARM_START_JOINT_VALUES)
        ):
            seed_state = dict(finger_state)
            seed_state.update(zip(_role_spec(plan, role)["joint_names"], seed))
            try:
                candidate_joints, seed_message = _compute_ik_with_complete_seed(
                    commanders[role], target, seed_state
                )
            except TypeError as exc:
                if "seed_robot_state" not in str(exc):
                    raise
                candidate_joints, seed_message = commanders[role].compute_ik(
                    target, seed_joint_positions=seed
                )
            messages.append(f"seed {seed_index}: {seed_message}")
            if candidate_joints is not None:
                joints = candidate_joints
                break
        message = (
            f"multi-seed IK succeeded after {len(messages)} attempts"
            if joints is not None
            else f"multi-seed IK exhausted {len(messages)} attempts: {'; '.join(messages)}"
        )
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
                for name, value in zip(_role_spec(plan, role)["joint_names"], joints)
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
        if active_aabbs:
            remove_ok, remove_message = commanders["holder"].remove_planning_scene_obstacles(
                [str(obstacle["id"]) for obstacle in active_aabbs],
                default_frame_id=frame_id,
            )
            record(
                name=f"preflight_remove_{target_name}_part_aabbs",
                role="shared",
                ok=remove_ok,
                message=remove_message,
            )
            if not remove_ok:
                ok = False
                message = f"{message}; could not remove phase part AABBs: {remove_message}"
        if ok and target_name == "inserter_pickup_grasp" and attached_objects:
            attach_ok, attach_message = _apply_attached_collision_objects(
                commanders["holder"],
                attached_objects,
                frame_id=frame_id,
            )
            record(
                name="preflight_attach_incoming_collision",
                role="shared",
                ok=attach_ok,
                message=attach_message,
            )
            if attach_ok:
                incoming_attached = True
            else:
                ok = False
                message = f"{message}; could not attach incoming collision geometry: {attach_message}"
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
    if incoming_attached:
        detach_ok, detach_message = _remove_attached_collision_objects(
            commanders["holder"],
            attached_objects,
            frame_id=frame_id,
        )
        record(
            name="preflight_cleanup_attached_incoming_collision",
            role="shared",
            ok=detach_ok,
            message=detach_message,
        )
        if not detach_ok:
            # A failed purge leaves the shared scene unknown.  Do not cache the
            # result or evaluate another candidate against potentially stale
            # incoming-part geometry.
            raise RuntimeError(
                f"Could not clean up incoming attached collision geometry: {detach_message}"
            )
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


def _trajectory_terminal_state(
    trajectory,
    *,
    current_state: Mapping[str, float],
    required_joint_names: Sequence[str],
) -> tuple[dict[str, float] | None, str]:
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    names = tuple(str(value) for value in getattr(joint_trajectory, "joint_names", ()))
    points = tuple(getattr(joint_trajectory, "points", ()))
    if not names or not points:
        return None, "trajectory contains no named joint points"
    positions = tuple(float(value) for value in getattr(points[-1], "positions", ()))
    if len(names) != len(positions):
        return None, "trajectory terminal point does not match its joint names"
    terminal = dict(current_state)
    terminal.update(zip(names, positions))
    missing = [str(name) for name in required_joint_names if str(name) not in terminal]
    if missing:
        return None, f"trajectory terminal state is missing active joints: {missing}"
    return terminal, "ok"


def _trajectory_joint_path_cost(trajectory) -> float:
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    points = tuple(getattr(joint_trajectory, "points", ()))
    positions = [tuple(float(value) for value in getattr(point, "positions", ())) for point in points]
    return sum(
        sum(abs(right - left) for left, right in zip(first, second))
        for first, second in zip(positions, positions[1:])
    )


def _compute_ik_with_complete_seed(commander, target, seed_state):
    try:
        return commander.compute_ik(target, seed_robot_state=seed_state, avoid_collisions=False)
    except TypeError as exc:
        if "avoid_collisions" not in str(exc):
            raise
        return commander.compute_ik(target, seed_robot_state=seed_state)


def _plan_free_space_with_multi_seed_ik(
    *, commander, target: PoseTarget, target_name: str, role: str,
    current_state: Mapping[str, float], config: DualRealExecutionConfig,
    plan: Mapping[str, object],
    kinematic_ik_cache: dict[tuple[object, ...], tuple[tuple[float, ...] | None, str]] | None = None,
) -> tuple[object | None, tuple[float, ...] | None, str]:
    """Try every distinct collision-valid IK branch before rejecting a pose."""

    joint_names = tuple(str(name) for name in _role_spec(plan, role)["joint_names"])
    start_joints = tuple(float(current_state[name]) for name in joint_names)
    seeds = kuka_iiwa_ik_seed_candidates(
        start_joints,
        candidate_count=config.ik_candidate_count,
        perturbation_rad=config.ik_seed_perturbation_rad,
    )
    solutions: list[tuple[float, ...]] = []
    plans: list[tuple[float, int, object, tuple[float, ...], str]] = []
    diagnostics: list[str] = []
    for seed_index, seed in enumerate(seeds):
        seed_state = dict(current_state)
        seed_state.update(zip(joint_names, seed))
        cache_key = (
            role,
            target.frame_id,
            *(round(float(value), 9) for value in (*target.position_xyz, *target.orientation_xyzw)),
            *(round(float(value), 9) for value in seed),
        )
        cached = None if kinematic_ik_cache is None else kinematic_ik_cache.get(cache_key)
        if cached is None:
            raw_joints, ik_message = _compute_ik_with_complete_seed(commander, target, seed_state)
            joints = None if raw_joints is None else tuple(float(value) for value in raw_joints)
            if kinematic_ik_cache is not None:
                kinematic_ik_cache[cache_key] = (joints, str(ik_message))
        else:
            joints, cached_message = cached
            ik_message = f"cached: {cached_message}"
        if joints is None:
            diagnostics.append(f"seed {seed_index}: IK failed ({ik_message})")
            continue
        solution = tuple(float(value) for value in joints)
        if any(max(abs(a - b) for a, b in zip(solution, old)) < config.ik_solution_dedup_tolerance_rad for old in solutions):
            continue
        solutions.append(solution)
        candidate_state = dict(current_state)
        candidate_state.update(zip(joint_names, solution))
        check_validity = getattr(commander, "check_state_validity", None)
        if callable(check_validity):
            validity, validity_message = check_validity(candidate_state, group_name="")
            if validity is None or not bool(validity.get("valid", False)):
                diagnostics.append(f"seed {seed_index}: IK state invalid ({validity_message})")
                continue
        trajectory, motion_message = commander.plan_to_joint_positions(
            solution, label=target_name, start_robot_state=current_state
        )
        if trajectory is None:
            diagnostics.append(f"seed {seed_index}: motion failed ({motion_message})")
            continue
        plans.append((_trajectory_joint_path_cost(trajectory), seed_index, trajectory, solution, motion_message))
        if len(plans) >= max(1, int(config.ik_beam_width)):
            break
    if not plans:
        detail = "; ".join(diagnostics) or "no distinct IK solutions"
        return None, None, f"multi-seed IK exhausted {len(seeds)} seeds and {len(solutions)} solutions: {detail}"
    _cost, seed_index, trajectory, solution, motion_message = min(plans, key=lambda item: (item[0], item[1]))
    return trajectory, solution, (
        f"multi-seed IK selected seed {seed_index} from {len(seeds)} seeds, "
        f"{len(solutions)} distinct solutions, and {len(plans)} motion-valid branches; motion: {motion_message}"
    )


def _same_orientation(first: PoseTarget, second: PoseTarget, *, tolerance: float = 1.0e-6) -> bool:
    dot = abs(
        sum(
            left * right
            for left, right in zip(
                first.orientation_xyzw,
                second.orientation_xyzw,
            )
        )
    )
    return 1.0 - min(dot, 1.0) <= float(tolerance)


def _preplan_connected_candidate(
    *,
    plan: Mapping[str, object],
    commanders: Mapping[str, object],
    initial_robot_state: Mapping[str, float],
    config: DualRealExecutionConfig,
    record: Callable[..., None],
    stop_after: str,
    candidate_rank: int,
    pair_id: str,
    kinematic_ik_cache: dict[tuple[object, ...], tuple[tuple[float, ...] | None, str]] | None = None,
) -> tuple[PreplannedDualSequence | None, str]:
    """Plan every connected segment from one live complete robot state."""

    targets = plan.get("targets")
    assert isinstance(targets, dict)
    current_state = {str(name): float(value) for name, value in initial_robot_state.items()}
    required_arm_joints = tuple(
        str(name)
        for role in ("holder", "inserter")
        for name in _role_spec(plan, role)["joint_names"]
    )
    missing = [name for name in required_arm_joints if name not in current_state]
    if missing:
        return None, f"live MoveIt state is missing dual-arm joints: {missing}"

    trajectories: dict[str, object] = {}
    joint_targets: dict[str, tuple[float, ...]] = {}
    segment_modes: dict[str, str] = {}
    start_states: dict[str, dict[str, float]] = {}
    contact_roles: set[str] = set()
    attached_objects = _attached_collision_objects(plan)
    incoming_attached = False
    previous_target_by_role: dict[str, PoseTarget] = {}
    failure = ""

    try:
        for role, target_name in _motion_sequence_for(stop_after, config.active_roles):
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
                candidate_rank=candidate_rank,
                pair_id=pair_id,
            )
            if not state_ok:
                raise RuntimeError(f"{target_name}: could not apply gripper state: {state_message}")
            current_state.update(finger_state)

            target = _target_from_payload(targets[target_name], frame_id=str(config.frame_id))
            active_aabbs = _pregrasp_aabb_obstacles(plan, target_name=target_name)
            if active_aabbs:
                scene_ok, scene_message = commanders["holder"].apply_planning_scene_obstacles(
                    active_aabbs,
                    default_frame_id=str(config.frame_id),
                )
                record(
                    name=f"preflight_apply_{target_name}_part_aabbs",
                    role="shared",
                    ok=scene_ok,
                    message=scene_message,
                    candidate_rank=candidate_rank,
                    pair_id=pair_id,
                )
                if not scene_ok:
                    raise RuntimeError(
                        f"{target_name}: could not apply phase obstacles: {scene_message}"
                    )
            try:
                start_states[target_name] = dict(current_state)
                if target_name in CARTESIAN_TARGETS:
                    previous_target = previous_target_by_role.get(role)
                    if previous_target is None:
                        trajectory = None
                        plan_message = "Cartesian segment has no preceding target"
                    elif not _same_orientation(previous_target, target):
                        trajectory = None
                        plan_message = "Cartesian segment changes TCP orientation"
                    else:
                        trajectory, plan_message = commanders[role].plan_cartesian_to_pose(
                            target,
                            label=target_name,
                            start_robot_state=current_state,
                            max_step_m=float(config.cartesian_max_step_m),
                            revolute_jump_threshold_rad=float(
                                config.cartesian_revolute_jump_threshold_rad
                            ),
                        )
                    segment_mode = "cartesian_linear"
                else:
                    trajectory, joints, plan_message = _plan_free_space_with_multi_seed_ik(
                        commander=commanders[role],
                        target=target,
                        target_name=target_name,
                        role=role,
                        current_state=current_state,
                        config=config,
                        plan=plan,
                        kinematic_ik_cache=kinematic_ik_cache,
                    )
                    segment_mode = "free_space"

                if trajectory is None:
                    failure = f"{target_name}: connected {segment_mode} planning failed: {plan_message}"
                    record(
                        name=f"preflight_plan_{target_name}",
                        role=role,
                        ok=False,
                        message=failure,
                        target=target,
                        candidate_rank=candidate_rank,
                        pair_id=pair_id,
                    )
                    break
                terminal_state, terminal_message = _trajectory_terminal_state(
                    trajectory,
                    current_state=current_state,
                    required_joint_names=_role_spec(plan, role)["joint_names"],
                )
                if terminal_state is None:
                    failure = f"{target_name}: {terminal_message}"
                    record(
                        name=f"preflight_plan_{target_name}",
                        role=role,
                        ok=False,
                        message=failure,
                        target=target,
                        candidate_rank=candidate_rank,
                        pair_id=pair_id,
                    )
                    break
                current_state = terminal_state
                trajectories[target_name] = trajectory
                joint_targets[target_name] = tuple(
                    current_state[str(name)] for name in _role_spec(plan, role)["joint_names"]
                )
                segment_modes[target_name] = segment_mode
                previous_target_by_role[role] = target
                record(
                    name=f"preflight_plan_{target_name}",
                    role=role,
                    ok=True,
                    message=f"{segment_mode} connected plan ready: {plan_message}",
                    target=target,
                    candidate_rank=candidate_rank,
                    pair_id=pair_id,
                )
            finally:
                if active_aabbs:
                    remove_ok, remove_message = commanders["holder"].remove_planning_scene_obstacles(
                        [str(obstacle["id"]) for obstacle in active_aabbs],
                        default_frame_id=str(config.frame_id),
                    )
                    record(
                        name=f"preflight_remove_{target_name}_part_aabbs",
                        role="shared",
                        ok=remove_ok,
                        message=remove_message,
                        candidate_rank=candidate_rank,
                        pair_id=pair_id,
                    )
                    if not remove_ok:
                        raise RuntimeError(
                            f"Could not remove temporary AABBs after preplanning {target_name}: {remove_message}"
                        )
            if failure:
                break

            close_role = GRIPPER_CLOSE_AFTER.get(target_name)
            if close_role is not None:
                contact_roles.add(close_role)
                closed_state = _moveit_gripper_state_for_plan(
                    plan,
                    contact_roles=frozenset(contact_roles),
                )
                current_state.update(closed_state)
                validity, validity_message = commanders[role].check_state_validity(
                    current_state,
                    group_name="",
                )
                if validity is None or not bool(validity["valid"]):
                    contacts = [] if validity is None else validity.get("contacts", [])
                    failure = (
                        f"{target_name}: closed complete state is invalid: "
                        f"{validity_message}; contacts={contacts}"
                    )
                    break
                closed_ok, closed_message = _apply_moveit_gripper_state(
                    commanders["holder"],
                    closed_state,
                )
                if not closed_ok:
                    raise RuntimeError(
                        f"{target_name}: could not apply closed gripper state: {closed_message}"
                    )
                if close_role == "inserter" and attached_objects:
                    attach_ok, attach_message = _apply_attached_collision_objects(
                        commanders["holder"],
                        attached_objects,
                        frame_id=str(config.frame_id),
                    )
                    record(
                        name="preflight_attach_incoming_collision",
                        role="shared",
                        ok=attach_ok,
                        message=attach_message,
                        candidate_rank=candidate_rank,
                        pair_id=pair_id,
                    )
                    if not attach_ok:
                        raise RuntimeError(
                            f"Could not attach incoming collision geometry: {attach_message}"
                        )
                    incoming_attached = True
    finally:
        if incoming_attached:
            detach_ok, detach_message = _remove_attached_collision_objects(
                commanders["holder"],
                attached_objects,
                frame_id=str(config.frame_id),
            )
            record(
                name="preflight_cleanup_attached_incoming_collision",
                role="shared",
                ok=detach_ok,
                message=detach_message,
                candidate_rank=candidate_rank,
                pair_id=pair_id,
            )
            if not detach_ok:
                raise RuntimeError(
                    f"Could not clean up incoming attached collision geometry: {detach_message}"
                )

    if failure:
        return None, failure
    expected_names = tuple(name for _role, name in _motion_sequence_for(stop_after, config.active_roles))
    if tuple(trajectories) != expected_names:
        return None, "connected preflight did not produce every requested trajectory segment"
    return (
        PreplannedDualSequence(
            trajectories=trajectories,
            joint_targets=joint_targets,
            segment_modes=segment_modes,
            start_states=start_states,
        ),
        "ok",
    )


def _select_ranked_preflight_candidate(
    *,
    plan: Mapping[str, object],
    commanders: Mapping[str, object],
    initial_robot_state: Mapping[str, float],
    config: DualRealExecutionConfig,
    frame_id: str,
    record: Callable[..., None],
    stop_after: str = "inserter_preinsertion",
    update_debug: Callable[..., None] | None = None,
) -> tuple[dict[str, object] | None, dict[str, object], PreplannedDualSequence | None]:
    """Select the first producer-ranked pair with a complete connected plan."""

    candidates = _ranked_candidate_plans(plan)
    candidate_records: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    selected_rank: int | None = None
    selected_preplanned: PreplannedDualSequence | None = None
    # A transition expansion often repeats an identical holder/pickup prefix
    # dozens of times. Cache only proven-failing connected prefixes; successful
    # trajectories remain candidate-local because later phases can change the
    # complete-state continuation and attached-object scene.
    failed_prefixes: dict[str, str] = {}
    kinematic_ik_cache: dict[
        tuple[object, ...], tuple[tuple[float, ...] | None, str]
    ] = {}
    cached_prefix_rejections = 0

    def prefix_signature(candidate: Mapping[str, object], through_target: str) -> str:
        prefix = []
        for role, target_name in _motion_sequence_for(through_target, config.active_roles):
            prefix.append(
                {
                    "role": role,
                    "name": target_name,
                    "target": candidate["targets"][target_name],
                    "obstacles": _pregrasp_aabb_obstacles(candidate, target_name=target_name),
                }
            )
        payload = {
            "prefix": prefix,
            # Only jaw widths affect the planning-scene finger state. Do not
            # include the transition-specific ``inserter_preinsertion`` grasp
            # payload when caching an earlier holder/pickup prefix.
            "role_robots": {
                role: candidate["roles"][role]["robot"]
                for role in ("holder", "inserter")
            },
            "jaw_widths_m": {
                role: _gripper_width_for_role(candidate, role, contact=True)
                for role in ("holder", "inserter")
            },
            "attached_objects": (
                _attached_collision_objects(candidate)
                if STOP_AFTER_CHOICES.index(through_target)
                >= STOP_AFTER_CHOICES.index("inserter_pickup_grasp")
                else {}
            ),
            "initial_robot_state": sorted(
                (str(name), round(float(value), 9))
                for name, value in initial_robot_state.items()
            ),
            "ik_candidate_count": int(config.ik_candidate_count),
            "ik_beam_width": int(config.ik_beam_width),
            "ik_seed_perturbation_rad": float(config.ik_seed_perturbation_rad),
            "ik_solution_dedup_tolerance_rad": float(config.ik_solution_dedup_tolerance_rad),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

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
                phase="motion_preflight",
                status="planning",
                message="Preplanning the complete connected dual-arm sequence before hardware motion.",
            )
        cached_failure = ""
        for _role, prefix_target in _motion_sequence_for(stop_after, config.active_roles):
            cached_failure = failed_prefixes.get(prefix_signature(candidate, prefix_target), "")
            if cached_failure:
                break
        if cached_failure:
            cached_prefix_rejections += 1
            preplanned = None
            failure = f"cached identical connected-prefix failure: {cached_failure}"
            print(
                f"[DUAL-REAL] candidate={candidate_id} skipped: {failure}",
                flush=True,
            )
        else:
            preplanned, failure = _preplan_connected_candidate(
                plan=candidate,
                commanders=commanders,
                initial_robot_state=initial_robot_state,
                config=config,
                record=record,
                stop_after=stop_after,
                candidate_rank=rank,
                pair_id=pair_id,
                kinematic_ik_cache=kinematic_ik_cache,
            )
            if preplanned is None:
                failed_target = failure.split(":", 1)[0]
                if failed_target in STOP_AFTER_CHOICES:
                    failed_prefixes[prefix_signature(candidate, failed_target)] = failure
        candidate_ok = preplanned is not None
        active_roles = tuple(dict.fromkeys(role for role, _ in _motion_sequence_for(stop_after, config.active_roles)))
        role_records = {
            role: {
                "grasp_id": _candidate_grasp_id(candidate, role=role),
                "cache_hit": bool(cached_failure),
                "ok": candidate_ok,
                "failure": failure,
            }
            for role in active_roles
        }

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
                phase="motion_preflight",
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
            selected_preplanned = preplanned
            break

    summary = {
        "policy": "producer_ranked_queue_connected_motion_preflight_before_motion",
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
        "selected_joint_targets": (
            {
                name: list(joints)
                for name, joints in selected_preplanned.joint_targets.items()
            }
            if selected_preplanned is not None
            else {}
        ),
        "preplanned_segments": (
            [
                {
                    "name": name,
                    "mode": selected_preplanned.segment_modes[name],
                    "point_count": len(
                        tuple(selected_preplanned.trajectories[name].joint_trajectory.points)
                    ),
                }
                for name in selected_preplanned.trajectories
            ]
            if selected_preplanned is not None
            else []
        ),
        "records": candidate_records,
        "cached_role_grasp_count": cached_prefix_rejections,
        "cached_prefix_rejections": cached_prefix_rejections,
        "kinematic_ik_cache_entries": len(kinematic_ik_cache),
        "stop_after": stop_after,
    }
    return selected, summary, selected_preplanned


def _validate_preplanned_trajectory_start(
    *,
    commander,
    trajectory,
    expected_start_state: Mapping[str, float],
    tolerance_rad: float,
    active_roles: Sequence[str] = ("holder", "inserter"),
    joint_names: Sequence[str] | None = None,
) -> tuple[bool, str]:
    get_state = getattr(commander, "get_current_robot_state", None)
    if not callable(get_state):
        return True, "Live-state guard unavailable in lightweight test adapter."
    current_state = get_state()
    del trajectory
    names = (
        tuple(str(name) for name in joint_names)
        if joint_names is not None
        else tuple(
            str(name)
            for role in _normalized_active_roles(active_roles)
            for name in ROLE_SPECS[role]["joint_names"]
        )
    )
    missing = [name for name in names if name not in current_state or name not in expected_start_state]
    if missing:
        return False, f"Live or preplanned complete state is missing dual-arm joints: {missing}"
    errors = {
        name: abs(float(current_state[name]) - float(expected_start_state[name]))
        for name in names
    }
    max_joint = max(errors, key=errors.get)
    max_error = errors[max_joint]
    if max_error > float(tolerance_rad):
        return (
            False,
            f"Live start state drifted from preflight: {max_joint} error={max_error:.4f} rad "
            f"> tolerance={float(tolerance_rad):.4f} rad.",
        )
    return True, f"Live start state matches preflight; max_error={max_error:.4f} rad."


def _execute_sequence(
    *,
    plan: Mapping[str, object],
    commanders: Mapping[str, object],
    grippers: Mapping[str, object],
    config: DualRealExecutionConfig,
    record: Callable[..., None],
    preplanned_sequence: PreplannedDualSequence,
    policy_approaches: Mapping[str, PreparedPolicyApproach] | None = None,
    candidate_rank: int | None = None,
    update_debug: Callable[..., None] | None = None,
) -> tuple[bool, str, str]:
    targets = plan["targets"]
    assert isinstance(targets, dict)
    last_completed = ""
    contact_roles: set[str] = set()
    attached_objects = _attached_collision_objects(plan)
    incoming_attached = False
    active_roles = _normalized_active_roles(config.active_roles)
    policy_approaches = dict(policy_approaches or {})
    requested_sequence = _motion_sequence_for(config.stop_after, active_roles)
    requested_targets = tuple(name for _role, name in requested_sequence)
    missing_preplans = [name for name in requested_targets if name not in preplanned_sequence.trajectories]
    if missing_preplans:
        return False, "connected_motion_preflight_missing", ""

    def _finish(success: bool, status: str, completed: str) -> tuple[bool, str, str]:
        nonlocal incoming_attached
        if incoming_attached:
            detach_ok, detach_message = _remove_attached_collision_objects(
                commanders["holder"],
                attached_objects,
                frame_id=str(config.frame_id),
            )
            record(
                name="cleanup_attached_incoming_collision",
                role="shared",
                ok=detach_ok,
                message=detach_message,
            )
            incoming_attached = False
            if not detach_ok:
                return False, "attached_collision_cleanup_failed", completed
        return success, status, completed

    for role in active_roles:
        if role not in grippers:
            record(
                name=f"skip_position_{role}_gripper_for_approach",
                role=role,
                ok=True,
                message="Gripper controller is unavailable; arm execution continues without this command.",
            )
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
            return _finish(False, f"{role}_gripper_approach_position_failed", last_completed)

    approach_state = _moveit_gripper_state_for_plan(plan)
    state_ok, state_message = _apply_moveit_gripper_state(commanders["holder"], approach_state)
    record(
        name="apply_approach_gripper_state",
        role="shared",
        ok=state_ok,
        message=state_message,
    )
    if not state_ok:
        return _finish(False, "approach_gripper_moveit_state_failed", last_completed)

    for role, target_name in requested_sequence:
        if update_debug is not None:
            update_debug(
                candidate=plan,
                attempt_index=candidate_rank,
                phase=target_name,
                status="planning",
                message=(
                    f"Executing preplanned {preplanned_sequence.segment_modes[target_name]} "
                    f"{role} target '{target_name}' on hardware."
                ),
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
                return _finish(
                    False,
                    f"{target_name}_aabb_apply_failed",
                    last_completed,
                )
        try:
            trajectory = preplanned_sequence.trajectories[target_name]
            start_ok, start_message = _validate_preplanned_trajectory_start(
                commander=commanders[role],
                trajectory=trajectory,
                expected_start_state=preplanned_sequence.start_states[target_name],
                tolerance_rad=float(config.execution_start_tolerance_rad),
                active_roles=active_roles,
                joint_names=tuple(
                    str(name)
                    for active_role in active_roles
                    for name in _role_spec(plan, active_role)["joint_names"]
                ),
            )
            record(
                name=f"validate_{target_name}_start_state",
                role=role,
                ok=start_ok,
                message=start_message,
                target=target,
            )
            if not start_ok:
                ok = False
                message = start_message
            elif role in policy_approaches and target_name in GRIPPER_CLOSE_AFTER:
                from grasp_planning.ros2.d405_visual_servo import run_d405_policy_visual_servo

                policy = policy_approaches[role]
                policy_result = run_d405_policy_visual_servo(
                    config_path=policy.config_path,
                    expected_grasp_id=policy.grasp_id,
                    expected_part_id=policy.part_id,
                    allow_real_motion=True,
                    preparation=policy.preparation,
                )
                ok = bool(
                    policy_result.completed
                    and policy_result.motion_applied
                    and policy_result.allow_gripper_close
                )
                message = (
                    f"policy state={policy_result.state} completed={policy_result.completed} "
                    f"motion_applied={policy_result.motion_applied} "
                    f"allow_gripper_close={policy_result.allow_gripper_close}; "
                    f"{policy_result.message}; run={policy_result.run_directory}"
                )
            else:
                ok, execute_message = commanders[role].execute_trajectory(
                    trajectory,
                    label=target_name,
                )
                message = (
                    f"executed preplanned {preplanned_sequence.segment_modes[target_name]} trajectory; "
                    f"{execute_message}"
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
            if update_debug is not None:
                update_debug(
                    candidate=plan,
                    attempt_index=candidate_rank,
                    phase=target_name,
                    status="failed",
                    message=message,
                )
            return _finish(False, f"{target_name}_failed", last_completed)
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
        if close_role is not None:
            if close_role in grippers:
                width = _gripper_fully_closed_width(plan, close_role)
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
                    return _finish(False, f"{close_role}_gripper_close_failed", last_completed)
            else:
                record(
                    name=f"skip_position_{close_role}_gripper_for_contact",
                    role=close_role,
                    ok=True,
                    message=(
                        "Gripper controller is unavailable; arm execution and the planned "
                        "MoveIt finger/object state continue without this hardware command."
                    ),
                )
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
                return _finish(False, f"{close_role}_gripper_moveit_state_failed", last_completed)
            if close_role == "inserter" and attached_objects:
                attach_ok, attach_message = _apply_attached_collision_objects(
                    commanders["holder"],
                    attached_objects,
                    frame_id=str(config.frame_id),
                )
                record(
                    name="attach_incoming_collision",
                    role="shared",
                    ok=attach_ok,
                    message=attach_message,
                )
                if not attach_ok:
                    return _finish(False, "incoming_collision_attach_failed", last_completed)
                incoming_attached = True

        if target_name == config.stop_after:
            return _finish(True, f"stopped_at_{target_name}", last_completed)

    return _finish(True, "completed", last_completed)


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
    active_roles = _normalized_active_roles(config.active_roles)
    if config.policy and not config.execute:
        print(
            "[DUAL-REAL] Policy assets and runtime goals will be validated after connected "
            "motion preflight; no policy command is emitted without --execute.",
            flush=True,
        )
    if float(config.cartesian_max_step_m) <= 0.0:
        raise ValueError("cartesian_max_step_m must be positive.")
    if float(config.cartesian_revolute_jump_threshold_rad) <= 0.0:
        raise ValueError("cartesian_revolute_jump_threshold_rad must be positive.")
    if float(config.execution_start_tolerance_rad) <= 0.0:
        raise ValueError("execution_start_tolerance_rad must be positive.")
    source_plan = load_and_validate_dual_plan(plan_json)
    if config.execute and not _has_loaded_part_collision_geometry(source_plan) and not config.allow_objectless_planning:
        raise RuntimeError(
            "Hardware execution plan has no phase-aware workbench-part and "
            "attached incoming-part collision geometry. Rebuild the task, or "
            "pass --allow-objectless-planning only for an explicitly reviewed legacy plan."
        )
    if config.execute and not config.grippers_enabled:
        requested = _motion_sequence_for(config.stop_after, active_roles)
        if any(target_name not in {"holder_pregrasp", "inserter_pickup_pregrasp"} for _, target_name in requested):
            raise RuntimeError("Execution without gripper control is limited to active-role pregrasp targets.")
    if rclpy is None:
        raise RuntimeError(
            "ROS2 MoveIt dependencies are unavailable. Source ROS2, lbr-stack, "
            "and this repository's ros2_ws overlay first."
        )

    plan_json = plan_json.expanduser().resolve()
    output_path = attempt_artifact_path.expanduser().resolve()
    ranked_candidates = _ranked_candidate_plans(source_plan)
    execution_plan = source_plan
    pair_id = ""
    pair_selection: dict[str, object] = {
        "policy": "producer_ranked_queue_connected_motion_preflight_before_motion",
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
    skipped_gripper_roles: tuple[str, ...] = ()
    preplanned_sequence: PreplannedDualSequence | None = None
    policy_approaches: dict[str, PreparedPolicyApproach] = {}
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
        commanders = {
            role: _make_commander(role=role, config=config, plan=source_plan)
            for role in ROLE_SPECS
        }
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

        initial_robot_state = commanders["holder"].get_current_robot_state()
        selected_plan, pair_selection, preplanned_sequence = _select_ranked_preflight_candidate(
            plan=source_plan,
            commanders=commanders,
            initial_robot_state=initial_robot_state,
            config=config,
            frame_id=str(config.frame_id),
            record=_record,
            stop_after=str(config.stop_after),
            update_debug=_update_debug,
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
                "motion_preflight_failed",
                (
                    "No ranked grasp pair passed complete connected motion "
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
                (
                    f"Ranked pair {pair_id} passed complete connected motion preflight; "
                    "hardware execution was not requested."
                ),
                pair_id,
                "",
                output_path,
            )
            return result

        if config.require_confirmation:
            if preplanned_sequence is None:
                raise RuntimeError("Connected motion preflight selected no executable trajectory sequence.")
            policy_approaches = _prepare_policy_approaches(
                plan=execution_plan,
                config=config,
                preplanned_sequence=preplanned_sequence,
                attempt_artifact_path=output_path,
                record=_record,
            )
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
        elif config.policy:
            if preplanned_sequence is None:
                raise RuntimeError("Connected motion preflight selected no executable trajectory sequence.")
            policy_approaches = _prepare_policy_approaches(
                plan=execution_plan,
                config=config,
                preplanned_sequence=preplanned_sequence,
                attempt_artifact_path=output_path,
                record=_record,
            )

        if config.grippers_enabled:
            configured_grippers = {
                role: _make_gripper(
                    role=role,
                    plan=execution_plan,
                    commander=commanders[role],
                    config=config,
                )
                for role in active_roles
            }
            grippers, skipped_gripper_roles = _activate_available_grippers(
                configured_grippers,
                timeout_s=float(config.wait_for_moveit_timeout_s),
                record=_record,
                allow_missing=bool(config.allow_missing_grippers),
            )

        if preplanned_sequence is None:
            raise RuntimeError("Connected motion preflight selected no executable trajectory sequence.")
        success, status, last_completed = _execute_sequence(
            plan=execution_plan,
            commanders=commanders,
            grippers=grippers,
            config=config,
            record=_record,
            preplanned_sequence=preplanned_sequence,
            policy_approaches=policy_approaches,
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
        if skipped_gripper_roles:
            message += (
                " Hardware gripper commands were skipped for unavailable role(s): "
                + ", ".join(skipped_gripper_roles)
                + "."
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
                "motion_preflight"
                if result.status in {"preflight_ok", "motion_preflight_failed"}
                else "complete"
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
    "MOTION_SEQUENCE",
    "STOP_AFTER_CHOICES",
    "execute_dual_real_plan",
    "load_and_validate_dual_plan",
]
