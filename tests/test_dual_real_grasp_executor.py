from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from grasp_planning.gripper_profiles import (
    SERVO_GRIPPER_CLOSED_WIDTH_M,
    SERVO_GRIPPER_OPEN_WIDTH_M,
)
from grasp_planning.pipeline.dual_robot_simple_sim import (
    DEFAULT_FLOOR_Z_WORLD_M,
    NoPoseFeasibleDualTasksError,
)
from grasp_planning.ros2 import dual_real_grasp_executor as dual_executor
from grasp_planning.ros2.dual_real_grasp_executor import (
    CARTESIAN_TARGETS,
    MOTION_SEQUENCE,
    DualRealExecutionConfig,
    PreplannedDualSequence,
    _activate_available_grippers,
    _command_gripper_width,
    _execute_sequence,
    _gripper_side_for_role,
    _preflight_targets,
    _select_ranked_preflight_candidate,
    _validate_preplanned_trajectory_start,
    _work_surface_obstacle,
    load_and_validate_dual_plan,
)
from grasp_planning.start_poses import (
    KUKA_MOVEIT_ARM_START_JOINT_VALUES,
    kuka_gripper_approach_width,
)
from scripts import build_simple_dual_robot_task as task_builder
from scripts.build_simple_dual_robot_task import (
    _include_nonretained_identity_fallbacks,
)


class _Commander:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.scene_calls: list[tuple[str, tuple[str, ...]]] = []
        self.executed: list[str] = []

    def move_to_pose(self, target, *, label: str, execute: bool):
        self.calls.append((label, execute))
        return True, f"{label} executed"

    def execute_trajectory(self, trajectory, *, label: str):
        del trajectory
        self.executed.append(label)
        return True, f"{label} executed"

    def apply_planning_scene_obstacles(
        self,
        obstacles,
        *,
        default_frame_id: str,
    ):
        del default_frame_id
        ids = tuple(str(obstacle["id"]) for obstacle in obstacles)
        self.scene_calls.append(("apply", ids))
        return True, f"applied {len(ids)}"

    def remove_planning_scene_obstacles(
        self,
        obstacle_ids,
        *,
        default_frame_id: str,
    ):
        del default_frame_id
        ids = tuple(str(value) for value in obstacle_ids)
        self.scene_calls.append(("remove", ids))
        return True, f"removed {len(ids)}"

    def apply_planning_scene_attached_obstacles(
        self,
        obstacles,
        *,
        default_frame_id: str,
    ):
        del default_frame_id
        ids = tuple(str(obstacle["id"]) for obstacle in obstacles)
        self.scene_calls.append(("attach", ids))
        return True, f"attached {len(ids)}"

    def remove_planning_scene_attached_obstacles(
        self,
        obstacles,
        *,
        default_frame_id: str,
    ):
        del default_frame_id
        ids = tuple(str(obstacle["id"]) for obstacle in obstacles)
        self.scene_calls.append(("detach", ids))
        return True, f"detached {len(ids)}"


class _Gripper:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def open(self, *, width: float):
        self.calls.append(("open", width))
        return True, "opened"

    def close(self, *, width: float):
        self.calls.append(("close", width))
        return True, "closed"


class _DiscoverableGripper:
    def __init__(self, role: str, events: list[str], *, available: bool) -> None:
        self.role = role
        self.events = events
        self.available = available

    def wait_for_server(self, *, timeout_s: float) -> None:
        assert timeout_s == 20.0
        self.events.append(f"wait:{self.role}")
        if not self.available:
            raise RuntimeError(f"Normalized gripper open service '/{self.role}/open' is unavailable.")

    def initialize_open(self) -> tuple[bool, str]:
        self.events.append(f"open:{self.role}")
        return True, "opened"


class _FallbackIkCommander:
    def __init__(self) -> None:
        self.seeds = []

    def compute_ik(self, target, seed_joint_positions=None):
        del target
        self.seeds.append(seed_joint_positions)
        if seed_joint_positions is None:
            return None, "live failed"
        return tuple(seed_joint_positions), "alternate ok"


class _FallbackIkExecutionCommander(_Commander, _FallbackIkCommander):
    def __init__(self) -> None:
        _Commander.__init__(self)
        _FallbackIkCommander.__init__(self)
        self.joint_plans: list[tuple[str, tuple[float, ...]]] = []
        self.executed: list[str] = []

    def move_to_pose(self, target, *, label: str, execute: bool):
        del target, label, execute
        raise AssertionError("Execution must not recompute pose IK after preflight.")

    def plan_to_joint_positions(self, joint_positions, *, label: str):
        values = tuple(float(value) for value in joint_positions)
        self.joint_plans.append((label, values))
        return object(), "joint plan ok"

    def execute_trajectory(self, trajectory, *, label: str):
        del trajectory
        self.executed.append(label)
        return True, f"{label} executed"


class _CleanupFailPreflightCommander(_Commander):
    def compute_ik(self, target, seed_joint_positions=None):
        del target, seed_joint_positions
        return tuple(KUKA_MOVEIT_ARM_START_JOINT_VALUES), "ok"

    def remove_planning_scene_attached_obstacles(
        self,
        obstacles,
        *,
        default_frame_id: str,
    ):
        del default_frame_id
        ids = tuple(str(obstacle["id"]) for obstacle in obstacles)
        self.scene_calls.append(("detach_failed", ids))
        return False, "synthetic world purge failure"


class _RankedIkCommander(_Commander):
    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role
        self.calls = 0

    def compute_ik(self, target, seed_joint_positions=None, seed_robot_state=None):
        del seed_joint_positions, seed_robot_state
        self.calls += 1
        return [float(target.x)] * 7, "ok"

    @property
    def joint_names(self) -> tuple[str, ...]:
        robot = "lbr_one" if self.role == "holder" else "lbr_two"
        return tuple(f"{robot}_A{index}" for index in range(1, 8))

    def plan_to_joint_positions(self, joints, *, label: str, start_robot_state):
        if float(joints[0]) > 0.9:
            return None, f"synthetic connected motion failure at {label}"
        start = tuple(float(start_robot_state[name]) for name in self.joint_names)
        return _fake_trajectory(self.joint_names, start, tuple(joints)), "ok"

    def plan_cartesian_to_pose(
        self,
        target,
        *,
        label: str,
        start_robot_state,
        max_step_m: float,
        revolute_jump_threshold_rad: float,
    ):
        del label, max_step_m, revolute_jump_threshold_rad
        if target.x > 0.9:
            return None, "synthetic Cartesian failure"
        start = tuple(float(start_robot_state[name]) for name in self.joint_names)
        terminal = tuple(value + 0.01 for value in start)
        return _fake_trajectory(self.joint_names, start, terminal), "ok"

    def check_state_validity(self, state, *, group_name: str):
        del state, group_name
        return {"valid": True, "contacts": []}, "ok"


def _recorded_steps():
    steps = []

    def record(**kwargs):
        steps.append(kwargs)

    return steps, record


def _fake_trajectory(
    joint_names: tuple[str, ...],
    start: tuple[float, ...],
    terminal: tuple[float, ...],
):
    return SimpleNamespace(
        joint_trajectory=SimpleNamespace(
            joint_names=list(joint_names),
            points=[
                SimpleNamespace(positions=list(start)),
                SimpleNamespace(positions=list(terminal)),
            ],
        )
    )


def _initial_robot_state() -> dict[str, float]:
    return {
        f"lbr_{robot}_A{index}": 0.0
        for robot in ("one", "two")
        for index in range(1, 8)
    }


def _preplanned_sequence(stop_after: str) -> PreplannedDualSequence:
    trajectories = {}
    joint_targets = {}
    segment_modes = {}
    start_states = {}
    full_state = _initial_robot_state()
    for role, name in MOTION_SEQUENCE:
        robot = "one" if role == "holder" else "two"
        names = tuple(f"lbr_{robot}_A{index}" for index in range(1, 8))
        trajectories[name] = _fake_trajectory(names, (0.0,) * 7, (0.1,) * 7)
        joint_targets[name] = (0.1,) * 7
        segment_modes[name] = "cartesian_linear" if name in CARTESIAN_TARGETS else "free_space"
        start_states[name] = dict(full_state)
        if name == stop_after:
            break
    return PreplannedDualSequence(
        trajectories=trajectories,
        joint_targets=joint_targets,
        segment_modes=segment_modes,
        start_states=start_states,
    )


def _plan_payload() -> dict[str, object]:
    targets = {
        target_name: {
            "position_world_m": [0.5, 0.0, 0.1],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        }
        for _, target_name in MOTION_SEQUENCE
    }
    return {
        "schema_version": 1,
        "kind": "dual_robot_simple_sim_task",
        "pair_id": "p001_h0001_i0_0002",
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
        "targets": targets,
        "grasps": {
            "holder": {"grasp_id": "h0001", "jaw_width_m": 0.040},
            "inserter_pickup": {
                "grasp_id": "i0_0002",
                "jaw_width_m": 0.043,
            },
        },
    }


def _write_plan(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    plan_path = tmp_path / "dual_plan.json"
    plan_path.write_text(
        json.dumps(_plan_payload() if payload is None else payload),
        encoding="utf-8",
    )
    return plan_path


def test_load_and_validate_dual_plan_accepts_saved_vertical_slice(
    tmp_path: Path,
) -> None:
    payload = load_and_validate_dual_plan(_write_plan(tmp_path))

    assert payload["pair_id"] == "p001_h0001_i0_0002"
    assert payload["roles"]["holder"]["robot"] == "lbr_one"
    assert payload["roles"]["inserter"]["robot"] == "lbr_two"


def test_load_and_validate_dual_plan_accepts_pdz_tcp_and_passive_state(
    tmp_path: Path,
) -> None:
    payload = _plan_payload()
    payload["gripper_model"] = "pdz_gripper"
    payload["roles"]["holder"]["tcp_link"] = "lbr_one_pdz_gripper_tcp"
    payload["roles"]["inserter"]["tcp_link"] = "lbr_two_pdz_gripper_tcp"

    loaded = load_and_validate_dual_plan(_write_plan(tmp_path, payload))

    assert dual_executor._role_spec(loaded, "holder")["pose_link"] == "lbr_one_pdz_gripper_tcp"
    assert dual_executor._role_spec(loaded, "inserter")["pose_link"] == "lbr_two_pdz_gripper_tcp"
    state = dual_executor._moveit_gripper_state_for_plan(loaded)
    assert set(state) == {
        "lbr_one_pdz_gripper_left_finger_joint",
        "lbr_two_pdz_gripper_left_finger_joint",
    }


def test_dual_plan_infers_pdz_model_from_tcp_for_compatibility() -> None:
    payload = _plan_payload()
    payload["roles"]["holder"]["tcp_link"] = "lbr_one_pdz_gripper_tcp"
    payload["roles"]["inserter"]["tcp_link"] = "lbr_two_pdz_gripper_tcp"

    dual_executor._validate_dual_plan_payload(payload, context="PDZ compatibility plan")
    assert dual_executor._gripper_model_for_plan(payload) == "pdz_gripper"


def test_load_and_validate_dual_plan_rejects_internally_inconsistent_role(tmp_path: Path) -> None:
    payload = _plan_payload()
    payload["roles"]["holder"]["robot"] = "lbr_two"

    try:
        load_and_validate_dual_plan(_write_plan(tmp_path, payload))
    except ValueError as exc:
        assert "expected 'arm_two'" in str(exc)
    else:
        raise AssertionError("Expected mismatched role mapping to fail.")


def test_load_and_validate_dual_plan_accepts_complete_role_swap(tmp_path: Path) -> None:
    payload = _plan_payload()
    payload["roles"] = {
        "holder": {
            "robot": "lbr_two",
            "planning_group": "arm_two",
            "tcp_link": "lbr_two_gripper_tcp",
        },
        "inserter": {
            "robot": "lbr_one",
            "planning_group": "arm_one",
            "tcp_link": "lbr_one_gripper_tcp",
        },
    }

    loaded = load_and_validate_dual_plan(_write_plan(tmp_path, payload))
    assert loaded["roles"]["holder"]["robot"] == "lbr_two"
    assert loaded["roles"]["inserter"]["robot"] == "lbr_one"
    assert _gripper_side_for_role(loaded, "holder") == "right"
    assert _gripper_side_for_role(loaded, "inserter") == "left"


def test_dual_gripper_clients_route_swapped_roles_by_physical_robot(monkeypatch) -> None:
    plan = _plan_payload()
    plan["roles"] = {
        "holder": {
            "robot": "lbr_two",
            "planning_group": "arm_two",
            "tcp_link": "lbr_two_gripper_tcp",
        },
        "inserter": {
            "robot": "lbr_one",
            "planning_group": "arm_one",
            "tcp_link": "lbr_one_gripper_tcp",
        },
    }
    created: list[dict[str, object]] = []

    class _Client:
        def __init__(self, node, **kwargs) -> None:
            created.append({"node": node, **kwargs})

    monkeypatch.setattr(dual_executor, "NormalizedPositionGripperClient", _Client)
    config = DualRealExecutionConfig()
    dual_executor._make_gripper(role="holder", plan=plan, commander="holder-node", config=config)
    dual_executor._make_gripper(role="inserter", plan=plan, commander="inserter-node", config=config)

    assert created[0]["position_command_topic"] == "/right/gripper_controller/position_command"
    assert created[0]["close_service_name"] == "/right/gripper_controller/close"
    assert created[1]["position_command_topic"] == "/left/gripper_controller/position_command"
    assert created[1]["close_service_name"] == "/left/gripper_controller/close"
    assert created[0]["closed_width_m"] == SERVO_GRIPPER_CLOSED_WIDTH_M
    assert created[0]["open_width_m"] == SERVO_GRIPPER_OPEN_WIDTH_M


def test_dual_gripper_uses_position_for_approach_and_close_service_for_contact() -> None:
    calls: list[tuple[str, float]] = []

    class _Client:
        def command_width(self, width_m, *, wait_for_feedback, settle_after_command):
            assert wait_for_feedback is True
            assert settle_after_command is False
            calls.append(("position", float(width_m)))
            return True, "positioned"

        def close(self, *, width):
            calls.append(("close", float(width)))
            return True, "closed"

    client = _Client()
    assert _command_gripper_width(client, 0.050, approach=True)[0]
    assert _command_gripper_width(client, SERVO_GRIPPER_CLOSED_WIDTH_M, approach=False)[0]
    assert calls == [("position", 0.050), ("close", SERVO_GRIPPER_CLOSED_WIDTH_M)]


def test_ranked_plan_accepts_higher_score_after_retained_queue_boundary(
    tmp_path: Path,
) -> None:
    retained = _plan_payload()
    retained["candidate_rank"] = 1
    retained["selection_score"] = 0.46

    fallback = _plan_payload()
    fallback["candidate_rank"] = 2
    fallback["pair_id"] = "p002_h0007_i0_0496"
    fallback["execution_candidate_id"] = "p002_h0007_i0_0496__tr_symmetric"
    fallback["selection_score"] = 0.64

    plan = dict(retained)
    plan["ranked_pair_candidates"] = [retained, fallback]

    loaded = load_and_validate_dual_plan(_write_plan(tmp_path, plan))

    assert loaded["ranked_pair_candidates"][1]["selection_score"] == 0.64


def test_ranked_plan_rejects_candidate_rank_that_disagrees_with_list_order(
    tmp_path: Path,
) -> None:
    first = _plan_payload()
    first["candidate_rank"] = 1

    second = _plan_payload()
    second["candidate_rank"] = 3
    second["pair_id"] = "p002_h0001_i0_0003"

    plan = dict(first)
    plan["ranked_pair_candidates"] = [first, second]

    try:
        load_and_validate_dual_plan(_write_plan(tmp_path, plan))
    except ValueError as exc:
        assert "candidate_rank=3; expected 2" in str(exc)
    else:
        raise AssertionError("Expected mismatched explicit candidate rank to fail.")


def test_execute_sequence_closes_each_gripper_and_stops_at_preinsertion(
    tmp_path: Path,
) -> None:
    plan = load_and_validate_dual_plan(_write_plan(tmp_path))
    commanders = {"holder": _Commander(), "inserter": _Commander()}
    grippers = {"holder": _Gripper(), "inserter": _Gripper()}
    steps, record = _recorded_steps()
    debug_updates = []
    config = DualRealExecutionConfig(
        execute=True,
        allow_objectless_planning=True,
        stop_after="inserter_preinsertion",
    )

    success, status, last_completed = _execute_sequence(
        plan=plan,
        commanders=commanders,
        grippers=grippers,
        config=config,
        record=record,
        preplanned_sequence=_preplanned_sequence(config.stop_after),
        candidate_rank=4,
        update_debug=lambda **entry: debug_updates.append(entry),
    )

    assert success is True
    assert status == "stopped_at_inserter_preinsertion"
    assert last_completed == "inserter_preinsertion"
    assert commanders["holder"].executed == ["holder_pregrasp", "holder_grasp"]
    assert commanders["inserter"].executed == [
        target_name for role, target_name in MOTION_SEQUENCE if role == "inserter"
    ]
    assert grippers["holder"].calls == [
        ("open", kuka_gripper_approach_width(plan["grasps"]["holder"]["jaw_width_m"])),
        ("close", SERVO_GRIPPER_CLOSED_WIDTH_M),
    ]
    assert grippers["inserter"].calls == [
        ("open", kuka_gripper_approach_width(plan["grasps"]["inserter_pickup"]["jaw_width_m"])),
        ("close", SERVO_GRIPPER_CLOSED_WIDTH_M),
    ]
    assert steps[-1]["name"] == "inserter_preinsertion"
    assert any(update["phase"] == "holder_pregrasp" and update["status"] == "planning" for update in debug_updates)
    assert debug_updates[-1]["phase"] == "inserter_preinsertion"
    assert debug_updates[-1]["status"] == "succeeded"
    assert all(update["attempt_index"] == 4 for update in debug_updates)


def test_execution_start_guard_checks_both_arms_against_preflight() -> None:
    class LiveCommander:
        def get_current_robot_state(self):
            state = _initial_robot_state()
            state["lbr_one_A4"] = 0.08
            return state

    trajectory = _fake_trajectory(
        tuple(f"lbr_two_A{index}" for index in range(1, 8)),
        (0.0,) * 7,
        (0.1,) * 7,
    )
    ok, message = _validate_preplanned_trajectory_start(
        commander=LiveCommander(),
        trajectory=trajectory,
        expected_start_state=_initial_robot_state(),
        tolerance_rad=0.05,
    )

    assert ok is False
    assert "lbr_one_A4" in message
    assert "0.0800 rad" in message


def test_gripper_activation_skips_unavailable_role_after_probing_both() -> None:
    events: list[str] = []
    steps, record = _recorded_steps()
    configured = {
        "holder": _DiscoverableGripper("holder", events, available=True),
        "inserter": _DiscoverableGripper("inserter", events, available=False),
    }

    available, skipped = _activate_available_grippers(
        configured,
        timeout_s=20.0,
        record=record,
        allow_missing=True,
    )

    assert tuple(available) == ("holder",)
    assert skipped == ("inserter",)
    assert events == ["wait:holder", "wait:inserter", "open:holder"]
    assert [step["name"] for step in steps] == [
        "wait_for_holder_gripper",
        "skip_inserter_gripper_unavailable",
        "initialize_holder_gripper_open",
    ]


def test_real_gripper_activation_requires_every_active_role_by_default() -> None:
    configured = {
        "inserter": _DiscoverableGripper("inserter", [], available=False),
    }

    with pytest.raises(RuntimeError, match="Required inserter gripper endpoint is unavailable"):
        _activate_available_grippers(
            configured,
            timeout_s=20.0,
            record=lambda **_kwargs: None,
        )


def test_execute_sequence_continues_when_inserter_gripper_is_unavailable(
    tmp_path: Path,
) -> None:
    payload = _plan_payload()
    payload["moveit"] = {
        "attached_collision_geometry": {
            "objects": {
                "incoming": {
                    "id": "attached_incoming",
                    "link_name": "lbr_two_gripper_tcp",
                }
            }
        }
    }
    plan = load_and_validate_dual_plan(_write_plan(tmp_path, payload))
    commanders = {"holder": _Commander(), "inserter": _Commander()}
    holder_gripper = _Gripper()
    steps, record = _recorded_steps()

    success, status, last_completed = _execute_sequence(
        plan=plan,
        commanders=commanders,
        grippers={"holder": holder_gripper},
        config=DualRealExecutionConfig(
            execute=True,
            stop_after="inserter_preinsertion",
        ),
        record=record,
        preplanned_sequence=_preplanned_sequence("inserter_preinsertion"),
    )

    assert success is True
    assert status == "stopped_at_inserter_preinsertion"
    assert last_completed == "inserter_preinsertion"
    assert commanders["inserter"].executed == [
        name for role, name in MOTION_SEQUENCE if role == "inserter"
    ]
    assert holder_gripper.calls == [
        ("open", kuka_gripper_approach_width(plan["grasps"]["holder"]["jaw_width_m"])),
        ("close", SERVO_GRIPPER_CLOSED_WIDTH_M),
    ]
    assert any(step["name"] == "skip_position_inserter_gripper_for_approach" for step in steps)
    assert any(step["name"] == "skip_position_inserter_gripper_for_contact" for step in steps)
    assert ("attach", ("attached_incoming",)) in commanders["holder"].scene_calls
    assert ("detach", ("attached_incoming",)) in commanders["holder"].scene_calls


def test_execute_sequence_stops_before_holder_close_at_pregrasp(
    tmp_path: Path,
) -> None:
    plan = load_and_validate_dual_plan(_write_plan(tmp_path))
    commanders = {"holder": _Commander(), "inserter": _Commander()}
    grippers = {"holder": _Gripper(), "inserter": _Gripper()}
    _, record = _recorded_steps()
    config = DualRealExecutionConfig(
        execute=True,
        allow_objectless_planning=True,
        stop_after="holder_pregrasp",
    )

    success, status, last_completed = _execute_sequence(
        plan=plan,
        commanders=commanders,
        grippers=grippers,
        config=config,
        record=record,
        preplanned_sequence=_preplanned_sequence(config.stop_after),
    )

    assert success is True
    assert status == "stopped_at_holder_pregrasp"
    assert last_completed == "holder_pregrasp"
    assert commanders["inserter"].executed == []
    assert grippers["holder"].calls == [
        ("open", kuka_gripper_approach_width(plan["grasps"]["holder"]["jaw_width_m"]))
    ]
    assert grippers["inserter"].calls == [
        ("open", kuka_gripper_approach_width(plan["grasps"]["inserter_pickup"]["jaw_width_m"]))
    ]


def test_execute_sequence_uses_aabbs_only_for_pregrasp_transit(
    tmp_path: Path,
) -> None:
    payload = _plan_payload()
    payload["moveit"] = {
        "pregrasp_aabb_collision_geometry": {
            "obstacles": {
                "subassembly": {"id": "base_aabb"},
                "incoming_pickup": {"id": "incoming_aabb"},
            },
            "active_by_target": {
                "holder_pregrasp": [
                    "subassembly",
                    "incoming_pickup",
                ],
                "inserter_pickup_pregrasp": [
                    "incoming_pickup",
                ],
            },
            "removed_before_grasp_approach": True,
        }
    }
    plan = load_and_validate_dual_plan(_write_plan(tmp_path, payload))
    commanders = {"holder": _Commander(), "inserter": _Commander()}
    grippers = {"holder": _Gripper(), "inserter": _Gripper()}
    _, record = _recorded_steps()

    success, status, _ = _execute_sequence(
        plan=plan,
        commanders=commanders,
        grippers=grippers,
        config=DualRealExecutionConfig(
            execute=True,
            allow_objectless_planning=True,
            stop_after="inserter_pickup_pregrasp",
        ),
        record=record,
        preplanned_sequence=_preplanned_sequence("inserter_pickup_pregrasp"),
    )

    assert success is True
    assert status == "stopped_at_inserter_pickup_pregrasp"
    assert commanders["holder"].scene_calls == [
        ("apply", ("base_aabb", "incoming_aabb")),
        ("remove", ("base_aabb", "incoming_aabb")),
        ("apply", ("incoming_aabb",)),
        ("remove", ("incoming_aabb",)),
    ]


def test_execute_sequence_keeps_subassembly_and_attached_incoming_in_transition_scene(
    tmp_path: Path,
) -> None:
    payload = _plan_payload()
    payload["moveit"] = {
        "pregrasp_aabb_collision_geometry": {
            "obstacles": {
                "base": {"id": "base_aabb"},
                "incoming": {"id": "incoming_aabb"},
            },
            "active_by_target": {
                "holder_pregrasp": ["base", "incoming"],
                "holder_grasp": ["base", "incoming"],
                "inserter_pickup_pregrasp": ["base", "incoming"],
                "inserter_pickup_grasp": ["base", "incoming"],
                "inserter_pickup_lift": ["base"],
                "inserter_above_preinsertion": ["base"],
                "inserter_preinsertion": ["base"],
            },
        },
        "attached_collision_geometry": {
            "objects": {
                "incoming": {
                    "id": "attached_incoming",
                    "link_name": "lbr_two_gripper_tcp",
                }
            }
        },
    }
    plan = load_and_validate_dual_plan(_write_plan(tmp_path, payload))
    commanders = {"holder": _Commander(), "inserter": _Commander()}
    grippers = {"holder": _Gripper(), "inserter": _Gripper()}
    _, record = _recorded_steps()

    success, status, last_completed = _execute_sequence(
        plan=plan,
        commanders=commanders,
        grippers=grippers,
        config=DualRealExecutionConfig(
            execute=True,
            stop_after="inserter_preinsertion",
        ),
        record=record,
        preplanned_sequence=_preplanned_sequence("inserter_preinsertion"),
    )

    assert success is True
    assert status == "stopped_at_inserter_preinsertion"
    assert last_completed == "inserter_preinsertion"
    scene_calls = commanders["holder"].scene_calls
    assert ("attach", ("attached_incoming",)) in scene_calls
    assert ("detach", ("attached_incoming",)) in scene_calls
    attach_index = scene_calls.index(("attach", ("attached_incoming",)))
    transition_apply_index = scene_calls.index(("apply", ("base_aabb",)), attach_index)
    assert attach_index < transition_apply_index


def test_dual_gripper_launch_has_stable_namespaces_and_usb_ids() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/gripper_computer/dual_grippers.launch.py").read_text(
        encoding="utf-8"
    )

    assert "namespace=role" in source
    assert '_gripper_node(role="left", port_argument="left_port")' in source
    assert '_gripper_node(role="right", port_argument="right_port")' in source
    assert "usb-1a86_USB_Single_Serial_5B3D047592-if00" in source
    assert "usb-1a86_USB_Single_Serial_5B3D044069-if00" in source


def test_dual_startup_scripts_force_one_shared_ros_domain() -> None:
    root = Path(__file__).resolve().parents[1]
    gripper_start = (root / "scripts/gripper_computer/start_dual_grippers.sh").read_text(encoding="utf-8")
    dual_run = (root / "scripts/run_dual_pipeline.sh").read_text(encoding="utf-8")
    moveit_start = (root / "start_dual_lbr_moveit.sh").read_text(encoding="utf-8")

    for source in (gripper_start, dual_run, moveit_start):
        assert 'export ROS_DOMAIN_ID="${ROS_DOMAIN' in source
        assert "export ROS_LOCALHOST_ONLY=0" in source
        assert "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" in source
        assert "unset ROS_DISCOVERY_SERVER" in source
    expected_local_domain_fallback = 'ROS_DOMAIN_VALUE="${DUAL_ROBOT_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}"'
    assert expected_local_domain_fallback in moveit_start
    assert expected_local_domain_fallback in dual_run
    assert 'export ROS_DOMAIN_ID="${ROS_DOMAIN_VALUE}"' in moveit_start


def test_one_command_runner_routes_fresh_sim_and_real_planning() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_dual_pipeline.sh").read_text(encoding="utf-8")
    public = (root / "run_pipeline.sh").read_text(encoding="utf-8")
    compatibility = (root / "run_simple_dual_robot.sh").read_text(encoding="utf-8")

    assert "scripts/plan_simple_dual_robot_sim.py" in source
    assert "scripts/run_simple_dual_robot_sim_in_isaac.py" in source
    assert "scripts/build_simple_dual_robot_task.py" in source
    assert "scripts/run_simple_dual_robot_real.py" in source
    assert "./start_dual_lbr_moveit.sh" in source
    assert "ros2 service list --no-daemon" in source
    assert 'grep -qx "/lbr_dual_arm/compute_ik"' in source
    assert 'grep -qx "/lbr_dual_arm/plan_kinematic_path"' in source
    assert 'FLOOR_Z="-0.030"' in source
    assert 'ASSEMBLY_Z=""' in source
    assert 'COMMON_TASK_ARGS+=(--assembly-z "${ASSEMBLY_Z}")' in source
    assert 'MAX_PAIR_ATTEMPTS="256"' in source
    assert 'PLANNING_DEBUG_GUI_PORT="${DUAL_REAL_PLANNING_DEBUG_GUI_PORT:-38825}"' in source
    assert "TASK_BUILD_ARGS+=(--debug-gui --debug-gui-port" in source
    assert "--no-debug-gui-open-browser" in source
    assert "scripts/run_unified_pipeline.py" in public
    assert 'exec "${SCRIPT_DIR}/run_pipeline.sh" --workflow dual "$@"' in compatibility

    moveit_start = (Path(__file__).resolve().parents[1] / "start_dual_lbr_moveit.sh").read_text(encoding="utf-8")
    assert "ros2 node list --no-daemon" in moveit_start


def test_real_task_adds_only_identity_fallbacks_without_a_fixed_pair() -> None:
    assert _include_nonretained_identity_fallbacks("") is True
    assert _include_nonretained_identity_fallbacks("p001_h0001_i0_0002") is False


def test_real_task_debugger_starts_before_empty_pickup_floor_filter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, object]] = []

    class FakeDebugServer:
        def __init__(self, *, port: int) -> None:
            events.append(("init", port))

        def start(self, *, open_browser: bool) -> str:
            events.append(("start", open_browser))
            return "http://127.0.0.1:38825/"

        def update(self, **payload) -> None:
            events.append(("update", payload))

        def close(self) -> None:
            events.append(("close", None))

    args = SimpleNamespace(
        artifact_root=tmp_path,
        artifact_dir=tmp_path,
        assembly="synthetic",
        incoming_part_id="1",
        step_id="step_001_part_1",
        pair_id="",
        max_pair_candidates=256,
        assembly_x=0.55,
        assembly_y=0.0,
        assembly_z=-0.03,
        assembly_yaw_deg=0.0,
        pickup_x=0.55,
        pickup_y=0.28,
        pickup_roll_deg=0.0,
        pickup_pitch_deg=0.0,
        pickup_yaw_deg=0.0,
        floor_z=-0.03,
        transport_clearance_m=0.08,
        output=tmp_path / "task.json",
        debug_gui=True,
        debug_gui_port=38825,
        debug_gui_open_browser=True,
    )
    selection = SimpleNamespace(
        artifact_dir=tmp_path,
        step_id="step_001_part_1",
    )

    def reject_all(**_kwargs):
        assert any(
            event == "update" and payload["status"] == "planning"
            for event, payload in events
            if isinstance(payload, dict)
        )
        raise NoPoseFeasibleDualTasksError(
            "all pickup grasps collide with the floor",
            candidate_filter_diagnostics={
                "pickup_floor_z_world_m": -0.03,
                "pickup_grasps_checked": 783,
                "pickup_grasps_accepted": 0,
                "pickup_grasps_rejected": 783,
            },
        )

    monkeypatch.setattr(task_builder, "_parse_args", lambda: args)
    monkeypatch.setattr(task_builder, "resolve_dual_robot_step_selection", lambda **_kwargs: selection)
    monkeypatch.setattr(task_builder, "DualRobotPlanningDebugServer", FakeDebugServer)
    monkeypatch.setattr(task_builder, "load_simple_dual_robot_pair_tasks", reject_all)
    monkeypatch.setattr(task_builder.time, "sleep", lambda _seconds: None)

    try:
        task_builder.main()
    except NoPoseFeasibleDualTasksError:
        pass
    else:
        raise AssertionError("Expected the synthetic pickup-floor rejection.")

    updates = [payload for event, payload in events if event == "update"]
    assert updates[0]["phase"] == "pickup_floor_check"
    assert updates[0]["status"] == "planning"
    assert updates[-1]["status"] == "fatal"
    assert updates[-1]["candidate_counts"]["pickup_grasps_checked"] == 783
    assert events[-1][0] == "close"


def test_default_dual_work_surface_top_is_minus_thirty_mm() -> None:
    obstacle = _work_surface_obstacle({})
    center_z = float(obstacle["xyz"][2])
    height = float(obstacle["size_m"][2])

    assert DEFAULT_FLOOR_Z_WORLD_M == -0.030
    assert math.isclose(center_z + 0.5 * height, -0.030)

    config_source = (Path(__file__).resolve().parents[1] / "configs" / "dual_grasp_planning.yaml").read_text(
        encoding="utf-8"
    )
    assert "floor_z_world_m: -0.030" in config_source


def test_preflight_retries_ik_with_known_start_seed(tmp_path: Path) -> None:
    plan = load_and_validate_dual_plan(_write_plan(tmp_path))
    commanders = {
        "holder": _FallbackIkCommander(),
        "inserter": _FallbackIkCommander(),
    }
    steps, record = _recorded_steps()

    assert (
        _preflight_targets(
            plan=plan,
            commanders=commanders,
            frame_id="base_link",
            record=record,
        )
        is True
    )
    target_steps = [step for step in steps if not step["name"].endswith("_gripper_state")]
    assert all("multi-seed IK succeeded" in step["message"] for step in target_steps)
    for commander in commanders.values():
        assert commander.seeds
        assert all(seed is not None for seed in commander.seeds)


def test_connected_preflight_uses_later_ik_seed_when_continuation_seed_fails() -> None:
    class LaterSeedCommander(_RankedIkCommander):
        def __init__(self, role: str) -> None:
            super().__init__(role)
            self.ik_seeds = []

        def compute_ik(self, target, *, seed_robot_state=None, avoid_collisions=None):
            del avoid_collisions
            seed = tuple(float(seed_robot_state[name]) for name in self.joint_names)
            self.ik_seeds.append(seed)
            if len(self.ik_seeds) == 1:
                return None, "synthetic continuation-branch failure"
            return [float(target.x)] * 7, "later branch ok"

    candidate = _plan_payload()
    candidate["ranked_pair_candidates"] = [dict(candidate)]
    commanders = {
        "holder": LaterSeedCommander("holder"),
        "inserter": LaterSeedCommander("inserter"),
    }
    steps, record = _recorded_steps()

    selected, _summary, preplanned = _select_ranked_preflight_candidate(
        plan=candidate,
        commanders=commanders,
        initial_robot_state=_initial_robot_state(),
        config=DualRealExecutionConfig(stop_after="holder_pregrasp"),
        frame_id="base_link",
        record=record,
        stop_after="holder_pregrasp",
    )

    assert selected is not None
    assert preplanned is not None
    assert len(commanders["holder"].ik_seeds) >= 2
    planned = next(step for step in steps if step["name"] == "preflight_plan_holder_pregrasp")
    assert "selected seed 1" in planned["message"]


def test_connected_preflight_tries_later_ik_solution_when_first_motion_plan_fails() -> None:
    class LaterMotionBranchCommander(_RankedIkCommander):
        def __init__(self, role: str) -> None:
            super().__init__(role)
            self.ik_calls = 0

        def compute_ik(self, target, *, seed_robot_state=None, avoid_collisions=None):
            del target, seed_robot_state, avoid_collisions
            self.ik_calls += 1
            return [0.8 if self.ik_calls == 1 else 0.5] * 7, "ok"

        def plan_to_joint_positions(self, joints, *, label: str, start_robot_state):
            if float(joints[0]) > 0.7:
                return None, "synthetic first-branch motion failure"
            return super().plan_to_joint_positions(joints, label=label, start_robot_state=start_robot_state)

    candidate = _plan_payload()
    candidate["ranked_pair_candidates"] = [dict(candidate)]
    commanders = {
        "holder": LaterMotionBranchCommander("holder"),
        "inserter": LaterMotionBranchCommander("inserter"),
    }
    steps, record = _recorded_steps()

    selected, _summary, preplanned = _select_ranked_preflight_candidate(
        plan=candidate,
        commanders=commanders,
        initial_robot_state=_initial_robot_state(),
        config=DualRealExecutionConfig(stop_after="holder_pregrasp"),
        frame_id="base_link",
        record=record,
        stop_after="holder_pregrasp",
    )

    assert selected is not None
    assert preplanned is not None
    planned = next(step for step in steps if step["name"] == "preflight_plan_holder_pregrasp")
    assert "selected seed 1" in planned["message"]
    assert "2 distinct solutions" in planned["message"]


def test_connected_preflight_trajectories_are_reused_without_replanning(
    tmp_path: Path,
) -> None:
    plan = load_and_validate_dual_plan(_write_plan(tmp_path))
    commanders = {
        "holder": _FallbackIkExecutionCommander(),
        "inserter": _FallbackIkExecutionCommander(),
    }
    _, record = _recorded_steps()
    preplanned = _preplanned_sequence("inserter_preinsertion")
    success, status, last_completed = _execute_sequence(
        plan=plan,
        commanders=commanders,
        grippers={"holder": _Gripper(), "inserter": _Gripper()},
        config=DualRealExecutionConfig(
            execute=True,
            allow_objectless_planning=True,
            stop_after="inserter_preinsertion",
        ),
        record=record,
        preplanned_sequence=preplanned,
    )

    assert success is True
    assert status == "stopped_at_inserter_preinsertion"
    assert last_completed == "inserter_preinsertion"
    for role, commander in commanders.items():
        expected_labels = [name for target_role, name in MOTION_SEQUENCE if target_role == role]
        assert commander.joint_plans == []
        assert commander.executed == expected_labels


def test_real_preflight_aborts_instead_of_caching_after_attached_cleanup_failure() -> None:
    plan = _plan_payload()
    plan["moveit"] = {
        "attached_collision_geometry": {
            "objects": {
                "incoming": {
                    "id": "attached_incoming",
                    "link_name": "lbr_two_gripper_tcp",
                }
            }
        }
    }
    commanders = {
        "holder": _CleanupFailPreflightCommander(),
        "inserter": _CleanupFailPreflightCommander(),
    }
    _steps, record = _recorded_steps()

    try:
        _preflight_targets(
            plan=plan,
            commanders=commanders,
            frame_id="base_link",
            record=record,
        )
    except RuntimeError as exc:
        assert "synthetic world purge failure" in str(exc)
    else:
        raise AssertionError("Expected dirty-scene cleanup failure to abort preflight.")

    assert ("attach", ("attached_incoming",)) in commanders["holder"].scene_calls
    assert ("detach_failed", ("attached_incoming",)) in commanders["holder"].scene_calls


def test_ranked_real_preflight_rejects_first_pair_and_selects_second() -> None:
    first = _plan_payload()
    first["selection_score"] = 0.9
    first["candidate_rank"] = 1
    first["targets"]["inserter_pickup_pregrasp"]["position_world_m"] = [1.0, 0.0, 0.1]

    second = _plan_payload()
    second["pair_id"] = "p002_h0001_i0_0003"
    second["selection_score"] = 0.8
    second["candidate_rank"] = 2
    second["grasps"]["inserter_pickup"]["grasp_id"] = "i0_0003"

    plan = dict(first)
    plan["ranked_pair_candidates"] = [first, second]
    commanders = {
        "holder": _RankedIkCommander("holder"),
        "inserter": _RankedIkCommander("inserter"),
    }
    steps, record = _recorded_steps()
    debug_updates = []

    selected, summary, preplanned = _select_ranked_preflight_candidate(
        plan=plan,
        commanders=commanders,
        initial_robot_state=_initial_robot_state(),
        config=DualRealExecutionConfig(stop_after="inserter_preinsertion"),
        frame_id="base_link",
        record=record,
        update_debug=lambda **entry: debug_updates.append(entry),
    )

    assert selected is not None
    assert preplanned is not None
    assert selected["pair_id"] == "p002_h0001_i0_0003"
    assert summary["candidates_checked"] == 2
    assert summary["selected_rank"] == 2
    assert summary["selected_pair_id"] == "p002_h0001_i0_0003"
    assert set(summary["selected_joint_targets"]) == {name for _, name in MOTION_SEQUENCE}
    assert summary["records"][1]["roles"]["holder"]["cache_hit"] is False
    assert {entry["mode"] for entry in summary["preplanned_segments"]} == {
        "free_space",
        "cartesian_linear",
    }
    assert [(update["attempt_index"], update["status"]) for update in debug_updates] == [
        (1, "planning"),
        (1, "failed"),
        (2, "planning"),
        (2, "succeeded"),
    ]
    assert any(
        step["name"] == "candidate_preflight" and step["candidate_rank"] == 1 and step["ok"] is False for step in steps
    )


def test_ranked_preflight_does_not_cache_same_grasp_across_transitions() -> None:
    first = _plan_payload()
    first["selection_score"] = 0.9
    first["candidate_rank"] = 1
    first["transition_id"] = "tr_left"
    first["execution_candidate_id"] = "p001_h0001_i0_0002__tr_left"
    first["targets"]["inserter_preinsertion"]["position_world_m"] = [1.0, 0.0, 0.1]

    second = _plan_payload()
    second["selection_score"] = 0.8
    second["candidate_rank"] = 2
    second["transition_id"] = "tr_right"
    second["execution_candidate_id"] = "p001_h0001_i0_0002__tr_right"

    plan = dict(first)
    plan["ranked_pair_candidates"] = [first, second]
    commanders = {
        "holder": _RankedIkCommander("holder"),
        "inserter": _RankedIkCommander("inserter"),
    }
    _, record = _recorded_steps()

    selected, summary, preplanned = _select_ranked_preflight_candidate(
        plan=plan,
        commanders=commanders,
        initial_robot_state=_initial_robot_state(),
        config=DualRealExecutionConfig(stop_after="inserter_preinsertion"),
        frame_id="base_link",
        record=record,
    )

    assert selected is not None
    assert preplanned is not None
    assert selected["execution_candidate_id"].endswith("__tr_right")
    assert summary["selected_transition_id"] == "tr_right"
    assert summary["records"][1]["roles"]["holder"]["cache_hit"] is False
    assert summary["records"][1]["roles"]["inserter"]["cache_hit"] is False


def test_ranked_preflight_caches_identical_failed_connected_prefix() -> None:
    first = _plan_payload()
    first["candidate_rank"] = 1
    first["execution_candidate_id"] = f"{first['pair_id']}__tr_first"
    first["targets"]["holder_pregrasp"]["position_world_m"] = [1.0, 0.0, 0.1]

    second = json.loads(json.dumps(first))
    second["candidate_rank"] = 2
    second["execution_candidate_id"] = second["execution_candidate_id"].replace("tr_first", "tr_second")
    second["transition_id"] = "tr_second"
    second["targets"]["inserter_preinsertion"]["position_world_m"] = [0.45, 0.0, 0.1]
    second["grasps"]["inserter_preinsertion"] = {
        "grasp_id": "transition_specific_grasp",
        "jaw_width_m": 0.041,
        "position_world_m": [0.45, 0.0, 0.1],
    }

    third = _plan_payload()
    third["candidate_rank"] = 3
    third["pair_id"] = "p003_h0004_i0_0005"
    third["execution_candidate_id"] = "p003_h0004_i0_0005__tr_good"
    plan = dict(first)
    plan["ranked_pair_candidates"] = [first, second, third]
    commanders = {
        "holder": _RankedIkCommander("holder"),
        "inserter": _RankedIkCommander("inserter"),
    }
    steps, record = _recorded_steps()

    selected, summary, preplanned = _select_ranked_preflight_candidate(
        plan=plan,
        commanders=commanders,
        initial_robot_state=_initial_robot_state(),
        config=DualRealExecutionConfig(stop_after="inserter_preinsertion"),
        frame_id="base_link",
        record=record,
    )

    assert selected is not None
    assert preplanned is not None
    assert selected["pair_id"] == "p003_h0004_i0_0005"
    assert summary["cached_prefix_rejections"] == 1
    assert summary["records"][1]["roles"]["holder"]["cache_hit"] is True
    assert not any(
        step["name"] == "preflight_plan_holder_pregrasp" and step.get("candidate_rank") == 2
        for step in steps
    )


def test_ranked_preflight_stops_before_unrequested_preinsertion_targets() -> None:
    candidate = _plan_payload()
    candidate["targets"]["inserter_above_preinsertion"]["position_world_m"] = [1.0, 0.0, 0.1]
    candidate["ranked_pair_candidates"] = [dict(candidate)]
    commanders = {
        "holder": _RankedIkCommander("holder"),
        "inserter": _RankedIkCommander("inserter"),
    }
    steps, record = _recorded_steps()

    selected, summary, preplanned = _select_ranked_preflight_candidate(
        plan=candidate,
        commanders=commanders,
        initial_robot_state=_initial_robot_state(),
        config=DualRealExecutionConfig(stop_after="inserter_pickup_grasp"),
        frame_id="base_link",
        record=record,
        stop_after="inserter_pickup_grasp",
    )

    assert selected is not None
    assert preplanned is not None
    assert selected["pair_id"] == "p001_h0001_i0_0002"
    assert summary["stop_after"] == "inserter_pickup_grasp"
    assert not any(step["name"] == "preflight_plan_inserter_above_preinsertion" for step in steps)
