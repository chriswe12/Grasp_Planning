from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from grasp_planning.pipeline.dual_robot_simple_sim import (
    DEFAULT_FLOOR_Z_WORLD_M,
    NoPoseFeasibleDualTasksError,
)
from grasp_planning.ros2 import dual_real_grasp_executor
from grasp_planning.ros2.dual_real_grasp_executor import (
    MOTION_SEQUENCE,
    DualRealExecutionConfig,
    _execute_sequence,
    _make_gripper,
    _preflight_targets,
    _select_ranked_preflight_candidate,
    _work_surface_obstacle,
    execute_dual_real_plan,
    load_and_validate_dual_plan,
)
from grasp_planning.ros2.mock_gripper_client import MockGripperClient
from grasp_planning.ros2.moveit_pose_commander import PoseTarget
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

    def move_to_pose(self, target, *, label: str, execute: bool):
        self.calls.append((label, execute))
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


class _Gripper:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def open(self, *, width: float):
        self.calls.append(("open", width))
        return True, "opened"

    def close(self, *, width: float):
        self.calls.append(("close", width))
        return True, "closed"


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


class _RankedIkCommander:
    def __init__(self) -> None:
        self.calls = 0

    def compute_ik(self, target, seed_joint_positions=None):
        del seed_joint_positions
        self.calls += 1
        if target.x > 0.9:
            return None, "synthetic no IK"
        return [0.0] * 7, "ok"


class _KdlLikeCommander:
    """Models a local single-seed solver's failure mode on a distant target.

    A seedless call only converges within `near_threshold` of the current
    pose. A call seeded with the fixed `KUKA_MOVEIT_ARM_START_JOINT_VALUES`
    generic retry seed (what the existing single-shot fallback uses) is no
    better - it only converges within `near_threshold` of *that* seed's own
    pose too, which this fake does not know, so it is treated the same as no
    seed. Any other seed is assumed to come from
    `solve_cartesian_waypoint_chain`'s previous, already-close waypoint
    solution and always converges - modeling why a walked chain succeeds on
    a jump a single direct call (with or without the generic retry) cannot.
    """

    def __init__(self, *, current_pose: PoseTarget, near_threshold: float = 0.15) -> None:
        self._current_pose = current_pose
        self._near_threshold = near_threshold
        self.calls: list[tuple[float, bool]] = []

    def get_current_pose(self, *, frame_id: str) -> PoseTarget:
        del frame_id
        return self._current_pose

    def compute_ik(self, target, seed_joint_positions=None):
        if seed_joint_positions is None or tuple(seed_joint_positions) == tuple(KUKA_MOVEIT_ARM_START_JOINT_VALUES):
            ok = abs(target.x - self._current_pose.x) <= self._near_threshold
        else:
            ok = True
        self.calls.append((target.x, ok))
        return ([0.0] * 7, "ok") if ok else (None, "no ik solution")


def _recorded_steps():
    steps = []

    def record(**kwargs):
        steps.append(kwargs)

    return steps, record


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


def test_load_and_validate_dual_plan_rejects_role_swap(tmp_path: Path) -> None:
    payload = _plan_payload()
    payload["roles"]["holder"]["robot"] = "lbr_two"

    try:
        load_and_validate_dual_plan(_write_plan(tmp_path, payload))
    except ValueError as exc:
        assert "expected 'lbr_one'" in str(exc)
    else:
        raise AssertionError("Expected mismatched role mapping to fail.")


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
        candidate_rank=4,
        update_debug=lambda **entry: debug_updates.append(entry),
    )

    assert success is True
    assert status == "stopped_at_inserter_preinsertion"
    assert last_completed == "inserter_preinsertion"
    assert commanders["holder"].calls == [
        ("holder_pregrasp", True),
        ("holder_grasp", True),
    ]
    assert commanders["inserter"].calls == [
        (target_name, True) for role, target_name in MOTION_SEQUENCE if role == "inserter"
    ]
    assert grippers["holder"].calls == [
        ("open", kuka_gripper_approach_width(plan["grasps"]["holder"]["jaw_width_m"])),
        ("close", plan["grasps"]["holder"]["jaw_width_m"]),
    ]
    assert grippers["inserter"].calls == [
        ("open", kuka_gripper_approach_width(plan["grasps"]["inserter_pickup"]["jaw_width_m"])),
        ("close", plan["grasps"]["inserter_pickup"]["jaw_width_m"]),
    ]
    assert steps[-1]["name"] == "inserter_preinsertion"
    assert any(update["phase"] == "holder_pregrasp" and update["status"] == "planning" for update in debug_updates)
    assert debug_updates[-1]["phase"] == "inserter_preinsertion"
    assert debug_updates[-1]["status"] == "succeeded"
    assert all(update["attempt_index"] == 4 for update in debug_updates)


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
    )

    assert success is True
    assert status == "stopped_at_holder_pregrasp"
    assert last_completed == "holder_pregrasp"
    assert commanders["inserter"].calls == []
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
    )

    assert success is True
    assert status == "stopped_at_inserter_pickup_pregrasp"
    assert commanders["holder"].scene_calls == [
        ("apply", ("base_aabb", "incoming_aabb")),
        ("remove", ("base_aabb", "incoming_aabb")),
        ("apply", ("incoming_aabb",)),
        ("remove", ("incoming_aabb",)),
    ]


def test_dual_gripper_launch_has_stable_namespaces_and_usb_ids() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/gripper_computer/dual_grippers.launch.py").read_text(
        encoding="utf-8"
    )

    assert "namespace=role" in source
    assert '_gripper_node(role="lbr_one"' in source
    assert '_gripper_node(role="lbr_two"' in source
    assert "usb-1a86_USB_Single_Serial_5B3D047592-if00" in source
    assert "usb-1a86_USB_Single_Serial_5B3D044069-if00" in source


def test_dual_startup_scripts_force_one_shared_ros_domain() -> None:
    root = Path(__file__).resolve().parents[1]
    gripper_start = (root / "scripts/gripper_computer/start_dual_grippers.sh").read_text(encoding="utf-8")
    dual_run = (root / "run_simple_dual_robot.sh").read_text(encoding="utf-8")
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
    source = (Path(__file__).resolve().parents[1] / "run_simple_dual_robot.sh").read_text(encoding="utf-8")

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
    assert all("alternate IK succeeded" in step["message"] for step in target_steps)
    for commander in commanders.values():
        assert len(commander.seeds) >= 2
        assert commander.seeds[0] is None
        assert commander.seeds[1] is not None


def test_preflight_targets_cartesian_waypoints_strategy_reaches_a_target_direct_cannot() -> None:
    far_target_name = "inserter_preinsertion"
    targets = {
        target_name: {
            "position_world_m": [1.0 if target_name == far_target_name else 0.0, 0.0, 0.1],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        }
        for _, target_name in MOTION_SEQUENCE
    }
    plan = {
        "roles": {
            "holder": {"robot": "lbr_one"},
            "inserter": {"robot": "lbr_two"},
        },
        "grasps": {
            "holder": {"grasp_id": "h0001", "jaw_width_m": 0.040},
            "inserter_pickup": {"grasp_id": "i0_0002", "jaw_width_m": 0.043},
        },
        "targets": targets,
    }
    current_pose = PoseTarget.from_quaternion(
        x=0.0, y=0.0, z=0.1, quaternion_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id="base_link"
    )

    direct_commanders = {
        "holder": _KdlLikeCommander(current_pose=current_pose),
        "inserter": _KdlLikeCommander(current_pose=current_pose),
    }
    direct_steps, direct_record = _recorded_steps()
    direct_ok = _preflight_targets(
        plan=plan,
        commanders=direct_commanders,
        frame_id="base_link",
        record=direct_record,
        ik_strategy="direct",
    )
    assert direct_ok is False
    direct_far_step = next(step for step in direct_steps if step["name"] == f"preflight_{far_target_name}")
    assert direct_far_step["ok"] is False

    waypoint_commanders = {
        "holder": _KdlLikeCommander(current_pose=current_pose),
        "inserter": _KdlLikeCommander(current_pose=current_pose),
    }
    waypoint_steps, waypoint_record = _recorded_steps()
    waypoint_ok = _preflight_targets(
        plan=plan,
        commanders=waypoint_commanders,
        frame_id="base_link",
        record=waypoint_record,
        ik_strategy="cartesian_waypoints",
        cartesian_waypoint_count=10,
    )
    assert waypoint_ok is True
    waypoint_far_step = next(step for step in waypoint_steps if step["name"] == f"preflight_{far_target_name}")
    assert waypoint_far_step["ok"] is True
    assert "cartesian_waypoint_chain succeeded" in waypoint_far_step["message"]
    # The chain reached x=1.0 through several small steps, not one direct
    # jump: more than the 1 call a "direct" attempt (plus its 1 retry) would
    # have made for this single target.
    assert len(waypoint_commanders["inserter"].calls) > 2


def test_fallback_preflight_joint_targets_are_reused_for_execution(
    tmp_path: Path,
) -> None:
    plan = load_and_validate_dual_plan(_write_plan(tmp_path))
    commanders = {
        "holder": _FallbackIkExecutionCommander(),
        "inserter": _FallbackIkExecutionCommander(),
    }
    resolved_joint_targets: dict[str, tuple[float, ...]] = {}
    _, record = _recorded_steps()

    assert _preflight_targets(
        plan=plan,
        commanders=commanders,
        frame_id="base_link",
        record=record,
        resolved_joint_targets=resolved_joint_targets,
    )
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
        preferred_joint_targets=resolved_joint_targets,
    )

    assert success is True
    assert status == "stopped_at_inserter_preinsertion"
    assert last_completed == "inserter_preinsertion"
    assert set(resolved_joint_targets) == {name for _, name in MOTION_SEQUENCE}
    for role, commander in commanders.items():
        expected_labels = [name for target_role, name in MOTION_SEQUENCE if target_role == role]
        assert [label for label, _ in commander.joint_plans] == expected_labels
        assert commander.executed == expected_labels
        assert all(joints == KUKA_MOVEIT_ARM_START_JOINT_VALUES for _, joints in commander.joint_plans)


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
        "holder": _RankedIkCommander(),
        "inserter": _RankedIkCommander(),
    }
    steps, record = _recorded_steps()
    debug_updates = []

    selected, summary = _select_ranked_preflight_candidate(
        plan=plan,
        commanders=commanders,
        frame_id="base_link",
        record=record,
        update_debug=lambda **entry: debug_updates.append(entry),
    )

    assert selected is not None
    assert selected["pair_id"] == "p002_h0001_i0_0003"
    assert summary["candidates_checked"] == 2
    assert summary["selected_rank"] == 2
    assert summary["selected_pair_id"] == "p002_h0001_i0_0003"
    assert set(summary["selected_joint_targets"]) == {name for _, name in MOTION_SEQUENCE}
    assert commanders["holder"].calls == 2
    assert commanders["inserter"].calls == 7
    assert summary["records"][1]["roles"]["holder"]["cache_hit"] is True
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
        "holder": _RankedIkCommander(),
        "inserter": _RankedIkCommander(),
    }
    _, record = _recorded_steps()

    selected, summary = _select_ranked_preflight_candidate(
        plan=plan,
        commanders=commanders,
        frame_id="base_link",
        record=record,
    )

    assert selected is not None
    assert selected["execution_candidate_id"].endswith("__tr_right")
    assert summary["selected_transition_id"] == "tr_right"
    assert commanders["holder"].calls == 2
    # Five targets for each transition, plus the seeded retry for the first
    # transition's failing pre-insertion target.
    assert commanders["inserter"].calls == 11
    assert summary["records"][1]["roles"]["holder"]["cache_hit"] is True
    assert summary["records"][1]["roles"]["inserter"]["cache_hit"] is False


def test_ranked_preflight_stops_before_unrequested_preinsertion_targets() -> None:
    candidate = _plan_payload()
    candidate["targets"]["inserter_above_preinsertion"]["position_world_m"] = [1.0, 0.0, 0.1]
    candidate["ranked_pair_candidates"] = [dict(candidate)]
    commanders = {
        "holder": _RankedIkCommander(),
        "inserter": _RankedIkCommander(),
    }
    steps, record = _recorded_steps()

    selected, summary = _select_ranked_preflight_candidate(
        plan=candidate,
        commanders=commanders,
        frame_id="base_link",
        record=record,
        stop_after="inserter_pickup_grasp",
    )

    assert selected is not None
    assert selected["pair_id"] == "p001_h0001_i0_0002"
    assert summary["stop_after"] == "inserter_pickup_grasp"
    assert commanders["holder"].calls == 2
    assert commanders["inserter"].calls == 2
    assert not any(step["name"] == "preflight_inserter_above_preinsertion" for step in steps)
