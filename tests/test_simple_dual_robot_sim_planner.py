from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.plan_simple_dual_robot_sim import (
    IK_PREFLIGHT_TARGETS,
    _complete_dual_arm_start_state,
    _configure_role_assignment,
    _exact_ik_seed_candidates,
    _ik_preflight_pair,
    _ik_search_targets,
    _inserter_diverse_task_prefix,
    _new_ik_preflight_state,
    _plan_and_execute,
    _pregrasp_aabb_obstacles_for_target,
    _rank_tasks_by_inserter_joint_path,
    _reset_active_roles,
)


def test_auto_role_assignment_uses_arm_nearest_pickup_side() -> None:
    holder, inserter, holder_base, inserter_base = _configure_role_assignment(
        requested_inserter_arm="auto",
        pickup_y=-0.26,
        assembly_y=0.0,
    )

    assert (holder, inserter) == ("lbr_two", "lbr_one")
    assert holder_base.position_world_m == (0.0, 0.42, 0.0)
    assert inserter_base.position_world_m == (0.0, -0.42, 0.0)

    holder, inserter, _, _ = _configure_role_assignment(
        requested_inserter_arm="auto",
        pickup_y=0.26,
        assembly_y=0.0,
    )
    assert (holder, inserter) == ("lbr_one", "lbr_two")


class _FakeCommander:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.calls = 0

    def compute_ik(
        self,
        _target,
        *,
        seed_joint_positions=None,
        seed_robot_state=None,
        avoid_collisions=None,
    ):
        del seed_joint_positions, seed_robot_state
        assert avoid_collisions is False
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return None, "synthetic no IK"
        return [0.0] * 7, "ok"

    def check_state_validity(self, robot_state, *, group_name=""):
        assert robot_state
        assert group_name == ""
        return {"valid": True, "contacts": []}, "valid"


class _CollisionDiagnosticCommander:
    def __init__(self) -> None:
        self.avoid_collisions: list[bool | None] = []
        self.validity_states: list[dict[str, float]] = []

    def compute_ik(
        self,
        _target,
        *,
        seed_joint_positions=None,
        seed_robot_state=None,
        avoid_collisions=None,
    ):
        assert seed_joint_positions is None
        assert seed_robot_state is not None
        self.avoid_collisions.append(avoid_collisions)
        return [0.0] * 7, "ok"

    def check_state_validity(self, robot_state, *, group_name=""):
        assert group_name == ""
        self.validity_states.append(dict(robot_state))
        return {
            "valid": False,
            "contacts": [
                {
                    "body_1": "lbr_one_left_finger_link",
                    "body_type_1": 0,
                    "body_2": "dual_sim_work_surface",
                    "body_type_2": 1,
                    "depth_m": 0.004,
                    "position_world_m": [0.5, 0.0, -0.03],
                    "normal_world": [0.0, 0.0, 1.0],
                }
            ],
        }, "invalid"


def test_collision_diagnostic_preflight_records_exact_contact_pair() -> None:
    holder = _CollisionDiagnosticCommander()
    state = _new_ik_preflight_state(
        pair_task_count=1,
        ik_candidate_count=1,
        ik_beam_width=1,
        collision_diagnostics=True,
    )

    ok, failure, joint_targets = _ik_preflight_pair(
        _task("pair_1", "holder_1", "inserter_1", 0.9),
        commanders={"holder": holder, "inserter": _FakeCommander()},
        feasible_cache={"holder": {}, "inserter": {}},
        state=state,
        rank=1,
        roles=("holder",),
        ik_candidate_count=1,
        ik_beam_width=1,
        collision_diagnostics=True,
    )

    assert not ok
    assert joint_targets == {}
    assert "dual_sim_work_surface <-> lbr_one_left_finger_link" in failure
    assert holder.avoid_collisions == [False]
    assert len(holder.validity_states) == 1
    diagnostics = state["collision_diagnostics"]
    assert diagnostics["contact_class_counts"] == {"finger_floor": 1}
    assert diagnostics["contact_pair_counts"] == {
        "dual_sim_work_surface <-> lbr_one_left_finger_link": 1,
    }
    assert diagnostics["invalid_states"] == 1


class _ResetCommander:
    def __init__(self, role: str, shared: dict[str, object]) -> None:
        self.role = role
        self.shared = shared

    def plan_to_joint_positions(self, _positions, *, label: str):
        calls = self.shared.setdefault("calls", [])
        assert isinstance(calls, list)
        calls.append(f"plan:{self.role}")
        if self.role == "holder" and not self.shared.get("inserter_cleared"):
            return None, "synthetic holder/inserter collision"
        return SimpleNamespace(), f"{label}: planned"

    def execute_trajectory(self, _trajectory, *, label: str):
        calls = self.shared.setdefault("calls", [])
        assert isinstance(calls, list)
        calls.append(f"execute:{self.role}")
        if self.role == "inserter":
            self.shared["inserter_cleared"] = True
        return True, f"{label}: execution complete"


def _task(
    pair_id: str,
    holder_id: str,
    inserter_id: str,
    score: float,
    *,
    preinsertion_x: float = 0.5,
    transition_id: str = "tr_identity",
    corridor_y: float = -1.0,
    transition_crosses: bool = False,
    holder_x: float = 0.5,
):
    pose = {
        "position_world_m": [0.5, 0.0, 0.2],
        "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
    }
    targets = {
        target_name: dict(pose) for target_names in IK_PREFLIGHT_TARGETS.values() for target_name in target_names
    }
    for target_name in IK_PREFLIGHT_TARGETS["holder"]:
        targets[target_name] = {
            **pose,
            "position_world_m": [holder_x, 0.0, 0.2],
        }
    targets["inserter_preinsertion"] = {
        **pose,
        "position_world_m": [preinsertion_x, 0.0, 0.2],
    }
    return SimpleNamespace(
        pair_id=pair_id,
        transition_id=transition_id,
        execution_candidate_id=f"{pair_id}__{transition_id}",
        transition_symmetry={
            "pre_to_final_translation_assembly_m": [0.0, corridor_y, 0.0],
        },
        selection_score=score,
        layout_proxy_components={
            "transition_segments_cross_xy": transition_crosses,
        },
        holder_candidate=SimpleNamespace(grasp_id=holder_id),
        inserter_candidate=SimpleNamespace(grasp_id=inserter_id),
        to_payload=lambda: {"targets": targets},
    )


def test_lazy_preflight_stops_on_failure_and_reuses_grasp_cache() -> None:
    holder = _FakeCommander(fail_first=True)
    inserter = _FakeCommander()
    commanders = {"holder": holder, "inserter": inserter}
    cache = {"holder": {}, "inserter": {}}
    state = _new_ik_preflight_state(pair_task_count=3)

    first_ok, first_failure, first_targets = _ik_preflight_pair(
        _task("pair_1", "holder_bad", "inserter_1", 0.9),
        commanders=commanders,
        feasible_cache=cache,
        state=state,
        rank=1,
    )
    assert not first_ok
    assert first_targets == {}
    assert "holder_bad failed holder_pregrasp" in first_failure
    assert "1 seed evaluation(s), 1 IK request(s)" in first_failure
    assert holder.calls == 1
    assert inserter.calls == 0

    second_ok, second_failure, second_targets = _ik_preflight_pair(
        _task("pair_2", "holder_bad", "inserter_2", 0.8),
        commanders=commanders,
        feasible_cache=cache,
        state=state,
        rank=2,
    )
    assert not second_ok
    assert second_targets == {}
    assert "cached IK failure" in second_failure
    assert holder.calls == 1
    assert inserter.calls == 0

    third_ok, third_failure, third_targets = _ik_preflight_pair(
        _task("pair_3", "holder_good", "inserter_3", 0.7),
        commanders=commanders,
        feasible_cache=cache,
        state=state,
        rank=3,
    )
    assert third_ok
    assert third_failure == ""
    assert set(third_targets) == {
        "holder_pregrasp",
        "holder_grasp",
        "inserter_pickup_pregrasp",
        "inserter_pickup_grasp",
        "inserter_pickup_lift",
        "inserter_above_preinsertion",
        "inserter_preinsertion",
    }
    assert holder.calls == 3
    assert inserter.calls == 5

    assert state["mode"] == "lazy_cached_kinematics_complete_state_multi_seed_beam"
    assert state["pair_tasks_checked"] == 3
    assert state["pair_tasks_after"] == 1
    assert state["holder_grasps_checked"] == 2
    assert state["holder_grasps_feasible"] == 1
    assert state["inserter_grasps_checked"] == 1
    assert state["inserter_grasps_feasible"] == 1
    assert [record["pair_id"] for record in state["pair_records"]] == ["pair_1", "pair_2", "pair_3"]
    assert state["pair_records"][1]["roles"]["holder"]["cache_hit"]


def test_exact_ik_seeds_include_bounded_a7_and_shoulder_branches() -> None:
    seeds = _exact_ik_seed_candidates(
        (0.0,) * 7,
        preferred_joint_positions=None,
        candidate_count=8,
        perturbation_rad=0.6,
    )

    assert len(seeds) == 8
    assert seeds[0] == pytest.approx((0.0,) * 7)
    assert any(seed[-1] == pytest.approx(3.0) for seed in seeds)
    assert any(seed[-1] == pytest.approx(-3.0) for seed in seeds)
    assert any(seed[0] == pytest.approx(0.6) for seed in seeds)
    assert any(seed[0] == pytest.approx(-0.6) for seed in seeds)
    assert any(seed[2] == pytest.approx(0.6) for seed in seeds)
    assert any(seed[2] == pytest.approx(-0.6) for seed in seeds)
    assert all(-3.05 <= seed[-1] <= 3.05 for seed in seeds)


def test_runtime_queue_visits_unique_inserter_pickups_before_pair_repeats() -> None:
    tasks = [
        _task("pair_a1", "holder_1", "inserter_a", 1.00),
        _task("pair_a2", "holder_2", "inserter_a", 0.99),
        _task("pair_a3", "holder_3", "inserter_a", 0.98),
        _task("pair_b1", "holder_1", "inserter_b", 0.80),
        _task("pair_c1", "holder_1", "inserter_c", 0.70),
    ]

    selected = _inserter_diverse_task_prefix(tasks, limit=4)

    assert [task.inserter_candidate.grasp_id for task in selected] == [
        "inserter_a",
        "inserter_b",
        "inserter_c",
        "inserter_a",
    ]
    assert [task.pair_id for task in selected] == ["pair_a1", "pair_b1", "pair_c1", "pair_a2"]


def test_pickup_approach_ik_targets_interpolate_and_only_publish_endpoint() -> None:
    targets = _task("pair", "holder", "inserter", 1.0).to_payload()["targets"]
    targets["inserter_pickup_pregrasp"]["position_world_m"] = [0.4, -0.2, 0.12]
    targets["inserter_pickup_grasp"]["position_world_m"] = [0.4, -0.2, 0.02]

    search_targets = _ik_search_targets(
        targets=targets,
        target_names=("inserter_pickup_pregrasp", "inserter_pickup_grasp", "inserter_pickup_lift"),
        pickup_approach_ik_steps=5,
    )

    assert len(search_targets) == 7
    approach = search_targets[1:6]
    assert [target.pose.z for target in approach] == pytest.approx([0.10, 0.08, 0.06, 0.04, 0.02])
    assert [target.result_target_name for target in approach] == [None, None, None, None, "inserter_pickup_grasp"]
    assert all(target.pose.frame_id == "base_link" for target in approach)


class _CoordinatedBranchCommander:
    def __init__(self, role: str) -> None:
        self.role = role
        self.seed_states: list[dict[str, float]] = []

    def compute_ik(
        self,
        _target,
        *,
        seed_joint_positions=None,
        seed_robot_state=None,
        avoid_collisions=None,
    ):
        assert seed_joint_positions is None
        assert seed_robot_state is not None
        assert avoid_collisions is False
        state = dict(seed_robot_state)
        self.seed_states.append(state)
        prefix = "lbr_one" if self.role == "holder" else "lbr_two"
        return [float(state[f"{prefix}_A{index}"]) for index in range(1, 8)], "ok"

    def check_state_validity(self, robot_state, *, group_name=""):
        assert group_name == ""
        state = dict(robot_state)
        if self.role == "inserter" and abs(float(state["lbr_one_A7"])) < 2.5:
            return {
                "valid": False,
                "contacts": [{"body_1": "lbr_one_link_5", "body_2": "lbr_two_link_5"}],
            }, "synthetic inactive-holder collision"
        return {"valid": True, "contacts": []}, "valid"


def test_multi_seed_preflight_keeps_holder_branch_that_unlocks_inserter() -> None:
    holder = _CoordinatedBranchCommander("holder")
    inserter = _CoordinatedBranchCommander("inserter")
    state = _new_ik_preflight_state(
        pair_task_count=1,
        ik_candidate_count=3,
        ik_beam_width=3,
    )

    ok, failure, joint_targets = _ik_preflight_pair(
        _task("pair_1", "holder_1", "inserter_1", 0.9),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache={"holder": {}, "inserter": {}},
        state=state,
        rank=1,
        ik_candidate_count=3,
        ik_beam_width=3,
    )

    assert ok
    assert failure == ""
    assert abs(joint_targets["holder_grasp"][-1]) >= 2.5
    holder_a7_states = {round(float(seed["lbr_one_A7"]), 3) for seed in inserter.seed_states}
    assert 0.0 in holder_a7_states
    assert any(abs(value) >= 2.5 for value in holder_a7_states)
    assert state["ik_seed_calls"] > len(IK_PREFLIGHT_TARGETS["holder"])
    assert state["pair_records"][0]["roles"]["holder"]["output_branches"] == 3


def test_holder_only_preflight_does_not_query_inserter() -> None:
    holder = _FakeCommander()
    inserter = _FakeCommander(fail_first=True)
    state = _new_ik_preflight_state(pair_task_count=1)

    ok, failure, joint_targets = _ik_preflight_pair(
        _task("pair_1", "holder_1", "inserter_1", 0.9),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache={"holder": {}, "inserter": {}},
        state=state,
        rank=1,
        roles=("holder",),
    )

    assert ok
    assert failure == ""
    assert set(joint_targets) == {"holder_pregrasp", "holder_grasp"}
    assert holder.calls == 2
    assert inserter.calls == 0
    assert state["holder_grasps_feasible"] == 1
    assert state["inserter_grasps_checked"] == 0
    assert set(state["pair_records"][0]["roles"]) == {"holder"}


def test_same_grasp_with_different_transition_targets_is_not_cached() -> None:
    holder = _FakeCommander()
    inserter = _FakeCommander()
    cache = {"holder": {}, "inserter": {}}
    state = _new_ik_preflight_state(pair_task_count=2)

    first_ok, _, _first_targets = _ik_preflight_pair(
        _task(
            "pair_1",
            "holder_1",
            "inserter_1",
            0.9,
            preinsertion_x=0.4,
            transition_id="tr_left",
        ),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache=cache,
        state=state,
        rank=1,
    )
    second_ok, _, _second_targets = _ik_preflight_pair(
        _task(
            "pair_1",
            "holder_1",
            "inserter_1",
            0.8,
            preinsertion_x=0.8,
            transition_id="tr_right",
        ),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache=cache,
        state=state,
        rank=2,
    )

    assert first_ok and second_ok
    assert holder.calls == 2
    assert inserter.calls == 10
    assert state["pair_records"][0]["execution_candidate_id"].endswith("tr_left")
    assert not state["pair_records"][1]["roles"]["inserter"]["cache_hit"]


class _StateTrackingCommander:
    def __init__(self, solution_value: float) -> None:
        self.solution_value = float(solution_value)
        self.seed_states: list[dict[str, float]] = []

    def compute_ik(
        self,
        target,
        *,
        seed_joint_positions=None,
        seed_robot_state=None,
        avoid_collisions=None,
    ):
        assert seed_joint_positions is None
        assert seed_robot_state is not None
        assert avoid_collisions is False
        self.seed_states.append(dict(seed_robot_state))
        value = self.solution_value if self.solution_value != 999.0 else float(target.x)
        return [value] * 7, "ok"

    def check_state_validity(self, robot_state, *, group_name=""):
        assert robot_state
        assert group_name == ""
        return {"valid": True, "contacts": []}, "valid"


class _CachedDiagnosticCommander(_StateTrackingCommander):
    def __init__(self, solution_value: float) -> None:
        super().__init__(solution_value)
        self.validity_states: list[dict[str, float]] = []

    def compute_ik(
        self,
        target,
        *,
        seed_joint_positions=None,
        seed_robot_state=None,
        avoid_collisions=None,
    ):
        assert avoid_collisions is False
        return super().compute_ik(
            target,
            seed_joint_positions=seed_joint_positions,
            seed_robot_state=seed_robot_state,
            avoid_collisions=False,
        )

    def check_state_validity(self, robot_state, *, group_name=""):
        assert group_name == ""
        self.validity_states.append(dict(robot_state))
        return {"valid": True, "contacts": []}, "valid"


def test_collision_diagnostic_reuses_active_arm_ik_but_revalidates_complete_state() -> None:
    holder = _CachedDiagnosticCommander(999.0)
    inserter = _CachedDiagnosticCommander(-0.4)
    role_cache = {"holder": {}, "inserter": {}}
    kinematic_cache = {}
    state = _new_ik_preflight_state(
        pair_task_count=2,
        ik_candidate_count=1,
        ik_beam_width=1,
        pickup_approach_ik_steps=1,
        collision_diagnostics=True,
    )

    first_ok, _, _ = _ik_preflight_pair(
        _task("pair_1", "holder_1", "same_inserter", 0.9, holder_x=0.4),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache=role_cache,
        kinematic_cache=kinematic_cache,
        state=state,
        rank=1,
        ik_candidate_count=1,
        ik_beam_width=1,
        pickup_approach_ik_steps=1,
        collision_diagnostics=True,
    )
    first_inserter_validity_count = len(inserter.validity_states)
    first_inserter_ik_count = len(inserter.seed_states)
    second_ok, _, _ = _ik_preflight_pair(
        _task("pair_2", "holder_2", "same_inserter", 0.8, holder_x=0.7),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache=role_cache,
        kinematic_cache=kinematic_cache,
        state=state,
        rank=2,
        ik_candidate_count=1,
        ik_beam_width=1,
        pickup_approach_ik_steps=1,
        collision_diagnostics=True,
    )

    assert first_ok and second_ok
    # Active-arm targets are solved only for the first holder state. This
    # synthetic task intentionally shares poses across several labels, so it
    # needs fewer calls than there are sequence targets.
    assert first_inserter_ik_count > 0
    assert len(inserter.seed_states) == first_inserter_ik_count
    # Every cached solution is nevertheless checked again with holder_2 at 0.7.
    assert len(inserter.validity_states) > first_inserter_validity_count
    assert all(
        complete_state["lbr_one_A1"] == pytest.approx(0.7)
        for complete_state in inserter.validity_states[first_inserter_validity_count:]
    )
    assert state["ik_kinematic_cache_hits"] >= len(IK_PREFLIGHT_TARGETS["inserter"])
    assert state["collision_diagnostics"]["kinematic_cache_hits"] == state["ik_kinematic_cache_hits"]


def test_normal_preflight_reuses_active_arm_ik_but_revalidates_complete_state() -> None:
    holder = _CachedDiagnosticCommander(999.0)
    inserter = _CachedDiagnosticCommander(-0.4)
    role_cache = {"holder": {}, "inserter": {}}
    kinematic_cache = {}
    state = _new_ik_preflight_state(pair_task_count=2)

    first_ok, _, _ = _ik_preflight_pair(
        _task("pair_1", "holder_1", "same_inserter", 0.9, holder_x=0.4),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache=role_cache,
        kinematic_cache=kinematic_cache,
        state=state,
        rank=1,
    )
    first_validity_count = len(inserter.validity_states)
    first_ik_count = len(inserter.seed_states)
    second_ok, _, _ = _ik_preflight_pair(
        _task("pair_2", "holder_2", "same_inserter", 0.8, holder_x=0.7),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache=role_cache,
        kinematic_cache=kinematic_cache,
        state=state,
        rank=2,
    )

    assert first_ok and second_ok
    assert len(inserter.seed_states) == first_ik_count
    assert len(inserter.validity_states) > first_validity_count
    assert state["ik_kinematic_cache_hits"] > 0
    assert (
        state["ik_state_validity_requests"] + state["post_grasp_state_validity_requests"]
        == len(holder.validity_states) + len(inserter.validity_states)
    )
    assert "collision_diagnostics" not in state


def test_preflight_carries_holder_solution_into_inserter_and_reuses_joint_targets() -> None:
    holder = _StateTrackingCommander(0.25)
    inserter = _StateTrackingCommander(-0.4)
    initial_state = _complete_dual_arm_start_state()

    ok, failure, joint_targets = _ik_preflight_pair(
        _task("pair_1", "holder_1", "inserter_1", 0.9),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache={"holder": {}, "inserter": {}},
        state=_new_ik_preflight_state(pair_task_count=1),
        rank=1,
        initial_robot_state=initial_state,
    )

    holder_joint_names = tuple(f"lbr_one_A{index}" for index in range(1, 8))
    inserter_joint_names = tuple(f"lbr_two_A{index}" for index in range(1, 8))
    assert ok
    assert failure == ""
    assert all(seed[name] == pytest.approx(0.25) for seed in inserter.seed_states for name in holder_joint_names)
    assert all(name in seed for seed in inserter.seed_states for name in (*holder_joint_names, *inserter_joint_names))
    assert joint_targets["holder_grasp"] == pytest.approx((0.25,) * 7)
    assert joint_targets["inserter_preinsertion"] == pytest.approx((-0.4,) * 7)


def test_inserter_preflight_cache_includes_solved_holder_state() -> None:
    holder = _StateTrackingCommander(999.0)
    inserter = _StateTrackingCommander(-0.4)
    cache = {"holder": {}, "inserter": {}}
    state = _new_ik_preflight_state(pair_task_count=2)

    first_ok, _, _ = _ik_preflight_pair(
        _task("pair_1", "holder_1", "same_inserter", 0.9, holder_x=0.4),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache=cache,
        state=state,
        rank=1,
    )
    second_ok, _, _ = _ik_preflight_pair(
        _task("pair_2", "holder_2", "same_inserter", 0.8, holder_x=0.7),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache=cache,
        state=state,
        rank=2,
    )

    assert first_ok and second_ok
    assert len(inserter.seed_states) == 10
    assert state["pair_records"][1]["roles"]["inserter"]["cache_hit"] is False


class _ValidatedTargetExecutionCommander:
    def __init__(self) -> None:
        self.joint_targets: list[tuple[float, ...]] = []

    def plan_to_joint_positions(self, joint_positions, *, label: str):
        del label
        values = tuple(float(value) for value in joint_positions)
        self.joint_targets.append(values)
        trajectory = SimpleNamespace(
            joint_trajectory=SimpleNamespace(
                joint_names=[f"joint_{index}" for index in range(7)],
                points=[SimpleNamespace(positions=list(values))],
            )
        )
        return trajectory, "joint target planned"

    def plan_to_pose(self, *_args, **_kwargs):
        raise AssertionError("A validated preflight target must not recompute pose IK.")

    def execute_trajectory(self, _trajectory, *, label: str):
        return True, f"{label}: execution complete"


def test_plan_and_execute_reuses_validated_preflight_joint_target() -> None:
    commander = _ValidatedTargetExecutionCommander()
    expected = tuple(0.1 * index for index in range(7))

    trajectory, message = _plan_and_execute(
        commander,
        target=SimpleNamespace(),
        label="holder_grasp",
        expected_joint_names=tuple(f"joint_{index}" for index in range(7)),
        preferred_joint_positions=expected,
    )

    assert trajectory == {
        "joint_names": [f"joint_{index}" for index in range(7)],
        "waypoints": [list(expected)],
    }
    assert message == "holder_grasp: execution complete"
    assert commander.joint_targets == [expected]


def test_joint_space_ranking_prefers_cheaper_transition_and_supplies_a7_seeds(
    monkeypatch,
) -> None:
    calls = []

    def fake_plan(
        _commander,
        *,
        targets,
        labels,
        start_joint_positions,
        joint_names,
        config,
        label_prefix,
    ):
        calls.append((labels, config, label_prefix))
        cost = 0.5
        if labels == ("inserter_above_preinsertion", "inserter_preinsertion"):
            cost = 1.0 if "tr_right" in label_prefix else 5.0
        terminal = tuple(float(value) for value in start_joint_positions)
        if labels == (
            "inserter_pickup_pregrasp",
            "inserter_pickup_grasp",
            "inserter_pickup_lift",
        ):
            terminal = (*terminal[:-1], 0.4)
        trajectories = {label: (tuple(float(index) * 0.01 for index in range(7)),) for label in labels}
        return SimpleNamespace(
            trajectories=trajectories,
            joint_path_cost=cost,
            terminal_joint_positions=terminal,
            diagnostics=(),
        )

    monkeypatch.setattr(
        "scripts.plan_simple_dual_robot_sim.plan_pose_sequence_multi_ik",
        fake_plan,
    )
    wrong = _task(
        "pair_1",
        "holder_1",
        "inserter_1",
        0.9,
        transition_id="tr_wrong",
    )
    right = _task(
        "pair_1",
        "holder_1",
        "inserter_1",
        0.8,
        transition_id="tr_right",
    )

    ranked, diagnostics, preferred = _rank_tasks_by_inserter_joint_path(
        [wrong, right],
        commander=object(),
        candidate_limit=2,
        ik_candidate_count=4,
        beam_width=1,
    )

    assert [task.transition_id for task in ranked] == ["tr_right", "tr_wrong"]
    assert diagnostics["candidate_count_planned"] == 2
    assert preferred[right.execution_candidate_id]["inserter_preinsertion"][-1] == pytest.approx(0.06)
    transition_configs = [
        config
        for labels, config, _prefix in calls
        if labels == ("inserter_above_preinsertion", "inserter_preinsertion")
    ]
    assert transition_configs
    assert all(config.seed_offsets_rad[0][-1] == pytest.approx(3.141592653589793) for config in transition_configs)
    assert all(config.seed_offsets_rad[1][-1] == pytest.approx(-3.141592653589793) for config in transition_configs)
    assert all(config.seed_offsets_rad[2][-1] == pytest.approx(2.6) for config in transition_configs)
    assert all(config.seed_offsets_rad[3][-1] == pytest.approx(-3.4) for config in transition_configs)
    assert all(config.continuous_joints == (False,) * 7 for config in transition_configs)


def test_joint_space_ranking_pool_preserves_insertion_corridor_diversity(
    monkeypatch,
) -> None:
    planned_execution_ids = []

    def fake_plan(
        _commander,
        *,
        targets,
        labels,
        start_joint_positions,
        joint_names,
        config,
        label_prefix,
    ):
        del targets, joint_names, config
        if labels == ("inserter_above_preinsertion", "inserter_preinsertion"):
            planned_execution_ids.append(label_prefix)
        trajectories = {label: ((0.0,) * 7,) for label in labels}
        return SimpleNamespace(
            trajectories=trajectories,
            joint_path_cost=1.0,
            terminal_joint_positions=tuple(float(value) for value in start_joint_positions),
            diagnostics=(),
        )

    monkeypatch.setattr(
        "scripts.plan_simple_dual_robot_sim.plan_pose_sequence_multi_ik",
        fake_plan,
    )
    identity_tasks = [
        _task(
            f"identity_{index}",
            f"holder_{index}",
            f"inserter_{index}",
            1.0 - index * 0.01,
            transition_id="tr_identity",
            corridor_y=-1.0,
        )
        for index in range(5)
    ]
    symmetric = _task(
        "symmetric",
        "holder_symmetric",
        "inserter_symmetric",
        0.5,
        transition_id="tr_symmetric",
        corridor_y=1.0,
    )

    _ranked, diagnostics, _preferred = _rank_tasks_by_inserter_joint_path(
        [*identity_tasks, symmetric],
        commander=object(),
        candidate_limit=2,
        ik_candidate_count=2,
        beam_width=1,
    )

    assert diagnostics["candidate_count_checked"] == 2
    assert len(diagnostics["corridors_checked"]) == 2
    assert any("symmetric__tr_symmetric" in label for label in planned_execution_ids)


def test_joint_space_ranking_keeps_planned_noncrossing_transition_first(
    monkeypatch,
) -> None:
    def fake_plan(
        _commander,
        *,
        targets,
        labels,
        start_joint_positions,
        joint_names,
        config,
        label_prefix,
    ):
        del targets, joint_names, config
        transition_cost = 0.1 if "crossed" in label_prefix else 2.0
        trajectories = {label: ((0.0,) * 7,) for label in labels}
        return SimpleNamespace(
            trajectories=trajectories,
            joint_path_cost=transition_cost,
            terminal_joint_positions=tuple(float(value) for value in start_joint_positions),
            diagnostics=(),
        )

    monkeypatch.setattr(
        "scripts.plan_simple_dual_robot_sim.plan_pose_sequence_multi_ik",
        fake_plan,
    )
    crossed = _task(
        "crossed",
        "holder_crossed",
        "inserter_crossed",
        0.9,
        transition_id="tr_crossed",
        transition_crosses=True,
    )
    clear = _task(
        "clear",
        "holder_clear",
        "inserter_clear",
        0.8,
        transition_id="tr_clear",
        transition_crosses=False,
    )

    ranked, diagnostics, _preferred = _rank_tasks_by_inserter_joint_path(
        [crossed, clear],
        commander=object(),
        candidate_limit=2,
        ik_candidate_count=2,
        beam_width=1,
    )

    assert [task.transition_id for task in ranked] == ["tr_clear", "tr_crossed"]
    assert diagnostics["primary_sort"] == (
        "strict_noncrossing_phase_then_preplan_status_then_transition_joint_path_cost"
    )


def test_joint_space_ranking_keeps_unranked_clear_before_planned_crossed(
    monkeypatch,
) -> None:
    def fake_plan(
        _commander,
        *,
        targets,
        labels,
        start_joint_positions,
        joint_names,
        config,
        label_prefix,
    ):
        del targets, joint_names, config, label_prefix
        trajectories = {label: ((0.0,) * 7,) for label in labels}
        return SimpleNamespace(
            trajectories=trajectories,
            joint_path_cost=0.1,
            terminal_joint_positions=tuple(float(value) for value in start_joint_positions),
            diagnostics=(),
        )

    monkeypatch.setattr(
        "scripts.plan_simple_dual_robot_sim.plan_pose_sequence_multi_ik",
        fake_plan,
    )
    crossed = _task(
        "crossed",
        "holder_crossed",
        "inserter_crossed",
        0.99,
        transition_id="tr_crossed",
        corridor_y=-1.0,
        transition_crosses=True,
    )
    clear = _task(
        "clear",
        "holder_clear",
        "inserter_clear",
        0.20,
        transition_id="tr_clear",
        corridor_y=-1.0,
        transition_crosses=False,
    )

    ranked, _diagnostics, _preferred = _rank_tasks_by_inserter_joint_path(
        [crossed, clear],
        commander=object(),
        candidate_limit=1,
        ik_candidate_count=2,
        beam_width=1,
    )

    assert [task.transition_id for task in ranked] == ["tr_clear", "tr_crossed"]


def test_pregrasp_aabb_schedule_avoids_intended_grasp_contacts() -> None:
    obstacles = {
        "holder_base_00": {
            "id": "base",
            "active_target": "holder_pregrasp",
        },
        "holder_incoming_00": {
            "id": "incoming_for_holder",
            "active_target": "holder_pregrasp",
        },
        "inserter_incoming_00": {
            "id": "incoming_for_inserter",
            "active_target": "inserter_pickup_pregrasp",
        },
    }

    assert [
        obstacle["id"]
        for obstacle in _pregrasp_aabb_obstacles_for_target(
            obstacles,
            target_name="holder_pregrasp",
        )
    ] == ["base", "incoming_for_holder"]
    assert [
        obstacle["id"]
        for obstacle in _pregrasp_aabb_obstacles_for_target(
            obstacles,
            target_name="inserter_pickup_pregrasp",
        )
    ] == ["incoming_for_inserter"]
    assert (
        _pregrasp_aabb_obstacles_for_target(
            obstacles,
            target_name="holder_grasp",
        )
        == []
    )


def test_candidate_recovery_retracts_inserter_before_holder() -> None:
    shared: dict[str, object] = {}
    commanders = {
        "holder": _ResetCommander("holder", shared),
        "inserter": _ResetCommander("inserter", shared),
    }

    ok, messages = _reset_active_roles(
        commanders,  # type: ignore[arg-type]
        active_roles=("holder", "inserter"),
        recovering_from_candidate=True,
    )

    assert ok
    assert list(messages) == ["inserter", "holder"]
    assert shared["calls"] == [
        "plan:inserter",
        "execute:inserter",
        "plan:holder",
        "execute:holder",
    ]
