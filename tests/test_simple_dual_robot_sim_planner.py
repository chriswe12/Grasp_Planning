from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.plan_simple_dual_robot_sim import (
    IK_PREFLIGHT_TARGETS,
    _configure_role_assignment,
    _ik_preflight_pair,
    _new_ik_preflight_state,
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

    def compute_ik(self, _target, *, seed_joint_positions=None):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return None, "synthetic no IK"
        return [0.0] * 7, "ok"


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
):
    pose = {
        "position_world_m": [0.5, 0.0, 0.2],
        "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
    }
    targets = {
        target_name: dict(pose) for target_names in IK_PREFLIGHT_TARGETS.values() for target_name in target_names
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

    first_ok, first_failure = _ik_preflight_pair(
        _task("pair_1", "holder_bad", "inserter_1", 0.9),
        commanders=commanders,
        feasible_cache=cache,
        state=state,
        rank=1,
    )
    assert not first_ok
    assert "holder_bad failed holder_pregrasp" in first_failure
    assert holder.calls == 1
    assert inserter.calls == 0

    second_ok, second_failure = _ik_preflight_pair(
        _task("pair_2", "holder_bad", "inserter_2", 0.8),
        commanders=commanders,
        feasible_cache=cache,
        state=state,
        rank=2,
    )
    assert not second_ok
    assert "cached IK failure" in second_failure
    assert holder.calls == 1
    assert inserter.calls == 0

    third_ok, third_failure = _ik_preflight_pair(
        _task("pair_3", "holder_good", "inserter_3", 0.7),
        commanders=commanders,
        feasible_cache=cache,
        state=state,
        rank=3,
    )
    assert third_ok
    assert third_failure == ""
    assert holder.calls == 3
    assert inserter.calls == 5

    assert state["mode"] == "lazy_strict_score_order"
    assert state["pair_tasks_checked"] == 3
    assert state["pair_tasks_after"] == 1
    assert state["holder_grasps_checked"] == 2
    assert state["holder_grasps_feasible"] == 1
    assert state["inserter_grasps_checked"] == 1
    assert state["inserter_grasps_feasible"] == 1
    assert [record["pair_id"] for record in state["pair_records"]] == ["pair_1", "pair_2", "pair_3"]
    assert state["pair_records"][1]["roles"]["holder"]["cache_hit"]


def test_holder_only_preflight_does_not_query_inserter() -> None:
    holder = _FakeCommander()
    inserter = _FakeCommander(fail_first=True)
    state = _new_ik_preflight_state(pair_task_count=1)

    ok, failure = _ik_preflight_pair(
        _task("pair_1", "holder_1", "inserter_1", 0.9),
        commanders={"holder": holder, "inserter": inserter},
        feasible_cache={"holder": {}, "inserter": {}},
        state=state,
        rank=1,
        roles=("holder",),
    )

    assert ok
    assert failure == ""
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

    first_ok, _ = _ik_preflight_pair(
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
    second_ok, _ = _ik_preflight_pair(
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
