from __future__ import annotations

from types import SimpleNamespace

from scripts.plan_simple_dual_robot_sim import (
    IK_PREFLIGHT_TARGETS,
    _ik_preflight_pair,
    _new_ik_preflight_state,
    _pregrasp_aabb_obstacles_for_target,
)


class _FakeCommander:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.calls = 0

    def compute_ik(self, _target):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            return None, "synthetic no IK"
        return [0.0] * 7, "ok"


def _task(
    pair_id: str,
    holder_id: str,
    inserter_id: str,
    score: float,
    *,
    preinsertion_x: float = 0.5,
    transition_id: str = "tr_identity",
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
        selection_score=score,
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
