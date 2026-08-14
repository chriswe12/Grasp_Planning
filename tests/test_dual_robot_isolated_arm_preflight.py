from __future__ import annotations

from grasp_planning.pipeline.dual_robot_isolated_arm_preflight import (
    compare_candidate_arm_isolation,
    disable_pairwise_collisions,
    links_by_prefix,
)
from grasp_planning.ros2.dual_real_grasp_executor import MOTION_SEQUENCE
from grasp_planning.ros2.moveit_pose_commander import PoseTarget


def test_links_by_prefix_filters_by_robot_prefix() -> None:
    entry_names = ("lbr_one_link_0", "lbr_one_gripper_base_link", "lbr_two_link_0", "table")

    assert links_by_prefix(entry_names, "lbr_one") == (
        "lbr_one_link_0",
        "lbr_one_gripper_base_link",
    )
    assert links_by_prefix(entry_names, "lbr_two") == ("lbr_two_link_0",)
    assert links_by_prefix(entry_names, "lbr_three") == ()


def test_disable_pairwise_collisions_preserves_existing_entries() -> None:
    entry_names = ("lbr_one_link_0", "lbr_one_link_1", "table")
    entry_rows = (
        (False, True, False),
        (True, False, False),
        (False, False, False),
    )

    names, rows = disable_pairwise_collisions(
        entry_names,
        entry_rows,
        group_a=("lbr_one_link_0", "lbr_one_link_1"),
        group_b=("lbr_two_link_0",),
    )

    assert names == ("lbr_one_link_0", "lbr_one_link_1", "table", "lbr_two_link_0")
    index = {name: position for position, name in enumerate(names)}
    # Pre-existing Adjacent pair is untouched.
    assert rows[index["lbr_one_link_0"]][index["lbr_one_link_1"]] is True
    # Table was never in either group; it stays disallowed against the newly
    # appended link too.
    assert rows[index["table"]][index["lbr_two_link_0"]] is False
    # Every group_a x group_b pair is now allowed, symmetrically.
    assert rows[index["lbr_one_link_0"]][index["lbr_two_link_0"]] is True
    assert rows[index["lbr_two_link_0"]][index["lbr_one_link_0"]] is True
    assert rows[index["lbr_one_link_1"]][index["lbr_two_link_0"]] is True


def test_disable_pairwise_collisions_is_idempotent_on_a_ragged_matrix() -> None:
    # Some real ACM responses can have entry rows shorter than entry_names
    # (a link only ever appeared as link1 in the SRDF); the helper must pad
    # rather than crash.
    entry_names = ("a", "b")
    entry_rows = ((False,),)  # only one ragged row supplied

    names, rows = disable_pairwise_collisions(entry_names, entry_rows, group_a=("a",), group_b=("b",))

    assert names == ("a", "b")
    assert rows[0][1] is True
    assert rows[1][0] is True


class _FakeIkCommander:
    def __init__(self, *, scene: dict[str, object], hard_target_x: float = 1.0) -> None:
        self._scene = scene
        self._hard_target_x = hard_target_x
        self.ik_calls: list[float] = []

    def compute_ik(self, target, seed_joint_positions=None):
        del seed_joint_positions
        self.ik_calls.append(target.x)
        if target.x == self._hard_target_x and not self._scene["isolated"]:
            return None, "blocked by the other arm's current configuration"
        return [0.0] * 7, "ok"

    def get_current_pose(self, *, frame_id: str) -> PoseTarget:
        return PoseTarget.from_quaternion(
            x=0.0, y=0.0, z=0.1, quaternion_xyzw=(0.0, 0.0, 0.0, 1.0), frame_id=frame_id
        )


class _FakeSceneCommander(_FakeIkCommander):
    """Adds the ACM get/apply surface; only the 'holder' role needs this."""

    def __init__(self, *, scene: dict[str, object], hard_target_x: float = 1.0) -> None:
        super().__init__(scene=scene, hard_target_x=hard_target_x)
        self.apply_calls: list[tuple[tuple[str, ...], tuple[tuple[bool, ...], ...]]] = []

    def get_allowed_collision_matrix(self):
        return ("lbr_one_link_0", "lbr_two_link_0"), ((False, False), (False, False))

    def apply_allowed_collision_matrix(self, entry_names, entry_rows):
        entry_names = tuple(entry_names)
        entry_rows = tuple(tuple(row) for row in entry_rows)
        self.apply_calls.append((entry_names, entry_rows))
        self._scene["isolated"] = any(any(row) for row in entry_rows)
        return True, "applied"


def _plan_with_hard_target(hard_target_name: str) -> dict[str, object]:
    targets = {}
    for _, target_name in MOTION_SEQUENCE:
        x = 1.0 if target_name == hard_target_name else 0.5
        targets[target_name] = {
            "position_world_m": [x, 0.0, 0.1],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        }
    return {
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


def test_compare_candidate_arm_isolation_flags_only_the_blocked_target() -> None:
    scene = {"isolated": False}
    holder = _FakeSceneCommander(scene=scene)
    inserter = _FakeIkCommander(scene=scene)
    commanders = {"holder": holder, "inserter": inserter}
    candidate = _plan_with_hard_target("inserter_preinsertion")

    results = compare_candidate_arm_isolation(
        candidate=candidate,
        commanders=commanders,
        frame_id="base_link",
        stop_after="inserter_preinsertion",
    )

    assert len(results) == len(MOTION_SEQUENCE)
    by_name = {result.target_name: result for result in results}

    blocked = by_name["inserter_preinsertion"]
    assert blocked.coupled_ok is False
    assert blocked.isolated_ok is True
    assert blocked.diverges is True

    for target_name, result in by_name.items():
        if target_name == "inserter_preinsertion":
            continue
        assert result.coupled_ok is True, target_name
        assert result.isolated_ok is True, target_name
        assert result.diverges is False, target_name

    # The ACM was flipped once to isolate and once more to restore, and the
    # scene ends up back in its coupled (non-isolated) state.
    assert len(holder.apply_calls) == 2
    assert scene["isolated"] is False


def test_compare_candidate_arm_isolation_reports_unexplained_failures() -> None:
    scene = {"isolated": False}
    holder = _FakeSceneCommander(scene=scene)
    # This role's own IK never succeeds for the hard target, isolated or not,
    # so it must not be misreported as arm-arm interference.
    inserter = _FakeIkCommander(scene=scene)
    inserter.compute_ik = lambda target, seed_joint_positions=None: (
        (None, "genuinely unreachable") if target.x == 1.0 else ([0.0] * 7, "ok")
    )
    commanders = {"holder": holder, "inserter": inserter}
    candidate = _plan_with_hard_target("inserter_preinsertion")

    results = compare_candidate_arm_isolation(
        candidate=candidate,
        commanders=commanders,
        frame_id="base_link",
        stop_after="inserter_preinsertion",
    )

    blocked = {result.target_name: result for result in results}["inserter_preinsertion"]
    assert blocked.coupled_ok is False
    assert blocked.isolated_ok is False
    assert blocked.diverges is False
    assert scene["isolated"] is False


def test_compare_candidate_arm_isolation_requires_prefixed_acm_entries() -> None:
    class _EmptyAcmCommander(_FakeSceneCommander):
        def get_allowed_collision_matrix(self):
            return ("table", "other_object"), ((False, False), (False, False))

    scene = {"isolated": False}
    commanders = {
        "holder": _EmptyAcmCommander(scene=scene),
        "inserter": _FakeIkCommander(scene=scene),
    }
    candidate = _plan_with_hard_target("inserter_preinsertion")

    try:
        compare_candidate_arm_isolation(
            candidate=candidate,
            commanders=commanders,
            frame_id="base_link",
            stop_after="inserter_preinsertion",
        )
    except RuntimeError as exc:
        assert "lbr_one_*/lbr_two_*" in str(exc)
    else:
        raise AssertionError("Expected a missing-link-prefix error.")


def test_compare_candidate_arm_isolation_threads_ik_strategy_to_both_passes() -> None:
    scene = {"isolated": False}
    holder = _FakeSceneCommander(scene=scene)
    inserter = _FakeIkCommander(scene=scene)
    commanders = {"holder": holder, "inserter": inserter}
    candidate = _plan_with_hard_target("inserter_preinsertion")

    results = compare_candidate_arm_isolation(
        candidate=candidate,
        commanders=commanders,
        frame_id="base_link",
        stop_after="inserter_preinsertion",
        ik_strategy="cartesian_waypoints",
        cartesian_waypoint_count=4,
    )

    blocked = {result.target_name: result for result in results}["inserter_preinsertion"]
    # get_current_pose only exists on this fake because the cartesian_waypoints
    # strategy was actually used for every target, proving the strategy
    # argument reached both the coupled and isolated preflight passes.
    assert len(inserter.ik_calls) > len(MOTION_SEQUENCE)
    assert blocked.coupled_ok is False
    assert blocked.isolated_ok is True
    assert blocked.diverges is True
