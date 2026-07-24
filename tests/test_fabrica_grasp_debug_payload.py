from __future__ import annotations

from grasp_planning.grasping.fabrica_grasp_debug import CandidateStatus, SavedGraspCandidate, candidate_payload


def _candidate() -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id="g0001",
        grasp_position_obj=(0.0, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.0, -0.01, 0.0),
        contact_point_b_obj=(0.0, 0.01, 0.0),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.02,
        roll_angle_rad=0.0,
    )


def test_candidate_payload_uses_selected_kuka_stl_visual_geometry() -> None:
    payload = candidate_payload(
        [CandidateStatus(grasp=_candidate(), status="accepted", reason="ok")],
        contact_gap_m=0.002,
        gripper_collision_model="kuka_y_gripper",
    )[0]

    assert payload["franka_left_boxes"] == []
    assert payload["franka_right_boxes"] == []
    assert len(payload["franka_hand_vertices_obj"]) > 50
    assert len(payload["franka_hand_faces"]) > 50


def test_candidate_payload_moves_kuka_visual_fingers_with_jaw_width() -> None:
    def y_extent(jaw_width: float) -> float:
        candidate = _candidate()
        candidate = SavedGraspCandidate(
            grasp_id=candidate.grasp_id,
            grasp_position_obj=candidate.grasp_position_obj,
            grasp_orientation_xyzw_obj=candidate.grasp_orientation_xyzw_obj,
            contact_point_a_obj=(0.0, -0.5 * jaw_width, 0.0),
            contact_point_b_obj=(0.0, 0.5 * jaw_width, 0.0),
            contact_normal_a_obj=candidate.contact_normal_a_obj,
            contact_normal_b_obj=candidate.contact_normal_b_obj,
            jaw_width=jaw_width,
            roll_angle_rad=candidate.roll_angle_rad,
        )
        payload = candidate_payload(
            [CandidateStatus(grasp=candidate, status="accepted", reason="ok")],
            contact_gap_m=0.002,
            gripper_collision_model="kuka_y_gripper",
        )[0]
        finger_vertices = [vertex for vertex in payload["franka_hand_vertices_obj"] if float(vertex[2]) > -0.055]
        ys = [float(vertex[1]) for vertex in finger_vertices]
        return max(ys) - min(ys)

    assert y_extent(0.08) > y_extent(0.01)


def test_candidate_payload_places_kuka_tcp_at_saved_grasp_position() -> None:
    candidate = SavedGraspCandidate(
        grasp_id="g0002",
        grasp_position_obj=(0.03, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.0, -0.01, 0.0),
        contact_point_b_obj=(0.0, 0.01, 0.0),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.02,
        roll_angle_rad=0.0,
    )

    payload = candidate_payload(
        [CandidateStatus(grasp=candidate, status="accepted", reason="ok")],
        contact_gap_m=0.002,
        gripper_collision_model="kuka_y_gripper",
    )[0]

    assert payload["franka_hand_origin_obj"] == [0.03, 0.0, -0.1505]
