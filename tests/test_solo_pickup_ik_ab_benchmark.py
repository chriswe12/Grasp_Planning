import numpy as np

from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from scripts.run_solo_pickup_ik_ab_benchmark import (
    _aggregate,
    _assembly_mesh_in_source_frame,
    _world_scene_overlays,
    _write_html,
    blocking_solo_contacts,
)


def test_assembly_mesh_is_expressed_in_saved_grasp_source_frame() -> None:
    mesh_assembly = TriangleMesh(
        vertices_obj=np.asarray(
            [
                [1.0, 2.0, 3.0],
                [2.0, 2.0, 3.0],
                [1.0, 3.0, 3.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    source_pose_assembly = ObjectWorldPose(
        position_world=(1.0, 2.0, 3.0),
        orientation_xyzw_world=(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
    )

    mesh_source = _assembly_mesh_in_source_frame(mesh_assembly, source_pose_assembly)

    np.testing.assert_allclose(
        mesh_source.vertices_obj,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(mesh_source.faces, mesh_assembly.faces)


def test_world_scene_overlays_show_both_physical_robot_bases_and_shoulders() -> None:
    overlays = _world_scene_overlays(
        {"assembly_x": 0.55, "assembly_y": 0.0, "assembly_z": -0.03},
        floor_z=-0.03,
    )

    markers = {marker["label"]: marker for marker in overlays["markers"]}
    assert markers["lbr_one base"]["position"] == [0.0, -0.42, -0.03]
    assert markers["lbr_two base"]["position"] == [0.0, 0.42, -0.03]
    assert markers["lbr_one shoulder"]["position"] == [0.0, -0.42, 0.31000000000000005]
    assert markers["lbr_two shoulder"]["connect_to_pregrasp"] is True
    assert overlays["axes"] == {"origin": [0.0, 0.0, -0.03], "length_m": 0.18}


def test_report_embeds_first_failed_pose_and_parent_grasp_navigation(tmp_path) -> None:
    output = tmp_path / "index.html"
    failed = {
        "case_id": "failed_pose",
        "status": "complete",
        "dual_success": False,
        "grasp_debug_html": "cases/failed/incoming_grasps.html",
        "incoming_part_id": "3",
        "orientation_id": "pitch_90_yaw_0",
        "placement_id": "right_inner_back",
        "floor_valid_candidates": 12,
        "solo": {},
        "solo_holder": {},
    }
    passed = {
        **failed,
        "case_id": "passed_pose",
        "dual_success": True,
        "grasp_debug_html": "cases/passed/incoming_grasps.html",
    }

    _write_html(
        output,
        {
            "records": [failed, passed],
            "aggregate": {
                "completed": 2,
                "dual_success": 1,
                "outcome_matrix": {},
            },
        },
    )

    report = output.read_text(encoding="utf-8")
    assert "<iframe id='graspViewer'" in report
    assert "src='cases/failed/incoming_grasps.html'" in report
    assert "part ${item.part} | ${item.orientation}" in report
    assert "fabrica-grasp-step" in report
    assert "event.key==='ArrowLeft'" in report
    assert 'const viewerCases=[{"case_id": "failed_pose"' in report
    assert '"case_id": "passed_pose"' not in report.split("const viewerCases=", 1)[1].split(";", 1)[0]


def test_blocking_solo_contacts_ignores_passive_and_inter_arm_contacts() -> None:
    contacts = [
        {"body_1": "lbr_one_link_5", "body_2": "lbr_one_link_7"},
        {"body_1": "lbr_one_link_5", "body_2": "lbr_two_link_5"},
        {"body_1": "lbr_two_link_6", "body_2": "solo_pickup_work_surface"},
        {"body_1": "lbr_one_link_6", "body_2": "solo_pickup_work_surface"},
    ]

    blocking, ignored = blocking_solo_contacts(contacts, active_robot="lbr_one")

    assert blocking == [contacts[0], contacts[3]]
    assert ignored == [contacts[1], contacts[2]]


def test_aggregate_separates_assigned_arm_from_either_arm_recovery() -> None:
    records = [
        {
            "status": "complete",
            "dual_success": False,
            "baseline_inserter_arm": "lbr_one",
            "baseline_holder_arm": "lbr_two",
            "either_solo_success": True,
            "assigned_roles_solo_success": False,
            "either_assignment_solo_success": True,
            "solo": {"lbr_one": {"success": False}, "lbr_two": {"success": True}},
            "solo_holder": {"lbr_one": {"success": True}, "lbr_two": {"success": True}},
        },
        {
            "status": "complete",
            "dual_success": False,
            "baseline_inserter_arm": "lbr_two",
            "baseline_holder_arm": "lbr_one",
            "either_solo_success": True,
            "assigned_roles_solo_success": True,
            "either_assignment_solo_success": True,
            "solo": {"lbr_one": {"success": False}, "lbr_two": {"success": True}},
            "solo_holder": {"lbr_one": {"success": True}, "lbr_two": {"success": False}},
        },
        {
            "status": "complete",
            "dual_success": True,
            "baseline_inserter_arm": "lbr_one",
            "baseline_holder_arm": "lbr_two",
            "either_solo_success": True,
            "assigned_roles_solo_success": True,
            "either_assignment_solo_success": True,
            "solo": {"lbr_one": {"success": True}, "lbr_two": {"success": False}},
            "solo_holder": {"lbr_one": {"success": False}, "lbr_two": {"success": True}},
        },
    ]

    aggregate = _aggregate(records)

    assert aggregate["completed"] == 3
    assert aggregate["dual_success"] == 1
    assert aggregate["assigned_inserter_solo_success"] == 2
    assert aggregate["assigned_pickups_solo_success"] == 2
    assert aggregate["either_assignment_solo_success"] == 3
    assert aggregate["incoming_either_arm_solo_success"] == 3
    assert aggregate["outcome_matrix"] == {
        "dual_fail__solo_pass": 2,
        "dual_pass__solo_pass": 1,
    }
