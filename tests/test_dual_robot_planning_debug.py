from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline import dual_robot_planning_debug as debug_module
from grasp_planning.pipeline.dual_robot_planning_debug import (
    DualRobotPlanningDebugServer,
    dual_robot_planning_scene_payload,
    dual_robot_planning_scene_payload_from_plan,
)


def _pose(x: float, y: float, z: float) -> ObjectWorldPose:
    return ObjectWorldPose(
        position_world=(x, y, z),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )


def _debug_task(tmp_path):
    subassembly_mesh = tmp_path / "base.obj"
    subassembly_mesh.write_text(
        "v 1 0 0\nv 1.1 0 0\nv 1 0.1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    incoming_mesh = tmp_path / "incoming.obj"
    incoming_mesh.write_text(
        "v 2 0 0\nv 2.1 0 0\nv 2 0.1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    targets = {
        "holder_pregrasp": {
            "position_world_m": [0.45, -0.2, 0.2],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        },
        "holder_grasp": {
            "position_world_m": [0.50, -0.2, 0.2],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        },
        "inserter_pickup_pregrasp": {
            "position_world_m": [0.55, 0.25, 0.2],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        },
        "inserter_pickup_grasp": {
            "position_world_m": [0.60, 0.25, 0.2],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        },
        "inserter_pickup_lift": {
            "position_world_m": [0.60, 0.25, 0.28],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        },
        "inserter_above_preinsertion": {
            "position_world_m": [0.60, 0.10, 0.28],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        },
        "inserter_preinsertion": {
            "position_world_m": [0.60, 0.10, 0.20],
            "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
        },
    }
    payload = {
        "assembly": "synthetic",
        "step_id": "step_001_part_1",
        "incoming_part_id": "1",
        "base_part_id": "0",
        "pair_id": "pair_1",
        "transition_id": "tr_opposite",
        "execution_candidate_id": "pair_1__tr_opposite",
        "selection_score": 0.8,
        "transition_motion_score": 0.7,
        "layout_proxy_components": {
            "pickup_segments_cross_xy": False,
            "transition_segments_cross_xy": True,
            "crossing_penalty_applied": 0.25,
        },
        "targets": targets,
        "grasps": {
            "holder": {"jaw_width_m": 0.04},
            "inserter_pickup": {"jaw_width_m": 0.05},
        },
        "layout": {
            "holder_base_world_m": [0.0, -0.42, 0.0],
            "inserter_base_world_m": [0.0, 0.42, 0.0],
            "pickup_floor_z_world_m": 0.0,
        },
        "objects": {
            "subassembly": {
                "base_part_id": "0",
                "mesh_scale": 1.0,
                "source_pose_assembly": {
                    "position_world_m": [1.0, 0.0, 0.0],
                    "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                },
                "source_pose_world": {
                    "position_world_m": [0.5, -0.2, 0.1],
                    "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                },
                "parts": [
                    {
                        "part_id": "0",
                        "mesh_path": str(subassembly_mesh),
                        "mesh_scale": 1.0,
                    }
                ],
            },
            "incoming": {
                "part_id": "1",
                "mesh_path": str(incoming_mesh),
                "mesh_scale": 1.0,
                "source_pose_assembly": {
                    "position_world_m": [2.0, 0.0, 0.0],
                    "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                },
                "pickup_source_pose_world": {
                    "position_world_m": [0.6, 0.25, 0.05],
                    "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                },
                "preinsertion_source_pose_world": {
                    "position_world_m": [0.6, 0.1, 0.05],
                    "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                },
                "final_source_pose_world": {
                    "position_world_m": [0.6, 0.0, 0.05],
                    "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
        "transition_symmetry": {"is_identity": False},
    }
    return SimpleNamespace(
        assembly="synthetic",
        step_id="step_001_part_1",
        incoming_part_id="1",
        base_part_id="0",
        pair_id="pair_1",
        transition_id="tr_opposite",
        execution_candidate_id="pair_1__tr_opposite",
        selection_score=0.8,
        transition_motion_score=0.7,
        layout_proxy_components={
            "pickup_segments_cross_xy": False,
            "transition_segments_cross_xy": True,
            "crossing_penalty_applied": 0.25,
        },
        subassembly_parts=(SimpleNamespace(part_id="0", mesh_path=subassembly_mesh),),
        mesh_scale=1.0,
        holder_source_pose_assembly=_pose(1.0, 0.0, 0.0),
        holder_source_pose_world=_pose(0.5, -0.2, 0.1),
        incoming_mesh_path=incoming_mesh,
        incoming_source_pose_assembly=_pose(2.0, 0.0, 0.0),
        incoming_pickup_source_pose_world=_pose(0.6, 0.25, 0.05),
        incoming_preinsertion_source_pose_world=_pose(0.6, 0.1, 0.05),
        incoming_final_source_pose_world=_pose(0.6, 0.0, 0.05),
        transport_clearance_m=0.08,
        pickup_floor_z_world_m=0.0,
        transition_symmetry={"is_identity": False},
        to_payload=lambda: payload,
    )


def test_live_debug_scene_uses_world_frame_part_poses(tmp_path) -> None:
    scene = dual_robot_planning_scene_payload(_debug_task(tmp_path))

    base_vertices = np.asarray(
        scene["subassembly_parts"][0]["vertices_world_m"],
        dtype=float,
    )
    incoming_vertices = np.asarray(
        scene["incoming"]["vertices_source_m"],
        dtype=float,
    )
    assert np.allclose(base_vertices[0], [0.5, -0.2, 0.1])
    assert np.allclose(incoming_vertices[0], [0.0, 0.0, 0.0])
    assert scene["incoming_poses"]["pickup"]["position_world_m"] == [
        0.6,
        0.25,
        0.05,
    ]
    assert scene["incoming_poses"]["above_preinsertion"]["position_world_m"] == [0.6, 0.1, 0.13]
    assert scene["layout_proxy_components"]["transition_segments_cross_xy"] is True
    assert scene["jaw_widths_m"] == {"holder": 0.04, "inserter": 0.05}
    assert np.allclose(
        [scene["approach_widths_m"]["holder"], scene["approach_widths_m"]["inserter"]],
        [0.05, 0.06],
    )
    assert "holder_grasp" in scene["gripper_floor_clearance_m"]
    assert np.isfinite(scene["gripper_floor_clearance_m"]["holder_grasp"])


def test_live_debug_scene_can_be_rebuilt_from_serialized_real_task(tmp_path) -> None:
    task = _debug_task(tmp_path)

    scene = dual_robot_planning_scene_payload_from_plan(task.to_payload())

    assert scene["execution_candidate_id"] == "pair_1__tr_opposite"
    assert scene["incoming_poses"]["pickup_lift"]["position_world_m"] == [0.6, 0.25, 0.13]
    assert scene["incoming_poses"]["preinsertion"]["position_world_m"] == [0.6, 0.1, 0.05]
    assert scene["robot_bases_world_m"]["holder"] == [0.0, -0.42, 0.0]


def test_live_debug_server_reports_candidate_and_phase(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeHttpServer:
        server_address = ("127.0.0.1", 43210)

        def __init__(self, _address, _handler) -> None:
            pass

        def serve_forever(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def server_close(self) -> None:
            pass

    monkeypatch.setattr(debug_module, "ThreadingHTTPServer", FakeHttpServer)
    task = _debug_task(tmp_path)
    server = DualRobotPlanningDebugServer(port=0)
    server.update(
        task=task,
        attempt_index=5,
        attempt_total=48,
        phase="inserter_preinsertion",
        status="planning",
        message="testing transition",
        candidate_counts={
            "pickup_grasps_checked": 783,
            "pickup_grasps_accepted": 434,
            "stage3_retained_pairs": 256,
            "stage3_retained_execution_candidates": 256,
            "pose_feasible_retained_execution_candidates": 10,
            "pose_feasible_validated_transition_fallback_candidates": 2,
            "pose_feasible_identity_fallback_candidates": 1556,
            "planner_queue_execution_candidates": 10,
            "planner_queue_unique_holder_grasps": 8,
            "planner_queue_unique_inserter_grasps": 5,
            "joint_rank_candidates_checked": 8,
            "joint_rank_candidates_planned": 3,
            "exact_ik_pair_tasks_checked": 5,
            "exact_ik_holder_grasps_checked": 4,
            "exact_ik_inserter_grasps_checked": 2,
        },
    )
    state, scene = server.snapshot()

    assert state["attempt_index"] == 5
    assert state["state_revision"] == 1
    assert state["server_id"]
    assert state["pair_id"] == "pair_1"
    assert state["transition_id"] == "tr_opposite"
    assert state["phase"] == "inserter_preinsertion"
    assert state["candidate_counts"]["pickup_grasps_checked"] == 783
    assert state["candidate_counts"]["planner_queue_execution_candidates"] == 10
    assert state["candidate_counts"]["exact_ik_pair_tasks_checked"] == 5
    assert scene["frame_id"] == "base_link"
    assert "1 · holder grasp" in debug_module._LIVE_HTML
    assert "2 · incoming grasp" in debug_module._LIVE_HTML
    assert "3 · transition" in debug_module._LIVE_HTML
    assert "Candidate checks" in debug_module._LIVE_HTML
    assert 'id="pickup-counts"' in debug_module._LIVE_HTML
    assert 'id="floor-plane"' in debug_module._LIVE_HTML
    assert 'id="stage3-counts"' in debug_module._LIVE_HTML
    assert 'id="fallback-counts"' in debug_module._LIVE_HTML
    assert 'id="queue-counts"' in debug_module._LIVE_HTML
    assert 'id="joint-counts"' in debug_module._LIVE_HTML
    assert 'id="ik-counts"' in debug_module._LIVE_HTML
    assert "exact_ik_seed_calls" in debug_module._LIVE_HTML
    assert "exact_ik_solutions_found" in debug_module._LIVE_HTML
    assert 'id="crossing"' in debug_module._LIVE_HTML
    assert 'id="holder-floor"' in debug_module._LIVE_HTML
    assert "gripper_floor_clearance_m" in debug_module._LIVE_HTML
    assert 'phase==="ik_preflight"' in debug_module._LIVE_HTML
    assert 'p==="joint_space_ranking"' in debug_module._LIVE_HTML
    assert "frameBounds=bounds(incomingVertices)" in debug_module._LIVE_HTML
    assert "const projected=vertices.map(project)" in debug_module._LIVE_HTML
    assert "requestAnimationFrame" in debug_module._LIVE_HTML
    assert "nextLive.state_revision!==lastStateRevision" in debug_module._LIVE_HTML
    assert "nextLive.server_id!==lastServerId" in debug_module._LIVE_HTML
    assert 'p==="pickup_floor_check"' in debug_module._LIVE_HTML
    assert "setTimeout(poll,100)" in debug_module._LIVE_HTML

    server.update(
        task=task,
        phase="inserter_preinsertion",
        status="failed",
        message="same candidate, new status",
        candidate_counts={"exact_ik_pair_tasks_checked": 6},
        record_event=False,
    )
    next_state, next_scene = server.snapshot()
    assert next_state["state_revision"] == 2
    assert next_state["scene_revision"] == state["scene_revision"]
    assert next_state["candidate_counts"]["pickup_grasps_checked"] == 783
    assert next_state["candidate_counts"]["exact_ik_pair_tasks_checked"] == 6
    assert next_scene["execution_candidate_id"] == scene["execution_candidate_id"]
