from __future__ import annotations

import copy
import json
import math
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import grasp_planning.pipeline.dual_robot_simple_sim as simple_sim
from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspCandidate
from grasp_planning.grasping.grasp_transforms import WorldFrameGraspCandidate
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline.dual_grasp_pair_planner import DualGraspPairConfig
from grasp_planning.pipeline.dual_robot_pair_scoring import MovableFrame
from grasp_planning.pipeline.dual_robot_simple_sim import (
    DEFAULT_HOLDER_PREGRASP_OFFSET_M,
    DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT,
    NoPoseFeasibleDualTasksError,
    compose_source_pose_world,
    load_simple_dual_robot_pair_tasks,
    resolve_dual_robot_step_selection,
    resolve_planar_runtime_layout,
    simple_dual_robot_attached_collision_objects,
    simple_dual_robot_pregrasp_aabb_obstacles,
    simple_dual_robot_pregrasp_aabb_schedule,
    source_local_subassembly_mesh,
    source_pose_resting_on_floor,
    translated_source_pose_world,
    with_inserter_pickup_pregrasp_offset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_holder_pregrasp_matches_offline_feasibility_config() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/dual_grasp_planning.yaml").read_text(encoding="utf-8"))

    assert DEFAULT_HOLDER_PREGRASP_OFFSET_M == 0.05
    assert config["holder_feasibility"]["pregrasp_offset_m"] == DEFAULT_HOLDER_PREGRASP_OFFSET_M


def test_runtime_pair_budget_matches_stage3_retention_budget() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/dual_grasp_planning.yaml").read_text(encoding="utf-8"))

    assert DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT == 256
    assert DualGraspPairConfig().max_accepted_pairs == DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT
    assert config["pair_planning"]["max_accepted_pairs"] == DEFAULT_RUNTIME_PAIR_CANDIDATE_LIMIT


def test_pickup_symmetry_bridge_expands_every_destination_over_exact_stage3_orbit() -> None:
    identity = np.eye(4, dtype=float)
    turn_x = np.diag((1.0, -1.0, -1.0, 1.0))
    turn_x[:3, 3] = (0.02, -0.01, 0.03)
    turn_z = np.diag((-1.0, -1.0, 1.0, 1.0))
    turn_z[:3, 3] = (-0.01, 0.03, 0.0)
    approximate = np.diag((-1.0, 1.0, -1.0, 1.0))

    def transition(name: str, matrix: np.ndarray) -> dict[str, object]:
        return {
            "incoming_destination_symmetry_name": name,
            "incoming_symmetry_source_m": matrix.tolist(),
            "incoming_destination_transform_assembly_m": matrix.tolist(),
        }

    sources, source_diagnostics = simple_sim._exact_pickup_symmetry_sources(
        (
            transition("identity", identity),
            transition("turn_x", turn_x),
            transition("turn_x", turn_x),
            transition("turn_z", turn_z),
            transition("approximate_y", approximate),
        ),
        exact_validations={
            "turn_x": {"accepted": True, "vertex_max_m": 1.0e-8},
            "turn_z": {"accepted": True, "vertex_max_m": 2.0e-8},
        },
    )

    identity_destination = SavedGraspCandidate(
        grasp_id="g_identity",
        grasp_position_obj=(0.10, 0.02, 0.03),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(0.10, 0.01, 0.03),
        contact_point_b_obj=(0.10, 0.03, 0.03),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.02,
        roll_angle_rad=0.0,
        score=0.8,
    )
    symmetry_sibling = simple_sim.transform_grasp_candidate_by_source_symmetry(
        identity_destination,
        symmetry_name="turn_x",
        matrix_source=turn_x,
    )
    aliases, alias_diagnostics = simple_sim._pickup_symmetry_bridge_candidates(
        (identity_destination, symmetry_sibling),
        symmetry_sources=sources,
    )

    assert [source["name"] for source in sources] == ["turn_x", "turn_z"]
    assert source_diagnostics["duplicate_matrix_count"] == 1
    assert source_diagnostics["rejected_names"] == {"approximate_y": "not_exactly_validated"}
    assert source_diagnostics["matrix_source"] == ("stage3_transition_artifact_scaled_source_frame")
    assert alias_diagnostics["raw_alias_count"] == 4
    assert alias_diagnostics["alias_count"] == 4

    destinations = {
        identity_destination.grasp_id: identity_destination,
        symmetry_sibling.grasp_id: symmetry_sibling,
    }
    matrices = {"turn_x": turn_x, "turn_z": turn_z}
    assert {
        dict(alias.metadata or {})["runtime_pickup_symmetry_bridge"]["destination_grasp_id"] for alias in aliases
    } == set(destinations)
    identity_aliases = []
    for alias in aliases:
        bridge = dict(alias.metadata or {})["runtime_pickup_symmetry_bridge"]
        destination = destinations[str(bridge["destination_grasp_id"])]
        symmetry_name = str(bridge["destination_symmetry_name"])
        np.testing.assert_allclose(
            simple_sim._candidate_part_to_tcp_matrix(alias),
            np.linalg.inv(matrices[symmetry_name]) @ simple_sim._candidate_part_to_tcp_matrix(destination),
            atol=1.0e-10,
        )
        if destination is identity_destination:
            identity_aliases.append(bridge)
    assert {bridge["destination_candidate_symmetry_name"] for bridge in identity_aliases} == {"identity"}


def test_shorter_pickup_pregrasp_rebuilds_truthful_task_payload() -> None:
    pickup_grasp = WorldFrameGraspCandidate(
        grasp_id="incoming",
        position_w=(0.60, -0.20, 0.02),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        normal_w=(0.0, 0.0, -1.0),
        pregrasp_offset=0.10,
        pregrasp_position_w=(0.60, -0.20, 0.12),
        gripper_width=0.04,
        jaw_width=0.03,
        roll_angle_rad=0.0,
        contact_point_a_w=(0.60, -0.21, 0.02),
        contact_point_b_w=(0.60, -0.19, 0.02),
    )
    Task = make_dataclass("Task", [("inserter_pickup_world_grasp", object)], frozen=True)
    task = Task(inserter_pickup_world_grasp=pickup_grasp)

    adjusted = with_inserter_pickup_pregrasp_offset(task, 0.075)

    assert adjusted is not task
    assert task.inserter_pickup_world_grasp.pregrasp_offset == 0.10
    assert adjusted.inserter_pickup_world_grasp.pregrasp_offset == 0.075
    assert adjusted.inserter_pickup_world_grasp.pregrasp_position_w == (
        0.60,
        -0.20,
        0.095,
    )


def test_pickup_pregrasp_offset_must_remain_positive() -> None:
    Task = make_dataclass("InvalidOffsetTask", [("inserter_pickup_world_grasp", object)], frozen=True)
    task = Task(
        inserter_pickup_world_grasp=SimpleNamespace(
            normal_w=(0.0, 0.0, 1.0),
            position_w=(0.0, 0.0, 0.0),
        )
    )

    with pytest.raises(ValueError, match="finite and > 0"):
        with_inserter_pickup_pregrasp_offset(task, 0.0)


def test_shorter_pickup_pregrasp_is_serialized_without_changing_other_targets() -> None:
    task = load_simple_dual_robot_pair_tasks(
        artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
        retained_only=True,
    )[0]
    original_payload = task.to_payload()

    adjusted = with_inserter_pickup_pregrasp_offset(task, 0.075)
    adjusted_payload = adjusted.to_payload()

    assert adjusted_payload["grasps"]["inserter_pickup"]["pregrasp_offset_m"] == 0.075
    assert adjusted_payload["targets"]["inserter_pickup_pregrasp"]["position_world_m"] == pytest.approx(
        adjusted.inserter_pickup_world_grasp.pregrasp_position_w
    )
    assert adjusted_payload["targets"]["inserter_pickup_grasp"] == original_payload["targets"]["inserter_pickup_grasp"]
    assert adjusted_payload["targets"]["inserter_pickup_lift"] == original_payload["targets"]["inserter_pickup_lift"]
    assert adjusted_payload["targets"]["inserter_preinsertion"] == original_payload["targets"]["inserter_preinsertion"]


def test_runtime_uses_stage3_declared_holder_source_instead_of_stale_library(
    tmp_path: Path,
) -> None:
    source_candidate = {
        "grasp_id": "h0001",
        "grasp_pose_obj": {
            "position": [0.1, 0.2, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "contact_points_obj": [[0.1, 0.19, 0.3], [0.1, 0.21, 0.3]],
        "contact_normals_obj": [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        "jaw_width": 0.02,
        "roll_angle_rad": 0.0,
        "contact_patch_offset_local": [0.0, 0.0],
        "score": 0.9,
        "score_components": {"score": 0.9},
        "metadata": {},
    }
    (tmp_path / "holder_state_feasibility.json").write_text(
        json.dumps(
            {
                "source_holder_cache_key": "current-cache",
                "source_frame_pose_assembly": {
                    "position": [0.4, 0.5, 0.6],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "candidates": {"h0001": source_candidate},
            }
        ),
        encoding="utf-8",
    )
    stale_candidate = simple_sim._saved_candidate_from_payload(
        {
            **source_candidate,
            "grasp_pose_obj": {
                "position": [-9.0, -9.0, -9.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    )

    candidates, source_pose, diagnostics = simple_sim._declared_holder_candidate_source(
        root=tmp_path,
        pair_payload={
            "candidate_sources": {
                "holder": {
                    "artifact": "holder_state_feasibility.json",
                    "candidate_collection": "candidates",
                }
            }
        },
        fallback_candidates=(stale_candidate,),
        fallback_source_pose_assembly=ObjectWorldPose(
            position_world=(-9.0, -9.0, -9.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    assert candidates[0].grasp_id == "h0001"
    assert candidates[0].grasp_position_obj == (0.1, 0.2, 0.3)
    assert source_pose.position_world == (0.4, 0.5, 0.6)
    assert diagnostics == {
        "artifact": "holder_state_feasibility.json",
        "candidate_collection": "candidates",
        "legacy_fallback": False,
        "candidate_count": 1,
        "source_holder_cache_key": "current-cache",
    }


def test_selected_order_request_resolves_every_plumbers_block_step() -> None:
    expected = {
        "0": ("step_001_part_0", ("2",)),
        "3": ("step_002_part_3", ("2", "0")),
        "1": ("step_003_part_1", ("2", "0", "3")),
        "4": ("step_004_part_4", ("2", "0", "3", "1")),
    }

    for incoming_part_id, (step_id, assembled_before) in expected.items():
        selection = resolve_dual_robot_step_selection(
            assembly="plumbers_block",
            incoming_part_id=incoming_part_id,
        )
        assert selection.base_part_id == "2"
        assert selection.step_id == step_id
        assert selection.assembled_part_ids_before == assembled_before


def test_source_local_subassembly_mesh_combines_prefix_in_base_frame(
    monkeypatch,
) -> None:
    meshes = {
        "base.obj": TriangleMesh(
            vertices_obj=np.asarray(
                ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                dtype=float,
            ),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        ),
        "added.obj": TriangleMesh(
            vertices_obj=np.asarray(
                ((1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
                dtype=float,
            ),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        ),
    }
    monkeypatch.setattr(
        "grasp_planning.pipeline.dual_robot_simple_sim.load_triangle_mesh",
        lambda path, *, scale: meshes[Path(path).name],
    )

    combined = source_local_subassembly_mesh(
        {
            "source_pose_assembly": {
                "position_world_m": [1.0, 0.0, 0.0],
                "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
            },
            "parts": [
                {
                    "part_id": "2",
                    "mesh_path": "base.obj",
                    "mesh_scale": 1.0,
                },
                {
                    "part_id": "0",
                    "mesh_path": "added.obj",
                    "mesh_scale": 1.0,
                },
            ],
        }
    )

    assert np.allclose(
        combined.vertices_obj,
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
    )
    assert np.array_equal(combined.faces, ((0, 1, 2), (3, 4, 5)))


def test_compose_source_pose_world_applies_assembly_yaw_and_translation() -> None:
    source_pose = ObjectWorldPose(
        position_world=(1.0, 0.0, 0.2),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )

    result = compose_source_pose_world(
        source_pose_assembly=source_pose,
        assembly_world=MovableFrame((0.5, -0.2, 0.0), 90.0),
        translation_assembly_m=(0.0, 0.1, 0.0),
    )

    assert np.allclose(result.position_world, (0.4, 0.8, 0.2))
    expected_quaternion = (
        0.0,
        0.0,
        math.sin(math.pi / 4.0),
        math.cos(math.pi / 4.0),
    )
    assert np.allclose(
        result.orientation_xyzw_world,
        expected_quaternion,
    )


def test_translated_source_pose_preserves_the_known_object_orientation() -> None:
    source_pose = ObjectWorldPose(
        position_world=(0.0, 0.0, 0.05),
        orientation_xyzw_world=(0.0, 0.0, 0.382683, 0.92388),
    )

    result = translated_source_pose_world(
        source_pose,
        position_world=(0.55, 0.28, 0.05),
    )

    assert result.position_world == (0.55, 0.28, 0.05)
    assert result.orientation_xyzw_world == source_pose.orientation_xyzw_world


def test_runtime_layout_recovers_assembly_yaw_and_relative_pickup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    identity = (0.0, 0.0, 0.0, 1.0)
    base_bundle = SimpleNamespace(
        source_frame_origin_obj_world=(1.0, 0.0, 0.0),
        source_frame_orientation_xyzw_obj_world=identity,
        target_stl_path="base.obj",
        stl_scale=1.0,
    )
    incoming_bundle = SimpleNamespace(
        source_frame_origin_obj_world=(0.0, 0.0, 0.0),
        source_frame_orientation_xyzw_obj_world=identity,
        target_stl_path="incoming.obj",
        stl_scale=1.0,
    )

    def _bundle(path):
        return base_bundle if Path(path).name == "holder_base_candidates.json" else incoming_bundle

    monkeypatch.setattr(
        "grasp_planning.pipeline.dual_robot_simple_sim.load_grasp_bundle",
        _bundle,
    )
    debug_mesh = TriangleMesh(
        vertices_obj=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    monkeypatch.setattr(
        "grasp_planning.pipeline.dual_robot_simple_sim.resolve_mesh_path",
        Path,
    )
    monkeypatch.setattr(
        "grasp_planning.pipeline.dual_robot_simple_sim.load_triangle_mesh",
        lambda _path, *, scale: debug_mesh,
    )
    yaw_90 = (
        0.0,
        0.0,
        math.sin(math.pi / 4.0),
        math.cos(math.pi / 4.0),
    )
    yaw_30 = (
        0.0,
        0.0,
        math.sin(math.pi / 12.0),
        math.cos(math.pi / 12.0),
    )
    layout = resolve_planar_runtime_layout(
        artifact_dir=tmp_path,
        step_id="step_001_part_0",
        base_source_pose_world=ObjectWorldPose(
            position_world=(0.5, 0.8, 0.0),
            orientation_xyzw_world=yaw_90,
        ),
        incoming_source_pose_world=ObjectWorldPose(
            position_world=(0.6, 0.2, 0.04),
            orientation_xyzw_world=yaw_30,
        ),
    )

    assert np.allclose(
        layout.assembly_world.position_world_m,
        (0.5, -0.2, 0.0),
    )
    assert math.isclose(layout.assembly_world.yaw_deg, 90.0)
    assert layout.pickup_source_world_xy == (0.6, 0.2)
    assert np.allclose(
        layout.pickup_orientation_rpy_deg,
        (0.0, 0.0, -60.0),
    )
    assert tuple(bounds.role for bounds in layout.perceived_part_aabbs) == (
        "base",
        "incoming",
    )
    assert layout.warnings == ()


def test_runtime_layout_warns_and_continues_for_nonplanar_base(
    monkeypatch,
    tmp_path: Path,
) -> None:
    identity = (0.0, 0.0, 0.0, 1.0)
    bundle = SimpleNamespace(
        source_frame_origin_obj_world=(0.0, 0.0, 0.0),
        source_frame_orientation_xyzw_obj_world=identity,
        target_stl_path="part.obj",
        stl_scale=1.0,
    )
    monkeypatch.setattr(
        "grasp_planning.pipeline.dual_robot_simple_sim.load_grasp_bundle",
        lambda _path: bundle,
    )
    debug_mesh = TriangleMesh(
        vertices_obj=np.asarray(
            ((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.0, 0.1, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    monkeypatch.setattr(
        "grasp_planning.pipeline.dual_robot_simple_sim.resolve_mesh_path",
        Path,
    )
    monkeypatch.setattr(
        "grasp_planning.pipeline.dual_robot_simple_sim.load_triangle_mesh",
        lambda _path, *, scale: debug_mesh,
    )
    roll_30 = (
        math.sin(math.pi / 12.0),
        0.0,
        0.0,
        math.cos(math.pi / 12.0),
    )

    layout = resolve_planar_runtime_layout(
        artifact_dir=tmp_path,
        step_id="step_001_part_0",
        base_source_pose_world=ObjectWorldPose(
            position_world=(0.5, 0.0, -0.01),
            orientation_xyzw_world=roll_30,
        ),
        incoming_source_pose_world=ObjectWorldPose(
            position_world=(0.6, 0.2, 0.04),
            orientation_xyzw_world=identity,
        ),
        maximum_assembly_tilt_deg=5.0,
    )

    assert math.isclose(layout.assembly_world.yaw_deg, 0.0, abs_tol=1e-9)
    assert len(layout.warnings) == 1
    assert "roll=30.000 deg pitch=-0.000 deg" in layout.warnings[0]
    assert "Continuing with the yaw-only assembly layout" in layout.warnings[0]


def test_source_pose_resting_on_floor_uses_the_mesh_lowest_point() -> None:
    mesh = TriangleMesh(
        vertices_obj=np.asarray(
            (
                (-0.2, -0.1, 0.5),
                (0.2, -0.1, 1.5),
                (0.0, 0.2, 1.0),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    source_pose = ObjectWorldPose(
        position_world=(0.0, 0.0, 1.0),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )

    result = source_pose_resting_on_floor(
        mesh_assembly=mesh,
        source_pose_assembly=source_pose,
        source_orientation_world=source_pose,
        xy_world=(0.55, 0.28),
        floor_z_world_m=0.2,
    )

    assert np.allclose(result.position_world, (0.55, 0.28, 0.7))


def test_simple_dual_sim_scripts_keep_moveit_and_physics_responsibilities_separate() -> None:
    planner = (REPO_ROOT / "scripts/plan_simple_dual_robot_sim.py").read_text(encoding="utf-8")
    isaac_runner = (REPO_ROOT / "scripts/run_simple_dual_robot_sim_in_isaac.py").read_text(encoding="utf-8")
    task_builder = (REPO_ROOT / "grasp_planning/pipeline/dual_robot_simple_sim.py").read_text(encoding="utf-8")

    assert '"object_collision_geometry_in_scene": True' in planner
    assert '"work_surface_collision_geometry_in_scene": True' in planner
    assert "apply_planning_scene_obstacles" in planner
    assert "remove_planning_scene_obstacles" in planner
    assert '"pregrasp_aabb_collision_geometry"' in planner
    assert '"attached_collision_geometry"' in planner
    assert 'parser.add_argument("--velocity-scale", type=float, default=0.35)' in planner
    assert 'parser.add_argument("--acceleration-scale", type=float, default=0.35)' in planner
    assert "KUKA_MOVEIT_ARM_START_JOINT_VALUES" in planner
    assert '"planning_group": (' in task_builder
    assert 'self.holder_robot_name == "lbr_one"' in task_builder
    assert 'self.inserter_robot_name == "lbr_one"' in task_builder
    assert "make_dual_kuka_assembly_scene_cfg" in isaac_runner
    assert "ground_height_m=floor_z_world_m" in isaac_runner
    assert "incoming_part.write_root_state_to_sim" in isaac_runner
    assert '"--unloaded-max-joint-speed-rad-s"' in isaac_runner
    assert "default=1.00" in isaac_runner
    assert '"--loaded-max-joint-speed-rad-s"' in isaac_runner
    assert "default=0.70" in isaac_runner
    assert "continuous_moveit_polyline_with_velocity_feedforward" in isaac_runner
    assert "_execute_segments" in isaac_runner
    assert '"--contact-pose-tolerance-rad"' in isaac_runner
    assert '"holder_grasp", "inserter_pickup_grasp"' in isaac_runner
    assert "default=0.005" in isaac_runner
    assert "default=0.030" in isaac_runner
    assert "default=2.0" in isaac_runner
    assert '"--close-width"' in isaac_runner
    assert "min(float(selected_jaw_width_m), minimum_width_m)" in isaac_runner
    assert "clearance_per_finger_m=args_cli.approach_clearance_per_finger_m" in isaac_runner
    assert '"tcp_position_error_m"' in isaac_runner
    assert '"duration_s"' in isaac_runner
    assert "inserter_pickup_lift" in isaac_runner
    assert "inserter_preinsertion" in isaac_runner


def test_default_supported_layout_places_both_object_aabbs_on_lowered_floor() -> None:
    floor_z = -0.030
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
        step_id="step_001_part_0",
        assembly_world=MovableFrame((0.55, 0.0, floor_z), 0.0),
        pickup_floor_z_world_m=floor_z,
        retained_only=True,
    )

    obstacles = simple_dual_robot_pregrasp_aabb_obstacles(tasks[0])
    schedule = simple_dual_robot_pregrasp_aabb_schedule(obstacles)

    assert schedule["holder_pregrasp"]
    assert schedule["holder_grasp"]
    assert schedule["inserter_pickup_pregrasp"]
    assert schedule["inserter_pickup_grasp"]
    assert schedule["inserter_pickup_lift"]
    assert schedule["inserter_above_preinsertion"]
    assert schedule["inserter_preinsertion"]
    assert set(schedule["holder_pregrasp"]).isdisjoint(schedule["inserter_pickup_pregrasp"])
    for obstacle in obstacles.values():
        center_z = float(obstacle["xyz"][2])
        height = float(obstacle["size_m"][2])
        assert center_z - 0.5 * height >= floor_z - 1.0e-9

    holder_incoming = [
        obstacles[key] for key in schedule["holder_pregrasp"] if obstacles[key]["role"] == "incoming_pickup"
    ]
    assert len(holder_incoming) == 1
    assert holder_incoming[0]["source"] == "world_aabb"
    assert math.isclose(
        float(holder_incoming[0]["xyz"][2]) - 0.5 * float(holder_incoming[0]["size_m"][2]),
        floor_z,
        abs_tol=1.0e-9,
    )

    attached = simple_dual_robot_attached_collision_objects(tasks[0])["incoming"]
    assert attached["link_name"] == f"{tasks[0].inserter_robot_name}_gripper_tcp"
    assert attached["attach_after_target"] == "inserter_pickup_grasp"
    assert attached["active_targets"] == [
        "inserter_pickup_lift",
        "inserter_above_preinsertion",
        "inserter_preinsertion",
    ]
    assert len(attached["touch_links"]) == 4


def test_runtime_task_can_swap_holder_and_inserter_robot_assignments() -> None:
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
        step_id="step_001_part_0",
        pickup_source_world_xy=(0.55, -0.26),
        holder_robot_name="lbr_two",
        inserter_robot_name="lbr_one",
        holder_robot_base_world=simple_sim.DEFAULT_INSERTER_BASE_WORLD,
        inserter_robot_base_world=simple_sim.DEFAULT_HOLDER_BASE_WORLD,
        retained_only=True,
    )

    payload = tasks[0].to_payload()

    assert payload["roles"]["holder"] == {
        "robot": "lbr_two",
        "planning_group": "arm_two",
        "tcp_link": "lbr_two_gripper_tcp",
    }
    assert payload["roles"]["inserter"] == {
        "robot": "lbr_one",
        "planning_group": "arm_one",
        "tcp_link": "lbr_one_gripper_tcp",
    }
    assert payload["layout"]["holder_base_world_m"] == [0.0, 0.42, 0.0]
    assert payload["layout"]["inserter_base_world_m"] == [0.0, -0.42, 0.0]


def test_pregrasp_aabb_pieces_exclude_selected_gripper_sweeps() -> None:
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
        step_id="step_001_part_0",
        retained_only=True,
    )

    obstacles = simple_dual_robot_pregrasp_aabb_obstacles(tasks[0])

    carved_obstacles = [
        obstacle for obstacle in obstacles.values() if obstacle["source"] == "world_aabb_minus_selected_gripper_sweep"
    ]
    assert carved_obstacles
    for obstacle in carved_obstacles:
        center = np.asarray(obstacle["xyz"], dtype=float)
        half_size = 0.5 * np.asarray(obstacle["size_m"], dtype=float)
        minimum = center - half_size
        maximum = center + half_size
        carved_sweep = obstacle["carved_sweep_aabb"]
        sweep_minimum = np.asarray(
            carved_sweep["minimum_world_m"],
            dtype=float,
        )
        sweep_maximum = np.asarray(
            carved_sweep["maximum_world_m"],
            dtype=float,
        )
        overlap_minimum = np.maximum(minimum, sweep_minimum)
        overlap_maximum = np.minimum(maximum, sweep_maximum)
        assert np.any(overlap_maximum <= overlap_minimum + 1.0e-12)


def test_simple_dual_sim_filters_and_records_grounded_pickup_floor_clearance() -> None:
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
        step_id="step_001_part_0",
        retained_only=False,
    )

    assert tasks
    task = tasks[0]
    floor_check = task.to_payload()["collision_checks"]["inserter_pickup_floor"]
    counts = task.to_payload()["candidate_filter_diagnostics"]
    assert floor_check["status"] == "accepted"
    assert floor_check["gripper_collision_model"] == "kuka_y_gripper"
    assert floor_check["floor_clearance_margin_m"] == 0.001
    assert counts["pickup_grasps_checked"] == (counts["pickup_grasps_accepted"] + counts["pickup_grasps_rejected"])
    assert counts["pose_feasible_execution_candidates"] == len(tasks)
    assert counts["pose_feasible_unique_holder_grasps"] <= len(tasks)
    assert counts["pose_feasible_unique_inserter_grasps"] <= len(tasks)
    assert counts["stage3_retained_execution_candidates"] >= counts["stage3_retained_pairs"]
    assert all(task.inserter_candidate.grasp_id != "i0_2040" for task in tasks)


def test_exact_pickup_aliases_follow_unchanged_direct_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=artifact_dir,
        step_id="step_001_part_0",
        retained_only=True,
    )

    first_bridge_index = next(
        index
        for index, task in enumerate(tasks)
        if isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict)
    )
    direct_tasks = tasks[:first_bridge_index]
    bridge_tasks = tasks[first_bridge_index:]
    assert direct_tasks
    assert bridge_tasks
    assert all(not isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict) for task in direct_tasks)
    assert all(isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict) for task in bridge_tasks)
    assert {str(task.transition_symmetry["source_execution_candidate_id"]) for task in direct_tasks} & {
        str(task.transition_symmetry["source_execution_candidate_id"]) for task in bridge_tasks
    }
    assert len({task.execution_candidate_id for task in tasks}) == len(tasks)

    counts = tasks[0].candidate_filter_diagnostics
    assert counts["pickup_symmetry_bridge_status"] == "used_with_direct_candidates"
    assert counts["pose_feasible_direct_execution_candidates"] == len(direct_tasks)
    assert counts["pose_feasible_bridge_execution_candidates"] == len(bridge_tasks)
    assert counts["pose_feasible_direct_pickup_options"] > 0
    assert counts["pose_feasible_bridge_pickup_options"] > 0

    monkeypatch.setattr(
        simple_sim,
        "_exact_pickup_symmetry_validations",
        lambda **_kwargs: ({}, {"status": "disabled_for_direct_order_test"}),
    )
    direct_only_tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=artifact_dir,
        step_id="step_001_part_0",
        retained_only=True,
    )
    assert [task.execution_candidate_id for task in direct_tasks] == [
        task.execution_candidate_id for task in direct_only_tasks
    ]


@pytest.mark.parametrize(
    "step_id,incoming_part_id",
    (
        ("step_001_part_0", "0"),
        ("step_002_part_3", "3"),
        ("step_003_part_1", "1"),
        ("step_004_part_4", "4"),
    ),
)
def test_exact_pickup_aliases_are_considered_for_every_incoming_part(
    step_id: str,
    incoming_part_id: str,
) -> None:
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block",
        step_id=step_id,
        pickup_orientation_rpy_deg=(0.0, 0.0, 0.0),
        retained_only=True,
    )

    assert tasks
    assert {task.incoming_part_id for task in tasks} == {incoming_part_id}
    first_bridge_index = next(
        index
        for index, task in enumerate(tasks)
        if isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict)
    )
    assert first_bridge_index > 0
    assert all(
        not isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict)
        for task in tasks[:first_bridge_index]
    )
    assert all(
        isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict) for task in tasks[first_bridge_index:]
    )

    counts = tasks[0].candidate_filter_diagnostics
    source_resolution = counts["pickup_symmetry_source_resolution"]
    alias_generation = counts["pickup_symmetry_alias_generation"]
    assert source_resolution["source_count"] > 0
    assert source_resolution["matrix_source"] == ("stage3_transition_artifact_scaled_source_frame")
    assert alias_generation["symmetry_source_count"] == source_resolution["source_count"]
    assert counts["pickup_symmetry_aliases_checked"] == alias_generation["alias_count"]
    assert counts["pickup_symmetry_aliases_accepted"] > 0
    assert counts["pose_feasible_direct_execution_candidates"] > 0
    assert counts["pose_feasible_bridge_execution_candidates"] > 0
    assert counts["pickup_symmetry_bridge_status"] == "used_with_direct_candidates"


def test_part0_roll_pickup_uses_exact_symmetry_bridge_and_preserves_stage3_tcp() -> None:
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
        step_id="step_001_part_0",
        pickup_orientation_rpy_deg=(90.0, 0.0, 0.0),
        retained_only=True,
    )

    assert tasks
    counts = tasks[0].candidate_filter_diagnostics
    assert counts["pickup_grasps_accepted"] == 0
    assert counts["pickup_symmetry_bridge_status"] == "used"
    assert counts["pickup_symmetry_aliases_accepted"] > 0
    assert counts["pickup_symmetry_alias_destination_grasps"] > 0
    assert not any(
        name.startswith("face_normal_") for name in counts["pickup_symmetry_bridge_validation"]["accepted_names"]
    )
    assert len({task.execution_candidate_id for task in tasks}) == len(tasks)

    task = tasks[0]
    payload = task.to_payload()
    bridge = task.transition_symmetry["pickup_symmetry_bridge"]
    assert str(bridge["destination_symmetry_name"]).startswith("object_")
    assert bridge["asset_validation"]["vertex_max_m"] <= 1.0e-6
    assert bridge["tcp_invariance_max_abs_error"] <= 1.0e-8
    assert "__pickup_bridge_" in task.execution_candidate_id
    assert (
        payload["grasps"]["inserter_pickup"]["part_to_tcp"] == payload["grasps"]["inserter_preinsertion"]["part_to_tcp"]
    )
    assert (
        bridge["nominal_final_source_pose_assembly"]["matrix_assembly_m"]
        != task.transition_symmetry["final_source_pose_assembly"]["matrix_assembly_m"]
    )


def test_empty_pickup_floor_queue_preserves_filter_diagnostics() -> None:
    try:
        load_simple_dual_robot_pair_tasks(
            artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
            step_id="step_001_part_0",
            retained_only=False,
            pickup_floor_clearance_margin_m=10.0,
        )
    except NoPoseFeasibleDualTasksError as exc:
        counts = exc.candidate_filter_diagnostics
    else:
        raise AssertionError("Expected an impossible floor margin to empty the runtime queue.")

    assert counts["pickup_floor_z_world_m"] == -0.030
    assert counts["pickup_floor_clearance_margin_m"] == 10.0
    assert counts["pickup_grasps_checked"] > 0
    assert counts["pickup_grasps_accepted"] == 0
    assert counts["pickup_grasps_rejected"] == counts["pickup_grasps_checked"]
    assert counts["pose_feasible_execution_candidates"] == 0
    assert counts["pickup_grasp_rejection_counts"]


def test_runtime_queue_uses_strict_clear_phase_then_only_validated_fallbacks() -> None:
    artifact_dir = REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"
    retained = load_simple_dual_robot_pair_tasks(
        artifact_dir=artifact_dir,
        step_id="step_001_part_0",
        retained_only=True,
    )
    expanded = load_simple_dual_robot_pair_tasks(
        artifact_dir=artifact_dir,
        step_id="step_001_part_0",
        retained_only=False,
        include_nonretained_identity_fallbacks=True,
    )

    retained_execution_ids = {task.execution_candidate_id for task in retained}
    assert len(expanded) > len(retained)
    first_bridge_index = next(
        (
            index
            for index, task in enumerate(expanded)
            if isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict)
        ),
        len(expanded),
    )
    assert all(
        not isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict)
        for task in expanded[:first_bridge_index]
    )
    assert all(
        isinstance(task.transition_symmetry.get("pickup_symmetry_bridge"), dict)
        for task in expanded[first_bridge_index:]
    )
    for ranked_partition in (
        expanded[:first_bridge_index],
        expanded[first_bridge_index:],
    ):
        crossing_flags = [
            bool(task.layout_proxy_components["transition_segments_cross_xy"]) for task in ranked_partition
        ]
        assert crossing_flags == sorted(crossing_flags)
    fallback = [task for task in expanded if task.execution_candidate_id not in retained_execution_ids]
    transformed_fallback = [task for task in fallback if not bool(task.transition_symmetry.get("is_identity"))]
    assert transformed_fallback
    assert all(
        dict(task.transition_symmetry.get("pair_collision_validation", {})).get("status") == "accepted"
        for task in transformed_fallback
    )
    counts = expanded[0].candidate_filter_diagnostics
    assert counts["pose_feasible_retained_execution_candidates"] == len(retained)
    assert counts["pose_feasible_identity_fallback_candidates"] + counts[
        "pose_feasible_validated_transition_fallback_candidates"
    ] == (len(expanded) - len(retained))
    assert counts["pose_feasible_validated_transition_fallback_candidates"] == len(transformed_fallback)
    assert [task.candidate_rank for task in expanded] == list(range(1, len(expanded) + 1))
    assert [task.to_payload()["candidate_rank"] for task in expanded] == list(range(1, len(expanded) + 1))


def test_later_step_task_contains_the_offline_checked_assembled_prefix() -> None:
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
        step_id="step_003_part_1",
        retained_only=True,
    )

    assert tasks
    payload = tasks[0].to_payload()
    subassembly = payload["objects"]["subassembly"]
    assert payload["schema_version"] == 3
    assert payload["transition_id"] == "tr_identity__part_identity"
    assert payload["execution_candidate_id"].endswith("__tr_identity__part_identity")
    assert (
        payload["grasps"]["inserter_pickup"]["part_to_tcp"]
        == (payload["grasps"]["inserter_preinsertion"]["part_to_tcp"])
    )
    assert subassembly["base_part_id"] == "2"
    assert subassembly["part_ids"] == ["2", "0", "3"]
    assert [part["part_id"] for part in subassembly["parts"]] == ["2", "0", "3"]
    assert subassembly["physics"] == "single_rigid_compound"
    assert payload["collision_checks"]["offline_dual_pair"]["assembled_part_ids_before"] == ["2", "0", "3"]


def test_task_expansion_uses_only_pair_compatible_transitions(
    monkeypatch,
) -> None:
    artifact_dir = REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"
    step_id = "step_001_part_0"
    baseline = load_simple_dual_robot_pair_tasks(
        artifact_dir=artifact_dir,
        step_id=step_id,
        retained_only=True,
    )
    assert baseline
    baseline_pair_ids = {task.pair_id for task in baseline}
    source_pose = baseline[0].incoming_source_pose_assembly
    canonical = simple_sim._legacy_transition_payload(
        source_pose_assembly=source_pose,
        final_to_pre_translation_assembly_m=(baseline[0].final_to_preinsertion_translation_assembly_m),
    )
    accepted = copy.deepcopy(canonical)
    accepted["transition_id"] = "tr_test_accepted"
    accepted["is_identity"] = False
    accepted["preinsertion_source_pose_assembly"]["matrix_assembly_m"][0][3] += 0.02
    accepted["preinsertion_source_pose_assembly"]["position_assembly_m"][0] += 0.02
    rejected = copy.deepcopy(canonical)
    rejected["transition_id"] = "tr_test_rejected"
    rejected["is_identity"] = False

    pair_path = artifact_dir / f"dual_grasp_pairs_{step_id}.json"
    pair_payload = json.loads(pair_path.read_text(encoding="utf-8"))
    pair_payload["transition_symmetry"] = {
        "enabled": True,
        "candidates": [canonical, accepted, rejected],
    }
    for evaluation in pair_payload["evaluations"]:
        if evaluation["status"] != "accepted":
            continue
        details = dict(evaluation.get("details", {}))
        details["compatible_transition_ids"] = [
            canonical["transition_id"],
            accepted["transition_id"],
        ]
        details["transition_validation"] = {
            canonical["transition_id"]: {"status": "accepted"},
            accepted["transition_id"]: {
                "status": "accepted",
                "gripper_sweep_checked": True,
            },
            rejected["transition_id"]: {"status": "rejected"},
        }
        evaluation["details"] = details
    pair_payload["retained_execution_candidate_ids"] = [
        f"{pair_id}__{transition_id}"
        for pair_id in sorted(baseline_pair_ids)
        for transition_id in (
            canonical["transition_id"],
            accepted["transition_id"],
        )
    ]

    original_read_json = simple_sim._read_json

    def read_json(path: Path):
        if Path(path).resolve() == pair_path.resolve():
            return pair_payload
        return original_read_json(path)

    monkeypatch.setattr(simple_sim, "_read_json", read_json)
    expanded = load_simple_dual_robot_pair_tasks(
        artifact_dir=artifact_dir,
        step_id=step_id,
        retained_only=True,
    )

    assert len(expanded) == 2 * len(baseline_pair_ids)
    assert {task.transition_id for task in expanded} == {
        canonical["transition_id"],
        "tr_test_accepted",
    }
    assert all(task.transition_id != "tr_test_rejected" for task in expanded)
    accepted_task = next(task for task in expanded if task.transition_id == "tr_test_accepted")
    payload = accepted_task.to_payload()
    assert payload["collision_checks"]["selected_transition"]["gripper_sweep_checked"] is True
    assert (
        payload["grasps"]["inserter_pickup"]["part_to_tcp"]
        == (payload["grasps"]["inserter_preinsertion"]["part_to_tcp"])
    )


def test_current_lowered_table_pickup_pose_keeps_feasible_pairs() -> None:
    tasks = load_simple_dual_robot_pair_tasks(
        artifact_dir=(REPO_ROOT / "artifacts/dual_grasp_planning/plumbers_block"),
        step_id="step_001_part_0",
        pickup_source_world_xy=(0.39187130331993103, 0.028867240995168686),
        pickup_orientation_rpy_deg=(
            -77.0324484408073,
            -75.24313954664626,
            175.26921099844273,
        ),
        pickup_floor_z_world_m=-0.030,
        retained_only=False,
    )

    assert tasks
    assert len({task.inserter_candidate.grasp_id for task in tasks}) > 1
    assert all(task.to_payload()["collision_checks"]["inserter_pickup_floor"]["status"] == "accepted" for task in tasks)
