from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from grasp_planning.grasping.collision import (
    MeshCollisionPrimitive,
    trimesh_fcl_backend_available,
)
from grasp_planning.grasping.fabrica_grasp_debug import (
    SavedGraspBundle,
    SavedGraspCandidate,
    load_grasp_bundle,
)
from grasp_planning.grasping.mesh_antipodal_grasp_generator import TriangleMesh
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline import (
    AssemblyPartSpec,
    AssemblySequence,
    AssemblySequenceStep,
    DualGraspPairConfig,
    HolderCandidateFeasibility,
    HolderFeasibilityConfig,
    HolderStateFeasibility,
    HolderStateFeasibilityResult,
    InserterCandidateStatus,
    InserterGraspLibrary,
    PlanningConfig,
    RetainedExecutionCandidate,
    Stage1Result,
    generate_inserter_grasp_library,
    plan_dual_grasp_pairs,
    write_dual_grasp_pair_step_html,
    write_dual_grasp_pair_step_json,
    write_holder_state_feasibility_json,
    write_inserter_grasp_library,
)
from grasp_planning.pipeline import dual_grasp_pair_planner as pair_planner


def _candidate(
    grasp_id: str,
    position: tuple[float, float, float],
    *,
    score: float,
) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=position,
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(position[0], position[1] - 0.005, position[2]),
        contact_point_b_obj=(position[0], position[1] + 0.005, position[2]),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.01,
        roll_angle_rad=0.0,
        score=score,
        score_components={"contact_support": score},
    )


def _step() -> AssemblySequenceStep:
    return AssemblySequenceStep(
        step_id="step_001_part_0",
        step_index=1,
        incoming_part_id="0",
        incoming_part_role="moving_part",
        assembled_part_ids_before=("2",),
        assembled_part_ids_after=("2", "0"),
        base_part_status="available",
        holder_base_available=True,
        final_to_pre_insertion_transform_m=(
            (1.0, 0.0, 0.0, -0.4),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        final_to_pre_insertion_translation_m=(-0.4, 0.0, 0.0),
        pre_to_final_insertion_vector_m=(0.4, 0.0, 0.0),
        insertion_distance_m=0.4,
        disassembly_path_waypoints=None,
    )


def _sequence(*, parts: tuple[AssemblyPartSpec, ...] = ()) -> AssemblySequence:
    return AssemblySequence(
        assembly="pair_fixture",
        base_part_id="2",
        base_part_source="selected_order[0]",
        base_part_order_index=0,
        first_holder_step_index=1,
        selected_order=("2", "0"),
        mesh_scale=1.0,
        table_z_assembly_m=0.0,
        table_contact_tolerance_m=1.0e-6,
        table_contact_part_ids=("2",),
        parts=parts,
        steps=(_step(),),
        precedence_plan_record={},
        pre_insertion_poses_record={},
        warnings=(),
        source_assembly_dir=Path("."),
    )


def _holder_result(
    holders: tuple[SavedGraspCandidate, ...],
    *,
    rejected_ids: tuple[str, ...] = (),
) -> HolderStateFeasibilityResult:
    candidate_results = tuple(
        HolderCandidateFeasibility(
            grasp_id=candidate.grasp_id,
            status=("rejected" if candidate.grasp_id in rejected_ids else "accepted"),
            reason=("table_collision" if candidate.grasp_id in rejected_ids else "accepted"),
            minimum_clearance_m=0.1,
        )
        for candidate in holders
    )
    counts = dict(
        sorted(
            {
                reason: sum(result.reason == reason for result in candidate_results)
                for reason in {result.reason for result in candidate_results}
            }.items()
        )
    )
    return HolderStateFeasibilityResult(
        assembly="pair_fixture",
        base_part_id="2",
        base_part_source="selected_order[0]",
        selected_order=("2", "0"),
        table_z_assembly_m=0.0,
        source_frame_pose_assembly=ObjectWorldPose(
            position_world=(0.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        ),
        config=HolderFeasibilityConfig(),
        candidates=holders,
        states=(
            HolderStateFeasibility(
                step_id="step_001_part_0",
                step_index=1,
                incoming_part_id="0",
                holder_base_available=True,
                assembled_part_ids_before=("2",),
                static_obstacle_part_ids=(),
                incoming_final_to_pre_translation_m=(-0.4, 0.0, 0.0),
                candidate_results=candidate_results,
                reason_counts=counts,
            ),
        ),
        collision_backend_name="fixture",
        source_holder_cache_key="holder-cache",
    )


def _inserter_library(
    candidates: tuple[SavedGraspCandidate, ...],
    *,
    rejected: tuple[SavedGraspCandidate, ...] = (),
) -> InserterGraspLibrary:
    statuses = tuple(
        InserterCandidateStatus(
            candidate=candidate,
            status="accepted",
            reason="accepted",
            minimum_clearance_m=0.1,
        )
        for candidate in candidates
    ) + tuple(
        InserterCandidateStatus(
            candidate=candidate,
            status="rejected",
            reason="inserter_table_collision",
            minimum_clearance_m=-0.01,
        )
        for candidate in rejected
    )
    pose = ObjectWorldPose(
        position_world=(0.0, 0.0, 0.0),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )
    return InserterGraspLibrary(
        step_id="step_001_part_0",
        step_index=1,
        incoming_part_id="0",
        bundle=SavedGraspBundle(
            target_mesh_path="pair_fixture/0.obj",
            mesh_scale=1.0,
            source_frame_origin_obj_world=pose.position_world,
            source_frame_orientation_xyzw_obj_world=pose.orientation_xyzw_world,
            candidates=candidates,
            metadata={"stage1_cache_key": "inserter-cache"},
        ),
        source_frame_pose_assembly=pose,
        candidate_statuses=statuses,
        raw_candidate_count=len(statuses),
        assembly_insertion_feasible_count=len(statuses),
        collision_backend_name="fixture",
        source_stage1_cache_key="inserter-cache",
        retreat_translation_assembly_m=(-0.05, 0.0, 0.0),
    )


class _SmallCubeGripper:
    _vertices = (
        np.asarray(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        * 0.01
    )
    _faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )

    def primitives_for_grasp(
        self,
        *,
        grasp_rotmat,
        contact_point_a,
        contact_point_b,
        grasp_center=None,
    ):
        del grasp_rotmat, contact_point_a, contact_point_b
        center = np.asarray(grasp_center, dtype=float)
        return (
            MeshCollisionPrimitive(
                name="small_cube",
                vertices_obj=self._vertices + center[None, :],
                faces=self._faces,
            ),
        )


def _planning() -> PlanningConfig:
    return PlanningConfig(
        gripper_collision_model="kuka_y_gripper",
        stage1_cache_enabled=False,
    )


def _config(**overrides) -> DualGraspPairConfig:
    defaults = {
        "max_holder_candidates_per_step": 4,
        "max_inserter_candidates_per_step": 4,
        "max_candidates_per_cluster": 4,
        "contact_position_bin_m": 0.01,
        "max_pair_checks": 16,
        "max_accepted_pairs": 4,
        "max_pairs_per_holder": 4,
        "max_pairs_per_inserter": 4,
        "matrix_unary_rejections_per_side": 2,
        "retreat_distance_m": 0.05,
        "path_samples": 21,
    }
    defaults.update(overrides)
    return DualGraspPairConfig(**defaults)


class InserterShortlistDiversityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_pose = ObjectWorldPose(
            position_world=(0.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )

    def _oriented_candidate(
        self,
        grasp_id: str,
        *,
        orientation_xyzw: tuple[float, float, float, float],
        score: float,
        x: float,
        symmetry: str = "identity",
    ) -> SavedGraspCandidate:
        return replace(
            _candidate(grasp_id, (x, 0.0, 0.2), score=score),
            grasp_orientation_xyzw_obj=orientation_xyzw,
            metadata={"symmetry_pickup_name": symmetry},
        )

    def test_shortlist_covers_every_available_signed_approach_axis(self) -> None:
        root_half = float(np.sqrt(0.5))
        candidates = tuple(
            self._oriented_candidate(
                f"z_high_{index}",
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                score=1.0 - index * 0.001,
                x=index * 0.02,
            )
            for index in range(12)
        ) + (
            self._oriented_candidate(
                "z_minus",
                orientation_xyzw=(1.0, 0.0, 0.0, 0.0),
                score=0.60,
                x=0.30,
            ),
            self._oriented_candidate(
                "x_plus",
                orientation_xyzw=(0.0, root_half, 0.0, root_half),
                score=0.59,
                x=0.32,
            ),
            self._oriented_candidate(
                "x_minus",
                orientation_xyzw=(0.0, -root_half, 0.0, root_half),
                score=0.58,
                x=0.34,
            ),
            self._oriented_candidate(
                "y_plus",
                orientation_xyzw=(-root_half, 0.0, 0.0, root_half),
                score=0.57,
                x=0.36,
            ),
            self._oriented_candidate(
                "y_minus",
                orientation_xyzw=(root_half, 0.0, 0.0, root_half),
                score=0.56,
                x=0.38,
            ),
        )

        selected = pair_planner._diverse_shortlist(
            candidates,
            source_pose_assembly=self.source_pose,
            config=_config(max_candidates_per_cluster=5),
            limit=6,
            balance_approach_directions=True,
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(
            {
                pair_planner._approach_direction_label(
                    pair_planner._candidate_approach_direction_key(
                        candidate,
                        source_pose_assembly=self.source_pose,
                    )
                )
                for candidate in selected
            },
            {"+x", "-x", "+y", "-y", "+z", "-z"},
        )

    def test_shortlist_round_robins_symmetries_inside_direction(self) -> None:
        candidates = tuple(
            self._oriented_candidate(
                f"identity_{index}",
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                score=1.0 - index * 0.01,
                x=index * 0.02,
            )
            for index in range(3)
        ) + tuple(
            self._oriented_candidate(
                f"symmetric_{index}",
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                score=0.60 - index * 0.01,
                x=0.10 + index * 0.02,
                symmetry="object_half_turn",
            )
            for index in range(3)
        )

        selected = pair_planner._diverse_shortlist(
            candidates,
            source_pose_assembly=self.source_pose,
            config=_config(max_candidates_per_cluster=5),
            limit=4,
            balance_approach_directions=True,
            balance_symmetry_transforms=True,
        )

        self.assertEqual(
            [pair_planner._candidate_symmetry_key(candidate) for candidate in selected],
            ["identity", "object_half_turn", "identity", "object_half_turn"],
        )

    def test_pair_budget_covers_inserters_before_repeating_one(self) -> None:
        holders = tuple(
            _candidate(f"h{index}", (index * 0.1, 0.0, 0.2), score=1.0 - index * 0.1)
            for index in range(3)
        )
        inserters = tuple(
            _candidate(f"i{index}", (index * 0.1, 0.1, 0.2), score=1.0 - index * 0.1)
            for index in range(4)
        )

        combinations = pair_planner._balanced_pair_combinations(
            holders,
            inserters,
            config=_config(max_pair_checks=6),
        )

        self.assertEqual(
            [(holder.grasp_id, inserter.grasp_id) for holder, inserter in combinations],
            [
                ("h0", "i0"),
                ("h0", "i1"),
                ("h0", "i2"),
                ("h0", "i3"),
                ("h1", "i0"),
                ("h1", "i1"),
            ],
        )


@unittest.skipUnless(trimesh_fcl_backend_available(), "python-fcl is unavailable")
class DualGraspPairPlannerTests(unittest.TestCase):
    def _plan(self, *, config: DualGraspPairConfig | None = None):
        holder = _candidate("h1", (0.0, 0.0, 0.2), score=0.9)
        holder_rejected = _candidate("h_rejected", (0.0, 0.3, 0.2), score=0.8)
        crossing = _candidate("i0_cross", (0.2, 0.0, 0.2), score=0.95)
        clear = _candidate("i0_clear", (0.2, 0.1, 0.2), score=0.85)
        inserter_rejected = _candidate("i0_table", (0.2, 0.2, 0.005), score=0.7)
        with mock.patch(
            "grasp_planning.pipeline.dual_grasp_pair_planner.make_gripper_collision_model",
            return_value=_SmallCubeGripper(),
        ):
            result = plan_dual_grasp_pairs(
                sequence=_sequence(),
                holder_feasibility=_holder_result(
                    (holder, holder_rejected),
                    rejected_ids=("h_rejected",),
                ),
                inserter_libraries=(
                    _inserter_library(
                        (crossing, clear),
                        rejected=(inserter_rejected,),
                    ),
                ),
                planning=_planning(),
                config=config or _config(),
            )
        return result

    def test_mid_sweep_collision_and_broadphase_clear_pair(self) -> None:
        result = self._plan()
        step = result.steps[0]
        by_inserter = {evaluation.inserter_grasp_id: evaluation for evaluation in step.evaluations}
        collision = by_inserter["i0_cross"]
        self.assertEqual(collision.reason, "end_effector_sweep_collision")
        self.assertEqual(collision.collision_check, "exact_fcl")
        self.assertEqual(
            collision.details["first_failing_phase"],
            "insertion",
        )
        progress = collision.details["first_failing_progress"]
        self.assertGreater(progress, 0.0)
        self.assertLess(progress, 1.0)

        clear = by_inserter["i0_clear"]
        self.assertEqual(clear.status, "accepted")
        self.assertEqual(clear.collision_check, "aabb_separation_proof")
        self.assertEqual(step.retained_pair_ids, (clear.pair_id,))
        self.assertIn("h_rejected", step.matrix_holder_ids)
        self.assertIn("i0_table", step.matrix_inserter_ids)

    def test_retained_pair_records_nonidentity_transition_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {part_id: root / f"{part_id}.obj" for part_id in ("2", "0")}
            for path, z in ((paths["2"], 5.0), (paths["0"], 0.0)):
                path.write_text(
                    "\n".join(
                        (
                            f"v -0.1 -0.1 {z}",
                            f"v 0.1 -0.1 {z}",
                            f"v 0.1 0.1 {z}",
                            f"v -0.1 0.1 {z}",
                            "f 1 2 3",
                            "f 1 3 4",
                            "",
                        )
                    ),
                    encoding="utf-8",
                )
            parts = tuple(
                AssemblyPartSpec(
                    part_id=part_id,
                    mesh_path=str(path),
                    role="base" if part_id == "2" else "moving_part",
                    bounds_min_assembly_m=(-0.1, -0.1, z),
                    bounds_max_assembly_m=(0.1, 0.1, z),
                    table_clearance_m=z,
                    touches_table=False,
                    vertex_count=4,
                    face_count=2,
                    asset_record={},
                    resolved_mesh_path=path,
                )
                for (part_id, path), z in zip(
                    paths.items(),
                    (5.0, 0.0),
                )
            )
            identity = np.eye(4)
            half_turn = np.diag((-1.0, -1.0, 1.0, 1.0))
            symmetry_records = [
                {
                    "name": name,
                    "description": name,
                    "source": "test",
                    "angle_deg": angle,
                    "matrix_obj": matrix.tolist(),
                }
                for name, angle, matrix in (
                    ("identity", 0.0, identity),
                    ("z180", 180.0, half_turn),
                )
            ]
            (root / "symmetries.json").write_text(
                json.dumps(
                    {
                        "assembly": "pair_fixture",
                        "mesh_scale": 1.0,
                        "parts": {
                            "2": {"symmetries": symmetry_records},
                            "0": {"symmetries": symmetry_records},
                        },
                    }
                ),
                encoding="utf-8",
            )
            sequence = _sequence(parts=parts)
            sequence = AssemblySequence(
                **{
                    **sequence.__dict__,
                    "source_assembly_dir": root,
                }
            )
            holder = _candidate("h1", (0.2, -0.1, 0.2), score=0.9)
            inserter = _candidate(
                "i0_clear",
                (0.0, 0.1, 0.2),
                score=0.85,
            )
            nonretained_inserter = _candidate(
                "i0_nonretained",
                (0.0, 0.3, 0.2),
                score=0.75,
            )
            with mock.patch(
                "grasp_planning.pipeline.dual_grasp_pair_planner.make_gripper_collision_model",
                return_value=_SmallCubeGripper(),
            ):
                result = plan_dual_grasp_pairs(
                    sequence=sequence,
                    holder_feasibility=_holder_result((holder,)),
                    inserter_libraries=(_inserter_library((inserter, nonretained_inserter)),),
                    planning=_planning(),
                    config=_config(
                        transition_symmetry_enabled=True,
                        max_accepted_pairs=1,
                    ),
                )

        step = result.steps[0]
        evaluation = step.retained_pairs[0]
        self.assertEqual(len(step.transition_candidates), 2)
        self.assertEqual(
            set(evaluation.details["compatible_transition_ids"]),
            {"tr_identity__part_identity"},
        )
        alternate_validation = evaluation.details["transition_validation"]["tr_identity__part_z180"]
        self.assertEqual(alternate_validation["status"], "rejected")
        self.assertEqual(
            alternate_validation["reason"],
            "end_effector_sweep_collision",
        )
        self.assertTrue(alternate_validation["gripper_sweep_checked"])
        self.assertTrue(alternate_validation["holder_gripper_checked"])
        nonretained = next(item for item in step.evaluations if item.inserter_grasp_id == "i0_nonretained")
        self.assertEqual(nonretained.status, "accepted")
        self.assertEqual(
            nonretained.details["compatible_transition_ids"],
            ["tr_identity__part_identity"],
        )
        self.assertIn(
            "identity-only",
            nonretained.details["transition_validation_policy"],
        )
        self.assertEqual(len(step.retained_execution_candidates), 1)
        self.assertEqual(
            step.retained_execution_candidates[0].execution_candidate_id,
            f"{evaluation.pair_id}__tr_identity__part_identity",
        )
        self.assertNotIn(
            nonretained.pair_id,
            {candidate.pair_id for candidate in step.retained_execution_candidates},
        )

    def test_pair_retention_covers_inserter_grasps_before_second_pairs(self) -> None:
        def evaluation(
            pair_id: str,
            holder_id: str,
            inserter_id: str,
            score: float,
        ) -> pair_planner.DualGraspPairEvaluation:
            return pair_planner.DualGraspPairEvaluation(
                pair_id=pair_id,
                holder_grasp_id=holder_id,
                inserter_grasp_id=inserter_id,
                status="accepted",
                reason="accepted",
                score=score,
                holder_score=score,
                inserter_score=score,
                clearance_score=1.0,
                minimum_clearance_m=0.01,
                collision_check="synthetic",
            )

        retained = pair_planner._retain_diverse_pairs(
            (
                evaluation("i1_best", "h1", "i1", 0.99),
                evaluation("i1_second", "h2", "i1", 0.98),
                evaluation("i1_third", "h3", "i1", 0.97),
                evaluation("i2_best", "h4", "i2", 0.80),
                evaluation("i3_best", "h5", "i3", 0.70),
            ),
            config=_config(
                max_accepted_pairs=3,
                max_pairs_per_holder=4,
                max_pairs_per_inserter=4,
            ),
        )

        self.assertEqual(retained, ("i1_best", "i2_best", "i3_best"))

    def test_pair_retention_balances_inserter_approach_directions(self) -> None:
        def evaluation(
            pair_id: str,
            inserter_id: str,
            score: float,
        ) -> pair_planner.DualGraspPairEvaluation:
            return pair_planner.DualGraspPairEvaluation(
                pair_id=pair_id,
                holder_grasp_id=f"h_{pair_id}",
                inserter_grasp_id=inserter_id,
                status="accepted",
                reason="accepted",
                score=score,
                holder_score=score,
                inserter_score=score,
                clearance_score=1.0,
                minimum_clearance_m=0.01,
                collision_check="synthetic",
            )

        root_half = float(np.sqrt(0.5))
        inserters = {
            "z1": _candidate("z1", (0.0, 0.0, 0.2), score=0.99),
            "z2": _candidate("z2", (0.1, 0.0, 0.2), score=0.98),
            "z3": _candidate("z3", (0.2, 0.0, 0.2), score=0.97),
            "x1": replace(
                _candidate("x1", (0.3, 0.0, 0.2), score=0.70),
                grasp_orientation_xyzw_obj=(0.0, root_half, 0.0, root_half),
            ),
            "y1": replace(
                _candidate("y1", (0.4, 0.0, 0.2), score=0.60),
                grasp_orientation_xyzw_obj=(-root_half, 0.0, 0.0, root_half),
            ),
        }
        source_pose = ObjectWorldPose(
            position_world=(0.0, 0.0, 0.0),
            orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
        )

        retained = pair_planner._retain_diverse_pairs(
            (
                evaluation("pair_z1", "z1", 0.99),
                evaluation("pair_z2", "z2", 0.98),
                evaluation("pair_z3", "z3", 0.97),
                evaluation("pair_x1", "x1", 0.70),
                evaluation("pair_y1", "y1", 0.60),
            ),
            config=_config(max_accepted_pairs=3),
            inserters_by_id=inserters,
            inserter_source_pose_assembly=source_pose,
        )

        self.assertEqual(retained, ("pair_z1", "pair_x1", "pair_y1"))

    def test_execution_candidate_retention_round_robins_corridors(self) -> None:
        def candidate(
            pair_id: str,
            transition_id: str,
            score: float,
            corridor: str,
            direction: tuple[float, float, float],
        ) -> RetainedExecutionCandidate:
            return RetainedExecutionCandidate(
                execution_candidate_id=f"{pair_id}__{transition_id}",
                pair_id=pair_id,
                transition_id=transition_id,
                holder_grasp_id=f"h_{pair_id}",
                inserter_grasp_id=f"i_{pair_id}",
                pair_score=score,
                corridor_key=corridor,
                corridor_direction_assembly=direction,
                is_identity=False,
                minimum_clearance_m=0.01,
            )

        retained = pair_planner._retain_diverse_execution_candidates(
            (
                candidate("left_1", "tr_left", 1.0, "left", (0.0, -1.0, 0.0)),
                candidate("left_2", "tr_left", 0.9, "left", (0.0, -1.0, 0.0)),
                candidate("left_3", "tr_left", 0.8, "left", (0.0, -1.0, 0.0)),
                candidate("right_1", "tr_right", 0.7, "right", (0.0, 1.0, 0.0)),
                candidate("right_2", "tr_right", 0.6, "right", (0.0, 1.0, 0.0)),
            ),
            limit=4,
        )

        self.assertEqual(
            [candidate.execution_candidate_id for candidate in retained],
            [
                "left_1__tr_left",
                "right_1__tr_right",
                "left_2__tr_left",
                "right_2__tr_right",
            ],
        )

    def test_pair_clearance_margin_rejects_geometrically_clear_pair(self) -> None:
        result = self._plan(config=_config(geometry_clearance_margin_m=0.09))
        clear = next(
            evaluation for evaluation in result.steps[0].evaluations if evaluation.inserter_grasp_id == "i0_clear"
        )
        self.assertEqual(clear.status, "rejected")
        self.assertEqual(clear.reason, "pair_clearance_margin_failed")

    def test_pair_limit_and_order_are_deterministic(self) -> None:
        first = self._plan(config=_config(max_pair_checks=1))
        second = self._plan(config=_config(max_pair_checks=1))
        self.assertEqual(first.steps[0].evaluations, second.steps[0].evaluations)
        self.assertEqual(len(first.steps[0].evaluations), 1)
        self.assertTrue(first.steps[0].metadata["pair_check_limit_reached"])

    def test_artifact_references_resolve_and_html_has_matrix_controls(self) -> None:
        result = self._plan()
        step = result.steps[0]
        library = result.inserter_libraries[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            holder_path = root / "holder_state_feasibility.json"
            inserter_path = root / "inserter.json"
            pair_path = root / "pairs.json"
            html_path = root / "pairs.html"
            write_holder_state_feasibility_json(
                result.holder_feasibility,
                holder_path,
            )
            write_inserter_grasp_library(library, inserter_path)
            write_dual_grasp_pair_step_json(
                step,
                pair_path,
                holder_source_artifact=holder_path.name,
                inserter_source_artifact=inserter_path.name,
            )
            fake_visual = {
                "visualization": {
                    "scene_bounds_assembly_m": {
                        "center": [0.0, 0.0, 0.2],
                        "extent": 1.0,
                    },
                    "table_vertices_assembly_m": [
                        [-1.0, -1.0, 0.0],
                        [1.0, -1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [-1.0, 1.0, 0.0],
                    ],
                    "parts": {
                        "2": {
                            "vertices_assembly_m": [
                                [0.0, 0.0, 0.0],
                                [0.1, 0.0, 0.0],
                                [0.0, 0.1, 0.0],
                            ],
                            "faces": [[0, 1, 2]],
                            "edges": [[0, 1], [1, 2], [0, 2]],
                        },
                        "0": {
                            "vertices_assembly_m": [
                                [0.0, 0.0, 0.2],
                                [0.1, 0.0, 0.2],
                                [0.0, 0.1, 0.2],
                            ],
                            "faces": [[0, 1, 2]],
                            "edges": [[0, 1], [1, 2], [0, 2]],
                        },
                    },
                }
            }
            with mock.patch(
                "grasp_planning.pipeline.dual_grasp_pair_debug_html.assembly_sequence_visual_payload",
                return_value=fake_visual,
            ):
                write_dual_grasp_pair_step_html(
                    result,
                    step,
                    _sequence(),
                    html_path,
                )

            pair_payload = json.loads(pair_path.read_text())
            holder_payload = json.loads(holder_path.read_text())
            inserter_bundle = load_grasp_bundle(inserter_path)
            holder_ids = set(holder_payload["candidates"])
            inserter_ids = {candidate.grasp_id for candidate in inserter_bundle.candidates}
            for pair in pair_payload["retained_pairs"]:
                self.assertIn(pair["holder_grasp_id"], holder_ids)
                self.assertIn(pair["inserter_grasp_id"], inserter_ids)
            html = html_path.read_text()
            self.assertIn("Holder × inserter compatibility matrix", html)
            self.assertIn("not checked by limit", html)
            self.assertIn("end_effector_sweep_collision", html)

    def test_wrong_gripper_model_is_rejected(self) -> None:
        result = self._plan()
        with self.assertRaisesRegex(ValueError, "KUKA Y-gripper"):
            plan_dual_grasp_pairs(
                sequence=_sequence(),
                holder_feasibility=result.holder_feasibility,
                inserter_libraries=result.inserter_libraries,
                planning=PlanningConfig(gripper_collision_model="franka_hand"),
                config=_config(),
            )


@unittest.skipUnless(trimesh_fcl_backend_available(), "python-fcl is unavailable")
class InserterUnaryFilteringTests(unittest.TestCase):
    def test_table_filter_runs_after_existing_assembly_insertion_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "2.obj"
            incoming_path = root / "0.obj"
            for path, z in ((base_path, 5.0), (incoming_path, 0.2)):
                path.write_text(
                    "\n".join(
                        [
                            f"v 0 0 {z}",
                            f"v 0.1 0 {z}",
                            f"v 0 0.1 {z}",
                            "f 1 2 3",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
            parts = (
                AssemblyPartSpec(
                    part_id="2",
                    mesh_path=str(base_path),
                    role="base",
                    bounds_min_assembly_m=(0.0, 0.0, 5.0),
                    bounds_max_assembly_m=(0.1, 0.1, 5.0),
                    table_clearance_m=5.0,
                    touches_table=False,
                    vertex_count=3,
                    face_count=1,
                    asset_record={},
                    resolved_mesh_path=base_path,
                ),
                AssemblyPartSpec(
                    part_id="0",
                    mesh_path=str(incoming_path),
                    role="moving_part",
                    bounds_min_assembly_m=(0.0, 0.0, 0.2),
                    bounds_max_assembly_m=(0.1, 0.1, 0.2),
                    table_clearance_m=0.2,
                    touches_table=False,
                    vertex_count=3,
                    face_count=1,
                    asset_record={},
                    resolved_mesh_path=incoming_path,
                ),
            )
            table = _candidate("g0001", (0.2, 0.0, 0.005), score=1.0)
            clear = _candidate("g0002", (0.2, 0.2, 0.2), score=0.9)
            lower_clear = _candidate("g0003", (0.2, 0.4, 0.2), score=0.8)
            pose = ObjectWorldPose(
                position_world=(0.0, 0.0, 0.0),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            )
            stage1 = Stage1Result(
                bundle=SavedGraspBundle(
                    target_mesh_path=str(incoming_path),
                    mesh_scale=1.0,
                    source_frame_origin_obj_world=pose.position_world,
                    source_frame_orientation_xyzw_obj_world=(pose.orientation_xyzw_world),
                    candidates=(table, clear, lower_clear),
                    metadata={"stage1_cache_key": "fixture"},
                ),
                target_mesh_local=TriangleMesh(
                    vertices_obj=np.asarray([[0.0, 0.0, 0.2], [0.1, 0.0, 0.2], [0.0, 0.1, 0.2]]),
                    faces=np.asarray([[0, 1, 2]], dtype=np.int64),
                ),
                target_pose_in_obj_world=pose,
                obstacle_mesh_world=None,
                collision_backend_name="fixture",
                raw_candidate_count=3,
                raw_candidates=(table, clear, lower_clear),
            )
            with (
                mock.patch(
                    "grasp_planning.pipeline.dual_grasp_pair_planner.generate_stage1_result",
                    return_value=stage1,
                ),
                mock.patch(
                    "grasp_planning.pipeline.dual_grasp_pair_planner.make_gripper_collision_model",
                    return_value=_SmallCubeGripper(),
                ),
                ):
                with mock.patch(
                    "grasp_planning.pipeline.dual_grasp_pair_planner._candidate_primitives_assembly",
                    wraps=pair_planner._candidate_primitives_assembly,
                ) as candidate_primitives:
                    library = generate_inserter_grasp_library(
                        sequence=_sequence(parts=parts),
                        step=_step(),
                        planning=_planning(),
                        config=_config(
                            max_inserter_candidates_per_step=1,
                            inserter_contact_offset_pairs_m=(),
                        ),
                    )
            by_id = {status.grasp_id: status for status in library.candidate_statuses}
            self.assertEqual(
                by_id["i0_0001"].reason,
                "inserter_table_collision",
            )
            self.assertEqual(by_id["i0_0002"].status, "accepted")
            self.assertEqual(
                by_id["i0_0003"].reason,
                "not_evaluated_shortlist_complete",
            )
            self.assertEqual(candidate_primitives.call_count, 2)
            self.assertEqual(
                [candidate.grasp_id for candidate in library.bundle.candidates],
                ["i0_0002"],
            )


if __name__ == "__main__":
    unittest.main()
