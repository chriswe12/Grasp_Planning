from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from grasp_planning.grasping import ObjectWorldPose, TriangleMesh
from grasp_planning.grasping.fabrica_grasp_debug import (
    CandidateStatus,
    SavedGraspBundle,
    SavedGraspCandidate,
    load_grasp_bundle,
    save_grasp_bundle,
)
from grasp_planning.pipeline.fabrica_pipeline import PlanningConfig, recheck_stage2_result


def _candidate(grasp_id: str = "g0001", *, x_position: float = 0.01) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=(x_position, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=(x_position, -0.01, 0.0),
        contact_point_b_obj=(x_position, 0.01, 0.0),
        contact_normal_a_obj=(0.0, 1.0, 0.0),
        contact_normal_b_obj=(0.0, -1.0, 0.0),
        jaw_width=0.02,
        roll_angle_rad=0.0,
    )


def _bundle(
    candidate: SavedGraspCandidate,
    *,
    mesh_scale: float = 1.0,
    source_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    source_orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> SavedGraspBundle:
    return SavedGraspBundle(
        target_mesh_path="obj/fabrica/beam/6.obj",
        mesh_scale=mesh_scale,
        source_frame_origin_obj_world=source_origin,
        source_frame_orientation_xyzw_obj_world=source_orientation,
        candidates=(candidate,),
        metadata={},
    )


def _run_stage2_symmetry_recheck(
    bundle: SavedGraspBundle,
    symmetry_payload: dict[str, object],
) -> tuple[object, list[SavedGraspCandidate]]:
    mesh = TriangleMesh(
        vertices_obj=np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    evaluated: list[SavedGraspCandidate] = []

    def accept_all(candidates, **_kwargs):
        evaluated.extend(tuple(candidates))
        return [CandidateStatus(grasp=candidate, status="accepted", reason="unit") for candidate in candidates]

    with tempfile.TemporaryDirectory() as temp_dir:
        symmetry_path = Path(temp_dir) / "symmetries.json"
        symmetry_path.write_text(json.dumps(symmetry_payload), encoding="utf-8")
        with (
            mock.patch("grasp_planning.pipeline.fabrica_pipeline.load_asset_mesh", return_value=mesh),
            mock.patch(
                "grasp_planning.pipeline.fabrica_pipeline.evaluate_saved_grasps_against_pickup_pose",
                side_effect=accept_all,
            ),
            mock.patch(
                "grasp_planning.pipeline.fabrica_pipeline.score_grasps",
                side_effect=lambda grasps, **_kwargs: list(grasps),
            ),
        ):
            result = recheck_stage2_result(
                bundle=bundle,
                pickup_spec=None,
                planning=PlanningConfig(
                    symmetry_pickup_enabled=True,
                    symmetry_asset_path=str(symmetry_path),
                    symmetry_next_orientation_limit=8,
                ),
                object_pose_world=ObjectWorldPose(
                    position_world=(0.0, 0.0, 0.0),
                    orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
                ),
            )
    return result, evaluated


class PipelineSymmetryPickupTests(unittest.TestCase):
    def test_grasp_bundle_preserves_candidate_metadata(self) -> None:
        candidate = _candidate()
        candidate = SavedGraspCandidate(**{**candidate.__dict__, "metadata": {"symmetry_pickup_name": "turn_z"}})
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bundle.json"
            save_grasp_bundle(_bundle(candidate), path)

            loaded = load_grasp_bundle(path)

        self.assertEqual(loaded.candidates[0].metadata, {"symmetry_pickup_name": "turn_z"})

    def test_stage2_expands_pickup_candidates_with_symmetry_grasps(self) -> None:
        symmetry_payload = {
            "schema_version": 1,
            "parts": {
                "6": {
                    "symmetries": [
                        {
                            "name": "identity",
                            "type": "identity",
                            "description": "Identity",
                            "angle_deg": 0.0,
                            "source": "identity",
                            "matrix_obj": np.eye(4).tolist(),
                        },
                        {
                            "name": "turn_z",
                            "type": "finite_rotation",
                            "description": "180 about +Z",
                            "angle_deg": 180.0,
                            "source": "unit",
                            "matrix_obj": [
                                [-1.0, 0.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        },
                    ]
                }
            },
        }
        mesh = TriangleMesh(
            vertices_obj=np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        evaluated: list[SavedGraspCandidate] = []

        def accept_all(candidates, **_kwargs):
            evaluated.extend(tuple(candidates))
            return [CandidateStatus(grasp=candidate, status="accepted", reason="unit") for candidate in candidates]

        with tempfile.TemporaryDirectory() as temp_dir:
            symmetry_path = Path(temp_dir) / "symmetries.json"
            symmetry_path.write_text(json.dumps(symmetry_payload), encoding="utf-8")
            with (
                mock.patch("grasp_planning.pipeline.fabrica_pipeline.load_asset_mesh", return_value=mesh),
                mock.patch(
                    "grasp_planning.pipeline.fabrica_pipeline.evaluate_saved_grasps_against_pickup_pose",
                    side_effect=accept_all,
                ),
                mock.patch(
                    "grasp_planning.pipeline.fabrica_pipeline.score_grasps",
                    side_effect=lambda grasps, **_kwargs: list(grasps),
                ),
            ):
                result = recheck_stage2_result(
                    bundle=_bundle(_candidate()),
                    pickup_spec=None,
                    planning=PlanningConfig(
                        symmetry_pickup_enabled=True,
                        symmetry_asset_path=str(symmetry_path),
                        symmetry_next_orientation_limit=8,
                    ),
                    object_pose_world=ObjectWorldPose(
                        position_world=(0.0, 0.0, 0.0),
                        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
                    ),
                )

        self.assertEqual(len(evaluated), 2)
        transformed = next(candidate for candidate in evaluated if candidate.grasp_id.endswith("__sym_turn_z"))
        np.testing.assert_allclose(transformed.grasp_position_obj, (-0.01, 0.0, 0.0), atol=1.0e-9)
        self.assertEqual(transformed.metadata["symmetry_pickup_parent_grasp_id"], "g0001")
        self.assertEqual(transformed.metadata["symmetry_pickup_name"], "turn_z")
        self.assertEqual(result.accepted_bundle.metadata["ground_original_input_count"], 1)
        self.assertEqual(result.accepted_bundle.metadata["ground_input_count"], 2)
        self.assertEqual(result.accepted_bundle.metadata["symmetry_pickup_derived_candidate_count"], 1)
        self.assertEqual(result.accepted_bundle.metadata["symmetry_pickup_load_status"], "loaded")
        self.assertEqual(
            result.accepted_bundle.metadata["symmetry_pickup_parent_summaries"][0]["feasible_variant_count"], 2
        )
        self.assertTrue(result.accepted_bundle.metadata["symmetry_next_orientation_options"])

    def test_stage2_converts_symmetry_transforms_into_translated_bundle_frame(self) -> None:
        symmetry_payload = {
            "schema_version": 1,
            "parts": {
                "6": {
                    "symmetries": [
                        {
                            "name": "identity",
                            "type": "identity",
                            "description": "Identity",
                            "angle_deg": 0.0,
                            "source": "identity",
                            "matrix_obj": np.eye(4).tolist(),
                        },
                        {
                            "name": "turn_z_about_asset_source",
                            "type": "finite_rotation",
                            "description": "180 about +Z around the original asset source point",
                            "angle_deg": 180.0,
                            "source": "unit",
                            "matrix_obj": [
                                [-1.0, 0.0, 0.0, 2.0],
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        },
                    ]
                }
            },
        }

        result, evaluated = _run_stage2_symmetry_recheck(
            _bundle(_candidate(), source_origin=(1.0, 0.0, 0.0)),
            symmetry_payload,
        )

        transformed = next(
            candidate for candidate in evaluated if candidate.grasp_id.endswith("__sym_turn_z_about_asset_source")
        )
        np.testing.assert_allclose(transformed.grasp_position_obj, (-0.01, 0.0, 0.0), atol=1.0e-9)
        np.testing.assert_allclose(transformed.contact_point_a_obj, (-0.01, 0.01, 0.0), atol=1.0e-9)
        np.testing.assert_allclose(transformed.contact_point_b_obj, (-0.01, -0.01, 0.0), atol=1.0e-9)
        np.testing.assert_allclose(transformed.contact_normal_a_obj, (0.0, -1.0, 0.0), atol=1.0e-9)
        np.testing.assert_allclose(transformed.contact_normal_b_obj, (0.0, 1.0, 0.0), atol=1.0e-9)
        turn_options = [
            option
            for option in result.accepted_bundle.metadata["symmetry_next_orientation_options"]
            if option["final_symmetry_name"] == "turn_z_about_asset_source"
        ]
        self.assertTrue(turn_options)
        for option in turn_options:
            np.testing.assert_allclose(option["final_object_pose"]["position_world"], [1.0, 0.0, 0.0], atol=1.0e-9)

    def test_stage2_converts_symmetry_transforms_through_bundle_rotation(self) -> None:
        symmetry_payload = {
            "schema_version": 1,
            "parts": {
                "6": {
                    "symmetries": [
                        {
                            "name": "identity",
                            "type": "identity",
                            "description": "Identity",
                            "angle_deg": 0.0,
                            "source": "identity",
                            "matrix_obj": np.eye(4).tolist(),
                        },
                        {
                            "name": "turn_x_asset",
                            "type": "finite_rotation",
                            "description": "180 about asset +X",
                            "angle_deg": 180.0,
                            "source": "unit",
                            "matrix_obj": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        },
                    ]
                }
            },
        }

        _result, evaluated = _run_stage2_symmetry_recheck(
            _bundle(_candidate(), source_orientation=(0.0, 0.0, 0.70710678, 0.70710678)),
            symmetry_payload,
        )

        transformed = next(candidate for candidate in evaluated if candidate.grasp_id.endswith("__sym_turn_x_asset"))
        np.testing.assert_allclose(transformed.grasp_position_obj, (-0.01, 0.0, 0.0), atol=1.0e-8)
        np.testing.assert_allclose(transformed.contact_point_a_obj, (-0.01, -0.01, 0.0), atol=1.0e-8)
        np.testing.assert_allclose(transformed.contact_point_b_obj, (-0.01, 0.01, 0.0), atol=1.0e-8)
        np.testing.assert_allclose(transformed.contact_normal_a_obj, (0.0, 1.0, 0.0), atol=1.0e-8)
        np.testing.assert_allclose(transformed.contact_normal_b_obj, (0.0, -1.0, 0.0), atol=1.0e-8)

    def test_stage2_scales_symmetry_translation_to_bundle_mesh_scale(self) -> None:
        symmetry_payload = {
            "schema_version": 1,
            "mesh_scale": 1.0,
            "parts": {
                "6": {
                    "symmetries": [
                        {
                            "name": "identity",
                            "type": "identity",
                            "description": "Identity",
                            "angle_deg": 0.0,
                            "source": "identity",
                            "matrix_obj": np.eye(4).tolist(),
                        },
                        {
                            "name": "turn_z_about_scaled_center",
                            "type": "finite_rotation",
                            "description": "180 about +Z around an asset-scale center",
                            "angle_deg": 180.0,
                            "source": "unit",
                            "matrix_obj": [
                                [-1.0, 0.0, 0.0, 2.0],
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        },
                    ]
                }
            },
        }

        result, evaluated = _run_stage2_symmetry_recheck(
            _bundle(_candidate(x_position=0.51), mesh_scale=0.5),
            symmetry_payload,
        )

        transformed = next(
            candidate for candidate in evaluated if candidate.grasp_id.endswith("__sym_turn_z_about_scaled_center")
        )
        np.testing.assert_allclose(transformed.grasp_position_obj, (0.49, 0.0, 0.0), atol=1.0e-9)
        np.testing.assert_allclose(transformed.contact_point_a_obj, (0.49, 0.01, 0.0), atol=1.0e-9)
        np.testing.assert_allclose(transformed.contact_point_b_obj, (0.49, -0.01, 0.0), atol=1.0e-9)
        self.assertEqual(result.accepted_bundle.metadata["symmetry_pickup_asset_mesh_scale"], 1.0)
        self.assertEqual(result.accepted_bundle.metadata["symmetry_pickup_bundle_mesh_scale"], 0.5)
        self.assertEqual(result.accepted_bundle.metadata["symmetry_pickup_translation_scale"], 0.5)
        self.assertEqual(transformed.metadata["symmetry_pickup_matrix_obj"][0][3], 1.0)


if __name__ == "__main__":
    unittest.main()
