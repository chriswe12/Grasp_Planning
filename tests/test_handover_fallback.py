from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from grasp_planning.grasping.fabrica_grasp_debug import SavedGraspCandidate
from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline.handover_fallback import (
    HandoverFallbackResult,
    HandoverGraspPair,
    _same_contact_pair,
    write_handover_fallback_result,
)


def _candidate(
    grasp_id: str,
    *,
    contact_a: tuple[float, float, float],
    contact_b: tuple[float, float, float],
    score: float = 1.0,
) -> SavedGraspCandidate:
    return SavedGraspCandidate(
        grasp_id=grasp_id,
        grasp_position_obj=(0.0, 0.0, 0.0),
        grasp_orientation_xyzw_obj=(0.0, 0.0, 0.0, 1.0),
        contact_point_a_obj=contact_a,
        contact_point_b_obj=contact_b,
        contact_normal_a_obj=(1.0, 0.0, 0.0),
        contact_normal_b_obj=(-1.0, 0.0, 0.0),
        jaw_width=0.02,
        roll_angle_rad=0.0,
        score=score,
    )


class HandoverFallbackTests(unittest.TestCase):
    def test_same_contact_pair_ignores_contact_order(self) -> None:
        left = _candidate("left", contact_a=(0.01, 0.0, 0.0), contact_b=(-0.01, 0.0, 0.0))
        right = _candidate("right", contact_a=(-0.01, 0.0, 0.0), contact_b=(0.01, 0.0, 0.0))

        self.assertTrue(_same_contact_pair(left, right))

    def test_handover_result_serializes_selected_pair(self) -> None:
        transfer = _candidate("transfer", contact_a=(0.0, 0.01, 0.0), contact_b=(0.0, -0.01, 0.0), score=0.7)
        final = _candidate("final", contact_a=(0.01, 0.0, 0.0), contact_b=(-0.01, 0.0, 0.0), score=0.9)
        pair = HandoverGraspPair(
            transfer_grasp=transfer,
            final_grasp=final,
            status="accepted",
            reason="transfer_floor_and_hand_clear",
            score=900000.7,
        )
        result = HandoverFallbackResult(
            target_mesh_path="obj/fabrica/example/0.obj",
            mesh_scale=0.01,
            source_frame_origin_obj_world=(0.0, 0.0, 0.0),
            source_frame_orientation_xyzw_obj_world=(0.0, 0.0, 0.0, 1.0),
            initial_object_pose_world=ObjectWorldPose(
                position_world=(0.0, 0.0, 0.0),
                orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
            ),
            accepted_pairs=(pair,),
            rejected_pairs=(),
            transfer_floor_status_counts={"accepted": 1},
            metadata={"checked_pair_count": 1},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "handover.json"
            write_handover_fallback_result(result, output)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["selected_pair"]["transfer_grasp"]["grasp_id"], "transfer")
        self.assertEqual(payload["selected_pair"]["final_grasp"]["grasp_id"], "final")
        self.assertEqual(payload["metadata"]["checked_pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
