from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grasp_planning.pipeline import (
    MovableFrame,
    ReachabilityProxyConfig,
    TaskTargetPose,
    pair_layout_score,
    transform_target_pose,
    workspace_pose_score,
    write_dual_robot_pair_score_debug_html,
)
from scripts.build_dual_robot_pair_score_debug import _reachability_config


def _target(
    name: str,
    position: tuple[float, float, float],
) -> TaskTargetPose:
    return TaskTargetPose(
        name=name,
        position_world_m=position,
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )


class DualRobotPairScoringTests(unittest.TestCase):
    def test_workspace_proxy_prefers_front_comfort_pose(self) -> None:
        config = ReachabilityProxyConfig()
        robot = MovableFrame((0.0, 0.0, 0.0))
        front = workspace_pose_score(
            _target("front", (0.5, 0.0, 0.25)),
            robot_base_world=robot,
            config=config,
        )
        behind = workspace_pose_score(
            _target("behind", (-0.5, 0.0, 0.25)),
            robot_base_world=robot,
            config=config,
        )
        unreachable = workspace_pose_score(
            _target("far", (1.2, 0.0, 0.25)),
            robot_base_world=robot,
            config=config,
        )

        self.assertGreater(front["score"], behind["score"])
        self.assertTrue(front["inside_reach_shell"])
        self.assertFalse(unreachable["inside_reach_shell"])
        self.assertEqual(unreachable["score"], 0.0)

    def test_layout_component_prefers_each_robot_owning_its_side(self) -> None:
        config = ReachabilityProxyConfig()
        holder_robot = MovableFrame((0.0, -0.42, 0.0), 0.0)
        inserter_robot = MovableFrame((0.0, 0.42, 0.0), 0.0)
        holder_own = _target("holder", (0.0, -0.1, 0.25))
        inserter_own = _target("inserter", (0.0, 0.1, 0.25))
        assigned = pair_layout_score(
            offline_pair_score=0.8,
            holder_targets=(holder_own,),
            inserter_targets=(inserter_own,),
            holder_grasp_target=holder_own,
            inserter_grasp_target=inserter_own,
            holder_robot_base_world=holder_robot,
            inserter_robot_base_world=inserter_robot,
            config=config,
        )
        crossed = pair_layout_score(
            offline_pair_score=0.8,
            holder_targets=(inserter_own,),
            inserter_targets=(holder_own,),
            holder_grasp_target=inserter_own,
            inserter_grasp_target=holder_own,
            holder_robot_base_world=holder_robot,
            inserter_robot_base_world=inserter_robot,
            config=config,
        )

        self.assertGreater(
            assigned["layout_score"],
            crossed["layout_score"],
        )
        self.assertGreater(
            assigned["ownership_score"],
            crossed["ownership_score"],
        )

    def test_transform_target_pose_applies_parent_yaw_and_translation(self) -> None:
        transformed = transform_target_pose(
            position_parent_m=(0.2, 0.0, 0.1),
            orientation_xyzw_parent=(0.0, 0.0, 0.0, 1.0),
            parent_frame_world=MovableFrame((1.0, 2.0, 0.3), 90.0),
            name="target",
        )

        self.assertAlmostEqual(transformed.position_world_m[0], 1.0)
        self.assertAlmostEqual(transformed.position_world_m[1], 2.2)
        self.assertAlmostEqual(transformed.position_world_m[2], 0.4)
        self.assertAlmostEqual(
            abs(transformed.orientation_xyzw_world[2]),
            2.0**-0.5,
        )
        self.assertAlmostEqual(
            abs(transformed.orientation_xyzw_world[3]),
            2.0**-0.5,
        )

    def test_yaml_debug_defaults_keep_840_mm_robot_offset(self) -> None:
        (
            holder,
            inserter,
            assembly,
            separation,
            locked,
            scoring,
        ) = _reachability_config({})

        self.assertEqual(separation, 0.84)
        self.assertEqual(holder.position_world_m, (0.0, -0.42, 0.0))
        self.assertEqual(inserter.position_world_m, (0.0, 0.42, 0.0))
        self.assertEqual(holder.yaw_deg, 0.0)
        self.assertEqual(inserter.yaw_deg, 0.0)
        self.assertEqual(assembly.position_world_m, (0.55, 0.0, 0.0))
        self.assertTrue(locked)
        self.assertEqual(scoring.offline_pair_weight, 0.40)

    def test_debug_html_contains_movable_frames_and_live_ranking(self) -> None:
        payload = {
            "assembly": "fixture",
            "selected_order": ["2", "0"],
            "base_part_id": "2",
            "scope_warning": "proxy only",
            "robot_layout_assumption": "fixture layout",
            "initial_layout": {
                "robot_separation_y_m": 0.84,
                "auto_select_top_pair": True,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "pair_debug.html"
            write_dual_robot_pair_score_debug_html(payload, output_path)
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("Pair Reachability Ranking", html)
        self.assertIn("H holder", html)
        self.assertIn("I inserter", html)
        self.assertIn("A assembly", html)
        self.assertIn("P pickup", html)
        self.assertIn("retainedOnly", html)
        self.assertIn("Current ranking", html)
        self.assertIn("Copy layout JSON", html)
        self.assertIn("H · HOLDER ROBOT", html)
        self.assertIn("I · INSERTER ROBOT", html)
        self.assertIn('id="graspScene"', html)
        self.assertIn("Selected grasp geometry", html)
        self.assertIn('id="holderRobot"', html)
        self.assertIn('id="autoTop"', html)
        self.assertIn('id="prevTop"', html)
        self.assertIn('id="nextTop"', html)
        self.assertIn("manual selection disables auto-follow", html)
        self.assertIn('"robot_separation_y_m":0.84', html)
        self.assertIn('"auto_select_top_pair":true', html)

    def test_invalid_reachability_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum < comfort < maximum"):
            ReachabilityProxyConfig(
                minimum_reach_m=0.6,
                comfort_reach_m=0.5,
                maximum_reach_m=0.9,
            )


if __name__ == "__main__":
    unittest.main()
