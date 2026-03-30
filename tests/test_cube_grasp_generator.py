from __future__ import annotations

import math
import unittest

import numpy as np

from grasp_planning.grasping import CubeFaceGraspGenerator


class CubeFaceGraspGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = CubeFaceGraspGenerator(
            cube_size=(0.05, 0.05, 0.05),
            pregrasp_offset=0.20,
        )
        self.candidates = self.generator.generate(
            cube_position_w=(-0.45, 0.0, 0.025),
            cube_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            robot_base_position_w=(0.0, 0.0, 0.0),
        )
        self.by_label = {candidate.label: candidate for candidate in self.candidates}

    def test_pos_z_grasp_has_expected_pregrasp_and_axes(self) -> None:
        grasp = self.by_label["+z"]

        self.assertEqual(grasp.position_w, (-0.45, 0.0, 0.025))
        self.assertEqual(grasp.pregrasp_position_w, (-0.45, 0.0, 0.225))
        self.assertEqual(grasp.normal_w, (-0.0, -0.0, -1.0))
        self.assertAlmostEqual(grasp.orientation_xyzw[0], 0.0, places=6)
        self.assertAlmostEqual(grasp.orientation_xyzw[1], math.sqrt(0.5), places=6)
        self.assertAlmostEqual(grasp.orientation_xyzw[2], 0.0, places=6)
        self.assertAlmostEqual(grasp.orientation_xyzw[3], math.sqrt(0.5), places=6)

        approach_axis, closing_axis, gripper_z_axis = grasp.gripper_axes_w()
        np.testing.assert_allclose(approach_axis, np.array([0.0, 0.0, -1.0]), atol=1e-6)
        np.testing.assert_allclose(closing_axis, np.array([0.0, 1.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(gripper_z_axis, np.array([1.0, 0.0, 0.0]), atol=1e-6)

    def test_neg_z_grasp_has_expected_pregrasp_and_axes(self) -> None:
        grasp = self.by_label["-z"]

        self.assertEqual(grasp.position_w, (-0.45, 0.0, 0.025))
        self.assertEqual(grasp.pregrasp_position_w, (-0.45, 0.0, -0.17500000000000002))
        self.assertEqual(grasp.normal_w, (-0.0, -0.0, 1.0))
        self.assertAlmostEqual(grasp.orientation_xyzw[0], math.sqrt(0.5), places=6)
        self.assertAlmostEqual(grasp.orientation_xyzw[1], 0.0, places=6)
        self.assertAlmostEqual(grasp.orientation_xyzw[2], math.sqrt(0.5), places=6)
        self.assertAlmostEqual(grasp.orientation_xyzw[3], 0.0, places=6)

        approach_axis, closing_axis, gripper_z_axis = grasp.gripper_axes_w()
        np.testing.assert_allclose(approach_axis, np.array([0.0, 0.0, 1.0]), atol=1e-6)
        np.testing.assert_allclose(closing_axis, np.array([0.0, -1.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(gripper_z_axis, np.array([1.0, 0.0, 0.0]), atol=1e-6)

    def test_pos_z_and_neg_z_share_gripper_z_axis_and_flip_approach(self) -> None:
        pos_z = self.by_label["+z"]
        neg_z = self.by_label["-z"]

        pos_axes = pos_z.gripper_axes_w()
        neg_axes = neg_z.gripper_axes_w()

        np.testing.assert_allclose(pos_axes[2], neg_axes[2], atol=1e-6)
        np.testing.assert_allclose(pos_axes[0], -neg_axes[0], atol=1e-6)
        np.testing.assert_allclose(pos_axes[1], -neg_axes[1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
