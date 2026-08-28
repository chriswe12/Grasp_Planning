from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from grasp_planning.d405_wrist_camera import (
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
)
from grasp_planning.rl.d405_goal_renderer import render_d405_goal_for_grasp
from grasp_planning.ros2.d405_visual_servo import D405VisualServoDeploymentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_renderer_uses_only_the_moveit_selected_grasp_and_validates_runtime_rgbd(tmp_path: Path) -> None:
    renderer_script = tmp_path / "renderer.py"
    launcher = tmp_path / "run_mujoco_filament.sh"
    robot_urdf = tmp_path / "robot.urdf"
    stage2_bundle = tmp_path / "stage2.json"
    output = tmp_path / "policy_goal_g0042.npz"
    for path in (launcher, renderer_script, robot_urdf, stage2_bundle):
        path.write_text("fixture\n", encoding="utf-8")

    config = SimpleNamespace(
        goal_renderer_launcher=launcher,
        goal_renderer_script=renderer_script,
        goal_renderer_robot_urdf=robot_urdf,
        goal_renderer_python_command="python3",
        goal_renderer_backend="filament",
        goal_renderer_timeout_s=240.0,
        expected_camera_profile=D405_VISUAL_SERVO_CAMERA_PROFILE,
        expected_observation_profile=D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    )

    def render(command, **kwargs):
        np.savez_compressed(
            output,
            goal_id=np.asarray("runtime__part_0__g0042"),
            part_id=np.asarray("0"),
            grasp_id=np.asarray("g0042"),
            jaw_width_m=np.asarray(0.031, dtype=np.float32),
            goal_rgb=np.zeros((144, 256, 3), dtype=np.uint8),
            goal_depth=np.full((144, 256), 0.12, dtype=np.float32),
            goal_camera_profile=np.asarray(D405_VISUAL_SERVO_CAMERA_PROFILE),
            goal_observation_profile=np.asarray(D405_VISUAL_SERVO_OBSERVATION_PROFILE),
            render_validation_passed=np.asarray(True, dtype=np.bool_),
        )
        return 0

    with (
        mock.patch.object(D405VisualServoDeploymentConfig, "from_yaml", return_value=config),
        mock.patch(
            "grasp_planning.rl.d405_goal_renderer.run_process_group",
            side_effect=render,
        ) as run,
    ):
        result = render_d405_goal_for_grasp(
            config_path=tmp_path / "visual.yaml",
            stage2_bundle_path=stage2_bundle,
            grasp_id="g0042",
            part_id="0",
            goal_joint_positions=(-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            goal_tcp_position=(-0.4, -0.1, 0.2),
            goal_tcp_orientation_xyzw=(
                -0.64623832798173575,
                0.7218592872184921,
                -0.16189657190189485,
                0.18730908389663303,
            ),
            approach_width_m=0.041,
            maximum_approach_width_m=0.064,
            output_path=output,
        )

    command = run.call_args.args[0]
    assert result.path == output
    assert result.goal_id == "runtime__part_0__g0042"
    assert result.grasp_id == "g0042"
    assert result.part_id == "0"
    assert command[:3] == [str(launcher), "python3", str(renderer_script)]
    assert command[command.index("--input-json") + 1] == str(stage2_bundle)
    assert command[command.index("--grasp-id") + 1] == "g0042"
    assert float(command[command.index("--maximum-approach-width-m") + 1]) == 0.064
    joint_argument = next(value for value in command if value.startswith("--goal-joint-positions="))
    position_argument = next(value for value in command if value.startswith("--goal-tcp-position="))
    orientation_argument = next(
        value for value in command if value.startswith("--goal-tcp-orientation-xyzw=")
    )
    np.testing.assert_allclose(
        [float(value) for value in joint_argument.split("=", 1)[1].split(",")],
        (-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    )
    np.testing.assert_allclose(
        [float(value) for value in position_argument.split("=", 1)[1].split(",")],
        (-0.4, -0.1, 0.2),
    )
    np.testing.assert_allclose(
        [float(value) for value in orientation_argument.split("=", 1)[1].split(",")],
        (
            -0.64623832798173575,
            0.7218592872184921,
            -0.16189657190189485,
            0.18730908389663303,
        ),
    )
    assert command[command.index("--robot-urdf") + 1] == str(robot_urdf)
    assert command[command.index("--renderer-backend") + 1] == "filament"
    assert "--headless" not in command
    assert not any("catalog" in value or "target-id" in value for value in command)


def test_renderer_reports_masked_mujoco_failure_instead_of_follow_on_file_not_found(tmp_path: Path) -> None:
    renderer_script = tmp_path / "renderer.py"
    launcher = tmp_path / "run_mujoco_filament.sh"
    robot_urdf = tmp_path / "robot.urdf"
    stage2_bundle = tmp_path / "stage2.json"
    for path in (launcher, renderer_script, robot_urdf, stage2_bundle):
        path.write_text("fixture\n", encoding="utf-8")
    config = SimpleNamespace(
        goal_renderer_launcher=launcher,
        goal_renderer_script=renderer_script,
        goal_renderer_robot_urdf=robot_urdf,
        goal_renderer_python_command="python3",
        goal_renderer_backend="filament",
        goal_renderer_timeout_s=240.0,
        expected_camera_profile=D405_VISUAL_SERVO_CAMERA_PROFILE,
        expected_observation_profile=D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    )

    with (
        mock.patch.object(D405VisualServoDeploymentConfig, "from_yaml", return_value=config),
        mock.patch(
            "grasp_planning.rl.d405_goal_renderer.run_process_group",
            return_value=0,
        ),
        pytest.raises(RuntimeError, match="produced no goal artifact"),
    ):
        render_d405_goal_for_grasp(
            config_path=tmp_path / "visual.yaml",
            stage2_bundle_path=stage2_bundle,
            grasp_id="g0042",
            part_id="0",
            goal_joint_positions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            goal_tcp_position=(0.4, -0.1, 0.2),
            goal_tcp_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            approach_width_m=0.041,
            maximum_approach_width_m=0.064,
            output_path=tmp_path / "missing.npz",
        )


def test_runtime_renderer_uses_moveit_joints_directly_in_imported_urdf() -> None:
    source = (REPO_ROOT / "scripts/render_d405_policy_goal.py").read_text(encoding="utf-8")

    assert "data.qpos[:7] = moveit_joints" in source
    assert "kuka_moveit_to_isaac_joint_positions" not in source
    assert "isaaclab" not in source.lower()


def test_runtime_pdz_renderer_uses_training_material_and_scene_contract() -> None:
    source = (REPO_ROOT / "scripts/render_d405_policy_goal.py").read_text(encoding="utf-8")

    assert "from grasp_planning.rl.goal_catalog_profiles import" in source
    assert "from grasp_planning.rl.goal_renderer_profiles import" not in source
    assert "_restore_pdz_gripper_visual_meshes" in source
    assert '"pdz_contact_white" if is_pad else "pdz_finger_black"' in source
    assert "FILAMENT_FALLBACK_HEAD_LIGHT_INTENSITY = 1000.0" in source
    assert "FILAMENT_FALLBACK_ENVIRONMENT_LIGHT_INTENSITY = 6500.0" in source
