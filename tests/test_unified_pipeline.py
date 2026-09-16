from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from grasp_planning.rl.d405_deployment_config import write_visual_servo_config
from grasp_planning.ros2.dual_real_grasp_executor import _motion_sequence_for
from grasp_planning.ros2.saved_stage2_moveit import config_from_backend_command
from scripts.run_unified_pipeline import (
    REPO_ROOT,
    UnifiedInvocation,
    _prepare_saved_isaac_invocation,
    resolve_invocation,
)


def _command(*args: str) -> tuple[str, ...]:
    invocation, dry_run = resolve_invocation((*args, "--dry-run"))
    assert dry_run is True
    return invocation.command


def test_default_real_route_is_dual_grasp_and_preinsertion() -> None:
    command = _command("--mode", "real", "--execute")

    assert command[0] == str(REPO_ROOT / "scripts/run_dual_pipeline.sh")
    assert command[command.index("--robots") + 1] == "both"
    assert command[command.index("--stop-after") + 1] == "inserter_preinsertion"
    assert "--execute" in command
    assert "--start-moveit" not in command


def test_moveit_reuse_is_default_and_temporary_ownership_is_explicit() -> None:
    reused = _command("--mode", "real")
    managed = _command("--mode", "real", "--start-moveit")

    assert "--start-moveit" not in reused
    assert "--reuse-moveit" not in reused
    assert "--start-moveit" in managed


def test_single_active_robot_defaults_to_inserter_and_grasp_only_includes_lift() -> None:
    command = _command("--mode", "real", "--robots", "left", "--grasp_only", "--execute")

    assert command[command.index("--single-role") + 1] == "inserter"
    assert command[command.index("--stop-after") + 1] == "inserter_pickup_lift"


def test_single_holder_stops_after_holder_grasp() -> None:
    command = _command("--mode", "real", "--robots", "right", "--role", "holder", "--execute")

    assert command[command.index("--single-role") + 1] == "holder"
    assert command[command.index("--stop-after") + 1] == "holder_grasp"


def test_policy_is_forwarded_once_for_every_active_dual_approach() -> None:
    command = _command(
        "--mode",
        "real",
        "--policy",
        "velocity-rotation",
        "--left-camera",
        "realsense_2",
        "--right-camera",
        "realsense_1",
    )

    assert command[command.index("--policy") + 1] == "velocity-rotation"
    assert command[command.index("--left-camera") + 1] == "realsense_2"
    assert command[command.index("--right-camera") + 1] == "realsense_1"


def test_dual_single_robot_bringup_keeps_shared_collision_scene() -> None:
    command = _command("--mode", "real", "--robots", "left", "--bringup-only", "--servo")

    assert command[0] == str(REPO_ROOT / "start_dual_lbr_moveit.sh")
    assert command[command.index("--robots") + 1] == "left"
    assert "--servo" in command


def test_action_server_uses_same_robot_role_stop_and_policy_contract() -> None:
    command = _command(
        "--mode",
        "real",
        "--robots",
        "left",
        "--serve-action",
        "--grasp-only",
        "--policy",
        "velocity",
    )

    assert command[:4] == ("ros2", "run", "robot_integration_ros", "grasp_assembly_action_server")
    assert command[command.index("--robots") + 1] == "left"
    assert command[command.index("--single-role") + 1] == "inserter"
    assert command[command.index("--stop-after") + 1] == "inserter_pickup_lift"
    assert command[command.index("--policy") + 1] == "velocity"


def test_benchmark_selection_is_public_pipeline_route() -> None:
    command = _command("--benchmark", "dual-assembly", "--limit-cases", "2")

    assert command[1] == str(REPO_ROOT / "scripts/run_dual_assembly_benchmark.py")
    assert command[-2:] == ("--limit-cases", "2")


def test_grasp_execution_benchmark_receives_backend_override() -> None:
    command = _command("--benchmark", "grasp-execution", "--backend", "isaac", "--limit-attempts", "1")

    assert command[command.index("--backend") + 1] == "isaac"
    assert command[-2:] == ("--limit-attempts", "1")


def test_saved_stage2_execution_routes_through_public_pipeline(tmp_path: Path) -> None:
    bundle = tmp_path / "stage2.json"
    attempt = tmp_path / "attempt.json"
    command = _command(
        "--workflow",
        "single-object",
        "--mode",
        "sim",
        "--backend",
        "mujoco",
        "--stage2-bundle",
        str(bundle),
        "--grasp-id",
        "g1",
        "--attempt-artifact",
        str(attempt),
    )

    assert command[1] == str(REPO_ROOT / "scripts/run_fabrica_grasp_in_mujoco.py")
    assert command[command.index("--input-json") + 1] == str(bundle.resolve())
    assert command[command.index("--grasp-id") + 1] == "g1"


def test_saved_isaac_moveit_config_is_derived_from_resolved_backend_command() -> None:
    config = config_from_backend_command(
        (
            "isaaclab.sh",
            "-p",
            "runner.py",
            "--moveit-planning-group",
            "arm",
            "--moveit-joint-names",
            "j1,j2",
            "--moveit-start-joint-positions",
            "0.1,0.2",
            "--moveit-target-position-signs",
            "1,-1,1",
            "--tcp-to-grasp-offset",
            "0.0",
            "0.0",
            "0.035",
            "--pregrasp-only",
        )
    )

    assert config.planning_group == "arm"
    assert config.joint_names == ("j1", "j2")
    assert config.start_joint_positions == (0.1, 0.2)
    assert config.target_position_signs == (1.0, -1.0, 1.0)
    assert config.tcp_to_grasp_offset == (0.0, 0.0, 0.035)
    assert config.pregrasp_only is True


def test_saved_isaac_preplan_failure_is_owned_and_recorded_by_unified_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import grasp_planning.ros2.saved_stage2_moveit as preplanner

    bundle = tmp_path / "stage2.json"
    attempt = tmp_path / "attempt.json"
    invocation = UnifiedInvocation(
        (
            "isaaclab.sh",
            "-p",
            "runner.py",
            "--input-json",
            str(bundle),
            "--grasp-id",
            "g1",
            "--attempt-artifact",
            str(attempt),
        ),
        "saved stage-2 isaac execution",
    )
    monkeypatch.setattr(
        preplanner,
        "preplan_saved_stage2_for_isaac",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("no collision-free plan")),
    )

    with pytest.raises(RuntimeError, match="no collision-free plan"):
        _prepare_saved_isaac_invocation(invocation)

    artifact = json.loads(attempt.read_text(encoding="utf-8"))
    assert artifact["execution"]["status"] == "moveit_preplan_failed"
    assert artifact["execution"]["success"] is False


def test_policy_and_single_active_dual_sim_are_rejected() -> None:
    with pytest.raises(ValueError, match="policy.*real"):
        resolve_invocation(("--mode", "sim", "--policy", "velocity"))
    with pytest.raises(ValueError, match="Single-active-robot.*real"):
        resolve_invocation(("--mode", "sim", "--robots", "left"))


def test_active_inserter_sequence_skips_holder_but_keeps_pickup_lift() -> None:
    assert _motion_sequence_for("inserter_pickup_lift", ("inserter",)) == (
        ("inserter", "inserter_pickup_pregrasp"),
        ("inserter", "inserter_pickup_grasp"),
        ("inserter", "inserter_pickup_lift"),
    )


def test_dual_policy_config_uses_prefixed_tf_feedback_and_servo_routes(tmp_path: Path) -> None:
    output = tmp_path / "visual.yaml"
    write_visual_servo_config(
        policy_name="velocity",
        assets={
            "checkpoint": tmp_path / "checkpoint.pth",
            "metadata": tmp_path / "metadata.json",
            "agent_config": tmp_path / "agent.yaml",
            "policy_rate_hz": 15.0,
            "action_delta_limit": 0.5,
            "camera_profile": "camera-profile",
            "gripper_model": "pdz_gripper",
        },
        template_path=REPO_ROOT / "configs/visual_servo_real_d405.yaml",
        output_path=output,
        output_root=tmp_path / "runs",
        model_device="cuda:0",
        camera_name="realsense_2",
        robot_name="lbr_two",
    )

    block = yaml.safe_load(output.read_text(encoding="utf-8"))["visual_servo"]
    assert block["camera_optical_frame"] == "lbr_two_camera_color_optical_frame"
    assert block["allow_camera_topic_frame_alias"] is True
    assert block["command_frame"] == "base_link"
    assert block["tcp_frame"] == "lbr_two_pdz_gripper_tcp"
    assert block["joint_state_topic"] == "/lbr_dual_arm/joint_states"
    assert block["force_topic"] == (
        "/lbr_dual_arm/lbr_two_control/lbr_two_force_torque_broadcaster/wrench"
    )
    assert block["expected_joint_names"] == [f"lbr_two_A{index}" for index in range(1, 8)]
    assert block["moveit_servo_twist_topic"] == "/lbr_dual_arm/lbr_two_servo_node/delta_twist_cmds"
