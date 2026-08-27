from __future__ import annotations

from pathlib import Path

import yaml

from grasp_planning.rl.policy_timing import POLICY_RATE_HZ
from grasp_planning.ros2.d405_visual_servo import D405VisualServoDeploymentConfig


def _write_config(tmp_path: Path, **changes) -> Path:
    files = {}
    for key, name in (
        ("checkpoint_path", "checkpoint.pth"),
        ("checkpoint_metadata_path", "checkpoint.json"),
        ("agent_config_path", "agent.yaml"),
        ("goal_catalog_path", "catalog.npz"),
    ):
        path = tmp_path / name
        path.write_bytes(b"fixture")
        files[key] = path.name
    payload = {
        **files,
        "target_id": "part_0__current__g1973",
        "command_sink": "dry_run",
        "require_deadman": False,
        "require_force_measurement": False,
        **changes,
    }
    config_path = tmp_path / "deployment.yaml"
    config_path.write_text(yaml.safe_dump({"visual_servo": payload}), encoding="utf-8")
    return config_path


def test_dry_run_config_resolves_artifact_paths_relative_to_yaml(tmp_path: Path) -> None:
    config = D405VisualServoDeploymentConfig.from_yaml(_write_config(tmp_path))

    assert config.command_sink == "dry_run"
    assert config.checkpoint_path == (tmp_path / "checkpoint.pth").resolve()
    assert config.target_id == "part_0__current__g1973"
    assert not config.real_motion_approved
    assert config.max_joint_state_age_s == 0.15
    assert config.max_force_age_s == 0.15
    assert config.max_operator_signal_age_s == 0.25
    assert config.policy_rate_hz == POLICY_RATE_HZ
    assert config.action_delta_limit == 0.50


def test_moveit_servo_config_fails_closed_without_explicit_motion_approval(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, command_sink="moveit_servo")

    try:
        D405VisualServoDeploymentConfig.from_yaml(config_path)
    except ValueError as exc:
        assert "real_motion_approved" in str(exc)
    else:
        raise AssertionError("Expected MoveIt Servo output to require explicit real-motion approval.")


def test_config_rejects_non_contract_policy_rate(tmp_path: Path) -> None:
    try:
        D405VisualServoDeploymentConfig.from_yaml(_write_config(tmp_path, policy_rate_hz=30.0))
    except ValueError as exc:
        assert "policy_rate_hz=15.0" in str(exc)
    else:
        raise AssertionError("Expected a policy rate other than 15 Hz to be rejected.")
