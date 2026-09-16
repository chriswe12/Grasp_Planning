from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PACKAGE_ROOT = REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros"

sys.path.insert(0, str(WORKSPACE_PACKAGE_ROOT))
try:
    server = importlib.import_module("robot_integration_ros.grasp_assembly_action_server")
finally:
    sys.path = [entry for entry in sys.path if entry != str(WORKSPACE_PACKAGE_ROOT)]


def _goal(*, assembly: str = "plumbers_block", part_id: int = 0, robot: str = "left"):
    return SimpleNamespace(
        assembly_name=assembly,
        base_part_id=4,
        insertion_part_id=part_id,
        holder_robot="right",
        inserter_robot=robot,
    )


def _payload() -> dict[str, object]:
    return {
        "geometry": {
            "target_mesh_path": "obj/fabrica/plumbers_block/0.obj",
            "mesh_scale": 0.01,
            "assembly_glob": "obj/fabrica/plumbers_block/*.obj",
        },
        "artifacts": {},
        "ros2": {
            "pose_base_topic": "/perception/fp/pose_base/fused/assembly",
            "assembly_name": "plumbers_block",
            "part_id": 0,
            "position_offset_m": [0.0, -0.840, 0.0],
        },
        "real_execution": {
            "enabled": True,
            "require_confirmation": True,
            "stop_after": "pregrasp",
            "frame_id": "lbr_link_0",
        },
        "grasp_assembly_action": {
            "output_root": "artifacts/action_test",
        },
    }


def test_single_robot_goal_uses_only_insertion_fields() -> None:
    parsed = server.SingleRobotGraspGoal.from_request(_goal())

    assert parsed.assembly_name == "plumbers_block"
    assert parsed.insertion_part_id == 0
    assert parsed.inserter_robot == "left"
    assert not hasattr(parsed, "base_part_id")
    assert not hasattr(parsed, "holder_robot")


def test_goal_validation_requires_left_robot_existing_mesh_and_topic(tmp_path: Path) -> None:
    mesh = tmp_path / "assets" / "obj" / "fabrica" / "plumbers_block" / "0.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("# mesh\n", encoding="utf-8")
    payload = _payload()

    assert (
        server._validate_goal(
            repo_root=tmp_path,
            payload=payload,
            goal=server.SingleRobotGraspGoal.from_request(_goal()),
        )
        is None
    )
    assert "inserter_robot='left'" in server._validate_goal(
        repo_root=tmp_path,
        payload=payload,
        goal=server.SingleRobotGraspGoal.from_request(_goal(robot="right")),
    )
    assert "does not exist" in server._validate_goal(
        repo_root=tmp_path,
        payload=payload,
        goal=server.SingleRobotGraspGoal.from_request(_goal(part_id=1)),
    )


def test_prepare_pipeline_payload_selects_part_pose_and_pickup_execution() -> None:
    goal = server.SingleRobotGraspGoal.from_request(_goal())

    payload, output_dir = server._prepare_pipeline_payload(
        base_payload=_payload(),
        goal=goal,
        goal_id="abc123",
    )

    assert payload["geometry"]["target_mesh_path"] == "obj/fabrica/plumbers_block/0.obj"
    assert payload["geometry"]["assembly_glob"] == "obj/fabrica/plumbers_block/*.obj"
    assert payload["ros2"]["assembly_name"] == "plumbers_block"
    assert payload["ros2"]["part_id"] == 0
    assert payload["ros2"]["position_offset_m"] == [0.0, -0.840, 0.0]
    assert payload["real_execution"]["enabled"] is True
    assert payload["real_execution"]["require_confirmation"] is False
    assert payload["real_execution"]["stop_after"] == "lift"
    assert payload["real_execution"]["gripper_enabled"] is True
    assert payload["real_execution"]["attempt_artifact"] == str(output_dir / "real_robot_pick_attempt.json")
    assert output_dir == Path("artifacts/action_test/plumbers_block/part_0/abc123")


def test_prepare_pipeline_payload_can_skip_gripper_commands() -> None:
    payload, _ = server._prepare_pipeline_payload(
        base_payload=_payload(),
        goal=server.SingleRobotGraspGoal.from_request(_goal()),
        goal_id="no-gripper",
        skip_gripper=True,
    )

    assert payload["real_execution"]["enabled"] is True
    assert payload["real_execution"]["gripper_enabled"] is False
    assert payload["real_execution"]["stop_after"] == "lift"


def test_pickup_only_attempt_does_not_satisfy_action_contract(tmp_path: Path) -> None:
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text(
        json.dumps(
            {
                "config": {"lift_height_m": 0.08},
                "object_pose_world": {
                    "position_world": [0.5, 0.0, 0.04],
                    "orientation_xyzw_world": [-0.7071, 0.0, 0.0, 0.7071],
                },
                "result": {
                    "success": True,
                    "status": "stopped_at_lift",
                    "message": "Reached lift pose.",
                    "lift_reached": True,
                },
            }
        ),
        encoding="utf-8",
    )

    outcome = server._outcome_from_attempt(
        return_code=0,
        attempt_path=attempt_path,
        frame_id="lbr_link_0",
        output_lines=[],
        cancelled=False,
    )

    assert outcome.success is False
    assert outcome.error_code == "TRANSPORT_UNSUPPORTED"
    assert "not transported to its pre-assembly pose" in outcome.message
    assert outcome.grasped_frame_id == ""
    assert outcome.grasped_position_xyz is None
    assert outcome.grasped_orientation_xyzw is None


def test_skip_gripper_attempt_does_not_satisfy_action_contract(tmp_path: Path) -> None:
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text(
        json.dumps(
            {
                "config": {"gripper_enabled": False, "lift_height_m": 0.08},
                "object_pose_world": {
                    "position_world": [0.5, 0.0, 0.04],
                    "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                },
                "result": {
                    "success": True,
                    "status": "stopped_at_lift",
                    "message": "Reached lift pose.",
                    "lift_reached": True,
                },
            }
        ),
        encoding="utf-8",
    )

    outcome = server._outcome_from_attempt(
        return_code=0,
        attempt_path=attempt_path,
        frame_id="lbr_link_0",
        output_lines=[],
        cancelled=False,
    )

    assert outcome.success is False
    assert outcome.error_code == "TRANSPORT_UNSUPPORTED"
    assert "not transported to its pre-assembly pose" in outcome.message
    assert outcome.grasped_position_xyz is None
    assert outcome.grasped_orientation_xyzw is None


def test_runner_refuses_unsupported_action_before_hardware_motion(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "run_pipeline.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "real.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    runner = server.RealPipelineRunner(
        repo_root=tmp_path,
        config_path=config_path,
        allow_execution=True,
    )

    def _unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("pickup-only pipeline must not be started")

    monkeypatch.setattr(server.subprocess, "Popen", _unexpected_subprocess)
    outcome = runner.run(
        request=_goal(),
        goal_id="unsupported",
        cancel_requested=lambda: False,
        publish_feedback=lambda _phase, _progress: None,
        publish_output=lambda _line: None,
    )

    assert outcome.success is False
    assert outcome.error_code == "TRANSPORT_UNSUPPORTED"
    assert "No hardware motion was started" in outcome.message


def test_failed_attempt_status_becomes_action_error_code(tmp_path: Path) -> None:
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text(
        json.dumps(
            {
                "result": {
                    "success": False,
                    "status": "grasp_failed",
                    "message": "IK failed.",
                }
            }
        ),
        encoding="utf-8",
    )

    outcome = server._outcome_from_attempt(
        return_code=1,
        attempt_path=attempt_path,
        frame_id="lbr_link_0",
        output_lines=[],
        cancelled=False,
    )

    assert outcome.success is False
    assert outcome.error_code == "GRASP_FAILED"
    assert outcome.message == "IK failed."


def test_server_requires_explicit_execute_flag() -> None:
    parser = server.build_argument_parser()

    assert parser.parse_args([]).execute is False
    assert parser.parse_args(["--execute"]).execute is True
    assert parser.parse_args([]).skip_gripper is False
    assert parser.parse_args(["--execute", "--skip-gripper"]).skip_gripper is True


def test_server_parser_selects_guarded_dual_modes() -> None:
    parser = server.build_argument_parser()

    pitl = parser.parse_args(["--dual-mode", "pitl", "--headless"])
    assert pitl.dual_mode == "pitl"
    assert pitl.headless is True
    assert pitl.execute is False

    real = parser.parse_args(
        [
            "--dual-mode",
            "real",
            "--execute",
            "--allow-objectless-planning",
            "--pair-id",
            "pair_1",
        ]
    )
    assert real.dual_mode == "real"
    assert real.execute is True
    assert real.allow_objectless_planning is True
    assert real.pair_id == "pair_1"
