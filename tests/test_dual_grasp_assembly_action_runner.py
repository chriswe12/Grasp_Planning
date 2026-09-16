from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from grasp_planning.grasping.world_constraints import ObjectWorldPose
from grasp_planning.pipeline.dual_robot_pair_scoring import MovableFrame
from grasp_planning.pipeline.dual_robot_simple_sim import (
    PlanarRuntimeLayout,
    RuntimePartAabb,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PACKAGE_ROOT = REPO_ROOT / "ros2_ws" / "src" / "robot_integration_ros"
sys.path.insert(0, str(WORKSPACE_PACKAGE_ROOT))
try:
    from robot_integration_ros.dual_grasp_assembly_action_runner import (
        DualPipelineRunner,
        DualRobotGraspGoal,
    )
finally:
    sys.path = [entry for entry in sys.path if entry != str(WORKSPACE_PACKAGE_ROOT)]


def _goal(
    *,
    base_part_id: int = 2,
    insertion_part_id: int = 0,
    holder_robot: str = "left",
    inserter_robot: str = "right",
):
    return SimpleNamespace(
        assembly_name="plumbers_block",
        base_part_id=base_part_id,
        insertion_part_id=insertion_part_id,
        holder_robot=holder_robot,
        inserter_robot=inserter_robot,
    )


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "configs").mkdir()
    (tmp_path / "run_pipeline.sh").write_text(
        "#!/usr/bin/env bash\n",
        encoding="utf-8",
    )
    for part_id in (0, 2):
        mesh = tmp_path / "assets" / "obj" / "fabrica" / "plumbers_block" / f"{part_id}.obj"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_text("# mesh\n", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts" / "dual_grasp_planning" / "plumbers_block"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "assembly_sequence.json").write_text(
        json.dumps(
            {
                "assembly": "plumbers_block",
                "base_part_id": "2",
                "steps": [
                    {
                        "step_id": "step_001_part_0",
                        "incoming_part_id": "0",
                        "holder_base_available": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "dual_grasp_pairs_step_001_part_0.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "configs" / "dual.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "grasp_assembly_action": {
                    "action_name": "/grasp_assembly",
                    "artifact_root": "artifacts/dual_grasp_planning",
                    "output_root": "artifacts/action",
                    "pose_base_topic": "/poses",
                    "holder_robot": "left",
                    "inserter_robot": "right",
                    "stop_after": "inserter_pickup_grasp",
                    "position_offset_m": [0.0, 0.0, 0.0],
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, config_path


def test_dual_goal_consumes_base_insertion_and_both_roles() -> None:
    goal = DualRobotGraspGoal.from_request(_goal())

    assert goal.assembly_name == "plumbers_block"
    assert goal.base_part_id == 2
    assert goal.insertion_part_id == 0
    assert goal.holder_robot == "left"
    assert goal.inserter_robot == "right"


def test_dual_runner_bootstraps_repo_package_for_ros_console_script(
    tmp_path: Path,
) -> None:
    script = f"""
import sys
from pathlib import Path
from robot_integration_ros.dual_grasp_assembly_action_runner import DualPipelineRunner

repo_root = Path({str(REPO_ROOT)!r})
assert str(repo_root) not in sys.path
DualPipelineRunner(
    repo_root=repo_root,
    config_path=repo_root / "configs" / "dual_grasp_planning.yaml",
    mode="pitl",
    allow_execution=False,
)
assert sys.path[0] == str(repo_root)
from grasp_planning.ros2.pose_listener import wait_for_debug_pose_item_messages
assert callable(wait_for_debug_pose_item_messages)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORKSPACE_PACKAGE_ROOT)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_dual_validation_rejects_wrong_base_step_or_role(
    tmp_path: Path,
) -> None:
    repo_root, config_path = _repo(tmp_path)
    runner = DualPipelineRunner(
        repo_root=repo_root,
        config_path=config_path,
        mode="pitl",
        allow_execution=False,
    )

    assert runner.validate(_goal()) is None
    assert "selected-order base" in runner.validate(_goal(base_part_id=0))
    assert "resolve exactly one" in runner.validate(_goal(insertion_part_id=2))
    assert "requires holder_robot='left'" in runner.validate(_goal(holder_robot="right", inserter_robot="left"))


def test_dual_real_requires_execution_acknowledgement(
    tmp_path: Path,
) -> None:
    repo_root, config_path = _repo(tmp_path)
    runner = DualPipelineRunner(
        repo_root=repo_root,
        config_path=config_path,
        mode="real",
        allow_execution=False,
    )
    outcome = runner.run(
        request=_goal(),
        goal_id="blocked",
        cancel_requested=lambda: False,
        publish_feedback=lambda _phase, _progress: None,
        publish_output=lambda _line: None,
    )
    assert outcome.error_code == "EXECUTION_DISABLED"

def test_dual_real_goal_builds_perceived_runtime_command_and_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root, config_path = _repo(tmp_path)
    runner = DualPipelineRunner(
        repo_root=repo_root,
        config_path=config_path,
        mode="real",
        allow_execution=True,
        allow_objectless_planning=True,
        pair_id="pair_1",
    )
    perceived = ObjectWorldPose(
        position_world=(0.5, 0.0, 0.0),
        orientation_xyzw_world=(0.0, 0.0, 0.0, 1.0),
    )
    monkeypatch.setattr(
        "grasp_planning.ros2.pose_listener.wait_for_debug_pose_item_messages",
        lambda **_kwargs: {2: perceived, 0: perceived},
    )
    monkeypatch.setattr(
        "grasp_planning.pipeline.dual_robot_simple_sim.resolve_planar_runtime_layout",
        lambda **_kwargs: PlanarRuntimeLayout(
            assembly_world=MovableFrame((0.55, 0.0, 0.0), 0.0),
            pickup_source_world_xy=(0.55, 0.28),
            pickup_orientation_rpy_deg=(0.0, 0.0, 0.0),
            perceived_part_aabbs=(
                RuntimePartAabb(
                    role="base",
                    minimum_world_m=(0.4, -0.1, -0.03),
                    maximum_world_m=(0.6, 0.1, 0.05),
                ),
                RuntimePartAabb(
                    role="incoming",
                    minimum_world_m=(0.6, 0.2, -0.03),
                    maximum_world_m=(0.7, 0.3, 0.02),
                ),
            ),
            warnings=("assembly tilt warning",),
        ),
    )
    commands = []
    output_lines = []
    debug_aabbs = []

    def _run_process(**kwargs):
        command = kwargs["command"]
        commands.append(command)
        task_path = Path(command[command.index("--task-output") + 1])
        attempt_path = Path(command[command.index("--attempt-output") + 1])
        task_path.write_text(
            json.dumps(
                {
                    "objects": {
                        "incoming": {
                            "preinsertion_source_pose_world": {
                                "position_world_m": [0.55, 0.0, 0.04],
                                "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        attempt_path.write_text(
            json.dumps(
                {
                    "result": {
                        "success": True,
                        "status": "stopped_at_inserter_preinsertion",
                        "message": "done",
                    }
                }
            ),
            encoding="utf-8",
        )
        return 0, [], False

    monkeypatch.setattr(runner, "_run_process", _run_process)
    outcome = runner.run(
        request=_goal(),
        goal_id="goal1",
        cancel_requested=lambda: False,
        publish_feedback=lambda _phase, _progress: None,
        publish_output=output_lines.append,
        publish_debug_aabbs=debug_aabbs.extend,
    )

    assert outcome.success is True
    assert outcome.grasped_position_xyz == (0.55, 0.0, 0.04)
    assert commands
    command = commands[0]
    assert command[command.index("--mode") + 1] == "real"
    assert "--reuse-moveit" in command
    assert "--execute" in command
    assert "--allow-objectless-planning" in command
    assert command[command.index("--pair-id") + 1] == "pair_1"
    assert command[command.index("--floor-z") + 1] == "-0.03"
    assert command[command.index("--stop-after") + 1] == "inserter_pickup_grasp"
    assert output_lines == ["[WARNING] assembly tilt warning"]
    assert [record["part_id"] for record in debug_aabbs] == [2, 0]
    assert [record["role"] for record in debug_aabbs] == [
        "base",
        "incoming",
    ]
    assert all(record["frame_id"] == "base_link" for record in debug_aabbs)


def test_dual_real_pickup_only_stop_is_not_reported_as_preinsertion(
    tmp_path: Path,
) -> None:
    repo_root, config_path = _repo(tmp_path)
    runner = DualPipelineRunner(
        repo_root=repo_root,
        config_path=config_path,
        mode="real",
        allow_execution=True,
        allow_objectless_planning=True,
    )
    task_path = tmp_path / "task.json"
    attempt_path = tmp_path / "attempt.json"
    task_path.write_text(json.dumps({"objects": {}}), encoding="utf-8")
    attempt_path.write_text(
        json.dumps(
            {
                "result": {
                    "success": True,
                    "status": "stopped_at_inserter_pickup_grasp",
                    "last_completed_phase": "inserter_pickup_grasp",
                }
            }
        ),
        encoding="utf-8",
    )

    outcome = runner._outcome_from_artifacts(
        return_code=0,
        task_path=task_path,
        attempt_path=attempt_path,
        output_lines=[],
        cancelled=False,
    )

    assert outcome.success is False
    assert outcome.error_code == "PARTIAL_EXECUTION"
    assert "inserter_pickup_grasp" in outcome.message
    assert outcome.grasped_position_xyz is None


def test_dual_real_result_uses_selected_fallback_candidate_pose(
    tmp_path: Path,
) -> None:
    repo_root, config_path = _repo(tmp_path)
    runner = DualPipelineRunner(
        repo_root=repo_root,
        config_path=config_path,
        mode="real",
        allow_execution=True,
        allow_objectless_planning=True,
    )

    def candidate(candidate_id: str, pair_id: str, x: float) -> dict[str, object]:
        return {
            "execution_candidate_id": candidate_id,
            "pair_id": pair_id,
            "objects": {
                "incoming": {
                    "preinsertion_source_pose_world": {
                        "position_world_m": [x, 0.0, 0.04],
                        "orientation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                    }
                }
            },
        }

    first = candidate("candidate_1", "pair_1", 0.51)
    second = candidate("candidate_2", "pair_2", 0.72)
    task_path = tmp_path / "task.json"
    attempt_path = tmp_path / "attempt.json"
    task_path.write_text(
        json.dumps({**first, "ranked_pair_candidates": [first, second]}),
        encoding="utf-8",
    )
    attempt_path.write_text(
        json.dumps(
            {
                "pair_id": "pair_2",
                "execution_candidate_id": "candidate_2",
                "pair_selection": {
                    "selected_rank": 2,
                    "selected_pair_id": "pair_2",
                    "selected_execution_candidate_id": "candidate_2",
                },
                "result": {
                    "success": True,
                    "status": "stopped_at_inserter_preinsertion",
                    "last_completed_phase": "inserter_preinsertion",
                },
            }
        ),
        encoding="utf-8",
    )

    outcome = runner._outcome_from_artifacts(
        return_code=0,
        task_path=task_path,
        attempt_path=attempt_path,
        output_lines=[],
        cancelled=False,
    )

    assert outcome.success is True
    assert outcome.grasped_position_xyz == (0.72, 0.0, 0.04)
    assert "pair_2" in outcome.message


def test_dual_runner_rejects_invalid_configured_stop_phase(
    tmp_path: Path,
) -> None:
    repo_root, config_path = _repo(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["grasp_assembly_action"]["stop_after"] = "unsupported"
    config_path.write_text(
        yaml.safe_dump(payload),
        encoding="utf-8",
    )

    try:
        DualPipelineRunner(
            repo_root=repo_root,
            config_path=config_path,
            mode="real",
            allow_execution=True,
        )
    except ValueError as exc:
        assert "grasp_assembly_action.stop_after" in str(exc)
    else:
        raise AssertionError("Expected invalid stop_after to fail.")
