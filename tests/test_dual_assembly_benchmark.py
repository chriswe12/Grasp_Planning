from __future__ import annotations

import json
from pathlib import Path

from scripts import run_dual_assembly_benchmark as benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict[str, object]:
    return benchmark._read_mapping(REPO_ROOT / "configs/dual_assembly_benchmark.yaml")


def test_default_benchmark_covers_every_step_side_position_and_orientation() -> None:
    specs = benchmark._case_specs(payload=_payload())

    assert len(specs) == 128
    assert {spec["incoming_part_id"] for spec in specs} == {"0", "3", "1", "4"}
    assert {spec["placement_id"] for spec in specs} == {
        "left_near",
        "left_far",
        "right_near",
        "right_far",
    }
    assert len({spec["orientation_id"] for spec in specs}) == 8
    assert all(spec["assembly_y"] == 0.0 for spec in specs)
    assert all(spec["assembly_yaw_deg"] == 0.0 for spec in specs)
    assert all(float(spec["pickup_x"]) > 0.0 for spec in specs)
    for spec in specs:
        expected_inserter = "lbr_one" if float(spec["pickup_y"]) < 0.0 else "lbr_two"
        assert spec["inserter_arm"] == expected_inserter
        assert spec["holder_arm"] != expected_inserter


def test_benchmark_command_is_headless_resumable_and_high_grip(tmp_path: Path) -> None:
    payload = _payload()
    spec = benchmark._case_specs(payload=payload, limit_cases=1)[0]
    paths = benchmark._case_paths(tmp_path, spec)

    command = benchmark._command(payload=payload, spec=spec, paths=paths)

    assert command[:3] == [str(REPO_ROOT / "run_simple_dual_robot.sh"), "--mode", "sim"]
    assert command[command.index("--inserter-arm") + 1] == "auto"
    assert command[command.index("--static-friction") + 1] == "5.0"
    assert command[command.index("--dynamic-friction") + 1] == "4.0"
    assert command[command.index("--gripper-effort-limit") + 1] == "200.0"
    assert command[command.index("--critical-damping-ratio") + 1] == "1.0"
    assert command[command.index("--gripper-close-duration-s") + 1] == "3.0"
    assert command[command.index("--finger-contact-min-force-n") + 1] == "0.25"
    assert command[command.index("--gripper-contact-preload-m") + 1] == "0.0004"
    assert "--headless" in command
    assert "--no-planning-debug-gui" in command
    assert command[command.index("--plan-output") + 1] == str(paths["plan"])
    assert command[command.index("--attempt-output") + 1] == str(paths["attempt"])
    assert paths["video"].suffix == ".webm"
    assert command[command.index("--record-video") + 1] == str(paths["video"])


def test_case_filters_select_one_named_matrix_cell() -> None:
    specs = benchmark._case_specs(
        payload=_payload(),
        selected_parts={"3"},
        selected_placements={"right_near"},
        selected_orientations={"upright_yaw_0"},
    )

    assert len(specs) == 1
    assert specs[0]["incoming_part_id"] == "3"
    assert specs[0]["placement_id"] == "right_near"
    assert specs[0]["orientation_id"] == "upright_yaw_0"
    assert specs[0]["inserter_arm"] == "lbr_two"


def test_incremental_outputs_keep_latest_case_state_and_embed_video(tmp_path: Path) -> None:
    specs = benchmark._case_specs(payload=_payload(), limit_cases=2)
    first = specs[0]
    events = tmp_path / "events.jsonl"
    running = {**first, "status": "running", "message": "active"}
    video = tmp_path / "cases" / str(first["case_id"]) / "scene.webm"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    completed = {
        **first,
        "status": "success",
        "success": True,
        "message": "done",
        "duration_s": 12.5,
        "video_path": str(video),
    }
    benchmark._append_jsonl(events, running)
    benchmark._append_jsonl(events, completed)
    latest = benchmark._latest_records(benchmark._jsonl_records(events))

    benchmark._refresh_outputs(output_dir=tmp_path, specs=specs, latest=latest)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    dashboard = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert summary["case_count"] == 2
    assert summary["completed_count"] == 1
    assert summary["success_count"] == 1
    assert latest[str(first["case_id"])]["status"] == "success"
    assert '<video controls preload="metadata">' in dashboard
    assert 'type="video/webm"' in dashboard
    assert "scene.webm" in dashboard
    assert "1 / 2" in dashboard


def test_dashboard_embeds_scene_image_and_failure_phase_without_video(tmp_path: Path) -> None:
    spec = benchmark._case_specs(payload=_payload(), limit_cases=1)[0]
    image = tmp_path / "cases" / str(spec["case_id"]) / "failure_scene.svg"
    image.parent.mkdir(parents=True)
    image.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    latest = {
        str(spec["case_id"]): {
            **spec,
            "status": "failed",
            "success": False,
            "duration_s": 2.0,
            "failure_stage": "holder_base_grasp",
            "failure_phase_label": "Holder/base grasp",
            "message": "holder did not establish contact",
            "image_path": str(image),
        }
    }

    benchmark._refresh_outputs(output_dir=tmp_path, specs=(spec,), latest=latest)

    dashboard = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert 'class="failure-image"' in dashboard
    assert "failure_scene.svg" in dashboard
    assert "Holder/base grasp" in dashboard


def test_csv_removes_sparse_log_nul_bytes(tmp_path: Path) -> None:
    spec = benchmark._case_specs(payload=_payload(), limit_cases=1)[0]
    latest = {
        str(spec["case_id"]): {
            **spec,
            "status": "failed",
            "success": False,
            "message": "planner stopped\x00during cleanup",
        }
    }

    benchmark._refresh_outputs(output_dir=tmp_path, specs=(spec,), latest=latest)

    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "\x00" not in csv_text
    assert "planner stoppedduring cleanup" in csv_text


def test_failure_phase_classifies_holder_and_grounded_pickup_failures() -> None:
    assert benchmark._failure_phase("holder_pregrasp: IK failed", has_plan=True) == (
        "holder_base_grasp",
        "Holder/base grasp",
    )
    assert benchmark._failure_phase(
        "No compatible grasp pairs remain after checking the grounded pickup pose; accepted=0/777",
    ) == ("incoming_grasp_planning", "Incoming-part grasp planning")


def test_failure_scene_renderer_draws_assembly_and_incoming_part(tmp_path: Path) -> None:
    payload = _payload()
    spec = benchmark._case_specs(payload=payload, limit_cases=1)[0]
    output = tmp_path / "failure_scene.svg"

    benchmark._render_failure_scene(
        path=output,
        payload=payload,
        spec=spec,
        plan_path=tmp_path / "missing-plan.json",
        failure_label="Incoming-part grasp planning",
        message="No compatible pickup grasps.",
    )

    source = output.read_text(encoding="utf-8")
    assert source.startswith("<svg")
    assert "assembled part 2" in source
    assert "incoming part 0" in source
    assert "Incoming-part grasp planning" in source


def test_wrapper_exposes_role_and_contact_physics_options() -> None:
    source = (REPO_ROOT / "run_simple_dual_robot.sh").read_text(encoding="utf-8")

    for flag in (
        "--inserter-arm",
        "--static-friction",
        "--dynamic-friction",
        "--gripper-effort-limit",
        "--critical-damping-ratio",
        "--gripper-close-duration-s",
        "--finger-contact-min-force-n",
        "--gripper-contact-preload-m",
    ):
        assert flag in source
