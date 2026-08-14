from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from scripts import run_dual_assembly_benchmark as benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict[str, object]:
    return benchmark._read_mapping(REPO_ROOT / "configs/dual_assembly_benchmark.yaml")


def test_default_benchmark_covers_every_step_side_position_and_orientation() -> None:
    specs = benchmark._case_specs(payload=_payload())

    assert len(specs) == 384
    assert {spec["incoming_part_id"] for spec in specs} == {"0", "3", "1", "4"}
    assert {spec["placement_id"] for spec in specs} == {
        "left_inner_front",
        "left_inner_middle",
        "left_inner_back",
        "left_outer_front",
        "left_outer_middle",
        "left_outer_back",
        "right_inner_front",
        "right_inner_middle",
        "right_inner_back",
        "right_outer_front",
        "right_outer_middle",
        "right_outer_back",
    }
    assert len({spec["orientation_id"] for spec in specs}) == 8
    assert all(spec["assembly_y"] == 0.0 for spec in specs)
    assert all(spec["assembly_yaw_deg"] == 0.0 for spec in specs)
    assert all(float(spec["pickup_x"]) > 0.0 for spec in specs)
    assert all(str(spec["incoming_mesh_path"]).endswith(f"/{spec['incoming_part_id']}.obj") for spec in specs)
    assert all(float(spec["incoming_mesh_scale"]) == 0.01 for spec in specs)
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
    assert command[command.index("--artifact-root") + 1] == str(
        (REPO_ROOT / "artifacts/dual_grasp_planning").resolve()
    )
    assert command[command.index("--inserter-arm") + 1] == "auto"
    assert command[command.index("--max-pair-attempts") + 1] == "256"
    assert command[command.index("--max-ik-screen-candidates") + 1] == "0"
    assert command[command.index("--static-friction") + 1] == "5.0"
    assert command[command.index("--dynamic-friction") + 1] == "4.0"
    assert command[command.index("--gripper-effort-limit") + 1] == "200.0"
    assert command[command.index("--critical-damping-ratio") + 1] == "1.0"
    assert command[command.index("--gripper-close-duration-s") + 1] == "3.0"
    assert command[command.index("--finger-contact-min-force-n") + 1] == "0.25"
    assert command[command.index("--gripper-contact-preload-m") + 1] == "0.0004"
    assert command[command.index("--ik-solver") + 1] == "kdl"
    assert command[command.index("--ik-timeout-s") + 1] == "0.35"
    assert command[command.index("--exact-ik-candidates") + 1] == "7"
    assert command[command.index("--exact-ik-beam-width") + 1] == "4"
    assert command[command.index("--exact-ik-seed-perturbation-rad") + 1] == "0.6"
    assert command[command.index("--pickup-approach-ik-steps") + 1] == "5"
    assert command[command.index("--pickup-pregrasp-offsets-m") + 1] == (
        "0.1,0.075,0.05,0.025"
    )
    assert command[command.index("--planning-time-s") + 1] == "15.0"
    assert command[command.index("--planning-attempts") + 1] == "16"
    assert "--headless" in command
    assert "--no-planning-debug-gui" in command
    assert command[command.index("--plan-output") + 1] == str(paths["plan"])
    assert command[command.index("--attempt-output") + 1] == str(paths["attempt"])
    assert paths["video"].suffix == ".webm"
    assert paths["thumbnail"].name == "scene_thumbnail.jpg"
    assert command[command.index("--record-video") + 1] == str(paths["video"])


def test_benchmark_command_forwards_alternate_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stage3"
    payload = _payload()
    payload["benchmark"] = {
        **dict(payload["benchmark"]),
        "artifact_root": str(artifact_root),
    }
    spec = {
        **benchmark._case_specs(payload=_payload(), limit_cases=1)[0],
        "assembly": "plumbers_block",
    }
    paths = benchmark._case_paths(tmp_path / "output", spec)

    command = benchmark._command(payload=payload, spec=spec, paths=paths)

    assert command[command.index("--artifact-root") + 1] == str(artifact_root.resolve())


def test_planning_only_command_skips_isaac_and_recording_options(tmp_path: Path) -> None:
    payload = _payload()
    spec = benchmark._case_specs(payload=payload, limit_cases=1)[0]
    paths = benchmark._case_paths(tmp_path, spec)

    command = benchmark._command(payload=payload, spec=spec, paths=paths, planning_only=True)

    assert "--planning-only" in command
    assert "--plan-output" in command
    assert "--attempt-output" not in command
    assert "--record-video" not in command
    assert "--isaac-python" not in command
    assert "--static-friction" not in command


def test_plan_summary_aggregates_exact_ik_collision_diagnostics(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "kind": "dual_robot_simple_sim_plan_failure",
                "ik_preflight": {
                    "ik_seed_calls": 3,
                    "ik_solutions_found": 1,
                    "collision_diagnostics": {
                        "ik_requests": 3,
                        "kinematic_cache_hits": 5,
                        "kinematic_cache_misses": 3,
                        "ik_request_duration_s": 0.25,
                        "collision_disabled_ik_solutions": 2,
                        "kinematic_or_numerical_failures": 1,
                        "state_validity_requests": 2,
                        "valid_states": 1,
                        "invalid_states": 1,
                        "invalid_states_without_contacts": 0,
                        "contact_class_counts": {"finger_floor": 1},
                        "contact_pair_counts": {
                            "dual_sim_work_surface <-> lbr_one_left_finger_link": 1,
                        },
                    },
                    "records": {
                        "holder": [
                            {
                                "targets": [
                                    {
                                        "target": "holder_pregrasp",
                                        "ok": False,
                                        "seed_attempts": 3,
                                        "kinematic_cache_hits": 5,
                                        "kinematic_cache_misses": 3,
                                        "collision_diagnostics": {
                                            "collision_disabled_ik_solutions": 2,
                                            "kinematic_or_numerical_failures": 1,
                                            "valid_states": 0,
                                            "invalid_states": 2,
                                            "invalid_states_without_contacts": 0,
                                            "contact_class_counts": {"finger_floor": 2},
                                            "contact_pair_counts": {
                                                "dual_sim_work_surface <-> lbr_one_left_finger_link": 2,
                                            },
                                        },
                                    }
                                ]
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = benchmark._plan_summary(plan_path)
    aggregate = benchmark._aggregate_ik_diagnostics([summary])

    assert summary["ik_failure_target_counts"] == {"holder:holder_pregrasp": 1}
    assert aggregate["ik_requests"] == 3
    assert aggregate["kinematic_cache_hits"] == 5
    assert aggregate["kinematic_cache_misses"] == 3
    assert aggregate["kinematic_or_numerical_failures"] == 1
    assert aggregate["invalid_states"] == 1
    assert aggregate["contact_class_counts"] == {"finger_floor": 1}
    assert aggregate["target_diagnostics"]["holder:holder_pregrasp"]["invalid_states"] == 2
    assert aggregate["target_diagnostics"]["holder:holder_pregrasp"]["kinematic_cache_hits"] == 5
    assert aggregate["target_diagnostics"]["holder:holder_pregrasp"]["ik_requests"] == 3


def test_plan_summary_keeps_joint_path_failure_primary_over_fallback_ik(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "kind": "dual_robot_simple_sim_plan_failure",
                "attempts": [
                    {
                        "success": False,
                        "failure": (
                            "ik_preflight: holder grasp h1 failed holder_pregrasp: "
                            "IK failed with code=-31"
                        ),
                    },
                    {
                        "success": False,
                        "failure": (
                            "ik_preflight: inserter grasp i1 failed inserter_pickup_pregrasp: "
                            "kinematic IK state is invalid (lbr_one_link_5 <-> work_surface)"
                        ),
                    },
                    {
                        "success": False,
                        "failure": (
                            "inserter_pickup_grasp: preferred joint target failed "
                            "(Planning failed with code=99999); pose fallback: "
                            "IK failed with code=-31"
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = benchmark._plan_summary(plan_path)

    assert summary["moveit_failure_kind"] == "joint_path_planning"
    assert summary["moveit_failure_label"] == "Joint-space path planning"
    assert summary["moveit_failure_kind_counts"] == {
        "exact_ik_preflight": 1,
        "complete_state_collision": 1,
        "joint_path_planning": 1,
    }
    assert summary["moveit_fallback_failure_kind_counts"] == {"fallback_pose_ik": 1}
    assert summary["moveit_primary_failure_message"] == (
        "inserter_pickup_grasp: Planning failed with code=99999"
    )
    assert "IK failed" not in str(summary["moveit_primary_failure_message"])
    assert summary["moveit_fallback_failure_kind"] == "fallback_pose_ik"
    assert summary["moveit_fallback_failure_message"] == "IK failed with code=-31"
    assert benchmark._failure_phase(
        "MoveIt could not plan any of 256 ranked pairs",
        has_plan=True,
        moveit_failure_kind=str(summary["moveit_failure_kind"]),
    ) == ("joint_path_planning", "Joint-space path planning")


def test_plan_summary_classifies_standalone_pose_ik_failure(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "kind": "dual_robot_simple_sim_plan_failure",
                "attempts": [
                    {
                        "success": False,
                        "failure": "holder_pregrasp: IK failed with code=-31",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = benchmark._plan_summary(plan_path)

    assert summary["moveit_failure_kind"] == "fallback_pose_ik"
    assert summary["moveit_primary_failure_message"] == "holder_pregrasp: IK failed with code=-31"
    assert summary["moveit_fallback_failure_kind"] == ""


def test_successful_plan_does_not_promote_rejected_candidate_to_case_failure(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "kind": "dual_robot_simple_sim_task",
                "moveit": {
                    "attempts": [
                        {
                            "success": False,
                            "failure": "ik_preflight: holder_pregrasp: IK failed with code=-31",
                        },
                        {"success": True, "failure": ""},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    summary = benchmark._plan_summary(plan_path)

    assert summary["moveit_failure_kind"] == ""
    assert summary["moveit_failure_kind_counts"] == {"exact_ik_preflight": 1}


def test_case_filters_select_one_named_matrix_cell() -> None:
    specs = benchmark._case_specs(
        payload=_payload(),
        selected_parts={"3"},
        selected_placements={"right_inner_middle"},
        selected_orientations={"upright_yaw_0"},
    )

    assert len(specs) == 1
    assert specs[0]["incoming_part_id"] == "3"
    assert specs[0]["placement_id"] == "right_inner_middle"
    assert specs[0]["orientation_id"] == "upright_yaw_0"
    assert specs[0]["inserter_arm"] == "lbr_two"


def test_failed_summary_selects_only_previous_failures_in_matrix_order(tmp_path: Path) -> None:
    specs = benchmark._case_specs(payload=_payload(), limit_cases=3)
    summary_path = tmp_path / "previous-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "records": [
                    {"case_id": specs[2]["case_id"], "status": "failed"},
                    {"case_id": specs[0]["case_id"], "status": "failed"},
                    {"case_id": specs[1]["case_id"], "status": "success"},
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = benchmark._failed_case_specs_from_summary(
        specs,
        summary_path=summary_path,
    )

    assert [spec["case_id"] for spec in selected] == [specs[0]["case_id"], specs[2]["case_id"]]


def test_failed_summary_can_select_one_failure_stage(tmp_path: Path) -> None:
    specs = benchmark._case_specs(payload=_payload(), limit_cases=3)
    summary_path = tmp_path / "previous-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "case_id": specs[0]["case_id"],
                        "status": "failed",
                        "failure_stage": "moveit_candidate_planning",
                    },
                    {
                        "case_id": specs[1]["case_id"],
                        "status": "failed",
                        "failure_stage": "transition",
                    },
                    {"case_id": specs[2]["case_id"], "status": "success"},
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = benchmark._failed_case_specs_from_summary(
        specs,
        summary_path=summary_path,
        failure_stages={"moveit_candidate_planning"},
    )

    assert [spec["case_id"] for spec in selected] == [specs[0]["case_id"]]


def test_incremental_outputs_keep_latest_case_state_and_embed_video(tmp_path: Path) -> None:
    specs = benchmark._case_specs(payload=_payload(), limit_cases=2)
    first = specs[0]
    events = tmp_path / "events.jsonl"
    running = {**first, "status": "running", "message": "active"}
    video = tmp_path / "cases" / str(first["case_id"]) / "scene.webm"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    thumbnail = video.with_name("scene_thumbnail.jpg")
    thumbnail.write_bytes(b"jpeg")
    completed = {
        **first,
        "status": "success",
        "success": True,
        "message": "done",
        "duration_s": 12.5,
        "video_path": str(video),
        "thumbnail_path": str(thumbnail),
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
    assert '<video class="lazy-video" controls preload="none"' in dashboard
    assert 'poster="cases/' in dashboard
    assert 'data-src="cases/' in dashboard
    assert 'type="video/webm"' in dashboard
    assert "▶ Play recording" in dashboard
    assert "loadAndPlay" in dashboard
    assert "scene.webm" in dashboard
    assert "scene_thumbnail.jpg" in dashboard
    assert "1 / 2" in dashboard
    assert 'id="failure"' in dashboard
    assert 'id="placement"' in dashboard
    assert 'id="orientation"' in dashboard
    assert 'id="inserter"' in dashboard
    assert 'id="case-sort"' in dashboard
    assert 'id="group-by"' in dashboard
    assert 'id="group-sort"' in dashboard
    assert 'id="guide-dimension"' in dashboard
    assert 'id="breakdown-body"' in dashboard
    assert 'data-guide="placement"' in dashboard
    assert 'data-guide="orientation"' in dashboard
    assert "updateBreakdown" in dashboard
    assert "sortCards" in dashboard


def test_outputs_serialize_and_display_moveit_failure_taxonomy(tmp_path: Path) -> None:
    spec = benchmark._case_specs(payload=_payload(), limit_cases=1)[0]
    record = {
        **spec,
        "status": "failed",
        "success": False,
        "failure_stage": "joint_path_planning",
        "failure_phase_label": "Joint-space path planning",
        "failure_substage": "joint_path_planning",
        "failure_substage_label": "Joint-space path planning",
        "moveit_failure_kind": "joint_path_planning",
        "moveit_failure_label": "Joint-space path planning",
        "moveit_failure_kind_counts": {
            "exact_ik_preflight": 2,
            "complete_state_collision": 3,
            "joint_path_planning": 1,
        },
        "moveit_primary_failure_message": "inserter_pickup_grasp: Planning failed with code=99999",
        "moveit_fallback_failure_kind": "fallback_pose_ik",
        "moveit_fallback_failure_message": "IK failed with code=-31",
        "moveit_fallback_failure_kind_counts": {"fallback_pose_ik": 1},
        "message": "MoveIt could not plan any ranked pair.",
    }

    benchmark._refresh_outputs(
        output_dir=tmp_path,
        specs=(spec,),
        latest={str(spec["case_id"]): record},
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    dashboard = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert summary["moveit_failure_taxonomy"] == {
        "case_kind_counts": {"joint_path_planning": 1},
        "attempt_kind_counts": {
            "exact_ik_preflight": 2,
            "complete_state_collision": 3,
            "joint_path_planning": 1,
        },
        "fallback_kind_counts": {"fallback_pose_ik": 1},
    }
    assert "moveit_primary_failure_message" in csv_text.splitlines()[0]
    assert 'data-failure="joint_path_planning"' in dashboard
    assert "Failed at: Joint-space path planning" in dashboard
    assert "inserter_pickup_grasp: Planning failed with code=99999" in dashboard
    assert "fallback diagnostic" in dashboard
    assert "IK failed with code=-31" in dashboard


def test_video_thumbnail_extraction_uses_a_real_recording_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "scene.webm"
    source.write_bytes(b"video")
    target = tmp_path / "scene_thumbnail.jpg"
    commands = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(command)
        Path(command[-1]).write_bytes(b"jpeg")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)

    benchmark._extract_video_thumbnail(
        ffmpeg=Path("/fake/ffmpeg"),
        source=source,
        target=target,
    )

    assert target.read_bytes() == b"jpeg"
    assert "-ss" in commands[0]
    assert "-frames:v" in commands[0]
    assert "scale=640:-2:flags=lanczos" in commands[0]


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
    assert 'class="failure-image-button"' in dashboard
    assert 'data-full-src="cases/' in dashboard
    assert "⛶ Enlarge scene" in dashboard
    assert '<dialog id="image-viewer">' in dashboard
    assert "viewer.showModal()" in dashboard
    assert "viewer.close()" in dashboard
    assert "failure_scene.svg" in dashboard
    assert "Holder/base grasp" in dashboard
    assert 'data-failure="holder_base_grasp"' in dashboard
    assert "Failed at: Holder/base grasp" in dashboard


def test_dashboard_visual_guides_explain_part_orientation_and_location() -> None:
    specs = benchmark._case_specs(payload=_payload(), limit_cases=2)

    part_icon = benchmark._part_icon_svg(specs[0])
    orientation_icon = benchmark._orientation_icon_svg(specs[0])
    placement_icon = benchmark._placement_icon_svg(specs[0])
    failure_icon = benchmark._failure_stage_icon_svg("transition", "Transport to pre-insertion")

    assert '<svg class="guide-svg"' in part_icon
    assert "Incoming part 0" in part_icon
    assert '<svg class="guide-svg"' in orientation_icon
    assert ">X</text>" in orientation_icon
    assert ">Y</text>" in orientation_icon
    assert ">Z</text>" in orientation_icon
    assert "pickup 0.44, -0.18 metres" in placement_icon
    assert "assembly" in placement_icon
    assert "LBR one" in placement_icon
    assert "Transport to pre-insertion" in failure_icon
    assert ">Move</text>" in failure_icon


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


def test_failure_phase_classifies_existing_moveit_stack_as_setup() -> None:
    message = (
        "[DUAL-RUN] Stop it first, or pass --reuse-moveit after confirming it matches mode=sim."
    )

    assert benchmark._failure_phase(message) == ("setup", "MoveIt/Isaac setup")
    assert benchmark._is_existing_stack_ownership_conflict(message)
    assert not benchmark._is_existing_stack_ownership_conflict(
        "--reuse-moveit was requested, but the live MoveIt services are not ready"
    )


def test_benchmark_stops_after_recording_first_existing_stack_conflict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    specs = benchmark._case_specs(payload=_payload(), limit_cases=2)
    calls: list[str] = []
    monkeypatch.setattr(
        benchmark,
        "_parse_args",
        lambda: SimpleNamespace(
            config=REPO_ROOT / "configs/dual_assembly_benchmark.yaml",
            artifact_root=None,
            output_dir=tmp_path,
            parts=None,
            placements=None,
            orientations=None,
            limit_cases=2,
            failed_from_summary=None,
            failure_stages=None,
            planning_only=True,
            ik_only=True,
            ik_collision_diagnostics=False,
            no_resume=True,
            retry_failed=False,
            repair_videos=False,
            repair_failure_evidence=False,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(benchmark, "_case_specs", lambda **_kwargs: specs)
    monkeypatch.setattr(benchmark, "_refresh_outputs", lambda **_kwargs: None)

    def fake_run_case(*, spec, **_kwargs):
        calls.append(str(spec["case_id"]))
        return (
            {
                **spec,
                "status": "failed",
                "success": False,
                "message": (
                    "[DUAL-RUN] Stop it first, or pass --reuse-moveit after "
                    "confirming it matches mode=sim."
                ),
                "duration_s": 0.1,
            },
            False,
        )

    monkeypatch.setattr(benchmark, "_run_case", fake_run_case)

    assert benchmark.main() == 2
    assert calls == [str(specs[0]["case_id"])]
    records = benchmark._jsonl_records(tmp_path / "events.jsonl")
    assert [record["status"] for record in records] == ["running", "failed"]


def test_case_cleanup_terminates_exact_nested_process_group_only(
    tmp_path: Path,
) -> None:
    nested_pid_path = tmp_path / "nested.pid"
    handoff_path = tmp_path / "moveit-process-group"
    nested_code = (
        "import signal,time; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(120)"
    )
    outer_code = (
        "import pathlib,signal,subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable, '-c', {nested_code!r}], start_new_session=True); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(130)); "
        "time.sleep(120)"
    )
    outer = subprocess.Popen(
        [sys.executable, "-c", outer_code, str(nested_pid_path)],
        start_new_session=True,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
    )
    nested_pid = 0
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not nested_pid_path.is_file():
            time.sleep(0.02)
        assert nested_pid_path.is_file()
        nested_pid = int(nested_pid_path.read_text(encoding="utf-8"))
        nested_start_time = benchmark._process_start_time_ticks(nested_pid)
        outer_start_time = benchmark._process_start_time_ticks(outer.pid)
        assert nested_start_time is not None
        assert outer_start_time is not None
        assert os.getpgid(nested_pid) == nested_pid
        handoff_path.write_text(
            f"{nested_pid} {nested_start_time}\n",
            encoding="utf-8",
        )

        benchmark._terminate_process_group(
            outer,
            owned_process_group_files=(handoff_path,),
            process_group_start_time_ticks=outer_start_time,
            timeout_s=0.15,
            term_timeout_s=0.15,
            kill_timeout_s=3.0,
        )

        assert outer.poll() is not None
        assert not benchmark._process_group_exists(nested_pid)
        assert unrelated.poll() is None
    finally:
        for process_group_id in (nested_pid, outer.pid, unrelated.pid):
            if process_group_id <= 1:
                continue
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in (outer, unrelated):
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)


def test_dual_moveit_wrappers_handoff_exact_nested_process_group() -> None:
    runner = (REPO_ROOT / "run_simple_dual_robot.sh").read_text(encoding="utf-8")
    launcher = (REPO_ROOT / "start_dual_lbr_moveit.sh").read_text(encoding="utf-8")

    assert "DUAL_MOVEIT_PROCESS_GROUP_FILE" in runner
    assert 'START_ARGS+=(--process-group-file "${MOVEIT_PROCESS_GROUP_FILE}")' in runner
    assert 'kill -TERM -- "-${moveit_process_group}"' in runner
    assert 'kill -KILL -- "-${moveit_process_group}"' in runner
    assert "--process-group-file" in launcher
    assert "launch_start_time" in launcher


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
        "--planning-only",
        "--ik-solver",
        "--ik-timeout-s",
        "--exact-ik-candidates",
        "--exact-ik-beam-width",
        "--exact-ik-seed-perturbation-rad",
        "--pickup-approach-ik-steps",
        "--pickup-pregrasp-offsets-m",
        "--planning-time-s",
        "--planning-attempts",
    ):
        assert flag in source
