from __future__ import annotations

import json
from pathlib import Path

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
    assert command[command.index("--inserter-arm") + 1] == "auto"
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
    assert command[command.index("--planning-time-s") + 1] == "15.0"
    assert command[command.index("--planning-attempts") + 1] == "16"
    assert "--headless" in command
    assert "--no-planning-debug-gui" in command
    assert command[command.index("--plan-output") + 1] == str(paths["plan"])
    assert command[command.index("--attempt-output") + 1] == str(paths["attempt"])
    assert paths["video"].suffix == ".webm"
    assert paths["thumbnail"].name == "scene_thumbnail.jpg"
    assert command[command.index("--record-video") + 1] == str(paths["video"])


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
        "--planning-time-s",
        "--planning-attempts",
    ):
        assert flag in source
