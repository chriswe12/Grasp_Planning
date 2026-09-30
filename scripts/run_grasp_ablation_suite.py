#!/usr/bin/env python3
"""Resumable, isolated thesis experiments; never connects to robot hardware.

Subcommands: prepare, run, worker, report. Results describe bounded geometric
search, except the explicitly labelled native-MuJoCo execution experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SUCCESS = {"direct_success", "fallback_success", "handover_fallback_success"}
REFERENCE_DIRS = [
    "artifacts/grasp_generation_fallback_franka_20260922_174714",
    "artifacts/grasp_generation_fallback_franka_remaining_jobs12_20260923_095500",
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_candidates(candidates, mode, seed=0):
    """Reorder a fixed feasible set; removing a term renormalizes its weights."""
    import numpy as np

    values = list(candidates)
    if mode == "full":
        return values
    if mode == "random":
        rng = np.random.default_rng(seed)
        return [values[int(i)] for i in rng.permutation(len(values))]
    weights = {"antipodal_alignment": 0.40, "centering": 0.25, "contact_support": 0.20, "com_offset": 0.15}
    if mode.startswith("without_") and mode != "without_top_down":
        weights.pop(mode.removeprefix("without_"))

    def score(c):
        s = c.score_components or {}
        object_score = sum(w * float(s[k]) for k, w in weights.items()) / sum(weights.values())
        if mode == "without_top_down":
            return object_score
        top = float(s.get("top_grasp_score_weight", 0.35))
        reach = float(s.get("reachability_proxy_score_weight", 0))
        return (
            (1 - top - reach) * object_score
            + top * float(s.get("top_down_approach", 0))
            + reach * float(s.get("reachability_proxy", 0))
        )

    return sorted(values, key=lambda c: (-score(c), c.grasp_id))


def read_reference(repo):
    rows, configs = [], []
    for relative in REFERENCE_DIRS:
        folder = repo / relative
        data = json.loads((folder / "results.json").read_text())
        configs.append(data["config"])
        for row in data["orientations"]:
            r = deepcopy(row)
            r["source_root"] = str(folder)
            r["source_stage2"] = str(folder / row["links"]["stage2_json"])
            r["source_stage1"] = str(folder / "parts" / row["assembly"] / row["part_id"] / "stage1/grasps.json")
            r["source_raw"] = str(Path(r["source_stage1"]).with_name("raw_grasps.json"))
            rows.append(r)
    keys = [(r["assembly"], r["part_id"], r["orientation_id"]) for r in rows]
    assert len(keys) == len(set(keys)) == 264
    return rows, configs[0]


def prepare(root, workers):
    from types import SimpleNamespace

    from scripts import run_grasp_generation_benchmark as bench

    root = root.resolve()
    if (root / "manifest.json").exists():
        raise RuntimeError("Existing suite; use run to resume, do not overwrite.")
    root.mkdir(parents=True, exist_ok=True)
    rows, base = read_reference(ROOT)
    base = deepcopy(base)
    base["planning"]["stage1_cache_enabled"] = False
    base["fallback"]["enabled"] = False
    base["handover_fallback"]["enabled"] = False
    targets = bench._discover_targets(base, SimpleNamespace(assembly=[], part=[], target=[], limit_parts=None))
    # Freeze the orientation identities before running any search ablation.
    expected = defaultdict(list)
    for r in rows:
        expected[r["assembly"] + "/" + r["part_id"]].append(r["orientation_id"])
    experiments = []

    def add(name, changes=None, extras=None, note="", priority=10):
        cfg = deepcopy(base)
        cfg["planning"].update(changes or {})
        experiments.append(dict(name=name, config=cfg, extras=extras or [], note=note, priority=priority))

    add(
        "reference_s0",
        extras=[
            dict(name="no_symmetry_s0", changes={"symmetry_pickup_enabled": False}),
            dict(name="floor_0mm_diagnostic", changes={"floor_clearance_margin_m": 0.0}),
            dict(name="floor_5mm_diagnostic", changes={"floor_clearance_margin_m": 0.005}),
            dict(name="floor_20mm_diagnostic", changes={"floor_clearance_margin_m": 0.020}),
        ],
        priority=0,
    )
    add("offset_center_s0", {"contact_lateral_offsets_m": [0.0], "contact_approach_offsets_m": [0.0]}, priority=1)
    add(
        "offset_wide_s0",
        {
            "contact_lateral_offsets_m": [
                -0.005833333333333334,
                -0.002916666666666667,
                0,
                0.002916666666666667,
                0.005833333333333334,
            ],
            "contact_approach_offsets_m": [
                -0.006166666666666667,
                -0.0030833333333333333,
                0,
                0.0030833333333333333,
                0.006166666666666667,
            ],
        },
        priority=2,
    )
    add("no_upright_s0", {"stage1_upright_axes_enabled": False}, priority=3)
    add("base_upright_only_s0", note="Retain base upright axes, omit stable-pose-specific axes.", priority=4)
    for n in [256, 512, 2048, 128]:
        add(f"samples_{n}_s0", {"num_surface_samples": n}, priority=5)
    for deg in [60, 30, 7.5]:
        add(f"roll_{str(deg).replace('.', 'p')}_s0", {"roll_angle_step_deg": deg}, priority=6)
    add(
        "offset_dense_s0",
        {
            "contact_lateral_offsets_m": [
                -0.002916666666666667,
                -0.0014583333333333335,
                0,
                0.0014583333333333335,
                0.002916666666666667,
            ],
            "contact_approach_offsets_m": [
                -0.0030833333333333333,
                -0.0015416666666666667,
                0,
                0.0015416666666666667,
                0.0030833333333333333,
            ],
        },
        priority=7,
    )
    for cap in [4096, 10240, 81920, 163840]:
        add(f"pairs_{cap}_s0", {"max_pair_checks": cap}, priority=8)
    add("samples_4096_s0", {"num_surface_samples": 4096}, priority=9)
    add("samples_4096_pairs_163840_s0", {"num_surface_samples": 4096, "max_pair_checks": 163840}, priority=10)
    for seed in [1, 2]:
        add(
            f"reference_s{seed}",
            {"rng_seed": seed},
            extras=[dict(name=f"no_symmetry_s{seed}", changes={"symmetry_pickup_enabled": False})],
            priority=5,
        )
        for n in [256, 512, 2048]:
            add(f"samples_{n}_s{seed}", {"rng_seed": seed, "num_surface_samples": n}, priority=11)
        for source in ["offset_center_s0", "offset_wide_s0", "no_upright_s0"]:
            cfg = next(e for e in experiments if e["name"] == source)["config"]["planning"]
            add(source[:-1] + str(seed), {**cfg, "rng_seed": seed}, priority=12)
    add(
        "final_pose_only_diagnostic_s0",
        note="Omit insertion sweep during generation, then audit the exact retained offsets against the full sweep.",
        priority=9,
    )
    add(
        "pdz_reference_s0",
        {
            "gripper_collision_model": "pdz_gripper",
            "min_jaw_width": 0.008,
            "max_jaw_width": 0.062,
            "detailed_finger_contact_gap_m": 0.005,
        },
        priority=13,
        note="Gripper-specific operating range; not a geometry-only comparison.",
    )
    jobs = []
    for e in sorted(experiments, key=lambda e: e["priority"]):
        for target in targets:
            key = target.assembly + "/" + target.part_id
            jobs.append(
                dict(
                    id=e["name"] + "/" + key,
                    kind="generation",
                    experiment=e["name"],
                    target=asdict(target),
                    config=e["config"],
                    extras=e["extras"],
                    expected=expected[key],
                    slots=4
                    if e["config"]["planning"]["num_surface_samples"] >= 4096
                    else (2 if e["config"]["planning"]["num_surface_samples"] >= 2048 else 1),
                    priority=e["priority"],
                    timeout_s=5400,
                )
            )
    # These use the exact saved baseline candidate pools; independent H and R searches.
    failures = [r for r in rows if r["status"] != "direct_success"]
    for r in failures:
        key = r["assembly"] + "/" + r["part_id"] + "/" + r["orientation_id"]
        jobs.append(
            dict(id="handover/" + key, kind="handover", reference=r, config=base, slots=1, priority=4, timeout_s=1800)
        )
        jobs.append(
            dict(id="regrasp/" + key, kind="regrasp", reference=r, config=base, slots=2, priority=14, timeout_s=7200)
        )
    # Fixed stratified execution subset; native controller is labelled separately.
    direct = defaultdict(list)
    for r in rows:
        if r["status"] == "direct_success":
            direct[r["assembly"]].append(r)
    for assembly, candidates in sorted(direct.items()):
        chosen = sorted(candidates, key=lambda r: (r["part_id"], r["orientation_id"]))
        for r in chosen:
            jobs.append(
                dict(
                    id="execution/" + assembly + "/" + r["part_id"] + "/" + r["orientation_id"],
                    kind="execution",
                    reference=r,
                    config=base,
                    slots=1,
                    priority=7,
                    timeout_s=3600,
                )
            )
    # Interleave parts across experiment families at the same priority.
    jobs.sort(key=lambda j: (j["priority"], j.get("target", {}).get("part_id", ""), j["id"]))
    snapshot = root / "source"
    snapshot.mkdir()
    shutil.copytree(ROOT / "grasp_planning", snapshot / "grasp_planning", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "scripts", snapshot / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", snapshot / "configs")
    shutil.copytree(
        ROOT / "ros2_ws/src/robot_integration_ros",
        snapshot / "ros2_ws/src/robot_integration_ros",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (snapshot / "assets").symlink_to(ROOT / "assets", target_is_directory=True)
    (snapshot / ".cache").symlink_to(ROOT / ".cache", target_is_directory=True)
    # Robot config resolves its model in the snapshot; assets are hash-recorded below.
    files = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob("*.py")}
    asset_files = set()
    for target in targets:
        folder = ROOT / "assets" / Path(target.target_mesh_path).parent
        asset_files.update(folder.glob("*.obj"))
        asset_files.update(folder.glob("*.json"))
    for folder in ["assets/urdf/franka_description", "assets/urdf/kuka_iiwa7_pdz_gripper"]:
        if (ROOT / folder).exists():
            asset_files.update(p for p in (ROOT / folder).rglob("*") if p.is_file())
    asset_files.update((ROOT / ".cache/generated_mujoco_models").glob("*.xml"))
    provenance = dict(
        created=now(),
        repo=str(ROOT),
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=files,
        asset_hashes={str(p): digest(p) for p in sorted(asset_files)},
        reference_hashes={str(ROOT / x / "results.json"): digest(ROOT / x / "results.json") for x in REFERENCE_DIRS},
    )
    save(root / "provenance.json", provenance)
    save(root / "reference.json", dict(rows=rows, expected=dict(expected)))
    manifest = dict(
        schema_version=1,
        root=str(root),
        workers=workers,
        created=now(),
        snapshot=str(snapshot),
        experiments=experiments,
        jobs=jobs,
        notes=[
            "Coverage is bounded-search geometric feasibility, not execution success.",
            "Fresh timing excludes HTML and includes per-stage wall time under shared CPU load; not isolated latency.",
            "All direct sweeps use identical baseline orientation identities; task errors and timeouts are not geometric failures.",
            "Wider offsets and relaxed floor margins are experimental settings, not promoted defaults.",
            "Regrasp/handover searches independently use baseline direct failures; retain overlap and truncation.",
            "Native MuJoCo execution covers the 79 baseline direct-success poses; conditional rates exclude generation failures and are separate from MoveIt/Isaac/hardware validation.",
            "Asset and source hashes pin provenance; source snapshot is used by workers.",
            "Three seeds for primary sampling/offset/upright comparisons; exploratory sweeps use seed zero.",
        ],
    )
    save(root / "manifest.json", manifest)
    refresh_report(root)
    print(root, "jobs=", len(jobs), "generation variants=", len(experiments), flush=True)


def build_stage1(job, folder):
    from grasp_planning.grasping.fabrica_grasp_debug import canonicalize_target_mesh, load_asset_mesh
    from grasp_planning.pipeline import GeometryConfig, generate_stage1_result, write_stage1_artifacts
    from grasp_planning.pipeline.stable_orientations import enumerate_stable_orientations
    from scripts import run_grasp_generation_benchmark as b
    from scripts.run_grasp_pipeline import _planning_config

    target = b.TargetSpec(**job["target"])
    cfg = job["config"]
    planning = _planning_config(cfg)
    geometry = GeometryConfig(
        target_mesh_path=target.target_mesh_path,
        mesh_scale=cfg["geometry"]["mesh_scale"],
        assembly_glob=target.assembly_glob,
        assembly_obstacle_paths=target.assembly_obstacle_paths,
        assembly_obstacle_sweep_vector_m=target.insertion_sweep_vector_m,
        assembly_obstacle_metadata=b._target_assembly_obstacle_metadata(target),
    )
    mesh, pose = canonicalize_target_mesh(load_asset_mesh(target.target_mesh_path, scale=geometry.mesh_scale))
    orientations = enumerate_stable_orientations(mesh, b._stable_orientation_config(cfg)).orientations
    assert [o.orientation_id for o in orientations] == job["expected"], "Orientation population changed"
    axes = (
        ()
        if job["experiment"].startswith("base_upright_only")
        else b._upright_approach_axes_obj(source_frame_pose_obj_world=pose, orientations=orientations)
    )
    if job["experiment"] == "final_pose_only_diagnostic_s0":
        geometry = replace(geometry, assembly_obstacle_sweep_vector_m=None)
    started = time.monotonic()
    stage1 = generate_stage1_result(geometry=geometry, planning=planning, upright_approach_axes_obj=axes)
    elapsed = time.monotonic() - started
    write_stage1_artifacts(
        stage1, geometry=geometry, planning=planning, output_json=folder / "stage1.json", output_html=None
    )
    return stage1, orientations, planning, elapsed


def generation(job, folder):
    from grasp_planning.pipeline import recheck_stage2_result, write_stage2_artifacts
    from scripts.run_grasp_pipeline import _planning_config

    s1, orientations, planning, t1 = build_stage1(job, folder)
    result = dict(
        job_id=job["id"],
        kind="generation",
        state="running",
        stage1_s=t1,
        stage1_raw=s1.raw_candidate_count,
        stage1_feasible=len(s1.bundle.candidates),
        cache_hit=s1.bundle.metadata.get("stage1_cache_hit", False),
        rows=[],
    )
    if job["experiment"] == "final_pose_only_diagnostic_s0":
        from grasp_planning.grasping.fabrica_grasp_debug import (
            filter_grasps_against_assembly,
            load_assembly_obstacle_mesh,
        )

        target = job["target"]
        obstacles, _ = load_assembly_obstacle_mesh(
            assembly_glob=target["assembly_glob"],
            assembly_paths=target["assembly_obstacle_paths"],
            obstacle_sweep_vector_m=target["insertion_sweep_vector_m"],
            target_stl_path=target["target_mesh_path"],
            stl_scale=job["config"]["geometry"]["mesh_scale"],
        )
        valid = filter_grasps_against_assembly(
            s1.bundle.candidates,
            object_pose_world=s1.target_pose_in_obj_world,
            obstacle_mesh_world=obstacles,
            contact_gap_m=planning.detailed_finger_contact_gap_m,
            gripper_collision_model=planning.gripper_collision_model,
            contact_lateral_offsets_m=(),
            contact_approach_offsets_m=(),
        )
        result["sweep_audit"] = dict(
            final_pose_accepted=len(s1.bundle.candidates),
            also_sweep_valid=len(valid),
            sweep_rejected=len(s1.bundle.candidates) - len(valid),
        )
    treatments = [dict(name=job["experiment"], changes={})] + job["extras"]
    for treatment in treatments:
        cfg = deepcopy(job["config"])
        cfg["planning"].update(treatment["changes"])
        p = _planning_config(cfg)
        for o in orientations:
            t = time.monotonic()
            s2 = recheck_stage2_result(
                bundle=s1.bundle, pickup_spec=None, planning=p, object_pose_world=o.object_pose_world
            )
            elapsed = time.monotonic() - t
            row = dict(
                experiment=treatment["name"],
                assembly=job["target"]["assembly"],
                part_id=job["target"]["part_id"],
                orientation_id=o.orientation_id,
                status="direct_success"
                if s2.accepted
                else ("stage1_failed" if not s1.bundle.candidates else "stage2_failed"),
                stage2_s=elapsed,
                stage1_s=t1,
                stage1_raw=s1.raw_candidate_count,
                stage1_feasible=len(s1.bundle.candidates),
                stage2_feasible=len(s2.accepted),
                reason_counts=dict(Counter(x.reason for x in s2.statuses)),
                symmetry_status=s2.accepted_bundle.metadata.get("symmetry_pickup_load_status"),
                symmetry_derived=s2.accepted_bundle.metadata.get("symmetry_pickup_derived_candidate_count", 0),
                seed=p.rng_seed,
            )
            if treatment["name"] == job["experiment"]:
                bundle_path = folder / o.orientation_id / "stage2.json"
                write_stage2_artifacts(s2, planning=p, output_json=bundle_path, output_html=None)
                row["stage2_json"] = str(bundle_path)
                if s2.accepted:
                    rankings = {
                        m: [c.grasp_id for c in rank_candidates(s2.accepted, m, p.rng_seed)[:10]]
                        for m in ["full", "without_contact_support", "without_top_down", "without_com_offset", "random"]
                    }
                    row["rankings"] = rankings
                    row["top1_components"] = {
                        m: rank_candidates(s2.accepted, m, p.rng_seed)[0].score_components for m in rankings
                    }
            result["rows"].append(row)
            save(folder / "checkpoint.json", result)
    return result


def load_fallback_inputs(job):
    from grasp_planning.grasping.fabrica_grasp_debug import load_asset_mesh, load_grasp_bundle
    from grasp_planning.grasping.world_constraints import ObjectWorldPose
    from grasp_planning.pipeline.fabrica_pipeline import GroundRecheckResult, Stage1Result, _mesh_in_source_frame
    from grasp_planning.pipeline.regrasp_fallback import _candidate_from_payload
    from scripts.run_grasp_pipeline import _planning_config

    r = job["reference"]
    bundle = load_grasp_bundle(r["source_stage1"])
    source_pose = ObjectWorldPose(
        position_world=bundle.source_frame_origin_obj_world,
        orientation_xyzw_world=bundle.source_frame_orientation_xyzw_obj_world,
    )
    mesh = _mesh_in_source_frame(load_asset_mesh(bundle.target_mesh_path, scale=bundle.mesh_scale), source_pose)
    raw = json.loads(Path(r["source_raw"]).read_text())
    s1 = Stage1Result(
        bundle=bundle,
        target_mesh_local=mesh,
        target_pose_in_obj_world=source_pose,
        obstacle_mesh_world=None,
        collision_backend_name="trimesh_fcl",
        raw_candidate_count=raw["raw_candidate_count"],
        raw_candidates=tuple(_candidate_from_payload(c) for c in raw["candidates"]),
    )
    direct = load_grasp_bundle(r["source_stage2"])
    pose = ObjectWorldPose(**direct.metadata["execution_world_pose"])
    s2 = GroundRecheckResult(
        source_bundle=bundle,
        accepted_bundle=direct,
        mesh_local=mesh,
        pickup_pose_world=pose,
        pickup_spec=None,
        statuses=[],
        accepted=list(direct.candidates),
    )
    return s1, s2, _planning_config(job["config"])


def fallback(job, folder):
    from grasp_planning.pipeline import plan_handover_fallback, plan_mujoco_regrasp_fallback
    from scripts.run_grasp_generation_benchmark import _fallback_summary, _handover_summary

    s1, s2, p = load_fallback_inputs(job)
    r = job["reference"]
    rows = []
    if job["kind"] == "handover":
        settings = [
            ("handover_1000", 1000, 0.0),
            ("handover_4000", 4000, 0.0),
            ("handover_16000", 16000, 0.0),
            ("handover_margin10mm", 4000, 0.01),
        ]
        for name, budget, margin in settings:
            t = time.monotonic()
            res = plan_handover_fallback(
                stage1=s1,
                direct_stage2=s2,
                planning=p,
                max_final_candidates=80,
                max_transfer_candidates=160,
                max_pair_checks=budget,
                max_accepted_pairs=48,
                max_rejected_pairs=0,
                transfer_floor_clearance_margin_m=margin,
            )
            rows.append(
                dict(
                    experiment=name,
                    assembly=r["assembly"],
                    part_id=r["part_id"],
                    orientation_id=r["orientation_id"],
                    success=bool(res and res.selected_pair),
                    duration_s=time.monotonic() - t,
                    summary=_handover_summary(res),
                    metadata={} if res is None else res.metadata,
                    transfer_floor_counts={} if res is None else res.transfer_floor_status_counts,
                )
            )
            save(folder / "checkpoint.json", dict(kind=job["kind"], rows=rows))
    else:
        for name, yaws in [("regrasp_yaw1_xy1", [0.0]), ("regrasp_yaw4_xy1", [0.0, 90.0, 180.0, 270.0])]:
            t = time.monotonic()
            res = plan_mujoco_regrasp_fallback(
                stage1=s1,
                direct_stage2=s2,
                planning=p,
                yaw_angles_deg=tuple(yaws),
                staging_xy_offsets_m=((0.0, 0.0),),
                max_orientations=96,
                max_placement_options=72,
                min_facet_area_m2=1e-6,
                stability_margin_m=0.0,
                coplanar_tolerance_m=1e-6,
            )
            rows.append(
                dict(
                    experiment=name,
                    assembly=r["assembly"],
                    part_id=r["part_id"],
                    orientation_id=r["orientation_id"],
                    success=res is not None,
                    duration_s=time.monotonic() - t,
                    summary=_fallback_summary(res),
                )
            )
            save(folder / "checkpoint.json", dict(kind=job["kind"], rows=rows))
    return dict(kind=job["kind"], rows=rows)


def execution(job, folder):
    from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle
    from scripts.run_grasp_execution_benchmark import _prepare_execution_stage2_json

    r = job["reference"]
    bundle = load_grasp_bundle(r["source_stage2"])
    modes = ["full", "without_contact_support", "without_top_down", "without_com_offset", "random"]
    rankings = {m: [c.grasp_id for c in rank_candidates(bundle.candidates, m, 0)[:5]] for m in modes}
    path, _, _ = _prepare_execution_stage2_json(
        source_stage2_json=Path(r["source_stage2"]), attempt_dir=folder, placement_xy_world=(0.5, 0.0)
    )
    outcomes = {}
    for gid in dict.fromkeys(gid for values in rankings.values() for gid in values):
        artifact = folder / (gid + "_attempt.json")
        log = folder / (gid + ".log")
        cmd = [
            sys.executable,
            str(ROOT / "scripts/run_fabrica_grasp_in_mujoco.py"),
            "--input-json",
            str(path),
            "--robot-config",
            str(ROOT / "configs/mujoco_fr3_with_hand.json"),
            "--simulation-config",
            str(ROOT / "configs/mujoco_simulation.yaml"),
            "--controller",
            "native",
            "--grasp-id",
            gid,
            "--object-density-kg-m3",
            "1240",
            "--attempt-artifact",
            str(artifact),
        ]
        t = time.monotonic()
        with log.open("w") as stream:
            proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        data = json.loads(artifact.read_text()) if artifact.exists() else {}
        outcomes[gid] = dict(
            returncode=proc.returncode, duration_s=time.monotonic() - t, artifact=str(artifact), result=data
        )
        save(folder / "checkpoint.json", dict(kind="execution", rankings=rankings, outcomes=outcomes))
        if proc.returncode and not artifact.exists():
            raise RuntimeError("Execution setup failed; inspect " + str(log))
    return dict(
        kind="execution",
        assembly=r["assembly"],
        part_id=r["part_id"],
        orientation_id=r["orientation_id"],
        controller="native",
        rankings=rankings,
        outcomes=outcomes,
    )


def worker(root, index):
    manifest = json.loads((root / "manifest.json").read_text())
    job = manifest["jobs"][index]
    folder = root / "jobs" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    save(folder / "started.json", dict(started=now(), pid=os.getpid(), job_id=job["id"]))
    try:
        if job["kind"] == "generation":
            result = generation(job, folder)
        elif job["kind"] in {"handover", "regrasp"}:
            result = fallback(job, folder)
        else:
            result = execution(job, folder)
        result.update(state="complete", finished=now(), wall_s=time.monotonic() - started, job_id=job["id"])
        save(folder / "result.json", result)
    except Exception:
        save(folder / "error.json", dict(state="error", error=traceback.format_exc(), job_id=job["id"], finished=now()))
        raise


def paired_interval(rows, baseline):
    """Part-cluster bootstrap of matched orientation differences, fixed corpus."""
    import numpy as np

    ref = {(r["assembly"], r["part_id"], r["orientation_id"]): r for r in baseline}
    grouped = defaultdict(list)
    for r in rows:
        key = (r["assembly"], r["part_id"], r["orientation_id"])
        if key not in ref:
            return None
        grouped[key[:2]].append(int(r["status"] == "direct_success") - int(ref[key]["status"] == "direct_success"))
    if len(rows) != len(ref):
        return None
    sums = np.array([sum(v) for v in grouped.values()])
    counts = np.array([len(v) for v in grouped.values()])
    rng = np.random.default_rng(20260923)
    ids = rng.integers(len(sums), size=(1000, len(sums)))
    deltas = sums[ids].sum(axis=1) / counts[ids].sum(axis=1)
    return [float(sum(sums) / sum(counts)), *[float(x) for x in np.quantile(deltas, [0.025, 0.975])]]


def report(root):
    import csv

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = json.loads((root / "reference.json").read_text())["rows"]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"jobs": [], "notes": []}
    grouped = defaultdict(list)
    fallback_rows = defaultdict(list)
    complete = errors = 0
    execution_results = []
    for job in manifest["jobs"]:
        folder = root / "jobs" / job["id"]
        if (folder / "result.json").exists():
            complete += 1
            data = json.loads((folder / "result.json").read_text())
            for r in data.get("rows", []):
                (grouped if job["kind"] == "generation" else fallback_rows)[r["experiment"]].append(r)
            if job["kind"] == "execution":
                execution_results.append(data)
        elif (folder / "error.json").exists():
            errors += 1
    reference_direct = [
        dict(r, status="direct_success" if r["status"] == "direct_success" else "stage2_failed") for r in reference
    ]
    rows = []
    baseline = grouped.get("reference_s0", reference_direct)
    if len(baseline) != len(reference):
        baseline = reference_direct
    for name, values in grouped.items():
        count = sum(r["status"] == "direct_success" for r in values)
        perpart = defaultdict(list)
        for r in values:
            perpart[(r["assembly"], r["part_id"])].append(r["status"] == "direct_success")
        seed = int(values[0].get("seed", 0))
        paired_baseline = grouped.get(f"reference_s{seed}", [])
        if len(paired_baseline) != len(reference):
            paired_baseline = reference_direct if seed == 0 else []
        ci = paired_interval(values, paired_baseline) if paired_baseline else None
        stage1 = {(r["assembly"], r["part_id"]): r["stage1_s"] for r in values}
        rows.append(
            dict(
                experiment=name,
                completed_orientations=len(values),
                expected_orientations=len(reference),
                complete=len(values) == len(reference),
                direct=count,
                coverage=count / len(values) if values else 0,
                macro_part_coverage=float(np.mean([np.mean(v) for v in perpart.values()])),
                stage1_median_s=float(np.median(list(stage1.values()))),
                stage2_median_s=float(np.median([r["stage2_s"] for r in values])),
                stage2_p95_s=float(np.percentile([r["stage2_s"] for r in values], 95)),
                paired_delta=None if ci is None else ci[0],
                paired_ci_low=None if ci is None else ci[1],
                paired_ci_high=None if ci is None else ci[2],
            )
        )
    if rows:
        with (root / "comparison.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    save(
        root / "comparison.json",
        dict(
            updated=now(),
            completed_jobs=complete,
            error_jobs=errors,
            total_jobs=len(manifest["jobs"]),
            experiments=rows,
            fallbacks={
                k: dict(
                    evaluated=len(v),
                    expected=sum(r["status"] != "direct_success" for r in reference),
                    solved=sum(x["success"] for x in v),
                )
                for k, v in fallback_rows.items()
            },
        ),
    )
    if fallback_rows:
        save(root / "fallback_results.json", dict(fallback_rows))
    if execution_results:
        save(root / "execution_results.json", execution_results)
    counts = Counter(r["status"] for r in reference)
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [
        counts["direct_success"],
        counts["direct_success"] + counts["fallback_success"],
        sum(counts[x] for x in SUCCESS),
    ]
    ax.bar(
        ["Direct + symmetry", "+ table regrasp", "+ handover"],
        np.array(vals) / len(reference) * 100,
        color=["#497da7", "#599c89", "#cb934c"],
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Geometric coverage (%)")
    ax.set_title("Existing baseline · 46 parts / 264 orientations")
    for i, v in enumerate(vals):
        ax.text(i, v / len(reference) * 100 + 2, f"{v}/{len(reference)}", ha="center")
    fig.tight_layout()
    fig.savefig(root / "baseline_coverage.png", dpi=180)
    fig.savefig(root / "baseline_coverage.pdf")
    plt.close(fig)
    finished = [r for r in rows if r["complete"]]
    if finished:
        fig, ax = plt.subplots(figsize=(9, max(3, len(finished) * 0.32)))
        ax.barh([r["experiment"] for r in finished], [r["coverage"] * 100 for r in finished], color="#497da7")
        ax.set_xlabel("Direct geometric coverage (%)")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(root / "ablation_coverage.png", dpi=180)
        fig.savefig(root / "ablation_coverage.pdf")
        plt.close(fig)
    # Presentation-ready assembly breakdown of the preserved full baseline.
    assemblies = sorted({r["assembly"] for r in reference})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(assemblies))
    for status, label, color in [
        ("direct_success", "Direct", "#497da7"),
        ("fallback_success", "Table regrasp", "#599c89"),
        ("handover_fallback_success", "Handover", "#cb934c"),
        ("stage2_failed_fallback_failed", "Unresolved", "#be6870"),
    ]:
        vals = np.array(
            [
                sum(r["assembly"] == a and r["status"] == status for r in reference)
                / sum(r["assembly"] == a for r in reference)
                * 100
                for a in assemblies
            ]
        )
        ax.bar(assemblies, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_ylabel("Orientations (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(root / "baseline_by_assembly.png", dpi=180)
    fig.savefig(root / "baseline_by_assembly.pdf")
    plt.close(fig)
    # Show individual matched seeds, without implying repeated poses are independent.
    curves = defaultdict(list)
    for r in finished:
        name = r["experiment"]
        if name.startswith("reference_s"):
            n = 1024
        elif name.startswith("samples_") and "_pairs_" not in name:
            n = int(name.split("_")[1])
        else:
            continue
        seed = int(name.rsplit("_s", 1)[1])
        curves[seed].append((n, r["coverage"] * 100, r["stage1_median_s"]))
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for seed, points in sorted(curves.items()):
            points.sort()
            axes[0].plot([p[0] for p in points], [p[1] for p in points], "o-", label=f"Seed {seed}")
            axes[1].plot([p[2] for p in points], [p[1] for p in points], "o-", label=f"Seed {seed}")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("Surface samples (pair cap 40,960)")
        axes[1].set_xlabel("Median stage-1 wall time / part (s)")
        for ax in axes:
            ax.set_ylabel("Direct coverage (%)")
            ax.legend()
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(root / "sampling_tradeoff.png", dpi=180)
        fig.savefig(root / "sampling_tradeoff.pdf")
        plt.close(fig)
    if execution_results:
        execution_summary = []
        for mode in ["full", "without_contact_support", "without_top_down", "without_com_offset", "random"]:
            first = top3 = top5 = held_first = held3 = held5 = 0
            statuses = Counter()
            for case in execution_results:
                ids = case["rankings"][mode]
                results = [case["outcomes"][gid]["result"].get("result", {}) for gid in ids]
                first += bool(results[0].get("success"))
                top3 += any(r.get("success", False) for r in results[:3])
                top5 += any(r.get("success", False) for r in results)
                held = []
                for r in results:
                    start = r.get("initial_object_position_world")
                    end = r.get("final_object_position_world")
                    held.append(bool(start and end and end[2] - start[2] >= r.get("target_lift_height_m", 0.05)))
                held_first += held[0]
                held3 += any(held[:3])
                held5 += any(held)
                statuses[results[0].get("status", "missing")] += 1
            execution_summary.append(
                dict(
                    ranking=mode,
                    cases=len(execution_results),
                    top1_success=first,
                    top3_success=top3,
                    top5_success=top5,
                    top1_final_hold_success=held_first,
                    top3_final_hold_success=held3,
                    top5_final_hold_success=held5,
                    top1_status_counts=dict(statuses),
                )
            )
        save(root / "execution_summary.json", execution_summary)
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(execution_summary))
        ax.bar(
            x - 0.18,
            [100 * r["top1_final_hold_success"] / r["cases"] for r in execution_summary],
            0.36,
            label="Top 1, final hold",
        )
        ax.bar(
            x + 0.18,
            [100 * r["top3_final_hold_success"] / r["cases"] for r in execution_summary],
            0.36,
            label="Within top 3, final hold",
        )
        ax.set_xticks(x, [r["ranking"] for r in execution_summary], rotation=20)
        ax.set_ylabel("Native MuJoCo success (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "execution_ranking.png", dpi=180)
        fig.savefig(root / "execution_ranking.pdf")
        plt.close(fig)
    # Intersections are evaluated only where both independent searches completed.
    overlap = {}
    h = {
        (r["assembly"], r["part_id"], r["orientation_id"]): r["success"] for r in fallback_rows.get("handover_4000", [])
    }
    for name in ["regrasp_yaw1_xy1", "regrasp_yaw4_xy1"]:
        rr = {(r["assembly"], r["part_id"], r["orientation_id"]): r["success"] for r in fallback_rows.get(name, [])}
        common = h.keys() & rr.keys()
        counts_overlap = Counter(
            ("both" if h[k] and rr[k] else "handover_only" if h[k] else "regrasp_only" if rr[k] else "neither")
            for k in common
        )
        overlap[name] = dict(evaluated=len(common), counts=dict(counts_overlap))
    save(root / "fallback_overlap.json", overlap)
    lines = (
        [
            "# Grasp thesis experiment queue",
            "",
            f"Updated: {now()}",
            "",
            f"Completed jobs: {complete}/{len(manifest['jobs'])}; errors/timeouts: {errors}.",
            "",
            "Existing baseline: 79 direct + 2 regrasp + 131 handover = 212/264 (80.3%).",
            "",
            "## Interpretation",
            "",
        ]
        + ["- " + n for n in manifest.get("notes", [])]
        + [
            "",
            "- Paired intervals resample parts within the fixed corpus; they do not establish generalization to unseen assemblies.",
            "- Partial experiments are explicitly incomplete and must not be compared as full-corpus percentages.",
            "- Score-ranking differences alone are not evidence of execution improvement.",
            "",
            "## Queue and results",
            "",
            "| Experiment | Cases complete | Direct successes | Coverage of completed cases |",
            "|---|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['completed_orientations']}/264 | {r['direct']} | {100 * r['coverage']:.1f}% |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    table = "".join(
        f"<tr><td>{html.escape(r['experiment'])}</td><td>{r['completed_orientations']}/264</td><td>{r['direct']}</td><td>{100 * r['coverage']:.1f}%"
        + ("" if r["complete"] else " (partial)")
        + "</td></tr>"
        for r in rows
    )
    page = f"""<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Grasp ablations</title><style>body{{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px;color:#213244}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style><h1>Grasp ablations</h1><p>{complete}/{len(manifest["jobs"])} jobs completed · {errors} errors/timeouts · refreshed {now()}</p><p>Geometric planning coverage. Partial experiments are not final comparisons.</p><p><a href="REPORT.md">Report and limitations</a> · <a href="comparison.csv">Comparison CSV</a> · <a href="status.json">Live queue status</a> · <a href="queue.log">Log</a></p><img src="baseline_coverage.png"><table><tr><th>Experiment</th><th>Cases</th><th>Direct</th><th>Coverage</th></tr>{table}</table>"""
    page += '<img src="baseline_by_assembly.png">'
    for figure in ["ablation_coverage", "sampling_tradeoff", "execution_ranking"]:
        if (root / (figure + ".png")).exists():
            page += f'<img src="{figure}.png">'
    page += '<p><a href="fallback_results.json">Independent fallback results</a> · <a href="fallback_overlap.json">Fallback overlap</a> · <a href="execution_summary.json">Execution summary</a></p>'
    (root / "index.html").write_text(page)


def refresh_report(root):
    env = dict(
        os.environ,
        PYTHONNOUSERSITE="1",
        MPLCONFIGDIR=str(root / ".matplotlib"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "report", "--root", str(root)], env=env, check=True, timeout=120
    )


def run(root):
    import psutil

    manifest = json.loads((root / "manifest.json").read_text())
    lock = (root / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    worker_script = Path(manifest["snapshot"]) / "scripts/run_grasp_ablation_suite.py"
    active = {}
    stop = False
    last_report = 0

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(now(), "QUEUE START", len(manifest["jobs"]), "jobs", manifest["workers"], "slots", flush=True)
    try:
        while True:
            for index, (proc, stream, start, job) in list(active.items()):
                elapsed = time.monotonic() - start
                if proc.poll() is None and elapsed > job["timeout_s"]:
                    os.killpg(proc.pid, signal.SIGKILL)
                    save(
                        root / "jobs" / job["id"] / "error.json",
                        dict(state="timeout", job_id=job["id"], elapsed_s=elapsed),
                    )
                if proc.poll() is not None:
                    stream.close()
                    folder = root / "jobs" / job["id"]
                    if not (folder / "result.json").exists() and not (folder / "error.json").exists():
                        save(folder / "error.json", dict(state="error", returncode=proc.returncode, job_id=job["id"]))
                    print(now(), "FINISHED", job["id"], "exit", proc.returncode, flush=True)
                    del active[index]
            pending = [
                (i, j)
                for i, j in enumerate(manifest["jobs"])
                if i not in active
                and not (root / "jobs" / j["id"] / "result.json").exists()
                and not (root / "jobs" / j["id"] / "error.json").exists()
            ]
            used = sum(j["slots"] for _, _, _, j in active.values())
            if (
                not stop
                and psutil.virtual_memory().available > 5 * 1024**3
                and shutil.disk_usage(root).free > 30 * 1024**3
            ):
                for index, job in pending:
                    if used + job["slots"] > manifest["workers"]:
                        continue
                    folder = root / "jobs" / job["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    stream = (folder / "run.log").open("a")
                    env = dict(
                        os.environ,
                        OMP_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        NUMEXPR_NUM_THREADS="1",
                        MPLBACKEND="Agg",
                    )
                    proc = subprocess.Popen(
                        [sys.executable, str(worker_script), "worker", "--root", str(root), "--index", str(index)],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        cwd=manifest["snapshot"],
                        env=env,
                        start_new_session=True,
                    )
                    active[index] = (proc, stream, time.monotonic(), job)
                    used += job["slots"]
                    print(now(), "START", job["id"], flush=True)
                    if used >= manifest["workers"]:
                        break
            save(
                root / "status.json",
                dict(
                    updated=now(),
                    state="stopping" if stop else "running",
                    active=[
                        dict(job_id=j["id"], pid=p.pid, elapsed_s=time.monotonic() - t)
                        for p, _, t, j in active.values()
                    ],
                    pending=len(pending),
                    workers=manifest["workers"],
                ),
            )
            if time.monotonic() - last_report > 120:
                try:
                    refresh_report(root)
                except Exception:
                    traceback.print_exc()
                last_report = time.monotonic()
            if not active and (stop or not pending):
                break
            time.sleep(5)
    finally:
        for proc, stream, _, _ in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            stream.close()
        refresh_report(root)
        save(root / "status.json", dict(updated=now(), state="stopped" if stop else "finished", active=[]))
        print(now(), "QUEUE EXIT", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "worker", "report"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        prepare(root, args.workers)
    elif args.command == "run":
        run(root)
    elif args.command == "worker":
        worker(root, args.index)
    else:
        report(root)


if __name__ == "__main__":
    main()
