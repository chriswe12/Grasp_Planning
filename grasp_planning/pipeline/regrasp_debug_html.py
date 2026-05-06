"""HTML visualization for MuJoCo regrasp fallback plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from grasp_planning.grasping.fabrica_grasp_debug import (
    CandidateStatus,
    SavedGraspCandidate,
    accepted_grasps,
    candidate_payload,
    evaluate_saved_grasps_against_pickup_pose,
    fmt_vec,
    object_point_to_world,
    quat_to_rotmat_xyzw,
    unique_edges,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose

from .fabrica_pipeline import PlanningConfig, Stage1Result, _score_grasps_for_world_top_approach
from .regrasp_fallback import MujocoRegraspFallbackPlan


def _candidate_key_tuple(candidate: SavedGraspCandidate) -> tuple[str, float, float]:
    return (
        str(candidate.grasp_id),
        round(float(candidate.contact_patch_lateral_offset_m), 9),
        round(float(candidate.contact_patch_approach_offset_m), 9),
    )


def _candidate_key(candidate: SavedGraspCandidate) -> str:
    grasp_id, lateral, approach = _candidate_key_tuple(candidate)
    return f"{grasp_id}|{lateral:.9f}|{approach:.9f}"


def _rescore_statuses(
    statuses: Iterable[CandidateStatus],
    scored_accepted: Iterable[SavedGraspCandidate],
) -> list[CandidateStatus]:
    scored_by_key = {_candidate_key_tuple(candidate): candidate for candidate in scored_accepted}
    rescored: list[CandidateStatus] = []
    for entry in statuses:
        rescored.append(
            CandidateStatus(
                grasp=scored_by_key.get(_candidate_key_tuple(entry.grasp), entry.grasp),
                status=entry.status,
                reason=entry.reason,
            )
        )
    return rescored


def _transfer_statuses(
    candidates: Iterable[SavedGraspCandidate],
    *,
    mesh_local: object,
    object_pose_world: ObjectWorldPose,
    planning: PlanningConfig,
    top_grasp_score_weight: float,
) -> list[CandidateStatus]:
    statuses = evaluate_saved_grasps_against_pickup_pose(
        candidates,
        object_pose_world=object_pose_world,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        floor_clearance_margin_m=planning.floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )
    scored = _score_grasps_for_world_top_approach(
        accepted_grasps(statuses),
        mesh_local=mesh_local,
        object_pose_world=object_pose_world,
        top_grasp_score_weight=top_grasp_score_weight,
    )
    return _rescore_statuses(statuses, scored)


def _final_statuses(
    candidates: Iterable[SavedGraspCandidate],
    *,
    mesh_local: object,
    object_pose_world: ObjectWorldPose,
    planning: PlanningConfig,
) -> list[CandidateStatus]:
    statuses = evaluate_saved_grasps_against_pickup_pose(
        candidates,
        object_pose_world=object_pose_world,
        contact_gap_m=planning.detailed_finger_contact_gap_m,
        floor_clearance_margin_m=planning.floor_clearance_margin_m,
        contact_lateral_offsets_m=planning.contact_lateral_offsets_m,
        contact_approach_offsets_m=planning.contact_approach_offsets_m,
    )
    scored = _score_grasps_for_world_top_approach(
        accepted_grasps(statuses),
        mesh_local=mesh_local,
        object_pose_world=object_pose_world,
        top_grasp_score_weight=planning.top_grasp_score_weight,
    )
    return _rescore_statuses(statuses, scored)


def _score_components_payload(candidate: SavedGraspCandidate) -> dict[str, float] | None:
    if candidate.score_components is None:
        return None
    return {str(key): round(float(value), 6) for key, value in candidate.score_components.items()}


def _candidate_marker_payload(
    entry: CandidateStatus,
    *,
    object_pose_world: ObjectWorldPose,
    selected_keys: set[str],
) -> dict[str, object]:
    candidate = entry.grasp
    grasp_rotation_world = object_pose_world.rotation_world_from_object @ quat_to_rotmat_xyzw(
        candidate.grasp_orientation_xyzw_obj
    )
    point_a_world = object_point_to_world(np.asarray(candidate.contact_point_a_obj, dtype=float), object_pose_world)
    point_b_world = object_point_to_world(np.asarray(candidate.contact_point_b_obj, dtype=float), object_pose_world)
    center_world = object_point_to_world(np.asarray(candidate.grasp_position_obj, dtype=float), object_pose_world)
    key = _candidate_key(candidate)
    return {
        "key": key,
        "grasp_id": candidate.grasp_id,
        "status": entry.status,
        "reason": entry.reason,
        "score": None if candidate.score is None else round(float(candidate.score), 6),
        "score_components": _score_components_payload(candidate),
        "grasp_position_world": fmt_vec(center_world.tolist()),
        "contact_point_a_world": fmt_vec(point_a_world.tolist()),
        "contact_point_b_world": fmt_vec(point_b_world.tolist()),
        "approach_axis_world": fmt_vec(grasp_rotation_world[:, 2].tolist()),
        "closing_axis_world": fmt_vec(grasp_rotation_world[:, 1].tolist()),
        "jaw_width": round(float(candidate.jaw_width), 6),
        "roll_angle_rad": round(float(candidate.roll_angle_rad), 6),
        "contact_patch_lateral_offset_m": round(float(candidate.contact_patch_lateral_offset_m), 6),
        "contact_patch_approach_offset_m": round(float(candidate.contact_patch_approach_offset_m), 6),
        "is_plan_selected": key in selected_keys,
    }


def _status_counts(statuses: Iterable[CandidateStatus]) -> dict[str, int]:
    accepted = 0
    rejected = 0
    for entry in statuses:
        if entry.status == "accepted":
            accepted += 1
        else:
            rejected += 1
    return {"accepted": accepted, "rejected": rejected, "total": accepted + rejected}


def _matching_status(
    statuses: Iterable[CandidateStatus],
    selected: SavedGraspCandidate,
) -> CandidateStatus | None:
    selected_key = _candidate_key_tuple(selected)
    by_id: CandidateStatus | None = None
    for entry in statuses:
        if _candidate_key_tuple(entry.grasp) == selected_key:
            return entry
        if entry.grasp.grasp_id == selected.grasp_id and by_id is None:
            by_id = entry
    return by_id


def _selected_hand_payload(
    statuses: Iterable[CandidateStatus],
    selected: SavedGraspCandidate,
    *,
    object_pose_world: ObjectWorldPose,
    contact_gap_m: float,
) -> dict[str, object]:
    entry = _matching_status(statuses, selected) or CandidateStatus(
        grasp=selected,
        status="accepted",
        reason="plan_selected",
    )
    return candidate_payload(
        [entry],
        contact_gap_m=contact_gap_m,
        object_pose_world=object_pose_world,
    )[0]


def _floor_plane_vertices_world(mesh_local: object, object_pose_world: ObjectWorldPose) -> list[list[float]]:
    vertices_world = object_pose_world.transform_points_to_world(np.asarray(mesh_local.vertices_obj, dtype=float))
    mins = vertices_world.min(axis=0)
    maxs = vertices_world.max(axis=0)
    extents = np.maximum(maxs - mins, 1.0e-3)
    padding = max(0.35 * float(np.max(extents[:2])), 0.10)
    return [
        fmt_vec((mins[0] - padding, mins[1] - padding, 0.0)),
        fmt_vec((maxs[0] + padding, mins[1] - padding, 0.0)),
        fmt_vec((maxs[0] + padding, maxs[1] + padding, 0.0)),
        fmt_vec((mins[0] - padding, maxs[1] + padding, 0.0)),
    ]


def _pose_payload(pose: ObjectWorldPose) -> dict[str, object]:
    return {
        "position_world": fmt_vec(pose.position_world),
        "orientation_xyzw_world": fmt_vec(pose.orientation_xyzw_world),
    }


def _support_payload(plan: MujocoRegraspFallbackPlan) -> dict[str, object]:
    pose = plan.staging_object_pose_world
    facet = plan.support_facet
    return {
        "normal_obj": fmt_vec(facet.normal_obj),
        "area_m2": round(float(facet.area_m2), 8),
        "stability_margin_m": round(float(facet.stability_margin_m), 8),
        "yaw_deg": round(float(facet.yaw_deg), 3),
        "vertices_world": [
            fmt_vec(object_point_to_world(np.asarray(vertex, dtype=float), pose).tolist())
            for vertex in facet.vertices_obj
        ],
        "com_world": fmt_vec(object_point_to_world(np.asarray(facet.com_obj, dtype=float), pose).tolist()),
        "com_projection_world": fmt_vec(
            object_point_to_world(np.asarray(facet.com_projection_obj, dtype=float), pose).tolist()
        ),
    }


def _group_payload(
    *,
    group_id: str,
    label: str,
    role: str,
    statuses: list[CandidateStatus],
    object_pose_world: ObjectWorldPose,
    selected: SavedGraspCandidate,
    contact_gap_m: float,
) -> dict[str, object]:
    selected_key = _candidate_key(selected)
    selected_keys = {selected_key}
    return {
        "id": group_id,
        "label": label,
        "role": role,
        "selected_key": selected_key,
        "selected_grasp_id": selected.grasp_id,
        "counts": _status_counts(statuses),
        "selected_hand": _selected_hand_payload(
            statuses,
            selected,
            object_pose_world=object_pose_world,
            contact_gap_m=contact_gap_m,
        ),
        "candidates": [
            _candidate_marker_payload(
                entry,
                object_pose_world=object_pose_world,
                selected_keys=selected_keys,
            )
            for entry in statuses
        ],
    }


def _scene_payload(
    *,
    scene_id: str,
    title: str,
    subtitle: str,
    mesh_local: object,
    object_pose_world: ObjectWorldPose,
    groups: list[dict[str, object]],
    support: dict[str, object] | None = None,
) -> dict[str, object]:
    vertices_world = object_pose_world.transform_points_to_world(np.asarray(mesh_local.vertices_obj, dtype=float))
    faces = np.asarray(mesh_local.faces, dtype=np.int64)
    return {
        "id": scene_id,
        "title": title,
        "subtitle": subtitle,
        "pose": _pose_payload(object_pose_world),
        "vertices_world": [fmt_vec(vertex.tolist()) for vertex in vertices_world],
        "faces": [[int(v) for v in face] for face in faces.tolist()],
        "edges": unique_edges(faces),
        "floor_world": _floor_plane_vertices_world(mesh_local, object_pose_world),
        "support": support,
        "groups": groups,
    }


def write_mujoco_regrasp_debug_html(
    *,
    plan: MujocoRegraspFallbackPlan,
    stage1: Stage1Result,
    planning: PlanningConfig,
    output_html: str | Path,
) -> None:
    """Write a side-by-side HTML view of the initial and staging regrasp poses."""

    raw_candidates = tuple(stage1.raw_candidates) if stage1.raw_candidates else tuple(stage1.bundle.candidates)
    transfer_initial_statuses = _transfer_statuses(
        raw_candidates,
        mesh_local=stage1.target_mesh_local,
        object_pose_world=plan.initial_object_pose_world,
        planning=planning,
        top_grasp_score_weight=planning.regrasp_transfer_top_grasp_score_weight,
    )
    transfer_staging_statuses = _transfer_statuses(
        raw_candidates,
        mesh_local=stage1.target_mesh_local,
        object_pose_world=plan.staging_object_pose_world,
        planning=planning,
        top_grasp_score_weight=planning.top_grasp_score_weight,
    )
    final_staging_statuses = _final_statuses(
        stage1.bundle.candidates,
        mesh_local=stage1.target_mesh_local,
        object_pose_world=plan.staging_object_pose_world,
        planning=planning,
    )

    data = {
        "title": "Fabrica MuJoCo Regrasp Debug",
        "subtitle": "Initial and staging object poses with floor, all evaluated grasp markers, and highlighted planned grasps.",
        "plan_metadata": dict(plan.metadata),
        "metadata_lines": [
            f"target_mesh:      {plan.target_mesh_path}",
            f"mesh_scale:       {plan.mesh_scale}",
            f"raw_transfer:     {len(raw_candidates)}",
            f"assembly_grasps:  {len(stage1.bundle.candidates)}",
            f"transfer_grasp:   {plan.transfer_grasp.grasp_id}",
            f"final_grasp:      {plan.final_grasp.grasp_id}",
            f"floor_clearance:  {planning.floor_clearance_margin_m:.6f} m",
            f"contact_offsets_x:{tuple(planning.contact_lateral_offsets_m)}",
            f"contact_offsets_z:{tuple(planning.contact_approach_offsets_m)}",
        ],
        "scenes": [
            _scene_payload(
                scene_id="initial",
                title="Initial Pose",
                subtitle="Raw stage-1 transfer grasps evaluated against the initial floor pose.",
                mesh_local=stage1.target_mesh_local,
                object_pose_world=plan.initial_object_pose_world,
                groups=[
                    _group_payload(
                        group_id="transfer_initial",
                        label="Transfer grasps",
                        role="transfer",
                        statuses=transfer_initial_statuses,
                        object_pose_world=plan.initial_object_pose_world,
                        selected=plan.transfer_grasp,
                        contact_gap_m=planning.detailed_finger_contact_gap_m,
                    )
                ],
            ),
            _scene_payload(
                scene_id="staging",
                title="Staging Pose",
                subtitle="Raw transfer grasps and assembly-feasible final grasps evaluated against the staging floor pose.",
                mesh_local=stage1.target_mesh_local,
                object_pose_world=plan.staging_object_pose_world,
                support=_support_payload(plan),
                groups=[
                    _group_payload(
                        group_id="transfer_staging",
                        label="Transfer grasps",
                        role="transfer",
                        statuses=transfer_staging_statuses,
                        object_pose_world=plan.staging_object_pose_world,
                        selected=plan.transfer_grasp,
                        contact_gap_m=planning.detailed_finger_contact_gap_m,
                    ),
                    _group_payload(
                        group_id="final_staging",
                        label="Final assembly grasps",
                        role="final",
                        statuses=final_staging_statuses,
                        object_pose_world=plan.staging_object_pose_world,
                        selected=plan.final_grasp,
                        contact_gap_m=planning.detailed_finger_contact_gap_m,
                    ),
                ],
            ),
        ],
    }
    data_json = json.dumps(data, indent=2)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fabrica MuJoCo Regrasp Debug</title>
  <style>
    :root {
      --bg: #f6f4ef;
      --panel: #fffdf8;
      --ink: #1f2522;
      --muted: #64706a;
      --line: #d8d2c5;
      --mesh: #465f55;
      --floor: #2563eb;
      --support: #d97706;
      --transfer: #0f766e;
      --final: #7c3aed;
      --rejected: #b91c1c;
      --hand: #9a5b13;
      --com: #db2777;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--bg); font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }
    .layout { display: grid; grid-template-columns: 340px minmax(0, 1fr); min-height: 100vh; }
    .sidebar { padding: 18px; background: var(--panel); border-right: 1px solid var(--line); overflow: auto; }
    h1 { margin: 0 0 8px; font-size: 25px; line-height: 1.15; }
    .subtitle { margin: 0 0 14px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }
    button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 7px; padding: 8px 10px; font: inherit; cursor: pointer; }
    button:hover, button.active { border-color: var(--transfer); }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    .main { padding: 16px; overflow: auto; }
    .scene-grid { display: grid; grid-template-columns: repeat(2, minmax(420px, 1fr)); gap: 14px; }
    .scene-panel { border: 1px solid var(--line); background: rgba(255,253,248,0.95); border-radius: 8px; padding: 12px; }
    .scene-head { display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 8px; }
    .scene-title { margin: 0; font-size: 18px; line-height: 1.2; }
    .scene-subtitle { margin: 4px 0 0; color: var(--muted); font-size: 12px; line-height: 1.35; }
    .group-buttons { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
    .group-buttons button { padding: 6px 8px; font-size: 12px; }
    .scene-svg { width: 100%; aspect-ratio: 1.25 / 1; display: block; border: 1px solid var(--line); border-radius: 6px; background: linear-gradient(180deg, #ffffff, #ece7dc); cursor: grab; }
    .scene-meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 8px; color: var(--muted); font-size: 12px; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; color: var(--muted); font-size: 12px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 12px; height: 12px; border-radius: 999px; display: inline-block; }
    @media (max-width: 1150px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .scene-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 id="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <div class="controls">
        <button id="isoBtn" type="button">ISO</button>
        <button id="topBtn" type="button">Top</button>
        <button id="allGroupsBtn" type="button">All Groups: On</button>
        <button id="rejectedBtn" type="button">Rejected: On</button>
      </div>
      <pre id="details" class="kv"></pre>
      <div class="legend">
        <span><i class="swatch" style="background: var(--mesh)"></i>Object</span>
        <span><i class="swatch" style="background: var(--floor)"></i>Floor</span>
        <span><i class="swatch" style="background: var(--transfer)"></i>Transfer</span>
        <span><i class="swatch" style="background: var(--final)"></i>Final</span>
        <span><i class="swatch" style="background: var(--rejected)"></i>Rejected</span>
        <span><i class="swatch" style="background: var(--support)"></i>Support facet</span>
        <span><i class="swatch" style="background: var(--com)"></i>COM</span>
      </div>
    </aside>
    <main class="main">
      <div id="scenes" class="scene-grid"></div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const titleNode = document.getElementById("title");
    const subtitleNode = document.getElementById("subtitle");
    const detailsNode = document.getElementById("details");
    const scenesNode = document.getElementById("scenes");
    const isoBtn = document.getElementById("isoBtn");
    const topBtn = document.getElementById("topBtn");
    const allGroupsBtn = document.getElementById("allGroupsBtn");
    const rejectedBtn = document.getElementById("rejectedBtn");

    titleNode.textContent = data.title;
    subtitleNode.textContent = data.subtitle;

    const state = {
      yaw: -0.82,
      pitch: 0.56,
      zoom: 1.0,
      showAllGroups: true,
      showRejected: true,
      activeGroup: {},
      selectedKey: {},
      drag: null,
    };

    for (const scene of data.scenes) {
      state.activeGroup[scene.id] = scene.groups[scene.groups.length - 1].id;
      state.selectedKey[scene.id] = {};
      for (const group of scene.groups) {
        state.selectedKey[scene.id][group.id] = group.selected_key;
      }
    }

    function fmt(value) {
      return value === null || value === undefined ? "n/a" : Number(value).toFixed(6);
    }

    function candidatePoints(group) {
      return group.candidates.flatMap((candidate) => [
        candidate.grasp_position_world,
        candidate.contact_point_a_world,
        candidate.contact_point_b_world,
      ]);
    }

    function scenePoints(scene) {
      const groupPoints = scene.groups.flatMap((group) => candidatePoints(group));
      const supportPoints = scene.support ? [
        ...scene.support.vertices_world,
        scene.support.com_world,
        scene.support.com_projection_world,
      ] : [];
      return [
        ...scene.vertices_world,
        ...scene.floor_world,
        ...groupPoints,
        ...supportPoints,
      ];
    }

    function boundsFor(points) {
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (const point of points) {
        point.forEach((value, axis) => {
          min[axis] = Math.min(min[axis], value);
          max[axis] = Math.max(max[axis], value);
        });
      }
      const center = min.map((value, axis) => 0.5 * (value + max[axis]));
      const extent = Math.max(...max.map((value, axis) => value - min[axis]), 0.16);
      return { center, extent };
    }

    for (const scene of data.scenes) {
      scene.bounds = boundsFor(scenePoints(scene));
    }

    function rotate(scene, point) {
      const shifted = point.map((value, axis) => value - scene.bounds.center[axis]);
      const cy = Math.cos(state.yaw);
      const sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch);
      const sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }

    function project(scene, point) {
      const [x, y, z] = rotate(scene, point);
      const scale = 420 * state.zoom / scene.bounds.extent;
      return { x: 360 + x * scale, y: 300 - y * scale, depth: z };
    }

    function addSvg(svg, tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      svg.appendChild(node);
      return node;
    }

    function drawLine(svg, scene, a, b, attrs = {}) {
      const pa = project(scene, a);
      const pb = project(scene, b);
      return addSvg(svg, "line", {
        x1: pa.x,
        y1: pa.y,
        x2: pb.x,
        y2: pb.y,
        stroke: attrs.stroke || "#555",
        "stroke-width": attrs.width || 1.5,
        "stroke-opacity": attrs.opacity ?? 1,
        "stroke-dasharray": attrs.dash || "",
      });
    }

    function drawPoint(svg, scene, point, attrs = {}) {
      const p = project(scene, point);
      return addSvg(svg, "circle", {
        cx: p.x,
        cy: p.y,
        r: attrs.radius || 3.5,
        fill: attrs.fill || "#000",
        "fill-opacity": attrs.opacity ?? 1,
        stroke: attrs.stroke || "#fff",
        "stroke-width": attrs.strokeWidth || 1,
      });
    }

    function drawPolygon(svg, scene, points, attrs = {}) {
      const projected = points.map((point) => project(scene, point));
      return addSvg(svg, "polygon", {
        points: projected.map((point) => `${point.x},${point.y}`).join(" "),
        fill: attrs.fill || "none",
        "fill-opacity": attrs.fillOpacity ?? 0,
        stroke: attrs.stroke || "none",
        "stroke-width": attrs.width || 1,
        "stroke-opacity": attrs.opacity ?? 1,
        "stroke-dasharray": attrs.dash || "",
      });
    }

    function drawLabel(svg, scene, point, text, color) {
      const p = project(scene, point);
      const node = addSvg(svg, "text", {
        x: p.x + 8,
        y: p.y - 8,
        fill: color,
        "font-size": 13,
        "font-family": "IBM Plex Mono, monospace",
        "font-weight": 600,
      });
      node.textContent = text;
    }

    function drawBox(svg, scene, corners, color) {
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      for (const [start, end] of edges) {
        drawLine(svg, scene, corners[start], corners[end], { stroke: color, width: 1.7, opacity: 0.86 });
      }
    }

    function groupColor(group, candidate, selected) {
      if (selected && group.role === "transfer") return "#0f766e";
      if (selected && group.role === "final") return "#7c3aed";
      if (candidate.status !== "accepted") return "#b91c1c";
      return group.role === "transfer" ? "#0f766e" : "#7c3aed";
    }

    function drawCandidate(svg, scene, group, candidate) {
      if (!state.showRejected && candidate.status !== "accepted") return;
      const selected = state.selectedKey[scene.id][group.id] === candidate.key || candidate.is_plan_selected;
      const color = groupColor(group, candidate, selected);
      const opacity = selected ? 1.0 : (candidate.status === "accepted" ? 0.28 : 0.11);
      const width = selected ? 3.2 : 1.35;
      const line = drawLine(svg, scene, candidate.contact_point_a_world, candidate.contact_point_b_world, {
        stroke: color,
        width,
        opacity,
      });
      const marker = drawPoint(svg, scene, candidate.grasp_position_world, {
        fill: color,
        radius: selected ? 5.5 : 2.5,
        opacity: selected ? 1.0 : opacity,
        strokeWidth: selected ? 1.4 : 0.7,
      });
      const title = `${group.label} ${candidate.grasp_id}\\n${candidate.status}: ${candidate.reason}\\nscore=${fmt(candidate.score)}`;
      for (const node of [line, marker]) {
        const titleNode = document.createElementNS("http://www.w3.org/2000/svg", "title");
        titleNode.textContent = title;
        node.appendChild(titleNode);
        node.addEventListener("click", () => {
          state.activeGroup[scene.id] = group.id;
          state.selectedKey[scene.id][group.id] = candidate.key;
          render();
        });
      }
      if (selected) {
        drawLabel(svg, scene, candidate.grasp_position_world, `${group.role}:${candidate.grasp_id}`, color);
      }
    }

    function selectedCandidate(scene, group) {
      const key = state.selectedKey[scene.id][group.id];
      return group.candidates.find((candidate) => candidate.key === key)
        || group.candidates.find((candidate) => candidate.is_plan_selected)
        || group.candidates[0];
    }

    function drawSelectedHand(svg, scene, group) {
      if (state.selectedKey[scene.id][group.id] !== group.selected_key) return;
      const hand = group.selected_hand;
      if (!hand) return;
      for (const box of hand.franka_left_boxes) drawBox(svg, scene, box.corners, "#9a5b13");
      for (const box of hand.franka_right_boxes) drawBox(svg, scene, box.corners, "#9a5b13");
      drawLine(svg, scene, hand.contact_point_a_obj, hand.contact_point_b_obj, {
        stroke: group.role === "transfer" ? "#0f766e" : "#7c3aed",
        width: 4,
        opacity: 0.95,
      });
      drawPoint(svg, scene, hand.franka_left_tip_anchor_obj, { fill: "#0f766e", radius: 4.5 });
      drawPoint(svg, scene, hand.franka_right_tip_anchor_obj, { fill: "#7c3aed", radius: 4.5 });
    }

    function visibleGroups(scene) {
      if (state.showAllGroups) return scene.groups;
      return scene.groups.filter((group) => group.id === state.activeGroup[scene.id]);
    }

    function drawScene(svg, scene) {
      svg.replaceChildren();
      drawPolygon(svg, scene, scene.floor_world, {
        fill: "#2563eb",
        fillOpacity: 0.14,
        stroke: "#2563eb",
        width: 2,
        opacity: 0.8,
        dash: "8 5",
      });
      if (scene.support && scene.support.vertices_world.length >= 3) {
        drawPolygon(svg, scene, scene.support.vertices_world, {
          fill: "#d97706",
          fillOpacity: 0.22,
          stroke: "#d97706",
          width: 2,
          opacity: 0.9,
        });
        drawPoint(svg, scene, scene.support.com_world, { fill: "#db2777", radius: 5 });
        drawPoint(svg, scene, scene.support.com_projection_world, { fill: "#fff", stroke: "#db2777", radius: 5, strokeWidth: 2 });
        drawLine(svg, scene, scene.support.com_world, scene.support.com_projection_world, { stroke: "#db2777", width: 1.5, opacity: 0.8, dash: "4 4" });
      }
      for (const [start, end] of scene.edges) {
        drawLine(svg, scene, scene.vertices_world[start], scene.vertices_world[end], {
          stroke: "#465f55",
          width: 1.35,
          opacity: 0.72,
        });
      }
      for (const group of visibleGroups(scene)) {
        for (const candidate of group.candidates) drawCandidate(svg, scene, group, candidate);
      }
      for (const group of visibleGroups(scene)) drawSelectedHand(svg, scene, group);
    }

    function renderDetails() {
      const lines = [
        ...data.metadata_lines,
        "",
        "plan_metadata:",
        ...Object.entries(data.plan_metadata).map(([key, value]) => `  ${key}: ${JSON.stringify(value)}`),
        "",
        "selection:",
      ];
      for (const scene of data.scenes) {
        const group = scene.groups.find((item) => item.id === state.activeGroup[scene.id]) || scene.groups[0];
        const candidate = selectedCandidate(scene, group);
        lines.push(
          `  ${scene.title} / ${group.label}:`,
          `    selected: ${candidate ? candidate.grasp_id : "n/a"}`,
          `    status:   ${candidate ? candidate.status : "n/a"}`,
          `    reason:   ${candidate ? candidate.reason : "n/a"}`,
          `    score:    ${candidate ? fmt(candidate.score) : "n/a"}`,
          `    accepted: ${group.counts.accepted}/${group.counts.total}`,
        );
      }
      detailsNode.textContent = lines.join("\\n");
    }

    function render() {
      scenesNode.replaceChildren();
      for (const scene of data.scenes) {
        const panel = document.createElement("section");
        panel.className = "scene-panel";
        const header = document.createElement("div");
        header.className = "scene-head";
        const titleWrap = document.createElement("div");
        const heading = document.createElement("h2");
        heading.className = "scene-title";
        heading.textContent = scene.title;
        const subtitle = document.createElement("p");
        subtitle.className = "scene-subtitle";
        subtitle.textContent = scene.subtitle;
        titleWrap.append(heading, subtitle);
        const buttons = document.createElement("div");
        buttons.className = "group-buttons";
        for (const group of scene.groups) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = `${group.label} ${group.counts.accepted}/${group.counts.total}`;
          button.className = state.activeGroup[scene.id] === group.id ? "active" : "";
          button.addEventListener("click", () => {
            state.activeGroup[scene.id] = group.id;
            render();
          });
          buttons.appendChild(button);
        }
        header.append(titleWrap, buttons);
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.classList.add("scene-svg");
        svg.setAttribute("viewBox", "0 0 720 600");
        svg.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          state.drag = { x: event.clientX, y: event.clientY, id: event.pointerId };
          svg.setPointerCapture(event.pointerId);
          svg.style.cursor = "grabbing";
        });
        svg.addEventListener("pointermove", (event) => {
          if (!state.drag || state.drag.id !== event.pointerId) return;
          const dx = event.clientX - state.drag.x;
          const dy = event.clientY - state.drag.y;
          state.drag.x = event.clientX;
          state.drag.y = event.clientY;
          state.yaw += dx * 0.01;
          state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch - dy * 0.01));
          render();
        });
        svg.addEventListener("pointerup", () => {
          state.drag = null;
          svg.style.cursor = "grab";
        });
        svg.addEventListener("wheel", (event) => {
          event.preventDefault();
          state.zoom = Math.max(0.35, Math.min(5.0, state.zoom * (event.deltaY < 0 ? 1.08 : 1 / 1.08)));
          render();
        }, { passive: false });
        const meta = document.createElement("div");
        meta.className = "scene-meta";
        meta.innerHTML = `
          <div><strong>position</strong><br>${scene.pose.position_world.join(", ")}</div>
          <div><strong>quat xyzw</strong><br>${scene.pose.orientation_xyzw_world.join(", ")}</div>
        `;
        panel.append(header, svg, meta);
        scenesNode.appendChild(panel);
        drawScene(svg, scene);
      }
      allGroupsBtn.textContent = `All Groups: ${state.showAllGroups ? "On" : "Off"}`;
      rejectedBtn.textContent = `Rejected: ${state.showRejected ? "On" : "Off"}`;
      renderDetails();
    }

    isoBtn.addEventListener("click", () => {
      state.yaw = -0.82;
      state.pitch = 0.56;
      state.zoom = 1.0;
      render();
    });
    topBtn.addEventListener("click", () => {
      state.yaw = 0.0;
      state.pitch = 0.0;
      state.zoom = 1.0;
      render();
    });
    allGroupsBtn.addEventListener("click", () => {
      state.showAllGroups = !state.showAllGroups;
      render();
    });
    rejectedBtn.addEventListener("click", () => {
      state.showRejected = !state.showRejected;
      render();
    });
    render();
  </script>
</body>
</html>
"""
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace("__DATA_JSON__", data_json), encoding="utf-8")
