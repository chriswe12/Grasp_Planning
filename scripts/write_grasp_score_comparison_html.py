#!/usr/bin/env python3
"""Write an HTML report comparing legacy and current grasp scoring for one part."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping import (  # noqa: E402
    AntipodalGraspGeneratorConfig,
    AntipodalMeshGraspGenerator,
    SavedGraspCandidate,
    TriangleMesh,
    load_grasp_bundle,
    score_grasps,
)
from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    DEFAULT_GRASP_SCORING_CONTACT_RADIUS_M,
    DEFAULT_GRASP_SCORING_SIGMA_CENTER_M,
    DEFAULT_GRASP_SCORING_SIGMA_COM_M,
    DEFAULT_GRASP_SCORING_SUPPORT_TARGET,
    CandidateStatus,
    candidate_payload,
    canonicalize_target_mesh,
    load_asset_mesh,
    mesh_area_weighted_triangle_centroid,
    quat_to_rotmat_xyzw,
    serialize_saved_candidate,
    unique_edges,
)


@dataclass(frozen=True)
class _LegacyMeshIndex:
    vertices_obj: np.ndarray
    tree: cKDTree
    center_of_mass_obj: np.ndarray


def _tuple_floats(values: object, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if values in ("", None):
        return default
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected list/tuple of floats, got {values!r}.")
    return tuple(float(value) for value in values)


def _roll_angles_from_planning(raw: dict[str, object]) -> tuple[float, ...]:
    if raw.get("roll_angle_step_deg") not in ("", None):
        step_deg = float(raw["roll_angle_step_deg"])
        if step_deg <= 0.0 or step_deg > 360.0:
            raise ValueError("planning.roll_angle_step_deg must be > 0 and <= 360.")
        count = max(1, int(math.ceil(360.0 / step_deg)))
        return tuple(float(math.radians(index * step_deg)) for index in range(count) if index * step_deg < 360.0)
    return _tuple_floats(raw.get("roll_angles_rad"), default=(0.0,))


def _generator_config_from_payload(payload: dict[str, object]) -> AntipodalGraspGeneratorConfig:
    raw = dict(payload.get("planning", {}))
    return AntipodalGraspGeneratorConfig(
        num_surface_samples=int(raw.get("num_surface_samples", 256)),
        min_jaw_width=float(raw.get("min_jaw_width", 0.002)),
        max_jaw_width=float(raw.get("max_jaw_width", 0.09)),
        antipodal_cosine_threshold=float(raw.get("antipodal_cosine_threshold", 0.984807753012208)),
        roll_angles_rad=_roll_angles_from_planning(raw),
        max_pair_checks=int(raw.get("max_pair_checks", 40960)),
        detailed_finger_contact_gap_m=float(raw.get("detailed_finger_contact_gap_m", 0.002)),
        rng_seed=int(raw.get("rng_seed", 0)),
    )


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in '{path}'.")
    return payload


def _mesh_center_of_mass(mesh: TriangleMesh) -> np.ndarray:
    try:
        import trimesh

        tri_mesh = trimesh.Trimesh(vertices=mesh.vertices_obj, faces=mesh.faces, process=False)
        center_mass = np.asarray(tri_mesh.center_mass, dtype=float)
        if center_mass.shape == (3,) and np.all(np.isfinite(center_mass)):
            return center_mass
    except Exception:
        pass
    return mesh_area_weighted_triangle_centroid(mesh)


def _build_legacy_index(mesh: TriangleMesh) -> _LegacyMeshIndex:
    vertices = np.asarray(mesh.vertices_obj, dtype=float)
    return _LegacyMeshIndex(
        vertices_obj=vertices,
        tree=cKDTree(vertices),
        center_of_mass_obj=_mesh_center_of_mass(mesh),
    )


def _project_onto_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    return np.asarray(vec, dtype=float) - float(np.dot(vec, normal)) * normal


def _legacy_contact_neighborhood_indices(
    index: _LegacyMeshIndex,
    contact_point_obj: np.ndarray,
    *,
    radius_m: float,
) -> np.ndarray:
    indices = index.tree.query_ball_point(np.asarray(contact_point_obj, dtype=float), r=float(radius_m))
    if indices:
        return np.asarray(indices, dtype=np.int64)
    _, nearest_index = index.tree.query(np.asarray(contact_point_obj, dtype=float), k=1)
    return np.asarray([int(nearest_index)], dtype=np.int64)


def _legacy_score_components(
    candidate: SavedGraspCandidate,
    *,
    mesh_index: _LegacyMeshIndex,
    sigma_center_m: float = DEFAULT_GRASP_SCORING_SIGMA_CENTER_M,
    sigma_com_m: float = DEFAULT_GRASP_SCORING_SIGMA_COM_M,
    support_target: int = DEFAULT_GRASP_SCORING_SUPPORT_TARGET,
    contact_radius_m: float = DEFAULT_GRASP_SCORING_CONTACT_RADIUS_M,
) -> dict[str, float]:
    grasp_center = np.asarray(candidate.grasp_position_obj, dtype=float)
    contact_right = np.asarray(candidate.contact_point_a_obj, dtype=float)
    contact_left = np.asarray(candidate.contact_point_b_obj, dtype=float)
    normal_right = np.asarray(candidate.contact_normal_a_obj, dtype=float)
    normal_left = np.asarray(candidate.contact_normal_b_obj, dtype=float)
    closing_axis = contact_left - contact_right
    closing_norm = float(np.linalg.norm(closing_axis))
    if closing_norm < 1.0e-12:
        raise ValueError(f"Candidate '{candidate.grasp_id}' has coincident contact points.")
    closing_axis /= closing_norm

    left_alignment = max(0.0, float(np.dot(normal_left, closing_axis)))
    right_alignment = max(0.0, float(np.dot(normal_right, -closing_axis)))
    s_align = 0.5 * (left_alignment + right_alignment)

    contact_midpoint = 0.5 * (contact_left + contact_right)
    center_offset_plane = _project_onto_plane(contact_midpoint - grasp_center, closing_axis)
    d_center = float(np.linalg.norm(center_offset_plane))
    s_center = math.exp(-((d_center * d_center) / (sigma_center_m * sigma_center_m)))

    left_indices = _legacy_contact_neighborhood_indices(mesh_index, contact_left, radius_m=contact_radius_m)
    right_indices = _legacy_contact_neighborhood_indices(mesh_index, contact_right, radius_m=contact_radius_m)
    n_left = int(left_indices.size)
    n_right = int(right_indices.size)
    s_support = min(1.0, float(n_left + n_right) / float(max(1, support_target)))

    com_offset_plane = _project_onto_plane(mesh_index.center_of_mass_obj - grasp_center, closing_axis)
    d_com = float(np.linalg.norm(com_offset_plane))
    s_com = math.exp(-((d_com * d_com) / (sigma_com_m * sigma_com_m)))

    total = 0.40 * s_align + 0.25 * s_center + 0.20 * s_support + 0.15 * s_com
    total = min(1.0, max(0.0, total))
    return {
        "antipodal_alignment": float(s_align),
        "centering": float(s_center),
        "contact_support": float(s_support),
        "com_offset": float(s_com),
        "contact_count_left": float(n_left),
        "contact_count_right": float(n_right),
        "center_offset_plane_m": float(d_center),
        "com_offset_plane_m": float(d_com),
        "score": float(total),
    }


def legacy_score_grasps(
    candidates: Iterable[SavedGraspCandidate],
    *,
    mesh_local: TriangleMesh,
) -> list[SavedGraspCandidate]:
    mesh_index = _build_legacy_index(mesh_local)
    scored: list[SavedGraspCandidate] = []
    for candidate in candidates:
        components = _legacy_score_components(candidate, mesh_index=mesh_index)
        scored.append(
            SavedGraspCandidate(
                grasp_id=candidate.grasp_id,
                grasp_position_obj=candidate.grasp_position_obj,
                grasp_orientation_xyzw_obj=candidate.grasp_orientation_xyzw_obj,
                contact_point_a_obj=candidate.contact_point_a_obj,
                contact_point_b_obj=candidate.contact_point_b_obj,
                contact_normal_a_obj=candidate.contact_normal_a_obj,
                contact_normal_b_obj=candidate.contact_normal_b_obj,
                jaw_width=candidate.jaw_width,
                roll_angle_rad=candidate.roll_angle_rad,
                contact_patch_lateral_offset_m=candidate.contact_patch_lateral_offset_m,
                contact_patch_approach_offset_m=candidate.contact_patch_approach_offset_m,
                score=components["score"],
                score_components=components,
            )
        )
    return sorted(
        scored,
        key=lambda item: (
            float("-inf") if item.score is None else float(item.score),
            item.grasp_id,
        ),
        reverse=True,
    )


def _mesh_in_bundle_source_frame(bundle) -> TriangleMesh:
    mesh_obj_world = load_asset_mesh(bundle.target_mesh_path, scale=bundle.mesh_scale)
    rotation_obj_world_from_source = np.asarray(bundle.source_frame_orientation_xyzw_obj_world, dtype=float)
    x, y, z, w = rotation_obj_world_from_source
    rot = quat_to_rotmat_xyzw((float(x), float(y), float(z), float(w)))
    translation = np.asarray(bundle.source_frame_origin_obj_world, dtype=float)
    vertices_source = (np.asarray(mesh_obj_world.vertices_obj, dtype=float) - translation[None, :]) @ rot
    return TriangleMesh(vertices_obj=vertices_source, faces=np.asarray(mesh_obj_world.faces, dtype=np.int64))


def load_candidates_from_bundle(input_json: Path) -> tuple[TriangleMesh, list[SavedGraspCandidate], str]:
    bundle = load_grasp_bundle(input_json)
    return _mesh_in_bundle_source_frame(bundle), list(bundle.candidates), str(input_json)


def generate_candidates_for_mesh(
    *,
    target_mesh_path: str,
    mesh_scale: float,
    generator_config: AntipodalGraspGeneratorConfig,
) -> tuple[TriangleMesh, list[SavedGraspCandidate], str]:
    mesh_obj_world = load_asset_mesh(target_mesh_path, scale=mesh_scale)
    mesh_local, _ = canonicalize_target_mesh(mesh_obj_world)
    generator = AntipodalMeshGraspGenerator(generator_config)
    raw_candidates = generator.generate(mesh_local)
    saved = [serialize_saved_candidate(f"g{index:04d}", candidate) for index, candidate in enumerate(raw_candidates, 1)]
    return mesh_local, saved, target_mesh_path


def _score_value(candidate: SavedGraspCandidate) -> float:
    return float("-inf") if candidate.score is None else float(candidate.score)


SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("new_rank", "New rank"),
    ("old_rank", "Old rank"),
    ("score_gain", "Most score gained"),
    ("score_loss", "Most score lost"),
)
SORT_OPTION_IDS = tuple(option_id for option_id, _ in SORT_OPTIONS)


def _ordered_records(records: list[dict[str, object]], sort_by: str) -> list[dict[str, object]]:
    if sort_by == "new_rank":
        return sorted(records, key=lambda item: (int(item["new_rank"]), str(item["grasp_id"])))
    if sort_by == "old_rank":
        return sorted(records, key=lambda item: (int(item["old_rank"]), str(item["grasp_id"])))
    if sort_by == "score_loss":
        return sorted(records, key=lambda item: (float(item["score_delta"]), str(item["grasp_id"])))
    return sorted(records, key=lambda item: (-float(item["score_delta"]), str(item["grasp_id"])))


def _comparison_candidate_payload(
    record: dict[str, object],
    *,
    contact_gap_m: float,
    display_index: int,
) -> dict[str, object]:
    candidate = record["_current_candidate"]
    if not isinstance(candidate, SavedGraspCandidate):
        raise TypeError("Internal comparison record is missing its current scored candidate.")
    payload = candidate_payload(
        [CandidateStatus(grasp=candidate, status="accepted", reason="score_comparison")],
        contact_gap_m=contact_gap_m,
    )[0]
    payload.update(
        {
            "rank": display_index,
            "comparison_rank": display_index,
            "reason": "score_comparison",
            "old_rank": int(record["old_rank"]),
            "new_rank": int(record["new_rank"]),
            "rank_delta": float(record["rank_delta"]),
            "old_score": float(record["old_score"]),
            "new_score": float(record["new_score"]),
            "score_delta": float(record["score_delta"]),
            "old_contact_support": float(record["old_contact_support"]),
            "new_contact_support": float(record["new_contact_support"]),
            "contact_support_delta": float(record["contact_support_delta"]),
            "old_components": record["old_components"],
            "new_components": record["new_components"],
        }
    )
    return payload


def build_score_comparison_payload(
    *,
    mesh_local: TriangleMesh,
    candidates: Iterable[SavedGraspCandidate],
    source_label: str,
    max_display: int,
    sort_by: str,
    contact_gap_m: float = 0.002,
) -> dict[str, object]:
    if sort_by not in SORT_OPTION_IDS:
        raise ValueError(f"Unsupported sort mode '{sort_by}'.")
    candidate_list = list(candidates)
    legacy_scored = legacy_score_grasps(candidate_list, mesh_local=mesh_local)
    current_scored = score_grasps(candidate_list, mesh_local=mesh_local)
    legacy_by_id = {candidate.grasp_id: candidate for candidate in legacy_scored}
    current_by_id = {candidate.grasp_id: candidate for candidate in current_scored}
    legacy_rank = {candidate.grasp_id: rank for rank, candidate in enumerate(legacy_scored, 1)}
    current_rank = {candidate.grasp_id: rank for rank, candidate in enumerate(current_scored, 1)}

    records: list[dict[str, object]] = []
    for grasp_id in sorted(current_by_id):
        current = current_by_id[grasp_id]
        legacy = legacy_by_id[grasp_id]
        old_score = _score_value(legacy)
        new_score = _score_value(current)
        old_components = dict(legacy.score_components or {})
        new_components = dict(current.score_components or {})
        old_support = float(old_components.get("contact_support", 0.0))
        new_support = float(new_components.get("contact_support", 0.0))
        records.append(
            {
                "grasp_id": grasp_id,
                "old_rank": legacy_rank[grasp_id],
                "new_rank": current_rank[grasp_id],
                "rank_delta": float(legacy_rank[grasp_id] - current_rank[grasp_id]),
                "old_score": old_score,
                "new_score": new_score,
                "score_delta": new_score - old_score,
                "old_contact_support": old_support,
                "new_contact_support": new_support,
                "contact_support_delta": new_support - old_support,
                "old_components": old_components,
                "new_components": new_components,
                "_current_candidate": current,
            }
        )

    sorted_records = _ordered_records(records, sort_by)
    display_records = sorted_records[: max(0, int(max_display))]
    display_payloads = [
        _comparison_candidate_payload(record, contact_gap_m=contact_gap_m, display_index=index)
        for index, record in enumerate(display_records, start=1)
    ]
    vertices = np.asarray(mesh_local.vertices_obj, dtype=float)
    faces = np.asarray(mesh_local.faces, dtype=np.int64)
    bounds_min = vertices.min(axis=0).tolist()
    bounds_max = vertices.max(axis=0).tolist()
    return {
        "source_label": source_label,
        "title": "Grasp Score Comparison",
        "subtitle": "Legacy vertex-count contact support versus current pad-footprint contact support.",
        "candidate_count": len(candidate_list),
        "display_count": len(display_payloads),
        "sort_by": sort_by,
        "sort_options": [{"id": option_id, "label": label} for option_id, label in SORT_OPTIONS],
        "old_top": None if not legacy_scored else legacy_scored[0].grasp_id,
        "new_top": None if not current_scored else current_scored[0].grasp_id,
        "bounds_min": [float(v) for v in bounds_min],
        "bounds_max": [float(v) for v in bounds_max],
        "metadata_lines": [
            f"source:           {source_label}",
            f"candidate_count:  {len(candidate_list)}",
            f"display_count:    {len(display_payloads)}",
            f"old_top:          {None if not legacy_scored else legacy_scored[0].grasp_id}",
            f"new_top:          {None if not current_scored else current_scored[0].grasp_id}",
            f"sort:             {sort_by}",
        ],
        "vertices_obj": [[float(v) for v in vertex] for vertex in vertices.tolist()],
        "faces": [[int(v) for v in face] for face in faces.tolist()],
        "edges": [[int(a), int(b)] for a, b in unique_edges(faces)],
        "edge_count_original": len(unique_edges(faces)),
        "obstacle_vertices_obj": [],
        "obstacle_edges": [],
        "obstacle_edge_count_original": 0,
        "obstacle_bounds_obj": [],
        "ground_plane_overlay": None,
        "candidates": display_payloads,
    }


def _legacy_flat_html_document(data_json: str) -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Grasp Score Comparison</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f8f7; color: #14211b; }
    header { padding: 18px 24px 12px; border-bottom: 1px solid #cfd8d3; background: #ffffff; }
    h1 { margin: 0 0 8px; font-size: 22px; font-weight: 680; }
    .meta { display: flex; flex-wrap: wrap; gap: 10px; color: #40534a; font-size: 13px; }
    .pill { border: 1px solid #cfd8d3; border-radius: 999px; padding: 4px 9px; background: #f9fbfa; }
    main { display: grid; grid-template-columns: minmax(420px, 1fr) 520px; gap: 16px; padding: 16px; }
    .panel { background: #ffffff; border: 1px solid #d5ddd9; border-radius: 8px; overflow: hidden; min-width: 0; }
    .panel h2 { margin: 0; padding: 12px 14px; font-size: 15px; border-bottom: 1px solid #d5ddd9; background: #fbfcfc; }
    .views { display: flex; gap: 8px; padding: 10px 12px; border-bottom: 1px solid #e2e8e5; }
    button { border: 1px solid #b9c7c1; background: #fff; color: #14211b; border-radius: 6px; padding: 6px 10px; cursor: pointer; }
    button.active { background: #174d3b; border-color: #174d3b; color: #fff; }
    svg { display: block; width: 100%; height: 620px; background: #fdfefe; }
    .content { padding: 12px; }
    .details { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre-wrap; line-height: 1.45; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 7px 8px; border-bottom: 1px solid #e1e7e4; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: #f2f7f4; }
    tbody tr.selected { background: #e8f2ed; }
    .pos { color: #0b6b45; }
    .neg { color: #a33a2b; }
    .zero { color: #5d6b65; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: #40534a; padding: 0 12px 12px; }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } svg { height: 520px; } }
  </style>
</head>
<body>
  <header>
    <h1>Grasp Score Comparison</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main>
    <section class="panel">
      <h2>Mesh And Grasp Candidates</h2>
      <div class="views">
        <button type="button" data-view="xy" class="active">XY</button>
        <button type="button" data-view="xz">XZ</button>
        <button type="button" data-view="yz">YZ</button>
      </div>
      <svg id="scene" viewBox="0 0 900 620" role="img" aria-label="Projected part mesh and grasp candidates"></svg>
      <div class="legend">
        <span><span class="swatch" style="background:#168354"></span>new score higher</span>
        <span><span class="swatch" style="background:#bf4938"></span>new score lower</span>
        <span><span class="swatch" style="background:#6d7d76"></span>near unchanged</span>
      </div>
    </section>
    <section class="panel">
      <h2>Score Deltas</h2>
      <div class="content">
        <table>
          <thead>
            <tr>
              <th>grasp</th>
              <th>old</th>
              <th>new</th>
              <th>delta</th>
              <th>support</th>
              <th>rank</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
      <h2>Selected Grasp</h2>
      <div class="content details" id="details"></div>
    </section>
  </main>
  <script>
    const data = __DATA_JSON__;
    let currentView = "xy";
    let selectedId = data.candidates.length ? data.candidates[0].grasp_id : null;
    const scene = document.getElementById("scene");
    const rows = document.getElementById("rows");
    const details = document.getElementById("details");

    function fmt(value, digits = 4) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }

    function cls(value) {
      if (value > 1e-6) return "pos";
      if (value < -1e-6) return "neg";
      return "zero";
    }

    function color(value) {
      if (value > 1e-6) return "#168354";
      if (value < -1e-6) return "#bf4938";
      return "#6d7d76";
    }

    function project(point) {
      if (currentView === "xz") return [point[0], point[2]];
      if (currentView === "yz") return [point[1], point[2]];
      return [point[0], point[1]];
    }

    function bounds2d() {
      const projected = data.vertices_obj.map(project);
      for (const c of data.candidates) {
        projected.push(project(c.grasp_position_obj), project(c.contact_point_a_obj), project(c.contact_point_b_obj));
      }
      const xs = projected.map(p => p[0]);
      const ys = projected.map(p => p[1]);
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      const padX = Math.max((maxX - minX) * 0.12, 0.02);
      const padY = Math.max((maxY - minY) * 0.12, 0.02);
      return [minX - padX, maxX + padX, minY - padY, maxY + padY];
    }

    function screenMapper() {
      const [minX, maxX, minY, maxY] = bounds2d();
      const width = 900, height = 620;
      const scale = Math.min(width / Math.max(maxX - minX, 1e-9), height / Math.max(maxY - minY, 1e-9));
      const usedW = (maxX - minX) * scale;
      const usedH = (maxY - minY) * scale;
      const offX = (width - usedW) * 0.5;
      const offY = (height - usedH) * 0.5;
      return (point) => {
        const [x, y] = project(point);
        return [offX + (x - minX) * scale, height - (offY + (y - minY) * scale)];
      };
    }

    function line(svg, p0, p1, attrs) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
      el.setAttribute("x1", p0[0]);
      el.setAttribute("y1", p0[1]);
      el.setAttribute("x2", p1[0]);
      el.setAttribute("y2", p1[1]);
      for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
      svg.appendChild(el);
      return el;
    }

    function circle(svg, p, attrs) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      el.setAttribute("cx", p[0]);
      el.setAttribute("cy", p[1]);
      for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
      svg.appendChild(el);
      return el;
    }

    function draw() {
      scene.replaceChildren();
      const toScreen = screenMapper();
      for (const [a, b] of data.edges) {
        line(scene, toScreen(data.vertices_obj[a]), toScreen(data.vertices_obj[b]), {
          stroke: "#b9c5c0", "stroke-width": 1.1, opacity: 0.7
        });
      }
      for (const c of data.candidates) {
        const isSelected = c.grasp_id === selectedId;
        const stroke = color(c.score_delta);
        const center = toScreen(c.grasp_position_obj);
        line(scene, toScreen(c.contact_point_a_obj), toScreen(c.contact_point_b_obj), {
          stroke, "stroke-width": isSelected ? 4 : 2, opacity: isSelected ? 1 : 0.68
        });
        circle(scene, center, {
          r: isSelected ? 6 : 3.5,
          fill: stroke,
          opacity: isSelected ? 1 : 0.78,
          "data-id": c.grasp_id
        }).addEventListener("click", () => select(c.grasp_id));
        if (isSelected) {
          const zEnd = [
            c.grasp_position_obj[0] + c.gripper_z_axis_obj[0] * 0.035,
            c.grasp_position_obj[1] + c.gripper_z_axis_obj[1] * 0.035,
            c.grasp_position_obj[2] + c.gripper_z_axis_obj[2] * 0.035
          ];
          line(scene, center, toScreen(zEnd), { stroke: "#1d4ed8", "stroke-width": 3, opacity: 0.95 });
        }
      }
    }

    function select(id) {
      selectedId = id;
      renderRows();
      renderDetails();
      draw();
    }

    function renderRows() {
      rows.replaceChildren();
      for (const c of data.candidates) {
        const tr = document.createElement("tr");
        if (c.grasp_id === selectedId) tr.classList.add("selected");
        tr.innerHTML = `
          <td>${c.grasp_id}</td>
          <td>${fmt(c.old_score)}</td>
          <td>${fmt(c.new_score)}</td>
          <td class="${cls(c.score_delta)}">${fmt(c.score_delta, 5)}</td>
          <td class="${cls(c.contact_support_delta)}">${fmt(c.old_contact_support)} -> ${fmt(c.new_contact_support)}</td>
          <td>${c.old_rank} -> ${c.new_rank}</td>
        `;
        tr.addEventListener("click", () => select(c.grasp_id));
        rows.appendChild(tr);
      }
    }

    function renderDetails() {
      const c = data.candidates.find(item => item.grasp_id === selectedId);
      if (!c) {
        details.textContent = "No candidate selected.";
        return;
      }
      const newPad = [
        `left pad:  score=${fmt(c.new_components.pad_support_left)} fraction=${fmt(c.new_components.pad_support_fraction_left)} normal=${fmt(c.new_components.pad_normal_consistency_left)}`,
        `right pad: score=${fmt(c.new_components.pad_support_right)} fraction=${fmt(c.new_components.pad_support_fraction_right)} normal=${fmt(c.new_components.pad_normal_consistency_right)}`
      ].join("\\n");
      details.textContent = [
        `grasp_id: ${c.grasp_id}`,
        `rank:     old ${c.old_rank} -> new ${c.new_rank}`,
        `score:    old ${fmt(c.old_score, 6)} -> new ${fmt(c.new_score, 6)}  delta ${fmt(c.score_delta, 6)}`,
        `support:  old ${fmt(c.old_contact_support, 6)} -> new ${fmt(c.new_contact_support, 6)}  delta ${fmt(c.contact_support_delta, 6)}`,
        "",
        "legacy vertex-count support:",
        `left vertices:  ${fmt(c.old_components.contact_count_left, 0)}`,
        `right vertices: ${fmt(c.old_components.contact_count_right, 0)}`,
        "",
        "new pad-footprint support:",
        newPad,
        "",
        "new score components:",
        JSON.stringify(c.new_components, null, 2),
        "",
        "old score components:",
        JSON.stringify(c.old_components, null, 2)
      ].join("\\n");
    }

    document.querySelectorAll("button[data-view]").forEach(button => {
      button.addEventListener("click", () => {
        currentView = button.dataset.view;
        document.querySelectorAll("button[data-view]").forEach(item => item.classList.toggle("active", item === button));
        draw();
      });
    });

    document.getElementById("meta").replaceChildren(
      ...[
        `source: ${data.source_label}`,
        `candidates: ${data.candidate_count}`,
        `displayed: ${data.display_count}`,
        `old top: ${data.old_top || "n/a"}`,
        `new top: ${data.new_top || "n/a"}`,
        `sort: ${data.sort_by}`
      ].map(text => {
        const span = document.createElement("span");
        span.className = "pill";
        span.textContent = text;
        return span;
      })
    );
    renderRows();
    renderDetails();
    draw();
  </script>
</body>
</html>
""".replace("__DATA_JSON__", data_json)


def _html_document(data_json: str) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Grasp Score Comparison</title>
  <style>
    :root {
      --bg: #f3efe4;
      --panel: #fffaf0;
      --ink: #1e1d1a;
      --accent: #b43f2c;
      --accent-soft: #e8b59f;
      --muted: #6f6a5f;
      --mesh: #4f6b5f;
      --obstacle: #64748b;
      --ground: #2563eb;
      --accepted: #15803d;
      --rejected: #b91c1c;
      --contact-a: #c8452d;
      --contact-b: #1f7c60;
      --franka: #d97706;
      --hand: #8f5a12;
      --axis: #1397a6;
      --delta-up: #168354;
      --delta-down: #bf4938;
      --delta-flat: #6d7d76;
      --line: #d9ceb8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8e8 0, transparent 30%),
        linear-gradient(135deg, #f7f2e7 0%, #efe7d4 100%);
    }
    .layout { display: grid; grid-template-columns: 390px minmax(0, 1fr); min-height: 100vh; }
    .sidebar { border-right: 1px solid var(--line); background: rgba(255,250,240,0.92); padding: 20px 18px; overflow: auto; }
    .title { margin: 0 0 8px; font-size: 28px; line-height: 1.1; }
    .subtitle { margin: 0 0 18px; color: var(--muted); font-size: 14px; line-height: 1.5; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; align-items: center; }
    button, select { border: 1px solid var(--line); background: white; color: var(--ink); border-radius: 999px; padding: 10px 14px; font: inherit; }
    button { cursor: pointer; }
    button:hover, select:hover { border-color: var(--accent); }
    .sort-control { display: grid; gap: 5px; width: 100%; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .sort-control select { width: 100%; border-radius: 10px; text-transform: none; letter-spacing: 0; color: var(--ink); }
    .list { display: grid; gap: 10px; margin-bottom: 18px; }
    .item { border: 1px solid var(--line); border-radius: 16px; padding: 12px 14px; background: rgba(255,255,255,0.7); cursor: pointer; transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease; text-align: left; }
    .item:hover { transform: translateY(-1px); border-color: var(--accent-soft); box-shadow: 0 8px 18px rgba(85,65,42,0.08); }
    .item.active { border-color: var(--accent); box-shadow: 0 10px 24px rgba(180,63,44,0.18); background: #fff; }
    .item-rank { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .item-main { display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px; gap: 10px; }
    .item-label { font-size: 20px; font-weight: 700; }
    .item-score { font-family: "IBM Plex Mono", monospace; font-size: 13px; white-space: nowrap; }
    .item-meta { margin-top: 8px; color: var(--muted); font-size: 13px; font-family: "IBM Plex Mono", monospace; line-height: 1.45; }
    .status.accepted { color: var(--accepted); }
    .status.rejected { color: var(--rejected); }
    .status.stage1_pass { color: var(--ground); }
    .delta.up { color: var(--delta-up); }
    .delta.down { color: var(--delta-down); }
    .delta.flat { color: var(--delta-flat); }
    .main { padding: 18px; overflow: auto; }
    .cards { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr); gap: 18px; align-items: start; }
    .card { border: 1px solid var(--line); border-radius: 20px; background: rgba(255,250,240,0.88); padding: 16px; box-shadow: 0 14px 32px rgba(72,51,28,0.08); }
    .card h2 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0.03em; text-transform: uppercase; }
    #scene {
      width: 100%;
      height: auto;
      aspect-ratio: 1.25 / 1;
      display: block;
      background:
        radial-gradient(circle at 20% 18%, rgba(255,255,255,0.9), rgba(255,255,255,0.55) 35%, rgba(233,226,208,0.65)),
        linear-gradient(180deg, rgba(255,255,255,0.2), rgba(223,214,194,0.18));
      border-radius: 16px;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; font-size: 13px; color: var(--muted); }
    .legend span { display: inline-flex; align-items: center; gap: 8px; }
    .swatch { width: 14px; height: 14px; border-radius: 999px; display: inline-block; }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 13px; line-height: 1.55; margin: 0; }
    .caption { margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.45; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .cards { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 id="title" class="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <div class="controls">
        <button id="prevBtn" type="button">Prev</button>
        <button id="nextBtn" type="button">Next</button>
        <button id="meshModeBtn" type="button">Solid Mesh</button>
        <button id="acceptedOnlyBtn" type="button">Accepted Only: Off</button>
        <label class="sort-control" for="sortSelect">
          Sort
          <select id="sortSelect"></select>
        </label>
      </div>
      <div id="graspList" class="list"></div>
    </aside>
    <main class="main">
      <div class="cards">
        <section class="card">
          <h2>Object Frame</h2>
          <svg id="scene" viewBox="0 0 960 760"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--mesh)"></i>Target mesh</span>
            <span><i class="swatch" style="background: var(--obstacle)"></i>Assembly obstacles</span>
            <span><i class="swatch" style="background: var(--ground)"></i>Ground plane</span>
            <span><i class="swatch" style="background: var(--accepted)"></i>Accepted</span>
            <span><i class="swatch" style="background: var(--rejected)"></i>Rejected</span>
            <span><i class="swatch" style="background: var(--franka)"></i>Gripper finger boxes</span>
            <span><i class="swatch" style="background: var(--hand)"></i>Gripper mesh</span>
            <span><i class="swatch" style="background: #0f766e"></i>3x3 contact grid</span>
            <span><i class="swatch" style="background: var(--delta-up)"></i>New score higher</span>
            <span><i class="swatch" style="background: var(--delta-down)"></i>New score lower</span>
          </div>
          <p class="caption">Left drag rotates, middle drag pans, scroll zooms, and arrow keys switch candidates.</p>
        </section>
        <section class="card">
          <h2>Selection</h2>
          <pre id="details" class="kv"></pre>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const title = document.getElementById("title");
    const subtitle = document.getElementById("subtitle");
    const graspList = document.getElementById("graspList");
    const scene = document.getElementById("scene");
    const details = document.getElementById("details");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const meshModeBtn = document.getElementById("meshModeBtn");
    const acceptedOnlyBtn = document.getElementById("acceptedOnlyBtn");
    const sortSelect = document.getElementById("sortSelect");
    title.textContent = data.title || "Grasp Score Comparison";
    subtitle.textContent = data.subtitle || "";
    (data.sort_options || []).forEach((optionInfo) => {
      const option = document.createElement("option");
      option.value = optionInfo.id;
      option.textContent = optionInfo.label;
      option.selected = optionInfo.id === data.sort_by;
      sortSelect.appendChild(option);
    });
    const state = {
      selectedIndex: 0,
      yaw: -0.82,
      pitch: 0.56,
      zoom: 1.0,
      panX: 0,
      panY: 0,
      dragging: false,
      dragMode: "rotate",
      lastPointerX: 0,
      lastPointerY: 0,
      pointerId: null,
      meshRenderMode: "wireframe",
      acceptedOnly: false,
      sortBy: data.sort_by || "new_rank",
    };
    function compareId(a, b) {
      return String(a.grasp_id).localeCompare(String(b.grasp_id));
    }
    function compareRank(a, b, key) {
      return Number(a[key]) - Number(b[key]) || compareId(a, b);
    }
    function orderedCandidates() {
      const candidates = data.candidates.slice();
      if (state.sortBy === "new_rank") return candidates.sort((a, b) => compareRank(a, b, "new_rank"));
      if (state.sortBy === "old_rank") return candidates.sort((a, b) => compareRank(a, b, "old_rank"));
      if (state.sortBy === "score_loss") return candidates.sort((a, b) => Number(a.score_delta) - Number(b.score_delta) || compareId(a, b));
      return candidates.sort((a, b) => Number(b.score_delta) - Number(a.score_delta) || compareId(a, b));
    }
    function visibleCandidates() {
      const candidates = orderedCandidates();
      return state.acceptedOnly ? candidates.filter((candidate) => candidate.status === "accepted") : candidates;
    }
    const points = [
      ...data.vertices_obj,
      ...(data.obstacle_vertices_obj || []),
      ...(data.obstacle_bounds_obj || []),
      ...(data.ground_plane_overlay ? data.ground_plane_overlay.corners_obj : []),
      ...data.candidates.flatMap((candidate) => [
        candidate.grasp_position_obj,
        candidate.contact_point_a_obj,
        candidate.contact_point_b_obj,
        ...(candidate.franka_hand_vertices_obj || []),
        ...((candidate.franka_left_boxes || []).flatMap((box) => box.corners)),
        ...((candidate.franka_right_boxes || []).flatMap((box) => box.corners)),
      ]),
    ];
    const bounds = points.reduce((acc, point) => {
      point.forEach((value, axis) => { acc.min[axis] = Math.min(acc.min[axis], value); acc.max[axis] = Math.max(acc.max[axis], value); });
      return acc;
    }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    const center = bounds.min.map((value, axis) => 0.5 * (value + bounds.max[axis]));
    const extent = Math.max(...bounds.max.map((value, axis) => value - bounds.min[axis]), 0.18);
    const baseScale = 520 / extent;
    function rotate(point) {
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function project(point) {
      const [x, y, z] = rotate(point);
      const scale = baseScale * state.zoom;
      return { x: 480 + state.panX + x * scale, y: 380 + state.panY - y * scale, depth: z };
    }
    function wrapAngle(angle) {
      const tau = Math.PI * 2;
      let wrapped = angle % tau;
      if (wrapped <= -Math.PI) wrapped += tau;
      else if (wrapped > Math.PI) wrapped -= tau;
      return wrapped;
    }
    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }
    function fmt(value, digits = 6) {
      if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }
    function fmtVec(vec) {
      return `(${vec.map((value) => value >= 0 ? `+${value.toFixed(4)}` : value.toFixed(4)).join(", ")})`;
    }
    function deltaClass(value) {
      if (Number(value) > 1e-6) return "up";
      if (Number(value) < -1e-6) return "down";
      return "flat";
    }
    function scoreDeltaColor(value) {
      if (Number(value) > 1e-6) return "#168354";
      if (Number(value) < -1e-6) return "#bf4938";
      return "#6d7d76";
    }
    function componentValue(components, key, digits = 6) {
      return components && components[key] !== undefined ? fmt(components[key], digits) : "n/a";
    }
    function addSvg(tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => {
        if (value !== null && value !== undefined) node.setAttribute(key, String(value));
      });
      scene.appendChild(node);
      return node;
    }
    function drawLine(a, b, options = {}) {
      const pa = project(a), pb = project(b);
      addSvg("line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: options.stroke || "#555", "stroke-width": options.strokeWidth || 2, "stroke-opacity": options.opacity ?? 1, "stroke-dasharray": options.dash || "", "marker-end": options.markerEnd || "" });
    }
    function drawPoint(point, options = {}) {
      const p = project(point);
      addSvg("circle", { cx: p.x, cy: p.y, r: options.radius || 6, fill: options.fill || "#000", "fill-opacity": options.opacity ?? 1, stroke: options.stroke || "white", "stroke-width": options.strokeWidth || 2 });
    }
    function drawPolygon(points, options = {}) {
      const projected = points.map((point) => project(point));
      addSvg("polygon", { points: projected.map((point) => `${point.x},${point.y}`).join(" "), fill: options.fill || "none", "fill-opacity": options.fillOpacity ?? 1, stroke: options.stroke || "none", "stroke-width": options.strokeWidth || 1, "stroke-opacity": options.strokeOpacity ?? 1 });
    }
    function drawLabel(point, text, fill, dx = 8, dy = -8) {
      const p = project(point);
      const node = addSvg("text", { x: p.x + dx, y: p.y + dy, fill, "font-size": 15, "font-family": "IBM Plex Mono, monospace", "font-weight": 600 });
      node.textContent = text;
    }
    function drawBox(corners, color) {
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      edges.forEach(([s,e]) => drawLine(corners[s], corners[e], { stroke: color, strokeWidth: 1.8, opacity: 0.8 }));
    }
    function shadeColor(hex, factor) {
      const clean = hex.replace("#", "");
      const value = Number.parseInt(clean, 16);
      const r = (value >> 16) & 255;
      const g = (value >> 8) & 255;
      const b = value & 255;
      const scale = clamp(factor, 0, 1.4);
      return `#${[r,g,b].map((channel) => clamp(Math.round(channel * scale), 0, 255)).map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
    }
    function drawMeshEdges(vertices, edges, stroke, width, opacity) {
      edges.forEach(([start, end]) => drawLine(vertices[start], vertices[end], { stroke, strokeWidth: width, opacity }));
    }
    function drawTargetMesh() {
      if (state.meshRenderMode === "solid") {
        const faces = data.faces.map((face) => {
          const points = face.map((index) => data.vertices_obj[index]);
          const rotated = points.map((point) => rotate(point));
          const edgeA = rotated[1].map((value, axis) => value - rotated[0][axis]);
          const edgeB = rotated[2].map((value, axis) => value - rotated[0][axis]);
          const normal = [
            edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
            edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
            edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
          ];
          const depth = rotated.reduce((sum, point) => sum + point[2], 0) / rotated.length;
          return { points, normal, depth };
        });
        faces
          .filter((face) => face.normal[2] > 0)
          .sort((a, b) => a.depth - b.depth)
          .forEach((face) => {
            const norm = Math.hypot(face.normal[0], face.normal[1], face.normal[2]) || 1;
            const light = 0.45 + 0.55 * (face.normal[2] / norm);
            drawPolygon(face.points, { fill: shadeColor("#4f6b5f", 0.7 + light * 0.45), fillOpacity: 0.92, stroke: "#32453d", strokeWidth: 1.2, strokeOpacity: 0.55 });
          });
        return;
      }
      drawMeshEdges(data.vertices_obj, data.edges, "#4f6b5f", 2.0, 0.8);
    }
    function drawHandMesh(candidate) {
      if (!candidate.franka_hand_faces || !candidate.franka_hand_vertices_obj) return;
      candidate.franka_hand_faces.forEach((face) => {
        drawLine(candidate.franka_hand_vertices_obj[face[0]], candidate.franka_hand_vertices_obj[face[1]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
        drawLine(candidate.franka_hand_vertices_obj[face[1]], candidate.franka_hand_vertices_obj[face[2]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
        drawLine(candidate.franka_hand_vertices_obj[face[2]], candidate.franka_hand_vertices_obj[face[0]], { stroke: "#8f5a12", strokeWidth: 1.1, opacity: 0.35 });
      });
    }
    function drawContactGrid(gridPoints, selectedPoint, gridColor, selectedColor) {
      (gridPoints || []).forEach((point) => {
        drawPoint(point, { fill: gridColor, radius: 2.4, opacity: 0.8, stroke: "white", strokeWidth: 0.8 });
      });
      if (selectedPoint) drawPoint(selectedPoint, { fill: selectedColor, radius: 4.2, opacity: 1.0, stroke: "white", strokeWidth: 1.2 });
    }
    function renderList() {
      graspList.replaceChildren();
      visibleCandidates().forEach((candidate, index) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = `item${index === state.selectedIndex ? " active" : ""}`;
        item.innerHTML = `
          <div class="item-rank">#${candidate.comparison_rank} ${candidate.grasp_id}</div>
          <div class="item-main">
            <div class="item-label status ${candidate.status}">${candidate.status}</div>
            <div class="item-score delta ${deltaClass(candidate.score_delta)}">delta=${fmt(candidate.score_delta, 4)}</div>
          </div>
          <div class="item-meta">old=${fmt(candidate.old_score, 4)} new=${fmt(candidate.new_score, 4)}<br>rank old ${candidate.old_rank} -> new ${candidate.new_rank}<br>support ${fmt(candidate.old_contact_support, 4)} -> ${fmt(candidate.new_contact_support, 4)}<br>roll=${fmt(candidate.roll_angle_rad, 3)} w=${fmt(candidate.jaw_width, 4)} center=${fmtVec(candidate.grasp_position_obj)}</div>
        `;
        item.addEventListener("click", () => { state.selectedIndex = index; render(); });
        graspList.appendChild(item);
      });
    }
    function renderScene(candidate) {
      scene.replaceChildren();
      const defs = addSvg("defs", {});
      const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
      marker.setAttribute("id", "arrow");
      marker.setAttribute("markerWidth", "8");
      marker.setAttribute("markerHeight", "8");
      marker.setAttribute("refX", "7");
      marker.setAttribute("refY", "4");
      marker.setAttribute("orient", "auto");
      marker.innerHTML = '<path d="M0,0 L8,4 L0,8 z" fill="currentColor"></path>';
      defs.appendChild(marker);
      if (data.ground_plane_overlay) {
        const corners = data.ground_plane_overlay.corners_obj;
        drawPolygon(corners, { fill: "#2563eb", fillOpacity: 0.16, stroke: "#2563eb", strokeWidth: 2, strokeOpacity: 0.75 });
        for (let i = 0; i < corners.length; i += 1) drawLine(corners[i], corners[(i + 1) % corners.length], { stroke: "#2563eb", strokeWidth: 2, opacity: 0.9, dash: "10 6" });
        drawLabel(corners[0], "z=0 plane", "#2563eb", 10, -8);
      }
      drawMeshEdges(data.obstacle_vertices_obj || [], data.obstacle_edges || [], "#64748b", 1.4, 0.45);
      drawTargetMesh();
      (candidate.franka_left_boxes || []).forEach((box) => drawBox(box.corners, "#d97706"));
      (candidate.franka_right_boxes || []).forEach((box) => drawBox(box.corners, "#d97706"));
      drawHandMesh(candidate);
      drawContactGrid(candidate.franka_left_contact_grid_obj, candidate.franka_left_tip_anchor_obj, "#0f766e", "#14b8a6");
      drawContactGrid(candidate.franka_right_contact_grid_obj, candidate.franka_right_tip_anchor_obj, "#0f766e", "#14b8a6");
      const deltaColor = scoreDeltaColor(candidate.score_delta);
      const statusColor = candidate.status === "accepted" ? "#15803d" : "#b91c1c";
      drawLine(candidate.contact_point_a_obj, candidate.contact_point_b_obj, { stroke: deltaColor, strokeWidth: 3, opacity: 0.95 });
      drawPoint(candidate.grasp_position_obj, { fill: statusColor, radius: 7 });
      drawPoint(candidate.contact_point_a_obj, { fill: "#c8452d", radius: 6 });
      drawPoint(candidate.contact_point_b_obj, { fill: "#1f7c60", radius: 6 });
      drawLabel(candidate.grasp_position_obj, `${candidate.grasp_id} d=${fmt(candidate.score_delta, 3)}`, deltaColor);
    }
    function renderDetails(candidate) {
      details.textContent = [
        ...data.metadata_lines,
        `grasp_id:         ${candidate.grasp_id}`,
        `status:           ${candidate.status}`,
        `comparison_rank:  ${candidate.comparison_rank}`,
        `old_rank:         ${candidate.old_rank}`,
        `new_rank:         ${candidate.new_rank}`,
        `rank_delta:       ${fmt(candidate.rank_delta, 0)}`,
        `old_score:        ${fmt(candidate.old_score, 6)}`,
        `new_score:        ${fmt(candidate.new_score, 6)}`,
        `score_delta:      ${fmt(candidate.score_delta, 6)}`,
        `old_support:      ${fmt(candidate.old_contact_support, 6)}`,
        `new_support:      ${fmt(candidate.new_contact_support, 6)}`,
        `support_delta:    ${fmt(candidate.contact_support_delta, 6)}`,
        "",
        "legacy vertex-count support:",
        `left vertices:    ${componentValue(candidate.old_components, "contact_count_left", 0)}`,
        `right vertices:   ${componentValue(candidate.old_components, "contact_count_right", 0)}`,
        "",
        "new pad-footprint support:",
        `left pad score:   ${componentValue(candidate.new_components, "pad_support_left")}`,
        `left fraction:    ${componentValue(candidate.new_components, "pad_support_fraction_left")}`,
        `left normal:      ${componentValue(candidate.new_components, "pad_normal_consistency_left")}`,
        `left samples:     ${componentValue(candidate.new_components, "pad_supported_samples_left", 0)} / ${componentValue(candidate.new_components, "pad_total_samples_left", 0)}`,
        `right pad score:  ${componentValue(candidate.new_components, "pad_support_right")}`,
        `right fraction:   ${componentValue(candidate.new_components, "pad_support_fraction_right")}`,
        `right normal:     ${componentValue(candidate.new_components, "pad_normal_consistency_right")}`,
        `right samples:    ${componentValue(candidate.new_components, "pad_supported_samples_right", 0)} / ${componentValue(candidate.new_components, "pad_total_samples_right", 0)}`,
        "",
        "new score components:",
        JSON.stringify(candidate.new_components, null, 2),
        "",
        "old score components:",
        JSON.stringify(candidate.old_components, null, 2),
        "",
        `jaw_width:        ${fmt(candidate.jaw_width, 6)} m`,
        `roll_angle_rad:   ${fmt(candidate.roll_angle_rad, 6)}`,
        `contact_offset_x: ${fmt(candidate.contact_patch_lateral_offset_m, 6)} m`,
        `contact_offset_z: ${fmt(candidate.contact_patch_approach_offset_m, 6)} m`,
        `grasp_position:   (${candidate.grasp_position_obj.join(", ")})`,
        `contact_a:        (${candidate.contact_point_a_obj.join(", ")})`,
        `contact_b:        (${candidate.contact_point_b_obj.join(", ")})`,
      ].join("\\n");
    }
    function render() {
      const candidates = visibleCandidates();
      if (candidates.length === 0) {
        details.textContent = [...data.metadata_lines, "No candidates to display."].join("\\n");
        graspList.replaceChildren();
        scene.replaceChildren();
        return;
      }
      if (state.selectedIndex >= candidates.length) {
        state.selectedIndex = 0;
      }
      const candidate = candidates[state.selectedIndex];
      renderList();
      renderScene(candidate);
      renderDetails(candidate);
    }
    let sceneRenderPending = false;
    function renderCurrentScene() {
      const candidates = visibleCandidates();
      if (candidates.length === 0) {
        scene.replaceChildren();
        return;
      }
      if (state.selectedIndex >= candidates.length) {
        state.selectedIndex = 0;
      }
      renderScene(candidates[state.selectedIndex]);
    }
    function scheduleSceneRender() {
      if (sceneRenderPending) return;
      sceneRenderPending = true;
      window.requestAnimationFrame(() => {
        sceneRenderPending = false;
        renderCurrentScene();
      });
    }
    window.addEventListener("keydown", (event) => {
      const candidates = visibleCandidates();
      if (candidates.length === 0) return;
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") { event.preventDefault(); state.selectedIndex = (state.selectedIndex - 1 + candidates.length) % candidates.length; render(); }
      if (event.key === "ArrowDown" || event.key === "ArrowRight") { event.preventDefault(); state.selectedIndex = (state.selectedIndex + 1) % candidates.length; render(); }
    });
    prevBtn.addEventListener("click", () => {
      const candidates = visibleCandidates();
      if (candidates.length === 0) return;
      state.selectedIndex = (state.selectedIndex - 1 + candidates.length) % candidates.length;
      render();
    });
    nextBtn.addEventListener("click", () => {
      const candidates = visibleCandidates();
      if (candidates.length === 0) return;
      state.selectedIndex = (state.selectedIndex + 1) % candidates.length;
      render();
    });
    meshModeBtn.addEventListener("click", () => {
      state.meshRenderMode = state.meshRenderMode === "wireframe" ? "solid" : "wireframe";
      meshModeBtn.textContent = state.meshRenderMode === "wireframe" ? "Solid Mesh" : "Wireframe Mesh";
      renderCurrentScene();
    });
    acceptedOnlyBtn.addEventListener("click", () => {
      state.acceptedOnly = !state.acceptedOnly;
      state.selectedIndex = 0;
      acceptedOnlyBtn.textContent = `Accepted Only: ${state.acceptedOnly ? "On" : "Off"}`;
      render();
    });
    sortSelect.addEventListener("change", () => {
      const current = visibleCandidates()[state.selectedIndex];
      state.sortBy = sortSelect.value;
      const candidates = visibleCandidates();
      const nextIndex = current ? candidates.findIndex((candidate) => candidate.grasp_id === current.grasp_id) : 0;
      state.selectedIndex = Math.max(0, nextIndex);
      render();
    });
    scene.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 1) return;
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 ? "pan" : "rotate";
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      state.pointerId = event.pointerId;
      scene.setPointerCapture(event.pointerId);
      scene.style.cursor = state.dragMode === "pan" ? "move" : "grabbing";
    });
    function stopDragging() {
      state.dragging = false;
      state.pointerId = null;
      scene.style.cursor = "grab";
    }
    scene.addEventListener("pointerup", (event) => {
      if (state.pointerId === event.pointerId) stopDragging();
    });
    scene.addEventListener("pointercancel", () => { stopDragging(); });
    scene.addEventListener("pointermove", (event) => {
      if (!state.dragging || (state.pointerId !== null && event.pointerId !== state.pointerId)) return;
      const dx = event.clientX - state.lastPointerX;
      const dy = event.clientY - state.lastPointerY;
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      if (state.dragMode === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.yaw = wrapAngle(state.yaw + dx * 0.01);
        state.pitch = wrapAngle(state.pitch - dy * 0.01);
      }
      scheduleSceneRender();
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 1.08 : 1 / 1.08;
      state.zoom = clamp(state.zoom * zoomFactor, 0.35, 4.0);
      scheduleSceneRender();
    }, { passive: false });
    scene.style.cursor = "grab";
    scene.addEventListener("contextmenu", (event) => event.preventDefault());
    render();
  </script>
</body>
</html>
""".replace("__DATA_JSON__", data_json)


def write_score_comparison_html(output_html: Path, payload: dict[str, object]) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, sort_keys=True)
    output_html.write_text(_html_document(data_json), encoding="utf-8")


def _source_from_args(args: argparse.Namespace) -> tuple[TriangleMesh, list[SavedGraspCandidate], str]:
    if args.input_json is not None:
        return load_candidates_from_bundle(args.input_json)

    config_payload: dict[str, object] = {}
    if args.config is not None:
        config_payload = _load_yaml(args.config)
    raw_geometry = dict(config_payload.get("geometry", {}))
    target_mesh_path = args.target_mesh or raw_geometry.get("target_mesh_path")
    if target_mesh_path in ("", None):
        raise ValueError("Provide --input-json, --target-mesh, or --config with geometry.target_mesh_path.")
    mesh_scale = float(args.mesh_scale if args.mesh_scale is not None else raw_geometry.get("mesh_scale", 1.0))
    generator_config = _generator_config_from_payload(config_payload)
    if args.num_surface_samples is not None:
        generator_config = AntipodalGraspGeneratorConfig(
            num_surface_samples=int(args.num_surface_samples),
            min_jaw_width=generator_config.min_jaw_width,
            max_jaw_width=generator_config.max_jaw_width,
            antipodal_cosine_threshold=generator_config.antipodal_cosine_threshold,
            roll_angles_rad=generator_config.roll_angles_rad,
            max_pair_checks=generator_config.max_pair_checks,
            detailed_finger_contact_gap_m=generator_config.detailed_finger_contact_gap_m,
            rng_seed=generator_config.rng_seed,
        )
    if args.max_pair_checks is not None:
        generator_config = AntipodalGraspGeneratorConfig(
            num_surface_samples=generator_config.num_surface_samples,
            min_jaw_width=generator_config.min_jaw_width,
            max_jaw_width=generator_config.max_jaw_width,
            antipodal_cosine_threshold=generator_config.antipodal_cosine_threshold,
            roll_angles_rad=generator_config.roll_angles_rad,
            max_pair_checks=int(args.max_pair_checks),
            detailed_finger_contact_gap_m=generator_config.detailed_finger_contact_gap_m,
            rng_seed=generator_config.rng_seed,
        )
    return generate_candidates_for_mesh(
        target_mesh_path=str(target_mesh_path),
        mesh_scale=mesh_scale,
        generator_config=generator_config,
    )


def _contact_gap_from_args(args: argparse.Namespace) -> float:
    if args.contact_gap_m is not None:
        return float(args.contact_gap_m)
    if args.config is None:
        return 0.002
    config_payload = _load_yaml(args.config)
    raw_planning = dict(config_payload.get("planning", {}))
    return float(raw_planning.get("detailed_finger_contact_gap_m", 0.002))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML report comparing legacy vertex-count and current pad-footprint grasp scores.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-json", type=Path, help="Existing saved grasp bundle JSON to compare.")
    source.add_argument("--target-mesh", help="Asset-relative or absolute mesh path to generate grasps for.")
    parser.add_argument("--config", type=Path, help="Pipeline YAML to reuse geometry/planning generation settings.")
    parser.add_argument(
        "--mesh-scale", type=float, help="Mesh scale for --target-mesh; overrides config geometry.mesh_scale."
    )
    parser.add_argument("--num-surface-samples", type=int, help="Override generation sample count for --target-mesh.")
    parser.add_argument("--max-pair-checks", type=int, help="Override generation pair-check cap for --target-mesh.")
    parser.add_argument(
        "--max-display", type=int, default=250, help="Maximum candidates to include in the HTML table/view."
    )
    parser.add_argument(
        "--contact-gap-m",
        type=float,
        help="Finger contact gap used for displayed gripper geometry; defaults to config planning value or 0.002.",
    )
    parser.add_argument(
        "--sort-by",
        choices=SORT_OPTION_IDS,
        default="new_rank",
        help="How to choose/order displayed candidates.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/grasp_score_comparison.html"),
        help="Output HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mesh_local, candidates, source_label = _source_from_args(args)
    contact_gap_m = _contact_gap_from_args(args)
    if not candidates:
        raise RuntimeError("No grasp candidates were available for comparison.")
    payload = build_score_comparison_payload(
        mesh_local=mesh_local,
        candidates=candidates,
        source_label=source_label,
        max_display=args.max_display,
        sort_by=args.sort_by,
        contact_gap_m=contact_gap_m,
    )
    write_score_comparison_html(args.output_html, payload)
    print(
        f"Wrote {args.output_html} with {payload['display_count']} / {payload['candidate_count']} candidates "
        f"(old_top={payload['old_top']}, new_top={payload['new_top']})."
    )


if __name__ == "__main__":
    main()
