#!/usr/bin/env python3
"""Write a mesh-only HTML debug view for the part local frame and execution transform."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    fmt_vec,
    load_asset_mesh,
    load_grasp_bundle,
    mesh_area_weighted_triangle_centroid,
    object_point_to_world,
    relative_asset_mesh_path,
    unique_edges,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline.fabrica_pipeline import _mesh_in_source_frame  # noqa: E402


def _tuple_floats(values: object, *, expected_len: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a list/tuple of floats, got {values!r}.")
    result = tuple(float(value) for value in values)
    if len(result) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(result)}.")
    return result


def _pose_from_metadata(metadata: dict[str, object]) -> ObjectWorldPose | None:
    raw = metadata.get("execution_world_pose")
    if not isinstance(raw, dict):
        return None
    return ObjectWorldPose(
        position_world=_tuple_floats(raw["position_world"], expected_len=3),
        orientation_xyzw_world=_tuple_floats(raw["orientation_xyzw_world"], expected_len=4),
    )


def _mesh_local_from_bundle(input_json: Path):
    bundle = load_grasp_bundle(input_json)
    source_frame_pose = ObjectWorldPose(
        position_world=bundle.source_frame_origin_obj_world,
        orientation_xyzw_world=bundle.source_frame_orientation_xyzw_obj_world,
    )
    mesh_obj_world = load_asset_mesh(bundle.target_mesh_path, scale=bundle.mesh_scale)
    mesh_local = _mesh_in_source_frame(mesh_obj_world, source_frame_pose)
    return bundle, mesh_local, source_frame_pose


def _bounds_lines(label: str, vertices: np.ndarray) -> list[str]:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return [
        f"{label}_min:       {tuple(round(float(v), 6) for v in mins)}",
        f"{label}_max:       {tuple(round(float(v), 6) for v in maxs)}",
    ]


def _mesh_metadata_lines(*, bundle, mesh_local, source_frame_pose, execution_pose_world: ObjectWorldPose | None):
    vertices_local = np.asarray(mesh_local.vertices_obj, dtype=float)
    local_centroid = mesh_area_weighted_triangle_centroid(mesh_local)
    lines = [
        f"target_mesh:      {relative_asset_mesh_path(bundle.target_mesh_path)}",
        f"mesh_scale:       {bundle.mesh_scale}",
        f"source_origin:    {tuple(round(v, 6) for v in source_frame_pose.position_world)}",
        f"source_quat_xyzw: {tuple(round(v, 6) for v in source_frame_pose.orientation_xyzw_world)}",
        f"local_centroid:   {tuple(round(float(v), 6) for v in local_centroid)}",
        *_bounds_lines("local", vertices_local),
        f"vertices:         {len(vertices_local)}",
        f"faces:            {len(mesh_local.faces)}",
    ]
    if execution_pose_world is not None:
        vertices_world = execution_pose_world.transform_points_to_world(vertices_local)
        transformed_centroid = object_point_to_world(local_centroid, execution_pose_world)
        lines.extend(
            [
                f"world_pos:       {tuple(round(v, 6) for v in execution_pose_world.position_world)}",
                f"world_quat_xyzw: {tuple(round(v, 6) for v in execution_pose_world.orientation_xyzw_world)}",
                f"world_centroid:  {tuple(round(float(v), 6) for v in transformed_centroid)}",
                *_bounds_lines("world", vertices_world),
                "floor_z_world:   0.0",
                f"floor_clearance: {round(float(vertices_world[:, 2].min()), 6)}",
            ]
        )
    return lines


def _floor_plane_vertices(vertices_world: np.ndarray, *, z_world: float = 0.0) -> list[list[float]]:
    mins = vertices_world.min(axis=0)
    maxs = vertices_world.max(axis=0)
    extents = np.maximum(maxs - mins, 1.0e-3)
    padding = max(0.25 * float(np.max(extents[:2])), 0.08)
    return [
        fmt_vec((mins[0] - padding, mins[1] - padding, z_world)),
        fmt_vec((maxs[0] + padding, mins[1] - padding, z_world)),
        fmt_vec((maxs[0] + padding, maxs[1] + padding, z_world)),
        fmt_vec((mins[0] - padding, maxs[1] + padding, z_world)),
    ]


def write_part_frame_debug_html(
    *,
    input_json: Path,
    output_html: Path,
    execution_pose_world: ObjectWorldPose | None = None,
) -> None:
    bundle, mesh_local, source_frame_pose = _mesh_local_from_bundle(input_json)
    if execution_pose_world is None:
        execution_pose_world = _pose_from_metadata(bundle.metadata)

    vertices_local = np.asarray(mesh_local.vertices_obj, dtype=float)
    faces = np.asarray(mesh_local.faces, dtype=np.int64)
    local_centroid = mesh_area_weighted_triangle_centroid(mesh_local)
    world_vertices: list[list[float]] = []
    world_centroid: list[float] | None = None
    world_origin: list[float] | None = None
    world_axis_directions: list[list[float]] | None = None
    floor_world: list[list[float]] = []
    if execution_pose_world is not None:
        vertices_world_array = execution_pose_world.transform_points_to_world(vertices_local)
        world_vertices = [fmt_vec(point) for point in vertices_world_array]
        world_centroid = fmt_vec(object_point_to_world(local_centroid, execution_pose_world))
        world_origin = fmt_vec(execution_pose_world.position_world)
        rotation = execution_pose_world.rotation_world_from_object
        world_axis_directions = [fmt_vec(rotation[:, axis]) for axis in range(3)]
        floor_world = _floor_plane_vertices(vertices_world_array)

    data = {
        "title": "Fabrica Part Frame Debug",
        "subtitle": "Target part in bundle-local coordinates and, when available, transformed into execution/world coordinates.",
        "input_json": str(input_json),
        "vertices_local": [fmt_vec(point) for point in vertices_local],
        "vertices_world": world_vertices,
        "faces": [[int(v) for v in face] for face in faces.tolist()],
        "edges": unique_edges(faces),
        "local_centroid": fmt_vec(local_centroid),
        "world_centroid": world_centroid,
        "world_origin": world_origin,
        "world_axis_directions": world_axis_directions,
        "floor_world": floor_world,
        "metadata_lines": _mesh_metadata_lines(
            bundle=bundle,
            mesh_local=mesh_local,
            source_frame_pose=source_frame_pose,
            execution_pose_world=execution_pose_world,
        ),
    }
    data_json = json.dumps(data, indent=2)
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fabrica Part Frame Debug</title>
  <style>
    :root {
      --bg: #f5f3ee;
      --panel: #fffdf8;
      --ink: #1f2522;
      --muted: #68716c;
      --local: #2f6f5e;
      --world: #b44732;
      --floor: #2563eb;
      --centroid: #7c3aed;
      --axis-x: #dc2626;
      --axis-y: #16a34a;
      --axis-z: #2563eb;
      --line: #d9d4c7;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .layout { display: grid; grid-template-columns: 340px minmax(0, 1fr); min-height: 100vh; }
    .sidebar { border-right: 1px solid var(--line); background: var(--panel); padding: 20px; overflow: auto; }
    h1 { margin: 0 0 8px; font-size: 26px; line-height: 1.15; }
    .subtitle { margin: 0 0 16px; color: var(--muted); font-size: 14px; line-height: 1.45; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
    button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 999px; padding: 9px 12px; font: inherit; cursor: pointer; }
    button:hover { border-color: var(--local); }
    .kv { white-space: pre-wrap; font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.55; margin: 0; }
    .main { padding: 18px; overflow: auto; }
    .cards { display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }
    .card { border: 1px solid var(--line); background: rgba(255,253,248,0.94); border-radius: 16px; padding: 14px; }
    .card h2 { margin: 0 0 10px; font-size: 14px; letter-spacing: 0.03em; text-transform: uppercase; }
    .scene { width: 100%; aspect-ratio: 1.6 / 1; display: block; border-radius: 12px; background: linear-gradient(180deg, #ffffff, #ebe7dc); }
    .split { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 13px; }
    .legend span { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 13px; height: 13px; border-radius: 999px; display: inline-block; }
    .caption { margin: 10px 0 0; color: var(--muted); font-size: 13px; line-height: 1.45; }
    @media (max-width: 1000px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 id="title"></h1>
      <p id="subtitle" class="subtitle"></p>
      <div class="controls">
        <button id="localBtn" type="button">Local</button>
        <button id="worldBtn" type="button">World</button>
        <button id="bothBtn" type="button">Both</button>
        <button id="solidBtn" type="button">Solid</button>
      </div>
      <pre id="details" class="kv"></pre>
    </aside>
    <main class="main">
      <div class="cards">
        <section class="card">
          <h2>Interactive View</h2>
          <svg id="scene" class="scene" viewBox="0 0 1100 700"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--local)"></i>Local mesh</span>
            <span><i class="swatch" style="background: var(--world)"></i>Transformed mesh</span>
            <span><i class="swatch" style="background: var(--floor)"></i>World floor z=0</span>
            <span><i class="swatch" style="background: var(--centroid)"></i>Centroid</span>
            <span><i class="swatch" style="background: linear-gradient(90deg, var(--axis-x), var(--axis-y), var(--axis-z))"></i>Frame axes</span>
          </div>
          <p class="caption">Left drag rotates, middle/right drag or Shift+drag pans, and scroll zooms.</p>
        </section>
        <section class="split">
          <div class="card">
            <h2>Local XY</h2>
            <svg id="localXY" class="scene" viewBox="0 0 700 440"></svg>
          </div>
          <div class="card">
            <h2>World XY</h2>
            <svg id="worldXY" class="scene" viewBox="0 0 700 440"></svg>
          </div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const title = document.getElementById("title");
    const subtitle = document.getElementById("subtitle");
    const details = document.getElementById("details");
    const scene = document.getElementById("scene");
    const localXY = document.getElementById("localXY");
    const worldXY = document.getElementById("worldXY");
    title.textContent = data.title;
    subtitle.textContent = data.subtitle;
    details.textContent = data.metadata_lines.join("\\n");

    const state = { yaw: -0.72, pitch: 0.52, zoom: 1.0, panX: 0, panY: 0, dragging: false, dragMode: "rotate", pointerId: null, lastX: 0, lastY: 0, mode: "both", solid: false };
    const axisLength = Math.max(0.025, extent(data.vertices_local) * 0.28);
    const localAxes = axes([0, 0, 0], axisLength);
    const worldAxes = data.world_axis_directions ? axesFromDirections(data.world_origin, data.world_axis_directions, axisLength) : null;

    function extent(vertices) {
      if (vertices.length === 0) return 1;
      const b = bounds(vertices);
      return Math.max(...b.max.map((v, i) => v - b.min[i]), 1e-3);
    }
    function bounds(vertices) {
      return vertices.reduce((acc, p) => {
        p.forEach((v, i) => { acc.min[i] = Math.min(acc.min[i], v); acc.max[i] = Math.max(acc.max[i], v); });
        return acc;
      }, { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
    }
    function axes(origin, length) {
      return {
        origin,
        x: [origin[0] + length, origin[1], origin[2]],
        y: [origin[0], origin[1] + length, origin[2]],
        z: [origin[0], origin[1], origin[2] + length],
      };
    }
    function axesFromDirections(origin, directions, length) {
      return {
        origin,
        x: origin.map((v, i) => v + directions[0][i] * length),
        y: origin.map((v, i) => v + directions[1][i] * length),
        z: origin.map((v, i) => v + directions[2][i] * length),
      };
    }
    function visiblePointSets() {
      const sets = [];
      if (state.mode === "local" || state.mode === "both") sets.push(...data.vertices_local, data.local_centroid, localAxes.x, localAxes.y, localAxes.z);
      if ((state.mode === "world" || state.mode === "both") && data.vertices_world.length > 0) sets.push(...data.vertices_world, data.world_centroid, worldAxes.x, worldAxes.y, worldAxes.z, ...data.floor_world);
      return sets.length ? sets : data.vertices_local;
    }
    function currentBounds() {
      const b = bounds(visiblePointSets());
      const c = b.min.map((v, i) => 0.5 * (v + b.max[i]));
      const e = Math.max(...b.max.map((v, i) => v - b.min[i]), 0.08);
      return { center: c, scale: 480 / e };
    }
    function rotate(point, center) {
      const shifted = point.map((v, i) => v - center[i]);
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];
      return [x1, cp * y1 + sp * z1, -sp * y1 + cp * z1];
    }
    function project(point, b) {
      const p = rotate(point, b.center);
      const scale = b.scale * state.zoom;
      return { x: 550 + state.panX + p[0] * scale, y: 350 + state.panY - p[1] * scale, depth: p[2] };
    }
    function add(svg, tag, attrs) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, String(v)));
      svg.appendChild(node);
      return node;
    }
    function line(svg, a, b, opts, projectionBounds) {
      const pa = project(a, projectionBounds), pb = project(b, projectionBounds);
      add(svg, "line", { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: opts.stroke, "stroke-width": opts.width || 2, "stroke-opacity": opts.opacity ?? 1 });
    }
    function point(svg, p, opts, projectionBounds) {
      const pp = project(p, projectionBounds);
      add(svg, "circle", { cx: pp.x, cy: pp.y, r: opts.radius || 6, fill: opts.fill, stroke: "#fff", "stroke-width": 1.5 });
    }
    function label(svg, p, text, fill, projectionBounds) {
      const pp = project(p, projectionBounds);
      const node = add(svg, "text", { x: pp.x + 8, y: pp.y - 8, fill, "font-size": 15, "font-family": "IBM Plex Mono, monospace", "font-weight": 700 });
      node.textContent = text;
    }
    function polygon(svg, points, fill, projectionBounds) {
      const projected = points.map((p) => project(p, projectionBounds));
      add(svg, "polygon", { points: projected.map((p) => `${p.x},${p.y}`).join(" "), fill, "fill-opacity": 0.82, stroke: "#233", "stroke-width": 0.8, "stroke-opacity": 0.35 });
    }
    function floorPolygon(svg, projectionBounds) {
      if (!data.floor_world || data.floor_world.length === 0) return;
      const projected = data.floor_world.map((p) => project(p, projectionBounds));
      add(svg, "polygon", { points: projected.map((p) => `${p.x},${p.y}`).join(" "), fill: "#2563eb", "fill-opacity": 0.14, stroke: "#2563eb", "stroke-width": 2, "stroke-opacity": 0.85 });
      data.floor_world.forEach((p, i) => line(svg, p, data.floor_world[(i + 1) % data.floor_world.length], { stroke: "#2563eb", width: 2, opacity: 0.95 }, projectionBounds));
      label(svg, data.floor_world[0], "floor z=0", "#2563eb", projectionBounds);
    }
    function drawMesh(svg, vertices, color, projectionBounds) {
      if (state.solid) {
        data.faces.map((face) => {
          const pts = face.map((i) => vertices[i]);
          const r = pts.map((p) => rotate(p, projectionBounds.center));
          const ea = r[1].map((v, i) => v - r[0][i]);
          const eb = r[2].map((v, i) => v - r[0][i]);
          const normalZ = ea[0] * eb[1] - ea[1] * eb[0];
          const depth = r.reduce((s, p) => s + p[2], 0) / r.length;
          return { pts, normalZ, depth };
        }).filter((f) => f.normalZ > 0).sort((a, b) => a.depth - b.depth).forEach((f) => polygon(svg, f.pts, color, projectionBounds));
      }
      data.edges.forEach(([a, b]) => line(svg, vertices[a], vertices[b], { stroke: color, width: 1.7, opacity: 0.78 }, projectionBounds));
    }
    function drawAxes(svg, axisSet, projectionBounds, opacity) {
      if (!axisSet) return;
      line(svg, axisSet.origin, axisSet.x, { stroke: "#dc2626", width: 3, opacity }, projectionBounds);
      line(svg, axisSet.origin, axisSet.y, { stroke: "#16a34a", width: 3, opacity }, projectionBounds);
      line(svg, axisSet.origin, axisSet.z, { stroke: "#2563eb", width: 3, opacity }, projectionBounds);
      label(svg, axisSet.x, "+X", "#dc2626", projectionBounds);
      label(svg, axisSet.y, "+Y", "#16a34a", projectionBounds);
      label(svg, axisSet.z, "+Z", "#2563eb", projectionBounds);
    }
    function renderMain() {
      scene.replaceChildren();
      const b = currentBounds();
      if (state.mode === "local" || state.mode === "both") {
        drawMesh(scene, data.vertices_local, "#2f6f5e", b);
        drawAxes(scene, localAxes, b, 1.0);
        point(scene, data.local_centroid, { fill: "#7c3aed", radius: 7 }, b);
        label(scene, data.local_centroid, "local centroid", "#7c3aed", b);
      }
      if ((state.mode === "world" || state.mode === "both") && data.vertices_world.length > 0) {
        floorPolygon(scene, b);
        drawMesh(scene, data.vertices_world, "#b44732", b);
        drawAxes(scene, worldAxes, b, 0.8);
        point(scene, data.world_centroid, { fill: "#7c3aed", radius: 7 }, b);
        label(scene, data.world_centroid, "world centroid", "#7c3aed", b);
      }
    }
    function renderOrtho(svg, vertices, centroid, color) {
      svg.replaceChildren();
      if (!vertices || vertices.length === 0 || !centroid) {
        const node = add(svg, "text", { x: 24, y: 42, fill: "#68716c", "font-size": 16, "font-family": "IBM Plex Mono, monospace" });
        node.textContent = "No execution/world pose available.";
        return;
      }
      const pts = [...vertices, centroid];
      if (vertices === data.vertices_world && data.floor_world.length > 0) pts.push(...data.floor_world);
      const min = [Math.min(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1]))];
      const max = [Math.max(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[1]))];
      const span = Math.max(max[0] - min[0], max[1] - min[1], 0.02);
      const center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
      const scale = 330 / span;
      function p2(p) { return [350 + (p[0] - center[0]) * scale, 220 - (p[1] - center[1]) * scale]; }
      data.edges.forEach(([a, b]) => {
        const pa = p2(vertices[a]), pb = p2(vertices[b]);
        add(svg, "line", { x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1], stroke: color, "stroke-width": 1.3, "stroke-opacity": 0.72 });
      });
      if (vertices === data.vertices_world && data.floor_world.length > 0) {
        const floorPts = data.floor_world.map(p2);
        add(svg, "polygon", { points: floorPts.map((p) => `${p[0]},${p[1]}`).join(" "), fill: "#2563eb", "fill-opacity": 0.12, stroke: "#2563eb", "stroke-width": 1.5, "stroke-opacity": 0.8 });
      }
      const pc = p2(centroid);
      add(svg, "circle", { cx: pc[0], cy: pc[1], r: 6, fill: "#7c3aed", stroke: "#fff", "stroke-width": 1.5 });
      add(svg, "line", { x1: pc[0], y1: pc[1], x2: pc[0] + axisLength * scale, y2: pc[1], stroke: "#dc2626", "stroke-width": 2.5 });
      add(svg, "line", { x1: pc[0], y1: pc[1], x2: pc[0], y2: pc[1] - axisLength * scale, stroke: "#16a34a", "stroke-width": 2.5 });
    }
    function render() {
      renderMain();
      renderOrtho(localXY, data.vertices_local, data.local_centroid, "#2f6f5e");
      renderOrtho(worldXY, data.vertices_world, data.world_centroid, "#b44732");
    }
    document.getElementById("localBtn").addEventListener("click", () => { state.mode = "local"; render(); });
    document.getElementById("worldBtn").addEventListener("click", () => { state.mode = "world"; render(); });
    document.getElementById("bothBtn").addEventListener("click", () => { state.mode = "both"; render(); });
    document.getElementById("solidBtn").addEventListener("click", () => { state.solid = !state.solid; render(); });
    scene.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 || event.button === 2 || event.shiftKey ? "pan" : "rotate";
      state.pointerId = event.pointerId;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      scene.setPointerCapture(event.pointerId);
      scene.style.cursor = state.dragMode === "pan" ? "move" : "grabbing";
    });
    function stopDragging() {
      state.dragging = false;
      state.pointerId = null;
      scene.style.cursor = "grab";
    }
    scene.addEventListener("pointerup", (event) => { if (state.pointerId === event.pointerId) stopDragging(); });
    scene.addEventListener("pointercancel", stopDragging);
    scene.addEventListener("pointermove", (event) => {
      if (!state.dragging || (state.pointerId !== null && state.pointerId !== event.pointerId)) return;
      const dx = event.clientX - state.lastX;
      const dy = event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      if (state.dragMode === "pan") {
        state.panX += dx;
        state.panY += dy;
      } else {
        state.yaw += dx * 0.01;
        state.pitch -= dy * 0.01;
      }
      render();
    });
    scene.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.3, Math.min(5, state.zoom * (event.deltaY < 0 ? 1.08 : 1 / 1.08)));
      render();
    }, { passive: false });
    scene.style.cursor = "grab";
    scene.addEventListener("contextmenu", (event) => event.preventDefault());
    render();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html.replace("__DATA_JSON__", data_json), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default="artifacts/pitl_pipeline_stage2_ground_feasible.json",
        help="Stage-1 or stage-2 grasp bundle JSON to inspect.",
    )
    parser.add_argument(
        "--output-html",
        default="artifacts/part_frame_debug.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--position-world",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional execution/world translation override.",
    )
    parser.add_argument(
        "--orientation-xyzw-world",
        nargs=4,
        type=float,
        default=None,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Optional execution/world quaternion override.",
    )
    args = parser.parse_args()

    if (args.position_world is None) != (args.orientation_xyzw_world is None):
        raise ValueError("--position-world and --orientation-xyzw-world must be provided together.")
    execution_pose_world = None
    if args.position_world is not None:
        execution_pose_world = ObjectWorldPose(
            position_world=tuple(args.position_world),
            orientation_xyzw_world=tuple(args.orientation_xyzw_world),
        )

    write_part_frame_debug_html(
        input_json=Path(args.input_json),
        output_html=Path(args.output_html),
        execution_pose_world=execution_pose_world,
    )
    print(f"[DEBUG] Wrote {args.output_html}")


if __name__ == "__main__":
    main()
