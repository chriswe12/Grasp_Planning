"""Self-contained interactive HTML debugger for compiled assembly sequences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grasp_planning.grasping.mesh_io import load_triangle_mesh

from .assembly_sequence import AssemblySequence

DEFAULT_MAX_EDGES_PER_PART = 400
DEFAULT_MAX_FACES_PER_PART = 2500


def _fmt_vec(values: np.ndarray) -> list[float]:
    return [round(float(value), 8) for value in values]


def _unique_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=np.int64):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.add(tuple(sorted((int(start), int(end)))))
    return sorted(edges)


def _sample_edges(edges: list[tuple[int, int]], *, limit: int) -> list[tuple[int, int]]:
    if limit <= 0 or len(edges) <= limit:
        return edges
    indices = np.linspace(0, len(edges) - 1, num=limit, dtype=np.int64)
    return [edges[int(index)] for index in indices]


def _sample_faces(faces: np.ndarray, *, limit: int) -> np.ndarray:
    if limit <= 0 or len(faces) <= limit:
        return faces
    indices = np.linspace(0, len(faces) - 1, num=limit, dtype=np.int64)
    return faces[indices]


def _mesh_centroid(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[np.asarray(faces, dtype=np.int64)]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas_twice = np.linalg.norm(raw_normals, axis=1)
    valid = areas_twice > 1.0e-12
    if not np.any(valid):
        return vertices.mean(axis=0)
    return np.average(triangles[valid].mean(axis=1), axis=0, weights=areas_twice[valid])


def assembly_sequence_visual_payload(
    sequence: AssemblySequence,
    *,
    max_edges_per_part: int = DEFAULT_MAX_EDGES_PER_PART,
    max_faces_per_part: int = DEFAULT_MAX_FACES_PER_PART,
) -> dict[str, object]:
    payload = sequence.to_payload()
    visual_parts: dict[str, object] = {}
    all_points: list[np.ndarray] = []
    for part in sequence.parts:
        mesh = load_triangle_mesh(part.resolved_mesh_path, scale=sequence.mesh_scale)
        vertices = np.asarray(mesh.vertices_obj, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        original_edges = _unique_edges(faces)
        edges = _sample_edges(original_edges, limit=max(0, int(max_edges_per_part)))
        visual_faces = _sample_faces(faces, limit=max(0, int(max_faces_per_part)))
        centroid = _mesh_centroid(vertices, faces)
        visual_parts[part.part_id] = {
            "vertices_assembly_m": [_fmt_vec(vertex) for vertex in vertices],
            "edges": [[start, end] for start, end in edges],
            "faces": [[int(index) for index in face] for face in visual_faces],
            "centroid_assembly_m": _fmt_vec(centroid),
            "original_edge_count": len(original_edges),
            "visual_edge_count": len(edges),
            "original_face_count": len(faces),
            "visual_face_count": len(visual_faces),
        }
        all_points.append(vertices)
        for step in sequence.steps:
            if step.incoming_part_id == part.part_id:
                all_points.append(vertices + np.asarray(step.final_to_pre_insertion_translation_m, dtype=float))
                break

    combined = np.vstack(all_points)
    scene_min = combined.min(axis=0)
    scene_max = combined.max(axis=0)
    extents = np.maximum(scene_max - scene_min, 1.0e-3)
    padding_xy = max(0.15 * float(max(extents[0], extents[1])), 0.03)
    table = [
        [float(scene_min[0] - padding_xy), float(scene_min[1] - padding_xy), sequence.table_z_assembly_m],
        [float(scene_max[0] + padding_xy), float(scene_min[1] - padding_xy), sequence.table_z_assembly_m],
        [float(scene_max[0] + padding_xy), float(scene_max[1] + padding_xy), sequence.table_z_assembly_m],
        [float(scene_min[0] - padding_xy), float(scene_max[1] + padding_xy), sequence.table_z_assembly_m],
    ]
    payload["visualization"] = {
        "parts": visual_parts,
        "table_vertices_assembly_m": table,
        "scene_bounds_assembly_m": {
            "min": _fmt_vec(scene_min),
            "max": _fmt_vec(scene_max),
            "center": _fmt_vec(0.5 * (scene_min + scene_max)),
            "extent": round(float(np.max(extents)), 8),
        },
        "max_edges_per_part": int(max_edges_per_part),
        "max_faces_per_part": int(max_faces_per_part),
    }
    return payload


def write_assembly_sequence_html(
    sequence: AssemblySequence,
    output_path: str | Path,
    *,
    max_edges_per_part: int = DEFAULT_MAX_EDGES_PER_PART,
    max_faces_per_part: int = DEFAULT_MAX_FACES_PER_PART,
) -> None:
    data = assembly_sequence_visual_payload(
        sequence,
        max_edges_per_part=max_edges_per_part,
        max_faces_per_part=max_faces_per_part,
    )
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fabrica Assembly Sequence</title>
  <style>
    :root {
      --bg: #f3f1eb;
      --panel: #fffdf8;
      --ink: #202722;
      --muted: #66706a;
      --line: #d7d2c7;
      --assembled: #65756d;
      --base: #18805f;
      --incoming: #dc7b20;
      --future: #b8b8b2;
      --table: #3478c7;
      --warning: #a33b2e;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }
    .layout { min-height: 100vh; display: grid; grid-template-columns: 370px minmax(0, 1fr); }
    aside { padding: 20px; background: var(--panel); border-right: 1px solid var(--line); overflow: auto; }
    main { padding: 18px; min-width: 0; }
    h1 { margin: 0 0 8px; font-size: 25px; line-height: 1.15; }
    h2 { margin: 20px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; }
    p { line-height: 1.45; }
    .subtitle { color: var(--muted); margin: 0 0 16px; font-size: 14px; }
    .control-row { display: flex; align-items: center; gap: 8px; margin: 9px 0; }
    .control-row label { min-width: 70px; color: var(--muted); font-size: 13px; }
    input[type="range"] { flex: 1; min-width: 0; }
    button {
      border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 999px;
      padding: 8px 11px; font: inherit; cursor: pointer;
    }
    button:hover { border-color: var(--base); }
    .buttons { display: flex; flex-wrap: wrap; gap: 7px; margin: 10px 0; }
    .check { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }
    .warnings { margin: 8px 0 0; padding-left: 19px; color: var(--warning); font-size: 13px; line-height: 1.45; }
    pre { white-space: pre-wrap; font: 12px/1.5 "IBM Plex Mono", "SFMono-Regular", monospace; margin: 0; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 12px; }
    canvas {
      width: 100%; height: calc(100vh - 72px); min-height: 540px; display: block;
      background: linear-gradient(180deg, #ffffff 0%, #ece9e1 100%); border-radius: 11px;
      cursor: grab; touch-action: none;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; padding: 10px 4px 0; color: var(--muted); font-size: 13px; }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
    @media (max-width: 950px) {
      .layout { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      canvas { height: 65vh; min-height: 440px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Assembly Sequence</h1>
      <p id="subtitle" class="subtitle"></p>

      <h2>Sequence controls</h2>
      <div class="control-row">
        <label for="stepSlider">Step</label>
        <input id="stepSlider" type="range" min="0" step="1">
        <output id="stepValue"></output>
      </div>
      <div class="control-row">
        <label for="progressSlider">Insertion</label>
        <input id="progressSlider" type="range" min="0" max="1" value="0" step="0.001">
        <output id="progressValue">0%</output>
      </div>
      <div class="buttons">
        <button id="prevStep" type="button">Previous</button>
        <button id="nextStep" type="button">Next</button>
        <button id="playButton" type="button">Play insertion</button>
        <button id="resetView" type="button">Reset view</button>
      </div>
      <label class="check"><input id="showFuture" type="checkbox"> Show future parts at final poses</label>

      <h2>Current state</h2>
      <pre id="details"></pre>

      <h2>Sequence warnings</h2>
      <ul id="warnings" class="warnings"></ul>
    </aside>
    <main>
      <section class="card">
        <canvas id="scene"></canvas>
        <div class="legend">
          <span><i class="dot" style="background:var(--base)"></i>Base part</span>
          <span><i class="dot" style="background:var(--assembled)"></i>Assembled</span>
          <span><i class="dot" style="background:var(--incoming)"></i>Incoming</span>
          <span><i class="dot" style="background:var(--future)"></i>Future</span>
          <span><i class="dot" style="background:var(--table)"></i>Table z=0</span>
        </div>
      </section>
    </main>
  </div>
  <script>
    const data = __DATA_JSON__;
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const stepSlider = document.getElementById("stepSlider");
    const progressSlider = document.getElementById("progressSlider");
    const stepValue = document.getElementById("stepValue");
    const progressValue = document.getElementById("progressValue");
    const details = document.getElementById("details");
    const warnings = document.getElementById("warnings");
    const showFuture = document.getElementById("showFuture");
    const playButton = document.getElementById("playButton");
    const bounds = data.visualization.scene_bounds_assembly_m;
    const colors = {
      assembled: getComputedStyle(document.documentElement).getPropertyValue("--assembled").trim(),
      base: getComputedStyle(document.documentElement).getPropertyValue("--base").trim(),
      incoming: getComputedStyle(document.documentElement).getPropertyValue("--incoming").trim(),
      future: getComputedStyle(document.documentElement).getPropertyValue("--future").trim(),
      table: getComputedStyle(document.documentElement).getPropertyValue("--table").trim(),
    };
    const state = {
      step: 0, progress: 0, yaw: -0.72, pitch: 0.52, zoom: 1,
      panX: 0, panY: 0, drag: null, playing: false, lastTime: null,
    };

    document.getElementById("subtitle").textContent =
      `${data.assembly}: ${data.selected_order.join(" → ")} · base ${data.base_part_id} (${data.base_part_source})`;
    for (const message of data.warnings) {
      const item = document.createElement("li");
      item.textContent = message;
      warnings.appendChild(item);
    }
    if (!data.warnings.length) {
      const item = document.createElement("li");
      item.textContent = "No sequence warnings.";
      warnings.appendChild(item);
    }
    stepSlider.max = Math.max(0, data.steps.length - 1);

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.round(rect.width * ratio));
      const height = Math.max(1, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function viewPoint(point) {
      const center = bounds.center;
      const x = point[0] - center[0];
      const y = point[1] - center[1];
      const z = point[2] - center[2];
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const xYaw = cy * x - sy * y;
      const yYaw = sy * x + cy * y;
      const yPitch = cp * yYaw - sp * z;
      const zPitch = sp * yYaw + cp * z;
      const scale = 0.72 * Math.min(canvas.width, canvas.height) / Math.max(bounds.extent, 1e-6) * state.zoom;
      return {
        x: 0.5 * canvas.width + state.panX + xYaw * scale,
        y: 0.52 * canvas.height + state.panY - zPitch * scale,
        depth: yPitch,
      };
    }

    function translated(point, offset) {
      return [point[0] + offset[0], point[1] + offset[1], point[2] + offset[2]];
    }

    function currentIncomingOffset(step) {
      const remaining = 1 - state.progress;
      return step.final_to_pre_insertion_translation_m.map((value) => value * remaining);
    }

    function line3(a, b, color, width, alpha = 1, dash = []) {
      const pa = viewPoint(a), pb = viewPoint(b);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width * Math.max(1, window.devicePixelRatio || 1);
      ctx.setLineDash(dash.map((value) => value * Math.max(1, window.devicePixelRatio || 1)));
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.restore();
    }

    function polygon3(points, fill, stroke, alpha) {
      const projected = points.map(viewPoint);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.5 * Math.max(1, window.devicePixelRatio || 1);
      ctx.beginPath();
      ctx.moveTo(projected[0].x, projected[0].y);
      for (const point of projected.slice(1)) ctx.lineTo(point.x, point.y);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function drawArrow(start, end, color) {
      line3(start, end, color, 3, 0.95);
      const a = viewPoint(start), b = viewPoint(end);
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      const size = 12 * Math.max(1, window.devicePixelRatio || 1);
      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(b.x - size * Math.cos(angle - 0.45), b.y - size * Math.sin(angle - 0.45));
      ctx.lineTo(b.x - size * Math.cos(angle + 0.45), b.y - size * Math.sin(angle + 0.45));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawPart(partId, color, alpha, offset = [0, 0, 0], width = 1.2) {
      const visual = data.visualization.parts[partId];
      const vertices = visual.vertices_assembly_m;
      const faceRecords = visual.faces.map((face) => {
        const points = face.map((index) => translated(vertices[index], offset));
        const projected = points.map(viewPoint);
        return {
          projected,
          depth: projected.reduce((sum, point) => sum + point.depth, 0) / projected.length,
        };
      }).sort((left, right) => left.depth - right.depth);
      ctx.save();
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.35 * Math.max(1, window.devicePixelRatio || 1);
      ctx.globalAlpha = 0.42 * alpha;
      for (const face of faceRecords) {
        ctx.beginPath();
        ctx.moveTo(face.projected[0].x, face.projected[0].y);
        ctx.lineTo(face.projected[1].x, face.projected[1].y);
        ctx.lineTo(face.projected[2].x, face.projected[2].y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
      for (const [start, end] of visual.edges) {
        line3(translated(vertices[start], offset), translated(vertices[end], offset), color, width, 0.38 * alpha);
      }
    }

    function drawAxes() {
      const axisLength = Math.max(0.025, bounds.extent * 0.22);
      line3([0, 0, 0], [axisLength, 0, 0], "#dc2626", 2.2, 0.9);
      line3([0, 0, 0], [0, axisLength, 0], "#16a34a", 2.2, 0.9);
      line3([0, 0, 0], [0, 0, axisLength], "#2563eb", 2.2, 0.9);
    }

    function renderDetails(step) {
      const incomingPart = data.parts[step.incoming_part_id];
      details.textContent = [
        `step:                 ${step.step_index + 1}/${data.steps.length} (${step.step_id})`,
        `incoming_part:        ${step.incoming_part_id} (${step.incoming_part_role || "role unknown"})`,
        `assembled_before:     ${JSON.stringify(step.assembled_part_ids_before)}`,
        `assembled_after:      ${JSON.stringify(step.assembled_part_ids_after)}`,
        `base_part:            ${data.base_part_id}`,
        `base_part_source:     ${data.base_part_source}`,
        `base_status:          ${step.base_part_status}`,
        `holder_base_available:${step.holder_base_available ? " yes" : " no"}`,
        `insertion_progress:   ${(state.progress * 100).toFixed(1)}%`,
        `final_to_pre_xyz_m:   ${JSON.stringify(step.final_to_pre_insertion_translation_m)}`,
        `pre_to_final_xyz_m:   ${JSON.stringify(step.pre_to_final_insertion_vector_m)}`,
        `insertion_distance_m: ${step.insertion_distance_m.toFixed(6)}`,
        `incoming_min_z_m:     ${incomingPart.bounds_assembly_m.min[2].toFixed(6)}`,
        `incoming_table_touch: ${incomingPart.touches_table}`,
        `table_contact_parts:  ${JSON.stringify(data.table.contact_part_ids)}`,
      ].join("\\n");
    }

    function render() {
      resizeCanvas();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const step = data.steps[state.step];
      polygon3(data.visualization.table_vertices_assembly_m, colors.table, colors.table, 0.12);
      drawAxes();

      if (showFuture.checked) {
        const visible = new Set([...step.assembled_part_ids_before, step.incoming_part_id]);
        for (const partId of data.selected_order) {
          if (!visible.has(partId)) drawPart(partId, colors.future, 0.25, [0, 0, 0], 0.8);
        }
      }
      for (const partId of step.assembled_part_ids_before) {
        drawPart(partId, partId === data.base_part_id ? colors.base : colors.assembled, 0.92);
      }
      const incomingOffset = currentIncomingOffset(step);
      drawPart(step.incoming_part_id, colors.incoming, 1, incomingOffset, 1.7);

      const centroidFinal = data.visualization.parts[step.incoming_part_id].centroid_assembly_m;
      const centroidPre = translated(centroidFinal, step.final_to_pre_insertion_translation_m);
      drawArrow(centroidPre, centroidFinal, colors.incoming);

      stepValue.textContent = `${state.step + 1}/${data.steps.length}`;
      progressValue.textContent = `${Math.round(state.progress * 100)}%`;
      renderDetails(step);
    }

    function setStep(index) {
      state.step = Math.max(0, Math.min(data.steps.length - 1, index));
      state.progress = 0;
      stepSlider.value = String(state.step);
      progressSlider.value = "0";
      state.playing = false;
      playButton.textContent = "Play insertion";
      render();
    }

    stepSlider.addEventListener("input", () => setStep(Number(stepSlider.value)));
    progressSlider.addEventListener("input", () => {
      state.progress = Number(progressSlider.value);
      state.playing = false;
      playButton.textContent = "Play insertion";
      render();
    });
    showFuture.addEventListener("change", render);
    document.getElementById("prevStep").addEventListener("click", () => setStep(state.step - 1));
    document.getElementById("nextStep").addEventListener("click", () => setStep(state.step + 1));
    document.getElementById("resetView").addEventListener("click", () => {
      Object.assign(state, { yaw: -0.72, pitch: 0.52, zoom: 1, panX: 0, panY: 0 });
      render();
    });
    playButton.addEventListener("click", () => {
      state.playing = !state.playing;
      state.lastTime = null;
      if (state.playing && state.progress >= 1) state.progress = 0;
      playButton.textContent = state.playing ? "Pause" : "Play insertion";
      requestAnimationFrame(tick);
    });

    function tick(timestamp) {
      if (!state.playing) return;
      if (state.lastTime !== null) state.progress += (timestamp - state.lastTime) / 2200;
      state.lastTime = timestamp;
      if (state.progress >= 1) {
        state.progress = 1;
        state.playing = false;
        playButton.textContent = "Play insertion";
      }
      progressSlider.value = String(state.progress);
      render();
      if (state.playing) requestAnimationFrame(tick);
    }

    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      state.drag = {
        id: event.pointerId, x: event.clientX, y: event.clientY,
        mode: event.shiftKey || event.button !== 0 ? "pan" : "rotate",
      };
      canvas.style.cursor = "grabbing";
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!state.drag || state.drag.id !== event.pointerId) return;
      const dx = event.clientX - state.drag.x;
      const dy = event.clientY - state.drag.y;
      state.drag.x = event.clientX;
      state.drag.y = event.clientY;
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      if (state.drag.mode === "pan") {
        state.panX += dx * ratio;
        state.panY += dy * ratio;
      } else {
        state.yaw += dx * 0.01;
        state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch - dy * 0.01));
      }
      render();
    });
    function endDrag(event) {
      if (state.drag && state.drag.id === event.pointerId) state.drag = null;
      canvas.style.cursor = "grab";
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.25, Math.min(8, state.zoom * Math.exp(-event.deltaY * 0.001)));
      render();
    }, { passive: false });
    window.addEventListener("resize", render);

    setStep(0);
  </script>
</body>
</html>
""".replace("__DATA_JSON__", data_json)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


__all__ = [
    "DEFAULT_MAX_EDGES_PER_PART",
    "DEFAULT_MAX_FACES_PER_PART",
    "assembly_sequence_visual_payload",
    "write_assembly_sequence_html",
]
