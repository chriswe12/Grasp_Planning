"""Generate a browser-based HTML debug view for cube grasp candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping import CubeFaceGraspGenerator, GraspCandidate
from grasp_planning.grasping.cube_grasp_generator import _quat_to_rotmat_xyzw


DEFAULT_CUBE_SIZE = (0.05, 0.05, 0.05)
DEFAULT_CUBE_POSITION = (0.45, 0.0, 0.025)
DEFAULT_CUBE_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
DEFAULT_ROBOT_BASE_POSITION = (0.0, 0.0, 0.0)
DEFAULT_PREGRASP_OFFSET = 0.10
DEFAULT_FINGER_CLEARANCE = 0.01
DEFAULT_OUTPUT_HTML = REPO_ROOT / "artifacts" / "cube_grasp_debug.html"


def _parse_vec3(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 comma-separated values, got '{raw}'.")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse '{raw}' as 3 floats.") from exc


def _parse_quat(raw: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Expected 4 comma-separated values, got '{raw}'.")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse '{raw}' as 4 floats.") from exc


parser = argparse.ArgumentParser(description="Generate an HTML debug view for ranked cube grasps.")
parser.add_argument("--cube-size", type=_parse_vec3, default=DEFAULT_CUBE_SIZE, help="Cube size as x,y,z in meters.")
parser.add_argument(
    "--cube-position",
    type=_parse_vec3,
    default=DEFAULT_CUBE_POSITION,
    help="Cube world position as x,y,z in meters.",
)
parser.add_argument(
    "--cube-orientation-xyzw",
    type=_parse_quat,
    default=DEFAULT_CUBE_ORIENTATION_XYZW,
    help="Cube world orientation quaternion as x,y,z,w.",
)
parser.add_argument(
    "--robot-base-position",
    type=_parse_vec3,
    default=DEFAULT_ROBOT_BASE_POSITION,
    help="Robot base position as x,y,z in meters.",
)
parser.add_argument(
    "--pregrasp-offset",
    type=float,
    default=DEFAULT_PREGRASP_OFFSET,
    help="Pregrasp offset applied opposite the approach direction.",
)
parser.add_argument(
    "--finger-clearance",
    type=float,
    default=DEFAULT_FINGER_CLEARANCE,
    help="Finger clearance added to the grasp closing span.",
)
parser.add_argument(
    "--output-html",
    type=Path,
    default=DEFAULT_OUTPUT_HTML,
    help=f"Output HTML path. Default: {DEFAULT_OUTPUT_HTML}",
)


@dataclass(frozen=True)
class _ViewerState:
    generator: CubeFaceGraspGenerator
    candidates: list[GraspCandidate]
    cube_size: tuple[float, float, float]
    cube_position_w: tuple[float, float, float]
    cube_orientation_xyzw: tuple[float, float, float, float]
    robot_base_position_w: tuple[float, float, float]


def _cube_corners_w(state: _ViewerState) -> np.ndarray:
    half_extents = 0.5 * np.asarray(state.cube_size, dtype=float)
    x0, y0, z0 = -half_extents
    x1, y1, z1 = half_extents
    corners_obj = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=float,
    )
    rotmat = _quat_to_rotmat_xyzw(state.cube_orientation_xyzw)
    cube_position = np.asarray(state.cube_position_w, dtype=float)
    return corners_obj @ rotmat.T + cube_position


def _fmt_vec(vec: tuple[float, ...]) -> list[float]:
    return [round(float(value), 6) for value in vec]


def _score_parts(state: _ViewerState, grasp: GraspCandidate) -> dict[str, float]:
    return state.generator.score_components(
        point_w=np.asarray(grasp.position_w, dtype=float),
        normal_w=np.asarray(grasp.normal_w, dtype=float),
        robot_base_position_w=np.asarray(state.robot_base_position_w, dtype=float),
        label=grasp.label,
    )


def _build_payload(state: _ViewerState) -> dict[str, object]:
    edge_ids = [
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7],
        [6, 7],
    ]
    candidates = []
    for rank, grasp in enumerate(state.candidates, start=1):
        left_finger, right_finger = grasp.finger_positions_w()
        approach_axis, closing_axis, z_axis = grasp.gripper_axes_w()
        candidates.append(
            {
                "rank": rank,
                "label": grasp.label,
                "score": round(float(grasp.score), 6),
                "score_parts": {key: round(float(value), 6) for key, value in _score_parts(state, grasp).items()},
                "position_w": _fmt_vec(grasp.position_w),
                "pregrasp_position_w": _fmt_vec(grasp.pregrasp_position_w),
                "normal_w": _fmt_vec(grasp.normal_w),
                "left_finger_w": _fmt_vec(left_finger),
                "right_finger_w": _fmt_vec(right_finger),
                "approach_axis_w": _fmt_vec(tuple(float(v) for v in approach_axis)),
                "closing_axis_w": _fmt_vec(tuple(float(v) for v in closing_axis)),
                "gripper_z_axis_w": _fmt_vec(tuple(float(v) for v in z_axis)),
                "gripper_width": round(float(grasp.gripper_width), 6),
            }
        )

    return {
        "cube_size": _fmt_vec(state.cube_size),
        "cube_position_w": _fmt_vec(state.cube_position_w),
        "cube_orientation_xyzw": _fmt_vec(state.cube_orientation_xyzw),
        "robot_base_position_w": _fmt_vec(state.robot_base_position_w),
        "cube_corners_w": [[round(float(v), 6) for v in corner] for corner in _cube_corners_w(state)],
        "cube_edges": edge_ids,
        "candidates": candidates,
    }


def _html_document(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, indent=2)
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cube Grasp Debug Viewer</title>
  <style>
    :root {{
      --bg: #f3efe4;
      --panel: #fffaf0;
      --ink: #1e1d1a;
      --accent: #b43f2c;
      --accent-soft: #e8b59f;
      --muted: #6f6a5f;
      --line: #d9ceb8;
      --cube: #d97a31;
      --pre: #2b8a57;
      --finger: #6d3cc6;
      --robot: #2563a6;
      --axis: #1397a6;
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff8e8 0, transparent 30%),
        linear-gradient(135deg, #f7f2e7 0%, #efe7d4 100%);
    }}

    .layout {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: 100vh;
    }}

    .sidebar {{
      border-right: 1px solid var(--line);
      background: rgba(255, 250, 240, 0.92);
      padding: 20px 18px;
      overflow: auto;
    }}

    .title {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.1;
    }}

    .subtitle {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}

    .controls {{
      display: flex;
      gap: 10px;
      margin-bottom: 14px;
    }}

    button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }}

    button:hover {{
      border-color: var(--accent);
    }}

    .list {{
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
    }}

    .item {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.7);
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }}

    .item:hover {{
      transform: translateY(-1px);
      border-color: var(--accent-soft);
      box-shadow: 0 8px 18px rgba(85, 65, 42, 0.08);
    }}

    .item.active {{
      border-color: var(--accent);
      box-shadow: 0 10px 24px rgba(180, 63, 44, 0.18);
      background: #fff;
    }}

    .item-rank {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}

    .item-main {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-top: 6px;
      gap: 10px;
    }}

    .item-label {{
      font-size: 24px;
      font-weight: 700;
    }}

    .item-score {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 14px;
    }}

    .item-meta {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      font-family: "IBM Plex Mono", monospace;
    }}

    .main {{
      padding: 18px;
      overflow: auto;
    }}

    .cards {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
      gap: 18px;
      align-items: start;
    }}

    .card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255, 250, 240, 0.88);
      padding: 16px;
      box-shadow: 0 14px 32px rgba(72, 51, 28, 0.08);
    }}

    .card h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}

    #scene {{
      width: 100%;
      height: auto;
      aspect-ratio: 1.25 / 1;
      display: block;
      background:
        radial-gradient(circle at 20% 18%, rgba(255,255,255,0.9), rgba(255,255,255,0.55) 35%, rgba(233,226,208,0.65)),
        linear-gradient(180deg, rgba(255,255,255,0.2), rgba(223,214,194,0.18));
      border-radius: 16px;
    }}

    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }}

    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}

    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      display: inline-block;
    }}

    .details {{
      display: grid;
      gap: 14px;
    }}

    .kv {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 13px;
      line-height: 1.55;
      white-space: pre-wrap;
      margin: 0;
    }}

    .score-grid {{
      display: grid;
      gap: 8px;
    }}

    .score-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      border-bottom: 1px dashed var(--line);
      padding-bottom: 6px;
      font-family: "IBM Plex Mono", monospace;
      font-size: 13px;
    }}

    .caption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}

    @media (max-width: 1100px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}

      .cards {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1 class="title">Cube Grasp Debug</h1>
      <p class="subtitle">
        Ranked deterministic cube grasps. Select a candidate to inspect the pregrasp point,
        approach direction, derived finger locations, and score breakdown.
      </p>
      <div class="controls">
        <button id="prevBtn" type="button">Prev</button>
        <button id="nextBtn" type="button">Next</button>
      </div>
      <div id="graspList" class="list"></div>
    </aside>

    <main class="main">
      <div class="cards">
        <section class="card">
          <h2>Scene</h2>
          <svg id="scene" viewBox="0 0 960 760" aria-label="Cube grasp debug scene"></svg>
          <div class="legend">
            <span><i class="swatch" style="background: var(--cube)"></i>Cube center</span>
            <span><i class="swatch" style="background: var(--robot)"></i>Robot base</span>
            <span><i class="swatch" style="background: var(--accent)"></i>Selected grasp</span>
            <span><i class="swatch" style="background: var(--pre)"></i>Pregrasp / approach path</span>
            <span><i class="swatch" style="background: var(--finger)"></i>Finger locations / closing axis</span>
            <span><i class="swatch" style="background: var(--axis)"></i>Gripper z-axis</span>
          </div>
          <p class="caption">
            The two finger points are derived from the grasp pose and gripper width, not stored directly by the generator.
          </p>
        </section>

        <section class="card">
          <h2>Selection</h2>
          <div class="details">
            <div>
              <h2 style="margin-bottom:10px;">Summary</h2>
              <pre id="summary" class="kv"></pre>
            </div>
            <div>
              <h2 style="margin-bottom:10px;">Score Components</h2>
              <div id="scoreGrid" class="score-grid"></div>
            </div>
            <div>
              <h2 style="margin-bottom:10px;">Geometry</h2>
              <pre id="geometry" class="kv"></pre>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script>
    const data = __DATA_JSON__;
    const scene = document.getElementById("scene");
    const graspList = document.getElementById("graspList");
    const summary = document.getElementById("summary");
    const geometry = document.getElementById("geometry");
    const scoreGrid = document.getElementById("scoreGrid");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");

    const state = {{
      selectedIndex: 0,
      yaw: -0.75,
      pitch: 0.62,
      zoom: 1.0,
      panX: 0,
      panY: 0,
      dragging: false,
      dragMode: "rotate",
      lastPointerX: 0,
      lastPointerY: 0,
    }};

    const worldPoints = [
      ...data.cube_corners_w,
      data.cube_position_w,
      data.robot_base_position_w,
      ...data.candidates.flatMap((candidate) => [
        candidate.position_w,
        candidate.pregrasp_position_w,
        candidate.left_finger_w,
        candidate.right_finger_w,
      ]),
    ];

    const bounds = worldPoints.reduce((acc, point) => {{
      point.forEach((value, axis) => {{
        acc.min[axis] = Math.min(acc.min[axis], value);
        acc.max[axis] = Math.max(acc.max[axis], value);
      }});
      return acc;
    }}, {{ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }});

    const center = bounds.min.map((value, axis) => 0.5 * (value + bounds.max[axis]));
    const extent = Math.max(...bounds.max.map((value, axis) => value - bounds.min[axis]), 0.25);
    const baseScale = 520 / extent;

    function rotate(point) {{
      const shifted = point.map((value, axis) => value - center[axis]);
      const cy = Math.cos(state.yaw);
      const sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch);
      const sp = Math.sin(state.pitch);

      const x1 = cy * shifted[0] + sy * shifted[1];
      const y1 = -sy * shifted[0] + cy * shifted[1];
      const z1 = shifted[2];

      const x2 = x1;
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      return [x2, y2, z2];
    }}

    function project(point) {{
      const [x, y, z] = rotate(point);
      const scale = baseScale * state.zoom;
      return {{
        x: 480 + state.panX + x * scale,
        y: 380 + state.panY - y * scale,
        depth: z,
      }};
    }}

    function fmtVec(vec) {{
      return `(${vec.map((value) => value >= 0 ? `+${value.toFixed(4)}` : value.toFixed(4)).join(", ")})`;
    }}

    function addSvg(tag, attrs) {{
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      scene.appendChild(node);
      return node;
    }}

    function drawLine(a, b, options = {{}}) {{
      const pa = project(a);
      const pb = project(b);
      addSvg("line", {{
        x1: pa.x,
        y1: pa.y,
        x2: pb.x,
        y2: pb.y,
        stroke: options.stroke || "#555",
        "stroke-width": options.strokeWidth || 2,
        "stroke-opacity": options.opacity ?? 1,
        "stroke-dasharray": options.dash || "",
        "marker-end": options.markerEnd || "",
      }});
    }}

    function drawPoint(point, options = {{}}) {{
      const p = project(point);
      addSvg("circle", {{
        cx: p.x,
        cy: p.y,
        r: options.radius || 6,
        fill: options.fill || "#000",
        "fill-opacity": options.opacity ?? 1,
        stroke: options.stroke || "white",
        "stroke-width": options.strokeWidth || 2,
      }});
    }}

    function drawLabel(point, text, fill, dx = 10, dy = -10) {{
      const p = project(point);
      const node = addSvg("text", {{
        x: p.x + dx,
        y: p.y + dy,
        fill,
        "font-size": 16,
        "font-family": "IBM Plex Mono, monospace",
        "font-weight": 600,
      }});
      node.textContent = text;
    }}

    function drawArrow(origin, vector, length, color, width) {{
      const target = origin.map((value, axis) => value + vector[axis] * length);
      drawLine(origin, target, {{ stroke: color, strokeWidth: width, markerEnd: "url(#arrow)" }});
    }}

    function renderList() {{
      graspList.replaceChildren();
      data.candidates.forEach((candidate, index) => {{
        const item = document.createElement("button");
        item.type = "button";
        item.className = `item${index === state.selectedIndex ? " active" : ""}`;
        item.innerHTML = `
          <div class="item-rank">Rank ${candidate.rank}</div>
          <div class="item-main">
            <div class="item-label">${{candidate.label}}</div>
            <div class="item-score">${{candidate.score.toFixed(4)}}</div>
          </div>
          <div class="item-meta">center=${{fmtVec(candidate.position_w)}}<br>normal=${{fmtVec(candidate.normal_w)}}</div>
        `;
        item.addEventListener("click", () => {{
          state.selectedIndex = index;
          render();
        }});
        graspList.appendChild(item);
      }});
    }}

    function renderDetails(candidate) {{
      summary.textContent =
        `rank:           ${{candidate.rank}} / ${{data.candidates.length}}\\n` +
        `label:          ${{candidate.label}}\\n` +
        `score:          ${{candidate.score.toFixed(6)}}\\n` +
        `gripper_width:  ${{candidate.gripper_width.toFixed(6)}} m`;

      geometry.textContent =
        `cube_position_w:    ${{fmtVec(data.cube_position_w)}}\\n` +
        `grasp_position_w:   ${{fmtVec(candidate.position_w)}}\\n` +
        `pregrasp_position:  ${{fmtVec(candidate.pregrasp_position_w)}}\\n` +
        `normal_w:           ${{fmtVec(candidate.normal_w)}}\\n` +
        `approach_axis_w:    ${{fmtVec(candidate.approach_axis_w)}}\\n` +
        `closing_axis_w:     ${{fmtVec(candidate.closing_axis_w)}}\\n` +
        `gripper_z_axis_w:   ${{fmtVec(candidate.gripper_z_axis_w)}}\\n` +
        `left_finger_w:      ${{fmtVec(candidate.left_finger_w)}}\\n` +
        `right_finger_w:     ${{fmtVec(candidate.right_finger_w)}}`;

      scoreGrid.replaceChildren();
      const entries = [
        ["distance_score", candidate.score_parts.distance_score],
        ["side_grasp_bonus", candidate.score_parts.side_grasp_bonus],
        ["top_grasp_penalty", candidate.score_parts.top_grasp_penalty],
        ["underside_penalty", candidate.score_parts.underside_penalty],
        ["horizontal_bonus", candidate.score_parts.horizontal_bonus],
        ["total", candidate.score_parts.total],
      ];
      entries.forEach(([label, value]) => {{
        const row = document.createElement("div");
        row.className = "score-row";
        row.innerHTML = `<span>${{label}}</span><strong>${{Number(value).toFixed(6)}}</strong>`;
        scoreGrid.appendChild(row);
      }});
    }}

    function renderScene(candidate) {{
      scene.replaceChildren();

      const defs = addSvg("defs", {{}});
      const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
      marker.setAttribute("id", "arrow");
      marker.setAttribute("markerWidth", "8");
      marker.setAttribute("markerHeight", "8");
      marker.setAttribute("refX", "7");
      marker.setAttribute("refY", "4");
      marker.setAttribute("orient", "auto");
      marker.innerHTML = '<path d="M0,0 L8,4 L0,8 z" fill="currentColor"></path>';
      defs.appendChild(marker);

      addSvg("rect", {{
        x: 0,
        y: 0,
        width: 960,
        height: 760,
        rx: 16,
        fill: "transparent",
      }});

      const segments = [];
      data.cube_edges.forEach(([start, end]) => {{
        const a = data.cube_corners_w[start];
        const b = data.cube_corners_w[end];
        const depth = 0.5 * (project(a).depth + project(b).depth);
        segments.push({{ kind: "cube", a, b, depth }});
      }});

      data.candidates.forEach((entry, index) => {{
        segments.push({{
          kind: "ghost",
          a: entry.pregrasp_position_w,
          b: entry.position_w,
          depth: 0.5 * (project(entry.pregrasp_position_w).depth + project(entry.position_w).depth),
          selected: index === state.selectedIndex,
        }});
      }});

      segments.sort((lhs, rhs) => lhs.depth - rhs.depth);
      segments.forEach((segment) => {{
        if (segment.kind === "cube") {{
          drawLine(segment.a, segment.b, {{ stroke: "#2f2a24", strokeWidth: 2, opacity: 0.9 }});
        }} else {{
          drawLine(segment.a, segment.b, {{
            stroke: segment.selected ? "#b43f2c" : "#b8b1a4",
            strokeWidth: segment.selected ? 3 : 1.4,
            opacity: segment.selected ? 1 : 0.55,
          }});
        }}
      }});

      drawPoint(data.cube_position_w, {{ fill: "#d97a31", radius: 7 }});
      drawLabel(data.cube_position_w, "cube", "#d97a31");
      drawPoint(data.robot_base_position_w, {{ fill: "#2563a6", radius: 7 }});
      drawLabel(data.robot_base_position_w, "robot", "#2563a6");

      data.candidates.forEach((entry, index) => {{
        drawPoint(entry.position_w, {{
          fill: index === state.selectedIndex ? "#b43f2c" : "#c6c0b4",
          radius: index === state.selectedIndex ? 7 : 4,
          opacity: index === state.selectedIndex ? 1 : 0.7,
          stroke: index === state.selectedIndex ? "white" : "#f5f1e6",
        }});
      }});

      drawPoint(candidate.pregrasp_position_w, {{ fill: "#2b8a57", radius: 6 }});
      drawPoint(candidate.position_w, {{ fill: "#b43f2c", radius: 7 }});
      drawPoint(candidate.left_finger_w, {{ fill: "#6d3cc6", radius: 6 }});
      drawPoint(candidate.right_finger_w, {{ fill: "#6d3cc6", radius: 6 }});

      drawLine(candidate.left_finger_w, candidate.right_finger_w, {{ stroke: "#6d3cc6", strokeWidth: 3 }});
      drawLine(candidate.pregrasp_position_w, candidate.position_w, {{
        stroke: "#2b8a57",
        strokeWidth: 3,
        markerEnd: "url(#arrow)",
      }});
      drawArrow(candidate.position_w, candidate.approach_axis_w, 0.08, "#b43f2c", 3);
      drawArrow(candidate.position_w, candidate.closing_axis_w, 0.05, "#6d3cc6", 2.5);
      drawArrow(candidate.position_w, candidate.gripper_z_axis_w, 0.05, "#1397a6", 2.5);

      drawLabel(candidate.pregrasp_position_w, "pre", "#2b8a57");
      drawLabel(candidate.position_w, candidate.label, "#b43f2c", 12, -14);
      drawLabel(candidate.left_finger_w, "L", "#6d3cc6", 10, 14);
      drawLabel(candidate.right_finger_w, "R", "#6d3cc6", 10, 14);
    }}

    function render() {{
      const candidate = data.candidates[state.selectedIndex];
      renderList();
      renderDetails(candidate);
      renderScene(candidate);
    }}

    function clamp(value, min, max) {{
      return Math.min(max, Math.max(min, value));
    }}

    prevBtn.addEventListener("click", () => {{
      state.selectedIndex = (state.selectedIndex - 1 + data.candidates.length) % data.candidates.length;
      render();
    }});

    nextBtn.addEventListener("click", () => {{
      state.selectedIndex = (state.selectedIndex + 1) % data.candidates.length;
      render();
    }});

    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") {{
        event.preventDefault();
        prevBtn.click();
      }}
      if (event.key === "ArrowDown" || event.key === "ArrowRight") {{
        event.preventDefault();
        nextBtn.click();
      }}
    }});

    scene.addEventListener("mousedown", (event) => {{
      if (event.button !== 0 && event.button !== 1) {{
        return;
      }}
      event.preventDefault();
      state.dragging = true;
      state.dragMode = event.button === 1 ? "pan" : "rotate";
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      scene.style.cursor = state.dragMode === "pan" ? "move" : "grabbing";
    }});

    window.addEventListener("mouseup", () => {{
      state.dragging = false;
      scene.style.cursor = "grab";
    }});

    window.addEventListener("mousemove", (event) => {{
      if (!state.dragging) {{
        return;
      }}
      const dx = event.clientX - state.lastPointerX;
      const dy = event.clientY - state.lastPointerY;
      state.lastPointerX = event.clientX;
      state.lastPointerY = event.clientY;
      if (state.dragMode === "pan") {{
        state.panX += dx;
        state.panY += dy;
      }} else {{
        state.yaw += dx * 0.01;
        state.pitch = clamp(state.pitch - dy * 0.01, -1.45, 1.45);
      }}
      render();
    }});

    scene.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 1.08 : 1 / 1.08;
      state.zoom = clamp(state.zoom * zoomFactor, 0.35, 4.0);
      render();
    }}, {{ passive: false }});

    scene.style.cursor = "grab";
    scene.addEventListener("contextmenu", (event) => event.preventDefault());

    render();
  </script>
</body>
</html>
"""
    return template.replace("{{", "{").replace("}}", "}").replace("__DATA_JSON__", data_json)


def _build_viewer_state(args: argparse.Namespace) -> _ViewerState:
    generator = CubeFaceGraspGenerator(
        cube_size=args.cube_size,
        pregrasp_offset=args.pregrasp_offset,
        finger_clearance=args.finger_clearance,
    )
    candidates = generator.generate(
        cube_position_w=args.cube_position,
        cube_orientation_xyzw=args.cube_orientation_xyzw,
        robot_base_position_w=args.robot_base_position,
    )
    return _ViewerState(
        generator=generator,
        candidates=candidates,
        cube_size=args.cube_size,
        cube_position_w=args.cube_position,
        cube_orientation_xyzw=args.cube_orientation_xyzw,
        robot_base_position_w=args.robot_base_position,
    )


def main() -> None:
    args = parser.parse_args()
    state = _build_viewer_state(args)
    output_path = args.output_html.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(_build_payload(state)), encoding="utf-8")
    print(f"[INFO] Wrote HTML grasp debug view to: {output_path}")


if __name__ == "__main__":
    main()
