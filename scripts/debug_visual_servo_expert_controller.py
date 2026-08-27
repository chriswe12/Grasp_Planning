#!/usr/bin/env python3
"""Compare the legacy and alignment-funnel visual-servo experts offline."""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.rl.visual_servo_curriculum import (  # noqa: E402
    VisualServoCurriculumConfig,
    alignment_funnel_expert_twist,
    expert_twist,
    smooth_trajectory_progress,
)


def _simulate(*, initial_transverse_error_m: float) -> list[dict[str, float | str]]:
    config = VisualServoCurriculumConfig()
    pregrasp = np.array([0.0, 0.0, 0.10])
    grasp = np.zeros(3)
    grasp_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    nominal_velocity = (grasp - pregrasp) / config.approach_duration_s
    states = {
        "legacy": pregrasp + np.array([0.0, initial_transverse_error_m, 0.0]),
        "funnel": pregrasp + np.array([0.0, initial_transverse_error_m, 0.0]),
    }
    rows: list[dict[str, float | str]] = []
    for step in range(config.step_count):
        elapsed = step * config.policy_dt_s
        progress, progress_rate = smooth_trajectory_progress(
            elapsed, config.approach_duration_s
        )
        target = pregrasp + progress * (grasp - pregrasp)
        nominal = np.concatenate(
            (
                nominal_velocity * progress_rate * config.approach_duration_s,
                np.zeros(3),
            )
        )
        for controller in ("legacy", "funnel"):
            error = np.concatenate((target - states[controller], np.zeros(3)))
            if controller == "legacy":
                full, _ = expert_twist(
                    nominal_twist=nominal, pose_error=error, config=config
                )
                diagnostics = {
                    "approach_scale": 1.0,
                    "funnel_half_width_m": 0.0,
                    "near_phase": 0.0,
                }
            else:
                full, _, diagnostics = alignment_funnel_expert_twist(
                    nominal_twist=nominal,
                    pose_error=error,
                    grasp_orientation_xyzw=grasp_quaternion,
                    trajectory_progress=progress,
                    config=config,
                )
            rows.append(
                {
                    "controller": controller,
                    "time_s": elapsed,
                    "progress": progress,
                    "tcp_closing_axis_mm": float(states[controller][1]) * 1000.0,
                    "tcp_approach_axis_mm": float(states[controller][2]) * 1000.0,
                    "target_approach_axis_mm": float(target[2]) * 1000.0,
                    "transverse_error_mm": abs(float(error[1])) * 1000.0,
                    "remaining_distance_mm": max(0.0, float(states[controller][2])) * 1000.0,
                    "approach_velocity_mm_s": -float(full[2]) * 1000.0,
                    "transverse_velocity_mm_s": float(full[1]) * 1000.0,
                    "approach_scale": diagnostics["approach_scale"],
                    "funnel_half_width_mm": diagnostics["funnel_half_width_m"] * 1000.0,
                    "near_phase": diagnostics["near_phase"],
                }
            )
            states[controller] = states[controller] + full[:3] * config.policy_dt_s
    return rows


def _polyline(
    rows: list[dict[str, float | str]],
    *,
    controller: str,
    key: str,
    x: float,
    y: float,
    width: float,
    height: float,
    maximum: float,
) -> str:
    selected = [row for row in rows if row["controller"] == controller]
    max_time = max(float(row["time_s"]) for row in selected)
    points = []
    for row in selected:
        px = x + width * float(row["time_s"]) / max_time
        py = y + height * (1.0 - min(max(float(row[key]), 0.0), maximum) / maximum)
        points.append(f"{px:.1f},{py:.1f}")
    color = "#d1495b" if controller == "legacy" else "#00798c"
    return (
        f'<polyline points="{" ".join(points)}" fill="none" '
        f'stroke="{color}" stroke-width="3"/>'
    )


def _write_html(
    path: Path,
    rows: list[dict[str, float | str]],
    *,
    initial_error_mm: float,
) -> None:
    panels = (
        ("transverse_error_mm", "Transverse error (mm)", 20.0),
        ("remaining_distance_mm", "Remaining approach distance (mm)", 110.0),
        ("approach_velocity_mm_s", "Approach speed (mm/s)", 55.0),
        ("funnel_half_width_mm", "Allowed funnel half-width (mm)", 10.0),
    )
    svg_parts = [
        '<svg viewBox="0 0 920 900" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="920" height="900" fill="#f7f5ef"/>',
        '<text x="45" y="42" font-size="25" font-family="sans-serif">Expert controller comparison</text>',
        f'<text x="45" y="70" font-size="15" font-family="sans-serif">initial transverse error: {initial_error_mm:.1f} mm</text>',
        '<line x1="610" y1="61" x2="650" y2="61" stroke="#d1495b" stroke-width="3"/>',
        '<text x="658" y="66" font-size="14" font-family="sans-serif">legacy P controller</text>',
        '<line x1="610" y1="82" x2="650" y2="82" stroke="#00798c" stroke-width="3"/>',
        '<text x="658" y="87" font-size="14" font-family="sans-serif">alignment funnel</text>',
    ]
    for panel_index, (key, title, maximum) in enumerate(panels):
        x, y, width, height = 70.0, 125.0 + panel_index * 185.0, 800.0, 130.0
        svg_parts.extend(
            [
                f'<text x="{x:.0f}" y="{y - 12:.0f}" font-size="16" font-family="sans-serif">{html.escape(title)}</text>',
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{width:.0f}" height="{height:.0f}" fill="white" stroke="#bbb"/>',
                f'<text x="{x - 8:.0f}" y="{y + 5:.0f}" text-anchor="end" font-size="12">{maximum:g}</text>',
                f'<text x="{x - 8:.0f}" y="{y + height:.0f}" text-anchor="end" font-size="12">0</text>',
                _polyline(
                    rows,
                    controller="legacy",
                    key=key,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    maximum=maximum,
                ),
                _polyline(
                    rows,
                    controller="funnel",
                    key=key,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    maximum=maximum,
                ),
            ]
        )
    svg_parts.append("</svg>")
    payload = "\n".join(svg_parts)
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Expert controller debug</title>"
        f"<body style='margin:0'>{payload}</body>\n",
        encoding="utf-8",
    )


def _write_path_html(
    path: Path,
    rows: list[dict[str, float | str]],
    *,
    initial_error_mm: float,
) -> None:
    """Write a spatial closing-axis-versus-approach-axis path report."""

    x, y, width, height = 90.0, 105.0, 740.0, 620.0

    def point(closing_mm: float, approach_mm: float) -> tuple[float, float]:
        px = x + width * (closing_mm + 20.0) / 40.0
        py = y + height * (1.0 - np.clip(approach_mm, 0.0, 110.0) / 110.0)
        return px, py

    parts = [
        '<svg viewBox="0 0 920 800" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="920" height="800" fill="#f7f5ef"/>',
        '<text x="45" y="42" font-size="25" font-family="sans-serif">TCP spatial path toward grasp</text>',
        f'<text x="45" y="70" font-size="15" font-family="sans-serif">initial closing-axis error: {initial_error_mm:.1f} mm</text>',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="white" stroke="#aaa"/>',
    ]
    safe_left, _ = point(-4.0, 0.0)
    safe_right, _ = point(4.0, 0.0)
    parts.append(
        f'<rect x="{safe_left:.1f}" y="{y:.1f}" width="{safe_right - safe_left:.1f}" '
        f'height="{height:.1f}" fill="#b8e0d2" opacity="0.35"/>'
    )
    for closing_mm in (-20, -10, 0, 10, 20):
        px, _ = point(float(closing_mm), 0.0)
        parts.extend(
            [
                f'<line x1="{px:.1f}" y1="{y:.1f}" x2="{px:.1f}" y2="{y + height:.1f}" stroke="#ddd"/>',
                f'<text x="{px:.1f}" y="{y + height + 22:.1f}" text-anchor="middle" font-size="12">{closing_mm}</text>',
            ]
        )
    for approach_mm in (0, 25, 50, 75, 100):
        _, py = point(0.0, float(approach_mm))
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{py:.1f}" x2="{x + width:.1f}" y2="{py:.1f}" stroke="#ddd"/>',
                f'<text x="{x - 10:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="12">{approach_mm}</text>',
            ]
        )
    for controller, color in (("legacy", "#d1495b"), ("funnel", "#00798c")):
        selected = [row for row in rows if row["controller"] == controller]
        points = [
            point(
                float(row["tcp_closing_axis_mm"]),
                float(row["tcp_approach_axis_mm"]),
            )
            for row in selected
        ]
        point_text = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="4"/>'
        )
        for row, (px, py) in zip(selected[::15], points[::15], strict=True):
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}">'
                f'<title>t={float(row["time_s"]):.2f}s, lateral={float(row["tcp_closing_axis_mm"]):.2f}mm, '
                f'approach={float(row["tcp_approach_axis_mm"]):.2f}mm</title></circle>'
            )
    object_x, object_y = point(0.0, 0.0)
    parts.extend(
        [
            f'<circle cx="{object_x:.1f}" cy="{object_y:.1f}" r="10" fill="#e9c46a" stroke="#8a5a00"/>',
            f'<text x="{object_x + 16:.1f}" y="{object_y - 8:.1f}" font-size="13">grasp/object center</text>',
            f'<text x="{x + width / 2:.1f}" y="775" text-anchor="middle" font-size="14">closing-axis offset (mm)</text>',
            '<text x="20" y="410" transform="rotate(-90 20 410)" text-anchor="middle" font-size="14">remaining approach distance (mm)</text>',
            '<line x1="620" y1="61" x2="660" y2="61" stroke="#d1495b" stroke-width="4"/>',
            '<text x="668" y="66" font-size="14">legacy</text>',
            '<line x1="735" y1="61" x2="775" y2="61" stroke="#00798c" stroke-width="4"/>',
            '<text x="783" y="66" font-size="14">funnel</text>',
            "</svg>",
        ]
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Expert TCP paths</title>"
        f"<body style='margin:0'>{''.join(parts)}</body>\n",
        encoding="utf-8",
    )


def _write_video(
    path: Path,
    rows: list[dict[str, float | str]],
    *,
    fps: float,
) -> None:
    """Animate legacy and funnel paths as a portable Pillow GIF."""

    from PIL import Image, ImageDraw

    width, height = 1280, 720
    grouped = {
        controller: [row for row in rows if row["controller"] == controller]
        for controller in ("legacy", "funnel")
    }
    colors = {"legacy": (209, 73, 91), "funnel": (0, 121, 140)}
    frames = []
    for frame_index in range(len(grouped["legacy"])):
        frame = Image.new("RGB", (width, height), (247, 245, 239))
        draw = ImageDraw.Draw(frame)
        draw.text(
            (35, 25),
            "Visual-servo expert: object-between-fingers approach",
            fill=(35, 35, 35),
            stroke_width=1,
        )
        for panel_index, controller in enumerate(("legacy", "funnel")):
            row = grouped[controller][frame_index]
            left = 35 + panel_index * 625
            top, panel_width, panel_height = 80, 590, 580
            draw.rectangle(
                (left, top, left + panel_width, top + panel_height),
                fill=(255, 255, 255),
                outline=(180, 180, 180),
            )
            draw.text(
                (left + 18, top + 18),
                "Legacy P controller" if controller == "legacy" else "Alignment funnel",
                fill=colors[controller],
                stroke_width=1,
            )
            origin_x, goal_y = left + panel_width // 2, top + panel_height - 65
            scale_x, scale_z = 11.0, 4.3
            corridor_half = float(row["funnel_half_width_mm"])
            if controller == "funnel":
                draw.rectangle(
                    (
                        int(origin_x - corridor_half * scale_x),
                        top + 75,
                        int(origin_x + corridor_half * scale_x),
                        goal_y,
                    ),
                    fill=(220, 242, 232),
                )
            draw.line((origin_x, top + 75, origin_x, goal_y), fill=(205, 205, 205))
            draw.ellipse(
                (origin_x - 16, goal_y - 16, origin_x + 16, goal_y + 16),
                fill=(235, 180, 80),
                outline=(120, 90, 20),
            )
            tcp_x = int(origin_x + float(row["tcp_closing_axis_mm"]) * scale_x)
            tcp_y = int(goal_y - float(row["tcp_approach_axis_mm"]) * scale_z)
            finger_gap = 44
            draw.rectangle(
                (tcp_x - finger_gap - 8, tcp_y - 12, tcp_x - finger_gap + 8, tcp_y + 42),
                fill=(35, 35, 35),
            )
            draw.rectangle(
                (tcp_x + finger_gap - 8, tcp_y - 12, tcp_x + finger_gap + 8, tcp_y + 42),
                fill=(35, 35, 35),
            )
            draw.line(
                (tcp_x - finger_gap, tcp_y, tcp_x + finger_gap, tcp_y),
                fill=colors[controller],
                width=3,
            )
            diagnostics = (
                f"t={float(row['time_s']):.2f} s",
                f"closing-axis error={float(row['transverse_error_mm']):.2f} mm",
                f"remaining distance={float(row['remaining_distance_mm']):.2f} mm",
                f"approach scale={float(row['approach_scale']):.2f}",
                f"funnel half-width={corridor_half:.2f} mm",
            )
            for line_index, line in enumerate(diagnostics):
                draw.text(
                    (left + 18, top + 410 + line_index * 27),
                    line,
                    fill=(50, 50, 50),
                )
            status = (
                "HOLD / ALIGN"
                if float(row["approach_scale"]) < 0.1
                else "SLOW APPROACH"
                if float(row["approach_scale"]) < 0.9
                else "APPROACH"
            )
            draw.text(
                (left + 400, top + 545),
                status,
                fill=colors[controller],
                stroke_width=1,
            )
        frames.append(frame)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(round(1000.0 / float(fps)))),
        loop=0,
        optimize=False,
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Debug video was not written to {path}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-transverse-error-mm", type=float, default=15.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/visual_servo_expert_debug"),
    )
    args = parser.parse_args()
    rows = _simulate(
        initial_transverse_error_m=float(args.initial_transverse_error_mm) / 1000.0
    )
    csv_path = args.output_prefix.with_suffix(".csv")
    html_path = args.output_prefix.with_suffix(".html")
    path_html_path = args.output_prefix.with_name(
        f"{args.output_prefix.name}_path"
    ).with_suffix(".html")
    video_path = args.output_prefix.with_suffix(".gif")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_html(
        html_path,
        rows,
        initial_error_mm=float(args.initial_transverse_error_mm),
    )
    _write_path_html(
        path_html_path,
        rows,
        initial_error_mm=float(args.initial_transverse_error_mm),
    )
    _write_video(video_path, rows, fps=30.0)
    for controller in ("legacy", "funnel"):
        selected = [row for row in rows if row["controller"] == controller]
        first_advance = next(
            (
                float(row["time_s"])
                for row in selected
                if float(row["approach_velocity_mm_s"]) > 1.0
            ),
            float("nan"),
        )
        print(
            f"{controller}: first_advance_s={first_advance:.3f} "
            f"final_transverse_error_mm={float(selected[-1]['transverse_error_mm']):.3f} "
            f"final_remaining_distance_mm={float(selected[-1]['remaining_distance_mm']):.3f}"
        )
    print(f"csv={csv_path}")
    print(f"html={html_path}")
    print(f"path_html={path_html_path}")
    print(f"video={video_path}")


if __name__ == "__main__":
    main()
