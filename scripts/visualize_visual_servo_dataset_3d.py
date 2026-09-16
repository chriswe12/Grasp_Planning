#!/usr/bin/env python3
"""Write an interactive 3D HTML overview of visual-servo dataset randomization."""

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
    load_asset_mesh,
    load_grasp_bundle,
    quat_to_rotmat_xyzw,
)
from grasp_planning.grasping.grasp_transforms import saved_grasp_to_world_grasp  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline.fabrica_pipeline import _mesh_in_source_frame  # noqa: E402


def _rotation_z(yaw_rad: float) -> np.ndarray:
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def _rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.array(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _episode_geometry(
    *,
    npz_path: Path,
    metadata: dict[str, object],
    nominal_object_pose: ObjectWorldPose,
    nominal_pregrasp: np.ndarray,
    nominal_grasp: np.ndarray,
) -> dict[str, object]:
    perturbation = metadata["object_perturbation"]
    dx = float(perturbation["dx_m"])
    dy = float(perturbation["dy_m"])
    yaw_rad = np.deg2rad(float(perturbation["yaw_deg"]))
    yaw_rotation = _rotation_z(yaw_rad)
    object_position = nominal_object_pose.translation_world + np.array((dx, dy, 0.0))
    object_rotation = yaw_rotation @ nominal_object_pose.rotation_world_from_object
    pregrasp = object_position + yaw_rotation @ (nominal_pregrasp - nominal_object_pose.translation_world)
    grasp = object_position + yaw_rotation @ (nominal_grasp - nominal_object_pose.translation_world)
    with np.load(npz_path) as episode:
        progress = np.asarray(episode["trajectory_progress"], dtype=np.float64)
        pose_error = np.asarray(episode["pose_error"], dtype=np.float64)
        goal_tcp_position = np.asarray(
            episode["goal_tcp_position_w"], dtype=np.float64
        )
        goal_tcp_rotation = quat_to_rotmat_xyzw(
            np.asarray(episode["goal_tcp_orientation_xyzw_w"], dtype=np.float64)
        )
        goal_object_position = np.asarray(
            episode["goal_object_position_w"], dtype=np.float64
        )
        goal_object_rotation = quat_to_rotmat_xyzw(
            np.asarray(episode["goal_object_orientation_xyzw_w"], dtype=np.float64)
        )
    tcp_position_object = goal_object_rotation.T @ (
        goal_tcp_position - goal_object_position
    )
    tcp_rotation_object = goal_object_rotation.T @ goal_tcp_rotation
    grasp_tcp = object_position + object_rotation @ tcp_position_object
    pregrasp_tcp = grasp_tcp + yaw_rotation @ (nominal_pregrasp - nominal_grasp)
    target_tcp_rotation = object_rotation @ tcp_rotation_object
    targets = (
        pregrasp_tcp[None, :]
        + progress[:, None] * (grasp_tcp - pregrasp_tcp)[None, :]
    )
    path_world = targets - pose_error[:, :3]
    initial_rotation_world = (
        _rotation_from_rotvec(pose_error[0, 3:]).T @ target_tcp_rotation
    )
    final_rotation_world = (
        _rotation_from_rotvec(pose_error[-1, 3:]).T @ target_tcp_rotation
    )
    # Normalize every sample into its own randomized part frame.  This removes
    # absolute scene placement and exposes only the TCP/target relationship to
    # the part, which is the relationship the visual servo policy must learn.
    world_to_part = object_rotation.T
    path = (path_world - object_position[None, :]) @ object_rotation
    pregrasp_part = world_to_part @ (pregrasp - object_position)
    grasp_part = world_to_part @ (grasp - object_position)
    return {
        "episode": int(metadata["episode_index"]),
        "split": str(metadata["split"]),
        "success": bool(metadata["success"]),
        "object_position": np.zeros(3, dtype=np.float64),
        "object_rotation": np.eye(3, dtype=np.float64),
        "pregrasp": pregrasp_part,
        "grasp": grasp_part,
        "path": path,
        "initial_rotation": world_to_part @ initial_rotation_world,
        "final_rotation": world_to_part @ final_rotation_world,
        "final_position_error_m": float(metadata["final_position_error_m"]),
        "final_rotation_error_deg": float(metadata["final_rotation_error_deg"]),
        "dx_m": dx,
        "dy_m": dy,
        "yaw_deg": float(perturbation["yaw_deg"]),
    }


def _axis_traces(go, *, origin: np.ndarray, rotation: np.ndarray, prefix: str, visible: bool, length: float):
    colors = ("#ef4444", "#22c55e", "#3b82f6")
    names = ("x", "y", "z")
    traces = []
    for axis in range(3):
        endpoint = origin + rotation[:, axis] * length
        traces.append(
            go.Scatter3d(
                x=(origin[0], endpoint[0]),
                y=(origin[1], endpoint[1]),
                z=(origin[2], endpoint[2]),
                mode="lines",
                line={"color": colors[axis], "width": 5},
                name=f"{prefix} {names[axis]}",
                legendgroup=prefix,
                showlegend=False,
                visible=visible,
                hoverinfo="skip",
            )
        )
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/pipeline_stage2_ground_feasible.json"),
        help="Stage-2 grasp bundle used to generate the dataset.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/curriculum_one_100_trajectory_debug.html"))
    parser.add_argument("--max-episodes", type=int, default=0, help="Limit episodes embedded in HTML; zero means all.")
    parser.add_argument("--pregrasp-offset", type=float, default=0.10)
    args = parser.parse_args()

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit("Plotly is required: install it with `python3 -m pip install plotly`.") from exc

    bundle = load_grasp_bundle(args.bundle)
    pose_raw = bundle.metadata.get("execution_world_pose")
    if not isinstance(pose_raw, dict):
        raise ValueError(f"{args.bundle} has no execution_world_pose metadata.")
    nominal_object_pose = ObjectWorldPose(
        position_world=tuple(float(value) for value in pose_raw["position_world"]),
        orientation_xyzw_world=tuple(float(value) for value in pose_raw["orientation_xyzw_world"]),
    )
    metadata_paths = sorted(args.dataset_dir.glob("episode_*.json"))
    if args.max_episodes > 0:
        metadata_paths = metadata_paths[: args.max_episodes]
    if not metadata_paths:
        raise ValueError(f"No episode metadata found under {args.dataset_dir}.")
    first_metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    grasp_id = str(first_metadata["grasp_id"])
    saved_grasp = next((candidate for candidate in bundle.candidates if candidate.grasp_id == grasp_id), None)
    if saved_grasp is None:
        raise ValueError(f"Grasp {grasp_id!r} is absent from {args.bundle}.")
    world_grasp = saved_grasp_to_world_grasp(
        saved_grasp,
        nominal_object_pose,
        pregrasp_offset=float(args.pregrasp_offset),
        gripper_width_clearance=0.01,
    )
    nominal_pregrasp = np.asarray(world_grasp.pregrasp_position_w)
    nominal_grasp = np.asarray(world_grasp.position_w)
    mesh_world = load_asset_mesh(bundle.target_mesh_path, scale=bundle.mesh_scale)
    source_pose = ObjectWorldPose(
        position_world=bundle.source_frame_origin_obj_world,
        orientation_xyzw_world=bundle.source_frame_orientation_xyzw_obj_world,
    )
    mesh_local = _mesh_in_source_frame(mesh_world, source_pose)
    vertices_local = np.asarray(mesh_local.vertices_obj, dtype=np.float64)
    faces = np.asarray(mesh_local.faces, dtype=np.int64)

    episodes = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        npz_path = metadata_path.with_suffix(".npz")
        episodes.append(
            _episode_geometry(
                npz_path=npz_path,
                metadata=metadata,
                nominal_object_pose=nominal_object_pose,
                nominal_pregrasp=nominal_pregrasp,
                nominal_grasp=nominal_grasp,
            )
        )

    figure = go.Figure()
    # Overview traces remain available through the first dropdown entry.
    for episode in episodes:
        path = episode["path"]
        color = "#dc2626" if not episode["success"] else "#2563eb"
        figure.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                line={"color": color, "width": 2},
                opacity=0.20,
                name=f"episode {episode['episode']}",
                showlegend=False,
                visible=True,
                hovertemplate=f"episode {episode['episode']}<extra></extra>",
            )
        )
    overview_trace_count = len(figure.data)
    initial_positions = np.stack([episode["path"][0] for episode in episodes])
    final_positions = np.stack([episode["path"][-1] for episode in episodes])
    labels = [
        (
            f"episode {episode['episode']}<br>split={episode['split']} success={episode['success']}"
            f"<br>dx={episode['dx_m'] * 1000:+.1f} mm dy={episode['dy_m'] * 1000:+.1f} mm"
            f"<br>yaw={episode['yaw_deg']:+.1f} deg"
        )
        for episode in episodes
    ]
    for points, name, color, symbol in (
        (initial_positions, "initial TCP", "#f59e0b", "circle"),
        (final_positions, "final TCP", "#7c3aed", "square"),
    ):
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": 4, "color": color, "symbol": symbol},
                name=name,
                text=labels,
                hovertemplate="%{text}<extra></extra>",
                visible=True,
            )
        )
    overview_trace_count = len(figure.data)

    detail_ranges: list[tuple[int, int]] = []
    for episode in episodes:
        start = len(figure.data)
        vertices = episode["object_position"] + vertices_local @ episode["object_rotation"].T
        path = episode["path"]
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#a16207",
                opacity=0.55,
                name="perturbed part",
                visible=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                line={"color": "#dc2626" if not episode["success"] else "#2563eb", "width": 7},
                name="actual TCP path",
                visible=False,
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=(episode["pregrasp"][0], episode["grasp"][0]),
                y=(episode["pregrasp"][1], episode["grasp"][1]),
                z=(episode["pregrasp"][2], episode["grasp"][2]),
                mode="lines+markers",
                line={"color": "#111827", "width": 4, "dash": "dash"},
                marker={"size": 4},
                name="privileged target path",
                visible=False,
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=(path[0, 0], path[-1, 0]),
                y=(path[0, 1], path[-1, 1]),
                z=(path[0, 2], path[-1, 2]),
                mode="markers+text",
                marker={"size": 7, "color": ("#f59e0b", "#7c3aed")},
                text=("initial TCP", "final TCP"),
                textposition="top center",
                name="initial/final TCP",
                visible=False,
            )
        )
        for trace in _axis_traces(
            go,
            origin=episode["object_position"],
            rotation=episode["object_rotation"],
            prefix="part frame",
            visible=False,
            length=0.025,
        ):
            figure.add_trace(trace)
        for trace in _axis_traces(
            go,
            origin=path[0],
            rotation=episode["initial_rotation"],
            prefix="initial TCP frame",
            visible=False,
            length=0.018,
        ):
            figure.add_trace(trace)
        for trace in _axis_traces(
            go,
            origin=path[-1],
            rotation=episode["final_rotation"],
            prefix="final TCP frame",
            visible=False,
            length=0.018,
        ):
            figure.add_trace(trace)
        detail_ranges.append((start, len(figure.data)))

    buttons = [
        {
            "label": "Overview - all episodes",
            "method": "update",
            "args": [
                {"visible": [index < overview_trace_count for index in range(len(figure.data))]},
                {"title": f"Part-frame curriculum overview - {len(episodes)} episodes"},
            ],
        }
    ]
    for episode, (start, end) in zip(episodes, detail_ranges, strict=True):
        visibility = [False] * len(figure.data)
        visibility[start:end] = [True] * (end - start)
        status = "success" if episode["success"] else "FAILED"
        buttons.append(
            {
                "label": f"Episode {episode['episode']:03d} - {status}",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "title": (
                            f"Episode {episode['episode']:03d} - {status} | "
                            f"dx={episode['dx_m'] * 1000:+.1f} mm, dy={episode['dy_m'] * 1000:+.1f} mm, "
                            f"yaw={episode['yaw_deg']:+.1f} deg | "
                            f"final={episode['final_position_error_m'] * 1000:.2f} mm, "
                            f"{episode['final_rotation_error_deg']:.2f} deg"
                        )
                    },
                ],
            }
        )

    figure.update_layout(
        title=f"Part-frame curriculum overview - {len(episodes)} episodes",
        template="plotly_white",
        scene={
            "xaxis_title": "part X [m]",
            "yaxis_title": "part Y [m]",
            "zaxis_title": "part Z [m]",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.4, "y": -1.5, "z": 1.1}},
        },
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.01,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            }
        ],
        legend={"x": 0.82, "y": 0.98},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Wrote {len(episodes)} episodes to {args.output}.")


if __name__ == "__main__":
    main()
