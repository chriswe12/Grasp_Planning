"""Recheck saved Fabrica grasps against a pickup-ground constraint."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    SavedGraspBundle,
    SavedGraspCandidate,
    evaluate_grasps_against_ground,
    ground_plane_overlay_obj,
    load_grasp_bundle,
    load_stl_mesh,
    pickup_pose_for_support_face,
    relative_stl_path,
    save_grasp_bundle,
    shifted_mesh,
    write_debug_html,
)


HARDCODED_PICKUP_SPECS: dict[str, dict[str, object]] = {
    "Fabrica/printing/beam/0.stl": {"support_face": "neg_x", "yaw_deg": 0.0, "xy_world": (0.0, 0.0)},
    "Fabrica/printing/beam/1.stl": {"support_face": "pos_y", "yaw_deg": 90.0, "xy_world": (0.0, 0.0)},
    "Fabrica/printing/beam/2.stl": {"support_face": "neg_y", "yaw_deg": 90.0, "xy_world": (0.0, 0.0)},
    "Fabrica/printing/beam/3.stl": {"support_face": "pos_x", "yaw_deg": 180.0, "xy_world": (0.0, 0.0)},
    "Fabrica/printing/beam/6.stl": {"support_face": "neg_y", "yaw_deg": 270.0, "xy_world": (0.0, 0.0)},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recheck saved Fabrica grasps against a pickup-ground constraint.")
    parser.add_argument("--input-json", type=Path, required=True, help="Input grasp JSON from generate_fabrica_assembly_grasps.py")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON for ground-feasible grasps")
    parser.add_argument("--output-html", type=Path, required=True, help="Output HTML showing accepted and rejected grasps")
    parser.add_argument("--stl-scale", type=float, default=None, help="Optional STL scale override; defaults to the bundle value")
    parser.add_argument("--detailed-finger-contact-gap-m", type=float, default=0.002, help="Detailed Franka finger contact gap")
    parser.add_argument(
        "--contact-lateral-offsets-m",
        type=lambda raw: tuple(float(part.strip()) for part in raw.split(",") if part.strip()),
        default=DEFAULT_CONTACT_LATERAL_OFFSETS_M,
        help="Comma-separated pad-local lateral contact offsets to try when center contact collides.",
    )
    parser.add_argument(
        "--contact-approach-offsets-m",
        type=lambda raw: tuple(float(part.strip()) for part in raw.split(",") if part.strip()),
        default=DEFAULT_CONTACT_APPROACH_OFFSETS_M,
        help="Comma-separated pad-local approach offsets to try when center contact collides.",
    )
    return parser


def _fallback_pickup_spec(relative_path: str) -> dict[str, object]:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    faces = ("pos_x", "neg_x", "pos_y", "neg_y", "neg_z")
    yaws = (0.0, 90.0, 180.0, 270.0)
    return {
        "support_face": faces[digest[0] % len(faces)],
        "yaw_deg": yaws[digest[1] % len(yaws)],
        "xy_world": (0.0, 0.0),
    }


def pickup_spec_for_stl(relative_path: str) -> dict[str, object]:
    return dict(HARDCODED_PICKUP_SPECS.get(relative_path, _fallback_pickup_spec(relative_path)))


def _accepted_bundle(source_bundle: SavedGraspBundle, accepted: list[SavedGraspCandidate], pickup_spec: dict[str, object]) -> SavedGraspBundle:
    metadata = dict(source_bundle.metadata)
    metadata.update(
        {
            "pickup_support_face": pickup_spec["support_face"],
            "pickup_yaw_deg": float(pickup_spec["yaw_deg"]),
            "ground_input_count": len(source_bundle.candidates),
            "ground_feasible_count": len(accepted),
        }
    )
    return SavedGraspBundle(
        target_stl_path=source_bundle.target_stl_path,
        stl_scale=source_bundle.stl_scale,
        local_frame_origin_world=source_bundle.local_frame_origin_world,
        local_frame_orientation_xyzw_world=source_bundle.local_frame_orientation_xyzw_world,
        candidates=tuple(accepted),
        metadata=metadata,
    )


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_grasp_bundle(args.input_json)
    stl_scale = bundle.stl_scale if args.stl_scale is None else float(args.stl_scale)
    mesh_global = load_stl_mesh(bundle.target_stl_path, scale=stl_scale)
    mesh_local = shifted_mesh(mesh_global, -np.asarray(bundle.local_frame_origin_world, dtype=float))
    pickup_spec = pickup_spec_for_stl(bundle.target_stl_path)
    pickup_pose_world = pickup_pose_for_support_face(
        mesh_local,
        support_face=str(pickup_spec["support_face"]),
        yaw_deg=float(pickup_spec["yaw_deg"]),
        xy_world=tuple(float(v) for v in pickup_spec["xy_world"]),
    )
    statuses = evaluate_grasps_against_ground(
        bundle.candidates,
        object_pose_world=pickup_pose_world,
        contact_gap_m=args.detailed_finger_contact_gap_m,
        contact_lateral_offsets_m=args.contact_lateral_offsets_m,
        contact_approach_offsets_m=args.contact_approach_offsets_m,
    )
    accepted = [entry.grasp for entry in statuses if entry.status == "accepted"]
    save_grasp_bundle(_accepted_bundle(bundle, accepted, pickup_spec), args.output_json)
    write_debug_html(
        title="Fabrica Pickup Ground Recheck",
        subtitle="Saved assembly-feasible grasps rechecked against the pickup-ground constraint. Accepted and rejected candidates are shown in the same local frame.",
        mesh_local=mesh_local,
        candidate_statuses=statuses,
        output_html=args.output_html,
        contact_gap_m=args.detailed_finger_contact_gap_m,
        ground_plane=ground_plane_overlay_obj(mesh_local, object_pose_world=pickup_pose_world, enabled=True),
        metadata_lines=[
            f"target_stl:       {relative_stl_path(bundle.target_stl_path)}",
            f"input_grasps:     {len(bundle.candidates)}",
            f"ground_feasible:  {len(accepted)}",
            f"support_face:     {pickup_spec['support_face']}",
            f"pickup_yaw_deg:   {float(pickup_spec['yaw_deg']):.1f}",
            f"contact_offsets_x:{tuple(args.contact_lateral_offsets_m)}",
            f"contact_offsets_z:{tuple(args.contact_approach_offsets_m)}",
            f"pickup_pos_w:     {tuple(round(v, 6) for v in pickup_pose_world.position_world)}",
        ],
    )
    print(
        "[INFO] Pickup ground recheck: "
        f"kept {len(accepted)} / {len(bundle.candidates)} candidates "
        f"for target '{relative_stl_path(bundle.target_stl_path)}'.",
        flush=True,
    )
    print(f"[INFO] Wrote feasible grasp JSON to: {args.output_json.resolve()}", flush=True)
    print(f"[INFO] Wrote HTML debug view to: {args.output_html.resolve()}", flush=True)


if __name__ == "__main__":
    main()
