"""Recheck saved Fabrica grasps against a pickup-ground constraint."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.grasping.fabrica_grasp_debug import (  # noqa: E402
    DEFAULT_CONTACT_APPROACH_OFFSETS_M,
    DEFAULT_CONTACT_LATERAL_OFFSETS_M,
    load_grasp_bundle,
    relative_asset_mesh_path,
)
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.pipeline import (  # noqa: E402
    PickupPoseConfig,
    PlanningConfig,
    recheck_stage2_result,
    write_stage2_artifacts,
)

HARDCODED_PICKUP_SPECS: dict[str, dict[str, object]] = {
    "Fabrica/printing/beam/0.obj": {"support_face": "neg_x", "yaw_deg": 0.0, "xy_world": (0.0, 0.0)},
    "Fabrica/printing/beam/1.obj": {"support_face": "pos_y", "yaw_deg": 90.0, "xy_world": (0.0, 0.0)},
    "Fabrica/printing/beam/2.obj": {"support_face": "neg_y", "yaw_deg": 90.0, "xy_world": (0.0, 0.0)},
    "Fabrica/printing/beam/3.obj": {"support_face": "pos_x", "yaw_deg": 180.0, "xy_world": (0.0, 0.0)},
    "Fabrica/printing/beam/6.obj": {"support_face": "neg_y", "yaw_deg": 270.0, "xy_world": (0.0, 0.0)},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recheck saved Fabrica grasps against a pickup-ground constraint.")
    parser.add_argument(
        "--input-json", type=Path, required=True, help="Input grasp JSON from generate_fabrica_assembly_grasps.py"
    )
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON for ground-feasible grasps")
    parser.add_argument(
        "--output-html", type=Path, required=True, help="Output HTML showing accepted and rejected grasps"
    )
    parser.add_argument(
        "--mesh-scale",
        "--stl-scale",
        dest="mesh_scale",
        type=float,
        default=None,
        help="Optional mesh scale override; defaults to the bundle value",
    )
    parser.add_argument(
        "--detailed-finger-contact-gap-m", type=float, default=0.002, help="Detailed Franka finger contact gap"
    )
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
    parser.add_argument("--support-face", type=str, default="", help="Optional explicit support face override.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Optional explicit pickup yaw override in degrees.")
    parser.add_argument("--xy-world", type=str, default="", help="Optional explicit world XY override as x,y.")
    parser.add_argument(
        "--object-position-world", type=str, default="", help="Optional explicit object world position as x,y,z."
    )
    parser.add_argument(
        "--object-orientation-xyzw", type=str, default="", help="Optional explicit object world orientation as x,y,z,w."
    )
    return parser


def _parse_vec2(raw: str) -> tuple[float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 2:
        raise ValueError(f"Expected exactly 2 comma-separated values, got '{raw}'.")
    return values


def _parse_vec3(raw: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 3:
        raise ValueError(f"Expected exactly 3 comma-separated values, got '{raw}'.")
    return values


def _parse_quat(raw: str) -> tuple[float, float, float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 4:
        raise ValueError(f"Expected exactly 4 comma-separated values, got '{raw}'.")
    return values


def _fallback_pickup_spec(relative_path: str) -> dict[str, object]:
    digest = hashlib.sha256(relative_path.encode("utf-8")).digest()
    faces = ("pos_x", "neg_x", "pos_y", "neg_y", "neg_z")
    yaws = (0.0, 90.0, 180.0, 270.0)
    return {
        "support_face": faces[digest[0] % len(faces)],
        "yaw_deg": yaws[digest[1] % len(yaws)],
        "xy_world": (0.0, 0.0),
    }


def pickup_spec_from_args(args: argparse.Namespace, *, relative_path: str) -> dict[str, object]:
    base_spec = dict(HARDCODED_PICKUP_SPECS.get(relative_path, _fallback_pickup_spec(relative_path)))
    if args.support_face:
        base_spec["support_face"] = str(args.support_face)
    if args.yaw_deg is not None:
        base_spec["yaw_deg"] = float(args.yaw_deg)
    if args.xy_world:
        base_spec["xy_world"] = _parse_vec2(args.xy_world)
    return base_spec


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_grasp_bundle(args.input_json)
    if args.mesh_scale is not None:
        bundle = type(bundle)(
            target_mesh_path=bundle.target_mesh_path,
            mesh_scale=float(args.mesh_scale),
            source_frame_origin_obj_world=bundle.source_frame_origin_obj_world,
            source_frame_orientation_xyzw_obj_world=bundle.source_frame_orientation_xyzw_obj_world,
            candidates=bundle.candidates,
            metadata=dict(bundle.metadata),
        )
    planning = PlanningConfig(
        detailed_finger_contact_gap_m=args.detailed_finger_contact_gap_m,
        contact_lateral_offsets_m=args.contact_lateral_offsets_m,
        contact_approach_offsets_m=args.contact_approach_offsets_m,
    )
    object_pose_world = None
    pickup_spec = None
    if args.object_position_world or args.object_orientation_xyzw:
        if not (args.object_position_world and args.object_orientation_xyzw):
            raise ValueError("--object-position-world and --object-orientation-xyzw must be provided together.")
        object_pose_world = ObjectWorldPose(
            position_world=_parse_vec3(args.object_position_world),
            orientation_xyzw_world=_parse_quat(args.object_orientation_xyzw),
        )
    else:
        pickup = pickup_spec_from_args(args, relative_path=bundle.target_mesh_path)
        pickup_spec = PickupPoseConfig(
            support_face=str(pickup["support_face"]),
            yaw_deg=float(pickup["yaw_deg"]),
            xy_world=tuple(float(v) for v in pickup["xy_world"]),
        ).to_spec()
    result = recheck_stage2_result(
        bundle=bundle,
        pickup_spec=pickup_spec,
        planning=planning,
        object_pose_world=object_pose_world,
    )
    write_stage2_artifacts(
        result,
        planning=planning,
        output_json=args.output_json,
        output_html=args.output_html,
    )
    print(
        "[INFO] Pickup ground recheck: "
        f"kept {len(result.accepted)} / {len(bundle.candidates)} candidates "
        f"for target '{relative_asset_mesh_path(bundle.target_mesh_path)}'.",
        flush=True,
    )
    print(f"[INFO] Wrote feasible grasp JSON to: {args.output_json.resolve()}", flush=True)
    print(f"[INFO] Wrote HTML debug view to: {args.output_html.resolve()}", flush=True)


if __name__ == "__main__":
    main()
