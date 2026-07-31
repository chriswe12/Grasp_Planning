#!/usr/bin/env python3
"""Build the movable-cell dual-grasp pair ranking debugger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.pipeline import (  # noqa: E402
    MovableFrame,
    ReachabilityProxyConfig,
    build_dual_robot_pair_score_debug_payload,
    compile_assembly_sequence,
    write_dual_robot_pair_score_debug_html,
)
from scripts.build_holder_grasp_library import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    _load_config,
    _repo_path,
)


def _frame_config(
    raw: object,
    *,
    default_position: tuple[float, float, float],
    default_yaw_deg: float,
) -> MovableFrame:
    values = dict(raw) if isinstance(raw, dict) else {}
    position_raw = values.get("position_world_m", default_position)
    if not isinstance(position_raw, (list, tuple)) or len(position_raw) != 3:
        raise ValueError("Movable-frame position_world_m must contain XYZ.")
    return MovableFrame(
        position_world_m=tuple(float(value) for value in position_raw),
        yaw_deg=float(values.get("yaw_deg", default_yaw_deg)),
    )


def _reachability_config(
    payload: dict[str, object],
) -> tuple[MovableFrame, MovableFrame, MovableFrame, float, bool, ReachabilityProxyConfig]:
    raw = dict(payload.get("reachability_debug", {}))
    separation = float(raw.get("robot_separation_y_m", 0.84))
    if separation <= 0.0:
        raise ValueError("reachability_debug.robot_separation_y_m must be > 0.")
    holder = _frame_config(
        raw.get("holder_base"),
        default_position=(0.0, -0.5 * separation, 0.0),
        default_yaw_deg=0.0,
    )
    inserter = _frame_config(
        raw.get("inserter_base"),
        default_position=(0.0, 0.5 * separation, 0.0),
        default_yaw_deg=0.0,
    )
    assembly = _frame_config(
        raw.get("assembly"),
        default_position=(0.55, 0.0, 0.0),
        default_yaw_deg=0.0,
    )
    scoring_raw = dict(raw.get("scoring", {}))
    if "shoulder_offset_base_m" in scoring_raw:
        offset = scoring_raw["shoulder_offset_base_m"]
        if not isinstance(offset, (list, tuple)) or len(offset) != 3:
            raise ValueError("reachability_debug.scoring.shoulder_offset_base_m must contain XYZ.")
        scoring_raw["shoulder_offset_base_m"] = tuple(float(value) for value in offset)
    scoring = ReachabilityProxyConfig(
        **{
            field_name: scoring_raw[field_name]
            for field_name in ReachabilityProxyConfig.__dataclass_fields__
            if field_name in scoring_raw
        }
    )
    return (
        holder,
        inserter,
        assembly,
        separation,
        bool(raw.get("lock_robot_separation", True)),
        scoring,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--assembly", help="Override assembly.name from YAML.")
    parser.add_argument(
        "--base-part-id",
        help="Override the default holder base, forward_assembly_orders[0][0].",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory containing the Stage-3 pair artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (default: <artifact-dir>/dual_robot_pair_score_debug.html).",
    )
    args = parser.parse_args(argv)

    payload = _load_config(args.config.expanduser().resolve())
    assembly_raw = dict(payload.get("assembly", {}))
    artifacts_raw = dict(payload.get("artifacts", {}))
    assembly_name = str(args.assembly or assembly_raw["name"])
    configured_base = assembly_raw.get("base_part_id")
    base_part_id = args.base_part_id if args.base_part_id is not None else configured_base
    sequence = compile_assembly_sequence(
        _repo_path(assembly_raw.get("asset_root", "assets/obj/fabrica")) / assembly_name,
        base_part_id=base_part_id,
        mesh_scale=float(assembly_raw.get("mesh_scale", 0.01)),
        table_z_assembly_m=float(assembly_raw.get("table_z_assembly_m", 0.0)),
        table_contact_tolerance_m=float(assembly_raw.get("table_contact_tolerance_m", 1.0e-6)),
    )
    (
        holder_base,
        inserter_base,
        assembly_world,
        separation,
        lock_separation,
        scoring,
    ) = _reachability_config(payload)
    if args.artifact_dir is None:
        artifact_dir = (
            _repo_path(
                artifacts_raw.get(
                    "output_root",
                    "artifacts/dual_grasp_planning",
                )
            )
            / assembly_name
        )
    else:
        artifact_dir = args.artifact_dir.expanduser().resolve()
    output_path = (
        artifact_dir / "dual_robot_pair_score_debug.html" if args.output is None else args.output.expanduser().resolve()
    )

    debug_payload = build_dual_robot_pair_score_debug_payload(
        sequence=sequence,
        artifact_dir=artifact_dir,
        holder_base_world=holder_base,
        inserter_base_world=inserter_base,
        assembly_world=assembly_world,
        robot_separation_y_m=separation,
        lock_robot_separation=lock_separation,
        scoring=scoring,
    )
    write_dual_robot_pair_score_debug_html(debug_payload, output_path)

    print(f"assembly: {sequence.assembly}")
    print(f"selected order: {' -> '.join(sequence.selected_order)}")
    print(
        "robot bases: "
        f"holder={holder_base.position_world_m} yaw={holder_base.yaw_deg:.1f}°, "
        f"inserter={inserter_base.position_world_m} "
        f"yaw={inserter_base.yaw_deg:.1f}°"
    )
    print(f"assembly: position={assembly_world.position_world_m} yaw={assembly_world.yaw_deg:.1f}°")
    print(f"configured Y separation: {separation:.3f} m")
    print("scope: ranking proxy only; no IK, full-arm collision, or trajectory feasibility")
    print(f"html: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
