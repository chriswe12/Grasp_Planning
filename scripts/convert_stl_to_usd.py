"""Convert an STL asset into an Isaac-native USD mesh asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from grasp_planning.grasping.fabrica_grasp_debug import resolve_stl_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert an STL asset into an Isaac-native USD mesh asset.")
    parser.add_argument(
        "--stl-path", type=Path, required=True, help="Input STL path, relative to assets/stl or absolute."
    )
    parser.add_argument("--output-usd", type=Path, required=True, help="Output USD path.")
    parser.add_argument("--stl-scale", type=float, default=1.0, help="Uniform STL scale applied during conversion.")
    parser.add_argument("--mass", type=float, default=0.15, help="Mass in kilograms for the rigid root.")
    parser.add_argument(
        "--collision-approximation",
        type=str,
        default="convex_decomposition",
        choices=("convex_hull", "convex_decomposition", "sdf"),
        help="Collision approximation used by Isaac Lab during mesh conversion.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402


def _mesh_collision_cfg(name: str):
    if name == "convex_hull":
        return schemas_cfg.ConvexHullPropertiesCfg()
    if name == "convex_decomposition":
        return schemas_cfg.ConvexDecompositionPropertiesCfg()
    if name == "sdf":
        return schemas_cfg.SDFMeshPropertiesCfg()
    raise ValueError(f"Unsupported collision approximation '{name}'.")


def run() -> Path:
    source_stl = resolve_stl_path(args_cli.stl_path)
    output_path = args_cli.output_usd.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter_cfg = MeshConverterCfg(
        asset_path=str(source_stl),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        force_usd_conversion=True,
        make_instanceable=False,
        scale=(float(args_cli.stl_scale), float(args_cli.stl_scale), float(args_cli.stl_scale)),
        mass_props=sim_utils.MassPropertiesCfg(mass=float(args_cli.mass)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        mesh_collision_props=_mesh_collision_cfg(args_cli.collision_approximation),
    )
    converter = MeshConverter(converter_cfg)
    converted_path = Path(converter.usd_path).resolve()
    print(
        f"[INFO] Converted STL '{source_stl}' -> '{converted_path}' (approximation={args_cli.collision_approximation})",
        flush=True,
    )
    return converted_path


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
