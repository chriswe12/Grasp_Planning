#!/usr/bin/env python3
"""Render a STEP/STL/OBJ part profile as an Isaac floor with aluminum material."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from isaaclab.app import AppLauncher
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mesh", type=Path, required=True, help="Input STEP/STL/OBJ path.")
parser.add_argument("--robot-usd", type=Path, default=Path("assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"))
parser.add_argument("--output-dir", type=Path, default=Path("artifacts/slot_floor_render"))
parser.add_argument("--output-prefix", type=str, default="slot_floor")
parser.add_argument("--frames", type=int, default=24, help="Number of RGB images to save.")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--stl-scale", type=float, default=1.0)
parser.add_argument("--repo-root", type=Path, default=Path("."))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.repo_root = args.repo_root.expanduser().resolve()
if str(args.repo_root) not in sys.path:
    sys.path.insert(0, str(args.repo_root))


ISAAC_MIN_CONTACT_OFFSET_M = 1.0e-5


def _ensure_image_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_command(cmd: list[str], *, expect_output: bool = False) -> str:
    """Run a command and return combined stdout+stderr on failure."""
    completed = subprocess.run(
        cmd,
        check=True,
        stdout=(subprocess.PIPE if expect_output else subprocess.DEVNULL),
        stderr=(subprocess.STDOUT if expect_output else subprocess.DEVNULL),
        text=True,
    )
    return completed.stdout if expect_output else ""


def _convert_step_to_stl(step_path: Path, stl_path: Path) -> None:
    """Try available command-line converters for STEP/STP -> STL."""
    if shutil.which("assimp"):
        try:
            _run_command(["assimp", "export", str(step_path), str(stl_path)])
            if stl_path.exists():
                return
        except Exception:
            pass

    if shutil.which("gmsh"):
        try:
            _run_command(["gmsh", "-3", str(step_path), "-o", str(stl_path)])
            if stl_path.exists():
                return
        except Exception:
            pass

    freecad_candidates = (
        "freecadcmd",
        "/snap/freecad/current/usr/bin/FreeCADCmd",
        "/snap/freecad/2337/usr/bin/FreeCADCmd",
        "/snap/freecad/2266/usr/bin/FreeCADCmd",
    )
    for freecad_cmd in freecad_candidates:
        freecad_path = freecad_cmd if shutil.which(freecad_cmd) else freecad_cmd
        if not Path(freecad_path).exists() and shutil.which(freecad_cmd) is None:
            continue
        # Keep this command minimal and robust for CLI-only environments.
        script = (
            "import Mesh, Part\n"
            f"shape = Part.read('{step_path.as_posix()}')\n"
            f"mesh = Mesh.Mesh(shape.tessellate(0.1))\n"
            f"mesh.write('{stl_path.as_posix()}')\n"
        )
        try:
            _run_command(
                [freecad_path, "--console", "-c", script],
                expect_output=True,
            )
            if stl_path.exists():
                return
        except Exception:
            continue

    raise RuntimeError(
        "No supported STEP/STP converter was available in this environment. "
        f"Install one (assimp/gmsh/freecadcmd) or export {step_path.name} to STL/OBJ in CAD first."
    )


def _convert_step_to_usd_via_isaac(
    step_path: Path,
    usd_path: Path,
    scale: float,
    *,
    sim_utils: object,
    mesh_converter_type: type,
    mesh_converter_cfg_type: type,
    schemas_cfg_module: object,
) -> Path:
    converter_cfg = mesh_converter_cfg_type(
        asset_path=str(step_path),
        usd_dir=str(usd_path.parent),
        usd_file_name=usd_path.name,
        force_usd_conversion=True,
        make_instanceable=False,
        scale=(float(scale), float(scale), float(scale)),
        mass_props=sim_utils.MassPropertiesCfg(mass=1200.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=True,
            kinematic_enabled=True,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=ISAAC_MIN_CONTACT_OFFSET_M,
            rest_offset=0.0,
        ),
        mesh_collision_props=schemas_cfg_module.ConvexDecompositionPropertiesCfg(),
    )
    converter = mesh_converter_type(converter_cfg)
    converted = Path(converter.usd_path).resolve()
    if not converted.exists():
        raise RuntimeError(f"MeshConverter did not produce an output for '{step_path}'.")
    return converted


def _bounds_from_usd(usd_path: Path) -> tuple[float, float, float]:
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    lower = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    upper = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    found = False
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        try:
            mesh = UsdGeom.Mesh(prim)
            extent = mesh.GetExtentAttr().Get()
            if not extent or len(extent) != 2:
                continue
            local_min = np.asarray(extent[0], dtype=np.float32)
            local_max = np.asarray(extent[1], dtype=np.float32)
            corners = np.array(
                [
                    [local_min[0], local_min[1], local_min[2]],
                    [local_min[0], local_min[1], local_max[2]],
                    [local_min[0], local_max[1], local_min[2]],
                    [local_min[0], local_max[1], local_max[2]],
                    [local_max[0], local_min[1], local_min[2]],
                    [local_max[0], local_min[1], local_max[2]],
                    [local_max[0], local_max[1], local_min[2]],
                    [local_max[0], local_max[1], local_max[2]],
                ],
                dtype=np.float32,
            )
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for corner in corners:
                point = transform.Transform(Gf.Vec3f(*corner))
                lower = np.minimum(lower, np.array([point[0], point[1], point[2]], dtype=np.float32))
                upper = np.maximum(upper, np.array([point[0], point[1], point[2]], dtype=np.float32))
            found = True
        except Exception:
            continue
    if not found:
        raise RuntimeError(f"Could not infer bounds from generated USD '{usd_path}'.")
    return (float(0.5 * (lower[0] + upper[0])), float(0.5 * (lower[1] + upper[1])), float(lower[2]))


def _load_mesh_to_stl(mesh_path: Path, out_dir: Path, scale: float) -> Path:
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Mesh file not found: {mesh_path}. "
            "When running through docker_env.sh, pass a path under the mounted repo (e.g. "
            "'/workspace/add_isaac/slot_profile/....')."
        )
    suffix = mesh_path.suffix.lower()
    if suffix == ".stl":
        return mesh_path
    if suffix not in {".step", ".stp", ".obj"}:
        raise ValueError(f"--mesh supports .stl/.obj/.step/.stp (got {mesh_path.suffix}).")
    if suffix in {".step", ".stp"}:
        out_path = out_dir / f"{mesh_path.stem}.stl"
        try:
            mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        except Exception:
            _convert_step_to_stl(mesh_path, out_path)
            try:
                mesh = trimesh.load(str(out_path), force="mesh", process=False)
            except Exception as exc:
                raise RuntimeError(
                    "STEP/STP import via fallback converter succeeded, but resulting STL could not be loaded. "
                    "Export the mesh to STL/OBJ manually and retry."
                ) from exc
        else:
            if isinstance(mesh, trimesh.Scene):
                if not mesh.geometry:
                    raise ValueError(f"No geometry found in {mesh_path}.")
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError(f"Unsupported mesh type from {mesh_path}.")
            if scale != 1.0:
                mesh.apply_scale(float(scale))
            mesh.export(out_path)
            return out_path
        # If we reached here from converter fallback, continue to validate and scale/export.
        out_path_loaded = trimesh.load(str(out_path), force="mesh", process=False)
        if isinstance(out_path_loaded, trimesh.Scene):
            if not out_path_loaded.geometry:
                raise ValueError(f"No geometry found in converted mesh {out_path}.")
            out_path_loaded = trimesh.util.concatenate(tuple(out_path_loaded.geometry.values()))
        if not isinstance(out_path_loaded, trimesh.Trimesh):
            raise TypeError(f"Unsupported mesh type from converted {out_path}.")
        if scale != 1.0:
            out_path_loaded.apply_scale(float(scale))
            out_path_loaded.export(out_path)
        return out_path

    try:
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {mesh_path}. If STEP import fails here, export STEP to STL first in your CAD tool."
        ) from exc
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"No geometry found in {mesh_path}.")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {mesh_path}.")
    if scale != 1.0:
        mesh.apply_scale(float(scale))
    out_path = out_dir / f"{mesh_path.stem}_as_floor.stl"
    mesh.export(out_path)
    return out_path


def _mesh_xy_center_bottom(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    return 0.5 * (mins[0] + maxs[0]), 0.5 * (mins[1] + maxs[1]), float(mins[2])


def _rgb_from_output(output_rgb) -> np.ndarray:
    frame = np.asarray(output_rgb[0, ..., :3].detach().cpu().numpy())
    if frame.dtype == np.uint8:
        return frame
    frame = frame.astype(float)
    if frame.max() <= 1.5:
        frame *= 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def main() -> None:
    if args.frames <= 0:
        raise ValueError("--frames must be > 0")
    if not args.robot_usd.exists():
        raise FileNotFoundError(f"Robot usd not found: {args.robot_usd}")
    if args.headless:
        os.environ.setdefault("HEADLESS", "1")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Omniverse/Isaac imports must happen after SimulationApp creation.
    import isaaclab.sim as sim_utils  # noqa: E402
    import omni.usd  # noqa: E402
    from isaaclab.scene import InteractiveScene  # noqa: E402
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
    from isaaclab.sim.schemas import schemas_cfg  # noqa: E402

    from grasp_planning.envs.fr3_part_env import (  # noqa: E402
        make_fr3_part_scene_cfg,
        make_robot_overview_camera_cfg,
    )
    from grasp_planning.isaac_visual_scene import make_visual_servo_render_cfg  # noqa: E402

    with tempfile.TemporaryDirectory(prefix="slot_floor_isaac_", dir="/tmp") as tmp_dir:
        tmp_root = Path(tmp_dir)
        mesh_path = args.mesh.expanduser().resolve()
        mesh_stl: Path | None = None
        part_usd: Path
        try:
            mesh_stl = _load_mesh_to_stl(mesh_path, tmp_root, args.stl_scale)
        except RuntimeError:
            if mesh_path.suffix.lower() in {".step", ".stp"}:
                part_usd = _convert_step_to_usd_via_isaac(
                    mesh_path,
                    tmp_root / "slot_floor.usd",
                    args.stl_scale,
                    sim_utils=sim_utils,
                    mesh_converter_type=MeshConverter,
                    mesh_converter_cfg_type=MeshConverterCfg,
                    schemas_cfg_module=schemas_cfg,
                )
            else:
                raise
        else:
            loaded = trimesh.load(str(mesh_stl), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            x_center, y_center, z_min = _mesh_xy_center_bottom(loaded)

            converter_cfg = MeshConverterCfg(
                asset_path=str(mesh_stl),
                usd_dir=str(tmp_root),
                usd_file_name="slot_floor.usd",
                force_usd_conversion=True,
                make_instanceable=False,
                scale=(1.0, 1.0, 1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=1200.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=True,
                    kinematic_enabled=True,
                    max_depenetration_velocity=5.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=ISAAC_MIN_CONTACT_OFFSET_M,
                    rest_offset=0.0,
                ),
                mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
            )
            part_usd = Path(MeshConverter(converter_cfg).usd_path).resolve()

        if mesh_stl is None:
            x_center, y_center, z_min = _bounds_from_usd(part_usd)

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / 60.0,
                device=args.device,
                render=make_visual_servo_render_cfg(),
            )
        )
        sim._app_control_on_stop_handle = None
        sim._disable_app_control_on_stop_handle = True

        scene_cfg = make_fr3_part_scene_cfg(
            fr3_asset_path=str(args.robot_usd.expanduser().resolve()),
            part_usd_path=str(part_usd),
            part_position=(-x_center, -y_center, -z_min),
            part_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            part_density_kg_m3=None,
            part_mass_kg=1200.0,
        )
        # Keep the slot profile fixed as static floor geometry.
        scene_cfg.part.spawn.rigid_props.disable_gravity = True
        scene_cfg.part.spawn.rigid_props.kinematic_enabled = True
        scene_cfg.num_envs = 1
        scene_cfg.overview_camera = make_robot_overview_camera_cfg(
            width=args.width,
            height=args.height,
        )
        scene_cfg.overview_camera.update_period = 0.0

        floor_scene = InteractiveScene(scene_cfg)
        sim.set_camera_view([1.45, -1.2, 1.0], [0.2, 0.0, 0.25])
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            simulation_app.update()

        sim.reset()
        floor_scene.reset()

        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/GroundPlane").IsValid():
            stage.RemovePrim("/World/GroundPlane")

        # Make slot profile look like aluminum.
        aluminum = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.92, 0.93, 0.96),
            roughness=0.14,
            metallic=0.95,
        )
        aluminum_look = "/World/Looks/slot_floor_aluminum"
        aluminum.func(aluminum_look, aluminum)
        bound_parts: list[str] = []
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            prim_name = prim.GetName()
            if "/World/envs/" not in prim_path:
                continue
            if prim_name in {"Part", "Part_0"} or prim_name.startswith("Part_"):
                sim_utils.bind_visual_material(
                    prim_path,
                    aluminum_look,
                    stage=stage,
                    stronger_than_descendants=True,
                )
                bound_parts.append(prim_path)
            elif prim_path.startswith("/World/envs/env_0/Part"):
                # Backward-compatible broad path prefix match for older authored USDs.
                sim_utils.bind_visual_material(
                    prim_path,
                    aluminum_look,
                    stage=stage,
                    stronger_than_descendants=True,
                )
                if prim_path not in bound_parts:
                    bound_parts.append(prim_path)
        if not bound_parts:
            raise RuntimeError(
                "Could not locate target part prims to apply aluminum material. "
                "Nothing was bound under /World/envs/... to names starting with Part."
            )
        print(f"[INFO] Applied aluminum material to: {', '.join(bound_parts)}")

        output_dir = args.output_dir.expanduser().resolve()
        _ensure_image_path(output_dir / "ok.txt")
        camera = floor_scene["overview_camera"]
        dt = sim.get_physics_dt()
        for frame_index in range(args.frames):
            floor_scene.write_data_to_sim()
            sim.step()
            floor_scene.update(dt)
            output_rgb = camera.data.output.get("rgb")
            if output_rgb is None:
                raise RuntimeError("Overview camera RGB output is missing.")
            image = _rgb_from_output(output_rgb)
            Image.fromarray(image).save(output_dir / f"{args.output_prefix}_{frame_index:04d}.png")
        print(f"[OK] Rendered {args.frames} frames -> {output_dir}")
        print(f"[OK] Floor USD: {part_usd}")
        simulation_app.close()


if __name__ == "__main__":
    main()
