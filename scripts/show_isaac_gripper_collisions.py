"""Open the KUKA iiwa7 gripper USD and visualize the gripper collision models."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = REPO_ROOT / "assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--usd",
    type=Path,
    default=DEFAULT_USD,
    help="Robot USD to open. Defaults to the generated KUKA iiwa7 + Y-gripper USD.",
)
parser.add_argument(
    "--show-arm-collisions",
    action="store_true",
    help="Also show arm/base authored USD collision input meshes.",
)
parser.add_argument(
    "--show-authored-collision-input",
    action="store_true",
    help=(
        "Show the active USD PhysicsCollisionAPI mesh input in addition to the planner/FCL hull overlay."
    ),
)
parser.add_argument(
    "--keep-gripper-visuals",
    action="store_true",
    help="Keep gripper visual meshes visible. By default they are hidden so collision meshes are obvious.",
)
parser.add_argument(
    "--play",
    action="store_true",
    help="Start simulation playback so PhysX runtime debug draw becomes visible.",
)
parser.add_argument(
    "--run-seconds",
    type=float,
    default=0.0,
    help="Keep Isaac open for this many seconds. Use 0 to keep it open until closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

from grasp_planning.grasping.collision import _load_kuka_y_gripper_mesh  # noqa: E402


GRIPPER_PATH_MARKERS = (
    "/gripper_base_link/",
    "/left_finger_link/",
    "/right_finger_link/",
)


def _wait_for_stage_load(timeout_s: float = 30.0) -> None:
    context = omni.usd.get_context()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        stage = context.get_stage()
        _, _, pending_files = context.get_stage_loading_status()
        if stage is not None and pending_files <= 0:
            for _ in range(3):
                simulation_app.update()
            return
        simulation_app.update()
    raise TimeoutError("Timed out waiting for USD stage to finish loading.")


def _set_display_style(prim, color: tuple[float, float, float], opacity: float) -> None:
    imageable = UsdGeom.Imageable(prim)
    imageable.MakeVisible()
    imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)

    primvars = UsdGeom.PrimvarsAPI(prim)
    color_pv = primvars.CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.constant,
    )
    color_pv.Set([Gf.Vec3f(*color)])

    opacity_pv = primvars.CreatePrimvar(
        "displayOpacity",
        Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.constant,
    )
    opacity_pv.Set([float(opacity)])


def _is_gripper_prim(path: str) -> bool:
    return any(marker in path for marker in GRIPPER_PATH_MARKERS)


def _define_mesh(parent_path: str, name: str, vertices: object, faces: object, color: tuple[float, float, float]) -> str:
    path = f"{parent_path}/{name}"
    mesh = UsdGeom.Mesh.Define(omni.usd.get_context().get_stage(), path)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.CreatePointsAttr().Set([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
    mesh.CreateFaceVertexCountsAttr().Set([3 for _ in faces])
    mesh.CreateFaceVertexIndicesAttr().Set([int(index) for face in faces for index in face])
    _set_display_style(mesh.GetPrim(), color, 0.55)
    return path


def _show_planner_hulls(stage) -> list[str]:
    specs = (
        ("base", "/kuka_iiwa7_y_gripper/gripper_base_link", "planner_base_convex_hull"),
        ("left_finger", "/kuka_iiwa7_y_gripper/left_finger_link", "planner_left_finger_component_hulls"),
        ("right_finger", "/kuka_iiwa7_y_gripper/right_finger_link", "planner_right_finger_component_hulls"),
    )
    paths: list[str] = []
    for key, parent_path, mesh_name in specs:
        if not stage.GetPrimAtPath(parent_path).IsValid():
            print(f"[WARN] Could not add planner hull under missing prim: {parent_path}")
            continue
        vertices, faces = _load_kuka_y_gripper_mesh(key)
        paths.append(_define_mesh(parent_path, mesh_name, vertices, faces, (1.0, 0.36, 0.0)))
    return paths


def _show_authored_collision_input_meshes(stage) -> int:
    shown = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        name = prim.GetName()
        path = str(prim.GetPath())
        is_gripper = _is_gripper_prim(path)
        is_collision = name.endswith("_collision_mesh")

        if is_collision and (is_gripper or args_cli.show_arm_collisions):
            if is_gripper:
                _set_display_style(prim, (1.0, 0.9, 0.0), 0.30)
            else:
                _set_display_style(prim, (0.1, 0.55, 1.0), 0.20)
            shown += 1
    return shown


def _hide_gripper_visuals(stage) -> int:
    hidden_visuals = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        name = prim.GetName()
        path = str(prim.GetPath())
        is_gripper = _is_gripper_prim(path)
        is_visual = name.endswith("_visual_mesh")
        if is_visual and is_gripper and not args_cli.keep_gripper_visuals:
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_visuals += 1

    return hidden_visuals


def _enable_physx_debug_draw() -> bool:
    try:
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.physx", True)
        manager.set_extension_enabled_immediate("omni.physx.ui", True)

        from omni.physx import get_physx_visualization_interface

        vis = get_physx_visualization_interface()
        vis.enable_visualization(True)
        vis.set_visualization_scale(1.0)
        for name in (
            "CollisionShapes",
            "CollisionEdges",
            "CollisionAxes",
            "CollisionAABBs",
            "ContactPoint",
            "ContactNormal",
        ):
            vis.set_visualization_parameter(name, True)
        return True
    except Exception as exc:  # pragma: no cover - Isaac extension availability varies.
        print(f"[WARN] Could not enable PhysX debug draw: {exc}")
        return False


def _open_physx_debug_window() -> bool:
    try:
        from omni.physxui.scripts.physxDebugView import PhysxDebugWindow

        # Keep a module-global reference so Kit does not immediately destroy it.
        globals()["_physx_debug_window"] = PhysxDebugWindow()
        globals()["_physx_debug_window"].visible = True
        globals()["_physx_debug_window"].focus()
        return True
    except Exception as exc:  # pragma: no cover - UI extension availability varies.
        print(f"[WARN] Could not open Physics Debug window: {exc}")
        return False


def main() -> None:
    usd_path = args_cli.usd.expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD not found: {usd_path}")

    context = omni.usd.get_context()
    if not context.open_stage(str(usd_path)):
        raise RuntimeError(f"Failed to open USD: {usd_path}")
    _wait_for_stage_load()

    stage = context.get_stage()
    hidden_visuals = _hide_gripper_visuals(stage)
    planner_hull_paths = _show_planner_hulls(stage)
    authored_collision_inputs = (
        _show_authored_collision_input_meshes(stage) if args_cli.show_authored_collision_input else 0
    )
    debug_draw = _enable_physx_debug_draw()
    debug_window = _open_physx_debug_window()

    print(f"[INFO] Opened: {usd_path}")
    print(f"[INFO] Added {len(planner_hull_paths)} planner/FCL collision hull mesh(es).")
    for path in planner_hull_paths:
        print(f"[INFO]   {path}")
    print(f"[INFO] Made {authored_collision_inputs} authored USD collision input mesh(es) visible.")
    print(f"[INFO] Hid {hidden_visuals} gripper visual mesh(es).")
    print(f"[INFO] PhysX debug draw enabled: {debug_draw}")
    print(f"[INFO] Physics Debug window opened: {debug_window}")
    print("[INFO] Orange meshes are the planner/FCL collision hulls used by benchmark collision checks.")
    print("[INFO] Yellow meshes, only with --show-authored-collision-input, are active USD collision inputs.")
    print("[INFO] Press Play, or rerun with --play, to see runtime PhysX collision/contact debug draw.")

    if args_cli.play:
        omni.timeline.get_timeline_interface().play()

    stop_at = None if args_cli.run_seconds <= 0.0 else time.monotonic() + args_cli.run_seconds
    while simulation_app.is_running():
        simulation_app.update()
        if stop_at is not None and time.monotonic() >= stop_at:
            break


if __name__ == "__main__":
    main()
