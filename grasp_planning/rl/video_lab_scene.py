"""Opt-in video-derived room asset and USD appearance randomization.

No simulator imports at module load. Call USD helpers after Isaac AppLauncher.
The existing RL scene is not replaced automatically. Robot/tasks stay caller-owned.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

DEFAULT_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets/scenes/video_lab"


def training_asset_digest(asset_dir) -> str:
    """Hash referenced USD/texture inputs, excluding editable sources and preview artifacts."""
    folder = Path(asset_dir).resolve()
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file() and (
            path.suffix.lower() in {".usd", ".usda", ".usdc"}
            or "textures" in path.relative_to(folder).parts
            or path.name == "manifest.json"
        ):
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def sample_prop_layout(seed: int, manifest: dict) -> list[dict]:
    """Non-overlapping footprints, bounded by table and excluded task/mount discs.

    May return fewer props than requested if free space cannot accommodate them.
    Layout is in the lab's local frame, with all prop bottoms at z=0.4 mm.
    """
    rng = random.Random(seed)
    cfg = manifest["randomization"]
    t = manifest["dimensions"]
    chosen = rng.sample(manifest["props"], min(rng.randint(*cfg["prop_count"]), len(manifest["props"])))
    reserved = cfg["reserved_robot_mounts_xy_radius"] + [cfg["reserved_task_center_xy_radius"]]
    placed = []
    for prop in chosen:
        radius = prop["footprint_radius_m"]
        w = t["width"]
        d = t["depth"]
        if 2 * (radius + 0.025) >= min(w, d):
            continue
        for _ in range(300):
            x = rng.uniform(-w / 2 + radius + 0.025, w / 2 - radius - 0.025)
            y = rng.uniform(-d / 2 + radius + 0.025, d / 2 - radius - 0.025)
            if any(
                math.hypot(x - a, y - b) < radius + r + 0.02
                for a, b, r in reserved + [(p["position"][0], p["position"][1], p["radius"]) for p in placed]
            ):
                continue
            placed.append(
                dict(
                    name=prop["name"],
                    asset=prop["asset"],
                    position=[x, y, 0.0004],
                    yaw_rad=rng.uniform(-math.pi, math.pi),
                    radius=radius,
                )
            )
            break
    return placed


def add_environment(stage, prim_path: str, asset_dir=DEFAULT_ASSET_DIR):
    """Reference static room under a unique, non-instanceable per-env root.

    Table z=0, floor z=-0.74 nominal. Place the robot relative to this frame.
    Do not also spawn the former tabletop/ground at z=0. Lights are separate.
    """
    from pxr import UsdGeom

    asset = Path(asset_dir).resolve() / "environment.usdc"
    if not asset.is_file():
        raise FileNotFoundError(asset)
    if stage.GetPrimAtPath(prim_path):
        raise ValueError(f"Prim already exists: {prim_path}")
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    root.GetReferences().AddReference(str(asset))
    root.SetInstanceable(False)
    repair_exported_curve_widths(stage, prim_path)
    return root


def repair_exported_curve_widths(stage, prim_path: str) -> list[dict]:
    """Restore bevel diameters lost by Blender's legacy-curve USD export.

    Only the three known cable/slot families are affected. Author overrides in
    the caller's edit layer, preserving every source asset and collision mesh.
    Radii match build_video_lab.py. Also suppress identical nested duplicates
    emitted by that exporter. Safe to repeat on an already corrected stage.
    """
    from pxr import Gf, Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root:
        raise ValueError(f"Missing environment prim: {prim_path}")
    changes = []
    for prim in list(Usd.PrimRange(root)):
        if not prim.IsA(UsdGeom.BasisCurves):
            continue
        name = prim.GetName()
        diameter = next(
            (
                d
                for prefix, d in (("Grommet_slot", 0.0013), ("Wall_power_lead", 0.006), ("Loose_floor_cable", 0.006))
                if name.startswith(prefix)
            ),
            None,
        )
        if diameter is None:
            continue
        curve = UsdGeom.BasisCurves(prim)
        points = curve.GetPointsAttr().Get()
        parent = prim.GetParent()
        original = parent.GetParent().GetChild(name)
        if parent.GetName() == name + "_0" and original.IsA(UsdGeom.BasisCurves):
            other = UsdGeom.BasisCurves(original)
            cache = UsdGeom.XformCache()
            if other.GetPointsAttr().Get() == points and cache.GetLocalToWorldTransform(
                original
            ) == cache.GetLocalToWorldTransform(prim):
                prim.SetActive(False)
                changes.append(dict(prim=str(prim.GetPath()), action="disable_duplicate"))
                continue
        previous = list(curve.GetWidthsAttr().Get() or [])
        curve.CreateWidthsAttr([diameter] * curve.ComputeVaryingDataSize(Usd.TimeCode.Default()))
        curve.SetWidthsInterpolation(UsdGeom.Tokens.varying)
        radius = diameter / 2
        curve.CreateExtentAttr(
            [
                Gf.Vec3f(*[min(p[i] for p in points) - radius for i in range(3)]),
                Gf.Vec3f(*[max(p[i] for p in points) + radius for i in range(3)]),
            ]
        )
        changes.append(
            dict(
                prim=str(prim.GetPath()), action="restore_bevel_diameter", previous_widths=previous, diameter_m=diameter
            )
        )
    return changes


def randomize_environment(stage, prim_path: str, seed: int, asset_dir=DEFAULT_ASSET_DIR):
    """Author local appearance overrides; never alter geometry, physics or source files.

    Suitable at scene setup/reset, not the per-frame fast path. Light and camera
    randomization remain owned by the caller's existing Isaac RL profile.
    Repeating a seed overwrites prior values without accumulating tint.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    manifest = json.loads((Path(asset_dir) / "manifest.json").read_text())
    cfg = manifest["randomization"]
    rng = random.Random(seed)
    gain = rng.uniform(*cfg["table_color_gain"])
    rough = rng.uniform(*cfg["table_roughness"])
    wg = rng.uniform(*cfg["wall_color_gain"])
    wear = rng.random() < cfg["wear_visibility_probability"]
    root = stage.GetPrimAtPath(prim_path)
    if not root:
        raise ValueError(f"Missing environment prim: {prim_path}")
    changes = 0
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            if "/Table_Laminate/" in path:
                if shader.GetIdAttr().Get() == "UsdUVTexture" and prim.GetName() == "Laminate_Albedo":
                    shader.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(gain, gain, gain, 1))
                    changes += 1
                if shader.GetIdAttr().Get() == "UsdPreviewSurface":
                    inp = shader.GetInput("roughness")
                    inp.DisconnectSource()
                    inp.Set(rough)
                    changes += 1
            if "/Warm_White_Partition/" in path and shader.GetIdAttr().Get() == "UsdPreviewSurface":
                inp = shader.GetInput("diffuseColor")
                attr = prim.GetAttribute("lab:nominalDiffuseColor")
                if not attr:
                    attr = prim.CreateAttribute("lab:nominalDiffuseColor", Sdf.ValueTypeNames.Color3f, custom=True)
                    attr.Set(inp.Get())
                inp.Set(Gf.Vec3f(*[min(1, float(c) * wg) for c in attr.Get()]))
        if prim.GetName().startswith(("Tape_Remnant", "Handling_Scuff", "Faded_fixture_outline")):
            img = UsdGeom.Imageable(prim)
            if img:
                img.CreateVisibilityAttr().Set("inherited" if wear else "invisible")
    if changes != 2:
        raise RuntimeError(f"Expected tabletop color and roughness inputs, found {changes}")
    result = dict(seed=seed, table_color_gain=gain, table_roughness=rough, wall_color_gain=wg, wear_visible=wear)
    root.CreateAttribute("lab:appearanceSample", Sdf.ValueTypeNames.String, custom=True).Set(
        json.dumps(result, sort_keys=True)
    )
    return result


def add_mock_props(stage, prim_path: str, seed: int, asset_dir=DEFAULT_ASSET_DIR):
    """Spawn optional dynamic USD props once, before simulation starts.

    For episode resets use the simulator's RigidObject state APIs; do not rebuild
    physics prims during stepping. Returned poses are local to prim_path.
    """
    from pxr import Gf, UsdGeom

    folder = Path(asset_dir).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if stage.GetPrimAtPath(prim_path):
        raise ValueError(f"Prim already exists: {prim_path}")
    UsdGeom.Xform.Define(stage, prim_path)
    layout = sample_prop_layout(seed, manifest)
    for pose in layout:
        xf = UsdGeom.Xform.Define(stage, prim_path + "/" + pose["name"])
        xf.GetPrim().GetReferences().AddReference(str(folder / pose["asset"]))
        xf.AddTranslateOp().Set(Gf.Vec3d(*pose["position"]))
        xf.AddRotateZOp().Set(math.degrees(pose["yaw_rad"]))
    return layout
