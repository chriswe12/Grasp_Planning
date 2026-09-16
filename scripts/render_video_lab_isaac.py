#!/usr/bin/env python3
"""Capture actual Isaac Sim RTX RGB/depth for each video-derived scene variant.

Use Isaac Sim 5.1's python.sh, outside a sandbox that hides GPU device access.
Source USD and Blender files are never saved or overwritten.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/video_lab_isaac")
parser.add_argument("--variants", nargs="+", default=["video_lab", "video_lab_weathered", "video_lab_pencil"])
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
parser.add_argument("--warmup-frames", type=int, default=4)
parser.add_argument("--subframes", type=int, default=16)
parser.add_argument("--props", choices=["both", "with", "without"], default="both")
parser.add_argument("--lighting", choices=["authored", "rl"], default="rl")
parser.add_argument(
    "--prepared-scene", type=Path, help="Render all cameras in an already composed USD, preserving its lights."
)
args = parser.parse_args()
if min(args.width, args.height, args.warmup_frames, args.subframes) < 1:
    parser.error("Sizes and frame counts must be positive.")
args.output_dir = args.output_dir.resolve()
args.output_dir.mkdir(parents=True, exist_ok=True)

faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
print("RENDER_START", flush=True)
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "width": args.width,
        "height": args.height,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
        "sync_loads": True,
        "disable_viewport_updates": True,
        "create_new_stage": False,
        "extra_args": [
            "--/app/file/ignoreUnsavedOnExit=true",
            "--/app/asyncRendering=false",
            "--/omni/replicator/asyncRendering=false",
        ],
    }
)

print("APP_INITIALIZED", flush=True)
import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdLux

sys.path.insert(0, str(ROOT))
from grasp_planning.rl.video_lab_scene import repair_exported_curve_widths

print("RENDER_IMPORTS_READY", flush=True)


def configure_renderer():
    print("CONFIGURE_RENDERER", flush=True)
    rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
    # Match the relevant current RL realtime settings, with fixed authored lights.
    settings = {
        "/rtx/translucency/enabled": False,
        "/rtx/reflections/enabled": False,
        "/rtx/indirectDiffuse/enabled": False,
        "/rtx-transient/dldenoiser/enabled": True,
        "/rtx/directLighting/enabled": True,
        "/rtx/directLighting/sampledLighting/samplesPerPixel": 4,
        "/rtx/shadows/enabled": True,
        "/rtx/ambientOcclusion/enabled": False,
        "/omni/replicator/captureOnPlay": False,
        "/omni/replicator/asyncRendering": False,
        "/app/asyncRendering": False,
    }
    for k, v in settings.items():
        carb.settings.get_settings().set(k, v)
    return settings


def setup_rl_lighting(stage):
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Lighting")).MakeInvisible()
    dome = UsdLux.DomeLight.Define(stage, "/World/RenderDome")
    dome.CreateIntensityAttr(450.0)
    dome.CreateColorAttr(Gf.Vec3f(0.76, 0.80, 0.86))
    key = UsdLux.DistantLight.Define(stage, "/World/RenderKey")
    key.CreateIntensityAttr(1200.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.82))
    key.CreateAngleAttr(8.0)
    key.AddOrientOp().Set(
        Gf.Quatf(0.9384303340467282, Gf.Vec3f(0.1620103355210184, -0.2851488516085399, -0.10858771455208008))
    )


def capture(stage, camera, output):
    print("CREATE_RENDER_PRODUCT", camera.GetPath(), flush=True)
    product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
    rgb.attach(product)
    depth.attach(product)
    print("CAPTURE_WARMUP", output, flush=True)
    for _ in range(args.warmup_frames):
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0, pause_timeline=True)
    pixels = np.asarray(rgb.get_data())
    distances = np.asarray(depth.get_data())
    print("FRAME_DATA", pixels.shape, pixels.dtype, distances.shape, flush=True)
    if pixels.ndim != 3 or pixels.shape[:2] != (args.height, args.width) or np.ptp(pixels[:, :, :3]) < 8:
        if pixels.ndim == 3 and pixels.shape[2] >= 3:
            Image.fromarray(pixels[:, :, :3]).save(output.with_name(output.name + "_raw_debug").with_suffix(".png"))
        raise RuntimeError(f"Unusable Isaac RGB frame: shape={pixels.shape}, range={np.ptp(pixels)}")
    if not np.isfinite(distances).any():
        raise RuntimeError("No finite camera depth")
    Image.fromarray(pixels[:, :, :3]).save(output.with_suffix(".png"))
    np.save(str(output) + "_depth_m.npy", distances)
    cam = UsdGeom.Camera(camera)
    record = {
        "rgb": str(output.with_suffix(".png").relative_to(args.output_dir)),
        "depth": str(Path(str(output) + "_depth_m.npy").relative_to(args.output_dir)),
        "camera_prim": str(camera.GetPath()),
        "resolution": [args.width, args.height],
        "focal_length": cam.GetFocalLengthAttr().Get(),
        "horizontal_aperture": cam.GetHorizontalApertureAttr().Get(),
        "vertical_aperture": cam.GetVerticalApertureAttr().Get(),
        "camera_world_matrix": list(map(list, UsdGeom.XformCache().GetLocalToWorldTransform(camera))),
        "stage_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "rgb_mean": pixels[:, :, :3].mean(axis=(0, 1)).tolist(),
        "finite_depth_fraction": float(np.isfinite(distances).mean()),
    }
    rgb.detach(product)
    depth.detach(product)
    product.destroy()
    print("CAPTURED", output.with_suffix(".png"), flush=True)
    return record


def main():
    settings = configure_renderer()
    report = {
        "renderer": "Isaac Sim 5.1 RTX RaytracedLighting",
        "antialiasing": "DLAA",
        "lighting": args.lighting,
        "render_settings": settings,
        "physics_stepped": False,
        "note": "Inspection cameras, not calibrated robot-mounted sensors. No RGB/depth augmentation. Source assets unchanged.",
        "captures": [],
        "curve_repairs": {},
    }
    if args.prepared_scene:
        source = args.prepared_scene.resolve()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        if not omni.usd.get_context().open_stage(str(source)):
            raise RuntimeError(f"Failed opening {source}")
        for _ in range(30):
            app.update()
        stage = omni.usd.get_context().get_stage()
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        assert UsdGeom.GetStageUpAxis(stage) == "Z"
        report["lighting"] = "preserved from prepared scene"
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdGeom.Camera) and str(prim.GetPath()).startswith("/World/"):
                report["captures"].append(capture(stage, prim, args.output_dir / prim.GetName()))
        assert report["captures"], "No cameras in prepared scene"
        assert before == hashlib.sha256(source.read_bytes()).hexdigest()
        report.update(status="complete", sources_unchanged=True, prepared_scene=str(source))
        (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("ISAAC_VIDEO_LAB_RENDERS_COMPLETE", len(report["captures"]), flush=True)
        return
    for variant in args.variants:
        folder = ROOT / "assets/scenes" / variant
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        source = folder / "preview.usda"
        print("HASH_SOURCE", folder, flush=True)
        hashes = {
            str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*")
            if p.is_file()
        }
        target = args.output_dir / variant
        target.mkdir(exist_ok=True)
        modes = [True, False] if args.props == "both" else [args.props == "with"]
        for with_props in modes:
            composition = target / ("isaac_preview.usda" if with_props else "isaac_empty.usda")
            composed = Usd.Stage.CreateNew(str(composition))
            composed.GetRootLayer().subLayerPaths = [os.path.relpath(source, target)]
            # Stage metadata is read from the root/session layer, not sublayers.
            original = Usd.Stage.Open(str(source))
            UsdGeom.SetStageMetersPerUnit(composed, UsdGeom.GetStageMetersPerUnit(original))
            UsdGeom.SetStageUpAxis(composed, UsdGeom.GetStageUpAxis(original))
            composed.SetDefaultPrim(composed.GetPrimAtPath("/World"))
            del original
            report["curve_repairs"][variant] = repair_exported_curve_widths(composed, "/World/Lab")
            if args.lighting == "rl":
                setup_rl_lighting(composed)
            # Author before opening in Kit: live interpolation/visibility changes
            # can remain stale in Hydra/Fabric during paused Replicator captures.
            if not with_props:
                composed.GetPrimAtPath("/World/Props").SetActive(False)
            composed.GetRootLayer().Save()
            del composed
            print("OPEN_STAGE", composition, flush=True)
            if not omni.usd.get_context().open_stage(str(composition)):
                raise RuntimeError(f"Failed opening {composition}")
            print("STAGE_OPENED", flush=True)
            for _ in range(30):
                app.update()
            stage = omni.usd.get_context().get_stage()
            assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
            assert UsdGeom.GetStageUpAxis(stage) == "Z"
            cameras = {}
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Camera):
                    for name, needle in [("overview", "Camera_Overview"), ("detail", "Camera_Table_Detail")]:
                        if needle in str(prim.GetPath()):
                            cameras[name] = prim
            if len(cameras) != 2:
                raise RuntimeError(f"Missing exported cameras: {cameras}")
            for name, camera in cameras.items():
                record = capture(stage, camera, target / (("with_props_" if with_props else "empty_") + name))
                record.update(variant=variant, with_props=with_props)
                report["captures"].append(record)
                (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
        assert hashes == {
            str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*")
            if p.is_file()
        }
    report["status"] = "complete"
    report["sources_unchanged"] = True
    (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ISAAC_VIDEO_LAB_RENDERS_COMPLETE", len(report["captures"]), flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
    app.close()
