#!/usr/bin/env python3
"""Convert the KUKA+PDZ URDF to USD and author the D405 depth camera."""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path

from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = _parse_args()
simulation_app = SimulationApp({"headless": ARGS.headless})

import omni.kit.app
import omni.kit.commands
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


ROBOT_PRIM = "/kuka_iiwa7_pdz_gripper"
CAMERA_FRAME = (
    f"{ROBOT_PRIM}/link7/pdz_gripper_base_link/camera_bottom_screw_frame/"
    "camera_link/camera_depth_frame/camera_depth_optical_frame"
)
CAMERA_OPTICAL_FRAME_NAME = "camera_depth_optical_frame"
LEFT_FINGER_JOINT = f"{ROBOT_PRIM}/joints/pdz_gripper_left_finger_joint"
KUKA_JOINT4 = f"{ROBOT_PRIM}/joints/joint4"


def _import_urdf(input_path: Path, output_path: Path) -> None:
    print(f"Importing URDF: {input_path}", flush=True)
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Isaac Sim failed to create a URDF import configuration")
    config.set_fix_base(True)
    config.set_make_default_prim(True)
    if hasattr(config, "set_make_instanceable"):
        config.set_make_instanceable(False)
    config.set_create_physics_scene(True)
    config.set_self_collision(False)
    status, _prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(input_path),
        import_config=config,
        dest_path=str(output_path),
    )
    if not status:
        raise RuntimeError(f"Isaac Sim failed to import {input_path}")
    print(f"Imported USD: {output_path}", flush=True)


def _remove_previous_import_layers(output_path: Path) -> None:
    """Remove only layers generated for this USD before a fresh URDF import.

    The URDF importer writes a root layer plus same-stem configuration layers.
    Leaving prior layers in place can compose old link paths into a new import.
    """

    generated = [output_path]
    config_dir = output_path.parent / "configuration"
    generated.extend(config_dir.glob(f"{output_path.stem}_*.usd"))
    for path in generated:
        if path.is_file():
            print(f"Removing stale generated layer: {path}", flush=True)
            path.unlink()


def _resolve_camera_frame(stage: Usd.Stage) -> str:
    """Resolve the optical frame from the newly imported link hierarchy."""

    if stage.GetPrimAtPath(CAMERA_FRAME).IsValid():
        return CAMERA_FRAME
    matches = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetName() == CAMERA_OPTICAL_FRAME_NAME
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"Imported USD is missing optical frame '{CAMERA_OPTICAL_FRAME_NAME}'.")
    raise RuntimeError(f"Imported USD has ambiguous optical frames: {matches}")


def _author_camera_and_drive(output_path: Path) -> None:
    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"Could not open generated USD {output_path}")
    camera_frame = _resolve_camera_frame(stage)
    camera_prim = f"{camera_frame}/Camera"
    print(f"Authoring camera: {camera_prim}", flush=True)

    camera = UsdGeom.Camera.Define(stage, camera_prim)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    # ROS optical frames look along +Z with +X right and +Y down. USD cameras
    # look along -Z with +Y up, so Rx(pi) aligns the optical axes exactly.
    xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
    camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.GetFocalLengthAttr().Set(1.93)
    camera.GetHorizontalApertureAttr().Set(2.0 * 1.93 * math.tan(math.radians(87.0 / 2.0)))
    camera.GetVerticalApertureAttr().Set(2.0 * 1.93 * math.tan(math.radians(58.0 / 2.0)))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))
    camera.GetPrim().CreateAttribute("pdz:nominalResolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(848, 480))
    camera.GetPrim().CreateAttribute("pdz:rosOpticalFrame", Sdf.ValueTypeNames.String).Set(
        "camera_depth_optical_frame"
    )

    # The established KUKA Isaac backend represents A4 as a +Y USD joint and
    # converts the physical MoveIt coordinate at the backend boundary.  The
    # generic URDF importer instead encodes the URDF's -Y axis by rotating both
    # joint frames 180 degrees about X.  Leaving that importer rotation in
    # place would apply the existing A4 sign conversion twice and put the end
    # effector far from every planned grasp.  Normalize this generated asset to
    # the same joint-coordinate convention as kuka_iiwa7_y_gripper.usda.
    kuka_joint4 = stage.GetPrimAtPath(KUKA_JOINT4)
    if not kuka_joint4.IsValid():
        raise RuntimeError(f"Imported USD is missing KUKA joint {KUKA_JOINT4}")
    identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    kuka_joint4.GetAttribute("physics:localRot0").Set(identity)
    kuka_joint4.GetAttribute("physics:localRot1").Set(identity)

    joint = stage.GetPrimAtPath(LEFT_FINGER_JOINT)
    if not joint.IsValid():
        raise RuntimeError(f"Imported USD is missing finger joint {LEFT_FINGER_JOINT}")
    drive = UsdPhysics.DriveAPI.Get(joint, "linear")
    drive.GetTypeAttr().Set("acceleration")
    drive.GetStiffnessAttr().Set(2500.0)
    drive.GetDampingAttr().Set(100.0)
    drive.GetMaxForceAttr().Set(100.0)

    stage.GetRootLayer().Save()


def main() -> None:
    input_path = ARGS.input.expanduser().resolve()
    output_path = ARGS.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_previous_import_layers(output_path)
    _import_urdf(input_path, output_path)
    _author_camera_and_drive(output_path)
    print(output_path, flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
