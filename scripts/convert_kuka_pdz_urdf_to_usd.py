#!/usr/bin/env python3
"""Convert the KUKA+PDZ URDF to USD and author the D405 depth camera."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from grasp_planning.isaac_visual_materials import (
    VISUAL_SERVO_CONTACT_PAD_COLOR,
    VISUAL_SERVO_CONTACT_PAD_ROUGHNESS,
    VISUAL_SERVO_FINGER_COLOR,
    VISUAL_SERVO_FINGER_ROUGHNESS,
)

ROBOT_PRIM = "/kuka_iiwa7_pdz_gripper"
CAMERA_FRAME = (
    f"{ROBOT_PRIM}/link7/pdz_gripper_base_link/camera_bottom_screw_frame/"
    "camera_link/camera_depth_frame/camera_depth_optical_frame"
)
CAMERA_OPTICAL_FRAME_NAME = "camera_depth_optical_frame"
LEFT_FINGER_JOINT = f"{ROBOT_PRIM}/joints/pdz_gripper_left_finger_joint"
KUKA_JOINT4 = f"{ROBOT_PRIM}/joints/joint4"


def _set_material_inputs(
    material: UsdShade.Material,
    *,
    color: tuple[float, float, float],
    roughness: float,
) -> None:
    """Override either UsdPreviewSurface or importer-authored OmniPBR inputs."""

    source = None
    # URDF importer materials are normally OmniPBR MDL materials.  Their
    # surface output is authored for the ``mdl`` render context, so the
    # universal/default query legitimately returns no shader.
    for render_context in ("mdl", "", "universal"):
        candidate = material.ComputeSurfaceSource(render_context)
        if candidate and candidate[0]:
            source = candidate
            break
    if source is None:
        outputs = [output.GetFullName() for output in material.GetOutputs()]
        raise RuntimeError(
            f"Material {material.GetPath()} has no supported surface shader; outputs={outputs}."
        )
    shader = UsdShade.Shader(source[0])
    color_written = False
    for name in ("diffuseColor", "diffuse_color_constant", "diffuse_color"):
        shader_input = shader.GetInput(name)
        if shader_input:
            shader_input.Set(Gf.Vec3f(*color))
            color_written = True
    if not color_written:
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        color_written = True
    roughness_written = False
    for name in ("roughness", "reflection_roughness_constant"):
        shader_input = shader.GetInput(name)
        if shader_input:
            shader_input.Set(float(roughness))
            roughness_written = True
    if not roughness_written:
        # OmniPBR importer materials omit inputs that retain their MDL default.
        # It is valid to author the standard input explicitly.
        shader.CreateInput(
            "reflection_roughness_constant", Sdf.ValueTypeNames.Float
        ).Set(float(roughness))
        roughness_written = True
    if not color_written or not roughness_written:
        raise RuntimeError(
            f"Material {material.GetPath()} does not expose supported color/roughness inputs; "
            f"shader={shader.GetPath()} inputs={[item.GetBaseName() for item in shader.GetInputs()]}"
        )


def _author_pdz_finger_materials(stage: Usd.Stage) -> None:
    """Make the editable importer base layer obey the black-white contract."""

    instance_roots = [
        prim
        for prim in stage.Traverse()
        if prim.IsInstance()
        and any(
            name in str(prim.GetPath()).lower()
            for name in (
                "pdz_gripper_left_finger_link",
                "pdz_gripper_right_finger_link",
            )
        )
    ]
    # The importer makes each visual scope instanceable even in its base layer.
    # Temporarily de-instance those four scopes so their authored material
    # shaders can be edited, then restore instanceability after saving values.
    for prim in instance_roots:
        prim.SetInstanceable(False)

    observed: dict[str, set[str]] = {"finger": set(), "pad": set()}
    try:
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            lowered = path.lower()
            if prim.GetTypeName() != "Mesh" or not any(
                name in lowered
                for name in (
                    "pdz_gripper_left_finger_link",
                    "pdz_gripper_right_finger_link",
                )
            ):
                continue
            is_pad = "tpu_pad" in lowered or "pad_8mm" in lowered
            material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material:
                raise RuntimeError(f"PDZ visual geometry has no bound material: {path}")
            key = "pad" if is_pad else "finger"
            observed[key].add(str(material.GetPath()))
            _set_material_inputs(
                material,
                color=VISUAL_SERVO_CONTACT_PAD_COLOR if is_pad else VISUAL_SERVO_FINGER_COLOR,
                roughness=(
                    VISUAL_SERVO_CONTACT_PAD_ROUGHNESS
                    if is_pad
                    else VISUAL_SERVO_FINGER_ROUGHNESS
                ),
            )
        if not observed["finger"] or not observed["pad"]:
            raise RuntimeError(
                "Imported PDZ USD did not expose both finger and TPU-pad visual materials: "
                f"{observed}"
            )
    finally:
        for prim in instance_roots:
            prim.SetInstanceable(True)
    print(
        "Authored PDZ visual materials: "
        f"finger={sorted(observed['finger'])} pad={sorted(observed['pad'])}",
        flush=True,
    )


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
    # Visual scopes in the composed robot stage are instance proxies even when
    # the top-level asset is not instanceable.  Author importer materials in
    # the editable base layer first, then let the root stage compose them.
    base_path = output_path.parent / "configuration" / f"{output_path.stem}_base.usd"
    base_stage = Usd.Stage.Open(str(base_path))
    if base_stage is None:
        raise RuntimeError(f"Could not open generated importer base layer {base_path}")
    _author_pdz_finger_materials(base_stage)
    base_stage.GetRootLayer().Save()

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
