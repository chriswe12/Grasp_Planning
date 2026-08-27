#!/usr/bin/env python3
"""Render close-grasp and workspace views through the mounted PDZ D405 frame."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-usd",
        type=Path,
        default=Path("assets/usd/kuka_iiwa7_pdz_gripper/kuka_iiwa7_pdz_gripper.usd"),
    )
    parser.add_argument(
        "--part-obj",
        type=Path,
        default=Path("assets/obj/fabrica/plumbers_block/0.obj"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pdz_gripper_camera_views"),
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = _parse_args()
simulation_app = SimulationApp(
    {
        "headless": bool(ARGS.headless),
        "width": 848,
        "height": 480,
        "anti_aliasing": 3,
    }
)

import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import trimesh
from isaacsim.core.prims import SingleArticulation
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_PRIM = "/World/Robot"
CAMERA_NAME = "Camera"
PDZ_BASE_NAME = "pdz_gripper_base_link"
PDZ_TCP_Z_M = 0.1355
KUKA_START_RAD = (0.0, 0.5, 0.0, 1.3962634015954636, 0.0, 1.1, 0.0)
OPEN_FINGER_M = 0.032


def _resolve(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _find_unique_prim(stage: Usd.Stage, name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one prim named '{name}', found {[str(p.GetPath()) for p in matches]}.")
    return matches[0]


def _set_material(stage: Usd.Stage, prim: Usd.Prim, name: str, color: tuple[float, float, float], roughness: float) -> None:
    material = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Materials/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _cube(
    stage: Usd.Stage,
    path: str,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    material_name: str,
    roughness: float = 0.55,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    _set_material(stage, cube.GetPrim(), material_name, color, roughness)
    return cube.GetPrim()


def _author_table(stage: Usd.Stage) -> None:
    _cube(
        stage,
        "/World/Worksurface",
        center=(0.55, 0.12, -0.018),
        size=(1.1, 0.9, 0.036),
        color=(0.32, 0.37, 0.39),
        material_name="TSlotAluminum",
        roughness=0.34,
    )
    for index, y in enumerate(np.linspace(-0.27, 0.51, 14)):
        _cube(
            stage,
            f"/World/TSlot_{index:02d}",
            center=(0.55, float(y), 0.001),
            size=(1.08, 0.006, 0.004),
            color=(0.055, 0.065, 0.07),
            material_name=f"TSlotGroove_{index:02d}",
            roughness=0.48,
        )


def _author_part(stage: Usd.Stage, obj_path: Path) -> tuple[UsdGeom.Xform, np.ndarray]:
    source = trimesh.load(obj_path, force="mesh", process=False)
    vertices = np.asarray(source.vertices, dtype=float) * 0.01
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    vertices -= 0.5 * (lower + upper)
    faces = np.asarray(source.faces, dtype=np.int64)

    part_xform = UsdGeom.Xform.Define(stage, "/World/PlumbersBlockPart0")
    mesh = UsdGeom.Mesh.Define(stage, "/World/PlumbersBlockPart0/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    _set_material(stage, mesh.GetPrim(), "PlumbersBlockGreen", (0.055, 0.29, 0.20), 0.43)
    return part_xform, upper - lower


def _sphere(
    stage: Usd.Stage,
    path: str,
    *,
    radius: float,
    color: tuple[float, float, float],
    material_name: str,
) -> UsdGeom.Sphere:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    _set_material(stage, sphere.GetPrim(), material_name, color, 0.28)
    return sphere


def _set_translation(prim: Usd.Prim, position: np.ndarray) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in position)))


def _set_visible(prim: Usd.Prim, visible: bool) -> None:
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _cylinder_between(
    stage: Usd.Stage,
    path: str,
    *,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    material_name: str,
) -> UsdGeom.Cylinder:
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-9:
        raise ValueError(f"Cannot author zero-length cylinder {path}.")
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(length)
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*(0.5 * (np.asarray(start) + np.asarray(end)))))
    rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), Gf.Vec3d(*(direction / length)))
    xform.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
    _set_material(stage, cylinder.GetPrim(), material_name, color, 0.30)
    return cylinder


def _set_xform_matrix(xform: UsdGeom.Xform, matrix: Gf.Matrix4d) -> None:
    xformable = UsdGeom.Xformable(xform.GetPrim())
    ordered_ops = xformable.GetOrderedXformOps()
    if len(ordered_ops) == 1 and ordered_ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
        ordered_ops[0].Set(matrix)
        return
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(matrix)


def _author_external_camera(
    stage: Usd.Stage,
    path: str,
    *,
    position: np.ndarray,
    target: np.ndarray,
) -> UsdGeom.Camera:
    camera = UsdGeom.Camera.Define(stage, path)
    camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.GetFocalLengthAttr().Set(28.0)
    camera.GetHorizontalApertureAttr().Set(36.0)
    camera.GetVerticalApertureAttr().Set(27.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))
    eye = Gf.Vec3d(*(float(value) for value in position))
    look_at = Gf.Vec3d(*(float(value) for value in target))
    transform = Gf.Matrix4d().SetLookAt(eye, look_at, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    _set_xform_matrix(camera, transform)
    return camera


def _camera_basis(stage: Usd.Stage, camera_prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    optical = camera_prim.GetParent()
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(optical)
    origin = np.asarray(matrix.ExtractTranslation(), dtype=float)
    rotation = np.asarray(matrix.ExtractRotationMatrix(), dtype=float)
    return origin, rotation


def _tcp_matrix(stage: Usd.Stage, base_prim: Usd.Prim) -> Gf.Matrix4d:
    base_matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(base_prim)
    local_tcp = Gf.Matrix4d(1.0)
    local_tcp.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, PDZ_TCP_Z_M))
    result = local_tcp * base_matrix
    expected = np.asarray(base_matrix.Transform(Gf.Vec3d(0.0, 0.0, PDZ_TCP_Z_M)), dtype=float)
    np.testing.assert_allclose(np.asarray(result.ExtractTranslation(), dtype=float), expected, atol=1.0e-8)
    return result


def _base_local_matrix(base_matrix: Gf.Matrix4d, xyz: tuple[float, float, float], yaw_deg: float = 0.0) -> Gf.Matrix4d:
    local = Gf.Matrix4d(1.0)
    local.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(yaw_deg)))
    local.SetTranslateOnly(Gf.Vec3d(*xyz))
    return local * base_matrix


def _camera_coordinates(origin: np.ndarray, rotation: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    # Gf matrices transform row vectors: each rotation row is a local axis in
    # world coordinates. Convert a world delta back to local optical XYZ.
    delta_world = np.asarray(point_world, dtype=float) - np.asarray(origin, dtype=float)
    return delta_world @ rotation.T


def _matrix_payload(matrix: Gf.Matrix4d) -> dict[str, object]:
    return {
        "translation_m": np.asarray(matrix.ExtractTranslation(), dtype=float).tolist(),
        "matrix_row_major": np.asarray(matrix, dtype=float).tolist(),
    }


def _set_drive_targets(stage: Usd.Stage) -> None:
    for index, target_rad in enumerate(KUKA_START_RAD, start=1):
        joint = _find_unique_prim(stage, f"joint{index}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        drive.GetTargetPositionAttr().Set(math.degrees(float(target_rad)))
    finger = _find_unique_prim(stage, "pdz_gripper_left_finger_joint")
    UsdPhysics.DriveAPI.Get(finger, "linear").GetTargetPositionAttr().Set(OPEN_FINGER_M)


def _save_rgb(path: Path, rgba: np.ndarray) -> None:
    pixels = np.asarray(rgba)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0.0, 1.0)
        pixels = np.rint(pixels * 255.0).astype(np.uint8)
    Image.fromarray(pixels[..., :3]).save(path)


def _save_depth(path: Path, depth: np.ndarray) -> None:
    values = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(values) & (values > 0.0)
    normalized = np.zeros_like(values)
    normalized[valid] = np.clip((values[valid] - 0.06) / (0.70 - 0.06), 0.0, 1.0)
    red = np.rint(255.0 * (1.0 - normalized)).astype(np.uint8)
    green = np.rint(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
    blue = np.rint(255.0 * normalized).astype(np.uint8)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb[~valid] = 0
    Image.fromarray(rgb).save(path)


def _capture(
    *,
    name: str,
    output_dir: Path,
    rgb_annotator,
    depth_annotator,
) -> None:
    simulation_app.update()
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=4)
    _save_rgb(output_dir / f"{name}_rgb.png", rgb_annotator.get_data())
    _save_depth(output_dir / f"{name}_depth.png", depth_annotator.get_data())


def _capture_rgb(*, name: str, output_dir: Path, rgb_annotator) -> None:
    simulation_app.update()
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=4)
    _save_rgb(output_dir / f"{name}_rgb.png", rgb_annotator.get_data())


def main() -> None:
    robot_usd = _resolve(ARGS.robot_usd)
    part_obj = _resolve(ARGS.part_obj)
    output_dir = _resolve(ARGS.output_dir)
    if not robot_usd.is_file():
        raise FileNotFoundError(robot_usd)
    if not part_obj.is_file():
        raise FileNotFoundError(part_obj)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()
    diagnostics_path = output_dir / "camera_pose_diagnostics.json"
    if diagnostics_path.is_file():
        diagnostics_path.unlink()

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    robot = stage.DefinePrim(ROBOT_PRIM, "Xform")
    robot.GetReferences().AddReference(str(robot_usd))
    # Referenced robot layers load asynchronously in Kit. Wait before resolving
    # joints and the camera hierarchy by name.
    for _ in range(30):
        simulation_app.update()
    print("robot reference loaded", flush=True)
    _author_table(stage)
    part, part_extents = _author_part(stage, part_obj)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1150.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2400.0)
    key.CreateAngleAttr(2.5)
    key_xform = UsdGeom.Xformable(key.GetPrim())
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, -25.0))

    _set_drive_targets(stage)
    print("drive targets authored", flush=True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(4):
        simulation_app.update()
    articulation = SingleArticulation(prim_path=ROBOT_PRIM, reset_xform_properties=False)
    articulation.initialize()
    joint_positions = np.asarray(articulation.get_joint_positions(), dtype=float)
    target_by_name = {
        **{f"joint{index}": value for index, value in enumerate(KUKA_START_RAD, start=1)},
        "pdz_gripper_left_finger_joint": OPEN_FINGER_M,
        "pdz_gripper_right_finger_joint": OPEN_FINGER_M,
    }
    for index, dof_name in enumerate(articulation.dof_names):
        if dof_name in target_by_name:
            joint_positions[index] = float(target_by_name[dof_name])
    articulation.set_joint_positions(joint_positions)
    articulation.set_joint_velocities(np.zeros_like(joint_positions))
    simulation_app.update()
    timeline.pause()
    actual_joint_positions = np.asarray(articulation.get_joint_positions(), dtype=float)
    print(
        f"robot pose teleported: {dict(zip(articulation.dof_names, actual_joint_positions.tolist(), strict=True))}",
        flush=True,
    )

    camera_prim = _find_unique_prim(stage, CAMERA_NAME)
    base_prim = _find_unique_prim(stage, PDZ_BASE_NAME)
    base_matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(base_prim)
    tcp_matrix = _tcp_matrix(stage, base_prim)
    tcp_origin = np.asarray(tcp_matrix.ExtractTranslation(), dtype=float)
    camera_origin, camera_rotation = _camera_basis(stage, camera_prim)
    # Gf uses row-vector transforms, so local optical +Z is row 2 in world
    # coordinates (not column 2 as in a column-vector convention).
    forward = camera_rotation[2, :]
    print(f"camera_origin_world_m={camera_origin.tolist()}", flush=True)
    print(f"camera_rotation_world={camera_rotation.tolist()}", flush=True)
    print(f"camera_forward_world={forward.tolist()}", flush=True)
    part_half_height = 0.5 * float(part_extents[2])
    table_distance = (part_half_height - camera_origin[2]) / forward[2]
    if table_distance <= 0.0:
        raise RuntimeError(f"Camera optical axis does not reach the table in front of the lens: {table_distance=}")
    table_target = camera_origin + float(table_distance) * forward
    table_target[2] = part_half_height

    part_poses: dict[str, Gf.Matrix4d] = {
        "03_part_tcp_base_aligned": tcp_matrix,
        "04_part_tcp_yaw90": _base_local_matrix(base_matrix, (0.0, 0.0, PDZ_TCP_Z_M), yaw_deg=90.0),
        "05_part_30mm_beyond_tcp": _base_local_matrix(base_matrix, (0.0, 0.0, PDZ_TCP_Z_M + 0.030)),
    }
    optical_axis_pose = Gf.Matrix4d(tcp_matrix)
    optical_axis_pose.SetTranslateOnly(Gf.Vec3d(*(camera_origin + 0.150 * forward)))
    part_poses["06_part_optical_axis_150mm"] = optical_axis_pose
    table_pose = Gf.Matrix4d(tcp_matrix)
    table_pose.SetTranslateOnly(Gf.Vec3d(*table_target))
    part_poses["07_part_table_axis"] = table_pose

    UsdGeom.Xform.Define(stage, "/World/DebugGeometry")
    tcp_marker = _sphere(
        stage,
        "/World/DebugGeometry/TcpMarker",
        radius=0.006,
        color=(0.90, 0.03, 0.03),
        material_name="TcpMarkerRed",
    )
    camera_marker = _sphere(
        stage,
        "/World/DebugGeometry/CameraMarker",
        radius=0.005,
        color=(0.03, 0.30, 0.95),
        material_name="CameraMarkerBlue",
    )
    _set_translation(tcp_marker.GetPrim(), tcp_origin)
    _set_translation(camera_marker.GetPrim(), camera_origin)
    camera_to_tcp = _cylinder_between(
        stage,
        "/World/DebugGeometry/CameraToTcp",
        start=camera_origin,
        end=tcp_origin,
        radius=0.0012,
        color=(0.02, 0.80, 0.85),
        material_name="CameraToTcpCyan",
    )
    axis_colors = (
        ("X", (0.92, 0.03, 0.03)),
        ("Y", (0.03, 0.78, 0.10)),
        ("Z", (0.03, 0.22, 0.95)),
    )
    axis_cylinders: list[UsdGeom.Cylinder] = []
    for axis_index, (axis_name, axis_color) in enumerate(axis_colors):
        endpoint_local = np.zeros(3, dtype=float)
        endpoint_local[axis_index] = 0.045
        endpoint_world = np.asarray(tcp_matrix.Transform(Gf.Vec3d(*endpoint_local)), dtype=float)
        axis_cylinders.append(
            _cylinder_between(
                stage,
                f"/World/DebugGeometry/TcpAxis{axis_name}",
                start=tcp_origin,
                end=endpoint_world,
                radius=0.0018,
                color=axis_color,
                material_name=f"TcpAxis{axis_name}",
            )
        )
    external_debug_prims = [
        tcp_marker.GetPrim(),
        camera_marker.GetPrim(),
        camera_to_tcp.GetPrim(),
        *(cylinder.GetPrim() for cylinder in axis_cylinders),
    ]
    for prim in external_debug_prims:
        _set_visible(prim, False)
    print(f"camera resolved at {camera_prim.GetPath()}", flush=True)

    render_product = rep.create.render_product(str(camera_prim.GetPath()), (848, 480))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb.attach([render_product])
    depth.attach([render_product])
    print("render product and annotators ready", flush=True)

    _set_visible(part.GetPrim(), False)
    _capture(
        name="01_empty_gripper",
        output_dir=output_dir,
        rgb_annotator=rgb,
        depth_annotator=depth,
    )

    _set_visible(tcp_marker.GetPrim(), True)
    _capture(
        name="02_tcp_marker",
        output_dir=output_dir,
        rgb_annotator=rgb,
        depth_annotator=depth,
    )
    _set_visible(tcp_marker.GetPrim(), False)

    _set_visible(part.GetPrim(), True)
    for name, pose in part_poses.items():
        _set_xform_matrix(part, pose)
        _capture(
            name=name,
            output_dir=output_dir,
            rgb_annotator=rgb,
            depth_annotator=depth,
        )

    # External views make the physical relationship independently visible:
    # blue is the optical origin, red is the TCP, and RGB rods are TCP axes.
    _set_xform_matrix(part, tcp_matrix)
    for prim in external_debug_prims:
        _set_visible(prim, True)
    external_target = tcp_origin
    external_specs = (
        ("08_external_camera_mount_side", (0.28, -0.34, 0.22)),
        ("09_external_camera_mount_front", (0.0, -0.42, 0.18)),
        ("10_external_finger_closing_axis", (0.42, -0.08, 0.16)),
    )
    external_camera_positions: dict[str, list[float]] = {}
    for name, local_position in external_specs:
        camera_position = np.asarray(base_matrix.Transform(Gf.Vec3d(*local_position)), dtype=float)
        external_camera_positions[name] = camera_position.tolist()
        external_camera = _author_external_camera(
            stage,
            f"/World/DiagnosticCameras/{name}",
            position=tuple(float(value) for value in camera_position),
            target=tuple(float(value) for value in external_target),
        )
        external_product = rep.create.render_product(str(external_camera.GetPath()), (960, 720))
        external_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        external_rgb.attach([external_product])
        _capture_rgb(name=name, output_dir=output_dir, rgb_annotator=external_rgb)

    tcp_in_camera = _camera_coordinates(camera_origin, camera_rotation, tcp_origin)
    diagnostics = {
        "robot_usd": str(robot_usd),
        "part_obj": str(part_obj),
        "part_extents_m": part_extents.tolist(),
        "isaac_joint_targets_rad": list(KUKA_START_RAD),
        "actual_joint_positions": dict(
            zip(articulation.dof_names, actual_joint_positions.tolist(), strict=True)
        ),
        "pdz_finger_position_m": OPEN_FINGER_M,
        "camera": {
            "prim": str(camera_prim.GetPath()),
            "ros_optical_origin_world_m": camera_origin.tolist(),
            "ros_optical_rotation_world": camera_rotation.tolist(),
            "ros_optical_forward_world": forward.tolist(),
            "nominal_resolution": [848, 480],
            "nominal_fov_deg": [87.0, 58.0],
        },
        "pdz_base": _matrix_payload(base_matrix),
        "tcp": {
            **_matrix_payload(tcp_matrix),
            "coordinates_in_ros_optical_frame_m": tcp_in_camera.tolist(),
            "range_from_camera_m": float(np.linalg.norm(tcp_in_camera)),
        },
        "part_pose_centers_world_m": {
            name: np.asarray(pose.ExtractTranslation(), dtype=float).tolist() for name, pose in part_poses.items()
        },
        "table_optical_axis_distance_m": float(table_distance),
        "external_camera_positions_world_m": external_camera_positions,
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")

    # The application closes immediately after capture, which tears these
    # annotators down. Explicit detach is avoided because Replicator 1.12
    # expects a render-product path here while create.render_product returns a
    # HydraTexture in this Isaac release.
    print(f"camera_prim={camera_prim.GetPath()}", flush=True)
    print(f"camera_origin_world_m={camera_origin.tolist()}", flush=True)
    print(f"camera_forward_world={forward.tolist()}", flush=True)
    print(f"tcp_origin_world_m={tcp_origin.tolist()}", flush=True)
    print(f"tcp_in_ros_optical_frame_m={tcp_in_camera.tolist()}", flush=True)
    print(diagnostics_path, flush=True)
    for path in sorted(output_dir.glob("*.png")):
        print(path, flush=True)


try:
    main()
except BaseException:
    # Kit overrides Python's exception hook in some headless launch modes. Emit
    # the traceback explicitly so render failures remain diagnosable in CI.
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
