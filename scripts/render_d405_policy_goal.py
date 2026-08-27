#!/usr/bin/env python3
"""Render one live MoveIt-selected D405 goal with MuJoCo Filament."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from mujoco.rendering.classic import gl_context
from scipy.spatial.transform import Rotation

# The experimental Filament backend does not create an OpenGL context, while
# MuJoCo's Python renderer still imports this name on the local build.
if not hasattr(gl_context, "GLContext"):
    gl_context.GLContext = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grasp_planning.d405_wrist_camera import (  # noqa: E402
    D405_VISUAL_SERVO_CAMERA_PROFILE,
    D405_VISUAL_SERVO_OBSERVATION_PROFILE,
    VISUAL_SERVO_RENDER_HEIGHT,
    VISUAL_SERVO_RENDER_WIDTH,
    D405WristCameraConfig,
    camera_mount_profile_from_camera_profile,
    camera_pose_in_link7,
)
from grasp_planning.grasping.fabrica_grasp_debug import load_grasp_bundle  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.isaac_visual_materials import VISUAL_SERVO_MATERIAL_PROFILE  # noqa: E402
from grasp_planning.mujoco import build_bundle_local_mesh  # noqa: E402
from grasp_planning.rl.goal_renderer_profiles import (  # noqa: E402
    GOAL_FILAMENT_MATERIALS,
    MUJOCO_GOAL_RENDERER_BACKEND,
    MUJOCO_GOAL_RENDERER_PROFILE,
)
from grasp_planning.start_poses import (  # noqa: E402
    gripper_joint_target_from_width,
)
from grasp_planning.visual_servo_workspace import VISUAL_SERVO_TSLOT_PROFILE  # noqa: E402

WIDTH = VISUAL_SERVO_RENDER_WIDTH
HEIGHT = VISUAL_SERVO_RENDER_HEIGHT


def _csv_floats(raw: str, *, count: int, label: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in str(raw).split(",") if value.strip())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(
            f"{label} must contain {count} finite comma-separated values."
        )
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--grasp-id", required=True)
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--goal-joint-positions", required=True)
    parser.add_argument("--goal-tcp-position", required=True)
    parser.add_argument("--goal-tcp-orientation-xyzw", required=True)
    parser.add_argument("--approach-width-m", type=float, required=True)
    parser.add_argument("--maximum-approach-width-m", type=float, required=True)
    parser.add_argument("--object-position", default="")
    parser.add_argument("--object-orientation-xyzw", default="")
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--camera-profile", default=D405_VISUAL_SERVO_CAMERA_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--renderer-backend",
        choices=(MUJOCO_GOAL_RENDERER_BACKEND,),
        default=MUJOCO_GOAL_RENDERER_BACKEND,
    )
    parser.add_argument("--maximum-position-error-m", type=float, default=0.002)
    parser.add_argument("--maximum-rotation-error-deg", type=float, default=1.0)
    parser.add_argument("--minimum-depth-std-m", type=float, default=0.01)
    return parser.parse_args()


def _bundle_object_pose(bundle: object) -> ObjectWorldPose:
    raw = dict(bundle.metadata).get("execution_world_pose")
    if not isinstance(raw, dict):
        raise ValueError("Stage-2 bundle has no execution_world_pose metadata.")
    return ObjectWorldPose(
        position_world=tuple(float(value) for value in raw["position_world"]),
        orientation_xyzw_world=tuple(
            float(value) for value in raw["orientation_xyzw_world"]
        ),
    )


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _export_robot_mjcf(robot_urdf: Path, output_mjcf: Path) -> None:
    """Canonicalize the URDF without the experimental Filament preload."""

    environment = os.environ.copy()
    for key in (
        "LD_PRELOAD",
        "MUJOCO_FILAMENT_ACTIVE",
        "MUJOCO_FILAMENT_ASSETS_DIR",
        "VK_ICD_FILENAMES",
    ):
        environment.pop(key, None)
    environment["MUJOCO_GL"] = "disable"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/export_robot_urdf_mjcf.py"),
            str(robot_urdf),
            str(output_mjcf),
        ],
        check=True,
        env=environment,
    )


def _fixed_link_transform(
    urdf_root: ET.Element,
    *,
    ancestor_link: str,
    descendant_link: str,
) -> tuple[np.ndarray, np.ndarray]:
    child_to_joint: dict[str, ET.Element] = {}
    for joint in urdf_root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            child_to_joint[str(child.get("link"))] = joint
    chain: list[ET.Element] = []
    current = str(descendant_link)
    while current != str(ancestor_link):
        joint = child_to_joint.get(current)
        if joint is None or str(joint.get("type")) != "fixed":
            raise ValueError(
                f"No fixed URDF transform from '{ancestor_link}' to '{descendant_link}'."
            )
        parent = joint.find("parent")
        if parent is None:
            raise ValueError(f"Joint '{joint.get('name')}' has no parent link.")
        chain.append(joint)
        current = str(parent.get("link"))

    transform = np.eye(4, dtype=np.float64)
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        )
        rpy = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        )
        local = np.eye(4, dtype=np.float64)
        local[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        local[:3, 3] = xyz
        transform = transform @ local
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def _ros_camera_quat_to_opengl(quat_wxyz: np.ndarray) -> np.ndarray:
    rotation_ros = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    rotation_gl = rotation_ros @ np.diag([1.0, -1.0, -1.0])
    quat_xyzw = Rotation.from_matrix(rotation_gl).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def _add_material(asset: ET.Element, *, name: str) -> None:
    material = GOAL_FILAMENT_MATERIALS[name]
    ET.SubElement(
        asset,
        "material",
        name=name,
        rgba=" ".join(str(value) for value in (*material.color, 1.0)),
        specular="0.18",
        shininess="0.25",
        metallic=str(material.metallic),
        roughness=str(material.roughness),
        emission=str(material.emission),
    )


def _scene_model(
    robot_mjcf: Path,
    robot_urdf: Path,
    part_mesh: Path,
    *,
    camera_profile: str,
) -> mujoco.MjModel:
    root = ET.parse(robot_mjcf).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Canonical robot MJCF has no worldbody.")
    link7_xml = worldbody.find(".//body[@name='link7']")
    if link7_xml is None:
        raise RuntimeError("Canonical robot MJCF has no link7 body.")

    camera_cfg = D405WristCameraConfig(
        enabled=True,
        include_privileged_mask=False,
        mount_profile=camera_mount_profile_from_camera_profile(camera_profile),
    )
    camera_position, camera_quat_ros = camera_pose_in_link7(camera_cfg)
    camera_quat_gl = _ros_camera_quat_to_opengl(
        np.asarray(camera_quat_ros, dtype=np.float64)
    )
    scale_x = WIDTH / float(camera_cfg.width)
    scale_y = HEIGHT / float(camera_cfg.height)
    ET.SubElement(
        link7_xml,
        "camera",
        name="d405",
        pos=" ".join(f"{value:.12g}" for value in camera_position),
        quat=" ".join(f"{value:.12g}" for value in camera_quat_gl),
        resolution=f"{WIDTH} {HEIGHT}",
        sensorsize=f"{WIDTH} {HEIGHT}",
        focalpixel=f"{camera_cfg.fx * scale_x:.12g} {camera_cfg.fy * scale_y:.12g}",
        principalpixel=(
            f"{camera_cfg.cx * scale_x - WIDTH / 2.0:.12g} "
            f"{HEIGHT / 2.0 - camera_cfg.cy * scale_y:.12g}"
        ),
    )

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(robot_urdf.parent))
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="filament.ao.enabled", data="0")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="selected_part_mesh", file=str(part_mesh))
    for name in GOAL_FILAMENT_MATERIALS:
        _add_material(asset, name=name)

    finger_names = (
        ("pdz_gripper_left_finger_link", "pdz_gripper_right_finger_link")
        if worldbody.find(".//body[@name='pdz_gripper_left_finger_link']") is not None
        else ("left_finger_link", "right_finger_link")
    )
    for finger_name in finger_names:
        finger = worldbody.find(f".//body[@name='{finger_name}']")
        if finger is None:
            raise RuntimeError(f"Canonical MJCF has no {finger_name} body.")
        for geom in finger.findall("geom"):
            geom.attrib.pop("rgba", None)
            geom.set("material", "finger_canonical")

    part_body = ET.SubElement(worldbody, "body", name="selected_part", mocap="true")
    ET.SubElement(
        part_body,
        "geom",
        name="selected_part_visual",
        type="mesh",
        mesh="selected_part_mesh",
        material="part_canonical",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="tslot_backing",
        type="box",
        size="0.325 0.30 0.002",
        pos="0.425 0.05 -0.009",
        material="tslot_slot",
        contype="0",
        conaffinity="0",
    )
    for index in range(25):
        x = 0.425 + (index - 12) * 0.0255
        ET.SubElement(
            worldbody,
            "geom",
            name=f"tslot_land_{index:02d}",
            type="box",
            size="0.01025 0.30 0.0015",
            pos=f"{x:.12g} 0.05 -0.003",
            material="tslot_aluminum",
            contype="0",
            conaffinity="0",
        )
    for name, position, color in (
        ("fill_top", "0.45 0.05 1.20", "1.00 0.98 0.95"),
        ("fill_camera", "0.20 -0.75 0.55", "1.00 0.96 0.91"),
        ("fill_back", "0.65 0.80 0.60", "0.92 0.96 1.00"),
        ("fill_left", "-0.35 0.05 0.55", "0.94 0.97 1.00"),
        ("fill_right", "1.20 0.05 0.55", "1.00 0.97 0.93"),
        ("fill_low", "0.45 0.05 0.18", "0.90 0.94 1.00"),
    ):
        ET.SubElement(
            worldbody,
            "light",
            name=name,
            type="point",
            pos=position,
            diffuse=color,
            intensity="1800",
            range="3.0",
            castshadow="false",
        )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _apply_filament_materials(model: mujoco.MjModel) -> None:
    for name, material in GOAL_FILAMENT_MATERIALS.items():
        material_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_MATERIAL, name
        )
        if material_id < 0:
            raise RuntimeError(f"MuJoCo material '{name}' was not compiled.")
        model.mat_rgba[material_id, :3] = material.color
        model.mat_metallic[material_id] = material.metallic
        model.mat_roughness[material_id] = material.roughness
        model.mat_emission[material_id] = material.emission
    model.light_castshadow[:] = 0


def _tcp_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tcp_position_link7: np.ndarray,
    tcp_rotation_link7: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    link7_rotation = data.xmat[link7_id].reshape(3, 3)
    return (
        data.xpos[link7_id] + link7_rotation @ tcp_position_link7,
        link7_rotation @ tcp_rotation_link7,
    )


def _set_gripper_width(model: mujoco.MjModel, data: mujoco.MjData, width_m: float) -> None:
    """Set whichever supported gripper joints survived URDF-to-MJCF import."""

    candidates = (
        "pdz_gripper_left_finger_joint",
        "pdz_gripper_right_finger_joint",
        "left_finger_joint",
        "right_finger_joint",
    )
    found = False
    for joint_name in candidates:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = gripper_joint_target_from_width(joint_name, width_m)
        found = True
    if not found:
        raise RuntimeError("Canonical MJCF contains no supported gripper joint.")


def main() -> None:  # noqa: C901
    args = _parse_args()
    if os.environ.get("MUJOCO_FILAMENT_ACTIVE") != "1":
        raise RuntimeError(
            "Runtime goal rendering must be launched through "
            "scripts/run_mujoco_filament.sh."
        )
    maximum_approach_width = float(args.maximum_approach_width_m)
    approach_width = float(args.approach_width_m)
    if not math.isfinite(maximum_approach_width) or maximum_approach_width <= 0.0:
        raise ValueError("--maximum-approach-width-m must be finite and positive.")
    if (
        not math.isfinite(approach_width)
        or approach_width <= 0.0
        or approach_width > maximum_approach_width + 1.0e-9
    ):
        raise ValueError(
            "Selected grasp exceeds the physical gripper approach aperture: "
            f"requested={approach_width:.6f} m maximum={maximum_approach_width:.6f} m."
        )

    input_json = args.input_json.expanduser().resolve()
    robot_urdf = args.robot_urdf.expanduser().resolve()
    for required in (input_json, robot_urdf):
        if not required.is_file():
            raise FileNotFoundError(required)
    moveit_joints = _csv_floats(
        args.goal_joint_positions,
        count=7,
        label="--goal-joint-positions",
    )
    goal_position = _csv_floats(
        args.goal_tcp_position,
        count=3,
        label="--goal-tcp-position",
    )
    goal_orientation = _csv_floats(
        args.goal_tcp_orientation_xyzw,
        count=4,
        label="--goal-tcp-orientation-xyzw",
    )
    bundle = load_grasp_bundle(input_json)
    candidate = next(
        (item for item in bundle.candidates if item.grasp_id == args.grasp_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Grasp '{args.grasp_id}' is absent from '{input_json}'.")
    if bool(args.object_position) != bool(args.object_orientation_xyzw):
        raise ValueError("Explicit object position and orientation must be provided together.")
    if args.object_position:
        object_pose = ObjectWorldPose(
            position_world=_csv_floats(
                args.object_position,
                count=3,
                label="--object-position",
            ),
            orientation_xyzw_world=_csv_floats(
                args.object_orientation_xyzw,
                count=4,
                label="--object-orientation-xyzw",
            ),
        )
    else:
        object_pose = _bundle_object_pose(bundle)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    urdf_root = ET.parse(robot_urdf).getroot()
    tcp_link = (
        "pdz_gripper_tcp"
        if urdf_root.find("./link[@name='pdz_gripper_tcp']") is not None
        else "gripper_tcp"
    )
    tcp_position_link7, tcp_rotation_link7 = _fixed_link_transform(
        urdf_root,
        ancestor_link="link7",
        descendant_link=tcp_link,
    )
    bundle_mesh = build_bundle_local_mesh(bundle)

    with tempfile.TemporaryDirectory(prefix="mujoco_runtime_d405_goal_") as temp_name:
        temporary = Path(temp_name)
        part_mesh = temporary / "selected_part_bundle_local.stl"
        trimesh.Trimesh(
            vertices=np.asarray(bundle_mesh.vertices_obj, dtype=np.float64),
            faces=np.asarray(bundle_mesh.faces, dtype=np.int64),
            process=False,
        ).export(part_mesh)
        robot_mjcf = temporary / "robot_canonical.xml"
        _export_robot_mjcf(robot_urdf, robot_mjcf)
        model = _scene_model(
            robot_mjcf,
            robot_urdf,
            part_mesh,
            camera_profile=str(args.camera_profile),
        )
        _apply_filament_materials(model)
        data = mujoco.MjData(model)
        data.qpos[:7] = moveit_joints
        _set_gripper_width(model, data, approach_width)
        data.mocap_pos[0] = object_pose.position_world
        object_xyzw = np.asarray(object_pose.orientation_xyzw_world, dtype=np.float64)
        data.mocap_quat[0] = object_xyzw[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)

        actual_position, actual_rotation = _tcp_pose(
            model,
            data,
            tcp_position_link7,
            tcp_rotation_link7,
        )
        desired_rotation = Rotation.from_quat(goal_orientation).as_matrix()
        position_error = float(
            np.linalg.norm(np.asarray(goal_position, dtype=np.float64) - actual_position)
        )
        rotation_error_deg = math.degrees(
            float(
                np.linalg.norm(
                    Rotation.from_matrix(desired_rotation @ actual_rotation.T).as_rotvec()
                )
            )
        )
        actual_orientation_xyzw = Rotation.from_matrix(actual_rotation).as_quat()
        actual_orientation_wxyz = actual_orientation_xyzw[[3, 0, 1, 2]]

        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        try:
            renderer.update_scene(data, camera="d405")
            rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="d405")
            depth = np.asarray(renderer.render(), dtype=np.float32).copy()
            renderer.disable_depth_rendering()
        finally:
            renderer.close()

    depth = np.nan_to_num(depth, nan=0.50, posinf=0.50, neginf=0.04).astype(
        np.float32
    )
    depth_std = float(depth.std())
    validation_passed = bool(
        position_error <= float(args.maximum_position_error_m)
        and rotation_error_deg <= float(args.maximum_rotation_error_deg)
        and depth_std >= float(args.minimum_depth_std_m)
    )
    goal_id = f"runtime__part_{args.part_id}__{args.grasp_id}"
    _atomic_savez(
        output,
        {
            "schema_version": np.asarray(2, dtype=np.int64),
            "goal_id": np.asarray(goal_id),
            "part_id": np.asarray(str(args.part_id)),
            "grasp_id": np.asarray(str(args.grasp_id)),
            "jaw_width_m": np.asarray(float(candidate.jaw_width), dtype=np.float32),
            "approach_width_m": np.asarray(approach_width, dtype=np.float32),
            "goal_rgb": rgb,
            "goal_depth": depth,
            "goal_camera_profile": np.asarray(str(args.camera_profile)),
            "goal_observation_profile": np.asarray(
                D405_VISUAL_SERVO_OBSERVATION_PROFILE
            ),
            "visual_material_profile": np.asarray(VISUAL_SERVO_MATERIAL_PROFILE),
            "visual_workspace_profile": np.asarray(VISUAL_SERVO_TSLOT_PROFILE),
            "renderer_backend": np.asarray("mujoco_filament"),
            "goal_renderer_backend": np.asarray(MUJOCO_GOAL_RENDERER_BACKEND),
            "goal_renderer_profile": np.asarray(MUJOCO_GOAL_RENDERER_PROFILE),
            "goal_joint_positions": np.asarray(moveit_joints, dtype=np.float32),
            "mujoco_joint_positions": np.asarray(moveit_joints, dtype=np.float32),
            "goal_tcp_position": np.asarray(goal_position, dtype=np.float64),
            "goal_tcp_orientation_xyzw": np.asarray(
                goal_orientation, dtype=np.float64
            ),
            "actual_tcp_position": actual_position.astype(np.float64),
            "actual_tcp_orientation_wxyz": actual_orientation_wxyz.astype(
                np.float64
            ),
            "tcp_position_error_m": np.asarray(position_error, dtype=np.float64),
            "tcp_rotation_error_deg": np.asarray(
                rotation_error_deg, dtype=np.float64
            ),
            "goal_depth_std_m": np.asarray(depth_std, dtype=np.float32),
            "render_validation_passed": np.asarray(
                validation_passed, dtype=np.bool_
            ),
        },
    )
    if not validation_passed:
        raise RuntimeError(
            "Runtime MuJoCo goal render validation failed: "
            f"position={position_error * 1000.0:.3f} mm, "
            f"rotation={rotation_error_deg:.4f} deg, depth_std={depth_std:.6f} m. "
            f"Diagnostic RGB-D was preserved at {output}."
        )
    print(
        f"[GOAL-RENDER] Wrote {goal_id} with MuJoCo Filament to {output}; "
        f"TCP error={position_error * 1000.0:.3f} mm/{rotation_error_deg:.4f} deg.",
        flush=True,
    )


if __name__ == "__main__":
    main()
