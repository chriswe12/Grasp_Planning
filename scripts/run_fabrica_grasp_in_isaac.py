"""Run a saved Fabrica grasp bundle through an Isaac pickup attempt."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Load saved grasps, sample a part placement, and execute a pickup in Isaac."
)
parser.add_argument("--input-json", type=Path, required=True, help="Input grasp bundle, typically from stage 2.")
parser.add_argument(
    "--part-usd",
    type=str,
    default="",
    help=(
        "Optional prebuilt bundle-local USD asset. By default the runner converts the saved bundle-local mesh to USD "
        "so Isaac uses the same frame as stage 2."
    ),
)
parser.add_argument(
    "--use-provided-part-usd",
    action="store_true",
    help="Use --part-usd directly instead of generating a bundle-local USD from the input bundle.",
)
parser.add_argument("--fr3-usd", type=str, default="", help="Optional override for the Franka Panda USD path.")
parser.add_argument(
    "--controller",
    type=str,
    default="moveit",
    choices=("moveit",),
    help="Execution controller. Isaac pickup execution uses MoveIt-planned waypoints.",
)
parser.add_argument("--pregrasp-offset", type=float, default=0.20, help="Pregrasp offset in meters.")
parser.add_argument("--grasp-id", type=str, default="", help="Optional explicit grasp id to execute.")
parser.add_argument(
    "--gripper-width-clearance",
    type=float,
    default=0.01,
    help="Clearance added to the saved grasp jaw width for the open approach width.",
)
parser.add_argument("--close-width", type=float, default=0.0, help="Finger joint target width for close.")
parser.add_argument(
    "--object-mass-kg",
    type=float,
    default=None,
    help="Optional target object mass in kg. Mutually exclusive with --object-density-kg-m3.",
)
parser.add_argument(
    "--object-density-kg-m3",
    type=float,
    default=None,
    help="Optional target object density in kg/m^3. Defaults to PLA-like density when mass is not set.",
)
parser.add_argument(
    "--success-height-margin-m",
    type=float,
    default=0.05,
    help="Minimum object lift height required to report Isaac pickup success.",
)
parser.add_argument(
    "--tcp-to-grasp-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Override the fixed TCP-to-grasp-center offset used for pose conversion and pickup execution.",
)
parser.add_argument("--support-face", type=str, default="", help="Optional explicit support face.")
parser.add_argument("--yaw-deg", type=float, default=None, help="Optional explicit pickup yaw in degrees.")
parser.add_argument("--xy-world", type=str, default="", help="Optional explicit world XY as x,y.")
parser.add_argument("--random-support-face", action="store_true", help="Sample support face from allowed set.")
parser.add_argument("--random-yaw", action="store_true", help="Sample yaw from allowed set.")
parser.add_argument(
    "--allowed-support-faces",
    type=str,
    default="pos_x,neg_x,pos_y,neg_y,neg_z",
    help="Comma-separated support faces used by the random sampler.",
)
parser.add_argument(
    "--allowed-yaw-deg",
    type=str,
    default="0,90,180,270",
    help="Comma-separated yaw values used by the random sampler.",
)
parser.add_argument("--xy-min-world", type=str, default="-0.45,-0.05", help="Random placement XY lower bound.")
parser.add_argument("--xy-max-world", type=str, default="-0.35,0.05", help="Random placement XY upper bound.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    "--detailed-finger-contact-gap-m",
    type=float,
    default=0.002,
    help="Detailed Franka finger contact gap used during the ground recheck.",
)
parser.add_argument(
    "--gripper-collision-model",
    type=str,
    default="",
    help="Gripper collision model for the pickup-pose ground recheck. Defaults to bundle metadata or franka_hand.",
)
parser.add_argument("--pregrasp-only", action="store_true", help="Stop after reaching pregrasp.")
parser.add_argument("--moveit-frame-id", type=str, default="base", help="MoveIt planning frame.")
parser.add_argument(
    "--moveit-target-position-signs",
    type=str,
    default="1,1,1",
    help="Comma-separated x,y,z signs applied to world grasp positions before MoveIt planning.",
)
parser.add_argument("--moveit-planning-group", type=str, default="fr3_arm", help="MoveIt planning group.")
parser.add_argument("--moveit-pose-link", type=str, default="fr3_hand_tcp", help="MoveIt pose link.")
parser.add_argument("--moveit-namespace", type=str, default="", help="Optional MoveIt namespace.")
parser.add_argument(
    "--moveit-joint-names",
    type=str,
    default="",
    help="Optional comma-separated arm joint names. Defaults to fr3_joint1..fr3_joint7.",
)
parser.add_argument(
    "--moveit-start-joint-positions",
    type=str,
    default="",
    help="Optional comma-separated MoveIt start joint positions for direct planning.",
)
parser.add_argument("--moveit-pipeline-id", type=str, default="", help="Optional MoveIt planning pipeline id.")
parser.add_argument("--moveit-planner-id", type=str, default="", help="Optional MoveIt planner id.")
parser.add_argument("--moveit-wait-for-moveit-timeout-s", type=float, default=15.0)
parser.add_argument("--moveit-ik-timeout-s", type=float, default=2.0)
parser.add_argument("--moveit-planning-time-s", type=float, default=5.0)
parser.add_argument("--moveit-num-planning-attempts", type=int, default=5)
parser.add_argument("--moveit-velocity-scale", type=float, default=0.05)
parser.add_argument("--moveit-acceleration-scale", type=float, default=0.05)
parser.add_argument(
    "--moveit-execution-speed-rad-s",
    type=float,
    default=0.35,
    help="Maximum Isaac joint playback speed for MoveIt waypoint execution.",
)
parser.add_argument(
    "--moveit-grasp-settle-time-s",
    type=float,
    default=0.0,
    help="Seconds to hold the final MoveIt grasp waypoint before closing the gripper.",
)
parser.add_argument(
    "--gripper-close-duration-s",
    type=float,
    default=1.5,
    help="Nominal gripper close command duration before contact/target settling.",
)
parser.add_argument(
    "--gripper-close-max-duration-s",
    type=float,
    default=10.0,
    help="Maximum gripper close duration before declaring close failure/contact.",
)
parser.add_argument(
    "--postclose-hold-s",
    type=float,
    default=1.0,
    help="Seconds to hold the closed gripper at the grasp waypoint before lift.",
)
parser.add_argument(
    "--moveit-plan-json",
    type=Path,
    default=None,
    help="Precomputed MoveIt joint waypoint plan. Used when IsaacLab Python cannot import ROS2.",
)
parser.add_argument(
    "--moveit-lift-height-m",
    type=float,
    default=0.08,
    help="Lift height for the MoveIt lift pose target, matching real_execution.lift_height_m by default.",
)
parser.add_argument("--moveit-allow-collisions", action="store_true")
parser.add_argument(
    "--run-seconds",
    type=float,
    default=0.0,
    help="Optional wall-clock duration to keep the simulation alive. Use 0 for until interrupted.",
)
parser.add_argument(
    "--attempt-artifact",
    type=Path,
    default=Path("artifacts/isaac_pick_attempt.json"),
    help="Optional JSON artifact for the selected attempt.",
)
parser.add_argument("--record-video", type=Path, default=None, help="Optional MP4/AVI path for Isaac RGB video.")
parser.add_argument(
    "--visual-servo-goal-image",
    type=Path,
    default=None,
    help="Save the wrist RGB view at the planned grasp waypoint before gripper closure.",
)
parser.add_argument(
    "--visual-servo-comparison-video",
    type=Path,
    default=None,
    help="After execution, write a side-by-side live-versus-goal wrist-camera video.",
)
parser.add_argument(
    "--curriculum-dataset-dir",
    type=Path,
    default=None,
    help="Generate first-curriculum pregrasp-to-grasp expert episodes instead of closing/lifting.",
)
parser.add_argument("--curriculum-episodes", type=int, default=0, help="Number of curriculum episodes to generate.")
parser.add_argument("--curriculum-seed", type=int, default=0, help="Curriculum perturbation seed.")
parser.add_argument(
    "--curriculum-episode-offset",
    type=int,
    default=0,
    help="First output episode index, for appendable multi-run dataset generation.",
)
parser.add_argument("--curriculum-num-envs", type=int, default=1, help="Parallel Isaac environments for curriculum.")
parser.add_argument("--curriculum-writer-workers", type=int, default=1, help="Parallel NPZ writer threads.")
parser.add_argument(
    "--curriculum-sample-hz",
    type=float,
    default=10.0,
    help="Saved RGB-D/label rate; the expert controller continues running at its full policy rate.",
)
parser.add_argument(
    "--curriculum-fixed-object-offset-xy-m",
    type=float,
    nargs=2,
    default=None,
    metavar=("DX", "DY"),
    help="Debug override for a deterministic object XY offset in every curriculum environment.",
)
parser.add_argument(
    "--curriculum-randomize-object-pose",
    action="store_true",
    help="Randomize object XY/yaw during curriculum generation; fixed object pose is the default.",
)
parser.add_argument(
    "--curriculum-fixed-object-yaw-deg",
    type=float,
    default=None,
    help="Debug override for a deterministic object yaw perturbation in every curriculum environment.",
)
parser.add_argument(
    "--curriculum-fixed-ee-offset-grasp-m",
    type=float,
    nargs=3,
    default=None,
    metavar=("DX", "DY", "DZ"),
    help="Deterministic initial TCP position offset, expressed in the target grasp frame.",
)
parser.add_argument(
    "--curriculum-fixed-ee-rotation-deg",
    type=float,
    nargs=3,
    default=None,
    metavar=("RX", "RY", "RZ"),
    help="Deterministic initial local TCP XYZ rotation offset in degrees.",
)
parser.add_argument(
    "--curriculum-ee-position-noise-grasp-m",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Uniform random initial TCP position half-ranges in the target grasp frame.",
)
parser.add_argument(
    "--curriculum-ee-rotation-noise-deg",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("RX", "RY", "RZ"),
    help="Uniform random initial local TCP XYZ rotation half-ranges in degrees.",
)
parser.add_argument(
    "--curriculum-video",
    type=Path,
    default=None,
    help="Record the first actual Isaac curriculum rollout from the wrist camera.",
)
parser.add_argument(
    "--visual-servo-policy-checkpoint",
    type=Path,
    default=None,
    help="Run the learned residual policy in the actual Isaac curriculum rollout.",
)
parser.add_argument(
    "--enable-d405-wrist-camera",
    action="store_true",
    help="Attach the calibrated RealSense D405 left optical camera to the robot link7 prim.",
)
parser.add_argument("--d405-width", type=int, default=848, help="D405 observation width.")
parser.add_argument("--d405-height", type=int, default=480, help="D405 observation height.")
parser.add_argument("--d405-fx", type=float, default=470.900, help="D405 focal length fx in pixels (nominal placeholder).")
parser.add_argument("--d405-fy", type=float, default=432.971, help="D405 focal length fy in pixels (nominal placeholder).")
parser.add_argument("--d405-cx", type=float, default=423.5, help="D405 principal point cx (nominal placeholder).")
parser.add_argument("--d405-cy", type=float, default=239.5, help="D405 principal point cy (nominal placeholder).")
parser.add_argument(
    "--d405-disable-privileged-mask",
    action="store_true",
    help="Do not request Isaac semantic segmentation for the target part.",
)
parser.add_argument("--video-fps", type=float, default=30.0, help="Recorded video frame rate.")
parser.add_argument("--video-width", type=int, default=960, help="Recorded video width in pixels.")
parser.add_argument("--video-height", type=int, default=540, help="Recorded video height in pixels.")
parser.add_argument(
    "--video-camera-eye",
    type=float,
    nargs=3,
    default=(1.6, -1.2, 1.0),
    metavar=("X", "Y", "Z"),
    help="Deprecated compatibility option; wrist-camera video ignores this world-space eye position.",
)
parser.add_argument(
    "--video-camera-target",
    type=float,
    nargs=3,
    default=(0.35, 0.0, 0.3),
    metavar=("X", "Y", "Z"),
    help="Deprecated compatibility option; wrist-camera video ignores this world-space target.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.object_mass_kg is not None and args_cli.object_density_kg_m3 is not None:
    parser.error("--object-mass-kg and --object-density-kg-m3 are mutually exclusive.")
if args_cli.object_mass_kg is not None and args_cli.object_mass_kg <= 0.0:
    parser.error("--object-mass-kg must be > 0.")
if args_cli.object_density_kg_m3 is not None and args_cli.object_density_kg_m3 <= 0.0:
    parser.error("--object-density-kg-m3 must be > 0.")
if (
    args_cli.record_video is not None
    or args_cli.visual_servo_goal_image is not None
    or args_cli.visual_servo_comparison_video is not None
    or args_cli.curriculum_dataset_dir is not None
    or args_cli.curriculum_video is not None
    or args_cli.enable_d405_wrist_camera
):
    args_cli.enable_cameras = True
if args_cli.visual_servo_comparison_video is not None and args_cli.record_video is None:
    parser.error("--visual-servo-comparison-video requires --record-video.")
if args_cli.visual_servo_comparison_video is not None and args_cli.visual_servo_goal_image is None:
    parser.error("--visual-servo-comparison-video requires --visual-servo-goal-image.")
if args_cli.curriculum_dataset_dir is not None and args_cli.curriculum_episodes <= 0:
    parser.error("--curriculum-dataset-dir requires --curriculum-episodes > 0.")
if args_cli.curriculum_video is not None and args_cli.curriculum_dataset_dir is None:
    parser.error("--curriculum-video requires --curriculum-dataset-dir.")
if args_cli.curriculum_num_envs <= 0:
    parser.error("--curriculum-num-envs must be > 0.")
if args_cli.curriculum_episode_offset < 0:
    parser.error("--curriculum-episode-offset must be >= 0.")
if args_cli.curriculum_writer_workers <= 0:
    parser.error("--curriculum-writer-workers must be > 0.")
if args_cli.curriculum_sample_hz <= 0.0:
    parser.error("--curriculum-sample-hz must be > 0.")
if any(value < 0.0 for value in args_cli.curriculum_ee_position_noise_grasp_m):
    parser.error("--curriculum-ee-position-noise-grasp-m values must be nonnegative.")
if any(value < 0.0 for value in args_cli.curriculum_ee_rotation_noise_deg):
    parser.error("--curriculum-ee-rotation-noise-deg values must be nonnegative.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors.camera import Camera  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402
from isaaclab.sim.schemas import schemas_cfg  # noqa: E402
from isaaclab.sim.utils import bind_physics_material  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

from grasp_planning import (  # noqa: E402
    accepted_grasps,
    build_pickup_pose_world,
    evaluate_saved_grasps_against_pickup_pose,
    load_grasp_bundle,
    sample_pickup_placement_spec,
    saved_grasp_to_world_grasp,
    score_grasps,
    select_first_feasible_grasp,
)
from grasp_planning.controllers.fr3_pick_controller import FR3PickController  # noqa: E402
from grasp_planning.d405_wrist_camera import D405_VISUAL_SERVO_OBSERVATION_PROFILE  # noqa: E402
from grasp_planning.envs import (  # noqa: E402
    DEFAULT_PART_DENSITY_KG_M3,
    ISAAC_MIN_CONTACT_OFFSET_M,
    D405WristCameraConfig,
    make_d405_wrist_camera_cfg,
    make_fr3_part_scene_cfg,
)
from grasp_planning.envs.franka_collisions import expose_franka_mesh_collisions  # noqa: E402
from grasp_planning.grasping.fabrica_grasp_debug import load_stl_mesh  # noqa: E402
from grasp_planning.grasping.world_constraints import ObjectWorldPose  # noqa: E402
from grasp_planning.isaac_visual_materials import apply_visual_servo_materials  # noqa: E402
from grasp_planning.isaac_visual_scene import make_visual_servo_render_cfg  # noqa: E402
from grasp_planning.mujoco.scene_builder import write_temporary_triangle_mesh_stl  # noqa: E402
from grasp_planning.planning.fr3_motion_context import FR3MotionContext  # noqa: E402
from grasp_planning.planning.pick_execution import (  # noqa: E402
    drive_robot_to_start_pose,
    execute_pick_from_moveit_joint_trajectories,
)
from grasp_planning.planning.types import PoseCommand  # noqa: E402
from grasp_planning.rl.visual_servo_curriculum import (  # noqa: E402
    VisualServoCurriculumConfig,
    alignment_funnel_expert_twist,
    interpolate_pose,
    pose_error_twist,
    precision_docking_expert_twist,
    smooth_trajectory_progress,
    write_episode_npz,
)
from grasp_planning.rl.visual_servo_dataset import (  # noqa: E402
    ANGULAR_ACTION_SCALE_RAD_S,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    LINEAR_ACTION_SCALE_M_S,
    camera_twist_to_world,
    normalize_twist,
    world_twist_to_camera,
)
from grasp_planning.rl.visual_servo_policy import ResidualVisualServoPolicy  # noqa: E402
from grasp_planning.ros2.moveit_pose_commander import (  # noqa: E402
    MoveItPoseCommander,
    MoveItPoseCommanderConfig,
    rclpy,
)
from grasp_planning.ros2.moveit_world_grasp import world_grasp_pose_targets  # noqa: E402
from grasp_planning.scene_defaults import ROBOT_BASE_ORIENTATION_XYZW, ROBOT_BASE_POSITION  # noqa: E402
from grasp_planning.start_poses import (  # noqa: E402
    DEFAULT_HAND_OPEN_WIDTH,
    KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M,
    kuka_isaac_to_moveit_joint_positions,
    kuka_moveit_to_isaac_joint_positions,
)
from grasp_planning.video import OpenCvVideoWriter  # noqa: E402


def _set_debug_display_color(prim, color: tuple[float, float, float]) -> None:
    """Author a constant viewport color on non-physical debug geometry."""

    primvars = UsdGeom.PrimvarsAPI(prim)
    display_color = primvars.CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.constant,
    )
    display_color.Set([Gf.Vec3f(*color)])


def _add_d405_debug_housing(camera_prim_path: str) -> str:
    """Add a visible, non-colliding D405-sized housing behind the optical frame."""

    stage = omni.usd.get_context().get_stage()
    housing_path = f"{camera_prim_path.rstrip('/')}/DebugHousing"
    housing = UsdGeom.Cube.Define(stage, housing_path)
    housing.CreateSizeAttr(1.0)
    # Approximate 42 x 42 x 23 mm D405 envelope. The camera prim uses the
    # OpenGL convention internally, so +Z is behind the -Z viewing direction.
    xform = UsdGeom.Xformable(housing)
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0115))
    xform.AddScaleOp().Set(Gf.Vec3d(0.042, 0.042, 0.023))
    _set_debug_display_color(housing.GetPrim(), (1.0, 0.25, 0.02))
    return housing_path


def _apply_matte_pla_materials() -> dict[str, object]:
    """Compatibility wrapper for the shared RL/execution material profile."""

    return apply_visual_servo_materials()


class GraspSelectionFailure(RuntimeError):
    def __init__(self, *, status: str, message: str, world_grasp=None) -> None:
        super().__init__(message)
        self.status = status
        self.world_grasp = world_grasp


def _effective_object_density_kg_m3() -> float | None:
    if args_cli.object_mass_kg is not None:
        return None
    if args_cli.object_density_kg_m3 is not None:
        return float(args_cli.object_density_kg_m3)
    return float(DEFAULT_PART_DENSITY_KG_M3)


def _object_mass_properties_cfg():
    if args_cli.object_mass_kg is not None:
        return sim_utils.MassPropertiesCfg(mass=float(args_cli.object_mass_kg))
    density_kg_m3 = _effective_object_density_kg_m3()
    if density_kg_m3 is None:
        return None
    return sim_utils.MassPropertiesCfg(density=float(density_kg_m3))


def _bind_high_friction_contact_material(stage, *, root_paths: tuple[str, ...]) -> int:
    material_path = "/World/Looks/high_friction_contact_material"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=3.0,
        dynamic_friction=2.5,
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )
    material_cfg.func(material_path, material_cfg)
    bound_count = 0
    for root_path in root_paths:
        prim = stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            print(f"[WARN]: Cannot bind contact material; prim does not exist: {root_path}", flush=True)
            continue
        try:
            bind_physics_material(root_path, material_path, stage=stage, stronger_than_descendants=True)
            bound_count += 1
        except ValueError as exc:
            print(f"[WARN]: Cannot bind contact material to {root_path}: {exc}", flush=True)
    return bound_count


def _parse_vec2(raw: str) -> tuple[float, float]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 2:
        raise ValueError(f"Expected exactly 2 comma-separated values, got '{raw}'.")
    return values


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated float, got '{raw}'.")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated token, got '{raw}'.")
    return values


def _moveit_joint_names_from_args() -> tuple[str, ...]:
    if not str(args_cli.moveit_joint_names).strip():
        return MoveItPoseCommanderConfig().joint_names
    return _parse_str_tuple(str(args_cli.moveit_joint_names))


def _using_kuka_lbr_moveit_joints() -> bool:
    joint_names = _moveit_joint_names_from_args()
    return len(joint_names) == 7 and all(name == f"lbr_A{index}" for index, name in enumerate(joint_names, start=1))


def _moveit_target_position_signs_from_args() -> tuple[float, float, float]:
    signs = _parse_float_tuple(str(args_cli.moveit_target_position_signs))
    if len(signs) != 3:
        raise ValueError(
            f"--moveit-target-position-signs must contain exactly 3 comma-separated values for x,y,z, got {len(signs)}."
        )
    return (float(signs[0]), float(signs[1]), float(signs[2]))


def _moveit_waypoint_to_isaac(waypoint: tuple[float, ...]) -> tuple[float, ...]:
    if _using_kuka_lbr_moveit_joints():
        return kuka_moveit_to_isaac_joint_positions(waypoint)
    return tuple(float(value) for value in waypoint)


def _isaac_waypoint_to_moveit(waypoint: tuple[float, ...]) -> tuple[float, ...]:
    if _using_kuka_lbr_moveit_joints():
        return kuka_isaac_to_moveit_joint_positions(waypoint)
    return tuple(float(value) for value in waypoint)


def _moveit_start_joint_positions_from_args() -> tuple[float, ...] | None:
    if not str(args_cli.moveit_start_joint_positions).strip():
        return None
    positions = _parse_float_tuple(str(args_cli.moveit_start_joint_positions))
    joint_names = _moveit_joint_names_from_args()
    if len(positions) != len(joint_names):
        raise ValueError(
            f"--moveit-start-joint-positions must match the configured MoveIt joint-name count ({len(joint_names)})."
        )
    return positions


def _approach_open_gripper_width(selected_world_grasp=None) -> float:
    del selected_world_grasp
    if _using_kuka_lbr_moveit_joints():
        return float(KUKA_Y_GRIPPER_SOURCE_OPEN_WIDTH_M)
    return float(DEFAULT_HAND_OPEN_WIDTH)


def _write_kuka_configured_start_state(scene) -> None:
    """Apply Isaac Lab's configured articulation defaults before the first rendered step."""

    if not _using_kuka_lbr_moveit_joints():
        return
    robot = scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    scene.write_data_to_sim()
    configured = {
        name: float(joint_pos[0, index]) for index, name in enumerate(robot.joint_names) if name.startswith("joint")
    }
    print(f"[INFO]: Applied configured KUKA articulation start state: {configured}", flush=True)


def _prepare_robot_start_pose(sim, scene, *, hand_open_width: float, step_callback=None) -> None:
    if _using_kuka_lbr_moveit_joints():
        print(
            "[INFO]: KUKA spawned at its configured articulation start state; "
            "skipping the legacy active start-pose drive.",
            flush=True,
        )
        return
    print("[INFO]: Driving robot to start pose...", flush=True)
    drive_robot_to_start_pose(
        sim,
        scene,
        hand_open_width=hand_open_width,
        step_callback=step_callback,
    )


def _effective_close_gripper_width(selected_grasp) -> float:
    requested_close_width = float(args_cli.close_width)
    if requested_close_width > 0.0 or not _using_kuka_lbr_moveit_joints():
        return requested_close_width
    jaw_width = float(getattr(selected_grasp, "jaw_width", 0.0))
    if not np.isfinite(jaw_width) or jaw_width <= 0.0:
        return requested_close_width
    return min(0.001, jaw_width)


def _is_generated_kuka_y_gripper_usd(path: str | Path) -> bool:
    resolved = str(path).replace("\\", "/")
    return "kuka_iiwa7_y_gripper" in resolved


def resolve_fr3_usd_path() -> str:
    if args_cli.fr3_usd:
        return args_cli.fr3_usd
    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Unable to resolve Isaac asset root for the built-in Franka Factory asset.")
    return assets_root_path + "/Isaac/IsaacLab/Factory/franka_mimic.usd"


def configure_grasp_tcp_calibration() -> None:
    tcp_to_grasp_offset = tuple(float(value) for value in args_cli.tcp_to_grasp_offset)
    FR3MotionContext._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset
    FR3PickController._TCP_TO_GRASP_CENTER_OFFSET = tcp_to_grasp_offset


def _gripper_collision_model_from_args_or_bundle(bundle) -> str:
    if str(args_cli.gripper_collision_model).strip():
        return str(args_cli.gripper_collision_model)
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    return str(metadata.get("gripper_collision_model", "franka_hand"))


def _moveit_config_from_args() -> MoveItPoseCommanderConfig:
    return MoveItPoseCommanderConfig(
        planning_group=str(args_cli.moveit_planning_group),
        pose_link=str(args_cli.moveit_pose_link),
        joint_names=_moveit_joint_names_from_args(),
        moveit_namespace=str(args_cli.moveit_namespace),
        pipeline_id=str(args_cli.moveit_pipeline_id),
        planner_id=str(args_cli.moveit_planner_id),
        wait_for_moveit_timeout_s=float(args_cli.moveit_wait_for_moveit_timeout_s),
        ik_timeout_s=float(args_cli.moveit_ik_timeout_s),
        fk_timeout_s=float(args_cli.moveit_ik_timeout_s),
        planning_time_s=float(args_cli.moveit_planning_time_s),
        num_planning_attempts=int(args_cli.moveit_num_planning_attempts),
        velocity_scale=float(args_cli.moveit_velocity_scale),
        acceleration_scale=float(args_cli.moveit_acceleration_scale),
        post_execute_sleep_s=0.0,
        avoid_collisions=not bool(args_cli.moveit_allow_collisions),
    )


def _trajectory_waypoints_for_joints(trajectory, *, joint_names: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    joint_trajectory = trajectory.joint_trajectory
    source_joint_names = tuple(str(name) for name in joint_trajectory.joint_names)
    name_to_index = {name: index for index, name in enumerate(source_joint_names)}
    missing = [joint_name for joint_name in joint_names if joint_name not in name_to_index]
    if missing:
        raise RuntimeError(f"MoveIt trajectory is missing arm joints: {missing}.")
    ordered_indices = [name_to_index[name] for name in joint_names]
    waypoints = tuple(
        tuple(float(point.positions[index]) for index in ordered_indices) for point in tuple(joint_trajectory.points)
    )
    if not waypoints:
        raise RuntimeError("MoveIt returned a trajectory with no points.")
    return waypoints


def _plan_moveit_target_sequence(
    *,
    targets,
    labels: tuple[str, ...],
    start_joint_positions: tuple[float, ...],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    if rclpy is None:
        raise RuntimeError("ROS2 MoveIt dependencies are unavailable. Source the ROS2 / MoveIt workspace first.")
    initialized_here = False
    commander = None
    try:
        if not rclpy.ok():
            print("[INFO]: Initializing ROS2 client for MoveIt planning.", flush=True)
            rclpy.init()
            initialized_here = True
        moveit_config = _moveit_config_from_args()
        print(
            f"[INFO]: Connecting to MoveIt group={moveit_config.planning_group} link={moveit_config.pose_link}.",
            flush=True,
        )
        commander = MoveItPoseCommander(moveit_config, node_name="isaac_moveit_trajectory")
        commander.wait_for_moveit(require_execute=False)
        planned: dict[str, tuple[tuple[float, ...], ...]] = {}
        current_start = start_joint_positions
        for label in labels:
            print(f"[INFO]: Requesting MoveIt plan for {label}.", flush=True)
            trajectory, message = commander.plan_to_pose(
                targets[label],
                label=f"isaac_{label}",
                start_joint_positions=current_start,
            )
            if trajectory is None:
                raise RuntimeError(f"MoveIt failed to plan {label}: {message}")
            waypoints = _trajectory_waypoints_for_joints(trajectory, joint_names=moveit_config.joint_names)
            print(f"[INFO]: MoveIt plan for {label} returned {len(waypoints)} waypoints.", flush=True)
            planned[label] = tuple(_moveit_waypoint_to_isaac(waypoint) for waypoint in waypoints)
            current_start = waypoints[-1]
        return planned
    finally:
        if commander is not None:
            commander.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def _plan_moveit_joint_trajectories(
    *,
    world_grasp,
    start_joint_positions: tuple[float, ...],
) -> dict[str, tuple[tuple[float, ...], ...]]:
    print("[INFO]: Building MoveIt pose targets for Isaac attempt.", flush=True)
    targets = world_grasp_pose_targets(
        world_grasp,
        frame_id=str(args_cli.moveit_frame_id),
        lift_height_m=float(args_cli.moveit_lift_height_m),
        position_signs=_moveit_target_position_signs_from_args(),
        tcp_to_grasp_offset=tuple(float(value) for value in args_cli.tcp_to_grasp_offset),
    )
    print("[INFO]: Built MoveIt pose targets for Isaac attempt.", flush=True)
    labels = ("pregrasp",) if args_cli.pregrasp_only else ("pregrasp", "grasp", "lift")
    return _plan_moveit_target_sequence(
        targets=targets,
        labels=labels,
        start_joint_positions=start_joint_positions,
    )


def _current_isaac_arm_joint_positions(*, sim, scene, robot, fixed_gripper_width: float) -> tuple[float, ...]:
    context = FR3MotionContext(
        robot=robot,
        scene=scene,
        sim=sim,
        fixed_gripper_width=fixed_gripper_width,
    )
    return tuple(float(value) for value in context.get_arm_q()[0].tolist())


def _print_moveit_joint_trajectory_summary(trajectories: dict[str, tuple[tuple[float, ...], ...]]) -> None:
    for label, waypoints in trajectories.items():
        print(
            f"[INFO]: MoveIt Isaac trajectory {label}: waypoints={len(waypoints)}",
            flush=True,
        )


def _load_moveit_plan_json(path: Path) -> tuple[dict[str, object], dict[str, tuple[tuple[float, ...], ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected MoveIt plan JSON object in '{path}'.")
    raw_trajectories = payload.get("trajectories")
    if not isinstance(raw_trajectories, dict):
        raise ValueError(f"MoveIt plan JSON '{path}' is missing a trajectories object.")
    trajectories: dict[str, tuple[tuple[float, ...], ...]] = {}
    for label, raw_waypoints in raw_trajectories.items():
        if not isinstance(raw_waypoints, list):
            raise ValueError(f"MoveIt plan trajectory '{label}' must be a list of waypoints.")
        waypoints = []
        for raw_waypoint in raw_waypoints:
            if not isinstance(raw_waypoint, list | tuple):
                raise ValueError(f"MoveIt plan trajectory '{label}' contains a non-list waypoint.")
            waypoints.append(_moveit_waypoint_to_isaac(tuple(float(value) for value in raw_waypoint)))
        trajectories[str(label)] = tuple(waypoints)
    return payload, trajectories


def _mesh_in_bundle_local_frame(bundle) -> object:
    mesh_global = load_stl_mesh(bundle.target_stl_path, scale=bundle.stl_scale)
    rotation = bundle.local_frame_orientation_xyzw_world
    object_pose_world = type("BundlePose", (), {})()
    object_pose_world.rotation_world_from_object = None
    # Use the same row-vector convention as ObjectWorldPose.transform_points_to_world.
    from grasp_planning.grasping.fabrica_grasp_debug import TriangleMesh, quat_to_rotmat_xyzw

    rot = quat_to_rotmat_xyzw(rotation)
    translation = np.asarray(bundle.local_frame_origin_world, dtype=float)
    vertices_local = (np.asarray(mesh_global.vertices_obj, dtype=float) - translation[None, :]) @ rot
    return TriangleMesh(vertices_obj=vertices_local, faces=np.asarray(mesh_global.faces, dtype=np.int64))


def _bundle_execution_pose_world(bundle) -> ObjectWorldPose | None:
    metadata = dict(bundle.metadata)
    raw_pose = metadata.get("execution_world_pose")
    if not isinstance(raw_pose, dict):
        return None
    position_world = raw_pose.get("position_world")
    orientation_xyzw_world = raw_pose.get("orientation_xyzw_world")
    if not isinstance(position_world, (list, tuple)) or not isinstance(orientation_xyzw_world, (list, tuple)):
        return None
    if len(position_world) != 3 or len(orientation_xyzw_world) != 4:
        return None
    return ObjectWorldPose(
        position_world=tuple(float(v) for v in position_world),
        orientation_xyzw_world=tuple(float(v) for v in orientation_xyzw_world),
    )


def _explicit_pose_spec(object_pose_world: ObjectWorldPose):
    return type(
        "PlacementSpec",
        (),
        {
            "support_face": "explicit_pose",
            "yaw_deg": 0.0,
            "xy_world": tuple(float(v) for v in object_pose_world.position_world[:2]),
        },
    )()


def _resolve_placement_spec():
    explicit_xy = _parse_vec2(args_cli.xy_world) if args_cli.xy_world else None
    if args_cli.support_face and args_cli.yaw_deg is not None and explicit_xy is not None:
        from grasp_planning.grasping.fabrica_grasp_debug import PickupPlacementSpec

        return PickupPlacementSpec(
            support_face=args_cli.support_face,
            yaw_deg=float(args_cli.yaw_deg),
            xy_world=explicit_xy,
        )

    rng = np.random.default_rng(args_cli.seed)
    return sample_pickup_placement_spec(
        rng=rng,
        allowed_support_faces=_parse_str_tuple(args_cli.allowed_support_faces),
        allowed_yaw_deg=_parse_float_tuple(args_cli.allowed_yaw_deg),
        xy_min_world=_parse_vec2(args_cli.xy_min_world),
        xy_max_world=_parse_vec2(args_cli.xy_max_world),
    )


def _resolve_placement_and_pose(bundle, mesh_local):
    bundle_pose_world = _bundle_execution_pose_world(bundle)
    has_pickup_override = bool(args_cli.support_face or args_cli.yaw_deg is not None or args_cli.xy_world)
    if bundle_pose_world is not None and not has_pickup_override:
        return _explicit_pose_spec(bundle_pose_world), bundle_pose_world

    placement_spec = _resolve_placement_spec()
    object_pose_world = build_pickup_pose_world(
        mesh_local,
        support_face=placement_spec.support_face,
        yaw_deg=placement_spec.yaw_deg,
        xy_world=placement_spec.xy_world,
    )
    return placement_spec, object_pose_world


def _mesh_collision_cfg():
    return schemas_cfg.ConvexDecompositionPropertiesCfg()


class IsaacVideoRecorder:
    def __init__(self, *, camera: Camera, sim, output_path: Path, fps: float, width: int, height: int) -> None:
        self._camera = camera
        self._sim = sim
        self._writer = OpenCvVideoWriter(output_path, fps=fps, width=width, height=height)
        self._capture_interval_s = 1.0 / float(fps)
        self._next_capture_time_s = 0.0
        self._elapsed_s = 0.0
        self.output_path = str(output_path)

    @property
    def frame_count(self) -> int:
        return int(self._writer.frame_count)

    def capture(self, *, force: bool = False) -> None:
        physics_dt = float(self._sim.get_physics_dt())
        self._elapsed_s += physics_dt
        if not force and self._elapsed_s + 1.0e-9 < self._next_capture_time_s:
            return
        self._camera.update(dt=physics_dt)
        raw_rgb = self._camera.data.output.get("rgb")
        if raw_rgb is None:
            return
        frame = raw_rgb[0]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        self._writer.append_rgb(np.asarray(frame))
        if force:
            self._next_capture_time_s = max(self._next_capture_time_s, self._elapsed_s + self._capture_interval_s)
            return
        while self._next_capture_time_s <= self._elapsed_s + 1.0e-9:
            self._next_capture_time_s += self._capture_interval_s

    def close(self) -> None:
        self._writer.close()


def _camera_rgb_frame(camera: Camera, *, physics_dt: float) -> np.ndarray:
    camera.update(dt=float(physics_dt))
    raw_rgb = camera.data.output.get("rgb")
    if raw_rgb is None:
        raise RuntimeError("D405 camera did not produce an RGB observation.")
    frame = raw_rgb[0]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _write_rgb_image(path: Path, frame_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Could not write visual-servo goal image to '{path}'.")


def _write_live_goal_comparison_video(
    *,
    live_video_path: Path,
    goal_image_path: Path,
    output_path: Path,
    fps: float,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> int:
    import cv2

    goal_bgr = cv2.imread(str(goal_image_path), cv2.IMREAD_COLOR)
    if goal_bgr is None:
        raise RuntimeError(f"Could not read goal image '{goal_image_path}'.")
    capture = cv2.VideoCapture(str(live_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not read live wrist video '{live_video_path}'.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    goal_bgr = cv2.resize(goal_bgr, (width, height), interpolation=cv2.INTER_AREA)
    writer = OpenCvVideoWriter(output_path, fps=float(fps), width=2 * width, height=height)
    frame_count = 0
    source_frame_index = 0
    try:
        while True:
            ok, live_bgr = capture.read()
            if not ok:
                break
            if source_frame_index < int(start_frame):
                source_frame_index += 1
                continue
            if end_frame is not None and source_frame_index > int(end_frame):
                break
            cv2.putText(live_bgr, "LIVE / SCRIPTED APPROACH", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 255), 2)
            goal_labeled = goal_bgr.copy()
            cv2.putText(goal_labeled, "GOAL / PLANNED GRASP", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 180, 30), 2)
            comparison_rgb = cv2.cvtColor(np.concatenate((live_bgr, goal_labeled), axis=1), cv2.COLOR_BGR2RGB)
            writer.append_rgb(comparison_rgb)
            frame_count += 1
            source_frame_index += 1
    finally:
        capture.release()
        writer.close()
    return frame_count


def _curriculum_camera_observation(camera: Camera, *, physics_dt: float) -> dict[str, np.ndarray]:
    import cv2

    camera.update(dt=float(physics_dt))
    output = camera.data.output

    def _numpy(name: str) -> np.ndarray:
        value = output.get(name)
        if value is None:
            raise RuntimeError(f"D405 curriculum observation is missing '{name}'.")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    rgb = _numpy("rgb")[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    depth = _numpy("distance_to_image_plane").astype(np.float32)
    mask = _numpy("semantic_segmentation")
    if mask.ndim == 4:
        mask = mask[..., 0]
    # Only the part is authored with a semantic class.  Replicator may assign
    # different integer IDs between runs, so identify the unlabelled background
    # as the modal ID instead of baking in a numeric target ID.
    width, height = 256, 144
    resized_rgb = []
    resized_depth = []
    resized_masks = []
    for env_index in range(rgb.shape[0]):
        semantic_ids, semantic_counts = np.unique(mask[env_index], return_counts=True)
        background_id = semantic_ids[int(np.argmax(semantic_counts))]
        binary_mask = (mask[env_index] != background_id).astype(np.uint8)
        resized_rgb.append(cv2.resize(rgb[env_index], (width, height), interpolation=cv2.INTER_AREA))
        resized_depth.append(
            np.nan_to_num(
                cv2.resize(
                    depth[env_index],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.float32),
                nan=0.50,
                posinf=0.50,
                neginf=0.04,
            )
        )
        resized_masks.append(
            cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        )
    return {
        "rgb": np.stack(resized_rgb),
        "depth": np.stack(resized_depth),
        "object_mask": np.stack(resized_masks),
    }


def _yaw_quaternion_xyzw(yaw_rad: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(0.5 * yaw_rad), np.cos(0.5 * yaw_rad)], dtype=np.float64)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def _euler_xyz_quaternion_xyzw(rotation_xyz_rad: np.ndarray) -> np.ndarray:
    """Return the quaternion for intrinsic local XYZ rotations."""

    rx, ry, rz = 0.5 * np.asarray(rotation_xyz_rad, dtype=np.float64)
    qx = np.array([np.sin(rx), 0.0, 0.0, np.cos(rx)], dtype=np.float64)
    qy = np.array([0.0, np.sin(ry), 0.0, np.cos(ry)], dtype=np.float64)
    qz = np.array([0.0, 0.0, np.sin(rz), np.cos(rz)], dtype=np.float64)
    quaternion = _quat_multiply_xyzw(_quat_multiply_xyzw(qx, qy), qz)
    return quaternion / np.linalg.norm(quaternion)


def _rotate_vector_by_quaternion_xyzw(
    vector: np.ndarray, quaternion_xyzw: np.ndarray
) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    vector_quaternion = np.array([*np.asarray(vector, dtype=np.float64), 0.0])
    conjugate = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
    return _quat_multiply_xyzw(
        _quat_multiply_xyzw(quaternion, vector_quaternion), conjugate
    )[:3]


def _rotate_z(vector: np.ndarray, yaw_rad: float) -> np.ndarray:
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return rotation @ np.asarray(vector, dtype=np.float64)


def _integrate_world_twist_pose(
    position_world: np.ndarray,
    orientation_xyzw_world: np.ndarray,
    twist_world: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a small world-frame TCP twist into the next absolute IK target."""

    twist = np.asarray(twist_world, dtype=np.float64)
    rotation_vector = twist[3:] * float(dt_s)
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 1.0e-12:
        delta_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        delta_quaternion = np.concatenate(
            (rotation_vector * (np.sin(0.5 * angle) / angle), [np.cos(0.5 * angle)])
        )
    next_orientation = _quat_multiply_xyzw(
        delta_quaternion, np.asarray(orientation_xyzw_world, dtype=np.float64)
    )
    next_orientation /= np.linalg.norm(next_orientation)
    return (
        np.asarray(position_world, dtype=np.float64) + twist[:3] * float(dt_s),
        next_orientation,
    )


def _measured_world_twist(
    previous_position_world: np.ndarray,
    previous_orientation_xyzw_world: np.ndarray,
    current_position_world: np.ndarray,
    current_orientation_xyzw_world: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Estimate world-frame linear/angular TCP velocity between policy frames."""

    dt = float(dt_s)
    linear = (
        np.asarray(current_position_world, dtype=np.float64)
        - np.asarray(previous_position_world, dtype=np.float64)
    ) / dt
    previous_quaternion = np.asarray(
        previous_orientation_xyzw_world, dtype=np.float64
    )
    delta_quaternion = _quat_multiply_xyzw(
        np.asarray(current_orientation_xyzw_world, dtype=np.float64),
        np.array(
            [
                -previous_quaternion[0],
                -previous_quaternion[1],
                -previous_quaternion[2],
                previous_quaternion[3],
            ],
            dtype=np.float64,
        ),
    )
    if delta_quaternion[3] < 0.0:
        delta_quaternion = -delta_quaternion
    vector_norm = float(np.linalg.norm(delta_quaternion[:3]))
    if vector_norm <= 1.0e-12:
        angular = 2.0 * delta_quaternion[:3] / dt
    else:
        angle = 2.0 * np.arctan2(vector_norm, float(delta_quaternion[3]))
        angular = delta_quaternion[:3] * (angle / (vector_norm * dt))
    return np.concatenate((linear, angular))


def _limit_position_command_lead(
    command_position_world: np.ndarray,
    measured_position_world: np.ndarray,
    *,
    max_lead_m: float,
) -> np.ndarray:
    """Bound Cartesian target accumulation ahead of the physical TCP."""

    command = np.asarray(command_position_world, dtype=np.float64)
    measured = np.asarray(measured_position_world, dtype=np.float64)
    lead = command - measured
    lead_norm = float(np.linalg.norm(lead))
    if lead_norm <= float(max_lead_m):
        return command
    return measured + lead * (float(max_lead_m) / lead_norm)


def _curriculum_dls_joint_velocity(context: FR3MotionContext, twist_world: np.ndarray, damping: float) -> torch.Tensor:
    from isaaclab.utils.math import matrix_from_quat, quat_inv

    jacobian = context.robot.root_physx_view.get_jacobians()[
        :, context.ee_jacobi_body_idx, :, context.arm_joint_ids
    ].clone()
    root_quat_w = context.robot.data.root_pose_w[:, 3:7]
    world_to_base = matrix_from_quat(quat_inv(root_quat_w))
    jacobian[:, :3, :] = torch.bmm(world_to_base, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(world_to_base, jacobian[:, 3:, :])
    twist_w = torch.as_tensor(
        np.asarray(twist_world, dtype=np.float32)[None, :],
        dtype=torch.float32,
        device=context.device,
    )
    twist_b = torch.cat(
        (
            torch.bmm(world_to_base, twist_w[:, :3].unsqueeze(-1)).squeeze(-1),
            torch.bmm(world_to_base, twist_w[:, 3:].unsqueeze(-1)).squeeze(-1),
        ),
        dim=1,
    )
    identity = torch.eye(6, dtype=torch.float32, device=context.device).unsqueeze(0)
    solve = torch.linalg.solve(
        torch.bmm(jacobian, jacobian.transpose(1, 2)) + float(damping) ** 2 * identity,
        twist_b.unsqueeze(-1),
    )
    return torch.bmm(jacobian.transpose(1, 2), solve).squeeze(-1)


def _run_first_curriculum(
    *,
    sim,
    scene,
    wrist_camera: Camera,
    moveit_joint_trajectories,
    selected_grasp,
    selected_world_grasp,
    object_pose_world,
    open_gripper_width: float,
) -> dict[str, object]:
    """Generate perturbed privileged-expert episodes for one part and one grasp."""

    config = VisualServoCurriculumConfig()
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(args_cli.curriculum_seed),
                int(args_cli.curriculum_episode_offset),
            ]
        )
    )
    context = FR3MotionContext(
        robot=scene["robot"],
        scene=scene,
        sim=sim,
        fixed_gripper_width=float(open_gripper_width),
    )
    expert_ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=context.device,
    )
    pregrasp_q = torch.tensor(
        [moveit_joint_trajectories["pregrasp"][-1]], dtype=torch.float32, device=context.device
    )
    grasp_q = torch.tensor([moveit_joint_trajectories["grasp"][-1]], dtype=torch.float32, device=context.device)
    part = scene["part"]
    nominal_object_position = np.asarray(object_pose_world.position_world, dtype=np.float64)
    nominal_object_quaternion = np.asarray(object_pose_world.orientation_xyzw_world, dtype=np.float64)
    nominal_pregrasp_position = np.asarray(selected_world_grasp.pregrasp_position_w, dtype=np.float64)
    nominal_grasp_position = np.asarray(selected_world_grasp.position_w, dtype=np.float64)
    nominal_grasp_quaternion = np.asarray(selected_world_grasp.orientation_xyzw, dtype=np.float64)

    # Render a single fixed goal observation at the nominal open-gripper grasp.
    context.reset_joint_state(grasp_q, q_hand=context.get_hand_q(), steps=5)
    goal_observation = _curriculum_camera_observation(wrist_camera, physics_dt=context.physics_dt)
    summary = {"episodes": [], "grasp_id": selected_grasp.grasp_id}
    policy_substeps = max(1, int(round(config.policy_dt_s / context.physics_dt)))
    nominal_velocity = (nominal_grasp_position - nominal_pregrasp_position) / config.approach_duration_s
    approach_nominal_twist = np.concatenate((nominal_velocity, np.zeros(3, dtype=np.float64)))

    for episode_index in range(int(args_cli.curriculum_episodes)):
        dx, dy = rng.uniform(-config.object_translation_xy_m, config.object_translation_xy_m, size=2)
        yaw = np.deg2rad(rng.uniform(-config.object_yaw_deg, config.object_yaw_deg))
        object_position = nominal_object_position + np.array([dx, dy, 0.0])
        object_quaternion = _quat_multiply_xyzw(_yaw_quaternion_xyzw(yaw), nominal_object_quaternion)
        root_pose = torch.tensor(
            [[*object_position, object_quaternion[3], *object_quaternion[:3]]],
            dtype=torch.float32,
            device=part.device,
        )
        part.write_root_pose_to_sim(root_pose)
        if hasattr(part, "write_root_velocity_to_sim"):
            part.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=part.device))

        perturb_q = pregrasp_q + torch.tensor(
            rng.uniform(
                -config.initial_joint_noise_rad,
                config.initial_joint_noise_rad,
                size=tuple(pregrasp_q.shape),
            ),
            dtype=torch.float32,
            device=context.device,
        )
        context.reset_joint_state(perturb_q, q_hand=context.get_hand_q(), steps=5)

        pregrasp_position = object_position + _rotate_z(
            nominal_pregrasp_position - nominal_object_position, yaw
        )
        grasp_position = object_position + _rotate_z(nominal_grasp_position - nominal_object_position, yaw)
        grasp_quaternion = _quat_multiply_xyzw(_yaw_quaternion_xyzw(yaw), nominal_grasp_quaternion)
        buffers: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "rgb_live",
                "depth_live",
                "object_mask",
                "joint_positions",
                "nominal_twist",
                "expert_twist",
                "expert_residual_twist",
                "pose_error",
                "trajectory_progress",
                "funnel_half_width_m",
                "funnel_transverse_error_m",
                "funnel_approach_scale",
                "funnel_near_phase",
            )
        }
        for step_index in range(config.step_count):
            elapsed_s = step_index * config.policy_dt_s
            progress, progress_rate = smooth_trajectory_progress(
                elapsed_s, config.approach_duration_s
            )
            nominal_twist = approach_nominal_twist * (
                progress_rate * config.approach_duration_s
            )
            target_position, target_quaternion = interpolate_pose(
                pregrasp_position,
                grasp_quaternion,
                grasp_position,
                grasp_quaternion,
                progress,
            )
            target_tcp_position, target_tcp_quaternion = context.grasp_pose_to_tcp_pose(
                tuple(float(value) for value in target_position),
                tuple(float(value) for value in target_quaternion),
            )
            current_position_t, current_quaternion_w = context.get_tcp_pose_w()
            current_position = current_position_t[0].detach().cpu().numpy()
            current_quaternion_w = current_quaternion_w[0].detach().cpu().numpy()
            current_quaternion = np.array(
                [current_quaternion_w[1], current_quaternion_w[2], current_quaternion_w[3], current_quaternion_w[0]]
            )
            error = pose_error_twist(
                current_position,
                current_quaternion,
                np.asarray(target_tcp_position),
                np.asarray(target_tcp_quaternion),
            )
            full_twist, residual_twist, funnel = alignment_funnel_expert_twist(
                nominal_twist=nominal_twist,
                pose_error=error,
                grasp_orientation_xyzw=grasp_quaternion,
                trajectory_progress=progress,
                config=config,
            )
            observation = _curriculum_camera_observation(wrist_camera, physics_dt=context.physics_dt)
            buffers["rgb_live"].append(observation["rgb"])
            buffers["depth_live"].append(observation["depth"])
            buffers["object_mask"].append(observation["object_mask"])
            buffers["joint_positions"].append(context.get_arm_q()[0].detach().cpu().numpy())
            buffers["nominal_twist"].append(nominal_twist.astype(np.float32))
            buffers["expert_twist"].append(full_twist.astype(np.float32))
            buffers["expert_residual_twist"].append(residual_twist.astype(np.float32))
            buffers["pose_error"].append(error.astype(np.float32))
            buffers["trajectory_progress"].append(np.array(progress, dtype=np.float32))
            buffers["funnel_half_width_m"].append(
                np.array(funnel["funnel_half_width_m"], dtype=np.float32)
            )
            buffers["funnel_transverse_error_m"].append(
                np.array(funnel["transverse_error_m"], dtype=np.float32)
            )
            buffers["funnel_approach_scale"].append(
                np.array(funnel["approach_scale"], dtype=np.float32)
            )
            buffers["funnel_near_phase"].append(
                np.array(funnel["near_phase"], dtype=np.float32)
            )

            command_position, command_quaternion = _integrate_world_twist_pose(
                current_position,
                current_quaternion,
                full_twist,
                config.policy_dt_s,
            )
            command_grasp_position, command_grasp_quaternion = (
                context.tcp_pose_to_grasp_pose(
                    tuple(float(value) for value in command_position),
                    tuple(float(value) for value in command_quaternion),
                )
            )
            target_command = PoseCommand(
                position_w=command_grasp_position,
                orientation_xyzw=command_grasp_quaternion,
            )
            for _ in range(policy_substeps):
                context.command_pose_via_differential_ik(expert_ik, target_command)
                context.command_fixed_gripper()
                scene.write_data_to_sim()
                sim.step()
                scene.update(context.physics_dt)

        final_position_t, final_quaternion_w = context.get_tcp_pose_w()
        final_quaternion_w = final_quaternion_w[0].detach().cpu().numpy()
        final_error = pose_error_twist(
            final_position_t[0].detach().cpu().numpy(),
            np.array([final_quaternion_w[1], final_quaternion_w[2], final_quaternion_w[3], final_quaternion_w[0]]),
            np.asarray(
                context.grasp_pose_to_tcp_pose(
                    tuple(float(value) for value in grasp_position),
                    tuple(float(value) for value in grasp_quaternion),
                )[0]
            ),
            np.asarray(
                context.grasp_pose_to_tcp_pose(
                    tuple(float(value) for value in grasp_position),
                    tuple(float(value) for value in grasp_quaternion),
                )[1]
            ),
        )
        success = bool(
            np.linalg.norm(final_error[:3]) <= config.success_position_tolerance_m
            and np.linalg.norm(final_error[3:]) <= np.deg2rad(config.success_rotation_tolerance_deg)
        )
        arrays = {name: np.stack(values) for name, values in buffers.items()}
        arrays.update(
            {
                "rgb_goal": goal_observation["rgb"],
                "depth_goal": goal_observation["depth"],
                "goal_object_mask": goal_observation["object_mask"],
            }
        )
        npz_path, _ = write_episode_npz(
            args_cli.curriculum_dataset_dir,
            episode_index=episode_index,
            arrays=arrays,
            metadata={
                "success": success,
                "final_position_error_m": float(np.linalg.norm(final_error[:3])),
                "final_rotation_error_deg": float(np.rad2deg(np.linalg.norm(final_error[3:]))),
                "object_perturbation": {"dx_m": float(dx), "dy_m": float(dy), "yaw_deg": float(np.rad2deg(yaw))},
                "grasp_id": selected_grasp.grasp_id,
                "action_frame": "world",
                "action_semantics": "end_effector_twist",
            },
            config=config,
        )
        episode_summary = {
            "episode": episode_index,
            "success": success,
            "final_position_error_m": float(np.linalg.norm(final_error[:3])),
            "final_rotation_error_deg": float(np.rad2deg(np.linalg.norm(final_error[3:]))),
            "path": str(npz_path),
        }
        summary["episodes"].append(episode_summary)
        print(f"[CURRICULUM]: {episode_summary}", flush=True)

    summary["success_count"] = sum(bool(item["success"]) for item in summary["episodes"])
    summary_path = Path(args_cli.curriculum_dataset_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_batched_first_curriculum(
    *,
    sim,
    scene,
    wrist_camera: Camera,
    moveit_joint_trajectories,
    selected_grasp,
    selected_world_grasp,
    object_pose_world,
    open_gripper_width: float,
) -> dict[str, object]:
    """Generate curriculum episodes with cloned robots, parts, cameras, and batched DLS."""

    from concurrent.futures import ThreadPoolExecutor

    config = VisualServoCurriculumConfig()
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(args_cli.curriculum_seed),
                int(args_cli.curriculum_episode_offset),
            ]
        )
    )
    context = FR3MotionContext(
        robot=scene["robot"],
        scene=scene,
        sim=sim,
        fixed_gripper_width=float(open_gripper_width),
    )
    env_count = int(scene.num_envs)
    expert_ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=env_count,
        device=context.device,
    )
    pregrasp_q = torch.tensor(
        [moveit_joint_trajectories["pregrasp"][-1]], dtype=torch.float32, device=context.device
    ).repeat(env_count, 1)
    grasp_q = torch.tensor(
        [moveit_joint_trajectories["grasp"][-1]], dtype=torch.float32, device=context.device
    ).repeat(env_count, 1)
    part = scene["part"]
    env_origins = scene.env_origins.detach().cpu().numpy().astype(np.float64)
    nominal_object_position = np.asarray(object_pose_world.position_world, dtype=np.float64)
    nominal_object_quaternion = np.asarray(object_pose_world.orientation_xyzw_world, dtype=np.float64)
    nominal_pregrasp_position = np.asarray(selected_world_grasp.pregrasp_position_w, dtype=np.float64)
    nominal_grasp_position = np.asarray(selected_world_grasp.position_w, dtype=np.float64)
    nominal_grasp_quaternion = np.asarray(selected_world_grasp.orientation_xyzw, dtype=np.float64)
    nominal_positions_w = nominal_object_position[None, :] + env_origins
    nominal_quaternions = np.repeat(nominal_object_quaternion[None, :], env_count, axis=0)

    def _write_part_poses(positions_w: np.ndarray, quaternions_xyzw: np.ndarray) -> None:
        root_pose = np.concatenate(
            (positions_w, quaternions_xyzw[:, 3:4], quaternions_xyzw[:, :3]), axis=1
        )
        part.write_root_pose_to_sim(torch.as_tensor(root_pose, dtype=torch.float32, device=part.device))
        if hasattr(part, "write_root_velocity_to_sim"):
            part.write_root_velocity_to_sim(
                torch.zeros((env_count, 6), dtype=torch.float32, device=part.device)
            )

    nominal_grasp_positions_w = nominal_grasp_position[None, :] + env_origins
    nominal_grasp_quaternions = np.repeat(
        nominal_grasp_quaternion[None, :], env_count, axis=0
    )
    _write_part_poses(nominal_positions_w, nominal_quaternions)
    context.reset_joint_state(grasp_q, q_hand=context.get_hand_q(), steps=5)
    goal_position_t = torch.as_tensor(
        nominal_grasp_positions_w, dtype=torch.float32, device=context.device
    )
    goal_quaternion_t = torch.as_tensor(
        nominal_grasp_quaternions, dtype=torch.float32, device=context.device
    )
    goal_convergence_steps = max(1, int(round(1.0 / context.physics_dt)))
    for _ in range(goal_convergence_steps):
        context.command_pose_batch_via_differential_ik(
            expert_ik, goal_position_t, goal_quaternion_t
        )
        context.command_fixed_gripper()
        scene.write_data_to_sim()
        sim.step()
        scene.update(context.physics_dt)
        _write_part_poses(nominal_positions_w, nominal_quaternions)
    goal_tcp_position_t, goal_tcp_quaternion_w_t = context.get_tcp_pose_w()
    goal_tcp_positions_w = goal_tcp_position_t.detach().cpu().numpy()
    goal_tcp_quaternions_wxyz = goal_tcp_quaternion_w_t.detach().cpu().numpy()
    goal_tcp_quaternions_xyzw = goal_tcp_quaternions_wxyz[:, (1, 2, 3, 0)]
    goal_part_pose_w = part.data.root_pose_w.detach().cpu().numpy()
    goal_observation = _curriculum_camera_observation(wrist_camera, physics_dt=context.physics_dt)
    if int(goal_observation["rgb"].shape[0]) != env_count:
        raise RuntimeError(
            f"Batched D405 returned {goal_observation['rgb'].shape[0]} images for {env_count} environments."
        )

    learned_policy = None
    policy_goal_rgbd = None
    policy_downsample = None
    if args_cli.visual_servo_policy_checkpoint is not None:
        checkpoint = torch.load(
            args_cli.visual_servo_policy_checkpoint,
            map_location=context.device,
            weights_only=False,
        )
        learned_policy = ResidualVisualServoPolicy().to(context.device)
        learned_policy.load_state_dict(checkpoint["model_state_dict"])
        learned_policy.eval()
        cache_dir_raw = checkpoint.get("training_config", {}).get(
            "training_cache_dir"
        )
        if not cache_dir_raw:
            raise ValueError(
                "Policy checkpoint lacks training_cache_dir; exact image preprocessing "
                "cannot be verified."
            )
        cache_manifest_path = Path(cache_dir_raw) / "manifest.json"
        if not cache_manifest_path.exists():
            raise FileNotFoundError(
                f"Training cache manifest is unavailable: {cache_manifest_path}"
            )
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cache_manifest.get("resampling") != "area":
            raise ValueError(
                "Policy training cache must use area resampling; rebuild it with "
                "scripts/build_visual_servo_training_cache.py."
            )
        if (
            cache_manifest.get("observation_profile")
            != D405_VISUAL_SERVO_OBSERVATION_PROFILE
        ):
            raise ValueError(
                "Policy training cache observation profile does not match the "
                f"current pipeline ({D405_VISUAL_SERVO_OBSERVATION_PROFILE})."
            )
        source_shape = tuple(int(value) for value in cache_manifest["source_image_shape"])
        expected_shape = tuple(int(value) for value in cache_manifest["image_shape"])
        observed_shape = tuple(int(value) for value in goal_observation["rgb"].shape[1:3])
        if observed_shape != source_shape:
            raise ValueError(
                f"Isaac policy image shape {observed_shape} does not match training source "
                f"shape {source_shape}."
            )
        policy_downsample = int(cache_manifest["downsample"])
        if (
            source_shape[0] // policy_downsample,
            source_shape[1] // policy_downsample,
        ) != expected_shape:
            raise ValueError(
                f"Training cache downsample={policy_downsample} does not produce "
                f"the declared shape {expected_shape} from {source_shape}."
            )

        def _policy_rgbd(observation: dict[str, np.ndarray]) -> torch.Tensor:
            rgb = torch.as_tensor(
                observation["rgb"], dtype=torch.float32, device=context.device
            ).permute(0, 3, 1, 2).div_(255.0)
            rgb = F.interpolate(rgb, size=expected_shape, mode="area")
            depth = torch.as_tensor(
                observation["depth"], dtype=torch.float32, device=context.device
            ).unsqueeze(1)
            depth = F.interpolate(depth, size=expected_shape, mode="area")
            depth = (
                depth.sub_(DEPTH_MIN_M)
                .div_(DEPTH_MAX_M - DEPTH_MIN_M)
                .clamp_(0.0, 1.0)
            )
            rgbd = torch.cat((rgb, depth), dim=1)
            if tuple(rgbd.shape[-2:]) != expected_shape:
                raise RuntimeError(
                    f"Policy RGB-D shape {tuple(rgbd.shape[-2:])} != {expected_shape}."
                )
            return rgbd

        policy_goal_rgbd = _policy_rgbd(goal_observation)
        print(
            f"[POLICY]: loaded {args_cli.visual_servo_policy_checkpoint} "
            f"epoch={checkpoint.get('epoch')} input_rgbd={tuple(policy_goal_rgbd.shape)} "
            f"downsample={policy_downsample}",
            flush=True,
        )

    summary = {"episodes": [], "grasp_id": selected_grasp.grasp_id, "parallel_envs": env_count}
    policy_substeps = max(1, int(round(config.policy_dt_s / context.physics_dt)))
    sample_stride = max(
        1, int(round(config.policy_hz / float(args_cli.curriculum_sample_hz)))
    )
    effective_sample_hz = config.policy_hz / sample_stride
    summary["sample_hz"] = effective_sample_hz
    nominal_velocity = (nominal_grasp_position - nominal_pregrasp_position) / config.approach_duration_s
    approach_nominal_twist = np.concatenate((nominal_velocity, np.zeros(3, dtype=np.float64)))
    output_dir = Path(args_cli.curriculum_dataset_dir)
    total_episodes = int(args_cli.curriculum_episodes)
    curriculum_video_recorder = None

    with ThreadPoolExecutor(max_workers=int(args_cli.curriculum_writer_workers)) as writer_pool:
        for batch_start in range(0, total_episodes, env_count):
            valid_count = min(env_count, total_episodes - batch_start)
            if args_cli.curriculum_randomize_object_pose:
                dx_dy = rng.uniform(
                    -config.object_translation_xy_m,
                    config.object_translation_xy_m,
                    size=(env_count, 2),
                )
                yaws = np.deg2rad(
                    rng.uniform(-config.object_yaw_deg, config.object_yaw_deg, size=env_count)
                )
            else:
                dx_dy = np.zeros((env_count, 2), dtype=np.float64)
                yaws = np.zeros(env_count, dtype=np.float64)
            if args_cli.curriculum_fixed_object_offset_xy_m is not None:
                dx_dy[:] = np.asarray(
                    args_cli.curriculum_fixed_object_offset_xy_m, dtype=np.float64
                )
            if args_cli.curriculum_fixed_object_yaw_deg is not None:
                yaws[:] = np.deg2rad(
                    float(args_cli.curriculum_fixed_object_yaw_deg)
                )
            object_positions_w = nominal_positions_w.copy()
            object_positions_w[:, :2] += dx_dy
            object_quaternions = np.stack(
                [
                    _quat_multiply_xyzw(_yaw_quaternion_xyzw(float(yaw)), nominal_object_quaternion)
                    for yaw in yaws
                ]
            )
            _write_part_poses(object_positions_w, object_quaternions)
            perturb_q = pregrasp_q + torch.as_tensor(
                rng.uniform(
                    -config.initial_joint_noise_rad,
                    config.initial_joint_noise_rad,
                    size=tuple(pregrasp_q.shape),
                ),
                dtype=torch.float32,
                device=context.device,
            )
            context.reset_joint_state(perturb_q, q_hand=context.get_hand_q(), steps=5)
            _write_part_poses(object_positions_w, object_quaternions)

            pregrasp_positions_w = np.stack(
                [
                    object_positions_w[index]
                    + _rotate_z(
                        nominal_pregrasp_position - nominal_object_position, float(yaws[index])
                    )
                    for index in range(env_count)
                ]
            )
            grasp_positions_w = np.stack(
                [
                    object_positions_w[index]
                    + _rotate_z(nominal_grasp_position - nominal_object_position, float(yaws[index]))
                    for index in range(env_count)
                ]
            )
            grasp_quaternions = np.stack(
                [
                    _quat_multiply_xyzw(
                        _yaw_quaternion_xyzw(float(yaw)), nominal_grasp_quaternion
                    )
                    for yaw in yaws
                ]
            )
            position_noise_half_ranges = np.asarray(
                args_cli.curriculum_ee_position_noise_grasp_m, dtype=np.float64
            )
            rotation_noise_half_ranges_deg = np.asarray(
                args_cli.curriculum_ee_rotation_noise_deg, dtype=np.float64
            )
            ee_position_offsets_grasp_m = rng.uniform(
                -position_noise_half_ranges,
                position_noise_half_ranges,
                size=(env_count, 3),
            )
            ee_rotation_offsets_deg = rng.uniform(
                -rotation_noise_half_ranges_deg,
                rotation_noise_half_ranges_deg,
                size=(env_count, 3),
            )
            if args_cli.curriculum_fixed_ee_offset_grasp_m is not None:
                ee_position_offsets_grasp_m[:] = np.asarray(
                    args_cli.curriculum_fixed_ee_offset_grasp_m, dtype=np.float64
                )
            if args_cli.curriculum_fixed_ee_rotation_deg is not None:
                ee_rotation_offsets_deg[:] = np.asarray(
                    args_cli.curriculum_fixed_ee_rotation_deg, dtype=np.float64
                )
            apply_cartesian_ee_perturbation = bool(
                np.any(ee_position_offsets_grasp_m)
                or np.any(ee_rotation_offsets_deg)
            )
            if apply_cartesian_ee_perturbation:
                nominal_pregrasp_tcp_poses = [
                    context.grasp_pose_to_tcp_pose(
                        tuple(float(value) for value in pregrasp_positions_w[index]),
                        tuple(float(value) for value in grasp_quaternions[index]),
                    )
                    for index in range(env_count)
                ]
                stressed_tcp_poses = [
                    (
                        np.asarray(nominal_pregrasp_tcp_poses[index][0])
                        + _rotate_vector_by_quaternion_xyzw(
                            ee_position_offsets_grasp_m[index],
                            grasp_quaternions[index],
                        ),
                        _quat_multiply_xyzw(
                            np.asarray(nominal_pregrasp_tcp_poses[index][1]),
                            _euler_xyz_quaternion_xyzw(
                                np.deg2rad(ee_rotation_offsets_deg[index])
                            ),
                        ),
                    )
                    for index in range(env_count)
                ]
                stressed_grasp_poses = [
                    context.tcp_pose_to_grasp_pose(
                        tuple(float(value) for value in position),
                        tuple(float(value) for value in quaternion),
                    )
                    for position, quaternion in stressed_tcp_poses
                ]
                stress_position_t = torch.as_tensor(
                    np.stack([pose[0] for pose in stressed_grasp_poses]),
                    dtype=torch.float32,
                    device=context.device,
                )
                stress_quaternion_t = torch.as_tensor(
                    np.stack([pose[1] for pose in stressed_grasp_poses]),
                    dtype=torch.float32,
                    device=context.device,
                )
                stress_convergence_steps = max(
                    1, int(round(1.5 / context.physics_dt))
                )
                for _ in range(stress_convergence_steps):
                    context.command_pose_batch_via_differential_ik(
                        expert_ik, stress_position_t, stress_quaternion_t
                    )
                    context.command_fixed_gripper()
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(context.physics_dt)
                    _write_part_poses(object_positions_w, object_quaternions)
            if batch_start == 0 and args_cli.curriculum_video is not None:
                curriculum_video_recorder = IsaacVideoRecorder(
                    camera=wrist_camera,
                    sim=sim,
                    output_path=args_cli.curriculum_video,
                    fps=float(args_cli.video_fps),
                    width=int(args_cli.video_width),
                    height=int(args_cli.video_height),
                )
                curriculum_video_recorder.capture(force=True)
            command_position_t, command_quaternion_w_t = context.get_tcp_pose_w()
            command_positions_w = command_position_t.detach().cpu().numpy()
            command_quaternions_wxyz = command_quaternion_w_t.detach().cpu().numpy()
            command_quaternions_xyzw = command_quaternions_wxyz[:, (1, 2, 3, 0)]
            previous_positions_w = command_positions_w.copy()
            previous_quaternions_xyzw = command_quaternions_xyzw.copy()
            buffers: dict[str, list[np.ndarray]] = {
                name: []
                for name in (
                    "rgb_live",
                    "depth_live",
                    "object_mask",
                    "joint_positions",
                    "tcp_position_w",
                    "tcp_orientation_xyzw_w",
                    "object_position_w",
                    "object_orientation_xyzw_w",
                    "nominal_twist",
                    "expert_twist",
                    "expert_residual_twist",
                    "pose_error",
                    "trajectory_progress",
                    "funnel_half_width_m",
                    "funnel_transverse_error_m",
                    "funnel_approach_scale",
                    "funnel_near_phase",
                    "measured_tcp_twist",
                    "controller_stage",
                )
            }
            phase_elapsed_s = np.zeros(env_count, dtype=np.float64)
            precision_active = np.zeros(env_count, dtype=bool)
            precision_step_counts = np.zeros(env_count, dtype=np.int32)
            required_precision_steps = max(
                1, int(round(config.precision_duration_s * config.policy_hz))
            )
            maximum_step_count = config.step_count + int(
                round(
                    (
                        config.capture_duration_s
                        + config.approach_duration_s
                        + config.settle_duration_s
                    )
                    * config.policy_hz
                )
            )
            for step_index in range(maximum_step_count):
                if np.all(
                    precision_step_counts[:valid_count] >= required_precision_steps
                ):
                    break
                progress_and_rates = [
                    smooth_trajectory_progress(
                        phase_elapsed_s[index], config.approach_duration_s
                    )
                    for index in range(env_count)
                ]
                progresses = np.asarray(
                    [item[0] for item in progress_and_rates], dtype=np.float64
                )
                precision_active |= progresses >= config.precision_start_progress
                precision_step_counts += precision_active.astype(np.int32)
                effective_progresses = np.where(precision_active, 1.0, progresses)
                progress_rates = np.asarray(
                    [item[1] for item in progress_and_rates], dtype=np.float64
                )
                nominal_twists = (
                    approach_nominal_twist[None, :]
                    * (progress_rates * config.approach_duration_s)[:, None]
                )
                nominal_twists[precision_active] = 0.0
                target_positions_w = (
                    pregrasp_positions_w
                    + effective_progresses[:, None]
                    * (grasp_positions_w - pregrasp_positions_w)
                )
                target_tcp_poses = [
                    context.grasp_pose_to_tcp_pose(
                        tuple(float(value) for value in target_positions_w[index]),
                        tuple(float(value) for value in grasp_quaternions[index]),
                    )
                    for index in range(env_count)
                ]
                target_tcp_positions_w = np.asarray(
                    [item[0] for item in target_tcp_poses], dtype=np.float64
                )
                target_tcp_quaternions = np.asarray(
                    [item[1] for item in target_tcp_poses], dtype=np.float64
                )
                current_position_t, current_quaternion_w_t = context.get_tcp_pose_w()
                current_positions_w = current_position_t.detach().cpu().numpy()
                current_quaternions_wxyz = current_quaternion_w_t.detach().cpu().numpy()
                current_quaternions_xyzw = current_quaternions_wxyz[:, (1, 2, 3, 0)]
                measured_twists = np.stack(
                    [
                        _measured_world_twist(
                            previous_positions_w[index],
                            previous_quaternions_xyzw[index],
                            current_positions_w[index],
                            current_quaternions_xyzw[index],
                            config.policy_dt_s,
                        )
                        for index in range(env_count)
                    ]
                )
                previous_positions_w = current_positions_w.copy()
                previous_quaternions_xyzw = current_quaternions_xyzw.copy()
                errors = np.stack(
                    [
                        pose_error_twist(
                            current_positions_w[index],
                            current_quaternions_xyzw[index],
                            target_tcp_positions_w[index],
                            target_tcp_quaternions[index],
                        )
                        for index in range(env_count)
                    ]
                )
                actions = []
                for index in range(env_count):
                    if precision_active[index]:
                        full_twist, precision_debug = precision_docking_expert_twist(
                            pose_error=errors[index],
                            measured_twist=measured_twists[index],
                            config=config,
                        )
                        actions.append(
                            (
                                full_twist,
                                full_twist.copy(),
                                {
                                    "near_phase": 1.0,
                                    "funnel_half_width_m": config.funnel_near_half_width_m,
                                    "transverse_error_m": precision_debug["position_error_m"],
                                    "approach_scale": 1.0,
                                },
                            )
                        )
                    else:
                        actions.append(
                            alignment_funnel_expert_twist(
                                nominal_twist=nominal_twists[index],
                                pose_error=errors[index],
                                grasp_orientation_xyzw=grasp_quaternions[index],
                                trajectory_progress=progresses[index],
                                config=config,
                                measured_twist=measured_twists[index],
                            )
                        )
                if learned_policy is not None:
                    policy_observation = _curriculum_camera_observation(
                        wrist_camera, physics_dt=context.physics_dt
                    )
                    nominal_camera = world_twist_to_camera(
                        nominal_twists, current_quaternions_xyzw
                    ).astype(np.float32)
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        predicted_normalized = learned_policy(
                            live_rgbd=_policy_rgbd(policy_observation),
                            goal_rgbd=policy_goal_rgbd,
                            joint_positions=context.get_arm_q(),
                            progress=torch.as_tensor(
                                effective_progresses[:, None],
                                dtype=torch.float32,
                                device=context.device,
                            ),
                            nominal_twist_camera=torch.as_tensor(
                                normalize_twist(nominal_camera),
                                dtype=torch.float32,
                                device=context.device,
                            ),
                        )
                    action_scale = np.asarray(
                        [LINEAR_ACTION_SCALE_M_S] * 3
                        + [ANGULAR_ACTION_SCALE_RAD_S] * 3,
                        dtype=np.float32,
                    )
                    residual_camera = (
                        predicted_normalized.float().cpu().numpy() * action_scale
                    )
                    residual_world = camera_twist_to_world(
                        residual_camera, current_quaternions_xyzw
                    )
                    full_twists = nominal_twists + residual_world
                    residual_twists = residual_world
                    funnel_diagnostics = [
                        {
                            "near_phase": float(precision_active[index]),
                            "funnel_half_width_m": config.funnel_near_half_width_m,
                            "transverse_error_m": float(
                                np.linalg.norm(errors[index, :3])
                            ),
                            "approach_scale": (
                                0.0 if precision_active[index] else 1.0
                            ),
                        }
                        for index in range(env_count)
                    ]
                else:
                    full_twists = np.stack([action[0] for action in actions])
                    residual_twists = np.stack([action[1] for action in actions])
                    funnel_diagnostics = [action[2] for action in actions]
                phase_elapsed_s = np.minimum(
                    config.approach_duration_s,
                    phase_elapsed_s
                    + config.policy_dt_s
                    * np.asarray(
                        [
                            0.0 if precision_active[index] else item["approach_scale"]
                            for index, item in enumerate(funnel_diagnostics)
                        ],
                        dtype=np.float64,
                    ),
                )
                record_sample = (
                    step_index % sample_stride == 0
                    or np.all(
                        precision_step_counts[:valid_count]
                        >= required_precision_steps
                    )
                )
                if record_sample:
                    observation = _curriculum_camera_observation(
                        wrist_camera, physics_dt=context.physics_dt
                    )
                    part_pose_w = part.data.root_pose_w.detach().cpu().numpy()
                    buffers["rgb_live"].append(observation["rgb"])
                    buffers["depth_live"].append(observation["depth"])
                    buffers["object_mask"].append(observation["object_mask"])
                    buffers["joint_positions"].append(
                        context.get_arm_q().detach().cpu().numpy()
                    )
                    buffers["tcp_position_w"].append(
                        current_positions_w.astype(np.float32)
                    )
                    buffers["tcp_orientation_xyzw_w"].append(
                        current_quaternions_xyzw.astype(np.float32)
                    )
                    buffers["object_position_w"].append(
                        part_pose_w[:, :3].astype(np.float32)
                    )
                    buffers["object_orientation_xyzw_w"].append(
                        part_pose_w[:, (4, 5, 6, 3)].astype(np.float32)
                    )
                    buffers["nominal_twist"].append(
                        nominal_twists.astype(np.float32)
                    )
                    buffers["expert_twist"].append(full_twists.astype(np.float32))
                    buffers["expert_residual_twist"].append(
                        residual_twists.astype(np.float32)
                    )
                    buffers["pose_error"].append(errors.astype(np.float32))
                    buffers["trajectory_progress"].append(
                        effective_progresses.astype(np.float32)
                    )
                    buffers["measured_tcp_twist"].append(
                        measured_twists.astype(np.float32)
                    )
                    buffers["controller_stage"].append(
                        precision_active.astype(np.int8)
                    )
                    for name, key in (
                        ("funnel_half_width_m", "funnel_half_width_m"),
                        ("funnel_transverse_error_m", "transverse_error_m"),
                        ("funnel_approach_scale", "approach_scale"),
                        ("funnel_near_phase", "near_phase"),
                    ):
                        buffers[name].append(
                            np.asarray(
                                [item[key] for item in funnel_diagnostics],
                                dtype=np.float32,
                            )
                        )

                integrated_commands = [
                    _integrate_world_twist_pose(
                        command_positions_w[index],
                        command_quaternions_xyzw[index],
                        full_twists[index],
                        config.policy_dt_s,
                    )
                    for index in range(env_count)
                ]
                command_positions_w = np.stack(
                    [
                        _limit_position_command_lead(
                            item[0],
                            current_positions_w[index],
                            max_lead_m=(
                                config.precision_max_command_lead_m
                                if precision_active[index]
                                else 0.006
                            ),
                        )
                        for index, item in enumerate(integrated_commands)
                    ]
                )
                command_quaternions_xyzw = np.stack(
                    [item[1] for item in integrated_commands]
                )
                integrated_grasp_commands = [
                    context.tcp_pose_to_grasp_pose(
                        tuple(float(value) for value in item[0]),
                        tuple(float(value) for value in item[1]),
                    )
                    for item in integrated_commands
                ]
                target_position_t = torch.as_tensor(
                    np.stack([item[0] for item in integrated_grasp_commands]),
                    dtype=torch.float32,
                    device=context.device,
                )
                target_quaternion_t = torch.as_tensor(
                    np.stack([item[1] for item in integrated_grasp_commands]),
                    dtype=torch.float32,
                    device=context.device,
                )
                for _ in range(policy_substeps):
                    context.command_pose_batch_via_differential_ik(
                        expert_ik, target_position_t, target_quaternion_t
                    )
                    context.command_fixed_gripper()
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(context.physics_dt)
                    _write_part_poses(object_positions_w, object_quaternions)
                    if curriculum_video_recorder is not None:
                        curriculum_video_recorder.capture()

            final_position_t, final_quaternion_w_t = context.get_tcp_pose_w()
            final_positions_w = final_position_t.detach().cpu().numpy()
            final_quaternions_wxyz = final_quaternion_w_t.detach().cpu().numpy()
            final_quaternions_xyzw = final_quaternions_wxyz[:, (1, 2, 3, 0)]
            final_errors = np.stack(
                [
                    pose_error_twist(
                        final_positions_w[index],
                        final_quaternions_xyzw[index],
                        np.asarray(
                            context.grasp_pose_to_tcp_pose(
                                tuple(float(value) for value in grasp_positions_w[index]),
                                tuple(float(value) for value in grasp_quaternions[index]),
                            )[0]
                        ),
                        np.asarray(
                            context.grasp_pose_to_tcp_pose(
                                tuple(float(value) for value in grasp_positions_w[index]),
                                tuple(float(value) for value in grasp_quaternions[index]),
                            )[1]
                        ),
                    )
                    for index in range(env_count)
                ]
            )
            stacked = {name: np.stack(values, axis=0) for name, values in buffers.items()}
            futures = []
            episode_summaries = []
            for env_index in range(valid_count):
                episode_index = (
                    int(args_cli.curriculum_episode_offset)
                    + batch_start
                    + env_index
                )
                final_error = final_errors[env_index]
                success = bool(
                    np.linalg.norm(final_error[:3]) <= config.success_position_tolerance_m
                    and np.linalg.norm(final_error[3:])
                    <= np.deg2rad(config.success_rotation_tolerance_deg)
                )
                arrays = {name: values[:, env_index].copy() for name, values in stacked.items()}
                arrays.update(
                    {
                        "rgb_goal": goal_observation["rgb"][env_index].copy(),
                        "depth_goal": goal_observation["depth"][env_index].copy(),
                        "goal_object_mask": goal_observation["object_mask"][env_index].copy(),
                        "goal_tcp_position_w": goal_tcp_positions_w[env_index].copy(),
                        "goal_tcp_orientation_xyzw_w": goal_tcp_quaternions_xyzw[env_index].copy(),
                        "goal_object_position_w": goal_part_pose_w[env_index, :3].copy(),
                        "goal_object_orientation_xyzw_w": goal_part_pose_w[
                            env_index, (4, 5, 6, 3)
                        ].copy(),
                    }
                )
                metadata = {
                    "success": success,
                    "final_position_error_m": float(np.linalg.norm(final_error[:3])),
                    "final_rotation_error_deg": float(
                        np.rad2deg(np.linalg.norm(final_error[3:]))
                    ),
                    "object_perturbation": {
                        "dx_m": float(dx_dy[env_index, 0]),
                        "dy_m": float(dx_dy[env_index, 1]),
                        "yaw_deg": float(np.rad2deg(yaws[env_index])),
                    },
                    "initial_ee_stress": {
                        "offset_grasp_m": ee_position_offsets_grasp_m[
                            env_index
                        ].tolist(),
                        "rotation_xyz_deg": ee_rotation_offsets_deg[
                            env_index
                        ].tolist(),
                    },
                    "grasp_id": selected_grasp.grasp_id,
                    "action_frame": "world",
                    "action_semantics": "end_effector_twist",
                    "controller": (
                        "learned_policy"
                        if learned_policy is not None
                        else "privileged_expert"
                    ),
                    "parallel_env_index": env_index,
                    "parallel_env_count": env_count,
                    "collection_sample_hz": effective_sample_hz,
                }
                futures.append(
                    writer_pool.submit(
                        write_episode_npz,
                        output_dir,
                        episode_index=episode_index,
                        arrays=arrays,
                        metadata=metadata,
                        config=config,
                    )
                )
                episode_summaries.append(
                    {
                        "episode": episode_index,
                        "success": success,
                        "final_position_error_m": metadata["final_position_error_m"],
                        "final_rotation_error_deg": metadata["final_rotation_error_deg"],
                    }
                )
            for future, episode_summary in zip(futures, episode_summaries, strict=True):
                npz_path, _ = future.result()
                episode_summary["path"] = str(npz_path)
                summary["episodes"].append(episode_summary)
                print(f"[CURRICULUM]: {episode_summary}", flush=True)
            print(
                f"[CURRICULUM]: completed batch {batch_start // env_count + 1} "
                f"({batch_start + valid_count}/{total_episodes} episodes)",
                flush=True,
            )
            if curriculum_video_recorder is not None:
                curriculum_video_recorder.capture(force=True)
                curriculum_video_recorder.close()
                curriculum_video_recorder = None

    summary["success_count"] = sum(bool(item["success"]) for item in summary["episodes"])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def resolve_part_usd_path(*, bundle, mesh_local) -> str:
    if args_cli.use_provided_part_usd:
        if not args_cli.part_usd:
            raise ValueError("--part-usd is required when --use-provided-part-usd is set.")
        print(
            "[WARN]: Using provided part USD directly. It must already be authored in the saved bundle-local frame.",
            flush=True,
        )
        return args_cli.part_usd

    output_dir = Path("artifacts/isaac_bundle_assets").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_usd = output_dir / f"{args_cli.input_json.stem}_bundle_local.usd"
    temp_stl = write_temporary_triangle_mesh_stl(
        mesh_local,
        prefix=f"{args_cli.input_json.stem}_bundle_local_",
        dir=output_dir,
    )
    converter_cfg = MeshConverterCfg(
        asset_path=str(temp_stl),
        usd_dir=str(output_dir),
        usd_file_name=output_usd.name,
        force_usd_conversion=True,
        make_instanceable=False,
        scale=(1.0, 1.0, 1.0),
        mass_props=_object_mass_properties_cfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=ISAAC_MIN_CONTACT_OFFSET_M,
            rest_offset=0.0,
        ),
        mesh_collision_props=_mesh_collision_cfg(),
    )
    converter = MeshConverter(converter_cfg)
    converted_path = str(Path(converter.usd_path).resolve())
    print(
        f"[INFO]: Generated bundle-local Isaac part USD from bundle={bundle.target_stl_path} output={converted_path}",
        flush=True,
    )
    try:
        temp_stl.unlink()
    except FileNotFoundError:
        pass
    return converted_path


def build_scene(
    *, object_pose_world, part_usd_path: str
) -> tuple[sim_utils.SimulationContext, InteractiveScene, Camera | None]:
    print("[INFO]: Creating simulation context...", flush=True)
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        render=make_visual_servo_render_cfg(),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim._app_control_on_stop_handle = None
    sim._disable_app_control_on_stop_handle = True
    sim.set_camera_view([1.6, -1.2, 1.0], [0.35, 0.0, 0.3])

    print("[INFO]: Building Franka Panda + part scene config...", flush=True)
    fr3_usd_path = resolve_fr3_usd_path()
    print(f"[INFO]: Isaac robot USD: {fr3_usd_path}", flush=True)
    wrist_camera_config = D405WristCameraConfig(
        enabled=bool(
            args_cli.enable_d405_wrist_camera
            or args_cli.record_video is not None
            or args_cli.visual_servo_goal_image is not None
            or args_cli.visual_servo_comparison_video is not None
        ),
        width=int(args_cli.d405_width),
        height=int(args_cli.d405_height),
        fx=float(args_cli.d405_fx),
        fy=float(args_cli.d405_fy),
        cx=float(args_cli.d405_cx),
        cy=float(args_cli.d405_cy),
        include_privileged_mask=not bool(args_cli.d405_disable_privileged_mask),
    )
    scene_cfg = make_fr3_part_scene_cfg(
        fr3_asset_path=fr3_usd_path,
        part_usd_path=part_usd_path,
        part_position=object_pose_world.position_world,
        part_orientation_xyzw=object_pose_world.orientation_xyzw_world,
        part_mass_kg=None if args_cli.object_mass_kg is None else float(args_cli.object_mass_kg),
        part_density_kg_m3=_effective_object_density_kg_m3(),
        robot_base_position=ROBOT_BASE_POSITION,
        robot_base_orientation_xyzw=ROBOT_BASE_ORIENTATION_XYZW,
    )
    curriculum_env_count = (
        min(int(args_cli.curriculum_num_envs), int(args_cli.curriculum_episodes))
        if args_cli.curriculum_dataset_dir is not None
        else 1
    )
    scene_cfg.num_envs = curriculum_env_count
    print("[INFO]: Creating interactive scene...", flush=True)
    scene = InteractiveScene(scene_cfg)
    print("[INFO]: Waiting for stage assets to finish loading...", flush=True)
    while omni.usd.get_context().get_stage_loading_status()[2] > 0:
        simulation_app.update()
    wrist_camera = None
    if wrist_camera_config.enabled:
        stage = omni.usd.get_context().get_stage()
        link7_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if str(prim.GetPath()).endswith("/link7") and "/Robot/" in str(prim.GetPath())
        ]
        if len(link7_paths) != curriculum_env_count:
            raise RuntimeError(
                f"Expected {curriculum_env_count} loaded robot link7 prims, found {link7_paths}."
            )
        wrist_camera = Camera(
            cfg=make_d405_wrist_camera_cfg(
                parent_prim_path="/World/envs/env_.*/Robot/link7",
                wrist_camera=wrist_camera_config,
            )
        )
        debug_housing_paths = []
        if curriculum_env_count == 1:
            debug_housing_paths.append(
                _add_d405_debug_housing(f"{link7_paths[0]}/D405LeftCamera")
            )
        debug_text = (
            f" with visible debug housing at {debug_housing_paths[0]}"
            if debug_housing_paths
            else " (debug housings disabled for batched collection)"
        )
        print(
            f"[INFO]: Attached D405 wrist camera under {len(link7_paths)} cloned link7 prims"
            f"{debug_text}.",
            flush=True,
        )
    material_bindings = _apply_matte_pla_materials()
    print(f"[INFO]: Applied matte PLA visual materials: {material_bindings}.", flush=True)
    if _is_generated_kuka_y_gripper_usd(fr3_usd_path):
        print(
            "[INFO]: Skipping Franka visual mesh collision exposure for generated KUKA/Y-gripper USD; "
            "using authored collision hulls.",
            flush=True,
        )
    else:
        enabled_collision_count, _enabled_collision_paths = expose_franka_mesh_collisions(
            mesh_path_patterns=(
                r"(?:panda|fr3)_hand",
                r"(?:panda|fr3)_leftfinger",
                r"(?:panda|fr3)_rightfinger",
                r"finger",
                r"gripper",
            )
        )
        if enabled_collision_count > 0:
            print(
                f"[INFO]: Enabled Isaac collision on {enabled_collision_count} Franka gripper mesh prims.",
                flush=True,
            )
        else:
            print(
                "[WARN]: No Franka gripper mesh prims were found while enabling Isaac robot collisions; "
                "gripper-object contacts may be missing.",
                flush=True,
            )
    stage = omni.usd.get_context().get_stage()
    bound_contact_material_count = _bind_high_friction_contact_material(
        stage,
        root_paths=tuple(
            path
            for env_index in range(curriculum_env_count)
            for path in (
                f"/World/envs/env_{env_index}/Robot",
                f"/World/envs/env_{env_index}/Part",
            )
        ),
    )
    print(
        "[INFO]: Bound high-friction Isaac contact material "
        f"to {bound_contact_material_count} robot/part collision subtrees.",
        flush=True,
    )
    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Resetting scene buffers...", flush=True)
    scene.reset()
    _write_kuka_configured_start_state(scene)
    print("[INFO]: Scene ready.", flush=True)
    return sim, scene, wrist_camera


def _write_attempt_artifact(
    *,
    bundle,
    placement_spec,
    object_pose_world,
    statuses,
    selected_world_grasp,
    execution_result,
    video_recorder=None,
) -> None:
    artifact = {
        "input_json": str(args_cli.input_json),
        "target_stl_path": bundle.target_stl_path,
        "part_usd": getattr(args_cli, "resolved_part_usd", args_cli.part_usd),
        "placement": {
            "support_face": placement_spec.support_face,
            "yaw_deg": placement_spec.yaw_deg,
            "xy_world": list(placement_spec.xy_world),
            "object_position_world": list(object_pose_world.position_world),
            "object_orientation_xyzw_world": list(object_pose_world.orientation_xyzw_world),
        },
        "counts": {
            "saved": len(bundle.candidates),
            "ground_feasible": len(accepted_grasps(statuses)),
        },
        "selected_grasp_id": None if selected_world_grasp is None else selected_world_grasp.grasp_id,
        "selected_world_grasp": None
        if selected_world_grasp is None
        else {
            "position_w": list(selected_world_grasp.position_w),
            "orientation_xyzw": list(selected_world_grasp.orientation_xyzw),
            "normal_w": list(selected_world_grasp.normal_w),
            "pregrasp_position_w": list(selected_world_grasp.pregrasp_position_w),
            "jaw_width": selected_world_grasp.jaw_width,
            "gripper_width": selected_world_grasp.gripper_width,
        },
        "execution": {
            "controller": args_cli.controller,
            "success": execution_result.success,
            "status": execution_result.status,
            "message": execution_result.message,
        },
    }
    object_lift_height_m = getattr(execution_result, "object_lift_height_m", None)
    target_lift_height_m = getattr(execution_result, "target_lift_height_m", None)
    if object_lift_height_m is not None:
        artifact["execution"]["object_lift_height_m"] = object_lift_height_m
    if target_lift_height_m is not None:
        artifact["execution"]["target_lift_height_m"] = target_lift_height_m
    diagnostics = getattr(execution_result, "diagnostics", None)
    if diagnostics:
        artifact["execution"]["diagnostics"] = dict(diagnostics)
    if video_recorder is not None:
        artifact["video"] = {
            "path": video_recorder.output_path,
            "frame_count": video_recorder.frame_count,
            "fps": float(args_cli.video_fps),
            "width": int(args_cli.video_width),
            "height": int(args_cli.video_height),
        }
    artifact["moveit"] = {
        "frame_id": args_cli.moveit_frame_id,
        "target_position_signs": list(_moveit_target_position_signs_from_args()),
        "tcp_to_grasp_offset": [float(value) for value in args_cli.tcp_to_grasp_offset],
        "planning_group": args_cli.moveit_planning_group,
        "pose_link": args_cli.moveit_pose_link,
        "namespace": args_cli.moveit_namespace,
        "joint_names": list(_moveit_joint_names_from_args()),
        "pipeline_id": args_cli.moveit_pipeline_id,
        "planner_id": args_cli.moveit_planner_id,
        "lift_height_m": args_cli.moveit_lift_height_m,
        "allow_collisions": bool(args_cli.moveit_allow_collisions),
        "plan_json": None if args_cli.moveit_plan_json is None else str(args_cli.moveit_plan_json),
    }
    output = args_cli.attempt_artifact
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def _candidate_world_grasp(grasp, object_pose_world):
    return saved_grasp_to_world_grasp(
        grasp,
        object_pose_world,
        pregrasp_offset=args_cli.pregrasp_offset,
        gripper_width_clearance=args_cli.gripper_width_clearance,
    )


def _select_executable_grasp(feasible_grasps, statuses, object_pose_world):
    candidates = _ordered_executable_grasp_candidates(feasible_grasps, statuses, object_pose_world)
    if not candidates:
        return None, None
    return candidates[0]


def _ordered_executable_grasp_candidates(feasible_grasps, statuses, object_pose_world):
    min_pregrasp_z = 0.05
    if args_cli.grasp_id:
        selected = next((grasp for grasp in feasible_grasps if grasp.grasp_id == args_cli.grasp_id), None)
        if selected is None:
            raise GraspSelectionFailure(
                status="requested_grasp_not_feasible",
                message=f"Requested grasp id '{args_cli.grasp_id}' is not ground-feasible for this pickup pose.",
            )
        world_grasp = _candidate_world_grasp(selected, object_pose_world)
        if world_grasp.pregrasp_position_w[2] <= min_pregrasp_z:
            raise GraspSelectionFailure(
                status="invalid_pregrasp",
                message=(
                    f"Requested grasp id '{args_cli.grasp_id}' has unsafe pregrasp height: "
                    f"pregrasp_position_w={world_grasp.pregrasp_position_w} required_min_z={min_pregrasp_z:.3f}"
                ),
                world_grasp=world_grasp,
            )
        return [(selected, world_grasp)]

    ordered = sorted(
        feasible_grasps, key=lambda grasp: float("-inf") if grasp.score is None else grasp.score, reverse=True
    )
    candidates = []
    for grasp in ordered:
        world_grasp = _candidate_world_grasp(grasp, object_pose_world)
        if world_grasp.pregrasp_position_w[2] > min_pregrasp_z:
            candidates.append((grasp, world_grasp))
    if candidates:
        return candidates

    fallback = select_first_feasible_grasp(statuses)
    if fallback is None:
        return []
    world_fallback = _candidate_world_grasp(fallback, object_pose_world)
    if world_fallback.pregrasp_position_w[2] <= min_pregrasp_z:
        return []
    return [(fallback, world_fallback)]


def run() -> None:
    configure_grasp_tcp_calibration()
    print("[INFO]: Loading grasp bundle...", flush=True)
    bundle = load_grasp_bundle(args_cli.input_json)
    print(f"[INFO]: Loaded {len(bundle.candidates)} saved grasps from '{args_cli.input_json}'.", flush=True)
    print("[INFO]: Loading local STL mesh for placement and filtering...", flush=True)
    mesh_local = _mesh_in_bundle_local_frame(bundle)
    args_cli.resolved_part_usd = resolve_part_usd_path(bundle=bundle, mesh_local=mesh_local)
    print("[INFO]: Resolving pickup placement...", flush=True)
    placement_spec, object_pose_world = _resolve_placement_and_pose(bundle, mesh_local)
    print(
        "[INFO]: Pickup placement "
        f"support_face={placement_spec.support_face} yaw_deg={placement_spec.yaw_deg:.1f} "
        f"xy_world={placement_spec.xy_world} object_pose_world={object_pose_world.position_world} "
        f"orientation_xyzw={object_pose_world.orientation_xyzw_world}",
        flush=True,
    )

    print("[INFO]: Rechecking saved grasps against the selected pickup pose...", flush=True)
    gripper_collision_model = _gripper_collision_model_from_args_or_bundle(bundle)
    statuses = evaluate_saved_grasps_against_pickup_pose(
        bundle.candidates,
        object_pose_world=object_pose_world,
        contact_gap_m=args_cli.detailed_finger_contact_gap_m,
        gripper_collision_model=gripper_collision_model,
    )
    rescored_feasible = score_grasps(accepted_grasps(statuses), mesh_local=mesh_local)
    rescored_by_id = {grasp.grasp_id: grasp for grasp in rescored_feasible}
    statuses = [
        type(entry)(
            grasp=rescored_by_id.get(entry.grasp.grasp_id, entry.grasp),
            status=entry.status,
            reason=entry.reason,
        )
        for entry in statuses
    ]
    feasible_grasps = accepted_grasps(statuses)
    try:
        executable_candidates = _ordered_executable_grasp_candidates(feasible_grasps, statuses, object_pose_world)
    except GraspSelectionFailure as exc:
        result = SimpleNamespace(success=False, status=exc.status, message=str(exc))
        _write_attempt_artifact(
            bundle=bundle,
            placement_spec=placement_spec,
            object_pose_world=object_pose_world,
            statuses=statuses,
            selected_world_grasp=exc.world_grasp,
            execution_result=result,
        )
        raise RuntimeError(result.message) from exc
    print(
        f"[INFO]: Ground recheck complete. feasible={len(feasible_grasps)} / saved={len(bundle.candidates)}",
        flush=True,
    )
    if not executable_candidates:
        result = SimpleNamespace(
            success=False,
            status="no_feasible_grasp",
            message="No saved grasp survives the pickup-ground recheck for the sampled placement.",
        )
        _write_attempt_artifact(
            bundle=bundle,
            placement_spec=placement_spec,
            object_pose_world=object_pose_world,
            statuses=statuses,
            selected_world_grasp=None,
            execution_result=result,
        )
        raise RuntimeError(result.message)
    selected_grasp, selected_world_grasp = executable_candidates[0]
    print(
        "[INFO]: Initial grasp candidate "
        f"id={selected_grasp.grasp_id} grasp_w={selected_world_grasp.position_w} "
        f"pregrasp_w={selected_world_grasp.pregrasp_position_w} "
        f"orientation_xyzw={selected_world_grasp.orientation_xyzw} "
        f"gripper_width={selected_world_grasp.gripper_width:.4f}",
        flush=True,
    )

    sim, scene, wrist_camera = build_scene(
        object_pose_world=object_pose_world, part_usd_path=args_cli.resolved_part_usd
    )
    physics_dt = sim.get_physics_dt()
    video_recorder = None
    if wrist_camera is not None and args_cli.record_video is not None:
        video_recorder = IsaacVideoRecorder(
            camera=wrist_camera,
            sim=sim,
            output_path=args_cli.record_video,
            fps=float(args_cli.video_fps),
            width=int(args_cli.video_width),
            height=int(args_cli.video_height),
        )
        video_recorder.capture(force=True)

    def _record_step() -> None:
        if video_recorder is not None:
            video_recorder.capture()

    approach_open_gripper_width = _approach_open_gripper_width(selected_world_grasp)
    print("[INFO]: Warming up simulation...", flush=True)
    for _ in range(max(1, int(0.1 / physics_dt))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)
        _record_step()
    _prepare_robot_start_pose(
        sim,
        scene,
        hand_open_width=approach_open_gripper_width,
        step_callback=_record_step,
    )

    robot = scene["robot"]
    part = scene["part"]
    part_pose_w = part.data.root_link_pose_w[0]
    print(
        "[INFO]: Spawned Isaac part pose "
        f"position=({float(part_pose_w[0]):.6f}, {float(part_pose_w[1]):.6f}, {float(part_pose_w[2]):.6f}) "
        f"orientation_wxyz=({float(part_pose_w[3]):.6f}, {float(part_pose_w[4]):.6f}, "
        f"{float(part_pose_w[5]):.6f}, {float(part_pose_w[6]):.6f})",
        flush=True,
    )
    print("[INFO]: Executing pick attempt...", flush=True)
    if args_cli.moveit_plan_json is not None:
        print(f"[INFO]: Loading precomputed MoveIt plan from {args_cli.moveit_plan_json}.", flush=True)
        moveit_plan_payload, moveit_joint_trajectories = _load_moveit_plan_json(args_cli.moveit_plan_json)
        planned_grasp_id = moveit_plan_payload.get("selected_grasp_id")
        if planned_grasp_id not in ("", None, selected_grasp.grasp_id):
            raise RuntimeError(
                "Precomputed MoveIt plan grasp id "
                f"'{planned_grasp_id}' does not match selected Isaac grasp '{selected_grasp.grasp_id}'."
            )
    else:
        start_joint_positions = _moveit_start_joint_positions_from_args()
        if start_joint_positions is None:
            start_joint_positions = _current_isaac_arm_joint_positions(
                sim=sim,
                scene=scene,
                robot=robot,
                fixed_gripper_width=approach_open_gripper_width,
            )
            start_joint_positions = _isaac_waypoint_to_moveit(start_joint_positions)
        moveit_joint_trajectories = None
        planning_errors: list[str] = []
        for candidate_grasp, candidate_world_grasp in executable_candidates:
            print(
                f"[INFO]: Planning Isaac attempt with MoveIt for grasp {candidate_grasp.grasp_id}.",
                flush=True,
            )
            try:
                moveit_joint_trajectories = _plan_moveit_joint_trajectories(
                    world_grasp=candidate_world_grasp,
                    start_joint_positions=start_joint_positions,
                )
            except RuntimeError as exc:
                planning_errors.append(f"{candidate_grasp.grasp_id}: {exc}")
                print(f"[WARN]: MoveIt rejected grasp {candidate_grasp.grasp_id}: {exc}", flush=True)
                continue
            selected_grasp = candidate_grasp
            selected_world_grasp = candidate_world_grasp
            print(f"[INFO]: Selected MoveIt-planned grasp {selected_grasp.grasp_id}.", flush=True)
            break
        if moveit_joint_trajectories is None:
            tried = ", ".join(planning_errors[:8])
            if len(planning_errors) > 8:
                tried += f", ... ({len(planning_errors)} total)"
            result = SimpleNamespace(
                success=False,
                status="moveit_planning_failed",
                message=f"MoveIt could not plan any ground-feasible Isaac grasp. Tried: {tried}",
            )
            _write_attempt_artifact(
                bundle=bundle,
                placement_spec=placement_spec,
                object_pose_world=object_pose_world,
                statuses=statuses,
                selected_world_grasp=selected_world_grasp,
                execution_result=result,
            )
            raise RuntimeError(result.message)
    effective_close_width = _effective_close_gripper_width(selected_grasp)
    print(
        "[INFO]: Isaac gripper command sequence "
        f"approach_open_width={approach_open_gripper_width:.4f} "
        f"candidate_grasp_width={selected_world_grasp.gripper_width:.4f} "
        f"requested_close_width={float(args_cli.close_width):.4f} "
        f"effective_close_width={effective_close_width:.4f}.",
        flush=True,
    )
    _print_moveit_joint_trajectory_summary(moveit_joint_trajectories)
    if args_cli.curriculum_dataset_dir is not None:
        if wrist_camera is None:
            raise RuntimeError("Curriculum generation requires the D405 wrist camera.")
        curriculum_summary = _run_batched_first_curriculum(
            sim=sim,
            scene=scene,
            wrist_camera=wrist_camera,
            moveit_joint_trajectories=moveit_joint_trajectories,
            selected_grasp=selected_grasp,
            selected_world_grasp=selected_world_grasp,
            object_pose_world=object_pose_world,
            open_gripper_width=approach_open_gripper_width,
        )
        if video_recorder is not None:
            video_recorder.close()
        print(f"[INFO]: First curriculum generation complete: {curriculum_summary}.", flush=True)
        return

    goal_image_path = args_cli.visual_servo_goal_image
    visual_servo_start_frame = 0
    visual_servo_end_frame = None

    def _mark_visual_servo_pregrasp() -> None:
        nonlocal visual_servo_start_frame
        if video_recorder is not None:
            video_recorder.capture(force=True)
            visual_servo_start_frame = max(0, video_recorder.frame_count - 1)
        print(
            f"[INFO]: Visual-servo demonstration starts at pregrasp frame "
            f"{visual_servo_start_frame}.",
            flush=True,
        )

    def _capture_visual_servo_goal() -> None:
        nonlocal visual_servo_end_frame
        if goal_image_path is None:
            return
        if wrist_camera is None:
            raise RuntimeError("Visual-servo goal rendering requires the D405 wrist camera.")
        goal_rgb = _camera_rgb_frame(wrist_camera, physics_dt=physics_dt)
        _write_rgb_image(goal_image_path, goal_rgb)
        if video_recorder is not None:
            video_recorder.capture(force=True)
            visual_servo_end_frame = max(visual_servo_start_frame, video_recorder.frame_count - 1)
        metadata_path = goal_image_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "image": str(goal_image_path),
                    "observation": "D405 RGB at planned grasp waypoint before gripper closure",
                    "robot_phase": "grasp",
                    "gripper_state": "open",
                    "visual_servo_approach_start_phase": "pregrasp",
                    "selected_grasp_id": selected_grasp.grasp_id,
                    "object_pose_world": {
                        "position": list(object_pose_world.position_world),
                        "orientation_xyzw": list(object_pose_world.orientation_xyzw_world),
                    },
                    "moveit_grasp_joint_waypoint": list(moveit_joint_trajectories["grasp"][-1]),
                    "privileged_training_label_only": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[INFO]: Rendered visual-servo goal image to {goal_image_path}.", flush=True)

    execution_result = execute_pick_from_moveit_joint_trajectories(
        sim=sim,
        scene=scene,
        robot=robot,
        object_asset=part,
        moveit_joint_trajectories=moveit_joint_trajectories,
        open_gripper_width=approach_open_gripper_width,
        closed_gripper_width=effective_close_width,
        pregrasp_only=bool(args_cli.pregrasp_only),
        success_height_margin_m=float(args_cli.success_height_margin_m),
        max_joint_speed_rad_s=float(args_cli.moveit_execution_speed_rad_s),
        grasp_settle_time_s=float(args_cli.moveit_grasp_settle_time_s),
        gripper_close_duration_s=float(args_cli.gripper_close_duration_s),
        gripper_close_max_duration_s=float(args_cli.gripper_close_max_duration_s),
        postclose_hold_s=float(args_cli.postclose_hold_s),
        selected_gripper_width_m=float(selected_world_grasp.jaw_width),
        step_callback=_record_step,
        pregrasp_observation_callback=_mark_visual_servo_pregrasp,
        grasp_observation_callback=_capture_visual_servo_goal,
    )
    if video_recorder is not None:
        video_recorder.capture(force=True)
    _write_attempt_artifact(
        bundle=bundle,
        placement_spec=placement_spec,
        object_pose_world=object_pose_world,
        statuses=statuses,
        selected_world_grasp=selected_world_grasp,
        execution_result=execution_result,
        video_recorder=video_recorder,
    )
    if video_recorder is not None:
        video_recorder.close()
    if args_cli.visual_servo_comparison_video is not None:
        if goal_image_path is None:
            raise RuntimeError("Comparison video requires --visual-servo-goal-image.")
        comparison_frame_count = _write_live_goal_comparison_video(
            live_video_path=args_cli.record_video,
            goal_image_path=goal_image_path,
            output_path=args_cli.visual_servo_comparison_video,
            fps=float(args_cli.video_fps),
            start_frame=visual_servo_start_frame,
            end_frame=visual_servo_end_frame,
        )
        print(
            f"[INFO]: Wrote visual-servo comparison video to "
            f"{args_cli.visual_servo_comparison_video} ({comparison_frame_count} frames).",
            flush=True,
        )

    print(
        "[INFO]: Fabrica Isaac pickup attempt "
        f"support_face={placement_spec.support_face} yaw_deg={placement_spec.yaw_deg:.1f} "
        f"saved={len(bundle.candidates)} feasible={len(feasible_grasps)} "
        f"selected={selected_grasp.grasp_id} status={execution_result.status} success={execution_result.success}",
        flush=True,
    )
    if not execution_result.success:
        raise SystemExit(1)

    if args_cli.headless and args_cli.run_seconds <= 0.0:
        return

    elapsed_s = 0.0
    while args_cli.run_seconds <= 0.0 or elapsed_s < args_cli.run_seconds:
        try:
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.step()
                continue
            scene.write_data_to_sim()
            sim.step()
            scene.update(physics_dt)
            elapsed_s += physics_dt
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_error: BaseException | None = None
    try:
        run()
    except SystemExit as exc:
        run_error = exc
    except BaseException as exc:
        run_error = exc
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    finally:
        try:
            simulation_app.close()
        except SystemExit:
            if run_error is None:
                raise
    if run_error is not None:
        raise run_error
