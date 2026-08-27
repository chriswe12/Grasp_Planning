"""Isaac Lab scene config for a Franka Panda robot, a plane, and one spawned part."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass

from grasp_planning.d405_wrist_camera import D405WristCameraConfig, camera_pose_in_link7
from grasp_planning.isaac_visual_scene import (
    VISUAL_SERVO_GROUND_COLOR,
    VISUAL_SERVO_KEY_ROTATION_WXYZ,
    make_visual_servo_dome_light_cfg,
    make_visual_servo_key_light_cfg,
)
from grasp_planning.start_poses import PDZ_GRIPPER_TRAVEL_M

from .fr3_cube_env import (
    DEFAULT_ARM_START_JOINT_POS,
    DEFAULT_HAND_START_JOINT_POS,
    DEFAULT_KUKA_ARM_START_JOINT_POS,
    DEFAULT_ROBOT_CFG,
    ISAAC_MIN_CONTACT_OFFSET_M,
)

DEFAULT_PART_DENSITY_KG_M3 = 1240.0
KUKA_ARM_ACTUATOR_PROFILE_WORKING = "working"
KUKA_ARM_ACTUATOR_PROFILE_SOURCE_USD = "source_usd"
KUKA_ARM_ACTUATOR_PROFILE_DEFAULT = KUKA_ARM_ACTUATOR_PROFILE_SOURCE_USD
KUKA_ARM_SOURCE_USD_DAMPING_DEFAULT = 80.0


def _spawn_local_ground_plane(
    prim_path: str,
    cfg: sim_utils.GroundPlaneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a local physics plane without requiring Isaac Nucleus/remote grid assets."""

    del orientation, kwargs
    import omni.usd
    from omni.physx.scripts import physicsUtils
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    position = Gf.Vec3f(*(translation or (0.0, 0.0, 0.0)))
    color = Gf.Vec3f(*(cfg.color if cfg.color is not None else (0.0, 0.0, 0.0)))
    size = float(max(cfg.size))
    physicsUtils.add_ground_plane(stage, prim_path, "Z", size, position, color)
    return stage.GetPrimAtPath(prim_path)


def _is_kuka_lbr_asset(asset_path: str) -> bool:
    normalized = str(asset_path).lower()
    return "kuka" in normalized or "iiwa" in normalized or "lbr" in normalized


def _is_pdz_gripper_asset(asset_path: str) -> bool:
    return "pdz_gripper" in str(asset_path).lower()


def _hand_start_joint_pos_for_asset(asset_path: str) -> dict[str, float]:
    if _is_pdz_gripper_asset(asset_path):
        return {
            "pdz_gripper_left_finger_joint": PDZ_GRIPPER_TRAVEL_M,
            "pdz_gripper_right_finger_joint": PDZ_GRIPPER_TRAVEL_M,
        }
    if _is_kuka_lbr_asset(asset_path):
        return {
            "left_finger_joint": DEFAULT_HAND_START_JOINT_POS["left_finger_joint"],
            "right_finger_joint": DEFAULT_HAND_START_JOINT_POS["right_finger_joint"],
        }
    return {"panda_finger_joint.*": DEFAULT_HAND_START_JOINT_POS["panda_finger_joint.*"]}


def _robot_start_joint_pos_for_asset(asset_path: str) -> dict[str, float]:
    if _is_kuka_lbr_asset(asset_path):
        return {**DEFAULT_KUKA_ARM_START_JOINT_POS, **_hand_start_joint_pos_for_asset(asset_path)}
    return {**DEFAULT_ARM_START_JOINT_POS, **_hand_start_joint_pos_for_asset(asset_path)}


def _kuka_arm_actuators(profile: str, *, damping_override: float | None = None) -> dict[str, ImplicitActuatorCfg]:
    if damping_override is not None and damping_override < 0.0:
        raise ValueError("KUKA arm damping override must be >= 0.")

    def _damping(default: float) -> float:
        return float(default if damping_override is None else damping_override)

    if profile == KUKA_ARM_ACTUATOR_PROFILE_SOURCE_USD:
        return {
            "arm_a1_a2": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                stiffness=625.0,
                damping=_damping(KUKA_ARM_SOURCE_USD_DAMPING_DEFAULT),
                effort_limit_sim=176.0,
                velocity_limit_sim=10.0,
            ),
            "arm_a3": ImplicitActuatorCfg(
                joint_names_expr=["joint3"],
                stiffness=625.0,
                damping=_damping(KUKA_ARM_SOURCE_USD_DAMPING_DEFAULT),
                effort_limit_sim=110.0,
                velocity_limit_sim=10.0,
            ),
            "arm_a4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                stiffness=625.0,
                damping=_damping(KUKA_ARM_SOURCE_USD_DAMPING_DEFAULT),
                effort_limit_sim=110.0,
                velocity_limit_sim=10.0,
            ),
            "arm_a5": ImplicitActuatorCfg(
                joint_names_expr=["joint5"],
                stiffness=625.0,
                damping=_damping(KUKA_ARM_SOURCE_USD_DAMPING_DEFAULT),
                effort_limit_sim=110.0,
                velocity_limit_sim=10.0,
            ),
            "arm_a6_a7": ImplicitActuatorCfg(
                joint_names_expr=["joint[6-7]"],
                stiffness=625.0,
                damping=_damping(KUKA_ARM_SOURCE_USD_DAMPING_DEFAULT),
                effort_limit_sim=40.0,
                velocity_limit_sim=10.0,
            ),
        }
    if profile != KUKA_ARM_ACTUATOR_PROFILE_WORKING:
        raise ValueError(
            "Unknown KUKA arm actuator profile "
            f"'{profile}'. Expected '{KUKA_ARM_ACTUATOR_PROFILE_WORKING}' "
            f"or '{KUKA_ARM_ACTUATOR_PROFILE_SOURCE_USD}'."
        )
    return {
        "arm_a1_a2": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            stiffness=8000.0,
            damping=_damping(800.0),
            effort_limit_sim=10000.0,
            velocity_limit_sim=10.0,
        ),
        "arm_a3": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            stiffness=8000.0,
            damping=_damping(800.0),
            effort_limit_sim=10000.0,
            velocity_limit_sim=10.0,
        ),
        "arm_a4": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            stiffness=8000.0,
            damping=_damping(800.0),
            effort_limit_sim=10000.0,
            velocity_limit_sim=10.0,
        ),
        "arm_a5": ImplicitActuatorCfg(
            joint_names_expr=["joint5"],
            stiffness=8000.0,
            damping=_damping(800.0),
            effort_limit_sim=10000.0,
            velocity_limit_sim=10.0,
        ),
        "arm_a6_a7": ImplicitActuatorCfg(
            joint_names_expr=["joint[6-7]"],
            stiffness=8000.0,
            damping=_damping(800.0),
            effort_limit_sim=10000.0,
            velocity_limit_sim=10.0,
        ),
    }


def _robot_actuators_for_asset(
    asset_path: str,
    *,
    kuka_arm_actuator_profile: str = KUKA_ARM_ACTUATOR_PROFILE_DEFAULT,
    kuka_arm_damping_override: float | None = None,
) -> dict[str, ImplicitActuatorCfg]:
    if _is_kuka_lbr_asset(asset_path):
        if _is_pdz_gripper_asset(asset_path):
            hand_actuators = {
                # The imported URDF exposes its mimic follower as a second DOF,
                # so both sides receive identical source-closed travel targets.
                "hand_left": ImplicitActuatorCfg(
                    joint_names_expr=["pdz_gripper_left_finger_joint"],
                    stiffness=7500.0,
                    damping=173.0,
                    effort_limit_sim=100.0,
                    velocity_limit_sim=0.05,
                ),
                "hand_right": ImplicitActuatorCfg(
                    joint_names_expr=["pdz_gripper_right_finger_joint"],
                    stiffness=7500.0,
                    damping=173.0,
                    effort_limit_sim=100.0,
                    velocity_limit_sim=0.05,
                ),
            }
        else:
            hand_actuators = {
                "hand_driver": ImplicitActuatorCfg(
                    joint_names_expr=["left_finger_joint"],
                    stiffness=7500.0,
                    damping=173.0,
                    effort_limit_sim=40.0,
                    velocity_limit_sim=0.04,
                ),
                "hand_passive": ImplicitActuatorCfg(
                    joint_names_expr=["right_finger_joint"],
                    stiffness=0.0,
                    damping=0.0,
                    effort_limit_sim=1.0,
                    velocity_limit_sim=0.04,
                ),
            }
        return {
            **_kuka_arm_actuators(
                kuka_arm_actuator_profile,
                damping_override=kuka_arm_damping_override,
            ),
            **hand_actuators,
        }
    return {
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=400.0,
            damping=80.0,
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=400.0,
            damping=80.0,
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            stiffness=7500.0,
            damping=173.0,
            effort_limit_sim=40.0,
            velocity_limit_sim=0.04,
        ),
    }


@configclass
class FR3PartSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 2.5

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            func=_spawn_local_ground_plane,
            color=VISUAL_SERVO_GROUND_COLOR,
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=make_visual_servo_dome_light_cfg(),
    )

    key_light = AssetBaseCfg(
        prim_path="/World/VisualServoKeyLight",
        spawn=make_visual_servo_key_light_cfg(),
        init_state=AssetBaseCfg.InitialStateCfg(rot=VISUAL_SERVO_KEY_ROTATION_WXYZ),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=ISAAC_MIN_CONTACT_OFFSET_M,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_ROBOT_CFG.base_pos,
            rot=DEFAULT_ROBOT_CFG.base_rot,
            joint_pos={**DEFAULT_ARM_START_JOINT_POS, **_hand_start_joint_pos_for_asset("")},
        ),
        actuators=_robot_actuators_for_asset(""),
    )

    part = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Part",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            semantic_tags=[("class", "target_part")],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=DEFAULT_PART_DENSITY_KG_M3),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=ISAAC_MIN_CONTACT_OFFSET_M,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # Keep sensors after their parent assets: InteractiveScene instantiates entries
    # in config declaration order, so link7 must exist before this camera spawns.
    wrist_camera: CameraCfg | None = None
    overview_camera: CameraCfg | None = None


def make_fr3_part_scene_cfg(
    *,
    fr3_asset_path: str,
    part_usd_path: str,
    part_position: tuple[float, float, float],
    part_orientation_xyzw: tuple[float, float, float, float],
    part_mass_kg: float | None = None,
    part_density_kg_m3: float | None = DEFAULT_PART_DENSITY_KG_M3,
    robot_base_position: tuple[float, float, float] = DEFAULT_ROBOT_CFG.base_pos,
    robot_base_orientation_xyzw: tuple[float, float, float, float] = DEFAULT_ROBOT_CFG.base_rot,
    kuka_arm_actuator_profile: str = KUKA_ARM_ACTUATOR_PROFILE_DEFAULT,
    kuka_arm_damping_override: float | None = None,
    wrist_camera: D405WristCameraConfig | None = None,
) -> FR3PartSceneCfg:
    """Build a configured scene for a single Franka Panda and rigid part."""

    if part_mass_kg is not None and part_density_kg_m3 is not None:
        raise ValueError("part_mass_kg and part_density_kg_m3 are mutually exclusive.")
    if part_mass_kg is not None and part_mass_kg <= 0.0:
        raise ValueError("part_mass_kg must be > 0 when set.")
    if part_density_kg_m3 is not None and part_density_kg_m3 <= 0.0:
        raise ValueError("part_density_kg_m3 must be > 0 when set.")

    def _resolve_path(asset_path: str) -> str:
        if "://" in asset_path:
            return asset_path
        resolved = Path(asset_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Asset not found at '{resolved}'.")
        return str(resolved)

    scene_cfg = FR3PartSceneCfg()
    resolved_robot_path = _resolve_path(fr3_asset_path)
    scene_cfg.robot.spawn.usd_path = resolved_robot_path
    scene_cfg.robot.init_state.pos = robot_base_position
    scene_cfg.robot.init_state.rot = robot_base_orientation_xyzw
    scene_cfg.robot.init_state.joint_pos = _robot_start_joint_pos_for_asset(resolved_robot_path)
    scene_cfg.robot.actuators = _robot_actuators_for_asset(
        resolved_robot_path,
        kuka_arm_actuator_profile=kuka_arm_actuator_profile,
        kuka_arm_damping_override=kuka_arm_damping_override,
    )
    scene_cfg.part.spawn.usd_path = _resolve_path(part_usd_path)
    if part_mass_kg is not None:
        scene_cfg.part.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=part_mass_kg)
    elif part_density_kg_m3 is not None:
        scene_cfg.part.spawn.mass_props = sim_utils.MassPropertiesCfg(density=part_density_kg_m3)
    scene_cfg.part.init_state.pos = part_position
    # Isaac Lab initial-state quaternions are wxyz, while pipeline world poses are xyzw.
    x, y, z, w = part_orientation_xyzw
    scene_cfg.part.init_state.rot = (w, x, y, z)
    if wrist_camera is not None and wrist_camera.enabled:
        camera_position, camera_orientation_wxyz = camera_pose_in_link7(wrist_camera)
        camera_data_types = ["rgb", "distance_to_image_plane"]
        if wrist_camera.include_privileged_mask:
            camera_data_types.append("semantic_segmentation")
        scene_cfg.wrist_camera = CameraCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/"
                f"{wrist_camera.parent_prim_path.strip('/')}/D405LeftCamera"
            ),
            update_period=float(wrist_camera.update_period_s),
            height=int(wrist_camera.height),
            width=int(wrist_camera.width),
            data_types=camera_data_types,
            semantic_filter=["class"],
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=wrist_camera.intrinsic_matrix_row_major,
                width=int(wrist_camera.width),
                height=int(wrist_camera.height),
                clipping_range=tuple(float(value) for value in wrist_camera.clipping_range_m),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=camera_position,
                rot=camera_orientation_wxyz,
                convention="ros",
            ),
        )
    return scene_cfg


def make_d405_wrist_camera_cfg(
    *, parent_prim_path: str, wrist_camera: D405WristCameraConfig
) -> CameraCfg:
    """Build a D405 sensor after the referenced robot USD has loaded."""

    camera_position, camera_orientation_wxyz = camera_pose_in_link7(wrist_camera)
    camera_data_types = ["rgb", "distance_to_image_plane"]
    if wrist_camera.include_privileged_mask:
        camera_data_types.append("semantic_segmentation")
    return CameraCfg(
        prim_path=f"{parent_prim_path.rstrip('/')}/D405LeftCamera",
        update_period=float(wrist_camera.update_period_s),
        height=int(wrist_camera.height),
        width=int(wrist_camera.width),
        data_types=camera_data_types,
        semantic_filter=["class"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=wrist_camera.intrinsic_matrix_row_major,
            width=int(wrist_camera.width),
            height=int(wrist_camera.height),
            clipping_range=tuple(float(value) for value in wrist_camera.clipping_range_m),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=camera_position,
            rot=camera_orientation_wxyz,
            convention="ros",
        ),
    )


def make_robot_overview_camera_cfg(
    *,
    width: int = 640,
    height: int = 480,
    prim_path: str = "{ENV_REGEX_NS}/OverviewCamera",
) -> CameraCfg:
    """Build a fixed perspective camera intended to show the complete robot."""

    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=int(height),
        width=int(width),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
    )
