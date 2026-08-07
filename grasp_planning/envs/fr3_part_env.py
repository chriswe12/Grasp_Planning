"""Isaac Lab scene config for a Franka Panda robot, a plane, and one spawned part."""

from __future__ import annotations

import copy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

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


def _hand_start_joint_pos_for_asset(asset_path: str) -> dict[str, float]:
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
    kuka_hand_effort_limit_sim: float = 40.0,
) -> dict[str, ImplicitActuatorCfg]:
    if _is_kuka_lbr_asset(asset_path):
        if float(kuka_hand_effort_limit_sim) <= 0.0:
            raise ValueError("KUKA hand effort limit must be positive.")
        return {
            **_kuka_arm_actuators(
                kuka_arm_actuator_profile,
                damping_override=kuka_arm_damping_override,
            ),
            "hand_driver": ImplicitActuatorCfg(
                joint_names_expr=["left_finger_joint"],
                stiffness=7500.0,
                damping=173.0,
                effort_limit_sim=float(kuka_hand_effort_limit_sim),
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
        spawn=sim_utils.GroundPlaneCfg(func=_spawn_local_ground_plane),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
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


_FR3_PART_SCENE_TEMPLATE = FR3PartSceneCfg()


@configclass
class DualKukaAssemblySceneCfg(InteractiveSceneCfg):
    """Two KUKA/Y-gripper articulations, one rigid prefix, and one incoming part."""

    num_envs = 1
    env_spacing = 2.5

    ground = copy.deepcopy(_FR3_PART_SCENE_TEMPLATE.ground)
    dome_light = copy.deepcopy(_FR3_PART_SCENE_TEMPLATE.dome_light)

    holder_robot = copy.deepcopy(_FR3_PART_SCENE_TEMPLATE.robot)
    holder_robot.prim_path = "{ENV_REGEX_NS}/HolderRobot"

    inserter_robot = copy.deepcopy(_FR3_PART_SCENE_TEMPLATE.robot)
    inserter_robot.prim_path = "{ENV_REGEX_NS}/InserterRobot"

    base_part = copy.deepcopy(_FR3_PART_SCENE_TEMPLATE.part)
    base_part.prim_path = "{ENV_REGEX_NS}/BasePart"

    incoming_part = copy.deepcopy(_FR3_PART_SCENE_TEMPLATE.part)
    incoming_part.prim_path = "{ENV_REGEX_NS}/IncomingPart"

    holder_left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/HolderRobot/left_finger_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/BasePart"],
    )
    holder_right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/HolderRobot/right_finger_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/BasePart"],
    )
    inserter_left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InserterRobot/left_finger_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/IncomingPart"],
    )
    inserter_right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/InserterRobot/right_finger_link",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/IncomingPart"],
    )


def _resolve_asset_path(asset_path: str) -> str:
    if "://" in asset_path:
        return asset_path
    resolved = Path(asset_path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Asset not found at '{resolved}'.")
    return str(resolved)


def _set_part_mass(
    part_cfg: RigidObjectCfg,
    *,
    part_mass_kg: float | None,
    part_density_kg_m3: float | None,
) -> None:
    if part_mass_kg is not None and part_density_kg_m3 is not None:
        raise ValueError("part_mass_kg and part_density_kg_m3 are mutually exclusive.")
    if part_mass_kg is not None and part_mass_kg <= 0.0:
        raise ValueError("part_mass_kg must be > 0 when set.")
    if part_density_kg_m3 is not None and part_density_kg_m3 <= 0.0:
        raise ValueError("part_density_kg_m3 must be > 0 when set.")
    if part_mass_kg is not None:
        part_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=part_mass_kg)
    elif part_density_kg_m3 is not None:
        part_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(density=part_density_kg_m3)


def _set_part_pose_xyzw(
    part_cfg: RigidObjectCfg,
    *,
    position: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
) -> None:
    part_cfg.init_state.pos = position
    x, y, z, w = orientation_xyzw
    part_cfg.init_state.rot = (w, x, y, z)


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
) -> FR3PartSceneCfg:
    """Build a configured scene for a single Franka Panda and rigid part."""

    scene_cfg = FR3PartSceneCfg()
    resolved_robot_path = _resolve_asset_path(fr3_asset_path)
    scene_cfg.robot.spawn.usd_path = resolved_robot_path
    scene_cfg.robot.init_state.pos = robot_base_position
    scene_cfg.robot.init_state.rot = robot_base_orientation_xyzw
    scene_cfg.robot.init_state.joint_pos = _robot_start_joint_pos_for_asset(resolved_robot_path)
    scene_cfg.robot.actuators = _robot_actuators_for_asset(
        resolved_robot_path,
        kuka_arm_actuator_profile=kuka_arm_actuator_profile,
        kuka_arm_damping_override=kuka_arm_damping_override,
    )
    scene_cfg.part.spawn.usd_path = _resolve_asset_path(part_usd_path)
    _set_part_mass(
        scene_cfg.part,
        part_mass_kg=part_mass_kg,
        part_density_kg_m3=part_density_kg_m3,
    )
    _set_part_pose_xyzw(
        scene_cfg.part,
        position=part_position,
        orientation_xyzw=part_orientation_xyzw,
    )
    return scene_cfg


def make_dual_kuka_assembly_scene_cfg(
    *,
    robot_asset_path: str,
    base_part_usd_path: str,
    incoming_part_usd_path: str,
    base_part_position: tuple[float, float, float],
    base_part_orientation_xyzw: tuple[float, float, float, float],
    incoming_part_position: tuple[float, float, float],
    incoming_part_orientation_xyzw: tuple[float, float, float, float],
    ground_height_m: float = 0.0,
    holder_robot_base_position: tuple[float, float, float] = (
        0.0,
        -0.42,
        0.0,
    ),
    inserter_robot_base_position: tuple[float, float, float] = (
        0.0,
        0.42,
        0.0,
    ),
    robot_base_orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    ),
    base_part_mass_kg: float | None = None,
    incoming_part_mass_kg: float | None = None,
    part_density_kg_m3: float | None = DEFAULT_PART_DENSITY_KG_M3,
    kuka_arm_actuator_profile: str = KUKA_ARM_ACTUATOR_PROFILE_DEFAULT,
    kuka_arm_damping_override: float | None = None,
    kuka_hand_effort_limit_sim: float = 40.0,
) -> DualKukaAssemblySceneCfg:
    """Build a dual-robot Fabrica prefix-holder/pickup physics scene.

    ``base_part_usd_path`` may contain either the bare base or a compound
    current subassembly rooted in the base source frame.
    """

    scene_cfg = DualKukaAssemblySceneCfg()
    scene_cfg.ground.init_state.pos = (
        0.0,
        0.0,
        float(ground_height_m),
    )
    resolved_robot_path = _resolve_asset_path(robot_asset_path)
    for robot_cfg, base_position in (
        (scene_cfg.holder_robot, holder_robot_base_position),
        (scene_cfg.inserter_robot, inserter_robot_base_position),
    ):
        robot_cfg.spawn.usd_path = resolved_robot_path
        robot_cfg.spawn.activate_contact_sensors = True
        robot_cfg.init_state.pos = base_position
        x, y, z, w = robot_base_orientation_xyzw
        robot_cfg.init_state.rot = (w, x, y, z)
        robot_cfg.init_state.joint_pos = _robot_start_joint_pos_for_asset(resolved_robot_path)
        robot_cfg.actuators = _robot_actuators_for_asset(
            resolved_robot_path,
            kuka_arm_actuator_profile=kuka_arm_actuator_profile,
            kuka_arm_damping_override=kuka_arm_damping_override,
            kuka_hand_effort_limit_sim=kuka_hand_effort_limit_sim,
        )

    scene_cfg.base_part.spawn.usd_path = _resolve_asset_path(base_part_usd_path)
    _set_part_mass(
        scene_cfg.base_part,
        part_mass_kg=base_part_mass_kg,
        part_density_kg_m3=(None if base_part_mass_kg is not None else part_density_kg_m3),
    )
    _set_part_pose_xyzw(
        scene_cfg.base_part,
        position=base_part_position,
        orientation_xyzw=base_part_orientation_xyzw,
    )

    scene_cfg.incoming_part.spawn.usd_path = _resolve_asset_path(incoming_part_usd_path)
    _set_part_mass(
        scene_cfg.incoming_part,
        part_mass_kg=incoming_part_mass_kg,
        part_density_kg_m3=(None if incoming_part_mass_kg is not None else part_density_kg_m3),
    )
    _set_part_pose_xyzw(
        scene_cfg.incoming_part,
        position=incoming_part_position,
        orientation_xyzw=incoming_part_orientation_xyzw,
    )
    return scene_cfg
