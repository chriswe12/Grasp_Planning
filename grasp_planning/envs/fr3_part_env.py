"""Isaac Lab scene config for a Franka Panda robot, a plane, and one spawned part."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .fr3_cube_env import (
    DEFAULT_ARM_START_JOINT_POS,
    DEFAULT_HAND_START_JOINT_POS,
    DEFAULT_KUKA_ARM_START_JOINT_POS,
    DEFAULT_ROBOT_CFG,
    ISAAC_MIN_CONTACT_OFFSET_M,
)

DEFAULT_PART_DENSITY_KG_M3 = 1240.0


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


def _robot_actuators_for_asset(asset_path: str) -> dict[str, ImplicitActuatorCfg]:
    if _is_kuka_lbr_asset(asset_path):
        return {
            "arm_a1_a2": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                stiffness=8000.0,
                damping=800.0,
                effort_limit_sim=10000.0,
                velocity_limit_sim=10.0,
            ),
            "arm_a3": ImplicitActuatorCfg(
                joint_names_expr=["joint3"],
                stiffness=8000.0,
                damping=800.0,
                effort_limit_sim=10000.0,
                velocity_limit_sim=10.0,
            ),
            "arm_a4": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                stiffness=8000.0,
                damping=800.0,
                effort_limit_sim=10000.0,
                velocity_limit_sim=10.0,
            ),
            "arm_a5": ImplicitActuatorCfg(
                joint_names_expr=["joint5"],
                stiffness=8000.0,
                damping=800.0,
                effort_limit_sim=10000.0,
                velocity_limit_sim=10.0,
            ),
            "arm_a6_a7": ImplicitActuatorCfg(
                joint_names_expr=["joint[6-7]"],
                stiffness=8000.0,
                damping=800.0,
                effort_limit_sim=10000.0,
                velocity_limit_sim=10.0,
            ),
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
    scene_cfg.robot.actuators = _robot_actuators_for_asset(resolved_robot_path)
    scene_cfg.part.spawn.usd_path = _resolve_path(part_usd_path)
    if part_mass_kg is not None:
        scene_cfg.part.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=part_mass_kg)
    elif part_density_kg_m3 is not None:
        scene_cfg.part.spawn.mass_props = sim_utils.MassPropertiesCfg(density=part_density_kg_m3)
    scene_cfg.part.init_state.pos = part_position
    # Isaac Lab initial-state quaternions are wxyz, while pipeline world poses are xyzw.
    x, y, z, w = part_orientation_xyzw
    scene_cfg.part.init_state.rot = (w, x, y, z)
    return scene_cfg
