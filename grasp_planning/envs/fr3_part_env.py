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
    DEFAULT_ROBOT_CFG,
)


@configclass
class FR3PartSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 2.5

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
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
                max_depenetration_velocity=0.05,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=64,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_ROBOT_CFG.base_pos,
            rot=DEFAULT_ROBOT_CFG.base_rot,
            joint_pos={**DEFAULT_ARM_START_JOINT_POS, **DEFAULT_HAND_START_JOINT_POS},
        ),
        actuators={
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
                joint_names_expr=["panda_finger_joint.*"],
                stiffness=500.0,
                damping=50.0,
                effort_limit_sim=50.0,
                velocity_limit_sim=0.2,
            ),
        },
    )

    part = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Part",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )


def make_fr3_part_scene_cfg(
    *,
    fr3_asset_path: str,
    part_usd_path: str,
    part_position: tuple[float, float, float],
    part_orientation_xyzw: tuple[float, float, float, float],
    robot_base_position: tuple[float, float, float] = DEFAULT_ROBOT_CFG.base_pos,
    robot_base_orientation_xyzw: tuple[float, float, float, float] = DEFAULT_ROBOT_CFG.base_rot,
) -> FR3PartSceneCfg:
    """Build a configured scene for a single Franka Panda and rigid part."""

    def _resolve_path(asset_path: str) -> str:
        if "://" in asset_path:
            return asset_path
        resolved = Path(asset_path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Asset not found at '{resolved}'.")
        return str(resolved)

    scene_cfg = FR3PartSceneCfg()
    scene_cfg.robot.spawn.usd_path = _resolve_path(fr3_asset_path)
    scene_cfg.robot.init_state.pos = robot_base_position
    scene_cfg.robot.init_state.rot = robot_base_orientation_xyzw
    scene_cfg.part.spawn.usd_path = _resolve_path(part_usd_path)
    scene_cfg.part.init_state.pos = part_position
    # Isaac Lab initial-state quaternions are wxyz, while pipeline world poses are xyzw.
    x, y, z, w = part_orientation_xyzw
    scene_cfg.part.init_state.rot = (w, x, y, z)
    return scene_cfg
