"""Isaac Lab scene config for a Franka Panda robot, a plane, and a graspable cube."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class DefaultRobotCfg:
    """Default Franka robot placement and initialization settings."""

    base_pos = (0.0, 0.0, 0.0)
    # Isaac Lab initial-state rotations use wxyz order. This rotates the robot 180 deg from the previous visual pose.
    base_rot = (1.0, 0.0, 0.0, 0.0)


@configclass
class DefaultCubeCfg:
    """Default cube geometry and physical properties."""

    size = (0.035, 0.035, 0.05)
    pos = (0.45, 0.0, 0.025)
    rot = (0.0, 0.0, 0.0, 1.0)
    mass = 0.05
    color = (0.8, 0.2, 0.2)


DEFAULT_ROBOT_CFG = DefaultRobotCfg()
DEFAULT_CUBE_CFG = DefaultCubeCfg()
DEFAULT_ARM_START_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
}
DEFAULT_HAND_START_JOINT_POS = {"panda_finger_joint.*": 0.04}
DEFAULT_HAND_OPEN_WIDTH = 0.04


@configclass
class FR3CubeSceneCfg(InteractiveSceneCfg):
    """Interactive scene with a ground plane, Franka Panda, cube, lights, and debug markers."""

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

    world_marker = AssetBaseCfg(
        prim_path="/World/Debug/WorldOrigin",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.02)),
    )

    robot_base_marker = AssetBaseCfg(
        prim_path="/World/Debug/RobotBase",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.9)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.025)),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=DEFAULT_CUBE_CFG.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=DEFAULT_CUBE_CFG.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0,
                dynamic_friction=2.5,
                friction_combine_mode="max",
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_CUBE_CFG.color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=DEFAULT_CUBE_CFG.pos, rot=DEFAULT_CUBE_CFG.rot),
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
                stiffness=1500.0,
                damping=120.0,
                effort_limit_sim=120.0,
                velocity_limit_sim=0.08,
            ),
        },
    )


def make_fr3_cube_scene_cfg(
    *,
    fr3_asset_path: str,
    cube_position: tuple[float, float, float],
    cube_orientation_xyzw: tuple[float, float, float, float],
    robot_base_position: tuple[float, float, float] = DEFAULT_ROBOT_CFG.base_pos,
    robot_base_orientation_xyzw: tuple[float, float, float, float] = DEFAULT_ROBOT_CFG.base_rot,
) -> FR3CubeSceneCfg:
    """Build a configured scene for a single Franka Panda and graspable cube."""

    if "://" in fr3_asset_path:
        asset_path_str = fr3_asset_path
    else:
        asset_path = Path(fr3_asset_path).expanduser()
        if not asset_path.is_file():
            raise FileNotFoundError(
                f"Franka asset not found at '{asset_path}'. Update the launch script or pass --fr3-usd."
            )
        asset_path_str = str(asset_path)

    scene_cfg = FR3CubeSceneCfg()
    scene_cfg.robot.spawn.usd_path = asset_path_str
    scene_cfg.robot.init_state.pos = robot_base_position
    scene_cfg.robot.init_state.rot = robot_base_orientation_xyzw
    scene_cfg.robot_base_marker.init_state.pos = (
        robot_base_position[0],
        robot_base_position[1],
        robot_base_position[2] + 0.025,
    )
    scene_cfg.cube.spawn.size = DEFAULT_CUBE_CFG.size
    scene_cfg.cube.init_state.pos = cube_position
    scene_cfg.cube.init_state.rot = cube_orientation_xyzw
    return scene_cfg
