"""Isaac Lab scene config for an FR3 robot, a plane, and a graspable cube."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class DefaultRobotCfg:
    """Default FR3 robot placement and initialization settings."""

    base_pos = (0.0, 0.0, 0.0)
    base_rot = (0.0, 0.0, 0.0, 1.0)


@configclass
class DefaultCubeCfg:
    """Default cube geometry and physical properties."""

    size = (0.05, 0.05, 0.05)
    pos = (-0.65, 0.0, 0.025)
    rot = (0.0, 0.0, 0.0, 1.0)
    mass = 0.15
    color = (0.8, 0.2, 0.2)


DEFAULT_ROBOT_CFG = DefaultRobotCfg()
DEFAULT_CUBE_CFG = DefaultCubeCfg()
DEFAULT_ARM_START_JOINT_POS = {
    "fr3_joint1": -0.35,
    "fr3_joint2": -0.45,
    "fr3_joint3": 0.0,
    "fr3_joint4": -1.85,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.25,
    "fr3_joint7": -1.2,
}
DEFAULT_HAND_START_JOINT_POS = {"fr3_finger_joint.*": 0.04}


@configclass
class FR3CubeSceneCfg(InteractiveSceneCfg):
    """Interactive scene with a ground plane, FR3, cube, lights, and debug markers."""

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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DEFAULT_CUBE_CFG.color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=DEFAULT_CUBE_CFG.pos, rot=DEFAULT_CUBE_CFG.rot),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=DEFAULT_ROBOT_CFG.base_pos,
            rot=DEFAULT_ROBOT_CFG.base_rot,
            joint_pos={**DEFAULT_ARM_START_JOINT_POS, **DEFAULT_HAND_START_JOINT_POS},
        ),
        actuators={
            "fr3_arm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-7]"],
                stiffness=800.0,
                damping=80.0,
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
            ),
            "fr3_hand": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                stiffness=2_000.0,
                damping=200.0,
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
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
    """Build a configured scene for a single FR3 and graspable cube."""

    if "://" in fr3_asset_path:
        asset_path_str = fr3_asset_path
    else:
        asset_path = Path(fr3_asset_path).expanduser()
        if not asset_path.is_file():
            raise FileNotFoundError(
                f"FR3 asset not found at '{asset_path}'. Update FR3_USD_PATH in the launch script or pass --fr3-usd."
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
