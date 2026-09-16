"""Reusable Panda + wrist RGB-D + white-table scene (import after AppLauncher).

The camera is the user-supplied mount from the local Isaac Lab unplug task:
panda_hand -> ROS optical frame, quaternion in WXYZ order. This scene does not
load the KUKA training catalogs or construct the T-slot workspace.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from grasp_planning.isaac_visual_scene import (
    VISUAL_SERVO_KEY_ROTATION_WXYZ,
    make_visual_servo_dome_light_cfg,
    make_visual_servo_key_light_cfg,
    make_visual_servo_render_cfg,
)

FRANKA_SCENE_PROFILE = "panda_hand_wrist_rgbd_white_table_blue_part_keylight_v2"
FRANKA_CAMERA_POSITION = (-0.1115, 0.0481, 0.1034 - 0.0883)
FRANKA_CAMERA_QUATERNION_WXYZ = (0.6435702, -0.2768679, 0.2724621, -0.6594891)


def make_franka_part_material_cfg():
    """Explicit provisional matte-blue material, shared by primitives and imported meshes."""
    return sim_utils.PreviewSurfaceCfg(diffuse_color=(0.045, 0.18, 0.42), roughness=0.65, metallic=0.0)


def make_franka_render_cfg():
    cfg = make_visual_servo_render_cfg()
    cfg.enable_ambient_occlusion = True
    return cfg


@configclass
class FrankaTrainingSceneCfg(InteractiveSceneCfg):
    """Vectorizable scene, with tabletop and robot mounting plane at world Z=0.

    Uses the Panda arm and Panda hand matching the source camera scene. An FR3
    arm requires its own articulation/joint configuration; it is not silently
    substituted for the Panda. The dynamic cube is a catalog-free smoke target.
    """

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
        "panda_finger_joint.*": 0.04,
    }

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.9, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.92, 0.92, 0.92),
                roughness=0.7,
                metallic=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
        ),
    )
    part = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Part",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.021)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            visual_material=make_franka_part_material_cfg(),
        ),
    )
    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        update_period=0.0,
        height=376,
        width=672,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.06,
            focus_distance=0.5,
            horizontal_aperture=4.8,
            vertical_aperture=3.6,
            clipping_range=(0.01, 1.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=FRANKA_CAMERA_POSITION,
            rot=FRANKA_CAMERA_QUATERNION_WXYZ,
            convention="ros",
        ),
    )
    hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        history_length=3,
    )
    left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Part"],
    )
    right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Part"],
    )
    light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=make_visual_servo_dome_light_cfg())
    light.spawn.intensity = 250.0
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=make_visual_servo_key_light_cfg(),
        init_state=AssetBaseCfg.InitialStateCfg(rot=VISUAL_SERVO_KEY_ROTATION_WXYZ),
    )
    key_light.spawn.intensity = 900.0
