"""Canonical Isaac lighting and RTX settings for wrist-camera observations."""

from __future__ import annotations

from typing import Any

VISUAL_SERVO_SCENE_PROFILE = (
    "small_tslot_dome_directional_dlaa_4spp_dldenoise_v6"
)

VISUAL_SERVO_DIRECT_LIGHT_SAMPLES = 4
VISUAL_SERVO_DL_DENOISER_ENABLED = True

VISUAL_SERVO_GROUND_COLOR = (0.075, 0.085, 0.10)
VISUAL_SERVO_DOME_INTENSITY = 450.0
VISUAL_SERVO_DOME_COLOR = (0.76, 0.80, 0.86)
VISUAL_SERVO_KEY_INTENSITY = 1200.0
VISUAL_SERVO_KEY_COLOR = (1.0, 0.91, 0.82)
VISUAL_SERVO_KEY_ANGLE_DEG = 8.0
# WXYZ quaternion for an angled top/front key. A distant light gives every
# cloned environment the same direction and intensity without one light per env.
VISUAL_SERVO_KEY_ROTATION_WXYZ = (
    0.9384303340467282,
    0.1620103355210184,
    -0.2851488516085399,
    -0.10858771455208008,
)


def make_visual_servo_render_cfg() -> Any:
    """Return the shared real-time RTX profile used by image-producing paths.

    Isaac imports stay local so non-Isaac catalog and test tooling can import
    the profile identifier without launching or installing Isaac Sim.
    """

    import isaaclab.sim as sim_utils

    return sim_utils.RenderCfg(
        enable_translucency=False,
        enable_reflections=False,
        enable_global_illumination=False,
        antialiasing_mode="DLAA",
        enable_dlssg=False,
        enable_dl_denoiser=VISUAL_SERVO_DL_DENOISER_ENABLED,
        enable_direct_lighting=True,
        samples_per_pixel=VISUAL_SERVO_DIRECT_LIGHT_SAMPLES,
        enable_shadows=True,
        enable_ambient_occlusion=False,
    )


def make_visual_servo_dome_light_cfg() -> Any:
    """Return the low-frequency ambient/fill light for the canonical scene."""

    import isaaclab.sim as sim_utils

    return sim_utils.DomeLightCfg(
        intensity=VISUAL_SERVO_DOME_INTENSITY,
        color=VISUAL_SERVO_DOME_COLOR,
        visible_in_primary_ray=False,
    )


def make_visual_servo_key_light_cfg() -> Any:
    """Return the global directional key that reveals matte object shape."""

    import isaaclab.sim as sim_utils

    return sim_utils.DistantLightCfg(
        intensity=VISUAL_SERVO_KEY_INTENSITY,
        color=VISUAL_SERVO_KEY_COLOR,
        angle=VISUAL_SERVO_KEY_ANGLE_DEG,
    )


def spawn_visual_servo_lights() -> dict[str, str]:
    """Spawn the canonical global dome and directional key lights."""

    dome_path = "/World/DomeLight"
    key_path = "/World/VisualServoKeyLight"
    dome_cfg = make_visual_servo_dome_light_cfg()
    key_cfg = make_visual_servo_key_light_cfg()
    dome_cfg.func(dome_path, dome_cfg)
    key_cfg.func(
        key_path,
        key_cfg,
        orientation=VISUAL_SERVO_KEY_ROTATION_WXYZ,
    )
    return {"dome": dome_path, "key": key_path}


__all__ = [
    "VISUAL_SERVO_DIRECT_LIGHT_SAMPLES",
    "VISUAL_SERVO_DL_DENOISER_ENABLED",
    "VISUAL_SERVO_GROUND_COLOR",
    "VISUAL_SERVO_KEY_ROTATION_WXYZ",
    "VISUAL_SERVO_SCENE_PROFILE",
    "make_visual_servo_dome_light_cfg",
    "make_visual_servo_key_light_cfg",
    "make_visual_servo_render_cfg",
    "spawn_visual_servo_lights",
]
