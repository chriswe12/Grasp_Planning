"""Physical light, shadow, and material randomization for live RL rendering."""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from grasp_planning.isaac_visual_materials import (
    PDZ_GRIPPER_APPEARANCE_VARIANT_SET,
    VISUAL_SERVO_CONTACT_PAD_COLOR,
    VISUAL_SERVO_CONTACT_PAD_ROUGHNESS,
    VISUAL_SERVO_FINGER_COLOR,
    VISUAL_SERVO_FINGER_ROUGHNESS,
    VISUAL_SERVO_PART_COLOR,
    VISUAL_SERVO_PART_ROUGHNESS,
    VISUAL_SERVO_WORK_SURFACE_COLOR,
    VISUAL_SERVO_WORK_SURFACE_ROUGHNESS,
    get_gripper_appearance_variant,
    nearest_gripper_appearance_variant,
)
from grasp_planning.isaac_visual_scene import (
    VISUAL_SERVO_DOME_COLOR,
    VISUAL_SERVO_DOME_INTENSITY,
    VISUAL_SERVO_KEY_ANGLE_DEG,
    VISUAL_SERVO_KEY_COLOR,
    VISUAL_SERVO_KEY_INTENSITY,
    VISUAL_SERVO_KEY_ROTATION_WXYZ,
)

LIVE_SCENE_APPEARANCE_PROFILE = "physical_light_shadow_plus_per_env_workspace_randomization_v5_visible_gripper_pad"
_SAMPLE_VALUE_COUNT = 19


def _ordered_range(name: str, value: tuple[float, float]) -> None:
    if len(value) != 2 or not math.isfinite(value[0]) or not math.isfinite(value[1]):
        raise ValueError(f"{name} must contain two finite values.")
    if value[0] > value[1]:
        raise ValueError(f"{name} must be ordered, got {value}.")


def _quat_mul(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_from_euler_xyz(roll_rad: float, pitch_rad: float, yaw_rad: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll_rad / 2.0), math.sin(roll_rad / 2.0)
    cp, sp = math.cos(pitch_rad / 2.0), math.sin(pitch_rad / 2.0)
    cy, sy = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _normalize_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in value))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero quaternion.")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


def _scaled_color(base: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(min(1.0, max(0.0, component * scale)) for component in base)  # type: ignore[return-value]


def _temperature_shifted_color(base: tuple[float, float, float], shift: float) -> tuple[float, float, float]:
    gains = (1.0 + shift, 1.0, 1.0 - shift)
    return tuple(min(1.0, max(0.0, component * gain)) for component, gain in zip(base, gains, strict=True))  # type: ignore[return-value]


def _hue_shifted_color(base: tuple[float, float, float], shift_deg: float) -> tuple[float, float, float]:
    hue, saturation, value = colorsys.rgb_to_hsv(*base)
    return colorsys.hsv_to_rgb((hue + shift_deg / 360.0) % 1.0, saturation, value)


@dataclass(frozen=True)
class SceneAppearanceRandomizationCfg:
    """Bounded global scene variations sampled at a slow training cadence."""

    enabled: bool = True
    interval_steps: int = 120
    key_yaw_delta_deg: tuple[float, float] = (-35.0, 35.0)
    key_pitch_delta_deg: tuple[float, float] = (-15.0, 15.0)
    key_intensity_scale: tuple[float, float] = (0.70, 1.30)
    key_angle_deg: tuple[float, float] = (
        VISUAL_SERVO_KEY_ANGLE_DEG - 3.0,
        VISUAL_SERVO_KEY_ANGLE_DEG + 4.0,
    )
    dome_intensity_scale: tuple[float, float] = (0.75, 1.25)
    light_temperature_shift: tuple[float, float] = (-0.08, 0.08)
    part_color_scale: tuple[float, float] = (0.90, 1.10)
    part_hue_shift_deg: tuple[float, float] = (-5.0, 5.0)
    part_roughness: tuple[float, float] = (
        VISUAL_SERVO_PART_ROUGHNESS - 0.15,
        VISUAL_SERVO_PART_ROUGHNESS + 0.10,
    )
    # Black requires a wider multiplicative range to make any visible change:
    # even the upper bound keeps the brightest canonical channel below 0.07.
    # The hardware identity remains black fingers with white contact pads.
    gripper_canonical_fraction: float = 0.20
    finger_color_scale: tuple[float, float] = (0.50, 2.00)
    finger_hue_shift_deg: tuple[float, float] = (-6.0, 6.0)
    finger_roughness: tuple[float, float] = (0.30, 0.70)
    pad_color_scale: tuple[float, float] = (0.88, 1.06)
    pad_temperature_shift: tuple[float, float] = (-0.03, 0.03)
    pad_roughness: tuple[float, float] = (0.55, 0.88)
    ground_color_scale: tuple[float, float] = (0.75, 1.25)
    ground_hue_shift_deg: tuple[float, float] = (-12.0, 12.0)
    ground_roughness: tuple[float, float] = (
        VISUAL_SERVO_WORK_SURFACE_ROUGHNESS - 0.15,
        min(1.0, VISUAL_SERVO_WORK_SURFACE_ROUGHNESS + 0.10),
    )

    def validate(self) -> None:
        if self.interval_steps < 1:
            raise ValueError("interval_steps must be positive.")
        if not math.isfinite(self.gripper_canonical_fraction) or not 0.0 <= self.gripper_canonical_fraction <= 1.0:
            raise ValueError("gripper_canonical_fraction must lie in [0, 1].")
        for name in (
            "key_yaw_delta_deg",
            "key_pitch_delta_deg",
            "key_intensity_scale",
            "key_angle_deg",
            "dome_intensity_scale",
            "light_temperature_shift",
            "part_color_scale",
            "part_hue_shift_deg",
            "part_roughness",
            "finger_color_scale",
            "finger_hue_shift_deg",
            "finger_roughness",
            "pad_color_scale",
            "pad_temperature_shift",
            "pad_roughness",
            "ground_color_scale",
            "ground_hue_shift_deg",
            "ground_roughness",
        ):
            _ordered_range(name, getattr(self, name))
        for name in (
            "key_intensity_scale",
            "key_angle_deg",
            "dome_intensity_scale",
            "part_color_scale",
            "finger_color_scale",
            "pad_color_scale",
            "ground_color_scale",
        ):
            if getattr(self, name)[0] <= 0.0:
                raise ValueError(f"{name} must stay positive.")
        for name in ("part_roughness", "finger_roughness", "pad_roughness", "ground_roughness"):
            lower, upper = getattr(self, name)
            if lower < 0.0 or upper > 1.0:
                raise ValueError(f"{name} must stay within [0, 1].")


@dataclass(frozen=True)
class SceneAppearanceSample:
    key_yaw_delta_deg: float
    key_pitch_delta_deg: float
    key_orientation_wxyz: tuple[float, float, float, float]
    key_intensity: float
    key_angle_deg: float
    key_color: tuple[float, float, float]
    dome_intensity: float
    dome_color: tuple[float, float, float]
    part_color: tuple[float, float, float]
    part_roughness: float
    finger_color: tuple[float, float, float]
    finger_roughness: float
    pad_color: tuple[float, float, float]
    pad_roughness: float
    gripper_variant: str
    gripper_canonical: bool
    ground_color: tuple[float, float, float]
    ground_roughness: float


def sample_scene_appearance(
    cfg: SceneAppearanceRandomizationCfg,
    unit_values: Sequence[float],
) -> SceneAppearanceSample:
    """Map reproducible unit samples to one physically plausible appearance."""

    cfg.validate()
    if len(unit_values) != _SAMPLE_VALUE_COUNT:
        raise ValueError(f"Expected {_SAMPLE_VALUE_COUNT} unit samples, got {len(unit_values)}.")
    values = tuple(float(value) for value in unit_values)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Every unit sample must be finite and lie in [0, 1].")

    def resolve(index: int, value_range: tuple[float, float]) -> float:
        lower, upper = value_range
        return lower + values[index] * (upper - lower)

    yaw_deg = resolve(0, cfg.key_yaw_delta_deg)
    pitch_deg = resolve(1, cfg.key_pitch_delta_deg)
    delta = _quat_from_euler_xyz(
        0.0,
        math.radians(pitch_deg),
        math.radians(yaw_deg),
    )
    orientation = _normalize_quaternion(_quat_mul(delta, VISUAL_SERVO_KEY_ROTATION_WXYZ))
    temperature_shift = resolve(5, cfg.light_temperature_shift)
    gripper_canonical = values[18] < cfg.gripper_canonical_fraction
    if gripper_canonical:
        gripper_variant = get_gripper_appearance_variant("canonical")
    else:
        requested_finger_color = _scaled_color(
            _hue_shifted_color(
                VISUAL_SERVO_FINGER_COLOR,
                resolve(13, cfg.finger_hue_shift_deg),
            ),
            resolve(8, cfg.finger_color_scale),
        )
        requested_pad_color = _scaled_color(
            _temperature_shifted_color(
                VISUAL_SERVO_CONTACT_PAD_COLOR,
                resolve(16, cfg.pad_temperature_shift),
            ),
            resolve(15, cfg.pad_color_scale),
        )
        gripper_variant = nearest_gripper_appearance_variant(
            finger_color=requested_finger_color,
            finger_roughness=resolve(14, cfg.finger_roughness),
            pad_color=requested_pad_color,
            pad_roughness=resolve(17, cfg.pad_roughness),
        )
    return SceneAppearanceSample(
        key_yaw_delta_deg=yaw_deg,
        key_pitch_delta_deg=pitch_deg,
        key_orientation_wxyz=orientation,
        key_intensity=VISUAL_SERVO_KEY_INTENSITY * resolve(2, cfg.key_intensity_scale),
        key_angle_deg=resolve(3, cfg.key_angle_deg),
        key_color=_temperature_shifted_color(VISUAL_SERVO_KEY_COLOR, temperature_shift),
        dome_intensity=VISUAL_SERVO_DOME_INTENSITY * resolve(4, cfg.dome_intensity_scale),
        dome_color=_temperature_shifted_color(VISUAL_SERVO_DOME_COLOR, -0.5 * temperature_shift),
        part_color=_scaled_color(
            _hue_shifted_color(VISUAL_SERVO_PART_COLOR, resolve(11, cfg.part_hue_shift_deg)),
            resolve(6, cfg.part_color_scale),
        ),
        part_roughness=resolve(7, cfg.part_roughness),
        finger_color=gripper_variant.finger_color,
        finger_roughness=gripper_variant.finger_roughness,
        pad_color=gripper_variant.pad_color,
        pad_roughness=gripper_variant.pad_roughness,
        gripper_variant=gripper_variant.name,
        gripper_canonical=gripper_variant.name == "canonical",
        ground_color=_scaled_color(
            _hue_shifted_color(
                VISUAL_SERVO_WORK_SURFACE_COLOR,
                resolve(12, cfg.ground_hue_shift_deg),
            ),
            resolve(9, cfg.ground_color_scale),
        ),
        ground_roughness=resolve(10, cfg.ground_roughness),
    )


class SceneAppearanceRandomizer:
    """Update shared light/finger prims without touching goal RGB-D.

    Part and T-slot colors are authored per environment by
    :class:`LiveWorkspaceAppearanceRandomizer`; changing their shared canonical
    materials here would incorrectly correlate all cloned environments.
    """

    def __init__(
        self,
        cfg: SceneAppearanceRandomizationCfg,
        *,
        light_paths: Mapping[str, str],
        material_paths: Mapping[str, str],
        gripper_variant_paths: Sequence[str] = (),
        device: torch.device | str,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(device)
        self.light_paths = dict(light_paths)
        self.material_paths = dict(material_paths)
        self.gripper_variant_paths = tuple(gripper_variant_paths)
        self.current_sample: SceneAppearanceSample | None = None
        self.last_randomized_step = -cfg.interval_steps

        import omni.usd

        self.stage = omni.usd.get_context().get_stage()
        required_lights = {"dome", "key"}
        required_materials = {"black_pla", "white_contact_pad", "work_surface"}
        if not required_lights.issubset(self.light_paths):
            raise ValueError("Scene appearance randomization requires dome and key lights.")
        if not required_materials.issubset(self.material_paths):
            raise ValueError("Scene appearance randomization requires shared finger, pad, and surface materials.")
        for path in self.gripper_variant_paths:
            prim = self._prim(path)
            if PDZ_GRIPPER_APPEARANCE_VARIANT_SET not in prim.GetVariantSets().GetNames():
                raise ValueError(f"Gripper appearance root lacks material variants: {path}")

    def _prim(self, path: str):
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Scene appearance prim does not exist: {path}")
        return prim

    @staticmethod
    def _set_attribute(prim, name: str, value) -> None:
        attribute = prim.GetAttribute(name)
        if not attribute.IsValid():
            raise RuntimeError(f"Missing appearance attribute {prim.GetPath()}.{name}")
        if not attribute.Set(value):
            raise RuntimeError(f"Failed to set appearance attribute {prim.GetPath()}.{name}")

    def _new_sample(self, *, strength: float) -> SceneAppearanceSample:
        unit_values = torch.rand(_SAMPLE_VALUE_COUNT, device=self.device).cpu().tolist()
        sample = sample_scene_appearance(self.cfg, unit_values)
        value = min(1.0, max(0.0, float(strength)))

        def lerp_scalar(start: float, end: float) -> float:
            return start + value * (end - start)

        def lerp_color(
            start: tuple[float, float, float], end: tuple[float, float, float]
        ) -> tuple[float, float, float]:
            return tuple(lerp_scalar(a, b) for a, b in zip(start, end, strict=True))  # type: ignore[return-value]

        yaw = value * sample.key_yaw_delta_deg
        pitch = value * sample.key_pitch_delta_deg
        delta = _quat_from_euler_xyz(0.0, math.radians(pitch), math.radians(yaw))
        orientation = _normalize_quaternion(_quat_mul(delta, VISUAL_SERVO_KEY_ROTATION_WXYZ))
        requested_finger_color = lerp_color(VISUAL_SERVO_FINGER_COLOR, sample.finger_color)
        requested_finger_roughness = lerp_scalar(
            VISUAL_SERVO_FINGER_ROUGHNESS, sample.finger_roughness
        )
        requested_pad_color = lerp_color(VISUAL_SERVO_CONTACT_PAD_COLOR, sample.pad_color)
        requested_pad_roughness = lerp_scalar(
            VISUAL_SERVO_CONTACT_PAD_ROUGHNESS, sample.pad_roughness
        )
        gripper_variant = (
            get_gripper_appearance_variant("canonical")
            if value <= 0.0 or sample.gripper_canonical
            else nearest_gripper_appearance_variant(
                finger_color=requested_finger_color,
                finger_roughness=requested_finger_roughness,
                pad_color=requested_pad_color,
                pad_roughness=requested_pad_roughness,
            )
        )
        return SceneAppearanceSample(
            key_yaw_delta_deg=yaw,
            key_pitch_delta_deg=pitch,
            key_orientation_wxyz=orientation,
            key_intensity=lerp_scalar(VISUAL_SERVO_KEY_INTENSITY, sample.key_intensity),
            key_angle_deg=lerp_scalar(VISUAL_SERVO_KEY_ANGLE_DEG, sample.key_angle_deg),
            key_color=lerp_color(VISUAL_SERVO_KEY_COLOR, sample.key_color),
            dome_intensity=lerp_scalar(VISUAL_SERVO_DOME_INTENSITY, sample.dome_intensity),
            dome_color=lerp_color(VISUAL_SERVO_DOME_COLOR, sample.dome_color),
            part_color=lerp_color(VISUAL_SERVO_PART_COLOR, sample.part_color),
            part_roughness=lerp_scalar(VISUAL_SERVO_PART_ROUGHNESS, sample.part_roughness),
            finger_color=gripper_variant.finger_color,
            finger_roughness=gripper_variant.finger_roughness,
            pad_color=gripper_variant.pad_color,
            pad_roughness=gripper_variant.pad_roughness,
            gripper_variant=gripper_variant.name,
            gripper_canonical=gripper_variant.name == "canonical",
            ground_color=lerp_color(VISUAL_SERVO_WORK_SURFACE_COLOR, sample.ground_color),
            ground_roughness=lerp_scalar(VISUAL_SERVO_WORK_SURFACE_ROUGHNESS, sample.ground_roughness),
        )

    def apply(self, sample: SceneAppearanceSample) -> None:
        """Author one sample into USD; rotating the key moves rendered shadows."""

        from pxr import Gf, Sdf

        key = self._prim(self.light_paths["key"])
        dome = self._prim(self.light_paths["dome"])
        finger = None
        pad = None
        if self.gripper_variant_paths:
            for path in self.gripper_variant_paths:
                variant_set = self._prim(path).GetVariantSets().GetVariantSet(
                    PDZ_GRIPPER_APPEARANCE_VARIANT_SET
                )
                if not variant_set.SetVariantSelection(sample.gripper_variant):
                    raise RuntimeError(
                        f"Could not select gripper material variant {sample.gripper_variant} on {path}"
                    )
        else:
            finger = self._prim(f"{self.material_paths['black_pla']}/Shader")
            pad = self._prim(f"{self.material_paths['white_contact_pad']}/Shader")
        ground = self._prim(f"{self.material_paths['work_surface']}/Shader")
        w, x, y, z = sample.key_orientation_wxyz
        with Sdf.ChangeBlock():
            self._set_attribute(
                key,
                "xformOp:orient",
                Gf.Quatd(w, Gf.Vec3d(x, y, z)),
            )
            self._set_attribute(key, "inputs:intensity", sample.key_intensity)
            self._set_attribute(key, "inputs:angle", sample.key_angle_deg)
            self._set_attribute(key, "inputs:color", sample.key_color)
            self._set_attribute(dome, "inputs:intensity", sample.dome_intensity)
            self._set_attribute(dome, "inputs:color", sample.dome_color)
            if finger is not None and pad is not None:
                self._set_attribute(finger, "inputs:diffuseColor", sample.finger_color)
                self._set_attribute(finger, "inputs:roughness", sample.finger_roughness)
                self._set_attribute(pad, "inputs:diffuseColor", sample.pad_color)
                self._set_attribute(pad, "inputs:roughness", sample.pad_roughness)
            self._set_attribute(ground, "inputs:diffuseColor", sample.ground_color)
            self._set_attribute(ground, "inputs:roughness", sample.ground_roughness)
        self.current_sample = sample

    def maybe_randomize(
        self,
        common_step_counter: int,
        *,
        force: bool = False,
        strength: float = 1.0,
    ) -> bool:
        """Sample at a slow global cadence so illumination is temporally stable."""

        step = int(common_step_counter)
        if not self.cfg.enabled:
            return False
        if not force and step - self.last_randomized_step < self.cfg.interval_steps:
            return False
        self.apply(self._new_sample(strength=strength))
        self.last_randomized_step = step
        return True


__all__ = [
    "LIVE_SCENE_APPEARANCE_PROFILE",
    "SceneAppearanceRandomizationCfg",
    "SceneAppearanceRandomizer",
    "SceneAppearanceSample",
    "sample_scene_appearance",
]
