"""Canonical render-only small-pitch T-slot workspace and live appearance variation."""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from grasp_planning.isaac_visual_materials import (
    VISUAL_SERVO_CANONICAL_PART_INDEX,
    VISUAL_SERVO_PART_PALETTE,
    VISUAL_SERVO_PART_ROUGHNESS,
    sample_weighted_part_palette_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_SERVO_TSLOT_ASSET = REPO_ROOT / "assets/usd/visual_servo/t_slot_surface_nominal.usda"
VISUAL_SERVO_TSLOT_PROFILE = "aluminum_tslot_half_scale_5mm_slot_20p5mm_land_flat_collision_v3"
VISUAL_SERVO_TSLOT_CENTER = (0.425, 0.05, 0.0)
VISUAL_SERVO_TSLOT_PITCH_M = 0.0255
VISUAL_SERVO_TSLOT_SCALE = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class TSlotLayoutVariant:
    """One fixed-per-environment visual layout; physics remains a flat plane."""

    name: str
    phase_m: float
    rotation_deg: float


@dataclass(frozen=True)
class TSlotBackgroundAppearance:
    """One plausible aluminum response under the randomized scene lighting."""

    name: str
    color: tuple[float, float, float]
    roughness: float
    weight: float


@dataclass(frozen=True)
class SurfaceAppearance:
    """One bounded continuous variation around a discrete material preset."""

    color: tuple[float, float, float]
    roughness: float


VISUAL_SERVO_TSLOT_BACKGROUNDS: tuple[TSlotBackgroundAppearance, ...] = (
    TSlotBackgroundAppearance("neutral", (0.58, 0.61, 0.64), 0.25, 0.75),
    TSlotBackgroundAppearance("cool_dim", (0.46, 0.52, 0.60), 0.27, 0.15),
    TSlotBackgroundAppearance("warm_bright", (0.64, 0.57, 0.48), 0.48, 0.10),
)
VISUAL_SERVO_CANONICAL_TSLOT_BACKGROUND_INDEX = 0


def validate_render_only_tslot_asset(path: Path = VISUAL_SERVO_TSLOT_ASSET) -> None:
    """Reject an absent asset or accidental collision schema before Isaac starts."""

    if not path.is_file():
        raise FileNotFoundError(f"Tracked T-slot visual asset does not exist: {path}")
    source = path.read_text(encoding="utf-8")
    forbidden = ("CollisionAPI", "PhysicsCollision", "RigidBodyAPI", "MeshCollisionAPI")
    found = [token for token in forbidden if token in source]
    if found:
        raise ValueError(
            f"T-slot asset must remain render-only; found collision/physics tokens {found} in {path}."
        )


def sample_tslot_layout_variants(
    num_envs: int,
    *,
    enabled: bool,
    seed: int,
    nominal_fraction: float = 0.60,
    phase_fraction: float = 0.20,
) -> tuple[TSlotLayoutVariant, ...]:
    """Return deterministic, weighted geometry layouts for cloned environments."""

    count = int(num_envs)
    if count < 1:
        raise ValueError("num_envs must be positive.")
    if not 0.0 <= nominal_fraction <= 1.0:
        raise ValueError("nominal_fraction must lie in [0, 1].")
    if not 0.0 <= phase_fraction <= 1.0:
        raise ValueError("phase_fraction must lie in [0, 1].")
    if nominal_fraction + phase_fraction > 1.0:
        raise ValueError("nominal_fraction + phase_fraction must not exceed 1.")
    if not enabled:
        return tuple(TSlotLayoutVariant("nominal", 0.0, 0.0) for _ in range(count))

    rng = random.Random(int(seed))
    order = list(range(count))
    rng.shuffle(order)
    nominal_count = round(count * nominal_fraction)
    phase_count = round(count * phase_fraction)
    if nominal_count + phase_count > count:
        phase_count = count - nominal_count
    # The part can be placed and grasped at different table-relative yaw
    # angles even though the robot and T-slot table are fixed.  A continuously
    # rotated visual background is therefore a proxy for the changing
    # table-to-wrist-camera relationship.  Slot orientation has 180-degree
    # symmetry, so [-90, 90] spans the complete unique range.
    result = [
        TSlotLayoutVariant(
            "rotated",
            rng.uniform(-0.5 * VISUAL_SERVO_TSLOT_PITCH_M, 0.5 * VISUAL_SERVO_TSLOT_PITCH_M),
            rng.uniform(-90.0, 90.0),
        )
        for _ in range(count)
    ]
    for env_index in order[:nominal_count]:
        result[env_index] = TSlotLayoutVariant("nominal", 0.0, 0.0)
    for env_index in order[nominal_count : nominal_count + phase_count]:
        result[env_index] = TSlotLayoutVariant(
            "phase_shifted",
            rng.uniform(-0.5 * VISUAL_SERVO_TSLOT_PITCH_M, 0.5 * VISUAL_SERVO_TSLOT_PITCH_M),
            0.0,
        )
    return tuple(result)


def sample_surface_appearance(
    base_color: tuple[float, float, float],
    base_roughness: float,
    unit_values: tuple[float, float, float, float],
    *,
    strength: float,
    color_scale: tuple[float, float],
    saturation_scale: tuple[float, float],
    hue_shift_deg: tuple[float, float],
    roughness: tuple[float, float],
) -> SurfaceAppearance:
    """Sample reset-stable HSV/value and roughness jitter around one preset."""

    values = tuple(float(value) for value in unit_values)
    if len(values) != 4 or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("unit_values must contain four values in [0, 1].")
    amount = float(strength)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("strength must lie in [0, 1].")
    for name, value_range in (
        ("color_scale", color_scale),
        ("saturation_scale", saturation_scale),
        ("hue_shift_deg", hue_shift_deg),
        ("roughness", roughness),
    ):
        if len(value_range) != 2 or value_range[0] > value_range[1]:
            raise ValueError(f"{name} must be an ordered pair.")
    if color_scale[0] <= 0.0 or saturation_scale[0] <= 0.0:
        raise ValueError("color and saturation scales must stay positive.")
    if roughness[0] < 0.0 or roughness[1] > 1.0:
        raise ValueError("roughness must stay within [0, 1].")

    def resolve(value: float, value_range: tuple[float, float]) -> float:
        return value_range[0] + value * (value_range[1] - value_range[0])

    hue, saturation, value = colorsys.rgb_to_hsv(*base_color)
    sampled_hue_shift = amount * resolve(values[0], hue_shift_deg) / 360.0
    sampled_saturation_scale = 1.0 + amount * (resolve(values[1], saturation_scale) - 1.0)
    sampled_color_scale = 1.0 + amount * (resolve(values[2], color_scale) - 1.0)
    sampled_roughness = resolve(values[3], roughness)
    varied_color = colorsys.hsv_to_rgb(
        (hue + sampled_hue_shift) % 1.0,
        min(1.0, max(0.0, saturation * sampled_saturation_scale)),
        min(1.0, max(0.0, value * sampled_color_scale)),
    )
    return SurfaceAppearance(
        color=tuple(min(1.0, max(0.0, channel)) for channel in varied_color),
        roughness=base_roughness + amount * (sampled_roughness - base_roughness),
    )


def sample_weighted_tslot_background_index(unit_value: float) -> int:
    value = float(unit_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("unit_value must lie in [0, 1].")
    total = sum(entry.weight for entry in VISUAL_SERVO_TSLOT_BACKGROUNDS)
    threshold = min(value, 1.0 - 1.0e-12) * total
    cumulative = 0.0
    for index, entry in enumerate(VISUAL_SERVO_TSLOT_BACKGROUNDS):
        cumulative += entry.weight
        if threshold < cumulative:
            return index
    return len(VISUAL_SERVO_TSLOT_BACKGROUNDS) - 1


def spawn_visual_servo_tslot_surfaces(
    num_envs: int,
    *,
    enabled: bool = True,
    geometry_randomization_enabled: bool = False,
    seed: int = 0,
    nominal_fraction: float = 0.60,
    phase_fraction: float = 0.20,
) -> dict[str, Any]:
    """Spawn the canonical small-pitch visual grooves over one flat collision plane."""

    count = int(num_envs)
    if count < 1:
        raise ValueError("num_envs must be positive.")
    if not enabled:
        return {
            "profile": VISUAL_SERVO_TSLOT_PROFILE,
            "asset": None,
            "variants": tuple(TSlotLayoutVariant("flat", 0.0, 0.0) for _ in range(count)),
            "prim_paths": {},
            "aluminum_shader_paths": {},
            "collision_surface": "/World/GroundPlane",
        }

    validate_render_only_tslot_asset()

    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sim.utils.transforms import standardize_xform_ops
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    ground = stage.GetPrimAtPath("/World/GroundPlane")
    if not ground.IsValid():
        raise RuntimeError("Expected /World/GroundPlane before spawning the T-slot visual surface.")
    # Visibility does not affect PhysX collision. This prevents the original
    # flat renderer geometry from occluding the shallow visual grooves.
    UsdGeom.Imageable(ground).MakeInvisible()

    variants = sample_tslot_layout_variants(
        num_envs,
        enabled=geometry_randomization_enabled,
        seed=seed,
        nominal_fraction=nominal_fraction,
        phase_fraction=phase_fraction,
    )
    cfg = sim_utils.UsdFileCfg(usd_path=str(VISUAL_SERVO_TSLOT_ASSET))
    first_prim_path = "/World/envs/env_0/VisualTSlotSurface"
    cfg.func(first_prim_path, cfg)
    shader_paths: dict[int, str] = {}
    prim_paths: dict[int, str] = {}
    for env_index, variant in enumerate(variants):
        env_path = f"/World/envs/env_{env_index}"
        if not stage.GetPrimAtPath(env_path).IsValid():
            raise RuntimeError(f"Expected cloned environment prim before T-slot spawn: {env_path}")
        prim_path = f"{env_path}/VisualTSlotSurface"
        half_angle = math.radians(variant.rotation_deg) * 0.5
        orientation = (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            # Some scene-cloning strategies do not propagate a child authored
            # after cloning. Add the same local reference explicitly there.
            cfg.func(prim_path, cfg)
            prim = stage.GetPrimAtPath(prim_path)
        phase_angle = math.radians(variant.rotation_deg)
        phase_x = variant.phase_m * math.cos(phase_angle)
        phase_y = variant.phase_m * math.sin(phase_angle)
        standardize_xform_ops(
            prim,
            (
                VISUAL_SERVO_TSLOT_CENTER[0] + phase_x,
                VISUAL_SERVO_TSLOT_CENTER[1] + phase_y,
                VISUAL_SERVO_TSLOT_CENTER[2],
            ),
            orientation,
            VISUAL_SERVO_TSLOT_SCALE,
        )
        prim_paths[env_index] = prim_path
        shader_paths[env_index] = f"{prim_path}/Looks/Aluminum/Shader"

    return {
        "profile": VISUAL_SERVO_TSLOT_PROFILE,
        "asset": str(VISUAL_SERVO_TSLOT_ASSET),
        "variants": variants,
        "prim_paths": prim_paths,
        "aluminum_shader_paths": shader_paths,
        "collision_surface": "/World/GroundPlane",
    }


class LiveWorkspaceAppearanceRandomizer:
    """Apply live part appearance and optional T-slot appearance per environment."""

    def __init__(
        self,
        *,
        part_shader_paths_by_env: Mapping[int, str],
        tslot_aluminum_shader_paths: Mapping[int, str],
        num_envs: int,
        device: torch.device | str,
        part_color_scale: tuple[float, float] = (0.90, 1.10),
        part_saturation_scale: tuple[float, float] = (0.90, 1.10),
        part_hue_shift_deg: tuple[float, float] = (-5.0, 5.0),
        part_roughness: tuple[float, float] = (0.65, 0.90),
        tslot_color_scale: tuple[float, float] = (0.88, 1.12),
        tslot_saturation_scale: tuple[float, float] = (0.90, 1.10),
        tslot_hue_shift_deg: tuple[float, float] = (-5.0, 5.0),
        tslot_roughness_delta: tuple[float, float] = (-0.08, 0.08),
    ) -> None:
        import omni.usd

        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.part_color_scale = tuple(part_color_scale)
        self.part_saturation_scale = tuple(part_saturation_scale)
        self.part_hue_shift_deg = tuple(part_hue_shift_deg)
        self.part_roughness = tuple(part_roughness)
        self.tslot_color_scale = tuple(tslot_color_scale)
        self.tslot_saturation_scale = tuple(tslot_saturation_scale)
        self.tslot_hue_shift_deg = tuple(tslot_hue_shift_deg)
        self.tslot_roughness_delta = tuple(tslot_roughness_delta)
        self.part_shader_paths_by_env = {
            int(env_index): str(path) for env_index, path in part_shader_paths_by_env.items()
        }
        self.tslot_aluminum_shader_paths = {
            int(env_index): str(path) for env_index, path in tslot_aluminum_shader_paths.items()
        }
        missing_parts = sorted(set(range(self.num_envs)) - set(self.part_shader_paths_by_env))
        if missing_parts:
            raise ValueError(
                "Live workspace randomization requires a part shader for every environment; "
                f"missing parts={missing_parts}."
            )
        self.stage = omni.usd.get_context().get_stage()
        self.part_palette_index = torch.full(
            (self.num_envs,), VISUAL_SERVO_CANONICAL_PART_INDEX, dtype=torch.long, device=self.device
        )
        # -1 identifies an explicitly disabled T-slot surface. Non-negative
        # values are indices into the canonical aluminum appearance palette.
        self.background_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    def sample(self, env_ids: torch.Tensor, *, strength: float | torch.Tensor) -> None:
        """Sample reset-stable live appearances with curriculum/clean masking."""

        from pxr import Gf

        ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        if isinstance(strength, torch.Tensor):
            strengths = strength.to(device=self.device, dtype=torch.float32).flatten()
            if strengths.numel() == 1:
                strengths = strengths.expand(ids.numel())
            elif strengths.numel() != ids.numel():
                raise ValueError("Per-environment appearance strength must match env_ids.")
        else:
            strengths = torch.full(
                (ids.numel(),), min(1.0, max(0.0, float(strength))), device=self.device
            )
        strengths = strengths.clamp(0.0, 1.0)
        active = torch.rand(ids.numel(), device=self.device) < strengths
        palette_units = torch.rand(ids.numel(), device=self.device).cpu().tolist()
        background_units = torch.rand(ids.numel(), device=self.device).cpu().tolist()
        material_units = torch.rand((ids.numel(), 8), device=self.device).cpu().tolist()
        active_cpu = active.cpu().tolist()
        strengths_cpu = strengths.cpu().tolist()

        for row, env_index_tensor in enumerate(ids.cpu()):
            env_index = int(env_index_tensor)
            if active_cpu[row]:
                palette_index = sample_weighted_part_palette_index(palette_units[row])
                background_index = sample_weighted_tslot_background_index(background_units[row])
            else:
                palette_index = VISUAL_SERVO_CANONICAL_PART_INDEX
                background_index = VISUAL_SERVO_CANONICAL_TSLOT_BACKGROUND_INDEX

            part_appearance = VISUAL_SERVO_PART_PALETTE[palette_index]
            varied_part = sample_surface_appearance(
                part_appearance.color,
                VISUAL_SERVO_PART_ROUGHNESS,
                tuple(material_units[row][:4]),
                strength=strengths_cpu[row] if active_cpu[row] else 0.0,
                color_scale=self.part_color_scale,
                saturation_scale=self.part_saturation_scale,
                hue_shift_deg=self.part_hue_shift_deg,
                roughness=self.part_roughness,
            )
            part_shader = self.stage.GetPrimAtPath(self.part_shader_paths_by_env[env_index])
            if not part_shader.IsValid():
                raise RuntimeError(
                    f"Per-environment part shader does not exist: {self.part_shader_paths_by_env[env_index]}"
                )
            part_color_attr = part_shader.GetAttribute("inputs:diffuseColor")
            part_roughness_attr = part_shader.GetAttribute("inputs:roughness")
            if not part_color_attr.Set(Gf.Vec3f(*varied_part.color)) or not part_roughness_attr.Set(
                varied_part.roughness
            ):
                raise RuntimeError(f"Failed to author part appearance for env {env_index}.")

            tslot_shader_path = self.tslot_aluminum_shader_paths.get(env_index)
            if tslot_shader_path is None:
                self.background_index[env_index] = -1
                self.part_palette_index[env_index] = palette_index
                continue

            background = VISUAL_SERVO_TSLOT_BACKGROUNDS[background_index]
            varied_background = sample_surface_appearance(
                background.color,
                background.roughness,
                tuple(material_units[row][4:]),
                strength=strengths_cpu[row] if active_cpu[row] else 0.0,
                color_scale=self.tslot_color_scale,
                saturation_scale=self.tslot_saturation_scale,
                hue_shift_deg=self.tslot_hue_shift_deg,
                roughness=(
                    max(0.0, background.roughness + self.tslot_roughness_delta[0]),
                    min(1.0, background.roughness + self.tslot_roughness_delta[1]),
                ),
            )
            shader = self.stage.GetPrimAtPath(tslot_shader_path)
            if not shader.IsValid():
                raise RuntimeError(
                    f"T-slot aluminum shader does not exist: {tslot_shader_path}"
                )
            color_attr = shader.GetAttribute("inputs:diffuseColor")
            roughness_attr = shader.GetAttribute("inputs:roughness")
            if not color_attr.Set(Gf.Vec3f(*varied_background.color)) or not roughness_attr.Set(
                varied_background.roughness
            ):
                raise RuntimeError(f"Failed to author T-slot appearance for env {env_index}.")
            self.part_palette_index[env_index] = palette_index
            self.background_index[env_index] = background_index


__all__ = [
    "LiveWorkspaceAppearanceRandomizer",
    "SurfaceAppearance",
    "TSlotBackgroundAppearance",
    "TSlotLayoutVariant",
    "VISUAL_SERVO_CANONICAL_TSLOT_BACKGROUND_INDEX",
    "VISUAL_SERVO_TSLOT_ASSET",
    "VISUAL_SERVO_TSLOT_BACKGROUNDS",
    "VISUAL_SERVO_TSLOT_CENTER",
    "VISUAL_SERVO_TSLOT_PITCH_M",
    "VISUAL_SERVO_TSLOT_PROFILE",
    "VISUAL_SERVO_TSLOT_SCALE",
    "sample_tslot_layout_variants",
    "sample_surface_appearance",
    "sample_weighted_tslot_background_index",
    "spawn_visual_servo_tslot_surfaces",
    "validate_render_only_tslot_asset",
]
