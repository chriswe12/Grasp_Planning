"""Render visual-servo curriculum episodes as diagnostic RGB frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeVisualization:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    depth_min_m: float
    depth_max_m: float

    @property
    def step_count(self) -> int:
        return int(self.arrays["rgb_live"].shape[0])


def load_episode_visualization(npz_path: str | Path) -> EpisodeVisualization:
    """Load one curriculum episode and choose stable depth display limits."""

    npz_path = Path(npz_path)
    metadata_path = npz_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"Missing episode metadata: {metadata_path}")
    with np.load(npz_path) as episode:
        arrays = {name: episode[name].copy() for name in episode.files}
    required = {
        "rgb_live",
        "depth_live",
        "object_mask",
        "rgb_goal",
        "nominal_twist",
        "expert_twist",
        "expert_residual_twist",
        "pose_error",
        "trajectory_progress",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"{npz_path} is missing visualization arrays: {missing}")
    finite_depth = arrays["depth_live"][np.isfinite(arrays["depth_live"])]
    if finite_depth.size == 0:
        raise ValueError(f"{npz_path} contains no finite depth values.")
    depth_min_m, depth_max_m = np.percentile(finite_depth, (2.0, 98.0))
    if depth_max_m <= depth_min_m:
        depth_max_m = depth_min_m + 1.0e-3
    return EpisodeVisualization(
        arrays=arrays,
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
    )


def _label_tile(image_rgb: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    tile = np.asarray(image_rgb, dtype=np.uint8).copy()
    image = Image.fromarray(tile)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, tile.shape[1], min(24, tile.shape[0])), fill=(0, 0, 0))
    draw.text((7, 5), label, fill=(255, 255, 255))
    return np.asarray(image)


def load_overview_frames(npz_path: str | Path, *, expected_steps: int) -> np.ndarray:
    """Load synchronized overview RGB frames produced by the Isaac replay tool."""

    with np.load(Path(npz_path)) as payload:
        if "overview_rgb" not in payload:
            raise ValueError(f"{npz_path} does not contain an 'overview_rgb' array.")
        frames = np.asarray(payload["overview_rgb"], dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"overview_rgb must have shape (T, H, W, 3), got {frames.shape}.")
    if frames.shape[0] != expected_steps:
        raise ValueError(
            f"Overview has {frames.shape[0]} frames but episode has {expected_steps} steps."
        )
    return frames


def prepend_overview_frame(
    diagnostic_rgb: np.ndarray,
    overview_rgb: np.ndarray,
    *,
    label: str = "ISAAC OVERVIEW / WHOLE ROBOT",
) -> np.ndarray:
    """Resize and prepend a labeled overview image to a diagnostic frame."""

    from PIL import Image

    diagnostic = np.asarray(diagnostic_rgb, dtype=np.uint8)
    overview = np.asarray(overview_rgb, dtype=np.uint8)
    target_height = diagnostic.shape[0]
    target_width = max(1, int(round(overview.shape[1] * target_height / overview.shape[0])))
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized = np.asarray(Image.fromarray(overview).resize((target_width, target_height), bilinear))
    return np.concatenate((_label_tile(resized, label), diagnostic), axis=1)


def _depth_rgb(depth_m: np.ndarray, *, minimum_m: float, maximum_m: float) -> np.ndarray:
    normalized = np.nan_to_num(
        (np.asarray(depth_m, dtype=np.float32) - minimum_m) / (maximum_m - minimum_m),
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    value = np.clip(1.0 - normalized, 0.0, 1.0)
    # Compact blue-cyan-yellow-red depth ramp without a plotting dependency.
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[np.asarray(mask) > 0] = (60, 230, 80)
    return image


def _vector_lines(name: str, vector: np.ndarray, *, scale: float) -> list[str]:
    values = np.asarray(vector, dtype=np.float64) * float(scale)
    return [
        f"{name} linear:  {values[0]:+7.3f} {values[1]:+7.3f} {values[2]:+7.3f}",
        f"{name} angular: {values[3]:+7.3f} {values[4]:+7.3f} {values[5]:+7.3f}",
    ]


def render_episode_frame(episode: EpisodeVisualization, step_index: int) -> np.ndarray:
    """Render one tiled diagnostic frame in RGB order."""

    from PIL import Image, ImageDraw

    if not 0 <= step_index < episode.step_count:
        raise IndexError(f"Step {step_index} is outside [0, {episode.step_count}).")
    arrays = episode.arrays
    live = np.asarray(arrays["rgb_live"][step_index], dtype=np.uint8)
    height, width = live.shape[:2]
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    goal = np.asarray(
        Image.fromarray(np.asarray(arrays["rgb_goal"], dtype=np.uint8)).resize((width, height), bilinear)
    )
    depth = _depth_rgb(
        arrays["depth_live"][step_index],
        minimum_m=episode.depth_min_m,
        maximum_m=episode.depth_max_m,
    )
    mask = _mask_rgb(arrays["object_mask"][step_index])
    tiles = np.concatenate(
        (
            np.concatenate((_label_tile(live, "LIVE RGB"), _label_tile(goal, "FIXED GOAL RGB")), axis=1),
            np.concatenate(
                (
                    _label_tile(depth, f"DEPTH {episode.depth_min_m:.2f}-{episode.depth_max_m:.2f} m"),
                    _label_tile(mask, "PRIVILEGED TARGET MASK"),
                ),
                axis=1,
            ),
        ),
        axis=0,
    )

    panel_width = 384
    panel_image = Image.new("RGB", (panel_width, 2 * height), (22, 22, 22))
    panel_draw = ImageDraw.Draw(panel_image)
    metadata = episode.metadata
    action_label = (
        "policy "
        if metadata.get("controller") == "learned_policy"
        else "expert "
    )
    progress = float(arrays["trajectory_progress"][step_index])
    position_error_mm = np.asarray(arrays["pose_error"][step_index, :3]) * 1000.0
    rotation_error_deg = np.rad2deg(np.asarray(arrays["pose_error"][step_index, 3:]))
    controller_stage = (
        "PRECISION DOCKING"
        if "controller_stage" in arrays and int(arrays["controller_stage"][step_index]) == 1
        else "ALIGNMENT FUNNEL"
    )
    lines = [
        f"Episode {int(metadata.get('episode_index', -1)):06d}",
        f"split={metadata.get('split', '?')} success={metadata.get('success', '?')}",
        f"step={step_index + 1}/{episode.step_count} progress={progress:.3f}",
        f"controller stage: {controller_stage}",
        "",
        f"position error [mm]: {np.linalg.norm(position_error_mm):.3f}",
        f"  xyz: {position_error_mm[0]:+6.2f} {position_error_mm[1]:+6.2f} {position_error_mm[2]:+6.2f}",
        f"rotation error [deg]: {np.linalg.norm(rotation_error_deg):.3f}",
        f"  xyz: {rotation_error_deg[0]:+6.2f} {rotation_error_deg[1]:+6.2f} {rotation_error_deg[2]:+6.2f}",
        "",
        *_vector_lines("nominal", arrays["nominal_twist"][step_index], scale=1.0),
        *_vector_lines(action_label, arrays["expert_twist"][step_index], scale=1.0),
        *_vector_lines("residual", arrays["expert_residual_twist"][step_index], scale=1.0),
        "",
        "twist units: m/s and rad/s",
    ]
    y = 18
    for line in lines:
        color = (240, 240, 240)
        if line.startswith("position") or line.startswith("rotation"):
            color = (110, 220, 255)
        panel_draw.text((12, y - 11), line, fill=color)
        y += 17
    panel = np.asarray(panel_image)
    return np.concatenate((tiles, panel), axis=1)
