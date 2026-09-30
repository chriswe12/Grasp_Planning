"""Cached, geometry-preserving reference variation, independent of live resets."""

import hashlib

import numpy as np

GOAL_VARIANTS_PROFILE = {
    "version": "franka_mixed_isaac_mujoco_goals_v1",
    "renderers": ["isaac_canonical", "isaac", "isaac", "mujoco", "mujoco"],
    "probabilities": [0.20, 0.15, 0.15, 0.25, 0.25],
    "geometry": "canonical_object_camera_grasp_no_live_world_pose",
    "color_sampling": "independent_of_live_appearance",
    "refresh": "once_per_episode",
    "storage": "four_float16_rgbd_variants_plus_refreshed_blue_canonical",
    "depth": "each_renderer_optical_z_metres_shared_packing",
}


def variant_digest(images):
    return hashlib.sha256(memoryview(np.ascontiguousarray(images)).cast("B")).hexdigest()


def validate_variants(data, profile):
    base = {k: v for k, v in profile.items() if k != "images_sha256"}
    if base != GOAL_VARIANTS_PROFILE:
        raise ValueError("Unsupported mixed goal profile")
    images = data["goal_rgbd_variants"]
    shape = data["goal_rgbd"].shape[1:] if "goal_rgbd" in data else (72, 128, 4)
    if images.shape != (len(data["target_ids"]), 4, *shape) or images.dtype != np.float16:
        raise ValueError("Invalid mixed goal bank shape or dtype")
    # Bounded temporary memory even for full catalogs.
    for chunk in images:
        if not np.isfinite(chunk).all() or chunk.min() < 0 or chunk.max() > 1:
            raise ValueError("Invalid mixed goal pixels")
    if variant_digest(images) != profile.get("images_sha256"):
        raise ValueError("Mixed goal bank digest mismatch")


class GoalVariantBank:
    """Keep variants on CPU; transfer only the images selected at reset."""

    def __init__(self, data, selected, profile, device):
        import torch

        validate_variants(data, profile)
        self.images = torch.from_numpy(np.ascontiguousarray(data["goal_rgbd_variants"][selected]))
        self.weights = torch.tensor(profile["probabilities"], device=device)
        self.device = device

    def sample(self, targets, canonical, fixed_index=-1):
        import torch

        n = len(targets)
        if fixed_index < -1 or fixed_index > 4:
            raise ValueError("Goal variant index must be -1 (sample) or 0..4")
        indices = (
            torch.multinomial(self.weights, n, replacement=True)
            if fixed_index == -1
            else torch.full((n,), fixed_index, device=self.device, dtype=torch.long)
        )
        result = canonical[targets].clone()
        mask = indices > 0
        if mask.any():
            result[mask] = self.images[targets[mask].cpu(), indices[mask].cpu() - 1].to(
                device=self.device, dtype=result.dtype
            )
        return result, indices
