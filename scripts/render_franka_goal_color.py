#!/usr/bin/env python3
"""Render one selected goal using the Isaac training task and an object material override."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac_rl/source/isaac_rl"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--request", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from PIL import Image
from pxr import Gf, Usd, UsdShade

from grasp_planning.rl.franka_fabrica import resolve_project_path
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg


def main():
    request = json.loads(args.request.read_text())
    contract = request["contract"]
    cfg = FrankaZedEnvCfg()
    cfg.camera_profile_data = contract.get("camera_profile_data")
    cfg.depth_source = contract.get("depth_source", "legacy_image_plane")
    if cfg.depth_source != "legacy_image_plane":
        cfg.appearance_randomization = contract.get("appearance_randomization")
        cfg.performance_profile = contract.get("performance_profile")
    cfg.seed = 42
    cfg.build_catalog = True
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.object_assets = [request["asset"]]
    cfg.env_part_indices = [0]
    cfg.robot_asset_manifest = "assets/usd/franka_panda_offline/manifest.json"
    cfg.gripper_open_width_m = contract["gripper_open_width_m"]
    lab = contract["lab_scene"]
    cfg.lab_asset_dir = str(resolve_project_path(lab["asset_dir"]))
    cfg.lab_translation = tuple(lab["translation_m"])
    cfg.lab_appearance_seed = lab["appearance_seed"]
    cfg.lab_props = lab["props"]
    # Catalog upgrades preserved the original canonical goal images. Reuse their
    # original balanced renderer and canonical lighting, not randomized live lighting.
    env = FrankaZedEnv(cfg)
    try:
        if cfg.depth_source != "legacy_image_plane" and env.appearance:
            # Same fixed appearance seed as the calibrated bank's canonical goal.
            env.appearance.apply_many([0], [4242])
        actual = env.contract()
        for key in ("camera_profile", "lab_scene", "robot_usd"):
            if actual.get(key) != contract.get(key):
                raise ValueError(f"Training goal scene changed: {key}")
        root = env.scene.env_prim_paths[0] + "/Part/material/"
        shaders = [
            UsdShade.Shader(p)
            for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath(root.rstrip("/")))
            if p.IsA(UsdShade.Shader) and UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"
        ]
        if len(shaders) != 1:
            raise ValueError("Expected one bound object material")
        color = request["metadata"]["color"]
        srgb = np.array([int(color[j : j + 2], 16) / 255 for j in (1, 3, 5)])
        linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        shaders[0].GetInput("diffuseColor").Set(Gf.Vec3f(*linear.tolist()))

        def tensor(value):
            return torch.as_tensor(value, device=env.device, dtype=torch.float32).unsqueeze(0)

        with torch.inference_mode():
            env.write_state(
                tensor(request["joints"]), tensor(request["object_pose"]), open_width=tensor(request["open_width"])
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            for _ in range(12):
                env.sim.render()
                env.scene.update(env.physics_dt)
            rgbd = env.rgbd()[0].cpu().numpy().astype(np.float32)
        if rgbd.shape != (72, 128, 4) or not np.isfinite(rgbd).all():
            raise ValueError("Invalid Isaac RGB-D render")
        rendered_depth = rgbd[..., 3].copy()
        # A color choice must not change the checkpoint's canonical depth input.
        # Retain the fresh sensor output separately for renderer diagnostics.
        rgbd[..., 3] = np.asarray(request["goal_depth"], dtype=np.float32)
        folder = args.request.parent
        Image.fromarray((rgbd[..., :3].clip(0, 1) * 255).astype("uint8")).save(folder / "goal.png")
        temporary = folder / "goal.partial.npz"
        np.savez_compressed(
            temporary,
            goal_rgbd=rgbd,
            rendered_depth=rendered_depth,
            metadata_json=np.asarray(json.dumps(request["metadata"], sort_keys=True)),
        )
        temporary.replace(folder / "goal.npz")
        print("COLOR GOAL COMPLETE", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close(wait_for_replicator=False)
