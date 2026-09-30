"""Exact training catalog, actor and camera contracts for manual-start deployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from grasp_planning.rl.deployment_model.completion_model import GraspCompletionModel
from grasp_planning.rl.deployment_model.resnet_rgbd_network import GraspRgbdResNetBuilder
from grasp_planning.rl.zed_mini import pack_zed_rgbd, resolve_zed_profile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "isaac_rl/data/franka_clutter_v5_fast_fxaa/catalog.npz"
DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts/franka_speedup_20260917/euler_results/franka/2026-09-18_04-16-08_job_14461123_4gpu/nn/last_franka_zed_ep_2000_rew_28.62624.pth"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-4:
        raise ValueError("Expected normalized WXYZ quaternion")
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform(pose):
    t = np.eye(4)
    t[:3, :3] = rotation(pose[3:])
    t[:3, 3] = pose[:3]
    return t


class Catalog:
    def __init__(self, path=DEFAULT_CATALOG):
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as d:
            keys = [
                "target_ids",
                "validated",
                "goal_rgbd",
                "goal_poses",
                "object_poses",
                "open_widths",
                "jaw_widths",
                "orientation_ids",
                "part_keys",
                "source_grasp_ids",
                "split",
                "joint_paths",
            ]
            self.data = {k: d[k].copy() for k in keys}
            self.contract = json.loads(str(d["contract_json"].item()))
            if self.contract.get("goal_randomization"):
                from grasp_planning.rl.franka_goal_variants import validate_variants

                self.data["goal_rgbd_variants"] = d["goal_rgbd_variants"].copy()
                validate_variants(self.data, self.contract["goal_randomization"])

        self.profile = resolve_zed_profile(self.contract)
        if self.contract["training_recipe"]["policy_hz"] != 15:
            raise ValueError("This deployment requires the 15 Hz recipe")
        if not np.isfinite(self.data["goal_rgbd"]).all():
            raise ValueError("Nonfinite catalog image")
        self.hash = sha256(self.path)

    def indices(self, part=None, orientation=None):
        mask = self.data["validated"].copy()
        if part is not None:
            mask &= self.data["part_keys"] == part
        if orientation is not None:
            mask &= self.data["orientation_ids"] == orientation
        return np.flatnonzero(mask)

    def index(self, target):
        i = np.flatnonzero(self.data["target_ids"] == target)
        if len(i) != 1 or not self.data["validated"][i[0]]:
            raise ValueError("Target is absent or not validated")
        return int(i[0])

    def diverse(self, indices, count=12):
        """Farthest-point coverage of translation, orientation and gripper aperture."""
        ids = np.asarray(indices, dtype=int)
        if len(ids) <= count:
            return ids.tolist()
        p = self.data["goal_poses"][ids]
        q = p[:, 3:]
        dist = np.linalg.norm(p[:, None, :3] - p[None, :, :3], axis=-1) / 0.03
        dist += 2 * np.arccos(np.clip(abs(q @ q.T), 0, 1)) / (np.pi / 4)
        widths = self.data["jaw_widths"][ids]
        dist += abs(widths[:, None] - widths[None, :]) / 0.02
        chosen = [0]
        nearest = dist[0].copy()
        while len(chosen) < count:
            nearest[chosen] = -1
            nxt = int(np.argmax(nearest))
            chosen.append(nxt)
            nearest = np.minimum(nearest, dist[nxt])
        return ids[chosen].tolist()

    def select(self, i, checkpoint=DEFAULT_CHECKPOINT):
        return dict(
            schema_version=1,
            catalog=str(self.path),
            catalog_sha256=self.hash,
            target_id=str(self.data["target_ids"][i]),
            checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_sha256=sha256(checkpoint),
            robot_ip="192.168.1.200",
            robot_model="fr3",
            camera_serial=13829658,
            mount_confirmed=False,
            workspace_min_m=[0.15, -0.65, 0.02],
            workspace_max_m=[0.85, 0.65, 0.85],
            max_linear_speed_m_s=0.01,
            max_angular_speed_rad_s=0.06,
            max_displacement_m=0.15,
            max_rotation_rad=0.2,
            max_duration_s=None,
            external_force_stop_n=10.0,
            external_torque_stop_nm=5.0,
            minimum_valid_depth_fraction=0.20,
            maximum_frame_age_s=0.15,
            manual_gripper_force_n=10.0,
        )


def live_to_training(rgb, depth, calibration, profile, device="cpu"):
    """Ray-preserving crop to the training K, then its exact RGB-D packing.

    ZED images are already rectified. No camera pose or depth-value scaling.
    Out-of-frame depth stays invalid, never a fabricated near surface.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB must be uint8 HWC")
    if depth.shape != rgb.shape[:2] or depth.dtype != np.float32:
        raise ValueError("Depth must be aligned float32 metres")
    h, w = rgb.shape[:2]
    if (w, h) != (calibration["width"], calibration["height"]):
        raise ValueError("Capture/calibration resolution mismatch")
    tw, th = profile["render_width"], profile["render_height"]
    fx = profile["fx"] * tw / profile["source_width"]
    fy = profile["fy"] * th / profile["source_height"]
    cx = profile["cx"] * tw / profile["source_width"]
    cy = profile["cy"] * th / profile["source_height"]
    yy, xx = torch.meshgrid(torch.arange(th, device=device), torch.arange(tw, device=device), indexing="ij")
    u = (xx - cx) / fx * calibration["fx"] + calibration["cx"]
    v = (yy - cy) / fy * calibration["fy"] + calibration["cy"]
    grid = torch.stack((2 * (u + 0.5) / w - 1, 2 * (v + 0.5) / h - 1), -1)[None]
    color = torch.from_numpy(rgb.copy()).to(device).float().permute(2, 0, 1)[None] / 255
    z = torch.from_numpy(depth.copy()).to(device)[None, None]
    valid = torch.isfinite(z) & (z >= profile["depth_min_m"]) & (z < profile["depth_max_m"])

    def sample(x, mode):
        return torch.nn.functional.grid_sample(x, grid, mode=mode, align_corners=False)

    color = sample(color, "bilinear").permute(0, 2, 3, 1)
    z = sample(torch.where(valid, z, 0), "nearest")
    valid = sample(valid.float(), "nearest") > 0.5
    z = torch.where(valid, z, float("nan")).permute(0, 2, 3, 1)
    return pack_zed_rgbd(color, z, profile)


class Actor:
    def __init__(self, catalog, checkpoint=DEFAULT_CHECKPOINT, device="cuda:0"):
        self.device = torch.device(device)
        self.catalog = catalog
        checkpoint = Path(checkpoint)
        contract = json.loads(checkpoint.with_suffix(".contract.json").read_text())
        if contract != catalog.contract:
            raise ValueError("Checkpoint contract does not match selected catalog")
        params = catalog.contract["training_recipe"]["agent"]["params"]
        network = dict(params["network"])
        network["pretrained"] = False
        b = GraspRgbdResNetBuilder()
        b.load(network)
        self.model = GraspCompletionModel(b).build(
            dict(
                actions_num=7,
                input_shape=(72 * 128 * 8 + 14,),
                num_seqs=1,
                value_size=1,
                normalize_value=params["config"]["normalize_value"],
                normalize_input=False,
            )
        )
        # Only user-owned local checkpoints; their saved optimizer uses numpy scalars.
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(data["model"], strict=True)
        self.model.to(self.device).eval()
        self.epoch = int(data.get("epoch", 0))
        self.previous = np.zeros(6, dtype=np.float32)
        self.goal = None

    def set_target(self, i, config=None):
        from .goal_color import load_goal

        self.goal = torch.from_numpy(load_goal(self.catalog, i, config or {})).to(self.device)[None]
        self.previous[:] = 0

    @torch.inference_mode()
    def infer(self, live):
        if self.goal is None:
            raise ValueError("Select a target before inference")
        obs = torch.cat(
            (
                torch.cat((live.to(self.device), self.goal), -1).flatten(1),
                torch.as_tensor(self.previous, device=self.device)[None],
                torch.zeros((1, 8), device=self.device),
            ),
            1,
        )
        a = self.model({"obs": obs, "is_train": False})["mus"][0].cpu().numpy()
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError("Nonfinite/invalid actor output")
        return a


def validate_execute_config(cfg):
    if cfg.get("robot_model") != "fr3":
        raise ValueError("This ROS connection uses the installed FR3 model")
    if not cfg.get("mount_confirmed"):
        raise ValueError("Confirm the hand/camera transform and command axes before setting mount_confirmed=true")
    for key, hi in [
        ("max_linear_speed_m_s", 0.04),
        ("max_angular_speed_rad_s", 0.24),
        ("max_displacement_m", 0.15),
        ("max_rotation_rad", 0.3),
        ("external_force_stop_n", 15.0),
        ("external_torque_stop_nm", 5.0),
    ]:
        if not 0 < float(cfg[key]) <= hi:
            raise ValueError(f"Invalid conservative limit: {key}")
    duration = cfg.get("max_duration_s")
    if duration is not None and (not np.isfinite(float(duration)) or float(duration) <= 0):
        raise ValueError("max_duration_s must be null (unlimited) or positive finite seconds")
    low = np.asarray(cfg["workspace_min_m"])
    high = np.asarray(cfg["workspace_max_m"])
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or not (low < high).all():
        raise ValueError("Invalid base-frame workspace")
    if not 0 < cfg["minimum_valid_depth_fraction"] <= 1 or not 0 < cfg["maximum_frame_age_s"] <= 0.2:
        raise ValueError("Invalid camera gate")
