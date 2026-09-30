#!/usr/bin/env python3
"""Measure actual randomized Panda reset errors and verify both curriculum endpoints."""

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "isaac_rl/source/isaac_rl")]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--catalog", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--rounds", type=int, default=16)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
import torch
from PIL import Image, ImageDraw

from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg


def main():
    cfg = FrankaZedEnvCfg()
    cfg.seed = 43
    cfg.scene.num_envs = args.num_envs
    cfg.catalog_path = str(args.catalog)
    cfg.sim.device = args.device
    cfg.robot_asset_manifest = str(ROOT / "assets/usd/franka_panda_offline/manifest.json")
    env = FrankaZedEnv(cfg)
    assert env.step_dt == 1 / 15 and env.physics_dt == 1 / 120
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        env.reset()
        assert env._pose_curriculum().fraction == 0
        # Explicitly exercise the final distribution before authorizing training.
        env.pose_curriculum_offset = int(192001 * 512 / env.num_envs)
        assert env._pose_curriculum().fraction == 1
        for round_i in range(args.rounds):
            env.reset()
            pe, re = env.pose_errors()
            pn = pe.norm(dim=-1)
            rn = re.norm(dim=-1)
            bank = env.reset_bank_index
            assert (bank >= 0).all(), "A reset bypassed the physics-validated bank"
            ids = env.target_index
            for i in range(env.num_envs):
                row = dict(
                    round=round_i,
                    slot=i,
                    target=env.target_ids[int(ids[i])],
                    mode=int(env.reset_mode[i]),
                    bank=int(bank[i]),
                    kind=int(env.pose_reset_kind[bank[i]]) if bank[i] >= 0 else -1,
                    progress=float(env.reset_progress[i]),
                    position_mm=float(pn[i] * 1000),
                    rotation_deg=float(rn[i] * 180 / torch.pi),
                    position_vector=pe[i].cpu().tolist(),
                    rotation_vector=re[i].cpu().tolist(),
                )
                if bank[i] >= 0:
                    row["lateral_mm"] = float(env.pose_reset_lateral_m[ids[i], bank[i]] * 1000)
                    row["stored_position_mm"] = float(env.pose_reset_position_error_m[ids[i], bank[i]] * 1000)
                    assert abs(row["stored_position_mm"] - row["position_mm"]) < 2.0, row
                    assert abs(float(env.pose_reset_rotation_error_rad[ids[i], bank[i]]) - float(rn[i])) < 0.03, row
                rows.append(row)
            for _ in range(12):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
            assert float(env.contact_force().max()) < 1.0, "Unsafe sampled reset"
            if round_i == 0:
                for _ in range(3):
                    env.sim.render()
                    env.scene.update(env.physics_dt)
                env.wrist_camera.update(env.physics_dt, force_recompute=True)
                images = env.wrist_camera.data.output["rgb"][..., :3].cpu().numpy()
                grid = Image.new("RGB", (4 * 256, 4 * 176), "#14202c")
                draw = ImageDraw.Draw(grid)
                for i in range(min(16, len(images))):
                    x = (i % 4) * 256
                    y = (i // 4) * 176
                    grid.paste(Image.fromarray(images[i]), (x, y))
                    draw.text(
                        (x + 3, y + 146),
                        f"{float(pn[i] * 1000):.1f}mm / {float(rn[i] * 180 / torch.pi):.1f}deg",
                        fill="white",
                    )
                grid.save(args.output / "randomized_starts.jpg")
    path = [r for r in rows if r["mode"] == 0 and r["kind"] == 0]
    assert len(path) > len(rows) * 0.35, "Full curriculum still mostly unperturbed"
    assert min(r["rotation_deg"] for r in path) > 4.0, "Rotational perturbation collapsed"
    assert max(r["rotation_deg"] for r in path) > 14.0, "Missing far orientation errors"
    assert max(r["lateral_mm"] for r in path) > 8.0, "Missing lateral variation"
    assert len(set(r["bank"] % 8 for r in path)) == 8, "Rotation axes missing"
    boundary = [r for r in rows if r["mode"] == 3]
    ready = [r for r in rows if r["mode"] == 2]
    assert ready and all(r["position_mm"] < 4 and r["rotation_deg"] < 3 for r in ready), (
        "Unsafe/mislabeled positive reset"
    )
    assert boundary
    assert sum(r["position_mm"] > 4 or r["rotation_deg"] > 3 for r in boundary) > 0.5 * len(boundary), (
        "Boundary pool collapsed to positive examples"
    )
    assert any(r["position_mm"] < 1 and r["rotation_deg"] > 4.5 for r in boundary), (
        "Missing isolated orientation negatives"
    )
    assert any(r["position_mm"] > 6 and r["rotation_deg"] < 0.5 for r in boundary), (
        "Missing isolated position negatives"
    )
    assert any(r["position_mm"] < 4 and r["rotation_deg"] < 0.5 for r in boundary), "Missing near-threshold positives"
    report = {
        "passed": True,
        "policy_hz": 1 / env.step_dt,
        "physics_hz": 1 / env.physics_dt,
        "samples": len(rows),
        "perturbed_path_samples": len(path),
        "mode_counts": {str(m): sum(r["mode"] == m for r in rows) for m in range(4)},
        "rotation_deg_range": [min(r["rotation_deg"] for r in path), max(r["rotation_deg"] for r in path)],
        "lateral_mm_range": [min(r["lateral_mm"] for r in path), max(r["lateral_mm"] for r in path)],
        "rows": rows,
    }
    (args.output / "reset_runtime_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[POSE RUNTIME AUDIT] PASSED", json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
