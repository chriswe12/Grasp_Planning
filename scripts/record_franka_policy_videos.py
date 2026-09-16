#!/usr/bin/env python3
"""Record trained-policy replays with overview, wrist RGB, goal RGB and honest terminal cards."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "isaac_rl/source/isaac_rl"))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cases", type=Path, required=True)
p.add_argument("--font", type=Path, help="Optional TrueType font for readable video labels")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--catalog", type=Path, default=ROOT / "isaac_rl/data/franka_fabrica_all_complete/catalog.npz")
p.add_argument(
    "--agent-config", type=Path, default=ROOT / "artifacts/franka_benchmark_20260916/epoch1800_validation/agent.yaml"
)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
import isaaclab.sim as sim_utils
import numpy as np
import torch
import yaml
from isaaclab.sensors import CameraCfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from PIL import Image, ImageDraw, ImageFont
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaac_rl.tasks.direct.isaac_rl.agents.completion_ppo import register_grasp_completion_runner
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg


def main():
    cases = json.loads(args.cases.read_text())
    cfg = FrankaZedEnvCfg()
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.num_envs = 34
    cfg.catalog_path = str(args.catalog)
    cfg.catalog_split = "validation"
    cfg.robot_asset_manifest = str(ROOT / "assets/usd/franka_panda_offline/manifest.json")
    cfg.rgb_gain_randomization = 0.0
    cfg.reset_ready_fraction = 0.0
    cfg.fixed_waypoint_index = 0
    cfg.sequential_target_assignment = True
    cfg.scene.inspection_camera = CameraCfg(
        prim_path="/World/InspectionCamera",
        width=960,
        height=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 15.0)),
    )
    env = FrankaZedEnv(cfg)
    # This single overview camera has one instance, not one per environment.
    # Manage its updates explicitly so batched environment resets cannot index it.
    camera = env.scene.sensors.pop("inspection_camera")
    wrapped = RlGamesVecEnvWrapper(env, args.device, 5.0, 1.0)
    vecenv.register("IsaacRlgWrapper", lambda name, actors, **kw: RlGamesGpuEnv(name, actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    config = yaml.safe_load(args.agent_config.read_text())
    config["params"]["config"].update(device=args.device, device_name=args.device, num_actors=34, multi_gpu=False)
    runner = Runner()
    register_grasp_completion_runner(runner)
    runner.load(config)
    runner.reset()
    player = runner.create_player()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(args.font), 18) if args.font else ImageFont.load_default()
    small = ImageFont.truetype(str(args.font), 14) if args.font else ImageFont.load_default()
    original_table = env.part_target_table.clone()
    original_counts = env.part_target_counts.clone()
    report = {
        "fps": 15,
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "episodes": [],
        "notes": [
            "Actual checkpoint inference; no scripted controller supplies policy motion.",
            "Inspection replays use fresh seeded appearance; outcomes are recorded rather than assumed from the benchmark.",
            "Object remains kinematic, hand remains open; success is alignment plus a completion declaration.",
            "Frames are captured before each action. The final card freezes the last pre-action frame and reports terminal physics metrics; no auto-reset frame is presented as terminal.",
        ],
    }
    try:
        for case in cases:
            checkpoint = Path(case["checkpoint"])
            assert json.loads(checkpoint.with_suffix(".contract.json").read_text()) == env.contract(), (
                "Checkpoint scene/camera mismatch"
            )
            player.restore(str(checkpoint))
            target = env.target_ids.index(case["target_id"])
            matches = (original_table == target).any(dim=1).nonzero().flatten()
            assert len(matches), "Target has no matching geometry environment"
            slot = int(matches[0])
            env.part_target_table.copy_(original_table)
            env.part_target_counts.copy_(original_counts)
            env.part_target_table[slot, :] = target
            env.part_target_counts[slot] = 1
            env.part_target_cursor.zero_()
            env.seed(case["seed"])
            result = wrapped.reset()
            obs = result["obs"] if isinstance(result, dict) else result
            player.get_batch_size(obs, 1)
            if player.is_rnn:
                player.init_rnn()
            assert int(env.target_index[slot]) == target
            origin = env.scene.env_origins[slot : slot + 1]
            camera.set_world_poses_from_view(
                origin + torch.tensor([[1.9, -2.2, 1.5]], device=env.device),
                origin + torch.tensor([[0.48, -0.10, 0.18]], device=env.device),
            )
            for _ in range(3):
                env.sim.render()
            camera.update(env.physics_dt, force_recompute=True)
            env.wrist_camera.update(env.physics_dt, force_recompute=True)
            initial_p, initial_r = env.pose_errors()
            with torch.no_grad():
                original = player.get_action(obs, is_deterministic=True)
                changed = obs.clone()
                changed[:, -8:] = torch.linspace(-10, 10, 8, device=env.device)
                altered = player.get_action(changed, is_deterministic=True)
            delta = float((original - altered).abs().max())
            assert delta < 1e-7, "Privileged labels affected inference actions"
            directory = args.output / case["name"]
            directory.mkdir(exist_ok=False)
            entry = dict(
                case,
                slot=slot,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                initial_position_mm=float(initial_p[slot].norm() * 1000),
                initial_rotation_deg=float(initial_r[slot].norm() * 180 / np.pi),
                privileged_label_action_max_delta=delta,
                appearance=env.appearance.samples[slot] if env.appearance else None,
                steps=[],
            )

            def frame(step, action=None):
                camera.update(env.step_dt, force_recompute=True)
                canvas = Image.new("RGB", (1280, 720), "#14202c")
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (14, 8),
                    f"Policy epoch {case['epoch']} | {case['part']} | 1x simulation time",
                    font=font,
                    fill="white",
                )
                draw.text((14, 34), case["target_id"], font=small, fill="#b9cddd")
                overview = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
                assert overview.std() > 2, "Blank overview image"
                canvas.paste(Image.fromarray(overview), (0, 72))
                wrist = env.wrist_camera.data.output["rgb"][slot, ..., :3].cpu().numpy()
                goal = (env.goal_rgbd[slot, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                for y, label, rgb in ((96, "Live wrist RGB", wrist), (342, "Goal reference RGB", goal)):
                    draw.text((968, y - 26), label, font=font, fill="white")
                    canvas.paste(Image.fromarray(rgb).resize((312, 176)), (968, y))
                p, r = env.pose_errors()
                pos = float(p[slot].norm() * 1000)
                rot = float(r[slot].norm() * 180 / np.pi)
                draw.text((968, 542), f"t = {step / 15:.2f} s", font=font, fill="white")
                draw.text((968, 570), f"Error: {pos:.2f} mm", font=font, fill="white")
                draw.text((968, 597), f"Rotation: {rot:.2f} deg", font=font, fill="white")
                draw.text((968, 636), "Goal: <4 mm and <3 deg", font=small, fill="#b9cddd")
                draw.text((968, 660), "Alignment task; fingers open", font=small, fill="#b9cddd")
                return canvas, pos, rot

            with torch.no_grad():
                for step in range(env.max_episode_length + 2):
                    assert int(env.target_index[slot]) == target
                    picture, pos, rot = frame(step)
                    picture.save(directory / f"{step:04d}.jpg", quality=92)
                    action = player.get_action(obs, is_deterministic=True)
                    assert torch.isfinite(action).all()
                    entry["steps"].append(
                        {"step": step, "position_mm": pos, "rotation_deg": rot, "action": action[slot].cpu().tolist()}
                    )
                    result, _, done, _ = wrapped.step(action)
                    obs = result["obs"] if isinstance(result, dict) else result
                    if done[slot]:
                        terminal = {k: v[slot].item() for k, v in env.last_transition.items()}
                        status = next(
                            k for k in ("success", "collision", "premature", "divergence", "timeout") if terminal[k]
                        )
                        entry.update(terminal=terminal, outcome=status, simulation_seconds=(step + 1) / 15)
                        card = picture.copy()
                        draw = ImageDraw.Draw(card)
                        draw.rectangle((0, 0, 1280, 105), fill="#14202c")
                        draw.text(
                            (14, 8),
                            f"Epoch {case['epoch']} | {status.upper()} | terminal error {terminal['position_error_m'] * 1000:.2f} mm / {terminal['rotation_error_rad'] * 180 / np.pi:.2f} deg",
                            font=font,
                            fill="white",
                        )
                        draw.text(
                            (14, 40),
                            "Frozen last pre-action frame. Terminal metrics above; automatic reset is excluded.",
                            font=font,
                            fill="#b9cddd",
                        )
                        draw.text(
                            (14, 69),
                            "This clip demonstrates visual alignment and stop timing, not a completed pick-and-lift.",
                            font=small,
                            fill="#b9cddd",
                        )
                        for hold in range(30):
                            card.save(directory / f"{step + 1 + hold:04d}.jpg", quality=92)
                        entry["frames"] = step + 31
                        report["episodes"].append(entry)
                        (args.output / "recording.json").write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            "[RECORDED]",
                            case["name"],
                            status,
                            entry["initial_position_mm"],
                            entry["initial_rotation_deg"],
                            terminal,
                            flush=True,
                        )
                        break
                else:
                    raise RuntimeError("Recorded episode did not terminate")
    finally:
        env.close()
    print("[VIDEOS] RECORDING COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        context = sim_utils.SimulationContext.instance()
        if context:
            context.clear_all_callbacks()
            context.clear_instance()
        app.close(wait_for_replicator=False)
