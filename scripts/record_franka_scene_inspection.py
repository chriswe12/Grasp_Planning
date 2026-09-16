#!/usr/bin/env python3
"""Record actual Fabrica target views and zero/scripted episodes for scene inspection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--catalog', type=Path, default=ROOT / 'isaac_rl/data/franka_fabrica_plumbers_v2/catalog_training_ready.npz')
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/franka_scene_inspection_v2')
parser.add_argument('--overview-width', type=int, default=960)
parser.add_argument('--overview-height', type=int, default=640)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.overview_width, args.overview_height) < 1:
    parser.error('Overview dimensions must be positive')
args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'isaac_rl/source/isaac_rl'))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaac_rl.tasks.direct.isaac_rl.franka_zed_env import FrankaZedEnv, FrankaZedEnvCfg


def main():
    cfg = FrankaZedEnvCfg()
    cfg.seed = 42
    cfg.sim.device = args.device
    # This inspection has one small scene; avoid large-batch PhysX allocations.
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**16
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**14
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**16
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**16
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16
    cfg.scene.num_envs = 1
    cfg.catalog_path = str(args.catalog)
    cfg.catalog_split = 'all'
    cfg.camera_profile_path = str(ROOT / 'configs/franka_zed_mini.json')
    cfg.fixed_target_index = 0
    cfg.fixed_waypoint_index = 0
    cfg.reset_ready_fraction = 0.
    cfg.rgb_gain_randomization = 0.
    cfg.scene.overview_camera = CameraCfg(
        prim_path='{ENV_REGEX_NS}/OverviewCamera', width=args.overview_width, height=args.overview_height,
        data_types=['rgb'], spawn=sim_utils.PinholeCameraCfg(focal_length=18., clipping_range=(.01, 10.)),
    )
    env = FrankaZedEnv(cfg)
    cfg = env.cfg  # The environment copies its input configuration.
    camera = env.scene['overview_camera']
    eye = [1.9, -2.2, 1.5] if env.lab_contract else [1.5, 1.3, 1.0]
    look = [.48, -.10, .18] if env.lab_contract else [.35, 0., .3]
    camera.set_world_poses_from_view(
        env.scene.env_origins + torch.tensor([eye], device=env.device),
        env.scene.env_origins + torch.tensor([look], device=env.device),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except OSError:
        font = ImageFont.load_default()

    def frame(title, subtitle):
        canvas = Image.new('RGB', (1280, 720), '#15202b')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), title, fill='white', font=font)
        draw.text((16, 35), subtitle, fill='#b8cedd', font=font)
        overview = camera.data.output['rgb'][0, ..., :3].cpu().numpy()
        assert overview.max() > overview.min(), 'Blank overview render'
        canvas.paste(Image.fromarray(overview).resize((960, 640)), (0, 72))
        live = env.wrist_camera.data.output['rgb'][0, ..., :3].cpu().numpy()
        goal = (env.goal_rgbd[0, ..., :3].clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
        for y, label, rgb in [(94, 'Live wrist camera', live), (346, 'Saved policy goal RGB', goal)]:
            draw.text((968, y-27), label, fill='white', font=font)
            canvas.paste(Image.fromarray(rgb).resize((320, 180)), (960, y))
        p, r = env.pose_errors()
        draw.text((968, 555), f'Position error: {p.norm().item()*1000:.1f} mm', fill='white', font=font)
        draw.text((968, 585), f'Rotation error: {r.norm().item()*180/np.pi:.1f} deg', fill='white', font=font)
        return canvas

    def reset(target, waypoint, seed=42):
        cfg.fixed_target_index = env.target_ids.index(target)
        cfg.fixed_waypoint_index = waypoint
        env.reset(seed=seed)
        for _ in range(5):
            env.sim.render()
        camera.update(env.physics_dt, force_recompute=True)
        env.wrist_camera.update(env.physics_dt, force_recompute=True)

    selected = ['orientation_000__g0544__p0', 'orientation_000__g1762__p1',
                'orientation_001__g0537__p0', 'orientation_001__g0533__p1',
                'orientation_002__g3192__p0', 'orientation_002__g2679__p1']
    report = {'catalog_sha256': hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
              'contract': env.contract(), 'physics_device': env.device,
              'overview_resolution': [args.overview_width, args.overview_height],
              'fps': 1/(cfg.sim.dt*cfg.decimation), 'goal_views': [], 'episodes': [],
              'notes': ['No trained policy used.', 'Object is kinematic and gripper stays open in alignment task.',
                        'RGB gain randomization disabled for inspection.', 'Camera calibration remains provisional.']}
    if env.appearance:
        report['notes'][2] = 'Per-episode physical appearance variation enabled; saved goals are canonical references.'
        report['variations'] = []
    with torch.inference_mode():
        if env.appearance:
            for i in range(12):
                reset(selected[0], env.catalog['joint_paths'].shape[1]-1, seed=1000+i)
                sample = env.appearance.samples[0]
                filename = f'variation_{i:02d}.png'
                frame(f'Appearance {i+1:02d} | {sample["part_color_name"]}',
                      f'Light {sample["light_intensity"]:.0f} | {sample["light_temperature_k"]:.0f} K | canonical goal on right').save(args.output / filename)
                report['variations'].append(dict(image=filename, sample=sample))
                print(f'[VARIATION] {i+1}: {sample["part_color_name"]}', flush=True)
        for target in selected:
            reset(target, env.catalog['joint_paths'].shape[1]-1)
            filename = f'goal_{target}.png'
            scene_label = 'Panda + pencil lab' if env.lab_contract else 'Panda + white table'
            frame('Fabrica target pose | ' + scene_label, target).save(args.output / filename)
            report['goal_views'].append({'target_id': target, 'image': filename})
            print(f'[PREVIEW] Goal {target}', flush=True)
        for mode in ('zero', 'scripted'):
            for number, target in enumerate((selected[0], selected[4]), 1):
                reset(target, 0, seed=100+number if env.appearance else 42)
                name = f'{mode}_{number:02d}'
                directory = args.output / name
                directory.mkdir(exist_ok=True)
                initial = float(env.pose_errors()[0].norm())
                for step in range(env.max_episode_length+2):
                    label = 'Zero actions (holds initial pose)' if mode == 'zero' else 'Scripted pose controller (privileged target)'
                    frame(label, f'{target} | t={step/report["fps"]:.2f} s').save(directory / f'{step:04d}.png')
                    action = torch.zeros((1, 7), device=env.device)
                    if mode == 'scripted':
                        p, r = env.pose_errors()
                        rot = env.camera_rotation().transpose(1, 2)
                        action[:, :3] = (rot @ (2*p)[..., None]).squeeze(-1) / cfg.linear_action_scale_m_s
                        action[:, 3:6] = (rot @ (2*r)[..., None]).squeeze(-1) / cfg.angular_action_scale_rad_s
                        for block in (slice(0, 3), slice(3, 6)):
                            action[:, block] /= action[:, block].abs().amax(-1, keepdim=True).clamp_min(1.)
                        action[:, 6] = env._labels(p.norm(dim=-1), r.norm(dim=-1)).ready.float()
                    _, reward, terminated, timeout, _ = env.step(action.clamp(-1, 1))
                    assert torch.isfinite(reward).all()
                    if (terminated | timeout).item():
                        # DirectRLEnv already reset; never record that reset as the terminal pose.
                        episode = {'name': name, 'mode': mode, 'target_id': target, 'frames': step+1,
                                   'initial_position_m': initial,
                                   'terminal': {k: v[0].item() for k, v in env.last_transition.items()}}
                        report['episodes'].append(episode)
                        if env.appearance:
                            # The terminal step auto-resets. The clip's starting sample is recorded by seed.
                            episode['reset_seed'] = 100+number
                        print(f'[PREVIEW] Episode {json.dumps(episode)}', flush=True)
                        break
                else:
                    raise RuntimeError(f'Episode {name} did not terminate')
    (args.output / 'recording.json').write_text(json.dumps(report, indent=2)+'\n')
    env.close()
    print(f'[PREVIEW] COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        context = sim_utils.SimulationContext.instance()
        if context:
            context.clear_all_callbacks()
            context.clear_instance()
        app.close(wait_for_replicator=False)
