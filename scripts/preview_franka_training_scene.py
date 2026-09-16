#!/usr/bin/env python3
"""Preview/smoke-test the Panda training scene without loading a policy or catalog."""

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=0, help="0 keeps the viewer running; headless defaults to 60.")
parser.add_argument("--output-dir", type=Path, default=Path("artifacts/franka_training_scene"))
parser.add_argument(
    "--physics-inspector", action="store_true",
    help="Use CPU physics and USD updates; let Physics Inspector own joint drive targets.",
)
parser.add_argument(
    "--test-inspector-control", action="store_true",
    help="With --physics-inspector, verify that a USD joint-drive edit moves joint 1 (120 steps).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 0:
    parser.error("num-envs must be positive and steps nonnegative")
if args.test_inspector_control and not args.physics_inspector:
    parser.error("--test-inspector-control requires --physics-inspector")
if args.physics_inspector:
    if args.num_envs != 1:
        parser.error("--physics-inspector requires --num-envs 1")
    args.device = "cpu"
if args.test_inspector_control:
    args.steps = 120
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp_planning.rl.franka_training_scene import (  # noqa: E402
    FRANKA_SCENE_PROFILE, FrankaTrainingSceneCfg, make_franka_render_cfg,
)


def handoff_to_inspector(sim):
    """Stop global physics without Isaac Lab's standalone STOP render loop."""
    sim.clear_all_callbacks()
    sim.clear_instance()
    sim.stop()


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1.0 / 120.0, device=args.device, render=make_franka_render_cfg(),
        use_fabric=not args.physics_inspector,
    ))
    sim.set_camera_view((1.5, 1.3, 1.0), (0.35, 0.0, 0.15))
    cfg = FrankaTrainingSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(cfg)
    inspector_drive = None
    if args.physics_inspector:
        # Seed the authored drives once so the UI starts at the same targets
        # as the configured pose. Angular USD drive targets are in degrees.
        for prim in sim.stage.Traverse():
            name = prim.GetName()
            if name.startswith("panda_joint") and prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                drive.GetTargetPositionAttr().Set(float(np.rad2deg(cfg.robot.init_state.joint_pos[name])))
                if name == "panda_joint1":
                    inspector_drive = drive
            elif name.startswith("panda_finger_joint") and prim.IsA(UsdPhysics.PrismaticJoint):
                UsdPhysics.DriveAPI.Get(prim, "linear").GetTargetPositionAttr().Set(0.04)
        if inspector_drive is None:
            raise RuntimeError("Could not find the Panda joint-1 USD drive")
    overview = Camera(CameraCfg(
        prim_path="/World/OverviewCamera", width=960, height=640,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, clipping_range=(0.01, 10.0),
        ),
    ))
    sim.reset()
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()
    origin = scene.env_origins[0:1]
    overview.set_world_poses_from_view(
        origin + torch.tensor([[1.5, 1.3, 1.0]], device=sim.device),
        origin + torch.tensor([[0.35, 0.0, 0.3]], device=sim.device),
    )
    sim.set_camera_view(
        (origin[0] + torch.tensor([1.5, 1.3, 1.0], device=sim.device)).cpu().tolist(),
        (origin[0] + torch.tensor([0.35, 0.0, 0.3], device=sim.device)).cpu().tolist(),
    )
    target = robot.data.default_joint_pos.clone()
    limit = args.steps or (60 if args.headless else 0)
    step = 0
    saved = False
    inspector_handoff = False
    while app.is_running() and (not limit or step < limit):
        if inspector_handoff:
            # Inspector runs its own isolated physics authoring simulation.
            # Keep the UI responsive without stepping the Isaac Lab scene.
            app.update()
            continue
        with torch.inference_mode():
            if not args.physics_inspector:
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
            elif args.test_inspector_control and step == 20:
                inspector_drive.GetTargetPositionAttr().Set(float(np.rad2deg(0.2)))
            sim.step()
            scene.update(sim.get_physics_dt())
            overview.update(sim.get_physics_dt())
        step += 1
        if not saved and step >= min(30, limit or 30):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            rgb = scene["wrist_camera"].data.output["rgb"].cpu().numpy()
            depth = scene["wrist_camera"].data.output["distance_to_image_plane"].cpu().numpy()
            if rgb.shape[:3] != (args.num_envs, cfg.wrist_camera.height, cfg.wrist_camera.width):
                raise RuntimeError(f"Unexpected RGB shape: {rgb.shape}")
            if not np.isfinite(depth).any() or int(rgb.max()) == int(rgb.min()):
                raise RuntimeError("Wrist camera did not produce usable RGB-D")
            Image.fromarray(rgb[0, ..., :3]).save(args.output_dir / "wrist_rgb.png")
            np.save(args.output_dir / "wrist_depth_m.npy", depth[0])
            Image.fromarray(overview.data.output["rgb"][0, ..., :3].cpu().numpy()).save(
                args.output_dir / "overview.png"
            )
            metadata = {
                "scene_profile": FRANKA_SCENE_PROFILE,
                "physics_device": sim.device,
                "physics_inspector": args.physics_inspector,
                "robot_usd": cfg.robot.spawn.usd_path,
                "camera_parent": "panda_hand",
                "camera_convention": cfg.wrist_camera.offset.convention,
                "camera_position_m": cfg.wrist_camera.offset.pos,
                "camera_quaternion_wxyz": cfg.wrist_camera.offset.rot,
                "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
                "finite_depth_fraction": float(np.isfinite(depth).mean()),
                "joint_names": robot.joint_names,
                "joint_positions": robot.data.joint_pos.cpu().tolist(),
                "part_position_w": scene["part"].data.root_pos_w.cpu().tolist(),
                "tabletop_z_m": 0.0,
            }
            (args.output_dir / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"[INFO] Franka scene RGB-D and overview saved to {args.output_dir}", flush=True)
            saved = True
            if args.physics_inspector and not args.headless and not args.steps:
                handoff_to_inspector(sim)
                inspector_handoff = True
                print(
                    "[INFO] Timeline stopped for Physics Inspector. Open Tools > Physics > "
                    "Physics Inspector and select /World/envs/env_0/Robot. "
                    "Use the Inspector drive sliders without starting the main timeline.",
                    flush=True,
                )
    if not saved:
        raise RuntimeError("Simulation closed before a camera frame was saved")
    if args.test_inspector_control:
        joint_id = robot.joint_names.index("panda_joint1")
        actual = float(robot.data.joint_pos[0, joint_id])
        result = {"target_rad": 0.2, "actual_rad": actual, "passed": abs(actual - 0.2) < 0.03}
        (args.output_dir / "inspector_control.json").write_text(json.dumps(result, indent=2) + "\n")
        if not result["passed"]:
            raise RuntimeError(f"USD drive control failed: {result}")
        print(f"[INFO] Inspector USD-drive control passed: {result}", flush=True)
        handoff_to_inspector(sim)
        app.update()
        if not sim.is_stopped():
            raise RuntimeError("Main timeline did not stop for Inspector handoff")
        print("[INFO] Inspector timeline handoff passed", flush=True)
    print(f"[INFO] Completed {step} simulation steps", flush=True)
    # Keep the scene available for orderly callback teardown before app.close.
    return sim, scene, overview


if __name__ == "__main__":
    try:
        scene_resources = main()
    finally:
        # Unsubscribe Isaac Lab's standalone STOP callback before Replicator
        # stops the timeline. Otherwise that callback renders indefinitely
        # waiting for playback to resume during application shutdown.
        sim_context = sim_utils.SimulationContext.instance()
        if sim_context is not None:
            sim_context.clear_all_callbacks()
            sim_context.clear_instance()
        # Images are written synchronously above; the live camera annotators
        # have no asynchronous writer jobs to drain at shutdown.
        print("[INFO] Closing scene preview", flush=True)
        app.close(wait_for_replicator=False)
