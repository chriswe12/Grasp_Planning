# Franka training scene

`grasp_planning/rl/franka_training_scene.py` provides `FrankaTrainingSceneCfg`,
an Isaac Lab `InteractiveSceneCfg` with a Panda arm and Panda hand, wrist RGB-D,
a plain matte white collision-enabled tabletop, a dynamic blue test cube, and
hand/finger contact sensors. Tabletop Z and the robot mounting plane are zero.
The table is 1.2 x 0.9 m and 50 mm thick. It has no T-slots or surface markings.

The robot matches the source of the supplied camera pose:
`/media/pdz/Elements1/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/unplug/unplug_env_cfg.py`.
That source uses the Panda asset, rather than an FR3 articulation. Its camera
parent is `panda_hand` and its default offset convention is ROS optical
(+Z forward, +Y down). The active supplied pose is preserved:

```python
offset=TiledCameraCfg.OffsetCfg(
    pos=(-0.1115, 0.0481, 0.1034 - 0.0883),
    rot=(0.6435702, -0.2768679, 0.2724621, -0.6594891),  # WXYZ
    convention="ros",
)
```

The scene retains the source camera's 672 x 376 resolution, focal length 3.06,
horizontal/vertical apertures 4.8/3.6, and 0.01–1 m clipping range, and adds
`distance_to_image_plane` depth alongside RGB. These are the source unplug/ZED
camera settings, not the current KUKA D405 calibration. The Panda high-PD preset
disables arm gravity; the cube remains dynamic with gravity.

The current v2 appearance uses an explicit matte-blue material for both the
preview cube and imported Fabrica meshes, a 900-intensity angled key light,
250-intensity dome fill, and ambient occlusion. The table stays white. This
replaces the first Fabrica pilot's uncolored USD and dome-only lighting. Restart
an already-open preview to apply it. Goal catalogs must be regenerated for the
new scene profile; see `FRANKA_FABRICA_TRAINING.md`.

## Preview

Run from this repository with Isaac Lab 2.3.2:

```bash
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p \
  scripts/preview_franka_training_scene.py --num-envs 1
```

The viewer holds the robot at its initial joint targets. For a bounded headless
smoke test, append `--headless --steps 60`. Cameras are enabled by the script.
`--output-dir` defaults to `artifacts/franka_training_scene`; outputs are
`overview.png`, `wrist_rgb.png`, metric `wrist_depth_m.npy`, and `scene.json`
with asset, camera, joint, image-shape, and object-position metadata.

### Move joints with Physics Inspector

Close the previous preview and relaunch in Inspector mode:

```bash
/media/pdz/Elements1/IsaacLab-2.3.2/isaaclab.sh -p \
  scripts/preview_franka_training_scene.py --physics-inspector
```

This uses one environment, CPU physics (`device="cpu"`) and USD synchronization
(`use_fabric=False`). Rendering still uses the GPU. It seeds the Panda drive
targets once and stops writing joint targets from the Python loop, allowing
Physics Inspector to change them. The regular GPU preview keeps holding its
configured joint pose and is unsuitable for Inspector drive edits; those edits
produce `setDriveTarget ... eENABLE_DIRECT_GPU_API` errors in GPU simulation.
Switching mode requires a fresh process, not merely pausing the existing scene.

After capturing the initial preview, the interactive mode stops the main
timeline and hands control to the Inspector's isolated authoring simulation.
Open **Tools > Physics > Physics Inspector**, select
`/World/envs/env_0/Robot`, and edit a drive target (angular targets use degrees).
Leave the main timeline stopped while using Inspector. This follows the
[Physics Inspector authoring workflow](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/joint_inspector.html).
For a repeatable native
drive check without the UI, run `--headless --physics-inspector
--test-inspector-control --output-dir artifacts/franka_inspector_smoke`.
This runs 120 physics steps, changes joint 1's USD target to 0.2 rad, checks the
measured joint position, and writes `inspector_control.json`.

The installed container can also run the headless check:

```bash
docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -v "$PWD:/workspace/project" -w /workspace/project \
  --entrypoint /bin/bash isaac-lab-euler:2.3.2 \
  /workspace/isaaclab/isaaclab.sh -p scripts/preview_franka_training_scene.py \
  --headless --num-envs 2 --steps 60
```

## Learning integration boundary

The standalone preview remains a scene inspection tool. A separate GPU
Panda/ZED Mini visual-servo pilot now provides catalog generation, the shared
RGB-D actor/completion PPO entrypoint, and controller/reset checks. See
[FRANKA_ZED_TRAINING.md](FRANKA_ZED_TRAINING.md) for its commands and remaining
validation. It has its own camera/catalog contract; existing KUKA datasets and
checkpoints are not compatible. A user-authorized local PPO smoke has run;
full training and cluster jobs have not been started.

## Verification

Validated on the local RTX 4090 with the `isaac-lab-euler:2.3.2` image:
two cloned environments rendered RGB-D and held their configured joint poses;
the final one-environment 30-step preview also exited successfully. The cube
settled with its center at Z=0.02 m. The wrist image contains the blue target,
table and fingers; 68.4% of pixels have finite depth. Python syntax checks and
the local launcher's `--help` also passed. These checks establish scene readiness,
not learned grasp or lift performance.

The Inspector control smoke test also passed using CPU physics: a 0.2 rad USD
drive target produced a measured joint-1 angle of 0.19718 rad after 120 steps,
with no Direct GPU API errors. This checks the native USD drive path; it does
not automate clicks in the Inspector UI.
