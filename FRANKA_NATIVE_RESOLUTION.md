# Native-resolution Franka visual policy

The ZED profile binds sensor calibration, render dimensions and policy dimensions to each catalog/checkpoint. Supported observation sizes are 128×72, 256×144 and 384×216. Old 128×72 profiles and checkpoint shapes remain unchanged. A higher-resolution profile requires regenerated references and new policy training; the loader still rejects catalog/profile/hash mismatches.

The ResNet layer-3 feature grid is preserved through fusion. The policy's first dense layer is sized to `128 * ceil(height/16) * ceil(width/16)`; no spatial pooling adapter transfers old policy weights into the new head. Pose reset history, observation spaces, and stereo-noise focal length follow the profile dimensions.

`isaac_rl/scripts/build_franka_mixed_goals.py --native-resolution --camera-profile configs/franka_zed_mini_sn13829658_384.json --feature-lighting` generates native Isaac and MuJoCo references, preserving validated grasp/reset geometry and splits. The renderer mixture remains 50% Isaac (including 20% blue canonical) and 50% MuJoCo; goal colors and canonical placement stay independent of live appearance/placement. Do not resize the old goal bank as a substitute for rebuilding it.

`feature_lighting` is an opt-in catalog field. `grasp_planning/rl/franka_feature_lighting.py` defines neutral key/fill/top softboxes, a dome, and two fills attached to the wrist camera. Per-episode intensities vary by 0.8–1.2 with independent seeded samples. Table/floor/material appearance randomization remains active; original competing light sources are disabled for this profile. MuJoCo uses corresponding soft fill lighting and independently seeded intensity variation, retaining its simplified lab materials.

The native-resolution inference benchmark is under `artifacts/franka_inference_capacity_20260922`; those eight-epoch probe weights only establish compute capacity. The new training and source/catalog provenance are recorded separately under `artifacts/franka_native384_training_20260922`.

Training is isolated from older Euler jobs using `euler/stage_franka.sh`; never synchronize a new catalog or changed sources into an older run's snapshot. The existing 163,840,000-transition learning schedule scales its epoch count to the actual global environment count.

Hardware deployment remains a separate gate: the current `real_franka.core.Actor` entrypoint still assumes the original observation shape. Loading a native-resolution checkpoint there requires adapting and validating the complete camera/observation contract first. Network-level inference timing alone does not establish grasp accuracy or hardware readiness.
