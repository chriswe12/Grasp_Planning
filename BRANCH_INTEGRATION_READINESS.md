# Branch integration readiness

Date: 2026-08-25

## Summary

The `new-gripper-urdf`, `hold_grasping`, and `grasping-rl` lines are architecturally compatible, but they should not be combined with a blind merge in their current state. The main risk is not an incompatible design: it is the amount of uncommitted work and the presence of two independently evolving copies of the D405 policy-deployment stack.

Current readiness is approximately **4/10**. Once the worktrees are committed in separable changes and the duplicate D405 deployment implementation is reconciled, the combined repository should be approximately **8/10** ready.

## Current branch state

| Worktree | Committed relationship | Dirty paths at audit time |
|---|---|---:|
| `new-gripper-urdf` | Same committed tip as `hold_grasping`: `b3ae830` | 102 |
| `hold_grasping` | Six commits ahead of `main` | 50 |
| `grasping-rl` | Three different commits ahead of `main` | 35 |

The new-gripper work is already based directly on hold-grasping, so those two lines are conceptually the easiest to combine. Much of the new-gripper implementation is still uncommitted, however, and would not be included by merging only the branch ref.

## Merge interference

A dry-run merge of the committed hold and RL tips reports conflicts in these core paths:

- `grasp_planning/envs/__init__.py`
- `grasp_planning/envs/fr3_part_env.py`
- `grasp_planning/planning/fr3_motion_context.py`
- `grasp_planning/planning/trajectory_executor.py`
- `grasp_planning/start_poses.py`
- `scripts/run_grasp_pipeline.py`
- `tests/test_trajectory_executor.py`

The uncommitted hold and RL work overlaps in 26 paths. Nineteen of those paths differ, including pipeline orchestration, D405 runtime/configuration, real execution, MoveIt launch files, packaging, and tests. Hold contains an uncommitted deployment-side copy of much of the D405 policy work, while the RL worktree contains the complete `isaac_rl` training environment. These implementations must be reconciled feature by feature.

There is no fundamental conflict between dual-arm holding, the PDZ gripper, and visual-servo RL. The interference is concentrated at shared robot-model, TCP, execution, and orchestration boundaries.

## PDZ gripper implications for RL

The RL policy concept is reusable. The deployed policy consumes live and goal RGB-D and outputs six camera-frame TCP velocity components plus a completion decision; it does not directly command finger joints.

The RL training environment and its generated assets are nevertheless tied to the old KUKA Y-gripper. Current assumptions include:

- a hard-coded Y-gripper robot USD;
- Y-gripper finger and hand link names for contact sensing;
- an `0.084 m` source-open aperture and Y-gripper-specific width conversion;
- `gripper_tcp` kinematics and transforms;
- a D405 pose authored relative to `link7`;
- a Y-gripper collision-validation profile;
- reset, rotation, path, and goal-image catalogs generated with the old geometry.

The new-gripper worktree already supplies the important PDZ embodiment pieces: PDZ joint detection, aperture conversion, collision geometry, TCP handling, camera frames, MoveIt assets, and KUKA+PDZ USD generation.

## Required PDZ RL migration

Add one gripper/embodiment profile shared by planning and RL. It should define:

- robot USD and URDF;
- TCP/body transform;
- finger joint and link names;
- open, closed, and approach apertures;
- contact-sensor prim patterns;
- collision-validation profile;
- camera parent, optical transform, and visual profile.

Then regenerate:

1. The PDZ multigrasp manifest.
2. MoveIt reset and approach trajectories.
3. Rotation and position reset assets using the PDZ collision model.
4. Isaac goal RGB-D images through the mounted PDZ D405 frame.
5. Checkpoint compatibility metadata.

The existing network shape remains compatible, so a zero-shot PDZ evaluation is useful. The old checkpoint should not be considered deployment-compatible without evaluation: the new gripper changes camera extrinsics, hand appearance, occlusion, TCP geometry, and collision behavior. Fine-tuning or retraining is likely to be necessary.

## Recommended integration order

1. Commit the current hold changes in separable commits.
2. Commit the PDZ assets, geometry, planning integration, execution integration, and tests as a series on top of hold.
3. Treat `grasping-rl` as canonical for `isaac_rl` training code.
4. Merge RL into the hold-plus-PDZ integration branch.
5. Preserve PDZ geometry/TCP/joint handling from `new-gripper-urdf`, RL task/catalog/curriculum behavior from `grasping-rl`, and dual-arm execution behavior from `hold_grasping` while resolving conflicts.
6. Reconcile the duplicated D405 deployment paths feature by feature.
7. Add the shared PDZ RL embodiment profile and regenerate its assets.
8. Run an Isaac zero-shot evaluation, fine-tune if needed, then validate ROS dry-run behavior before enabling hardware motion.

## Verification performed during the audit

- `git diff --check` passed in all three worktrees.
- Selected new-gripper tests: 137 passed.
- Selected hold deployment tests: 25 passed.
- The selected RL test run could not complete collection in the current shell because the optional `rl_games.algos_torch.network_builder` dependency was unavailable. This is an environment/setup gap, not evidence that the RL logic failed.
