# Agent Guidance

## Orientation

YAML-driven Fabrica grasp planning for FR3 and KUKA iiwa7. Start with
`README.md`; detailed operations and contracts live in `docs/`.

- `run_pipeline.sh` is the public entrypoint. Dual-arm is the default;
  select `--workflow single-object` explicitly. Modes are `sim`, `pitl`, `real`.
- Planning: `grasp_planning/pipeline/`; execution: `scripts/run_fabrica_grasp_in_{mujoco,isaac}.py`
  and `grasp_planning/ros2/real_grasp_executor.py`; hardware nodes: `ros2_ws/src/robot_integration_ros/`.
- Standalone RL is the pinned `isaac_rl/` submodule, with shared helpers in
  `grasp_planning/rl/` and cluster deployment in `euler/`. It is not bundle execution.
- See `docs/execution-paths.md`, `docs/dual-arm-symmetry.md`, and
  `docs/training.md` instead of restoring retired plans or experiment diaries.

## Scope And Safety

- Audits/proposals do not authorize fixes. Deployment/smoke preparation does
  not authorize training, cluster submission, or hardware motion.
- Verify current paths/configs against this checkout; wiki, memory, benchmark
  results, and cluster observations can be dated. Report verification limits.
- Preserve `configs/grasp_pipeline_real.yaml` defaults: execution disabled,
  confirmation required, stop at pregrasp, gripper disabled. Software, fake
  hardware, networking, and FCI checks do not establish physical readiness.
- Generation benchmark success is planning-only, not execution or lift success.
- Keep standalone benchmarks separate from mode behavior; do not restore the
  retired simulator stack. Do not commit caches, logs, or `__pycache__`.

## Companion Wiki

For broad architecture, pipeline, frame, backend, ROS2, config, asset, or safety
changes, read `../mt_wiki/index.md` first. Code is the source of truth.
Do not edit durable wiki pages or its `log.md` from this repository.
When follow-up is needed, maintain one session note under
`../mt_wiki/agent-changelogs/`, following its `README.md` and `TEMPLATE.md`.
Before committing, check the note's paths, behavior, verification, risks, and
commit/status accuracy. Trivial formatting, generated outputs, and scratch
experiments do not need a note. Agents operating from the wiki may read this
code but must not modify, commit, or push it.

## Contracts That Must Not Regress

- Single-object backends consume the same saved stage-2 bundle; do not add
  a second grasp serialization path. Rebuild MuJoCo meshes and generated
  collision-enabled Isaac USD in the saved bundle-local frame. A supplied USD
  is valid only if authored in that frame.
- Saved-bundle Isaac execution uses MoveIt joint waypoints, not local direct
  controllers. MuJoCo's optional MoveIt controller plans only; MuJoCo owns physics.
- Bump `GRASP_SCORING_ALGORITHM_VERSION` when `score_grasps()` changes so
  stage-1 caches cannot reuse stale pad-footprint `contact_support` scores.
- `--skip-stage1-collision-checks` skips only assembly filtering, not
  object/gripper/floor checks. `roll_angle_step_deg` covers a full 360-degree sweep.
- Regrasp candidates are geometry-filtered then MoveIt-ranked at execution.
  Score reachability per actual staging XY offset, not just the base pose.
  Keep candidate-plan artifacts separate from execution-attempt diagnostics.
- Dual runtime queues preserve producer safety/corridor tiers and explicit
  `candidate_rank`; execute the exact collision-aware IK joints accepted in preflight.
- Propagate `--inserter-arm` into real task construction and consume declared
  roles; `auto` swaps roles across assembly Y, so hard-coded roles break symmetric cases.
- Ground rotated dual pickup meshes on the configured floor before filtering.
  Start real-mode live diagnostics before filtering so empty queues remain explainable.
- Dual Isaac streams the MoveIt polyline with critically damped drives;
  release the pickup fixture after bilateral intended-object contact, not during transport.

## Environment And Verification

- Interpreter precedence: `PIPELINE_PYTHON`, `python3`, `python`.
- `sim` uses YAML `execution_world_pose`; `pitl`/`real` use shared ROS2 intake.
  `--backend {config,mujoco,isaac,both,none}` overrides sim/pitl execution.
- ROS2 hardware nodes need an external FR3/MoveIt underlay. Pipeline discovery
  defaults to domain 0 and clears localhost/static discovery unless
  `GRASP_KEEP_ROS_DISCOVERY_ENV=1`.
- Menagerie `franka_fr3` is arm-only; use `scripts/build_mujoco_fr3_hand_models.py`
  for FR3+Panda hand XML under `.cache/generated_mujoco_models/`.
- Expose tuning in `configs/mujoco_simulation.yaml`, sim/pitl `isaac_execution`,
  or real `real_execution`, matching the affected backend.
- Initialize pinned submodules before testing. CPU regressions:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q`.
  Broaden checks for changed contracts; CPU results do not replace Isaac/ROS/hardware validation.
- Documentation moves must update local links and script error hints. Keep
  newcomer guidance in README and deeper reference in `docs/`.
