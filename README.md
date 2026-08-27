# Fabrica Grasp Planning And Dual-Arm Assembly

This repository plans collision-checked parallel-jaw grasps for Fabrica parts
and executes them with Franka Research 3 or KUKA iiwa7 robots. One public
command exposes two related workflows:

1. The **single-object pipeline** generates grasps, filters them at a known
   world pose, and can execute a selected stage-2 bundle in MuJoCo, Isaac, or
   on a real robot.
2. The **dual-arm assembly pipeline** selects one grasp that holds the partial
   assembly and another that picks and transports the incoming part to a
   symmetry-aware pre-insertion pose.

Both workflows are YAML-driven and write JSON/HTML artifacts that explain what
was selected and why alternatives were rejected. Generated artifacts under
`artifacts/` are local outputs and are not the source code.

## Start Here

Use one environment and one entrypoint:

```bash
source ./setup_robot_env.sh
./run_pipeline.sh --help
```

| Goal | Entrypoint | Main config |
| --- | --- | --- |
| Dual real grasp, lift, and move to pre-insertion | `./run_pipeline.sh --mode real --robots both --execute` | `configs/dual_grasp_planning.yaml` |
| Dual real grasp and lift only | `./run_pipeline.sh --mode real --robots both --grasp-only --execute` | `configs/dual_grasp_planning.yaml` |
| One active real robot | `./run_pipeline.sh --mode real --robots left --grasp-only --execute` | `configs/dual_grasp_planning.yaml` |
| Use one policy for every active grasp approach | `./run_pipeline.sh --mode real --policy clutter-v5 --execute` | policy registry/metadata |
| Bring up one or both physical arms | `./run_pipeline.sh --mode real --robots both --bringup-only --rviz` | ROS2 MoveIt config |
| Generate and execute one legacy single-object grasp | `./run_pipeline.sh --workflow single-object --mode sim` | `configs/grasp_pipeline_sim.yaml` |
| Use a ROS2 perception pose in simulation | `./run_pipeline.sh --workflow single-object --mode pitl` | `configs/grasp_pipeline_pitl.yaml` |
| Build dual holder/inserter planning artifacts | `scripts/build_dual_grasp_pairs.py` | `configs/dual_grasp_planning.yaml` |
| Run the dual-arm vertical slice in Isaac | `./run_pipeline.sh --mode sim --robots both` | `configs/dual_grasp_planning.yaml` |
| Benchmark every dual assembly step | `./run_pipeline.sh --benchmark dual-assembly` | `configs/dual_assembly_benchmark.yaml` |
| Evaluate grasp generation without execution | `./run_pipeline.sh --benchmark grasp-generation` | `configs/grasp_generation_benchmark.yaml` |
| Execute saved benchmark grasps | `./run_pipeline.sh --benchmark grasp-execution` | `configs/grasp_execution_benchmark.yaml` |

See [EXECUTION_PATHS.md](EXECUTION_PATHS.md) for the complete single-object,
policy, dual-arm, ROS action, and benchmark execution map.

The single-object and dual-arm pipelines share grasp generation and collision
geometry, but their numbered stages are different. When this README says
"dual Stage 3," it means holder/inserter pair construction, not the
single-object stage-2 bundle.

## What Is Implemented

The single-object path supports:

- object-local antipodal grasp generation and scoring;
- assembly and world-pose collision filtering;
- finite-symmetry pickup-grasp expansion;
- stage-2 bundles as the shared MuJoCo, Isaac, and real-execution contract;
- optional MuJoCo regrasp fallback; and
- planning-only and execution benchmarks.

The dual-arm path supports:

- compiling an authored assembly order into explicit partial-assembly states;
- reusable holder grasps checked against each changing assembly state;
- incoming-part grasps and collision-checked holder/inserter pairs;
- two distinct uses of symmetry:
  - **pickup symmetry** creates equivalent object-local grasp choices before
    the incoming part is grasped;
  - **transition symmetry** creates equivalent final/pre-insertion corridors
    after the grasp is fixed;
  - a **pickup symmetry bridge** expands every Stage-3 destination grasp across
    the complete effectively exact incoming-part symmetry orbit, checks those
    equivalent pickup grasps alongside the direct pickup, and compensates the
    final/pre-insertion object frame. This keeps the already validated TCP
    corridor unchanged while the rigidly held part rotates during transport;
- cheap distance/rotation pool ordering and corridor-diverse joint-space
  pre-ranking from the planned pickup lift. The simulator/benchmark candidate
  planner then checks exact MoveIt IK through a complete 14-joint hypothetical
  state: holder targets are solved first and frozen while inserter
  pickup/transition targets are checked. Incoming pickup approach IK follows
  short Cartesian continuation waypoints from pregrasp to grasp instead of one
  large numerical jump. Its request-local cache key includes the detected
  world pose and complete input state, and full trajectory planning reuses the
  validated holder and inserter joint targets instead of recomputing a
  different IK branch;
- execution of the validated vertical slice in Isaac; and
- guarded real-robot planning/execution through pre-insertion.

The dual-arm path does **not** yet execute constrained insertion, release the
incoming part, retreat, or coordinate an arbitrary complete assembly. Real
execution stops at the configured phase, no later than pre-insertion. Read
`DUAL_ROBOT_HOLD_GRASPING_PLAN.md` for implementation status and
`DUAL_ROBOT_TRANSITION_SYMMETRY_PLAN.md` for the symmetry/frame contract.

## Architecture And Artifact Flow

The single-object path is:

```text
Fabrica OBJ + assembly metadata + YAML
              |
              v
stage 1: object-local grasp generation and assembly filtering
              |
              v
stage 2: execution-pose floor filtering and scoring
              |
              v
saved stage-2 bundle
       |             |              |
       v             v              v
    MuJoCo         Isaac         real MoveIt
```

MuJoCo, Isaac, and the real executor consume the same saved stage-2 bundle;
do not create backend-specific grasp serialization.

The dual-arm path is:

```text
Fabrica assembly order, pre-insertion poses, meshes, and symmetries
              |
              v
dual Stage 0: explicit partial-assembly sequence
              |
              v
dual Stage 1: reusable holder-grasp library
              |
              v
dual Stage 2: holder feasibility in every assembly state
              |
              v
dual Stage 3: incoming grasps + collision-checked grasp pairs
              |       + symmetry-equivalent transition corridors
              v
runtime layout ranking -> MoveIt IK/path fallback -> Isaac or guarded real run
```

For `plumbers_block`, the selected order is `2 -> 0 -> 3 -> 1 -> 4`.
Part `2` is the initial base. Holder-active insertion steps therefore use
incoming parts `0`, `3`, `1`, and `4`.

## Installation

Install the Python package and test dependencies:

```bash
python3 -m pip install -e ".[test]"
```

Simulation and real-robot planning additionally require ROS2 and a compatible
MoveIt workspace. The current KUKA launchers default to ROS Humble and
`/home/pdz/lbr-stack`. Isaac commands default to
`/media/pdz/Elements1/IsaacLab/isaaclab.sh`; override `--lbr-ws` or
`--isaac-python` when using another installation.

Fetch the pinned ROS2 message dependency and build the repository overlay:

```bash
bash scripts/download_ros2_dependencies.sh

source /opt/ros/humble/setup.bash
source /home/pdz/lbr-stack/install/setup.bash
cd ros2_ws
colcon build --packages-select fp_debug_msgs robot_integration_ros --symlink-install
cd ..
```

Source the repository environment helper in every ROS terminal:

```bash
source ./setup_robot_env.sh
```

The older setup helpers are compatibility shims to this file.

## Single-Object Quick Start

The user-facing single-object entrypoint has three modes:

- `sim`: take the execution-world pose from YAML, then optionally execute in
  MuJoCo and/or Isaac;
- `pitl`: wait for a ROS2 perception pose, then optionally execute in MuJoCo
  and/or Isaac; and
- `real`: wait for the same ROS2 pose, write the same planning artifacts, and
  optionally execute on hardware.

```bash
./run_pipeline.sh --workflow single-object --mode sim
./run_pipeline.sh --workflow single-object --mode pitl
./run_pipeline.sh --workflow single-object --mode real
./run_pipeline.sh --workflow single-object --mode sim --headless
./run_pipeline.sh --workflow single-object --mode sim --backend isaac --headless
```

Default configs are `configs/grasp_pipeline_sim.yaml`,
`configs/grasp_pipeline_pitl.yaml`, and `configs/grasp_pipeline_real.yaml`.
Use `--backend {config,mujoco,isaac,both,none}` to override simulation backend
selection for one run.

`sim` and `pitl` both run stages 1 and 2 before executing the resulting
stage-2 bundle. `real` writes the same artifacts and executes only when
`real_execution.enabled: true`. The checked-in real config remains safe by
default: execution is disabled, confirmation is required, motion stops at
pregrasp, and gripper actuation is disabled.

The default sim config reproduces corrected KUKA execution-benchmark run 3:
`plumbers_block/0` at stable `orientation_002`. Start
`./start_lbr_moveit.sh` in another terminal, then run
`./run_pipeline.sh --workflow single-object --mode sim`. Pass `--headless` when the Isaac GUI is not
wanted.

For `pitl` and `real`, the planning local frame is the OBJ frame translated by
the arithmetic mean of all OBJ vertices. A matching
`fp_debug_msgs/msg/DebugPoseItem.pose_base` is treated as the world pose of
that centroid-centered frame.

## Dual-Arm Quick Start

### 1. Build the offline planning artifacts

Run the four dual stages in order after changing meshes, assembly metadata,
grasp settings, pair settings, or symmetry settings:

```bash
python3 scripts/build_assembly_sequence.py --assembly plumbers_block
python3 scripts/build_holder_grasp_library.py \
  --config configs/dual_grasp_planning.yaml
python3 scripts/build_holder_state_feasibility.py \
  --config configs/dual_grasp_planning.yaml
python3 scripts/build_dual_grasp_pairs.py \
  --config configs/dual_grasp_planning.yaml
```

`build_dual_grasp_pairs.py` can build/load the earlier stages, but listing all
four commands makes the dependency order explicit. Outputs are written to
`artifacts/dual_grasp_planning/plumbers_block/`. The most useful entrypoints
are:

- `assembly_sequence.html` for the authored sequence and insertion poses;
- `holder_base_candidates.html` for reusable base grasps;
- `holder_validity_matrix.html` for per-state holder feasibility;
- `dual_grasp_pair_summary.html` for per-step holder/inserter pairs; and
- `dual_robot_pair_score_debug.html` after running
  `python3 scripts/build_dual_robot_pair_score_debug.py`.

Stage-3 JSON files contain both `transition_symmetry.candidates` and a bounded
`retained_execution_candidates` list. Each retained execution ID names a
collision-validated `pair_id + transition_id`; retention round-robins across
distinct insertion-corridor directions so one canonical approach cannot crowd
out an equivalent opposite-side approach. The Stage-3 builder republishes the
matching `holder_base_candidates.json`, and each pair artifact declares its
exact holder/inserter candidate source. Runtime lookup follows that declaration;
sequential IDs such as `h0460` must never be resolved through a different or
stale generated library. Regenerate these files after
enabling or changing transition symmetry; the runtime runner does not silently
rebuild stale Stage-3 artifacts. Schema-2 artifacts remain readable, but the
runtime then falls back to the transition-validated retained pair subset.
Pickup symmetry bridges are derived from the symmetry provenance already in
the declared inserter bundle and transition artifact, so current Stage-3
artifacts do not need a separate raw-grasp serialization. Runtime retains the
unchanged direct queue first, then adds floor-feasible aliases from every exact
nonidentity symmetry for every Stage-3 destination grasp. Bridges retain the
Stage-3-proven contact-patch offset, are geometrically deduplicated, and receive
unique execution IDs before exact MoveIt screening. Runtime admits only asset
symmetries whose transformed-vertex error is at most `1e-6 m`; approximate
face-normal symmetries remain disabled until their carried-object sweep is
explicitly revalidated.

Stage-3 insertion filtering is adaptive. It evaluates candidates in score and
diversity order, but score is applied within coverage buckets rather than
globally: signed assembly-frame approach axes are round-robined first and
pickup-symmetry transforms are round-robined within each axis. The same
ordering is used for unary filtering, the pairing shortlist, and retained pair
selection, so one high-scoring physical side cannot consume every bounded
stage. Empty or exhausted buckets donate their capacity to the remaining
feasible directions. The default inserter library cap is 512. Stage 3 prefers
robust corridors whose individual gripper-component AABBs prove separation
from the assembled prefix, and invokes exact FCL only when those robust
candidates cannot fill the configured inserter shortlist.
The endpoint-AABB test is exact for the linear insertion/retreat sweep because
the swept AABB is the union of its endpoint AABBs. Independent assembly steps
run in separate worker processes; `pair_planning.stage3_worker_count: 0`
selects up to one worker per step from the available CPU affinity. Set
`adaptive_inserter_shortlist: false` and
`prefer_aabb_clear_inserter_candidates: false` only when an exhaustive
diagnostic artifact is more important than build time. The coverage policy is
controlled by `balance_inserter_approach_directions` and
`balance_inserter_symmetry_transforms`; disabling both restores global
score-first ordering.

For a faster collision-safe test build, keep the artifacts separate from the
full profile:

```bash
python3 scripts/build_dual_grasp_pairs.py \
  --config configs/dual_grasp_planning.yaml \
  --assembly plumbers_block \
  --output-dir artifacts/dual_grasp_planning_fast/plumbers_block \
  --max-inserter-candidates-per-step 320 \
  --max-pair-checks 8000 \
  --skip-exact-pair-clearance-ranking
```

This still performs exact FCL collision rejection. It skips only exact
separation-distance queries for overlapping, non-colliding pair AABBs, so such
pairs receive no exact-clearance ranking bonus. Run it with
`--artifact-root artifacts/dual_grasp_planning_fast`. The full profile uses the
YAML defaults (512 inserters, 16,000 pair checks, exact clearance ranking) and
can write to `artifacts/dual_grasp_planning_overnight/plumbers_block` without
overwriting the fast artifacts.

For the KUKA Y-gripper, every offline assembly, holder-state, pair, and pickup-
floor decision must pass at two finger states: the selected contact width and
the approach width, which is 5 mm farther open per finger (10 mm total jaw
clearance). `planning.detailed_finger_contact_gap_m: 0.005` is that per-finger
approach clearance. Changing it invalidates the KUKA Stage-1 collision cache;
rebuild the artifacts before sim or real execution.
Stage-1 target-object self-collision is the deliberate exception: it checks the
partially open approach geometry, because a closed whole-gripper query would
classify the intended finger-pad/object contacts as collisions. Closed and
approach states are both required for external obstacles such as the floor,
assembled parts, and the other gripper.

### 2. Run the dual-arm Isaac vertical slice

For the first holder-active step, keep mock MoveIt in one terminal:

```bash
./run_pipeline.sh --mode sim --robots both --bringup-only --rviz
```

Then run the task from a second terminal:

```bash
./run_pipeline.sh \
  --mode sim \
  --robots both \
  --assembly plumbers_block \
  --incoming-part-id 0
```

The task reuses the shared dual-arm mock MoveIt stack and pre-plans the
first eight inserter transitions in joint space. Stage 3 retains up to 256
complete pair/transition candidates. After the actual-pose floor check, the
runtime queue is split into a strict non-crossing phase followed by a crossed
fallback phase; retained candidates lead only within their phase. Other non-identity corridors are
eligible only when the artifact contains an explicit accepted pair-conditioned
transition validation; collision-checked canonical identity pairs extend the
finite exact-IK screening pool beyond the retained Stage-3 prefix. Simulation
screens that complete diversity-ordered pool until it has admitted up to
`--max-pair-attempts` exact-IK-feasible candidates to path planning. IK failures
therefore no longer consume the path-attempt budget. An optional
`--max-ik-screen-candidates` bound can cap the screening work; its default `0`
checks the finite pool until it is exhausted or the path pool is full. A
transformed corridor is never inferred for an identity-only pair. Before either simulation or real planning,
the loader rechecks the full saved inserter grasp library
against the supplied pickup position, roll, pitch, yaw, and floor height; this
is deliberately not tied to one simulated orientation. Stage-3 pair retention
round-robins across inserter grasps before taking second pairs for the same
grasp, so arbitrary detected orientations do not collapse the validated pool
onto a few similar pickup approaches. That eight-candidate pool is itself
round-robined across retained insertion corridors, so the earlier Cartesian
score cannot hide the opposite-side option before joint planning. Runtime
layout scoring evaluates shoulder-to-grasp lines at both pickup and
pre-insertion and records a soft score penalty when either phase crosses the
arms. The bounded queue and the post-MoveIt pre-rank order additionally enforce
the hard phase boundary, so a successfully pre-planned crossed corridor cannot
jump ahead of an unchecked or cheap-preplan-failed clear corridor. The
pre-ranker starts at each planned pickup-lift joint state, probes A7
`+pi`/`-pi`, and also seeds valid A7 `+3.0`/`-3.0` rad branches because the
iiwa limit is slightly below pi. IK is free to adjust every joint to make up
the remaining orientation. It sorts successful candidates with non-crossing
pre-insertion phase first and then by velocity-weighted transition joint-path
cost. All candidates not successfully pre-planned retain producer order within
their clear/crossed phase: the smaller pre-rank search may promote positive
evidence, but a missed branch cannot demote a top grasp behind the entire
unchecked queue. Exact complete-state IK remains authoritative. It then starts Isaac and streams
each MoveIt polyline continuously with position and velocity targets. It
settles only at the holder grasp, incoming-part grasp, and final pre-insertion
pose; intermediate MoveIt points and transport checkpoints no longer become
stop-and-oscillate commands. Loaded transport defaults to `0.70 rad/s`. The
incoming part is pinned at its authored pickup pose only until the inserter
establishes contact; loaded transport then relies on physics contact instead of
overwriting the part root state. Final validation checks both position and
orientation for the held base and incoming part. Incoming orientation is
accepted against the selected target and finite symmetry-equivalent targets for
the same held-base pose. The default angular tolerance is `0.20 rad` for each
object. The wrapper stops the MoveIt stack when finished. Add `--headless` for a non-GUI run or
`--record-video /tmp/dual_part_0.mp4` to save a video. Use
`--joint-rank-candidates N` to change the pre-plan bound or
`--skip-joint-space-ranking` only for a comparison run.

The dual iiwa MoveIt stack defaults to tuned KDL: its redundant-space search
resolution is `0.03 rad` instead of the former `0.005 rad`, and its solver
timeout is `0.10 s`. Exact preflight starts four complete-state searches at
the first pose, retains up to four coordinated branches, and then preserves
one continuation per branch through five pickup approach waypoints. These
solutions are cached only for the current detected world pose and planning
request; a new perception pose creates new targets and new IK searches.

PickIK global mode is retained as an explicit A/B option. Install the optional
ROS plugin before selecting it:

```bash
sudo apt install ros-humble-pick-ik
cd ros2_ws
colcon build --packages-select robot_integration_ros --symlink-install
```

Then pass `--ik-solver pick_ik`. Its optional configuration uses global mode
with 1 mm position and 0.01 rad orientation thresholds. In the measured hard
failure, PickIK found no additional feasible grasp and was slower than tuned
KDL, so it is not the operational default.

The MoveIt launch is isolated in its own process group. The launcher hands the
exact group ID and Linux process start-time token back to the wrapper, so the
wrapper can still terminate the owned ROS group if an inner launcher shell has
already exited. On success, planner failure, Isaac failure, Ctrl-C, or shell
exit, cleanup sends TERM and then KILL only to that validated group. The
benchmark applies the same ownership-aware teardown and stops after the first
explicit existing-stack conflict instead of recording the rest of the matrix as
planner failures. Persistent-stack reuse is the normal behavior. Pass
`--start-moveit` only for an intentionally temporary, wrapper-owned stack;
`--keep-moveit` is valid only with that option.

During visible simulation and guarded real execution, a localhost browser
debugger opens before candidate preflight or Isaac. In real mode it starts
before the actual-pose pickup-floor filter, so even an empty candidate queue
shows the configured floor plane, checked/accepted counts, rejection message,
and `pickup_floor_check` failure stage. The same tab reconnects to the real
MoveIt executor after task construction.
It follows the currently attempted pair, renders the partial assembly,
incoming part, and both selected grippers in `base_link`, and highlights
whether MoveIt is planning the holder grasp, incoming-part pickup, or
pickup-to-pre-insertion transition. Joint-space candidate pre-ranking is shown
as part of the transition stage. The debugger also shows the exact target phase,
transition ID, failure message, and recent fallback history. Its candidate-check
card distinguishes pickup grasps accepted by the runtime floor check, retained
Stage-3 pairs/executions, the pose-filtered queue's clear/crossed split and unique holder/inserter
grasps, joint-space pre-ranking results, and cumulative exact-IK checks. Drag
to orbit and use the mouse wheel to zoom. The candidate card explicitly marks
pickup or insertion crossing, and IK preflight renders the pre-insertion side
rather than the pickup pose. It also reports holder-gripper floor clearance in
millimetres from the exact displayed mesh and colors a penetrating gripper red.
Geometry is bounded, projected once
per rendered frame, and redrawn only when the visual planner state or camera
changes, while the status panel polls independently at 10 Hz. Pass
`--no-planning-debug-gui` to suppress only this browser view; `--headless`
suppresses it together with the Isaac GUI.

Do not pass `--holder-only` when testing transition symmetry; that option
intentionally skips the inserter transport and pre-insertion phases.

The other selected-order incoming parts are `3`, `1`, and `4`:

```bash
./run_pipeline.sh --mode sim --robots both --incoming-part-id 3
./run_pipeline.sh --mode sim --robots both --incoming-part-id 1
./run_pipeline.sh --mode sim --robots both --incoming-part-id 4
```

The plan and attempt artifacts are written next to the Stage-3 artifacts. For
part `0`, inspect the chosen grasp pair and transition with:

```bash
jq '{
  pair_id,
  transition_id,
  is_identity: .transition_symmetry.is_identity,
  insertion_vector: .transition_symmetry.pre_to_final_translation_assembly_m,
  transition_motion_score,
  transition_motion_components,
  joint_space_ranking: .moveit.joint_space_ranking
}' artifacts/dual_grasp_planning/plumbers_block/simple_dual_robot_sim_plan_part_0.json
```

The selected pickup grasp retains its parent and symmetry provenance. Once the
gripper closes, its part-to-TCP transform is fixed: transition fallback may
choose another compatible destination corridor, but cannot silently change
the grasp or regrasp the part. A pickup symmetry bridge preserves this rule:
it uses one equivalent part-to-TCP transform for pickup and pre-insertion, then
right-composes the object source pose by the inverse transform so the gripper
targets remain exactly the Stage-3-validated targets. A 180-degree A7 seed is considered only while
solving a symmetry-validated destination pose, is discarded when it exceeds
the bounded iiwa joint limits, and never bypasses MoveIt path planning.

If a candidate fails after partial mock execution, recovery retracts the
inserter before returning the holder to its start pose. This ordering clears
the central shared workspace before the holder moves. A recovery failure is
still fatal because the next candidate must never be planned from an unknown
or partially reset state; its messages are saved in the failed plan artifact.

### 2a. Run the resumable dual-assembly benchmark

The default benchmark runs all four `plumbers_block` insertion steps at twelve
front-of-robot pickup locations and eight incoming-part orientations: 384
MoveIt-plus-Isaac cases. Each side samples three forward depths and two lateral
offsets. The assembled prefix remains upright at
`(0.55, 0.0)` for every case. Negative-Y pickups assign `lbr_one` as inserter;
positive-Y pickups assign `lbr_two`, and the opposite arm holds the assembly.

Start with a small smoke slice:

```bash
./run_pipeline.sh --benchmark dual-assembly --limit-cases 4
```

Named filters make a single matrix cell reproducible without changing YAML:

```bash
./run_pipeline.sh --benchmark dual-assembly \
  --parts 3 \
  --placements right_inner_middle \
  --orientations upright_yaw_0
```

Then run or resume the complete benchmark:

```bash
./run_pipeline.sh --benchmark dual-assembly
```

Each case has its own plan JSON, Isaac attempt JSON, combined log,
browser-playable WebM scene recording, and terminal `case.json`. If planning
fails before Isaac can record a frame, the case instead receives a
`failure_scene.svg` rendered from the actual Fabrica meshes, assembled prefix,
world placement, pickup orientation, and arm roles. The record and dashboard
also name the failed phase, such as `Holder/base grasp`, `Incoming-part grasp`,
or `Transport to pre-insertion`, and retain the actionable exception rather
than the wrapper's final cleanup line. `events.jsonl`, `summary.json`,
`summary.csv`, and the HTML dashboard are updated after a case starts and again
when it finishes. Pressing Ctrl-C stops the current wrapper cleanly, records the
case as interrupted, and leaves all prior results viewable. Running the same
command resumes interrupted and pending cases while skipping completed cases.
Use `--retry-failed` to rerun failures or `--no-resume` to start the event log
again.

To measure fixes without mixing old successes into the new report, select the
failed case IDs from a completed summary and write them to a separate output:

```bash
./run_pipeline.sh --benchmark dual-assembly \
  --failed-from-summary artifacts/dual_assembly_benchmark/plumbers_block_384_positions/summary.json \
  --output-dir artifacts/dual_assembly_benchmark/plumbers_block_failed_retry \
  --no-resume
```

The new manifest and dashboard contain only the previously failed cases, so
every success in that dashboard is a recovered case.

For a fast planner-only regression, skip Isaac entirely and select just the
cases that previously failed during MoveIt candidate planning:

```bash
./run_pipeline.sh --benchmark dual-assembly \
  --failed-from-summary artifacts/dual_assembly_benchmark/plumbers_block_384_positions/summary.json \
  --failure-stages moveit_candidate_planning \
  --planning-only \
  --output-dir artifacts/dual_assembly_benchmark/plumbers_block_moveit_planning_only \
  --no-resume
```

In this mode, a case succeeds only when the planner writes a complete MoveIt
plan and returns successfully. Isaac is never launched, no video or attempt
artifact is expected, and the resumable HTML/CSV report still records the
candidate, timing, precise planning failure, and a static scene image for both
successful and failed planner-only cases.

To benchmark the stronger real-executor safety boundary offline, do not start
MoveIt separately. The benchmark owns one shared mock stack for the complete
run, reuses it across cases, and stops it on completion or Ctrl-C:

```bash
source ./setup_robot_env.sh
./run_pipeline.sh --benchmark dual-assembly \
  --real-preflight-only \
  --ros-domain-id 43 \
  --parts 0 \
  --placements right_inner_middle \
  --orientations upright_yaw_0 \
  --output-dir artifacts/dual_assembly_benchmark/real_connected_preflight_smoke \
  --no-resume
```

To plan from the physical robots' live current joint state instead, start the
hardware MoveIt stack yourself and use `--ros-domain-id 0` for the benchmark.
Persistent-stack reuse is already the default and fails fast if that stack is
unavailable.

`--real-preflight-only` calls the real executor without `--execute`, plans all
seven connected segments through
`inserter_preinsertion`, and never launches Isaac. OMPL is used for free-space
segments and the four approach/lift/insertion segments must be complete
straight-line Cartesian paths. The attempt JSON and dashboard distinguish a
`real_connected_preflight` failure and retain the exact failing segment. These
benchmark poses come from the YAML matrix rather than live vision; use the
action-server pipeline for the separate live-perception test.

To separate single-arm pickup reachability from dual-arm coordination, first
start the mock MoveIt stack and then run the isolated pickup A/B benchmark
against an existing benchmark summary:

```bash
./start_dual_lbr_moveit.sh --mode mock --ros-domain-id 43

# In a sourced second terminal on the same ROS domain:
ROS_DOMAIN_ID=43 ./run_pipeline.sh --benchmark solo-pickup-ik \
  --baseline-summary artifacts/dual_assembly_benchmark/plumbers_block_ik_after_gripper_fix_20260811/summary.json \
  --output-dir artifacts/dual_assembly_benchmark/solo_pickup_ik_ab_20260812
```

The A side is the saved full dual-arm result. The B side runs every
floor-valid Stage-3 incoming-part grasp through pickup pregrasp, interpolated
approach, closed grasp, and lift for `lbr_one` and `lbr_two` independently. It
also runs the unary-valid holder/base grasps through pregrasp, interpolated
approach, and closed contact for each arm; the holder does not lift because the
production holder sequence ends at contact. It keeps the tested arm's joint
limits, self-collision, work-surface collision, exact TCP target, and
candidate-specific gripper widths, while ignoring the passive robot, inter-arm
contacts, and the other separately placed object. `summary.json` and
`index.html` checkpoint after every case. The outcome matrix distinguishes
dual failures where both assigned pickup tasks work independently, cases
recovered only by swapping arm roles, and cases where at least one isolated
pickup remains unreachable.

By default the A/B runner evaluates every floor-valid incoming grasp for both
robots, even after finding a successful grasp. Each dashboard row links to a
collision-debug-style `incoming_grasps.html`: the actual global part pose,
full work surface, `base_link` axes, both physical robot bases and shoulders,
contacts, complete KUKA gripper collision mesh, and per-arm IK outcome for
every grasp. The cyan ghost is the exact 10-cm pregrasp TCP target and the
brown gripper is the contact pose; dashed lines connect both shoulders to the
selected pregrasp. Use the dashboard's `world` link for the complete scene or
`focus` for a close object/gripper view, and switch between them inside the
viewer. Green means at least one isolated arm completed the pickup sequence;
red means both failed. The candidate details identify the failed target and
the counts of IK requests, no-solution responses, valid states, and
collision-invalid states. Pass `--no-grasp-debug` only when a faster
aggregate-only rerun is desired.

The A/B `index.html` embeds the first failed pose's world viewer immediately.
Its pose selector contains every completed dual-failed bundle; after selecting
a pose, use the report's previous/next grasp buttons or the left/right arrow
keys to traverse all of that pose's floor-valid grasps without opening another
page. The statistics table remains below the viewer.

Exact preflight is reliability-first. For every target it tries seven balanced
bounded seeds spanning the current/pre-ranked state, both valid joint-7
branches, both joint-1 shoulder directions, and both joint-3 upper-arm
directions. It retains four low-motion holder branches through grasp, then
searches the complete inserter sequence against each frozen 14-joint holder
state. KDL solves only the active arm; that kinematic result is cached across
pair variants, inserted into the current complete two-arm/finger state, and
revalidated by MoveIt on every reuse. Thus passive-arm collisions are never
cached, while repeated pair combinations do not repeat the same expensive KDL
solve. Pickup IK prefers a 10 cm pregrasp, then tries 7.5, 5, and 2.5 cm only
when the preceding pickup pregrasp/approach fails with pure kinematic no-IK.
Collision-invalid states never trigger this shortening. Every shorter approach
is checked again through the complete dual-arm state, target-specific object
geometry, and all five continuation poses; the selected offset and exact
continuation joints are serialized and executed. The selected complete branch
is reused by trajectory planning. The
wrapper uses a short `0.35 s` timeout per distinct active-arm seed; after IK
succeeds, OMPL receives up to `15 s` and `16` planning attempts. Override these with
`--ik-timeout-s`, `--exact-ik-candidates`, `--exact-ik-beam-width`,
`--exact-ik-seed-perturbation-rad`, `--pickup-pregrasp-offsets-m`,
`--planning-time-s`, and
`--planning-attempts` when reproducing a solver boundary.

Open the incremental dashboard at:

```text
artifacts/dual_assembly_benchmark/plumbers_block/index.html
```

The dashboard provides live statistics for the currently visible cases. Filter
by failure phase, part, pickup location, source-frame orientation, status, or
inserter arm; search error text and candidate IDs; and sort the case cards by
phase, runtime, part, location, or orientation. The configurable group table
shows pass rate, median runtime, and the dominant failure phases for any of
those dimensions. Clicking a table row or visual-guide tile filters the cases.
The guide uses actual OBJ thumbnails for parts, top-down workspace diagrams for
locations and arm assignment, oriented XYZ diagrams for RPY, and a phase
timeline for failure groups. The gallery retains lazy-loaded inline video
playback or failure-scene stills, explicit per-case failure phases, and links to
each plan, attempt, and log. Both the benchmark and one-case simulator
use static/dynamic contact friction `5.0/4.0` and a KUKA finger effort limit of
`200`, so weak simulated contact is not the limiting factor. Finger motion is
not commanded as one high-force step: it follows a three-second quintic close,
latches on bilateral contact with the selected object, and then smoothly
finishes the same three-second ramp to the configured high-force hold target;
there is no instantaneous post-contact target jump. Arm and driven-finger implicit position drives
use configuration-adaptive diagonal critical damping, recomputed from the
current generalized inertia as `D = 2 sqrt(K I)` (`zeta = 1`). Edit
`configs/dual_assembly_benchmark.yaml` to change placements, RPY orientations,
physics, or case limits; incoming-part Z is still computed from the rotated mesh
so it rests on the floor.

Each recorded case has a JPEG scene thumbnail used as the video poster. Videos
are loaded only after pressing `Play recording`, so large reports do not create
hundreds of simultaneous browser media players. The direct `open video` link is
also retained as a fallback. Failed cases without video show their rendered
scene image directly; click it to open a near-full-screen view, then use Escape,
the close button, or the dark backdrop to dismiss it.

For a completed report, these commands repair browser video paths/posters and
backfill failure-stage scene stills without rerunning MoveIt or Isaac:

```bash
./run_pipeline.sh --benchmark dual-assembly --repair-videos
./run_pipeline.sh --benchmark dual-assembly --repair-failure-evidence
```

### 3. Inspect or run MoveIt separately

To keep MoveIt running for RViz or repeated planner calls:

```bash
./run_pipeline.sh --mode sim --robots both --bringup-only --rviz
```

The default solver is `kdl`; use `--ik-solver pick_ik` for an explicit global
solver comparison after installing the optional plugin. When reusing an
existing stack, the runner cannot change its solver—restart MoveIt with the
requested solver first.

In another terminal:

```bash
source ./setup_robot_env.sh
python3 scripts/smoke_test_dual_lbr_moveit.py
./run_pipeline.sh --mode sim --robots both --incoming-part-id 0
```

The shared MoveIt model contains `arm_one`, `arm_two`, and `both_arms` in one
planning scene, so cross-arm collisions remain enabled. The normal default maps
`lbr_one` at `Y=-0.42 m` to holder and `lbr_two` at `Y=+0.42 m` to inserter.
Simulation can pass `--inserter-arm auto` to swap those logical roles according
to pickup Y; the saved task, debugger, MoveIt group, and Isaac scene retain the
resolved physical-arm provenance.

Hardware MoveIt continuously receives the measured physical gripper closure
from `/left/gripper_controller/position` and
`/right/gripper_controller/position`. The bridge republishes the last valid
reading at 20 Hz, including after the gripper stops moving; before the first
reading it uses the fully-open model envelope and warns. Candidate planning
then sets the exact requested finger state explicitly. It uses the
5-mm-per-finger approach opening through pregrasp/grasp IK and planning, then
switches that gripper to the selected contact width after the grasp. Both states
are checked against the complete shared planning scene; unspecified finger
joints do not fall back to MoveIt's fully-open default.

### 4. Real dual-arm safety boundary

Without `--execute`, real mode performs the complete non-moving connected
motion preflight:

```bash
./run_pipeline.sh --mode real --robots both --incoming-part-id 0
```

Real mode uses the same pose-dependent pickup-floor filter and retained-first,
identity-fallback queue as simulation. It writes at most 256 candidates to the
preflight task by default; use `--max-pair-attempts N` to override that real-task
bound. In simulation, the same option instead limits exact-IK-feasible candidates
admitted to expensive path planning, after the broader pose-feasible pool has
been screened. Real preflight starts from MoveIt's complete live dual-arm state
and plans every requested segment before motion. Free-space segments use OMPL;
holder pregrasp-to-grasp, inserter pickup approach, pickup lift, and the final
pre-insertion descent require complete collision-aware Cartesian paths with
fixed TCP orientation. Each plan starts from the preceding trajectory endpoint,
so disconnected endpoint IK cannot admit a candidate. Execution checks both
arms against each saved segment start (default tolerance `0.05 rad`) and plays
the exact preflight trajectories without replanning. Cartesian paths default to
a `0.005 m` interpolation step and reject revolute jumps above `0.35 rad`; the
real wrapper exposes `--cartesian-max-step-m`,
`--cartesian-revolute-jump-threshold-rad`, and
`--execution-start-tolerance-rad`. The ROS action reports
success only after `inserter_preinsertion`; an intentionally earlier
`stop_after` is returned as partial execution rather than a reached pose. When
a later ranked candidate is selected, the action result pose comes from that
candidate rather than the task's rank-1 compatibility fields.

Real task construction preserves the selected physical roles. With
`--inserter-arm auto`, pickup Y below assembly Y assigns `lbr_one` as inserter
and `lbr_two` as holder; pickup Y at or above assembly Y assigns the reverse.
The executor validates and uses the task-declared robot, MoveIt group, TCP, and
joint names. Downstream real-execution paths must not hard-code logical
holder/inserter roles.

Real mode opens the same live browser debugger after the perception poses are
resolved but before the pickup-floor filter runs. If every grasp is rejected,
the page still reports the configured floor Z, filter counts, rejection reasons,
and fatal stage. If candidates survive, the same tab reconnects after the task
handoff, changes scene as exact IK rejects or selects ranked candidates, then
follows holder approach/grasp, incoming pickup, and transport through the
configured stop phase. Use `--debug-gui-port N` to override the stable real-mode
port (`38825` by default). Direct wrapper runs can pass
`--no-planning-debug-gui`; the action server can pass `--headless` (or set
`grasp_assembly_action.headless: true`) to suppress it.

Hardware execution requires the correctly configured hardware MoveIt stack,
explicit `--execute`, and confirmation unless `--yes` is supplied. Generated
dual tasks now add phase-aware collision boxes for both detected workbench
parts. The stationary held subassembly remains a carved world obstacle, and
the incoming part becomes an attached TCP-frame collision body after pickup so
MoveIt checks its loaded lift and pre-insertion path. The carved regions admit
only the selected gripper-contact and insertion corridors. The legacy
`--allow-objectless-planning` flag is needed only to run an older task artifact
that lacks this geometry. Review `./run_pipeline.sh --help` and the
KUKA hardware runbook below before enabling motion.

Candidate cleanup detaches that incoming collision body and then explicitly
removes the world object with the same ID. MoveIt restores a detached body to
the world, so detach alone would leave stale incoming-part geometry in later
holder IK checks. Either cleanup operation failing is fatal: subsequent
candidates are not evaluated and their collision results are not cached against
an unknown scene.

The dual real executor publishes normalized positions to
`/left/gripper_controller/position_command` for `lbr_one` and
`/right/gripper_controller/position_command` for `lbr_two`
(`std_msgs/msg/Float64`): `0` is 74 mm fully open and `1` is 7 mm fully closed.
Each side must first be calibrated separately with empty jaws. Execution calls
each namespaced `open` Trigger service at startup, commands the same
candidate-specific approach width used by MoveIt, and monitors the matching
`/position` feedback topic. The grasp then uses that side's `close` endpoint so
motor contact can stop it before 7 mm. Repeated identical positions are not
republished; `stop` remains the emergency interruption endpoint.

Gripper availability is physical-side-local during dual real execution. The executor
checks both controllers before opening either one, opens and commands every
controller that is present, and resolves holder/inserter through the current
plan's robot assignment (`lbr_one -> left`, `lbr_two -> right`). Thus automatic
holder/inserter role swaps cannot send a command to the other robot's gripper.
An unavailable physical side is recorded against its current task role while
the same arm trajectory continues. MoveIt still
uses the planned finger state and attached incoming-part collision body. Thus a
missing inserter controller produces an empty-arm motion diagnostic, not a
physical pickup, even though the arm can continue through pre-insertion. Once a
controller is discovered, a later open, position, feedback, or stop failure is
still treated as a real gripper fault rather than an optional absence.

## Grasp Generation Benchmark

Run the standalone benchmark to evaluate grasp generation over Fabrica OBJ parts and robust stable orientations without executing in MuJoCo, Isaac, MoveIt, or hardware:

```bash
./run_pipeline.sh --benchmark grasp-generation --limit-parts 1
./run_pipeline.sh --benchmark grasp-generation --assembly plumbers_block --clean
```

The default config is `configs/grasp_generation_benchmark.yaml`; outputs go to `artifacts/grasp_generation_benchmark/` with `results.json`, `summary.csv`, `summary.md`, `index.html`, per-part stage artifacts, stable-orientation metadata, and optional generation-only fallback plans. The benchmark requires the same collision backend as normal stage-1 filtering.

## Grasp Execution Benchmark

After running the generation benchmark, execute selected stage-2 feasible grasps in MuJoCo and/or Isaac with per-attempt artifacts and videos:

```bash
./run_pipeline.sh --benchmark grasp-execution --assembly beam --part 0 --limit-orientations 1 --max-grasps-per-orientation 2
./run_pipeline.sh --benchmark grasp-execution --backend both --assembly beam --part 0 --orientation orientation_003 --max-grasps-per-orientation 1
./run_pipeline.sh --benchmark grasp-execution --backend both --assembly beam --part 0 --orientation orientation_003 --max-grasps-per-orientation 9 --placement-xy-world 0.5,0.0 --no-resume
./run_pipeline.sh --benchmark grasp-execution --backend mujoco --record-video all --limit-attempts 10
```

For the KUKA iiwa7 Isaac path, start the LBR mock state/controller stack and namespaced MoveIt planning server in one terminal:

```bash
./start_lbr_moveit.sh
```

The helper launches `robot_state_publisher`, mock `ros2_control`, and MoveGroup together. The default `pdz_gripper` embodiment is derived from the same URDF used by current policy training, so MoveIt plans to `pdz_gripper_tcp`. Pass `--gripper-model y_gripper` only for a legacy Y-gripper policy or artifact.

Then execute the highest-scored direct grasp for every `plumbers_block` orientation that has a stage-2 feasible grasp:

```bash
export ROS_LOG_DIR=/tmp/ros-log
source /opt/ros/humble/setup.bash
source /home/pdz/lbr-stack/install/setup.bash
source ros2_ws/install/setup.bash
./run_pipeline.sh --benchmark grasp-execution \
  --backend isaac \
  --assembly plumbers_block \
  --max-grasps-per-orientation 1 \
  --no-resume \
  --output-dir artifacts/grasp_execution_benchmark_kuka_plumbers_block_corrected
```

The KUKA conversion currently applies only to the benchmark's `isaac` block. Do not use `--backend mujoco` or `--backend both` for this run. Omit `--max-grasps-per-orientation 1` only when intentionally running every direct feasible grasp; the current `plumbers_block` generation artifacts contain thousands of such candidates.

The default config is `configs/grasp_execution_benchmark.yaml`; outputs go to `artifacts/grasp_execution_benchmark/` with resumable `attempts.jsonl`, `results.json`, `summary.csv`, `index.html`, per-attempt `attempt.json`, logs, and browser-playable `attempt.webm` when video recording is enabled. The benchmark consumes the generation benchmark's stage-2 bundles and runs direct stage-2 feasible grasps in descending score order; orientations with no stage-2 feasible grasp are skipped rather than converted into regrasp attempts. By default, execution attempts keep each stage-2 bundle's saved stable orientation and Z height but shift object XY to the normal robot workspace at `[0.5, 0.0]`; pass `--placement-xy-world x,y` to choose another table location or `--use-bundle-placement` to use the generation bundle pose verbatim.

## ROS2 Workspace

The repo now contains a dedicated ROS2 workspace for hardware-facing integration:

- `ros2_ws/src/robot_integration_ros`
- `ros2_ws/dependencies.repos` for pinned external ROS2 package sources

This keeps the real-robot entrypoints and ROS packaging under `colcon`, while the rest of the project stays a normal Python repo.

Before building the overlay, fetch the pinned ROS2 source dependency used by `pitl` and `real` mode `DebugPoseItem` intake:

```bash
bash scripts/download_ros2_dependencies.sh
```

Build and source it as an overlay on top of your FR3 / MoveIt workspace:

```bash
source /opt/ros/<distro>/setup.bash
source /path/to/your/fr3_moveit_ws/install/setup.bash

cd ros2_ws
colcon build --packages-select fp_debug_msgs robot_integration_ros --symlink-install
source install/setup.bash
```

Once sourced, you can run the real-robot EE mover with:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --execute
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation --execute
```

### Two-Shell Launch For Real Hardware

Use normal ROS2 discovery unless you deliberately need an isolated domain. The repo launcher defaults to `ROS_DOMAIN_ID=0` and clears localhost-only discovery settings.

Terminal 1: launch the FR3 MoveIt stack

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/franka_ros2_ws/install/setup.bash

ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=<robot_ip> use_fake_hardware:=false
```

Terminal 2: source the robot integration overlay and run the script

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/franka_ros2_ws/install/setup.bash
source /media/pdz/Elements1/perception_bag_test/ros2_ws/install/setup.bash
```

Optional check:

```bash
ros2 control list_controllers
```

You want `fr3_arm_controller` to be `active`.

Plan only:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation
```

Execute on hardware:

```bash
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --execute
ros2 run robot_integration_ros move_real_robot_ee --x 0.35 --y 0.00 --z 0.40 --keep-current-orientation --execute
```

## Standalone Real-Robot EE Motion

For direct MoveIt-controlled FR3 end-effector testing there is a standalone script:

```bash
python scripts/move_real_robot_ee.py --x 0.35 --y 0.00 --z 0.40
python scripts/move_real_robot_ee.py --x 0.35 --y 0.00 --z 0.40 --execute
python scripts/move_real_robot_ee.py --x 0.32 --y -0.10 --z 0.28 --roll 3.14159 --pitch 0.0 --yaw 1.5708 --execute
```

Notes:
- The canonical ROS2-packaged entrypoint is now `ros2 run robot_integration_ros move_real_robot_ee ...`.
- `scripts/move_real_robot_ee.py` is a source-tree shim around the same workspace package.
- The script assumes the FR3 MoveIt stack is already running elsewhere and exposing `/compute_ik`, `/plan_kinematic_path`, and `/execute_trajectory`.
- `--keep-current-orientation` queries the current EE pose from MoveIt FK and reuses only its orientation while applying your requested `x/y/z`.
- Without `--execute`, the script only checks IK and planning. Add `--execute` to move the hardware.
- The default motion scales are intentionally slow: `velocity_scale=0.05` and `acceleration_scale=0.05`.
- `Ctrl-C` only sends a best-effort cancel to MoveIt. Do not rely on it as a safety stop; use the robot-side stop/pause for emergency interruption.
- If you do not provide orientation, the script uses the thesis neutral/top-down orientation: `roll=pi`, `pitch=0`, `yaw=pi/2`.
- The default planning frame is `base` and the default end-effector link is `fr3_hand_tcp`. Override these if your MoveIt setup uses different names.

## Setup

For local Isaac execution, use the local Isaac Lab launcher. The default
Isaac configs call:

```bash
/media/pdz/Elements1/IsaacLab/isaaclab.sh -p
```

through `isaac_execution.python_executable`. This field may include launcher
arguments, so a local Isaac Lab command such as
`/media/pdz/Elements1/IsaacLab/isaaclab.sh -p` is supported. Docker remains an
optional helper for reproducing the older containerized environment, not a
requirement for Isaac sim execution.

Bootstrap the MuJoCo assets:

```bash
bash scripts/download_required_assets.sh
```

This does two things:
- sparse-clones the required MuJoCo Menagerie assets under `.cache/robot_descriptions/mujoco_menagerie`
- builds `.cache/generated_mujoco_models/fr3_with_panda_hand.xml`

Bootstrap the pinned ROS2 message dependency used by `pitl` / `real` `DebugPoseItem` intake:

```bash
bash scripts/download_ros2_dependencies.sh
```

This does two things:
- imports `ros2_ws/dependencies.repos` when `vcstool` is available
- ensures `ros2_ws/src/fp_debug_msgs` is present and pinned even without `vcstool`

Override `FP_DEBUG_MSGS_REMOTE` and `FP_DEBUG_MSGS_REF` if you need to bootstrap from a mirror, a local bare repo, or a different pinned ref.

The pipeline expects the vendored Franka hand collision mesh at:
- `assets/urdf/franka_description/meshes/robot_ee/franka_hand_black/collision/hand.stl`

The current KUKA iiwa7 PDZ-gripper configs use the trained local gripper meshes and generated robot USD/URDF:
- `assets/urdf/kuka_iiwa7_pdz_gripper/meshes/`
- `assets/urdf/kuka_iiwa7_pdz_gripper/urdf/kuka_iiwa7_pdz_gripper.urdf`
- `assets/usd/kuka_iiwa7_pdz_gripper/configuration/kuka_iiwa7_pdz_gripper_base.usd`

The legacy Y-gripper remains available for older bundles and checkpoints:
- `assets/urdf/kuka_iiwa7_y_gripper/meshes/{hand.STL,left_finger.STL,right_finger.STL}`
- `assets/urdf/kuka_iiwa7_y_gripper/urdf/kuka_iiwa7_y_gripper.urdf`
- `assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda`

The PDZ URDF is the current kinematic source of truth. Its calibrated `pdz_gripper_tcp` is `0.1705 m` along local Z from `lbr_link_7`, and it carries the named D405 mount used by the new training runs. `python3 scripts/build_kuka_moveit_description.py` regenerates both repo-local MoveIt/ros2_control xacros used by mock simulation and real hardware. The checked-in USD and MoveIt descriptions are covered by FK-equivalence and mount-contract regression tests.

Changing the physical mount also changes collision geometry. The KUKA Stage-1 cache key includes a mount-geometry version, so rebuild the dual artifacts before sim or real testing rather than reusing grasps checked against the old hand orientation.

During dual Isaac close, selected-width finger travel remains a useful geometry check. If physics stalls slightly early, the replay also accepts measured contact only when both role-specific fingertip sensors report force against the intended object (`BasePart` for the holder or `IncomingPart` for the inserter). Floor, other-part, and one-sided contacts cannot satisfy that fallback.

For `gripper_collision_model: kuka_y_gripper`, saved bundles identify `robot_model: kuka_iiwa7`, `gripper_model: kuka_y_gripper`, and `tcp_link: gripper_tcp`.

The visual-servo RL task uses the PDZ parallel gripper instead. Its source
URDF and meshes are under `assets/urdf/kuka_iiwa7_pdz_gripper/` and
`assets/urdf/pdz_gripper_description/`, and its Isaac asset is
`assets/usd/kuka_iiwa7_pdz_gripper/kuka_iiwa7_pdz_gripper.usd`. The aperture
is 12--76 mm; goal contacts are limited to 62 mm so the approach command can
add 5 mm per finger. The named TCP is 135.5 mm above the PDZ base and rotated
-90 degrees about local Z. The checked-in MoveIt xacro and SRDF use that same
named frame, and `configs/grasp_generation_benchmark_pdz.yaml` records the
matching collision-generation contract.

RL goal RGB-D is generated from the collision-validated PDZ trajectories with
MuJoCo Filament via `python3 isaac_rl/scripts/prepare_plumbers_block_catalog.py
--stage mujoco`. The renderer uses the deployed D405 intrinsics and wrist
transform, hides only the camera enclosure surfaces that contain the optical
origin, and retains the visible gripper fingers/pads, part, and T-slot surface.

### Copying Local State To A New Worktree

Tracked assets are checked out by Git, but the MuJoCo cache, pinned ROS2 source dependency, and `colcon` build/install/log directories are ignored local state. After creating a new worktree, copy those directories from this worktree with:

```bash
git worktree add /path/to/new-worktree <branch>
scripts/copy_worktree_local_state.sh --to /path/to/new-worktree
```

You can also opt into an automatic `post-checkout` hook for future worktrees:

```bash
git config core.hooksPath .githooks
git config grasp-planning.localStateSource "$(pwd -P)"
```

Git does not enable repository hooks from tracked files by default. Once enabled, the hook runs after `git worktree add` performs the initial checkout and copies the ignored local state from `grasp-planning.localStateSource`.

## Config Layout

Pipeline configs:
- `configs/grasp_pipeline_sim.yaml`
- `configs/grasp_pipeline_sim_cumotion.yaml`
- `configs/grasp_pipeline_sim_isaac.yaml`
- `configs/grasp_pipeline_pitl.yaml`
- `configs/grasp_pipeline_pitl_isaac.yaml`
- `configs/grasp_pipeline_real.yaml`

Shared MuJoCo execution config:
- `configs/mujoco_simulation.yaml`

Isaac execution config:
- `isaac_execution` block inside `configs/grasp_pipeline_sim.yaml`
- `isaac_execution` block inside `configs/grasp_pipeline_pitl.yaml`
- `scripts/run_fabrica_grasp_in_isaac.py`
- `scripts/convert_stl_to_usd.py`

Mesh/frame debug view:

```bash
./scripts/write_part_frame_debug_html.py \
  --input-json artifacts/pitl_pipeline_stage2_ground_feasible.json \
  --output-html artifacts/part_frame_debug.html
```

This writes a mesh-only HTML view showing the saved bundle-local part, its
area-weighted centroid, and the transformed execution/world pose when the input
bundle contains `metadata.execution_world_pose`.

Real hardware execution config:
- `real_execution` block inside `configs/grasp_pipeline_real.yaml`

### D405 policy-assisted real grasp approach

The real executor can keep normal MoveIt IK/path planning for the selected
stage-2 pregrasp and replace only the final pregrasp-to-grasp segment with the
trained D405 PPO policy:

```text
stage-2 bundle target
-> MoveIt plan/execute to pregrasp
-> synchronized D405 RGB-D + explicit catalogue target
-> deterministic policy twists
-> live optical-frame TF into lbr_link_0
-> safety supervisor
-> collision-checking MoveIt Servo
-> four-frame learned completion hold
-> explicitly approved gripper close
-> optional MoveIt lift
```

Set these fields in the selected real-pipeline YAML:

```yaml
real_execution:
  grasp_approach_controller: "d405_policy"
  visual_servo_config: "configs/visual_servo_real_d405.yaml"
  stop_after: "grasp"
```

Then fill the checkpoint, checkpoint-metadata sidecar, and exact `target_id` in
`configs/visual_servo_real_d405.yaml`. Startup fails if the target's part or
grasp ID differs from the selected stage-2 bundle, if the D405 serial/profile
does not match, or if the checkpoint/catalogue hashes and observation/action
contracts do not match the sidecar.

After reviewing and selecting a checkpoint, create its sidecar with:

```bash
python3 scripts/write_d405_checkpoint_metadata.py \
  --checkpoint /path/to/reviewed-checkpoint.pth \
  --output /path/to/reviewed-checkpoint.deployment.json \
  --source-commit "$(git rev-parse HEAD)"
```

The robot-side Python environment needs the lightweight policy dependencies but
does not need Isaac Sim:

```bash
python3 -m pip install -e '.[deployment]'
```

The checked-in deployment config is intentionally non-moving:
`command_sink: dry_run`, `real_motion_approved: false`, and
`allow_gripper_close_on_completion: false`. Start the aligned LBR stack with
the opt-in Servo node for dry-run/status checks and later staged motion:

```bash
./start_lbr_moveit.sh --mode hardware --servo
python3 scripts/run_d405_ppo_visual_servo.py \
  --config configs/visual_servo_real_d405.yaml \
  --expected-part-id 0 \
  --expected-grasp-id g1973
```

The dry-run preflight also requires fresh deadman and E-stop heartbeats. For a
stationary, non-moving dry-run only, these can be synthetic publishers in two
additional terminals:

```bash
ros2 topic pub -r 10 /d405_visual_servo/deadman std_msgs/msg/Bool '{data: true}'
ros2 topic pub -r 10 /d405_visual_servo/emergency_stop std_msgs/msg/Bool '{data: false}'
```

Do not use synthetic operator signals for real motion. Connect those topics to
the reviewed physical operator controls, set a real `WrenchStamped`
`force_topic`, and keep all three streams publishing as heartbeats; stale
joint, force, deadman, or E-stop data latches a zero-motion hold.

MoveIt Servo consumes speed-unit `TwistStamped` commands on
`/lbr/servo_node/delta_twist_cmds`, commands `gripper_tcp` in `lbr_link_0`, and
monitors the same planning scene published by MoveGroup. It therefore applies
joint/singularity and self/scene-collision slowdown or halt behavior during the
learned approach. A full motion planner is still used for pregrasp and lift;
calling a global planner for every 15 Hz policy step is not supported.

Real Servo output additionally requires `command_sink: moveit_servo`,
`real_motion_approved: true`, the CLI/caller real-motion confirmation, live
deadman and emergency-stop topics, reviewed workspace/joint/force limits, and
a healthy Servo status/command subscriber. Gripper closing occurs only after
the learned completion probability is at least `0.95` for four consecutive
low-speed frames and both `gripper_enabled` and
`allow_gripper_close_on_completion` are true.

After the standalone dry-run and guarded no-close trials pass, select
`d405_policy` in `configs/grasp_pipeline_real_lbr_iiwa7.yaml` and run the normal
real pipeline. It performs MoveIt pregrasp planning first, starts the policy
approach, closes only after the completion gate, deactivates Servo, and then
uses MoveIt for the configured lift:

```bash
./run_pipeline.sh --mode real --config configs/grasp_pipeline_real_lbr_iiwa7.yaml
```

Use the `planning` block in `configs/grasp_pipeline_*.yaml` to tune grasp generation and filtering:
- `stage1_cache_enabled` and `stage1_cache_dir` cache the generated stage-1 grasps plus surface samples per object mesh and stage-1 planning settings. Cache hits still write the normal stage artifacts.
- `roll_angle_step_deg` expands roll samples over a full 360 degrees. For example, `15.0` generates 24 roll angles from 0 through 345 degrees.
- `stage1_pose_upright_axis_enabled` adds a live-pose-derived world-upright roll sample during stage 1. Stage-1 caching stores the pose-independent base grasps and augments cache hits with only the missing per-run upright roll variants, so real/PITL pose jitter does not force a full regeneration.
- `detailed_finger_contact_gap_m` changes the gripper contact geometry used during detailed checks.
- `floor_clearance_margin_m` is a stage-2 filtering margin: the full hand/finger collision geometry must stay at least this far above the world `z=0` floor. This does not change MuJoCo execution settings.
- `top_grasp_score_weight` is applied during stage-2 scoring after the real/execution pose is known. It boosts grasps whose pregrasp-to-grasp approach is top-down in world coordinates, with movement mostly along `-Z`.
- `regrasp_transfer_top_grasp_score_weight` applies the same top-down score to the regrasp transfer pickup in the initial pose. It defaults higher than `top_grasp_score_weight` because the first pickup is more sensitive to wrist orientation than setting the object back down.
- `skip_stage1_collision_checks: true` keeps all generated stage-1 grasps and skips offline assembly collision filtering. For a one-off run, pass `--skip-stage1-collision-checks`.

Use `configs/mujoco_simulation.yaml` to tune:
- grasp approach settings such as `pregrasp_offset_m` and `gripper_width_clearance_m`
- scene contact settings such as object mass, friction, `solref`, `solimp`, margin, and gap
- robot timing and speed such as `timestep_s`, `control_substeps`, `speed_scale`, adaptive final-approach slowdown, IK, and trajectory settings
- gripper actuation and settle behavior such as `open_ctrl`, `closed_ctrl`, and `close_steps`

MuJoCo also has an optional one-placement regrasp fallback for the case where stage 1 finds assembly-feasible grasps but stage 2 rejects all of them because of the floor. Configure it in the `mujoco_execution` block:
- `regrasp_fallback_enabled: true`
- `force_regrasp_fallback: true` or `./run_pipeline.sh ... --force-regrasp-fallback` to test the fallback even when direct stage-2 grasps exist; forced mode skips the surface currently on the floor and fails clearly if no feasible different-surface plan exists
- `regrasp_plan_artifact` for the JSON plan containing the transfer grasp, staging pose, and final grasp
- `regrasp_html_artifact` for the side-by-side debug view of the initial/staging poses, floor plane, all evaluated grasp markers, and highlighted planned transfer/final grasps
- `regrasp_staging_xy_offsets_m` samples multiple table XY locations around `regrasp_staging_xy_world`
- `regrasp_max_placement_options` caps how many feasible placement options are written into the regrasp plan
- `regrasp_moveit_max_candidate_plans`, `regrasp_moveit_transfer_candidates_per_placement`, and `regrasp_moveit_final_candidates_per_placement` cap the runtime MoveIt candidate plans scored before MuJoCo execution

The fallback computes convex-hull support facets, checks homogeneous-COM stability for candidate resting poses, samples staging XY locations, searches for a final assembly-feasible grasp in each staging pose, and then searches raw stage-1 grasps for a transfer grasp that is floor-feasible in both the initial and staging poses. MuJoCo can execute this fallback either with native IK or with MoveIt-planned transfer/place/final-pick trajectories. With MoveIt, execution first plans and scores a capped set of placement/transfer/final-grasp combinations, ranks them by joint path length, joint jumps, and the static placement score, then executes the cheapest candidate in MuJoCo and falls back to the next cheapest if needed. During regrasp transport, the runner inserts higher lift/rotate/translate waypoints before descending to the staging placement; tune that with `robot.regrasp_transport_clearance_m` in `configs/mujoco_simulation.yaml`. The MuJoCo attempt artifact includes `planned_candidates`, `attempts`, and trajectory diagnostics for debugging path choice.

For a planning-only regrasp visualization that uses a known different-surface case, run:

```bash
./run_pipeline.sh --mode sim --config configs/grasp_pipeline_sim_plumbers_regrasp.yaml --backend none --headless
```

This writes `artifacts/plumbers_mujoco_regrasp_plan.html` without launching MuJoCo execution.

MuJoCo can either use its native damped-IK arm controller or MoveIt-planned arm trajectories:
- default sim config: `mujoco_execution.controller: "moveit"`
- native MuJoCo IK: set `mujoco_execution.controller: "native"`
- cuMotion through MoveIt: use `configs/grasp_pipeline_sim_cumotion.yaml`

The MoveIt-backed MuJoCo path requires the FR3 MoveIt stack to be running and sourced, but only uses MoveIt planning services. For direct pickups, MoveIt plans `pregrasp`, `grasp`, and `lift` trajectories from the stage-2 bundle. For regrasp fallback, it plans transfer, staging placement, retreat, and final-pick trajectories. MuJoCo still executes those joint waypoints, closes/opens the gripper, simulates contacts, and evaluates pickup success by object lift height.

The cuMotion sim config keeps MoveIt as the ROS2 integration layer and asks `/plan_kinematic_path` to use `moveit_pipeline_id: "isaac_ros_cumotion"` and `moveit_planner_id: "cuMotion"`. This requires a running MoveIt stack that has NVIDIA Isaac ROS cuMotion installed and configured under those identifiers. If your local MoveIt config uses different names, override `moveit_pipeline_id` or `moveit_planner_id` in the YAML.

Start the local fake-hardware cuMotion + MoveIt stack in one terminal:

```bash
./start_cumotion_moveit.sh
```

Leave that script running, then run the MuJoCo sim from another terminal.

Run MuJoCo sim with cuMotion-backed MoveIt planning:

```bash
./run_pipeline.sh --mode sim --config configs/grasp_pipeline_sim_cumotion.yaml --backend mujoco --headless
```

The default sim config uses Isaac execution; other configs can opt in with `isaac_execution.enabled: true`. The runner generates a collision-enabled bundle-local USD from the stage-2 bundle by default, so the spawned Isaac asset uses the same frame as the ground recheck. With no `isaac_execution.fr3_usd` override it uses Isaac Lab's Factory Franka mimic USD because that asset has manipulation-ready finger contact geometry. It also exposes the spawned gripper mesh prims as PhysX collision geometry before simulation reset, then validates success from the part lift height using `isaac_execution.success_height_margin_m`. Disable `mujoco_execution.enabled` if you want Isaac only. Isaac direct pickups use `isaac_execution.controller: "moveit"`: MoveIt plans the same `pregrasp`, `grasp`, and `lift` pose targets used by real execution, then Isaac streams the returned joint waypoints in simulation.

The real, single-arm policy, dual-arm, and PDZ benchmark configs now use `gripper_collision_model: pdz_gripper`, `pdz_gripper_tcp`, and the trained 62 mm contact-width limit. Legacy Y-gripper configs and `--gripper-model y_gripper` remain available for old artifacts; do not mix those with a PDZ checkpoint.

### KUKA iiwa7 Real Hardware Runbook

This runbook uses the working three-host network layout:

- `192.170.20.1`: pipeline, LBR ROS2 control, MoveIt, and RViz computer
- `192.170.20.2`: KUKA controller / FRI peer
- `192.170.20.3`: gripper computer

All ROS2 processes on `.1` and `.3` must start with the same `ROS_DOMAIN_ID`, non-localhost discovery, and compatible DDS configuration. Environment changes do not affect processes that are already running; restart a process after changing its ROS environment.

On the pipeline computer, source the repository helper in every new ROS2 terminal:

```bash
cd /media/pdz/Elements1/Grasp_Planning_hold_grasping
source ./setup_robot_env.sh
```

It sources ROS Humble, `/home/pdz/lbr-stack`, and the repository overlay, then selects domain `0`, network discovery, and Fast DDS. It also sets `GRASP_KEEP_ROS_DISCOVERY_ENV=1` so `run_pipeline.sh` preserves this hardware network configuration.

Before launching ROS, start the FRI client application from the KUKA SmartPAD with the configured client IP and a 10 ms send period. Use position control or joint impedance with client command mode `POSITION`. Place the real arm in a collision-free starting configuration; real execution plans from the current `/lbr/joint_states` and does not reset the physical robot to the Isaac ready pose.

Terminal 1 on `.1`: start the physical robot, trajectory controller, robot-state publisher, and namespaced MoveIt server from the aligned description.

```bash
cd /media/pdz/Elements1/Grasp_Planning_hold_grasping
source ./setup_robot_env.sh
./run_pipeline.sh --mode real --robots left --bringup-only --rviz
```

Terminal 2 on the gripper computer: start both new persistent controllers. The
verified adapter mapping is `left/lbr_one -> 5B3D047592` and
`right/lbr_two -> 5B3D044069`; both servos use ID 1.

```bash
cd ~/Workspaces/new_gripper/ros2
source servo_gripper/setup_gripper_env.sh

ros2 launch servo_gripper dual_grippers.launch.py
```

Calibrate each controller separately with empty jaws before its first pickup;
the pipeline deliberately does not recalibrate automatically:

```bash
cd ~/Workspaces/new_gripper/ros2
source servo_gripper/setup_gripper_env.sh
ros2 service call /left/gripper_controller/calibrate std_srvs/srv/Trigger "{}"
ros2 service call /right/gripper_controller/calibrate std_srvs/srv/Trigger "{}"
```

MoveIt bringup subscribes to both normalized `position` topics and republishes
the last measured passive finger coordinates at 20 Hz. The value therefore
remains available after motion finishes even if the gripper only reports on a
change. Before the first feedback sample, the scene uses a warned fully-open
fallback.

Terminal 3 on `.1`: publish the expected Fabrica `plumbers_block/0`
perception pose in the shared dual-cell `base_link` frame. Standalone
`lbr_one` is mounted at `base_link` Y=`-0.420 m`, so
`configs/grasp_pipeline_real_lbr_iiwa7.yaml` adds `+0.420 m` to the received Y
coordinate to express the pose in the single-arm `lbr_link_0` frame.

```bash
cd /media/pdz/Elements1/Grasp_Planning_hold_grasping
source ./setup_robot_env.sh
ros2 topic pub -r 2 \
  /perception/fp/pose_base/fused/assembly \
  fp_debug_msgs/msg/DebugPoseItem \
  "{assembly_name: plumbers_block, part_id: 0, mode: config_test, score: 1.0, pose_base: {pose: {position: {x: 0.5, y: 0.0, z: 0.04}, orientation: {x: -0.7071067811865475, y: 0.0, z: 0.0, w: 0.7071067811865476}}}}"
```

Terminal 4 on `.1`: verify discovery before enabling hardware execution.

```bash
cd /media/pdz/Elements1/Grasp_Planning_hold_grasping
source ./setup_robot_env.sh

ros2 service type /left/gripper_controller/calibrate
ros2 service type /left/gripper_controller/open
ros2 service type /left/gripper_controller/close
ros2 service type /left/gripper_controller/stop
ros2 topic type /left/gripper_controller/position_command
ros2 topic type /left/gripper_controller/position
ros2 service type /right/gripper_controller/calibrate
ros2 service type /right/gripper_controller/open
ros2 service type /right/gripper_controller/close
ros2 service type /right/gripper_controller/stop
ros2 topic type /right/gripper_controller/position_command
ros2 topic type /right/gripper_controller/position
ros2 service list | grep -E '^/lbr/(compute_ik|plan_kinematic_path)$'
ros2 action list | grep '^/lbr/execute_trajectory$'
ros2 topic list | grep -E '^/(lbr/joint_states|perception/fp/pose_base/fused/assembly)$'
```

All eight gripper services must print `std_srvs/srv/Trigger`; the command and
feedback topics must print `std_msgs/msg/Float64`. Every `grep` must return its
requested interface. Then run:

```bash
./run_pipeline.sh --mode real --robots left --grasp-only --execute
```

Review the printed grasp target and type `yes` only when the workspace is
clear. The current KUKA config positions the gripper to the selected approach
opening, moves through pregrasp and grasp, closes through the endpoint service,
lifts `0.08 m`, and stops while holding the object. It retains confirmation and
`0.05` velocity/acceleration scaling.

### Single-arm D405 policy pickup on `lbr_one` (`192.170.10.2`)

Eight PPO checkpoints are installed locally under `.cache/d405_policy_deployment/`
and selected through `configs/d405_policy_registry.yaml`. The four legacy
Y-gripper policies retain their recorded validation scores. The four current
PDZ policies use the final epoch-3810 checkpoints; held-out selection is still
marked pending in the registry rather than being presented as validated. List
them without touching ROS or the robot:

```bash
./run_pipeline.sh --list-policies
```

Build the updated hardware package once, then start only the `.10`-subnet arm
as the physical member of the shared dual scene. `--servo` starts MoveIt Servo but
does not activate it until the pickup process reaches the policy handoff. Stop
`start_dual_lbr_moveit.sh` first: the standalone and dual control stacks cannot
own the same `lbr_one` FRI UDP connection at the same time.

```bash
source /opt/ros/humble/setup.bash
source /home/pdz/lbr-stack/install/setup.bash
cd ros2_ws
colcon build --packages-select robot_integration_ros --symlink-install
cd ..

./run_pipeline.sh --mode real --robots left --policy velocity-rotation --bringup-only --rviz
```

This launch uses only FRI peer `192.170.10.2`; `lbr_two` remains a mock
collision participant and does not claim a physical FRI connection.
The robot-side Python environment also needs PyTorch and torchvision. The
deployment actor is self-contained and does not import the RL-Games training
framework:

```bash
python3 -m pip install -e '.[deployment]'
```

Start the calibrated `realsense_1` D405 stack. The single-arm profile consumes
JPEG-compressed rectified color plus lossless `compressedDepth` aligned depth;
both transports preserve the source timestamps used by the synchronizer. The
runtime decodes them directly without `cv_bridge`. A live comparison on this
cell measured about `1.7 ms` median RGB decode and `4.0 ms` median depth decode;
compressed transport reduced median arrival age from roughly `604 ms` to
`237 ms` for RGB and from `297 ms` to `243 ms` for depth. The generated profile
therefore allows at most `0.50 s` image age and still holds Servo on longer
stalls. Between policy frames, the runtime republishes only the last
safety-approved twist at the Servo watchdog rate; it stops refreshing that
command as soon as the original RGB-D source timestamp exceeds the same
`0.50 s` bound.

The required robot feedback is configured in
`configs/visual_servo_real_d405.yaml`: the current `realsense_1` camera serial
`260522275434`,
`/lbr/joint_states`, the live `lbr_link_0 -> pdz_gripper_tcp` TF, and live TF from
the camera optical frame into `lbr_link_0`. The runtime derives TCP pose and
speed from the MoveIt robot TF tree; it no longer depends on the external
`/left/ee_pose` topic. Real motion also consumes the estimated-wrench output at
`/lbr/force_torque_broadcaster/wrench`; the single-arm hardware launch starts
that broadcaster automatically.

Application-level deadman and emergency-stop Bool topics are optional. The
single-pick wrapper leaves them disabled because this cell uses its reviewed
hardware safety chain. If physical operator controls are later connected, set
`require_deadman: true` and configure both topics as fresh heartbeats. Do not
use synthetic always-true/always-false `ros2 topic pub` processes as substitutes
for physical controls during real motion.

The wrapper does not contain stored grasp targets or stored policy goal
images. It runs the ordinary stage-1/stage-2 pipeline for the perceived part,
then passes every accepted stage-2 grasp to the real executor in score order.
List every installed policy and its embodiment/context with:

```bash
./run_pipeline.sh --list-policies
```

The current PDZ-trained policies are `baseline`, `background`, `velocity`, and
`velocity-rotation`. Run one guarded pickup with, for example:

```bash
./run_pipeline.sh --mode real --robots left --grasp-only --policy velocity-rotation --execute
```

Override the complete RGB-D source when the physical camera is currently under
the other RealSense namespace:

```bash
./run_pipeline.sh --mode real --robots left --grasp-only \
  --policy velocity-rotation --left-camera realsense_2 --execute
```

This switches compressed color, aligned compressed depth, both `CameraInfo`
topics, the camera parameter node, and the serial provenance hint together.

Part `0` is the default. Select another part with:

```bash
./run_pipeline.sh --workflow single-object --mode real --robots left \
  --policy velocity-rotation --part-id 2
```

The command starts perception, planning, runtime rendering, and the real
executor; the existing typed `yes` confirmation is still required before any
arm or gripper motion. To validate policy assets and write the resolved YAML
without starting perception, MoveIt, Isaac, or hardware, use:

```bash
./run_pipeline.sh --workflow single-object --mode real --robots left \
  --policy velocity-rotation --part-id 2 --generate-config-only
```

For each accepted grasp, the executor first plans a collision-aware MoveIt path
to its automatic 10 cm pregrasp. It then solves collision-aware grasp IK seeded
from that retained pregrasp trajectory. A failure rejects only that grasp and
the executor tries the next stage-2 grasp. Thus the first grasp used is the
highest live-score grasp for which both operations succeed; there is no policy
compatibility whitelist and no branch-specific target lookup.

The new physical gripper is calibrated from 7 mm fully closed to 74 mm fully
open. The single-arm planner rejects contact grasps outside that interval. It
requests the contact width plus 10 mm total approach clearance and clamps the
approach opening to 74 mm. The 84 mm aperture in the KUKA mesh is only the
source geometry/model limit and is no longer treated as reachable hardware;
for example, an 80.037 mm grasp is now rejected before MoveIt execution.

Candidate-specific approach openings for single-arm `lbr_one` use
`/left/gripper_controller/position_command`: `0.0` is 74 mm open and `1.0` is 7 mm
closed, with linear interpolation between them. The final grasp uses
`/left/gripper_controller/close` so the controller can stop on object contact. The
matching `/left/gripper_controller/position` feedback is required before arm motion
continues from an arbitrary approach command.

Only after MoveIt accepts a grasp does the wrapper launch the same MuJoCo
Filament goal-rendering path used by the RL branch's current training data. It
rebuilds the current stage-2 bundle-local part mesh, places it at the live
perceived pose, imports the configured robot URDF, sets the exact MoveIt grasp
joints and approach aperture, and renders RGB plus aligned metric D405 depth
from the calibrated wrist-camera pose. Because MuJoCo imports the MoveIt URDF
joint axes directly, no Isaac A4 sign conversion is applied. The renderer
validates the resulting TCP pose, camera/observation profiles, and depth
variation, then writes one per-run `policy_goal_<grasp-id>.npz`. That file is
the policy goal for this attempt; it is generated on demand and is not reused
to select a grasp.

The renderer launcher and the complete pickup pipeline each run inside a
private process group. Success, failure, Ctrl-C, and timeout all terminate and
reap that group, so renderer descendants cannot remain after the pickup
command exits. The old Isaac/Kit launcher is not used by this entrypoint.

The live left-arm wrist stream is D405 `realsense_1`, currently serial
`260522275434` after the physical `realsense_1`/`realsense_2` swap.
Small calibration differences between its rectified `CameraInfo` and the
training intrinsics are reported as warnings and the policy continues with the
live projection. Color and aligned depth must still use the same projection
grid; a disagreement there remains a hard failure because the RGB-D pixels
would no longer correspond.
The serial query is retained only in the run artifact for provenance. A
missing parameter service or future serial mismatch warns and continues;
camera routing is defined by the `realsense_1` topics, live `CameraInfo`, and
TF chain.

Before stage 2, the perceived object pose is floor-settled by translating it
in Z until the rotated mesh's lowest vertex lies on the configured table plane.
This corrects estimated table penetration or hovering for collision checks and
the runtime goal render. It deliberately does not invent corrections for XY or
orientation errors in perception.

A focused `policy_execution_debug.html` opens before confirmation and is
refreshed after MoveIt selection and rendering. It shows the actual selected
grasp/pregrasp together with the newly rendered goal RGB and depth images. The
generic `part_frame.html` remains a secondary artifact. Pass
`--no-planning-debug-gui` to suppress browser tabs while still writing both
HTML artifacts.

Execution is: normal MoveIt IK/path planning to pregrasp, policy twists from
pregrasp until the learned four-frame completion gate returns, Servo stop,
gripper close, and a normal MoveIt lift. The collision-aware pregrasp
trajectory is planned before typed confirmation and retained; after confirmation
the executor opens the physical gripper and executes that exact accepted plan
instead of replanning after the first hardware command. MoveIt Servo consumes the same
`move_group` planning scene with collision checking enabled, 10 mm self and
20 mm scene proximity thresholds. A stale sensor/operator stream, collision
halt, missing command consumer, timeout, joint/workspace violation, or policy
failure commands zero motion and prevents gripper closure. The standard typed
`yes` confirmation remains required before the initial gripper/pregrasp motion.
The selected checkpoint is strict-loaded after the runtime goal is rendered
and before confirmation or any gripper/arm command. An incompatible checkpoint,
unavailable CUDA device, or broken CUDA/cuDNN runtime therefore still fails
before hardware motion.
The pregrasp is fully automatic: no separate pregrasp command is needed. The
persistent bridge supplies MoveIt's passive
`pdz_gripper_left_finger_joint` from normalized physical feedback and keeps
publishing the final measured position after motion stops. Candidate planning
still applies its exact selected approach/contact width. Planning-scene diffs
are idempotent and retry once if the first Fast DDS service response is lost;
the initial attempt is bounded at two seconds instead of waiting the full scene
timeout. The pregrasp-to-grasp segment is deliberately generated online by the
visual policy through MoveIt Servo.
Before confirmation or any hardware command, MoveIt tries all surviving
stage-2 candidates in live score order. An IK or path-planning failure rejects
only that candidate; the first collision-aware pregrasp plan is retained and
executed after confirmation. The focused debug page is rewritten for that
actually planned candidate, including its world-frame grasp/pregrasp view and
the goal RGB/depth images rendered for that run.

The ROS2 package also exposes `fp_debug_msgs/action/GraspAssembly`. Start its
adapter through the same public pipeline boundary:

```bash
source ./setup_robot_env.sh
./run_pipeline.sh --mode real --robots both --serve-action --execute
```

Then send a dual task goal:

```bash
ros2 action send_goal --feedback \
  /grasp_assembly \
  fp_debug_msgs/action/GraspAssembly \
  "{assembly_name: plumbers_block, base_part_id: 2, insertion_part_id: 0, holder_robot: left, inserter_robot: right}"
```

The action adapter consumes both perception poses and delegates the task to
`run_pipeline.sh`; it does not contain a second real executor. For
perception-in-the-loop Isaac, use:

```bash
./run_pipeline.sh --mode pitl --robots both --serve-action --headless
```

For one active inserter that stops after lift and uses a policy:

```bash
./run_pipeline.sh --mode real --robots left --serve-action --grasp-only \
  --policy velocity-rotation --execute
```

The default dual action reaches pre-insertion but does not perform insertion,
release, or retreat. `--grasp-only` ends at pickup lift with no transport.

Manual gripper commands from any correctly configured ROS2 terminal:

```bash
# Release the object.
ros2 service call /left/gripper_controller/open std_srvs/srv/Trigger "{}"

# Close toward the 7 mm endpoint; motor contact may stop it earlier.
ros2 service call /left/gripper_controller/close std_srvs/srv/Trigger "{}"

# Command an arbitrary closure fraction (0=open, 1=closed).
ros2 topic pub --once /left/gripper_controller/position_command \
  std_msgs/msg/Float64 "{data: 0.7}"

# Stop the gripper motor immediately.
ros2 service call /left/gripper_controller/stop std_srvs/srv/Trigger "{}"
```

Stop the pose publisher after the pipeline reports that it received the pose. Shut down the pipeline first, then MoveIt, the hardware launch, and finally the SmartPAD FRI application.

#### Why The ROS Environment Must Match

ROS2 domain and RMW selection happen when each process starts. `ROS_DOMAIN_ID` separates independent DDS graphs, so domain `42` cannot discover domain `0`. `ROS_LOCALHOST_ONLY=1` prevents discovery across `.1` and `.3`. DDS discovery also has to select the correct network interface on a multihomed robot computer.

Using the same middleware implementation is not a fundamental ROS2 requirement: Fast DDS and Cyclone DDS both implement DDS/RTPS and can interoperate in principle. In this setup, however, the gripper was running Cyclone DDS while the pipeline computer had only the Fast DDS RMW installed, and remote service discovery did not succeed reliably. Domain and RMW settings were corrected together, so the session did not prove that the vendor mismatch alone caused the failure. Standardizing both hosts on the already-installed `rmw_fastrtps_cpp` removes that variable and its vendor-specific discovery and interface differences.

The alternatives are to install Cyclone DDS RMW on `.1` and run every process with Cyclone DDS, or deliberately configure mixed-vendor discovery, multicast routing, interfaces, and firewall rules. A Fast DDS discovery server can replace multicast when the network blocks it, but every participant still needs a consistent discovery-server configuration. For this two-computer ROS graph, one shared environment in launch wrappers or system services is simpler than exporting variables interactively and does not require changing middleware on every run.

Run Isaac-backed sim locally:

```bash
./run_pipeline.sh --mode sim
```

If you want the MuJoCo backend instead, bootstrap its generated robot XML first:

```bash
bash scripts/download_required_assets.sh
```

## Development And Verification

Run the same Python checks used by CI:

```bash
python3 -m pip install -e ".[dev,test]"
pre-commit run --all-files
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

The FCL-backed collision tests require the native `libccd`/`libfcl` packages
and `python-fcl`. ROS2-independent tests use mocks where possible; the dual
MoveIt smoke test is separate because it requires a live shared MoveIt stack:

```bash
./start_dual_lbr_moveit.sh --mode mock
# In a second sourced terminal:
python3 scripts/smoke_test_dual_lbr_moveit.py
```

## Repo Shape

The main source areas are:

- `run_pipeline.sh` - the one public dual/single, bringup, action, policy, and benchmark entrypoint;
- `setup_robot_env.sh` - the one public sourced ROS environment;
- `run_simple_dual_robot.sh` - compatibility shim to the unified dual workflow;
- `start_lbr_moveit.sh` and `start_dual_lbr_moveit.sh` - single/shared MoveIt
  launchers;
- `configs/` - single-object, benchmark, backend, and dual-arm YAML settings;
- `grasp_planning/grasping/` - mesh loading, antipodal generation, scoring,
  transforms, and gripper collision geometry;
- `grasp_planning/pipeline/` - single-object stages, regrasp planning, dual
  assembly stages, pair scoring, and transition symmetry;
- `grasp_planning/planning/` - backend-neutral execution data and helpers;
- `grasp_planning/mujoco/` and `grasp_planning/envs/` - MuJoCo and Isaac
  execution support;
- `grasp_planning/ros2/` - perception, MoveIt, gripper, multi-IK, and guarded
  real execution adapters;
- `scripts/` - build, run, benchmark, debug, and model-generation commands;
- `ros2_ws/src/robot_integration_ros/` - ROS2 package, launch descriptions,
  MoveIt configuration, and action-server entrypoints;
- `assets/obj/fabrica/` - per-assembly OBJ meshes, precedence plans,
  pre-insertion poses, and finite symmetry assets;
- `assets/urdf/` and `assets/usd/` - robot and gripper models; and
- `tests/` - unit/regression coverage, including ROS-independent MoveIt mocks.

Deep dual-arm implementation notes live in
`DUAL_ROBOT_HOLD_GRASPING_PLAN.md`; the shorter
`DUAL_ROBOT_TRANSITION_SYMMETRY_PLAN.md` is the authoritative explanation of
the two symmetry boundaries. `KUKA_dual_arm_CHEATSHEET.md` is an operational
lab reference, not the architecture source of truth.

## Notes

- The default Fabrica OBJ scale in the pipeline configs is `0.01`.
- The MuJoCo runner uses the exact `execution_world_pose` stored in the stage-2 bundle unless you override placement explicitly.
- `pitl` and `real` use one ROS2 subscriber: set `ros2.pose_base_topic`, `ros2.assembly_name`, and numeric `ros2.part_id` in the pipeline YAML before running those modes. The configured topic must publish `fp_debug_msgs/msg/DebugPoseItem`. Optional `ros2.position_offset_m: [x, y, z]` adds a fixed world-axis translation to the received pose before planning; the standalone `lbr_one` real config uses `[0.0, 0.420, 0.0]` to convert shared dual-cell `base_link` coordinates into `lbr_link_0` coordinates.
