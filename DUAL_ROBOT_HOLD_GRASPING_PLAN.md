# Dual-Robot Hold-Grasping Implementation Plan

Status: Stages 0 through 3, Stage 4A, the shared dual-arm MoveIt foundation,
and selected-order per-step exact-IK, Isaac, and guarded real pre-insertion
episodes implemented; full object-aware Stage 5 trajectories, insertion
execution, and the generalized
Stage 7 coordinator proposed

## Implementation Progress

Stage 0 is implemented by:

- [`grasp_planning/pipeline/assembly_sequence.py`](grasp_planning/pipeline/assembly_sequence.py)
  - validates and compiles `forward_assembly_orders[0]`;
  - records explicit before/after states, insertion transforms, base
    availability, table contact, source-asset metadata, and warnings;
- [`grasp_planning/pipeline/assembly_sequence_debug_html.py`](grasp_planning/pipeline/assembly_sequence_debug_html.py)
  - writes the self-contained step/insertion viewer;
- [`scripts/build_assembly_sequence.py`](scripts/build_assembly_sequence.py)
  - writes `assembly_sequence.json` and `assembly_sequence.html`;
- [`tests/test_assembly_sequence.py`](tests/test_assembly_sequence.py)
  - covers synthetic validation, artifact controls, and the real
    `plumbers_block` sequence with selected-order base part `2`.

Generate the current vertical-slice artifacts with:

```bash
python3 scripts/build_assembly_sequence.py \
  --assembly plumbers_block
```

The generated files are written under
`artifacts/dual_grasp_planning/plumbers_block/`. For this assembly, Stage 0
records selected order `2 -> 0 -> 3 -> 1 -> 4`, chooses the first selected part
`2` as the base, and begins base-only holder availability at step index `1`.
Part `2` is also the asset mesh touching the configured `z=0` table plane.
An explicit base override remains available for exceptional assemblies and is
recorded as `base_part_source: explicit_override`.

Stage 1 is implemented by:

- [`grasp_planning/pipeline/holder_grasp_library.py`](grasp_planning/pipeline/holder_grasp_library.py)
  - reuses the existing Stage-1 antipodal generator, scoring, KUKA Y-gripper
    object collision checks, and cache;
  - deliberately disables changing-assembly collision checks so the generated
    base contacts remain a reusable input to Stage 2;
  - assigns stable holder IDs, records the base/assembly frame contract, and
    preserves contacts, normals, jaw width, roll, offsets, score components,
    candidate metadata, and cache identity;
- [`grasp_planning/pipeline/holder_grasp_debug_html.py`](grasp_planning/pipeline/holder_grasp_debug_html.py)
  - embeds all holder candidates once and shares the KUKA collision meshes
    across candidates, keeping the real 1,241-candidate HTML usable;
  - provides candidate search, score filtering, next/previous navigation,
    all-candidate overlay, mesh rendering controls, contact normals, grasp
    axes, and a debug pregrasp;
- [`scripts/build_holder_grasp_library.py`](scripts/build_holder_grasp_library.py)
  - builds `holder_base_candidates.json` and
    `holder_base_candidates.html`;
- [`configs/dual_grasp_planning.yaml`](configs/dual_grasp_planning.yaml)
  - selects `plumbers_block` and the current KUKA generation settings; the base
    defaults from the selected order rather than duplicating it in YAML;
- [`tests/test_holder_grasp_library.py`](tests/test_holder_grasp_library.py)
  - covers the state-independent reuse boundary, field preservation, required
    gripper model, artifact round trip, compact viewer, and CLI/config path.

Generate both implemented stages with:

```bash
python3 scripts/build_assembly_sequence.py \
  --assembly plumbers_block
python3 scripts/build_holder_grasp_library.py
```

For the selected-order base part `2`, Stage 1 currently saves 1,241 candidates.
Their jaw widths range from approximately `0.02006 m` to `0.06600 m`; all saved
contact points were numerically checked against the base surface with maximum
distance below `1.9e-12 m`. The Stage-1 artifact remains intentionally
unfiltered by assembly state, table, and insertion sweep; Stage 2 applies those
geometric filters without changing the reusable contact library.

Stage 2 is implemented by:

- [`grasp_planning/pipeline/holder_state_feasibility.py`](grasp_planning/pipeline/holder_state_feasibility.py)
  - evaluates the reusable holder library against every selected-order state;
  - checks the table, static assembled prefix, holder pregrasp and linear
    approach, and the incoming part's complete linear insertion sweep;
  - separates static and linearly moving obstacle specifications and records
    deterministic, machine-readable rejection reasons;
- [`grasp_planning/pipeline/holder_state_debug_html.py`](grasp_planning/pipeline/holder_state_debug_html.py)
  - writes one self-contained debugger that combines a candidate-by-step
    validity matrix with a step, insertion-progress, candidate, and rejection
    reason explorer;
  - highlights the table or part responsible for a rejection and draws the
    selected holder grasp, its pregrasp, and the incoming part;
- [`scripts/build_holder_state_feasibility.py`](scripts/build_holder_state_feasibility.py)
  - reuses the cached Stage-1 library and writes the Stage-2 JSON, matrix HTML,
    and one HTML entrypoint per sequence step;
- [`tests/test_holder_state_feasibility.py`](tests/test_holder_state_feasibility.py)
  - covers base availability, table collision and clearance, static assembly
    collision, holder-approach collision, and a midpoint-only incoming sweep
    collision that endpoint checks would miss.

Generate all three implemented stages with:

```bash
python3 scripts/build_assembly_sequence.py --assembly plumbers_block
python3 scripts/build_holder_grasp_library.py
python3 scripts/build_holder_state_feasibility.py
```

For `plumbers_block`, holder availability begins when part `0` is inserted.
Stage 2 accepts `154`, `95`, `82`, and `69` candidates for the four holder-active
steps, respectively. The rejected candidates include table collisions,
configured table-clearance failures, collisions with the assembled prefix, and
incoming insertion-sweep collisions. Several incoming-part failures occur only
at intermediate insertion progress, confirming that the full sweep changes the
result compared with endpoint-only checking.

Stage 3 is implemented by:

- [`grasp_planning/pipeline/dual_grasp_pair_planner.py`](grasp_planning/pipeline/dual_grasp_pair_planner.py)
  - reuses the existing assembly/insertion-filtered KUKA grasp generator for
    each incoming part, then adds table and post-insertion retreat checks;
  - diversity-shortlists holder and inserter candidates by contact region,
    grasp pose, and jaw/approach axes before Cartesian pairing;
  - enumerates pairs best-first, proves clearly separated pairs safe with an
    AABB distance lower bound, caches one swept-geometry FCL manager per
    inserter, and caps total pair evaluation deterministically;
  - ranks compatible pairs by holder score, inserter score, and clearance, then
    retains alternatives with per-holder and per-inserter caps;
- [`grasp_planning/pipeline/dual_grasp_pair_debug_html.py`](grasp_planning/pipeline/dual_grasp_pair_debug_html.py)
  - writes a per-step interactive holder-by-inserter compatibility matrix and
    combined table/assembly/part/two-gripper scene;
  - supports insertion and retreat scrubbing, compatible/rejected filters,
    sampled collision diagnostics, and clickable matrix cells;
- [`scripts/build_dual_grasp_pairs.py`](scripts/build_dual_grasp_pairs.py)
  - builds or loads the previous-stage inputs, republishes the exact Stage-1
    holder bundle used by Stage 2/3, writes one inserter bundle and pair
    JSON/HTML per holder-active step, and writes master JSON/HTML summaries;
- [`tests/test_dual_grasp_pair_planner.py`](tests/test_dual_grasp_pair_planner.py)
  - covers midpoint-only gripper collision, AABB-proven compatibility,
    clearance rejection, deterministic limits, KUKA model enforcement, table
    filtering, artifact references, and debugger controls.

Generate all four implemented stages with:

```bash
python3 scripts/build_assembly_sequence.py --assembly plumbers_block
python3 scripts/build_holder_grasp_library.py
python3 scripts/build_holder_state_feasibility.py
python3 scripts/build_dual_grasp_pairs.py
```

Stage 3 first retains a bounded, diverse pair pool for expensive transformed
corridor checks. It covers distinct inserter grasps before assigning second
pairs to one inserter, because runtime pickup-floor feasibility changes with
the detected world orientation. It then flattens every accepted
`pair_id + transition_id` into a complete execution candidate. A second bounded retention pass round-robins
those candidates by normalized pre-to-final corridor direction and by pair, so
both insertion sides survive whenever collision-valid alternatives exist.
Non-retained compatible pairs remain canonical identity-only records. At
runtime, pose-feasible candidates are partitioned into a strict clear phase and
a crossed fallback phase. Retained executions lead within each phase, followed
by exact non-identity transitions that were explicitly validated for a retained
pair but fell beyond the execution-retention cap. Already pair-checked canonical
records fill the remaining bounded fallback budget. A non-identity transform is
never inferred for an identity-only pair. Pair artifacts
reference the holder feasibility artifact and per-step inserter bundle by
candidate ID rather than duplicating grasp serialization. Runtime lookup honors
those declared sources because generated sequential IDs are scoped to one
specific library and can name different poses after a rebuild.

Stage 4A is implemented by:

- [`grasp_planning/pipeline/dual_robot_pair_scoring.py`](grasp_planning/pipeline/dual_robot_pair_scoring.py)
  - adds a deliberately approximate, frame-aware workspace score for both
    robots;
  - scores every required holder and inserter TCP phase using reach distance,
    height, front-of-base placement, and approach alignment;
  - combines the weakest and mean target scores so one bad phase cannot be
    hidden by several easy poses;
  - adds arm-ownership and top-down non-crossing layout preferences, then
    combines these with the Stage-3 geometric pair score;
- [`grasp_planning/pipeline/dual_robot_pair_score_debug_html.py`](grasp_planning/pipeline/dual_robot_pair_score_debug_html.py)
  - loads all Stage-3 compatible pairs and their referenced grasp artifacts;
  - writes a self-contained top-down cell debugger with live pair reranking;
  - makes the holder robot, inserter robot, rigid assembly, and incoming-part
    pickup source movable by drag or numeric XYZ/yaw controls;
  - defaults physical Robot 1 at negative Y to `H` holder and Robot 2 at
    positive Y to `I` inserter, while allowing their roles to be swapped
    explicitly;
  - shows the selected real KUKA gripper meshes in an orbitable close-up with
    final, approach, and retreat poses;
  - can automatically follow the highest-scoring pair after layout changes,
    or quickly cycle through the top 30 pairs with previous/next controls;
- [`scripts/build_dual_robot_pair_score_debug.py`](scripts/build_dual_robot_pair_score_debug.py)
  - defaults the holder to `Y=-0.420 m` and inserter to `Y=+0.420 m`,
    side by side and both facing world `+X`, with an optional locked `0.840 m`
    Y offset;
- [`tests/test_dual_robot_pair_scoring.py`](tests/test_dual_robot_pair_scoring.py)
  - covers reach-shell gating, front placement, target ownership, frame
    transforms, the 840 mm default, config validation, and HTML controls.

Generate the movable-cell ranking debugger after Stage 3 with:

```bash
python3 scripts/build_dual_robot_pair_score_debug.py
```

The generated
`artifacts/dual_grasp_planning/plumbers_block/dual_robot_pair_score_debug.html`
starts by ranking the retained Stage-3 pair set for each step. Its diagnostic
retained-only filter can be disabled to inspect every checked-compatible pair.
Runtime planning prioritizes the transition-complete
`retained_execution_candidates` list, then adds explicitly pair-validated
transition overflow and canonical identity-only pairs until its configured
queue bound is reached.
The offline/reachability/layout weights are adjustable, the selected pair shows
per-phase diagnostics and grasp geometry, and the current layout can be copied
as JSON. Manual pair selection or cycling disables automatic rank-1 following
so the selected alternative remains visible.

Stage 4A is only an attempt-ordering heuristic. It does not make an IK,
joint-limit, singularity, robot-link collision, or trajectory-feasibility
claim. Stage 4B below remains required to turn a high-scoring pair into an
accepted robot solution.

The shared dual-arm MoveIt prerequisite for Stages 4B and 5 is implemented by:

- [`scripts/build_kuka_moveit_description.py`](scripts/build_kuka_moveit_description.py)
  - deterministically generates both the existing single-arm description and
    one `lbr_dual_arm` model containing two fully prefixed hardware-canonical
    iiwa7 chains;
  - instantiates the calibrated Y-gripper independently as
    `lbr_one_gripper_tcp` and `lbr_two_gripper_tcp`;
  - places the bases at `Y=-0.420 m` and `Y=+0.420 m` under a common
    `base_link`, with identical zero base rotations;
- [`ros2_ws/src/robot_integration_ros/launch/dual_aligned_lbr_moveit.launch.py`](ros2_ws/src/robot_integration_ros/launch/dual_aligned_lbr_moveit.launch.py)
  - launches one robot-state publisher, one shared planning scene, one
    MoveGroup, two LBR hardware interfaces, and one independent trajectory
    controller per arm;
  - provides `arm_one`, `arm_two`, and `both_arms` SRDF groups while leaving
    all cross-arm collision pairs enabled;
  - bridges each persistent controller's normalized position feedback into the
    corresponding passive gripper driver joint, republishes the last measured
    value continuously, and uses a warned fully-open fallback before the first
    sample so MoveIt receives a complete 16-joint state;
- [`start_dual_lbr_moveit.sh`](start_dual_lbr_moveit.sh)
  - starts the shared model in mock or hardware mode and optionally opens a
    dual-arm RViz configuration;
  - RViz uses one stable `both_arms` MotionPlanning display and exposes one
    interactive goal marker for each gripper TCP. Two MotionPlanning displays
    are not used because they share the same marker namespace and crash RViz;
- [`scripts/smoke_test_dual_lbr_moveit.py`](scripts/smoke_test_dual_lbr_moveit.py)
  - verifies FK, IK, and a small Cartesian plan for each gripper TCP;
  - checks that a known overlapping configuration is rejected specifically
    for contacts between the two grippers;
  - can execute the small motions in mock mode to verify per-arm controller
    routing.

Build and inspect this foundation with:

```bash
cd ros2_ws
colcon build --packages-select robot_integration_ros --symlink-install
cd ..
./start_dual_lbr_moveit.sh --mode mock --rviz
```

With the stack running, verify it from another sourced terminal with:

```bash
python3 scripts/smoke_test_dual_lbr_moveit.py
python3 scripts/smoke_test_dual_lbr_moveit.py --execute  # mock mode only
```

This foundation makes full arm-arm collision checking possible in one MoveIt
scene. The first-step real vertical slice now coordinates two independently
namespaced Y-gripper Trigger-service endpoints; general action-server and
multi-step coordination remain future work.

The first exact-IK and simulation vertical slice is implemented by:

- [`grasp_planning/pipeline/dual_robot_simple_sim.py`](grasp_planning/pipeline/dual_robot_simple_sim.py)
  - resolves one accepted Stage-3 pair into world-frame holder, pickup, lift,
    transport, and pre-insertion targets for `step_001_part_0`;
  - places base part `2` at the centered assembly pose and part `0` at a known
    pickup pose close to the inserter;
  - derives the pickup root Z from the transformed source-local mesh minimum,
    so the incoming part's lowest point lies exactly on the configured floor;
- [`scripts/plan_simple_dual_robot_sim.py`](scripts/plan_simple_dual_robot_sim.py)
  - checks exact MoveIt IK and plans each holder/inserter phase in the shared
    two-arm scene;
  - resets both arms to the same shared KUKA start joint pose used by the
    single-arm Isaac path;
  - pre-plans a bounded set of complete inserter candidates from pickup through
    pre-insertion, caching shared pickup prefixes and enforcing a strict
    non-crossing phase before any crossed fallback, then ranking by pre-plan
    status and velocity-weighted joint-path cost within each phase;
  - seeds equivalent 180-degree transition solutions with bounded A7 `+pi` and
    `-pi` offsets, while allowing IK to adjust every joint and rejecting seeds
    outside the actual iiwa limits;
  - after a partially executed candidate fails, immediately retracts the
    inserter before resetting the holder so the holder's home path is not
    planned through an arm still occupying the transition corridor; recovery
    remains fatal if either arm cannot reach its known start state;
  - adds conservative world AABBs for transit to each pregrasp, then removes
    them before the corresponding grasp approach so intended contact is not
    rejected; exact object meshes remain omitted;
  - saves the exact per-arm MoveIt joint waypoints to
    `simple_dual_robot_sim_plan.json`;
- [`grasp_planning/pipeline/dual_robot_planning_debug.py`](grasp_planning/pipeline/dual_robot_planning_debug.py)
  - serves a localhost-only live browser debugger during visible simulation
    and guarded real planning;
  - real task construction starts it before the actual-pose pickup-floor filter,
    preserving counts and the terminal rejection reason even when no task can
    be serialized, then hands the stable port to the real executor so one tab
    follows the full run;
  - renders the partial assembly, incoming part, and selected holder/inserter
    grippers in the actual `base_link` world poses for the active candidate;
  - distinguishes holder grasp, incoming-part grasp, and transition stages and
    reports exact target phases, transition IDs, failures, resets, and fallback
    history;
  - reports separate counts for pickup-floor grasp filtering, retained Stage-3
    pair/transition candidates, the actual pose-filtered runtime queue's
    clear/crossed split and unique grasps, joint-space pre-ranking, and
    cumulative exact-IK screening;
  - reports whether the pickup or pre-insertion shoulder-to-target proxy
    crosses the holder corridor and renders pre-insertion during IK preflight;
  - reports exact displayed-mesh holder floor clearance and colors negative
    clearance red, avoiding judgments based only on the projected table edge;
- [`scripts/run_simple_dual_robot_sim_in_isaac.py`](scripts/run_simple_dual_robot_sim_in_isaac.py)
  - creates two KUKA/Y-gripper articulations and collision-enabled dynamic
    meshes for base part `2` and incoming part `0`;
  - places the Isaac ground plane at the task's configured support height
    (default `base_link z=-0.030 m`), matching MoveIt's work-surface top and
    both freshly generated object placements;
  - keeps the grounded incoming part in its known staging fixture until
    physical gripper contact, preventing the approach from pushing it away,
    then releases it for normal physics;
  - physically closes the holder on the base and the inserter on part `0`
    using the same effective 1 mm close command as `run_pipeline`; selected jaw
    width remains the primary contact check, with a fallback requiring
    bilateral role-filtered fingertip force against the intended object;
  - streams each saved MoveIt polyline as one continuous position/velocity
    reference instead of independently stopping at every planner waypoint;
    consecutive pregrasp/grasp and lift/transport/pre-insertion segments are
    grouped, with settling only at the two grasp actions and final
    pre-insertion;
  - uses separate `1.00 rad/s` unloaded and `0.70 rad/s` loaded defaults, a
    strict `0.005 rad` contact-pose tolerance, a `0.030 rad` transit tolerance,
    and a `2.0 s` final settling window;
  - records contact diagnostics, final grouped-transport part poses, base
    displacement, transport distance, and pre-insertion error in
    `simple_dual_robot_sim_attempt.json`;
- [`tests/test_dual_robot_simple_sim.py`](tests/test_dual_robot_simple_sim.py)
  - covers source/assembly/world transforms, support-plane alignment, and the
    boundary between pregrasp AABB transit checks and collision-enabled Isaac
    physics.
- [`grasp_planning/ros2/dual_real_grasp_executor.py`](grasp_planning/ros2/dual_real_grasp_executor.py)
  - consumes the same saved dual task targets but replans each phase from the
    live shared MoveIt state instead of replaying mock joint trajectories;
  - applies the table, uses the same temporary AABB lifecycle for pregrasp
    motion, preflights pairs in the producer's ranked queue order (clear
    corridors, retained Stage-3 executions, then score) before any motion,
    caches repeated holder/inserter grasp IK results, selects the first
    pair whose complete target set passes, requires explicit acknowledgement
    of the omitted exact object meshes, and records selection plus execution
    phases in a real-attempt artifact;
  - preflights both namespaced gripper Open/Close/Stop service sets before arm
    motion and best-effort stops both endpoints after a failure;
- [`scripts/run_simple_dual_robot_real.py`](scripts/run_simple_dual_robot_real.py)
  - defaults to a non-moving IK preflight;
  - requires `--execute`, `--allow-objectless-planning`, and typed
    confirmation for hardware motion, limits velocity and acceleration
    scaling to at most 20%, and supports a stop after every phase;
- [`scripts/build_simple_dual_robot_task.py`](scripts/build_simple_dual_robot_task.py)
  - resolves up to 256 transition-validated execution candidates and the
    current perceived pickup orientation/layout directly into a target-only
    hardware candidate artifact, without depending on previously saved mock
    trajectories;
- [`run_simple_dual_robot.sh`](run_simple_dual_robot.sh)
  - provides one entrypoint for this vertical slice in `sim` and `real` modes;
  - reuses the appropriate persistent mock or hardware MoveIt stack by default,
    generates a fresh plan or target task, and executes it; `--start-moveit`
    explicitly opts into a temporary owned stack;
  - uses an explicit CLI domain first, then
    `DUAL_ROBOT_ROS_DOMAIN_ID`, then the calling shell's `ROS_DOMAIN_ID`, and
    finally `0`, while enforcing matching Fast DDS discovery settings so
    MoveIt, the two arms, and the relevant execution process stay together;
- [`scripts/gripper_computer/dual_grippers.launch.py`](scripts/gripper_computer/dual_grippers.launch.py)
  - is the repository copy deployed to the gripper computer's
    `servo_gripper` package;
  - starts USB adapter `5B3D047592` as `/left/gripper_controller` for
    `lbr_one`, and `5B3D044069` as `/right/gripper_controller` for `lbr_two`,
    each with independent Calibrate/Open/Close/Stop Trigger services and
    closure-fraction position command/feedback topics;
- [`scripts/gripper_computer/start_dual_grippers.sh`](scripts/gripper_computer/start_dual_grippers.sh)
  - sources the remote ROS workspace and deliberately overrides the gripper
    computer's unrelated domain `42` with the shared default domain `0`;
  - clears discovery-server/profile overrides and selects Fast DDS over UDP;
- [`tests/test_dual_real_grasp_executor.py`](tests/test_dual_real_grasp_executor.py)
  - covers plan/role validation, exact phase routing, both gripper closes, and
    the conservative holder-pregrasp stop;
  - reproduces a rank-1 inserter-pregrasp IK failure and verifies that rank 2
    is selected without repeating an already-cached holder grasp check;
  - checks the shared ROS environment contract and unified runner routing.

Run planning and Isaac execution together from one command:

```bash
./run_simple_dual_robot.sh \
  --mode sim \
  --pair-id p001_h0450_i0_0422 \
  --pickup-x 0.55 \
  --pickup-y 0.28 \
  --headless
```

The same entrypoint resolves any holder-active selected-order step from the
assembly and incoming part:

```bash
./run_simple_dual_robot.sh \
  --mode sim \
  --assembly plumbers_block \
  --incoming-part-id 3 \
  --headless
```

The saved task records `assembled_part_ids_before` and the mesh path for every
part in that prefix. Isaac combines those final-coordinate meshes in the base
source frame and spawns them as one rigid compound subassembly. The holder
still contacts the selected-order base. This reuses the Stage-2/Stage-3
offline checks for the changing prefix while supporting independent,
reset-between-steps episodes that stop at pre-insertion.

For hardware, first run `./start_dual_grippers.sh` in the remote
`servo_gripper` workspace and start both SmartPAD `LBRServer` apps. Then use
the same local entrypoint with `--mode real`. With no `--execute` it performs a
non-moving live IK preflight; guarded staged motion is enabled with
`--execute --allow-objectless-planning --stop-after PHASE`.

The assembled prefix, incoming part, MoveIt work-surface top, and Isaac ground
now share the calibrated runtime support plane `base_link z=-0.030 m` by
default. Both grippers establish measured two-sided contact, and the dual
replay uses the same strong close semantics as the working single-arm
pipeline. This runtime calibration is separate from the Fabrica asset-frame
table plane at `z=0`. This is not a complete insertion: MoveIt uses only
temporary conservative AABBs for the two pregrasp transits, not exact object
meshes for all trajectory phases, and the test stops at pre-insertion without
insertion contact, release, or retreat.

## Purpose

Extend the Fabrica pipeline from single-part grasp generation to coordinated
two-robot assembly:

- the holder robot grasps the base or current subassembly to stabilize it;
- the inserter robot grasps and inserts the next part;
- both grasps and robot motions must be feasible at the same time;
- planning follows only the selected Fabrica assembly order,
  `forward_assembly_orders[0]`.

This plan deliberately separates geometry, pair compatibility, robot
kinematics, trajectories, simulation, and real execution so that each layer can
be tested and debugged independently.

## Confirmed Operating Assumptions

- The base and current assembly rest stably on a table throughout assembly.
- The holder may release and acquire a new grasp between insertion steps.
- A holder grasp that remains feasible for multiple steps is useful, but not
  required.
- The first part in `forward_assembly_orders[0]` is the base by default.
- The first implementation will generate holder contacts on that base part
  only. An explicit override is retained for exceptional assets.
- Allowing holder contacts on any already assembled part is a possible later
  extension.
- Both robots are expected to be KUKA iiwa7 robots with the same Y-gripper.
- The robot bases are approximately `0.840 m` apart in the Y direction.
- The exact six-degree-of-freedom transform between robot bases is not yet
  calibrated.
- No insertion-force bounds, part material data, or calibrated friction values
  are currently available.

The selected precedence order, not the README action example or the
pre-insertion role label, is authoritative for the default base.

## Intended Step Sequence

For assembly step `k`:

```text
assembly rests on table
  -> holder releases its previous grasp
  -> inserter moves to a safe/parked configuration
  -> holder moves to holder grasp h_k and closes
  -> inserter uses insertion grasp g_k
  -> inserter transports the part to the pre-insertion pose
  -> inserter follows the insertion path
  -> inserter releases and retreats
  -> holder may release
  -> proceed to assembly state k+1
```

No holder/inserter handover of the assembly is required.

## Existing Code to Reuse

- Assembly-order and pre-insertion asset loading:
  [`scripts/run_grasp_generation_benchmark.py`](scripts/run_grasp_generation_benchmark.py)
- Antipodal generation, detailed gripper collision geometry, scoring, and
  self-contained HTML rendering:
  [`grasp_planning/grasping/fabrica_grasp_debug.py`](grasp_planning/grasping/fabrica_grasp_debug.py)
- Stage-1 assembly filtering and cache patterns:
  [`grasp_planning/pipeline/fabrica_pipeline.py`](grasp_planning/pipeline/fabrica_pipeline.py)
- Bounded pair search and hand-hand collision-checking structure:
  [`grasp_planning/pipeline/handover_fallback.py`](grasp_planning/pipeline/handover_fallback.py)
- Stable-pose and multi-scene HTML patterns:
  [`grasp_planning/pipeline/regrasp_debug_html.py`](grasp_planning/pipeline/regrasp_debug_html.py)
- KUKA collision geometry:
  [`grasp_planning/grasping/collision.py`](grasp_planning/grasping/collision.py)
- MoveIt IK, planning, execution, and planning-scene client:
  [`ros2_ws/src/robot_integration_ros/robot_integration_ros/moveit_pose_commander.py`](ros2_ws/src/robot_integration_ros/robot_integration_ros/moveit_pose_commander.py)

The current `GraspAssembly` action remains single-robot and ignores holder/base
fields. It should not be expanded until the planning and simulation stages below
are validated.

## Stage 0: Assembly Sequence Compiler and Viewer

Status: implemented

### Goal

Turn the selected precedence order and pre-insertion assets into one explicit,
validated sequence of assembly states.

### Proposed model

For every step, record:

- assembly name;
- step ID and index;
- base-part ID derived from `selected_order[0]`, or an explicit override;
- already assembled part IDs;
- incoming part ID;
- asset paths and mesh scale;
- final assembled transforms;
- pre-insertion transform;
- pre-to-final insertion vector and distance;
- table-plane definition;
- source asset hashes or metadata needed for cache invalidation.

Only `forward_assembly_orders[0]` is used.

### Artifacts

- `assembly_sequence.json`
- `assembly_sequence.html`

### Visual debugger

The sequence HTML should contain:

- assembly-step slider;
- insertion-progress slider;
- base part highlighted separately;
- already assembled parts in gray or green;
- incoming part in orange;
- insertion direction and path;
- table plane;
- assembly and part frame axes;
- selected order, part IDs, and transforms in a details panel.

### Tests

- The selected order exactly matches `forward_assembly_orders[0]`.
- Each state prefix contains the expected parts.
- Each incoming part is absent from the current subassembly.
- Pre-insertion translations and insertion-vector signs are mutually
  consistent.
- Missing part meshes fail with a precise error.
- Unsupported rotational insertion transforms fail explicitly.
- All current Fabrica precedence assets compile.
- The generated HTML embeds the expected states, controls, and frame data.

### Completion gate

Do not continue until a human can use the HTML to verify one complete real
assembly sequence and its insertion directions.

## Stage 1: Reusable Base-Part Holder Candidate Library

Status: implemented

### Goal

Generate raw holder grasps on the configured base part once, independently of
the changing assembly state.

### Behavior

- Use the existing antipodal generator.
- Use the KUKA Y-gripper collision model.
- Generate candidates in a stable base/assembly-relative frame.
- Preserve contact points, normals, jaw width, roll, offsets, and detailed score
  components.
- Reject invalid collisions with the base itself using the existing generator
  behavior.
- Cache the raw holder library by base mesh, scale, gripper model, generator
  settings, scoring version, and random seed.

Do not regenerate base-surface contacts for every assembly step. Later stages
will determine in which states each candidate remains feasible.

### Artifacts

- `holder_base_candidates.json`
- `holder_base_candidates.html`

### Visual debugger

- Show all raw candidates on the base.
- Select a candidate by ID.
- Show detailed KUKA base and finger collision meshes.
- Show contact points, normals, jaw axis, approach axis, and pregrasp.
- Display jaw width and all score components.
- Filter candidates by status or score.

### Tests

- Fixed seeds produce deterministic candidate geometry and IDs.
- Contact points lie on the base mesh within tolerance.
- Jaw-width limits are enforced.
- KUKA contact offsets are preserved in the saved candidate.
- Synthetic box fixtures produce known accepted and rejected contacts.
- Cache identity changes when relevant mesh, gripper, or algorithm inputs
  change.
- Saved candidates round-trip through the artifact schema.

### Completion gate

The raw library must contain visually plausible, diverse holder contacts on the
chosen real base part before state-specific filtering begins.

## Stage 2: Per-State Holder Feasibility

Status: implemented

### Goal

Evaluate every raw base holder candidate against each actual assembly step.

### Checks

For holder candidate `h` at step `k`, check:

1. holder gripper versus already assembled parts other than its intended base
   contact;
2. holder gripper versus the table;
3. holder linear pregrasp-to-grasp approach versus the subassembly and table;
4. static holder gripper versus the incoming part over its full insertion
   sweep;
5. configured clearance margins around static and moving geometry.

### Required geometry change

The current `assembly_obstacle_sweep_vector_m` applies one sweep to all obstacle
meshes. Holder planning needs per-obstacle motion:

- current subassembly parts: static;
- table: static;
- holder gripper: static during insertion;
- incoming part: moving along its insertion path.

Introduce an explicit obstacle/motion specification rather than overloading the
single global sweep vector.

### Artifacts

- `holder_state_feasibility.json`
- `holder_state_<step-id>.html`
- `holder_validity_matrix.html`

The JSON should preserve accepted and rejected candidates with machine-readable
reason codes.

### Suggested reason codes

- `accepted`
- `base_collision`
- `assembled_part_collision`
- `table_collision`
- `holder_pregrasp_collision`
- `holder_approach_sweep_collision`
- `incoming_part_sweep_collision`
- `clearance_margin_failed`

### Visual debugger

- Candidate-by-step validity heatmap.
- Step and candidate selectors.
- Current subassembly and incoming-part swept volume.
- Selected holder gripper and pregrasp path.
- Collision geometry highlighted in red.
- First failing insertion-path position.
- Minimum-clearance location and value when available.
- Rejection-reason filters and counts.

### Tests

- Adding an occluding part rejects the expected holder candidate.
- An incoming part that collides only midway through insertion is detected.
- Endpoint-only collision checks cannot pass a midpoint collision fixture.
- Insertion sweep direction is correct.
- Table clearance and inflation margins behave predictably.
- Static and moving obstacle specifications are handled differently.
- Candidate ordering and reason counts are deterministic.

### Completion gate

At least one real assembly step must show both deliberately accepted and
deliberately rejected holder candidates with visually correct reasons.

Completed for `plumbers_block`: every holder-active state contains accepted and
rejected candidates, the final state retains 69 accepted holder grasps, and the
generated matrix/scene debugger was browser-rendered and visually inspected.
This is geometric holder feasibility only. Inserter-gripper compatibility,
dual-robot IK, robot-link collision, and coordinated motion remain in Stages 3
through 5.

## Stage 3: Holder/Inserter End-Effector Pair Planner

Status: implemented

### Goal

Produce bounded, ranked holder/inserter grasp combinations that are geometrically
compatible throughout insertion.

### Inputs

- Stage-2 holder candidates for the selected assembly step.
- Existing insertion grasps that survived assembly/insertion filtering.
- Actual KUKA gripper collision models for both robots.
- Incoming-part insertion path.

The offline pair graph may reference stage-1 insertion candidate IDs. Runtime
pickup feasibility can later intersect this graph with the candidates that
survive the actual stage-2 pickup pose.

### Pair checks

- Static holder gripper versus swept inserter gripper.
- Static holder gripper versus swept incoming-part mesh.
- Inserter gripper versus current subassembly.
- Both grippers at pre-insertion, final insertion, and retreat.
- Both grippers versus the table.
- Configured robust clearance margins.

This stage checks end effectors and task geometry, not complete robot arms.

### Search strategy

Avoid an unrestricted Cartesian product:

1. apply all unary holder and inserter filters;
2. cluster by contact region, jaw axis, approach axis, and wrist pose;
3. retain a configurable number of diverse candidates per side;
4. compute swept AABBs or another cheap broad phase;
5. run exact FCL checks only for potentially overlapping pairs;
6. enumerate pairs best-first using an upper-bound score;
7. stop evaluation at the configured pair-check budget and retain only the
   configured accepted-pair limit;
8. retain diversity so alternatives do not all share the same contact region.

The current configuration permits `16000` pair checks and retains up to `256`
complete pair/transition execution candidates. The larger retained set is
intentional: the cheap runtime pickup-floor filter depends on the perceived
part orientation, so it needs enough holder, inserter, and corridor diversity
left for exact IK fallback.

### Artifacts

- `dual_grasp_pairs_step_<step-id>.json`
- `dual_grasp_pairs_step_<step-id>.html`
- master per-step pair summary

The pair artifact should reference holder and insertion candidate IDs rather
than duplicate the existing insertion-grasp serialization.

### Visual debugger

Build an interactive compatibility matrix:

- rows: holder candidates;
- columns: insertion candidates;
- green: accepted;
- red: exact collision;
- gray: unary rejection;
- blue: not checked because of configured limits;
- click a cell to inspect the pair;
- animate or scrub insertion progress;
- highlight the first collision pose and colliding primitives;
- show unary scores, pair score, rejection reason, and clearance.

### Tests

- Known compatible and incompatible synthetic gripper pairs.
- Mid-sweep gripper collision with collision-free endpoints.
- Correct handling of moving part and moving inserter gripper.
- Correct KUKA rather than hard-coded Franka geometry.
- Broad-phase rejection agrees with exact checks on retained pairs.
- Pair limits are deterministic and reported in metadata.
- Accepted pairs are sorted deterministically.
- Candidate references resolve to their source bundles.

### Completion gate

One real assembly step must produce a visually verified list of compatible
holder/inserter pairs without invoking MoveIt.

Completed for all four holder-active `plumbers_block` steps with regenerated
schema-v3 artifacts. The steps retain 256, 208, 188, and 172 pairs respectively
and 256 complete pair/transition execution candidates each. Every shortlisted
Cartesian product fits below the configured 16000-pair check budget. The master
summary and detailed matrix/scene pages were generated. No MoveIt, robot IK,
robot-link collision, or trajectory claim is made at this stage.

## Stage 4: Layout-Aware Ranking and Individual Two-Robot IK

Status: Stage 4A implemented; one first-step Stage 4B exact-IK and planned
trajectory vertical slice implemented; generalized per-step diagnostics and
debugger proposed

### Stage 4A: Layout-aware attempt ordering

The current debugger uses the approximate `0.840 m` base separation to answer
the practical question “which already collision-compatible pair should we try
first for this object and cell placement?” It operates on movable world frames
and preserves Stage 3 as the geometric source of truth.

The ranking includes:

- Stage-3 holder/inserter/clearance score;
- holder pregrasp and grasp workspace proxies;
- inserter pre-insertion, final, and retreat workspace proxies;
- optional incoming-part pickup proxy;
- minimum-plus-mean aggregation across phases and across arms;
- target ownership by the intended side;
- a top-down crossed-arm penalty.

The UI can move the cell layout without regenerating grasps or collision pairs.
Moving the assembly is rigid: already assembled parts keep their relative
transforms. The pickup frame moves the current incoming part separately.

### Stage 4B: Exact individual two-robot IK

The common MoveIt robot model, per-arm IK groups, gripper TCPs, controllers,
mock launch, and collision-scene smoke tests are implemented. The simple
vertical slice consumes a retained-transition-first queue completed by safe
canonical identity fallbacks. It builds a configurable pre-plan prefix by
round-robin insertion corridor, then
evaluates that prefix with multi-seed IK and
joint-space path cost, then performs exact shared-scene IK/path/execution in
that order. Repeated pickup prefixes are cached, bounded A7 half-turn seeds are
included for symmetric opposite-side corridors, and valid A7 `+3.0/-3.0` rad
near-limit branches cover the case where literal pi exceeds the joint bound.
Every failure falls back to the next retained execution candidate. The
selected plan artifact records the pre-ranking diagnostics and exact executed
joint waypoints. Generalizing
that flow to every step with attached-object collision geometry and HTML
trajectory diagnostics remains.

#### Goal

Reject geometrically compatible pairs that either robot cannot reach.

#### Required frame contract

Define and calibrate:

```text
T_world_left_robot_base
T_world_right_robot_base
T_world_assembly
```

The approximate `0.840 m` Y separation may be used for early simulation, but it
is not sufficient for real acceptance. The existing perception position offset
is not a complete two-robot calibration.

#### Checks

- Holder pregrasp and grasp IK on the holder robot.
- Inserter pickup, transport, pre-insertion, final insertion, and retreat IK on
  the inserter robot.
- Multiple IK solutions where practical.
- Joint-limit margin.
- FK reconstruction error at each requested TCP pose.
- Simple manipulability or singularity proxy.

This stage still does not claim arm-arm trajectory feasibility.

#### Artifacts

- per-pair IK solutions and diagnostics;
- updated pair artifact with `ik_feasible` status;
- `dual_robot_ik_step_<step-id>.html`.

#### Visual debugger

- Both robot bases in the common world frame.
- Simplified link chains or collision meshes for the selected IK solutions.
- TCP and assembly frames.
- Joint-limit warnings.
- Reachable and unreachable pair filters.
- FK error and alternate IK solution selector.

#### Tests

- Mocked MoveIt success, timeout, missing-joint, and failure responses.
- Known reachable and unreachable poses.
- Correct holder/inserter namespace and joint-name routing.
- FK reconstructs each accepted target within tolerance.
- Integration smoke test against the simulated two-robot MoveIt setup.

#### Completion gate

At least one geometrically compatible real-asset pair must have independently
verified IK for every required phase.

## Stage 5: Full Trajectory and Arm-Arm Collision Validation

### Goal

Validate complete robot motion for both robots in a shared collision world.

### Planned scheduling

Because the table supports the assembly:

1. park the inserter;
2. plan holder release, pregrasp, grasp, and close;
3. freeze the holder configuration;
4. plan inserter pickup, transport, insertion, release, and retreat;
5. validate the entire sequence in one shared scene.

The core expensive check is the inserter trajectory while the holder remains
fixed.

### Required planning model

Prefer one shared two-robot collision model or combined MoveIt robot model.
Two unrelated planning scenes are insufficient for reliable arm-arm collision
checking.

Model:

- both complete KUKA arms and grippers;
- table and cell geometry;
- current subassembly;
- incoming part attached to the inserter where appropriate;
- holder touch links and intended base contact;
- current holder robot configuration;
- interpolated trajectory states between planner waypoints.

### Artifacts

- holder and inserter joint trajectories;
- start states and IK selections;
- shared-scene collision results;
- minimum-clearance timeline;
- plan duration and joint-motion cost;
- alternate planned trajectories;
- `dual_robot_trajectory_step_<step-id>.html`.

### Visual debugger

- Full two-robot trajectory-time slider.
- Both collision models and attached objects.
- Phase markers for holder approach, close, insertion, release, and retreat.
- Colliding links highlighted in red.
- Clearance graph over time.
- Planner waypoints versus interpolated validation samples.

### Tests

- Seeded arm-arm collision is rejected.
- A known collision-free two-robot configuration is accepted.
- Fixed holder links remain present in the inserter collision scene.
- Attached-part collision semantics are correct.
- Collision between recorded waypoints is caught by interpolation.
- Parked-inserter and parked-holder scheduling constraints are respected.
- Trajectory artifacts round-trip and retain their exact start states.

### Completion gate

One real-asset pair must pass shared-scene trajectory validation with both full
robots before physics simulation or hardware integration.

## Stage 6: Simulation

Status: first-step holder plus pickup-to-pre-insertion vertical slice
implemented; complete insertion, release, regrasp, and multi-step simulation
proposed

### Goal

Validate planned coordination before changing the real action server.

### Progression

1. kinematic two-robot trajectory playback;
2. collision-enabled playback;
3. incoming part attached to the inserter;
4. simulated holder gripper contact;
5. insertion contact and release;
6. several consecutive assembly states with holder regrasping.

Start with one middle assembly step rather than a full assembly.

### Measurements

- assembly displacement on the table;
- holder slip;
- unexpected contacts;
- insertion pose error;
- planner versus executed joint error;
- contact forces when the simulator exposes them;
- minimum robot and gripper clearances;
- execution phase timing.

### Artifacts

- per-attempt JSON;
- videos;
- contact and trajectory traces;
- selected and fallback pair IDs;
- final part and assembly poses.

### Tests

- Kinematic playback follows saved trajectories.
- Simulation loads the same candidate IDs and transforms as the planner.
- Holder remains closed and static during insertion.
- The assembly remains within configured displacement limits.
- Failure cases produce precise phase and contact diagnostics.
- A second holder grasp can be selected for the next table-supported state.

### Completion gate

At least one complete simulated insertion and one table-supported holder regrasp
must succeed with inspectable artifacts.

## Stage 7: Real Runtime Coordinator

Status: guarded selected-step holder/pickup-to-pre-insertion runner,
pre-motion ranked-pair IK fallback, and `GraspAssembly` action adapter
implemented for PITL/Isaac and real execution; insertion, release, retreat,
post-motion perception verification, post-motion fallback, role swapping, and
multi-step execution proposed

### Goal

Extend `GraspAssembly` only after the earlier stages have produced validated
pair and trajectory artifacts.

### Runtime behavior

- Read the requested assembly, base part, holder robot, and inserter robot.
- Resolve the selected assembly step.
- Recheck the assembly pose and relevant candidate transforms.
- Select a prevalidated holder/inserter pair.
- Execute the holder regrasp while the assembly remains on the table.
- Execute insertion while the holder remains fixed.
- Record actual outcomes and failure phase.

The implemented first slice uses the existing
`fp_debug_msgs/action/GraspAssembly` endpoint and all five goal fields. It
validates the selected-order base and incoming step, requires the currently
validated `left` holder (`lbr_one`) and `right` inserter (`lbr_two`), waits for
both fused `DebugPoseItem` poses, derives the planar assembly and pickup
transforms, and invokes the same dual planner/executor as the standalone
wrapper. Real mode embeds up to 256 candidates after rechecking grasps against
the actual perceived pickup pose and orientation: retained symmetry-validated
executions first, then other explicitly pair-validated transition corridors
and canonical identity-only compatible pairs. It checks them
lazily and starts hardware motion only after one candidate passes all target IK
checks. A fixed `--pair-id` intentionally disables this fallback.
PITL returns the Isaac-measured final incoming pose. Real execution currently
returns the commanded pre-insertion pose because post-motion perception
verification is not implemented.

### Fallback order

1. retry another trajectory or IK solution for the same grasp pair;
2. try another insertion grasp compatible with the current holder grasp;
3. release and acquire another holder grasp;
4. try that holder's compatible insertion alternatives;
5. abort safely when the bounded candidate list is exhausted.

The implemented slice applies this ordering only before hardware motion. Once
either robot starts moving, a planning, execution, or gripper failure aborts
the attempt; it does not try a different grasp from a changed live state.

### Safety requirements

- Exact common-frame calibration.
- Conservative collision and pose-uncertainty margins.
- Explicit parked configurations.
- Cancellation and best-effort gripper stop behavior.
- Low initial velocity and acceleration scales.
- Insertion force/torque limits or another contact-abort mechanism before
  claiming physical insertion robustness.
- No action success until transport, insertion, release, and retreat satisfy the
  action contract.

### Tests

- Pure coordinator unit tests with mocked robot interfaces.
- Cancellation in every execution phase.
- Pair fallback without changing holder.
- Table-supported holder change followed by pair fallback.
- Perception/TF drift rejection.
- Planning-scene application failure.
- Hardware-disabled integration smoke test.
- Supervised single-step execution before multi-step assembly.

## Scoring Strategy

### Hard feasibility gates

Do not trade these constraints against score:

- collision;
- table clearance;
- complete insertion-path clearance;
- individual IK;
- shared-scene arm-arm collision;
- valid trajectories;
- configured uncertainty margins.

### Initial holder score

Because the table supports gravity, holder scoring should initially emphasize:

- antipodal/contact quality;
- pad support;
- jaw geometry;
- table and subassembly clearance;
- holder pregrasp clearance;
- distance from insertion swept geometry;
- orientation relative to the insertion direction;
- lever arm between holder contact and insertion region;
- later, IK and trajectory robustness.

Without force, friction, or material data, insertion-direction restraint is a
ranking heuristic rather than a hard stability guarantee.

### Pair score

Keep component scores in the artifact. One initial ranking heuristic is:

```text
pair_robustness =
    0.6 * min(holder_quality, insertion_quality, clearance_quality, motion_quality)
  + 0.4 * mean(holder_quality, insertion_quality, clearance_quality, motion_quality)
```

Before Stage 4, omit `motion_quality` and renormalize.

Add small bonuses for:

- holder grasp also feasible in adjacent states;
- pair diversity;
- an already planned/reliable holder configuration.

Holder persistence must not be a feasibility requirement because the table
supports regrasping.

## Debug Artifact Structure

Suggested output layout:

```text
artifacts/dual_grasp_planning/<assembly>/
  index.html
  assembly_sequence.json
  assembly_sequence.html
  holder_base_candidates.json
  holder_base_candidates.html
  holder_state_feasibility.json
  holder_validity_matrix.html
  steps/
    step_001/
      holder_state.html
      dual_grasp_pairs.json
      pair_matrix.html
      ik_diagnostics.json
      ik.html
      trajectories.json
      trajectory.html
    step_002/
      ...
```

The master page should show the count after each filter:

```text
raw holder candidates
  -> base-valid
  -> state-valid
  -> incoming-sweep-valid
  -> holder/inserter compatible
  -> dual IK feasible
  -> shared-trajectory feasible
  -> simulated success
```

The existing self-contained SVG/canvas HTML approach should be reused first:
it works offline, embeds its JSON payload, and already supports rotation, pan,
zoom, candidate selection, and detailed geometry. Live RViz markers can be
added later as a complementary robot-level debugger.

## Implementation Boundary

Keep the coordination work as a standalone artifact pipeline and explicitly
guarded vertical slice. Do not change `run_pipeline.sh` mode behavior or claim
the complete `GraspAssembly` hardware contract until Stages 0 through 6 have
passed their completion gates. The standalone real runner may stop at each
phase for supervised hardware validation, but it must not report insertion
success because insertion, release, and retreat are not implemented.

The existing insertion stage-2 bundle remains the insertion-grasp source of
truth. New coordination artifacts should reference its candidates rather than
create a second insertion-grasp serialization path.

## Inputs Needed for the First Vertical Slice

- Assembly name.
- Confirmation that the first selected-order part is the physical base, or an
  explicit override when it is not.
- Confirmation of whether the assembled OBJ asset frame places the table at
  `z = 0`; otherwise an explicit table pose is required.

The exact two-robot calibration is not needed for Stages 0 through 3, but it is
required before Stage 4 can produce real-world reachability results.
