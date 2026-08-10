# Dual-Robot Grasp And Transition Symmetry Plan

Status: implemented for the current Stage-3 and pre-insertion vertical slice

## Goal

Make dual-arm assembly planning insensitive to symmetry-equivalent perception
poses while keeping the holder fixed and preferring the simplest reachable
motion from pickup to pre-insertion.

The design uses symmetry at two distinct boundaries:

1. **Pickup-grasp symmetry** expands an incoming part's object-frame grasp into
   symmetry-equivalent contact/TCP choices before the gripper closes.
2. **Transition symmetry** expands a selected assembly step into
   symmetry-equivalent final and pre-insertion corridors after the grasp has
   been fixed.

Once the inserter closes, the selected part-to-TCP transform is immutable.
Transition fallback may change the destination corridor, but it must not switch
to a different grasp without an explicit regrasp.

## Frame Contract

Let:

- `T_A_B` map the base-part source frame into the canonical assembly frame;
- `S_B` be a finite base-part symmetry in that source frame;
- `G_A = T_A_B @ S_B @ inverse(T_A_B)` be the corresponding assembly-frame
  transform;
- `T_A_i` be a part source pose in the canonical final assembly;
- `S_i` be a finite symmetry of part `i`.
- `D_A` be a selected incoming-part symmetry expressed in the assembly/object
  coordinates used by the authored OBJ mesh.

`G_A` is a valid symmetry for a particular assembly step only when every part
already present remains geometrically equivalent:

```text
G_A @ T_A_i ~= T_A_i @ S_i
```

The incoming final pose must satisfy the same relation for at least one of its
symmetries. Equality here means equivalent occupied geometry within configured
tolerances, not identical numeric SE(3) coordinates. Individual geometric
part symmetry is not by itself proof of functional mating equivalence.

For an accepted step symmetry, transform the complete canonical corridor. If
`T_A_pre` is the canonical pre-insertion source pose:

```text
T_A_final_candidate = G_A @ D_A @ T_A_final
T_A_pre_candidate   = G_A @ D_A @ T_A_pre
```

The incoming symmetry must be left-applied to both endpoints; post-composing
it onto each source frame would change the wrist representative without
rotating the insertion direction. The insertion vector and every sampled
insertion pose therefore rotate with the selected corridor.

Raw `matrix_obj` values from `symmetries.json` must first be converted from the
asset object frame into the saved source frame, including mesh-scale handling.

## Offline Planning

For each holder-active assembly step:

1. Load finite proper-rotation symmetries for the base, assembled prefix, and
   incoming part.
2. Convert the base transforms into assembly-frame candidates.
3. Reject candidates that do not preserve every already assembled part.
4. Match each surviving candidate to an incoming-part final symmetry.
5. Generate the corresponding final/pre-insertion corridor.
6. For every retained grasp pair, re-run the transformed inserter-gripper
   insertion/retreat sweep against the table, assembled prefix, and selected
   holder gripper. Robot-link checks remain the runtime planner's job.
7. Flatten every accepted pair/transition combination into an execution
   candidate and retain a bounded final set by round-robin corridor direction
   and pair diversity.
8. Save stable execution-candidate and transition IDs, transforms, matching
   symmetry names, validation evidence, and rejection reasons in the per-step
   pair artifact.

Identity must always remain available. Continuous symmetry hints are not
expanded until a bounded sampling policy is explicitly introduced.

## Runtime Planning

At runtime:

1. Keep the detected base and incoming-part poses exactly as reported.
2. Recheck symmetry-expanded pickup grasps against the actual pickup pose.
3. Select a dual grasp pair and preserve its parent/variant provenance.
4. Lock the selected part-to-TCP transform after close.
5. Instantiate all compatible transition corridors in the perceived world
   assembly frame using that fixed grasp.
6. Use the Cartesian layout score only to order within each corridor, then
   round-robin corridors to bound the online pre-ranking pool. This guarantees
   that an identity corridor cannot consume the whole pool before joint-space
   evaluation reaches an equivalent opposite-side corridor.
7. Plan the shared pickup prefix once per incoming grasp. From its terminal
   lift joints, use multi-seed IK and path planning for every candidate's
   above-pre-insertion and pre-insertion targets.
8. Probe A7 `+pi` and `-pi` seed offsets for symmetry candidates and reject
   out-of-limit probes. Because iiwa A7 is bounded slightly below pi, also seed
   valid `+3.0` and `-3.0` rad branch targets derived from the reached lift
   state. Partition the complete bounded queue into a strict non-crossing
   phase and a crossed fallback phase. Within each phase, rank successful
   pre-plans by velocity-weighted joint-path cost before unchecked and failed
   cheap-preplan fallbacks. A crossed successful pre-plan must not jump ahead
   of any remaining non-crossing candidate.
   A7 is a cheap seed direction, not a commanded single-joint shortcut; IK may
   adjust every joint.
9. Execute in joint-space rank order with exact shared-scene replanning and
   retain the remaining complete candidates as fallbacks. The real executor
   continues to plan the selected target online as each guarded motion executes.
10. After hardware
   motion starts, a path failure stops execution rather than switching
   candidates from a changed physical state.

Fallback has two levels:

- before close, another holder/inserter grasp pair may be selected;
- after close, only another transition candidate compatible with the already
  selected grasp may be selected.

Non-identity corridors are pair-conditionally checked for the bounded retained
pair pool, then retained online as complete pair/transition execution IDs. The
default bound is 256 so the orientation-dependent pickup-floor check can remove
invalid grasps without collapsing runtime fallback to only a few holders.
Retained-pair selection covers distinct inserter grasps before taking repeated
pairs for one inserter, preserving pickup-orientation diversity without baking
in a simulated world pose. The
same actual-pose check and retained pool are used in simulation and real mode;
no particular pickup orientation is compiled into the offline artifacts.
Other Stage-3 accepted pairs remain identity-only records. Default runtime
planning may also use a non-retained execution when its retained pair contains
an explicit accepted validation for that exact transition; already
collision-checked canonical targets from identity-only pairs fill the rest of
the bounded queue. It must never infer transformed compatibility where the
artifact contains no such evidence.

The online layout proxy evaluates both the pickup target and pre-insertion
target. It combines ownership/non-crossing scores across the two phases and
records a soft crossing penalty if either shoulder-to-target line crosses the
holder line in XY. Runtime ordering adds a strict phase boundary around that
proxy: every bounded non-crossing candidate is tried before the crossed phase.
This is still not collision proof; exact shared MoveIt planning remains
authoritative, and crossed candidates remain available only after the clear
phase fails.

IK/preflight caches must include the transition identity or complete target
signature. Caching only by grasp ID is invalid because one grasp can lead to
multiple pre-insertion targets.

## Artifact Provenance

Every executable candidate must preserve at least:

```text
parent_grasp_id
selected_pickup_grasp_id
pickup_grasp_symmetry
execution_candidate_id
transition_id
partial_assembly_symmetry
incoming_final_symmetry
part_to_tcp_transform
final_part_pose
preinsertion_part_pose
insertion_vector
collision_validation
```

The chosen transition ID must remain in simulation, real-task, and execution
attempt artifacts so later insertion uses the same final-pose representative.

## Safety And Validation

- Symmetry preserves object-local grasp validity only when the geometry match
  is valid; environment collision checks must always be rerun.
- The complete partial assembly, table support, holder gripper, both robot
  arms, and the attached incoming object remain collision constraints.
- Stage-3 checks the incoming object and both selected end effectors. The
  current simple MoveIt scene still omits exact attached-object geometry, so
  its joint-space ranking is a robot/work-surface pre-plan and Isaac remains
  responsible for physics/contact validation. This limitation must be removed
  before treating the same path as hardware-safe.
- A pointwise IK success is only a prefilter. Hardware execution requires a
  planned path from the current joint state.
- The real executor currently selects a candidate by collision-aware IK before
  motion, then asks MoveIt to plan each selected target online. It cannot
  safely fall back to another transition after partial hardware motion.
- The current real vertical slice stops no later than pre-insertion; actual
  constrained insertion remains a separate future phase.
- Runtime assembly layout is currently yaw-only. Non-planar symmetries must be
  rejected until full six-degree-of-freedom assembly placement is supported.

## Verification

Tests must cover:

- source-frame and mesh-scale conversion of symmetry transforms;
- identity fallback when assets are missing or no non-identity transform is
  step-compatible;
- partial-assembly symmetry reduction as parts are inserted;
- incoming final-pose equivalence and alternate pre-insertion sides;
- invariant part-to-TCP transform across pickup and every transition;
- collision rejection for a transformed corridor;
- deterministic ranking and fallback;
- distinct IK cache entries for the same grasp with different transitions;
- complete symmetry provenance through saved artifacts.

Implemented coverage is in `tests/test_transition_symmetry.py`,
`tests/test_dual_grasp_pair_planner.py`,
`tests/test_dual_robot_simple_sim.py`,
`tests/test_simple_dual_robot_sim_planner.py`, and
`tests/test_dual_real_grasp_executor.py`.

## Current Plumbers-Block Result

With the configured 1 mm geometry-equivalence tolerance, the four
holder-active steps compile 6, 2, 6, and 6 transition candidates before
pair-conditioned collision filtering. Step 1 contains both the canonical
`+Y` insertion corridor and symmetry-equivalent `-Y` corridors.

The base part's Z180 transform is not accepted as a symmetry of the step-1
partial assembly: it moves incoming part 0 by about 2.532 mm. The opposite-side
step-1 choices instead come from part 0's own validated finite symmetries. This
is intentional; raising the partial-assembly tolerance would change the final
placement and needs an explicit mating-level decision.

**Why 2.532 mm specifically:** each part's Z180 symmetry axis is validated
independently against its own mesh, and part 0's and part 2's axes do not sit
at the same Y coordinate. `assets/obj/fabrica/plumbers_block/symmetries.json`
records `object_z_bounds_center_order2_step1` at `center_obj_m` Y `= 0.0` for
part 0 and Y `= 0.0012658` for part 2 (all four candidate centers - bounds
center, area-weighted centroid, vertex mean, volumetric center of mass - agree
per part, so this is not a center-convention bug). A 180 degree rotation
doubles any perpendicular pivot offset, so applying part 2's axis to part 0
lands `2 x 0.0012658 m = 0.0025316 m` away from part 0's own valid symmetric
placement - over the 1 mm tolerance by design. The root cause is that part
2's mesh is centered ~1.27 mm off the assembly's Y = 0 symmetry plane (a small
CAD/mesh-authoring asymmetry, likely a feature near one Y edge), not a
frame/origin bug in the symmetry code. Fixing it for real would mean
re-centering part 2's mesh (or its accepted symmetry's pivot) to Y = 0, not
adjusting tolerances or composition code.
