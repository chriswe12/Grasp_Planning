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
7. Save stable transition IDs, transforms, matching symmetry names, validation
   evidence, and rejection reasons in the per-step pair artifact.

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
6. Rank cheaply from the actual pickup/post-lift TCP using translation and
   rotation cost.
7. Try collision-aware IK in rank order. The simulation planning path also
   plans complete target sequences in rank order; the real executor plans the
   selected target online as each guarded motion executes.
8. Retain compatible alternatives as pre-motion fallbacks. After hardware
   motion starts, a path failure stops execution rather than switching
   candidates from a changed physical state.

Fallback has two levels:

- before close, another holder/inserter grasp pair may be selected;
- after close, only another transition candidate compatible with the already
  selected grasp may be selected.

Non-identity corridors are pair-conditionally checked for the bounded retained
fallback set. Other Stage-3 accepted pairs remain available through their
already checked identity corridor only; runtime must not infer transformed
compatibility where the pair artifact contains no such evidence.

IK/preflight caches must include the transition identity or complete target
signature. Caching only by grasp ID is invalid because one grasp can lead to
multiple pre-insertion targets.

## Artifact Provenance

Every executable candidate must preserve at least:

```text
parent_grasp_id
selected_pickup_grasp_id
pickup_grasp_symmetry
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
