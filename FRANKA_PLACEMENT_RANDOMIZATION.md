# Independent object placement with canonical goals

Added 2026-09-21 after finding that the earlier pose bank perturbed the robot
around only two catalog object locations. Previous scores demonstrate alignment
within that distribution; they do not establish unknown-location robustness.

## Placement contract

`isaac_rl/data/franka_clutter_v5_placement/catalog.npz` uses the same Panda,
ZED Mini mount/intrinsics, 15 Hz actions, 120 Hz physics, parts, pencil table,
floor colors, renderer and existing per-episode appearance variation as the
previous FXAA catalog. The following changes are explicit in its contract:

- Every target samples the **same absolute tabletop region**, X = 0.33–0.58 m,
  Y = −0.185–0.185 m, relative to the fixed robot base. Sampling does not center
  each target on its old catalog XY location.
- Object yaw varies by up to ±90 degrees from its selected upright orientation.
  Z and upright tilt remain grounded as in the source target.
- Offline IK retargets the desired start and goal TCP with the object. The
  robot base and table do not move. Both poses must pass 12 physics steps with
  contact below 0.5 N and pose error below 0.8 mm / 0.012 rad.
- Goal RGB-D pixels remain **byte-identical** to their canonical source images.
  Runtime reset changes the live object pose and the simulation scoring goal;
  it never rerenders a goal using the randomized world placement.
- Reset mixture, curriculum, nominal starts, ready starts, and hard negatives
  use the accepted placement bank. There is no original-location fallback.
  The bank is finite (up to eight candidates per selected kind/progress), and
  runtime randomly selects accepted entries. It is not fresh continuous IK
  on every reset.
- Rejected candidates and excluded targets are reported. Consequently the
  accepted distribution is conditioned on IK/contact feasibility; it is not
  claimed to cover every point or yaw uniformly after filtering.

This removes matching absolute scene layout as a requirement. Object identity,
CAD shape, selected upright orientation and object-local grasp are still known
inputs to goal selection. Simulator truth remains necessary for resets,
rewards and scoring; it is excluded from the actor input path. This is an
alignment task with open fingers. The source catalog's lift/path validation
flags refer to its **canonical** paths, not new randomized lift trajectories.

## Rebuild (inside the repository Isaac runtime)

```bash
bash scripts/franka_isaac_python.sh isaac_rl/scripts/build_franka_placement_resets.py \
  --output isaac_rl/data/franka_clutter_v5_placement/catalog.npz \
  --num-envs 128 --variants 8 --headless --device cuda:0
```

The builder reads the exact robot joint transforms from USD, verifies FK against
PhysX and the TCP Jacobian against finite differences, then validates candidates
in physics. It does not alter the existing policy controller. A discrepancy
between that controller's PhysX point-Jacobian and the USD point-Jacobian was
observed during construction; its cause is not established by this experiment.
The same existing controller is used in both evaluation conditions.

The output `.placement.json` includes source catalog hash, goal hash, coverage,
exclusions, acceptance counts and sampled ranges. The original catalog is retained.

## Frozen latest-policy comparison

```bash
bash scripts/franka_isaac_python.sh isaac_rl/scripts/benchmark_franka_placement.py \
  --catalog isaac_rl/data/franka_clutter_v5_placement/catalog.npz \
  --checkpoint artifacts/franka_speedup_20260917/euler_results/franka/2026-09-19_01-25-35_job_14461129_4gpu/nn/last_franka_zed_ep_5000_rew__45.270706_.pth \
  --agent artifacts/franka_placement_20260921/agent.yaml \
  --output artifacts/franka_placement_20260921/benchmark \
  --split test --headless --device cuda:0
python3 scripts/report_franka_placement.py artifacts/franka_placement_20260921/benchmark
```

This explicit diagnostic permits only the new placement contract field to differ
from the checkpoint. Ordinary training/resume checks remain strict. It verifies
identical canonical goal pixels, actor invariance to hidden labels, and normal
training reset pools. Original/new placement trials share target, source
perturbation and appearance seed. Initial relative errors are checked within
1.5 mm / 0.025 rad; small IK/physics differences can remain. Results contain
termination categories, error distributions, per-part outcomes and paired
part-bootstrap uncertainty, plus actual Isaac render pairs.

No checkpoint is retrained by this command. To train a new policy, select the new
catalog explicitly with `train_franka_zed.py --catalog ...`; an old checkpoint
cannot be silently resumed with changed randomization. Hardware calibration and
real-world robustness still require a separate test.

## September 21 verification and frozen-policy result

The completed bank retains 1,592 of 1,615 targets, all 46 parts, and 141,337
accepted reset states. All five perturbed approach distances remain represented
for every retained target. Both the independent-placement audit and the existing
source geometry/split audit pass. The 23 exclusions are listed in the placement
report; one part (`plumbers_block__part_2`) has no remaining **test** targets,
although it remains represented in other splits. Do not present this as the
complete original held-out cohort.

`artifacts/franka_placement_20260921/evaluation_catalog.npz` contains retained test
targets plus the 40 validation targets for `plumbers_block__part_3`. It is a row
subset, with identical contract and goal pixels. The recorded test uses three
seeds, three distances, and 29 parallel parts; validation results stay separate.

| Cohort | Original placement | Randomized placement |
| --- | ---: | ---: |
| Held-out test, 28 parts | 117/252 (46.4%) | 0/252 |
| Plumbers block part 3, validation | 1/9 | 0/9 |

This is 522 episodes, not training. Initial paired error differences were at most
0.0108 mm in position-error magnitude and 0.159 degrees in rotation-error
magnitude. On test targets the median terminal errors increased from 3.15 mm /
2.49 degrees to 16.13 mm / 21.62 degrees. Randomized failures were 138 timeouts,
74 collisions, 35 divergences and 5 premature declarations. The checkpoint is
not robust to this placement change; these observations alone do not isolate
background shortcuts from every possible posture/dynamics contribution.

All 348 normal-curriculum reset smoke samples passed, covering all four modes.
Actor outputs were exactly unchanged when the eight hidden scoring/supervision
values were zeroed. The report has 36 actual Isaac scene images (18 paired views)
and full trajectories/part breakdowns. Far-start examples can put the object
partly outside the wrist view; those trials were not removed based on visibility.

A separate `--oracle-check` uses true pose errors for a low-gain Cartesian
controller, solely to check scene controllability. It is never counted as policy
performance. `--reference-results .../benchmark/results.json` replays the exact
near-start target/reset entries from the policy benchmark. No training,
checkpoint modification, or hardware motion was performed in this change.

The matched oracle check succeeded on 29/29 canonical and 29/29 randomized
near starts. Its target IDs, source-bank entries and placement deltas were
explicitly checked against the frozen-policy records. This confirms those
randomized states are controllable under the existing controller; it is not a
replacement for a learned-policy score. Raw evidence is in
`artifacts/franka_placement_20260921/oracle_matched/results.json`.
