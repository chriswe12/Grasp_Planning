# Franka GPU scaling

The local workstation exposes one RTX 4090 with 24 GiB VRAM. Multiple workers
on that one GPU are not multi-GPU training. The launcher discovers available
GPUs, uses all of them by default (`--gpu-count 0`), and refuses a request for
more GPUs than are installed. `--num-envs` is the number per GPU.

## CUDA library selection

The `isaac-lab-euler:2.3.2` image contains two cuDNN installations. Isaac's
ML archive provides the PyTorch-pinned cuDNN 9.7.1.26, but the Kit Python
site-packages can shadow it with cuDNN 9.20. The latter failed with
`CUDNN_STATUS_NOT_INITIALIZED` on the local NVIDIA 570.211.01 driver.

`scripts/franka_isaac_python.sh` selects the bundled library through process-local
`LD_PRELOAD` and `LD_LIBRARY_PATH`. It does not install packages or change the
host driver. Forward/backward CUDA convolution passed with runtime version
90701. Full PPO benchmarks also verify optimizer steps, finite checkpoint tensors
and loss metrics. NVIDIA lists driver 570.26 or newer for the CUDA 12.8/cuDNN 9.7.1
combination in its [support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.1/reference/support-matrix.html).

The launcher accepts `--cudnn-mode compatible` (default), `native` (ordinary
loader selection), or `disabled` (the original slower CUDA convolution fallback).
Each trainer writes `backend_rank_N.json` after constructing its model.

## Benchmark and resume

`scripts/benchmark_franka_training.py` runs complete short PPO training jobs in
a dedicated reusable Isaac container, verifies the resulting checkpoints and
records median transitions/sec after epoch one plus sampled peak whole-device
VRAM. `artifacts/franka_scaling/benchmark.json` retains both passing runs and
setup failures; a failure before scene creation is not an environment-count limit.
Growth stops on a failed run or less than approximately 2 GiB device headroom.

The selected configuration should maximize measured throughput with memory
headroom. More environments increase rollout size and transitions per epoch;
keep total transition budget in mind when comparing runs. PPO's effective global
minibatch stays near 1024 regardless of rank count, using the existing batching
helper to select a divisor of each rank's rollout batch.

The launcher supports `--checkpoint /path/in/repository/model.pth`. A matching
catalog contract sidecar is mandatory. Resume restores model/optimizer state;
new environments reset normally. Saved pilot checkpoints remain available under
`logs/franka_fabrica/20260914_145932/`.

## Multi-GPU execution

For multiple visible GPUs, the container entrypoint invokes `torch.distributed.run`
with one Isaac/PPO worker per GPU. Each worker uses its local CUDA device and a
rank-specific seed. Only rank zero writes shared run metadata and checkpoint
sidecars; each rank writes its own backend/device record. RL-Games synchronizes
actor and critic updates through the existing distributed PPO implementation.

For direct use inside the Isaac container, for example with two actual GPUs:

```bash
bash scripts/franka_isaac_python.sh -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  isaac_rl/scripts/train_franka_zed.py --headless --distributed \
  --experiment-name franka_two_gpu --num-envs 32 \
  --catalog isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz
```

Local benchmarks use one physical GPU. A real multi-GPU execution still requires
a machine with multiple GPUs; one-rank distributed smoke checks cannot establish
cross-GPU throughput or correctness. Validation/test evaluation after training
runs separately on one GPU.

### Euler Franka deployment

`euler/stage_franka.sh [--prepare-only] [checkpoint.pth]` builds a separate
source/data/checkpoint snapshot, verifies its catalog, object, scene and robot
hashes, and optionally uploads it to a new directory in the configured Euler
account. `--prepare-only` performs no network transfer or submission. The existing
generic Euler checkout is preserved. The generated `project/euler/euler.env`
selects that deployment and its own scratch log directory.

The Panda mirror under `assets/usd/franka_panda_offline/` contains the original
USD bytes and relative dependencies, downloaded by
`scripts/prepare_franka_offline_asset.py`. Its manifest identifies the original
source URL and checksums every file. `--robot-asset-manifest` verifies that identity
and uses the mirror for loading; catalog/checkpoint robot identity remains the
original source URL. Isaac supplies the built-in `OmniPBR.mdl` module.

After staging, set `EULER_CONFIG_PATH` to the generated configuration and
`EULER_SKIP_SYNC=1`, then use:

```bash
bash euler/submit.sh franka-smoke --gpu-count 6 --num-envs 32
# After the smoke succeeds, run a short optimizer check:
bash euler/submit.sh franka-train --gpu-count 6 --iterations 3
# After checkpoint/optimizer verification, resume the staged policy:
bash euler/submit.sh franka-train --gpu-count 6 \
  --checkpoint checkpoints/resume.pth --iterations END_EPOCH
```

Compute `END_EPOCH` from the resume epoch/frames and intended transition budget;
six ranks with 32 environments and a 64-step rollout add 12,288 transitions per
epoch. The effective global PPO minibatch is 1,536 (256 per rank), a divisible
batch for six ranks. The smoke creates all six scenes, runs inference and ten
simulation steps per rank, checks finite observations/rewards and performs an
NCCL all-reduce. It performs no optimizer updates. It does not measure rendering
quality or learned performance; inspect images and subsequent PPO artifacts too.

Slurm gives each renderer a one-GPU task cgroup and private writable caches.
Every rank uses logical `cuda:0`, retaining global/original rank and physical GPU
in its metadata. Native application/environment startup is serialized and a
file barrier waits for all environments before NCCL rendezvous. Each rank must
emit its completion marker; training additionally requires a finite checkpoint,
optimizer and loss verification report. Logs live under the generated Euler
runs directory in `franka/<experiment>/`.

On 2026-09-14, read-only access confirmed the `gimenol` account, the staged SIF,
pretrained weights and available RTX 4090 node type; no account jobs were active
at inspection. A local package was prepared at
`artifacts/franka_euler/20260914_161209/`: 381 files, 120,823,595 bytes, matching
catalog/scene/robot identity, and a finite epoch153 resume checkpoint with454
actor updates. `resume_plan.json` proposes end epoch301 for approximately2.048M
total transitions. Five asset/appearance tests and mocked six-GPU submission
argument checks passed. Automatic approval review rejected the source/assets/
checkpoint upload pending explicit transfer approval. No Euler smoke, PPO test
or training was submitted; multi-GPU runtime remains unverified. Local training
was left running.

## Reset update batching

`FrankaAppearanceRandomizer.apply_many()` prepares all resetting environments'
attribute edits, then commits cached Sdf attribute specs inside one change block.
No Usd queries or mutations occur inside that block, following the
[OpenUSD change-block contract](https://openusd.org/release/api/class_sdf_change_block.html).
Live RGB gains are copied to CUDA in one batch as well. The material/light values,
seeded samples and canonical goals are unchanged. Tests cover one-notice updates,
composed values and preservation of the referenced source layer; the four-room
render-isolation/deterministic-reset check is recorded in
`artifacts/franka_scaling/batched_appearance_check.json`.

The same image also shadows bundled NCCL 2.26.2 with NCCL 2.29.7, which failed
collective initialization on the local driver. The compatible mode selects both
bundled libraries. `isaaclab.sh` overwrites `LD_PRELOAD` with its `libcarb.so`;
`scripts/franka_python_bootstrap.py` therefore re-execs Python after that setup,
preserving libcarb and applying the CUDA preloads before importing Torch/Isaac.
A real NCCL broadcast and three-epoch, one-rank distributed PPO smoke passed:
`artifacts/franka_scaling/distributed_smoke.verification.json` (768 transitions,
286 finite tensors, three actor updates). No second physical GPU was available.

On 2026-09-14, the accelerated batched-reset sweep measured median full-PPO
throughput after the first epoch:

| Environments | Transitions/sec | Sampled peak device MiB |
| --- | ---: | ---: |
| 16 | 222 | 9,725 |
| 32 | 238 | 12,787 |
| 48 | 227 | 11,840 |
| 64 | 192.5 | 12,519 |

All four runs completed three epochs and passed optimizer/checkpoint checks.
Short-run peak allocations can vary with renderer/cuDNN caching; they are not
per-environment constants. An earlier 128-environment probe fit at about 17,209
MiB but slowed to 90 transitions/sec and was intentionally stopped. It was not
an out-of-memory failure or a completed optimizer validation. The chosen default
is 32 environments per GPU, based on throughput rather than filling VRAM.

Resume verification compares optimizer steps and transition count against the
source checkpoint, so merely re-saving old weights cannot pass. A resumed run
preserves the supplied checkpoint as an evaluation fallback. The launcher copies
it and its sidecar into the new run directory for independent verification. The detached entrypoint
requires a trainer completion marker, a finite final checkpoint/optimizer report,
and actual evaluation artifacts before it writes `COMPLETED`; Isaac's exit code
alone is insufficient. Trainer exceptions are printed before Kit teardown.

The accelerated persistent run launched as `franka-fabrica-zed-20260914_155609`,
32 environments on the available GPU, resuming the best policy from the resize
probe. That probe advanced epoch 77 to 80 with 12 new actor updates and 84,992 total
transitions. The new epoch limit 1,039 preserves the original approximately 2.048
million total-transition budget despite doubling environments. Live metadata:
`artifacts/franka_fabrica/latest_launch.json`; resume proof:
`artifacts/franka_scaling/resume_probe.verification.json`.

The persistent run passed its own startup check at checkpoint epoch79:82,944
transitions,286 finite tensors and4 new actor updates beyond its resume snapshot.
Evidence: `logs/franka_fabrica/20260914_155609/startup_check.json`.

## Euler execution (2026-09-14)

The user approved source, scene and checkpoint transfer to the private `gimenol`
Euler account. The isolated deployment is
`/cluster/home/gimenol/grasping_rl_franka_20260914_161209`; results are under
`/cluster/scratch/gimenol/grasping-rl-runs/franka_20260914_161209`.
The generic remote checkout is preserved.

The six-GPU chain (14149453, 14149636, 14149727) was cancelled after the
four-GPU fallback acquired resources. The active configuration uses four RTX4090
GPUs, 32 environments per rank (128 total), and a global PPO minibatch of1024.

| Job | Purpose | Verified state |
| --- | --- | --- |
| 14153579 | Four-rank camera, inference and NCCL smoke | Completed; all four ranks passed |
| 14153793 | Three-epoch distributed PPO validation | Completed; synchronized actor/critic weights and finite checkpoints |
| 14153881 | Resume epoch153 through375 | Running; new optimizer updates and downloaded checkpoint verified |

The PPO check collected24576 transitions, completed45 actor optimizer updates,
and reached approximately689 transitions/sec at epoch3. Sampled peak GPU usage
was8.3GiB or less. The full run's time limit was reduced from4hours to2hours
based on the measured throughput, retaining substantial startup/runtime margin.
These short-run results do not establish long-run memory stability or task success.

`watch_and_pull.sh` successfully downloaded the real PPO checkpoints automatically;
an independent CPU verification of the downloaded files passed. Evidence:
`artifacts/franka_euler/20260914_161209/ppo_pulled_verification.json`.
The detached full-run watcher is recorded in `launch.json` and writes
`watcher_four_gpu.log`. It pulls results on termination, including failure logs.
Keep the workstation on for automatic downloading; the Slurm job itself runs
independently. Do not start a second watcher when the existing one is alive.

```bash
ssh euler-gimenol 'squeue --me'
tail -f artifacts/franka_euler/20260914_161209/watcher_four_gpu.log
# Restart the watcher only if needed:
EULER_CONFIG_PATH="$PWD/artifacts/franka_euler/20260914_161209/project/euler/euler.env" \
EULER_LOCAL_RESULTS_DIR="$PWD/logs/euler_franka_20260914_161209" \
bash euler/watch_and_pull.sh 14153881
```

Each rank records live Torch allocation, allocator reservation, CUDA device
memory and process resident RAM every optimizer epoch. CUDA total-minus-free
measurement respects Slurm's task-local GPU mapping. The observation gate skips
25 warm-up epochs, requires75 further samples per rank, limits sustained device
and live-tensor growth to0.5MiB/epoch, and requires2GiB free VRAM. A passing bounded
window means no leak was observed during that window, not proof for an unlimited run.
`verify_franka_training_run.py --in-progress` additionally checks new optimizer
updates beyond the source checkpoint and progress on four distinct GPU UUIDs;
final actor/critic weight agreement remains a completion-only check.

### Setup fixes and validation boundaries

The initial auxiliary one-GPU jobs failed before useful validation:14149955 hit
argparse's handling of a separate `--/plugins/...` Kit argument;14152370 exposed
missing catalog-referenced stage-2 source bundles. The joined Kit argument and
portable copy/hash verification of all four original bundles fix those issues.
Both four-GPU validation jobs subsequently passed. The source bundles remain the
original Fabrica serialization; no second grasp format was introduced.

Each rank has private writable Kit caches, and native simulator startup is
serialized before a distributed barrier. The four-rank startup takes around11minutes
on the observed node. Every backend report includes its actual GPU UUID.

The Euler host NGX library was absent from Apptainer's default `--nv` mounts.
`euler/franka_apptainer.sh` exposes the matching host driver library, and
`scripts/franka_isaac_python.sh` puts that directory on the loader path. Loading
that library inside the actual SIF and the active simulator was verified. All
four full-run workers initialized without the previous NGX errors, and the
first training image was visually inspected successfully.

Smoke image grids from ranks0 and3 were visually inspected: colored parts,
Franka fingers and pencil table were visible. The full trainer also saves
`first_training_rgb.png` after its first optimizer epoch and rejects nearly
uniform images. Appearance variation and synchronized training do not establish
physical grasp/lift success; this task still trains visual alignment.

The full run started on eu-g6-022 at17:40 CEST. Independent inspection of its
downloaded epoch159 checkpoint found286 finite tensors,283648 transitions and
96 new actor optimizer updates beyond the resume snapshot. All four distinct
GPU workers reached epoch162; finite loss metrics and catalog/contract identity
passed. Evidence: `full_training_startup_check.json` in the deployment artifacts.
The four-rank memory observation passed through epoch255:102 new epochs over
26.6minutes, including77 samples/rank after25 warm-up epochs. Whole-device VRAM
and allocator reservations had zero measured slope on every rank; live Torch
allocation was effectively flat. Each GPU retained at least15.24GiB free.
Post-warm-up process RSS rose by at most6.2MiB on three ranks and decreased on
rank0. This is a bounded observation, not a guarantee for unlimited training.

The downloaded epoch250 checkpoint also passed:1029120 transitions,286 finite
tensors,1550 new actor optimizer updates, and a matching remote/local SHA256.
All captured full-run logs were scanned with no renderer/Python/CUDA/NCCL error
matches; normal headless/deprecation warnings remain. The full job continues
through epoch375. Slurm accepted a90minute minimum backfill window and retained
the full2hour allocation. Final rank weight hashes are checked by the batch job
at completion before the watcher reports success.

Evidence in `artifacts/franka_euler/20260914_161209/`:
`full_training_health.json`, `full_training_memory.json`,
`full_training_epoch_250_check.json`, `memory_observation.png` and
`ngx_library_check.json`. The first real training camera grid is in
`logs/euler_franka_20260914_161209/franka/2026-09-14_17-40-22_job_14153881_4gpu/first_training_rgb.png`.
The original local32-environment training was preserved and remains a separate run.
