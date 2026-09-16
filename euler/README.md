# ETH Euler deployment

This directory deploys the Isaac Lab 2.3.2 RGB-D task to ETH Euler without
copying the local training-log tree. The immutable Apptainer image holds Isaac
Sim and Isaac Lab. The external `isaac_rl` project is synchronized separately
and made importable through `PYTHONPATH` at job startup.

The deployment uses the `gimenol` staff login and the `es_meboldt` shareholder
allocation. The safe default is still 8 CPU cores, 24 GiB system RAM, and one
RTX 4090, while `submit.sh` accepts explicit GPU, memory, CPU, and time selectors
for controlled capacity probes. The six-GPU, 48-core, and 144-GiB allocation is
an aggregate ceiling. For two or more GPUs, the wrapper launches one Slurm task,
Isaac Sim process, and RL-Games rank per GPU; gradients are synchronized by
RL-Games, while each rank owns its own simulator, environment batch, policy
replica, and VRAM allocation. Slurm isolates each renderer in a one-GPU task
cgroup before Apptainer or Vulkan starts.

## What happens in each layer

1. `build_image.sh` derives `isaac-lab-euler:2.3.2` from the existing local
   `isaac-lab-base:latest` image and explicitly installs the missing core
   `isaaclab` package.
2. `push_image.sh` converts that Docker image to a SIF, runs an import check and
   a one-environment simulator smoke test locally, then uploads the verified
   SIF to Euler scratch.
3. `submit.sh` synchronizes this repository to Euler home and calls `sbatch`.
   The resource request and job submission happen in this single `sbatch`
   command; Slurm queues the job until those resources are available.
4. `job.sbatch` copies the large SIF and caches to node-local `$TMPDIR`, starts
   a one-second GPU-memory sampler, then launches the requested test or train
   command through Apptainer. Training also records allocator and whole-device
   VRAM once per PPO epoch for long-run leak detection.
5. The source checkout is mounted read-only, but its `logs/` directory is
   replaced by a writable bind mount to persistent Euler scratch. Checkpoints,
   TensorBoard events, Slurm logs, and GPU measurements therefore survive the
   compute node shutting down.

## First-time preparation

Create the ignored site-specific configuration from the checked-in template:

```bash
cp euler/euler.env.example euler/euler.env
```

Set the login, Slurm account, home path, and scratch paths for your allocation.

Verify SSH authentication first:

```bash
ssh euler-gimenol
```

Enter the ETH password when prompted, then run `exit`. For repeated project
syncs and submissions, configure an SSH key as described in the ETH HPC SSH
documentation. `push_image.sh` performs this authentication check before any
long image work and reuses the connection for its directory creation, upload,
and remote validation.

Build the corrected image and replace the old SIF that failed with
`No module named 'isaaclab'`:

```bash
./euler/build_image.sh
./euler/push_image.sh --force
```

The image only needs rebuilding when Isaac Lab, Isaac Sim, or Python package
dependencies change. Ordinary edits to `isaac_rl` are synchronized by every
submission and do not require another 10 GB image upload.

## Test before a long training run

First submit a simulator-only smoke test. This creates the task, resets it, and
steps it ten times, but does not initialize PPO or save weights:

```bash
./euler/submit.sh smoke
```

The command prints `Submitted batch job JOB_ID`. Monitor it with:

```bash
ssh euler-gimenol 'squeue --me'
ssh euler-gimenol 'tail -n 100 /cluster/scratch/gimenol/grasping-rl-runs/slurm-JOB_ID.out'
ssh euler-gimenol 'tail -n 100 /cluster/scratch/gimenol/grasping-rl-runs/slurm-JOB_ID.err'
```

A successful smoke log contains several `[SMOKE]` lines and a final line with
`steps=10`. Warnings about GLFW or a missing display are normal for a headless
run; a Python traceback, `CUDA out of memory`, or a non-zero Slurm exit status
is not.

Next run short, real training probes. These execute five complete PPO
iterations, so their VRAM measurement is representative of training:

```bash
./euler/submit.sh probe 16
./euler/submit.sh probe 32
./euler/submit.sh probe 48
./euler/submit.sh probe 64
```

To probe the separate five-part task, pass the complete training arguments:

```bash
./euler/submit.sh probe \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 64 \
    --max_iterations 5 \
    --headless \
    --enable_cameras
```

Run them progressively rather than all at once. After a probe completes, read
its peak memory:

```bash
ssh euler-gimenol 'cat /cluster/scratch/gimenol/grasping-rl-runs/metrics/gpu-JOB_ID.summary.txt'
```

There is no reliable environment count that can be calculated in advance:
camera render targets, simulator buffers, the ResNet, and PPO batches do not
all scale linearly. Choose the largest count that completes with roughly
10-20% VRAM still free, then compare
`performance/step_inference_rl_update_fps` in TensorBoard. The fastest safe
count is more useful than the count that merely fills the most VRAM. The
one-second sampler can miss a very short peak, so do not use the last few MiB.

The default requests an RTX 4090 and at least 20 GiB. Override the resource
selector before the training arguments, for example:

```bash
./euler/submit.sh probe \
    --gpu-type rtx_pro_6000 \
    --gpu-memory 90G \
    --time-limit 00:30:00 \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 1024 \
    --max_iterations 5 \
    --headless \
    --enable_cameras
```

`--gpu-type`, `--gpu-count`, `--gpu-memory`, `--cpus-per-gpu`,
`--memory-per-cpu`, and `--time-limit` configure Slurm; all remaining flags are
forwarded unchanged to the Isaac training script. The job log records the
requested and assigned GPU. See [GPU_BENCHMARKS.md](GPU_BENCHMARKS.md) for the
measured environment counts and active target-GPU probes.

`--num_envs` is **per GPU** when `--gpu-count` is greater than one. For example,
this two-GPU probe simulates 448 environments and collects a global
`448 * 64 = 28,672`-sample rollout per PPO iteration:

```bash
./euler/submit.sh probe \
    --gpu-type rtx_4090 \
    --gpu-count 2 \
    --gpu-memory 20G \
    --time-limit 00:30:00 \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 224 \
    --max_iterations 5 \
    --headless \
    --enable_cameras
```

Do not add `--distributed`; the wrapper adds it and creates the matching Slurm
tasks automatically. Each rank gets a distinct seed and an isolated GPU.
RL-Games aggregates gradients and reports global throughput from rank zero.
The batch job verifies every rank's completion marker and records peak VRAM for
every allocated GPU. Multi-GPU does not combine GPU memory into one pool.

Distributed training changes the global rollout and effective gradient batch.
The configured 1,024-sample minibatch is treated as the target effective global
batch, so four ranks automatically use 256 samples each. This keeps optimizer
updates proportional to the larger rollout instead of silently making every
gradient four times larger. The resolved local/global batch sizes and updates
per epoch are printed and saved in `params/{agent,sim2real_profile}.yaml`.
Independent one-GPU seeds and hyperparameter jobs remain preferable when the
goal is more experimental evidence rather than lower wall-clock time for one policy.

For a group of benchmark submissions from one unchanged checkout, perform the
first submission normally. `EULER_SKIP_SYNC=1 ./euler/submit.sh ...` may then
reuse that exact remote source snapshot. Never use this shortcut after editing
the local project: a running job mounts the synchronized checkout read-only,
and a later sync would change what queued jobs execute.

The five-iteration probes normally do not produce weights: the current agent
configuration starts best-model saving after iteration 20 and saves periodic
checkpoints every 50 iterations. Their purpose is task, PPO, and VRAM
validation. Probe RL-Games and Hydra artifacts are written to node-local
`$TMPDIR` and discarded when the allocation ends; only their Slurm output and
GPU metrics remain in global scratch.

A five-iteration probe proves only that the initial allocation fits. Use a
long probe to prove that 256 environments per RTX 4090 remain stable:

```bash
./euler/submit.sh probe \
    --gpu-type rtx_4090 \
    --gpu-count 4 \
    --gpu-memory 20G \
    --time-limit 02:00:00 \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 256 \
    --max_iterations 200 \
    --headless \
    --enable_cameras
```

The automatic report is pulled to
`logs/euler/metrics/gpu-JOB_ID.training-memory.json`. A `PASS` result means all
ranks retained at least 1 GiB free and their fitted post-warm-up growth stayed
below 0.50 MiB per epoch. This is a targeted regression check: the previous
approximately 2 MiB/epoch growth is large enough to detect over the 175
post-warm-up samples. Ordinary five-epoch probes report
`INSUFFICIENT_SAMPLES`; this is not a failure, but it is also not a stability
result.

The rank-asymmetric growth came from the stock Isaac RL-Games observer retaining
CUDA-backed episode dictionaries on nonzero ranks, which never execute the
rank-zero-only statistics callback. The training entrypoint uses a rank-safe
observer that skips those unused statistics on worker ranks while preserving
rank-zero TensorBoard output.

Two controlled full-training variants are selectable without changing the
task or goal catalog:

```bash
# Combined baseline plus peripheral RGB-D clutter in 60% of environments.
./euler/submit.sh train \
    --gpu-type rtx_4090 --gpu-count 4 --gpu-memory 20G \
    --time-limit 16:00:00 \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 256 --global_minibatch_size 1024 \
    --max_iterations 2500 --seed 43 \
    --sim2real_profile combined_clutter \
    --headless --enable_cameras

# Combined baseline with stronger structured depth corruption and no clutter.
./euler/submit.sh train \
    --gpu-type rtx_4090 --gpu-count 4 --gpu-memory 20G \
    --time-limit 16:00:00 \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 256 --global_minibatch_size 1024 \
    --max_iterations 2500 --seed 44 \
    --sim2real_profile combined_depth_robust \
    --headless --enable_cameras
```

Each collects the same approximately 163.84 million transitions as the
four-GPU 2,500-iteration combined baseline. The clutter is render/depth-only
and peripheral; it does not replace real-robot collision checking. The depth
variant changes only depth-error ranges, retaining the baseline RGB,
appearance, camera-warp, timing, control, reset, reward, and PPO settings.

`sync_project.sh` also downloads the TorchVision ResNet-18 weights once into
`.cache/euler/torch/`, verifies their checksum, and stages them in the Euler
runtime cache. This is required because the policy uses a pretrained backbone
and Euler compute nodes cannot reach the public PyTorch download URL.

## Full training

Once the chosen environment count is known, submit the full command explicitly:

```bash
./euler/submit.sh train \
    --task Grasp-Visual-Servo-RGBD-Direct-v0 \
    --num_envs 64 \
    --max_iterations 5000 \
    --headless \
    --enable_cameras
```

`submit.sh train` uses those same values as defaults when no additional
arguments are supplied.

Full training requests the `EULER_TRAIN_TIME_LIMIT` value from `euler.env`,
currently two days. At the measured 256-environment multi-part throughput,
10,000 epochs require about 32 hours and therefore do not fit in a one-day
allocation. Smoke and probe limits remain 30 minutes and one hour.

The command above is the single-part 50-target task. A fresh multi-part run
uses five physical parts and only the checked-in catalog's 1,012-target train
split. The shared geometry/context network is not load-compatible with older
multi-part checkpoints. Re-run the five-iteration 256-environment probe for the
new architecture, then start this revision without `--checkpoint`:

```bash
./euler/submit.sh train \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --gpu-type rtx_4090 \
    --gpu-count 4 \
    --gpu-memory 20G \
    --num_envs 256 \
    --max_iterations 3000 \
    --headless \
    --enable_cameras
```

## Results and durability

All durable run output is written directly to:

```text
/cluster/scratch/gimenol/grasping-rl-runs/
├── rl_games/grasp_visual_servo_rgbd/<timestamp>/
│   ├── nn/*.pth                 policy checkpoints
│   ├── params/{agent,env}.yaml exact run configuration
│   └── events.*                 TensorBoard feedback
├── rl_games/grasp_visual_servo_rgbd_multipart/<timestamp>/
│   └── ...                      multi-part policy artifacts
├── metrics/gpu-JOB_ID.csv       one-second GPU samples
├── metrics/gpu-JOB_ID.summary.txt
├── metrics/gpu-JOB_ID.training-memory.json
├── metrics/memory-JOB_ID/gpu_memory_rank_*.csv
├── slurm-JOB_ID.out             stdout
└── slurm-JOB_ID.err             stderr
```

These files remain after the compute node closes because the directory is on
Euler global scratch, not node-local storage. `$TMPDIR` is deleted at job end;
it contains only the copied SIF and disposable runtime caches. A failed job
also leaves its Slurm logs, GPU samples, and any checkpoints already written.
If a job is killed before its first checkpoint interval, there may be no
weights yet.

Download everything to local `logs/euler/` promptly:

```bash
./euler/pull_results.sh
find logs/euler -type f -name '*.pth'
tensorboard --logdir logs/euler/rl_games
```

Every submission prints a job-specific watcher command. Run it in a local
terminal to wait for Slurm and pull results automatically, for example:

```bash
./euler/watch_and_pull.sh JOB_ID
```

The watcher downloads results after both successful and failed jobs, then
returns a failure status when Slurm did not report `COMPLETED` or the expected
application completion marker is absent. This second check matters because an
Isaac Sim launcher can mask a child Python failure and still return zero. The
local PC must remain powered on, awake, network-connected, and able to SSH to Euler.
While training is active it reports the current/total epoch, percentage,
total FPS, and an ETA estimated from observed progress. The first estimate
includes simulator startup and becomes more accurate after another poll. It
normally reads unbuffered RL-Games output and falls back to incrementally
copying the active TensorBoard event file when an already-running job has
buffered its epoch lines.
Without a local watcher, the compute node cannot initiate a transfer back to a
PC behind SSH/NAT; the durable files remain in Euler global scratch until
`pull_results.sh` is run later.

For a multi-part run, use the validation-aware watcher instead of
`watch_and_pull.sh`:

```bash
./euler/watch_validate_and_pull.sh JOB_ID 1000
```

At epochs 1,000, 2,000, and so on, it pulls only that checkpoint and evaluates
all 125 held-out validation targets in far/mid/close conditions on this PC's
GPU while Euler keeps training. Keep the PC awake and its RTX GPU free. Reports
are written under the run's
`evaluations/periodic_validation/epoch_<N>/`. The selector ranks checkpoints by
validation success with penalties for unsafe collision, premature completion,
divergence, and timeout, then writes `checkpoint_selection.json`,
`checkpoint_selection.md`, and `best_checkpoint.txt`. It never evaluates the
119-target test split. When training ends it also pulls all remaining Euler
artifacts. While training is active, the watcher reports epoch progress, FPS,
training ETA, and the next validation epoch with its remaining epoch distance.
`local_policy.sh` and the full benchmark prefer the selected checkpoint
automatically.

If Slurm reports `COMPLETED` but the application failed before creating an
RL-Games run, the validation-aware watcher exits with an error and points to
the pulled `logs/euler/slurm-JOB_ID.{out,err}` files instead of printing an
empty result directory.

Euler scratch is not backed up and files older than 15 days are purged. Treat
it as persistent across compute jobs, not as archival storage.

## Local policy inspection

After pulling a completed run, `local_policy.sh` automatically selects the
held-out-validation winner when present and otherwise falls back to the newest
downloaded RL-Games best checkpoint. Show TensorBoard at
`http://localhost:6006`, record one deterministic playback, run the full
50-target evaluation, or make composite debug videos with:

```bash
./euler/local_policy.sh tensorboard
./euler/local_policy.sh play
./euler/local_policy.sh evaluate
./euler/local_policy.sh debug-videos
```

Set `ISAAC_RL_CHECKPOINT=/absolute/path/to/checkpoint.pth` to evaluate a
specific checkpoint. Additional arguments are appended after the defaults, so
for example `./euler/local_policy.sh play --target_index 4 --video_length 900`
selects a different target and longer recording.

The single-part evaluator covers all 50 training targets because that catalog
does not have held-out splits. After pulling a multi-part run, evaluate its 125
held-out validation targets with:

```bash
ISAAC_RL_POLICY=multipart ./euler/local_policy.sh evaluate
```

This automatically selects the held-out-validation winner when one exists,
otherwise the newest multi-part training checkpoint, and the multi-part play
task. Add `--catalog_split test` only for the final 119-target held-out test
result, after checkpoint selection is complete.
