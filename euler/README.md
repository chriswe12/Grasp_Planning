# ETH Euler deployment

This directory deploys the Isaac Lab 2.3.2 RGB-D task to ETH Euler without
copying the local training-log tree. The immutable Apptainer image holds Isaac
Sim and Isaac Lab. The external `isaac_rl` project is synchronized separately
and made importable through `PYTHONPATH` at job startup.

The configured Slurm job stays within the `mavt-pdz-euler-student` limits on
the `es_meboldt` share: 8 CPU cores, 24 GiB system RAM, and one GPU. It asks for
a GPU with at least 20 GiB of memory because the task uses RGB-D cameras.

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
   command through Apptainer.
5. The source checkout is mounted read-only, but its `logs/` directory is
   replaced by a writable bind mount to persistent Euler scratch. Checkpoints,
   TensorBoard events, Slurm logs, and GPU measurements therefore survive the
   compute node shutting down.

## First-time preparation

Verify SSH authentication first:

```bash
ssh cwellan@euler.ethz.ch
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
ssh cwellan@euler.ethz.ch 'squeue --me'
ssh cwellan@euler.ethz.ch 'tail -n 100 /cluster/scratch/cwellan/grasping-rl-runs/slurm-JOB_ID.out'
ssh cwellan@euler.ethz.ch 'tail -n 100 /cluster/scratch/cwellan/grasping-rl-runs/slurm-JOB_ID.err'
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
ssh cwellan@euler.ethz.ch 'cat /cluster/scratch/cwellan/grasping-rl-runs/metrics/gpu-JOB_ID.summary.txt'
```

There is no reliable environment count that can be calculated in advance:
camera render targets, simulator buffers, the ResNet, and PPO batches do not
all scale linearly. Choose the largest count that completes with roughly
10-20% VRAM still free, then compare
`performance/step_inference_rl_update_fps` in TensorBoard. The fastest safe
count is more useful than the count that merely fills the most VRAM. The
one-second sampler can miss a very short peak, so do not use the last few MiB.

The job requests at least 20 GiB, not a specific GPU model. The job log records
the exact assigned GPU and its total memory. Compare probes from the same GPU
memory class before using their results to size a full run.

The five-iteration probes normally do not produce weights: the current agent
configuration starts best-model saving after iteration 20 and saves periodic
checkpoints every 50 iterations. Their purpose is task, PPO, and VRAM
validation. Probe RL-Games and Hydra artifacts are written to node-local
`$TMPDIR` and discarded when the allocation ends; only their Slurm output and
GPU metrics remain in global scratch.

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
    --num_envs 256 \
    --max_iterations 10000 \
    --headless \
    --enable_cameras
```

## Results and durability

All durable run output is written directly to:

```text
/cluster/scratch/cwellan/grasping-rl-runs/
├── rl_games/grasp_visual_servo_rgbd/<timestamp>/
│   ├── nn/*.pth                 policy checkpoints
│   ├── params/{agent,env}.yaml exact run configuration
│   └── events.*                 TensorBoard feedback
├── rl_games/grasp_visual_servo_rgbd_multipart/<timestamp>/
│   └── ...                      multi-part policy artifacts
├── metrics/gpu-JOB_ID.csv       one-second GPU samples
├── metrics/gpu-JOB_ID.summary.txt
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
returns a failure status when Slurm did not report `COMPLETED`. The local PC
must remain powered on, awake, network-connected, and able to SSH to Euler.
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
