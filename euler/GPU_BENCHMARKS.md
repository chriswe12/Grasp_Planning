# Euler GPU Capacity Benchmarks

Last updated: 2026-08-20

This file records measured capacity for the camera-enabled multipart PPO task.
Do not infer a production environment count solely from nominal GPU memory.
Isaac Sim renderer allocations, PPO tensors, host RAM, GPU architecture, and
throughput all affect the useful limit.

## Benchmark contract

All new comparison jobs use the same synchronized checkout and command:

```bash
./euler/submit.sh probe \
    --gpu-type GPU_TYPE \
    --gpu-memory GPU_MEMORY \
    --time-limit 00:30:00 \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs NUM_ENVS \
    --max_iterations 5 \
    --headless \
    --enable_cameras
```

Peak VRAM is sampled once per second. Stable FPS below excludes the first PPO
iteration because it includes additional warm-up. Five iterations are enough
to compare startup, PPO execution, and approximate peak memory, but they cannot
rule out gradual allocator or distributed-communication growth. Long probes
also write `gpu_memory_rank_<RANK>.csv` and receive an automatic slope verdict.

## Confirmed results

| GPU | Job | Envs | Peak / total VRAM | Peak use | Stable total FPS | Training time | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| RTX 4090 | 10402203 | 192 | 18,808 / 24,564 MiB | 76.6% | 1,131--1,174 | 61.14 s | completed |
| RTX 4090 | 10404371 | 256 | 22,732 / 24,564 MiB | 92.5% | 1,368--1,408 | 67.82 s | completed |
| RTX 4090 | 11319881 | 256 | 22,656 / 24,564 MiB | 92.2% | 1,249--1,281 | 75.9 s | completed with current randomization |

The task fits **256 envs** on a 24 GB RTX 4090 with about 1.8 GiB remaining.
That is the intended production count after the reusable distributed-gradient
buffers pass the long probe below. Until that probe reports `PASS`, 224 remains
the conservative fallback rather than evidence that the old growth is fixed.

## Long-run four-GPU memory proof

Job 11329112 demonstrated why a five-epoch capacity probe is insufficient:
three ranks gradually reached 24,080 MiB and the run failed near epoch 1,095,
while one rank stayed near 22,404 MiB. The training integration now reuses the
actor and central-value all-reduce buffers, enables expandable allocator
segments, and records synchronized allocator plus whole-device VRAM at every
epoch. Follow-up job 11408451 exposed the remaining rank-asymmetric source:
RL-Games' stock Isaac observer retained CUDA-backed episode dictionaries on
nonzero ranks because only rank zero executes the statistics callback that
clears them. The training entrypoint now uses a distributed-safe observer that
collects episode logging tensors only on rank zero.

Run the proof at the desired production count:

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

After pulling the job, inspect:

```bash
cat logs/euler/metrics/gpu-JOB_ID.training-memory.json
```

`PASS` requires at least 150 post-warm-up samples on every rank, no more than
0.50 MiB/epoch fitted whole-device growth, and at least 1,024 MiB free at every
sample. The old approximately 2 MiB/epoch regression would add roughly 350 MiB
over this analysis window, so 200 epochs are sufficient to detect that same
failure mode without repeating a near-production run. `INSUFFICIENT_SAMPLES`
is expected for ordinary five-epoch probes. `FAIL` makes the Euler job fail
even when the training process itself reached its requested final epoch.

## Active target-GPU probes

These jobs were submitted from the same immutable project snapshot after one
full synchronization. Their status and queue estimates are transient.

| GPU request | Job | Envs | Status at 2026-08-20 | Purpose |
|---|---:|---:|---|---|
| RTX PRO 6000 96 GB | 11319924 | 512 | pending; estimated Aug 23 08:25 | conservative target baseline |
| RTX PRO 6000 96 GB | 11319963 | 1,024 | pending; estimated Aug 23 08:55 | expected useful range |
| RTX PRO 6000 96 GB | 11319992 | 1,280 | pending; estimated Aug 23 09:25 | upper-capacity/throughput probe |
| A100 PCIe 80 GB | 11320012 | 256 | pending; estimated Aug 22 04:20 | architecture and renderer comparison |

## Measured single-node distributed probes

These use the current one-Slurm-task-per-GPU launch contract and 224
environments **per GPU**. The task-level GPU cgroup is required because the
USD-RT RGB-D renderer only accepts its sole visible card as logical `cuda:0`.

| GPU request | Job | Envs/GPU | Total envs | Global rollout | Stable total FPS | Scaling | Peak VRAM/GPU | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1x RTX 4090 | 11319881 | 256 | 256 | 16,384 | 1,263 | 1.00x | 22,656 MiB (92.2%) | completed |
| 2x RTX 4090 | 11327083 | 224 | 448 | 28,672 | 2,243 | 1.78x | 21,992--22,328 MiB (89.5--90.9%) | completed, 2/2 ranks |
| 4x RTX 4090 | 11327086 | 224 | 896 | 57,344 | 4,506 | 3.57x | 21,874--22,354 MiB (89.0--91.0%) | completed, 4/4 ranks |

Stable FPS is the mean of epochs 2--5, excluding the first warm-up epoch. The
one-GPU baseline has a slightly larger per-GPU environment batch. Normalized by
total environments, the three results are 4.93, 5.01, and 5.03 FPS per
environment respectively, so distributed synchronization did not introduce a
meaningful steady-state throughput penalty in these probes.

The first torchrun implementation exposed every allocated Vulkan device to
every Isaac process. Rank one then failed in the USD-RT camera path with `GPU 1
requested`, while changing `CUDA_VISIBLE_DEVICES` inside the child process was
too late for Vulkan initialization. The final launcher allocates one Slurm task
and one GPU cgroup per rank before Apptainer starts. Each task consequently sees
one card as logical `cuda:0`, while RL-Games still receives distinct global
ranks and synchronizes gradients through NCCL.

The four-GPU job showed about 100 seconds of one-time startup skew because
multiple Kit processes contended for the shared key-value/cache directory; Kit
disabled the locked database for three processes and continued. This did not
affect the steady PPO epochs or completion, and is negligible for a long run.
All four rank timers completed; the fastest two reported 83--92 seconds and
the earlier-started ranks included their collective wait (about 196 seconds).

The local 4090 measurements imply roughly 52 MiB of additional VRAM per
environment after a large fixed allocation. That predicts about 62 GiB at
1,024 envs and 75 GiB at 1,280 envs, but these are explicitly **unconfirmed**.
The RTX PRO probes are required to establish the real ceiling and to determine
whether additional environments improve samples per second or merely increase
PPO/update latency. The A100 comparison starts at 256 envs because its compute
and ray-tracing balance differs from Ada RTX GPUs.

Inspect or pull an active result with:

```bash
ssh euler-gimenol 'squeue -j 11319924,11319963,11319992,11320012'
./euler/watch_and_pull.sh JOB_ID
cat logs/euler/metrics/gpu-JOB_ID.summary.txt
```

For one policy, four RTX 4090 GPUs are the measured throughput choice when the
allocation is available. Two GPUs are a useful lower-queue-cost alternative.
The configured 1,024-sample PPO minibatch is now interpreted as a target
effective global minibatch: four ranks use 256 samples each before averaging
their gradients. At 224 environments/rank this produces 112 synchronized
optimizer updates per epoch, preserving the old single-GPU update-to-sample
ratio. Compare runs using total frames, optimizer updates, and held-out
validation rather than epoch count alone.

## Resource selection

The default remains one RTX 4090. Select a target GPU per submission:

```bash
# RTX PRO 6000, 96 GB
./euler/submit.sh probe \
    --gpu-type rtx_pro_6000 --gpu-memory 90G \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 1024 --max_iterations 5 --headless --enable_cameras

# A100 PCIe, 80 GB
./euler/submit.sh probe \
    --gpu-type a100_80gb_pcie --gpu-memory 75G \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0 \
    --num_envs 256 --max_iterations 5 --headless --enable_cameras
```

`--gpu-memory` is the minimum scheduler constraint, not an application memory
limit. The exact assigned GPU and total VRAM are recorded in the Slurm output.
For `--gpu-count N`, the wrapper starts `N` Slurm tasks and RL-Games ranks on
one node, and `--num_envs` applies to every rank. The summary records peak
memory separately for each GPU. Independent one-GPU jobs remain available for
parallel seeds and ablations without changing PPO's global rollout batch.
