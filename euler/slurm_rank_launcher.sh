#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    echo "usage: slurm_rank_launcher.sh COMMAND [arguments...]" >&2
    exit 2
fi

: "${SLURM_PROCID:?Slurm did not provide SLURM_PROCID}"
: "${SLURM_LOCALID:?Slurm did not provide SLURM_LOCALID}"
: "${SLURM_NTASKS:?Slurm did not provide SLURM_NTASKS}"
: "${MASTER_ADDR:?job.sbatch did not provide MASTER_ADDR}"
: "${MASTER_PORT:?job.sbatch did not provide MASTER_PORT}"

# srun --gpus-per-task=1 and --gpu-bind=single:1 establish a one-GPU task
# cgroup before Apptainer, Python, CUDA, or Vulkan starts. Forward the task's
# visibility and distributed identity explicitly because Apptainer's --nv
# setup otherwise sanitizes CUDA_VISIBLE_DEVICES.
task_gpu="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${task_gpu}" || "${task_gpu}" == *,* ]]; then
    echo "[ERROR] Slurm rank ${SLURM_PROCID} received CUDA_VISIBLE_DEVICES=${task_gpu:-unset}; expected exactly one GPU." >&2
    exit 1
fi

export APPTAINERENV_CUDA_VISIBLE_DEVICES="${task_gpu}"
export APPTAINERENV_RANK="${SLURM_PROCID}"
export APPTAINERENV_WORLD_SIZE="${SLURM_NTASKS}"
export APPTAINERENV_LOCAL_RANK=0
export APPTAINERENV_ISAAC_RL_ORIGINAL_LOCAL_RANK="${SLURM_LOCALID}"
export APPTAINERENV_ISAAC_RL_SELECTED_GPU="${task_gpu}"
export APPTAINERENV_MASTER_ADDR="${MASTER_ADDR}"
export APPTAINERENV_MASTER_PORT="${MASTER_PORT}"

echo "[EULER_DISTRIBUTED] global_rank=${SLURM_PROCID}/${SLURM_NTASKS} original_local_rank=${SLURM_LOCALID} physical_gpu=${task_gpu} logical_device=cuda:0"
exec "$@"
