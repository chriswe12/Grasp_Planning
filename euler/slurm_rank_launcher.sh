#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: slurm_rank_launcher.sh CACHE_ROOT COMMAND [arguments...]" >&2
    exit 2
fi

cache_root="${1}"
shift
if [[ "${cache_root}" != /* ]]; then
    echo "[ERROR] Distributed cache root must be an absolute path: ${cache_root}" >&2
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

# Each Isaac process needs private writable Kit, OV, shader, and user-data
# directories. A job-local cache shared by ranks is still unsafe: concurrent
# Kit processes contend for the same key-value database and can abort inside a
# native allocator. Only the immutable pretrained Torch weights are copied
# from the cache root.
rank_cache="${cache_root}/rank_${SLURM_PROCID}"
mkdir -p \
    "${rank_cache}/cache/kit" \
    "${rank_cache}/cache/ov" \
    "${rank_cache}/cache/pip" \
    "${rank_cache}/cache/torch" \
    "${rank_cache}/cache/glcache" \
    "${rank_cache}/cache/computecache" \
    "${rank_cache}/logs" \
    "${rank_cache}/data" \
    "${rank_cache}/documents"
if [[ -d "${cache_root}/cache/torch" ]]; then
    rsync -a "${cache_root}/cache/torch/" "${rank_cache}/cache/torch/"
fi

runtime_args=()
for runtime_arg in "$@"; do
    if [[ "${runtime_arg}" == "${cache_root}"* ]]; then
        runtime_arg="${rank_cache}${runtime_arg#"${cache_root}"}"
    fi
    runtime_args+=("${runtime_arg}")
done

echo "[EULER_DISTRIBUTED] global_rank=${SLURM_PROCID}/${SLURM_NTASKS} original_local_rank=${SLURM_LOCALID} physical_gpu=${task_gpu} logical_device=cuda:0 cache=${rank_cache}"

if [[ -n "${APPTAINERENV_FRANKA_CUDNN_MODE:-}" && "${runtime_args[0]}" == apptainer ]]; then
    runtime_args=(bash "$(dirname "$0")/franka_apptainer.sh" "${runtime_args[@]:1}")
fi
exec "${runtime_args[@]}"
