#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:20G
#SBATCH --time=24:00:00
#SBATCH --tmp=50G

set -euo pipefail

if [[ -z "${EULER_CONFIG_PATH:-}" ]]; then
    echo "[ERROR] EULER_CONFIG_PATH was not exported by euler/submit.sh" >&2
    exit 1
fi
# shellcheck source=euler.env
source "${EULER_CONFIG_PATH}"

mode="${1:-}"
if [[ -z "${mode}" ]]; then
    echo "usage: job.sbatch {smoke|probe|train} [arguments...]" >&2
    exit 2
fi
shift

if [[ ! -f "${EULER_IMAGE_PATH}" ]]; then
    echo "[ERROR] Apptainer image not found: ${EULER_IMAGE_PATH}" >&2
    exit 1
fi

mkdir -p "${EULER_CACHE_DIR}" "${EULER_RUNS_DIR}/metrics"

# Probe runs only measure startup, throughput, and VRAM. Keep their RL-Games
# and Hydra artifacts on node-local storage; Slurm output and GPU metrics remain
# durable in EULER_RUNS_DIR. Full training writes all artifacts to global scratch.
if [[ "${mode}" == "probe" ]]; then
    runtime_logs_dir="${TMPDIR}/probe-logs"
    runtime_outputs_dir="${TMPDIR}/probe-outputs"
else
    runtime_logs_dir="${EULER_RUNS_DIR}"
    runtime_outputs_dir="${EULER_RUNS_DIR}/hydra-outputs"
fi
mkdir -p "${runtime_logs_dir}" "${runtime_outputs_dir}"

local_image="${TMPDIR}/$(basename "${EULER_IMAGE_PATH}")"
local_cache="${TMPDIR}/docker-isaac-sim"

echo "[INFO] Node: $(hostname)"
echo "[INFO] CUDA allocation: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
    --format=csv,noheader
echo "[INFO] Copying the Apptainer image to node-local storage"
cp "${EULER_IMAGE_PATH}" "${local_image}"

mkdir -p \
    "${local_cache}/cache/kit" \
    "${local_cache}/cache/ov" \
    "${local_cache}/cache/pip" \
    "${local_cache}/cache/torch" \
    "${local_cache}/cache/glcache" \
    "${local_cache}/cache/computecache" \
    "${local_cache}/logs" \
    "${local_cache}/data" \
    "${local_cache}/documents"

if [[ -d "${EULER_CACHE_DIR}" ]]; then
    rsync -a "${EULER_CACHE_DIR}/" "${local_cache}/"
fi

gpu_metrics="${EULER_RUNS_DIR}/metrics/gpu-${SLURM_JOB_ID}.csv"
gpu_summary="${EULER_RUNS_DIR}/metrics/gpu-${SLURM_JOB_ID}.summary.txt"
gpu_selector="${CUDA_VISIBLE_DEVICES:-}"
gpu_selector="${gpu_selector%%,*}"
nvidia_smi_args=(
    --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu
    --format=csv,noheader,nounits
    --loop=1
)
if [[ -n "${gpu_selector}" ]]; then
    nvidia_smi_args+=(--id="${gpu_selector}")
fi
nvidia-smi "${nvidia_smi_args[@]}" >"${gpu_metrics}" &
gpu_sampler_pid=$!

finish_job() {
    kill "${gpu_sampler_pid}" 2>/dev/null || true
    wait "${gpu_sampler_pid}" 2>/dev/null || true
    if [[ -s "${gpu_metrics}" ]]; then
        awk -F, '
            {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
                if ($3 + 0 > peak + 0) {
                    peak = $3
                    total = $4
                    peak_line = $0
                }
            }
            END {
                if (peak_line != "") {
                    printf "peak_memory_mib=%s\ntotal_memory_mib=%s\npeak_sample=%s\n", peak, total, peak_line
                }
            }
        ' "${gpu_metrics}" | tee "${gpu_summary}"
    fi
    rsync -a "${local_cache}/" "${EULER_CACHE_DIR}/" || true
}
trap finish_job EXIT

export APPTAINERENV_ACCEPT_EULA=Y
export APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONPATH="${EULER_CONTAINER_PYTHONPATH}"
export APPTAINERENV_TERM=xterm

apptainer_args=(
    exec
    --nv
    --containall
    --writable-tmpfs
    --bind "${EULER_PROJECT_DIR}:/workspace/grasping_rl:ro"
    --bind "${runtime_logs_dir}:/workspace/grasping_rl/logs:rw"
    --bind "${runtime_outputs_dir}:/workspace/grasping_rl/outputs:rw"
    --bind "${local_cache}/cache/kit:/isaac-sim/kit/cache:rw"
    --bind "${local_cache}/cache/ov:${HOME}/.cache/ov:rw"
    --bind "${local_cache}/cache/pip:${HOME}/.cache/pip:rw"
    --bind "${local_cache}/cache/torch:${HOME}/.cache/torch:rw"
    --bind "${local_cache}/cache/glcache:${HOME}/.cache/nvidia/GLCache:rw"
    --bind "${local_cache}/cache/computecache:${HOME}/.nv/ComputeCache:rw"
    --bind "${local_cache}/logs:${HOME}/.nvidia-omniverse/logs:rw"
    --bind "${local_cache}/data:${HOME}/.local/share/ov/data:rw"
    --bind "${local_cache}/documents:${HOME}/Documents:rw"
    --pwd /workspace/grasping_rl
    "${local_image}"
    /isaac-sim/python.sh
)

case "${mode}" in
    smoke)
        if (( $# == 0 )); then
            set -- --steps 10 --num_envs 2 --headless
        fi
        command_path=isaac_rl/scripts/smoke_env.py
        ;;
    probe)
        if (( $# > 0 )) && [[ "${1}" == --* ]]; then
            # Full train.py arguments allow probing another registered task,
            # including the held-out-split multi-part task.
            :
        else
            env_count="${1:-}"
            iterations="${2:-5}"
            if [[ ! "${env_count}" =~ ^[1-9][0-9]*$ ]]; then
                echo "[ERROR] Probe requires a positive environment count" >&2
                exit 2
            fi
            if [[ ! "${iterations}" =~ ^[1-9][0-9]*$ ]]; then
                echo "[ERROR] Probe iterations must be a positive integer" >&2
                exit 2
            fi
            set -- \
                --task Grasp-Visual-Servo-RGBD-Direct-v0 \
                --num_envs "${env_count}" \
                --max_iterations "${iterations}" \
                --headless \
                --enable_cameras
        fi
        command_path=isaac_rl/scripts/rl_games/train.py
        ;;
    train)
        if (( $# == 0 )); then
            set -- \
                --task Grasp-Visual-Servo-RGBD-Direct-v0 \
                --num_envs 64 \
                --max_iterations 5000 \
                --headless \
                --enable_cameras
        fi
        command_path=isaac_rl/scripts/rl_games/train.py
        ;;
    *)
        echo "[ERROR] Unknown mode: ${mode}" >&2
        exit 2
        ;;
esac

echo "[INFO] Running mode=${mode}: ${command_path} $*"
runtime_status=0
if [[ "${mode}" == "smoke" ]]; then
    smoke_output="${TMPDIR}/isaaclab-smoke.stdout"
    set +e
    apptainer "${apptainer_args[@]}" "${command_path}" "$@" | tee "${smoke_output}"
    runtime_status=${PIPESTATUS[0]}
    set -e
    if ! grep -q '^\[SMOKE\] steps=' "${smoke_output}"; then
        echo "[ERROR] Smoke process exited without its completion marker" >&2
        runtime_status=1
    fi
else
    set +e
    apptainer "${apptainer_args[@]}" "${command_path}" "$@"
    runtime_status=$?
    set -e
fi
exit "${runtime_status}"
