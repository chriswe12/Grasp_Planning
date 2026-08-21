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

requested_gpu_count="${EULER_REQUESTED_GPU_COUNT:-1}"
if [[ ! "${requested_gpu_count}" =~ ^[1-6]$ ]]; then
    echo "[ERROR] Invalid EULER_REQUESTED_GPU_COUNT=${requested_gpu_count}" >&2
    exit 2
fi
if [[ "${mode}" == "smoke" && "${requested_gpu_count}" != "1" ]]; then
    echo "[ERROR] Smoke mode does not use distributed training and requires one GPU." >&2
    exit 2
fi

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
echo "[INFO] Requested GPU: ${EULER_REQUESTED_GPU_TYPE:-script-default}:${EULER_REQUESTED_GPU_COUNT:-1}, gpumem=${EULER_REQUESTED_GPUMEM:-20G}"
visible_gpu_selector="${CUDA_VISIBLE_DEVICES:-}"
gpu_filter_args=()
if [[ -n "${visible_gpu_selector}" ]]; then
    IFS=',' read -r -a visible_gpu_ids <<<"${visible_gpu_selector}"
    if (( ${#visible_gpu_ids[@]} != requested_gpu_count )); then
        echo "[ERROR] Slurm exposed ${#visible_gpu_ids[@]} GPUs but ${requested_gpu_count} were requested." >&2
        exit 1
    fi
    gpu_filter_args+=(--id="${visible_gpu_selector}")
fi
nvidia-smi "${gpu_filter_args[@]}" --query-gpu=index,name,uuid,memory.total,driver_version \
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
nvidia_smi_args=(
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu
    --format=csv,noheader,nounits
    --loop=1
)
nvidia_smi_args+=("${gpu_filter_args[@]}")
nvidia-smi "${nvidia_smi_args[@]}" >"${gpu_metrics}" &
gpu_sampler_pid=$!

finish_job() {
    kill "${gpu_sampler_pid}" 2>/dev/null || true
    wait "${gpu_sampler_pid}" 2>/dev/null || true
    if [[ -s "${gpu_metrics}" ]]; then
        awk -F, '
            {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5)
                gpu = $2
                used = $4 + 0
                if (!(gpu in seen)) {
                    seen[gpu] = 1
                    gpu_count += 1
                }
                if (used > gpu_peak[gpu] + 0) {
                    gpu_peak[gpu] = used
                    gpu_total[gpu] = $5
                }
                if (used > peak + 0) {
                    peak = used
                    total = $5
                    peak_line = $0
                }
            }
            END {
                if (peak_line != "") {
                    printf "gpu_count=%s\npeak_memory_mib=%s\ntotal_memory_mib=%s\npeak_sample=%s\n", gpu_count, peak, total, peak_line
                    for (gpu in seen) {
                        printf "gpu_%s_peak_memory_mib=%s\ngpu_%s_total_memory_mib=%s\n", gpu, gpu_peak[gpu], gpu, gpu_total[gpu]
                    }
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
experiment_name="$(date '+%Y-%m-%d_%H-%M-%S')_job_${SLURM_JOB_ID}_${requested_gpu_count}gpu"
export APPTAINERENV_ISAAC_RL_EXPERIMENT_NAME="${experiment_name}"
export APPTAINERENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR="$(hostname)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

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
            set -- \
                --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0 \
                --steps 10 --num_envs 2 --headless
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

runtime_command=("${command_path}" "$@")
if (( requested_gpu_count > 1 )); then
    runtime_command+=(--distributed)
fi

runtime_exec=(apptainer "${apptainer_args[@]}" "${runtime_command[@]}")
if (( requested_gpu_count > 1 )); then
    runtime_exec=(
        srun
        --exact
        --kill-on-bad-exit=1
        --ntasks="${requested_gpu_count}"
        --cpus-per-task="${SLURM_CPUS_PER_TASK}"
        --gpus-per-task=1
        --gpu-bind=single:1
        euler/slurm_rank_launcher.sh
        "${runtime_exec[@]}"
    )
fi

echo "[INFO] Running mode=${mode}: gpu_processes=${requested_gpu_count} ${runtime_exec[*]}"

# Isaac Sim's python.sh has returned zero for a failed child interpreter in the
# past. Require an explicit import marker before paying the simulator startup
# cost, and require a mode-specific completion marker below.
preflight_output="${TMPDIR}/isaaclab-import-preflight.stdout"
set +e
apptainer "${apptainer_args[@]}" -c \
    "import importlib.util; import grasp_planning; assert importlib.util.find_spec('isaaclab'); assert importlib.util.find_spec('isaac_rl'); print('[PREFLIGHT] project imports OK')" \
    | tee "${preflight_output}"
preflight_status=${PIPESTATUS[0]}
set -e
if (( preflight_status != 0 )) || ! grep -q '^\[PREFLIGHT\] project imports OK$' "${preflight_output}"; then
    echo "[ERROR] Container project import preflight failed" >&2
    exit 1
fi

runtime_status=0
if [[ "${mode}" == "smoke" ]]; then
    smoke_output="${TMPDIR}/isaaclab-smoke.stdout"
    set +e
    "${runtime_exec[@]}" | tee "${smoke_output}"
    runtime_status=${PIPESTATUS[0]}
    set -e
    if ! grep -q '^\[SMOKE\] steps=' "${smoke_output}"; then
        echo "[ERROR] Smoke process exited without its completion marker" >&2
        runtime_status=1
    fi
else
    training_output="${TMPDIR}/isaaclab-${mode}.stdout"
    set +e
    "${runtime_exec[@]}" | tee "${training_output}"
    runtime_status=${PIPESTATUS[0]}
    set -e
    completion_count="$(grep -c '^Training time: ' "${training_output}" || true)"
    if (( completion_count < requested_gpu_count )); then
        echo "[ERROR] ${mode^} completed ${completion_count}/${requested_gpu_count} distributed ranks" >&2
        runtime_status=1
    fi

    # A short capacity probe can establish peak allocation, but only hundreds
    # of PPO epochs can rule out the gradual distributed-memory growth that
    # killed job 11329112. Preserve each rank's epoch samples and make the
    # stability verdict part of the Slurm output.
    memory_run_dir="$(find "${runtime_logs_dir}/rl_games" -type d -name "${experiment_name}" -print -quit 2>/dev/null || true)"
    if [[ -n "${memory_run_dir}" ]]; then
        durable_memory_dir="${EULER_RUNS_DIR}/metrics/memory-${SLURM_JOB_ID}"
        mkdir -p "${durable_memory_dir}"
        find "${memory_run_dir}" -maxdepth 1 -type f -name 'gpu_memory_rank_*.csv' \
            -exec cp -t "${durable_memory_dir}" {} +
        memory_analysis_status=0
        python3 euler/analyze_training_memory.py \
            "${memory_run_dir}" \
            --output "${EULER_RUNS_DIR}/metrics/gpu-${SLURM_JOB_ID}.training-memory.json" \
            || memory_analysis_status=$?
        if (( memory_analysis_status != 0 )); then
            echo "[ERROR] Long-run GPU-memory stability check failed" >&2
            runtime_status=1
        fi
    else
        echo "[GPU_MEMORY][WARNING] No RL-Games experiment directory found for ${experiment_name}" >&2
    fi
fi
exit "${runtime_status}"
