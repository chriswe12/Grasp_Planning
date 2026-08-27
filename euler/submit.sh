#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

mode="${1:-}"
case "${mode}" in
    smoke|probe|train)
        shift
        ;;
    *)
        echo "usage: $0 {smoke|probe|train} [arguments...]" >&2
        exit 2
        ;;
esac

gpu_type="${EULER_GPU_TYPE:-rtx_4090}"
gpu_count="${EULER_GPU_COUNT:-1}"
gpu_memory="${EULER_GPUMEM:-20G}"
cpus_per_gpu="${EULER_CPUS_PER_GPU:-8}"
memory_per_cpu="${EULER_MEM_PER_CPU:-3G}"
requested_time=""
job_label=""
training_args=()
while (( $# > 0 )); do
    case "${1}" in
        --gpu-type)
            [[ $# -ge 2 ]] || { echo "[ERROR] --gpu-type requires a value" >&2; exit 2; }
            gpu_type="${2}"
            shift 2
            ;;
        --gpu-count)
            [[ $# -ge 2 ]] || { echo "[ERROR] --gpu-count requires a value" >&2; exit 2; }
            gpu_count="${2}"
            shift 2
            ;;
        --gpu-memory)
            [[ $# -ge 2 ]] || { echo "[ERROR] --gpu-memory requires a value" >&2; exit 2; }
            gpu_memory="${2}"
            shift 2
            ;;
        --cpus-per-gpu)
            [[ $# -ge 2 ]] || { echo "[ERROR] --cpus-per-gpu requires a value" >&2; exit 2; }
            cpus_per_gpu="${2}"
            shift 2
            ;;
        --memory-per-cpu)
            [[ $# -ge 2 ]] || { echo "[ERROR] --memory-per-cpu requires a value" >&2; exit 2; }
            memory_per_cpu="${2}"
            shift 2
            ;;
        --time-limit)
            [[ $# -ge 2 ]] || { echo "[ERROR] --time-limit requires a value" >&2; exit 2; }
            requested_time="${2}"
            shift 2
            ;;
        --job-label)
            [[ $# -ge 2 ]] || { echo "[ERROR] --job-label requires a value" >&2; exit 2; }
            job_label="${2}"
            shift 2
            ;;
        *)
            training_args+=("${1}")
            shift
            ;;
    esac
done
set -- "${training_args[@]}"

if [[ ! "${gpu_count}" =~ ^[1-6]$ ]]; then
    echo "[ERROR] --gpu-count must be between 1 and the six-GPU shareholder limit" >&2
    exit 2
fi
if [[ ! "${cpus_per_gpu}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --cpus-per-gpu must be a positive integer" >&2
    exit 2
fi

if [[ "${mode}" == "smoke" && "${gpu_count}" != "1" ]]; then
    echo "[ERROR] Smoke mode is simulator-only and requires --gpu-count 1." >&2
    exit 2
fi
for argument in "$@"; do
    if [[ "${argument}" == "--distributed" ]]; then
        echo "[ERROR] Do not pass --distributed directly; --gpu-count greater than one configures the Slurm ranks automatically." >&2
        exit 2
    fi
done
cpu_count=$((gpu_count * cpus_per_gpu))
if (( cpu_count > 48 )); then
    echo "[ERROR] Requested ${cpu_count} CPU cores exceeds the 48-core shareholder limit" >&2
    exit 2
fi

if [[ "${EULER_SKIP_SYNC:-0}" == "1" ]]; then
    echo "[INFO] Reusing the already synchronized Euler project (EULER_SKIP_SYNC=1)"
else
    "${SCRIPT_DIR}/sync_project.sh"
fi

if ! ssh "${EULER_LOGIN}" "test -f '${EULER_IMAGE_PATH}'"; then
    echo "[ERROR] Euler image is missing. Run: ${SCRIPT_DIR}/push_image.sh" >&2
    exit 1
fi

job_name="grasp-rgbd-${mode}-${gpu_count}gpu"
if [[ -n "${job_label}" ]]; then
    if [[ ! "${job_label}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "[ERROR] --job-label may contain only letters, digits, dot, underscore, and hyphen" >&2
        exit 2
    fi
    job_name="grasp-${job_label}"
fi
time_limit="${EULER_TRAIN_TIME_LIMIT:-2-00:00:00}"
if [[ "${mode}" == "smoke" ]]; then
    time_limit=00:30:00
elif [[ "${mode}" == "probe" ]]; then
    time_limit=01:00:00
fi
if [[ -n "${requested_time}" ]]; then
    time_limit="${requested_time}"
fi

remote_command=(
    sbatch
    --account="${EULER_ACCOUNT}"
    --job-name="${job_name}"
    --time="${time_limit}"
    --nodes=1
    --ntasks="${gpu_count}"
    --cpus-per-task="${cpus_per_gpu}"
    --mem-per-cpu="${memory_per_cpu}"
    --gpus="${gpu_type}:${gpu_count}"
    --gres="gpumem:${gpu_memory}"
    --chdir="${EULER_PROJECT_DIR}"
    --export="ALL,EULER_CONFIG_PATH=${EULER_PROJECT_DIR}/euler/euler.env,EULER_REQUESTED_GPU_TYPE=${gpu_type},EULER_REQUESTED_GPU_COUNT=${gpu_count},EULER_REQUESTED_GPUMEM=${gpu_memory}"
    --output="${EULER_RUNS_DIR}/slurm-%j.out"
    --error="${EULER_RUNS_DIR}/slurm-%j.err"
    "${EULER_PROJECT_DIR}/euler/job.sbatch"
    "${mode}"
    "$@"
)
printf -v quoted_remote_command '%q ' "${remote_command[@]}"

echo "[INFO] Submitting ${mode} job to account ${EULER_ACCOUNT}: gpu=${gpu_type}:${gpu_count} gpumem=${gpu_memory} cpus=${cpu_count} time=${time_limit}"
submission_output="$(ssh "${EULER_LOGIN}" "${quoted_remote_command}")"
echo "${submission_output}"

if [[ "${submission_output}" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
    job_id="${BASH_REMATCH[1]}"
    echo "[INFO] Monitor and automatically pull when finished:"
    echo "${SCRIPT_DIR}/watch_and_pull.sh ${job_id}"
    if [[ "${mode}" == "train" && " $* " == *MultiPart* ]]; then
        echo "[INFO] Or periodically validate saved checkpoints on this PC and select the best:"
        echo "${SCRIPT_DIR}/watch_validate_and_pull.sh ${job_id} 1000"
    fi
fi
