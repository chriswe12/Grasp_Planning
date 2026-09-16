#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUITE_ID="${WEEKEND_SUITE_ID:-$(date '+%Y%m%d-%H%M%S')}"
NUM_ENVS="${WEEKEND_NUM_ENVS:-224}"
MAX_ITERATIONS="${WEEKEND_MAX_ITERATIONS:-10000}"
SEED="${WEEKEND_SEED:-42}"
DRY_RUN="${WEEKEND_DRY_RUN:-0}"
GPU_COUNT="${WEEKEND_GPU_COUNT:-4}"
GLOBAL_MINIBATCH_SIZE="${WEEKEND_GLOBAL_MINIBATCH_SIZE:-$((GPU_COUNT * 256))}"
GPU_TYPE="rtx_4090"
GPU_MEMORY="20G"
TIME_LIMIT="2-00:00:00"
TASK="Grasp-Visual-Servo-RGBD-FabricaAll-Direct-v0"
EXPERIMENT_FAMILY="grasp_visual_servo_rgbd_multipart"
MANIFEST="${REPO_ROOT}/euler/weekend_suite_${SUITE_ID}.json"

if [[ ! "${SUITE_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "[ERROR] WEEKEND_SUITE_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
fi
if [[ ! "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] WEEKEND_NUM_ENVS and WEEKEND_MAX_ITERATIONS must be positive integers" >&2
    exit 2
fi
if [[ "${GPU_COUNT}" != "4" && "${GPU_COUNT}" != "6" ]]; then
    echo "[ERROR] WEEKEND_GPU_COUNT must be 4 or 6 for a complete Fabrica shard layout" >&2
    exit 2
fi
if [[ ! "${GLOBAL_MINIBATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] WEEKEND_GLOBAL_MINIBATCH_SIZE must be a positive integer" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "[ERROR] WEEKEND_DRY_RUN must be 0 or 1" >&2
    exit 2
fi

common_args=(
    --gpu-type "${GPU_TYPE}"
    --gpu-count "${GPU_COUNT}"
    --gpu-memory "${GPU_MEMORY}"
    --time-limit "${TIME_LIMIT}"
    --task "${TASK}"
    --num_envs "${NUM_ENVS}"
    --global_minibatch_size "${GLOBAL_MINIBATCH_SIZE}"
    --max_iterations "${MAX_ITERATIONS}"
    --seed "${SEED}"
    --policy-context action
    --headless
    --enable_cameras
)

run_records=()

write_manifest() {
    python3 - "${MANIFEST}" "${SUITE_ID}" "${TASK}" "${EXPERIMENT_FAMILY}" \
        "${GPU_TYPE}" "${GPU_COUNT}" "${NUM_ENVS}" "${MAX_ITERATIONS}" "${SEED}" \
        "${run_records[@]}" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    suite_id,
    task,
    experiment_family,
    gpu_type,
    gpu_count,
    num_envs,
    max_iterations,
    seed,
    *records,
) = sys.argv[1:]
runs = []
for record in records:
    label, job_id, experiment, sim2real_profile, training_profile, policy_context = record.split("|")
    runs.append(
        {
            "label": label,
            "job_id": int(job_id),
            "experiment": experiment,
            "sim2real_profile": sim2real_profile,
            "training_profile": training_profile,
            "policy_context": policy_context,
        }
    )
payload = {
    "suite_id": suite_id,
    "task": task,
    "experiment_family": experiment_family,
    "gpu_type": gpu_type,
    "gpu_count": int(gpu_count),
    "num_envs_per_rank": int(num_envs),
    "max_iterations": int(max_iterations),
    "seed": int(seed),
    "runs": runs,
}
path = Path(output)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

submit_one() {
    local label="$1"
    local sim2real_profile="$2"
    local training_profile="$3"
    local skip_sync="$4"
    local experiment="fabrica_all_weekend_${SUITE_ID}_${label}_seed${SEED}"
    local command=(
        "${SCRIPT_DIR}/submit.sh" train
        --job-label "weekend-${label}"
        "${common_args[@]}"
        --sim2real_profile "${sim2real_profile}"
        --training-profile "${training_profile}"
        --experiment-name "${experiment}"
    )

    echo "[WEEKEND] ${label}: sim2real=${sim2real_profile} training=${training_profile} experiment=${experiment}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'EULER_SKIP_SYNC=%q ' "${skip_sync}"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi

    local output
    if ! output="$(EULER_SKIP_SYNC="${skip_sync}" "${command[@]}")"; then
        printf '%s\n' "${output}"
        echo "[ERROR] Submission failed for ${label}; previously submitted jobs remain in ${MANIFEST}" >&2
        exit 1
    fi
    printf '%s\n' "${output}"
    if [[ ! "${output}" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
        echo "[ERROR] Could not parse the Slurm job ID for ${label}" >&2
        exit 1
    fi
    local job_id="${BASH_REMATCH[1]}"
    run_records+=("${label}|${job_id}|${experiment}|${sim2real_profile}|${training_profile}|action")
    # Keep a usable partial manifest even if a later submission fails.
    write_manifest
}

echo "[INFO] Fabrica weekend suite ${SUITE_ID}"
echo "[INFO] Each run: ${GPU_COUNT}x ${GPU_TYPE}, ${NUM_ENVS} envs/GPU, global minibatch ${GLOBAL_MINIBATCH_SIZE}, ${MAX_ITERATIONS} epochs, seed ${SEED}."
echo "[INFO] A complete run collects $((GPU_COUNT * NUM_ENVS * 64 * MAX_ITERATIONS)) transitions."
echo "[WARN] Euler's ${TIME_LIMIT} limit may stop these 10k-epoch runs early; checkpoints are saved every 100 epochs."

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry run only; no jobs will be submitted and no manifest will be written."
fi

submit_one baseline combined_sim2real baseline 0
submit_one improved combined_sim2real long_run_improved 1
submit_one clutter-improved combined_clutter long_run_improved 1

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry run complete; no jobs were submitted."
else
    echo "[INFO] Submitted all three jobs. Manifest: ${MANIFEST}"
    echo "[INFO] Watch, pull, and verify the entire suite with:"
    echo "${SCRIPT_DIR}/watch_ablation_suite.sh ${MANIFEST} 60"
fi
