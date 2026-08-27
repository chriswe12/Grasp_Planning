#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ID="${ABLATION_SUITE_ID:-$(date '+%Y%m%d-%H%M%S')}"
NUM_ENVS="${ABLATION_NUM_ENVS:-224}"
GPU_COUNT="${ABLATION_GPU_COUNT:-3}"
# 3 ranks * 224 envs/rank * 3810 epochs ~= the reference 256 envs *
# 10000 epochs. The matched global rollout and PPO-update volume completes
# safely inside Euler's 48-hour limit at measured PDZ throughput.
MAX_ITERATIONS="${ABLATION_MAX_ITERATIONS:-3810}"
SEED="${ABLATION_SEED:-42}"

if [[ ! "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${GPU_COUNT}" =~ ^[1-6]$ ]]; then
    echo "[ERROR] ABLATION_NUM_ENVS/ABLATION_MAX_ITERATIONS must be positive and ABLATION_GPU_COUNT must be 1--6" >&2
    exit 2
fi

common_args=(
    --gpu-type rtx_4090
    --gpu-count "${GPU_COUNT}"
    --gpu-memory 20G
    --time-limit 2-00:00:00
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-v0
    --num_envs "${NUM_ENVS}"
    --max_iterations "${MAX_ITERATIONS}"
    --seed "${SEED}"
    --headless
    --enable_cameras
)

submit_one() {
    local label="$1"
    local profile="$2"
    local context="$3"
    local skip_sync="$4"
    local experiment="ablation_${SUITE_ID}_${label}_seed${SEED}"
    echo "[ABLATION] ${label}: profile=${profile} context=${context} experiment=${experiment}"
    EULER_SKIP_SYNC="${skip_sync}" "${SCRIPT_DIR}/submit.sh" train \
        --job-label "abl-${label}" \
        "${common_args[@]}" \
        --sim2real_profile "${profile}" \
        --policy-context "${context}" \
        --experiment-name "${experiment}"
}

echo "[INFO] Submitting policy-context ablation suite ${SUITE_ID}"
echo "[INFO] Each run uses ${GPU_COUNT} GPUs x ${NUM_ENVS} environments/rank for ${MAX_ITERATIONS} PPO epochs (seed ${SEED})."
submit_one baseline combined_sim2real action 0
submit_one background combined_busy_background action 1
submit_one velocity combined_sim2real action_twist 1
submit_one velocity-rotation combined_sim2real action_twist_rotation 1

echo "[INFO] All four jobs were submitted. Their readable Slurm names are:"
echo "       grasp-abl-baseline, grasp-abl-background, grasp-abl-velocity, grasp-abl-velocity-rotation"
