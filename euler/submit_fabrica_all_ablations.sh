#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The default 4 ranks * 128 envs/rank * 5000 epochs = 2.56M
# environment-epochs, matching the previous 3 * 224 * 3810 suite while loading
# one disjoint Fabrica part shard per GPU. ABLATION_GPU_COUNT=6 selects the
# complete six-shard layout. Set ABLATION_DRY_RUN=1 to inspect every command.
export ABLATION_TASK="Grasp-Visual-Servo-RGBD-FabricaAll-Direct-v0"
export ABLATION_GPU_COUNT="${ABLATION_GPU_COUNT:-4}"
export ABLATION_NUM_ENVS="${ABLATION_NUM_ENVS:-128}"
export ABLATION_MAX_ITERATIONS="${ABLATION_MAX_ITERATIONS:-5000}"

exec "${SCRIPT_DIR}/submit_policy_context_ablations.sh"
