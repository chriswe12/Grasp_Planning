#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

for command_name in curl rsync sha256sum ssh; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: ${command_name}" >&2
        exit 1
    fi
done

if [[ "${EULER_LOCAL_TORCH_CACHE}" = /* ]]; then
    local_torch_cache="${EULER_LOCAL_TORCH_CACHE}"
else
    local_torch_cache="${REPO_ROOT}/${EULER_LOCAL_TORCH_CACHE}"
fi
resnet18_path="${local_torch_cache}/hub/checkpoints/${EULER_RESNET18_FILE}"
mkdir -p "$(dirname "${resnet18_path}")"

if [[ ! -f "${resnet18_path}" ]]; then
    download_path="${resnet18_path}.part"
    echo "[INFO] Downloading the pretrained ResNet-18 weights for offline Euler jobs"
    curl --fail --location --retry 3 \
        --output "${download_path}" "${EULER_RESNET18_URL}"
    mv -f "${download_path}" "${resnet18_path}"
fi

resnet18_sha256="$(sha256sum "${resnet18_path}")"
resnet18_sha256="${resnet18_sha256%% *}"
if [[ "${resnet18_sha256}" != "${EULER_RESNET18_SHA256_PREFIX}"* ]]; then
    echo "[ERROR] Invalid ResNet-18 weight checksum: ${resnet18_sha256}" >&2
    exit 1
fi

ssh "${EULER_LOGIN}" \
    "mkdir -p '${EULER_PROJECT_DIR}/logs' \
        '${EULER_CACHE_DIR}/cache/torch/hub/checkpoints' '${EULER_RUNS_DIR}'"

echo "[INFO] Synchronizing the grasping project to ${EULER_LOGIN}:${EULER_PROJECT_DIR}"
rsync -azh --info=progress2 --delete-delay \
    --exclude-from="${SCRIPT_DIR}/rsync-excludes.txt" \
    "${REPO_ROOT}/" "${EULER_LOGIN}:${EULER_PROJECT_DIR}/"

echo "[INFO] Staging the pretrained ResNet-18 weights in the Euler cache"
rsync -azh --info=progress2 --partial \
    "${resnet18_path}" \
    "${EULER_LOGIN}:${EULER_CACHE_DIR}/cache/torch/hub/checkpoints/${EULER_RESNET18_FILE}"

ssh "${EULER_LOGIN}" \
    "test -f '${EULER_PROJECT_DIR}/isaac_rl/scripts/rl_games/train.py' && \
     test -f '${EULER_PROJECT_DIR}/isaac_rl/data/multigrasp_50_catalog.npz' && \
     test -f '${EULER_PROJECT_DIR}/isaac_rl/data/plumbers_block/goal_catalog.npz' && \
     test -f '${EULER_PROJECT_DIR}/isaac_rl/data/plumbers_block/rotation_resets.npz' && \
     test -f '${EULER_PROJECT_DIR}/isaac_rl/data/plumbers_block/usd/part_0_bundle_local.usd' && \
     test -f '${EULER_PROJECT_DIR}/isaac_rl/data/plumbers_block/usd/part_1_bundle_local.usd' && \
     test -f '${EULER_PROJECT_DIR}/isaac_rl/data/plumbers_block/usd/part_2_bundle_local.usd' && \
     test -f '${EULER_PROJECT_DIR}/isaac_rl/data/plumbers_block/usd/part_3_bundle_local.usd' && \
     test -f '${EULER_PROJECT_DIR}/isaac_rl/data/plumbers_block/usd/part_4_bundle_local.usd' && \
     test -f '${EULER_PROJECT_DIR}/assets/usd/kuka_iiwa7_pdz_gripper/kuka_iiwa7_pdz_gripper.usd' && \
     test -f '${EULER_PROJECT_DIR}/assets/usd/kuka_iiwa7_y_gripper/kuka_iiwa7_y_gripper.usda' && \
     test -f '${EULER_PROJECT_DIR}/artifacts/isaac_bundle_assets/pipeline_stage2_ground_feasible_bundle_local.usd' && \
     echo '[INFO] Required Euler project files are present'"
