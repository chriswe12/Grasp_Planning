#!/usr/bin/env bash
# Stage a separate immutable-by-convention source/data snapshot for Franka jobs.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/euler.env"
cd "${repo_root}"
prepare_only=0
if [[ "${1:-}" == --prepare-only ]]; then
    prepare_only=1
    shift
fi
stamp="$(date +%Y%m%d_%H%M%S)"
staging="artifacts/franka_euler/${stamp}"
remote_project="${EULER_PROJECT_DIR}_franka_${stamp}"
remote_runs="${EULER_RUNS_DIR}/franka_${stamp}"
mkdir -p "${staging}/project/checkpoints"
catalog="${EULER_FRANKA_CATALOG:-isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz}"
test -f "${catalog}"
rsync -aR --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    grasp_planning scripts configs euler isaac_rl/source isaac_rl/scripts \
    "$(dirname "${catalog}")" \
    isaac_rl/data/franka_fabrica_plumbers/part_bundle_local.usd \
    assets/scenes/video_lab_pencil assets/usd/franka_panda_offline \
    "${staging}/project/"
printf '\nEULER_PROJECT_DIR=%q\nEULER_RUNS_DIR=%q\n' "${remote_project}" "${remote_runs}" >> "${staging}/project/euler/euler.env"
test -f assets/usd/franka_panda_offline/manifest.json
if [[ -n "${1:-}" ]]; then
    test -f "${1%.pth}.contract.json"
    cp "${1}" "${staging}/project/checkpoints/resume.pth"
    cp "${1%.pth}.contract.json" "${staging}/project/checkpoints/resume.contract.json"
fi
python3 euler/verify_franka_deployment.py "${staging}/project" \
    --catalog "${catalog}" --copy-sources-from "${repo_root}" --output "${staging}/verification.json"
printf '[FRANKA DEPLOYMENT] %s/%s/project/euler/euler.env\n' "${repo_root}" "${staging}"
if (( prepare_only )); then
    echo '[INFO] Local package prepared; no network transfer or submission performed.'
    exit 0
fi
ssh "${EULER_LOGIN}" "mkdir -p '${remote_project}/logs' '${remote_project}/outputs' '${remote_runs}'"
rsync -az "${staging}/project/" "${EULER_LOGIN}:${remote_project}/"
weights="${EULER_LOCAL_TORCH_CACHE}/hub/checkpoints/${EULER_RESNET18_FILE}"
test -f "${weights}"
ssh "${EULER_LOGIN}" "mkdir -p '${EULER_CACHE_DIR}/cache/torch/hub/checkpoints'"
rsync -az "${weights}" "${EULER_LOGIN}:${EULER_CACHE_DIR}/cache/torch/hub/checkpoints/"
