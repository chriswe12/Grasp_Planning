#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MENAGERIE_ROOT="${REPO_ROOT}/.cache/robot_descriptions/mujoco_menagerie"
GENERATED_MODELS_DIR="${REPO_ROOT}/.cache/generated_mujoco_models"
MENAGERIE_REMOTE="${MENAGERIE_REMOTE:-https://github.com/google-deepmind/mujoco_menagerie.git}"
MENAGERIE_REF="${MENAGERIE_REF:-main}"
FORCE_CLONE=0
FORCE_REBUILD=0

usage() {
  cat <<EOF
Usage:
  bash scripts/download_required_assets.sh

Options:
  --force-clone    Recreate the MuJoCo Menagerie checkout from scratch.
  --force-rebuild  Rebuild the generated FR3+hand MuJoCo XMLs.
  -h, --help       Show this message.

Environment:
  MENAGERIE_REMOTE  Git remote to clone from.
                    Default: ${MENAGERIE_REMOTE}
  MENAGERIE_REF     Git branch, tag, or commit to check out.
                    Default: ${MENAGERIE_REF}

What this script does:
  1. Creates .cache directories under the repo root.
  2. Clones a sparse MuJoCo Menagerie checkout containing only:
     - franka_fr3
     - franka_fr3_v2
     - franka_emika_panda
  3. Builds the merged FR3+Panda-hand XMLs used by the MuJoCo pipeline.
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

ensure_vendored_franka_description() {
  local vendor_dir="${REPO_ROOT}/assets/urdf/franka_description"
  if [[ ! -d "${vendor_dir}" ]]; then
    echo "Vendored franka_description assets are missing at '${vendor_dir}'." >&2
    echo "This repo is expected to contain them already." >&2
    exit 1
  fi
}

clone_or_update_menagerie() {
  mkdir -p "$(dirname "${MENAGERIE_ROOT}")"

  if [[ "${FORCE_CLONE}" -eq 1 && -d "${MENAGERIE_ROOT}" ]]; then
    rm -rf "${MENAGERIE_ROOT}"
  fi

  if [[ ! -d "${MENAGERIE_ROOT}/.git" ]]; then
    echo "[INFO] Cloning MuJoCo Menagerie into ${MENAGERIE_ROOT}"
    git clone --filter=blob:none --sparse "${MENAGERIE_REMOTE}" "${MENAGERIE_ROOT}"
  fi

  echo "[INFO] Configuring sparse checkout"
  git -C "${MENAGERIE_ROOT}" sparse-checkout set \
    franka_emika_panda \
    franka_fr3 \
    franka_fr3_v2

  echo "[INFO] Fetching MuJoCo Menagerie ref ${MENAGERIE_REF}"
  git -C "${MENAGERIE_ROOT}" fetch --depth 1 origin "${MENAGERIE_REF}"
  git -C "${MENAGERIE_ROOT}" checkout --force FETCH_HEAD
}

build_generated_models() {
  if [[ "${FORCE_REBUILD}" -eq 0 ]] && \
     [[ -f "${GENERATED_MODELS_DIR}/fr3_with_panda_hand.xml" ]] && \
     [[ -f "${GENERATED_MODELS_DIR}/fr3v2_with_panda_hand.xml" ]]; then
    echo "[INFO] Reusing generated MuJoCo models in ${GENERATED_MODELS_DIR}"
    return 0
  fi

  echo "[INFO] Building merged FR3+Panda-hand MuJoCo XMLs"
  python3 "${REPO_ROOT}/scripts/build_mujoco_fr3_hand_models.py" \
    --menagerie-root "${MENAGERIE_ROOT}" \
    --output-dir "${GENERATED_MODELS_DIR}"
}

verify_outputs() {
  local required_paths=(
    "${MENAGERIE_ROOT}/franka_emika_panda/hand.xml"
    "${MENAGERIE_ROOT}/franka_fr3/fr3.xml"
    "${MENAGERIE_ROOT}/franka_fr3_v2/fr3v2.xml"
    "${GENERATED_MODELS_DIR}/fr3_with_panda_hand.xml"
    "${GENERATED_MODELS_DIR}/fr3v2_with_panda_hand.xml"
  )

  local missing=0
  local path
  for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
      echo "[ERROR] Missing expected asset: ${path}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-clone)
      FORCE_CLONE=1
      shift
      ;;
    --force-rebuild)
      FORCE_REBUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_cmd git
require_cmd python3
ensure_vendored_franka_description
clone_or_update_menagerie
build_generated_models
verify_outputs

echo "[INFO] MuJoCo asset bootstrap complete."
echo "[INFO] Menagerie checkout: ${MENAGERIE_ROOT}"
echo "[INFO] Generated models:   ${GENERATED_MODELS_DIR}"
