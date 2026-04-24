#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROS2_WS_ROOT="${REPO_ROOT}/ros2_ws"
ROS2_WS_SRC="${ROS2_WS_ROOT}/src"
ROS2_REPOS_FILE="${ROS2_WS_ROOT}/dependencies.repos"
FP_DEBUG_MSGS_ROOT="${ROS2_WS_SRC}/fp_debug_msgs"
DEFAULT_FP_DEBUG_MSGS_REMOTE="https://github.com/Moreno-Nautilus/fp_debug_msgs.git"
FP_DEBUG_MSGS_REMOTE="${FP_DEBUG_MSGS_REMOTE:-${DEFAULT_FP_DEBUG_MSGS_REMOTE}}"
FP_DEBUG_MSGS_REF="${FP_DEBUG_MSGS_REF:-7cab8c96effad8f3489fa509dfe5cd2795242c37}"
FORCE_CLONE=0
TEMP_FILES=()

usage() {
  cat <<EOF
Usage:
  bash scripts/download_ros2_dependencies.sh

Options:
  --force-clone  Recreate the fp_debug_msgs checkout from scratch.
  -h, --help     Show this message.

Environment:
  FP_DEBUG_MSGS_REMOTE  Git remote to clone from.
                        Default: ${FP_DEBUG_MSGS_REMOTE}
  FP_DEBUG_MSGS_REF     Git branch, tag, or commit to check out.
                        Default: ${FP_DEBUG_MSGS_REF}

What this script does:
  1. Ensures ros2_ws/src exists under the repo root.
  2. Imports the pinned ROS2 dependency manifest from ros2_ws/dependencies.repos
     when vcstool is available.
  3. Falls back to a direct git checkout for fp_debug_msgs when vcstool is not installed.
  4. Pins ros2_ws/src/fp_debug_msgs to the requested ref.
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

cleanup_temp_files() {
  local path
  for path in "${TEMP_FILES[@]}"; do
    [[ -e "${path}" ]] && rm -f "${path}"
  done
}

trap cleanup_temp_files EXIT

prepare_checkout_root() {
  mkdir -p "${ROS2_WS_SRC}"

  if [[ "${FORCE_CLONE}" -eq 1 && -d "${FP_DEBUG_MSGS_ROOT}" ]]; then
    rm -rf "${FP_DEBUG_MSGS_ROOT}"
  fi

  if [[ -e "${FP_DEBUG_MSGS_ROOT}" && ! -d "${FP_DEBUG_MSGS_ROOT}/.git" ]]; then
    echo "Path exists but is not a git checkout: ${FP_DEBUG_MSGS_ROOT}" >&2
    exit 1
  fi
}

import_with_vcs_if_available() {
  if ! command -v vcs >/dev/null 2>&1; then
    echo "[INFO] vcstool is not installed; falling back to a direct git checkout for fp_debug_msgs."
    return 0
  fi

  local import_file="${ROS2_REPOS_FILE}"
  if [[ "${FP_DEBUG_MSGS_REMOTE}" != "${DEFAULT_FP_DEBUG_MSGS_REMOTE}" ]]; then
    local temp_repos_file escaped_remote
    temp_repos_file="$(mktemp)"
    TEMP_FILES+=("${temp_repos_file}")
    escaped_remote="$(printf '%s' "${FP_DEBUG_MSGS_REMOTE}" | sed -e 's/[&|\\]/\\&/g')"
    sed "s|${DEFAULT_FP_DEBUG_MSGS_REMOTE}|${escaped_remote}|g" "${ROS2_REPOS_FILE}" > "${temp_repos_file}"
    import_file="${temp_repos_file}"
  fi

  echo "[INFO] Importing ROS2 dependency manifest from ${import_file}"
  vcs import --force --input "${import_file}" "${ROS2_WS_SRC}"
}

clone_if_missing() {
  if [[ -d "${FP_DEBUG_MSGS_ROOT}/.git" ]]; then
    git -C "${FP_DEBUG_MSGS_ROOT}" remote set-url origin "${FP_DEBUG_MSGS_REMOTE}"
    return 0
  fi

  echo "[INFO] Cloning fp_debug_msgs into ${FP_DEBUG_MSGS_ROOT}"
  git clone --filter=blob:none "${FP_DEBUG_MSGS_REMOTE}" "${FP_DEBUG_MSGS_ROOT}"
}

pin_fp_debug_msgs_checkout() {
  clone_if_missing

  local current_head requested_commit
  current_head="$(git -C "${FP_DEBUG_MSGS_ROOT}" rev-parse --verify HEAD 2>/dev/null || true)"
  requested_commit="$(git -C "${FP_DEBUG_MSGS_ROOT}" rev-parse --verify "${FP_DEBUG_MSGS_REF}^{commit}" 2>/dev/null || true)"
  if [[ -n "${current_head}" && -n "${requested_commit}" && "${current_head}" == "${requested_commit}" ]]; then
    echo "[INFO] fp_debug_msgs is already pinned to ${FP_DEBUG_MSGS_REF}; skipping fetch."
    return 0
  fi

  echo "[INFO] Fetching fp_debug_msgs ref ${FP_DEBUG_MSGS_REF}"
  git -C "${FP_DEBUG_MSGS_ROOT}" fetch --depth 1 origin "${FP_DEBUG_MSGS_REF}"
  git -C "${FP_DEBUG_MSGS_ROOT}" checkout --force FETCH_HEAD
}

verify_outputs() {
  local required_paths=(
    "${ROS2_REPOS_FILE}"
    "${FP_DEBUG_MSGS_ROOT}/package.xml"
    "${FP_DEBUG_MSGS_ROOT}/msg/DebugFrame.msg"
  )

  local missing=0
  local path
  for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
      echo "[ERROR] Missing expected ROS2 dependency path: ${path}" >&2
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
prepare_checkout_root
import_with_vcs_if_available
pin_fp_debug_msgs_checkout
verify_outputs

echo "[INFO] ROS2 dependency bootstrap complete."
echo "[INFO] Workspace source root: ${ROS2_WS_SRC}"
echo "[INFO] fp_debug_msgs path:    ${FP_DEBUG_MSGS_ROOT}"
