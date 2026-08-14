#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

local_results="${REPO_ROOT}/logs/euler"
mkdir -p "${local_results}"

echo "[INFO] Downloading Euler results to ${local_results}"
rsync -azh --info=progress2 --partial \
    "${EULER_LOGIN}:${EULER_RUNS_DIR}/" "${local_results}/"
