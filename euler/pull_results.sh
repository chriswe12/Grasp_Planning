#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

local_results="${REPO_ROOT}/logs/euler"
mkdir -p "${local_results}"

retry_seconds="${EULER_PULL_RETRY_SECONDS:-60}"
if [[ ! "${retry_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] EULER_PULL_RETRY_SECONDS must be a positive integer" >&2
    exit 2
fi

# Several detached job watchers can reach this helper within the same minute.
# Serialize them so independent rsync processes never update the same local
# checkpoint or event file concurrently.  Keep retrying after transient SSH or
# network failures: these watchers are the unattended result handoff.
lock_file="${local_results}/.pull.lock"
exec 9>"${lock_file}"
echo "[INFO] Waiting for the Euler result-pull lock"
flock 9

attempt=1
while true; do
    echo "[INFO] Downloading Euler results to ${local_results} (attempt ${attempt})"
    if rsync -azh --info=progress2 --partial \
        "${EULER_LOGIN}:${EULER_RUNS_DIR}/" "${local_results}/"; then
        break
    fi
    echo "[WARNING] Euler result download failed; retrying in ${retry_seconds}s" >&2
    sleep "${retry_seconds}"
    attempt=$((attempt + 1))
done
