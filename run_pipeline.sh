#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/setup_robot_env.sh" ]]; then
  # The public entrypoint has one environment path. The helper is deliberately
  # soft when optional ROS underlays are absent so help and offline tools work.
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/setup_robot_env.sh" --quiet
fi

if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
  PYTHON_BIN="${PIPELINE_PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Could not find a usable Python interpreter." >&2
  exit 1
fi

exec "${PYTHON_BIN}" scripts/run_unified_pipeline.py "$@"
