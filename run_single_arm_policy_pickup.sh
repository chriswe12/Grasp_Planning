#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" scripts/run_single_arm_policy_pickup.py "$@"
