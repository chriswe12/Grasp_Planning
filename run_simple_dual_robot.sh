#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[COMPAT] run_simple_dual_robot.sh delegates to run_pipeline.sh." >&2
exec "${SCRIPT_DIR}/run_pipeline.sh" --workflow dual "$@"
