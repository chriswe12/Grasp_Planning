#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_pipeline.sh --mode local
  ./run_pipeline.sh --mode real
  ./run_pipeline.sh --mode local --config configs/grasp_pipeline_local.yaml
EOF
}

MODE=""
CONFIG=""

resolve_python() {
  if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
    printf '%s\n' "${PIPELINE_PYTHON}"
    return 0
  fi
  if [[ -x "/isaac-sim/python.sh" ]]; then
    printf '%s\n' "/isaac-sim/python.sh"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "Could not find a usable Python interpreter." >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
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

if [[ -z "$MODE" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$CONFIG" ]]; then
  case "$MODE" in
    local)
      CONFIG="configs/grasp_pipeline_local.yaml"
      ;;
    real)
      CONFIG="configs/grasp_pipeline_real.yaml"
      ;;
    *)
      echo "Unsupported mode: $MODE" >&2
      exit 1
      ;;
  esac
fi

PYTHON_BIN="$(resolve_python)"
exec "${PYTHON_BIN}" scripts/run_grasp_pipeline.py --mode "$MODE" --config "$CONFIG"
