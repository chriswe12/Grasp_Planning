#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_pipeline.sh --mode sim
  ./run_pipeline.sh --mode pitl
  ./run_pipeline.sh --mode real
  ./run_pipeline.sh --mode sim --config configs/grasp_pipeline_sim.yaml
  ./run_pipeline.sh --mode sim --headless
  ./run_pipeline.sh --mode sim --backend isaac --headless
  ./run_pipeline.sh --mode pitl --skip-stage1-collision-checks
  ./run_pipeline.sh --mode sim --backend mujoco --force-regrasp-fallback

Backends for sim/pitl:
  config  Honor YAML execution blocks
  mujoco  Run MuJoCo only
  isaac   Run Isaac only
  both    Run MuJoCo then Isaac
  none    Plan/write artifacts only
EOF
}

MODE=""
CONFIG=""
BACKEND="config"
HEADLESS=0
SKIP_STAGE1_COLLISION_CHECKS=0
FORCE_REGRASP_FALLBACK=0

source_if_exists() {
  local setup_file="$1"
  if [[ -f "${setup_file}" ]]; then
    local nounset_was_enabled=0
    case "$-" in
      *u*) nounset_was_enabled=1 ;;
    esac
    set +u
    # shellcheck source=/dev/null
    source "${setup_file}"
    if [[ "${nounset_was_enabled}" -eq 1 ]]; then
      set -u
    fi
  fi
}

source_ros_environment() {
  if [[ -n "${ROS_DISTRO:-}" ]]; then
    source_if_exists "/opt/ros/${ROS_DISTRO}/setup.bash"
  else
    for distro in humble jazzy iron rolling; do
      if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
        source_if_exists "/opt/ros/${distro}/setup.bash"
        break
      fi
    done
  fi
  if [[ "${GRASP_SKIP_WORKSPACE_ROS_OVERLAY:-0}" != "1" ]]; then
    if [[ -f "ros2_ws/install/local_setup.bash" ]]; then
      source_if_exists "ros2_ws/install/local_setup.bash"
    else
      source_if_exists "ros2_ws/install/setup.bash"
    fi
  fi
}

configure_ros_discovery() {
  if [[ "${GRASP_KEEP_ROS_DISCOVERY_ENV:-0}" == "1" ]]; then
    return 0
  fi

  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
  unset ROS_LOCALHOST_ONLY
  unset ROS_STATIC_PEERS
  unset RMW_IMPLEMENTATION
  unset CYCLONEDDS_URI
  unset FASTRTPS_DEFAULT_PROFILES_FILE
  export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
  if [[ -n "${ROS_AUTOMATIC_DISCOVERY_RANGE:-}" ]]; then
    unset ROS_AUTOMATIC_DISCOVERY_RANGE
  fi
}

resolve_python() {
  if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
    printf '%s\n' "${PIPELINE_PYTHON}"
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
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    --skip-stage1-collision-checks)
      SKIP_STAGE1_COLLISION_CHECKS=1
      shift
      ;;
    --force-regrasp-fallback)
      FORCE_REGRASP_FALLBACK=1
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

if [[ -z "$MODE" ]]; then
  usage >&2
  exit 1
fi

case "$MODE" in
  simulation)
    MODE="sim"
    ;;
  perception_in_the_loop|perception-in-the-loop)
    MODE="pitl"
    ;;
esac

if [[ -z "$CONFIG" ]]; then
  case "$MODE" in
    sim)
      CONFIG="configs/grasp_pipeline_sim.yaml"
      ;;
    pitl)
      CONFIG="configs/grasp_pipeline_pitl.yaml"
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

source_ros_environment
configure_ros_discovery
PYTHON_BIN="$(resolve_python)"
ARGS=(scripts/run_grasp_pipeline.py --mode "$MODE" --config "$CONFIG")
ARGS+=(--backend "$BACKEND")
if [[ "${HEADLESS}" -eq 1 ]]; then
  ARGS+=(--headless)
fi
if [[ "${SKIP_STAGE1_COLLISION_CHECKS}" -eq 1 ]]; then
  ARGS+=(--skip-stage1-collision-checks)
fi
if [[ "${FORCE_REGRASP_FALLBACK}" -eq 1 ]]; then
  ARGS+=(--force-regrasp-fallback)
fi
exec "${PYTHON_BIN}" "${ARGS[@]}"
