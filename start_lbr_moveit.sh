#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./start_lbr_moveit.sh
  ./start_lbr_moveit.sh --mode hardware --rviz

Starts the KUKA iiwa7 robot state/controller stack and namespaced MoveIt server
from the hardware-canonical iiwa7 + Y-gripper description also used by Isaac. Leave
this running while the pipeline plans or executes from another terminal.

Options:
  --lbr-ws PATH               LBR ROS2 workspace. Default: /home/pdz/lbr-stack
  --model NAME                LBR model. Default: iiwa7
  --mode MODE                 mock or hardware. Default: mock
  --rviz                      Launch RViz with the aligned MoveIt description.
  --robot-name NAME           ROS namespace used by the control stack. Default: lbr
  --skip-ros-graph-check      Start even if LBR/MoveIt nodes already exist.
  -h, --help                  Show this help.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LBR_WS="/home/pdz/lbr-stack"
MODEL="iiwa7"
MODE="mock"
RVIZ=0
ROBOT_NAME="lbr"
CHECK_ROS_GRAPH=1
PIDS=()

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

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "[LBR-MOVEIT] Stopping background ROS processes..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${status}"
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
  unset ROS_AUTOMATIC_DISCOVERY_RANGE
  export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lbr-ws)
      LBR_WS="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --rviz)
      RVIZ=1
      shift
      ;;
    --robot-name)
      ROBOT_NAME="${2:-}"
      shift 2
      ;;
    --skip-ros-graph-check)
      CHECK_ROS_GRAPH=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[LBR-MOVEIT] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LBR_WS}" || -z "${MODEL}" || -z "${ROBOT_NAME}" ]]; then
  echo "[LBR-MOVEIT] Empty --lbr-ws, --model, or --robot-name is not allowed." >&2
  exit 1
fi
if [[ "${MODEL}" != "iiwa7" ]]; then
  echo "[LBR-MOVEIT] The aligned description currently supports only --model iiwa7." >&2
  exit 1
fi
if [[ "${MODE}" != "mock" && "${MODE}" != "hardware" ]]; then
  echo "[LBR-MOVEIT] --mode must be 'mock' or 'hardware'." >&2
  exit 1
fi

source_if_exists "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source_if_exists "${LBR_WS}/install/setup.bash"
source_if_exists "${SCRIPT_DIR}/ros2_ws/install/setup.bash"
configure_ros_discovery
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros-log}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[LBR-MOVEIT] Missing required command: ros2" >&2
  exit 1
fi
if ! ros2 pkg prefix lbr_bringup >/dev/null 2>&1; then
  echo "[LBR-MOVEIT] lbr_bringup is unavailable after sourcing ${LBR_WS}/install/setup.bash." >&2
  exit 1
fi
if ! ros2 pkg prefix robot_integration_ros >/dev/null 2>&1; then
  echo "[LBR-MOVEIT] robot_integration_ros is unavailable after sourcing ros2_ws/install/setup.bash." >&2
  echo "[LBR-MOVEIT] Build it first with: cd ros2_ws && colcon build --packages-select robot_integration_ros --symlink-install" >&2
  exit 1
fi

if [[ "${CHECK_ROS_GRAPH}" -eq 1 ]]; then
  EXISTING_NODES="$(timeout 5s ros2 node list 2>/dev/null || true)"
  if grep -qxE "/${ROBOT_NAME}/(move_group|robot_state_publisher|controller_manager)" <<<"${EXISTING_NODES}"; then
    echo "[LBR-MOVEIT] Existing ${ROBOT_NAME} control or MoveIt node detected." >&2
    echo "[LBR-MOVEIT] Stop the old launch first, or rerun with --skip-ros-graph-check." >&2
    exit 1
  fi
fi

trap cleanup EXIT INT TERM

echo "[LBR-MOVEIT] Starting aligned ${MODEL} ${MODE} control and MoveIt stack."
ros2 launch robot_integration_ros aligned_lbr_moveit.launch.py \
  mode:="${MODE}" \
  robot_name:="${ROBOT_NAME}" \
  rviz:="$([[ "${RVIZ}" -eq 1 ]] && printf true || printf false)" &
PIDS+=("$!")

echo "[LBR-MOVEIT] Started. Leave this terminal running. Press Ctrl-C to stop the stack."
set +e
wait -n "${PIDS[@]}"
status=$?
echo "[LBR-MOVEIT] The ROS launch exited with status ${status}."
exit "${status}"
