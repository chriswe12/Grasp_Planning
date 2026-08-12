#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./start_dual_lbr_moveit.sh
  ./start_dual_lbr_moveit.sh --mode hardware --rviz

Starts one MoveIt planning scene containing two side-by-side KUKA iiwa7 arms.
Each arm carries the repository's calibrated Y-gripper and has its own planning
group and trajectory controller.

Options:
  --lbr-ws PATH               LBR ROS2 workspace. Default: /home/pdz/lbr-stack
  --mode MODE                 mock or hardware. Default: mock
  --ik-solver NAME            kdl or pick_ik. Default: kdl
  --rviz                      Launch the dual-arm RViz configuration.
  --robot-namespace NAME      Shared ROS namespace. Default: lbr_dual_arm
  --ros-domain-id ID          Force the ROS domain for this stack.
                              Default: DUAL_ROBOT_ROS_DOMAIN_ID, then
                              ROS_DOMAIN_ID, then 0
  --skip-ros-graph-check      Start even if dual-arm nodes already exist.
  -h, --help                  Show this help.

Physical-arm mapping:
  lbr_one: y=-0.42 m, controller 192.170.10.2, FRI port 30200
  lbr_two: y=+0.42 m, controller 192.170.20.2, FRI port 30201
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LBR_WS="/home/pdz/lbr-stack"
MODE="mock"
IK_SOLVER="kdl"
RVIZ=0
ROBOT_NAMESPACE="lbr_dual_arm"
ROS_DOMAIN_VALUE="${DUAL_ROBOT_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}"
CHECK_ROS_GRAPH=1
PIDS=()
PROCESS_GROUPS=()

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
  if [[ "${#PROCESS_GROUPS[@]}" -gt 0 ]]; then
    echo "[DUAL-LBR-MOVEIT] Stopping background ROS process groups..."
    for process_group in "${PROCESS_GROUPS[@]}"; do
      kill -TERM -- "-${process_group}" 2>/dev/null || true
    done
    # ros2 launch can exit before all launched nodes. Give every member the
    # same grace period, then guarantee that no orphaned MoveIt/controller
    # process survives after the launcher returns.
    sleep 2
    for process_group in "${PROCESS_GROUPS[@]}"; do
      kill -KILL -- "-${process_group}" 2>/dev/null || true
    done
  fi
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${status}"
}

configure_ros_discovery() {
  export ROS_DOMAIN_ID="${ROS_DOMAIN_VALUE}"
  if [[ "${GRASP_KEEP_ROS_DISCOVERY_ENV:-0}" == "1" ]]; then
    return 0
  fi

  export ROS_LOCALHOST_ONLY=0
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
  unset ROS_DISCOVERY_SERVER
  unset ROS_STATIC_PEERS
  unset CYCLONEDDS_URI
  unset FASTRTPS_DEFAULT_PROFILES_FILE
  unset ROS_AUTOMATIC_DISCOVERY_RANGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lbr-ws)
      LBR_WS="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --ik-solver)
      IK_SOLVER="${2:-}"
      shift 2
      ;;
    --rviz)
      RVIZ=1
      shift
      ;;
    --robot-namespace)
      ROBOT_NAMESPACE="${2:-}"
      shift 2
      ;;
    --ros-domain-id)
      ROS_DOMAIN_VALUE="${2:-}"
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
      echo "[DUAL-LBR-MOVEIT] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LBR_WS}" || -z "${ROBOT_NAMESPACE}" || -z "${ROS_DOMAIN_VALUE}" ]]; then
  echo "[DUAL-LBR-MOVEIT] Empty workspace, namespace, or ROS domain is not allowed." >&2
  exit 1
fi
if [[ ! "${ROS_DOMAIN_VALUE}" =~ ^[0-9]+$ ]]; then
  echo "[DUAL-LBR-MOVEIT] --ros-domain-id must be a non-negative integer." >&2
  exit 1
fi
if [[ "${MODE}" != "mock" && "${MODE}" != "hardware" ]]; then
  echo "[DUAL-LBR-MOVEIT] --mode must be 'mock' or 'hardware'." >&2
  exit 1
fi
if [[ "${IK_SOLVER}" != "pick_ik" && "${IK_SOLVER}" != "kdl" ]]; then
  echo "[DUAL-LBR-MOVEIT] --ik-solver must be 'pick_ik' or 'kdl'." >&2
  exit 1
fi

source_if_exists "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source_if_exists "${LBR_WS}/install/setup.bash"
source_if_exists "${SCRIPT_DIR}/ros2_ws/install/setup.bash"
configure_ros_discovery
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros-log}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[DUAL-LBR-MOVEIT] Missing required command: ros2" >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "[DUAL-LBR-MOVEIT] Missing required command: setsid" >&2
  exit 1
fi
if ! ros2 pkg prefix lbr_bringup >/dev/null 2>&1; then
  echo "[DUAL-LBR-MOVEIT] lbr_bringup is unavailable after sourcing ${LBR_WS}/install/setup.bash." >&2
  exit 1
fi
if ! ros2 pkg prefix robot_integration_ros >/dev/null 2>&1; then
  echo "[DUAL-LBR-MOVEIT] robot_integration_ros is unavailable after sourcing ros2_ws/install/setup.bash." >&2
  echo "[DUAL-LBR-MOVEIT] Build it first with: cd ros2_ws && colcon build --packages-select robot_integration_ros --symlink-install" >&2
  exit 1
fi
if [[ "${IK_SOLVER}" == "pick_ik" ]] && ! ros2 pkg prefix pick_ik >/dev/null 2>&1; then
  echo "[DUAL-LBR-MOVEIT] pick_ik is unavailable after sourcing ROS." >&2
  echo "[DUAL-LBR-MOVEIT] Install it with: sudo apt install ros-${ROS_DISTRO:-humble}-pick-ik" >&2
  echo "[DUAL-LBR-MOVEIT] Or use the tuned fallback: --ik-solver kdl" >&2
  exit 1
fi

if [[ "${CHECK_ROS_GRAPH}" -eq 1 ]]; then
  # Bypass the long-lived ros2cli daemon here. Its cached graph can retain a
  # stopped MoveGroup briefly and used to make a clean restart look occupied.
  EXISTING_NODES="$(
    timeout 5s ros2 node list --no-daemon --spin-time 1.0 2>/dev/null || true
  )"
  if grep -qxE "/${ROBOT_NAMESPACE}/(move_group|robot_state_publisher|controller_manager)" <<<"${EXISTING_NODES}"; then
    echo "[DUAL-LBR-MOVEIT] Existing ${ROBOT_NAMESPACE} control or MoveIt node detected." >&2
    echo "[DUAL-LBR-MOVEIT] Stop the old launch first, or rerun with --skip-ros-graph-check." >&2
    exit 1
  fi
fi

trap cleanup EXIT INT TERM

echo "[DUAL-LBR-MOVEIT] Starting both iiwa7/Y-gripper arms in ${MODE} mode with ${IK_SOLVER} on ROS domain ${ROS_DOMAIN_ID}."
setsid ros2 launch robot_integration_ros dual_aligned_lbr_moveit.launch.py \
  mode:="${MODE}" \
  ik_solver:="${IK_SOLVER}" \
  robot_namespace:="${ROBOT_NAMESPACE}" \
  rviz:="$([[ "${RVIZ}" -eq 1 ]] && printf true || printf false)" &
launch_pid="$!"
PIDS+=("${launch_pid}")
launch_process_group=""
for _ in $(seq 1 50); do
  launch_process_group="$(ps -o pgid= -p "${launch_pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${launch_process_group}" == "${launch_pid}" ]]; then
    break
  fi
  if ! kill -0 "${launch_pid}" 2>/dev/null; then
    break
  fi
  sleep 0.02
done
if [[ ! "${launch_process_group}" =~ ^[1-9][0-9]*$ || "${launch_process_group}" != "${launch_pid}" ]]; then
  echo "[DUAL-LBR-MOVEIT] Failed to isolate the ROS launch in its own process group." >&2
  kill "${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
  exit 1
fi
PROCESS_GROUPS+=("${launch_process_group}")

echo "[DUAL-LBR-MOVEIT] Started. Leave this terminal running. Press Ctrl-C to stop the stack."
set +e
wait -n "${PIDS[@]}"
status=$?
echo "[DUAL-LBR-MOVEIT] The ROS launch exited with status ${status}."
exit "${status}"
