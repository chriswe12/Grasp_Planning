#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./start_cumotion_moveit.sh
  ./start_cumotion_moveit.sh --robot-ip dont-care

Starts the fake-hardware FR3 MoveIt stack with the Isaac ROS cuMotion action
server. Leave this running in one terminal, then run the grasp pipeline from a
second terminal.

Options:
  --robot-ip VALUE              Robot IP passed to the Franka launch/xacro.
                                Default: dont-care
  --franka-ws PATH              External FR3 ROS2 workspace.
                                Default: /home/pdz/franka_ros2_ws
  --cuda-home PATH              CUDA root used by cuRobo JIT.
                                Default: /usr/local/cuda-12.8
  --tool-frame NAME             cuMotion tool frame. Default: fr3_hand_tcp
  --joint-states-topic TOPIC    cuMotion joint states topic. Default: /joint_states
  --urdf-path PATH              Generated/static URDF path.
                                Default: /tmp/fr3_cumotion/fr3.urdf
  --xrdf-path PATH              cuMotion XRDF path.
                                Default: package franka_fr3_moveit_config/config/fr3_cumotion.xrdf
  --skip-ros-graph-check        Do not check for already-running MoveIt/cuMotion nodes.
  -h, --help                    Show this help.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRANKA_WS="/home/pdz/franka_ros2_ws"
ROBOT_IP="dont-care"
CUDA_HOME_VALUE="/usr/local/cuda-12.8"
TOOL_FRAME="fr3_hand_tcp"
JOINT_STATES_TOPIC="/joint_states"
URDF_PATH="/tmp/fr3_cumotion/fr3.urdf"
XRDF_PATH=""
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
    echo "[CUMOTION-MOVEIT] Stopping background ROS processes..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${status}"
}

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "[CUMOTION-MOVEIT] Missing required command: ${name}" >&2
    exit 1
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-ip)
      ROBOT_IP="${2:-}"
      shift 2
      ;;
    --franka-ws)
      FRANKA_WS="${2:-}"
      shift 2
      ;;
    --cuda-home)
      CUDA_HOME_VALUE="${2:-}"
      shift 2
      ;;
    --tool-frame)
      TOOL_FRAME="${2:-}"
      shift 2
      ;;
    --joint-states-topic)
      JOINT_STATES_TOPIC="${2:-}"
      shift 2
      ;;
    --urdf-path)
      URDF_PATH="${2:-}"
      shift 2
      ;;
    --xrdf-path)
      XRDF_PATH="${2:-}"
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
      echo "[CUMOTION-MOVEIT] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${ROBOT_IP}" || -z "${FRANKA_WS}" || -z "${CUDA_HOME_VALUE}" ]]; then
  echo "[CUMOTION-MOVEIT] Empty --robot-ip, --franka-ws, or --cuda-home is not allowed." >&2
  exit 1
fi

source_if_exists "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source_if_exists "${FRANKA_WS}/install/setup.bash"
source_if_exists "${SCRIPT_DIR}/ros2_ws/install/setup.bash"
configure_ros_discovery

require_command ros2
require_command xacro

export CUDA_HOME="${CUDA_HOME_VALUE}"
PATCH_DIR="${CUMOTION_PYTHON_PATCH_DIR:-/home/pdz/.local/cumotion_humble_patches}"
if [[ -d "${PATCH_DIR}" ]]; then
  export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"
else
  echo "[CUMOTION-MOVEIT] Warning: ${PATCH_DIR} does not exist; Warp compatibility shim will not be loaded." >&2
fi

if [[ -z "${XRDF_PATH}" ]]; then
  XRDF_PATH="$(ros2 pkg prefix --share franka_fr3_moveit_config)/config/fr3_cumotion.xrdf"
fi
FRANKA_DESCRIPTION_SHARE="$(ros2 pkg prefix --share franka_description)"
FR3_XACRO="${FRANKA_DESCRIPTION_SHARE}/robots/fr3/fr3.urdf.xacro"

if [[ ! -f "${XRDF_PATH}" ]]; then
  echo "[CUMOTION-MOVEIT] Missing XRDF: ${XRDF_PATH}" >&2
  echo "[CUMOTION-MOVEIT] Rebuild/source the patched franka_fr3_moveit_config workspace first." >&2
  exit 1
fi
if [[ ! -f "${FR3_XACRO}" ]]; then
  echo "[CUMOTION-MOVEIT] Missing FR3 xacro: ${FR3_XACRO}" >&2
  exit 1
fi

if [[ "${CHECK_ROS_GRAPH}" -eq 1 ]]; then
  EXISTING_NODES="$(timeout 5s ros2 node list 2>/dev/null || true)"
  if grep -qxE '/move_group|/cumotion_planner|/cumotion_action_server' <<<"${EXISTING_NODES}"; then
    echo "[CUMOTION-MOVEIT] Existing /move_group or cuMotion node detected." >&2
    echo "[CUMOTION-MOVEIT] Stop the old launch first, or rerun with --skip-ros-graph-check." >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "${URDF_PATH}")"
echo "[CUMOTION-MOVEIT] Generating FR3 URDF: ${URDF_PATH}"
xacro "${FR3_XACRO}" \
  hand:=true \
  arm_id:=fr3 \
  ros2_control:=false \
  robot_ip:="${ROBOT_IP}" \
  use_fake_hardware:=true \
  fake_sensor_commands:=true \
  > "${URDF_PATH}"

trap cleanup EXIT INT TERM

echo "[CUMOTION-MOVEIT] Starting cuMotion planner node."
ros2 run isaac_ros_cumotion cumotion_planner_node --ros-args \
  -p robot:="${XRDF_PATH}" \
  -p urdf_path:="${URDF_PATH}" \
  -p tool_frame:="${TOOL_FRAME}" \
  -p joint_states_topic:="${JOINT_STATES_TOPIC}" &
PIDS+=("$!")

echo "[CUMOTION-MOVEIT] Starting FR3 MoveIt fake-hardware launch."
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:="${ROBOT_IP}" \
  use_fake_hardware:=true \
  fake_sensor_commands:=true &
PIDS+=("$!")

echo "[CUMOTION-MOVEIT] Started. Leave this terminal running. Press Ctrl-C to stop both processes."
set +e
wait -n "${PIDS[@]}"
status=$?
echo "[CUMOTION-MOVEIT] One process exited with status ${status}; stopping the rest."
exit "${status}"
