#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SERVO_GRIPPER_WORKSPACE:-${SCRIPT_DIR}}"
ROS_DOMAIN="${DUAL_ROBOT_ROS_DOMAIN_ID:-0}"

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

source_if_exists "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source_if_exists "${WORKSPACE_ROOT}/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
unset ROS_DISCOVERY_SERVER
unset ROS_STATIC_PEERS
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset CYCLONEDDS_URI
unset FASTRTPS_DEFAULT_PROFILES_FILE

echo "[DUAL-GRIPPERS] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[DUAL-GRIPPERS] ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
echo "[DUAL-GRIPPERS] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[DUAL-GRIPPERS] Starting lbr_one and lbr_two gripper endpoints."

exec ros2 launch servo_gripper dual_grippers.launch.py "$@"
