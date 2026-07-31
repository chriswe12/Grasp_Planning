#!/usr/bin/env bash

# Source this file in each terminal:
#   source ./setup_dual_robot_env.sh
#
# It only configures the shell. It never starts ROS nodes or sends commands.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced so its environment remains in your shell:" >&2
  echo "  source ./setup_dual_robot_env.sh" >&2
  exit 2
fi

_dual_env_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_dual_env_source_if_exists() {
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

_dual_env_source_if_exists "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
_dual_env_source_if_exists "${DUAL_LBR_WORKSPACE:-/home/pdz/lbr-stack}/install/setup.bash"
_dual_env_source_if_exists "${_dual_env_repo_root}/ros2_ws/install/setup.bash"

export GRASP_REPO="${_dual_env_repo_root}"
export DUAL_ROBOT_ROS_DOMAIN_ID="${DUAL_ROBOT_ROS_DOMAIN_ID:-0}"
export ROS_DOMAIN_ID="${DUAL_ROBOT_ROS_DOMAIN_ID}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros-log}"

unset ROS_DISCOVERY_SERVER
unset ROS_STATIC_PEERS
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset CYCLONEDDS_URI
unset FASTRTPS_DEFAULT_PROFILES_FILE

echo "[DUAL-ENV] GRASP_REPO=${GRASP_REPO}"
echo "[DUAL-ENV] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[DUAL-ENV] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"

unset _dual_env_repo_root
unset -f _dual_env_source_if_exists
