#!/usr/bin/env bash

# One ROS environment for single-arm, dual-arm, simulation, PITL, real,
# action-server, and benchmark entrypoints. Source it for interactive ros2 use;
# run_pipeline.sh sources it automatically.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file so its environment remains in the current shell:" >&2
  echo "  source ./setup_robot_env.sh [--robots left|right|both] [--ros-domain-id N]" >&2
  exit 2
fi

_robot_env_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_robot_env_lbr_ws="${LBR_WORKSPACE:-/home/pdz/lbr-stack}"
_robot_env_robots="${GRASP_ROBOTS:-both}"
_robot_env_domain="${GRASP_ROS_DOMAIN_ID:-${DUAL_ROBOT_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}}"
_robot_env_quiet=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robots) _robot_env_robots="${2:-}"; shift 2 ;;
    --ros-domain-id) _robot_env_domain="${2:-}"; shift 2 ;;
    --lbr-ws) _robot_env_lbr_ws="${2:-}"; shift 2 ;;
    --quiet) _robot_env_quiet=1; shift ;;
    -h|--help)
      echo "source ./setup_robot_env.sh [--robots left|right|both] [--ros-domain-id N] [--lbr-ws PATH]"
      return 0
      ;;
    *)
      echo "[ROBOT-ENV] Unknown argument: $1" >&2
      return 2
      ;;
  esac
done

if [[ "${_robot_env_robots}" != "left" && "${_robot_env_robots}" != "right" && "${_robot_env_robots}" != "both" ]]; then
  echo "[ROBOT-ENV] --robots must be left, right, or both." >&2
  return 2
fi
if [[ ! "${_robot_env_domain}" =~ ^[0-9]+$ ]]; then
  echo "[ROBOT-ENV] --ros-domain-id must be a non-negative integer." >&2
  return 2
fi

_robot_env_source_if_exists() {
  local setup_file="$1"
  if [[ ! -f "${setup_file}" ]]; then
    return 0
  fi
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
}

_robot_env_source_if_exists "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
_robot_env_source_if_exists "${_robot_env_lbr_ws}/install/setup.bash"
if [[ -f "${_robot_env_repo_root}/ros2_ws/install/local_setup.bash" ]]; then
  _robot_env_source_if_exists "${_robot_env_repo_root}/ros2_ws/install/local_setup.bash"
else
  _robot_env_source_if_exists "${_robot_env_repo_root}/ros2_ws/install/setup.bash"
fi

export GRASP_REPO="${_robot_env_repo_root}"
export GRASP_ROBOTS="${_robot_env_robots}"
export GRASP_ROS_DOMAIN_ID="${_robot_env_domain}"
export DUAL_ROBOT_ROS_DOMAIN_ID="${_robot_env_domain}"
export ROS_DOMAIN_ID="${_robot_env_domain}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION="${GRASP_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros-log}"
export GRASP_KEEP_ROS_DISCOVERY_ENV=1

unset ROS_DISCOVERY_SERVER
unset ROS_STATIC_PEERS
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset CYCLONEDDS_URI
unset FASTRTPS_DEFAULT_PROFILES_FILE

if [[ "${_robot_env_quiet}" -eq 0 ]]; then
  echo "[ROBOT-ENV] repo=${GRASP_REPO} robots=${GRASP_ROBOTS} domain=${ROS_DOMAIN_ID} rmw=${RMW_IMPLEMENTATION}"
fi

unset _robot_env_repo_root _robot_env_lbr_ws _robot_env_robots _robot_env_domain _robot_env_quiet
unset -f _robot_env_source_if_exists
