#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file so its ROS2 environment remains in the current shell:" >&2
  echo "  source ./setup_ros2_hardware_env.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
LBR_SETUP="/home/pdz/lbr-stack/install/setup.bash"
REPO_SETUP="${SCRIPT_DIR}/ros2_ws/install/setup.bash"

for setup_file in "${ROS_SETUP}" "${LBR_SETUP}" "${REPO_SETUP}"; do
  if [[ ! -f "${setup_file}" ]]; then
    echo "[ROS2-ENV] Missing required setup file: ${setup_file}" >&2
    return 1
  fi
done

nounset_was_enabled=0
case "$-" in
  *u*) nounset_was_enabled=1; set +u ;;
esac
# shellcheck source=/dev/null
source "${ROS_SETUP}"
# shellcheck source=/dev/null
source "${LBR_SETUP}"
# shellcheck source=/dev/null
source "${REPO_SETUP}"
if [[ "${nounset_was_enabled}" -eq 1 ]]; then
  set -u
fi
unset nounset_was_enabled setup_file

export ROS_DOMAIN_ID="${GRASP_ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION="${GRASP_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros-log}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"

# Keep every hardware-facing process on one explicit multicast DDS setup.
unset ROS_DISCOVERY_SERVER
unset ROS_STATIC_PEERS
unset ROS_AUTOMATIC_DISCOVERY_RANGE
unset CYCLONEDDS_URI
unset FASTRTPS_DEFAULT_PROFILES_FILE

# run_pipeline.sh normally sanitizes discovery variables. Preserve this known
# hardware setup after sourcing the helper.
export GRASP_KEEP_ROS_DISCOVERY_ENV=1

if ! ros2 pkg prefix "${RMW_IMPLEMENTATION}" >/dev/null 2>&1; then
  echo "[ROS2-ENV] ROS RMW package '${RMW_IMPLEMENTATION}' is not installed." >&2
  return 1
fi

echo "[ROS2-ENV] domain=${ROS_DOMAIN_ID} localhost=${ROS_LOCALHOST_ONLY} rmw=${RMW_IMPLEMENTATION}"
