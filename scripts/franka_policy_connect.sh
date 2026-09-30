#!/usr/bin/env bash
set -e
# Do not run alongside another hardware interface to the same arm.
robot_ip=192.168.1.200
fake=false
inspect_only=false
for arg in "$@"; do
  case "$arg" in
    robot_ip:=*) robot_ip="${arg#robot_ip:=}" ;;
    fake:=*) fake="${arg#fake:=}" ;;
    --help|-h|--show-args|-s|--show-arguments|--print-description|-p) inspect_only=true ;;
  esac
done
if [[ "$fake" != true && "$inspect_only" != true ]]; then
  if ! ping -n -c 1 -W 2 "$robot_ip" >/dev/null 2>&1; then
    echo "Robot $robot_ip did not answer the network preflight; ROS was not started." >&2
    ip route get "$robot_ip" >&2 || true
    echo "Check the control-box Ethernet cable and give that interface an address on the robot subnet." >&2
    echo "See docs/franka-policy.md (Connection troubleshooting). No network settings were changed." >&2
    exit 1
  fi
fi
source "$(dirname "$(realpath "$0")")/setup_franka_policy_env.sh"
exec ros2 launch "$(dirname "$(realpath "$0")")/franka_policy_connect.launch.py" "$@"
