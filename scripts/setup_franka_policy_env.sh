#!/usr/bin/env bash
# Source the isolated protocol-10 stack, without chaining the old home workspace.
franka_policy_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
franka_policy_stack="$franka_policy_root/.cache/franka_compatible"
if [[ ! -f "$franka_policy_stack/install/local_setup.bash" ]]; then
  echo "Compatible Franka workspace missing; run scripts/build_franka_policy_stack.sh" >&2
  return 1
fi
# A terminal may already have sourced the incompatible home workspace.
for franka_env_var in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH PYTHONPATH PATH; do
  franka_env_clean=""
  IFS=: read -ra franka_env_parts <<< "${!franka_env_var}"
  for franka_env_part in "${franka_env_parts[@]}"; do
    [[ "$franka_env_part" == "$HOME/franka_ros2_ws"* ]] && continue
    [[ -z "$franka_env_part" ]] && continue
    franka_env_clean+="${franka_env_clean:+:}$franka_env_part"
  done
  export "$franka_env_var=$franka_env_clean"
done
source /opt/ros/humble/setup.bash
source "$franka_policy_stack/install/local_setup.bash"
if [[ -f "$franka_policy_root/.cache/franka_velocity_driver/install/local_setup.bash" ]]; then
  source "$franka_policy_root/.cache/franka_velocity_driver/install/local_setup.bash"
fi
export LD_LIBRARY_PATH="$franka_policy_stack/libfranka/lib:/opt/openrobots/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUTF8=1
