#!/usr/bin/env bash
set -e
export PYTHONUTF8=1
cd "$(dirname "$(realpath "$0")")"
if [[ "${1:-}" == connect ]]; then shift; exec scripts/franka_policy_connect.sh "$@"; fi
source scripts/setup_franka_policy_env.sh
if [[ "${1:-}" == probe ]]; then
  shift
  exec timeout -k 2s 12s "$franka_policy_stack/read_state" "${1:-192.168.1.200}"
fi
exec python3 scripts/franka_real_policy.py "$@"
