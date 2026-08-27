#!/usr/bin/env bash

# Compatibility shim. New shells should source setup_robot_env.sh directly.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file so its environment remains in the current shell:" >&2
  echo "  source ./setup_robot_env.sh --robots both" >&2
  exit 2
fi

_compat_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[COMPAT] setup_dual_robot_env.sh delegates to setup_robot_env.sh --robots both." >&2
# shellcheck source=setup_robot_env.sh
source "${_compat_repo_root}/setup_robot_env.sh" --robots both "$@"
unset _compat_repo_root
