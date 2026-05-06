#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Copy ignored local assets and ROS2 build state from one worktree to another.

Usage:
  scripts/copy_worktree_local_state.sh --to NEW_WORKTREE [--from SOURCE_WORKTREE]
  scripts/copy_worktree_local_state.sh NEW_WORKTREE

Defaults:
  --from defaults to the current repository root.

Copied paths:
  .cache/generated_mujoco_models/
  .cache/robot_descriptions/
  ros2_ws/src/fp_debug_msgs/
  ros2_ws/build/
  ros2_ws/install/
  ros2_ws/log/

Notes:
  This script copies only when the source path exists. It does not delete files
  from the destination.
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

repo_root() {
  git -C "$1" rev-parse --show-toplevel 2>/dev/null
}

copy_dir() {
  local src_root=$1
  local dst_root=$2
  local rel=$3
  local src=$src_root/$rel
  local dst=$dst_root/$rel

  if [[ ! -e "$src" ]]; then
    printf 'skip missing %s\n' "$rel"
    return 0
  fi
  if [[ ! -d "$src" ]]; then
    printf 'skip non-directory %s\n' "$rel"
    return 0
  fi

  mkdir -p "$dst"
  rsync -a "$src"/ "$dst"/
  printf 'copied %s\n' "$rel"
}

from_arg=
to_arg=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      [[ $# -ge 2 ]] || die '--from requires a path'
      from_arg=$2
      shift 2
      ;;
    --to)
      [[ $# -ge 2 ]] || die '--to requires a path'
      to_arg=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      [[ -z "$to_arg" ]] || die "unexpected extra argument: $1"
      to_arg=$1
      shift
      ;;
  esac
done

if [[ -z "$from_arg" ]]; then
  from_arg=$(repo_root .) || die 'current directory is not inside a git repository'
fi
[[ -n "$to_arg" ]] || die 'missing destination worktree path'

from_root=$(repo_root "$from_arg") || die "source is not inside a git repository: $from_arg"
to_root=$(repo_root "$to_arg") || die "destination is not inside a git repository: $to_arg"

from_root=$(cd "$from_root" && pwd -P)
to_root=$(cd "$to_root" && pwd -P)

[[ "$from_root" != "$to_root" ]] || die 'source and destination worktrees are the same'

paths=(
  ".cache/generated_mujoco_models"
  ".cache/robot_descriptions"
  "ros2_ws/src/fp_debug_msgs"
  "ros2_ws/build"
  "ros2_ws/install"
  "ros2_ws/log"
)

printf 'source:      %s\n' "$from_root"
printf 'destination: %s\n' "$to_root"

for rel in "${paths[@]}"; do
  copy_dir "$from_root" "$to_root" "$rel"
done
