#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_SH="${ISAACLAB_SH:-/media/pdz/Elements1/IsaacLab/isaaclab.sh}"

exec "${ISAACLAB_SH}" -p "${REPO_ROOT}/scripts/show_isaac_gripper_collisions.py" "$@"
