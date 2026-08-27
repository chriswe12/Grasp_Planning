#!/usr/bin/env bash
set -euo pipefail

FILAMENT_LIBRARY="${MUJOCO_FILAMENT_LIBRARY:-/tmp/mujoco-filament-clang-build/lib/libmujoco_filament.so}"
FILAMENT_ASSETS="${MUJOCO_FILAMENT_ASSETS_DIR:-/tmp/mujoco-filament-clang-build/src/experimental/filament/assets}"

if [[ ! -f "${FILAMENT_LIBRARY}" ]]; then
  echo "[ERROR] MuJoCo Filament library not found: ${FILAMENT_LIBRARY}" >&2
  exit 2
fi
if [[ ! -d "${FILAMENT_ASSETS}" ]]; then
  echo "[ERROR] MuJoCo Filament assets not found: ${FILAMENT_ASSETS}" >&2
  exit 2
fi
if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 COMMAND [ARGS...]" >&2
  exit 2
fi

export LD_PRELOAD="${FILAMENT_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
export MUJOCO_GL=disable
export MUJOCO_FILAMENT_ACTIVE=1
export MUJOCO_FILAMENT_ASSETS_DIR="${FILAMENT_ASSETS}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}"
exec "$@"
