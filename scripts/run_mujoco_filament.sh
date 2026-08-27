#!/usr/bin/env bash
set -euo pipefail

DEFAULT_FILAMENT_LIBRARY="/tmp/mujoco-filament-370-clang11-build/lib/libmujoco.so"
if [[ ! -f "${DEFAULT_FILAMENT_LIBRARY}" ]]; then
    DEFAULT_FILAMENT_LIBRARY="/tmp/mujoco-filament-370-build/lib/libmujoco.so"
fi
if [[ ! -f "${DEFAULT_FILAMENT_LIBRARY}" ]]; then
    DEFAULT_FILAMENT_LIBRARY="/tmp/mujoco-filament-clang-build/lib/libmujoco.so"
fi
FILAMENT_LIBRARY="${MUJOCO_FILAMENT_LIBRARY:-${DEFAULT_FILAMENT_LIBRARY}}"

DEFAULT_FILAMENT_ASSETS="/tmp/mujoco-filament-370-clang11-build/src/experimental/filament/assets"
if [[ ! -d "${DEFAULT_FILAMENT_ASSETS}" ]]; then
    DEFAULT_FILAMENT_ASSETS="/tmp/mujoco-filament-370-build/src/experimental/filament/assets"
fi
if [[ ! -d "${DEFAULT_FILAMENT_ASSETS}" ]]; then
    DEFAULT_FILAMENT_ASSETS="/tmp/mujoco-filament-clang-build/src/render/filament/assets"
fi
if [[ ! -d "${DEFAULT_FILAMENT_ASSETS}" ]]; then
    DEFAULT_FILAMENT_ASSETS="/tmp/mujoco-filament-clang-build/src/experimental/filament/assets"
fi
FILAMENT_ASSETS="${MUJOCO_FILAMENT_ASSETS_DIR:-${DEFAULT_FILAMENT_ASSETS}}"

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
