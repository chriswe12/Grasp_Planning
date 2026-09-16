#!/usr/bin/env bash
# Select the PyTorch-pinned CUDA libraries shipped with this Isaac image.
set -euo pipefail
if [[ -n "${FRANKA_NGX_LIBRARY:-}" ]]; then
    test -f "${FRANKA_NGX_LIBRARY}"
    export LD_LIBRARY_PATH="$(dirname "${FRANKA_NGX_LIBRARY}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
bootstrap=()
case "${FRANKA_CUDNN_MODE:-compatible}" in
  compatible)
    franka_cudnn=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cudnn/lib
    franka_nccl=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib
    test -f "$franka_cudnn/libcudnn.so.9"
    test -f "$franka_nccl/libnccl.so.2"
    export LD_LIBRARY_PATH="$franka_cudnn:$franka_nccl${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export FRANKA_CUDA_PRELOAD="$franka_cudnn/libcudnn.so.9:$franka_nccl/libnccl.so.2"
    bootstrap=("$(dirname "$0")/franka_python_bootstrap.py")
    export ISAAC_RL_DISABLE_CUDNN=0
    ;;
  native) export ISAAC_RL_DISABLE_CUDNN=0 ;;
  disabled) export ISAAC_RL_DISABLE_CUDNN=1 ;;
  *) echo 'FRANKA_CUDNN_MODE must be compatible, native, or disabled' >&2; exit 2 ;;
esac
exec /workspace/isaaclab/isaaclab.sh -p "${bootstrap[@]}" "$@"
