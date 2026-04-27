ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAAC_SIM_IMAGE}

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

ARG ISAACLAB_VERSION=2.3.2.post1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    git-lfs \
    libccd-dev \
    libasound2t64 \
    libegl1 \
    libfcl-dev \
    libglib2.0-0 \
    libglvnd0 \
    libgl1 \
    libnss3 \
    libsm6 \
    libxext6 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxi6 \
    pkg-config \
    wget \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip install \
    --no-deps \
    flatdict==4.0.0 \
    h5py \
    "isaaclab==${ISAACLAB_VERSION}" \
    python-fcl \
    --extra-index-url https://pypi.nvidia.com

COPY . /workspace/add_isaac
WORKDIR /workspace/add_isaac

RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip install -e ".[test]"

RUN /isaac-sim/python.sh - <<'PY'
import os
import torch
from isaaclab.app import AppLauncher

isaaclab_source_root = "/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab"
if not os.path.exists(isaaclab_source_root):
    raise SystemExit(f"Missing Isaac Lab source root: {isaaclab_source_root}")

print(torch.__version__)
print(torch.__file__)
print(AppLauncher)
print(isaaclab_source_root)
PY

ENV GRASP_WORKSPACE=/workspace/add_isaac
ENV ISAACLAB_PYTHON_ROOT=/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab
ENV PIPELINE_PYTHON=/isaac-sim/python.sh
ENV PYTHONPATH=${GRASP_WORKSPACE}:${ISAACLAB_PYTHON_ROOT}

RUN cat >/etc/profile.d/grasp_pythonpath.sh <<'EOF'
export GRASP_WORKSPACE=/workspace/add_isaac
export ISAACLAB_PYTHON_ROOT=/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab
export PIPELINE_PYTHON=/isaac-sim/python.sh
export PYTHONPATH="${GRASP_WORKSPACE}:${ISAACLAB_PYTHON_ROOT}"
EOF

CMD ["/bin/bash"]
