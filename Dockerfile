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
    gnupg \
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
    python3 \
    python3-pip \
    python3-venv \
    wget \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      python3-vcstool \
      ros-jazzy-moveit-msgs \
      ros-jazzy-ros-base \
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

RUN bash scripts/download_ros2_dependencies.sh \
    && mkdir -p /opt/grasp_ros2_ws/src \
    && cp -a ros2_ws/src/fp_debug_msgs /opt/grasp_ros2_ws/src/fp_debug_msgs \
    && source /opt/ros/jazzy/setup.bash \
    && cd /opt/grasp_ros2_ws \
    && colcon build --packages-select fp_debug_msgs --symlink-install

RUN python3 -m venv --system-site-packages /opt/grasp-pipeline-venv \
    && /opt/grasp-pipeline-venv/bin/python -m pip install --upgrade pip \
    && /opt/grasp-pipeline-venv/bin/python -m pip install -e ".[test]"

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
ENV PIPELINE_PYTHON=/opt/grasp-pipeline-venv/bin/python
ENV PYTHONPATH=${GRASP_WORKSPACE}:${ISAACLAB_PYTHON_ROOT}

RUN cat >/etc/profile.d/grasp_pythonpath.sh <<'EOF'
export GRASP_WORKSPACE=/workspace/add_isaac
export ISAACLAB_PYTHON_ROOT=/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab
export PIPELINE_PYTHON=/opt/grasp-pipeline-venv/bin/python
export PYTHONPATH="${GRASP_WORKSPACE}:${ISAACLAB_PYTHON_ROOT}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
source /opt/ros/jazzy/setup.bash
source /opt/grasp_ros2_ws/install/setup.bash
if [[ "${GRASP_SOURCE_WORKSPACE_ROS_OVERLAY:-0}" == "1" ]]; then
  if [[ -f "${GRASP_WORKSPACE}/ros2_ws/install/local_setup.bash" ]]; then
    source "${GRASP_WORKSPACE}/ros2_ws/install/local_setup.bash"
  elif [[ -f "${GRASP_WORKSPACE}/ros2_ws/install/setup.bash" ]]; then
    source "${GRASP_WORKSPACE}/ros2_ws/install/setup.bash"
  fi
fi
EOF

CMD ["/bin/bash"]
