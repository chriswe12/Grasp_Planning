#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-add-isaac-pipeline}"
CONTAINER_NAME="${CONTAINER_NAME:-add-isaac-pipeline}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace/add_isaac}"
XAUTH_FILE="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

check_gpu_runtime() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is not available on the host." >&2
        exit 1
    fi
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "Host GPU check failed: nvidia-smi could not communicate with the NVIDIA driver." >&2
        exit 1
    fi
    if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
        echo "Docker does not report an 'nvidia' runtime. Configure NVIDIA Container Toolkit first." >&2
        exit 1
    fi
}

usage() {
    cat <<EOF
Usage: $0 <command> [args...]

Commands:
  build              Build the Docker image
  run [command...]   Run the container, optionally executing command
  stop               Stop the running container
  remove             Remove the container
  shell              Open a shell in the running container
  help               Show this message

Environment overrides:
  IMAGE_NAME          Docker image name (default: ${IMAGE_NAME})
  CONTAINER_NAME      Docker container name (default: ${CONTAINER_NAME})
  CONTAINER_WORKSPACE Container mount path (default: ${CONTAINER_WORKSPACE})
  XAUTHORITY          X11 authority file to mount (default: ${XAUTH_FILE})
EOF
}

build_image() {
    docker build -t "${IMAGE_NAME}" "${WORKSPACE_DIR}"
}

run_container() {
    local xauth_args=()
    local revoke_xhost=0
    local container_command=("$@")

    check_gpu_runtime

    if [[ -n "${DISPLAY:-}" ]] && command -v xhost >/dev/null 2>&1; then
        xhost +SI:localuser:root >/dev/null
        revoke_xhost=1
        trap 'xhost -SI:localuser:root >/dev/null 2>&1 || true' EXIT
    fi

    if [[ -f "${XAUTH_FILE}" ]]; then
        xauth_args=(
            -e XAUTHORITY=/tmp/.docker.xauth
            -v "${XAUTH_FILE}:/tmp/.docker.xauth:ro"
        )
    fi

    local docker_args=(
        --name "${CONTAINER_NAME}"
        -it
        --runtime=nvidia
        --gpus all
        --network host
        --ipc host
        --entrypoint /bin/bash
        -e ACCEPT_EULA=Y
        -e PRIVACY_CONSENT=Y
        -e NVIDIA_VISIBLE_DEVICES=all
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -e DISPLAY="${DISPLAY:-:0}"
        -e QT_X11_NO_MITSHM=1
        -e GRASP_WORKSPACE="${CONTAINER_WORKSPACE}"
        -e PIPELINE_PYTHON=/opt/grasp-pipeline-venv/bin/python
        -e PYTHONPATH="${CONTAINER_WORKSPACE}:/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab"
        "${xauth_args[@]}"
        -v "${WORKSPACE_DIR}:${CONTAINER_WORKSPACE}"
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
        -w "${CONTAINER_WORKSPACE}"
        "${IMAGE_NAME}"
    )

    if [[ "${#container_command[@]}" -gt 0 ]]; then
        docker run "${docker_args[@]}" -lc "${container_command[*]}"
    else
        docker run "${docker_args[@]}" -l
    fi

    if [[ "${revoke_xhost}" -eq 1 ]]; then
        xhost -SI:localuser:root >/dev/null 2>&1 || true
        trap - EXIT
    fi
}

stop_container() {
    docker stop "${CONTAINER_NAME}"
}

remove_container() {
    docker rm "${CONTAINER_NAME}"
}

open_shell() {
    docker exec -it "${CONTAINER_NAME}" /bin/bash -l
}

COMMAND="${1:-help}"
shift || true

case "${COMMAND}" in
    build)
        build_image
        ;;
    run)
        run_container "$@"
        ;;
    stop)
        stop_container
        ;;
    remove)
        remove_container
        ;;
    shell)
        open_shell
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown command: ${COMMAND}" >&2
        usage >&2
        exit 1
        ;;
esac
