#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-grasp-planning-isaac}"
CONTAINER_NAME="${CONTAINER_NAME:-grasp-planning}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XAUTH_FILE="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
  build    Build the Docker image
  run      Run the container interactively
  stop     Stop the running container
  remove   Remove the container
  shell    Open a shell in the running container
  help     Show this message

Environment overrides:
  IMAGE_NAME       Docker image name (default: ${IMAGE_NAME})
  CONTAINER_NAME   Docker container name (default: ${CONTAINER_NAME})
  XAUTHORITY       X11 authority file to mount (default: ${XAUTH_FILE})
EOF
}

build_image() {
    docker build -t "${IMAGE_NAME}" "${WORKSPACE_DIR}"
}

run_container() {
    local xauth_args=()
    local revoke_xhost=0

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

    docker run --name "${CONTAINER_NAME}" -it \
        --gpus all \
        --network host \
        --ipc host \
        --entrypoint /bin/bash \
        -e ACCEPT_EULA=Y \
        -e PRIVACY_CONSENT=Y \
        -e DISPLAY="${DISPLAY:-:0}" \
        -e QT_X11_NO_MITSHM=1 \
        "${xauth_args[@]}" \
        -v "${WORKSPACE_DIR}:/workspace/Grasp_Planning" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${IMAGE_NAME}" \
        -l

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

case "${COMMAND}" in
    build)
        build_image
        ;;
    run)
        run_container
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
