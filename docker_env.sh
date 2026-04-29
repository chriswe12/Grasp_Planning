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
  pipeline [args...] Run run_pipeline.sh inside the container
  play-bag <bag>     Play a ROS2 bag from a matching ROS2 container
  stop               Stop the running container
  remove             Remove the container
  shell              Open a shell in the running container
  help               Show this message

Environment overrides:
  IMAGE_NAME          Docker image name (default: ${IMAGE_NAME})
  CONTAINER_NAME      Docker container name (default: ${CONTAINER_NAME})
  CONTAINER_WORKSPACE Container mount path (default: ${CONTAINER_WORKSPACE})
  ROS_DOMAIN_ID       ROS2 domain passed into Docker (default: ${ROS_DOMAIN_ID:-0})
  GRASP_PIPELINE_ROS_DOMAIN_ID
                      ROS2 domain for pipeline/play-bag commands (default: ROS_DOMAIN_ID)
  FASTDDS_BUILTIN_TRANSPORTS
                      Fast DDS transport passed into Docker (default: ${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4})
  XAUTHORITY          X11 authority file to mount (default: ${XAUTH_FILE})
EOF
}

build_image() {
    docker build -t "${IMAGE_NAME}" "${WORKSPACE_DIR}"
}

docker_tty_args() {
    if [[ -t 0 && -t 1 ]]; then
        printf '%s\n' -it
    else
        printf '%s\n' -i
    fi
}

run_container() {
    local xauth_args=()
    local ros_env_args=()
    local ros_domain_id="${ROS_DOMAIN_ID:-0}"
    local fastdds_transports="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
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

    ros_env_args=(
        -e "ROS_DOMAIN_ID=${ros_domain_id}"
        -e "FASTDDS_BUILTIN_TRANSPORTS=${fastdds_transports}"
    )
    if [[ "${GRASP_FORWARD_ROS_ENV:-0}" == "1" ]]; then
        ros_env_args=()
        for env_name in \
            ROS_DOMAIN_ID \
            ROS_AUTOMATIC_DISCOVERY_RANGE \
            ROS_STATIC_PEERS \
            RMW_IMPLEMENTATION \
            CYCLONEDDS_URI \
            FASTRTPS_DEFAULT_PROFILES_FILE \
            FASTDDS_BUILTIN_TRANSPORTS
        do
            if [[ -n "${!env_name:-}" ]]; then
                ros_env_args+=(-e "${env_name}=${!env_name}")
            fi
        done
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
        -e GRASP_SKIP_WORKSPACE_ROS_OVERLAY=1
        -e PIPELINE_PYTHON=/opt/grasp-pipeline-venv/bin/python
        -e PYTHONPATH="${CONTAINER_WORKSPACE}:/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source/isaaclab"
        "${xauth_args[@]}"
        "${ros_env_args[@]}"
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

container_exists() {
    docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
    [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

ensure_container_running() {
    if container_running; then
        return 0
    fi
    if container_exists; then
        docker start "${CONTAINER_NAME}" >/dev/null
        return 0
    fi

    echo "Container '${CONTAINER_NAME}' does not exist. Start it first with: $0 run" >&2
    exit 1
}

exec_in_container() {
    local ros_domain_id="${ROS_DOMAIN_ID:-0}"
    local fastdds_transports="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
    local tty_args=(-i)
    if [[ -t 0 && -t 1 ]]; then
        tty_args=(-it)
    fi

    docker exec "${tty_args[@]}" \
        -e "ROS_DOMAIN_ID=${ros_domain_id}" \
        -e "FASTDDS_BUILTIN_TRANSPORTS=${fastdds_transports}" \
        "${CONTAINER_NAME}" \
        "$@"
}

run_pipeline() {
    ensure_container_running
    ROS_DOMAIN_ID="${GRASP_PIPELINE_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}" exec_in_container /bin/bash -lc \
        'cd "${GRASP_WORKSPACE:-/workspace/add_isaac}"; exec ./run_pipeline.sh "$@"' \
        bash "$@"
}

play_bag() {
    if [[ $# -lt 1 ]]; then
        echo "Usage: $0 play-bag <bag_path> [ros2 bag play args...]" >&2
        exit 1
    fi

    local bag_path="$1"
    shift
    local bag_abs
    bag_abs="$(realpath "${bag_path}")"
    local bag_parent
    bag_parent="$(dirname "${bag_abs}")"
    local ros_domain_id="${GRASP_PIPELINE_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}"
    local fastdds_transports="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
    local tty_args=()
    mapfile -t tty_args < <(docker_tty_args)

    docker run "${tty_args[@]}" --rm \
        --runtime=nvidia \
        --gpus all \
        --network host \
        --ipc host \
        --entrypoint /bin/bash \
        -e ACCEPT_EULA=Y \
        -e PRIVACY_CONSENT=Y \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e "ROS_DOMAIN_ID=${ros_domain_id}" \
        -e "FASTDDS_BUILTIN_TRANSPORTS=${fastdds_transports}" \
        -v "${bag_parent}:${bag_parent}:ro" \
        "${IMAGE_NAME}" \
        -lc 'source /opt/ros/jazzy/setup.bash; source /opt/grasp_ros2_ws/install/setup.bash; exec ros2 bag play --loop "$@"' \
        bash "${bag_abs}" "$@"
}

open_shell() {
    ensure_container_running
    exec_in_container /bin/bash -l
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
    pipeline)
        run_pipeline "$@"
        ;;
    play-bag)
        play_bag "$@"
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
