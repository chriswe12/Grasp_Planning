#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

ssh_control_dir="$(mktemp -d /tmp/isaaclab-euler-ssh.XXXXXX)"
ssh_control_path="${ssh_control_dir}/control"
ssh_options=(
    -o ControlMaster=auto
    -o ControlPersist=30m
    -o "ControlPath=${ssh_control_path}"
)
smoke_dir=""

cleanup() {
    ssh -o "ControlPath=${ssh_control_path}" -O exit \
        "${EULER_LOGIN}" >/dev/null 2>&1 || true
    if [[ -n "${smoke_dir}" ]]; then
        rm -rf -- "${smoke_dir}"
    fi
    rm -rf -- "${ssh_control_dir}"
}
trap cleanup EXIT

for command_name in apptainer docker rsync ssh; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: ${command_name}" >&2
        exit 1
    fi
done

echo "[INFO] Checking SSH authentication before local image work"
ssh "${ssh_options[@]}" "${EULER_LOGIN}" true

if ! docker image inspect "${EULER_DOCKER_IMAGE}" >/dev/null 2>&1; then
    echo "[ERROR] Docker image not found: ${EULER_DOCKER_IMAGE}" >&2
    exit 1
fi

if [[ "${EULER_LOCAL_IMAGE}" = /* ]]; then
    local_image="${EULER_LOCAL_IMAGE}"
else
    local_image="${REPO_ROOT}/${EULER_LOCAL_IMAGE}"
fi

mkdir -p "$(dirname "${local_image}")"

force_rebuild=false
case "${1:-}" in
    --force)
        force_rebuild=true
        ;;
    "")
        ;;
    *)
        echo "usage: $0 [--force]" >&2
        exit 2
        ;;
esac

if [[ -e "${local_image}" && "${force_rebuild}" == false ]]; then
    echo "[INFO] Reusing existing local image: ${local_image}"
fi

if [[ ! -e "${local_image}" || "${force_rebuild}" == true ]]; then
    building_image="${local_image}.building"
    rm -f "${building_image}"
    echo "[INFO] Converting ${EULER_DOCKER_IMAGE} to ${local_image}"
    if ! apptainer build --fakeroot "${building_image}" "docker-daemon://${EULER_DOCKER_IMAGE}"; then
        rm -f "${building_image}"
        echo "[ERROR] Image conversion failed; the previous SIF was left unchanged" >&2
        exit 1
    fi
    mv -f "${building_image}" "${local_image}"
fi

echo "[INFO] Validating the external Isaac project inside the Apptainer image"
APPTAINERENV_ACCEPT_EULA=Y \
APPTAINERENV_PRIVACY_CONSENT=Y \
APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 \
APPTAINERENV_PYTHONPATH="${EULER_CONTAINER_PYTHONPATH}" \
    apptainer exec --nv --containall \
    --bind "${REPO_ROOT}:/workspace/grasping_rl:ro" \
    "${local_image}" \
    /isaac-sim/python.sh -c \
    "import importlib.util; import grasp_planning; assert importlib.util.find_spec('isaaclab'); assert importlib.util.find_spec('isaac_rl'); print('Isaac Lab, isaac_rl, and grasp_planning package paths OK')"

smoke_dir="$(mktemp -d /tmp/isaaclab-euler-smoke.XXXXXX)"
mkdir -p \
    "${smoke_dir}/cache/kit" \
    "${smoke_dir}/cache/ov" \
    "${smoke_dir}/cache/pip" \
    "${smoke_dir}/cache/glcache" \
    "${smoke_dir}/cache/computecache" \
    "${smoke_dir}/logs" \
    "${smoke_dir}/data" \
    "${smoke_dir}/documents"

echo "[INFO] Running a one-environment, one-step simulator smoke test"
APPTAINERENV_ACCEPT_EULA=Y \
APPTAINERENV_PRIVACY_CONSENT=Y \
APPTAINERENV_PYTHONDONTWRITEBYTECODE=1 \
APPTAINERENV_PYTHONPATH="${EULER_CONTAINER_PYTHONPATH}" \
APPTAINERENV_TERM=xterm \
    apptainer exec --nv --containall --writable-tmpfs \
    --bind "${REPO_ROOT}:/workspace/grasping_rl:ro" \
    --bind "${smoke_dir}/cache/kit:/isaac-sim/kit/cache:rw" \
    --bind "${smoke_dir}/cache/ov:${HOME}/.cache/ov:rw" \
    --bind "${smoke_dir}/cache/pip:${HOME}/.cache/pip:rw" \
    --bind "${smoke_dir}/cache/glcache:${HOME}/.cache/nvidia/GLCache:rw" \
    --bind "${smoke_dir}/cache/computecache:${HOME}/.nv/ComputeCache:rw" \
    --bind "${smoke_dir}/logs:${HOME}/.nvidia-omniverse/logs:rw" \
    --bind "${smoke_dir}/data:${HOME}/.local/share/ov/data:rw" \
    --bind "${smoke_dir}/documents:${HOME}/Documents:rw" \
    --pwd /workspace/grasping_rl \
    "${local_image}" \
    /isaac-sim/python.sh isaac_rl/scripts/smoke_env.py \
    --task Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0 \
    --steps 1 --num_envs 1 --headless

remote_image_dir="$(dirname "${EULER_IMAGE_PATH}")"
ssh "${ssh_options[@]}" "${EULER_LOGIN}" "mkdir -p '${remote_image_dir}'"
echo "[INFO] Uploading image to ${EULER_LOGIN}:${EULER_IMAGE_PATH}"
rsync -ah --info=progress2 --partial \
    -e "ssh -o ControlPath=${ssh_control_path}" \
    "${local_image}" "${EULER_LOGIN}:${EULER_IMAGE_PATH}"

ssh "${ssh_options[@]}" "${EULER_LOGIN}" \
    "apptainer inspect '${EULER_IMAGE_PATH}' >/dev/null && echo '[INFO] Remote Apptainer image OK'"
