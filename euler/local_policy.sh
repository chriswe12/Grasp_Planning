#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

mode="${1:-}"
if [[ -z "${mode}" ]]; then
    echo "usage: $0 {tensorboard|play|evaluate|debug-videos} [arguments...]" >&2
    exit 2
fi
shift

if [[ "${mode}" == "tensorboard" ]]; then
    exec tensorboard \
        --logdir "${REPO_ROOT}/logs/euler/rl_games" \
        --port "${TENSORBOARD_PORT:-6006}"
fi

for command_name in docker nvidia-smi; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: ${command_name}" >&2
        exit 1
    fi
done

policy="${ISAAC_RL_POLICY:-single}"
case "${policy}" in
    single)
        experiment_name=grasp_visual_servo_rgbd
        checkpoint_name=grasp_visual_servo_rgbd.pth
        play_task=Grasp-Visual-Servo-RGBD-Direct-Play-v0
        catalog_split=all
        ;;
    multipart)
        experiment_name=grasp_visual_servo_rgbd_multipart
        checkpoint_name=grasp_visual_servo_rgbd_multipart.pth
        play_task=Grasp-Visual-Servo-RGBD-MultiPart-Direct-Play-v0
        catalog_split=validation
        ;;
    *)
        echo "[ERROR] ISAAC_RL_POLICY must be 'single' or 'multipart', got: ${policy}" >&2
        exit 2
        ;;
esac

if [[ -n "${ISAAC_RL_CHECKPOINT:-}" ]]; then
    checkpoint="$(realpath "${ISAAC_RL_CHECKPOINT}")"
else
    checkpoint=""
    if [[ "${policy}" == multipart ]]; then
        selection_record="$(
            find "${REPO_ROOT}/logs/euler/rl_games/${experiment_name}" \
                -type f -path '*/evaluations/periodic_validation/best_checkpoint.txt' \
                -printf '%T@|%p\n' 2>/dev/null | sort -nr | head -n 1 || true
        )"
        selection_file="${selection_record#*|}"
        if [[ -n "${selection_record}" && "${selection_file}" != "${selection_record}" ]]; then
            selected_checkpoint="$(head -n 1 "${selection_file}")"
            if [[ -f "${selected_checkpoint}" ]]; then
                checkpoint="$(realpath "${selected_checkpoint}")"
                echo "[INFO] Using held-out-validation checkpoint selection"
            fi
        fi
    fi
fi

if [[ -z "${checkpoint:-}" ]]; then
    checkpoint_record="$(
        find "${REPO_ROOT}/logs/euler/rl_games/${experiment_name}" \
            -type f -path "*/nn/${checkpoint_name}" \
            -printf '%T@|%p\n' 2>/dev/null | sort -nr | head -n 1 || true
    )"
    checkpoint="${checkpoint_record#*|}"
fi

if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "[ERROR] No downloaded ${policy} best checkpoint found under logs/euler/rl_games" >&2
    exit 1
fi
case "${checkpoint}" in
    "${REPO_ROOT}"/*)
        container_checkpoint="/workspace/grasping_rl/${checkpoint#"${REPO_ROOT}/"}"
        ;;
    *)
        echo "[ERROR] Checkpoint must be inside this repository: ${checkpoint}" >&2
        exit 1
        ;;
esac

common_args=(
    --checkpoint "${container_checkpoint}"
    --headless
)
case "${mode}" in
    play)
        python_script=isaac_rl/scripts/rl_games/play.py
        default_args=(
            --task "${play_task}"
            --catalog_split "${catalog_split}"
            --num_envs 1
            --target_index 27
            --reset_progress 0.85
            --reset_noise_rad 0.005
            --reset_rotation_deg 15
            --video
            --video_length 450
            --enable_cameras
        )
        ;;
    evaluate)
        python_script=isaac_rl/scripts/rl_games/evaluate_multigrasp.py
        default_args=(
            --task "${play_task}"
            --catalog_split "${catalog_split}"
            --runs_per_target 3
            --episode_seconds 15
            --conditions far mid close
            --rotation_deg 15
        )
        ;;
    debug-videos)
        python_script=isaac_rl/scripts/rl_games/record_debug_videos.py
        default_args=(
            --task "${play_task}"
            --catalog_split "${catalog_split}"
            --conditions far mid close exact
            --episode_seconds 15
        )
        ;;
    *)
        echo "usage: $0 {tensorboard|play|evaluate|debug-videos} [arguments...]" >&2
        exit 2
        ;;
esac

echo "[INFO] Using checkpoint: ${checkpoint}"
docker run --rm \
    --gpus all \
    --ipc host \
    --network none \
    --entrypoint /isaac-sim/python.sh \
    -e ACCEPT_EULA=Y \
    -e PRIVACY_CONSENT=Y \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONUNBUFFERED=1 \
    -e ISAAC_RL_DISABLE_CUDNN=1 \
    -e "PYTHONPATH=${EULER_CONTAINER_PYTHONPATH}" \
    -v "${REPO_ROOT}:/workspace/grasping_rl:rw" \
    -v "${REPO_ROOT}/.cache/euler/torch:/root/.cache/torch:ro" \
    -w /workspace/grasping_rl \
    "${EULER_DOCKER_IMAGE}" \
    "${python_script}" \
    "${common_args[@]}" \
    "${default_args[@]}" \
    "$@"
