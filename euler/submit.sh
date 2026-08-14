#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

mode="${1:-}"
case "${mode}" in
    smoke|probe|train)
        shift
        ;;
    *)
        echo "usage: $0 {smoke|probe|train} [arguments...]" >&2
        exit 2
        ;;
esac

"${SCRIPT_DIR}/sync_project.sh"

if ! ssh "${EULER_LOGIN}" "test -f '${EULER_IMAGE_PATH}'"; then
    echo "[ERROR] Euler image is missing. Run: ${SCRIPT_DIR}/push_image.sh" >&2
    exit 1
fi

job_name="grasp-rgbd-${mode}"
time_limit="${EULER_TRAIN_TIME_LIMIT:-2-00:00:00}"
if [[ "${mode}" == "smoke" ]]; then
    time_limit=00:30:00
elif [[ "${mode}" == "probe" ]]; then
    time_limit=01:00:00
fi

remote_command=(
    sbatch
    --account="${EULER_ACCOUNT}"
    --job-name="${job_name}"
    --time="${time_limit}"
    --chdir="${EULER_PROJECT_DIR}"
    --export="ALL,EULER_CONFIG_PATH=${EULER_PROJECT_DIR}/euler/euler.env"
    --output="${EULER_RUNS_DIR}/slurm-%j.out"
    --error="${EULER_RUNS_DIR}/slurm-%j.err"
    "${EULER_PROJECT_DIR}/euler/job.sbatch"
    "${mode}"
    "$@"
)
printf -v quoted_remote_command '%q ' "${remote_command[@]}"

echo "[INFO] Submitting ${mode} job to account ${EULER_ACCOUNT}"
submission_output="$(ssh "${EULER_LOGIN}" "${quoted_remote_command}")"
echo "${submission_output}"

if [[ "${submission_output}" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
    job_id="${BASH_REMATCH[1]}"
    echo "[INFO] Monitor and automatically pull when finished:"
    echo "${SCRIPT_DIR}/watch_and_pull.sh ${job_id}"
    if [[ "${mode}" == "train" && " $* " == *MultiPart* ]]; then
        echo "[INFO] Or periodically validate saved checkpoints on this PC and select the best:"
        echo "${SCRIPT_DIR}/watch_validate_and_pull.sh ${job_id} 1000"
    fi
fi
