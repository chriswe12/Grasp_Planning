#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

job_id="${1:-}"
poll_seconds="${2:-60}"

if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 JOB_ID [POLL_SECONDS]" >&2
    exit 2
fi
if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] POLL_SECONDS must be a positive integer" >&2
    exit 2
fi

echo "[INFO] Waiting for Euler job ${job_id}; keep this PC and terminal running"

watch_cache="$(mktemp -d "/tmp/euler-watch-${job_id}.XXXXXX")"
cleanup() {
    rm -rf -- "${watch_cache}"
}
trap cleanup EXIT

format_duration() {
    local total_seconds="${1}"
    local days hours minutes seconds
    days=$((total_seconds / 86400))
    hours=$(((total_seconds % 86400) / 3600))
    minutes=$(((total_seconds % 3600) / 60))
    seconds=$((total_seconds % 60))
    if (( days > 0 )); then
        printf '%dd %02dh %02dm' "${days}" "${hours}" "${minutes}"
    elif (( hours > 0 )); then
        printf '%dh %02dm %02ds' "${hours}" "${minutes}" "${seconds}"
    else
        printf '%dm %02ds' "${minutes}" "${seconds}"
    fi
}

previous_epoch=0
previous_elapsed=0
max_iterations_pattern='--max_iterations[[:space:]]+([0-9]+)'

while true; do
    job_record="$(
        ssh "${EULER_LOGIN}" \
            "sacct -X -j '${job_id}' --noheader --parsable2 --format=State,ExitCode,ElapsedRaw | head -n 1" \
            2>/dev/null || true
    )"
    IFS='|' read -r state exit_code elapsed_raw <<<"${job_record}"
    elapsed_raw="${elapsed_raw:-0}"

    case "${state}" in
        PENDING*|RUNNING*|CONFIGURING*|COMPLETING*|SUSPENDED*|RESIZING*|REQUEUED*|REQUEUE_*|SIGNALING*|STAGE_OUT*)
            progress_line="$(
                ssh "${EULER_LOGIN}" \
                    "grep -E 'fps step:.*epoch: [0-9]+/[0-9]+' '${EULER_RUNS_DIR}/slurm-${job_id}.out' 2>/dev/null | tail -n 1" \
                    2>/dev/null || true
            )"
            current_epoch=""
            total_epochs=""
            fps="unknown"
            if [[ "${progress_line}" =~ epoch:[[:space:]]+([0-9]+)/([0-9]+) ]]; then
                current_epoch="${BASH_REMATCH[1]}"
                total_epochs="${BASH_REMATCH[2]}"
                if [[ "${progress_line}" =~ fps[[:space:]]total:[[:space:]]([0-9]+) ]]; then
                    fps="${BASH_REMATCH[1]}"
                fi
            else
                command_line="$(
                    ssh "${EULER_LOGIN}" \
                        "grep -m1 'Running mode=train:' '${EULER_RUNS_DIR}/slurm-${job_id}.out' 2>/dev/null" \
                        2>/dev/null || true
                )"
                if [[ "${command_line}" =~ ${max_iterations_pattern} ]]; then
                    total_epochs="${BASH_REMATCH[1]}"
                fi

                container_run_dir="$(
                    ssh "${EULER_LOGIN}" \
                        "grep -m1 '^Exact experiment name requested from command line: /' '${EULER_RUNS_DIR}/slurm-${job_id}.out' 2>/dev/null | sed 's/^Exact experiment name requested from command line: //'" \
                        2>/dev/null || true
                )"
                event_record=""
                container_logs_prefix=/workspace/grasping_rl/logs
                if [[ "${container_run_dir}" == "${container_logs_prefix}"/* ]]; then
                    remote_run_dir="${EULER_RUNS_DIR}${container_run_dir#"${container_logs_prefix}"}"
                    event_record="$(
                        ssh "${EULER_LOGIN}" \
                            "find '${remote_run_dir}' -type f -name 'events.out.tfevents.*' -printf '%T@|%p\\n' 2>/dev/null | sort -nr | head -n 1" \
                            2>/dev/null || true
                    )"
                fi
                event_file="${event_record#*|}"
                if [[ -n "${event_record}" && "${event_file}" != "${event_record}" ]]; then
                    local_event="${watch_cache}/$(basename "${event_file}")"
                    if rsync -az --partial \
                        "${EULER_LOGIN}:${event_file}" "${local_event}" \
                        >/dev/null 2>&1; then
                        tensorboard_progress="$(
                            python3 "${SCRIPT_DIR}/read_tensorboard_progress.py" "${local_event}" \
                                2>/dev/null || true
                        )"
                        IFS='|' read -r current_epoch fps <<<"${tensorboard_progress}"
                    fi
                fi
            fi

            if [[ -n "${current_epoch}" && -n "${total_epochs}" ]]; then
                percent=$((100 * current_epoch / total_epochs))

                eta_seconds=0
                rollout_batch="$(
                    ssh "${EULER_LOGIN}" \
                        "grep -m1 -oE 'RL-Games rollout batch=[0-9]+' '${EULER_RUNS_DIR}/slurm-${job_id}.out' 2>/dev/null | sed 's/.*=//'" \
                        2>/dev/null || true
                )"
                eta=calculating
                if [[ "${fps}" =~ ^[1-9][0-9]*$ && "${rollout_batch}" =~ ^[1-9][0-9]*$ ]]; then
                    eta_seconds=$(((total_epochs - current_epoch) * rollout_batch / fps))
                    eta="$(format_duration "${eta_seconds}")"
                elif (( current_epoch > previous_epoch && previous_epoch > 0 && elapsed_raw > previous_elapsed )); then
                    eta_seconds=$((
                        (total_epochs - current_epoch) * (elapsed_raw - previous_elapsed) /
                        (current_epoch - previous_epoch)
                    ))
                    eta="$(format_duration "${eta_seconds}")"
                elif (( current_epoch >= 10 && elapsed_raw > 0 )); then
                    eta_seconds=$((elapsed_raw * (total_epochs - current_epoch) / current_epoch))
                    eta="$(format_duration "${eta_seconds}")"
                fi
                printf \
                    '[%(%Y-%m-%d %H:%M:%S)T] job=%s state=%s epoch=%s/%s (%s%%) fps=%s eta~%s\n' \
                    -1 "${job_id}" "${state}" "${current_epoch}" "${total_epochs}" \
                    "${percent}" "${fps}" "${eta}"
                previous_epoch="${current_epoch}"
                previous_elapsed="${elapsed_raw}"
            else
                elapsed="$(format_duration "${elapsed_raw}")"
                printf \
                    '[%(%Y-%m-%d %H:%M:%S)T] job=%s state=%s elapsed=%s waiting-for-first-epoch\n' \
                    -1 "${job_id}" "${state}" "${elapsed}"
            fi
            sleep "${poll_seconds}"
            ;;
        "")
            echo "[INFO] Job ${job_id} is not visible yet; retrying"
            sleep "${poll_seconds}"
            ;;
        *)
            echo "[INFO] Job ${job_id} finished: state=${state} exit_code=${exit_code}"
            break
            ;;
    esac
done

"${SCRIPT_DIR}/pull_results.sh"

if [[ "${state}" != COMPLETED* ]]; then
    echo "[ERROR] Results were downloaded, but job ${job_id} did not complete successfully" >&2
    exit 1
fi

echo "[INFO] Job ${job_id} completed and results are available under logs/euler/"
