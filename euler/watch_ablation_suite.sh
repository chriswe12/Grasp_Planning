#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=euler.env
source "${SCRIPT_DIR}/euler.env"

manifest="${1:-}"
poll_seconds="${2:-60}"
if [[ ! -f "${manifest}" ]]; then
    echo "usage: $0 MANIFEST.json [POLL_SECONDS]" >&2
    exit 2
fi
if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] POLL_SECONDS must be a positive integer" >&2
    exit 2
fi

mapfile -t job_ids < <(
    python3 -c 'import json,sys; print(*[run["job_id"] for run in json.load(open(sys.argv[1]))["runs"]], sep="\n")' \
        "${manifest}"
)
if (( ${#job_ids[@]} == 0 )); then
    echo "[ERROR] Manifest contains no jobs" >&2
    exit 2
fi
job_list="$(IFS=,; echo "${job_ids[*]}")"
suite_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["suite_id"])' "${manifest}")"
status_path="${REPO_ROOT}/logs/euler/ablation-suite-${suite_id}.sacct.txt"
verification_path="${REPO_ROOT}/logs/euler/ablation-suite-${suite_id}.verification.json"

echo "[INFO] Watching Euler ablation suite ${suite_id}: ${job_list}"
while true; do
    records="$(
        ssh "${EULER_LOGIN}" \
            "sacct -X -j '${job_list}' --noheader --parsable2 --format=JobIDRaw,State,ExitCode" \
            2>/dev/null || true
    )"
    all_terminal=1
    for job_id in "${job_ids[@]}"; do
        record="$(printf '%s\n' "${records}" | awk -F'|' -v id="${job_id}" '$1 == id {print; exit}')"
        IFS='|' read -r _ state exit_code <<<"${record}"
        state="${state:-UNKNOWN}"
        exit_code="${exit_code:-unknown}"
        printf '[%(%Y-%m-%d %H:%M:%S)T] job=%s state=%s exit=%s\n' -1 "${job_id}" "${state}" "${exit_code}"
        case "${state}" in
            COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*)
                ;;
            *)
                all_terminal=0
                ;;
        esac
    done
    (( all_terminal == 1 )) && break
    sleep "${poll_seconds}"
done

mkdir -p "${REPO_ROOT}/logs/euler"
ssh "${EULER_LOGIN}" \
    "sacct -X -j '${job_list}' --noheader --parsable2 --format=JobIDRaw,JobName,State,Elapsed,ExitCode,NodeList" \
    >"${status_path}"

# This helper serializes against every per-job watcher and retries transient
# SSH failures indefinitely.
"${SCRIPT_DIR}/pull_results.sh"

terminal_failure=0
while IFS='|' read -r job_id _ state _ exit_code _; do
    if [[ "${state}" != COMPLETED* || "${exit_code}" != "0:0" ]]; then
        echo "[ERROR] Job ${job_id} ended state=${state} exit=${exit_code}" >&2
        terminal_failure=1
    fi
done <"${status_path}"

verification_status=0
python3 "${SCRIPT_DIR}/verify_pulled_ablation_suite.py" \
    "${manifest}" \
    --logs-root "${REPO_ROOT}/logs/euler" \
    --output "${verification_path}" \
    || verification_status=$?
if (( terminal_failure != 0 || verification_status != 0 )); then
    echo "[ERROR] Ablation suite ${suite_id} failed final verification" >&2
    exit 1
fi
echo "[INFO] Ablation suite ${suite_id} completed, pulled, and verified locally"
