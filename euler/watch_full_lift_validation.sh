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
status_path="${REPO_ROOT}/logs/euler/full-lift-suite-${suite_id}.sacct.txt"
output_dir="${REPO_ROOT}/artifacts/scripted_grasp_lift_validation/full_${suite_id}"

echo "[INFO] Watching full lift suite ${suite_id}: ${#job_ids[@]} jobs"
while true; do
    records="$(
        ssh "${EULER_LOGIN}" \
            "sacct -X -j '${job_list}' --noheader --parsable2 --format=JobIDRaw,State,ExitCode" \
            2>/dev/null || true
    )"
    all_terminal=1
    state_counts=""
    for state_name in PENDING RUNNING COMPLETED FAILED CANCELLED TIMEOUT OUT_OF_MEMORY UNKNOWN; do
        count=0
        for job_id in "${job_ids[@]}"; do
            record="$(printf '%s\n' "${records}" | awk -F'|' -v id="${job_id}" '$1 == id {print; exit}')"
            IFS='|' read -r _ state _ <<<"${record}"
            state="${state:-UNKNOWN}"
            if [[ "${state}" == "${state_name}"* ]]; then
                count=$((count + 1))
            fi
            case "${state}" in
                COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*)
                    ;;
                *)
                    all_terminal=0
                    ;;
            esac
        done
        if (( count > 0 )); then
            state_counts+=" ${state_name}=${count}"
        fi
    done
    printf '[%(%Y-%m-%d %H:%M:%S)T]%s\n' -1 "${state_counts}"
    (( all_terminal == 1 )) && break
    sleep "${poll_seconds}"
done

mkdir -p "${REPO_ROOT}/logs/euler"
ssh "${EULER_LOGIN}" \
    "sacct -X -j '${job_list}' --noheader --parsable2 --format=JobIDRaw,JobName,State,Elapsed,ExitCode,NodeList" \
    >"${status_path}"
"${SCRIPT_DIR}/pull_results.sh"

while IFS='|' read -r job_id _ state _ exit_code _; do
    if [[ "${state}" != COMPLETED* || "${exit_code}" != "0:0" ]]; then
        echo "[ERROR] Job ${job_id} ended state=${state} exit=${exit_code}; reports were pulled." >&2
        exit 1
    fi
done <"${status_path}"

python3 "${REPO_ROOT}/scripts/verify_full_lift_validation.py" \
    "${manifest}" \
    --logs-root "${REPO_ROOT}/logs/euler" \
    --output-dir "${output_dir}"
PYTHONPATH="${REPO_ROOT}" python3 "${REPO_ROOT}/scripts/build_scripted_liftability_labels.py" \
    --attempt-root "${output_dir}/attempts.csv" \
    --output-dir "${output_dir}/liftability_labels"
echo "[INFO] Full lift suite ${suite_id} completed, pulled, and verified: ${output_dir}"
