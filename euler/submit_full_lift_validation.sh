#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUITE_ID="${LIFT_SUITE_ID:-$(date '+%Y%m%d-%H%M%S')}"
BATCH_SIZE="${LIFT_BATCH_SIZE:-96}"
DRY_RUN="${LIFT_DRY_RUN:-0}"
GPU_TYPE="${LIFT_GPU_TYPE:-rtx_4090}"
GPU_MEMORY="${LIFT_GPU_MEMORY:-20G}"
TIME_LIMIT="${LIFT_TIME_LIMIT:-01:00:00}"
TASK="Grasp-Visual-Servo-RGBD-FabricaAll-Direct-v0"
DATASET_INDEX="${REPO_ROOT}/isaac_rl/data/fabrica_all_v1/dataset_index.json"
MANIFEST="${REPO_ROOT}/euler/full_lift_suite_${SUITE_ID}.json"

if [[ ! "${SUITE_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "[ERROR] LIFT_SUITE_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
fi
if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] LIFT_BATCH_SIZE must be a positive integer" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "[ERROR] LIFT_DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ -e "${MANIFEST}" ]]; then
    echo "[ERROR] Suite manifest already exists: ${MANIFEST}" >&2
    echo "[ERROR] Choose another LIFT_SUITE_ID so an existing suite is never overwritten." >&2
    exit 2
fi

mapfile -t dataset_values < <(
    python3 - "${DATASET_INDEX}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["dataset_sha256"])
print(payload["selected_target_count"])
print(payload["shards"]["count"])
PY
)
DATASET_SHA256="${dataset_values[0]}"
EXPECTED_TARGET_COUNT="${dataset_values[1]}"
SHARD_COUNT="${dataset_values[2]}"

mapfile -t work_items < <(
    python3 - "${DATASET_INDEX}" "${BATCH_SIZE}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
batch_size = int(sys.argv[2])
for shard in payload["shards"]["items"]:
    for split in ("train", "validation", "test"):
        count = int(shard["split_counts"][split])
        for offset in range(0, count, batch_size):
            print(f'{shard["shard_index"]}|{split}|{offset}|{min(batch_size, count - offset)}')
PY
)

run_records=()

write_manifest() {
    python3 - "${MANIFEST}" "${SUITE_ID}" "${TASK}" "${DATASET_INDEX}" \
        "${DATASET_SHA256}" "${EXPECTED_TARGET_COUNT}" "${SHARD_COUNT}" "${BATCH_SIZE}" \
        "${GPU_TYPE}" "${GPU_MEMORY}" "${TIME_LIMIT}" "${run_records[@]}" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    suite_id,
    task,
    dataset_index,
    dataset_sha256,
    expected_target_count,
    shard_count,
    batch_size,
    gpu_type,
    gpu_memory,
    time_limit,
    *records,
) = sys.argv[1:]
runs = []
for record in records:
    label, job_id, shard, split, offset, expected = record.split("|")
    runs.append(
        {
            "label": label,
            "job_id": int(job_id),
            "dataset_shard": int(shard),
            "catalog_split": split,
            "target_offset": int(offset),
            "expected_targets": int(expected),
        }
    )
payload = {
    "schema_version": 1,
    "suite_id": suite_id,
    "task": task,
    "dataset_index": dataset_index,
    "dataset_sha256": dataset_sha256,
    "expected_target_count": int(expected_target_count),
    "dataset_shard_count": int(shard_count),
    "batch_size": int(batch_size),
    "gpu_type": gpu_type,
    "gpu_memory": gpu_memory,
    "time_limit": time_limit,
    "protocol": {
        "dynamic_part": True,
        "videos": False,
        "lift_height_m": 0.060,
        "lift_speed_m_s": 0.050,
        "postlift_hold_s": 0.20,
        "minimum_final_lift_m": 0.040,
        "fixture_part_pose_until_close": False,
        "disable_part_gravity_until_close": False,
    },
    "runs": runs,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

echo "[INFO] Full Fabrica lift suite ${SUITE_ID}: ${EXPECTED_TARGET_COUNT} targets in ${#work_items[@]} jobs"
echo "[INFO] First pass uses dynamic parts, normal gravity, a 60 mm lift at 50 mm/s, and no fixtures."
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry run only; no jobs will be submitted."
fi

skip_sync=0
for item in "${work_items[@]}"; do
    IFS='|' read -r shard split offset expected <<<"${item}"
    split_short="${split:0:1}"
    label="lift-${SUITE_ID}-s${shard}${split_short}${offset}"
    command=(
        "${SCRIPT_DIR}/submit.sh" lift
        --job-label "${label}"
        --gpu-type "${GPU_TYPE}"
        --gpu-memory "${GPU_MEMORY}"
        --time-limit "${TIME_LIMIT}"
        --task "${TASK}"
        --dataset-shard "${shard}"
        --catalog-split "${split}"
        --target-offset "${offset}"
        --max-targets "${expected}"
        --lift-height-m 0.060
        --lift-speed-m-s 0.050
        --postlift-hold-s 0.20
        --minimum-final-lift-m 0.040
        --no-lift-videos
    )
    echo "[LIFT-SUITE] shard=${shard} split=${split} offset=${offset} targets=${expected}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'EULER_SKIP_SYNC=%q ' "${skip_sync}"
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        output="$(EULER_SKIP_SYNC="${skip_sync}" "${command[@]}")"
        printf '%s\n' "${output}"
        if [[ ! "${output}" =~ Submitted[[:space:]]batch[[:space:]]job[[:space:]]([0-9]+) ]]; then
            echo "[ERROR] Could not parse Slurm job ID; already submitted jobs remain recorded." >&2
            exit 1
        fi
        job_id="${BASH_REMATCH[1]}"
        run_records+=("${label}|${job_id}|${shard}|${split}|${offset}|${expected}")
        write_manifest
    fi
    skip_sync=1
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] Dry run complete."
else
    echo "[INFO] Submitted ${#run_records[@]} jobs. Manifest: ${MANIFEST}"
    echo "[INFO] Watch, pull, verify exact coverage, and build labels with:"
    echo "${SCRIPT_DIR}/watch_full_lift_validation.sh ${MANIFEST} 60"
fi
