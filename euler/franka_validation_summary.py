"""Compare checkpoints only across complete, matching validation conditions."""

import json
from collections import defaultdict


def summarize_validation(results):
    records = []
    groups = defaultdict(list)
    for path in sorted(results.rglob("evaluation.json")):
        value = json.loads(path.read_text())
        if value.get("catalog_split") != "validation" or not value.get("coverage_complete"):
            continue
        record = dict(
            report=str(path),
            checkpoint=value["checkpoint"],
            macro_part_success=value["macro_part_success"],
            pose_evaluation_progress=value.get("pose_evaluation_progress"),
        )
        records.append(record)
        groups[record["checkpoint"]].append(record)
    pose_mode = any(r["pose_evaluation_progress"] is not None for r in records)
    expected = {0.0, 0.5, 0.94} if pose_mode else {None}
    candidates = []
    for checkpoint, group in groups.items():
        conditions = {r["pose_evaluation_progress"] for r in group}
        if conditions != expected or len(group) != len(expected):
            continue
        candidates.append(
            dict(
                checkpoint=checkpoint,
                macro_part_success=sum(r["macro_part_success"] for r in group) / len(group),
                aggregation="equal mean across complete validation conditions",
                conditions=group,
            )
        )
    return dict(
        validation=records,
        validation_by_checkpoint=candidates,
        best_validation=max(candidates, key=lambda x: x["macro_part_success"]) if candidates else None,
    )
