"""Extract the small dual-planning regression fixture from a recorded run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Recorded artifacts/dual_grasp_planning/plumbers_block directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "dual_grasp_planning/plumbers_block")
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}

    def read(name):
        raw = (source / name).read_bytes()
        provenance[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def portable(obj):
        if isinstance(obj, list):
            return [portable(v) for v in obj]
        if isinstance(obj, dict):
            return {k: portable(v) for k, v in obj.items()}
        if isinstance(obj, str) and obj.startswith(str(source.parents[2]) + "/"):
            return obj.removeprefix(str(source.parents[2]) + "/")
        return obj

    def write(name, data):
        (out / name).write_text(json.dumps(portable(data), indent=2) + "\n")
        print(name, (out / name).stat().st_size)

    seq = read("assembly_sequence.json")
    write("assembly_sequence.json", seq)
    holder = read("holder_base_candidates.json")
    states = read("holder_state_feasibility.json")
    holder_ids = set()
    for step in seq["steps"]:
        sid = step["step_id"]
        name = f"dual_grasp_pairs_{sid}.json"
        if not (source / name).is_file():
            continue
        pairs = read(name)
        bundle = read(f"inserter_candidates_{sid}.json")
        valid = {c["grasp_id"] for c in bundle["candidates"]}
        es = [e for e in pairs["evaluations"] if e["status"] == "accepted" and e["inserter_grasp_id"] in valid]
        by_id = {e["pair_id"]: e for e in es}
        chosen = []
        # Preserve the producer's retained order and a sample of genuine nonretained fallbacks.
        for pid in pairs["retained_pair_ids"]:
            if pid in by_id:
                chosen.append(by_id[pid])
        chosen = chosen[:24]
        retained = set(pairs["retained_pair_ids"])
        execution_ids = set(pairs["retained_execution_candidate_ids"])
        chosen += [
            e
            for e in es
            if e["pair_id"] in retained
            and any(
                t != "tr_identity__part_identity" and e["pair_id"] + "__" + t not in execution_ids
                for t in e.get("details", {}).get("compatible_transition_ids", [])
            )
        ][:8]
        referenced = {e["inserter_grasp_id"] for e in chosen}
        chosen += [e for e in es if e["pair_id"] not in retained and e["inserter_grasp_id"] in referenced][:8]
        chosen = list({e["pair_id"]: e for e in chosen}.values())
        ids = {e["pair_id"] for e in chosen}
        holder_ids.update(e["holder_grasp_id"] for e in chosen)
        pair = {
            k: pairs[k]
            for k in [
                "schema_version",
                "kind",
                "step_id",
                "step_index",
                "incoming_part_id",
                "assembled_part_ids_before",
                "motion",
                "transition_symmetry",
                "candidate_sources",
            ]
        }
        pair["evaluations"] = chosen
        pair["retained_pair_ids"] = [x for x in pairs["retained_pair_ids"] if x in ids]
        pair["retained_execution_candidate_ids"] = [
            x for x in pairs["retained_execution_candidate_ids"] if any(x.startswith(pid + "__") for pid in ids)
        ]
        write(name, pair)
        # Keep canonical object-frame candidates and contract metadata, omitting score explanation tables.
        referenced = {e["inserter_grasp_id"] for e in chosen}
        # Include additional raw proposals so pickup rejection diagnostics retain coverage.
        referenced.update(c["grasp_id"] for c in bundle["candidates"][:20])
        if sid == "step_001_part_0":
            referenced.add("i0_1808")
        bundle["candidates"] = [c for c in bundle["candidates"] if c["grasp_id"] in referenced]
        for c in bundle["candidates"]:
            c.pop("score_components", None)
        write(f"inserter_candidates_{sid}.json", bundle)
        print(sid, "pairs", len(chosen), "retained", len(pair["retained_pair_ids"]), "candidate IDs", len(valid))
    holder["candidates"] = [c for c in holder["candidates"] if c["grasp_id"] in holder_ids]
    write("holder_base_candidates.json", holder)
    write(
        "holder_state_feasibility.json",
        {
            k: v
            for k, v in states.items()
            if k in ["schema_version", "kind", "source_frame_pose_assembly", "source_holder_cache_key"]
        }
        | {"candidates": {k: v for k, v in states["candidates"].items() if k in holder_ids}},
    )
    write("source_sha256.json", provenance)


if __name__ == "__main__":
    main()
