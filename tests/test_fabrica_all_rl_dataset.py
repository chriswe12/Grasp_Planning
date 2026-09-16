from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "isaac_rl/scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assembly_manifest = _load_script("build_assembly_multigrasp_manifest.py")
merge_manifest = _load_script("merge_fabrica_multigrasp_manifests.py")
fabrica_manifest = _load_script("build_fabrica_manifest_from_benchmark.py")


def test_global_ids_namespace_repeated_local_ids_across_assemblies() -> None:
    assert (
        assembly_manifest._part_key("beam", "0")
        == "beam__part_0"
    )
    assert assembly_manifest._prefixed_orientation(
        "0", "orientation_003", assembly_name="beam"
    ) == "beam__part_0__orientation_003"
    assert assembly_manifest._prefixed_target(
        "0", "orientation_003", "g0147", assembly_name="beam"
    ) == "beam__part_0__orientation_003__g0147"
    assert assembly_manifest._prefixed_target(
        "0", "orientation_003", "g0147", assembly_name="car"
    ) != assembly_manifest._prefixed_target(
        "0", "orientation_003", "g0147", assembly_name="beam"
    )


def _write_manifest(path: Path, assembly: str) -> None:
    part_key = f"{assembly}__part_0"
    orientation_id = f"{part_key}__orientation_000"
    targets = []
    for index, split in enumerate(("train", "validation", "test")):
        grasp_id = f"g{index:04d}"
        targets.append(
            {
                "target_id": f"{orientation_id}__{grasp_id}",
                "orientation_id": orientation_id,
                "local_orientation_id": "orientation_000",
                "grasp_id": grasp_id,
                "assembly_name": assembly,
                "local_part_id": "0",
                "part_key": part_key,
                "part_id": part_key,
                "part_index": 0,
                "orientation_selection_rank": index,
                "split": split,
            }
        )
    payload = {
        "schema_version": 5,
        "assembly_name": assembly,
        "configured_part_count": 1,
        "parts": [
            {
                "assembly_name": assembly,
                "local_part_id": "0",
                "part_key": part_key,
                "part_id": part_key,
                "part_index": 0,
            }
        ],
        "orientations": [
            {
                "orientation_id": orientation_id,
                "local_orientation_id": "orientation_000",
                "assembly_name": assembly,
                "local_part_id": "0",
                "part_key": part_key,
                "part_id": part_key,
                "part_index": 0,
            }
        ],
        "targets": targets,
        "alternates": [],
        "exclusions": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_is_deterministic_unique_and_isolates_holdouts(tmp_path: Path) -> None:
    beam = tmp_path / "beam.json"
    car = tmp_path / "car.json"
    _write_manifest(beam, "beam")
    _write_manifest(car, "car")

    first = merge_manifest.merge_manifests(
        [car, beam],
        split_seed=7,
        held_out_part_fraction=0.5,
        held_out_assemblies={"car"},
    )
    second = merge_manifest.merge_manifests(
        [beam, car],
        split_seed=7,
        held_out_part_fraction=0.5,
        held_out_assemblies={"car"},
    )

    assert first == second
    assert [part["part_key"] for part in first["parts"]] == [
        "beam__part_0",
        "car__part_0",
    ]
    target_ids = [target["target_id"] for target in first["targets"]]
    assert len(target_ids) == len(set(target_ids))
    for target in first["targets"]:
        expected = "test" if target["assembly_name"] == "car" else "train"
        assert target["assembly_holdout_split"] == expected
    held_out_parts = set(
        first["split_scheme_metadata"]["part_holdout"]["held_out_part_keys"]
    )
    assert len(held_out_parts) == 1
    assert all(
        target["part_holdout_split"]
        == ("test" if target["part_key"] in held_out_parts else "train")
        for target in first["targets"]
    )


def test_merge_rejects_non_namespaced_legacy_manifest(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"schema_version": 4, "assembly_name": "plumbers_block"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="globally namespaced"):
        merge_manifest.merge_manifests(
            [path],
            split_seed=1,
            held_out_part_fraction=0.0,
            held_out_assemblies=set(),
        )


def test_fabrica_config_respects_pdz_closed_gap() -> None:
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs/fabrica_all_v1.yaml").read_text(encoding="utf-8")
    )
    selection = config["selection"]
    assert selection["min_training_jaw_width_m"] >= (
        fabrica_manifest.PDZ_GRIPPER_CLOSED_WIDTH_M
    )
    assert selection["min_training_jaw_width_m"] <= selection[
        "max_training_jaw_width_m"
    ]
