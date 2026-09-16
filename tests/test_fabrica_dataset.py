from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_planning.rl.fabrica_dataset import (
    assign_parts_to_shards,
    canonical_json_sha256,
    configure_fabrica_env_cfg,
    load_fabrica_shard_config,
    resolve_fabrica_shard_layout,
    select_fabrica_shard,
    sha256_file,
    subset_target_arrays,
)


def test_part_assignment_is_deterministic_balanced_and_disjoint() -> None:
    part_ids = np.asarray(["a"] * 8 + ["b"] * 7 + ["c"] * 6 + ["d"] * 5)
    split_ids = np.asarray(
        ["train"] * 6
        + ["test"] * 2
        + ["train"] * 5
        + ["test"] * 2
        + ["train"] * 4
        + ["validation"] * 2
        + ["train"] * 3
        + ["test"] * 2
    )
    first = assign_parts_to_shards(part_ids, split_ids, shard_count=2)
    second = assign_parts_to_shards(part_ids, split_ids, shard_count=2)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == {"a", "b", "c", "d"}
    train_loads = [sum(part_ids[split_ids == "train"].tolist().count(part) for part in shard) for shard in first]
    assert abs(train_loads[0] - train_loads[1]) <= 2


def test_subset_target_arrays_remaps_part_and_orientation_indices() -> None:
    arrays = {
        "target_ids": np.asarray(["t0", "t1", "t2", "t3"]),
        "part_names": np.asarray(["a", "b", "c"]),
        "part_usd_paths": np.asarray(["a.usd", "b.usd", "c.usd"]),
        "part_ids": np.asarray(["a", "b", "c", "b"]),
        "part_indices": np.asarray([0, 1, 2, 1]),
        "orientation_names": np.asarray(["o0", "o1", "o2"]),
        "orientation_ids": np.asarray(["o0", "o1", "o2", "o1"]),
        "orientation_indices": np.asarray([0, 1, 2, 1]),
        "target_values": np.arange(8).reshape(4, 2),
        "waypoint_values": np.arange(3),
    }
    result = subset_target_arrays(
        arrays,
        np.asarray([1, 3]),
        part_names=("b",),
        part_usd_paths=("assets/b.usd",),
    )
    assert result["target_ids"].tolist() == ["t1", "t3"]
    assert result["part_names"].tolist() == ["b"]
    assert result["part_indices"].tolist() == [0, 0]
    assert result["orientation_names"].tolist() == ["o1"]
    assert result["orientation_indices"].tolist() == [0, 0]
    assert result["target_values"].tolist() == [[2, 3], [6, 7]]
    assert result["waypoint_values"].tolist() == [0, 1, 2]


def test_rank_selection_requires_one_rank_per_shard() -> None:
    assert select_fabrica_shard(rank=3, world_size=4, shard_count=4) == 3
    assert select_fabrica_shard(rank=0, world_size=1, shard_count=4) == 0
    assert select_fabrica_shard(rank=0, world_size=1, shard_count=4, explicit_shard=2) == 2
    with pytest.raises(ValueError, match="exactly 4 ranks"):
        select_fabrica_shard(rank=0, world_size=3, shard_count=4)
    with pytest.raises(ValueError, match="single-rank"):
        select_fabrica_shard(rank=0, world_size=4, shard_count=4, explicit_shard=2)


def test_layout_resolution_selects_rank_count_and_preserves_default() -> None:
    four = {"count": 4, "items": [{"shard_index": index} for index in range(4)]}
    six = {"count": 6, "items": [{"shard_index": index} for index in range(6)]}
    index = {
        "shards": four,
        "shard_layouts": {
            "default_count": 4,
            "items": {"4": four, "6": six},
        },
    }
    assert resolve_fabrica_shard_layout(index, world_size=1) is four
    assert resolve_fabrica_shard_layout(index, world_size=4) is four
    assert resolve_fabrica_shard_layout(index, world_size=6) is six
    with pytest.raises(ValueError, match="available rank counts: 4, 6"):
        resolve_fabrica_shard_layout(index, world_size=5)


def test_six_rank_layout_loads_the_matching_rank(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data/layout_06/shard_05/part_inventory.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "target_count": 7,
        "split_counts": {"train": 5, "validation": 1, "test": 1},
        "part_names": ["assembly__part_5"],
        "part_usd_paths": ["assets/part_5.usd"],
        "part_xy_rotation_radii_m": [0.05],
        "artifacts": {
            "goal_catalog": {"path": "data/layout_06/shard_05/goal_catalog.npz"},
            "paths": {"path": "data/layout_06/shard_05/paths.npz"},
            "rotation_resets": {"path": "data/layout_06/shard_05/rotation_resets.npz"},
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    four = {"count": 4, "items": [{"shard_index": index} for index in range(4)]}
    six_items = [{"shard_index": index} for index in range(6)]
    six_items[5] = {
        "shard_index": 5,
        "manifest": "data/layout_06/shard_05/part_inventory.json",
        "manifest_sha256": sha256_file(manifest_path),
    }
    index = {
        "dataset_name": "fabrica_all_v1",
        "training_ready": True,
        "shards": four,
        "shard_layouts": {
            "default_count": 4,
            "items": {"4": four, "6": {"count": 6, "items": six_items}},
        },
    }
    index["dataset_sha256"] = canonical_json_sha256(index)
    index_path = tmp_path / "dataset_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    shard = load_fabrica_shard_config(
        rank=5,
        world_size=6,
        index_path=index_path,
        repo_root=tmp_path,
    )
    assert shard.shard_index == 5
    assert shard.shard_count == 6
    assert shard.target_count == 7
    assert shard.part_names == ("assembly__part_5",)


def test_portable_index_configures_environment(tmp_path: Path) -> None:
    manifest_path = tmp_path / "data/shard_00/part_inventory.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "target_count": 12,
        "split_counts": {"train": 10, "validation": 1, "test": 1},
        "part_names": ["assembly__part_0"],
        "part_usd_paths": ["assets/part.usd"],
        "part_xy_rotation_radii_m": [0.04],
        "artifacts": {
            "goal_catalog": {"path": "data/shard_00/goal_catalog.npz"},
            "paths": {"path": "data/shard_00/paths.npz"},
            "rotation_resets": {"path": "data/shard_00/rotation_resets.npz"},
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index = {
        "dataset_name": "fabrica_all_v1",
        "training_ready": True,
        "shards": {
            "count": 1,
            "items": [
                {
                    "shard_index": 0,
                    "manifest": "data/shard_00/part_inventory.json",
                    "manifest_sha256": sha256_file(manifest_path),
                }
            ],
        },
    }
    index["dataset_sha256"] = canonical_json_sha256(index)
    index_path = tmp_path / "dataset_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    shard = load_fabrica_shard_config(index_path=index_path, repo_root=tmp_path)
    assert shard.target_count == 12
    assert shard.part_names == ("assembly__part_0",)
    assert shard.part_usd_paths == ((tmp_path / "assets/part.usd").resolve(),)

    cfg = SimpleNamespace()
    configure_fabrica_env_cfg(cfg, index_path=index_path, repo_root=tmp_path)
    assert cfg.dataset_shard_index == 0
    assert cfg.goal_catalog_data_path == str((tmp_path / "data/shard_00/goal_catalog.npz").resolve())
    assert cfg.part_xy_rotation_radii_m == (0.04,)


def test_merged_index_configures_complete_catalog(tmp_path: Path) -> None:
    merged = tmp_path / "data/merged"
    merged.mkdir(parents=True)
    manifest_path = merged / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    index = {
        "dataset_name": "fabrica_all_v1",
        "training_ready": True,
        "validated_target_count": 12,
        "merged_manifest_sha256": sha256_file(manifest_path),
        "split_scheme_metadata": {"primary": {"target_counts": {"train": 10, "validation": 1, "test": 1}}},
        "artifacts": {
            "merged_manifest": "data/merged/manifest.json",
            "merged_goal_catalog": "data/merged/goal_catalog.npz",
            "merged_paths": "data/merged/paths.npz",
            "merged_rotation_resets": "data/merged/rotation_resets.npz",
        },
        "parts": [
            {
                "part_name": "assembly__part_0",
                "usd_path": "assets/part.usd",
                "xy_rotation_radius_m": 0.04,
            }
        ],
        "shards": {"count": 0, "items": []},
    }
    index["dataset_sha256"] = canonical_json_sha256(index)
    index_path = tmp_path / "dataset_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    cfg = SimpleNamespace()
    merged_config = configure_fabrica_env_cfg(
        cfg,
        merged=True,
        index_path=index_path,
        repo_root=tmp_path,
    )
    assert merged_config.shard_count == 1
    assert merged_config.target_count == 12
    assert merged_config.split_counts == {"train": 10, "validation": 1, "test": 1}
    assert cfg.goal_catalog_data_path == str((merged / "goal_catalog.npz").resolve())
