"""Portable, rank-aware contracts for the Fabrica-all visual-servo dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FABRICA_TASK_ID = "Grasp-Visual-Servo-RGBD-FabricaAll-Direct-v0"
FABRICA_PLAY_TASK_ID = "Grasp-Visual-Servo-RGBD-FabricaAll-Direct-Play-v0"
FABRICA_DATASET_NAME = "fabrica_all_v1"
FABRICA_DATASET_SCHEMA_VERSION = 3
FABRICA_SHARD_SCHEMA_VERSION = 1
FABRICA_SHARD_COUNT = 4
FABRICA_SUPPORTED_SHARD_COUNTS = (4, 6)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_INDEX = REPO_ROOT / "isaac_rl/data/fabrica_all_v1/dataset_index.json"


@dataclass(frozen=True)
class FabricaShardConfig:
    """Resolved files and scene assets for one target/part shard."""

    dataset_name: str
    dataset_sha256: str
    index_path: Path
    shard_index: int
    shard_count: int
    shard_manifest_path: Path
    shard_manifest_sha256: str
    goal_catalog_path: Path
    paths_path: Path
    rotation_resets_path: Path
    part_names: tuple[str, ...]
    part_usd_paths: tuple[Path, ...]
    part_xy_rotation_radii_m: tuple[float, ...]
    target_count: int
    split_counts: Mapping[str, int]

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_sha256": self.dataset_sha256,
            "index_path": str(self.index_path),
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "shard_manifest": str(self.shard_manifest_path),
            "shard_manifest_sha256": self.shard_manifest_sha256,
            "target_count": self.target_count,
            "split_counts": dict(self.split_counts),
            "part_count": len(self.part_names),
            "part_names": list(self.part_names),
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("dataset_sha256", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def repo_relative(path: str | Path, *, repo_root: str | Path = REPO_ROOT) -> str:
    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Dataset artifact must live below the repository root: {resolved}") from exc


def resolve_repo_path(path: str | Path, *, repo_root: str | Path = REPO_ROOT) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(repo_root).expanduser().resolve() / candidate).resolve()


def assign_parts_to_shards(
    part_ids: Iterable[str],
    split_ids: Iterable[str],
    *,
    shard_count: int = FABRICA_SHARD_COUNT,
) -> tuple[tuple[str, ...], ...]:
    """Greedily balance train targets, then total targets, without splitting a part."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    counts: dict[str, dict[str, int]] = {}
    for part_id, split_id in zip(part_ids, split_ids, strict=True):
        by_split = counts.setdefault(str(part_id), {"train": 0, "total": 0})
        by_split["total"] += 1
        if str(split_id) == "train":
            by_split["train"] += 1
    if len(counts) < shard_count:
        raise ValueError(f"Cannot distribute {len(counts)} parts over {shard_count} non-empty shards.")

    assignments: list[list[str]] = [[] for _ in range(shard_count)]
    train_load = [0] * shard_count
    total_load = [0] * shard_count
    ordered = sorted(
        counts,
        key=lambda name: (-counts[name]["train"], -counts[name]["total"], name),
    )
    for part_name in ordered:
        shard_index = min(
            range(shard_count),
            key=lambda index: (train_load[index], total_load[index], len(assignments[index]), index),
        )
        assignments[shard_index].append(part_name)
        train_load[shard_index] += counts[part_name]["train"]
        total_load[shard_index] += counts[part_name]["total"]
    return tuple(tuple(sorted(names)) for names in assignments)


def subset_target_arrays(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    part_names: tuple[str, ...] | None = None,
    part_usd_paths: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Subset every target-leading array and remap part/orientation indices."""

    target_count = int(np.asarray(arrays["target_ids"]).shape[0])
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("A shard must contain at least one target.")
    if np.any(selected < 0) or np.any(selected >= target_count):
        raise ValueError("Shard target index is outside the source arrays.")
    result = {
        name: (
            np.asarray(value)[selected].copy()
            if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == target_count
            else np.asarray(value).copy()
        )
        for name, value in arrays.items()
    }

    if "part_ids" in result:
        names = part_names or tuple(sorted(set(result["part_ids"].astype(str).tolist())))
        name_to_index = {name: index for index, name in enumerate(names)}
        result["part_names"] = np.asarray(names)
        if part_usd_paths is not None:
            if len(part_usd_paths) != len(names):
                raise ValueError("part_usd_paths must align with part_names")
            result["part_usd_paths"] = np.asarray(part_usd_paths)
        result["part_indices"] = np.asarray(
            [name_to_index[name] for name in result["part_ids"].astype(str)], dtype=np.int64
        )

    if "orientation_ids" in result:
        orientation_names = tuple(sorted(set(result["orientation_ids"].astype(str).tolist())))
        orientation_to_index = {name: index for index, name in enumerate(orientation_names)}
        result["orientation_names"] = np.asarray(orientation_names)
        result["orientation_indices"] = np.asarray(
            [orientation_to_index[name] for name in result["orientation_ids"].astype(str)],
            dtype=np.int64,
        )
    return result


def select_fabrica_shard(
    *,
    rank: int,
    world_size: int,
    shard_count: int,
    explicit_shard: int | None = None,
) -> int:
    if shard_count <= 0:
        raise ValueError("Dataset index contains no shards.")
    if explicit_shard is not None:
        if not 0 <= explicit_shard < shard_count:
            raise ValueError(f"dataset shard must be in [0, {shard_count - 1}], got {explicit_shard}")
        if world_size > 1:
            raise ValueError("--dataset-shard is only valid for a single-rank probe or evaluation.")
        return explicit_shard
    if world_size == 1:
        return 0
    if world_size != shard_count:
        raise ValueError(
            f"Fabrica-all distributed training requires exactly {shard_count} ranks so every "
            f"part shard participates once; received WORLD_SIZE={world_size}."
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside WORLD_SIZE={world_size}")
    return rank


def resolve_fabrica_shard_layout(
    index: Mapping[str, Any],
    *,
    world_size: int,
) -> Mapping[str, Any]:
    """Select a complete, part-disjoint shard layout for this rank count.

    Schema-v2 indices exposed one layout through ``shards``. Schema-v3 keeps
    that field as the default/single-rank compatibility layout and adds
    ``shard_layouts.items`` keyed by distributed world size.
    """

    if world_size < 1:
        raise ValueError("world_size must be positive")
    legacy_layout = index.get("shards", {})
    if not isinstance(legacy_layout, Mapping):
        raise ValueError("Fabrica dataset index has an invalid shards record.")

    layouts: dict[int, Mapping[str, Any]] = {}
    legacy_count = int(legacy_layout.get("count", 0))
    if legacy_count > 0:
        layouts[legacy_count] = legacy_layout

    layout_group = index.get("shard_layouts", {})
    if layout_group:
        if not isinstance(layout_group, Mapping):
            raise ValueError("Fabrica dataset shard_layouts must be a mapping.")
        layout_items = layout_group.get("items", {})
        if not isinstance(layout_items, Mapping):
            raise ValueError("Fabrica dataset shard_layouts.items must be a mapping.")
        for count_text, layout in layout_items.items():
            if not isinstance(layout, Mapping):
                raise ValueError(f"Fabrica shard layout {count_text!r} is not a mapping.")
            count = int(count_text)
            if count < 1 or int(layout.get("count", -1)) != count:
                raise ValueError(f"Fabrica shard layout {count_text!r} has a mismatched count.")
            layouts[count] = layout

    if world_size == 1:
        default_count = int(
            layout_group.get("default_count", legacy_count) if isinstance(layout_group, Mapping) else legacy_count
        )
        desired_count = default_count
    else:
        desired_count = world_size
    if desired_count not in layouts:
        supported = ", ".join(str(count) for count in sorted(layouts)) or "none"
        raise ValueError(
            f"Fabrica-all distributed training has no {desired_count}-rank shard layout; "
            f"available rank counts: {supported}."
        )
    return layouts[desired_count]


def load_fabrica_shard_config(
    *,
    rank: int = 0,
    world_size: int = 1,
    explicit_shard: int | None = None,
    merged: bool = False,
    index_path: str | Path = DEFAULT_DATASET_INDEX,
    repo_root: str | Path = REPO_ROOT,
) -> FabricaShardConfig:
    resolved_index = resolve_repo_path(index_path, repo_root=repo_root)
    index = json.loads(resolved_index.read_text(encoding="utf-8"))
    if index.get("dataset_name") != FABRICA_DATASET_NAME:
        raise ValueError(f"Unexpected Fabrica dataset name: {index.get('dataset_name')}")
    if not bool(index.get("training_ready")):
        raise ValueError(f"Fabrica dataset is not training-ready: {resolved_index}")
    recorded_hash = str(index.get("dataset_sha256", ""))
    actual_hash = canonical_json_sha256(index)
    if not recorded_hash or recorded_hash != actual_hash:
        raise ValueError("Fabrica dataset index hash is missing or stale; rebuild/verify the shards.")

    if merged:
        artifacts = dict(index.get("artifacts", {}))
        required_artifacts = (
            "merged_manifest",
            "merged_goal_catalog",
            "merged_paths",
            "merged_rotation_resets",
        )
        missing = [name for name in required_artifacts if not artifacts.get(name)]
        if missing:
            raise ValueError(f"Fabrica dataset index is missing merged artifacts: {', '.join(missing)}")
        parts = list(index.get("parts", ()))
        if not parts:
            raise ValueError("Fabrica dataset index contains no merged part metadata.")
        manifest_path = resolve_repo_path(artifacts["merged_manifest"], repo_root=repo_root)
        manifest_hash = sha256_file(manifest_path)
        expected_manifest_hash = str(index.get("merged_manifest_sha256", ""))
        if expected_manifest_hash and manifest_hash != expected_manifest_hash:
            raise ValueError(f"Fabrica merged manifest checksum mismatch: {manifest_path}")
        split_counts = dict(index.get("split_scheme_metadata", {}).get("primary", {}).get("target_counts", {}))
        return FabricaShardConfig(
            dataset_name=str(index["dataset_name"]),
            dataset_sha256=recorded_hash,
            index_path=resolved_index,
            shard_index=0,
            shard_count=1,
            shard_manifest_path=manifest_path,
            shard_manifest_sha256=manifest_hash,
            goal_catalog_path=resolve_repo_path(artifacts["merged_goal_catalog"], repo_root=repo_root),
            paths_path=resolve_repo_path(artifacts["merged_paths"], repo_root=repo_root),
            rotation_resets_path=resolve_repo_path(artifacts["merged_rotation_resets"], repo_root=repo_root),
            part_names=tuple(str(part["part_name"]) for part in parts),
            part_usd_paths=tuple(resolve_repo_path(part["usd_path"], repo_root=repo_root) for part in parts),
            part_xy_rotation_radii_m=tuple(float(part["xy_rotation_radius_m"]) for part in parts),
            target_count=int(index["validated_target_count"]),
            split_counts={str(key): int(value) for key, value in split_counts.items()},
        )

    shard_layout = resolve_fabrica_shard_layout(index, world_size=world_size)
    shard_records = list(shard_layout.get("items", ()))
    shard_count = int(shard_layout.get("count", len(shard_records)))
    if len(shard_records) != shard_count:
        raise ValueError("Fabrica dataset index shard count does not match its shard records.")
    shard_index = select_fabrica_shard(
        rank=rank,
        world_size=world_size,
        shard_count=shard_count,
        explicit_shard=explicit_shard,
    )
    record = shard_records[shard_index]
    if int(record.get("shard_index", -1)) != shard_index:
        raise ValueError("Fabrica shard records are not stored in index order.")
    manifest_path = resolve_repo_path(record["manifest"], repo_root=repo_root)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != record.get("manifest_sha256"):
        raise ValueError(f"Fabrica shard manifest checksum mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    part_names = tuple(str(value) for value in manifest["part_names"])
    part_usd_paths = tuple(resolve_repo_path(value, repo_root=repo_root) for value in manifest["part_usd_paths"])
    radii = tuple(float(value) for value in manifest["part_xy_rotation_radii_m"])
    if not (len(part_names) == len(part_usd_paths) == len(radii)):
        raise ValueError("Fabrica shard part names, USDs, and radii are not aligned.")
    return FabricaShardConfig(
        dataset_name=str(index["dataset_name"]),
        dataset_sha256=recorded_hash,
        index_path=resolved_index,
        shard_index=shard_index,
        shard_count=shard_count,
        shard_manifest_path=manifest_path,
        shard_manifest_sha256=manifest_hash,
        goal_catalog_path=resolve_repo_path(manifest["artifacts"]["goal_catalog"]["path"], repo_root=repo_root),
        paths_path=resolve_repo_path(manifest["artifacts"]["paths"]["path"], repo_root=repo_root),
        rotation_resets_path=resolve_repo_path(manifest["artifacts"]["rotation_resets"]["path"], repo_root=repo_root),
        part_names=part_names,
        part_usd_paths=part_usd_paths,
        part_xy_rotation_radii_m=radii,
        target_count=int(manifest["target_count"]),
        split_counts={str(key): int(value) for key, value in manifest["split_counts"].items()},
    )


def configure_fabrica_env_cfg(
    env_cfg: Any,
    *,
    rank: int = 0,
    world_size: int = 1,
    explicit_shard: int | None = None,
    merged: bool = False,
    index_path: str | Path = DEFAULT_DATASET_INDEX,
    repo_root: str | Path = REPO_ROOT,
) -> FabricaShardConfig:
    shard = load_fabrica_shard_config(
        rank=rank,
        world_size=world_size,
        explicit_shard=explicit_shard,
        merged=merged,
        index_path=index_path,
        repo_root=repo_root,
    )
    env_cfg.goal_catalog_data_path = str(shard.goal_catalog_path)
    env_cfg.rotation_reset_data_path = str(shard.rotation_resets_path)
    env_cfg.part_names = shard.part_names
    env_cfg.part_usd_paths = tuple(str(path) for path in shard.part_usd_paths)
    env_cfg.part_xy_rotation_radii_m = shard.part_xy_rotation_radii_m
    env_cfg.dataset_name = shard.dataset_name
    env_cfg.dataset_sha256 = shard.dataset_sha256
    env_cfg.dataset_shard_index = shard.shard_index
    env_cfg.dataset_shard_count = shard.shard_count
    return shard
