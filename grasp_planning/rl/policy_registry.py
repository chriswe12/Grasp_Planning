"""Resolve named or explicit D405 deployment checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a YAML mapping in '{path}'.")
    return dict(payload)


def resolve_from(path_text: object, *, base: Path) -> Path:
    path = Path(str(path_text)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_policy_registry(path: Path) -> tuple[dict[str, object], Path]:
    registry = load_yaml_mapping(path)
    if int(registry.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported D405 policy registry schema.")
    asset_root = resolve_from(registry.get("asset_root", ""), base=path.parent)
    policies = registry.get("policies")
    if not isinstance(policies, Mapping) or not policies:
        raise ValueError("The D405 policy registry contains no policies.")
    return registry, asset_root


def _validated_assets(
    *,
    policy_name: str,
    checkpoint: Path,
    metadata_path: Path,
    agent_config: Path,
    record: Mapping[str, object],
) -> dict[str, object]:
    for label, path in (
        ("checkpoint", checkpoint),
        ("metadata", metadata_path),
        ("agent_config", agent_config),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} for policy '{policy_name}': {path}")
    metadata = load_yaml_mapping(metadata_path)
    expected_checkpoint_hash = str(metadata.get("checkpoint_sha256", ""))
    actual_checkpoint_hash = sha256(checkpoint)
    if not expected_checkpoint_hash or actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError(
            f"checkpoint hash mismatch for policy '{policy_name}': "
            f"expected={expected_checkpoint_hash or 'missing'} actual={actual_checkpoint_hash}."
        )
    context_mode = str(metadata.get("policy_context_mode", "action"))
    declared_context = str(record.get("policy_context", context_mode))
    if declared_context != context_mode:
        raise ValueError(
            f"Policy '{policy_name}' registry context '{declared_context}' does not match "
            f"its checkpoint sidecar '{context_mode}'."
        )
    return {
        "checkpoint": checkpoint,
        "metadata": metadata_path,
        "agent_config": agent_config,
        "policy_context_mode": context_mode,
        "policy_rate_hz": float(record.get("policy_rate_hz", metadata.get("policy_rate_hz", 30.0))),
        "action_delta_limit": float(
            record.get("action_delta_limit", metadata.get("action_delta_limit", 0.25))
        ),
        "camera_profile": str(metadata.get("camera_profile", "")),
        "gripper_model": str(record.get("gripper_model", metadata.get("gripper_model", "y_gripper"))),
    }


def resolve_policy_assets(
    registry: Mapping[str, object],
    *,
    registry_path: Path,
    asset_root: Path,
    policy_name: str,
) -> dict[str, object]:
    policies = registry["policies"]
    assert isinstance(policies, Mapping)
    if policy_name not in policies:
        available = ", ".join(sorted(str(name) for name in policies))
        raise ValueError(f"Unknown policy '{policy_name}'. Available: {available}.")
    record = policies[policy_name]
    if not isinstance(record, Mapping):
        raise ValueError(f"Policy record '{policy_name}' is malformed.")
    return _validated_assets(
        policy_name=policy_name,
        checkpoint=resolve_from(record.get("checkpoint", ""), base=asset_root),
        metadata_path=resolve_from(record.get("metadata", ""), base=asset_root),
        agent_config=resolve_from(
            record.get("agent_config", registry.get("agent_config", "")),
            base=registry_path.parent,
        ),
        record=record,
    )


def resolve_policy_reference(
    reference: str,
    *,
    registry_path: Path,
) -> tuple[str, dict[str, object]]:
    """Resolve a registry name or checkpoint path with a mandatory sidecar."""

    raw = str(reference).strip()
    checkpoint = Path(raw).expanduser()
    if checkpoint.is_file():
        checkpoint = checkpoint.resolve()
        metadata_candidates = (
            checkpoint.with_name(f"{checkpoint.stem}.deployment.json"),
            checkpoint.with_suffix(".json"),
            checkpoint.with_suffix(".yaml"),
            checkpoint.with_suffix(".yml"),
            checkpoint.parent / f"{checkpoint.stem}_metadata.json",
        )
        metadata = next((path for path in metadata_candidates if path.is_file()), None)
        if metadata is None:
            raise FileNotFoundError(
                "Explicit policy checkpoints require a hash-bearing metadata sidecar next to "
                f"the checkpoint; checked: {', '.join(str(path) for path in metadata_candidates)}"
            )
        metadata_payload = load_yaml_mapping(metadata)
        agent_raw = metadata_payload.get("agent_config_path", "")
        agent_config = (
            resolve_from(agent_raw, base=metadata.parent)
            if str(agent_raw).strip()
            else registry_path.parent / "rl_games_multipart_ppo_deployment.yaml"
        )
        policy_name = checkpoint.stem
        return policy_name, _validated_assets(
            policy_name=policy_name,
            checkpoint=checkpoint,
            metadata_path=metadata,
            agent_config=agent_config.resolve(),
            record=metadata_payload,
        )
    registry, asset_root = load_policy_registry(registry_path)
    policy_name = raw.lower()
    return policy_name, resolve_policy_assets(
        registry,
        registry_path=registry_path,
        asset_root=asset_root,
        policy_name=policy_name,
    )


__all__ = [
    "load_policy_registry",
    "load_yaml_mapping",
    "resolve_from",
    "resolve_policy_assets",
    "resolve_policy_reference",
    "sha256",
]
