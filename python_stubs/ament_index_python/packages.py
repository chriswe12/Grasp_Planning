from __future__ import annotations

import os
from pathlib import Path


class PackageNotFoundError(LookupError):
    pass


def _candidate_prefixes() -> list[Path]:
    prefixes: list[Path] = []
    seen: set[Path] = set()
    for env_var in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"):
        raw_value = os.environ.get(env_var, "")
        for entry in raw_value.split(os.pathsep):
            if not entry:
                continue
            prefix = Path(entry).resolve()
            if prefix not in seen:
                prefixes.append(prefix)
                seen.add(prefix)
    return prefixes


def get_package_prefix(package_name: str) -> str:
    for prefix in _candidate_prefixes():
        marker = prefix / "share" / "ament_index" / "resource_index" / "packages" / package_name
        direct_share = prefix / "share" / package_name
        if marker.exists() or direct_share.exists():
            return str(prefix)
    raise PackageNotFoundError(f"Package '{package_name}' was not found in the configured ament prefixes.")


def get_package_share_directory(package_name: str) -> str:
    prefix = Path(get_package_prefix(package_name))
    share_dir = prefix / "share" / package_name
    if share_dir.exists():
        return str(share_dir)
    raise PackageNotFoundError(f"Package '{package_name}' does not have a share directory under '{prefix}'.")
