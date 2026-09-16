#!/usr/bin/env python3
"""Mirror the exact Isaac Panda USD and relative dependencies for offline jobs.

Run with Isaac's USD Python libraries, without starting SimulationApp.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

from pxr import Sdf

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/"
ENTRY = "panda_instanceable.usd"
DEST = ROOT / "assets/usd/franka_panda_offline"


def main():
    pending = [ENTRY]
    files = {}
    builtins = set()
    while pending:
        relative = pending.pop()
        if relative in files:
            continue
        url = urljoin(BASE, relative)
        if not url.startswith(BASE):
            raise ValueError(f"Asset escaped the Panda directory: {url}")
        path = DEST / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Re-fetch every file so an old local copy cannot masquerade as source.
        with urlopen(url, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
        files[relative] = hashlib.sha256(data).hexdigest()
        print(f"[PANDA ASSET] {relative}: {len(data)} bytes", flush=True)
        if path.suffix in (".usd", ".usda", ".usdc"):
            layer = Sdf.Layer.FindOrOpen(str(path))
            if layer is None:
                raise ValueError(f"Cannot read USD: {path}")
            for dependency in re.findall(r"@([^@]+)@", layer.ExportToString()):
                if dependency.endswith(".mdl") and "/" not in dependency:
                    builtins.add(dependency)
                    continue
                resolved = urljoin(url, dependency)
                if not resolved.startswith(BASE):
                    raise ValueError(f"Nonrelative dependency needs explicit packaging: {resolved}")
                pending.append(resolved.removeprefix(BASE))
    (DEST / "manifest.json").write_text(
        json.dumps(
            dict(source_url=BASE + ENTRY, entry=ENTRY, files=files, builtin_material_modules=sorted(builtins)), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
