#!/usr/bin/env python3
"""Opt a pencil-lab catalog into live appearance variation with canonical saved goals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grasp_planning.rl.franka_appearance import load_profile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, default=ROOT/'isaac_rl/data/franka_fabrica_pencil/catalog.npz')
p.add_argument('--output', type=Path, default=ROOT/'isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz')
p.add_argument('--profile', type=Path, default=ROOT/'configs/franka_pencil_randomization.json')
a = p.parse_args()
if a.source.resolve() == a.output.resolve(): p.error('Preserve the canonical source catalog')
with np.load(a.source, allow_pickle=False) as f:
    data = {k: f[k].copy() for k in f.files}
contract = json.loads(str(data['contract_json'].item()))
if not contract.get('lab_scene') or not data['validated'].all() or not data['lab_approach_validated'].all():
    raise ValueError('Use a validated pencil-lab catalog')
contract['appearance_randomization'] = load_profile(a.profile)
data['contract_json'] = np.asarray(json.dumps(contract, sort_keys=True))
a.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.output, **data)
report = dict(source=str(a.source), source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
              catalog_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
              goal_images='Unchanged canonical references; live scene randomized independently each episode',
              profile=contract['appearance_randomization'], targets=len(data['target_ids']))
a.output.with_suffix('.appearance.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
