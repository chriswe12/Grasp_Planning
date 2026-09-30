#!/usr/bin/env python3
"""Reproduce and verify the batched GPU depth-layout correction, without Isaac."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from grasp_planning.rl.zed_mini import load_zed_profile, pack_zed_rgbd, reproject_intrinsics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
profile = load_zed_profile()
device = "cuda:0"
values = torch.tensor([0.2, 0.5, 0.8], device=device)
depth = values[:, None, None, None].expand(-1, 144, 256, 1).clone()
rgb = torch.zeros((3, 144, 256, 3), device=device)
k = torch.tensor([[163.0, 0.0, 128.0], [0.0, 163.0, 72.0], [0.0, 0.0, 1.0]], device=device)
color, metric = reproject_intrinsics(rgb, depth, k, k)
legacy = pack_zed_rgbd(color, metric, profile, legacy_batch_layout=True)[0][:, 20, 10, 3] * 0.9 + 0.1
fixed = pack_zed_rgbd(color, metric, profile)[0][:, 20, 10, 3] * 0.9 + 0.1
torch.testing.assert_close(fixed, values)
result = dict(
    torch=torch.__version__,
    cuda=torch.version.cuda,
    input_depth_m=values.cpu().tolist(),
    legacy_packed_depth_m=legacy.cpu().tolist(),
    fixed_packed_depth_m=fixed.cpu().tolist(),
    reprojected_depth_stride=list(metric.stride()),
    passed=True,
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
