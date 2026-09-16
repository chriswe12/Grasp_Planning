# Dual planning regression fixtures

These files are a reduced sample of recorded plumbers-block planning artifacts,
not deployable plans. Tests use this directory explicitly instead of depending
on ignored files under `artifacts/`. Runtime defaults are unchanged.

The source was the `Grasp_Planning_hold_grasping` worktree's
`artifacts/dual_grasp_planning/plumbers_block` directory, read on 2026-09-16.
`plumbers_block/source_sha256.json` records each original JSON file hash.
Meshes and exact symmetry validation remain the repository's tracked assets.

For each insertion step the sample preserves:

- The first 24 retained pairs present in the saved inserter library, in producer order.
- Up to eight retained pairs with validated nonidentity transitions outside the retained execution set.
- Up to eight nonretained accepted pairs using the sampled inserter grasps.
- Referenced holder and inserter candidates, plus the first 20 raw inserter proposals.
- Part-0 candidate `i0_1808`, the floor-feasible raw proposal used by the rolled-pickup regression; it does not yield a retained direct execution.
- Original scores, poses, collision validations, retention membership, and transition matrices.

Absolute checkout prefixes are made relative, and verbose inserter score
component tables are omitted. Source metadata counters describe the original
run, not this sample. No acceptance result or geometric transform is synthesized.
The fixture exercises loader, geometry, ordering, fallback, and serialization
contracts; it does not establish robot reachability or hardware safety.

Regenerate from the recorded source directory:

```bash
python3 tests/fixtures/build_dual_fixture.py /path/to/artifacts/dual_grasp_planning/plumbers_block
```

Run the corresponding regressions:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dual_robot_simple_sim.py tests/test_dual_assembly_benchmark.py
```
