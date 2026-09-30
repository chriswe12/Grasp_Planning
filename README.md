# Fabrica Grasp Planning

YAML-driven grasp planning for Franka Research 3 and KUKA iiwa7, with
MuJoCo/Isaac simulation and guarded ROS2 execution. The dual-arm workflow
holds a partial assembly and transports an incoming part to pre-insertion.
Standalone RGB-D visual-servo RL is a separate workflow.

## Start Here

Clone with submodules, or initialize them in an existing clone:

```bash
git submodule update --init --recursive
./run_pipeline.sh --help
```

The default workflow is **dual-arm**. Choose `--workflow single-object`
explicitly for the single-object pipeline. Run all commands below from the
repository root.

Before running a pipeline, follow the [dependency and asset checklist](docs/deployment.md).
Git alone does not provide the external ROS2/MoveIt and Isaac installations,
training catalogs, or policy checkpoints. The [pipeline reference](docs/pipeline-guide.md)
contains detailed simulator setup and configuration examples.

Start without robot or simulator execution:

```bash
./run_pipeline.sh --workflow single-object --mode sim --backend none
```

This generates planning artifacts using the configured object pose. Inspect
the JSON/HTML outputs under `artifacts/` before enabling a backend.
Planning success is not evidence of simulation or hardware execution success.

## Choose A Workflow

| Goal | Entrypoint | Configuration / guide |
| --- | --- | --- |
| Single-object planning or simulation | `./run_pipeline.sh --workflow single-object --mode sim` | `configs/grasp_pipeline_sim.yaml` |
| ROS2 perception with simulated execution | `./run_pipeline.sh --workflow single-object --mode pitl` | `configs/grasp_pipeline_pitl.yaml` |
| Dual-arm Isaac simulation | `./run_pipeline.sh --mode sim --robots both` | `configs/dual_grasp_planning.yaml` |
| Planning-only generation benchmark | `./run_pipeline.sh --benchmark grasp-generation` | `configs/grasp_generation_benchmark.yaml` |
| All-step dual-arm benchmark | `./run_pipeline.sh --benchmark dual-assembly` | `configs/dual_assembly_benchmark.yaml` |
| Standalone Franka policy setup | `./franka_policy.sh configure` | [Franka workbench](docs/franka-policy.md) |
| Visual-servo training and evaluation | `isaac_rl/` submodule | [Training guide](docs/training.md) |

The [execution map](docs/execution-paths.md) explains the entrypoints,
artifact contracts, backend selection, and benchmark boundaries.

## Hardware And Scope

Read the [deployment checklist](docs/deployment.md) and the applicable
[KUKA runbook](docs/kuka-runbook.md) or [Franka workbench](docs/franka-policy.md)
before connecting hardware. Sourcing `setup_robot_env.sh` sets up ROS discovery;
it does not verify network, FCI, safety/brake, robot-model, or realtime readiness.

Single-object real execution is disabled in the default config, requires
confirmation, stops at pregrasp, and leaves the gripper disabled. Do not bypass
these gates to test installation. The Franka workbench defaults to observe-only.
Dual-arm execution currently ends no later than pre-insertion: constrained
insertion, release, retreat, and arbitrary full-assembly execution are not implemented.

Saved stage-2 bundles are the source of truth for single-object execution.
Standalone RL alignment and completion do not establish reliable physical
pickup or lift. Training and Euler job submission are explicit separate actions.

## Repository Map

| Directory / file | Purpose |
| --- | --- |
| `run_pipeline.sh` | Public planning/execution and benchmark entrypoint |
| `configs/` | Pipeline, robot, benchmark, and training recipes |
| `grasp_planning/` | Planning, frames, simulator, ROS2, and RL helpers |
| `scripts/` | Asset builders, runners, evaluation, and diagnostics |
| `ros2_ws/` | [Hardware-facing ROS2 workspace](ros2_ws/README.md) |
| `isaac_rl/` | [Standalone Isaac Lab training project](isaac_rl/README.md) |
| `euler/` | [Cluster staging, smoke checks, and submission](euler/README.md) |
| `assets/` | Fabrica geometry, robot models, and scenes |
| `tests/` | Unit and regression coverage |
| `artifacts/`, `.cache/`, `logs/` | Local generated outputs, not source code |

## Reference

- [Pipeline setup and operations](docs/pipeline-guide.md)
- [Execution paths and artifact flow](docs/execution-paths.md)
- [Deployment dependencies and safety prerequisites](docs/deployment.md)
- [Dual-arm symmetry and frame invariants](docs/dual-arm-symmetry.md)
- [KUKA lab runbook](docs/kuka-runbook.md)
- [Franka real-policy workbench](docs/franka-policy.md)
- [Training and dataset contracts](docs/training.md)

Historical implementation plans, completion diaries, and dated experiment
notes are available in Git history. Use current configs, code, and saved
artifact metadata rather than old job IDs or checkpoint paths.
