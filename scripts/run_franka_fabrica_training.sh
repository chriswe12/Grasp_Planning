#!/usr/bin/env bash
# Container entrypoint: preserve logs, then evaluate held-out targets after training.
set -euo pipefail
catalog=$1
run_root=$2
num_envs=$3
iterations=$4
gpu_count=${5:-1}
resume_checkpoint=${6:-}
mkdir -p "$run_root"
exec > >(tee -a "$run_root/session.log") 2>&1
python_runner=(bash /workspace/project/scripts/franka_isaac_python.sh)
kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8'
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
train_args=(isaac_rl/scripts/train_franka_zed.py --headless
  --catalog "$catalog" --num-envs "$num_envs" --iterations "$iterations" --run-dir "$run_root/train" --kit_args "$kit_args")
if [[ -n "$resume_checkpoint" ]]; then train_args+=(--checkpoint "$resume_checkpoint"); fi
if (( gpu_count > 1 )); then
  "${python_runner[@]}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$gpu_count" \
    "${train_args[@]}" --distributed --experiment-name distributed
else
  "${python_runner[@]}" "${train_args[@]}"
fi
completion=$(find "$run_root/train" -name training_completed.json -print -quit)
test -n "$completion"
"${python_runner[@]}" scripts/verify_franka_training_run.py --run "$(dirname "$completion")" \
  --output "$run_root/final_training_check.json"
checkpoint=$(find "$run_root/train" -path '*/nn/franka_zed.pth' -print -quit)
test -n "$checkpoint"
for split in validation test; do
  "${python_runner[@]}" isaac_rl/scripts/train_franka_zed.py --headless \
    --catalog "$catalog" --num-envs 8 --evaluate --catalog-split "$split" \
    --checkpoint "$checkpoint" --evaluation-steps 600 --run-dir "$run_root/evaluate_$split" --kit_args "$kit_args"
  evaluation=$(find "$run_root/evaluate_$split" -name evaluation.json -print -quit)
  test -n "$evaluation"
done
touch "$run_root/COMPLETED"
