#!/usr/bin/env python3
"""Submit a verified multipart training continuation chain and persistent result watcher."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def segment_plan(segments, epochs_per_segment, time_limit, startup_epochs=0, startup_time_limit="04:00:00"):
    """Optional short first allocation without changing the final learning budget."""
    if segments < 1 or epochs_per_segment <= 3:
        raise ValueError("Need positive segments and more than three epochs per segment")
    if startup_epochs and not 3 < startup_epochs < epochs_per_segment:
        raise ValueError("Startup milestone must be above the gate and below the first regular milestone")
    plan = [(i, i * epochs_per_segment, time_limit) for i in range(1, segments + 1)]
    return ([(0, startup_epochs, startup_time_limit)] if startup_epochs else []) + plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--catalog-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-job",
        type=int,
        required=True,
        help="Successful production-catalog PPO gate; never a diagnostic dataset",
    )
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--epochs-per-segment", type=int, default=1024)
    parser.add_argument("--time-limit", default="24:00:00")
    parser.add_argument(
        "--startup-epochs",
        type=int,
        default=0,
        help="Optional short first continuation; final epoch budget is unchanged",
    )
    parser.add_argument("--startup-time-limit", default="04:00:00")
    args = parser.parse_args()
    audit = json.loads(args.catalog_audit.read_text())
    assert audit["passed"] and audit["splits"]["train"]["targets"] > 0
    assert hashlib.sha256((ROOT / args.catalog).read_bytes()).hexdigest() == audit["catalog_sha256"], (
        "Catalog changed after its audit"
    )
    plan = segment_plan(
        args.segments, args.epochs_per_segment, args.time_limit, args.startup_epochs, args.startup_time_limit
    )
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), "Refusing to duplicate an existing submission chain"
    env = dict(os.environ, EULER_CONFIG_PATH=str(args.config.resolve()), EULER_SKIP_SYNC="1")
    login = subprocess.check_output(
        ["bash", "-c", 'source "$1"; printf "%s" "$EULER_LOGIN"', "bash", str(args.config.resolve())], text=True
    ).strip()
    gate_record = (
        subprocess.check_output(
            [
                "ssh",
                login,
                "sacct",
                "-X",
                "-j",
                str(args.initial_job),
                "--noheader",
                "--parsable2",
                "--format=State,ExitCode",
            ],
            text=True,
            timeout=60,
        )
        .strip()
        .splitlines()
    )
    assert gate_record, "Initial gate is absent from Slurm accounting"
    gate_state, gate_exit, *_ = gate_record[0].split("|")
    initial_completed = gate_state == "COMPLETED" and gate_exit == "0:0"
    assert initial_completed or gate_state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING"), gate_record
    report = dict(
        catalog=args.catalog,
        catalog_sha256=audit["catalog_sha256"],
        config=str(args.config.resolve()),
        initial_job=args.initial_job,
        jobs=[],
        initial_gate_state=gate_state,
        initial_gate_exit_code=gate_exit,
        gpu_count=4,
        environments_per_gpu=32,
        global_rollout_frames=8192,
        final_epoch=args.segments * args.epochs_per_segment,
        target_transitions=args.segments * args.epochs_per_segment * 8192,
        planned_milestones=[dict(segment=i, final_epoch=epoch, time_limit=limit) for i, epoch, limit in plan],
        budget_note="A substantial learning budget, not a convergence guarantee; inspect held-out evaluation between segments.",
    )

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    previous = args.initial_job
    for segment, final_epoch, time_limit in plan:
        command = [
            "bash",
            str(ROOT / "euler/submit.sh"),
            "franka-train",
            "--gpu-count",
            "4",
            "--global-minibatch-size",
            "1024",
            "--num-envs",
            "32",
            "--catalog",
            args.catalog,
            "--iterations",
            str(final_epoch),
            "--save-frequency",
            "100",
            "--resume-job",
            str(previous),
            "--evaluate-after",
            "--job-label",
            f"franka-all-long-s{segment}",
            "--time-limit",
            time_limit,
        ]
        # Completed jobs can disappear from the controller before submission
        # (MinJobAge). Accounting plus strict resume-artifact checks suffice;
        # pending/running predecessors still require an afterok dependency.
        if not (previous == args.initial_job and initial_completed):
            command.extend(["--afterok", str(previous)])
        if segment == args.segments:
            command.append("--evaluate-test-after")
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, flush=True)
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if result.returncode or not match:
            raise RuntimeError("Submission failed; completed submissions remain recorded in manifest")
        job = int(match.group(1))
        report["jobs"].append(
            dict(
                segment=segment,
                job_id=job,
                predecessor=previous,
                final_epoch=final_epoch,
                time_limit=time_limit,
                command=command,
            )
        )
        save()
        previous = job
    log = args.output.with_suffix(".watch.log")
    with log.open("a") as output:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "euler/watch_franka_chain.py"), str(args.output)],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    report["watcher_pid"] = process.pid
    report["watcher_log"] = str(log)
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
