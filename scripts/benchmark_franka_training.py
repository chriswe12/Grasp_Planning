#!/usr/bin/env python3
"""Benchmark full Franka PPO updates in an existing dedicated Isaac container."""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", required=True)
    p.add_argument("--counts", type=int, nargs="+", default=[32, 64, 96, 128])
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--cudnn-mode", choices=["compatible", "native", "disabled"], default="disabled")
    p.add_argument("--label", default="standard")
    p.add_argument(
        "--stop-on-slowdown",
        action="store_true",
        help="Stop growth when throughput drops more than 15 percent below the best count in this sweep",
    )
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/franka_scaling")
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    report_path = a.output / "benchmark.json"
    results = json.loads(report_path.read_text()) if report_path.exists() else []
    kit = "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    best_fps = 0.0
    for n in a.counts:
        busy = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if busy:
            raise RuntimeError(f"GPU 0 still has compute processes; refusing a confounded benchmark: {busy}")
        stem = f"envs_{n}_{time.strftime('%Y%m%d_%H%M%S')}"
        log = a.output / (stem + ".log")
        runs = a.output / stem
        command = [
            "docker",
            "exec",
            "-e",
            f"FRANKA_CUDNN_MODE={a.cudnn_mode}",
            a.container,
            "bash",
            "scripts/franka_isaac_python.sh",
            "isaac_rl/scripts/train_franka_zed.py",
            "--headless",
            "--catalog",
            "isaac_rl/data/franka_fabrica_pencil_randomized/catalog.npz",
            "--num-envs",
            str(n),
            "--iterations",
            str(a.iterations),
            "--run-dir",
            str(Path("/workspace/project") / runs.relative_to(ROOT)),
            "--kit_args",
            kit,
        ]
        peak = 0.0
        samples = []
        start = time.monotonic()
        source_hash = hashlib.sha256((ROOT / "grasp_planning/rl/franka_appearance.py").read_bytes()).hexdigest()
        with log.open("w") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                query = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                )
                if query.returncode == 0:
                    used, util = map(float, query.stdout.splitlines()[0].split(","))
                    peak = max(peak, used)
                    samples.append(util)
                if time.monotonic() - start > 900:
                    subprocess.run(
                        ["docker", "exec", a.container, "pkill", "-KILL", "-f", "isaac_rl/scripts/train_franka_zed.py"],
                        check=False,
                    )
                    proc.wait(timeout=30)
                    break
                time.sleep(0.5)
        body = log.read_text()
        values = [float(x) for x in re.findall(r"fps total: ([0-9.]+)", body)]
        record = dict(
            num_envs=n,
            iterations=a.iterations,
            exit_code=proc.returncode,
            peak_device_mib=peak,
            cudnn_mode=a.cudnn_mode,
            label=a.label,
            appearance_source_sha256=source_hash,
            median_gpu_utilization_percent=statistics.median(samples) if samples else None,
            median_fps=statistics.median(values[1:]) if len(values) > 1 else None,
            epochs_recorded=len(values),
            log=str(log.relative_to(ROOT)),
            wall_seconds=time.monotonic() - start,
            passed=False,
        )
        if len(values) == a.iterations and "[FRANKA TRAIN] Completed bounded run:" in body:
            run = next(runs.iterdir())
            verification = a.output / (stem + ".verification.json")
            check = subprocess.run(
                [
                    "docker",
                    "exec",
                    a.container,
                    "/isaac-sim/python.sh",
                    "scripts/verify_franka_training_run.py",
                    "--run",
                    str(Path("/workspace/project") / run.relative_to(ROOT)),
                    "--output",
                    str(Path("/workspace/project") / verification.relative_to(ROOT)),
                ],
                capture_output=True,
                text=True,
            )
            record["passed"] = (
                check.returncode == 0 and verification.is_file() and json.loads(verification.read_text())["passed"]
            )
            record["verification"] = str(verification.relative_to(ROOT))
            if not record["passed"]:
                record["verification_error"] = check.stdout + check.stderr
        else:
            record["error_tail"] = body[-4000:]
        results.append(record)
        report_path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if not record["passed"] or peak > 22500:
            print("Stopping growth at failed run or <2 GiB device headroom.", flush=True)
            break
        if a.stop_on_slowdown and record["median_fps"] < 0.85 * best_fps:
            print("Stopping growth: larger scene is over 15 percent slower.", flush=True)
            break
        best_fps = max(best_fps, record["median_fps"])


if __name__ == "__main__":
    main()
