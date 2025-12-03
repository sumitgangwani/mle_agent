import argparse
import json
import math
import os
import subprocess
import sys
from typing import List, Tuple


def get_latest_run_dir(output_dir: str) -> str:
    """Return the most recently modified run_ directory inside output_dir."""
    runs = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full) and name.startswith("run_"):
            runs.append(full)
    if not runs:
        raise RuntimeError(f"No run_ directories found in {output_dir}")
    runs.sort(key=os.path.getmtime)
    return runs[-1]


def get_metric_from_log(run_dir: str) -> Tuple[str, float]:
    """Read agent_log.jsonl and return (metric_name, metric_value) from the last result record."""
    log_path = os.path.join(run_dir, "agent_log.jsonl")
    if not os.path.exists(log_path):
        raise RuntimeError(f"agent_log.jsonl not found in {run_dir}")

    metric_name = None
    metric_value = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "result":
                result = rec.get("result", {})
                metric_name = result.get("metric_name")
                metric_value = result.get("metric_value")

    if metric_name is None or metric_value is None:
        raise RuntimeError(f"Could not find metric in {log_path}")

    return metric_name, float(metric_value)


def run_once(data_dir: str, output_dir: str, seed: int) -> Tuple[str, float]:
    """Call run_agent.py for a single seed, then extract metric."""
    cmd = [
        sys.executable,
        "run_agent.py",
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
        "--seed",
        str(seed),
    ]
    print(f"\n=== Running seed {seed} ===")
    print("Command:", " ".join(cmd))

    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"run_agent.py failed for seed {seed} with code {proc.returncode}")

    run_dir = get_latest_run_dir(output_dir)
    print(f"Latest run dir: {run_dir}")

    metric_name, metric_value = get_metric_from_log(run_dir)
    print(f"Seed {seed}  {metric_name} = {metric_value:.6f}")
    return metric_name, metric_value


def mean_and_sem(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    return mean, sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to MLEbench dataset directory")
    parser.add_argument("--output_dir", default="outputs", help="Where run_agent.py stores runs")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="List of seeds to evaluate, e.g. --seeds 0 1 2",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metrics: List[float] = []
    metric_name: str = ""

    for seed in args.seeds:
        name, value = run_once(args.data_dir, args.output_dir, seed)
        metric_name = name
        metrics.append(value)

    mean, sem = mean_and_sem(metrics)
    print("\n=== Summary ===")
    print(f"Seeds: {args.seeds}")
    print(f"Metric: {metric_name}")
    print(f"Values: {metrics}")
    print(f"Mean  SEM: {mean:.6f}  {sem:.6f}")


if __name__ == "__main__":
    main()
