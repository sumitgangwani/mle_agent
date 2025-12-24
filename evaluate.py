import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple, Optional


def get_latest_run_dir(output_dir: str) -> str:
    """Return the most recently modified run_ directory inside output_dir."""
    runs = []
    if not os.path.exists(output_dir):
        raise RuntimeError(f"output_dir does not exist: {output_dir}")

    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full) and name.startswith("run_"):
            runs.append(full)

    if not runs:
        raise RuntimeError(f"No run_ directories found in {output_dir}")

    runs.sort(key=os.path.getmtime)
    return runs[-1]


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


def run_agent_once(data_dir: str, output_dir: str, seed: int) -> str:
    """
    Call run_agent.py for a single seed, return path to produced submission.csv
    (assumes run_agent creates a new run_* directory under output_dir).
    """
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
    submission_path = os.path.join(run_dir, "submission.csv")
    if not os.path.exists(submission_path):
        raise RuntimeError(f"submission.csv not found in {run_dir}")

    print(f"Latest run dir: {run_dir}")
    print(f"Submission: {submission_path}")
    return submission_path


def _extract_json_object(text: str) -> Dict:
    """
    Extract the competition report JSON object from mlebench output.
    We look for the first {...} block and parse it.
    """
    # This is robust to logs before/after JSON
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise RuntimeError("Could not find JSON object in mlebench output.")

    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # help debugging
        raise RuntimeError(f"Failed to parse JSON from mlebench output: {e}\nExtracted:\n{json_str}") from e


def grade_submission_with_mlebench(submission_path: str, competition_id: str) -> Dict:
    """
    Run: mlebench grade-sample <submission_path> <competition_id>
    Return parsed JSON report.
    """
    cmd = ["mlebench", "grade-sample", submission_path, competition_id]
    print("Grading Command:", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0:
        # show output for debugging
        raise RuntimeError(
            f"mlebench grade-sample failed (code={proc.returncode}). Output:\n{combined}"
        )

    report = _extract_json_object(combined)
    return report


def save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_table_csv(path: str, rows: List[Dict]) -> None:
    """
    Write rows (dicts) to CSV with union of keys across rows as columns.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Union of keys, stable-ish order: keep first-seen ordering
    cols: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser()

    # Same as your old script
    parser.add_argument("--data_dir", required=True, help="Path to prepared MLEbench (lite) dataset directory")
    parser.add_argument("--output_dir", default="outputs", help="Where run_agent.py stores run_* directories")

    # NEW: must tell evaluate.py which competition to grade against
    parser.add_argument("--competition_id", required=True, help="MLEbench competition id, e.g. siim-isic-melanoma-classification")

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="List of seeds to evaluate, e.g. --seeds 0 1 2",
    )

    # NEW: save JSONs + aggregated CSV
    parser.add_argument(
        "--reports_dir",
        default="reports",
        help="Directory to save mlebench JSON reports (one per seed).",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    all_reports: List[Dict] = []
    scores: List[float] = []
    any_medals: List[float] = []

    for seed in args.seeds:
        submission_path = run_agent_once(args.data_dir, args.output_dir, seed)

        report = grade_submission_with_mlebench(submission_path, args.competition_id)

        # Keep track
        all_reports.append(report)

        # Extract fields we need for stats
        score = float(report.get("score"))
        any_medal = bool(report.get("any_medal"))

        scores.append(score)
        any_medals.append(1.0 if any_medal else 0.0)

        print(f"Seed {seed}: score={score:.6f} any_medal={any_medal}")

        # Save JSON per seed (helpful for Hexo)
        out_json = os.path.join(args.reports_dir, f"{args.competition_id}_seed_{seed}.json")
        save_json(out_json, report)
        print(f"Saved report: {out_json}")

    # Summary stats
    score_mean, score_sem = mean_and_sem(scores)
    medal_mean, medal_sem = mean_and_sem(any_medals)

    print("\n=== Summary (official mlebench grade-sample) ===")
    print(f"Competition: {args.competition_id}")
    print(f"Seeds: {args.seeds}")
    print(f"Scores: {scores}")
    print(f"Score mean ± SEM: {score_mean:.6f} ± {score_sem:.6f}")
    print(f"Any Medal (%) mean ± SEM: {medal_mean*100:.2f}% ± {medal_sem*100:.2f}%")

    # Also save an aggregated table CSV (one row per seed for this competition)
    table_csv = os.path.join(args.reports_dir, f"{args.competition_id}_all_seeds_table.csv")
    write_table_csv(table_csv, all_reports)
    print(f"Saved table CSV: {table_csv}")


if __name__ == "__main__":
    main()
