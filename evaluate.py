# evaluate.py
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path
from typing import Any, Dict


def infer_competition_id(data_dir: str) -> str:
    """
    Best-effort inference from folder name.
    Update mapping if your local folder names differ.
    """
    name = Path(data_dir).name

    mapping = {
        "SIIM_ISIC_Melanoma_Classification": "siim-isic-melanoma-classification",
        "siim_isic_melanoma_classification": "siim-isic-melanoma-classification",
        "spooky_author": "spooky-author-identification",
        "spooky_author_identification": "spooky-author-identification",
        "tabular-playground-series-may-2022": "tabular-playground-series-may-2022",
        "text_normalization": "text-normalization-challenge-english-language",
        "text-normalization-challenge-english-language": "text-normalization-challenge-english-language",
        "whale_challenge": "the-icml-2013-whale-challenge-right-whale-redux",
        "the-icml-2013-whale-challenge-right-whale-redux": "the-icml-2013-whale-challenge-right-whale-redux",
    }
    return mapping.get(name, name)


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def find_run_dir(output_dir: str, seed: int, stdout: str) -> Path:
    """
    Locate the run directory created by run_agent.py.
    We try:
    1) parse from stdout if it prints it
    2) fallback to most-recent run_*_seed{seed} directory
    """
    # Common patterns you previously had:
    # "Latest run dir: outputs\\run_...._seed0"
    m = re.search(r"Latest run dir:\s*(.+)", stdout)
    if m:
        candidate = m.group(1).strip().strip('"')
        p = Path(candidate)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists() and p.is_dir():
            return p

    # "[AGENT] Done. Submission written to: outputs\\run_..._seed0\\submission.csv"
    m = re.search(r"Submission written to:\s*(.+submission\.csv)", stdout, re.IGNORECASE)
    if m:
        sub_path = m.group(1).strip().strip('"')
        p = Path(sub_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return p.parent

    out = Path(output_dir)
    if not out.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # fallback: choose newest dir matching run_*_seed{seed}
    pattern = re.compile(rf"run_\d+_\d+_seed{seed}$")
    candidates = [d for d in out.iterdir() if d.is_dir() and pattern.search(d.name)]
    if not candidates:
        # last resort: any run_*_seed{seed}
        candidates = [d for d in out.iterdir() if d.is_dir() and d.name.endswith(f"_seed{seed}") and d.name.startswith("run_")]

    if not candidates:
        raise FileNotFoundError(f"Could not locate run directory for seed {seed} in {output_dir}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_submission_csv(run_dir: Path) -> Path:
    """
    Find submission.csv in run_dir. Handles minor variations.
    """
    direct = run_dir / "submission.csv"
    if direct.exists():
        return direct

    # sometimes scripts save into a nested folder
    for p in run_dir.rglob("submission.csv"):
        return p

    raise FileNotFoundError(f"submission.csv not found under {run_dir}")


def grade_with_mlebench(competition_id: str, submission_csv: Path, run_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """
    Grade using MLE-bench CLI.

    IMPORTANT:
    `mlebench grade` expects --submission to be a JSONL file where each line contains:
      - competition_id
      - submission_path (path to the CSV)
    (not the CSV itself).
    """

    out_dir = run_dir / "mlebench_grade"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create the JSONL that mlebench expects
    submissions_jsonl = out_dir / "submissions.jsonl"
    payload = {
        "competition_id": competition_id,
        "submission_path": str(submission_csv),
    }
    submissions_jsonl.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    cmd = [
        "mlebench",
        "grade",
        "--submission",
        str(submissions_jsonl),
        "--output-dir",
        str(out_dir),
        "--data-dir",
        str(data_dir),
    ]

    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(
            "mlebench grade failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{out}\n"
            f"STDERR:\n{err}\n"
        )

    # MLEbench often writes results to JSON/JSONL in output-dir.
    # Try to load the newest JSON/JSONL produced.
    candidates = list(out_dir.rglob("*.json")) + list(out_dir.rglob("*.jsonl"))
    if candidates:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)

        if newest.suffix == ".json":
            return json.loads(newest.read_text(encoding="utf-8"))

        # jsonl: return last json object line
        lines = [ln.strip() for ln in newest.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in reversed(lines):
            if ln.startswith("{") and ln.endswith("}"):
                return json.loads(ln)

    # Fallback: sometimes it prints JSON to stdout
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            return json.loads(ln)

    raise RuntimeError(
        "mlebench grade succeeded but no result JSON/JSONL was found.\n"
        f"Looked in: {out_dir}\n"
        f"STDOUT:\n{out}\n"
        f"STDERR:\n{err}\n"
    )


def append_csv_row(csv_path: Path, header: List[str], row: Dict[str, Any]) -> None:
    exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def mean_sem(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    sem = math.sqrt(var) / math.sqrt(len(values))
    return m, sem


def run_once(data_dir: str, output_dir: str, seed: int) -> Tuple[Path, Path, str]:
    """
    Runs your existing run_agent.py exactly like before.
    Returns (run_dir, submission_csv, combined_stdout)
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

    rc, out, err = run_cmd(cmd)
    combined = (out or "") + ("\n" + err if err else "")

    if rc != 0:
        # Still show logs to help debugging
        print("run_agent.py failed.")
        print("STDOUT:\n", out)
        print("STDERR:\n", err)
        raise RuntimeError(f"run_agent.py failed for seed {seed} with code {rc}")

    run_dir = find_run_dir(output_dir, seed, out)
    submission_csv = find_submission_csv(run_dir)
    return run_dir, submission_csv, combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--competition_id",
        type=str,
        default=None,
        help="MLEbench competition id. If omitted, inferred from data_dir folder name.",
    )
    args = parser.parse_args()

    competition_id = args.competition_id or infer_competition_id(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV summary file (per seed rows)
    summary_csv = output_dir / "grading_summary.csv"
    header = [
        "competition_id",
        "seed",
        "score",
        "gold_threshold",
        "silver_threshold",
        "bronze_threshold",
        "median_threshold",
        "any_medal",
        "submission_csv",
        "run_dir",
    ]

    scores: List[float] = []
    any_medal_flags: List[bool] = []

    # save per-run grader json
    grading_json_dir = output_dir / "grading_json"
    grading_json_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        run_dir, sub_csv, logs = run_once(args.data_dir, args.output_dir, seed)

        grade = grade_with_mlebench(
            competition_id=competition_id,
            submission_csv=sub_csv,
            run_dir=run_dir,
            data_dir=Path(args.data_dir),
        )

        # Persist full JSON for auditability
        json_path = grading_json_dir / f"{competition_id}_seed{seed}.json"
        json_path.write_text(json.dumps(grade, indent=2), encoding="utf-8")

        score = grade.get("score", None)
        any_medal = bool(grade.get("any_medal", False))

        if isinstance(score, (int, float)):
            scores.append(float(score))
        else:
            # Some graders might return string – try coercion
            try:
                scores.append(float(score))
            except Exception:
                print(f"Warning: could not parse score as float for seed {seed}: {score}")

        any_medal_flags.append(any_medal)

        row = {
            "competition_id": competition_id,
            "seed": seed,
            "score": grade.get("score"),
            "gold_threshold": grade.get("gold_threshold"),
            "silver_threshold": grade.get("silver_threshold"),
            "bronze_threshold": grade.get("bronze_threshold"),
            "median_threshold": grade.get("median_threshold"),
            "any_medal": grade.get("any_medal"),
            "submission_csv": str(sub_csv),
            "run_dir": str(run_dir),
        }
        append_csv_row(summary_csv, header, row)

        print(f"Seed {seed} grade score = {grade.get('score')} | any_medal = {grade.get('any_medal')}")

    m, sem = mean_sem(scores)
    any_medal_pct = 100.0 * (sum(1 for x in any_medal_flags if x) / len(any_medal_flags)) if any_medal_flags else 0.0

    print("\n=== Final Summary ===")
    print("competition_id:", competition_id)
    print("seeds:", args.seeds)
    print("scores:", scores)
    print(f"Score Mean ± SEM: {m:.6f} ± {sem:.6f}")
    print(f"Any Medal (%): {any_medal_pct:.2f}")
    print(f"Wrote per-seed grading table to: {summary_csv}")
    print(f"Wrote full grading JSONs to: {grading_json_dir}")


if __name__ == "__main__":
    main()
