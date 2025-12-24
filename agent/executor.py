import os
import subprocess
import sys

def run_training_script(script_path: str, run_dir: str):
    stdout_path = os.path.join(run_dir, "train_stdout.txt")
    stderr_path = os.path.join(run_dir, "train_stderr.txt")

    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        proc = subprocess.run([sys.executable, os.path.basename(script_path)],
                              cwd=run_dir, text=True, stdout=out, stderr=err)

    if proc.returncode != 0:
        # show last stderr lines to make debugging fast
        try:
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                tail = f.read()[-4000:]
        except Exception:
            tail = "<could not read stderr>"
        raise RuntimeError(f"Training script failed (code={proc.returncode}). Stderr tail:\n{tail}")

    submission_path = os.path.join(run_dir, "submission.csv")
    if not os.path.exists(submission_path):
        raise RuntimeError(f"Training script finished but did not write submission.csv in {run_dir}")

    return {"status": "ok", "submission_path": submission_path}
