import os
import subprocess
import sys
from typing import Dict


def run_training_script(script_path: str, run_dir: str) -> Dict:
    # Ensure we run from the directory where the script lives
    script_abs_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_abs_path)
    script_name = os.path.basename(script_abs_path)

    cmd = [sys.executable, script_name]
    print(f"[EXECUTOR] Running: {' '.join(cmd)} in {script_dir}")

    proc = subprocess.run(
        cmd,
        cwd=script_dir,          # run *in* the script directory
        capture_output=True,
        text=True,
    )

    stdout_path = os.path.join(run_dir, "train_stdout.txt")
    stderr_path = os.path.join(run_dir, "train_stderr.txt")
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(proc.stderr)

    metric_value = None
    metric_name = None
    for line in proc.stdout.splitlines():
        if "VAL_METRIC:" in line:
            part = line.split("VAL_METRIC:")[-1].strip()
            if "=" in part:
                metric_name = part.split("=")[0].strip()
                val = part.split("=")[1].strip().split(",")[0]
                try:
                    metric_value = float(val)
                except ValueError:
                    metric_value = None
            break

    result: Dict = {
        "returncode": proc.returncode,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }
    if proc.returncode != 0:
        result["error"] = "Training script failed"

    return result
