import json
import os
import time
from dataclasses import dataclass


@dataclass
class RunInfo:
    run_id: str
    run_dir: str
    log_path: str
    seed: int


def init_run(output_root: str, seed: int) -> RunInfo:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{ts}_seed{seed}"
    run_dir = os.path.join(output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "agent_log.jsonl")
    with open(log_path, "w") as f:
        f.write("")  # create empty file

    return RunInfo(run_id=run_id, run_dir=run_dir, log_path=log_path, seed=seed)


def _write_log(run: RunInfo, record: dict):
    record["run_id"] = run.run_id
    record["timestamp"] = time.time()
    with open(run.log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_step(run: RunInfo, step: str, info: dict):
    _write_log(run, {"type": "step", "step": step, "info": info})


def log_result(run: RunInfo, result: dict):
    _write_log(run, {"type": "result", "result": result})
