import argparse
import os

from agent.logging_utils import init_run, log_step, log_result
from agent.inspector import inspect_dataset
from agent.planner import make_plan
from agent.codegen import generate_training_script
from agent.executor import run_training_script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to MLEbench dataset directory")
    parser.add_argument("--output_dir", default="outputs", help="Where to store runs/logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run = init_run(args.output_dir, seed=args.seed)
    log_step(run, "start", {"data_dir": args.data_dir, "seed": args.seed})

    spec = inspect_dataset(args.data_dir)
    log_step(run, "inspect_dataset", spec.__dict__)

    plan = make_plan(spec, seed=args.seed)
    log_step(run, "make_plan", plan.__dict__)

    script_path = generate_training_script(plan, args.data_dir, run.run_dir)
    log_step(run, "codegen", {"script_path": script_path})

    result = run_training_script(script_path, run.run_dir)
    log_result(run, result)

    submission_path = os.path.join(run.run_dir, "submission.csv")
    if not os.path.exists(submission_path):
        raise RuntimeError(f"submission.csv not found in {run.run_dir}")

    print(f"[AGENT] Done. Submission written to: {submission_path}")


if __name__ == "__main__":
    main()
