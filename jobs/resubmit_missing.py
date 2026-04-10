"""
Utility to detect participants without saved fit outputs and resubmit their
SLURM job scripts.

Use after a batch run to catch failures: compares ``data/{dataset}.pkl`` pids
against per-participant ``*_params.pkl`` files, then ``sbatch`` the matching
``jobs/{model_type}_{dataset}_{pid}.sh`` scripts when not in dry-run mode.

Entry point::

    python -m jobs.resubmit_missing {dataset} {model_type}

Pass ``--dry-run`` anywhere on the command line to only list missing pids
without submitting.
"""

import subprocess
import sys
import time

import pandas as pd

from utils.paths import PROJECT_ROOT, data_path


def find_missing(dataset: str, model_type: str) -> list[int]:
    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    pids = sorted(int(x) for x in human["pid"].unique())
    missing: list[int] = []
    for pid in pids:
        params_path = data_path(f"{model_type}_{dataset}_{pid}_params.pkl")
        if not params_path.is_file():
            missing.append(pid)
    return missing


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python -m jobs.resubmit_missing {dataset} {model_type} [--dry-run]",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset = sys.argv[1]
    model_type = sys.argv[2]
    dry_run = "--dry-run" in sys.argv

    missing = find_missing(dataset, model_type)
    if not missing:
        print(f"All participants complete for {model_type} {dataset}")
        return

    n = len(missing)
    print(f"Missing {n} participants: {missing}")

    if dry_run:
        return

    jobs_dir = PROJECT_ROOT / "jobs"
    resubmitted = 0
    for pid in missing:
        script_path = jobs_dir / f"{model_type}_{dataset}_{pid}.sh"
        if not script_path.is_file():
            print(
                f"Warning: no job script for pid={pid}, skipping: {script_path}",
                file=sys.stderr,
            )
            continue
        result = subprocess.run(["sbatch", str(script_path)], check=False)
        if result.returncode == 0:
            resubmitted += 1
        else:
            print(
                f"Warning: sbatch failed for pid={pid} (exit {result.returncode})",
                file=sys.stderr,
            )
        time.sleep(0.5)

    print(f"Resubmitted {resubmitted} job(s)")


if __name__ == "__main__":
    main()
