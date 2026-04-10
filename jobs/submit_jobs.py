"""
SLURM job submission helper: ``sbatch`` every generated script for a
dataset/model pair.

Entry point::

    python -m jobs.submit_jobs {dataset} {model_type}
"""

import subprocess
import sys
import time

import pandas as pd

from utils.paths import PROJECT_ROOT, data_path


def main() -> None:
    dataset = sys.argv[1]
    model_type = sys.argv[2]

    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    pids = human["pid"].unique()

    jobs_dir = PROJECT_ROOT / "jobs"
    submitted = 0
    for pid in pids:
        pid = int(pid)
        script_path = jobs_dir / f"{model_type}_{dataset}_{pid}.sh"
        if not script_path.is_file():
            print(
                f"Warning: missing job script, skipping pid={pid}: {script_path}",
                file=sys.stderr,
            )
            continue
        result = subprocess.run(["sbatch", str(script_path)], check=False)
        if result.returncode == 0:
            submitted += 1
        else:
            print(
                f"Warning: sbatch failed for pid={pid} (exit {result.returncode})",
                file=sys.stderr,
            )
        time.sleep(0.5)

    print(f"Submitted {submitted} job(s)")


if __name__ == "__main__":
    main()
