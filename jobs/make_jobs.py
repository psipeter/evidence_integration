"""
SLURM job script generator for participant-level fitting and rerun.

Writes one ``.sh`` file per participant under ``jobs/``, with SBATCH
directives and commands to run ``fitting.fit`` then ``fitting.rerun``.

Entry point::

    python -m jobs.make_jobs {dataset} {model_type} [n_trials] [n_runs]
"""

import sys

import pandas as pd

from utils.paths import PROJECT_ROOT, data_path

TIME_LIMITS = {
    "Bayes": "1:0:0",
    "NoisyCounting": "24:0:0",
    "RL": "4:0:0",
    "DeGroot": "4:0:0",
    "Mean": "1:0:0",
    "ADM": "4:0:0",
    "recurrent": "48:0:0",
    "synaptic": "48:0:0",
}

PARAM_FREE = {"Bayes", "Mean"}


def main() -> None:
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    if model_type not in TIME_LIMITS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; add a TIME_LIMITS entry or fix typo."
        )

    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    pids = human["pid"].unique()

    jobs_dir = PROJECT_ROOT / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    time_limit = TIME_LIMITS[model_type]
    root = str(PROJECT_ROOT)
    count = 0
    for pid in pids:
        pid = int(pid)
        fit_cmd = f"python -m fitting.fit {dataset} {model_type} {pid} {n_trials}"
        if model_type not in PARAM_FREE and n_runs > 1:
            fit_cmd = (
                f"python -m fitting.fit {dataset} {model_type} {pid} "
                f"{n_trials} mse {n_runs}"
            )

        rerun_cmd = f"python -m fitting.rerun {dataset} {model_type} {pid}"

        script = f"""#!/bin/bash
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time={time_limit}
#SBATCH --output={root}/jobs/logs/%j.out


# assumes PY311 conda env and venv are inherited from submitting shell
cd {root}

{fit_cmd}
{rerun_cmd}
"""
        out_path = jobs_dir / f"{model_type}_{dataset}_{pid}.sh"
        out_path.write_text(script, encoding="utf-8")
        out_path.chmod(0o755)
        count += 1

    print(f"Generated {count} job script(s) in {jobs_dir}")


if __name__ == "__main__":
    main()
