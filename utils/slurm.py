"""Shared SLURM job submission utilities."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TIME_LIMITS = {
    "Mean": "0:30:0",
    "NoisyCounting": "24:0:0",
    "RL": "2:0:0",
    "RL_lambda": "2:0:0",
    "NoisyRL_lambda": "6:0:0",
    "Mean": "0:30:0",
    "LeakyIntegrator": "1:0:0",
    "PrimacyRecency": "2:0:0",
    "PearceHall": "2:0:0",
    "NEF": "72:0:0",
}

DEFAULT_MEM_LIMITS = {
    "Mean": "8G",
    "NoisyCounting": "16G",
    "RL": "8G",
    "RL_lambda": "8G",
    "NoisyRL_lambda": "8G",
    "Mean": "8G",
    "LeakyIntegrator": "4G",
    "PrimacyRecency": "8G",
    "PearceHall": "8G",
    "NEF": "32G",
}

# DEFAULT_TIME_LIMITS["NEF"]=72h is sized for a full 200-trial Optuna fit --
# far too long for utils/save_activities.py's own job, which is ONE forward
# pass per trial (no hyperparameter search at all), the same shape as
# utils/save_responses.py's job. Requesting 72h for either causes exactly the
# problem seen in practice: SLURM's scheduler can't guarantee any node for
# that long a walltime if a maintenance window falls anywhere inside it,
# leaving the job stuck pending ("ReqNodeNotAvail, Reserved for maintenance")
# even though the job itself would finish in minutes. Memory is UNCHANGED
# (same network, same n_neurons, so the same footprint) -- only walltime
# scales with "how many total passes", which drops from 200 to 1 here.
SINGLE_PASS_TIME_LIMIT = "2:0:0"

def make_job_script(
    root: str,
    commands: list[str],
    time_limit: str = "4:0:0",
    mem: str = "8G",
    log_dir: str | None = None,
) -> str:
    if log_dir is None:
        log_dir = f"{root}/logs"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --mem={mem}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={log_dir}/%j.out",
        "",
        "# assumes PY311 conda env and venv are inherited from submitting shell",
        f"cd {root}",
        "",
    ]
    lines.extend(commands)
    return "\n".join(lines) + "\n"

def submit_script(script_path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[dry_run] would submit: {script_path}")
        return
    # --export=ALL is explicit here rather than relying on sbatch's own
    # cluster-configured default -- this job script already assumes the
    # submitting shell's conda/venv gets inherited (see its own comment);
    # making that guaranteed rather than implicit also covers any other
    # env var a caller sets before submitting.
    result = subprocess.run(["sbatch", "--export=ALL", str(script_path)], check=False)
    if result.returncode != 0:
        print(f"Warning: sbatch failed for {script_path}", file=sys.stderr)
    time.sleep(0.5)
