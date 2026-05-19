"""Shared SLURM job submission utilities."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TIME_LIMITS = {
    "Bayes": "0:30:0",
    "NoisyCounting": "24:0:0",
    "RL": "2:0:0",
    "RL_lambda": "2:0:0",
    "Mean": "0:30:0",
    "ADM": "2:0:0",
    "PearceHall": "2:0:0",
    "NEF_recurrent": "72:0:0",
    "NEF_synaptic": "72:0:0",
    "NEF2d": "72:0:0",
}

DEFAULT_MEM_LIMITS = {
    "Bayes": "8G",
    "NoisyCounting": "16G",
    "RL": "8G",
    "RL_lambda": "8G",
    "Mean": "8G",
    "ADM": "8G",
    "PearceHall": "8G",
    "NEF_recurrent": "32G",
    "NEF_synaptic": "32G",
    "NEF2d": "32G",
}

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
    result = subprocess.run(["sbatch", str(script_path)], check=False)
    if result.returncode != 0:
        print(f"Warning: sbatch failed for {script_path}", file=sys.stderr)
    time.sleep(0.5)
