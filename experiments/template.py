#!/usr/bin/env python3
"""
Template for NEF experiment scripts.

Usage:
    python experiments/template.py --pid 1 --local   # run locally
    python experiments/template.py                    # submit all pids
    python experiments/template.py --collect Apr21_X  # collect results
    python experiments/template.py --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.NEF import PARAM_DEFAULTS, _pretrain, build_network
from utils.paths import RUNS_DIR, data_path
from utils.slurm import DEFAULT_TIME_LIMITS, make_job_script, submit_script

EXPERIMENT_NAME = "template"
DATASET = "carrabin"
MODEL_TYPE = "NEF_recurrent"
RUN_FOLDER = "MSE"


def simulate_experiment(pid: int, params: dict) -> pd.DataFrame:
    """Run experiment for one pid. Add probes/analysis here."""
    raise NotImplementedError


def run_local(pid: int) -> None:
    params = pd.read_pickle(
        RUNS_DIR / RUN_FOLDER / f"{MODEL_TYPE}_{DATASET}_{pid}_params.pkl"
    ).loc[0].to_dict()
    params = {**PARAM_DEFAULTS, **params}
    out_dir = data_path("experiments") / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    df = simulate_experiment(pid, params)
    df.to_pickle(out_dir / f"{EXPERIMENT_NAME}_{DATASET}_{pid}.pkl")
    print(f"Saved {out_dir}/{EXPERIMENT_NAME}_{DATASET}_{pid}.pkl")


def submit(pids: list[int], dry_run: bool = False) -> None:
    from utils.paths import DATA_DIR

    root = str(DATA_DIR.parent)
    logs_dir = Path(root) / "logs"
    logs_dir.mkdir(exist_ok=True)
    jobs_dir = Path(root) / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    for pid in pids:
        cmd = f"python experiments/{EXPERIMENT_NAME}.py --pid {pid} --local"
        script = make_job_script(
            root=root,
            commands=[cmd],
            time_limit=DEFAULT_TIME_LIMITS.get(MODEL_TYPE, "4:0:0"),
        )
        script_path = jobs_dir / f"{EXPERIMENT_NAME}_{DATASET}_{pid}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        submit_script(script_path, dry_run=dry_run)


def collect(out_dir: Path) -> None:
    import glob

    dfs = [
        pd.read_pickle(f)
        for f in sorted(glob.glob(str(out_dir / f"{EXPERIMENT_NAME}_{DATASET}_*.pkl")))
    ]
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_pickle(out_dir / f"{EXPERIMENT_NAME}_{DATASET}.pkl")
        print(
            f"Collected {len(dfs)} files -> "
            f"{out_dir}/{EXPERIMENT_NAME}_{DATASET}.pkl"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--local", action="store_true")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.local and args.pid is not None:
        run_local(args.pid)
    elif args.collect:
        out_dir = data_path("experiments") / EXPERIMENT_NAME
        collect(out_dir)
    else:
        human = pd.read_pickle(data_path(f"{DATASET}.pkl"))
        pids = [int(pid) for pid in human["pid"].unique()]
        if args.pid is not None:
            pids = [args.pid]
        submit(pids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
