"""
Single entry point for submitting, resubmitting, and collecting
model fitting jobs.

Usage::

    # Submit a new run
    python -m jobs.run all [--n_trials N] [--n_runs N] [--k K] [--loss_type L]
    python -m jobs.run {dataset} {model_type} [--n_trials N] ...
    python -m jobs.run {dataset} {model_type} {pid} [--n_trials N] ...

    # Resubmit missing jobs from an existing run
    python -m jobs.run --resubmit {run_folder}

    # Collect results from a run folder
    python -m jobs.run --collect {run_folder}

    # Run locally without SLURM
    python -m jobs.run all --local [--n_trials N] [--k K] ...

    # Dry run (print without executing)
    python -m jobs.run all --dry_run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# Allow running both as module (`python -m jobs.run`) and script path
# (`python jobs/run.py`) by ensuring project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.fit import DEFAULT_LOSS, fit
from fitting.param_ranges import MODEL_PARAMS
from models import recurrent
from models.math_models import run as model_run
from utils.paths import DATA_DIR, RUNS_DIR, data_path

TIME_LIMITS = {
    "Bayes":         "0:30:0",
    "NoisyCounting": "24:0:0",
    "RL":            "2:0:0",
    "DeGroot":       "1:0:0",
    "Mean":          "0:30:0",
    "ADM":           "2:0:0",
    "recurrent":     "48:0:0",
    "synaptic":      "48:0:0",
    "NEF_recurrent": "48:0:0",
}

PROTECTED = frozenset(
    {
        "carrabin.pkl",
        "jiang.pkl",
        "yoo.pkl",
        "jiang_networks.npy",
    }
)


def _make_run_folder() -> Path:
    """Create a new timestamped run folder under data/runs/."""
    now = datetime.now()
    month = now.strftime("%b")  # e.g. "Apr"
    day = now.strftime("%d").lstrip("0")  # e.g. "12"
    hour = now.hour
    minute = now.strftime("%M")
    ampm = "am" if hour < 12 else "pm"
    hour12 = hour % 12 or 12
    name = f"{month}{day}_{hour12}{minute}{ampm}"
    folder = RUNS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_run_config(
    run_folder: Path,
    jobs: list[dict],
) -> None:
    """Write run_config.json to the run folder."""
    import subprocess as sp

    try:
        commit = (
            sp.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=sp.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"
    config = {
        "run_folder": run_folder.name,
        "timestamp": run_folder.name,
        "git_commit": commit,
        "jobs": jobs,
    }
    (run_folder / "run_config.json").write_text(json.dumps(config, indent=2))


def _resolve_jobs(
    dataset: str | None,
    model_type: str | None,
    pid: int | None,
    n_trials: int,
    n_runs: int,
    k: int,
    loss_type: str | None,
) -> list[dict]:
    """
    Build the list of job dicts for a run.
    Each dict has: dataset, model_type, pid, n_trials, n_runs, k, loss_type.
    """
    jobs = []
    datasets = (
        list(MODEL_PARAMS.keys())
        if dataset == "all" or dataset is None
        else [dataset]
    )
    for ds in datasets:
        models = (
            list(MODEL_PARAMS[ds].keys())
            if model_type is None or model_type == "all"
            else [model_type]
        )
        for mt in models:
            lt = loss_type if loss_type is not None else DEFAULT_LOSS.get(ds, "mse")
            pids_all = pd.read_pickle(data_path(f"{ds}.pkl"))["pid"].unique()
            pids = [int(pid)] if pid is not None else [int(p) for p in pids_all]
            for p in pids:
                jobs.append(
                    {
                        "dataset": ds,
                        "model_type": mt,
                        "pid": p,
                        "n_trials": n_trials,
                        "n_runs": n_runs,
                        "k": k,
                        "loss_type": lt,
                    }
                )
    return jobs


def _submit_job(job: dict, run_folder: Path, dry_run: bool = False) -> None:
    """Generate and submit one SLURM job script."""
    ds = job["dataset"]
    mt = job["model_type"]
    pid = job["pid"]
    n_trials = job["n_trials"]
    n_runs = job["n_runs"]
    k = int(job.get("k", 5))
    lt = job["loss_type"]
    root = str(DATA_DIR.parent)
    time_limit = TIME_LIMITS.get(mt, "4:0:0")

    fit_cmd = (
        f"python -m fitting.fit {ds} {mt} {pid} {n_trials} "
        f"{lt} {n_runs} {k} {run_folder}"
    )
    rerun_cmd = f"python -m jobs.run --rerun {ds} {mt} {pid} {run_folder}"

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
    jobs_dir = DATA_DIR.parent / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    (jobs_dir / "logs").mkdir(exist_ok=True)
    script_path = jobs_dir / f"{mt}_{ds}_{pid}.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    if dry_run:
        print(f"[dry_run] would submit: {script_path}")
        return
    result = subprocess.run(["sbatch", str(script_path)], check=False)
    if result.returncode != 0:
        print(
            f"Warning: sbatch failed for {mt} {ds} pid={pid}",
            file=sys.stderr,
        )
    time.sleep(0.5)


def _run_local(job: dict, run_folder: Path, dry_run: bool = False) -> None:
    """Run one job locally without SLURM."""
    ds = job["dataset"]
    mt = job["model_type"]
    pid = job["pid"]
    n_trials = job["n_trials"]
    n_runs = job["n_runs"]
    k = int(job.get("k", 5))
    lt = job["loss_type"]

    if dry_run:
        print(f"[dry_run] would run locally: {mt} {ds} pid={pid}")
        return

    print(f"Running {mt} {ds} pid={pid}...")
    fit(
        ds,
        mt,
        pid,
        n_trials=n_trials,
        k=k,
        loss_type=lt,
        n_runs=n_runs,
        run_folder=run_folder,
    )
    _rerun_single(ds, mt, pid, run_folder)


def _rerun_single(
    dataset: str, model_type: str, pid: int, run_folder: Path
) -> None:
    """Load best params from run_folder and generate full model responses."""
    params_path = run_folder / f"{model_type}_{dataset}_{pid}_params.pkl"
    if not params_path.exists():
        print(f"Warning: params not found for {model_type} {dataset} pid={pid}")
        return
    params = pd.read_pickle(params_path).loc[0].to_dict()
    model_spec = MODEL_PARAMS.get(dataset, {}).get(model_type, {})
    fixed = model_spec.get("fixed", {})
    params = {**fixed, **params}
    params["seed"] = abs(hash((int(params["pid"]), 0))) % (2**31)
    if model_type == "NEF_recurrent":
        df = recurrent.run(params)
    else:
        df = model_run(params)
    out_path = run_folder / f"{model_type}_{dataset}_{pid}_responses.pkl"
    df.to_pickle(out_path)


def _collect(run_folder: Path) -> None:
    """
    Aggregate per-participant files in run_folder into combined files.
    Reads run_config.json to know which jobs were run.
    """
    config_path = run_folder / "run_config.json"
    if not config_path.exists():
        print(f"No run_config.json found in {run_folder}", file=sys.stderr)
        return
    config = json.loads(config_path.read_text())
    jobs = config["jobs"]

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for job in jobs:
        groups[(job["dataset"], job["model_type"])].append(job["pid"])

    for (ds, mt), pids in groups.items():
        responses_dfs, params_dfs, perf_dfs = [], [], []
        for pid in pids:
            pid = int(pid)
            rp = run_folder / f"{mt}_{ds}_{pid}_responses.pkl"
            pp = run_folder / f"{mt}_{ds}_{pid}_params.pkl"
            fp = run_folder / f"{mt}_{ds}_{pid}_performance.pkl"
            if not all(p.exists() for p in [rp, pp, fp]):
                print(f"Warning: missing files for {mt} {ds} pid={pid}, skipping")
                continue
            responses_dfs.append(pd.read_pickle(rp))
            params_dfs.append(pd.read_pickle(pp))
            perf_dfs.append(pd.read_pickle(fp))

        def _save(dfs, name):
            if not dfs:
                return
            df = pd.concat(dfs, ignore_index=True)
            df.to_pickle(run_folder / name)
            print(f"  Saved {name}: {df.shape}")

        print(f"Collecting {mt} {ds}...")
        _save(responses_dfs, f"{mt}_{ds}_responses.pkl")
        _save(params_dfs, f"{mt}_{ds}_params.pkl")
        _save(perf_dfs, f"{mt}_{ds}_performance.pkl")


def _resubmit(run_folder: Path, dry_run: bool = False) -> None:
    """
    Read run_config.json, find missing participants, and resubmit their jobs.
    """
    config_path = run_folder / "run_config.json"
    if not config_path.exists():
        print(f"No run_config.json in {run_folder}", file=sys.stderr)
        return
    config = json.loads(config_path.read_text())
    missing = []
    for job in config["jobs"]:
        params_path = (
            run_folder
            / f"{job['model_type']}_{job['dataset']}_{job['pid']}_params.pkl"
        )
        if not params_path.exists():
            missing.append(job)

    if not missing:
        print("No missing jobs found.")
        return
    print(f"Found {len(missing)} missing jobs:")
    for job in missing:
        print(f"  {job['model_type']} {job['dataset']} pid={job['pid']}")

    if not dry_run:
        for job in missing:
            _submit_job(job, run_folder, dry_run=False)
        print(f"Resubmitted {len(missing)} jobs.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="jobs.run")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resubmit", metavar="RUN_FOLDER")
    mode.add_argument("--collect", metavar="RUN_FOLDER")
    mode.add_argument(
        "--rerun",
        nargs=4,
        metavar=("DATASET", "MODEL_TYPE", "PID", "RUN_FOLDER"),
    )

    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="'all', dataset, or 'dataset model_type'",
    )
    parser.add_argument("model_type", nargs="?", default=None)
    parser.add_argument("pid", nargs="?", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    if args.resubmit is not None:
        _resubmit(RUNS_DIR / args.resubmit, dry_run=args.dry_run)
    elif args.collect is not None:
        _collect(RUNS_DIR / args.collect)
    elif args.rerun is not None:
        dataset, model_type, pid, run_folder_str = args.rerun
        _rerun_single(dataset, model_type, int(pid), Path(run_folder_str))
    else:
        dataset = args.target
        jobs = _resolve_jobs(
            dataset,
            args.model_type,
            args.pid,
            args.n_trials,
            args.n_runs,
            args.k,
            args.loss_type,
        )
        run_folder = _make_run_folder()
        _write_run_config(run_folder, jobs)
        for job in jobs:
            if args.local:
                _run_local(job, run_folder, dry_run=args.dry_run)
            else:
                _submit_job(job, run_folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
