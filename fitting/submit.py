"""
Submit, resubmit, or locally run model fitting jobs.

Usage::

    # Submit all models/datasets in one command to a named folder
    python -m fitting.submit all --n_trials 100 --loss_type joint --run_folder joint_fits

    # Add more jobs to the same folder later
    python -m fitting.submit carrabin NoisyCounting --n_trials 200 --run_folder joint_fits

    # submit fitting jobs (new timestamped folder if --run_folder omitted)
    python -m fitting.submit carrabin NEF_recurrent --n_trials 300
    python -m fitting.submit carrabin NEF_recurrent --n_trials 300 --pid 1
    python -m fitting.submit carrabin NEF_recurrent --n_trials 300 --pid 1 --local

    # rerun best-fit params to generate responses
    python -m fitting.submit carrabin NEF_recurrent --rerun Apr21_1200pm
    python -m fitting.submit carrabin NEF_recurrent --rerun Apr21_1200pm --pid 1

    # resubmit missing jobs
    python -m fitting.submit --resubmit joint_fits

    # collect all results
    python -m fitting.collect joint_fits

    # dry run
    python -m fitting.submit --dry_run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from fitting.fit import DEFAULT_LOSS, fit
from fitting.param_ranges import MODEL_PARAMS
from models import NEF
from models.math_models import run as model_run
from utils.paths import DATA_DIR, RUNS_DIR, data_path
from utils.slurm import DEFAULT_TIME_LIMITS, make_job_script, submit_script


def _make_run_folder() -> Path:
    """Create a new timestamped run folder under data/runs/."""
    now = datetime.now()
    month = now.strftime("%b")
    day = now.strftime("%d").lstrip("0")
    hour = now.hour
    minute = now.strftime("%M")
    ampm = "am" if hour < 12 else "pm"
    hour12 = hour % 12 or 12
    name = f"{month}{day}_{hour12}{minute}{ampm}"
    folder = RUNS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_run_config(run_folder: Path, jobs: list[dict]) -> None:
    """Write or merge run_config.json in the run folder."""
    import subprocess as sp

    try:
        commit = (
            sp.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=sp.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    config_path = run_folder / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        prev_jobs = list(existing.get("jobs", []))
        existing_keys = {
            (j["dataset"], j["model_type"], j["pid"]) for j in prev_jobs
        }
        new_jobs = [
            j
            for j in jobs
            if (j["dataset"], j["model_type"], j["pid"]) not in existing_keys
        ]
        prev_jobs.extend(new_jobs)
        existing["jobs"] = prev_jobs
        config_path.write_text(json.dumps(existing, indent=2))
    else:
        config = {
            "run_folder": run_folder.name,
            "timestamp": run_folder.name,
            "git_commit": commit,
            "jobs": jobs,
        }
        config_path.write_text(json.dumps(config, indent=2))


def _resolve_jobs(
    dataset: str | None,
    model_type: str | None,
    pid: int | None,
    n_trials: int,
    n_runs: int,
    k: int,
    loss_type: str | None,
    optuna_seed: int,
) -> list[dict]:
    """Build the list of job dicts for a run."""
    jobs = []
    datasets = (
        list(MODEL_PARAMS.keys()) if dataset == "all" or dataset is None else [dataset]
    )
    for ds in datasets:
        models = (
            list(MODEL_PARAMS[ds].keys())
            if model_type is None or model_type == "all"
            else [model_type]
        )
        for mt in models:
            lt = loss_type if loss_type is not None else DEFAULT_LOSS.get(ds, "response")
            pids_all = pd.read_pickle(data_path(f"{ds}.pkl"))["pid"].unique()
            pids = [int(pid)] if pid is not None else [int(p) for p in pids_all]
            model_spec = MODEL_PARAMS[ds].get(mt, {})
            has_params = any(k != "fixed" for k in model_spec)
            effective_n_trials = n_trials if has_params else 1
            for p in pids:
                jobs.append(
                    {
                        "dataset": ds,
                        "model_type": mt,
                        "pid": p,
                        "n_trials": effective_n_trials,
                        "n_runs": n_runs,
                        "k": k,
                        "loss_type": lt,
                        "optuna_seed": optuna_seed,
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
    time_limit = DEFAULT_TIME_LIMITS.get(mt, "4:0:0")

    fit_cmd = (
        f"python -m fitting.fit {ds} {mt} {pid} {n_trials} "
        f"{lt} {n_runs} {k} {run_folder} {job.get('optuna_seed', 42)}"
    )
    # TODO: prompt text requested fitting.collect --rerun, but collect is
    # specified as aggregation-only. Keep rerun in fitting.submit.
    rerun_cmd = f"python -m fitting.submit {ds} {mt} --rerun {run_folder} --pid {pid}"
    logs_dir = Path(root) / "logs"
    logs_dir.mkdir(exist_ok=True)
    jobs_dir = Path(root) / "jobs"
    jobs_dir.mkdir(exist_ok=True)

    script = make_job_script(
        root=root,
        commands=[fit_cmd, rerun_cmd],
        time_limit=time_limit,
        log_dir=f"{root}/logs",
    )
    script_path = jobs_dir / f"{mt}_{ds}_{pid}.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    submit_script(script_path, dry_run=dry_run)


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
        optuna_seed=job.get("optuna_seed", 42),
    )
    _rerun_single(ds, mt, pid, run_folder)


def _rerun_single(dataset: str, model_type: str, pid: int, run_folder: Path) -> None:
    """Load best params from run_folder and generate full model responses."""
    params_path = run_folder / f"{model_type}_{dataset}_{pid}_params.pkl"
    if not params_path.exists():
        print(f"Warning: params not found for {model_type} {dataset} pid={pid}")
        return
    params = pd.read_pickle(params_path).loc[0].to_dict()
    model_spec = MODEL_PARAMS.get(dataset, {}).get(model_type, {})
    fixed = model_spec.get("fixed", {})
    params = {**fixed, **params}
    if model_type in ("NEF_recurrent", "NEF_synaptic"):
        df = NEF.run(params)
    else:
        df = model_run(params)
    out_path = run_folder / f"{model_type}_{dataset}_{pid}_responses.pkl"
    df.to_pickle(out_path)


def _resubmit(run_folder: Path, dry_run: bool = False) -> None:
    """Read run_config.json, find missing participants, and resubmit their jobs."""
    config_path = run_folder / "run_config.json"
    if not config_path.exists():
        print(f"No run_config.json in {run_folder}")
        return
    config = json.loads(config_path.read_text())
    missing = []
    for job in config["jobs"]:
        mt = job["model_type"]
        ds = job["dataset"]
        pid = job["pid"]
        params_path = run_folder / f"{mt}_{ds}_{pid}_params.pkl"
        responses_path = run_folder / f"{mt}_{ds}_{pid}_responses.pkl"
        perf_path = run_folder / f"{mt}_{ds}_{pid}_performance.pkl"
        if not all(p.exists() for p in [params_path, responses_path, perf_path]):
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
    parser = argparse.ArgumentParser(prog="fitting.submit")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resubmit", metavar="RUN_FOLDER")
    mode.add_argument("--rerun", metavar="RUN_FOLDER")

    parser.add_argument("target", nargs="?", default="all")
    parser.add_argument("model_type", nargs="?", default=None)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--optuna_seed", type=int, default=42)
    parser.add_argument(
        "--run_folder",
        type=str,
        default=None,
        help="Use existing run folder by name instead of creating a new timestamped one.",
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    if args.resubmit is not None:
        _resubmit(RUNS_DIR / args.resubmit, dry_run=args.dry_run)
        return

    if args.rerun is not None:
        if args.target in (None, "all") or args.model_type is None:
            raise ValueError(
                "--rerun requires dataset and model_type, e.g. "
                "`python -m fitting.submit carrabin NEF_recurrent --rerun Apr21_1200pm`"
            )
        run_folder = RUNS_DIR / args.rerun
        pids = [args.pid] if args.pid is not None else [
            int(p)
            for p in pd.read_pickle(data_path(f"{args.target}.pkl"))["pid"].unique()
        ]
        for pid in pids:
            _rerun_single(args.target, args.model_type, int(pid), run_folder)
        return

    dataset = args.target
    jobs = _resolve_jobs(
        dataset,
        args.model_type,
        args.pid,
        args.n_trials,
        args.n_runs,
        args.k,
        args.loss_type,
        args.optuna_seed,
    )
    if args.run_folder is not None:
        run_folder = RUNS_DIR / args.run_folder
        run_folder.mkdir(parents=True, exist_ok=True)
    else:
        run_folder = _make_run_folder()
    _write_run_config(run_folder, jobs)
    for job in jobs:
        if args.local:
            _run_local(job, run_folder, dry_run=args.dry_run)
        else:
            _submit_job(job, run_folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
