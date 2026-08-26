"""
Submit, resubmit, or locally run model fitting jobs.

Jobs are enumerated from ``MODEL_PARAMS`` (datasets: carrabin, yoo,
soltani_numbers, soltani_colors); NEF models use the same SLURM templates as
other datasets.

``--datafile`` selects which build of a dataset's human data to fit against
(``data/{dataset}_{datafile}.pkl``) and is recorded in each job dict, so
``run_config.json`` -- and therefore ``fitting.collect`` -- knows which data
version a run folder's outputs belong to.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from fitting.fit import fit
from fitting.model_params import MODEL_PARAMS
from utils.paths import DATA_DIR, RUNS_DIR, data_path, dataset_stem
from utils.slurm import (
    DEFAULT_MEM_LIMITS,
    DEFAULT_TIME_LIMITS,
    make_job_script,
    submit_script,
)


def _make_run_folder() -> Path:
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
            (j["dataset"], j.get("datafile"), j["model_type"], j["pid"])
            for j in prev_jobs
        }
        new_jobs = [
            j
            for j in jobs
            if (j["dataset"], j.get("datafile"), j["model_type"], j["pid"])
            not in existing_keys
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
    k: int,
    optuna_seed: int,
    datafile: str | None = None,
    loss_fn: str = "rmse",
    n_sims: int = 100,
) -> list[dict]:
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
            stem = dataset_stem(ds, datafile)
            pids_all = pd.read_pickle(data_path(f"{stem}.pkl"))["pid"].unique()
            pids = [int(pid)] if pid is not None else [int(p) for p in pids_all]
            model_spec = MODEL_PARAMS[ds].get(mt, {})
            has_params = any(k_ != "fixed" for k_ in model_spec)
            effective_n_trials = n_trials if has_params else 1
            for p in pids:
                jobs.append(
                    {
                        "dataset": ds,
                        "datafile": datafile,
                        "model_type": mt,
                        "pid": p,
                        "n_trials": effective_n_trials,
                        "k": k,
                        "optuna_seed": optuna_seed,
                        # Recorded even at the rmse default, so run_config.json
                        # (and hence resubmit/dedup) always distinguishes an NLL
                        # job from an RMSE job of the same model_type/pid -- the
                        # two are NOT the same fit and must never be treated as
                        # duplicates of each other.
                        "loss_fn": loss_fn,
                        "n_sims": n_sims,
                    }
                )
    return jobs


def _submit_command(
    *,
    script_name: str,
    command: str,
    time_limit: str,
    mem: str,
    dry_run: bool,
) -> None:
    root = str(DATA_DIR.parent)
    jobs_dir = Path(root) / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    script = make_job_script(
        root=root,
        commands=[command],
        time_limit=time_limit,
        mem=mem,
        log_dir=f"{root}/logs",
    )
    script_path = jobs_dir / script_name
    script_path.write_text(script)
    script_path.chmod(0o755)
    submit_script(script_path, dry_run=dry_run)


def _submit_job(
    job: dict,
    run_folder: Path,
    dry_run: bool = False,
) -> None:
    ds = job["dataset"]
    mt = job["model_type"]
    pid = job["pid"]
    n_trials = job["n_trials"]
    k = int(job.get("k", 5))
    seed = job.get("optuna_seed", 42)
    datafile = job.get("datafile")
    loss_fn = job.get("loss_fn", "rmse")
    n_sims = job.get("n_sims", 100)
    stem = dataset_stem(ds, datafile)
    cmd = (
        f"python -m fitting.fit {ds} {mt} {pid} --n_trials {n_trials} "
        f"--k {k} --run_folder {run_folder} --optuna_seed {seed}"
    )
    if datafile:
        cmd += f" --datafile {datafile}"
    if loss_fn != "rmse":
        cmd += f" --loss {loss_fn} --n_sims {n_sims}"
    # file_stem (with the _nll suffix fitting.fit inserts for loss_fn='nll')
    # matters for the job SCRIPT name too, exactly as it does for fit.py's own
    # output filenames -- otherwise an NLL job and an RMSE job for the same
    # model_type/pid would collide on the SAME jobs/*.sh script name.
    file_stem = f"{stem}_nll" if loss_fn == "nll" else stem
    _submit_command(
        script_name=f"{mt}_{file_stem}_{pid}.sh",
        command=cmd,
        time_limit=DEFAULT_TIME_LIMITS.get(mt, "4:0:0"),
        mem=DEFAULT_MEM_LIMITS.get(mt, "8G"),
        dry_run=dry_run,
    )


def _run_local(job: dict, run_folder: Path, dry_run: bool = False) -> None:
    import logging

    logging.basicConfig(level=logging.INFO)
    ds = job["dataset"]
    mt = job["model_type"]
    pid = job["pid"]
    n_trials = job["n_trials"]
    k = int(job.get("k", 5))
    datafile = job.get("datafile")
    loss_fn = job.get("loss_fn", "rmse")
    n_sims = job.get("n_sims", 100)
    stem = dataset_stem(ds, datafile)
    file_stem = f"{stem}_nll" if loss_fn == "nll" else stem

    if dry_run:
        print(f"[dry_run] would run locally: {mt} {file_stem} pid={pid}")
        return

    print(f"Running {mt} {file_stem} pid={pid}...")
    fit(
        ds,
        mt,
        pid,
        n_trials=n_trials,
        k=k,
        run_folder=run_folder,
        optuna_seed=job.get("optuna_seed", 42),
        datafile=datafile,
        loss_fn=loss_fn,
        n_sims=n_sims,
    )


def _jobs_from_config(run_folder: Path) -> list[dict]:
    config_path = run_folder / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No run_config.json in {run_folder}")
    return list(json.loads(config_path.read_text()).get("jobs", []))


def _resubmit(
    *,
    resubmit_type: str,
    run_folder: Path,
    dry_run: bool,
    local: bool = False,
    ensembles: list[str],
    timing: str,
    dt_sample: float,
    dataset: str | None = None,
    model_type: str | None = None,
    pid: int | None = None,
    datafile: str | None = None,
) -> None:
    jobs = _jobs_from_config(run_folder)
    # apply filters
    if dataset is not None:
        jobs = [j for j in jobs if j["dataset"] == dataset]
    if model_type is not None:
        jobs = [j for j in jobs if j["model_type"] == model_type]
    if pid is not None:
        jobs = [j for j in jobs if int(j["pid"]) == int(pid)]
    # Filter on the data version too. run_config.json ACCUMULATES across every
    # submit into this run folder, so one folder legitimately holds jobs for
    # several datafiles (e.g. pilot5 and complete_pairs); without this filter a
    # resubmit scoped to one of them also picks up the others, which is at best
    # redundant work and at worst refits a data version you did not ask about.
    # As with the other filters, omitting --datafile means "do not filter".
    if datafile is not None:
        jobs = [j for j in jobs if j.get("datafile") == datafile]
    missing: list[dict] = []
    for job in jobs:
        ds = job["dataset"]
        mt = job["model_type"]
        pid = int(job["pid"])
        stem = dataset_stem(ds, job.get("datafile"))
        params_path = run_folder / f"{mt}_{stem}_{pid}_params.pkl"

        if resubmit_type == "params":
            if not params_path.exists():
                missing.append(job)
        elif resubmit_type == "responses":
            responses_path = run_folder / f"{mt}_{stem}_{pid}_responses.pkl"
            if params_path.exists() and not responses_path.exists():
                missing.append(job)
        elif resubmit_type == "activities":
            if not params_path.exists():
                continue
            # Activities come from models.NEF's own build_network/_pretrain, so
            # only NEF and its variants (NEF_recurrent, NEF_synaptic -- passed
            # via --extra_models; see utils.run_params' nef_type branch) can
            # produce them. startswith rather than an explicit tuple: the tuple
            # here read ("NEF", "NEF") -- a duplicated element, so it only ever
            # matched plain "NEF" and silently skipped every variant.
            if not mt.startswith("NEF"):
                continue
            if timing == "once_per_dt" and ds != "carrabin":
                continue  # windowed once_per_dt activities only implemented for carrabin
            out_dir = run_folder
            ens_missing = False
            for ens in ensembles:
                if timing == "once_per_dt":
                    p = out_dir / f"activities_windowed_{ens}_{stem}_{pid}.npz"
                else:
                    p = out_dir / f"activities_{ens}_{stem}_{pid}.pkl"
                if not p.exists():
                    ens_missing = True
                    break
            if ens_missing:
                missing.append(job)

    if not missing:
        print(f"No missing jobs found for type={resubmit_type}.")
        return

    print(f"Found {len(missing)} missing jobs for type={resubmit_type}:")
    for job in missing:
        print(f"  {job['model_type']} "
              f"{dataset_stem(job['dataset'], job.get('datafile'))} "
              f"pid={job['pid']}")

    for job in missing:
        ds = job["dataset"]
        mt = job["model_type"]
        pid = int(job["pid"])
        stem = dataset_stem(ds, job.get("datafile"))
        if resubmit_type == "params":
            if local:
                _run_local(job, run_folder, dry_run=dry_run)
            else:
                _submit_job(job, run_folder, dry_run=dry_run)
        elif resubmit_type == "responses":
            if local:
                if not dry_run:
                    from utils.save_responses import save as save_responses

                    save_responses(int(pid), ds, run_folder, mt,
                                   job.get("datafile"))
            else:
                cmd = f"python -m utils.save_responses {ds} {mt} {pid} {run_folder}"
                if job.get("datafile"):
                    cmd += f" --datafile {job['datafile']}"
                _submit_command(
                    script_name=f"responses_{mt}_{stem}_{pid}.sh",
                    command=cmd,
                    time_limit=DEFAULT_TIME_LIMITS.get(mt, "4:0:0"),
                    mem=DEFAULT_MEM_LIMITS.get(mt, "8G"),
                    dry_run=dry_run,
                )
        elif resubmit_type == "activities":
            if local:
                if not dry_run:
                    from utils.save_activities import run as run_activities

                    run_activities(
                        int(pid), ds, ensembles, str(run_folder), timing,
                        dt_sample, mt, job.get("datafile")
                    )
            else:
                ensembles_str = ",".join(ensembles)
                cmd = (
                    f"python -m utils.save_activities {ds} {mt} {pid} {run_folder} "
                    f"{ensembles_str} {timing} {dt_sample}"
                )
                if job.get("datafile"):
                    cmd += f" --datafile {job['datafile']}"
                _submit_command(
                    script_name=f"activities_{mt}_{stem}_{pid}.sh",
                    command=cmd,
                    time_limit=DEFAULT_TIME_LIMITS.get(mt, "4:0:0"),
                    mem=DEFAULT_MEM_LIMITS.get(mt, "8G"),
                    dry_run=dry_run,
                )


def main() -> None:
    parser = argparse.ArgumentParser(prog="fitting.submit")
    parser.add_argument(
        "--resubmit",
        type=str,
        choices=["params", "responses", "activities"],
        default=None,
    )

    parser.add_argument("target", nargs="?", default="all")
    parser.add_argument("model_type", nargs="?", default=None)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--optuna_seed", type=int, default=42)
    parser.add_argument(
        "--loss", dest="loss", choices=("rmse", "nll"), default="rmse",
        help="Loss fitting.fit minimises -- see that module's docstring. 'nll' "
             "only for models in models.math_models._STOCHASTIC_ENSEMBLE_MODELS "
             "or a '<model>_resp_noise' name; fitting.fit itself validates this "
             "before creating an Optuna study, so an invalid combination fails "
             "per-job rather than blocking the whole submit call.",
    )
    parser.add_argument(
        "--n_sims", type=int, default=100,
        help="Ensemble size per NLL evaluation (ignored for --loss rmse). See "
             "fitting.fit's own --n_sims help for the stability/cost tradeoff.",
    )
    parser.add_argument("--run_folder", type=str, default=None)
    parser.add_argument(
        "--datafile",
        type=str,
        default=None,
        help="Data-version suffix: fit against data/{dataset}_{datafile}.pkl "
             "and tag every output with it. Omit for data/{dataset}.pkl.",
    )
    parser.add_argument("--ensembles", nargs="+", default=["error"])
    parser.add_argument("--timing", type=str, default="once_per_obs")
    parser.add_argument("--dt_sample", type=float, default=0.01)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.resubmit is not None:
        if args.run_folder is None:
            raise ValueError("--resubmit requires --run_folder")
        _resubmit(
            resubmit_type=args.resubmit,
            run_folder=RUNS_DIR / args.run_folder,
            dry_run=args.dry_run,
            local=args.local,
            ensembles=args.ensembles,
            timing=args.timing,
            dt_sample=args.dt_sample,
            dataset=args.target if args.target != "all" else None,
            model_type=args.model_type,
            pid=args.pid,
            datafile=args.datafile,
        )
        return

    jobs = _resolve_jobs(
        args.target,
        args.model_type,
        args.pid,
        args.n_trials,
        args.k,
        args.optuna_seed,
        datafile=args.datafile,
        loss_fn=args.loss,
        n_sims=args.n_sims,
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
            _submit_job(
                job,
                run_folder,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
