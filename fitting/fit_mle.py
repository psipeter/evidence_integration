#!/usr/bin/env python3
"""
MLE fitting via shared simulation database and Optuna TPE.

Each process fits one pid. All processes share:
  - A simulation database: {model}/params_{hash}.pkl storing
    (params, responses[32 seqs × n_sims × n_obs])
  - An Optuna SQLite storage: one study per pid, all in one file

Loop per process:
  1. Scan database for any entries not yet in this pid's Optuna study
     → compute sim_db_loss(params, pid) → inject as completed trial
  2. Ask this pid's study for next params (TPE)
  3. If params already in database: skip to 1 (free reuse)
  4. Simulate params → save to database
  5. Cross-report: compute sim_db_loss(params, pid=k) for all pids
     → inject into each pid's study
  Repeat n_fits times.

Usage (NoisyCounting, pid 1):
    python -m fitting.fit_mle carrabin NoisyCounting 1 \\
        --n_fits 50 --n_sims 100 \\
        --db_folder data/sim_db \\
        --optuna_db data/optuna/NoisyCounting_carrabin.db \\
        --run_folder carrabin

SLURM: one job per pid, all pointing to the same db_folder and optuna_db.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import create_trial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.losses import compute_sim_db_loss
from fitting.model_params import MODEL_PARAMS, MLE_PARAMS
from scripts.build_sim_db import params_hash, simulate_param_point
from utils.paths import data_path, resolve_run_folder

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s pid=%(pid)s %(message)s",
    datefmt="%H:%M:%S",
)


def _log(msg: str, pid: int) -> None:
    logging.info(msg, extra={"pid": pid})


# ── Parameter space ───────────────────────────────────────────────────────────

def _get_distributions(model_type: str, dataset: str) -> dict:
    """Return Optuna distributions for the free parameters of a model."""
    params_source = MLE_PARAMS if dataset in MLE_PARAMS and model_type in MLE_PARAMS.get(dataset, {}) else MODEL_PARAMS
    spec = params_source[dataset][model_type]
    dists = {}
    for name, bounds in spec.items():
        if name == "fixed":
            continue
        low, high, step = bounds
        if step is None or float(step) == 0.0:
            dists[name] = FloatDistribution(float(low), float(high))
        elif float(step).is_integer() and float(low).is_integer():
            dists[name] = IntDistribution(int(low), int(high), step=int(step))
        else:
            dists[name] = FloatDistribution(float(low), float(high), step=float(step))
    return dists


def _full_params(model_type: str, dataset: str, free: dict, pid: int) -> dict:
    """Merge free params with fixed params."""
    params_source = MLE_PARAMS if dataset in MLE_PARAMS and model_type in MLE_PARAMS.get(dataset, {}) else MODEL_PARAMS
    fixed = params_source[dataset][model_type].get("fixed", {})
    return {**fixed, **free,
            "model_type": model_type, "dataset": dataset, "pid": pid}


# ── Database helpers ──────────────────────────────────────────────────────────

def _db_path(db_dir: Path, model_type: str, ph: str) -> Path:
    return db_dir / model_type / f"{model_type}_{ph}.pkl"


def _list_db_entries(db_dir: Path, model_type: str) -> list[Path]:
    """Return all simulation database files for this model."""
    model_dir = db_dir / model_type
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob(f"{model_type}_*.pkl"))


def _load_db_entry(path: Path) -> dict:
    return pd.read_pickle(path)


def _simulate_and_save(
    model_type: str,
    params: dict,
    db_dir: Path,
    n_sims: int,
    run_folder: str,
) -> Path:
    """Simulate params and save to shared database using atomic rename.

    Uses write-to-tempfile + atomic rename instead of file locking.
    Atomic rename is safe on NFS and avoids stale locks if a job dies.
    If two processes simulate the same params simultaneously, one will
    overwrite the other's file — this is safe since both produce the
    same params hash and the content is statistically equivalent.
    """
    ph        = params_hash(model_type, params)
    model_dir = db_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path  = model_dir / f"{model_type}_{ph}.pkl"

    if out_path.exists():
        return out_path

    # Simulate into a temp file in the same directory (same filesystem)
    # then atomically rename into place.
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        dir=model_dir, suffix=".pkl.tmp", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        simulate_param_point(
            model_type=model_type,
            params=params,
            n_sims=n_sims,
            db_dir=db_dir,
            run_folder=run_folder,
            overwrite=False,
            out_path_override=tmp_path,  # write to temp first
        )
        # Atomic rename — safe even if out_path now exists (other process won race)
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return out_path


# ── Loss evaluation ───────────────────────────────────────────────────────────

def _eval_loss(
    entry: dict,
    human_pids: dict[int, pd.DataFrame],
    model_type: str,
    db_dir: Path,
) -> dict[int, float]:
    """Return {pid: loss} for all pids given one database entry."""
    params = entry["params"]
    losses = {}
    for pid, human_pid in human_pids.items():
        try:
            loss = compute_sim_db_loss(
                model_type=model_type,
                params=params,
                human_pid=human_pid,
                db_dir=db_dir,
            )
            losses[pid] = loss
        except Exception as e:
            logging.warning(f"Loss eval failed pid={pid}: {e}")
    return losses


# ── Main fitting loop ─────────────────────────────────────────────────────────

def fit_pid(
    dataset: str,
    model_type: str,
    target_pid: int,
    n_fits: int = 50,
    n_sims: int = 100,
    db_folder: str = "data/sim_db",
    optuna_db: str = "data/optuna/carrabin.db",
    run_folder: str = "carrabin",
    optuna_seed: int = 42,
) -> None:
    db_dir      = Path(db_folder)
    # Use absolute path for SQLite
    _abs_db    = Path(optuna_db).resolve()
    optuna_url = f"sqlite:///{_abs_db}"
    Path(optuna_db).parent.mkdir(parents=True, exist_ok=True)
    Path(optuna_db).touch(exist_ok=True)

    # Brief random sleep to stagger concurrent job starts and reduce
    # SQLite schema-creation race condition
    import random
    time.sleep(random.uniform(0, 5))

    human_df    = pd.read_pickle(data_path(f"{dataset}.pkl"))
    all_pids    = sorted(human_df["pid"].unique())
    human_pids  = {p: human_df[human_df["pid"] == p].copy() for p in all_pids}

    dists       = _get_distributions(model_type, dataset)

    # Create/load studies for all pids (needed for cross-reporting)
    studies: dict[int, optuna.Study] = {}
    for pid in all_pids:
        for attempt in range(10):
            try:
                studies[pid] = optuna.create_study(
                    study_name=f"{model_type}_{dataset}_pid{pid}",
                    storage=optuna_url,
                    direction="minimize",
                    load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(
                        seed=optuna_seed + pid,
                        n_startup_trials=20,
                    ),
                )
                break
            except Exception as e:
                if attempt < 9:
                    time.sleep(1 + attempt)
                else:
                    raise

    target_study  = studies[target_pid]
    _log(f"Starting fit: model={model_type} dataset={dataset} "
         f"n_fits={n_fits} n_sims={n_sims}", target_pid)

    # Track which database hashes have already been injected into each study
    injected: dict[int, set[str]] = {pid: set() for pid in all_pids}
    # Pre-populate from existing study trials
    for pid, study in studies.items():
        for t in study.trials:
            if t.system_attrs.get("db_hash"):
                injected[pid].add(t.system_attrs["db_hash"])

    for fit_idx in range(n_fits):
        t0 = time.time()

        # ── Step 1: scan database → inject new entries into all pid studies ──
        # This is the cross-pid sharing mechanism. Each process picks up
        # simulations from all other concurrent processes automatically.
        for db_path in _list_db_entries(db_dir, model_type):
            entry = _load_db_entry(db_path)
            ph    = params_hash(model_type, entry["params"])
            pids_needing = [p for p in all_pids if ph not in injected[p]]
            if not pids_needing:
                continue
            losses = _eval_loss(
                entry, {p: human_pids[p] for p in pids_needing},
                model_type, db_dir,
            )
            for pid, loss in losses.items():
                free_e = {k: v for k, v in entry["params"].items() if k in dists}
                studies[pid].add_trial(create_trial(
                    params=free_e, distributions=dists, value=loss,
                    system_attrs={"db_hash": ph},
                ))
                injected[pid].add(ph)

        n_trials = len(target_study.trials)
        _log(f"fit {fit_idx+1}/{n_fits}: {n_trials} trials in study", target_pid)

        # ── Step 2: ask TPE for next params to try ───────────────────────────
        trial  = target_study.ask(fixed_distributions=dists)
        free   = trial.params
        params = _full_params(model_type, dataset, free, target_pid)
        ph     = params_hash(model_type, params)

        # ── Step 3: cache hit — params already in database, skip simulation ──
        db_path_new = _db_path(db_dir, model_type, ph)
        if db_path_new.exists():
            _log(f"  cache hit hash={ph[:8]} — no simulation needed", target_pid)
            loss = compute_sim_db_loss(model_type, params,
                                       human_pids[target_pid], db_dir)
            target_study.tell(trial, loss)
            continue   # next iteration step 1 will inject into all other studies

        # ── Step 4: simulate n_sims times, save to shared database ───────────
        _log(f"  simulating hash={ph[:8]} params={free}", target_pid)
        _simulate_and_save(model_type, params, db_dir, n_sims, run_folder)
        elapsed = time.time() - t0

        # ── Step 5: evaluate loss for this pid, report to its study ──────────
        # Other pids will pick this entry up at their next step 1 — no
        # explicit cross-reporting needed.
        loss = compute_sim_db_loss(model_type, params,
                                   human_pids[target_pid], db_dir)
        target_study.tell(trial, loss)
        injected[target_pid].add(ph)
        _log(f"  loss={loss:.4f} ({elapsed:.0f}s)", target_pid)

    # ── Save best params ──────────────────────────────────────────────────────
    best       = target_study.best_trial
    best_free  = best.params
    best_full  = _full_params(model_type, dataset, best_free, target_pid)
    out_dir    = resolve_run_folder(run_folder)

    pd.to_pickle(
        pd.DataFrame([{**best_full, "mle_loss": best.value}]),
        out_dir / f"{model_type}_{dataset}_{target_pid}_params_mle.pkl",
    )
    pd.to_pickle(
        pd.DataFrame([{"pid": target_pid, "mle_loss": best.value,
                       "n_trials": len(target_study.trials)}]),
        out_dir / f"{model_type}_{dataset}_{target_pid}_performance_mle.pkl",
    )
    _log(f"Done. best_loss={best.value:.4f} params={best_free}", target_pid)


# ── CLI ───────────────────────────────────────────────────────────────────────

def init_db(
    dataset: str,
    model_type: str,
    optuna_db: str,
    optuna_seed: int = 42,
) -> None:
    """Pre-initialise all pid studies in the SQLite DB before parallel jobs start.
    Run this once from the submit script before sbatch array submission.
    """
    import pandas as pd
    _abs_db    = Path(optuna_db).resolve()
    optuna_url = f"sqlite:///{_abs_db}"
    Path(optuna_db).parent.mkdir(parents=True, exist_ok=True)
    Path(optuna_db).touch(exist_ok=True)

    human_df = pd.read_pickle(data_path(f"{dataset}.pkl"))
    all_pids = sorted(human_df["pid"].unique())
    dists    = _get_distributions(model_type, dataset)

    print(f"Initialising {len(all_pids)} studies in {_abs_db}")
    for pid in all_pids:
        study = optuna.create_study(
            study_name=f"{model_type}_{dataset}_pid{pid}",
            storage=optuna_url,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=optuna_seed + pid,
                                               n_startup_trials=20),
        )
        print(f"  pid={pid}: {len(study.trials)} existing trials")
    print("Done — safe to submit parallel jobs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",     type=str)
    parser.add_argument("model_type",  type=str)
    parser.add_argument("pid",         type=int)
    parser.add_argument("--n_fits",    type=int, default=50)
    parser.add_argument("--n_sims",   type=int, default=100)
    parser.add_argument("--db_folder", type=str, default="data/sim_db")
    parser.add_argument("--optuna_db", type=str,
                        default="data/optuna/carrabin.db")
    parser.add_argument("--run_folder", type=str, default="carrabin")
    parser.add_argument("--optuna_seed", type=int, default=42)
    parser.add_argument("--init_db", action="store_true",
                        help="Pre-initialise all pid studies and exit (run before sbatch)")
    args = parser.parse_args()

    if args.init_db:
        init_db(dataset=args.dataset, model_type=args.model_type,
                optuna_db=args.optuna_db, optuna_seed=args.optuna_seed)
        return

    fit_pid(
        dataset=args.dataset,
        model_type=args.model_type,
        target_pid=args.pid,
        n_fits=args.n_fits,
        n_sims=args.n_sims,
        db_folder=args.db_folder,
        optuna_db=args.optuna_db,
        run_folder=args.run_folder,
        optuna_seed=args.optuna_seed,
    )


if __name__ == "__main__":
    main()
