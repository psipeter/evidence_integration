#!/usr/bin/env python3
"""
MLE fitting via shared simulation database and per-process in-memory Optuna.

Each process fits one pid using its own in-memory Optuna study.
Processes share only the simulation database (pickle files — no SQLite).
Cross-pid sharing happens at step 1: each process scans all simulation files
and evaluates loss for its own pid, injecting any new entries into its study.

Architecture:
  - Simulation database: data/sim_db/{model}/{model}_{hash}.pkl
    One file per parameter point, shared read/write across all processes.
    Written via atomic rename (NFS-safe, no stale locks).
  - Per-pid checkpoint: data/sim_db/{model}/checkpoint_pid{pid}.pkl
    Saves study state after each fit so jobs can resume if interrupted.
  - No SQLite / no shared Optuna storage.

Loop per process (n_fits iterations):
  1. Scan simulation database → evaluate loss for this pid → inject into study
  2. Ask in-memory TPE for next params
  3. Cache hit: params already in database → evaluate loss, report, continue
  4. Simulate n_sims times → save to database (atomic rename)
  5. Evaluate loss for this pid → report to study

Usage:
    python -m fitting.fit_mle carrabin NoisyCounting 1 \\
        --n_fits 500 --n_sims 100 \\
        --db_folder data/sim_db \\
        --run_folder carrabin

SLURM: one job per pid, all pointing to the same db_folder.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
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
    from optuna.distributions import CategoricalDistribution
    spec = (MLE_PARAMS if dataset in MLE_PARAMS
            and model_type in MLE_PARAMS.get(dataset, {})
            else MODEL_PARAMS)[dataset][model_type]
    dists = {}
    for name, bounds in spec.items():
        if name == "fixed":
            continue
        # List value or "categorical" marker -> CategoricalDistribution
        if isinstance(bounds, list):
            dists[name] = CategoricalDistribution(bounds)
            continue
        if bounds == "categorical":
            # Resolve from NEF_N_NEURONS_VALUES
            from fitting.model_params import NEF_N_NEURONS_VALUES
            dists[name] = CategoricalDistribution(NEF_N_NEURONS_VALUES)
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
    spec = (MLE_PARAMS if dataset in MLE_PARAMS
            and model_type in MLE_PARAMS.get(dataset, {})
            else MODEL_PARAMS)[dataset][model_type]
    fixed = spec.get("fixed", {})
    params = {**fixed, **free,
              "model_type": model_type, "dataset": dataset, "pid": pid}
    # For NEF MLE: tie n_neurons_counting to n_neurons
    if model_type == "NEF" and "n_neurons" in free:
        params["n_neurons_counting"] = int(free["n_neurons"])
        params["n_neurons"]          = int(free["n_neurons"])
    return params


# ── Database helpers ──────────────────────────────────────────────────────────

def _db_path(db_dir: Path, model_type: str, ph: str) -> Path:
    return db_dir / model_type / f"{model_type}_{ph}.pkl"


def _list_db_entries(db_dir: Path, model_type: str) -> list[Path]:
    model_dir = db_dir / model_type
    if not model_dir.exists():
        return []
    # Exclude temp files and checkpoints
    return sorted(p for p in model_dir.glob(f"{model_type}_*.pkl")
                  if not p.name.startswith("checkpoint"))


def _simulate_and_save(
    model_type: str,
    params: dict,
    db_dir: Path,
    n_sims: int,
    run_folder: str,
) -> Path:
    """Simulate and save using atomic rename (NFS-safe)."""
    ph        = params_hash(model_type, params)
    model_dir = db_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path  = model_dir / f"{model_type}_{ph}.pkl"

    if out_path.exists():
        return out_path

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
            out_path_override=tmp_path,
        )
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return out_path


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(db_dir: Path, model_type: str, pid: int) -> Path:
    return db_dir / model_type / f"checkpoint_pid{pid}.pkl"


def _save_checkpoint(study: optuna.Study, db_dir: Path,
                     model_type: str, pid: int,
                     injected: set[str],
                     sim_times: list[float] | None = None) -> None:
    """Save study trials and injected hashes to a per-pid checkpoint file."""
    trials_data = [
        {"params": t.params, "value": t.value}
        for t in study.trials
        if t.value is not None
    ]
    data = {"trials": trials_data, "injected": list(injected),
            "sim_times": sim_times or []}
    path = _checkpoint_path(db_dir, model_type, pid)
    tmp  = path.with_suffix(".tmp")
    pd.to_pickle(data, tmp)
    os.replace(tmp, path)


def _load_checkpoint(study: optuna.Study, db_dir: Path,
                     model_type: str, pid: int,
                     dists: dict) -> set[str]:
    """Restore study from checkpoint. Returns injected set."""
    path = _checkpoint_path(db_dir, model_type, pid)
    if not path.exists():
        return set(), []
    data     = pd.read_pickle(path)
    injected = set(data.get("injected", []))
    for td in data.get("trials", []):
        if td["value"] is None:
            continue
        try:
            study.add_trial(create_trial(
                params=td["params"],
                distributions=dists,
                value=td["value"],
            ))
        except Exception:
            pass   # duplicate trial — already in study
    return injected, data.get("sim_times", [])


# ── Main fitting loop ─────────────────────────────────────────────────────────

def fit_pid(
    dataset: str,
    model_type: str,
    target_pid: int,
    n_fits: int = 500,
    n_sims: int = 100,
    db_folder: str = "data/sim_db",
    run_folder: str = "carrabin",
    optuna_seed: int = 42,
) -> None:
    db_dir     = Path(db_folder)
    human_df   = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human_pid  = human_df[human_df["pid"] == target_pid].copy()
    dists      = _get_distributions(model_type, dataset)

    # In-memory study — no SQLite, no contention
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=optuna_seed + target_pid,
            n_startup_trials=20,
        ),
    )

    # Restore from checkpoint if available
    injected, sim_times = _load_checkpoint(study, db_dir, model_type, target_pid, dists)
    _log(f"Starting fit: model={model_type} n_fits={n_fits} n_sims={n_sims} "
         f"(resuming from {len(study.trials)} trials)", target_pid)

    db_entries: list[Path] = []   # cached directory listing, refreshed every 5 fits
    sim_times:  list[float] = []  # elapsed seconds per simulation (step 4 only)

    for fit_idx in range(n_fits):
        t0 = time.time()

        # ── Step 1: scan database → inject new entries into this study ────────
        # Re-glob every 5 iterations to pick up new files from other processes
        # while avoiding expensive NFS directory reads every iteration.
        if fit_idx % 5 == 0:
            db_entries = _list_db_entries(db_dir, model_type)

        for db_path in db_entries:
            # Derive hash from filename — avoids loading file just to check
            ph = db_path.stem[len(model_type) + 1:]  # strip "{model_type}_"
            if ph in injected:
                continue
            try:
                entry    = pd.read_pickle(db_path)
                params_e = entry["params"]
                free_e   = {k: v for k, v in params_e.items() if k in dists}
                loss     = compute_sim_db_loss(model_type, params_e, human_pid, db_dir)
                # Skip system_attrs — we track injected ourselves
                study.add_trial(create_trial(
                    params=free_e, distributions=dists, value=loss,
                ))
                injected.add(ph)
            except Exception as e:
                _log(f"  Warning: could not evaluate {ph[:8]}: {e}", target_pid)

        n_trials = len(study.trials)
        _log(f"fit {fit_idx+1}/{n_fits}: {n_trials} trials in study", target_pid)

        # ── Step 2: ask TPE for next params ───────────────────────────────────
        trial  = study.ask(fixed_distributions=dists)
        free   = trial.params
        params = _full_params(model_type, dataset, free, target_pid)
        ph     = params_hash(model_type, params)

        # ── Step 3: cache hit ─────────────────────────────────────────────────
        db_path_new = _db_path(db_dir, model_type, ph)
        if db_path_new.exists():
            _log(f"  cache hit hash={ph[:8]}", target_pid)
            try:
                loss = compute_sim_db_loss(model_type, params, human_pid, db_dir)
            except Exception:
                loss = 1e6
            study.tell(trial, loss)
            continue

        # ── Step 4: simulate ──────────────────────────────────────────────────
        _log(f"  simulating hash={ph[:8]} params={free}", target_pid)
        t_sim = time.time()
        _simulate_and_save(model_type, params, db_dir, n_sims, run_folder)
        elapsed = time.time() - t_sim   # pure simulation time only
        sim_times.append(elapsed)

        # ── Step 5: evaluate and report ───────────────────────────────────────
        loss = compute_sim_db_loss(model_type, params, human_pid, db_dir)
        study.tell(trial, loss)
        injected.add(ph)
        _log(f"  loss={loss:.4f} ({elapsed:.1f}s)", target_pid)

        # Checkpoint every 10 fits
        if (fit_idx + 1) % 10 == 0:
            _save_checkpoint(study, db_dir, model_type, target_pid, injected, sim_times)

    # Final checkpoint
    _save_checkpoint(study, db_dir, model_type, target_pid, injected, sim_times)

    # Save best params
    best      = study.best_trial
    best_free = best.params
    best_full = _full_params(model_type, dataset, best_free, target_pid)
    out_dir   = resolve_run_folder(run_folder)

    pd.to_pickle(
        pd.DataFrame([{**best_full, "mle_loss": best.value}]),
        out_dir / f"{model_type}_{dataset}_{target_pid}_params_mle.pkl",
    )
    sim_times_arr = np.array(sim_times) if sim_times else np.array([0.0])
    pd.to_pickle(
        pd.DataFrame([{
            "pid":              target_pid,
            "mle_loss":         best.value,
            "n_trials":         len(study.trials),
            "n_sims_done":      len(sim_times),
            "sim_time_mean_s":  float(sim_times_arr.mean()),
            "sim_time_total_s": float(sim_times_arr.sum()),
            "sim_time_median_s":float(np.median(sim_times_arr)),
        }]),
        out_dir / f"{model_type}_{dataset}_{target_pid}_performance_mle.pkl",
    )
    _log(f"Done. best_loss={best.value:.4f} params={best_free}", target_pid)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",      type=str)
    parser.add_argument("model_type",   type=str)
    parser.add_argument("pid",          type=int)
    parser.add_argument("--n_fits",     type=int, default=500)
    parser.add_argument("--n_sims",     type=int, default=100)
    parser.add_argument("--db_folder",  type=str, default="data/sim_db")
    parser.add_argument("--run_folder", type=str, default="carrabin")
    parser.add_argument("--optuna_seed", type=int, default=42)
    args = parser.parse_args()

    fit_pid(
        dataset=args.dataset,
        model_type=args.model_type,
        target_pid=args.pid,
        n_fits=args.n_fits,
        n_sims=args.n_sims,
        db_folder=args.db_folder,
        run_folder=args.run_folder,
        optuna_seed=args.optuna_seed,
    )


if __name__ == "__main__":
    main()
