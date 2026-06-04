#!/usr/bin/env python3
"""Build a simulation database for carrabin PMMH-style likelihood fitting.

For a given model and parameter set, simulates all 32 binary sequences
n_sims times each, storing the complete response trajectory (5 responses)
per simulation. Simulations are shared across all pids.

Database structure (one file per parameter point):
    data/sim_db/{model_type}/{model_type}_{params_hash}.pkl
    Contents: {
        "params":     dict of model parameters,
        "model_type": str,
        "n_sims":    int,
        "elapsed_s":  float,
        "data": {
            seq_tuple: np.ndarray of shape (n_sims, n_obs)
                       each row is one simulated response trajectory
        }
    }

The likelihood for a given pid is then:
    For each trial the pid ran with sequence S:
        Compare observed response trajectory (5 values) against the
        simulated distribution data[S] — a (n_sims, 5) array.
        Evaluate a Gaussian (or multivariate Gaussian) log-likelihood.

Usage:
    python scripts/build_sim_db.py \\
        --model NEF \\
        --params_json '{"alpha_0": 0.3, "lambda_": 0.5}' \\
        --n_sims 100 \\
        --db_folder data/sim_db \\
        --run_folder carrabin

    python scripts/build_sim_db.py \\
        --model NoisyCounting \\
        --grid_file data/sim_db/NoisyCounting_grid.pkl \\
        --grid_idx 42 \\
        --n_sims 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS, _NEF_FIXED
from utils.paths import data_path, RUNS_DIR

# All 32 binary sequences of length 5, as tuples of +1/-1
ALL_SEQUENCES: list[tuple] = [
    tuple((1 if (i >> (4 - j)) & 1 else -1) for j in range(5))
    for i in range(32)
]


def params_hash(model_type: str, params: dict) -> str:
    """Stable 12-char hash of (model_type, free params) for filenames."""
    SKIP = {"pid", "model_type", "dataset", "seed", "base_seed"}
    free = {k: v for k, v in params.items() if k not in SKIP}
    key  = json.dumps({"model": model_type, "params": free}, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def simulate_param_point(
    model_type: str,
    params: dict,
    n_sims: int,
    db_dir: Path,
    run_folder: str = "carrabin",
    overwrite: bool = False,
    out_path_override: Path | None = None,
) -> Path:
    """Simulate all 32 sequences × n_sims for one parameter point.

    For each sequence, produces an (n_sims, n_obs) array of response
    trajectories. Saves to db_dir/{model_type}_{hash}.pkl.

    Returns the output path.
    """
    ph        = params_hash(model_type, params)
    model_dir = db_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_path_override or (model_dir / f"{model_type}_{ph}.pkl")

    if out_path.exists() and not overwrite and out_path_override is None:
        print(f"Already exists: {out_path.name} — skipping")
        return out_path

    print(f"Simulating {model_type}  hash={ph}")
    print(f"  params: { {k:v for k,v in params.items() if k not in ('model_type','dataset','pid')} }")
    print(f"  n_sims={n_sims}, n_sequences={len(ALL_SEQUENCES)}")

    DETERMINISTIC = {"Mean", "PrimacyRecency", "LeakyIntegrator", "RL"}
    t0 = time.time()
    db_data: dict[tuple, np.ndarray] = {}

    for seq in ALL_SEQUENCES:
        if model_type in DETERMINISTIC:
            traj = _simulate_deterministic(model_type, params, seq)
            # Replicate n_sims times — distribution is a delta
            db_data[seq] = np.tile(traj, (n_sims, 1))  # (n_sims, n_obs)

        elif model_type == "NoisyCounting":
            db_data[seq] = _simulate_noisy_counting(params, seq, n_sims)

        elif model_type == "NEF":
            db_data[seq] = _simulate_nef(params, seq, n_sims, run_folder)

        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

    elapsed = time.time() - t0
    out = {
        "params":     params,
        "model_type": model_type,
        "n_sims":    n_sims,
        "elapsed_s":  elapsed,
        "data":       db_data,  # {seq_tuple: (n_sims, n_obs) array}
    }
    pd.to_pickle(out, out_path)
    print(f"  Saved in {elapsed:.0f}s → {out_path.name}")
    return out_path


# ── Model-specific simulators ──────────────────────────────────────────────────

def _simulate_deterministic(model_type: str, params: dict,
                             seq: tuple) -> np.ndarray:
    """Return shape (n_obs,) response trajectory for a deterministic model."""
    import models.math_models as mm
    rows = [{"pid": 0, "trial": 1, "observation": i + 1,
             "value": v, "response": 0.0, "qid": ""}
            for i, v in enumerate(seq)]
    human_pid = pd.DataFrame(rows)
    traj = []
    for obs in range(1, len(seq) + 1):
        r = float(mm.run_trial(params, human_pid, trial=1, observation=obs))
        traj.append(r)
    return np.array(traj)


def _simulate_noisy_counting(params: dict, seq: tuple,
                              n_sims: int) -> np.ndarray:
    """Return shape (n_sims, n_obs) for NoisyCounting.

    Matches the implementation in math_models.py exactly:
      r = cognitive state (cumulative count)
      p_hat = response estimate on [-1, 1]
      Update: r += x * mu + xi (xi ~ N(0, sigma_c))
              p_hat += (r - p_hat) * exp(epsilon) (epsilon ~ N(0, nu))
              p_hat = clip(p_hat, -1, 1)

    No carrabin shrinkage transform is applied (NoisyCounting is excluded).
    Seeds are 0..n_sims-1 to match a trial-index-based seeding scheme.
    """
    mu      = float(params["mu"])
    sigma_c = float(params["sigma_c"])
    nu      = float(params["nu"])
    n_obs   = len(seq)
    trajs   = np.zeros((n_sims, n_obs))

    for seed in range(n_sims):
        rng   = np.random.RandomState(seed)
        r     = 0.0
        p_hat = 0.0
        for obs_idx, val in enumerate(seq):
            xi    = rng.normal(0.0, sigma_c)
            r     = r + float(val) * mu + xi
            eps   = rng.normal(0.0, nu)
            p_hat = p_hat + (r - p_hat) * float(np.exp(eps))
            p_hat = float(np.clip(p_hat, -1.0, 1.0))
            trajs[seed, obs_idx] = p_hat

    return trajs


def _simulate_nef(params: dict, seq: tuple, n_sims: int,
                  run_folder: str) -> np.ndarray:
    """Return shape (n_sims, n_obs) for the NEF model."""
    from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
    from models.counting_integrator import fast_decode, load_activities

    n  = int(params.get("n_neurons", 100))
    nc = int(params.get("n_neurons_counting", n))
    try:
        activity_map = load_activities(
            n_neurons=n, n_neurons_counting=nc, dataset="carrabin"
        )
    except FileNotFoundError:
        activity_map = None

    full_params = {
        **PARAM_DEFAULTS, **_NEF_FIXED, **params,
        "dataset": "carrabin", "model_type": "NEF",
    }
    n_obs  = len(seq)
    trajs  = []
    failed = 0

    for seed in range(n_sims):
        p = {**full_params, "seed": seed}
        if activity_map is not None:
            act = activity_map.get(seed % len(activity_map))
            decoders = (fast_decode(act,
                                    alpha_0=float(params["alpha_0"]),
                                    lambda_=float(params["lambda_"]))
                        if act is not None
                        else _pretrain({**p, "base_seed": seed}))
        else:
            decoders = _pretrain({**p, "base_seed": seed})
        try:
            responses, _ = _simulate_trial(
                np.array(seq, dtype=float), p, decoders, return_probes=True
            )
            trajs.append([float(r) for r in responses])
        except Exception:
            failed += 1

    if failed:
        print(f"  Warning: {failed}/{n_sims} seeds failed")

    return np.array(trajs) if trajs else np.zeros((0, n_obs))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["Mean", "PrimacyRecency", "LeakyIntegrator",
                                 "NoisyCounting", "NEF"])
    parser.add_argument("--params_json", type=str, default=None,
                        help="JSON string of free parameters")
    parser.add_argument("--grid_file", type=str, default=None,
                        help="Path to pkl of parameter grid (list of dicts)")
    parser.add_argument("--grid_idx", type=int, default=None,
                        help="Index into grid_file to simulate")
    parser.add_argument("--n_sims", type=int, default=100)
    parser.add_argument("--db_folder", type=str, default="data/sim_db")
    parser.add_argument("--run_folder", type=str, default="carrabin")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    db_dir = Path(args.db_folder) / args.model
    db_dir.mkdir(parents=True, exist_ok=True)

    # Resolve free parameters
    if args.params_json is not None:
        free_params = json.loads(args.params_json)
    elif args.grid_file is not None and args.grid_idx is not None:
        grid = pd.read_pickle(args.grid_file)
        free_params = grid[args.grid_idx]
    else:
        parser.error("Provide either --params_json or --grid_file + --grid_idx")

    # Merge with fixed params from MODEL_PARAMS
    dataset = "carrabin"
    fixed   = MODEL_PARAMS.get(dataset, {}).get(args.model, {}).get("fixed", {})
    params  = {**fixed, **free_params,
               "model_type": args.model, "dataset": dataset, "pid": 0}

    simulate_param_point(
        model_type=args.model,
        params=params,
        n_sims=args.n_sims,
        db_dir=db_dir,
        run_folder=args.run_folder,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
