#!/usr/bin/env python3
"""
Generate supplementary NEF response-noise simulations for figure_yoo.py.

Runs NEF_recurrent with multiple random seeds per participant to estimate
response variability. Output is used by panel H of figure_yoo (mean response
noise vs mean response change). Must be run before figure_yoo.py.

=============================================================================
CLUSTER USAGE
=============================================================================

  Submit all jobs (one SLURM job per pid × seed_index):
      bash jobs/submit_yoo_noise.sh

  Collect (after all jobs complete):
      python scripts/extras_yoo.py --mode collect --n_seeds 10 \\
          --run_folder nef200

=============================================================================
OUTPUT FILES  (written to data/runs/nef200/)
=============================================================================

  NEF_recurrent_yoo_<pid>_seed<N>_responses.pkl  — per-pid per-seed responses
  NEF_recurrent_yoo_all_responses.pkl             — combined file (after collect)

The combined file is read by figure_yoo.py --noise_folder nef200.
=============================================================================
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS
from models import NEF
from utils.paths import RUNS_DIR, data_path

MODEL_TYPE = "NEF_recurrent"
DATASET = "yoo"
RESP_RE = re.compile(rf"^{MODEL_TYPE}_{DATASET}_(\d+)_seed(\d+)_responses\.pkl$")


def _yoo_pids() -> list[int]:
    return sorted(int(p) for p in pd.read_pickle(data_path(f"{DATASET}.pkl"))["pid"].unique())


def _source_params(source_folder: str, pid: int) -> tuple[float, float, int]:
    p = RUNS_DIR / source_folder / f"{MODEL_TYPE}_{DATASET}_{pid}_params.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Missing source params: {p}")
    row = pd.read_pickle(p).iloc[0]
    for key in ("lambda_", "alpha_0"):
        if key not in row:
            raise KeyError(f"Expected {key!r} in {p}")
    # fall back to 0 if legacy per-pid params do not store seed
    base_seed = int(row["seed"]) if "seed" in row and pd.notna(row["seed"]) else 0
    return float(row["lambda_"]), float(row["alpha_0"]), base_seed


def _total_seed(pid: int, seed_index: int, base_seed: int) -> int:
    # Unique across pid×seed_index; base_seed contributes lower-order variation.
    return int(pid) * 1_000_000 + int(seed_index) * 1_000 + (int(base_seed) % 1000)


def _run_one(pid: int, seed_index: int, run_folder: str, source_folder: str) -> Path:
    lambda_, alpha_0, base_seed = _source_params(source_folder, pid)
    seed = _total_seed(pid, seed_index, base_seed)
    fixed = dict(MODEL_PARAMS[DATASET][MODEL_TYPE].get("fixed", {}))
    params = {
        "model_type": MODEL_TYPE,
        "dataset": DATASET,
        "pid": int(pid),
        **fixed,
        "lambda_": float(lambda_),
        "alpha_0": float(alpha_0),
        "seed": int(seed),
    }
    df = NEF.run(params, save=False)
    df = df[["model_type", "pid", "trial", "observation", "response"]].copy()
    df["seed"] = int(seed)
    df = df[["model_type", "pid", "seed", "trial", "observation", "response"]]

    out_dir = RUNS_DIR / run_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODEL_TYPE}_{DATASET}_{pid}_seed{seed_index}_responses.pkl"
    df.to_pickle(out_path)
    print(f"Saved {out_path}")
    print("JOB_COMPLETE")
    return out_path


def _run_mode(args: argparse.Namespace) -> None:
    """Run a single pid×seed simulation locally."""
    if args.pid is None or args.seed_index is None:
        raise ValueError("--mode run requires --pid and --seed_index")
    _run_one(args.pid, args.seed_index, args.run_folder, args.source_folder)


def _collect_mode(args: argparse.Namespace) -> None:
    run_dir = RUNS_DIR / args.run_folder
    run_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(run_dir.glob(f"{MODEL_TYPE}_{DATASET}_*_seed*_responses.pkl"))
    if not files:
        print(f"No response files found in {run_dir}")
        return

    # Reuses fitting.collect pattern: sorted glob + pd.concat([...], ignore_index=True).
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    out = run_dir / f"{MODEL_TYPE}_{DATASET}_all_responses.pkl"
    df.to_pickle(out)

    expected_pids = _yoo_pids()
    expected = {(int(pid), int(seed_idx)) for pid in expected_pids for seed_idx in range(args.n_seeds)}
    found: set[tuple[int, int]] = set()
    for f in files:
        m = RESP_RE.match(f.name)
        if not m:
            continue
        found.add((int(m.group(1)), int(m.group(2))))
    missing = sorted(expected - found)
    if missing:
        print(f"Warning: missing {len(missing)} pid×seed files for n_seeds={args.n_seeds}")
        for pid, seed_idx in missing[:20]:
            print(f"  missing pid={pid} seed_index={seed_idx}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    seeds_per_pid = (
        df[["pid", "seed"]].drop_duplicates().groupby("pid").size().to_dict()
        if {"pid", "seed"}.issubset(df.columns)
        else {}
    )
    unique_seed_counts = sorted(set(seeds_per_pid.values())) if seeds_per_pid else []
    print(f"Collected {len(files)} -> {out} ({df.shape})")
    print(
        f"Summary: n pids={df['pid'].nunique() if 'pid' in df.columns else 0}, "
        f"n seeds per pid={unique_seed_counts}, total rows={len(df)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="scripts.extras_yoo")
    parser.add_argument(
        "--mode",
        choices=["run", "collect"],
        required=True,
    )
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--run_folder", type=str, default="nef200")
    parser.add_argument("--source_folder", type=str, default="refit")
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--seed_index", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "run":
        _run_mode(args)
    elif args.mode == "collect":
        _collect_mode(args)


if __name__ == "__main__":
    main()
