#!/usr/bin/env python3
"""
Submit and collect NEF_recurrent yoo response-noise simulations.
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
from utils.paths import DATA_DIR, RUNS_DIR, data_path
from utils.slurm import DEFAULT_MEM_LIMITS, make_job_script, submit_script

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


def _single_pass_time_limit() -> str:
    return "1:00:00"


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


def _submit_command(*, script_name: str, command: str, time_limit: str, mem: str, dry_run: bool) -> None:
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


def _submit_mode(args: argparse.Namespace) -> None:
    if args.local:
        if args.pid is None or args.seed_index is None:
            raise ValueError("--mode submit --local requires --pid and --seed_index")
        _run_one(args.pid, args.seed_index, args.run_folder, args.source_folder)
        return

    pids = [int(args.pid)] if args.pid is not None else _yoo_pids()
    seed_indices = [int(args.seed_index)] if args.seed_index is not None else list(range(args.n_seeds))
    time_limit = _single_pass_time_limit()
    mem = DEFAULT_MEM_LIMITS.get(MODEL_TYPE, "8G")
    print(
        f"Submitting {len(pids) * len(seed_indices)} jobs "
        f"({len(pids)} pids × {len(seed_indices)} seeds). "
        f"Using walltime={time_limit} per job."
    )

    for pid in pids:
        for seed_index in seed_indices:
            cmd = (
                "python scripts/yoo_response_noise.py "
                f"--mode submit --local --pid {int(pid)} --seed_index {int(seed_index)} "
                f"--run_folder {args.run_folder} --source_folder {args.source_folder} "
                f"--n_seeds {int(args.n_seeds)}"
            )
            _submit_command(
                script_name=f"yoo_noise_{MODEL_TYPE}_{pid}_seed{seed_index}.sh",
                command=cmd,
                time_limit=time_limit,
                mem=mem,
                dry_run=args.dry_run,
            )


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
    parser = argparse.ArgumentParser(prog="scripts.yoo_response_noise")
    parser.add_argument("--mode", choices=["submit", "collect"], required=True)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--run_folder", type=str, default="yoo_response_noise")
    parser.add_argument("--source_folder", type=str, default="response")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--seed_index", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "submit":
        _submit_mode(args)
    else:
        _collect_mode(args)


if __name__ == "__main__":
    main()
