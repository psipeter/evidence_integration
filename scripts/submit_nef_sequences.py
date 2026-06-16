"""
scripts/submit_nef_sequences.py
================================
Submit NEF sequence runs as individual SLURM jobs on the cluster.
Each job runs one (alpha_0, lambda_) param set across all trials and tasks,
saving to data/runs/test_sequences/nef_runs/nef_a{...}_l{...}.pkl

Usage:
    # Submit 20 random param sets
    python scripts/submit_nef_sequences.py --n_params 20

    # Dry run — print job scripts without submitting
    python scripts/submit_nef_sequences.py --n_params 20 --dry_run

    # Submit specific params
    python scripts/submit_nef_sequences.py \
        --params 0.5,0.15 0.5,0.40 0.5,0.67 0.5,0.88

    # After jobs complete, collect results:
    python scripts/collect_nef_sequences.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.slurm import make_job_script, submit_script
from utils.paths import RUNS_DIR, PROJECT_ROOT


CLUSTER_ROOT = "/dartfs-hpc/rc/home/n/f007qzn/evidence_integration"
JOBS_DIR     = PROJECT_ROOT / "jobs" / "nef_sequences"


def param_tag(alpha_0: float, lambda_: float) -> str:
    return f"a{alpha_0:.3f}_l{lambda_:.3f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n_params", type=int, default=20,
                        help="Number of random (alpha_0, lambda_) pairs to sample")
    parser.add_argument("--alpha_0_min", type=float, default=0.2)
    parser.add_argument("--alpha_0_max", type=float, default=1.0)
    parser.add_argument("--lambda_min",  type=float, default=0.01)
    parser.add_argument("--lambda_max",  type=float, default=1.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--tasks",       nargs="+",
                        default=["task_continuous", "task_binary"])
    parser.add_argument("--params", nargs="+", default=None,
                        help="Explicit param pairs as alpha_0,lambda_ e.g. 0.5,0.15")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = CLUSTER_ROOT + "/logs"

    # ── Build param list ─────────────────────────────────────────────────────
    if args.params:
        param_list = []
        for p in args.params:
            a, l = p.split(",")
            param_list.append((float(a), float(l)))
    else:
        rng = np.random.default_rng(args.seed)
        alpha_0s = rng.uniform(args.alpha_0_min, args.alpha_0_max, args.n_params)
        lambdas  = rng.uniform(args.lambda_min,  args.lambda_max,  args.n_params)
        param_list = list(zip(alpha_0s.tolist(), lambdas.tolist()))

    # ── Skip already-completed runs ──────────────────────────────────────────
    nef_dir = RUNS_DIR / "test_sequences" / "nef_runs"
    nef_dir.mkdir(parents=True, exist_ok=True)

    to_submit = []
    for alpha_0, lambda_ in param_list:
        tag      = param_tag(alpha_0, lambda_)
        out_path = nef_dir / f"nef_{tag}.pkl"
        if out_path.exists():
            print(f"  [skip] already done: {tag}")
        else:
            to_submit.append((alpha_0, lambda_))

    print(f"Submitting {len(to_submit)}/{len(param_list)} param sets "
          f"({'dry run' if args.dry_run else 'live'})")

    # ── Write and submit one job per param set ───────────────────────────────
    submitted = 0
    for alpha_0, lambda_ in to_submit:
        tag = param_tag(alpha_0, lambda_)
        cmd = (
            f"venv/bin/python scripts/run_nef_sequences.py "
            f"--alpha_0 {alpha_0:.6f} --lambda_ {lambda_:.6f} "
            f"--tasks {' '.join(args.tasks)}"
        )
        script = make_job_script(
            root      = CLUSTER_ROOT,
            commands  = [cmd],
            time_limit = "4:0:0",
            mem       = "32G",
            log_dir   = log_dir,
        )
        script_path = JOBS_DIR / f"nef_{tag}.sh"
        script_path.write_text(script)

        if args.dry_run:
            print(f"  [dry_run] {script_path.name}: alpha_0={alpha_0:.3f} lambda_={lambda_:.3f}")
            print(f"    cmd: {cmd}")
        else:
            print(f"  Submitting {script_path.name}: "
                  f"alpha_0={alpha_0:.3f} lambda_={lambda_:.3f}")
            submit_script(script_path, dry_run=False)
            submitted += 1

    print(f"\nDone. {submitted} jobs submitted.")
    if not args.dry_run:
        print(f"Job scripts in: {JOBS_DIR}")
        print(f"Results will appear in: {nef_dir}")
        print(f"\nAfter completion, collect with:")
        print(f"  python scripts/collect_nef_sequences.py")


if __name__ == "__main__":
    main()
