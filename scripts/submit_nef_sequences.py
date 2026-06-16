"""
scripts/submit_nef_sequences.py
================================
Submit NEF sequence runs as individual SLURM jobs on the cluster.
Each job runs one lambda_ index (fixed alpha_0=0.5), matching the RL_lambda
simulation approach. One job per lambda, results collected afterwards.

Usage:
    # Submit all 100 lambda jobs
    python scripts/submit_nef_sequences.py

    # Submit a subset
    python scripts/submit_nef_sequences.py --lambda_indices 0 10 20 50 99

    # Dry run
    python scripts/submit_nef_sequences.py --dry_run

    # After all jobs complete:
    python scripts/collect_nef_sequences.py
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.slurm import make_job_script, submit_script
from utils.paths import RUNS_DIR, PROJECT_ROOT

CLUSTER_ROOT = "/dartfs-hpc/rc/home/n/f007qzn/evidence_integration"
JOBS_DIR     = PROJECT_ROOT / "jobs" / "nef_sequences"
N_LAMBDAS    = 100


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_lambdas', type=int, default=N_LAMBDAS,
                        help='Total number of lambda values in grid')
    parser.add_argument('--lambda_indices', type=int, nargs='+', default=None,
                        help='Specific indices to submit (default: all)')
    parser.add_argument('--tasks', nargs='+',
                        default=['task_continuous', 'task_binary'])
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir  = CLUSTER_ROOT + "/logs"
    nef_dir  = RUNS_DIR / "test_sequences" / "nef_runs"
    nef_dir.mkdir(parents=True, exist_ok=True)

    indices = args.lambda_indices if args.lambda_indices else list(range(args.n_lambdas))

    # Skip already-completed runs
    to_submit = [i for i in indices
                 if not (nef_dir / f'nef_l{i:03d}.pkl').exists()]

    lambdas = np.linspace(0.01, 0.99, args.n_lambdas)
    print(f"Submitting {len(to_submit)}/{len(indices)} lambda jobs "
          f"({'dry run' if args.dry_run else 'live'})")

    submitted = 0
    for idx in to_submit:
        lam = lambdas[idx]
        cmd = (f"venv/bin/python scripts/run_nef_sequences.py "
               f"--lambda_index {idx} "
               f"--tasks {' '.join(args.tasks)}")

        script = make_job_script(
            root       = CLUSTER_ROOT,
            commands   = [cmd],
            time_limit = "4:0:0",
            mem        = "32G",
            log_dir    = log_dir,
        )
        script_path = JOBS_DIR / f"nef_l{idx:03d}.sh"
        script_path.write_text(script)

        if args.dry_run:
            print(f"  [dry_run] {script_path.name}: lambda_={lam:.4f}")
        else:
            print(f"  Submitting {script_path.name}: lambda_={lam:.4f}")
            submit_script(script_path, dry_run=False)
            submitted += 1

    print(f"\nDone. {submitted} jobs submitted.")
    if not args.dry_run:
        print(f"Job scripts: {JOBS_DIR}")
        print(f"Results:     {nef_dir}")
        print(f"\nAfter completion:")
        print(f"  python scripts/collect_nef_sequences.py")


if __name__ == "__main__":
    main()
