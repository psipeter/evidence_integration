#!/usr/bin/env python3
"""
Check SLURM job logs for JOB_COMPLETE and identify queued jobs
that are finished (cancelable) vs still running.

Usage:
    python scripts/check_jobs.py              # report only
    python scripts/check_jobs.py --cancel     # cancel cancelable jobs only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import PROJECT_ROOT

LOGS_DIR = PROJECT_ROOT / "logs"


def get_all_slurm_jobs() -> set[str]:
    """Return set of all job IDs currently in SLURM queue (any state)."""
    result = subprocess.run(
        ["squeue", "--me", "--format=%i", "--noheader"],
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.strip().splitlines() if line.strip()}


def scan_complete_logs(queued_job_ids: set[str]) -> list[str]:
    """Return queued job IDs whose log tails contain JOB_COMPLETE."""
    complete: list[str] = []
    for job_id in sorted(queued_job_ids):
        log_file = LOGS_DIR / f"{job_id}.out"
        if not log_file.exists():
            continue
        with log_file.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 1024))
            tail = f.read()
        if b"JOB_COMPLETE" in tail:
            complete.append(job_id)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Cancel jobs that have JOB_COMPLETE but are still listed in squeue",
    )
    args = parser.parse_args()

    queued_ids = sorted(get_all_slurm_jobs())
    complete_ids = scan_complete_logs(set(queued_ids))
    complete_set = set(complete_ids)

    cancelable = sorted(set(queued_ids) & complete_set)
    running = [job_id for job_id in queued_ids if job_id not in complete_set]

    print("SLURM / log summary\n")
    print(f"  Jobs currently in queue (squeue --me): {len(queued_ids)}")

    if queued_ids:
        print("\nQueued job status:")
        for job_id in cancelable:
            print(f"  [DONE]    {job_id}")
        for job_id in running:
            print(f"  [RUNNING] {job_id}")

    if not cancelable:
        print("\nNo cancelable jobs (none still in queue with JOB_COMPLETE).")
        return

    print(f"\n{len(cancelable)} job(s) can be cancelled with --cancel.")
    if args.cancel:
        for job_id in cancelable:
            subprocess.run(["scancel", job_id], check=False)
            print(f"  Cancelled {job_id}")
    else:
        print("Run with --cancel to cancel them.")
        print(f"  scancel {' '.join(cancelable)}")


if __name__ == "__main__":
    main()
