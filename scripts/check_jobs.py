#!/usr/bin/env python3
"""
Check SLURM job logs for JOB_COMPLETE and identify jobs still in the queue
that have finished (cancelable) vs jobs already removed from the queue.

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


def scan_complete_logs() -> list[str]:
    """Return list of job IDs whose log files contain JOB_COMPLETE."""
    complete = []
    for log_file in sorted(LOGS_DIR.glob("*.out")):
        job_id = log_file.stem
        if "JOB_COMPLETE" in log_file.read_text(errors="replace"):
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

    queued_ids = get_all_slurm_jobs()
    complete_ids = scan_complete_logs()
    complete_set = set(complete_ids)

    cancelable = sorted(complete_set & queued_ids)
    already_done = sorted(complete_set - queued_ids)

    print("SLURM / log summary\n")
    print(f"  Jobs currently in queue (squeue --me):     {len(queued_ids)}")
    print(f"  Logs with JOB_COMPLETE:                    {len(complete_set)}")
    print(
        f"  JOB_COMPLETE and still in queue (cancel): {len(cancelable)}"
    )
    print(
        f"  JOB_COMPLETE but gone from queue (done):  {len(already_done)}"
    )

    if cancelable:
        print("\nCancelable job IDs (log complete, still in queue):")
        for job_id in cancelable:
            print(f"    {job_id}")
    if already_done:
        print("\nAlready finished (JOB_COMPLETE, not in queue):")
        for job_id in already_done:
            print(f"    {job_id}")

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
