#!/usr/bin/env python3
"""
Check running SLURM jobs and identify ones that have already finished.
A job is considered complete when its log file contains "JOB_COMPLETE".
Optionally cancel completed jobs to free cluster resources.

Usage:
    python scripts/check_jobs.py              # report only
    python scripts/check_jobs.py --cancel     # cancel completed jobs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import DATA_DIR

PROJECT_ROOT = DATA_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def get_running_jobs() -> list[dict]:
    """Return list of RUNNING jobs for current user via squeue."""
    result = subprocess.run(
        ["squeue", "--me", "--format=%i %j %T", "--noheader"],
        capture_output=True,
        text=True,
    )
    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "RUNNING":
            jobs.append({"job_id": parts[0], "name": parts[1]})
    return jobs


def job_is_complete(job_id: str) -> bool:
    """Return True if the job log contains the JOB_COMPLETE sentinel."""
    log_file = LOGS_DIR / f"{job_id}.out"
    if not log_file.exists():
        return False
    return "JOB_COMPLETE" in log_file.read_text(errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancel", action="store_true", help="Cancel jobs whose log contains JOB_COMPLETE"
    )
    args = parser.parse_args()

    jobs = get_running_jobs()
    if not jobs:
        print("No running jobs found.")
        return

    print(f"Found {len(jobs)} running jobs.\n")
    to_cancel = []

    for job in jobs:
        complete = job_is_complete(job["job_id"])
        status = "DONE   " if complete else "RUNNING"
        print(f"  [{status}] job={job['job_id']:>10}  name={job['name']}")
        if complete:
            to_cancel.append(job["job_id"])

    if not to_cancel:
        print("\nNo completed jobs to cancel.")
        return

    print(f"\n{len(to_cancel)} completed jobs found.")
    if args.cancel:
        for job_id in to_cancel:
            subprocess.run(["scancel", job_id])
            print(f"  Cancelled {job_id}")
    else:
        print("Run with --cancel to cancel them.")
        print(f"  scancel {' '.join(to_cancel)}")


if __name__ == "__main__":
    main()
