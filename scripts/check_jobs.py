#!/usr/bin/env python3
"""
Check running SLURM jobs and identify ones that have already saved all
required output files (params, responses, performance). Optionally cancel
them to free up cluster resources.

Usage:
    python scripts/check_jobs.py [--cancel] [--run_folder NAME]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import DATA_DIR, RUNS_DIR

PROJECT_ROOT = DATA_DIR.parent


def get_running_jobs() -> list[dict]:
    """
    Return list of running jobs for current user via squeue.
    Each dict has: job_id, name, state, node.
    """
    result = subprocess.run(
        ["squeue", "--me", "--format=%i %j %T %R", "--noheader"],
        capture_output=True,
        text=True,
    )
    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            jobs.append(
                {
                    "job_id": parts[0],
                    "name": parts[1],
                    "state": parts[2],
                    "reason": parts[3] if len(parts) > 3 else "",
                }
            )
    return [j for j in jobs if j["state"] == "RUNNING"]


def find_run_folder_for_job(job_id: str) -> Path | None:
    """
    Search logs/ for {job_id}.out and extract run_folder from the log.
    Returns the run folder Path if found, else None.
    """
    log_file = PROJECT_ROOT / "logs" / f"{job_id}.out"
    if not log_file.exists():
        return None
    # look for run_folder path in log output
    text = log_file.read_text(errors="replace")
    for line in text.splitlines():
        if "data/runs/" in line:
            # extract folder name from path like data/runs/response_loss/...
            parts = line.split("data/runs/")
            if len(parts) > 1:
                folder_name = parts[1].split("/")[0]
                candidate = RUNS_DIR / folder_name
                if candidate.exists():
                    return candidate
    return None


def parse_job_name(name: str) -> tuple[str, str, str] | None:
    """
    Parse model_type, dataset, pid from job script name like
    'NEF_recurrent_yoo_2.sh' or 'ADM_carrabin_15.sh'.
    Returns (model_type, dataset, pid) or None if unparseable.
    """
    name = name.replace(".sh", "")
    datasets = ("carrabin", "jiang", "yoo")
    for ds in datasets:
        marker = f"_{ds}_"
        if marker in name:
            idx = name.index(marker)
            model_type = name[:idx]
            pid = name[idx + len(marker) :]
            if model_type and pid.isdigit():
                return model_type, ds, pid
    return None


def job_is_complete(job: dict, run_folder_filter: str | None) -> tuple[bool, str]:
    """
    Check if a job has saved all required output files.
    Returns (is_complete, reason_string).
    """
    log_file = PROJECT_ROOT / "logs" / f"{job['job_id']}.out"
    if not log_file.exists():
        return False, "no log file found"

    parsed = parse_job_name(job["name"])
    if parsed is None:
        return False, f"could not parse job name: {job['name']}"
    model_type, dataset, pid = parsed

    candidates = []
    for folder in RUNS_DIR.iterdir():
        if not folder.is_dir():
            continue
        if run_folder_filter and folder.name != run_folder_filter:
            continue
        params_f = folder / f"{model_type}_{dataset}_{pid}_params.pkl"
        if params_f.exists():
            candidates.append(folder)

    if not candidates:
        return False, f"{model_type} {dataset} pid={pid} — no params file found"

    for folder in candidates:
        params_f = folder / f"{model_type}_{dataset}_{pid}_params.pkl"
        responses_f = folder / f"{model_type}_{dataset}_{pid}_responses.pkl"
        perf_f = folder / f"{model_type}_{dataset}_{pid}_performance.pkl"
        missing = [f.name for f in [params_f, responses_f, perf_f] if not f.exists()]
        if not missing:
            return True, f"{model_type} {dataset} pid={pid} in {folder.name}"

    return False, f"{model_type} {dataset} pid={pid} — missing output files"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Cancel jobs that have completed their output",
    )
    parser.add_argument(
        "--run_folder",
        type=str,
        default=None,
        help="Only check jobs from this run folder",
    )
    args = parser.parse_args()

    jobs = get_running_jobs()
    if not jobs:
        print("No running jobs found.")
        return

    print(f"Found {len(jobs)} running jobs.\n")
    to_cancel = []

    for job in jobs:
        complete, reason = job_is_complete(job, args.run_folder)
        status = "DONE  " if complete else "RUNNING"
        print(f"  [{status}] job={job['job_id']:>10}  name={job['name']:<20}  {reason}")
        if complete:
            to_cancel.append(job["job_id"])

    if not to_cancel:
        print("\nNo completed jobs to cancel.")
        return

    print(f"\n{len(to_cancel)} jobs appear complete.")
    if args.cancel:
        for job_id in to_cancel:
            subprocess.run(["scancel", job_id], check=False)
            print(f"  Cancelled {job_id}")
    else:
        print("Run with --cancel to cancel them.")
        print(f"  scancel {' '.join(to_cancel)}")


if __name__ == "__main__":
    main()
