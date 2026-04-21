"""
Aggregate per-participant fitting outputs into combined run-level files.

Usage::

    python -m fitting.collect Apr21_1200pm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from utils.paths import RUNS_DIR


def _collect(run_folder: Path) -> None:
    """
    Aggregate per-participant files in run_folder into combined files.
    Reads run_config.json to know which jobs were run.
    """
    config_path = run_folder / "run_config.json"
    if not config_path.exists():
        print(f"No run_config.json found in {run_folder}", file=sys.stderr)
        return
    config = json.loads(config_path.read_text())
    jobs = config["jobs"]

    groups: dict[tuple[str, str], list] = defaultdict(list)
    for job in jobs:
        groups[(job["dataset"], job["model_type"])].append(job["pid"])

    for (ds, mt), pids in groups.items():
        responses_dfs, params_dfs, perf_dfs = [], [], []
        for pid in pids:
            pid = int(pid)
            rp = run_folder / f"{mt}_{ds}_{pid}_responses.pkl"
            pp = run_folder / f"{mt}_{ds}_{pid}_params.pkl"
            fp = run_folder / f"{mt}_{ds}_{pid}_performance.pkl"
            if not all(p.exists() for p in [rp, pp, fp]):
                print(f"Warning: missing files for {mt} {ds} pid={pid}, skipping")
                continue
            responses_dfs.append(pd.read_pickle(rp))
            params_dfs.append(pd.read_pickle(pp))
            perf_dfs.append(pd.read_pickle(fp))

        def _save(dfs: list[pd.DataFrame], name: str) -> None:
            if not dfs:
                return
            df = pd.concat(dfs, ignore_index=True)
            df.to_pickle(run_folder / name)
            print(f"  Saved {name}: {df.shape}")

        print(f"Collecting {mt} {ds}...")
        _save(responses_dfs, f"{mt}_{ds}_responses.pkl")
        _save(params_dfs, f"{mt}_{ds}_params.pkl")
        _save(perf_dfs, f"{mt}_{ds}_performance.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fitting.collect")
    parser.add_argument("run_folder", help="Run folder name under data/runs/")
    args = parser.parse_args()
    _collect(RUNS_DIR / args.run_folder)


if __name__ == "__main__":
    main()
