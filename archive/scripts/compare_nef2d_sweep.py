#!/usr/bin/env python3
"""Compare NEF2d counting sweep results. Usage: python scripts/compare_nef2d_sweep.py"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", type=str, default="nef2d_sweep")
    args = parser.parse_args()
    folder = RUNS_DIR / args.out_folder
    files = sorted(folder.glob("counting_test_*.pkl"))
    if not files:
        print(f"No files found in {folder}")
        return
    rows = []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        rows.append(d)
    rows.sort(key=lambda d: d["mean_rmse_alpha"])
    hdr = (
        f"{'radius_c':>10} {'n_count':>8} "
        f"{'rmse_count':>12} {'rmse_alpha':>12} {'train_s':>10} {'sim_s':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for d in rows:
        print(
            f"{d['radius_c']:>10.0f} {d['n_neurons_counting']:>8d} "
            f"{d['mean_rmse_count']:>12.4f} {d['mean_rmse_alpha']:>12.4f} "
            f"{d.get('train_time_s', 0):>10.1f} {d.get('sim_time_s', 0):>8.1f}"
        )


if __name__ == "__main__":
    main()
