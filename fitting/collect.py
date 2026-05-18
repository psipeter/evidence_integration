"""
Aggregate per-participant fitting outputs into combined run-level files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paths import RUNS_DIR, data_path


def _load_groups(run_folder: Path) -> dict[tuple[str, str], list[int]]:
    config_path = run_folder / "run_config.json"
    if not config_path.exists():
        print(f"No run_config.json found in {run_folder}", file=sys.stderr)
        return {}
    config = json.loads(config_path.read_text())
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for job in config.get("jobs", []):
        groups[(job["dataset"], job["model_type"])].append(int(job["pid"]))
    return groups


def _collect_params(run_folder: Path) -> None:
    groups = _load_groups(run_folder)
    for (dataset, model_type), pids in groups.items():
        params_dfs: list[pd.DataFrame] = []
        perf_dfs: list[pd.DataFrame] = []
        folds_dfs: list[pd.DataFrame] = []
        for pid in pids:
            pp = run_folder / f"{model_type}_{dataset}_{pid}_params.pkl"
            fp = run_folder / f"{model_type}_{dataset}_{pid}_performance.pkl"
            cp = run_folder / f"{model_type}_{dataset}_{pid}_folds.pkl"
            if pp.exists():
                params_dfs.append(pd.read_pickle(pp))
            if fp.exists():
                perf_dfs.append(pd.read_pickle(fp))
            if cp.exists():
                folds_dfs.append(pd.read_pickle(cp))

        if params_dfs:
            out = run_folder / f"{model_type}_{dataset}_params.pkl"
            df = pd.concat(params_dfs, ignore_index=True)
            df.to_pickle(out)
            print(f"Collected {len(params_dfs)} -> {out} ({df.shape})")
        if perf_dfs:
            out = run_folder / f"{model_type}_{dataset}_performance.pkl"
            df = pd.concat(perf_dfs, ignore_index=True)
            df.to_pickle(out)
            print(f"Collected {len(perf_dfs)} -> {out} ({df.shape})")
        if folds_dfs:
            out = run_folder / f"{model_type}_{dataset}_folds.pkl"
            df = pd.concat(folds_dfs, ignore_index=True)
            df.to_pickle(out)
            print(f"Collected {len(folds_dfs)} -> {out} ({df.shape})")


def _collect_responses(run_folder: Path) -> None:
    groups = _load_groups(run_folder)
    available_datasets = sorted({dataset for dataset, _ in groups.keys()})
    available_model_types = sorted({model_type for _, model_type in groups.keys()})
    for dataset in available_datasets:
        for model_type in available_model_types:
            files = sorted(run_folder.glob(f"{model_type}_{dataset}_*_responses.pkl"))
            if files:
                df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
                out = run_folder / f"{model_type}_{dataset}_responses.pkl"
                df.to_pickle(out)
                print(f"Collected {len(files)} -> {out} ({df.shape})")


def _collect_activities(run_folder: Path, ensembles: list[str], timing: str) -> None:
    out_dir = run_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted({dataset for dataset, _ in _load_groups(run_folder).keys()})
    for dataset in datasets:
        for ens_name in ensembles:
            if timing == "once_per_dt":
                npz_files = sorted(
                    out_dir.glob(f"activities_windowed_{ens_name}_{dataset}_*.npz")
                )
                if npz_files:
                    arrays = [np.load(f)["activities"] for f in npz_files]
                    pid_ids = np.array([int(f.stem.split("_")[-1]) for f in npz_files])
                    max_trials = max(a.shape[0] for a in arrays)
                    padded = []
                    for a in arrays:
                        n_trials = a.shape[0]
                        if n_trials < max_trials:
                            pad_shape = (max_trials - n_trials,) + a.shape[1:]
                            padding = np.full(pad_shape, np.nan, dtype=np.float32)
                            a = np.concatenate([a, padding], axis=0)
                        padded.append(a.astype(np.float32))
                    combined = np.stack(padded, axis=0)
                    out_path = out_dir / f"activities_windowed_{ens_name}_{dataset}.npz"
                    np.savez_compressed(out_path, activities=combined, pid_ids=pid_ids)
                    print(
                        f"Collected {len(npz_files)} -> {out_path} shape {combined.shape}"
                    )
            else:
                activity_files = sorted(
                    out_dir.glob(f"activities_{ens_name}_{dataset}_*.pkl")
                )
                if activity_files:
                    activities_df = pd.concat(
                        [pd.read_pickle(f) for f in activity_files], ignore_index=True
                    )
                    activities_out = out_dir / f"activities_{ens_name}_{dataset}.pkl"
                    activities_df.to_pickle(activities_out)
                    print(
                        f"Collected {len(activity_files)} -> {activities_out} "
                        f"({activities_df.shape})"
                    )

            encoder_files = sorted(out_dir.glob(f"encoders_{ens_name}_{dataset}_*.pkl"))
            if encoder_files:
                encoders_df = pd.concat(
                    [pd.read_pickle(f) for f in encoder_files], ignore_index=True
                )
                encoders_out = out_dir / f"encoders_{ens_name}_{dataset}.pkl"
                encoders_df.to_pickle(encoders_out)
                print(f"Collected {len(encoder_files)} -> {encoders_out} ({encoders_df.shape})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fitting.collect")
    parser.add_argument("run_folder", help="Run folder name under data/runs/")
    parser.add_argument(
        "--type",
        type=str,
        choices=["params", "responses", "activities"],
        required=True,
    )
    parser.add_argument("--ensembles", nargs="+", default=["error"])
    parser.add_argument("--timing", type=str, default="once_per_obs")
    args = parser.parse_args()

    run_folder = RUNS_DIR / args.run_folder
    if args.type == "params":
        _collect_params(run_folder)
    elif args.type == "responses":
        _collect_responses(run_folder)
    else:
        _collect_activities(run_folder, args.ensembles, args.timing)


if __name__ == "__main__":
    main()
