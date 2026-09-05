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

from utils.paths import RUNS_DIR, dataset_stem


def _load_groups(run_folder: Path) -> dict[tuple[str, str], list[int]]:
    """Map (dataset_stem, model_type) -> pids, read from run_config.json.

    Keyed on the STEM (dataset family + optional --datafile suffix, see
    utils.paths.dataset_stem, PLUS a `_nll` suffix when the job used
    --loss nll) rather than the bare dataset name, because that is what every
    output filename actually uses. This lets one run folder hold fits against
    several builds of the same dataset, and both loss functions, without them
    colliding.

    The `_nll` suffix mirrors fitting.fit's OWN file_stem construction exactly
    (`f"{stem}_nll" if loss_fn == "nll" else stem`) and fitting.submit's
    corresponding job-script naming -- this function drifting from either of
    those silently means every NLL job's outputs exist on disk but never get
    aggregated, since the stem this function computes would not match the
    actual filenames. That happened once already: this function originally had
    no knowledge of loss_fn at all, so it built the RMSE stem for every job
    regardless, and _collect_params/_collect_responses silently collected
    nothing for every NLL fit even though the run_config.json entries and the
    per-pid files were both present.
    """
    config_path = run_folder / "run_config.json"
    if not config_path.exists():
        print(f"No run_config.json found in {run_folder}", file=sys.stderr)
        return {}
    config = json.loads(config_path.read_text())
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for job in config.get("jobs", []):
        stem = dataset_stem(job["dataset"], job.get("datafile"))
        if job.get("loss_fn") == "nll":
            stem = f"{stem}_nll"
        groups[(stem, job["model_type"])].append(int(job["pid"]))
    return groups


def _collect_params(run_folder: Path) -> None:
    groups = _load_groups(run_folder)
    for (stem, model_type), pids in groups.items():
        dataset = stem  # filenames are stem-based; see _load_groups
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
    """Concatenate per-pid response files into one file per (stem, model_type).

    Driven by the explicit pid list from run_config.json rather than a
    `{model_type}_{stem}_*_responses.pkl` glob: with an UNSUFFIXED stem the `*`
    (meant to match only a pid) also matches a suffixed stem's files, so
    `Mean_soltani_numbers_*_responses.pkl` would swallow
    `Mean_soltani_numbers_pilot5_3_responses.pkl` and silently merge two
    different data versions into one output.
    """
    groups = _load_groups(run_folder)
    for (stem, model_type), pids in sorted(groups.items()):
        files = [
            p for p in (
                run_folder / f"{model_type}_{stem}_{pid}_responses.pkl"
                for pid in sorted(set(pids))
            )
            if p.exists()
        ]
        if files:
            df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
            out = run_folder / f"{model_type}_{stem}_responses.pkl"
            df.to_pickle(out)
            print(f"Collected {len(files)} -> {out} ({df.shape})")


def _collect_activities(run_folder: Path, ensembles: list[str], timing: str) -> None:
    """Concatenate per-pid NEF activity files into one file per (stem, ensemble).

    Driven by the explicit pid list from run_config.json rather than a
    `activities_{ens}_{stem}_*` glob, for the same reason as
    _collect_responses: with an UNSUFFIXED stem the `*` (meant to match only a
    pid) also matches a suffixed stem's files, so
    `activities_windowed_error_soltani_numbers_*.npz` would swallow
    `activities_windowed_error_soltani_numbers_complete_pairs_3.npz` and merge
    two different data versions into one output. That was unreachable while NEF
    was unwired for the soltani datasets; it is reachable now.
    """
    out_dir = run_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _load_groups(run_folder)
    # Activity filenames carry no model_type, so take the union of pids across
    # every model_type for a stem and let the existence check below filter to
    # the ones that actually produced activities (NEF-only in practice).
    pids_by_stem: dict[str, set[int]] = defaultdict(set)
    for (stem, _model_type), pids in groups.items():
        pids_by_stem[stem].update(int(p) for p in pids)
    for dataset in sorted(pids_by_stem):
        stem_pids = sorted(pids_by_stem[dataset])
        for ens_name in ensembles:
            if timing == "once_per_dt":
                pid_paths = [
                    (pid, out_dir / f"activities_windowed_{ens_name}_{dataset}_{pid}.npz")
                    for pid in stem_pids
                ]
                pid_paths = [(pid, f) for pid, f in pid_paths if f.exists()]
                npz_files = [f for _, f in pid_paths]
                if npz_files:
                    arrays = [np.load(f)["activities"] for f in npz_files]
                    # pid_ids comes from run_config.json, not from parsing the
                    # filename tail -- a suffixed stem puts extra underscored
                    # tokens before the pid, so split("_")[-1] was only correct
                    # for unsuffixed names.
                    pid_ids = np.array([pid for pid, _ in pid_paths])
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
                activity_files = [
                    f for f in (
                        out_dir / f"activities_{ens_name}_{dataset}_{pid}.pkl"
                        for pid in stem_pids
                    )
                    if f.exists()
                ]
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
