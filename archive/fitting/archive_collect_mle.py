"""
MLE-only collection functions extracted from fitting/collect.py
(2026-09-05), completing the MLE-pipeline retirement whose main decision
and rationale live in docs/DECISIONS.md ("State-noise models,
NoisyCounting, and their MLE/NLL pipelines retired from active analysis")
and archive/fitting/archive_fit_mle.py. `fitting/collect.py` itself
remains active (it still serves the RMSE/NLL `params`/`responses`/
`activities` collection paths) — only these three MLE-only functions,
their `--type` choices, and the CLI args that existed solely to serve
them were removed from it and archived here, unchanged.

Extracted:
- `_collect_mle_params`: concatenates per-pid `*_params_mle.pkl` /
  `*_performance_mle.pkl` files into combined run-level files.
- `_generate_mle_responses`: simulates response trajectories at each
  pid's best-fit MLE params (via `scripts.build_sim_db.simulate_param_point`
  / `ALL_SEQUENCES`, now archived at `archive/scripts/build_sim_db.py`).
- `_collect_mle_from_db`: scans the full `data/sim_db/` simulation
  database (via `scripts.build_sim_db.params_hash`, `fitting.losses.
  compute_sim_db_loss`) to find each pid's best-fit params exhaustively,
  then calls `_generate_mle_responses`.

Also archived: the `--type` choices `"mle_params"`, `"mle_responses"`,
`"mle_from_db"` and the CLI args `--model_type`, `--dataset`,
`--db_folder` that existed only to serve these three branches (confirmed
against `fitting/collect.py`'s remaining `params`/`responses`/
`activities` branches, none of which read `args.model_type`,
`args.dataset`, or `args.db_folder`).

How to restore: merge `_collect_mle_params`/`_generate_mle_responses`/
`_collect_mle_from_db` back into `fitting/collect.py`, re-add the three
`--type` choices and their `elif` dispatch branches in `main()`, and
re-add the `--model_type`/`--dataset`/`--db_folder` argparse args shown
below. Also requires `scripts/build_sim_db.py` to be restored from
`archive/scripts/build_sim_db.py` (NOT `archive/scripts/
build_sim_db_early_draft.py` — see that file's own header for why there
are two archived versions).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from utils.paths import RUNS_DIR, data_path


def _collect_mle_params(run_folder: Path, dataset: str,
                        model_type: str) -> None:
    """Collect per-pid MLE params/performance into single combined files.

    Globs {model_type}_{dataset}_{pid}_params_mle.pkl and
    {model_type}_{dataset}_{pid}_performance_mle.pkl from run_folder,
    concatenates them, and saves combined files.

    Output:
        {model_type}_{dataset}_params_mle.pkl
        {model_type}_{dataset}_performance_mle.pkl
    """
    params_files = sorted(
        run_folder.glob(f"{model_type}_{dataset}_*_params_mle.pkl")
    )
    perf_files = sorted(
        run_folder.glob(f"{model_type}_{dataset}_*_performance_mle.pkl")
    )

    if not params_files:
        print(f"No MLE params files found for {model_type}/{dataset} in {run_folder}")
        return

    params_df = pd.concat([pd.read_pickle(f) for f in params_files],
                           ignore_index=True)
    out_params = run_folder / f"{model_type}_{dataset}_params_mle.pkl"
    params_df.to_pickle(out_params)
    print(f"Collected {len(params_files)} params -> {out_params.name}")
    print(params_df.sort_values("pid")[
        ["pid", "mle_loss"] +
        [c for c in params_df.columns
         if c not in ("pid", "mle_loss", "model_type", "dataset")]
    ].to_string(index=False))

    if perf_files:
        perf_df = pd.concat([pd.read_pickle(f) for f in perf_files],
                             ignore_index=True)
        out_perf = run_folder / f"{model_type}_{dataset}_performance_mle.pkl"
        perf_df.to_pickle(out_perf)
        print(f"Collected {len(perf_files)} performance -> {out_perf.name}")



def _generate_mle_responses(run_folder: Path, dataset: str,
                             model_type: str, n_sims: int = 200,
                             db_folder: str = "data/sim_db") -> None:
    """Generate responses at MLE best-fit params for each pid.

    Reads {model_type}_{dataset}_params_mle.pkl, simulates n_sims trajectories
    per pid at their best params, and saves a combined responses file matching
    the format of RMSE-fitted {model_type}_{dataset}_responses.pkl.

    The responses are the mean trajectory across n_sims simulations, so they
    represent the model's expected response given its best-fit distribution.

    Output: {model_type}_{dataset}_responses_mle.pkl
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.build_sim_db import simulate_param_point, ALL_SEQUENCES
    from utils.paths import data_path

    params_path = run_folder / f"{model_type}_{dataset}_params_mle.pkl"
    if not params_path.exists():
        print(f"No MLE params file found: {params_path}")
        return

    all_params = pd.read_pickle(params_path)
    human      = pd.read_pickle(data_path(f"{dataset}.pkl"))
    db_dir     = Path(db_folder)

    print(f"Generating MLE responses: {model_type} {dataset} "
          f"({len(all_params)} pids, n_sims={n_sims})")

    rows = []
    for _, prow in all_params.iterrows():
        pid    = int(prow["pid"])
        params = {k: v for k, v in prow.items()
                  if k not in ("mle_loss",)}
        params["model_type"] = model_type
        params["dataset"]    = dataset
        params["pid"]        = pid

        # Simulate at best params (uses cache if already in db)
        db_path = simulate_param_point(
            model_type=model_type,
            params=params,
            n_sims=n_sims,
            db_dir=db_dir,
            run_folder=dataset,
            overwrite=False,
        )
        db = pd.read_pickle(db_path)["data"]   # {seq: (n_sims, n_obs)}

        # Reconstruct per-trial responses: one row per (sim, trial, obs)
        # so that qid_resp_std can measure within-group variability.
        # We use trial indices as sim seeds — each trial maps to one simulation.
        for trial, tdf in human[human["pid"] == pid].groupby("trial"):
            tdf  = tdf.sort_values("observation")
            seq  = tuple(tdf["value"].values)
            if seq not in db:
                continue
            trajs   = db[seq]           # (n_sims, n_obs)
            sim_idx = (int(trial) - 1) % trajs.shape[0]
            traj    = trajs[sim_idx]    # one trajectory matching this trial slot
            for obs_idx in range(len(traj)):
                rows.append({
                    "model_type":  f"{model_type}_mle",
                    "pid":         pid,
                    "trial":       trial,
                    "observation": obs_idx + 1,
                    "response":    float(traj[obs_idx]),
                })

        print(f"  pid={pid}: done")

    if not rows:
        print("No responses generated.")
        return

    out = run_folder / f"{model_type}_{dataset}_responses_mle.pkl"
    pd.DataFrame(rows).to_pickle(out)
    print(f"Saved {len(rows)} rows -> {out.name}")



def _collect_mle_from_db(
    run_folder: Path,
    dataset: str,
    model_type: str,
    db_folder: str = "data/sim_db",
) -> None:
    """Find best-fit params for every pid by scanning the full simulation database.

    For each pid, evaluates compute_sim_db_loss at every (params, responses)
    entry in the database, finds the minimum-loss entry, and saves combined
    params and performance files. Also generates response trajectories at the
    best params for each pid.

    This is the definitive collect step — it is equivalent to running n_fits=1
    on a job that has already seen the full database, but does it locally and
    exhaustively without any new simulations.

    Output:
        {model_type}_{dataset}_params_mle.pkl
        {model_type}_{dataset}_performance_mle.pkl
        {model_type}_{dataset}_responses_mle.pkl
    """
    import sys, time
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fitting.losses import compute_sim_db_loss
    from scripts.build_sim_db import params_hash
    from utils.paths import data_path

    db_dir   = Path(db_folder)
    model_dir = db_dir / model_type
    if not model_dir.exists():
        print(f"No simulation database found at {model_dir}")
        return

    db_files = sorted(
        p for p in model_dir.glob(f"{model_type}_*.pkl")
        if not p.name.startswith("checkpoint")
    )
    if not db_files:
        print(f"No simulation files found in {model_dir}")
        return

    print(f"Scanning {len(db_files)} database entries for {model_type}/{dataset}")

    human_df = pd.read_pickle(data_path(f"{dataset}.pkl"))
    all_pids = sorted(human_df["pid"].unique())
    human_pids = {pid: human_df[human_df["pid"] == pid].copy() for pid in all_pids}

    # Load each db file once, evaluate loss for all pids simultaneously
    # This avoids re-reading 4000 files 21 times (21x speedup)
    best_loss  = {pid: float("inf") for pid in all_pids}
    best_entry = {pid: None         for pid in all_pids}

    t0 = time.time()
    for i, db_path in enumerate(db_files):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(db_files)} entries scanned  ({elapsed:.0f}s elapsed)", flush=True)
        try:
            entry  = pd.read_pickle(db_path)
            params = entry["params"]
        except Exception:
            continue
        for pid in all_pids:
            try:
                loss = compute_sim_db_loss(model_type, params, human_pids[pid], db_dir)
                if loss < best_loss[pid]:
                    best_loss[pid]  = loss
                    best_entry[pid] = params
            except Exception:
                continue

    print(f"Scan complete in {time.time()-t0:.0f}s")

    best_params_rows = []
    best_perf_rows   = []
    for pid in all_pids:
        if best_entry[pid] is None:
            print(f"  pid={pid}: no valid entry found — skipping")
            continue
        print(f"  pid={pid}: best_loss={best_loss[pid]:.4f}")
        row = {k: v for k, v in best_entry[pid].items()
               if k not in ("model_type", "dataset")}
        row["model_type"] = model_type
        row["dataset"]    = dataset
        row["pid"]        = pid
        row["mle_loss"]   = best_loss[pid]
        best_params_rows.append(row)
        best_perf_rows.append({
            "pid":          pid,
            "mle_loss":     best_loss[pid],
            "n_db_entries": len(db_files),
        })

    if not best_params_rows:
        print("No valid results — aborting")
        return

    params_df = pd.DataFrame(best_params_rows)
    perf_df   = pd.DataFrame(best_perf_rows)

    out_params = run_folder / f"{model_type}_{dataset}_params_mle.pkl"
    out_perf   = run_folder / f"{model_type}_{dataset}_performance_mle.pkl"
    params_df.to_pickle(out_params)
    perf_df.to_pickle(out_perf)
    print(f"Saved {out_params.name}  ({len(params_df)} pids)")
    print(f"Saved {out_perf.name}")
    print(params_df.sort_values("pid")[
        ["pid", "mle_loss"] +
        [c for c in params_df.columns
         if c not in ("pid", "mle_loss", "model_type", "dataset")]
    ].to_string(index=False))

    # Generate responses at best params
    print("\nGenerating responses at best-fit params...")
    _generate_mle_responses(run_folder, dataset, model_type, db_folder=db_folder)


# ── CLI (for restoring into fitting/collect.py's own main(), not run standalone) ──

def main() -> None:
    """Mirrors the MLE branch of fitting.collect's former main() exactly, for
    reference when restoring. Not wired to fitting/collect.py's own CLI
    anymore -- see this file's header for how to merge back in."""
    parser = argparse.ArgumentParser(prog="fitting.collect (MLE branch, archived)")
    parser.add_argument("run_folder", help="Run folder name under data/runs/")
    parser.add_argument(
        "--type",
        type=str,
        choices=["mle_params", "mle_responses", "mle_from_db"],
        required=True,
    )
    parser.add_argument("--model_type", type=str, default=None,
                        help="Model type for mle_params collection")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset for mle_params collection")
    parser.add_argument("--db_folder", type=str, default="data/sim_db",
                        help="Simulation database folder for mle_responses")
    args = parser.parse_args()

    run_folder = RUNS_DIR / args.run_folder
    if not args.model_type or not args.dataset:
        parser.error("--model_type and --dataset required")
    if args.type == "mle_params":
        _collect_mle_params(run_folder, args.dataset, args.model_type)
    elif args.type == "mle_responses":
        _generate_mle_responses(run_folder, args.dataset, args.model_type,
                                 db_folder=args.db_folder)
    elif args.type == "mle_from_db":
        _collect_mle_from_db(run_folder, args.dataset, args.model_type,
                              db_folder=args.db_folder)


if __name__ == "__main__":
    main()
