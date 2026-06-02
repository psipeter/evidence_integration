#!/usr/bin/env python3
"""
Generate supplementary NEF simulation data for figure_carrabin.py.

This script runs computationally expensive NEF simulations that produce data
for the bottom panels of figure_carrabin. Run simulations on the cluster, then
collect per-pid outputs into combined pickle files before plotting via
``scripts/figure_carrabin.py``.

The script is carrabin-specific: it reads ``data/carrabin.pkl``, uses the
``qid`` column, and loads NEF_carrabin params from the run folder.

=============================================================================
CLUSTER USAGE — run each .sh file as a single command, then collect
=============================================================================

Two experiments. Each has a corresponding job submission script in jobs/.

----------------------------------------------------------------------
1. probe_pids  →  bash jobs/submit_probe_pids.sh
----------------------------------------------------------------------
Runs the full NEF simulation for all carrabin pids with probe saving enabled.
One SLURM job per pid.

  Submit all jobs:
      bash jobs/submit_probe_pids.sh

  Collect (after all jobs complete):
      python scripts/extras_carrabin.py --experiment probe_pids --mode collect --out_folder refit

----------------------------------------------------------------------
2. n_neurons_scan  (panel H)
----------------------------------------------------------------------
Scans n_neurons for a representative (alpha_0, lambda_) and pid.
For each n_neurons value, n_neurons_counting is set to the same value.
Precomputes counting activities if needed, simulates all trials,
fits an RNN, and saves sigma and std(PE) to a single pkl file.

  Run (locally, sequential):
      python scripts/extras_carrabin.py --experiment n_neurons_scan \
          --run_folder carrabin --out_folder carrabin

  Output: data/runs/<out_folder>/n_neurons_scan.pkl

----------------------------------------------------------------------
3. pe_readout  (panels F, G)
----------------------------------------------------------------------
Runs the standard NEF simulation for one pid, extracting only the decoded
prediction error at the readout moment for each (trial, observation).
Output is a small DataFrame: pid, trial, observation, qid, pe_at_readout.
One job per pid; very fast (same cost as standard fitting run).

  Run:
      python scripts/extras_carrabin.py --experiment pe_readout \
          --pid 1 --run_folder carrabin --out_folder carrabin

  Collect (after all pids complete):
      python scripts/extras_carrabin.py --experiment pe_readout \
          --mode collect --out_folder carrabin

  Output: data/runs/<out_folder>/pe_readout_NEF_carrabin_<pid>.pkl
          data/runs/<out_folder>/pe_readout_NEF_carrabin.pkl  (combined)

----------------------------------------------------------------------
4. probe_timeseries  (panel E)
----------------------------------------------------------------------
Runs the full NEF simulation for one pid with once_per_dt probe saving.
Builds a flat DataFrame with columns:
  pid, trial, observation, qid, t_within_obs, decoded_pe, decoded_value

  Run (single pid, locally or on cluster):
      python scripts/extras_carrabin.py --experiment probe_timeseries \
          --pid 1 --run_folder carrabin --out_folder carrabin

  Output: data/runs/<out_folder>/probe_timeseries_NEF_carrabin_<pid>.pkl

=============================================================================
OUTPUT FILES  (written to data/runs/<out_folder>/)
=============================================================================

Per-pid files (from ``--mode run``):

probe_pids experiment:
  probe_NEF_carrabin_<pid>.pkl          — raw NEF probe timeseries per pid

n_neurons_scan experiment:
  n_neurons_scan.pkl  — sigma and std(PE) per n_neurons value

pe_readout experiment:
  pe_readout_NEF_carrabin_<pid>.pkl     — per-pid: pid, trial, observation, qid, pe_at_readout

probe_timeseries experiment:
  probe_timeseries_NEF_carrabin_<pid>.pkl  — flat DataFrame: pid, trial, observation,
                                             qid, t_within_obs, decoded_pe, decoded_value

Combined files (from ``--mode collect``):
  probe_pids_carrabin.pkl         — all probe_pids probes
  scan_responses_carrabin.pkl     — combined compact scan rows

Per-pid files are read by ``scripts/figure_carrabin.py`` for the bottom panels.
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import RUNS_DIR, data_path

N_NEURONS_LIST = [25, 50, 100, 200, 400]
READOUT_OFFSET = 0.5  # seconds into observation window for readout


def _run_probe_pids_simulate(
    pids: list[int],
    run_folder: Path,
    out_folder: str,
) -> None:
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, run as nef_run

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in pids:
        params = pd.read_pickle(
            run_folder / f"NEF_carrabin_{pid}_params.pkl"
        ).iloc[0].to_dict()
        fixed = MODEL_PARAMS["carrabin"]["NEF"].get("fixed", {})
        params = {**PARAM_DEFAULTS, **fixed, **params}
        params["nef_type"] = "recurrent"
        params["dataset"] = "carrabin"
        params["model_type"] = "NEF"
        print(
            f"Running pid={pid} (alpha_0={params['alpha_0']:.3f}, "
            f"lambda_={params['lambda_']:.3f})..."
        )
        nef_run(params, save_probes=True)
        src = data_path(f"probe_NEF_carrabin_{pid}.pkl")
        dst = out_dir / f"probe_NEF_carrabin_{pid}.pkl"
        if src.exists():
            Path(src).rename(dst)
            print(f"  Saved to {dst}")


def _run_n_neurons_scan(
    pid: int,
    alpha_0: float,
    lambda_: float,
    n_neurons_list: list[int],
    run_folder: Path,
    out_folder: str,
) -> None:
    """Scan n_neurons for panel H.

    For each n_neurons value (n_neurons_counting set to same value):
    - Precompute counting activities if not already cached
    - Simulate all trials for the given pid/alpha_0/lambda_
    - Compute std(PE at readout) within (obs, qid) groups
    - Fit RNN to responses and compute sigma_NEF
    - Save results to n_neurons_scan.pkl

    Output columns: n_neurons, sigma, pe_std, cv_rmse, elapsed_s
    """
    import pickle
    import time as _time
    from fitting.model_params import _NEF_FIXED
    from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
    from models.RNN import fit as rnn_fit
    from models.counting_integrator import (
        fast_decode, precompute_activities,
    )
    from utils.carrabin_transform import apply_carrabin_transform

    out_dir  = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"n_neurons_scan_{pid}.pkl"
    if out_path.exists():
        print(f"Already exists: {out_path.name} — skipping (delete to rerun)")
        return

    human   = pd.read_pickle(data_path("carrabin.pkl"))
    h_pid   = human[human["pid"] == pid]
    qid_map = (
        h_pid[["trial", "observation", "qid"]]
        .drop_duplicates()
        .set_index(["trial", "observation"])["qid"]
    )
    n_trials = h_pid["trial"].nunique()
    n_obs    = int(h_pid["observation"].max())

    print(f"n_neurons scan  pid={pid}  alpha_0={alpha_0:.3f}  lambda_={lambda_:.3f}")
    print(f"n_neurons values: {n_neurons_list}\n")

    results = []

    for n_neurons in n_neurons_list:
        print(f"=== n_neurons={n_neurons} ===", flush=True)

        params = {
            **PARAM_DEFAULTS,
            **_NEF_FIXED,
            "model_type":         "NEF",
            "dataset":            "carrabin",
            "pid":                pid,
            "alpha_0":            alpha_0,
            "lambda_":            lambda_,
            "n_neurons":          n_neurons,
            "n_neurons_counting": n_neurons,
            "radius_c":           5,
        }

        # Load or precompute counting activities
        act_path = data_path(f"counting_activities_n{n_neurons}_nc{n_neurons}.pkl")
        if act_path.exists():
            with open(act_path, "rb") as f:
                activity_map = pickle.load(f)
            print(f"  Loaded activities: {act_path.name}")
        else:
            print(f"  Precomputing activities (n={n_neurons}, nc={n_neurons})...", flush=True)
            t0 = _time.time()
            precompute_activities(n_trials=200, params=params, out_path=act_path)
            print(f"  Done in {_time.time()-t0:.0f}s")
            with open(act_path, "rb") as f:
                activity_map = pickle.load(f)

        # Simulate
        t0 = _time.time()
        rows_resp, rows_pe = [], []

        for ti, (trial, trial_data) in enumerate(h_pid.groupby("trial"), 1):
            trial_data = trial_data.sort_values("observation")
            obs_values = trial_data["value"].to_numpy(dtype=float)
            p          = {**params, "seed": int(trial)}

            activity = activity_map.get(int(trial))
            if activity is not None:
                decoders = fast_decode(activity, alpha_0=alpha_0, lambda_=lambda_)
            else:
                decoders = _pretrain({**p, "base_seed": int(trial)})

            try:
                responses, probe = _simulate_trial(
                    obs_values, p, decoders, return_probes=True
                )
            except Exception as e:
                print(f"\n  Warning: trial {trial} failed ({e}), skipping")
                continue

            t_arr    = probe["t"]
            error    = probe["error"]
            pe_trace = error[:, 1] if error.ndim > 1 else error.ravel()
            t_iti_   = float(p["t_iti"])
            t_obs_   = float(p["t_obs"])
            t_step   = t_obs_ + t_iti_

            for obs in range(1, n_obs + 1):
                t_readout = t_iti_ + (obs - 1) * t_step + READOUT_OFFSET
                idx = int(np.argmin(np.abs(t_arr - t_readout)))
                qid = qid_map.get((trial, obs), np.nan)
                rows_resp.append({
                    "n_neurons":   n_neurons,
                    "trial":       int(trial),
                    "observation": obs,
                    "qid":         qid,
                    "response":    float(responses[obs - 1]),
                })
                rows_pe.append({
                    "n_neurons":   n_neurons,
                    "trial":       int(trial),
                    "observation": obs,
                    "qid":         qid,
                    "pe":          float(pe_trace[idx]),
                })

            sys.stdout.write(
                f"\r  trial {ti:3d}/{n_trials}  elapsed={_time.time()-t0:.0f}s"
            )
            sys.stdout.flush()

        print()
        elapsed = _time.time() - t0

        # std(PE) within (obs, qid) groups
        pe_df  = pd.DataFrame(rows_pe)
        pe_std = float(
            pe_df.groupby(["observation", "qid"])["pe"]
            .apply(lambda x: x.std() if len(x) >= 3 else np.nan)
            .dropna()
            .mean()
        )

        # Apply carrabin transform and fit RNN
        resp_for_rnn = apply_carrabin_transform(
            pd.DataFrame([{
                "model_type": "NEF", "pid": pid,
                "trial": r["trial"], "observation": r["observation"],
                "response": r["response"],
            } for r in rows_resp]),
            "carrabin",
        )
        source   = f"NEF_scan_n{n_neurons}"
        resp_src = out_dir / f"{source}_carrabin_responses.pkl"
        resp_for_rnn.to_pickle(resp_src)

        print(f"  Fitting RNN (source={source})...", flush=True)
        rnn_result = rnn_fit(
            pid=pid,
            source=source,
            run_folder=out_folder,
            max_epochs=5000,
            patience=300,
            verbose=False,
        )
        sigma   = float(rnn_result["sigma"]["sigma"].iloc[0])
        cv_rmse = float(rnn_result["params"]["cv_rmse"].iloc[0])

        print(f"  sigma={sigma:.4f}  pe_std={pe_std:.4f}  "
              f"cv_rmse={cv_rmse:.4f}  t={elapsed:.0f}s")
        results.append({
            "pid":       pid,
            "n_neurons": n_neurons,
            "sigma":     sigma,
            "pe_std":    pe_std,
            "cv_rmse":   cv_rmse,
            "elapsed_s": elapsed,
        })

    df = pd.DataFrame(results)
    df.to_pickle(out_path)
    print(f"\nSaved -> {out_path}")
    print(df[["n_neurons", "sigma", "pe_std", "cv_rmse"]].to_string(index=False))


def _collect_probe_pids(out_dir: Path) -> None:
    probe_files = sorted(out_dir.glob("probe_NEF_carrabin_*.pkl"))
    combined: list[dict] = []
    for path in probe_files:
        pid = int(path.stem.split("_")[-1])
        probes_raw = pd.read_pickle(path)
        probes = probes_raw if isinstance(probes_raw, list) else [probes_raw]
        for probe in probes:
            entry = dict(probe)
            entry["pid"] = pid
            combined.append(entry)

    out_path = out_dir / "probe_pids_carrabin.pkl"
    pd.to_pickle(combined, out_path)
    print(
        f"Collected {len(probe_files)} probe file(s), "
        f"{len(combined)} probe entries -> {out_path}"
    )



def _collect_n_neurons_scan(out_dir: Path) -> None:
    """Collect per-pid n_neurons_scan files into a single combined pkl."""
    files = sorted(out_dir.glob("n_neurons_scan_[0-9]*.pkl"))
    if not files:
        print("No n_neurons_scan per-pid files found.")
        return
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    out = out_dir / "n_neurons_scan.pkl"
    df.to_pickle(out)
    print(f"Collected {len(files)} files -> {out.name}")
    print(df.groupby("n_neurons")[["sigma","pe_std"]].mean().round(4).to_string())



def _run_pe_readout(
    pid: int,
    run_folder: Path,
    out_folder: str,
) -> None:
    """Run NEF for one pid; save decoded PE at readout moment per (trial, obs).

    Output: pe_readout_NEF_carrabin_<pid>.pkl
    Columns: pid, trial, observation, qid, pe_at_readout

    Same simulation cost as a standard fitting run — no full timeseries saved.
    """
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
    from models.counting_integrator import fast_decode, load_activities

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pe_readout_NEF_carrabin_{pid}.pkl"
    if out_path.exists():
        print(f"Already exists: {out_path.name} — skipping (delete to rerun)")
        return

    human = pd.read_pickle(data_path("carrabin.pkl"))
    human_pid = human[human["pid"] == pid]
    qid_map = (
        human_pid[["trial", "observation", "qid"]]
        .drop_duplicates()
        .set_index(["trial", "observation"])["qid"]
    )

    per_pid_path = run_folder / f"NEF_carrabin_{pid}_params.pkl"
    combined_path = run_folder / "NEF_carrabin_params.pkl"
    if per_pid_path.exists():
        params_df = pd.read_pickle(per_pid_path)
    elif combined_path.exists():
        params_df = pd.read_pickle(combined_path)
        params_df = params_df[params_df["pid"] == pid]
        if params_df.empty:
            raise ValueError(f"No params found for pid={pid}")
    else:
        raise FileNotFoundError(f"No params file for pid={pid} in {run_folder}")

    fixed = MODEL_PARAMS["carrabin"]["NEF"].get("fixed", {})
    params = {**PARAM_DEFAULTS, **fixed, **params_df.iloc[0].to_dict()}
    params["dataset"] = "carrabin"
    params["model_type"] = "NEF"
    params["pid"] = int(pid)

    try:
        activity_map = load_activities(
            n_neurons=int(params["n_neurons"]),
            n_neurons_counting=int(params["n_neurons_counting"]),
        )
    except FileNotFoundError:
        activity_map = None

    t_obs_  = float(params["t_obs"])
    t_iti_  = float(params["t_iti"])
    t_step  = t_obs_ + t_iti_
    dt      = float(params["dt"])
    n_obs   = int(human_pid["observation"].max())
    n_trials = human_pid["trial"].nunique()

    rows = []
    for ti, (trial, trial_data) in enumerate(human_pid.groupby("trial"), 1):
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        p = {**params, "seed": int(trial)}

        if activity_map is not None:
            activity = activity_map.get(int(trial))
            decoders = (
                fast_decode(activity, alpha_0=float(params["alpha_0"]),
                            lambda_=float(params["lambda_"]))
                if activity is not None
                else _pretrain({**p, "base_seed": int(trial)})
            )
        else:
            decoders = _pretrain({**p, "base_seed": int(trial)})

        try:
            _, probe = _simulate_trial(obs_values, p, decoders, return_probes=True)
        except Exception as e:
            print(f"\n  Warning: trial {trial} failed ({e}), skipping")
            continue

        t_arr    = probe["t"]
        error    = probe["error"]                       # (T, 2)
        pe_trace = error[:, 1] if error.ndim > 1 else error.ravel()

        for obs in range(1, n_obs + 1):
            t_readout = t_iti_ + (obs - 1) * t_step + READOUT_OFFSET
            idx = int(np.argmin(np.abs(t_arr - t_readout)))
            qid = qid_map.get((trial, obs), np.nan)
            rows.append({
                "pid":          pid,
                "trial":        int(trial),
                "observation":  obs,
                "qid":          qid,
                "pe_at_readout": float(pe_trace[idx]),
            })

        sys.stdout.write(f"\r  pid={pid}  trial {ti:3d}/{n_trials}")
        sys.stdout.flush()

    print()
    df = pd.DataFrame(rows)
    df.to_pickle(out_path)
    print(f"Saved {len(df)} rows -> {out_path.name}")


def _collect_pe_readout(out_dir: Path) -> None:
    files = sorted(out_dir.glob("pe_readout_NEF_carrabin_[0-9]*.pkl"))
    if not files:
        print("No pe_readout files found.")
        return
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    out = out_dir / "pe_readout_NEF_carrabin.pkl"
    df.to_pickle(out)
    print(f"Collected {len(files)} files -> {out.name} ({df.shape})")


def _run_probe_timeseries(
    pid: int,
    run_folder: Path,
    out_folder: str,
) -> None:
    """Run NEF for one pid with per-dt probing; save flat timeseries DataFrame.

    Output DataFrame columns:
      pid, trial, observation, qid, t_within_obs, decoded_pe, decoded_value

    - t_within_obs: time in seconds since start of observation window (0..t_obs)
    - decoded_pe:   decoded error[:, 1] — the prediction error signal
    - decoded_value: decoded value ensemble output

    All 200 trials are simulated. The qid column enables filtering to specific
    repeated sequences for reliability analysis.
    """
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
    from models.counting_integrator import fast_decode, load_activities

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"probe_timeseries_NEF_carrabin_{pid}.pkl"
    if out_path.exists():
        print(f"Already exists: {out_path.name} — skipping (delete to rerun)")
        return

    human = pd.read_pickle(data_path("carrabin.pkl"))
    human_pid = human[human["pid"] == pid]
    qid_map = (
        human_pid[["trial", "observation", "qid"]]
        .drop_duplicates()
        .set_index(["trial", "observation"])["qid"]
    )

    # Try per-pid params file first, fall back to combined params file
    per_pid_path = run_folder / f"NEF_carrabin_{pid}_params.pkl"
    combined_path = run_folder / "NEF_carrabin_params.pkl"
    if per_pid_path.exists():
        params_df = pd.read_pickle(per_pid_path)
    elif combined_path.exists():
        params_df = pd.read_pickle(combined_path)
        params_df = params_df[params_df["pid"] == pid]
        if params_df.empty:
            raise ValueError(f"No params found for pid={pid} in {combined_path}")
    else:
        raise FileNotFoundError(f"No params file found for pid={pid} in {run_folder}")
    fixed = MODEL_PARAMS["carrabin"]["NEF"].get("fixed", {})
    params = {**PARAM_DEFAULTS, **fixed, **params_df.iloc[0].to_dict()}
    params["dataset"] = "carrabin"
    params["model_type"] = "NEF"
    params["pid"] = int(pid)

    # Load precomputed counting activities if available
    try:
        activity_map = load_activities(
            n_neurons=int(params["n_neurons"]),
            n_neurons_counting=int(params["n_neurons_counting"]),
        )
    except FileNotFoundError:
        activity_map = None

    t_obs_  = float(params["t_obs"])
    t_iti_  = float(params["t_iti"])
    t_step  = t_obs_ + t_iti_
    dt      = float(params["dt"])
    n_obs   = int(human_pid["observation"].max())
    n_trials = human_pid["trial"].nunique()

    rows = []
    for ti, (trial, trial_data) in enumerate(human_pid.groupby("trial"), 1):
        trial_data  = trial_data.sort_values("observation")
        obs_values  = trial_data["value"].to_numpy(dtype=float)
        p           = {**params, "seed": int(trial)}

        if activity_map is not None:
            activity = activity_map.get(int(trial))
            if activity is not None:
                decoders = fast_decode(
                    activity,
                    alpha_0=float(params["alpha_0"]),
                    lambda_=float(params["lambda_"]),
                )
            else:
                decoders = _pretrain({**p, "base_seed": int(trial)})
        else:
            decoders = _pretrain({**p, "base_seed": int(trial)})

        _, probe = _simulate_trial(obs_values, p, decoders, return_probes=True)

        t_arr       = probe["t"]                            # (T,)
        value_dec   = np.asarray(probe["value"]).ravel()   # (T,)
        error_dec   = probe["error"]                        # (T, 2)
        pe          = error_dec[:, 1] if error_dec.ndim > 1 else error_dec

        for obs in range(1, n_obs + 1):
            t_win_start = t_iti_ + (obs - 1) * t_step
            t_win_end   = t_win_start + t_obs_
            mask = (t_arr >= t_win_start) & (t_arr < t_win_end)
            t_within    = t_arr[mask] - t_win_start
            qid = qid_map.get((trial, obs), np.nan)

            for j in range(mask.sum()):
                rows.append({
                    "pid":           pid,
                    "trial":         int(trial),
                    "observation":   obs,
                    "qid":           qid,
                    "t_within_obs":  float(t_within[j]),
                    "decoded_pe":    float(pe[mask][j]),
                    "decoded_value": float(value_dec[mask][j]),
                })

        print(
            f"  trial {ti:3d}/{n_trials}",
            end="\r", flush=True,
        )

    print()
    df = pd.DataFrame(rows)
    df.to_pickle(out_path)
    print(
        f"Saved {len(df):,} rows "
        f"({n_trials} trials × {n_obs} obs × {int(t_obs_/dt)} timesteps) "
        f"-> {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="probe_pids",
        choices=["probe_pids", "n_neurons_scan", "pe_readout", "probe_timeseries"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "collect"],
        default=None,
        help=(
            "'run': execute NEF simulations (one job per pid on cluster). "
            "'collect': aggregate per-pid output files into combined files. "
            "Not required for probe_timeseries (always runs directly)."
        ),
    )
    parser.add_argument(
        "--run_folder",
        type=str,
        default="refit",
        help="Source folder for fitted NEF params (run mode only)",
    )
    parser.add_argument("--out_folder", type=str, default="refit")
    parser.add_argument(
        "--pids",
        type=int,
        nargs="+",
        default=[6, 7],
        help="PIDs to simulate for probe_pids experiment",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Single PID for pe_readout / probe_timeseries / n_neurons_scan",
    )
    parser.add_argument(
        "--n_neurons_list",
        type=int,
        nargs="+",
        default=list(N_NEURONS_LIST),
    )
    parser.add_argument("--alpha_0", type=float, default=None,
                        help="alpha_0 for n_neurons_scan (default: load from fitted params)")
    parser.add_argument("--lambda_", type=float, default=None,
                        help="lambda_ for n_neurons_scan (default: load from fitted params)")
    parser.add_argument("--scan_pid", type=int, default=18,
                        help="pid for n_neurons_scan (default: 18)")
    args = parser.parse_args()

    out_folder = args.out_folder
    run_folder = RUNS_DIR / args.run_folder

    # n_neurons_scan: per-pid run or collect
    if args.experiment == "n_neurons_scan":
        if args.mode == "collect":
            _collect_n_neurons_scan(RUNS_DIR / out_folder)
        else:
            # Load fitted params for this pid if alpha_0/lambda_ not overridden
            pid = args.scan_pid
            alpha_0 = args.alpha_0
            lambda_ = args.lambda_
            if alpha_0 is None or lambda_ is None:
                combined = run_folder / "NEF_carrabin_params.pkl"
                if combined.exists():
                    p = pd.read_pickle(combined)
                    row = p[p["pid"]==pid]
                    if not row.empty:
                        alpha_0 = float(row["alpha_0"].iloc[0])
                        lambda_ = float(row["lambda_"].iloc[0])
            _run_n_neurons_scan(
                pid=pid,
                alpha_0=alpha_0,
                lambda_=lambda_,
                n_neurons_list=args.n_neurons_list,
                run_folder=run_folder,
                out_folder=out_folder,
            )
    # pe_readout: single-pid run or collect
    elif args.experiment == "pe_readout":
        if args.mode == "collect":
            _collect_pe_readout(RUNS_DIR / out_folder)
        else:
            if args.pid is None:
                parser.error("--pid required for pe_readout")
            _run_pe_readout(args.pid, run_folder, out_folder)
    # probe_timeseries: no collect step, always runs directly
    elif args.experiment == "probe_timeseries":
        if args.pid is None:
            parser.error("--pid required for probe_timeseries")
        _run_probe_timeseries(args.pid, run_folder, out_folder)
    elif args.mode == "run":
        if args.experiment == "probe_pids":
            _run_probe_pids_simulate(args.pids, run_folder, out_folder)
    elif args.mode == "collect":
        out_dir = RUNS_DIR / out_folder
        if args.experiment == "probe_pids":
            _collect_probe_pids(out_dir)
    else:
        parser.error("--mode run or --mode collect required for this experiment")


if __name__ == "__main__":
    main()
