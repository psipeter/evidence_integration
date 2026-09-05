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
    from models.counting_integrator import (
        fast_decode, precompute_activities,
    )
    from utils.carrabin_transform import apply_carrabin_transform

    out_dir  = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

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
        single_path = out_dir / f"n_neurons_scan_{pid}_{n_neurons}.pkl"
        if single_path.exists():
            print(f"Already exists: {single_path.name} — skipping")
            continue
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
        act_path = data_path(f"counting_activities_n{n_neurons}_nc{n_neurons}_carrabin.pkl")
        # LEGACY -- DO NOT CARRY THIS PATTERN FORWARD. Unlike models/NEF.py's
        # run() (which now REQUIRES the activity file and raises a clear error
        # if it's missing -- see that file's own comment), this helper silently
        # calls precompute_activities() inline and keeps going. That's not the
        # seed-mismatch bug run() used to have, but it's the other failure mode
        # we're trying to kill project-wide: a script quietly kicking off a
        # real, possibly-long precompute job because a file happened to be
        # absent, rather than that being a deliberate decision. Feeds
        # figure_carrabin_neural.py's Panel C (N3, "Response variability and PE
        # variability vs n_neurons scan"). This whole script (extras_carrabin.py)
        # is carrabin-specific and predates the generic dataset-agnostic NEF
        # testing/dynamics tooling -- refactor or retire it once we reach that
        # stage of the new pipeline (all 4 tasks refit fresh; see CLAUDE.md/
        # docs/HISTORY.md), rather than patching this fallback in place now.
        def _load_or_precompute():
            if act_path.exists():
                try:
                    with open(act_path, "rb") as f:
                        m = pickle.load(f)
                    print(f"  Loaded activities: {act_path.name}")
                    return m
                except (EOFError, Exception) as e:
                    print(f"  Corrupt activity file ({e}), regenerating...", flush=True)
                    act_path.unlink(missing_ok=True)
            print(f"  Precomputing activities (n={n_neurons}, nc={n_neurons})...", flush=True)
            t0 = _time.time()
            precompute_activities(n_trials=None, params=params, out_path=act_path)
            print(f"  Done in {_time.time()-t0:.0f}s")
            with open(act_path, "rb") as f:
                return pickle.load(f)

        activity_map = _load_or_precompute()

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

        out = {
            "pid":        pid,
            "n_neurons":  n_neurons,
            "elapsed_s":  elapsed,
            "responses":  pd.DataFrame(rows_resp),    # cols: n_neurons, trial, observation, qid, response
            "pe_readout": pd.DataFrame(rows_pe),      # cols: n_neurons, trial, observation, qid, pe
        }
        # One file per (pid, n_neurons)
        single_path = out_dir / f"n_neurons_scan_{pid}_{n_neurons}.pkl"
        pd.to_pickle(out, single_path)
        print(f"  Saved -> {single_path.name}  (t={elapsed:.0f}s)")


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
    """Collect per-(pid, n_neurons) scan files into a single combined pkl.

    Input files: n_neurons_scan_{pid}_{n_neurons}.pkl
    Each contains: {"pid", "n_neurons", "elapsed_s", "responses", "pe_readout"}

    Output: n_neurons_scan.pkl
    Format: {n_neurons: {"responses": DataFrame, "pe_readout": DataFrame}}
    where DataFrames have columns: pid, trial, observation, qid, response/pe
    Metrics (resp_std, pe_std) are computed at plot time in figure_carrabin.py.
    """
    # Match both new format n_neurons_scan_{pid}_{n_neurons}.pkl
    # and old format n_neurons_scan_{pid}.pkl for backwards compat
    files = sorted(out_dir.glob("n_neurons_scan_[0-9]*_[0-9]*.pkl"))
    if not files:
        print("No n_neurons_scan per-(pid,n_neurons) files found.")
        print("Looking for old format...")
        files = sorted(out_dir.glob("n_neurons_scan_[0-9]*.pkl"))
        if not files:
            print("No scan files found at all.")
            return

    raw_resp: dict[int, list] = {}
    raw_pe:   dict[int, list] = {}

    for f in files:
        data = pd.read_pickle(f)
        if not isinstance(data, dict) or "responses" not in data:
            print(f"  Skipping {f.name} (old format)")
            continue
        n  = int(data["n_neurons"])
        pid = int(data["pid"])
        resp = data["responses"].copy()
        resp["pid"] = pid
        pe   = data["pe_readout"].copy()
        pe["pid"] = pid
        raw_resp.setdefault(n, []).append(resp)
        raw_pe.setdefault(n, []).append(pe)

    if not raw_resp:
        print("No valid scan data found.")
        return

    combined = {
        n: {
            "responses":  pd.concat(raw_resp[n],  ignore_index=True),
            "pe_readout": pd.concat(raw_pe[n],    ignore_index=True),
        }
        for n in sorted(raw_resp.keys())
    }

    out = out_dir / "n_neurons_scan.pkl"
    pd.to_pickle(combined, out)
    n_pids = len(set(combined[list(combined.keys())[0]]["responses"]["pid"]))
    print(f"Collected {len(files)} files  ({n_pids} pids x {len(combined)} n_neurons) -> {out.name}")



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
            dataset=str(params.get("dataset","carrabin")),
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
            dataset=str(params.get("dataset","carrabin")),
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



def _run_pe_dynamics(
    pids: list[int],
    run_folder: Path,
    out_folder: str,
    n_seeds: int = 10,
    alpha_0_list: list[float] | None = None,
    n_neurons_list_pe: list[int] | None = None,
    lambda_fixed: float = 0.0,
) -> None:
    """Simulate PE dynamics for a set of pids or explicit param combinations.

    Two modes:
      1. pid mode (default): load MLE-fitted params per pid, simulate with those.
         Output filename: pe_dynamics_NEF_carrabin_pid{pid}.pkl
      2. param grid mode (alpha_0_list + n_neurons_list_pe provided): simulate
         all combinations of alpha_0 x n_neurons, ignoring pids.
         Output filename: pe_dynamics_NEF_carrabin_a{alpha_0}_n{n_neurons}.pkl

    In both modes, runs n_seeds trials with a constant +1 artificial input
    (single observation per trial). Saves full once_per_dt timeseries of:
      - pe_product: error[:, 0] * error[:, 1]  (alpha * (obs − value))
      - pe_raw:     error[:, 1]                 (obs − value)
      - weight:     error[:, 0]                 (alpha(t))
      - value:      value ensemble output
    """
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, _pretrain, build_network
    import nengo

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = MODEL_PARAMS["carrabin"]["NEF"].get("fixed", {})

    # ── Build list of (label, params_dict) to simulate ───────────────────────
    jobs = []  # list of (out_path, params_dict, label)

    if alpha_0_list is not None and n_neurons_list_pe is not None:
        # Param grid mode — ignore pids
        from fitting.model_params import _NEF_FIXED
        base = {**PARAM_DEFAULTS, **fixed,
                "dataset": "carrabin", "model_type": "NEF", "pid": 0,
                "lambda_": lambda_fixed}
        for a0 in alpha_0_list:
            for nn in n_neurons_list_pe:
                tag = f"a{str(a0).replace('.','p')}_n{nn}"
                out_path = out_dir / f"pe_dynamics_NEF_carrabin_{tag}.pkl"
                p = {**base, "alpha_0": a0, "n_neurons": nn,
                     "n_neurons_counting": nn}
                jobs.append((out_path, p, f"alpha_0={a0} n_neurons={nn}"))
    else:
        # Pid mode — load fitted params
        for suffix in ("_mle", ""):
            params_path = run_folder / f"NEF_carrabin_params{suffix}.pkl"
            if params_path.exists():
                all_params = pd.read_pickle(params_path)
                params_label = "mle" if suffix == "_mle" else "rmse"
                break
        else:
            raise FileNotFoundError(f"No NEF params file found in {run_folder}")
        print(f"Using {params_label} params from {params_path.name}")

        for pid in pids:
            row = all_params[all_params["pid"] == pid]
            if row.empty:
                print(f"  pid={pid}: no params found — skipping")
                continue
            p = {**PARAM_DEFAULTS, **fixed, **row.iloc[0].to_dict()}
            p["dataset"] = "carrabin"; p["model_type"] = "NEF"; p["pid"] = int(pid)
            out_path = out_dir / f"pe_dynamics_NEF_carrabin_pid{pid}.pkl"
            jobs.append((out_path, p,
                         f"pid={pid} alpha_0={p['alpha_0']:.3f} "
                         f"n_neurons={int(p['n_neurons'])}"))

    # ── Simulate each job ─────────────────────────────────────────────────────
    for out_path, params, label in jobs:
        if out_path.exists():
            print(f"  {label}: already exists — skipping (delete to rerun)")
            continue

        alpha_0   = float(params["alpha_0"])
        lambda_   = float(params["lambda_"])
        n_neurons = int(params["n_neurons"])
        dt        = float(params["dt"])
        t_obs_    = float(params["t_obs"])
        t_iti_    = float(params["t_iti"])
        print(f"  {label}")

        obs_values = np.array([1.0])
        rows = []

        for seed in range(n_seeds):
            p = {**params, "seed": seed}
            decoders = _pretrain({**p})
            net      = build_network(obs_values, p, decoders)
            t_total  = t_obs_ + t_iti_
            with nengo.Simulator(net, dt=dt, seed=seed, progress_bar=False) as sim:
                sim.run(t_total)

            t_arr      = np.arange(len(sim.data[net.probe_value])) * dt
            value_dec  = sim.data[net.probe_value].squeeze()
            error_dec  = sim.data[net.probe_error]
            obs_probe  = sim.data[net.probe_obs].squeeze()
            weight_dec = error_dec[:, 0]
            pe_raw     = error_dec[:, 1]
            pe_prod    = weight_dec * pe_raw

            mask     = (t_arr >= t_iti_) & (t_arr < t_iti_ + t_obs_)
            t_within = t_arr[mask] - t_iti_

            for j in range(mask.sum()):
                rows.append({
                    "seed":        seed,
                    "t":           float(t_within[j]),
                    "pe_product":  float(pe_prod[mask][j]),
                    "pe_raw":      float(pe_raw[mask][j]),
                    "weight":      float(weight_dec[mask][j]),
                    "value":       float(value_dec[mask][j]),
                    "obs":         float(obs_probe[mask][j]),
                    "alpha_0":     alpha_0,
                    "lambda_":     lambda_,
                    "n_neurons":   n_neurons,
                })

            print(f"    seed {seed+1}/{n_seeds}", end="\r", flush=True)

        print()
        df = pd.DataFrame(rows)
        df.to_pickle(out_path)
        print(f"  Saved {len(df):,} rows ({n_seeds} seeds × {mask.sum()} timesteps)"
              f" -> {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="probe_pids",
        choices=["probe_pids", "n_neurons_scan", "pe_readout", "probe_timeseries", "pe_dynamics"],
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
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of network seeds for pe_dynamics")
    parser.add_argument("--alpha_0_list", type=float, nargs="+", default=None,
                        help="Explicit alpha_0 values for pe_dynamics param grid")
    parser.add_argument("--n_neurons_pe", type=int, nargs="+", default=None,
                        help="Explicit n_neurons values for pe_dynamics param grid")
    parser.add_argument("--lambda_fixed", type=float, default=0.0,
                        help="Fixed lambda_ for pe_dynamics param grid (default 0.0)")
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
    parser.add_argument("--n_neurons", type=int, default=None,
                        help="Single n_neurons value for n_neurons_scan (overrides --n_neurons_list)")
    args = parser.parse_args()

    out_folder = args.out_folder
    run_folder = RUNS_DIR / args.run_folder

    # pe_dynamics: no mode required
    if args.experiment == "pe_dynamics":
        _run_pe_dynamics(
            pids=args.pids,
            run_folder=run_folder,
            out_folder=out_folder,
            n_seeds=args.n_seeds,
            alpha_0_list=args.alpha_0_list,
            n_neurons_list_pe=args.n_neurons_pe,
            lambda_fixed=args.lambda_fixed,
        )
        return

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
            n_neurons_list = [args.n_neurons] if args.n_neurons is not None else args.n_neurons_list
            _run_n_neurons_scan(
                pid=pid,
                alpha_0=alpha_0,
                lambda_=lambda_,
                n_neurons_list=n_neurons_list,
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
