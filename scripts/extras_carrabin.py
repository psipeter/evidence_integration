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
2. n_neurons_scan  →  bash jobs/submit_neurons_scan.sh
----------------------------------------------------------------------
Runs the NEF for selected pids across neuron counts 50–500.
One SLURM job per (pid, n_neurons) combination.

  Submit all jobs:
      bash jobs/submit_neurons_scan.sh

  Collect (after all jobs complete):
      python scripts/extras_carrabin.py --experiment n_neurons_scan --mode collect --out_folder refit

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
  scan_compact_carrabin_<pid>_n<N>.pkl  — readout response + abs_pred_error per obs

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

N_NEURONS_LIST = [50, 75, 100, 150, 200, 300, 500]
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


def _run_n_neurons_scan_simulate(
    scan_pids: list[int],
    n_neurons_list: list[int],
    run_folder: Path,
    out_folder: str,
) -> None:
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
    from utils.run_params import trial_seed as _trial_seed

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    human = pd.read_pickle(data_path("carrabin.pkl"))

    for pid in scan_pids:
        base_params = pd.read_pickle(
            run_folder / f"NEF_carrabin_{pid}_params.pkl"
        ).iloc[0].to_dict()
        fixed = MODEL_PARAMS["carrabin"]["NEF"].get("fixed", {})
        base_params = {**PARAM_DEFAULTS, **fixed, **base_params}
        base_params["nef_type"] = "recurrent"
        base_params["dataset"] = "carrabin"
        base_params["model_type"] = "NEF"
        base_params["pid"] = int(pid)
        human_pid = human.query("pid == @pid")

        for n_neurons in n_neurons_list:
            print(f"Simulating pid={pid}, n_neurons={n_neurons}...")
            p = {**base_params, "n_neurons": n_neurons}
            decoders = _pretrain(p)
            compact_rows: list[dict] = []

            for trial, trial_data in human_pid.groupby("trial"):
                trial_data = trial_data.sort_values("observation")
                obs_values = trial_data["value"].to_numpy(dtype=float)
                trial_seed = _trial_seed(int(p["seed"]), int(trial))
                p_trial = {**p, "seed": trial_seed}
                _, probe_data = _simulate_trial(
                    obs_values, p_trial, decoders, return_probes=True
                )
                t = probe_data["t"]
                value_decoded = np.asarray(probe_data["value"]).squeeze()
                error = probe_data["error"]
                error1 = error[:, 1] if error.ndim > 1 else error
                t_iti = float(p_trial["t_iti"])
                t_obs = float(p_trial["t_obs"])
                t_step = t_obs + t_iti
                n_obs = len(obs_values)

                for obs in range(1, n_obs + 1):
                    t_readout = t_iti + (obs - 1) * t_step + READOUT_OFFSET
                    idx = int(np.argmin(np.abs(t - t_readout)))
                    compact_rows.append(
                        {
                            "pid": pid,
                            "n_neurons": n_neurons,
                            "trial": int(trial),
                            "observation": obs,
                            "response": float(value_decoded[idx]),
                            "abs_pred_error": float(np.abs(error1[idx])),
                        }
                    )

            compact_df = pd.DataFrame(compact_rows)
            out_path = out_dir / f"scan_compact_carrabin_{pid}_n{n_neurons}.pkl"
            compact_df.to_pickle(out_path)
            print(f"  Saved {len(compact_df)} rows to {out_path}")


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


def _collect_n_neurons_scan(out_dir: Path, _n_neurons_list: list[int]) -> None:
    compact_files = sorted(out_dir.glob("scan_compact_carrabin_*_n*.pkl"))
    if not compact_files:
        print("No compact scan files found.")
        return
    df = pd.concat([pd.read_pickle(f) for f in compact_files], ignore_index=True)
    out = out_dir / "scan_responses_carrabin.pkl"
    df.to_pickle(out)
    print(f"Collected {len(compact_files)} files -> {out} ({df.shape})")



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

        _, probe = _simulate_trial(obs_values, p, decoders, return_probes=True)

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
        help="Single PID for probe_timeseries experiment",
    )
    parser.add_argument(
        "--scan_pid",
        type=int,
        default=None,
        help="Single PID alias for --scan_pids",
    )
    parser.add_argument(
        "--scan_pids",
        type=int,
        nargs="+",
        default=[14],
        help="PIDs to use for n_neurons_scan (default: [14])",
    )
    parser.add_argument(
        "--n_neurons_list",
        type=int,
        nargs="+",
        default=list(N_NEURONS_LIST),
    )
    args = parser.parse_args()

    out_folder = args.out_folder
    if args.scan_pid is not None:
        args.scan_pids = [args.scan_pid]

    run_folder = RUNS_DIR / args.run_folder

    # pe_readout: single-pid run or collect
    if args.experiment == "pe_readout":
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
        elif args.experiment == "n_neurons_scan":
            _run_n_neurons_scan_simulate(
                args.scan_pids, args.n_neurons_list, run_folder, out_folder
            )
    elif args.mode == "collect":
        out_dir = RUNS_DIR / out_folder
        if args.experiment == "probe_pids":
            _collect_probe_pids(out_dir)
        elif args.experiment == "n_neurons_scan":
            _collect_n_neurons_scan(out_dir, args.n_neurons_list)
    else:
        parser.error("--mode run or --mode collect required for this experiment")


if __name__ == "__main__":
    main()
