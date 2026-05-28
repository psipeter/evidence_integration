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

=============================================================================
OUTPUT FILES  (written to data/runs/<out_folder>/)
=============================================================================

Per-pid files (from ``--mode run``):

probe_pids experiment:
  probe_NEF_carrabin_<pid>.pkl   — raw NEF probe timeseries per pid

n_neurons_scan experiment:
  scan_compact_carrabin_<pid>_n<N>.pkl     — readout response + abs_pred_error per obs

Combined files (from ``--mode collect``):

  probe_pids_carrabin.pkl                  — all probe_pids probes (pid on each entry)
  scan_responses_carrabin.pkl              — combined compact scan rows (response, abs_pred_error)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="probe_pids",
        choices=["probe_pids", "n_neurons_scan"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "collect"],
        required=True,
        help=(
            "'run': execute NEF simulations (one job per pid on cluster). "
            "'collect': aggregate per-pid output files into combined files."
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

    if args.mode == "run":
        run_folder = RUNS_DIR / args.run_folder
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


if __name__ == "__main__":
    main()
