#!/usr/bin/env python3
"""Run noisy representation experiments for NEF carrabin fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import RUNS_DIR, data_path


def _sensitivity_analysis(run_folder: Path) -> None:
    from scipy.stats import pearsonr, spearmanr

    from fitting.losses import _mean_qid_std

    nef_params = pd.read_pickle(run_folder / "NEF_recurrent_carrabin_params.pkl")
    nef_resp = pd.read_pickle(run_folder / "NEF_recurrent_carrabin_responses.pkl")
    human = pd.read_pickle(data_path("carrabin.pkl"))
    qid_map = human[["pid", "trial", "observation", "qid"]].drop_duplicates()

    rows = []
    for pid, grp in nef_resp.groupby("pid"):
        grp_qid = grp.merge(qid_map, on=["pid", "trial", "observation"], how="left")
        noise = _mean_qid_std(grp_qid)
        p = nef_params[nef_params["pid"] == pid].iloc[0]
        rows.append(
            {
                "pid": pid,
                "response_noise": noise,
                "alpha_0": float(p["alpha_0"]),
                "lambda_": float(p["lambda_"]),
            }
        )

    df = pd.DataFrame(rows).dropna()
    print(f"\nSensitivity analysis (NEF carrabin, n={len(df)} pids):")
    for col in ("alpha_0", "lambda_"):
        r, p_val = pearsonr(df[col], df["response_noise"])
        rs, ps = spearmanr(df[col], df["response_noise"])
        print(
            f"  response_noise vs {col}: "
            f"pearson r={r:.3f} (p={p_val:.4f}), "
            f"spearman r={rs:.3f} (p={ps:.4f})"
        )


def _run_probe_pids(
    pids: list[int],
    run_folder: Path,
    out_folder: str,
    run_simulation: bool,
) -> None:
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, run as nef_run

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_simulation:
        for pid in pids:
            params = pd.read_pickle(
                run_folder / f"NEF_recurrent_carrabin_{pid}_params.pkl"
            ).iloc[0].to_dict()
            fixed = MODEL_PARAMS["carrabin"]["NEF_recurrent"].get("fixed", {})
            params = {**PARAM_DEFAULTS, **fixed, **params}
            params["nef_type"] = "recurrent"
            params["dataset"] = "carrabin"
            params["model_type"] = "NEF_recurrent"
            print(
                f"Running pid={pid} (alpha_0={params['alpha_0']:.3f}, "
                f"lambda_={params['lambda_']:.3f})..."
            )
            nef_run(params, save_probes=True)
            src = data_path(f"probe_NEF_recurrent_carrabin_{pid}.pkl")
            dst = out_dir / f"probe_NEF_recurrent_carrabin_{pid}.pkl"
            if src.exists():
                Path(src).rename(dst)
                print(f"  Saved to {dst}")

    print("\nProbe data analysis TBD.")
    for pid in pids:
        probe_path = out_dir / f"probe_NEF_recurrent_carrabin_{pid}.pkl"
        if probe_path.exists():
            probes = pd.read_pickle(probe_path)
            n_trials = len(probes) if isinstance(probes, list) else 1
            print(f"  pid={pid}: {n_trials} trials of probe data available")


def _run_n_neurons_scan(
    scan_pid: int,
    n_neurons_list: list[int],
    run_folder: Path,
    out_folder: str,
    run_simulation: bool,
) -> None:
    from fitting.model_params import MODEL_PARAMS
    from models.NEF import PARAM_DEFAULTS, run as nef_run

    out_dir = RUNS_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_simulation:
        base_params = pd.read_pickle(
            run_folder / f"NEF_recurrent_carrabin_{scan_pid}_params.pkl"
        ).iloc[0].to_dict()
        fixed = MODEL_PARAMS["carrabin"]["NEF_recurrent"].get("fixed", {})
        base_params = {**PARAM_DEFAULTS, **fixed, **base_params}
        base_params["nef_type"] = "recurrent"
        base_params["dataset"] = "carrabin"
        base_params["model_type"] = "NEF_recurrent"

        for n_neurons in n_neurons_list:
            print(f"Simulating n_neurons={n_neurons}...")
            p = {**base_params, "n_neurons": n_neurons}
            responses = nef_run(p)
            responses["n_neurons"] = n_neurons
            out_path = out_dir / f"responses_carrabin_{scan_pid}_n{n_neurons}.pkl"
            responses.to_pickle(out_path)
            print(f"  Saved {out_path}")

    print("\nn_neurons scan analysis TBD.")
    for n_neurons in n_neurons_list:
        out_path = out_dir / f"responses_carrabin_{scan_pid}_n{n_neurons}.pkl"
        if out_path.exists():
            print(f"  n_neurons={n_neurons}: responses available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="probe_pids",
        choices=["probe_pids", "n_neurons_scan"],
        help="Which experiment to run",
    )
    parser.add_argument("--run_simulation", action="store_true", default=False)
    parser.add_argument(
        "--run_folder",
        type=str,
        default="response",
        help="Source folder for fitted NEF params",
    )
    parser.add_argument("--out_folder", type=str, default="noisy_representations")
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
        default=14,
        help="PID to use for n_neurons_scan experiment",
    )
    parser.add_argument(
        "--n_neurons_list",
        type=int,
        nargs="+",
        default=[50, 75, 100, 150, 200, 300, 500],
    )
    args = parser.parse_args()

    run_folder = RUNS_DIR / args.run_folder
    out_folder = args.out_folder

    _sensitivity_analysis(run_folder)

    if args.experiment == "probe_pids":
        _run_probe_pids(args.pids, run_folder, out_folder, args.run_simulation)
    elif args.experiment == "n_neurons_scan":
        _run_n_neurons_scan(
            args.scan_pid,
            args.n_neurons_list,
            run_folder,
            out_folder,
            args.run_simulation,
        )


if __name__ == "__main__":
    main()
