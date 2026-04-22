#!/usr/bin/env python3
"""
Experiment 01: error-population activity vs prediction error.

Measures the relationship between mean error population activity and decoded
prediction error at the beginning of each observation window across all trials
and participants for a dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nengo
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.NEF import PARAM_DEFAULTS, _pretrain, build_network
from utils.paths import RUNS_DIR, data_path
from utils.slurm import DEFAULT_TIME_LIMITS, make_job_script, submit_script

EXPERIMENT_NAME = "experiment_01_error_activity"
DATASET = "carrabin"  # can be overridden via CLI
MODEL_TYPE = "NEF_recurrent"
RUN_FOLDER = "joint_loss"  # folder containing best-fit params


def simulate_experiment(pid: int, params: dict) -> pd.DataFrame:
    decoders = _pretrain(params)
    rows: list[dict] = []
    human_pid = pd.read_pickle(data_path(f"{params['dataset']}.pkl")).query("pid == @pid")

    for trial, trial_data in human_pid.groupby("trial"):
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        alpha_bias = (
            trial_data["rd"].to_numpy(dtype=float)
            if params["dataset"] == "jiang"
            else np.zeros(len(obs_values))
        )
        p = {**params, "alpha_bias_array": alpha_bias}
        n_obs = len(obs_values)
        t_obs = float(params["t_obs"])
        t_iti = float(params["t_iti"])
        t_step = t_obs + t_iti
        t_total = n_obs * t_step
        dt = float(params["dt"])
        tau_probe = float(params["tau_probe"])

        net = build_network(obs_values, p, decoders)

        # Additional probe for error-population activity.
        with net:
            probe_error_neurons = nengo.Probe(net.error.neurons, synapse=tau_probe)

        with nengo.Simulator(net, dt=dt, seed=int(p["seed"]), progress_bar=False) as sim:
            sim.run(t_total)

        error_decoded = sim.data[net.probe_error]  # (T, 2)
        spike_data = sim.data[probe_error_neurons]  # (T, n_neurons), probe-filtered

        firing_rates = spike_data

        for n_idx, (_, row) in enumerate(trial_data.iterrows()):
            obs = int(row["observation"])
            # Readout at start of obs period (before value update).
            t_readout = t_iti + n_idx * t_step
            idx = int(np.round(t_readout / float(p["dt"])))
            idx = int(np.clip(idx, 0, min(len(error_decoded), len(firing_rates)) - 1))

            mean_act = float(firing_rates[idx].mean())
            pred_err = float(error_decoded[idx, 1])  # o - v

            entry: dict[str, int | float | str] = {
                "model_type": MODEL_TYPE,
                "pid": pid,
                "trial": int(trial),
                "observation": obs,
                "mean_activity": mean_act,
                "prediction_error": pred_err,
            }
            if params["dataset"] == "jiang":
                entry["stage"] = int(row["stage"])
            rows.append(entry)

    return pd.DataFrame(rows)


def run_local(pid: int, dataset: str, run_folder: str) -> None:
    from fitting.param_ranges import MODEL_PARAMS

    params_path = RUNS_DIR / run_folder / f"{MODEL_TYPE}_{dataset}_params.pkl"
    all_params = pd.read_pickle(params_path)
    params = all_params[all_params["pid"] == pid].iloc[0].to_dict()
    fixed = MODEL_PARAMS.get(dataset, {}).get(MODEL_TYPE, {}).get("fixed", {})
    params = {**PARAM_DEFAULTS, **fixed, **params}
    params["nef_type"] = "recurrent" if "recurrent" in MODEL_TYPE else "synaptic"
    params["dataset"] = dataset
    params["model_type"] = MODEL_TYPE

    out_dir = data_path("experiments") / EXPERIMENT_NAME / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    df = simulate_experiment(pid, params)
    df.to_pickle(out_dir / f"{EXPERIMENT_NAME}_{dataset}_{pid}.pkl")
    print(f"Saved {out_dir}/{EXPERIMENT_NAME}_{dataset}_{pid}.pkl")


def submit(
    pids: list[int],
    dataset: str,
    run_folder: str,
    dry_run: bool = False,
) -> None:
    from utils.paths import DATA_DIR

    root = str(DATA_DIR.parent)
    jobs_dir = Path(root) / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    for pid in pids:
        cmd = (
            f"python experiments/{EXPERIMENT_NAME}.py "
            f"--pid {pid} --dataset {dataset} "
            f"--run_folder {run_folder} --local"
        )
        script = make_job_script(
            root=root,
            commands=[cmd],
            time_limit=DEFAULT_TIME_LIMITS.get(MODEL_TYPE, "4:0:0"),
        )
        script_path = jobs_dir / f"{EXPERIMENT_NAME}_{dataset}_{pid}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        submit_script(script_path, dry_run=dry_run)


def collect(dataset: str) -> None:
    out_dir = data_path("experiments") / EXPERIMENT_NAME / dataset
    files = sorted(out_dir.glob(f"{EXPERIMENT_NAME}_{dataset}_*.pkl"))
    if not files:
        print(f"No files found in {out_dir}")
        return
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    out = out_dir / f"{EXPERIMENT_NAME}_{dataset}.pkl"
    df.to_pickle(out)
    print(f"Collected {len(files)} files -> {out} ({df.shape})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--dataset", type=str, default=DATASET)
    p.add_argument("--run_folder", type=str, default=RUN_FOLDER)
    p.add_argument("--local", action="store_true")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.collect:
        collect(args.dataset)
    elif args.local and args.pid is not None:
        run_local(args.pid, args.dataset, args.run_folder)
    else:
        human = pd.read_pickle(data_path(f"{args.dataset}.pkl"))
        pids = [int(pid) for pid in human["pid"].unique()]
        if args.pid is not None:
            pids = [args.pid]
        submit(pids, args.dataset, args.run_folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
