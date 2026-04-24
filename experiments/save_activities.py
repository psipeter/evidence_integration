#!/usr/bin/env python3
"""
Save per-neuron activities and encoders for NEF ensembles.
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

EXPERIMENT_NAME = "save_activities"
MODEL_TYPE = "NEF_recurrent"
RUN_FOLDER = "joint_loss"
READOUT_OFFSET = 0.1  # seconds into observation window for once-per-obs readout

VALID_ENSEMBLES = {"error", "value", "counting"}


def _get_ensemble_obj(net, ens_name: str):
    if ens_name == "counting":
        return net.counting.memory
    return getattr(net, ens_name)


def simulate_and_save(
    pid: int,
    params: dict,
    ensembles: list[str],
    timing: str,
    out_dir: Path,
) -> None:
    decoders = _pretrain(params)
    human_pid = pd.read_pickle(data_path(f"{params['dataset']}.pkl")).query("pid == @pid")

    activities_rows: dict[str, list[dict]] = {ens_name: [] for ens_name in ensembles}
    encoders_by_ens: dict[str, np.ndarray | None] = {ens_name: None for ens_name in ensembles}

    for trial, trial_data in human_pid.groupby("trial"):
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        rd_values = (
            trial_data["rd"].to_numpy(dtype=float)
            if params["dataset"] == "jiang"
            else np.zeros(len(obs_values))
        )
        p = {**params, "alpha_bias_array": rd_values}

        n_obs = len(obs_values)
        t_obs = float(params["t_obs"])
        t_iti = float(params["t_iti"])
        t_step = t_obs + t_iti
        t_total = n_obs * t_step
        dt = float(params["dt"])
        tau_probe = float(params["tau_probe"])

        net = build_network(obs_values, p, decoders)
        probes = {}
        with net:
            for ens_name in ensembles:
                ens = _get_ensemble_obj(net, ens_name)
                probes[ens_name] = nengo.Probe(ens.neurons, synapse=tau_probe)

        with nengo.Simulator(net, dt=dt, seed=int(p["seed"]), progress_bar=False) as sim:
            sim.run(t_total)

            if timing == "once_per_obs":
                for n_idx, (_, row) in enumerate(trial_data.iterrows()):
                    t_readout = t_iti + n_idx * t_step + READOUT_OFFSET
                    n_timesteps = len(sim.data[probes[ensembles[0]]])
                    idx = int(np.clip(np.round(t_readout / dt), 0, n_timesteps - 1))

                    for ens_name in ensembles:
                        activity = sim.data[probes[ens_name]][idx]
                        out_row: dict[str, int | float] = {
                            "pid": int(pid),
                            "trial": int(trial),
                            "observation": int(row["observation"]),
                        }
                        if params["dataset"] == "jiang":
                            out_row["stage"] = int(row["stage"])
                            out_row["trial_obs_idx"] = n_idx
                        for j, val in enumerate(activity):
                            out_row[f"n{j}"] = float(val)
                        activities_rows[ens_name].append(out_row)
            else:
                raise NotImplementedError("per_timestep timing not yet implemented")

            for ens_name in ensembles:
                if encoders_by_ens[ens_name] is None:
                    ens = _get_ensemble_obj(net, ens_name)
                    encoders_by_ens[ens_name] = np.array(sim.data[ens].encoders, copy=True)

    for ens_name in ensembles:
        activities_df = pd.DataFrame(activities_rows[ens_name])
        activities_path = out_dir / f"activities_{ens_name}_{params['dataset']}_{pid}.pkl"
        activities_df.to_pickle(activities_path)
        print(f"Saved {activities_path}")

        enc = encoders_by_ens[ens_name]
        if enc is None:
            raise RuntimeError(f"No encoders captured for ensemble {ens_name!r}")
        enc_rows = []
        for neuron_idx in range(enc.shape[0]):
            row = {
                "pid": int(pid),
                "ensemble": ens_name,
                "neuron_idx": int(neuron_idx),
            }
            for d in range(enc.shape[1]):
                row[f"enc_dim_{d}"] = float(enc[neuron_idx, d])
            enc_rows.append(row)
        enc_df = pd.DataFrame(enc_rows)
        encoders_path = out_dir / f"encoders_{ens_name}_{params['dataset']}_{pid}.pkl"
        enc_df.to_pickle(encoders_path)
        print(f"Saved {encoders_path}")


def run_local(
    pid: int,
    dataset: str,
    ensembles: list[str],
    run_folder: str,
    timing: str,
) -> None:
    from fitting.param_ranges import MODEL_PARAMS

    params_path = RUNS_DIR / run_folder / f"{MODEL_TYPE}_{dataset}_params.pkl"
    all_params = pd.read_pickle(params_path)
    params = all_params[all_params["pid"] == pid].iloc[0].to_dict()
    fixed = MODEL_PARAMS.get(dataset, {}).get(MODEL_TYPE, {}).get("fixed", {})
    params = {**PARAM_DEFAULTS, **fixed, **params}
    params["nef_type"] = "recurrent" if "recurrent" in MODEL_TYPE else "synaptic"
    params["dataset"] = dataset
    params["model_type"] = MODEL_TYPE

    out_dir = data_path("experiments") / "save_activities"
    out_dir.mkdir(parents=True, exist_ok=True)
    simulate_and_save(pid, params, ensembles, timing, out_dir)


def submit(
    pids: list[int],
    dataset: str,
    ensembles: list[str],
    run_folder: str,
    timing: str,
    dry_run: bool = False,
) -> None:
    from utils.paths import DATA_DIR

    root = str(DATA_DIR.parent)
    jobs_dir = Path(root) / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    ensembles_str = " ".join(ensembles)
    for pid in pids:
        cmd = (
            f"python experiments/{EXPERIMENT_NAME}.py "
            f"--pid {pid} --dataset {dataset} "
            f"--ensembles {ensembles_str} "
            f"--run_folder {run_folder} --timing {timing} --local"
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


def collect(dataset: str, ensembles: list[str]) -> None:
    out_dir = data_path("experiments") / "save_activities"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ens_name in ensembles:
        activity_files = sorted(out_dir.glob(f"activities_{ens_name}_{dataset}_*.pkl"))
        if activity_files:
            activities_df = pd.concat(
                [pd.read_pickle(f) for f in activity_files], ignore_index=True
            )
            activities_out = out_dir / f"activities_{ens_name}_{dataset}.pkl"
            activities_df.to_pickle(activities_out)
            print(
                f"Collected {len(activity_files)} files -> {activities_out} "
                f"({activities_df.shape})"
            )
        else:
            print(f"No activity files found for {ens_name} in {out_dir}")

        encoder_files = sorted(out_dir.glob(f"encoders_{ens_name}_{dataset}_*.pkl"))
        if encoder_files:
            encoders_df = pd.concat(
                [pd.read_pickle(f) for f in encoder_files], ignore_index=True
            )
            encoders_out = out_dir / f"encoders_{ens_name}_{dataset}.pkl"
            encoders_df.to_pickle(encoders_out)
            print(
                f"Collected {len(encoder_files)} files -> {encoders_out} "
                f"({encoders_df.shape})"
            )
        else:
            print(f"No encoder files found for {ens_name} in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="carrabin")
    parser.add_argument("--ensembles", nargs="+", default=["error"])
    parser.add_argument("--run_folder", type=str, default=RUN_FOLDER)
    parser.add_argument("--timing", type=str, default="once_per_obs")
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.timing == "per_timestep":
        raise NotImplementedError("per_timestep timing not yet implemented")
    if args.timing != "once_per_obs":
        raise ValueError(f"Unknown timing mode: {args.timing!r}")

    invalid = [ens for ens in args.ensembles if ens not in VALID_ENSEMBLES]
    if invalid:
        raise ValueError(
            f"Unknown ensembles: {invalid}. "
            f"Valid values: {sorted(VALID_ENSEMBLES)}"
        )

    if args.collect:
        collect(args.dataset, args.ensembles)
    elif args.local and args.pid is not None:
        run_local(args.pid, args.dataset, args.ensembles, args.run_folder, args.timing)
    else:
        human = pd.read_pickle(data_path(f"{args.dataset}.pkl"))
        pids = [int(pid) for pid in human["pid"].unique()]
        if args.pid is not None:
            pids = [args.pid]
        submit(
            pids,
            args.dataset,
            args.ensembles,
            args.run_folder,
            args.timing,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
