#!/usr/bin/env python3
"""
Save per-neuron activities and encoders for NEF ensembles.

Works for any dataset with ``{dataset}.pkl`` containing ``trial``, ``observation``,
and ``value`` columns (carrabin, yoo).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nengo
import numpy as np
import pandas as pd

from models.NEF import PARAM_DEFAULTS, _pretrain, build_network
from utils.paths import data_path, resolve_run_folder

READOUT_OFFSET = 0.5  # seconds into observation window for once-per-obs readout

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
    dt_sample: float = 0.01,
) -> None:
    decoders = _pretrain(params)
    human_pid = pd.read_pickle(data_path(f"{params['dataset']}.pkl")).query("pid == @pid")

    activities_rows: dict[str, list[dict]] = {ens_name: [] for ens_name in ensembles}
    encoders_by_ens: dict[str, np.ndarray | None] = {ens_name: None for ens_name in ensembles}
    windowed_data_by_ens: dict[str, list[np.ndarray]] = {
        ens_name: [] for ens_name in ensembles
    }

    for trial in sorted(human_pid["trial"].unique()):
        t_trial = time.time()
        trial_data = human_pid[human_pid["trial"] == trial].sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        p = {**params}

        n_obs = len(obs_values)
        t_obs = float(params["t_obs"])
        t_iti = float(params["t_iti"])
        t_step = t_obs + t_iti
        t_total = n_obs * t_step
        dt = float(params["dt"])
        tau_probe = float(params["tau_probe"])

        net = build_network(obs_values, p, decoders)
        probes: dict[str, nengo.Probe] = {}
        probes_dt: dict[str, nengo.Probe] = {}
        with net:
            for ens_name in ensembles:
                ens = _get_ensemble_obj(net, ens_name)
                if timing == "once_per_obs":
                    probes[ens_name] = nengo.Probe(ens.neurons, synapse=tau_probe)
                elif timing == "once_per_dt":
                    probes_dt[ens_name] = nengo.Probe(
                        ens.neurons,
                        synapse=float(params["tau_probe"]),
                        sample_every=dt_sample,
                    )

        with nengo.Simulator(net, dt=dt, seed=int(p["seed"]), progress_bar=False) as sim:
            sim.run(t_total)
            elapsed_trial = time.time() - t_trial
            print(f"  pid={pid} trial {int(trial)}: {elapsed_trial:.1f}s", flush=True)

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
                        for j, val in enumerate(activity):
                            out_row[f"n{j}"] = float(val)
                        activities_rows[ens_name].append(out_row)
            elif timing == "once_per_dt":
                n_samples_obs = int(round(float(params["t_obs"]) / dt_sample))
                n_samples_step = int(round(t_step / dt_sample))
                for ens_name in ensembles:
                    spike_data = sim.data[probes_dt[ens_name]]
                    obs_windows = []
                    for n_idx in range(n_obs):
                        n_samples_iti = int(round(float(params["t_iti"]) / dt_sample))
                        start = n_idx * n_samples_step + n_samples_iti
                        end = start + n_samples_obs
                        obs_windows.append(spike_data[start:end])
                    windowed_arr = np.stack(obs_windows, axis=0)
                    windowed_data_by_ens[ens_name].append(windowed_arr)
            else:
                raise ValueError(f"Unknown timing mode in simulate_and_save: {timing!r}")

            for ens_name in ensembles:
                if encoders_by_ens[ens_name] is None:
                    ens = _get_ensemble_obj(net, ens_name)
                    encoders_by_ens[ens_name] = np.array(sim.data[ens].encoders, copy=True)

    if timing == "once_per_obs":
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
    elif timing == "once_per_dt":
        trial_ids = np.array(sorted(human_pid["trial"].unique()), dtype=int)
        for ens_name in ensembles:
            trial_arrays = windowed_data_by_ens[ens_name]
            max_obs = max(a.shape[0] for a in trial_arrays)
            padded = []
            for a in trial_arrays:
                n_obs = a.shape[0]
                if n_obs < max_obs:
                    pad_shape = (max_obs - n_obs,) + a.shape[1:]
                    padding = np.full(pad_shape, np.nan, dtype=np.float32)
                    a = np.concatenate([a, padding], axis=0)
                padded.append(a.astype(np.float32))
            activities_arr = np.stack(padded, axis=0)
            enc = encoders_by_ens[ens_name]
            if enc is None:
                raise RuntimeError(f"No encoders captured for ensemble {ens_name!r}")
            out_path = (
                out_dir / f"activities_windowed_{ens_name}_{params['dataset']}_{pid}.npz"
            )
            np.savez_compressed(
                out_path,
                activities=activities_arr,
                encoders=enc,
                trial_ids=trial_ids,
                dt_sample=np.array(dt_sample),
            )
            print(f"Saved {out_path} — shape {activities_arr.shape}")
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
    else:
        raise ValueError(f"Unknown timing mode in simulate_and_save: {timing!r}")


def run(
    pid: int,
    dataset: str,
    ensembles: list[str],
    run_folder: str,
    timing: str,
    dt_sample: float = 0.01,
    model_type: str = "NEF_recurrent",
) -> None:
    from utils.run_params import load_run_params

    params = load_run_params(pid, dataset, model_type, run_folder)

    out_dir = resolve_run_folder(run_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    simulate_and_save(pid, params, ensembles, timing, out_dir, dt_sample=dt_sample)


if __name__ == "__main__":
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    pid = int(sys.argv[3])
    run_folder = sys.argv[4]
    ensembles = sys.argv[5].split(",")
    timing = sys.argv[6] if len(sys.argv) > 6 else "once_per_obs"
    dt_sample = float(sys.argv[7]) if len(sys.argv) > 7 else 0.01
    run(pid, dataset, ensembles, run_folder, timing, dt_sample, model_type)
    print("JOB_COMPLETE")
