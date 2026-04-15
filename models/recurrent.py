#!/usr/bin/env python3
"""
NEF recurrent model of evidence integration.
Internal populations are TODO stubs.

Usage:
    from models.recurrent import run
    responses = run(params)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nengo
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path


def _make_obs_input(obs_values: np.ndarray, t_obs: float, t_iti: float) -> callable:
    """Input function presenting obs_values[i] for t_obs s, then 0 for t_iti s."""
    t_step = t_obs + t_iti
    n_obs = len(obs_values)

    def obs_fn(t: float) -> float:
        step = int(t / t_step)
        phase = t - step * t_step
        if step < n_obs and phase < t_obs:
            return float(obs_values[step])
        return 0.0

    return obs_fn


def _extract_responses(
    t_arr: np.ndarray,
    value_decoded: np.ndarray,
    n_obs: int,
    params: dict,
) -> np.ndarray:
    """Compute readout times and return averaged responses."""
    t_obs = params["t_obs"]
    t_iti = params["t_iti"]
    t_step = t_obs + t_iti
    readout_times = np.array([i * t_step + t_obs for i in range(n_obs)])
    return np.array(
        [
            float(np.mean(value_decoded[np.abs(t_arr - rt) < params["probe_dt"] * 3]))
            for rt in readout_times
        ]
    )


def build_network(obs_values: np.ndarray, params: dict) -> nengo.Network:
    """Build the recurrent NEF network for a single trial. Internals are TODO."""
    t_obs = params["t_obs"]
    t_iti = params["t_iti"]
    seed = params["seed"]
    obs_fn = _make_obs_input(obs_values, t_obs, t_iti)

    with nengo.Network(label="recurrent", seed=seed) as net:
        net.node_obs = nengo.Node(obs_fn, label="node_obs")

        # TODO: internal populations and connections

        net.value = nengo.Node(0.0, label="value_stub")
        net.probe_value = nengo.Probe(
            net.value,
            synapse=params["probe_syn"],
            sample_every=params["probe_dt"],
        )

    return net


def _simulate_trial(obs_values: np.ndarray, params: dict) -> np.ndarray:
    """Simulate one trial, return model responses (one per observation)."""
    t_obs = params["t_obs"]
    t_iti = params["t_iti"]
    n_obs = len(obs_values)
    t_total = n_obs * (t_obs + t_iti) + t_iti

    net = build_network(obs_values, params)
    with nengo.Simulator(net, dt=params["dt"], seed=params["seed"], progress_bar=False) as sim:
        sim.run(t_total)

    t_arr = sim.trange(dt=params["probe_dt"])
    value_decoded = sim.data[net.probe_value].squeeze()
    responses = _extract_responses(t_arr, value_decoded, n_obs, params)
    return responses


def run(params: dict, save: bool = False, trials: list | None = None) -> pd.DataFrame:
    """Run the recurrent NEF model for a single participant."""
    required = (
        "model_type",
        "dataset",
        "pid",
        "t_obs",
        "t_iti",
        "dt",
        "probe_syn",
        "probe_dt",
        "seed",
    )
    for key in required:
        if key not in params:
            raise KeyError(f"params must include {key!r}")

    dataset = params["dataset"]
    pid = int(params["pid"])

    human_pid = pd.read_pickle(data_path(f"{dataset}.pkl")).query("pid == @pid")
    if trials is not None:
        human_pid = human_pid[human_pid["trial"].isin(trials)]

    rows = []
    t0 = time.time()

    for trial, trial_data in human_pid.groupby("trial"):
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        responses = _simulate_trial(obs_values, params)
        for observation, response in zip(trial_data["observation"], responses):
            rows.append(
                {
                    "model_type": params["model_type"],
                    "pid": pid,
                    "trial": int(trial),
                    "observation": int(observation),
                    "response": float(response),
                }
            )

    elapsed = time.time() - t0
    n_trials = human_pid["trial"].nunique()
    print(f"  {n_trials} trials in {elapsed:.1f}s ({elapsed/n_trials:.2f}s/trial)")

    out = pd.DataFrame(rows)
    if save:
        out.to_pickle(data_path(f"{params['model_type']}_{dataset}_{pid}.pkl"))
    return out
