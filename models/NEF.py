#!/usr/bin/env python3
"""
NEF model of evidence integration.

Supports **carrabin** and **yoo**: sequential scalar ``value`` inputs per
observation.

Architecture (per trial):
    counting subnetwork (LMU or integrator, pretrained decoders)
    counting → error[dim 0]   (alpha(n) via W_weight decoder)
    node_input[0] → error[dim 1]   (observation o(t))
    node_input[1] → error.neurons  (ITI inhibition)
    value → error[dim 1]      (transform=-1, subtracts v)

Recurrent value dynamics: multiplicative error→value connection and recurrent
self-connection on ``value``.

Usage:
    from models.NEF import run
    responses = run(params)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import nengo
import numpy as np
import pandas as pd

nengo.rc.set("decoder_cache", "enabled", "False")

for _logger_name in (
    "nengo",
    "nengo.simulator",
    "nengo.builder",
    "nengo.builder.network",
    "nengo.builder.optimizer",
    "nengo.builder.connection",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.counting_integrator import (
    build_network as build_counting_integrator,
    decode_outputs as decode_counting_integrator,
    fast_decode as fast_decode_counting,
    load_activities as load_counting_activities,
    load_decoders as load_counting_decoders,
    simulate_network as simulate_counting_integrator,
)
from utils.paths import data_path

from fitting.model_params import _NEF_FIXED
from utils.binary_transform import (
    apply_binary_transform,
    nef_obs_values,
    nef_response_to_model_scale,
)

PARAM_DEFAULTS: dict = {
    **_NEF_FIXED,
    "n_obs": 30,
    "lambda_": 0.5,
    "alpha_0": 1,
    "T_error": 0.5,
    "tau_error": 0.1,
}


def _make_input(obs_values: np.ndarray, params: dict) -> callable:
    """Returns [obs(t), inh(t)] where inh=1 during ITI, 0 during observation."""
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = len(obs_values)

    def fn(t: float) -> list[float]:
        if t < t_iti:
            return [0.0, 1.0]
        step = int((t - t_iti) / t_step)
        phase = (t - t_iti) - step * t_step
        if step < n_obs and phase < t_obs:
            return [float(obs_values[step]), 0.0]
        return [0.0, 1.0]

    return fn


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
    readout_times = np.array([t_iti + i * t_step + t_obs for i in range(n_obs)])
    return np.array(
        [
            float(np.mean(value_decoded[np.abs(t_arr - rt) < params["dt"] * 3]))
            for rt in readout_times
        ]
    )


def _pretrain(params: dict) -> dict:
    """Run counting pretraining for the integrator subnetwork."""
    seed = int(params.get("seed", 0))
    p = {**params, "n_obs": params["radius_c"], "seed": seed}
    net = build_counting_integrator(p, train=True)
    raw = simulate_counting_integrator(net, p, train=True)
    return decode_counting_integrator(raw, p)


def build_network(
    obs_values: np.ndarray,
    params: dict,
    decoders: dict,
) -> nengo.Network:
    seed = int(params["seed"])
    tau_fb = float(params["tau_fb"])
    T_error = float(params["T_error"])

    _build_c = build_counting_integrator

    with nengo.Network(label=str(params["model_type"]), seed=seed) as net:
        net.node_input = nengo.Node(
            _make_input(obs_values, params), size_out=2, label="node_input"
        )

        # Counting uses n_neurons_counting for memory and n_neurons for
        # onset_detector (error and value use this n_neurons only).
        c_params = {
            **params,
            "n_obs": int(params["radius_c"]),
        }
        net.counting = _build_c(c_params, train=False, decoders=decoders)
        # probe counting weight and count decoded outputs
        net.probe_counting_weight = nengo.Probe(
            net.counting.weight_out,
            synapse=float(params["tau_probe"]),
            sample_every=float(params["dt"]),
        )
        net.probe_counting_count = nengo.Probe(
            net.counting.count_out,
            synapse=float(params["tau_probe"]),
            sample_every=float(params["dt"]),
        )

        net.error = nengo.Ensemble(
            n_neurons=int(params["n_neurons"]),
            dimensions=2,
            radius=float(params["radius_e"]),
            seed=seed,
            label="error",
        )
        nengo.Connection(
                net.counting.memory.neurons,
                net.error[0],
                transform=decoders["W_weight"],
                synapse=float(params["tau_ff"]),
                seed=seed,
            )

        nengo.Connection(net.node_input[0], net.error[1], synapse=None, seed=seed)
        w_inh = -10.0 * np.ones((net.error.n_neurons, 1))
        nengo.Connection(
            net.node_input[1],
            net.error.neurons,
            transform=w_inh,
            synapse=float(params["tau_error"]),
            seed=seed,
        )

        net.value = nengo.Ensemble(
            n_neurons=int(params["n_neurons"]),
            dimensions=1,
            radius=float(params["radius_v"]),
            seed=seed,
            label="value",
        )

        nengo.Connection(
            net.value,
            net.error[1],
            transform=-1,
            synapse=float(params["tau_ff"]),
            seed=seed,
        )

        nengo.Connection(
            net.error,
            net.value,
            function=lambda x: x[0] * x[1],
            transform=T_error,
            synapse=tau_fb,
            seed=seed,
        )
        nengo.Connection(
            net.value,
            net.value,
            synapse=tau_fb,
            seed=seed,
        )

        net.probe_value = nengo.Probe(
            net.value,
            synapse=float(params["tau_probe"]),
            sample_every=float(params["dt"]),
        )
        net.probe_error = nengo.Probe(
            net.error,
            synapse=float(params["tau_probe"]),
            sample_every=float(params["dt"]),
        )
        net.probe_error_neurons = nengo.Probe(net.error.neurons, synapse=None)
        net.probe_obs = nengo.Probe(
            net.node_input[0],
            synapse=None,
            sample_every=float(params["dt"]),
        )

    return net


def _simulate_trial(
    obs_values: np.ndarray,
    params: dict,
    decoders: dict,
    *,
    return_probes: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Simulate one trial, return model responses (one per observation).

    If ``return_probes`` is True, return ``(responses, probe_data)`` instead.
    """
    n_obs = len(obs_values)
    t_total = n_obs * (float(params["t_obs"]) + float(params["t_iti"]))

    net = build_network(obs_values, params, decoders)
    # optional ITI noise injection (used by iti_perturbation.py)
    if float(params.get("iti_noise_amplitude", 0.0)) > 0:
        try:
            from scripts.iti_perturbation import _add_iti_noise

            _add_iti_noise(net, params, len(obs_values))
        except ImportError:
            pass
    with nengo.Simulator(
        net,
        dt=float(params["dt"]),
        seed=int(params["seed"]),
        progress_bar=False,
    ) as sim:
        sim.run(t_total)

    t_arr = np.arange(len(sim.data[net.probe_value])) * float(params["dt"])
    value_decoded = sim.data[net.probe_value].squeeze()
    responses = _extract_responses(t_arr, value_decoded, n_obs, params)
    if not return_probes:
        return responses
    probe_data = {
        "obs": sim.data[net.probe_obs].squeeze(),
        "error": sim.data[net.probe_error],
        "value": sim.data[net.probe_value].squeeze(),
        "counting_weight": sim.data[net.probe_counting_weight].squeeze(),
        "counting_count": sim.data[net.probe_counting_count].squeeze(),
        "t": np.arange(len(sim.data[net.probe_value])) * float(params["dt"]),
    }
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    dt = float(params["dt"])
    readout_offset = 0.5
    error_neuron_data = sim.data[net.probe_error_neurons]
    readout_indices = []
    for n in range(n_obs):
        t_readout = t_iti + n * t_step + readout_offset
        idx = int(np.round(t_readout / dt))
        idx = int(np.clip(idx, 0, len(error_neuron_data) - 1))
        readout_indices.append(idx)
    probe_data["error_neurons"] = error_neuron_data[readout_indices]
    if return_probes and hasattr(net, "probe_iti_noise"):
        probe_data["iti_noise"] = sim.data[net.probe_iti_noise].squeeze()
    return responses, probe_data


def run(
    params: dict,
    save: bool = False,
    trials: list | None = None,
    save_probes: bool = False,
) -> pd.DataFrame:
    """Run the NEF model for a single participant."""
    pfull = {**PARAM_DEFAULTS, **params}
    pfull["nef_type"] = "recurrent"

    required = (
        "model_type",
        "dataset",
        "pid",
        "t_obs",
        "t_iti",
        "dt",
        "tau_probe",
        "seed",
    )
    for key in required:
        if key not in pfull:
            raise KeyError(f"params must include {key!r}")

    dataset = pfull["dataset"]
    pid = int(pfull["pid"])

    # Every dataset -- carrabin, yoo, soltani_numbers, soltani_colors -- loads
    # real per-participant human data from its own pkl. An earlier version of
    # this function special-cased the soltani datasets to read the RETIRED
    # task/sequences/{continuous,binary}_sequences.pkl files with a dummy pid,
    # from back when no real human data existed for them; that branch is gone.
    # It silently discarded the `pid` argument, so an NEF fit would have
    # simulated old task/ sequences while fitting.losses scored the result
    # against real participant responses.
    #
    # NOT YET SAFE FOR THE SOLTANI DATASETS -- two known issues, deliberately
    # left for the NEF integration pass rather than fixed blind here:
    #   1. utils/binary_transform.nef_obs_values / nef_response_to_model_scale
    #      still assume soltani_numbers `value`/`response` are on the NATIVE
    #      [0,100] scale, but scripts/build_model_inputs.build_from_df already
    #      rescales them to [-1,1]. Running as-is double-rescales observations
    #      (collapsing the stimulus range to ~[-1.02,-0.98]) and returns
    #      responses on [0,1] against human responses on [-1,1].
    #   2. soltani trials/observations are 0-indexed (trial 0-31, obs 0-14),
    #      unlike carrabin/yoo (1-indexed). The counting-activity map below is
    #      keyed 1..n_trials, so trial 0 MISSES and falls through to the slow
    #      _pretrain path (which also passes base_seed, itself deprecated).
    human_pid = pd.read_pickle(data_path(f"{dataset}.pkl")).query("pid == @pid")
    if trials is not None:
        human_pid = human_pid[human_pid["trial"].isin(trials)]

    # Load precomputed counting network activities (Gram matrices).
    # If available, W_weight is recomputed per-trial via fast_decode using
    # the current (alpha_0, lambda_) — 300x faster than re-running Nengo.
    # Falls back to _pretrain if the activity file is not found.
    _activity_map: dict | None = None
    try:
        _activity_map = load_counting_activities(
            n_neurons=int(pfull["n_neurons"]),
            n_neurons_counting=int(pfull["n_neurons_counting"]),
            dataset=str(pfull.get("dataset", "carrabin")),
        )
    except FileNotFoundError:
        decoders = _pretrain(pfull)

    rows = []
    all_probe_data: list[dict] = []

    for trial, trial_data in human_pid.groupby("trial"):
        t_trial = time.time()
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        obs_values = nef_obs_values(obs_values, dataset)
        # seed = trial number directly
        p = {**pfull, "seed": int(trial)}
        if _activity_map is not None:
            activity = _activity_map.get(int(trial))
            if activity is not None:
                decoders = fast_decode_counting(
                    activity,
                    alpha_0=float(pfull["alpha_0"]),
                    lambda_=float(pfull["lambda_"]),
                )
            else:
                decoders = _pretrain({**p, "base_seed": int(trial)})
        if save_probes:
            responses, probe_data = _simulate_trial(
                obs_values, p, decoders, return_probes=True
            )
            probe_data["trial"] = int(trial)
            probe_data["params"] = dict(p)
            all_probe_data.append(probe_data)
        else:
            responses = _simulate_trial(obs_values, p, decoders)
        elapsed_trial = time.time() - t_trial
        print(f"  pid={pid} trial {int(trial)}: {elapsed_trial:.1f}s", flush=True)
        for i, (_, row) in enumerate(trial_data.iterrows()):
            entry = {
                "model_type": pfull["model_type"],
                "pid": pid,
                "trial": int(trial),
                "observation": int(row["observation"]),
                "response": nef_response_to_model_scale(float(responses[i]), dataset),
            }
            rows.append(entry)

    out = apply_binary_transform(pd.DataFrame(rows), dataset)
    if save_probes and all_probe_data:
        fname = f"probe_{pfull['model_type']}_{dataset}_{pid}.pkl"
        pd.to_pickle(all_probe_data, data_path(fname))
        print(f"  Saved probe data ({len(all_probe_data)} trials) to data/{fname}")
    if save:
        out.to_pickle(data_path(f"{pfull['model_type']}_{dataset}_{pid}.pkl"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEF evidence integration")
    p.add_argument(
        "--dataset",
        type=str,
        default="carrabin",
        choices=("carrabin", "yoo", "soltani_numbers", "soltani_colors"),
    )
    p.add_argument("--pid", type=int, default=1)
    p.add_argument("--model_type", type=str, default="NEF")
    p.add_argument("--n_obs", type=int, default=PARAM_DEFAULTS["n_obs"])
    p.add_argument("--n_neurons", type=int, default=PARAM_DEFAULTS["n_neurons"])
    p.add_argument(
        "--n_neurons_counting",
        type=int,
        default=PARAM_DEFAULTS["n_neurons_counting"],
    )
    p.add_argument("--lambda_", type=float, default=PARAM_DEFAULTS["lambda_"])
    p.add_argument("--tau_ff", type=float, default=PARAM_DEFAULTS["tau_ff"])
    p.add_argument("--tau_fb", type=float, default=PARAM_DEFAULTS["tau_fb"])
    p.add_argument("--T_error", type=float, default=PARAM_DEFAULTS["T_error"])
    p.add_argument("--tau_error", type=float, default=PARAM_DEFAULTS["tau_error"])
    p.add_argument("--onset_detector_amp", type=float, default=PARAM_DEFAULTS["onset_detector_amp"])
    p.add_argument("--tau_fast", type=float, default=PARAM_DEFAULTS["tau_fast"])
    p.add_argument("--tau_slow", type=float, default=PARAM_DEFAULTS["tau_slow"])
    p.add_argument("--tau_probe", type=float, default=PARAM_DEFAULTS["tau_probe"])
    p.add_argument("--radius_e", type=float, default=PARAM_DEFAULTS["radius_e"])
    p.add_argument("--radius_v", type=float, default=PARAM_DEFAULTS["radius_v"])
    p.add_argument("--pes_learning_rate", type=float, default=PARAM_DEFAULTS["pes_learning_rate"])
    p.add_argument("--dt", type=float, default=PARAM_DEFAULTS["dt"])
    p.add_argument("--t_obs", type=float, default=PARAM_DEFAULTS["t_obs"])
    p.add_argument("--t_iti", type=float, default=PARAM_DEFAULTS["t_iti"])
    p.add_argument("--seed", type=int, default=PARAM_DEFAULTS["seed"])
    p.add_argument("--alpha_0", type=float, default=PARAM_DEFAULTS["alpha_0"])
    p.add_argument("--save_probes", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = run(vars(args), save_probes=args.save_probes)
    print(df.head())
