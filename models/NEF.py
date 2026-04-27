#!/usr/bin/env python3
"""
NEF model of evidence integration (recurrent or synaptic value dynamics).

Architecture (per trial):
    counting subnetwork (LMU or integrator, pretrained decoders)
    counting → error[dim 0]   (alpha(n) via W_weight decoder)
    node_input[0] → error[dim 1]   (observation o(t))
    node_input[1] → error.neurons  (ITI inhibition)
    value → error[dim 1]      (transform=-1, subtracts v)

``nef_type`` (from ``model_type``): ``recurrent`` uses multiplicative
error→value and recurrent self-connection on ``value``; ``synaptic`` uses a
``spikes`` population and PES learning onto ``value``.

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
    simulate_network as simulate_counting_integrator,
)
from models.counting_lmu import (
    build_network as build_counting_lmu,
    decode_outputs as decode_counting_lmu,
    simulate_network as simulate_counting_lmu,
)
from utils.paths import data_path

from fitting.model_params import _NEF_FIXED

PARAM_DEFAULTS: dict = {
    **_NEF_FIXED,
    "n_obs": 30,
    "lambda_": 0.5,
    "alpha_0": 1,
    "T_error": 0.5,  # transform on error→value connection; defaults to tau_fb for backward compatibility
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


def _make_alpha_bias_input(obs_values: np.ndarray, params: dict) -> callable:
    """
    Outputs alpha_bias_array[step] during each observation, 0 during ITI.
    For non-jiang datasets, alpha_bias_array is all zeros so output is always 0.
    """
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    n_obs = len(obs_values)
    bias = np.array(params.get("alpha_bias_array", np.zeros(n_obs)), dtype=float)

    def fn(t: float) -> float:
        if t < t_iti:
            return 0.0
        step = int((t - t_iti) / t_step)
        phase = (t - t_iti) - step * t_step
        if step < n_obs and phase < t_obs:
            return float(bias[step])
        return 0.0

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
    """Run counting pretraining using lmu_n_obs_max observations."""
    p = {**params, "n_obs": params["lmu_n_obs_max"]}
    if p["counting"] == "lmu":
        net = build_counting_lmu(p, train=True)
        raw = simulate_counting_lmu(net, p, train=True)
        return decode_counting_lmu(raw, p)
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

    if params["counting"] == "lmu":
        _build_c = build_counting_lmu
    else:
        _build_c = build_counting_integrator

    with nengo.Network(label=str(params["model_type"]), seed=seed) as net:
        net.node_input = nengo.Node(
            _make_input(obs_values, params), size_out=2, label="node_input"
        )

        # Counting uses n_neurons_counting for memory / lmu_ea and n_neurons for
        # onset_detector (error and value use this n_neurons only).
        c_params = {**params, "n_obs": params["lmu_n_obs_max"]}
        net.counting = _build_c(c_params, train=False, decoders=decoders)

        net.error = nengo.Ensemble(
            n_neurons=int(params["n_neurons"]),
            dimensions=2,
            radius=float(params["radius_e"]),
            seed=seed,
            label="error",
        )
        net.node_alpha_bias = nengo.Node(
            _make_alpha_bias_input(obs_values, params),
            label="node_alpha_bias",
        )
        nengo.Connection(
            net.node_alpha_bias,
            net.error[0],
            synapse=None,
            seed=seed,
        )

        if params["counting"] == "lmu":
            nengo.Connection(
                net.counting.lmu_ea.output,
                net.error[0],
                transform=decoders["W_weight_neural"],
                synapse=float(params["lmu_tau"]),
                seed=seed,
            )
        else:
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

        if params["nef_type"] == "recurrent":
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

        elif params["nef_type"] == "synaptic":
            net.spikes = nengo.Ensemble(
                n_neurons=int(params["n_neurons"]),
                dimensions=1,
                seed=seed,
                label="spikes",
            )
            conn_value = nengo.Connection(
                net.spikes,
                net.value,
                synapse=tau_fb,
                learning_rule_type=nengo.PES(
                    learning_rate=float(params["pes_learning_rate"])
                ),
                function=lambda x: 0.0,
                seed=seed,
            )
            nengo.Connection(
                net.error,
                conn_value.learning_rule,
                function=lambda x: x[0] * x[1],
                transform=-1,
                synapse=tau_fb,
                seed=seed,
            )
        else:
            raise ValueError(
                f"nef_type must be 'recurrent' or 'synaptic', got {params['nef_type']!r}"
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
        "t": np.arange(len(sim.data[net.probe_value])) * float(params["dt"]),
    }
    return responses, probe_data


def run(
    params: dict,
    save: bool = False,
    trials: list | None = None,
    save_probes: bool = False,
) -> pd.DataFrame:
    """Run the NEF model for a single participant."""
    pfull = {**PARAM_DEFAULTS, **params}
    pfull["nef_type"] = (
        "synaptic" if "synaptic" in pfull["model_type"] else "recurrent"
    )

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

    human_pid = pd.read_pickle(data_path(f"{dataset}.pkl")).query("pid == @pid")
    if trials is not None:
        human_pid = human_pid[human_pid["trial"].isin(trials)]

    first_trial = int(sorted(human_pid["trial"].unique())[0]) if len(human_pid) else 0
    decoders = _pretrain(pfull)
    rows = []

    for trial, trial_data in human_pid.groupby("trial"):
        t_trial = time.time()
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        alpha_bias = (
            trial_data["rd"].to_numpy(dtype=float)
            if pfull["dataset"] == "jiang"
            else np.zeros(len(obs_values))
        )
        p = {**pfull, "alpha_bias_array": alpha_bias}
        if save_probes and int(trial) == first_trial:
            responses, probe_data = _simulate_trial(
                obs_values, p, decoders, return_probes=True
            )
            probe_data["params"] = dict(p)
            fname = f"probe_{pfull['model_type']}_{dataset}_{pid}.pkl"
            pd.to_pickle(probe_data, data_path(fname))
            print(f"  Saved probe data to data/{fname}")
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
                "response": float(responses[i]),
            }
            if pfull["dataset"] == "jiang":
                entry["stage"] = int(row["stage"])
            rows.append(entry)

    out = pd.DataFrame(rows)
    if save:
        out.to_pickle(data_path(f"{pfull['model_type']}_{dataset}_{pid}.pkl"))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEF evidence integration (recurrent / synaptic)")
    p.add_argument("--dataset", type=str, default="carrabin")
    p.add_argument("--pid", type=int, default=1)
    p.add_argument("--model_type", type=str, default="NEF_recurrent")
    p.add_argument("--counting", type=str, default=PARAM_DEFAULTS["counting"], choices=("lmu", "integrator"))
    p.add_argument("--n_seeds", type=int, default=PARAM_DEFAULTS["n_seeds"])
    p.add_argument("--n_obs", type=int, default=PARAM_DEFAULTS["n_obs"])
    p.add_argument("--n_neurons", type=int, default=PARAM_DEFAULTS["n_neurons"])
    p.add_argument(
        "--n_neurons_counting",
        type=int,
        default=PARAM_DEFAULTS["n_neurons_counting"],
    )
    p.add_argument("--lambda_", type=float, default=PARAM_DEFAULTS["lambda_"])
    p.add_argument("--lmu_order", type=int, default=PARAM_DEFAULTS["lmu_order"])
    p.add_argument("--lmu_tau", type=float, default=PARAM_DEFAULTS["lmu_tau"])
    p.add_argument("--lmu_n_obs_max", type=int, default=PARAM_DEFAULTS["lmu_n_obs_max"])
    p.add_argument("--lmu_theta_mult", type=float, default=PARAM_DEFAULTS["lmu_theta_mult"])
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
