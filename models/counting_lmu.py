#!/usr/bin/env python3
"""LMU-based counting and weight decoding testbed."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nengo
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette


def make_pulse_input(params: dict) -> callable:
    n_obs = int(params["n_obs"])
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti

    def pulse(t: float) -> float:
        step = int(t / t_step)
        phase = t - step * t_step
        if step < n_obs and phase < t_obs:
            return 1.0
        return 0.0

    return pulse


def _compute_lmu_matrices(params: dict) -> tuple[np.ndarray, np.ndarray]:
    from nengo.utils.filter_design import cont2discrete as nengo_c2d

    n_obs = int(params["lmu_n_obs_max"])
    lmu_order = int(params["lmu_order"])
    theta = n_obs * (float(params["t_obs"]) + float(params["t_iti"])) * float(params["lmu_theta_mult"])
    dt = float(params["dt"])

    q = np.arange(lmu_order, dtype=np.float64)
    r = (2 * q + 1)[:, None] / theta
    j, i = np.meshgrid(q, q)
    a_cont = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * r
    b_cont = ((-1.0) ** q)[:, None] * r
    c = np.ones((1, lmu_order))
    d = np.zeros((1,))
    a_disc, b_disc, _, _, _ = nengo_c2d((a_cont, b_cont, c, d), dt=dt, method="zoh")
    return a_disc, b_disc


def _ideal_stream(params: dict) -> callable:
    lambda_ = float(params["lambda_"])
    state = {"last_in": 0.0, "count": 0.0}

    def fn(t: float, x: np.ndarray) -> np.ndarray:
        inp = float(x[0])
        if state["last_in"] <= 0.0 and inp > 0.0:
            state["count"] += 1.0
        state["last_in"] = inp
        c = state["count"]
        w = 1.0 / max(c, 1.0) ** lambda_
        return np.array([c, w], dtype=np.float64)

    return fn


def build_network(params: dict, train: bool, decoders: dict | None = None) -> nengo.Network:
    seed = int(params["seed"])
    n_neurons = int(params["n_neurons"])
    lmu_order = int(params["lmu_order"])
    lmu_tau = float(params["lmu_tau"])
    tau_probe = float(params["tau_probe"])
    a_disc, b_disc = _compute_lmu_matrices(params)
    radius = 2.0

    with nengo.Network(label="counting_lmu", seed=seed) as net:
        net.node_input = nengo.Node(make_pulse_input(params), label="node_input")
        net.ideal = nengo.Node(_ideal_stream(params), size_in=1, size_out=2, label="ideal")
        nengo.Connection(net.node_input, net.ideal, synapse=None, seed=seed)
        net.probe_ideal_raw = nengo.Probe(net.ideal, synapse=None)

        # Stream 2 — lmu_math (Legendre coefficients on Node)
        net.lmu_math_node = nengo.Node(size_in=lmu_order, label="lmu_math_node")
        nengo.Connection(
            net.node_input, net.lmu_math_node, transform=b_disc, synapse=None, seed=seed
        )
        nengo.Connection(
            net.lmu_math_node, net.lmu_math_node, transform=a_disc, synapse=0.0, seed=seed
        )
        net.probe_lmu_math_raw = nengo.Probe(net.lmu_math_node, synapse=None)

        # Stream 3 — lmu_neural (coefficients decoded from spikes inside EnsembleArray)
        net.lmu_ea = nengo.networks.EnsembleArray(
            n_neurons=n_neurons,
            n_ensembles=lmu_order,
            neuron_type=nengo.SpikingRectifiedLinear(),
            radius=radius,
            seed=seed,
            label="lmu_ea",
        )
        nengo.Connection(
            net.node_input, net.lmu_ea.input, transform=b_disc, synapse=lmu_tau, seed=seed
        )
        nengo.Connection(
            net.lmu_ea.output, net.lmu_ea.input, transform=a_disc, synapse=lmu_tau, seed=seed
        )
        net.probe_lmu_neural_raw = nengo.Probe(net.lmu_ea.output, synapse=lmu_tau)

        if not train:
            if decoders is None:
                raise ValueError("decoders required when train=False")
            net.lmu_math_count_out = nengo.Ensemble(
                1, 1, neuron_type=nengo.Direct(), label="lmu_math_count_out", seed=seed
            )
            net.lmu_math_weight_out = nengo.Ensemble(
                1, 1, neuron_type=nengo.Direct(), label="lmu_math_weight_out", seed=seed
            )
            nengo.Connection(
                net.lmu_math_node,
                net.lmu_math_count_out,
                transform=decoders["W_count_math"],
                synapse=tau_probe,
                seed=seed,
            )
            nengo.Connection(
                net.lmu_math_node,
                net.lmu_math_weight_out,
                transform=decoders["W_weight_math"],
                synapse=tau_probe,
                seed=seed,
            )
            net.lmu_neural_count_out = nengo.Ensemble(
                1, 1, neuron_type=nengo.Direct(), label="lmu_neural_count_out", seed=seed
            )
            net.lmu_neural_weight_out = nengo.Ensemble(
                1, 1, neuron_type=nengo.Direct(), label="lmu_neural_weight_out", seed=seed
            )
            nengo.Connection(
                net.lmu_ea.output,
                net.lmu_neural_count_out,
                transform=decoders["W_count_neural"],
                synapse=tau_probe,
            )
            nengo.Connection(
                net.lmu_ea.output,
                net.lmu_neural_weight_out,
                transform=decoders["W_weight_neural"],
                synapse=tau_probe,
            )

            net.probe_lmu_math_count = nengo.Probe(net.lmu_math_count_out, synapse=tau_probe)
            net.probe_lmu_math_weight = nengo.Probe(net.lmu_math_weight_out, synapse=tau_probe)
            net.probe_lmu_neural_count = nengo.Probe(net.lmu_neural_count_out, synapse=tau_probe)
            net.probe_lmu_neural_weight = nengo.Probe(net.lmu_neural_weight_out, synapse=tau_probe)

    return net


def simulate_network(net: nengo.Network, params: dict, train: bool) -> dict:
    dt = float(params["dt"])
    n_obs = int(params["n_obs"])
    t_step = float(params["t_obs"]) + float(params["t_iti"])
    t_total = n_obs * t_step + float(params["t_iti"])
    with nengo.Simulator(net, dt=dt, seed=int(params["seed"]), progress_bar=False) as sim:
        sim.run(t_total)

    if train:
        return {
            "ideal": sim.data[net.probe_ideal_raw],
            "lmu_math": sim.data[net.probe_lmu_math_raw],
            "lmu_neural": sim.data[net.probe_lmu_neural_raw],
        }
    return {
        "ideal_count": sim.data[net.probe_ideal_raw][:, 0],
        "ideal_weight": sim.data[net.probe_ideal_raw][:, 1],
        "lmu_math_count": sim.data[net.probe_lmu_math_count].squeeze(),
        "lmu_math_weight": sim.data[net.probe_lmu_math_weight].squeeze(),
        "lmu_neural_count": sim.data[net.probe_lmu_neural_count].squeeze(),
        "lmu_neural_weight": sim.data[net.probe_lmu_neural_weight].squeeze(),
    }


def decode_outputs(raw: dict, params: dict) -> dict:
    dt = float(params["dt"])
    tau_probe = float(params["tau_probe"])
    syn = nengo.Lowpass(tau_probe)

    # 1D ideal columns need (T, 1) for Lowpass.filt (same pattern as counting_integrator)
    ideal_count_filt = syn.filt(raw["ideal"][:, 0:1], dt=dt).ravel()
    ideal_weight_filt = syn.filt(raw["ideal"][:, 1:2], dt=dt).ravel()
    lmu_math_filt = syn.filt(raw["lmu_math"], dt=dt)
    lmu_neural_filt = syn.filt(raw["lmu_neural"], dt=dt)

    solver = nengo.solvers.LstsqL2(reg=1e-3)
    W_count_math, _ = solver(lmu_math_filt, ideal_count_filt[:, np.newaxis])
    W_weight_math, _ = solver(lmu_math_filt, ideal_weight_filt[:, np.newaxis])
    W_count_neural, _ = solver(lmu_neural_filt, ideal_count_filt[:, np.newaxis])
    W_weight_neural, _ = solver(lmu_neural_filt, ideal_weight_filt[:, np.newaxis])

    return {
        "W_count_math": W_count_math.T,
        "W_weight_math": W_weight_math.T,
        "W_count_neural": W_count_neural.T,
        "W_weight_neural": W_weight_neural.T,
    }


def _eval_idx(params: dict, t_len: int) -> np.ndarray:
    dt = float(params["dt"])
    n_obs = int(params["n_obs"])
    t_step = float(params["t_obs"]) + float(params["t_iti"])
    t_mid = float(params["t_obs"]) + float(params["t_iti"]) / 2.0
    times = np.array([i * t_step + t_mid for i in range(n_obs)])
    idx = np.clip(np.rint(times / dt).astype(int), 0, t_len - 1)
    return idx


def analysis(all_test_outputs: list[dict], params: dict) -> None:
    apply_style()
    palette = get_palette()
    nef_color = palette["NEF"]
    grey = "0.45"

    ideal_count = all_test_outputs[0]["ideal_count"]
    ideal_weight = all_test_outputs[0]["ideal_weight"]
    math_count = np.stack([o["lmu_math_count"] for o in all_test_outputs], axis=0)
    neural_count = np.stack([o["lmu_neural_count"] for o in all_test_outputs], axis=0)
    math_weight = np.stack([o["lmu_math_weight"] for o in all_test_outputs], axis=0)
    neural_weight = np.stack([o["lmu_neural_weight"] for o in all_test_outputs], axis=0)

    math_count_mean = math_count.mean(axis=0)
    neural_count_mean = neural_count.mean(axis=0)
    neural_count_std = neural_count.std(axis=0)
    math_weight_mean = math_weight.mean(axis=0)
    neural_weight_mean = neural_weight.mean(axis=0)
    neural_weight_std = neural_weight.std(axis=0)

    idx = _eval_idx(params, len(ideal_count))
    n_true_eval = ideal_count[idx]
    w_true_eval = ideal_weight[idx]
    err_count_math = math_count_mean[idx] - n_true_eval
    err_count_neural = neural_count_mean[idx] - n_true_eval
    err_weight_math = math_weight_mean[idx] - w_true_eval
    err_weight_neural = neural_weight_mean[idx] - w_true_eval
    rmse_cm = float(np.sqrt(np.mean(err_count_math**2)))
    rmse_cn = float(np.sqrt(np.mean(err_count_neural**2)))
    rmse_wm = float(np.sqrt(np.mean(err_weight_math**2)))
    rmse_wn = float(np.sqrt(np.mean(err_weight_neural**2)))

    t = np.arange(len(ideal_count)) * float(params["dt"])
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_SIZE[0] * 1.35, FIGURE_SIZE[1] * 1.35), constrained_layout=True)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    ax00.plot(t, ideal_count, "--", color="0.5", label="ideal count")
    ax00.plot(t, math_count_mean, color=grey, label="lmu_math")
    ax00.fill_between(t, neural_count_mean - neural_count_std, neural_count_mean + neural_count_std, color=nef_color, alpha=0.25)
    ax00.plot(t, neural_count_mean, color=nef_color, label="lmu_neural")
    ax00.set_title("Count dynamics")
    ax00.legend(frameon=False, fontsize=7)

    ax01.axhline(0, color="0.7", linestyle="--")
    ax01.scatter(n_true_eval, err_count_math, color=grey, s=20, label=f"lmu_math (RMSE={rmse_cm:.3f})")
    ax01.scatter(n_true_eval, err_count_neural, color=nef_color, s=20, label=f"lmu_neural (RMSE={rmse_cn:.3f})")
    ax01.set_title("Count error vs n")
    ax01.set_xlabel("True n")
    ax01.set_ylabel("Decoded - True")
    ax01.legend(frameon=False, fontsize=7)

    ax10.plot(t, ideal_weight, "--", color="0.5", label="ideal weight")
    ax10.plot(t, math_weight_mean, color=grey, label="lmu_math")
    ax10.fill_between(t, neural_weight_mean - neural_weight_std, neural_weight_mean + neural_weight_std, color=nef_color, alpha=0.25)
    ax10.plot(t, neural_weight_mean, color=nef_color, label="lmu_neural")
    ax10.set_title("Weight dynamics")
    ax10.legend(frameon=False, fontsize=7)

    ax11.axhline(0, color="0.7", linestyle="--")
    ax11.scatter(n_true_eval, err_weight_math, color=grey, s=20, label=f"lmu_math (RMSE={rmse_wm:.3f})")
    ax11.scatter(n_true_eval, err_weight_neural, color=nef_color, s=20, label=f"lmu_neural (RMSE={rmse_wn:.3f})")
    ax11.set_title("Weight error vs n")
    ax11.set_xlabel("True n")
    ax11.set_ylabel("Decoded - True")
    ax11.legend(frameon=False, fontsize=7)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "counting_lmu.png", dpi=300)
    plt.savefig(FIGURES_DIR / "counting_lmu.pdf")
    print("Saved figures/counting_lmu.{png,pdf}")


def run(params: dict) -> None:
    all_outputs = []
    for s in range(int(params["seed"]), int(params["seed"]) + int(params["n_seeds"])):
        p = {**params, "seed": s}
        net_train = build_network(p, train=True)
        raw = simulate_network(net_train, p, train=True)
        decoders = decode_outputs(raw, p)
        net_test = build_network(p, train=False, decoders=decoders)
        outputs = simulate_network(net_test, p, train=False)
        all_outputs.append(outputs)
    analysis(all_outputs, params)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LMU counting and weight decoding")
    p.add_argument("--n_obs", type=int, default=30)
    p.add_argument("--n_neurons", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--lmu_order", type=int, default=24)
    p.add_argument("--lmu_tau", type=float, default=0.2)
    p.add_argument("--lmu_n_obs_max", type=int, default=30)
    p.add_argument("--lmu_theta_mult", type=float, default=1.1)
    p.add_argument("--tau_probe", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--t_obs", type=float, default=1.0)
    p.add_argument("--t_iti", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(vars(args))
