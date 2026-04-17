#!/usr/bin/env python3
"""Integrator-based counting and weight decoding testbed."""

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
    rng = np.random.default_rng(int(params["seed"]))
    signs = rng.choice([-1.0, 1.0], size=n_obs)

    def pulse(t: float) -> float:
        if t < t_iti:
            return 0.0
        step = int((t - t_iti) / t_step)
        phase = (t - t_iti) - step * t_step
        if step < n_obs and phase < t_obs:
            return float(signs[step])
        return 0.0

    return pulse


def _ideal_stream(params: dict) -> callable:
    lambda_ = float(params["lambda_"])
    alpha_0 = float(params["alpha_0"])
    state = {"last_in": 0.0, "count": 0.0}

    def fn(t: float, x: np.ndarray) -> np.ndarray:
        inp = float(x[0])
        if abs(state["last_in"]) < 1e-12 and abs(inp) > 1e-12:
            state["count"] += 1.0
        state["last_in"] = inp
        c = state["count"]
        w = alpha_0 / max(c, 1.0) ** lambda_
        return np.array([c, w], dtype=np.float64)

    return fn


def build_network(params: dict, train: bool, decoders: dict | None = None) -> nengo.Network:
    n_obs = int(params["n_obs"])
    n_neurons = int(params["n_neurons"])
    seed = int(params["seed"])
    tau_fb = float(params["tau_fb"])
    tau_probe = float(params["tau_probe"])

    with nengo.Network(label="counting_integrator", seed=seed) as net:
        net.node_input = nengo.Node(make_pulse_input(params), label="node_input")
        net.ideal = nengo.Node(_ideal_stream(params), size_in=1, size_out=2, label="ideal")
        nengo.Connection(net.node_input, net.ideal, synapse=None, seed=seed)
        net.probe_ideal_raw = nengo.Probe(net.ideal, synapse=None)

        net.memory = nengo.Ensemble(
            n_neurons=n_neurons, dimensions=1, radius=n_obs, label="memory", seed=seed
        )
        # onset_detector: fires only on 0→±1 transitions
        # positive-only encoders + intercepts ensure it only fires when
        # fast-filtered |input| > slow-filtered |input| (i.e. rising edge)
        net.onset_detector = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=1,
            encoders=nengo.dists.Choice([[1]]),
            intercepts=nengo.dists.Uniform(0, 1),
            seed=seed,
            label="onset_detector",
        )
        net.onset_to_memory = nengo.Ensemble(
            n_neurons=1,
            dimensions=1,
            label="onset_to_memory",
            seed=seed,
            neuron_type=nengo.Direct(),
        )
        nengo.Connection(
            net.node_input,
            net.onset_detector,
            synapse=float(params["tau_fast"]),
            function=lambda x: np.abs(x),
            seed=seed,
        )
        nengo.Connection(
            net.node_input,
            net.onset_detector,
            synapse=float(params["tau_slow"]),
            function=lambda x: -np.abs(x),
            seed=seed,
        )
        amp = float(params["onset_detector_amp"])
        nengo.Connection(
            net.onset_detector,
            net.memory,
            synapse=tau_fb,
            function=lambda x, amp=amp: amp,
            seed=seed,
        )
        nengo.Connection(
            net.onset_detector,
            net.onset_to_memory,
            synapse=tau_fb,
            function=lambda x, amp=amp: amp,
            seed=seed,
        )
        nengo.Connection(net.memory, net.memory, transform=1.0, synapse=tau_fb, seed=seed)
        net.probe_onset_to_memory = nengo.Probe(net.onset_to_memory, synapse=None)
        net.probe_memory_raw = nengo.Probe(net.memory.neurons, synapse=None)
        net.probe_onset_detector = nengo.Probe(net.onset_detector, synapse=tau_probe)
        net.probe_memory_default_decoded = nengo.Probe(net.memory, synapse=tau_probe)

        if not train:
            if decoders is None:
                raise ValueError("decoders required when train=False")
            net.count_out = nengo.Ensemble(
                1, 1, neuron_type=nengo.Direct(), label="count_out", seed=seed
            )
            net.weight_out = nengo.Ensemble(
                1, 1, neuron_type=nengo.Direct(), label="weight_out", seed=seed
            )
            nengo.Connection(
                net.memory.neurons,
                net.count_out,
                transform=decoders["W_count"],
                synapse=tau_probe,
                seed=seed,
            )
            nengo.Connection(
                net.memory.neurons,
                net.weight_out,
                transform=decoders["W_weight"],
                synapse=tau_probe,
                seed=seed,
            )
            net.probe_count = nengo.Probe(net.count_out, synapse=None)
            net.probe_weight = nengo.Probe(net.weight_out, synapse=None)

    return net


def simulate_network(net: nengo.Network, params: dict, train: bool) -> dict:
    dt = float(params["dt"])
    n_obs = int(params["n_obs"])
    t_step = float(params["t_obs"]) + float(params["t_iti"])
    t_total = n_obs * t_step
    with nengo.Simulator(net, dt=dt, seed=int(params["seed"]), progress_bar=False) as sim:
        sim.run(t_total)

    if train:
        return {
            "ideal": sim.data[net.probe_ideal_raw],
            "memory": sim.data[net.probe_memory_raw],
            "onset_to_memory": sim.data[net.probe_onset_to_memory].squeeze(),
            "onset_detector": sim.data[net.probe_onset_detector].squeeze(),
            "memory_default_decoded": sim.data[net.probe_memory_default_decoded].squeeze(),
        }
    return {
        "ideal_count": sim.data[net.probe_ideal_raw][:, 0],
        "ideal_weight": sim.data[net.probe_ideal_raw][:, 1],
        "count": sim.data[net.probe_count].squeeze(),
        "weight": sim.data[net.probe_weight].squeeze(),
        "onset_to_memory": sim.data[net.probe_onset_to_memory].squeeze(),
        "onset_detector": sim.data[net.probe_onset_detector].squeeze(),
        "memory_default_decoded": sim.data[net.probe_memory_default_decoded].squeeze(),
    }


def decode_outputs(raw: dict, params: dict) -> dict:
    dt = float(params["dt"])
    tau_probe = float(params["tau_probe"])

    syn = nengo.Lowpass(tau_probe)
    ideal_count_filt = syn.filt(raw["ideal"][:, 0:1], dt=dt).ravel()
    ideal_weight_filt = syn.filt(raw["ideal"][:, 1:2], dt=dt).ravel()
    memory_filt = syn.filt(raw["memory"], dt=dt)

    solver = nengo.solvers.LstsqL2(reg=1e-3)
    W_count, _ = solver(memory_filt, ideal_count_filt[:, np.newaxis])
    W_weight, _ = solver(memory_filt, ideal_weight_filt[:, np.newaxis])
    return {"W_count": W_count.T, "W_weight": W_weight.T}


def _eval_idx(params: dict, t_len: int) -> np.ndarray:
    dt = float(params["dt"])
    n_obs = int(params["n_obs"])
    t_step = float(params["t_obs"]) + float(params["t_iti"])
    t_mid = (
        float(params["t_iti"])
        + float(params["t_obs"])
        + float(params["t_iti"]) / 2.0
    )
    times = np.array([i * t_step + t_mid for i in range(n_obs)])
    idx = np.clip(np.rint(times / dt).astype(int), 0, t_len - 1)
    return idx


def analysis(all_test_outputs: list[dict], params: dict) -> None:
    apply_style()
    palette = get_palette()
    nef_color = palette["NEF"]
    grey = "0.45"

    count_all = np.stack([o["count"] for o in all_test_outputs], axis=0)
    weight_all = np.stack([o["weight"] for o in all_test_outputs], axis=0)
    ideal_count = all_test_outputs[0]["ideal_count"]
    ideal_weight = all_test_outputs[0]["ideal_weight"]

    count_mean = count_all.mean(axis=0)
    count_std = count_all.std(axis=0)
    weight_mean = weight_all.mean(axis=0)
    weight_std = weight_all.std(axis=0)

    idx = _eval_idx(params, len(ideal_count))
    n_true_eval = ideal_count[idx]
    w_true_eval = ideal_weight[idx]
    err_count = count_mean[idx] - n_true_eval
    err_weight = weight_mean[idx] - w_true_eval
    rmse_count = float(np.sqrt(np.mean(err_count**2)))
    rmse_weight = float(np.sqrt(np.mean(err_weight**2)))

    t = np.arange(len(ideal_count)) * float(params["dt"])
    fig, axes = plt.subplots(2, 2, figsize=(FIGURE_SIZE[0] * 1.35, FIGURE_SIZE[1] * 1.35), constrained_layout=True)
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    ax00.plot(t, ideal_count, "--", color="0.5", label="ideal count")
    ax00.fill_between(t, count_mean - count_std, count_mean + count_std, color=nef_color, alpha=0.25)
    ax00.plot(t, count_mean, color=nef_color, label="integrator")
    ax00.set_title("Count dynamics")
    ax00.legend(frameon=False, fontsize=7)

    ax01.axhline(0, color="0.7", linestyle="--")
    ax01.scatter(n_true_eval, err_count, color=nef_color, s=20, label=f"integrator (RMSE={rmse_count:.3f})")
    ax01.set_title("Count error vs n")
    ax01.set_xlabel("True n")
    ax01.set_ylabel("Decoded - True")
    ax01.legend(frameon=False, fontsize=7)

    ax10.plot(t, ideal_weight, "--", color="0.5", label="ideal weight")
    ax10.fill_between(t, weight_mean - weight_std, weight_mean + weight_std, color=nef_color, alpha=0.25)
    ax10.plot(t, weight_mean, color=nef_color, label="integrator")
    ax10.set_title("Weight dynamics")
    ax10.legend(frameon=False, fontsize=7)

    ax11.axhline(0, color="0.7", linestyle="--")
    ax11.scatter(n_true_eval, err_weight, color=nef_color, s=20, label=f"integrator (RMSE={rmse_weight:.3f})")
    ax11.set_title("Weight error vs n")
    ax11.set_xlabel("True n")
    ax11.set_ylabel("Decoded - True")
    ax11.legend(frameon=False, fontsize=7)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "counting_integrator.png", dpi=300)
    plt.savefig(FIGURES_DIR / "counting_integrator.pdf")
    print("Saved figures/counting_integrator.{png,pdf}")

    otm_all = np.stack([o["onset_to_memory"] for o in all_test_outputs], axis=0)
    otm_mean = otm_all.mean(axis=0)
    otm_std = otm_all.std(axis=0)
    onset_all = np.stack([o["onset_detector"] for o in all_test_outputs], axis=0)
    onset_mean = onset_all.mean(axis=0)
    onset_std = onset_all.std(axis=0)
    mem_dec_all = np.stack([o["memory_default_decoded"] for o in all_test_outputs], axis=0)
    mem_dec_mean = mem_dec_all.mean(axis=0)
    mem_dec_std = mem_dec_all.std(axis=0)
    pulse_fn = make_pulse_input(params)
    true_pulse = np.array([pulse_fn(float(ti)) for ti in t])

    fig2, (ax0, ax1, ax2, ax3) = plt.subplots(
        4, 1, figsize=(FIGURE_SIZE[0] * 1.35, FIGURE_SIZE[1] * 1.6), constrained_layout=True
    )
    ax0.plot(t, true_pulse, color="0.35", linewidth=0.8)
    ax0.set_ylabel("pulse")
    ax0.set_title("Stimulus (node_input)")
    ax1.fill_between(t, otm_mean - otm_std, otm_mean + otm_std, color="0.55", alpha=0.25)
    ax1.plot(t, otm_mean, color="0.35", linewidth=0.8, label="onset→memory")
    ax1.set_ylabel("drive")
    ax1.set_title("Onset → memory (probe)")
    ax1.legend(frameon=False, fontsize=7)
    ax2.fill_between(t, onset_mean - onset_std, onset_mean + onset_std, color=nef_color, alpha=0.25)
    ax2.plot(t, onset_mean, color=nef_color, linewidth=0.8, label="onset (default decoded)")
    ax2.set_ylabel("onset")
    ax2.set_title("Onset detector (default decoded)")
    ax2.legend(frameon=False, fontsize=7)
    ax3.fill_between(t, mem_dec_mean - mem_dec_std, mem_dec_mean + mem_dec_std, color=nef_color, alpha=0.2)
    ax3.plot(t, mem_dec_mean, color=nef_color, linewidth=0.8, label="memory (default decoded)")
    ax3.plot(t, ideal_count, "--", color="0.5", linewidth=0.8, label="ideal count")
    ax3.set_ylabel("count")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Memory (default decoded)")
    ax3.legend(frameon=False, fontsize=7)
    plt.savefig(FIGURES_DIR / "counting_integrator_onset.png", dpi=300)
    plt.savefig(FIGURES_DIR / "counting_integrator_onset.pdf")
    plt.close(fig2)
    print("Saved figures/counting_integrator_onset.{png,pdf}")


def run(params: dict) -> None:
    all_outputs = []
    for s in range(int(params["seed"]), int(params["seed"]) + int(params["n_seeds"])):
        p = {**params, "seed": s}
        # train with this seed
        net_train = build_network(p, train=True)
        raw = simulate_network(net_train, p, train=True)
        decoders = decode_outputs(raw, p)
        # test with same seed — guarantees neuron tuning curves match
        net_test = build_network(p, train=False, decoders=decoders)
        outputs = simulate_network(net_test, p, train=False)
        all_outputs.append(outputs)
    analysis(all_outputs, params)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrator counting and weight decoding")
    p.add_argument("--n_obs", type=int, default=30)
    p.add_argument("--n_neurons", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--alpha_0", type=float, default=1.0)
    p.add_argument("--tau_fb", type=float, default=0.2)
    p.add_argument("--onset_detector_amp", type=float, default=0.3)
    p.add_argument("--tau_fast", type=float, default=0.01)
    p.add_argument("--tau_slow", type=float, default=0.2)
    p.add_argument("--tau_probe", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--t_obs", type=float, default=1.0)
    p.add_argument("--t_iti", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(vars(args))
