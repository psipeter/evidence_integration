#!/usr/bin/env python3
"""Integrator-based counting and weight decoding testbed."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nengo
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import _NEF_FIXED

# Number of trials per dataset -- precompute one activity set per (trial, sim)
# seed. Also used as the per-sim BLOCK SIZE in activity_key_for_trial's
# sim-offset formula (added for NEF's NLL branch -- see docs/HISTORY.md):
# sim s's keys occupy [(s-1)*N + 1, s*N], so this value must stay in sync
# with how many trial seeds precompute_activities actually generates per sim.
_DATASET_N_TRIALS = {"carrabin": 200, "yoo": 30, "soltani_numbers": 40, "soltani_colors": 40}

# Datasets whose `trial` column is 0-indexed. carrabin and yoo are 1-indexed
# (trial 1..N); the soltani datasets are 0-indexed (trial 0..31), inherited from
# task_backend's own 0-based trial_index. Everything else about the activity
# pipeline assumes 1-based keys, so this set is what reconciles the two -- see
# activity_key_for_trial below.
_ZERO_INDEXED_DATASETS = frozenset({"soltani_numbers", "soltani_colors"})


def activity_key_for_trial(dataset: str, trial: int, sim: int = 1) -> int:
    """Map a (dataset, trial, sim) triple to its counting-activity key AND seed.

    precompute_activities() builds entry `k` by simulating a network with
    ``seed = k``. So an activity key is not an arbitrary index -- it
    IDENTIFIES A SEED. Its stored MtM is the Gram matrix of that network's
    filtered memory activity, and decoders solved from it via fast_decode()
    are valid ONLY for a network built with the same seed.

    That makes this function the single source of truth for both halves of the
    pairing: callers must use the returned value as the activity-map key AND as
    the `seed` passed to the simulation network. Using it for only one of the two
    silently mismatches the decoders against different tuning curves, which
    produces plausible-looking but meaningless output rather than an error.

    For 1-indexed datasets (carrabin, yoo) the base key is the identity. For
    0-indexed datasets (soltani_*) it is trial+1, so trials 0..31 use keys/seeds
    1..32 -- all within the 40 precomputed entries. The alternative, leaving
    trial 0 to miss the map, sends it down the ~300x slower _pretrain path and
    gives that one trial decoders derived by a different procedure than its 31
    siblings.

    `sim` (added for NEF's NLL branch -- see docs/HISTORY.md and
    models.NEF.simulate_ensemble). For sim=1 (the default) this is IDENTICAL
    to the original single-seed-per-trial behaviour above -- a pure
    extension, not a format change. For sim>1, the key is offset by a full
    dataset-sized BLOCK per sim: (sim-1)*_DATASET_N_TRIALS[dataset] + base.
    This gives every (trial, sim) pair a GENUINELY DISTINCT seed, and
    therefore genuinely distinct neural tuning curves, rather than reusing
    one seed across different trials -- reusing a seed across trials would
    silently CORRELATE supposedly-independent ensemble members, since that
    seed's idiosyncratic tuning-curve bias would show up identically in
    every trial that reused it rather than as independent noise per trial.
    This is why a genuine NLL ensemble for NEF needs n_trials * n_sims
    precomputed entries, not just n_sims -- see precompute_activities' own
    n_sims parameter.

    NEVER hand-derive a key with this formula inline, for either the
    activity-map lookup or the seed passed to the simulation -- always call
    this function for both. The two have already drifted apart once in this
    codebase's history (a bare `_activity_map.get(trial)` silently missing
    0-indexed trial 0); a second, hand-rolled copy of the sim-offset
    arithmetic is the same class of risk.
    """
    t = int(trial)
    base = t + 1 if str(dataset) in _ZERO_INDEXED_DATASETS else t
    if sim <= 1:
        return base
    return (int(sim) - 1) * _DATASET_N_TRIALS[str(dataset)] + base
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
    n_neurons_counting = int(params["n_neurons_counting"])
    seed = int(params["seed"])
    tau_fb = float(params["tau_fb"])
    tau_probe = float(params["tau_probe"])

    with nengo.Network(label="counting_integrator", seed=seed) as net:
        net.node_input = nengo.Node(make_pulse_input(params), label="node_input")
        net.ideal = nengo.Node(_ideal_stream(params), size_in=1, size_out=2, label="ideal")
        nengo.Connection(net.node_input, net.ideal, synapse=None, seed=seed)
        net.probe_ideal_raw = nengo.Probe(net.ideal, synapse=None)

        # radius_c: representational range of counting memory ensemble.
        # Set per-dataset (carrabin=5, yoo=30) so neurons are tuned
        # to the exact count range needed.
        radius_memory = float(params.get("radius_c", n_obs))
        net.memory = nengo.Ensemble(
            n_neurons=n_neurons_counting,
            dimensions=1,
            radius=radius_memory,
            label="memory",
            seed=seed,
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
    pal = get_palette(10)
    nef_color = pal[3]
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
    p.add_argument("--n_neurons", type=int, default=_NEF_FIXED["n_neurons"])
    p.add_argument(
        "--n_neurons_counting",
        type=int,
        default=_NEF_FIXED["n_neurons_counting"],
    )
    p.add_argument("--seed", type=int, default=_NEF_FIXED["seed"])
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--lambda_", type=float, default=0.5)
    p.add_argument("--alpha_0", type=float, default=1.0)
    p.add_argument("--tau_fb", type=float, default=_NEF_FIXED["tau_fb"])
    p.add_argument(
        "--onset_detector_amp",
        type=float,
        default=_NEF_FIXED["onset_detector_amp"],
    )
    p.add_argument("--tau_fast", type=float, default=_NEF_FIXED["tau_fast"])
    p.add_argument("--tau_slow", type=float, default=_NEF_FIXED["tau_slow"])
    p.add_argument("--tau_probe", type=float, default=_NEF_FIXED["tau_probe"])
    p.add_argument("--dt", type=float, default=_NEF_FIXED["dt"])
    p.add_argument("--t_obs", type=float, default=_NEF_FIXED["t_obs"])
    p.add_argument("--t_iti", type=float, default=_NEF_FIXED["t_iti"])
    p.add_argument("--precompute", action="store_true",
                   help="Precompute and save decoder sets for N trial seeds")
    p.add_argument("--precompute_activities", action="store_true",
                   help="Precompute and save counting network activity Gram matrices")
    p.add_argument("--n_sims", type=int, default=1,
                   help="Distinct seeds PER TRIAL to precompute (default 1, the "
                        "original single-seed-per-trial behaviour). >1 is for "
                        "NEF's NLL branch, where an ensemble needs n_trials*n_sims "
                        "genuinely distinct seeds -- see activity_key_for_trial's "
                        "own docstring for why reusing seeds across trials would "
                        "silently correlate supposedly-independent ensemble members.")
    p.add_argument("--plot_activities", action="store_true",
                   help="Load saved activities, decode, and plot ideal vs decoded")
    p.add_argument("--dataset", type=str, default=None,
                   choices=("carrabin", "yoo", "soltani_numbers", "soltani_colors"),
                   help="Task dataset — sets radius_c automatically "
                        "(carrabin=5, yoo=30). Overrides _NEF_FIXED default.")
    p.add_argument("--base_seed", type=int, default=0,
                   help="Base seed for generating trial seeds (default 0)")
    return p.parse_args()


def precompute_decoders(
    seeds: list[tuple[int, int]],  # list of (trial_number, seed) pairs
    params: dict,
    verbose: bool = True,
) -> dict[int, dict]:
    """Pretrain counting decoders for a list of seeds.

    Each seed produces an independent set of decoders (W_count, W_weight)
    that can be loaded per-trial in NEF.run() instead of retraining per trial
    or using a single shared base_seed.

    Parameters
    ----------
    seeds  : list of integer seeds (one per trial)
    params : base params dict (must include timing, n_neurons, n_neurons_counting)

    Returns
    -------
    dict mapping seed -> {"W_count": ..., "W_weight": ...}
    """
    import time
    decoder_map = {}
    n = len(seeds)
    for i, (trial, seed) in enumerate(seeds):
        p = {**params, "seed": seed, "n_obs": int(params["radius_c"])}
        t0  = time.time()
        net = build_network(p, train=True)
        raw = simulate_network(net, p, train=True)
        dec = decode_outputs(raw, p)
        decoder_map[trial] = dec   # keyed by trial number
        if verbose and (i % 20 == 0 or i == n - 1):
            print(f"  [{i+1}/{n}] trial={trial} seed={seed}  {time.time()-t0:.1f}s",
                  flush=True)
    return decoder_map


def save_decoders(
    seeds: list[int],
    params: dict,
    out_path: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Precompute and save decoder map to disk.

    File is named counting_decoders_n{n_neurons}_nc{n_neurons_counting}.pkl
    and saved to data/ by default.
    """
    import pickle
    from utils.paths import data_path

    n  = int(params["n_neurons"])
    nc = int(params["n_neurons_counting"])
    if out_path is None:
        out_path = data_path(f"counting_decoders_n{n}_nc{nc}.pkl")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Precomputing {len(seeds)} decoder sets  "
              f"(n_neurons={n}, n_neurons_counting={nc})")
    decoder_map = precompute_decoders(seeds, params, verbose=verbose)
    with open(out_path, "wb") as f:
        pickle.dump(decoder_map, f, protocol=4)
    print(f"Saved {len(decoder_map)} decoder sets -> {out_path}")
    return out_path


def load_decoders(
    n_neurons: int | None = None,
    n_neurons_counting: int | None = None,
    path: str | Path | None = None,
) -> dict[int, dict]:
    """Load precomputed decoder map from disk."""
    import pickle
    from utils.paths import data_path

    if path is None:
        n  = n_neurons  or _NEF_FIXED["n_neurons"]
        nc = n_neurons_counting or _NEF_FIXED["n_neurons_counting"]
        path = data_path(f"counting_decoders_n{n}_nc{nc}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def precompute_activities(
    n_trials: int | None,
    params: dict,
    out_path: str | Path | None = None,
    verbose: bool = True,
    n_sims: int = 1,
) -> Path:
    """Simulate the counting network for each (sim, trial) seed and save
    Gram matrices -- one entry per DISTINCT NETWORK REALIZATION, keyed by
    activity_key_for_trial(dataset, trial, sim).

    n_sims (added for NEF's NLL branch -- see docs/HISTORY.md and
    models.NEF.simulate_ensemble): a genuine ensemble of NEF responses for
    one trial needs n_sims DIFFERENT seeds simulating that trial's stimulus,
    not n_sims copies of one seed -- see activity_key_for_trial's own
    docstring for why reusing a seed across trials would silently correlate
    supposedly-independent ensemble members. n_sims=1 (the default)
    reproduces the exact original single-seed-per-trial behaviour and file
    contents -- this is a pure extension, not a format change.

    RESUMABLE: if out_path already exists, its entries are loaded first and
    only MISSING keys are simulated. This matters because these files are
    genuinely expensive: at n_neurons_counting=2000, a carrabin file (200
    trials) at n_sims=50 is on the order of tens of GB and real compute --
    growing an existing n_sims=1 file up to n_sims=50 must not re-pay the
    cost of the keys it already has.

    Saves per key: MtM (Gram matrix), Mty_count, ideal_count_filt. These are
    sufficient to recompute W_count and W_weight for any (alpha_0, lambda_)
    without re-running the Nengo simulation.

    File: data/counting_activities_n{n}_nc{nc}_{dataset}.pkl
    Keys: activity_key_for_trial(dataset, trial, sim) for trial in
          1..n_trials, sim in 1..n_sims -- NOT necessarily contiguous with
          the real dataset's own 0/1-indexed trial numbers; see
          activity_key_for_trial's own docstring.
    """
    import pickle
    import time
    from utils.paths import data_path

    dataset = str(params.get("dataset", "")).lower()
    if n_trials is None:
        n_trials = _DATASET_N_TRIALS.get(dataset)
        if n_trials is None:
            raise ValueError(f"n_trials not provided and dataset {dataset!r} not in _DATASET_N_TRIALS")

    n  = int(params["n_neurons"])
    nc = int(params["n_neurons_counting"])
    dataset  = str(params.get("dataset", "")).lower() or "unknown"
    if out_path is None:
        out_path = data_path(f"counting_activities_n{n}_nc{nc}_{dataset}.pkl")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    radius_c = int(params["radius_c"])
    p_base = {**params, "n_obs": radius_c}

    # Resume: load whatever's already there, skip keys we already have.
    activities: dict[int, dict] = {}
    if out_path.exists():
        try:
            with open(out_path, "rb") as f:
                activities = pickle.load(f)
            if verbose:
                print(f"Found existing {out_path.name} with {len(activities)} "
                      f"entries -- resuming, will only simulate missing keys.")
        except Exception as e:
            if verbose:
                print(f"Existing {out_path.name} unreadable ({e}); regenerating from scratch.")
            activities = {}

    all_keys = [
        (sim, trial, (sim - 1) * n_trials + trial)
        for sim in range(1, n_sims + 1)
        for trial in range(1, n_trials + 1)
    ]
    todo = [(sim, trial, key) for sim, trial, key in all_keys if key not in activities]

    if verbose:
        print(f"Precomputing counting activity sets  "
              f"(n_neurons={n}, n_neurons_counting={nc}, radius_c={radius_c}, "
              f"n_trials={n_trials}, n_sims={n_sims}) -- "
              f"{len(todo)} of {len(all_keys)} keys needed")

    if not todo:
        if verbose:
            print(f"Nothing to do -- all {len(all_keys)} keys already present in {out_path.name}")
        return out_path

    t_total = time.time()
    for i, (sim, trial, key) in enumerate(todo, start=1):
        p = {**p_base, "seed": key}
        t0  = time.time()
        net = build_network(p, train=True)
        raw = simulate_network(net, p, train=True)

        dt        = float(p["dt"])
        tau_probe = float(p["tau_probe"])
        syn       = nengo.Lowpass(tau_probe)

        mem_filt  = syn.filt(raw["memory"], dt=dt)           # (T, n_neurons)
        ic_filt   = syn.filt(raw["ideal"][:, 0:1], dt=dt).ravel()  # (T,) count

        MtM       = mem_filt.T @ mem_filt                    # (n, n)
        Mty_count = mem_filt.T @ ic_filt                     # (n,)

        # Compact basis representation: instead of storing the full
        # (n, T) activity matrix, store (n, radius_c+1) basis vectors
        # Mty_basis[k] = mem_filt.T @ (ic_filt == k).
        # At inference: Mty_weight = sum_k w(k) * Mty_basis[k]
        # where w(k) = alpha_0 / max(k,1)^lambda_ — no information lost.
        radius_c_int = int(p.get("radius_c", p["n_obs"]))
        Mty_basis = np.zeros((mem_filt.shape[1], radius_c_int + 1))
        for k in range(radius_c_int + 1):
            mask = (np.round(ic_filt).astype(int) == k)
            if mask.any():
                Mty_basis[:, k] = mem_filt.T @ mask.astype(float)

        # Store activity at readout timepoints only: (n_neurons, n_obs)
        # Used for plotting decoded count/weight per observation.
        idx_readout = _eval_idx(p, len(ic_filt))
        mem_readout = mem_filt[idx_readout, :].T    # (n, n_obs)
        ic_readout  = ic_filt[idx_readout]          # (n_obs,)

        activities[key] = {
            "MtM":              MtM,
            "Mty_count":        Mty_count,
            "ideal_count_filt": ic_filt,
            "Mty_basis":        Mty_basis,       # (n, radius_c+1) — for fast_decode
            "mem_readout":      mem_readout,     # (n, n_obs) — for plotting
            "ic_readout":       ic_readout,      # (n_obs,) — ideal count at readout
        }
        if verbose:
            t_trial = time.time() - t0
            elapsed_total = time.time() - t_total
            avg = elapsed_total / i
            eta = avg * (len(todo) - i)
            pct = 100 * i / len(todo)
            bar_len = 30
            filled = int(bar_len * i / len(todo))
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{bar}] {pct:5.1f}%  key {key:5d} (sim={sim} trial={trial})  "
                  f"{i:5d}/{len(todo)}  {t_trial:.1f}s  avg={avg:.1f}s  ETA={eta/60:.1f}min",
                  end="\r", flush=True)

    if verbose:
        print()  # newline after progress bar

    # Write to a temp file then atomically rename to avoid corruption
    # from concurrent jobs writing to the same path.
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(activities, f, protocol=4)
    tmp_path.rename(out_path)

    elapsed = time.time() - t_total
    size_mb = out_path.stat().st_size / 1024**2
    print(f"Saved {len(activities)} activity sets ({len(todo)} newly simulated) -> {out_path}  "
          f"({size_mb:.1f} MB, {elapsed/60:.1f}min)")
    return out_path


def fast_decode(
    activity: dict,
    alpha_0: float,
    lambda_: float,
    reg: float = 1e-3,
) -> dict:
    """Compute W_count and W_weight from precomputed Gram matrix.

    Uses the saved MtM and mem_filt_T to solve for decoders without
    re-running the Nengo simulation.

    Parameters
    ----------
    activity : dict with keys MtM, Mty_count, ideal_count_filt, mem_filt_T
    alpha_0, lambda_ : cognitive parameters for this Optuna trial
    reg : regularisation fraction (matches nengo LstsqL2 default)

    Returns
    -------
    dict with W_count and W_weight (same format as decode_outputs)
    """
    MtM       = activity["MtM"]
    Mty_count = activity["Mty_count"]
    ic_filt   = activity["ideal_count_filt"]

    n   = MtM.shape[0]
    lam = reg * np.trace(MtM) / n
    A   = MtM + lam * np.eye(n)

    # W_count — independent of alpha_0 / lambda_
    W_count = np.linalg.solve(A, Mty_count)[np.newaxis, :]

    # Reconstruct Mty_weight from compact basis (no mem_filt_T needed)
    if "Mty_basis" in activity:
        Mty_basis  = activity["Mty_basis"]          # (n, radius_c+1)
        radius_c   = Mty_basis.shape[1] - 1
        k_vals     = np.arange(radius_c + 1, dtype=float)
        w_vals     = alpha_0 / np.maximum(k_vals, 1.0) ** lambda_   # (radius_c+1,)
        Mty_weight = Mty_basis @ w_vals                              # (n,)
    else:
        # Legacy files with mem_filt_T
        mem_filt_T = activity["mem_filt_T"]
        iw_filt    = alpha_0 / np.maximum(ic_filt, 1.0) ** lambda_
        Mty_weight = mem_filt_T @ iw_filt

    W_weight = np.linalg.solve(A, Mty_weight)[np.newaxis, :]
    return {"W_count": W_count, "W_weight": W_weight}


def load_activities(
    n_neurons: int | None = None,
    n_neurons_counting: int | None = None,
    dataset: str = "carrabin",
    path: str | Path | None = None,
) -> dict[int, dict]:
    """Load precomputed activity data from disk."""
    import pickle
    from utils.paths import data_path

    if path is None:
        n  = n_neurons          or _NEF_FIXED["n_neurons"]
        nc = n_neurons_counting or _NEF_FIXED["n_neurons_counting"]
        path = data_path(f"counting_activities_n{n}_nc{nc}_{dataset}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_from_activities(
    activities: dict,
    alpha_0: float,
    lambda_: float,
    params: dict,
    out_path: Path | None = None,
) -> None:
    """Load precomputed activities, decode per trial, plot ideal vs decoded.

    Uses fast_decode to compute W_count and W_weight for each trial using
    the given (alpha_0, lambda_), then decodes count and weight signals and
    plots mean ± CI across trials using seaborn lineplot.
    """
    import seaborn as sns
    from utils.plot_style import apply_style, get_palette
    from utils.paths import FIGURES_DIR

    apply_style()
    pal = get_palette(6)
    nef_color  = pal[3]
    ideal_color = "0.4"

    n_obs     = int(params["radius_c"])
    dt        = float(params["dt"])
    t_obs_    = float(params["t_obs"])
    t_iti_    = float(params["t_iti"])
    t_step    = t_obs_ + t_iti_
    t_total   = n_obs * t_step
    T         = int(round(t_total / dt))
    t_arr     = np.arange(T) * dt

    # ── Decode count and weight for each trial ────────────────────────────────
    rows_count, rows_weight = [], []
    ideal_count_all, ideal_weight_all = [], []

    tau_probe = float(params["tau_probe"])
    syn       = nengo.Lowpass(tau_probe)

    for trial, act in sorted(activities.items()):
        dec = fast_decode(act, alpha_0=alpha_0, lambda_=lambda_)
        W_count  = dec["W_count"]   # (1, n)
        W_weight = dec["W_weight"]  # (1, n)

        # Ideal signals
        ic = act["ideal_count_filt"]
        iw = alpha_0 / np.maximum(ic, 1.0) ** lambda_

        # Sample at observation midpoints
        idx = _eval_idx(params, len(ic))

        if "mem_filt_T" in act:
            # Legacy: full timeseries available
            mem_filt_T = act["mem_filt_T"]
            count_dec  = (W_count  @ mem_filt_T).ravel()
            weight_dec = (W_weight @ mem_filt_T).ravel()
            for i, obs in enumerate(range(1, n_obs + 1)):
                if i >= len(idx): continue
                k = idx[i]
                rows_count.append({"trial": trial, "observation": obs,
                                    "decoded": float(count_dec[k]),
                                    "ideal":   float(ic[k])})
                rows_weight.append({"trial": trial, "observation": obs,
                                     "decoded": float(weight_dec[k]),
                                     "ideal":   float(iw[k])})
        else:
            # New format: use mem_readout (n, n_obs) — exact activity at readout
            mem_ro = act["mem_readout"]   # (n, n_obs)
            ic_ro  = act["ic_readout"]    # (n_obs,) ideal count at readout
            iw_ro  = alpha_0 / np.maximum(ic_ro, 1.0) ** lambda_
            count_dec_ro  = (W_count  @ mem_ro).ravel()  # (n_obs,)
            weight_dec_ro = (W_weight @ mem_ro).ravel()  # (n_obs,)
            for i, obs in enumerate(range(1, n_obs + 1)):
                if i >= len(count_dec_ro): continue
                rows_count.append({"trial": trial, "observation": obs,
                                    "decoded": float(count_dec_ro[i]),
                                    "ideal":   float(ic_ro[i])})
                rows_weight.append({"trial": trial, "observation": obs,
                                     "decoded": float(weight_dec_ro[i]),
                                     "ideal":   float(iw_ro[i])})

    df_c = pd.DataFrame(rows_count)
    df_w = pd.DataFrame(rows_weight)
    n_trials = len(activities)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    fig.suptitle(
        f"Counting circuit: ideal vs decoded  "
        f"(α₀={alpha_0:.2f}, λ={lambda_:.2f}, "
        f"n={int(params['n_neurons'])}, nc={int(params['n_neurons_counting'])}, "
        f"n_trials={n_trials})",
        fontsize=9,
    )

    for ax, df, ylabel, title in [
        (axes[0], df_c, "count", "Observation count"),
        (axes[1], df_w, "weight  α(t) = α₀/t^λ", "Learning-rate weight"),
    ]:
        # ideal line (same across trials)
        ideal_vals = df.groupby("observation")["ideal"].first()
        ax.plot(ideal_vals.index, ideal_vals.values,
                "--", color=ideal_color, lw=1.5, label="ideal", zorder=3)

        # decoded: lineplot with 95% CI across trials
        sns.lineplot(
            data=df, x="observation", y="decoded",
            color=nef_color, ax=ax,
            errorbar=("ci", 95), err_style="band",
            label="decoded (mean ± 95% CI)",
        )
        ax.set_xlabel("Observation")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax, top=True, right=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        n  = int(params["n_neurons"])
        nc = int(params["n_neurons_counting"])
        dataset_str = str(params.get("dataset","")).lower() or "unknown"
        out_path = FIGURES_DIR / f"counting_accuracy_n{n}_nc{nc}_{dataset_str}.png"
    plt.savefig(out_path, dpi=300)
    plt.savefig(str(out_path).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"Saved {out_path}")



# Map dataset name to radius_c
_DATASET_RADIUS_C = {"carrabin": 5, "yoo": 30, "soltani_numbers": 15, "soltani_colors": 15}


if __name__ == "__main__":
    args = parse_args()
    params_base = {**_NEF_FIXED, **vars(args)}
    # Derive radius_c from dataset if specified
    if args.dataset is not None:
        params_base["radius_c"] = _DATASET_RADIUS_C[args.dataset]
    if args.precompute:
        n_t = _DATASET_N_TRIALS.get(str(getattr(args,"dataset","") or ""), 200)
        seeds = [(t, t) for t in range(1, n_t + 1)]
        save_decoders(seeds, params_base)
    elif args.precompute_activities:
        params_base["dataset"] = args.dataset or "unknown"
        precompute_activities(None, params_base, n_sims=args.n_sims)
    elif args.plot_activities:
        acts = load_activities(
            n_neurons=args.n_neurons,
            n_neurons_counting=args.n_neurons_counting,
            dataset=args.dataset or "carrabin",
        )
        plot_from_activities(
            acts,
            alpha_0=args.alpha_0,
            lambda_=args.lambda_,
            params=params_base,
        )
    else:
        run(vars(args))
