#!/usr/bin/env python3
"""
Diagnostic test for the 2D counting integrator in NEF2d.py.

Tests count integrator and alpha decoder against held-out sessions.

Saves diagnostic plots to figures/nef2d_counting_test.png.

Usage:
    python scripts/test_nef2d_counting.py
    python scripts/test_nef2d_counting.py --n_test 3 --alpha_0 0.3 --lambda_ 0.5
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nengo
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitting.model_params import MODEL_PARAMS
from models.NEF2d import (
    PARAM_DEFAULTS,
    _build_1d_counting_circuit,
    _ideal_targets_at_times,
    _make_session_input,
    _pretrain_counting_1d,
    _session_distrib_map,
    _session_duration,
    _session_input_timeseries,
    _validate_1d_decoders,
)
from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import apply_style, get_palette


def _readout_indices(n_rows: int, params: dict, n_steps: int) -> list[int]:
    dt = float(params["dt"])
    t_obs = float(params["t_obs"])
    t_iti = float(params["t_iti"])
    t_step = t_obs + t_iti
    return [
        min(int((t_iti + i * t_step + t_obs * 0.8) / dt), n_steps - 1)
        for i in range(n_rows)
    ]


def _build_counting_network_2d_diag(
    params: dict,
    input_fn: callable,
    decoders: dict,
) -> nengo.Network:
    """
    Diagnostic network: two independent 1D counting circuits (A and B)
    with default-decoded count probes.
    """
    seed = int(params["seed"])
    tau_probe = float(params["tau_probe"])
    n_neurons_counting = int(params["n_neurons_counting"])

    _validate_1d_decoders(decoders, n_neurons_counting)

    with nengo.Network(label="counting_2d_diag", seed=seed) as net:
        net.input_node = nengo.Node(input_fn, size_out=4, label="input_4d")
        _build_1d_counting_circuit(net, "A", 2, params, seed, decoders)
        _build_1d_counting_circuit(net, "B", 3, params, seed, decoders)

        net.probe_count_A_default = nengo.Probe(
            net.count_A,
            synapse=tau_probe,
            sample_every=float(params["dt"]),
        )
        net.probe_count_B_default = nengo.Probe(
            net.count_B,
            synapse=tau_probe,
            sample_every=float(params["dt"]),
        )

    return net


def evaluate_session(
    pid: int,
    session: int,
    human: pd.DataFrame,
    params: dict,
    decoders: dict,
    rng: np.random.Generator,
) -> dict:
    sess = human[(human["pid"] == pid) & (human["session"] == session)].copy()
    rows, da, db, _ = _session_input_timeseries(sess, params)

    input_fn = _make_session_input(rows, da, db, params)
    net = _build_counting_network_2d_diag(params, input_fn, decoders)
    t_total = _session_duration(len(rows), params)
    dt = float(params["dt"])

    with nengo.Simulator(
        net, dt=dt, seed=int(params["seed"]), progress_bar=False
    ) as sim:
        sim.run(t_total)

    alpha_a_dec = sim.data[net.probe_alpha_A].squeeze()
    alpha_b_dec = sim.data[net.probe_alpha_B].squeeze()
    count_a_neurons = sim.data[net.probe_count_A_neurons]
    count_b_neurons = sim.data[net.probe_count_B_neurons]
    count_def_a = sim.data[net.probe_count_A_default].squeeze()
    count_def_b = sim.data[net.probe_count_B_default].squeeze()
    count_def = np.column_stack([count_def_a, count_def_b])

    n_steps = len(alpha_a_dec)
    t_arr = np.arange(n_steps) * dt
    targets = _ideal_targets_at_times(t_arr, rows, da, db, params)

    ri = _readout_indices(len(rows), params, n_steps)

    ca_true = targets[ri, 0];  cb_true = targets[ri, 1]
    aa_true = targets[ri, 2];  ab_true = targets[ri, 3]

    ca_dec = (decoders["W_count"] @ count_a_neurons.T).squeeze()[ri]
    cb_dec = (decoders["W_count"] @ count_b_neurons.T).squeeze()[ri]
    aa_dec = alpha_a_dec[ri]
    ab_dec = alpha_b_dec[ri]

    # Count default-decoded diagnostic: does count grow at all?
    count_final_A = float(count_def[-100:, 0].mean())
    count_final_B = float(count_def[-100:, 1].mean())
    count_true_final_A = float(targets[-1, 0])
    count_true_final_B = float(targets[-1, 1])

    return {
        "pid": pid, "session": session,
        "ca_true": ca_true, "cb_true": cb_true,
        "aa_true": aa_true, "ab_true": ab_true,
        "ca_dec": ca_dec, "cb_dec": cb_dec,
        "aa_dec": aa_dec, "ab_dec": ab_dec,
        "rmse_ca": float(np.sqrt(np.mean((ca_dec - ca_true) ** 2))),
        "rmse_cb": float(np.sqrt(np.mean((cb_dec - cb_true) ** 2))),
        "rmse_aa": float(np.sqrt(np.mean((aa_dec - aa_true) ** 2))),
        "rmse_ab": float(np.sqrt(np.mean((ab_dec - ab_true) ** 2))),
        # Timeseries for plotting
        "t_arr": t_arr,
        "targets": targets,
        "count_def": count_def,
        "alpha_a_dec_full": alpha_a_dec,
        "alpha_b_dec_full": alpha_b_dec,
        "ri": ri,
        "count_final_A": count_final_A,
        "count_final_B": count_final_B,
        "count_true_final_A": count_true_final_A,
        "count_true_final_B": count_true_final_B,
    }


def plot_results(results: list[dict], params: dict) -> None:
    apply_style()
    pal = get_palette()
    col_A, col_B = pal[0], pal[1]

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n), constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, r in enumerate(results):
        t = r["t_arr"]

        # Col 0: Count (default decoded vs true)
        ax = axes[i, 0]
        ax.plot(t, r["targets"][:, 0], "--", color=col_A, linewidth=0.8, alpha=0.6, label="count_A true")
        ax.plot(t, r["count_def"][:, 0], color=col_A, linewidth=1.2,
                label=f"count_A def (final: dec={r['count_final_A']:.1f} true={r['count_true_final_A']:.0f})")
        ax.plot(t, r["targets"][:, 1], "--", color=col_B, linewidth=0.8, alpha=0.6, label="count_B true")
        ax.plot(t, r["count_def"][:, 1], color=col_B, linewidth=1.2,
                label=f"count_B def (final: dec={r['count_final_B']:.1f} true={r['count_true_final_B']:.0f})")
        ax.set_title(
            f"pid={r['pid']} s={r['session']} — count "
            f"RMSE A={r['rmse_ca']:.2f} B={r['rmse_cb']:.2f}"
        )
        ax.set_xlabel("time (s)")
        ax.legend(frameon=False, fontsize=6, ncols=2)

        # Col 1: Alpha (decoded vs true)
        ax = axes[i, 1]
        ax.plot(t, r["targets"][:, 2], "--", color=col_A, linewidth=0.8, alpha=0.6)
        ax.plot(t, r["alpha_a_dec_full"], color=col_A, linewidth=1.2,
                label=f"alpha_A (RMSE={r['rmse_aa']:.4f})")
        ax.plot(t, r["targets"][:, 3], "--", color=col_B, linewidth=0.8, alpha=0.6)
        ax.plot(t, r["alpha_b_dec_full"], color=col_B, linewidth=1.2,
                label=f"alpha_B (RMSE={r['rmse_ab']:.4f})")
        ax.set_title("Alpha(n) decoded vs true")
        ax.set_xlabel("time (s)")
        ax.legend(frameon=False, fontsize=6)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "nef2d_counting_test.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test",  type=int, default=3)
    parser.add_argument("--alpha_0", type=float, default=0.3)
    parser.add_argument("--lambda_", type=float, default=0.5)
    parser.add_argument("--seed",    type=int,   default=0)
    parser.add_argument("--onset_detector_amp", type=float, default=0.3)
    parser.add_argument("--n_neurons", type=int, default=200)
    parser.add_argument("--n_neurons_counting", type=int, default=1000)
    parser.add_argument("--radius_c", type=float, default=60.0)
    parser.add_argument("--out_folder", type=str, default="nef2d_sweep")
    parser.add_argument("--count_leak", type=float, default=0.0)
    args = parser.parse_args()

    fixed = dict(MODEL_PARAMS["diederen"]["NEF2d"].get("fixed", {}))
    params = {
        **PARAM_DEFAULTS, **fixed,
        "alpha_0": args.alpha_0,
        "lambda_": args.lambda_,
        "seed": args.seed,
        "base_seed": args.seed,
        "onset_detector_amp": args.onset_detector_amp,
        "n_neurons": args.n_neurons,
        "n_neurons_counting": args.n_neurons_counting,
        "radius_c": args.radius_c,
        "count_leak": args.count_leak,
    }

    human = pd.read_pickle(data_path("diederen.pkl"))
    human = human[~human["missed"]].copy()

    sessions = []
    for (pid, session), _ in human.groupby(["pid", "session"], sort=False):
        sess = human[(human["pid"] == pid) & (human["session"] == session)]
        try:
            _session_distrib_map(sess)
            sessions.append((int(pid), int(session)))
        except ValueError:
            pass

    rng = np.random.default_rng(args.seed)

    print("Pretraining...")
    t0 = time.time()
    decoders = _pretrain_counting_1d(params)
    train_time = time.time() - t0
    print(f"Done in {train_time:.1f}s")
    print("Decoder shapes:", {k: v.shape for k, v in decoders.items()})
    print()

    n_test = min(args.n_test, len(sessions))
    test_sessions = [
        sessions[i] for i in rng.choice(len(sessions), n_test, replace=False)
    ]

    results = []
    print(f"Evaluating on {n_test} held-out sessions...")
    sim_t0 = time.time()
    for pid, session in test_sessions:
        r = evaluate_session(pid, session, human, params, decoders, rng)
        results.append(r)
        print(
            f"  pid={pid} sess={session}: "
            f"RMSE count_A={r['rmse_ca']:.3f} count_B={r['rmse_cb']:.3f} "
            f"alpha_A={r['rmse_aa']:.4f} alpha_B={r['rmse_ab']:.4f} | "
            f"count final A: dec={r['count_final_A']:.1f} true={r['count_true_final_A']:.0f} "
            f"B: dec={r['count_final_B']:.1f} true={r['count_true_final_B']:.0f}"
        )
    sim_time = time.time() - sim_t0

    print()
    print(f"Mean RMSE count_A: {np.mean([r['rmse_ca'] for r in results]):.3f}")
    print(f"Mean RMSE count_B: {np.mean([r['rmse_cb'] for r in results]):.3f}")
    print(f"Mean RMSE alpha_A: {np.mean([r['rmse_aa'] for r in results]):.4f}")
    print(f"Mean RMSE alpha_B: {np.mean([r['rmse_ab'] for r in results]):.4f}")

    plot_results(results, params)

    out_dir = RUNS_DIR / args.out_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"counting_test"
        f"_rc{int(args.radius_c)}"
        f"_nc{args.n_neurons_counting}"
        f".pkl"
    )
    summary = {
        "radius_c": args.radius_c,
        "n_neurons_counting": args.n_neurons_counting,
        "n_test": args.n_test,
        "alpha_0": args.alpha_0,
        "lambda_": args.lambda_,
        "train_time_s": train_time,
        "sim_time_s": sim_time,
        "rmse_count_A": [r["rmse_ca"] for r in results],
        "rmse_count_B": [r["rmse_cb"] for r in results],
        "rmse_alpha_A": [r["rmse_aa"] for r in results],
        "rmse_alpha_B": [r["rmse_ab"] for r in results],
        "mean_rmse_count": float(
            np.mean([r["rmse_ca"] + r["rmse_cb"] for r in results]) / 2
        ),
        "mean_rmse_alpha": float(
            np.mean([r["rmse_aa"] + r["rmse_ab"] for r in results]) / 2
        ),
    }
    out_path = out_dir / fname
    with open(out_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"Saved summary -> {out_path}")


if __name__ == "__main__":
    main()