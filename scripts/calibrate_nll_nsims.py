#!/usr/bin/env python3
"""
Calibrate n_sims for NLL fitting using CHEAP math-model proxies -- no Nengo,
no activity files, runs in seconds. Meant to give a ballpark n_sims before
committing to NEF's own activity-file scaling (n_trials * n_sims distinct
seeds -- see docs/HISTORY.md / this session's discussion for why that's
expensive enough to be worth bounding first).

Two complementary views, both scored against REAL human data (no synthetic
ground truth -- this mirrors fitting/fit.py's own established validation
convention: checking whether independent Monte Carlo reps AGREE with each
other, not whether they recover a known-in-advance truth):

1. --mode nll_convergence: at a FIXED (alpha_0, lambda_, sigma) point, how
   does the Monte Carlo NLL estimate (mean +/- spread across independent
   reps) converge toward a large-n_sims reference as n_sims grows? Answers
   "how many sims until the NLL number itself stops moving."

2. --mode argmin_stability: sweep a sigma grid, find the argmin at each
   n_sims level, across several independent reps -- how much does the
   RECOVERED sigma vary across reps as n_sims grows? Answers the more
   practically relevant question: "how many sims until a real Optuna fit
   would reliably land on the same answer." This generalises the exact
   criterion fitting/fit.py's own docstring used to validate n_sims=100 for
   sigma_resp (5 reseeded reps, same argmin) into an actual sweep with a
   continuous stability metric instead of one yes/no check.

Runs on NoisyRL_lambda (state/compounding noise) by default -- the closer
structural analogy to NEF's own noise: NEF's value ensemble is a recurrent
integrator, so a given sim's idiosyncrasies persist and compound through a
trial, much like sigma_state, rather than behaving like i.i.d.
per-observation sigma_resp. Pass --model_type RL_lambda_resp_noise (or both,
space-separated) to compare the i.i.d. mechanism directly -- if the two need
similar n_sims, the choice of proxy mechanism doesn't matter much for
picking NEF's n_sims; if they diverge, that's worth knowing before trusting
either one as NEF's stand-in.

This script does NOT auto-pick a "recommended" n_sims -- it hands back the
numbers and a plot for a human judgement call, same as every other
diagnostic tool in this project.

Usage:
    python scripts/calibrate_nll_nsims.py --dataset soltani_numbers --pid 13 \\
        --alpha_0 0.371 --lambda_ 0.654 --sigmas 0.02 0.05 0.1 0.2 \\
        --n_sims_grid 5 10 20 40 80 160 --n_reps 10

    # Compare both noise mechanisms in one run:
    python scripts/calibrate_nll_nsims.py --dataset carrabin --pid 5 \\
        --alpha_0 0.502 --lambda_ 0.613 --model_type NoisyRL_lambda RL_lambda_resp_noise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitting.losses as losses
from models.math_models import (
    _NOISE_WRAPPABLE_BASE_MODELS,
    _STOCHASTIC_ENSEMBLE_MODELS,
    add_noise,
    base_model_of,
    is_resp_noise_model,
    simulate_ensemble,
)
from utils.paths import FIGURES_DIR, data_path, dataset_stem
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette

DATASETS = ("carrabin", "yoo", "soltani_numbers", "soltani_colors")


def _sigma_param_name(model_type: str) -> str:
    """Which param name this proxy's noise level lives under."""
    if model_type in _STOCHASTIC_ENSEMBLE_MODELS:
        return "sigma_state"
    if is_resp_noise_model(model_type) and base_model_of(model_type) in _NOISE_WRAPPABLE_BASE_MODELS:
        return "sigma_resp"
    raise ValueError(
        f"{model_type!r} is not a supported noise proxy for this script -- use "
        f"one of {sorted(_STOCHASTIC_ENSEMBLE_MODELS)} or '<model>_resp_noise' "
        f"for a model in {sorted(_NOISE_WRAPPABLE_BASE_MODELS)}."
    )


def _make_ensemble(params: dict, sigma_value: float, n_total: int) -> np.ndarray:
    """Dispatch to the real production ensemble generator for this model_type
    -- simulate_ensemble for genuinely stochastic models (compounding state
    noise), add_noise for the '<model>_resp_noise' wrapper (i.i.d. response
    noise). Same functions fitting.fit's --loss nll path calls; no
    reimplementation here."""
    model_type = params["model_type"]
    if model_type in _STOCHASTIC_ENSEMBLE_MODELS:
        p = {**params, "sigma_state": float(sigma_value)}
        return simulate_ensemble(p, n_total)
    return add_noise(params, n_total, sigma_resp=float(sigma_value))


def _load_human_y(dataset: str, pid: int, datafile: str | None) -> np.ndarray:
    stem = dataset_stem(dataset, datafile)
    human = pd.read_pickle(data_path(f"{stem}.pkl"))
    hp = human[human["pid"] == pid]
    if hp.empty:
        raise ValueError(f"No rows for pid={pid} in data/{stem}.pkl")
    hp = hp.sort_values(["trial", "observation"])
    return hp["response"].to_numpy(float)


# ── Mode 1: NLL value convergence ───────────────────────────────────────────

def nll_convergence(
    params: dict, sigma_value: float, y: np.ndarray,
    n_sims_grid: list[int], n_reps: int, n_sims_ref: int,
) -> tuple[pd.DataFrame, float]:
    """For a FIXED (params, sigma), estimate NLL at each n_sims in the grid,
    n_reps independent times (independent = non-overlapping blocks of a
    single larger draw -- cheap, no re-simulation needed), plus one
    high-n_sims reference estimate."""
    max_n = max(n_sims_grid)
    ens_all = _make_ensemble(params, sigma_value, n_reps * max_n)

    rows = []
    for n_sims in n_sims_grid:
        for r in range(n_reps):
            block = ens_all[r * max_n: r * max_n + n_sims]
            nll = losses.nll_from_ensemble(block, y)
            rows.append({"sigma": sigma_value, "n_sims": n_sims, "rep": r, "nll": nll})

    ref_ens = _make_ensemble(params, sigma_value, n_sims_ref)
    nll_ref = losses.nll_from_ensemble(ref_ens, y)
    return pd.DataFrame(rows), nll_ref


# ── Mode 2: argmin recovery stability ───────────────────────────────────────

def argmin_stability(
    params: dict, sigma_grid: list[float], y: np.ndarray,
    n_sims_grid: list[int], n_reps: int,
) -> pd.DataFrame:
    """Sweep sigma_grid, find the argmin NLL at each n_sims level, n_reps
    independent times. No ground truth involved -- this measures
    SELF-CONSISTENCY (would repeated fits agree), same criterion fitting/
    fit.py's own docstring used (5 reseeded reps, same argmin), generalised
    into a sweep with a continuous spread metric."""
    max_n = max(n_sims_grid)
    big_by_sigma = {s: _make_ensemble(params, s, n_reps * max_n) for s in sigma_grid}

    rows = []
    for n_sims in n_sims_grid:
        for r in range(n_reps):
            nlls = []
            for s in sigma_grid:
                block = big_by_sigma[s][r * max_n: r * max_n + n_sims]
                nlls.append(losses.nll_from_ensemble(block, y))
            best_idx = int(np.argmin(nlls))
            rows.append({
                "n_sims": n_sims, "rep": r,
                "argmin_sigma": sigma_grid[best_idx],
                "min_nll": nlls[best_idx],
            })
    return pd.DataFrame(rows)


# ── Reporting ────────────────────────────────────────────────────────────────

def print_convergence_report(model_type: str, sigma: float, df: pd.DataFrame, nll_ref: float) -> None:
    print(f"\n[{model_type}] sigma={sigma:.4f}  NLL convergence (reference NLL @ large n_sims = {nll_ref:.4f})")
    agg = df.groupby("n_sims")["nll"].agg(["mean", "std"])
    for n_sims, row in agg.iterrows():
        gap = row["mean"] - nll_ref
        print(f"  n_sims={n_sims:4d}   mean NLL={row['mean']:.4f}  std={row['std']:.4f}  "
              f"gap vs ref={gap:+.4f}")


def print_argmin_report(model_type: str, df: pd.DataFrame) -> None:
    print(f"\n[{model_type}] argmin-sigma recovery stability across reps")
    agg = df.groupby("n_sims")["argmin_sigma"].agg(["mean", "std", "min", "max"])
    for n_sims, row in agg.iterrows():
        print(f"  n_sims={n_sims:4d}   recovered sigma: mean={row['mean']:.4f}  "
              f"std={row['std']:.4f}  range=[{row['min']:.4f}, {row['max']:.4f}]")


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(
    conv_results: dict[str, list[tuple[float, pd.DataFrame, float]]],
    argmin_results: dict[str, pd.DataFrame],
    out_stem: str,
) -> None:
    apply_style()
    model_types = list(conv_results.keys())
    linestyles = ["-", "--", ":", "-."]
    ls_by_model = {m: linestyles[i % len(linestyles)] for i, m in enumerate(model_types)}

    all_sigmas = sorted({s for entries in conv_results.values() for s, _, _ in entries})
    pal = get_palette(max(len(all_sigmas), 3))
    color_by_sigma = {s: pal[i % len(pal)] for i, s in enumerate(all_sigmas)}

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(FIGURE_SIZE[0] * 1.9, FIGURE_SIZE[1]), constrained_layout=True
    )

    # Panel 1: NLL convergence
    for model_type, entries in conv_results.items():
        for sigma, df, nll_ref in entries:
            agg = df.groupby("n_sims")["nll"].agg(["mean", "std"]).reset_index()
            color = color_by_sigma[sigma]
            ax0.errorbar(
                agg["n_sims"], agg["mean"], yerr=agg["std"],
                color=color, linestyle=ls_by_model[model_type], marker="o", markersize=3,
                capsize=2, linewidth=1.2,
            )
            ax0.axhline(nll_ref, color=color, linestyle=ls_by_model[model_type],
                        linewidth=0.5, alpha=0.5)
    ax0.set_xscale("log")
    ax0.set_xlabel("n_sims")
    ax0.set_ylabel("NLL (mean \u00b1 std across reps)")
    ax0.set_title("NLL estimate convergence\n(colour = sigma, style = model, thin line = large-n_sims ref)")
    for s in all_sigmas:
        ax0.plot([], [], color=color_by_sigma[s], label=f"sigma={s}")
    for m in model_types:
        ax0.plot([], [], color="0.3", linestyle=ls_by_model[m], label=m)
    ax0.legend(frameon=False, fontsize=7, loc="best")

    # Panel 2: argmin recovery stability
    for model_type, df in argmin_results.items():
        agg = df.groupby("n_sims")["argmin_sigma"].agg(["mean", "std"]).reset_index()
        ax1.errorbar(
            agg["n_sims"], agg["std"], marker="o", markersize=3,
            linestyle=ls_by_model[model_type], linewidth=1.2, label=model_type,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("n_sims")
    ax1.set_ylabel("std of recovered argmin-sigma across reps")
    ax1.set_title("Argmin recovery stability\n(lower = more repeatable fit at that n_sims)")
    ax1.legend(frameon=False, fontsize=8)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{out_stem}.pdf")
    plt.close(fig)
    print(f"\nSaved figures/{out_stem}.pdf")


# ── Orchestration ────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--datafile", type=str, default=None)
    p.add_argument("--alpha_0", type=float, required=True)
    p.add_argument("--lambda_", type=float, required=True)
    p.add_argument("--model_type", type=str, nargs="+", default=["NoisyRL_lambda"],
                    help="One or more of NoisyRL_lambda / <base>_resp_noise "
                         "(e.g. RL_lambda_resp_noise). Space-separated to compare.")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2],
                    help="Sigma values to test. Used both as fixed evaluation "
                         "points (nll_convergence) and as the search grid "
                         "(argmin_stability). Pick values that bracket the "
                         "expected real noise magnitude -- e.g. soltani's "
                         "qid-grouped human response std is ~0.055.")
    p.add_argument("--n_sims_grid", type=int, nargs="+", default=[5, 10, 20, 40, 80, 160],
                    help="Candidate n_sims values to test.")
    p.add_argument("--n_reps", type=int, default=10,
                    help="Independent Monte Carlo reps per n_sims level.")
    p.add_argument("--n_sims_ref", type=int, default=2000,
                    help="Large n_sims used as the convergence reference.")
    p.add_argument("--mode", choices=["nll_convergence", "argmin_stability", "both"],
                    default="both")
    args = p.parse_args()

    for mt in args.model_type:
        _sigma_param_name(mt)  # validate early, fail loudly before any work

    y = _load_human_y(args.dataset, args.pid, args.datafile)
    print(f"Loaded {len(y)} real response rows for {args.dataset} pid={args.pid}")

    conv_results: dict[str, list[tuple[float, pd.DataFrame, float]]] = {}
    argmin_results: dict[str, pd.DataFrame] = {}

    for model_type in args.model_type:
        base_params = {
            "model_type": model_type,
            "dataset": args.dataset,
            "pid": int(args.pid),
            "alpha_0": float(args.alpha_0),
            "lambda_": float(args.lambda_),
            "datafile": args.datafile,
            "seed": 0,
        }

        if args.mode in ("nll_convergence", "both"):
            entries = []
            for sigma in args.sigmas:
                df, nll_ref = nll_convergence(
                    base_params, sigma, y, args.n_sims_grid, args.n_reps, args.n_sims_ref
                )
                print_convergence_report(model_type, sigma, df, nll_ref)
                entries.append((sigma, df, nll_ref))
            conv_results[model_type] = entries

        if args.mode in ("argmin_stability", "both"):
            df = argmin_stability(base_params, args.sigmas, y, args.n_sims_grid, args.n_reps)
            print_argmin_report(model_type, df)
            argmin_results[model_type] = df

    if conv_results or argmin_results:
        out_stem = f"nll_nsims_calibration_{args.dataset}_{args.pid}"
        if conv_results and argmin_results:
            plot_results(conv_results, argmin_results, out_stem)
            pd.to_pickle(
                {"convergence": conv_results, "argmin": argmin_results, "params": vars(args)},
                FIGURES_DIR / f"{out_stem}.pkl",
            )
            print(f"Saved figures/{out_stem}.pkl")
        elif conv_results:
            print("\n(--mode nll_convergence only: skipping combined plot, "
                  "which needs both panels' data. Data printed above.)")
        else:
            print("\n(--mode argmin_stability only: skipping combined plot, "
                  "which needs both panels' data. Data printed above.)")


if __name__ == "__main__":
    main()
