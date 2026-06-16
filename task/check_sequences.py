"""
task/check_sequences.py
=======================
Generate n_tries sequence sets and pick the one with the lowest combined
sampling noise score. Two criteria:

  1. RMSE(optimal_response, ground_truth) across obs
     Measures how much sequence noise causes the optimal Bayesian agent
     to deviate from true_mean / true_p. Targets panel A smoothness.

  2. RMSE(|Δoptimal|, A/t) across obs
     Fits A/t to the observed |Δoptimal| curve, then measures deviation.
     Captures how much the decay curve departs from ideal power-law.
     Targets panel C smoothness.

Optimal Bayesian agent (parameter-free):
  continuous : response(t) = running_mean(x_1..x_t) * (t+1)/(t+3)
               ground truth: true_mean / 100  (normalised to [-1,1])
  binary     : response(t) = (n_blue + 1) / (t + 2) * 2 - 1
               ground truth: true_p * 2 - 1

Usage:
    python task/check_sequences.py --task both --n_tries 500
    python task/check_sequences.py --task binary --n_tries 200 \\
        --n_unique_sequences 8 --n_repeats 5
"""

from __future__ import annotations
import argparse, subprocess, sys, tempfile, shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import PROJECT_ROOT as ROOT_DIR


# ── Bayesian agent responses ──────────────────────────────────────────────────

def bayesian_continuous(values_raw):
    """Running mean with Laplace smoothing toward 0. Input: raw [-100,100]."""
    resps = []
    running = 0.0
    for t, v in enumerate(values_raw, 1):
        v_norm = v / 100.0
        running += (v_norm - running) / t
        resp = running * (t + 1) / (t + 3)
        resps.append(resp)
    return resps   # on [-1,1]


def bayesian_binary(values):
    """Laplace-smoothed proportion mapped to [-1,1]. Input: {-1,+1}."""
    resps = []
    n_pos = 0
    for t, v in enumerate(values, 1):
        n_pos += (1 if v == 1 else 0)
        p_hat = (n_pos + 1) / (t + 2)   # Beta(1,1) prior
        resps.append(p_hat * 2 - 1)
    return resps   # on [-1,1]


# ── Scoring ───────────────────────────────────────────────────────────────────

def power_law(t, A):
    return A / np.asarray(t, float)


def score_sequences(seq_df, task):
    """
    Score a sequence set for panel A and C quality.

    Score 1 — RMSE decay quality (panel A):
      Compute RMSE(bayesian_response, ground_truth) per obs.
      Fit a power-law A/t^b to this decay curve.
      Penalise: RMSE of observed curve vs fit  (bumps/non-monotonicity)
               + any early-obs rise (obs 1-5 increase in RMSE)
      We do NOT minimise absolute RMSE level — that would select easy
      sequences. Instead we reward smooth *decay shape*.

    Score 2 — delta decay quality (panel C):
      Compute mean |Δbayesian| per obs, fit A/t, penalise deviations.

    Combined: score1 + score2 (lower = better shaped curves)
    """
    rmse_by_obs   = {}    # obs -> list of per-trial RMSE
    delta_curves  = []

    for trial_id in seq_df['trial'].unique():
        tdf = seq_df[seq_df['trial'] == trial_id].sort_values('observation')
        values = tdf['value'].tolist()
        obs_list = tdf['observation'].tolist()

        if task == 'continuous':
            gt    = tdf['true_mean'].iloc[0] / 100.0
            resps = bayesian_continuous(values)
        else:
            gt    = tdf['true_p'].iloc[0] * 2 - 1
            resps = bayesian_binary(values)

        for obs, r in zip(obs_list, resps):
            rmse_by_obs.setdefault(obs, []).append(abs(r - gt))

        deltas = [abs(resps[i] - resps[i-1]) for i in range(1, len(resps))]
        delta_curves.append(deltas)

    obs_sorted  = sorted(rmse_by_obs.keys())
    rmse_curve  = np.array([np.mean(rmse_by_obs[o]) for o in obs_sorted])
    t_vals      = np.arange(1, len(rmse_curve) + 1, dtype=float)

    # Fit power law to RMSE curve and penalise deviation
    try:
        popt, _ = curve_fit(lambda t, A, b: A / t**b,
                            t_vals, rmse_curve,
                            p0=[rmse_curve[0], 0.5],
                            bounds=([0, 0.01], [np.inf, 2.0]))
        fitted_rmse = popt[0] / t_vals**popt[1]
        score_rmse_shape = float(np.sqrt(np.mean((rmse_curve - fitted_rmse)**2)))
    except Exception:
        score_rmse_shape = float(np.std(np.diff(rmse_curve)))

    # Penalise early-obs rises (obs 1-5): sum of upward steps
    early = rmse_curve[:min(5, len(rmse_curve))]
    early_rise = float(np.sum(np.maximum(np.diff(early), 0)))

    # Score 2: delta curve vs A/t
    mean_delta = np.mean(delta_curves, axis=0)
    t_d = np.arange(2, len(mean_delta) + 2, dtype=float)
    try:
        popt2, _ = curve_fit(power_law, t_d, mean_delta,
                             p0=[mean_delta[0]], bounds=([0], [np.inf]))
        fitted_d = power_law(t_d, popt2[0])
        score_delta = float(np.sqrt(np.mean((mean_delta - fitted_d)**2)))
    except Exception:
        score_delta = float(np.std(mean_delta))

    combined = score_rmse_shape + early_rise + score_delta
    return score_rmse_shape, score_delta, combined


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_diagnostic(seq_df, task, attempt, out_dir, score):
    """Plot 4 diagnostic curves for the current best sequences."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    fig.suptitle(f"{task} | attempt={attempt} | score={score:.5f}", fontsize=9)

    rmse_by_obs, delta_curves = {}, []
    for trial_id in seq_df['trial'].unique():
        tdf  = seq_df[seq_df['trial']==trial_id].sort_values('observation')
        vals = tdf['value'].tolist()
        if task == 'continuous':
            gt    = tdf['true_mean'].iloc[0]/100.0
            resps = bayesian_continuous(vals)
        else:
            gt    = tdf['true_p'].iloc[0]*2-1
            resps = bayesian_binary(vals)
        for obs, r in zip(tdf['observation'].tolist(), resps):
            rmse_by_obs.setdefault(obs, []).append(abs(r-gt))
        delta_curves.append([abs(resps[i]-resps[i-1]) for i in range(1,len(resps))])

    obs_sorted = sorted(rmse_by_obs.keys())
    rmse_curve = np.array([np.mean(rmse_by_obs[o]) for o in obs_sorted])
    mean_delta = np.mean(delta_curves, axis=0)
    t_d        = np.arange(2, len(mean_delta)+2, dtype=float)

    # Panel 1: RMSE vs obs
    axes[0].plot(obs_sorted, rmse_curve, 'b-o', ms=3, lw=1.5)
    axes[0].set_title('RMSE vs true target'); axes[0].set_xlabel('obs')

    # Panel 2: RMSE with power-law fit
    axes[1].plot(obs_sorted, rmse_curve, 'b-o', ms=3, lw=1.5, label='data')
    try:
        popt,_ = curve_fit(lambda t,A,b: A/t**b, np.array(obs_sorted,float),
                           rmse_curve, p0=[rmse_curve[0],0.5], bounds=([0,0.01],[np.inf,2]))
        fit = popt[0]/np.array(obs_sorted,float)**popt[1]
        axes[1].plot(obs_sorted, fit, 'r--', lw=1.5, label=f'A/t^{popt[1]:.2f}')
    except: pass
    axes[1].set_title('RMSE + power-law fit'); axes[1].set_xlabel('obs')
    axes[1].legend(fontsize=7)

    # Panel 3: |Δresponse| vs obs
    axes[2].plot(t_d, mean_delta, 'g-o', ms=3, lw=1.5, label='data')
    try:
        popt2,_ = curve_fit(power_law, t_d, mean_delta, p0=[mean_delta[0]], bounds=([0],[np.inf]))
        axes[2].plot(t_d, power_law(t_d, popt2[0]), 'r--', lw=1.5, label='A/t fit')
    except: pass
    axes[2].set_title('Mean |Δresponse|'); axes[2].set_xlabel('obs')
    axes[2].legend(fontsize=7)

    # Panel 4: true_p / true_mean distribution
    if task == 'binary':
        tp = seq_df[seq_df.observation==1]['true_p'].values
        axes[3].hist(tp, bins=10, color='purple', alpha=0.7)
        axes[3].set_title('true_p distribution'); axes[3].set_xlabel('true_p')
    else:
        tm = seq_df[seq_df.observation==1]['true_mean'].values
        axes[3].hist(tm, bins=10, color='teal', alpha=0.7)
        axes[3].set_title('true_mean distribution'); axes[3].set_xlabel('true_mean')

    diag_dir = Path(out_dir) / 'diagnostics'
    diag_dir.mkdir(exist_ok=True)
    out_pdf = diag_dir / f'{task}_attempt{attempt:04d}.pdf'
    plt.savefig(out_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', choices=['continuous','binary','both'],
                        default='both')
    parser.add_argument('--n_tries',            type=int, default=200)
    parser.add_argument('--n_unique_sequences', type=int, default=8)
    parser.add_argument('--n_repeats',          type=int, default=5)
    parser.add_argument('--n_random',           type=int, default=0)
    parser.add_argument('--seq_length',         type=int, default=20)
    parser.add_argument('--prefix_length',      type=int, default=3)
    parser.add_argument('--mean_range', type=float, nargs=2, default=[-60., 60.])
    parser.add_argument('--std_low',    type=float, default=10.0)
    parser.add_argument('--std_high',   type=float, default=40.0)
    parser.add_argument('--p_range',    type=float, nargs=2, default=[0.2,  0.8])
    parser.add_argument('--max_outliers', type=int, default=2)
    parser.add_argument('--output_dir', default='task/sequences',
                        help='Where to save the winning sequences')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    tasks = ['continuous','binary'] if args.task == 'both' else [args.task]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    python  = sys.executable

    print(f"Checking {args.n_tries} sequence sets per task")
    print(f"Config: u{args.n_unique_sequences}_r{args.n_repeats}_"
          f"rnd{args.n_random}_o{args.seq_length}_p{args.prefix_length}"
          f" std=[{args.std_low},{args.std_high}]")

    for task in tasks:
        print(f"\n{'='*55}\nTask: {task.upper()}")
        best_score   = np.inf
        best_scores  = (np.inf, np.inf)
        best_seed    = None
        best_seq_df  = None
        all_scores   = []

        tmpdir = Path(tempfile.mkdtemp())
        try:
            for attempt in range(args.n_tries):
                seed = attempt  # deterministic: seed = attempt index

                # Generate sequences into temp dir
                cmd = [
                    python, 'task/generate_sequences.py',
                    '--task', task,
                    '--n_unique_sequences', str(args.n_unique_sequences),
                    '--n_repeats',          str(args.n_repeats),
                    '--n_random',           str(args.n_random),
                    '--seq_length',         str(args.seq_length),
                    '--prefix_length',      str(args.prefix_length),
                    '--mean_range'] + [str(v) for v in args.mean_range] + [
                    '--std_low',  str(args.std_low),
                    '--std_high', str(args.std_high),]  + [
                    '--p_range']    + [str(v) for v in args.p_range]    + [
                    '--max_outliers', str(args.max_outliers),
                    '--seed',         str(seed),
                    '--output_dir',   str(tmpdir),
                    '--overwrite',
                ]
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        cwd=str(ROOT_DIR))
                if result.returncode != 0:
                    continue

                pkl = tmpdir / f'{task}_sequences.pkl'
                if not pkl.exists():
                    continue

                seq_df = pd.read_pickle(pkl)
                r_resp, r_decay, combined = score_sequences(seq_df, task)
                all_scores.append(combined)

                if combined < best_score:
                    best_score  = combined
                    best_scores = (r_resp, r_decay)
                    best_seed   = seed
                    best_seq_df = seq_df.copy()
                    # Copy winning files
                    shutil.copy(pkl, out_dir / f'{task}_sequences.pkl')
                    json_src = tmpdir / f'{task}_sequences.json'
                    if json_src.exists():
                        shutil.copy(json_src,
                                    out_dir / f'{task}_sequences.json')
                    # Copy to Vite src
                    vite = ROOT_DIR/'task'/'src'/task/'sequences.json'
                    if json_src.exists() and vite.parent.exists():
                        shutil.copy(json_src, vite)

                if (attempt+1) % 50 == 0:
                    plot_diagnostic(best_seq_df, task, attempt+1, out_dir, best_score)

                if args.verbose or (attempt+1) % 50 == 0:
                    pct = np.mean(np.array(all_scores) <= best_score) * 100
                    print(f"  [{attempt+1:4d}/{args.n_tries}]  "
                          f"best seed={best_seed}  "
                          f"combined={best_score:.5f}  "
                          f"(rmse_shape={best_scores[0]:.5f}  "
                          f"delta={best_scores[1]:.5f})")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        all_arr = np.array(all_scores)
        print(f"\nResults ({args.n_tries} attempts):")
        print(f"  Best seed:        {best_seed}")
        print(f"  Best combined:    {best_score:.5f}")
        print(f"  Best rmse_shape:  {best_scores[0]:.5f}")
        print(f"  Best delta_score: {best_scores[1]:.5f}")
        print(f"  Median combined:  {np.median(all_arr):.5f}")
        print(f"  Best percentile:  {np.mean(all_arr >= best_score)*100:.1f}th")
        print(f"  Saved to: {out_dir}/{task}_sequences.{{pkl,json}}")
        if best_seq_df is not None:
            plot_diagnostic(best_seq_df, task, args.n_tries, out_dir, best_score)
            print(f"  Diagnostics: {out_dir}/diagnostics/")

    print("\nJOB_COMPLETE")


if __name__ == '__main__':
    main()
