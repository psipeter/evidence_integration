"""
scripts/test_sequences_subsampling.py
======================================
Subsampling power analysis for lambda parameter recovery.

For each combination of (n_trials, n_obs), bootstrap-subsample the full
dataset and measure:
  1. Fraction of pids with significant decay (p<0.05 Spearman)
  2. Pearson r between subsampled fitted lambda and reference lambda
     (lambda fitted from full dataset — ground truth on measurement scale)
  3. RMSE between subsampled and reference lambda

Separately for: continuous/binary × RL_lambda/PrimacyRecency.

Output: figures/test_sequences_subsampling.pdf  (2×2 grid of heatmaps)
        data/runs/test_sequences/subsampling_results.pkl

Usage:
    python scripts/test_sequences_subsampling.py
    python scripts/test_sequences_subsampling.py --n_boots 50
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES_DIR, resolve_run_folder
from utils.plot_style import apply_style, label_panels, FIGURE_SIZE

# ── Grid ─────────────────────────────────────────────────────────────────────
N_TRIALS_GRID = [10, 20, 30, 40, 50]
N_OBS_GRID    = [10, 15, 20]
MODELS        = ['RL_lambda']
TASKS         = ['continuous', 'binary']


# ── Fitting helpers ───────────────────────────────────────────────────────────

def fit_lambda(g, min_obs=2):
    """Fit power law A*n^(-lam) to |Δresponse| curve. Returns (lam, p_spearman)."""
    rows = []
    for trial, tg in g.groupby('trial'):
        tg = tg.sort_values('observation')
        delta = tg['response'].diff().abs()
        for obs, d in zip(tg['observation'], delta):
            if pd.notna(d) and obs >= min_obs:
                rows.append({'observation': int(obs), 'delta': float(d)})
    if len(rows) < 5:
        return np.nan, np.nan
    ddf = pd.DataFrame(rows)
    curve = ddf.groupby('observation')['delta'].mean().sort_index()
    if len(curve) < 3:
        return np.nan, np.nan
    n, y = curve.index.values.astype(float), curve.values.astype(float)
    try:
        popt, _ = curve_fit(lambda n, A, lam: A * n**(-lam),
                            n, y, p0=[0.1, 0.5], bounds=([0,0],[2,2]), maxfev=2000)
        lam = float(popt[1])
    except:
        return np.nan, np.nan
    _, p = spearmanr(n, y)
    return lam, float(p)


def fit_lambda_from_curve(obs_array, delta_array):
    """Fit power law directly from precomputed (obs, delta) arrays."""
    n = obs_array.astype(float)
    y = delta_array.astype(float)
    if len(n) < 3 or not (np.all(np.isfinite(n)) and np.all(np.isfinite(y))):
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(lambda n, A, lam: A * n**(-lam),
                            n, y, p0=[0.1, 0.5], bounds=([0,0],[2,2]), maxfev=2000)
        lam = float(popt[1])
    except:
        return np.nan, np.nan
    _, p = spearmanr(n, y)
    return lam, float(p)


def precompute_delta_tables(df, min_obs=2):
    """
    Precompute per-(task, model_type, model_id, trial) delta tables.
    Returns dict: (task, model_type, model_id, trial) ->
                  DataFrame with columns [observation, delta].
    Fast lookup avoids repeated groupby/diff in the inner boot loop.
    """
    tables = {}
    for (task, model_type, mid, trial), g in df.groupby(
            ['task', 'model_type', 'model_id', 'trial']):
        g = g.sort_values('observation').copy()
        g['delta'] = g['response'].diff().abs()
        sub = g[['observation', 'delta']].dropna()
        sub = sub[sub['observation'] >= min_obs]
        tables[(task, model_type, mid, trial)] = sub
    return tables


def compute_reference_lambdas(df):
    """Fit lambda from full data per model_id — ground truth on measurement scale."""
    ref = {}
    for (task, model_type, mid), g in df.groupby(['task','model_type','model_id']):
        lam, _ = fit_lambda(g)
        ref[(task, model_type, mid)] = lam
    return ref


# ── Subsampling ───────────────────────────────────────────────────────────────

def subsample_stats(delta_tables, ref_lambdas, task, model_type,
                    all_trials, model_ids, n_trials, n_obs, n_boots, rng):
    """
    Bootstrap n_boots times using precomputed delta tables.
    For each boot: subsample n_trials trials, truncate to n_obs,
    aggregate delta per (model_id, obs), fit lambda, compare to reference.
    """
    nt = min(n_trials, len(all_trials))

    frac_sig_boots, r_boots, rmse_boots = [], [], []

    for _ in range(n_boots):
        trials_s = set(rng.choice(all_trials, size=nt, replace=False).tolist())

        lam_sub, lam_ref = [], []
        n_sig = 0

        for mid in model_ids:
            # Aggregate delta across sampled trials, truncated to n_obs
            pieces = []
            for trial in trials_s:
                t = delta_tables.get((task, model_type, mid, trial))
                if t is not None and not t.empty:
                    pieces.append(t[t['observation'] < n_obs])
            if not pieces:
                continue
            agg = pd.concat(pieces, ignore_index=True)
            curve = agg.groupby('observation')['delta'].mean().sort_index()
            if len(curve) < 3:
                continue
            lam, p = fit_lambda_from_curve(curve.index.values, curve.values)
            ref = ref_lambdas.get((task, model_type, mid), np.nan)
            if np.isfinite(lam) and np.isfinite(ref):
                lam_sub.append(lam)
                lam_ref.append(ref)
                if np.isfinite(p) and p < 0.05:
                    n_sig += 1

        frac_sig = n_sig / len(model_ids) if model_ids else np.nan
        frac_sig_boots.append(frac_sig)

        if len(lam_sub) >= 5:
            r, _  = pearsonr(lam_ref, lam_sub)
            rmse  = float(np.sqrt(np.mean((np.array(lam_ref) - np.array(lam_sub))**2)))
            r_boots.append(r)
            rmse_boots.append(rmse)

    return {
        'frac_sig': float(np.nanmean(frac_sig_boots)),
        'r':        float(np.nanmean(r_boots)) if r_boots else np.nan,
        'rmse':     float(np.nanmean(rmse_boots)) if rmse_boots else np.nan,
    }


# ── Heatmap helper ────────────────────────────────────────────────────────────

def make_heatmap(ax, pivot, title, cmap, vmin, vmax, fmt='.2f', annot=True):
    sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=annot, fmt=fmt, linewidths=0.4, linecolor='0.9',
                cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=8, fontweight='bold')
    ax.set_xlabel('n_obs'); ax.set_ylabel('n_trials')
    ax.tick_params(labelsize=7)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_boots', type=int, default=10)
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    apply_style()
    rng = np.random.default_rng(args.seed)
    out_folder = resolve_run_folder('test_sequences')

    print("Loading data...")
    df_all = pd.read_pickle(out_folder / 'test_sequences_responses.pkl')
    df = df_all[df_all['model_type'].isin(MODELS)].copy()
    print(f"  {len(df)} rows, {df['model_id'].nunique()} model_ids "
          f"({df['model_type'].unique().tolist()}), "
          f"obs 0-{df['observation'].max()}, trials 0-{df['trial'].max()}")

    print("Computing reference lambdas (full data)...")
    ref_lambdas = compute_reference_lambdas(df)
    n_valid = sum(1 for v in ref_lambdas.values() if np.isfinite(v))
    print(f"  {n_valid}/{len(ref_lambdas)} valid reference lambdas")

    # ── Precompute delta tables ───────────────────────────────────────────────
    print("Precomputing delta tables...")
    delta_tables = precompute_delta_tables(df)
    print(f"  {len(delta_tables)} tables cached")

    # ── Run subsampling grid ──────────────────────────────────────────────────
    results = []
    total = len(TASKS) * len(MODELS) * len(N_TRIALS_GRID) * len(N_OBS_GRID)
    done  = 0
    for task in TASKS:
        for model_type in MODELS:
            sub      = df[(df.task==task) & (df.model_type==model_type)]
            all_tr   = sub['trial'].unique().tolist()
            model_ids = sub['model_id'].unique().tolist()
            print(f"\n{task} / {model_type}  ({len(model_ids)} pids, {len(all_tr)} trials)")
            for n_trials in N_TRIALS_GRID:
                for n_obs in N_OBS_GRID:
                    stats = subsample_stats(
                        delta_tables, ref_lambdas, task, model_type,
                        all_tr, model_ids, n_trials, n_obs, args.n_boots, rng
                    )
                    results.append({
                        'task': task, 'model_type': model_type,
                        'n_trials': n_trials, 'n_obs': n_obs,
                        **stats
                    })
                    done += 1
                    print(f"  [{done:3d}/{total}] n_trials={n_trials:3d} n_obs={n_obs:2d}  "
                          f"frac_sig={stats['frac_sig']:.2f}  r={stats['r']:.3f}  "
                          f"rmse={stats['rmse']:.3f}")

    res_df = pd.DataFrame(results)
    pkl_path = out_folder / 'subsampling_results.pkl'
    res_df.to_pickle(pkl_path)
    print(f"\nSaved: {pkl_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    metrics = [
        ('r',        'r (fitted λ vs reference λ)', 'YlOrRd', 0, 1),
        ('frac_sig', 'Fraction pids p<0.05',         'YlOrRd', 0, 1),
        ('rmse',     'RMSE (fitted vs reference λ)',  'YlOrRd_r', 0, None),
    ]

    n_rows = len(TASKS) * len(MODELS)   # 4
    n_cols = len(metrics)                # 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIGURE_SIZE[0] * 0.85, FIGURE_SIZE[1] * 1.1),
        constrained_layout=True
    )

    row = 0
    for task in TASKS:
        for model_type in MODELS:
            sub = res_df[(res_df.task==task) & (res_df.model_type==model_type)]
            for col, (metric, title, cmap, vmin, vmax) in enumerate(metrics):
                ax = axes[row, col]
                pivot = (sub.pivot(index='n_trials', columns='n_obs', values=metric)
                           .sort_index(ascending=False))
                vmax_use = pivot.values.max() if vmax is None else vmax
                make_heatmap(ax, pivot,
                             f"{task} / {model_type}\n{title}",
                             cmap, vmin, vmax_use)
            row += 1

    label_panels(axes)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES_DIR / 'test_sequences_subsampling.pdf'
    plt.savefig(out_pdf)
    print(f"Saved: {out_pdf}")
    print("JOB_COMPLETE")


if __name__ == '__main__':
    main()
