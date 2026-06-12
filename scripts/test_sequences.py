"""
scripts/test_sequences.py
=========================
Run math models on generated task sequences (continuous and/or binary)
without requiring human data, then produce inspection figures.

Figures saved:
    figures/test_sequences_continuous.pdf
    figures/test_sequences_binary.pdf

Each figure has 5 panels:
    A — RMSE vs ground truth, per observation
    B — Response variability for identical inputs (prefix reps only)
    C — Mean |Δresponse| vs observation (decay curve)
    D — Split-half reliability of response variability (prefix reps)
    E — Split-half reliability of λ (|Δresponse| decay)

Parameter modes (--param_mode):
    fixed       : midpoint defaults (or --params key=value overrides)
    random      : n random draws from param ranges
    sweep       : grid sweep over param ranges
    extreme     : min/mid/max of each param independently
    fitted_yoo  : load fitted params from a yoo run folder
    fitted_carrabin : load fitted params from a carrabin run folder

Usage examples
--------------
    python scripts/test_sequences.py
    python scripts/test_sequences.py --param_mode sweep --sweep_n 3
    python scripts/test_sequences.py --param_mode random --n_random_draws 5
    python scripts/test_sequences.py --param_mode fitted_yoo --run_folder refit
    python scripts/test_sequences.py --tasks continuous --models RL_lambda
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit as scipy_curve_fit
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import FIGURES_DIR, RUNS_DIR, resolve_run_folder
from utils.binary_transform import apply_binary_transform, normalise_continuous
from utils.plot_style import (
    FIGURE_SIZE, apply_style, get_palette, label_panels, fit_power_law_params,
    pvalue_to_stars,
)


# ── Constants ─────────────────────────────────────────────────────────────────

PARAM_RANGES = {
    'RL_lambda':       {'alpha_0': (0.01, 1.0), 'lambda_': (0.01, 1.0)},
    'LeakyIntegrator': {'gamma':   (0.001, 0.999)},
    'PrimacyRecency':  {'eps_p':   (0.001, 1.0), 'eps_r': (0.001, 1.0)},
    'RL':              {'alpha':   (0.001, 1.0)},
    'Mean':            {},
}

ALL_MODELS = ['Mean', 'RL_lambda', 'LeakyIntegrator', 'PrimacyRecency']


# ── Normalisation ─────────────────────────────────────────────────────────────

def norm(task, v):
    return normalise_continuous(v) if task == 'continuous' else float(v)


# ── Model functions (operate on normalised [-1,1] values) ─────────────────────

def run_mean(vals):
    return float(np.mean(vals))

def run_rl(vals, alpha):
    est = 0.0
    for v in vals:
        est += float(alpha) * (float(v) - est)
    return float(np.clip(est, -1, 1))

def run_rl_lambda(vals, alpha_0, lambda_):
    est = 0.0
    for n, v in enumerate(vals, 1):
        est += (float(alpha_0) / n**float(lambda_)) * (float(v) - est)
    return float(np.clip(est, -1, 1))

def run_leaky(vals, gamma):
    est = 0.0
    for v in vals:
        est = float(gamma)*est + (1-float(gamma))*float(v)
    return float(np.clip(est, -1, 1))

def run_primacy_recency(vals, eps_p, eps_r, eta=0.01):
    n = len(vals)
    w = np.array([(1-(1-eps_p**(o+1))*(1-eps_r**(n-o)))*(1-eta)+eta
                  for o in range(n)])
    return float(np.dot(w, vals) / np.sum(w))

def run_model(model_type, vals_norm, params):
    if model_type == 'Mean':           return run_mean(vals_norm)
    if model_type == 'RL':             return run_rl(vals_norm, params['alpha'])
    if model_type == 'RL_lambda':      return run_rl_lambda(vals_norm, params['alpha_0'], params['lambda_'])
    if model_type == 'LeakyIntegrator':return run_leaky(vals_norm, params['gamma'])
    if model_type == 'PrimacyRecency': return run_primacy_recency(vals_norm, params['eps_p'], params['eps_r'])
    raise ValueError(f'Unknown model: {model_type}')


# ── Parameter set generators ──────────────────────────────────────────────────

def params_fixed(model_type, overrides=None):
    defaults = {
        'RL_lambda':       {'alpha_0': 0.3, 'lambda_': 0.5},
        'LeakyIntegrator': {'gamma': 0.7},
        'PrimacyRecency':  {'eps_p': 0.3, 'eps_r': 0.3},
        'RL':              {'alpha': 0.2},
        'Mean':            {},
    }
    p = dict(defaults.get(model_type, {}))
    if overrides:
        p.update(overrides)
    return [p]

def params_random(model_type, n=5, seed=42):
    ranges = PARAM_RANGES[model_type]
    if not ranges: return [{}]
    rng = np.random.default_rng(seed)
    return [{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}
            for _ in range(n)]

def params_sweep(model_type, n_per_param=3):
    ranges = PARAM_RANGES[model_type]
    if not ranges: return [{}]
    grids = {k: np.linspace(lo, hi, n_per_param).tolist()
             for k, (lo, hi) in ranges.items()}
    keys = list(grids.keys())
    return [dict(zip(keys, combo))
            for combo in itertools.product(*[grids[k] for k in keys])]

def params_extreme(model_type):
    ranges = PARAM_RANGES[model_type]
    if not ranges: return [{}]
    defaults = params_fixed(model_type)[0]
    sets = []
    for param, (lo, hi) in ranges.items():
        for val in [lo, (lo+hi)/2, hi]:
            p = dict(defaults); p[param] = float(val)
            sets.append(p)
    return sets

def params_from_fitted(model_type, run_folder):
    folder = resolve_run_folder(run_folder)
    pkls   = sorted(folder.glob(f'{model_type}_*_params.pkl'))
    if not pkls: return params_fixed(model_type)
    sets = []
    for pkl in pkls:
        try:
            row = pd.read_pickle(pkl).iloc[0].to_dict()
            p   = {k: v for k, v in row.items() if k in PARAM_RANGES.get(model_type, {})}
            if p or model_type == 'Mean': sets.append(p)
        except Exception: pass
    return sets or params_fixed(model_type)

def get_param_sets(model_type, args):
    mode = args.param_mode
    if mode == 'fixed':
        overrides = {}
        if args.params:
            for kv in args.params:
                k, v = kv.split('='); overrides[k] = float(v)
        return params_fixed(model_type, overrides)
    if mode == 'random':  return params_random(model_type, args.n_random_draws, args.seed)
    if mode == 'sweep':   return params_sweep(model_type, args.sweep_n)
    if mode == 'extreme': return params_extreme(model_type)
    if mode in ('fitted_yoo', 'fitted_carrabin'):
        return params_from_fitted(model_type, args.run_folder)
    raise ValueError(f'Unknown param_mode: {mode}')


# ── Run models on sequences ───────────────────────────────────────────────────

def run_on_sequences(task, seq_df, models, get_params_fn):
    rows = []
    for model_type in models:
        param_sets = get_params_fn(model_type)
        for ps_id, params in enumerate(param_sets):
            params_str = ' '.join(f'{k}={v:.3f}' for k, v in sorted(params.items())) or 'none'
            # Unique model_id combining model + param set
            model_id = model_type if len(param_sets) == 1 \
                       else f'{model_type}[{ps_id}]'

            for trial_id in seq_df['trial'].unique():
                tdf = seq_df[seq_df['trial']==trial_id].sort_values('observation')
                vals_raw  = tdf['value'].tolist()
                vals_norm = [norm(task, v) for v in vals_raw]
                meta = tdf.iloc[0]

                for oi, obs in enumerate(tdf['observation'].tolist()):
                    resp = run_model(model_type, vals_norm[:oi+1], params)
                    rows.append({
                        'task':         task,
                        'model_type':   model_type,
                        'model_id':     model_id,
                        'param_set_id': ps_id,
                        'params_str':   params_str,
                        'trial':        int(trial_id),
                        'qid':          meta.get('qid'),
                        'trial_type':   meta.get('trial_type'),
                        'prefix_length':meta.get('prefix_length', 0),
                        'observation':  int(obs),
                        'value':        vals_raw[oi],
                        'value_norm':   vals_norm[oi],
                        'response':     resp,
                        'response_raw': resp,
                        'true_mean':    meta.get('true_mean'),
                        'true_std':     meta.get('true_std'),
                        'true_p':       meta.get('true_p'),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = apply_binary_transform(df, f'task_{task}')
    return df


# ── Metric helpers ────────────────────────────────────────────────────────────

def ground_truth_rmse_per_obs(df, task):
    """Panel A: RMSE vs ground truth (cumulative mean for continuous, true_p for binary)."""
    rows = []
    for (model_id, trial), g in df.groupby(['model_id','trial']):
        g = g.sort_values('observation').copy()
        vals_norm = g['value_norm'].tolist()
        for oi, (_, row) in enumerate(g.iterrows()):
            gt = float(np.mean(vals_norm[:oi+1]))  # cumulative mean = optimal
            err = abs(row['response'] - gt)
            rows.append({'model_id': row['model_id'], 'model_type': row['model_type'],
                         'observation': row['observation'], 'sq_err': err**2})
    if not rows: return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.groupby(['model_id','model_type','observation'])['sq_err'].mean().apply(np.sqrt).reset_index(name='rmse')


def prefix_variability(df):
    """Panel B: std of responses across repeats of the same qid×observation,
    within the prefix region only."""
    prefix_df = df[df['trial_type']=='structured'].copy()
    if prefix_df.empty: return pd.DataFrame()
    # Only observations within prefix
    prefix_df = prefix_df[prefix_df['observation'] < prefix_df['prefix_length']]
    if prefix_df.empty: return pd.DataFrame()
    return (prefix_df.groupby(['model_id','model_type','qid','observation'])['response']
            .std().reset_index(name='resp_std'))


def abs_delta(df):
    """Panel C: |Δresponse| per observation."""
    rows = []
    for (model_id, trial), g in df.groupby(['model_id','trial']):
        g = g.sort_values('observation').copy()
        g['delta'] = g['response'].diff().abs()
        rows.append(g)
    if not rows: return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[out['observation'] >= 1].dropna(subset=['delta'])


def split_half_variability(df):
    """Panel D: test-retest of mean prefix variability across trial halves."""
    rows = []
    for model_id, g in df.groupby('model_id'):
        trials = sorted(g['trial'].unique())
        mid    = len(trials) // 2
        if mid < 2: continue
        for half, trial_set in [('first', trials[:mid]), ('second', trials[mid:])]:
            sub = g[g['trial'].isin(trial_set)]
            pv  = prefix_variability(sub)
            if pv.empty: continue
            mean_var = pv.groupby(['qid','observation'])['resp_std'].mean().mean()
            rows.append({'model_id': model_id,
                         'model_type': g['model_type'].iloc[0],
                         'half': half, 'mean_var': mean_var})
    if not rows: return pd.DataFrame()
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=['model_id','model_type'],
                            columns='half', values='mean_var').reset_index()
    wide.columns.name = None
    return wide.dropna(subset=['first','second'])


def fit_lambda(df):
    """Fit power-law λ to |Δresponse| curve per model_id. Returns Series."""
    def power_law(n, A, lam):
        return A * np.power(np.asarray(n, float), -lam)
    out = {}
    for model_id, g in df.groupby('model_id'):
        dlt = abs_delta(g)
        if dlt.empty: continue
        curve = dlt.groupby('observation')['delta'].mean().dropna()
        curve = curve[curve.index >= 1]
        if len(curve) < 3: continue
        try:
            popt, _ = scipy_curve_fit(power_law, curve.index.values.astype(float),
                                      curve.values.astype(float),
                                      p0=[0.1, 0.5], bounds=([0,0],[2,2]), maxfev=2000)
            out[model_id] = float(popt[1])
        except Exception: pass
    return pd.Series(out, name='lambda_')


def split_half_lambda(df):
    """Panel E: test-retest of λ across trial halves."""
    rows = []
    for model_id, g in df.groupby('model_id'):
        trials = sorted(g['trial'].unique())
        mid    = len(trials) // 2
        if mid < 3: continue
        for half, trial_set in [('first', trials[:mid]), ('second', trials[mid:])]:
            sub  = g[g['trial'].isin(trial_set)].copy()
            lam  = fit_lambda(sub)
            if model_id in lam.index:
                rows.append({'model_id': model_id,
                             'model_type': g['model_type'].iloc[0],
                             'half': half, 'lambda_': float(lam[model_id])})
    if not rows: return pd.DataFrame()
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=['model_id','model_type'],
                            columns='half', values='lambda_').reset_index()
    wide.columns.name = None
    return wide.dropna(subset=['first','second'])


# ── Figure panels ─────────────────────────────────────────────────────────────

def _blank(ax, msg='No data\n(noisy models only)'):
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, color='0.5', style='italic', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_panel_a(ax, df, palette):
    """A: RMSE vs ground truth per observation.
    For each model_type, shaded band spans from the mean of the weak-U subgroup
    (bottom N_GROUP model_ids by U-strength) to the mean of the strong-U subgroup
    (top N_GROUP), with thin edge lines — matching the yoo figure convention.
    """
    N_GROUP = 10
    SMOOTH  = 3

    # Compute per-(model_id, trial, obs) task error
    rows = []
    for (mid, trial), g in df.groupby(['model_id','trial']):
        g = g.sort_values('observation')
        vals = g['value_norm'].tolist()
        for oi, (_, row) in enumerate(g.iterrows()):
            gt = float(np.mean(vals[:oi+1]))
            rows.append({'model_id': row['model_id'],
                         'model_type': row['model_type'],
                         'trial': trial,
                         'observation': int(row['observation']),
                         'err': abs(row['response'] - gt)})
    if not rows: _blank(ax, 'No responses'); return
    err_df = pd.DataFrame(rows)

    # Per model_id: mean error curve, U-strength = late_error - min(smoothed)
    def u_strength(g):
        curve = g.groupby('observation')['err'].mean().sort_index()
        sm    = pd.Series(curve.values).rolling(SMOOTH, min_periods=1, center=True).mean().values
        late  = float(np.mean(curve.values[-3:]))
        return late - float(np.min(sm))

    handles, labels = [], []
    for model_type in sorted(err_df['model_type'].unique()):
        color = palette.get(model_type, '0.5')
        mt_df = err_df[err_df['model_type']==model_type]
        strengths = mt_df.groupby('model_id').apply(u_strength).sort_values()

        n = len(strengths)
        if n < N_GROUP * 2:
            # Not enough param sets — just plot mean
            curve = mt_df.groupby('observation')['err'].mean()
            ax.plot(curve.index, curve.values, color=color, lw=1.8,
                    label='_nolegend_')
        else:
            weak_ids   = strengths.index[:N_GROUP]
            strong_ids = strengths.index[-N_GROUP:]
            obs = sorted(mt_df['observation'].unique())
            weak_mean   = (mt_df[mt_df['model_id'].isin(weak_ids)]
                           .groupby('observation')['err'].mean().reindex(obs))
            strong_mean = (mt_df[mt_df['model_id'].isin(strong_ids)]
                           .groupby('observation')['err'].mean().reindex(obs))
            ax.fill_between(obs, weak_mean.values, strong_mean.values,
                            color=color, alpha=0.18, zorder=1, linewidth=0)
            ax.plot(obs, weak_mean.values,   color=color, lw=1.8, zorder=2,
                    label='_nolegend_')
            ax.plot(obs, strong_mean.values, color=color, lw=1.8, zorder=2,
                    label='_nolegend_')

        handles.append(Line2D([0],[0], color=color, lw=1.8))
        labels.append(model_type)

    ax.set_xlabel('Observation'); ax.set_ylabel('RMSE vs cumulative mean')
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_panel_b(ax, df, palette):
    """B: Response variability within prefix region (std across qid repeats)."""
    pv = prefix_variability(df)
    if pv.empty or pv['resp_std'].isna().all():
        _blank(ax); return
    handles, labels = [], []
    for model_type, g in pv.groupby('model_type'):
        color = palette.get(model_type, '0.5')
        mean_by_obs = g.groupby('observation')['resp_std'].mean()
        ax.plot(mean_by_obs.index, mean_by_obs.values, color=color, lw=1.8, label='_nolegend_')
        handles.append(Line2D([0],[0], color=color, lw=1.8))
        labels.append(model_type)
    ax.set_xlabel('Observation (prefix only)')
    ax.set_ylabel('Response std (within qid repeats)')
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_panel_c(ax, df, palette):
    """C: Mean |Δresponse| vs observation — lineplot with CI across trials."""
    dlt = abs_delta(df)
    if dlt.empty: _blank(ax, 'No responses'); return
    handles, labels = [], []
    mt_map = df[['model_id','model_type']].drop_duplicates().set_index('model_id')['model_type']
    dlt['model_type'] = dlt['model_id'].map(mt_map)
    for model_type in dlt['model_type'].unique():
        g = dlt[dlt['model_type']==model_type]
        color = palette.get(model_type, '0.5')
        sns.lineplot(data=g, x='observation', y='delta',
                     color=color, lw=1.8, errorbar='ci', ax=ax, legend=False)
        handles.append(Line2D([0],[0], color=color, lw=1.8))
        labels.append(model_type)
    ax.set_xlabel('Observation'); ax.set_ylabel('Mean |Δresponse|')
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_panel_d(ax, df, palette):
    """D: Split-half reliability of prefix response variability.
    Models with variance: regplot line. Models with zero variance: single point."""
    wide = split_half_variability(df)
    if wide.empty:
        _blank(ax); return

    handles, labels = [], []
    any_plotted = False
    var_order = (wide.groupby('model_type')['first'].std()
                 .fillna(0).sort_values().index.tolist())

    for model_type in var_order:
        g = wide[wide['model_type']==model_type]
        color = palette.get(model_type, '0.5')
        std_first = g['first'].std()

        if len(g) >= 3 and std_first > 1e-6:
            sns.regplot(data=g, x='first', y='second', ax=ax,
                        color=color, ci=95, scatter=False,
                        line_kws={'lw': 2.0})
            r, p = pearsonr(g['first'], g['second'])
            handles.append(Line2D([0],[0], color=color, lw=2.0))
            labels.append(f'{model_type} r={r:.2f}{pvalue_to_stars(p)}')
            any_plotted = True
        elif len(g) >= 1:
            mx, my = g['first'].mean(), g['second'].mean()
            ax.scatter([mx], [my], color=color, s=60, zorder=5,
                       marker='D', edgecolors='white', linewidths=0.5)
            handles.append(Line2D([0],[0], color=color, lw=0, marker='D',
                                  ms=6, markeredgecolor='white'))
            labels.append(f'{model_type} (var≈0)')
            any_plotted = True

    if not any_plotted:
        _blank(ax); return

    all_v = pd.concat([wide['first'], wide['second']])
    lo, hi = max(0, all_v.min()-0.005), all_v.max()+0.005
    ax.plot([lo, hi], [lo, hi], color='0.7', lw=0.8, ls='--', zorder=0)
    ax.set_xlabel('Mean var (1st half)'); ax.set_ylabel('Mean var (2nd half)')
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_panel_e(ax, df, palette):
    """E: Split-half reliability of λ.
    Models with variance: regplot line. Models with no variance: single point.
    Drawn in ascending std order so low-variance models aren't buried."""
    wide = split_half_lambda(df)
    if wide.empty:
        _blank(ax); return

    handles, labels = [], []
    any_plotted = False

    # Sort model_types by variance so high-var models draw last (on top)
    var_order = (wide.groupby('model_type')['first'].std()
                 .fillna(0).sort_values().index.tolist())

    for model_type in var_order:
        g = wide[wide['model_type']==model_type]
        color = palette.get(model_type, '0.5')
        std_first = g['first'].std()

        if len(g) >= 3 and std_first > 1e-6:
            # Enough variance — draw regplot line
            sns.regplot(data=g, x='first', y='second', ax=ax,
                        color=color, ci=95, scatter=False,
                        line_kws={'lw': 2.0})
            r, p = pearsonr(g['first'], g['second'])
            handles.append(Line2D([0],[0], color=color, lw=2.0))
            labels.append(f'{model_type} r={r:.2f}{pvalue_to_stars(p)}')
            any_plotted = True
        elif len(g) >= 1:
            # Near-zero variance — single point at mean
            mx, my = g['first'].mean(), g['second'].mean()
            ax.scatter([mx], [my], color=color, s=60, zorder=5,
                       marker='D', edgecolors='white', linewidths=0.5)
            handles.append(Line2D([0],[0], color=color, lw=0, marker='D',
                                  ms=6, markeredgecolor='white'))
            labels.append(f'{model_type} (λ≈0)')
            any_plotted = True

    if not any_plotted:
        _blank(ax); return

    # Identity line
    all_v = pd.concat([wide['first'], wide['second']])
    lo, hi = max(0, all_v.min()-0.05), all_v.max()+0.05
    ax.plot([lo, hi], [lo, hi], color='0.7', lw=0.8, ls='--', zorder=0)

    ax.set_xlabel('λ (1st half of trials)'); ax.set_ylabel('λ (2nd half of trials)')
    ax.legend(handles, labels, fontsize=7, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def make_figure(task, df, palette, out_path):
    fig, axes = plt.subplots(1, 5,
                             figsize=(FIGURE_SIZE[0] * 1.1, FIGURE_SIZE[1] / 2),
                             constrained_layout=True)
    fig.suptitle(f'Task: {task}', fontsize=10, fontweight='bold')

    plot_panel_a(axes[0], df, palette)
    plot_panel_b(axes[1], df, palette)
    plot_panel_c(axes[2], df, palette)
    plot_panel_d(axes[3], df, palette)
    plot_panel_e(axes[4], df, palette)

    label_panels(axes.reshape(1, -1))
    plt.savefig(out_path)
    print(f'Saved {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tasks', nargs='+', default=['continuous','binary'],
                        choices=['continuous','binary'])
    parser.add_argument('--models', nargs='+', default=ALL_MODELS)
    parser.add_argument('--seq_dir', default='task/sequences')
    parser.add_argument('--output_dir', default='test_sequences')

    parser.add_argument('--param_mode',
                        choices=['fixed','random','sweep','extreme',
                                 'fitted_yoo','fitted_carrabin'],
                        default='fixed')
    parser.add_argument('--params', nargs='*', metavar='KEY=VALUE')
    parser.add_argument('--n_random_draws', type=int, default=5)
    parser.add_argument('--sweep_n',        type=int, default=3)
    parser.add_argument('--run_folder',     default='yoo')
    parser.add_argument('--apply_transform', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    apply_style()
    out_folder = resolve_run_folder(args.output_dir)
    seq_dir    = Path(args.seq_dir)
    all_dfs    = []

    # Build a stable colour palette keyed by model_type
    pal     = get_palette(len(args.models) + 1)
    palette = {m: pal[i] for i, m in enumerate(args.models)}

    for task in args.tasks:
        pkl = seq_dir / f'{task}_sequences.pkl'
        if not pkl.exists():
            print(f'[skip] {pkl} not found'); continue
        seq_df = pd.read_pickle(pkl)
        print(f'\n{"="*55}\nTask: {task.upper()} | {seq_df["trial"].nunique()} trials')

        df = run_on_sequences(
            task, seq_df, args.models,
            lambda mt: get_param_sets(mt, args),
        )
        if df.empty:
            print('  No responses generated'); continue

        all_dfs.append(df)

        # Build palette extended to model_ids
        extended = dict(palette)
        for mid in df['model_id'].unique():
            mt = df[df['model_id']==mid]['model_type'].iloc[0]
            if mid not in extended:
                extended[mid] = palette.get(mt, '0.5')

        stem = f'test_sequences_{task}'
        make_figure(task, df, extended,
                    FIGURES_DIR / f'{stem}.pdf')

    # Save combined pkl
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = out_folder / 'test_sequences_responses.pkl'
        combined.to_pickle(out_path)
        print(f'\nSaved responses: {out_path}')

    print('JOB_COMPLETE')


if __name__ == '__main__':
    main()
