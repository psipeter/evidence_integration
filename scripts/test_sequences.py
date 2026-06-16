"""
scripts/test_sequences_combined.py
===================================
Combined inspection figure for both tasks + cross-task panels.

Layout: 3 rows × 5 cols + 3 extra panels
  Row 1 (A–E): Binary task    — RMSE, prefix var, |Δresponse|, split-half var, split-half λ
  Row 2 (F–J): Continuous task — same panels
  Row 3 (K–M): Cross-task     — λ correlation, prefix variability correlation, λ → late error

Usage:
    python scripts/test_sequences_combined.py
    python scripts/test_sequences_combined.py --param_mode random --n_random_draws 100
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit as scipy_curve_fit
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES_DIR, resolve_run_folder
from utils.plot_style import (
    FIGURE_SIZE, apply_style, get_palette, label_panels,
    pvalue_to_stars,
)


# ── RL_lambda parameter scan ─────────────────────────────────────────────────
# Fixed alpha_0, uniform grid of lambda_ values.
# model_id encodes the lambda_ value for easy downstream splitting.

def _lambda_scan_params(alpha_0=0.5, n_lambdas=50):
    """Return list of param dicts: fixed alpha_0, lambda_ on uniform grid [0.01, 0.99]."""
    return [{'alpha_0': alpha_0, 'lambda_': float(lam)}
            for lam in np.linspace(0.01, 0.99, n_lambdas)]

def _run_model(model_type, vals_norm, params):
    est = 0.0
    if model_type == 'Mean':
        return float(np.mean(vals_norm))
    for n, v in enumerate(vals_norm, 1):
        if model_type == 'RL_lambda':
            alpha = params['alpha_0'] / n**params['lambda_']
        elif model_type == 'LeakyIntegrator':
            alpha = 1 - params['gamma']
        elif model_type == 'RL':
            alpha = params['alpha']
        else:
            raise ValueError(f"Unknown model: {model_type}")
        est += alpha * (float(v) - est)
        est = float(np.clip(est, -1, 1))
    return est

def run_models_on_sequences(seq_dir, output_dir, tasks, models,
                             alpha_0=0.5, n_lambdas=50, seed=42):
    """Run math models on generated sequences, save responses pkl."""
    from utils.binary_transform import apply_binary_transform
    seq_dir    = Path(seq_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for task in tasks:
        pkl = seq_dir / f'{task}_sequences.pkl'
        if not pkl.exists():
            print(f"[skip] {pkl} not found"); continue
        seq_df = pd.read_pickle(pkl)
        print(f"\n{'='*50}\nTask: {task.upper()} | {seq_df['trial'].nunique()} trials")

        for model_type in models:
            param_sets = _lambda_scan_params(alpha_0=alpha_0, n_lambdas=n_lambdas)
            print(f"  {model_type}: {len(param_sets)} param set(s) "
                  f"(alpha_0={alpha_0}, lambda_ grid 0.01-0.99 n={n_lambdas})")

            for ps_id, params in enumerate(param_sets):
                params_str = ' '.join(f'{k}={v:.3f}' for k,v in sorted(params.items())) or 'none'
                model_id   = model_type if len(param_sets)==1 else f'{model_type}[{ps_id}]'

                for trial_id in seq_df['trial'].unique():
                    tdf = seq_df[seq_df['trial']==trial_id].sort_values('observation')
                    vals_raw  = tdf['value'].tolist()
                    vals_norm = [v/100.0 for v in vals_raw] if task=='continuous' \
                                else [float(v) for v in vals_raw]
                    meta = tdf.iloc[0]

                    for oi, obs in enumerate(tdf['observation'].tolist()):
                        resp = _run_model(model_type, vals_norm[:oi+1], params)
                        rows.append({
                            'task': task, 'model_type': model_type,
                            'model_id': model_id, 'param_set_id': ps_id,
                            'params_str': params_str, 'trial': int(trial_id),
                            'qid': meta.get('qid'), 'trial_type': meta.get('trial_type'),
                            'prefix_length': meta.get('prefix_length', 0),
                            'observation': int(obs),
                            'value': vals_raw[oi], 'value_norm': vals_norm[oi],
                            'response': resp, 'response_raw': resp,
                            'true_mean': meta.get('true_mean'),
                            'true_std':  meta.get('true_std'),
                            'true_p':    meta.get('true_p'),
                            'std_condition': meta.get('std_condition'),
                        })

    if not rows:
        print("No results generated."); return None
    df = pd.DataFrame(rows)
    df = apply_binary_transform(df, 'task_binary')
    out_path = output_dir / 'test_sequences_responses.pkl'
    df.to_pickle(out_path)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")
    return df


# 4 lambda quartile styles: Q1 (lowest λ) → Q4 (highest λ)
# Colour encodes model type; linestyle+weight encode quartile
LAMBDA_Q_STYLES = {
    0: {'lw': 0.8,  'ls': ':',  'alpha': 0.5,  'label': 'Q1 (low λ)'},
    1: {'lw': 1.2,  'ls': '--', 'alpha': 0.7,  'label': 'Q2'},
    2: {'lw': 1.6,  'ls': '-.', 'alpha': 0.85, 'label': 'Q3'},
    3: {'lw': 2.2,  'ls': '-',  'alpha': 1.0,  'label': 'Q4 (high λ)'},
}

def std_condition_label(val):
    """Convert numeric std_condition to 'low'/'high' string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return 'low' if float(val) < 15 else 'high'

N_GROUP       = 10
SMOOTH        = 3       # window for U-shape detection
SMOOTH_WINDOW = 3       # window for response smoothing before all metrics
ALL_MODELS = ['RL_lambda', 'NEF']


def subsample_df(df, n_trials, n_obs, seed=42):
    """Randomly subsample n_trials trials and truncate to first n_obs observations."""
    rng = np.random.default_rng(seed)
    all_trials = df['trial'].unique()
    nt = min(n_trials, len(all_trials))
    chosen = rng.choice(all_trials, size=nt, replace=False)
    return df[(df['trial'].isin(chosen)) & (df['observation'] < n_obs)].copy()


# ── Metric helpers ─────────────────────────────────────────────────────────

def smooth_responses_per_pid(df, window=SMOOTH_WINDOW):
    """Apply rolling mean over observations within each (model_id, trial),
    replacing 'response' with the smoothed version. Applied before all
    metric computations so RMSE, |Δ|, λ fitting etc. all use smoothed data."""
    if window <= 1:
        return df
    out = []
    for (mid, trial), g in df.groupby(['model_id', 'trial'], sort=False):
        g = g.sort_values('observation').copy()
        g['response'] = (g['response']
                         .rolling(window, min_periods=1, center=False)
                         .mean().values)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def compute_task_error(df):
    """Per-(model_id, trial, obs): |response - true_mean_norm|.
    Uses the true generative mean (normalised) as ground truth, not the
    cumulative sample mean, so early-obs error reflects model uncertainty
    rather than sample noise in the ground truth."""
    rows = []
    for (mid, trial), g in df.groupby(['model_id', 'trial']):
        g = g.sort_values('observation')
        # True mean: normalised true_mean for continuous, true_p mapped to
        # {-1,+1} scale for binary (true_p*2-1)
        task = g['task'].iloc[0]
        if task == 'continuous':
            gt = float(g['true_mean'].iloc[0]) / 100.0
        else:
            gt = float(g['true_p'].iloc[0]) * 2 - 1
        for _, row in g.iterrows():
            rows.append({'model_id': row['model_id'],
                         'model_type': row['model_type'],
                         'trial': trial,
                         'observation': int(row['observation']),
                         'err': abs(row['response'] - gt)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def u_strength(err_df_mid):
    """Late error minus smoothed minimum — U-shape strength per model_id."""
    curve = err_df_mid.groupby('observation')['err'].mean().sort_index()
    sm    = pd.Series(curve.values).rolling(SMOOTH, min_periods=1, center=True).mean().values
    return float(np.mean(curve.values[-3:])) - float(np.nanmin(sm))


def compute_abs_delta(df):
    """Per-(model_id, trial, obs): |Δresponse|."""
    rows = []
    for (mid, trial), g in df.groupby(['model_id', 'trial']):
        g = g.sort_values('observation').copy()
        g['delta'] = g['response'].diff().abs()
        rows.append(g[['model_id','model_type','trial','observation','delta']])
    if not rows: return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).dropna(subset=['delta'])



def late_delta_quartiles(df, n_late=3):
    """Per-model_id mean |Δresponse| in the last n_late observations.
    Returns dict: model_id -> quartile (0=Q1 lowest, 3=Q4 highest).
    Matches the yoo temporal A / neural D quartile split approach."""
    dlt = compute_abs_delta(df)
    if dlt.empty: return {}
    n_obs = dlt['observation'].max()
    late_start = n_obs - n_late + 1
    late_dlt = (dlt[dlt.observation >= late_start]
                .groupby('model_id')['delta'].mean()
                .dropna())
    if len(late_dlt) < 4: return {}
    bins = pd.qcut(late_dlt, q=4, labels=False, duplicates='drop')
    return bins.to_dict()

def true_lambda_quartiles(df):
    """Per-model_id true lambda from params_str → quartile (0=Q1 lowest, 3=Q4 highest)."""
    lam_map = {}
    for mid, g in df.groupby('model_id'):
        ps = g['params_str'].iloc[0]
        try:
            lam = float(ps.split('lambda_=')[1].split()[0])
            lam_map[mid] = lam
        except (IndexError, ValueError):
            pass
    if len(lam_map) < 4:
        return {}
    lam_s = pd.Series(lam_map)
    bins  = pd.qcut(lam_s, q=4, labels=False, duplicates='drop')
    return bins.to_dict()


def fit_lambda_mid(g, min_obs=2):
    """Fit power-law to |Δresponse| curve for one model_id. Returns (lambda, p)."""
    rows = []
    for trial, tg in g.groupby('trial'):
        tg = tg.sort_values('observation')
        delta = tg['response'].diff().abs()
        for obs, d in zip(tg['observation'], delta):
            if pd.notna(d) and obs >= min_obs:
                rows.append({'observation': int(obs), 'delta': float(d)})
    if not rows: return np.nan, np.nan
    ddf = pd.DataFrame(rows)
    curve = ddf.groupby('observation')['delta'].mean().sort_index()
    if len(curve) < 3: return np.nan, np.nan
    n, y = curve.index.values.astype(float), curve.values.astype(float)
    try:
        popt, _ = scipy_curve_fit(lambda n, A, lam: A * n**(-lam),
                                  n, y, p0=[0.1,0.5], bounds=([0,0],[2,2]), maxfev=2000)
        lam = float(popt[1])
    except: return np.nan, np.nan
    _, p = spearmanr(n, y)
    return lam, float(p)


def prefix_var_per_mid(df):
    """Mean response std within prefix region per model_id."""
    pdf = df[(df.trial_type=='structured') &
             (df.observation < df.prefix_length)].copy()
    if pdf.empty: return pd.Series(dtype=float)
    pv = (pdf.groupby(['model_id','model_type','qid','observation'])['response']
           .std().reset_index(name='resp_std'))
    return pv.groupby('model_id')['resp_std'].mean()


def split_half_var(df):
    """Split-half mean prefix variability per model_id → (first, second)."""
    rows = []
    for mid, g in df.groupby('model_id'):
        trials = sorted(g['trial'].unique())
        mid_t  = len(trials) // 2
        if mid_t < 2: continue
        for half, tset in [('first',trials[:mid_t]),('second',trials[mid_t:])]:
            pdf = g[(g['trial'].isin(tset)) & (g['trial_type']=='structured') &
                    (g['observation'] < g['prefix_length'])]
            if pdf.empty: continue
            pv = (pdf.groupby(['qid','observation'])['response'].std()
                  .mean())
            rows.append({'model_id': mid, 'model_type': g['model_type'].iloc[0],
                         'half': half, 'mean_var': float(pv)})
    if not rows: return pd.DataFrame()
    wide = (pd.DataFrame(rows)
            .pivot_table(index=['model_id','model_type'], columns='half', values='mean_var')
            .reset_index())
    wide.columns.name = None
    return wide.dropna(subset=['first','second'])


def split_half_lambda(df):
    """Split-half fitted λ per model_id → (first, second)."""
    rows = []
    for mid, g in df.groupby('model_id'):
        trials = sorted(g['trial'].unique())
        mid_t  = len(trials) // 2
        if mid_t < 3: continue
        for half, tset in [('first',trials[:mid_t]),('second',trials[mid_t:])]:
            lam, _ = fit_lambda_mid(g[g['trial'].isin(tset)])
            if np.isfinite(lam):
                rows.append({'model_id': mid, 'model_type': g['model_type'].iloc[0],
                             'half': half, 'lambda_': lam})
    if not rows: return pd.DataFrame()
    wide = (pd.DataFrame(rows)
            .pivot_table(index=['model_id','model_type'], columns='half', values='lambda_')
            .reset_index())
    wide.columns.name = None
    return wide.dropna(subset=['first','second'])


# ── Per-task panels ─────────────────────────────────────────────────────────

def _blank(ax, msg='No data\n(noisy models only)'):
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, color='0.5', style='italic', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_A(ax, df, palette, title='', sub_df=None, sub_label=None):
    """A/H: RMSE vs obs, split by true lambda quartile (Q1=lowest, Q4=highest).
    Linestyle+weight → quartile. Colour → model type."""
    err_df = compute_task_error(df)
    if err_df.empty: _blank(ax, 'No data'); return

    handles, labels = [], []
    for mt in [m for m in ALL_MODELS if m in err_df['model_type'].unique()]:
        color    = palette.get(mt, '0.5')
        df_mt    = df[df.model_type == mt]
        err_mt   = err_df[err_df['model_type'] == mt]
        bins_dict = true_lambda_quartiles(df_mt)
        if len(bins_dict) < 4:
            sns.lineplot(data=err_mt, x='observation', y='err',
                         color=color, lw=1.8, errorbar='ci', ax=ax, legend=False)
            handles.append(Line2D([0],[0], color=color, lw=1.8))
            labels.append(mt); continue

        bins = pd.Series(bins_dict)
        for q in range(4):
            q_mids = bins[bins == q].index.tolist()
            if not q_mids: continue
            q_data = err_mt[err_mt['model_id'].isin(q_mids)]
            if q_data.empty: continue
            sty = LAMBDA_Q_STYLES[q]
            sns.lineplot(data=q_data, x='observation', y='err',
                         color=color, lw=sty['lw'], alpha=sty['alpha'],
                         linestyle=sty['ls'], errorbar='ci', ax=ax, legend=False)
            lbl = f"{mt}: {sty['label']}"
            handles.append(Line2D([0],[0], color=color, lw=sty['lw'],
                                  alpha=sty['alpha'], ls=sty['ls']))
            labels.append(lbl)

    ax.set_xlabel('Observation'); ax.set_ylabel('RMSE vs true mean')
    ax.set_ylim(bottom=0)
    if title: ax.set_title(title, fontsize=8, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_B(ax, df, palette, sub_df=None, sub_label=None):
    """B/I: KDE of per-pid mean response variability for noisy models (NEF only).
    RL_lambda is deterministic so excluded — shows zero and clutters the plot."""
    pdf = df[(df.trial_type=='structured') &
             (df.observation < df.prefix_length)].copy()
    if pdf.empty: _blank(ax); return

    handles, labels = [], []
    any_plotted = False
    for mt in [m for m in ALL_MODELS if m in pdf['model_type'].unique()]:
        color = palette.get(mt, '0.5')
        pv = (pdf[pdf.model_type==mt]
              .groupby(['model_id', 'qid', 'observation'])['response']
              .std().reset_index(name='resp_var'))
        pv_pid = pv.groupby('model_id')['resp_var'].mean().dropna().values
        if len(pv_pid) == 0: continue
        if pv_pid.std() < 1e-9:
            # Deterministic — skip rather than cluttering with a spike at 0
            continue
        sns.kdeplot(pv_pid, ax=ax, color=color, lw=1.8, fill=True, alpha=0.2)
        handles.append(Line2D([0],[0], color=color, lw=1.8))
        labels.append(mt)
        any_plotted = True

    if not any_plotted:
        _blank(ax, 'No noisy models')
        return
    ax.set_xlabel('Mean response variability')
    ax.set_ylabel('Density')
    if handles:
        ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_C(ax, df, palette, sub_df=None, sub_label=None):
    """C/J: |Δresponse| vs obs, split by true lambda quartile (Q1=lowest, Q4=highest).
    Linestyle+weight → quartile. Colour → model type."""
    dlt_all = compute_abs_delta(df)
    if dlt_all.empty: _blank(ax, 'No data'); return

    handles, labels = [], []
    for mt in [m for m in ALL_MODELS if m in dlt_all['model_type'].unique()]:
        color  = palette.get(mt, '0.5')
        df_mt  = df[df.model_type == mt]
        dlt_mt = dlt_all[dlt_all['model_type'] == mt]
        bins_dict = true_lambda_quartiles(df_mt)
        if len(bins_dict) < 4:
            sns.lineplot(data=dlt_mt, x='observation', y='delta',
                         color=color, lw=1.8, errorbar='ci', ax=ax, legend=False)
            handles.append(Line2D([0],[0], color=color, lw=1.8))
            labels.append(mt); continue

        bins = pd.Series(bins_dict)
        for q in range(4):
            q_mids = bins[bins == q].index.tolist()
            if not q_mids: continue
            q_data = dlt_mt[dlt_mt['model_id'].isin(q_mids)]
            if q_data.empty: continue
            sty = LAMBDA_Q_STYLES[q]
            sns.lineplot(data=q_data, x='observation', y='delta',
                         color=color, lw=sty['lw'], alpha=sty['alpha'],
                         linestyle=sty['ls'], errorbar='ci', ax=ax, legend=False)
            lbl = f"{mt}: {sty['label']}"
            handles.append(Line2D([0],[0], color=color, lw=sty['lw'],
                                  alpha=sty['alpha'], ls=sty['ls']))
            labels.append(lbl)

    ax.set_xlabel('Observation'); ax.set_ylabel('Mean |Δresponse|')
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_D(ax, df, palette, sub_df=None, sub_label=None):
    """D/I: Split-half prefix variability reliability."""
    wide = split_half_var(df)
    if wide.empty: _blank(ax); return
    handles, labels = [], []
    var_order = (wide.groupby('model_type')['first'].std()
                 .fillna(0).sort_values().index.tolist())
    any_plotted = False
    for mt in var_order:
        g = wide[wide['model_type']==mt]
        color = palette.get(mt, '0.5')
        std_f = g['first'].std()
        if len(g) >= 3 and std_f > 1e-6:
            sns.regplot(data=g, x='first', y='second', ax=ax,
                        color=color, ci=95, scatter=False, line_kws={'lw': 2.0})
            r, p = pearsonr(g['first'], g['second'])
            handles.append(Line2D([0],[0], color=color, lw=2.0))
            labels.append(f'{mt} r={r:.2f}{pvalue_to_stars(p)}')
            any_plotted = True
        # skip deterministic models (var/λ≈0) — no point to plot
    if not any_plotted: _blank(ax); return
    all_v = pd.concat([wide['first'], wide['second']])
    lo, hi = max(0, all_v.min()-0.005), all_v.max()+0.005
    ax.plot([lo,hi], [lo,hi], color='0.7', lw=0.8, ls='--', zorder=0)
    if sub_df is not None:
        sub_wide = split_half_var(sub_df)
        if not sub_wide.empty:
            for mt, g in sub_wide.groupby('model_type'):
                color = palette.get(mt, '0.5')
                std_f = g['first'].std()
                if len(g) >= 3 and std_f > 1e-6:
                    sns.regplot(data=g, x='first', y='second', ax=ax,
                                color=color, ci=None, scatter=False,
                                line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.7})
                # skip deterministic models in sub overlay
    ax.set_xlabel('Var (1st half)'); ax.set_ylabel('Var (2nd half)')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_E(ax, df, palette, sub_df=None, sub_label=None):
    """E/J: Split-half λ reliability."""
    wide = split_half_lambda(df)
    if wide.empty: _blank(ax); return
    handles, labels = [], []
    var_order = (wide.groupby('model_type')['first'].std()
                 .fillna(0).sort_values().index.tolist())
    any_plotted = False
    for mt in var_order:
        g = wide[wide['model_type']==mt]
        color = palette.get(mt, '0.5')
        std_f = g['first'].std()
        if len(g) >= 3 and std_f > 1e-6:
            sns.regplot(data=g, x='first', y='second', ax=ax,
                        color=color, ci=95, scatter=False, line_kws={'lw': 2.0})
            r, p = pearsonr(g['first'], g['second'])
            handles.append(Line2D([0],[0], color=color, lw=2.0))
            labels.append(f'{mt} r={r:.2f}{pvalue_to_stars(p)}')
            any_plotted = True
        # skip deterministic models (var/λ≈0) — no point to plot
    if not any_plotted: _blank(ax); return
    all_v = pd.concat([wide['first'], wide['second']])
    lo, hi = max(0, all_v.min()-0.05), all_v.max()+0.05
    ax.plot([lo,hi], [lo,hi], color='0.7', lw=0.8, ls='--', zorder=0)
    if sub_df is not None:
        sub_wide = split_half_lambda(sub_df)
        if not sub_wide.empty:
            for mt, g in sub_wide.groupby('model_type'):
                color = palette.get(mt, '0.5')
                std_f = g['first'].std()
                if len(g) >= 3 and std_f > 1e-6:
                    sns.regplot(data=g, x='first', y='second', ax=ax,
                                color=color, ci=None, scatter=False,
                                line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.7})
                # skip deterministic models in sub overlay
    ax.set_xlabel('λ (1st half)'); ax.set_ylabel('λ (2nd half)')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Cross-task panels ───────────────────────────────────────────────────────

def plot_F_late(ax, df, palette, sub_df=None, sub_label=None):
    """F/M: True λ vs late RMSE — scatter + regplot per model type.
    X: true lambda (from params_str). Y: mean RMSE in last 3 obs."""
    err_df = compute_task_error(df)
    if err_df.empty: _blank(ax, 'No data'); return

    n_obs      = df['observation'].max()
    late_start = n_obs - 2  # last 3 obs

    handles, labels = [], []
    for mt in [m for m in ALL_MODELS if m in df['model_type'].unique()]:
        color  = palette.get(mt, '0.5')
        df_mt  = df[df.model_type == mt]

        # Build per-model_id (true_lambda, late_rmse) pairs
        rows = []
        for mid, g in df_mt.groupby('model_id'):
            ps = g['params_str'].iloc[0]
            try:
                lam = float(ps.split('lambda_=')[1].split()[0])
            except (IndexError, ValueError):
                continue
            late_err = (err_df[(err_df.model_id == mid) &
                               (err_df.observation >= late_start)]['err'].mean())
            if np.isfinite(late_err):
                rows.append({'lambda_': lam, 'late_rmse': late_err})
        if len(rows) < 5: continue
        plot_df = pd.DataFrame(rows)

        r, p = pearsonr(plot_df['lambda_'], plot_df['late_rmse'])
        ax.scatter(plot_df['lambda_'], plot_df['late_rmse'],
                   color=color, s=12, alpha=0.5, zorder=3)
        sns.regplot(data=plot_df, x='lambda_', y='late_rmse', scatter=False,
                    color=color, ci=95, ax=ax, line_kws={'lw': 1.8})
        handles.append(Line2D([0],[0], color=color, lw=1.8))
        labels.append(f"{mt}, r={r:.2f}{pvalue_to_stars(p)}")

    if not handles: _blank(ax, 'No data'); return
    ax.set_xlabel('True λ')
    ax.set_ylabel('Late RMSE')
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_G_lambda_recovery(ax, df, palette, sub_df=None, sub_label=None):
    """G/N: Fitted λ vs true λ — parameter recovery scatter + regplot.
    True λ from params_str; fitted λ from |Δresponse| power-law fit.
    Shows how reliably the response data recovers the model parameter."""
    handles, labels = [], []
    any_plotted = False

    for mt in [m for m in ALL_MODELS if m in df['model_type'].unique()]:
        color = palette.get(mt, '0.5')
        params = df[['model_id','params_str']].drop_duplicates().set_index('model_id')

        rows = []
        for mid, g in df[df.model_type==mt].groupby('model_id'):
            lam_fit, p = fit_lambda_mid(g)
            ps = params.loc[mid, 'params_str']
            # Extract true lambda from params_str
            parts = {kv.split('=')[0]: float(kv.split('=')[1])
                     for kv in ps.split() if '=' in kv}
            lam_true = parts.get('lambda_', np.nan)
            if np.isfinite(lam_fit) and np.isfinite(lam_true):
                rows.append({'lam_true': lam_true, 'lam_fit': lam_fit, 'p_fit': p})
        if not rows: continue
        rdf = pd.DataFrame(rows)

        if len(rdf) >= 3:
            # Filled = significant fit (p<0.05), hollow = non-significant
            sig   = rdf[rdf['p_fit'] <  0.05]
            nonsig = rdf[rdf['p_fit'] >= 0.05]
            ax.scatter(sig['lam_true'],    sig['lam_fit'],
                       color=color, s=14, alpha=0.5, zorder=3)
            ax.scatter(nonsig['lam_true'], nonsig['lam_fit'],
                       color=color, s=14, alpha=0.3, zorder=3,
                       facecolors='none', edgecolors=color, linewidths=0.8)
            sns.regplot(data=rdf, x='lam_true', y='lam_fit', ax=ax,
                        color=color, ci=95, scatter=False,
                        line_kws={'lw': 2.0}, label='_nolegend_')
            r, p = pearsonr(rdf['lam_true'], rdf['lam_fit'])
            rmse = np.sqrt(((rdf['lam_true'] - rdf['lam_fit'])**2).mean())
            n_sig = (rdf['p_fit'] < 0.05).sum()
            handles.append(Line2D([0],[0], color=color, lw=2.0))
            labels.append(f'{mt} r={r:.2f}{pvalue_to_stars(p)} ({n_sig}/{len(rdf)} sig)')
            any_plotted = True

    if not any_plotted: _blank(ax, 'No data'); return

    # Identity line
    ax.plot([0,1],[0,1], color='0.7', lw=0.8, ls='--', zorder=0)

    if sub_df is not None:
        sub_params = sub_df[['model_id','params_str']].drop_duplicates().set_index('model_id')
        for mt in [m for m in ALL_MODELS if m in sub_df['model_type'].unique()]:
            color = palette.get(mt, '0.5')
            rows_s = []
            for mid, g in sub_df[sub_df.model_type==mt].groupby('model_id'):
                lam_fit, _ = fit_lambda_mid(g)
                ps = sub_params.loc[mid, 'params_str']
                parts = {kv.split('=')[0]: float(kv.split('=')[1])
                         for kv in ps.split() if '=' in kv}
                lam_true = parts.get('lambda_', np.nan)
                if np.isfinite(lam_fit) and np.isfinite(lam_true):
                    rows_s.append({'lam_true': lam_true, 'lam_fit': lam_fit})
            if len(rows_s) >= 3:
                rdf_s = pd.DataFrame(rows_s)
                sns.regplot(data=rdf_s, x='lam_true', y='lam_fit', ax=ax,
                            color=color, ci=None, scatter=False,
                            line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.7},
                            label='_nolegend_')
        if sub_label and sub_label not in labels:
            handles.append(Line2D([0],[0], color='0.4', lw=1.5, ls='--'))
            labels.append(sub_label)

    ax.set_xlabel('True λ (model param)')
    ax.set_ylabel('Fitted λ (from responses)')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.35)
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_K(ax, df_bin, df_cont, palette, sub_bin=None, sub_cont=None, sub_label=None):
    """K: Cross-task λ correlation — binary vs continuous λ per model_id."""
    handles, labels = [], []
    any_plotted = False
    for mt in [m for m in ALL_MODELS
               if m in df_bin['model_type'].unique()
               and m in df_cont['model_type'].unique()]:
        color = palette.get(mt, '0.5')
        lam_b = {mid: fit_lambda_mid(g)[0]
                 for mid, g in df_bin[df_bin.model_type==mt].groupby('model_id')}
        lam_c = {mid: fit_lambda_mid(g)[0]
                 for mid, g in df_cont[df_cont.model_type==mt].groupby('model_id')}
        mids  = [m for m in lam_b if m in lam_c
                 and np.isfinite(lam_b[m]) and np.isfinite(lam_c[m])]
        if len(mids) < 3: continue
        xv = np.array([lam_b[m] for m in mids])
        yv = np.array([lam_c[m] for m in mids])
        std_x = xv.std()
        if std_x > 1e-6:
            sns.regplot(x=xv, y=yv, ax=ax, color=color, ci=95,
                        scatter=False, line_kws={'lw': 2.0})
            r, p = pearsonr(xv, yv)
            handles.append(Line2D([0],[0], color=color, lw=2.0))
            labels.append(f'{mt} r={r:.2f}{pvalue_to_stars(p)}')
            any_plotted = True
        # skip models with near-zero lambda variance (deterministic)
    if not any_plotted: _blank(ax, 'No cross-task data'); return
    if sub_bin is not None and sub_cont is not None:
        for mt in [m for m in ALL_MODELS
                   if m in sub_bin['model_type'].unique()
                   and m in sub_cont['model_type'].unique()]:
            color = palette.get(mt, '0.5')
            lam_b = {mid: fit_lambda_mid(g)[0]
                     for mid, g in sub_bin[sub_bin.model_type==mt].groupby('model_id')}
            lam_c = {mid: fit_lambda_mid(g)[0]
                     for mid, g in sub_cont[sub_cont.model_type==mt].groupby('model_id')}
            mids = [m for m in lam_b if m in lam_c
                    and np.isfinite(lam_b[m]) and np.isfinite(lam_c[m])]
            if len(mids) >= 3 and np.std([lam_b[m] for m in mids]) > 1e-6:
                xv = np.array([lam_b[m] for m in mids])
                yv = np.array([lam_c[m] for m in mids])
                sns.regplot(x=xv, y=yv, ax=ax, color=color, ci=None,
                            scatter=False, line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.7})
    ax.set_xlabel('λ (binary)'); ax.set_ylabel('λ (continuous)')
    ax.set_title('Cross-task λ correlation', fontsize=8, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_L(ax, df_bin, df_cont, palette, sub_bin=None, sub_cont=None, sub_label=None):
    """L: Cross-task prefix variability correlation."""
    handles, labels = [], []
    any_plotted = False
    for mt in [m for m in ALL_MODELS
               if m in df_bin['model_type'].unique()
               and m in df_cont['model_type'].unique()]:
        color = palette.get(mt, '0.5')
        pv_b = prefix_var_per_mid(df_bin[df_bin.model_type==mt])
        pv_c = prefix_var_per_mid(df_cont[df_cont.model_type==mt])
        mids = [m for m in pv_b.index if m in pv_c.index
                and np.isfinite(pv_b[m]) and np.isfinite(pv_c[m])]
        if len(mids) < 3: continue
        xv = pv_b[mids].values
        yv = pv_c[mids].values
        if xv.std() > 1e-6 and yv.std() > 1e-6:
            sns.regplot(x=xv, y=yv, ax=ax, color=color, ci=95,
                        scatter=False, line_kws={'lw': 2.0})
            r, p = pearsonr(xv, yv)
            handles.append(Line2D([0],[0], color=color, lw=2.0))
            labels.append(f'{mt} r={r:.2f}{pvalue_to_stars(p)}')
            any_plotted = True
        # skip models with near-zero variability (deterministic)
    if not any_plotted: _blank(ax, 'No cross-task data'); return
    if sub_bin is not None and sub_cont is not None:
        for mt in [m for m in ALL_MODELS
                   if m in sub_bin['model_type'].unique()
                   and m in sub_cont['model_type'].unique()]:
            color = palette.get(mt, '0.5')
            pv_b = prefix_var_per_mid(sub_bin[sub_bin.model_type==mt])
            pv_c = prefix_var_per_mid(sub_cont[sub_cont.model_type==mt])
            mids = [m for m in pv_b.index if m in pv_c.index
                    and np.isfinite(pv_b[m]) and np.isfinite(pv_c[m])]
            if len(mids) >= 3:
                xv, yv = pv_b[mids].values, pv_c[mids].values
                if xv.std() > 1e-6 and yv.std() > 1e-6:
                    sns.regplot(x=xv, y=yv, ax=ax, color=color, ci=None,
                                scatter=False,
                                line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.7})
    ax.set_xlabel('Prefix var (binary)'); ax.set_ylabel('Prefix var (continuous)')
    ax.set_title('Cross-task prefix variability', fontsize=8, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_M(ax, df_task, palette, task_label='continuous', sub_df=None, sub_label=None):
    """M/N: λ predicts late-trial error (U-shape strength) for one task."""
    err_df = compute_task_error(df_task)
    if err_df.empty: _blank(ax, 'No data'); return
    handles, labels = [], []
    any_plotted = False
    for mt in [m for m in ALL_MODELS if m in df_task['model_type'].unique()]:
        color = palette.get(mt, '0.5')
        g_err = err_df[err_df['model_type']==mt]
        g_df  = df_task[df_task.model_type==mt]
        u_vals, lam_vals = [], []
        for mid in g_df['model_id'].unique():
            us  = u_strength(g_err[g_err.model_id==mid])
            lam, _ = fit_lambda_mid(g_df[g_df.model_id==mid])
            if np.isfinite(us) and np.isfinite(lam):
                u_vals.append(us); lam_vals.append(lam)
        if len(lam_vals) < 3: continue
        xv, yv = np.array(lam_vals), np.array(u_vals)
        if xv.std() > 1e-6:
            sns.regplot(x=xv, y=yv, ax=ax, color=color, ci=95,
                        scatter=False, line_kws={'lw': 2.0})
            r, p = pearsonr(xv, yv)
            handles.append(Line2D([0],[0], color=color, lw=2.0))
            labels.append(f'{mt} r={r:.2f}{pvalue_to_stars(p)}')
            any_plotted = True
        # skip models with near-zero lambda variance (deterministic)
    if not any_plotted: _blank(ax, 'No data'); return
    if sub_df is not None:
        sub_err = compute_task_error(sub_df)
        if not sub_err.empty:
            for mt in [m for m in ALL_MODELS if m in sub_df['model_type'].unique()]:
                color = palette.get(mt, '0.5')
                g_err = sub_err[sub_err['model_type']==mt]
                g_df  = sub_df[sub_df.model_type==mt]
                u_vals, lam_vals = [], []
                for mid in g_df['model_id'].unique():
                    us  = u_strength(g_err[g_err.model_id==mid])
                    lam, _ = fit_lambda_mid(g_df[g_df.model_id==mid])
                    if np.isfinite(us) and np.isfinite(lam):
                        u_vals.append(us); lam_vals.append(lam)
                if len(lam_vals) >= 3:
                    xv, yv = np.array(lam_vals), np.array(u_vals)
                    if xv.std() > 1e-6:
                        sns.regplot(x=xv, y=yv, ax=ax, color=color, ci=None,
                                    scatter=False,
                                    line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.7})
    ax.set_xlabel(f'λ ({task_label})'); ax.set_ylabel('U-shape strength')
    ax.set_title(f'λ → late error ({task_label})', fontsize=8, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_models', action='store_true',
                        help='Generate responses pkl before plotting')
    parser.add_argument('--tasks', nargs='+', default=['continuous','binary'])
    parser.add_argument('--models', nargs='+', default=['RL_lambda'])
    parser.add_argument('--seq_dir', default='task/sequences')
    parser.add_argument('--alpha_0',    type=float, default=1.0,
                        help='Fixed alpha_0 for RL_lambda scan')
    parser.add_argument('--n_lambdas',  type=int,   default=50,
                        help='Number of lambda_ values in uniform grid 0.01-0.99')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', default=None,
                        help='Path to responses pkl (default: auto)')
    parser.add_argument('--out_pdf', default=None,
                        help='Output PDF path (default: figures/test_sequences_combined.pdf)')
    parser.add_argument('--subsample', action='store_true',
                        help='Overlay a subsampled version (n_trials, n_obs) on each panel')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of trials in subsample overlay')
    parser.add_argument('--n_obs', type=int, default=15,
                        help='Number of observations per trial in subsample overlay')
    parser.add_argument('--sub_seed', type=int, default=42,
                        help='Random seed for subsampling')
    args = parser.parse_args()

    apply_style()
    out_folder = resolve_run_folder('test_sequences')
    pkl = Path(args.data_path) if args.data_path else \
          out_folder / 'test_sequences_responses.pkl'

    if args.run_models or not pkl.exists():
        print("Running models on sequences...")
        run_models_on_sequences(
            seq_dir    = args.seq_dir,
            output_dir = out_folder,
            tasks      = args.tasks,
            models     = args.models,
            alpha_0    = args.alpha_0,
            n_lambdas  = args.n_lambdas,
            seed       = args.seed,
        )

    print(f"Loading {pkl}...")
    df = pd.read_pickle(pkl)
    df = df[df['model_type'].isin(ALL_MODELS)].copy()
    print(f"  {len(df)} rows, models={df['model_type'].unique().tolist()}, "
          f"tasks={df['task'].unique().tolist()}, "
          f"model_ids={df['model_id'].nunique()}")

    # All panels use raw responses — no smoothing applied
    df_bin  = df[df.task=='binary'].copy()
    df_cont = df[df.task=='continuous'].copy()
    raw_bin  = df_bin
    raw_cont = df_cont

    # Subsampled versions (None if flag not set)
    if args.subsample:
        sub_label = f'n={args.n_trials}t×{args.n_obs}obs'

        sub_cache = out_folder / f'sub_{args.n_trials}t_{args.n_obs}obs_s{args.sub_seed}.pkl'
        if sub_cache.exists():
            print(f"  Loading cached subsample: {sub_cache}")
            sub_data = pd.read_pickle(sub_cache)
            sub_raw_bin  = sub_data[sub_data.task=='binary']
            sub_raw_cont = sub_data[sub_data.task=='continuous']
        else:
            sub_raw_bin  = subsample_df(df_bin,  args.n_trials, args.n_obs, args.sub_seed)
            sub_raw_cont = subsample_df(df_cont, args.n_trials, args.n_obs, args.sub_seed)
            sub_data = pd.concat([sub_raw_bin, sub_raw_cont], ignore_index=True)
            sub_data.to_pickle(sub_cache)
            print(f"  Saved subsample cache: {sub_cache}")

        sub_bin  = sub_raw_bin
        sub_cont = sub_raw_cont
        print(f"  Subsample: {len(sub_bin['trial'].unique())} trials × {args.n_obs} obs per task")
    else:
        sub_raw_bin = sub_raw_cont = None
        sub_bin = sub_cont = None
        sub_label = None

    # Palette keyed by model_type — RL_lambda first, NEF second
    model_order = ['RL_lambda', 'NEF'] + [m for m in sorted(df['model_type'].unique()) if m not in ('RL_lambda','NEF')]
    model_order = [m for m in model_order if m in df['model_type'].unique()]
    pal     = get_palette(len(model_order) + 1)
    palette = {m: pal[i] for i, m in enumerate(model_order)}

    # ── Layout: 3 rows × 6 cols
    # Row 1 (A–F): Binary task
    # Row 2 (G–L): Continuous task
    # Row 3 (M–N): Cross-task metrics (2 panels, centred)
    fig = plt.figure(figsize=(FIGURE_SIZE[0] * 1.7, FIGURE_SIZE[1] * 1.5))
    gs  = gridspec.GridSpec(3, 7, figure=fig,
                            hspace=0.55, wspace=0.45,
                            left=0.06, right=0.98,
                            top=0.94, bottom=0.06)

    # Row 1: Binary (A–G)
    ax_bin = [fig.add_subplot(gs[0, c]) for c in range(7)]
    # Row 2: Continuous (H–N)
    ax_cont = [fig.add_subplot(gs[1, c]) for c in range(7)]
    # Row 3: Cross-task (M, N) — span cols 1-2 and 3-4
    ax_K = fig.add_subplot(gs[2, 1:4])   # cross-task λ correlation
    ax_L = fig.add_subplot(gs[2, 4:7])   # cross-task prefix var correlation

    # ── Plot binary row ──────────────────────────────────────────────────────
    plot_A(ax_bin[0], df_bin,  palette, title='Binary',      sub_df=sub_bin,  sub_label=sub_label)
    plot_B(ax_bin[1], raw_bin,  palette,                      sub_df=sub_raw_bin,  sub_label=sub_label)
    plot_C(ax_bin[2], df_bin,  palette,                       sub_df=sub_bin,  sub_label=sub_label)
    plot_D(ax_bin[3], raw_bin,  palette,                      sub_df=sub_raw_bin,  sub_label=sub_label)
    plot_E(ax_bin[4], df_bin,  palette,                       sub_df=sub_bin,  sub_label=sub_label)
    plot_F_late(ax_bin[5], df_bin,  palette, sub_df=sub_bin,  sub_label=sub_label)
    plot_G_lambda_recovery(ax_bin[6], df_bin,  palette, sub_df=sub_bin,  sub_label=sub_label)

    # ── Plot continuous row ──────────────────────────────────────────────────
    plot_A(ax_cont[0], df_cont, palette, title='Continuous', sub_df=sub_cont, sub_label=sub_label)
    plot_B(ax_cont[1], raw_cont, palette,                     sub_df=sub_raw_cont, sub_label=sub_label)
    plot_C(ax_cont[2], df_cont, palette,                      sub_df=sub_cont, sub_label=sub_label)
    plot_D(ax_cont[3], raw_cont, palette,                     sub_df=sub_raw_cont, sub_label=sub_label)
    plot_E(ax_cont[4], df_cont, palette,                      sub_df=sub_cont, sub_label=sub_label)
    plot_F_late(ax_cont[5], df_cont, palette, sub_df=sub_cont, sub_label=sub_label)
    plot_G_lambda_recovery(ax_cont[6], df_cont, palette, sub_df=sub_cont, sub_label=sub_label)

    # ── Plot cross-task row ──────────────────────────────────────────────────
    plot_K(ax_K, df_bin, df_cont, palette, sub_bin=sub_bin,  sub_cont=sub_cont, sub_label=sub_label)
    plot_L(ax_L, raw_bin, raw_cont, palette, sub_bin=sub_raw_bin, sub_cont=sub_raw_cont, sub_label=sub_label)

    # ── Label panels A–M ────────────────────────────────────────────────────
    all_axes = ax_bin + ax_cont + [ax_K, ax_L]
    for i, ax in enumerate(all_axes):
        ax.text(-0.12, 1.05, chr(65+i), transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = Path(args.out_pdf) if args.out_pdf else FIGURES_DIR / 'test_sequences.pdf'
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf)
    print(f"Saved: {out_pdf}")
    print("JOB_COMPLETE")


if __name__ == '__main__':
    main()
