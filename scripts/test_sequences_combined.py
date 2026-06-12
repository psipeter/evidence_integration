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

N_GROUP = 10
SMOOTH  = 3
ALL_MODELS = ['RL_lambda']


def subsample_df(df, n_trials, n_obs, seed=42):
    """Randomly subsample n_trials trials and truncate to first n_obs observations."""
    rng = np.random.default_rng(seed)
    all_trials = df['trial'].unique()
    nt = min(n_trials, len(all_trials))
    chosen = rng.choice(all_trials, size=nt, replace=False)
    return df[(df['trial'].isin(chosen)) & (df['observation'] < n_obs)].copy()


# ── Metric helpers ─────────────────────────────────────────────────────────

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
    """A/F: RMSE vs obs — mean and CI across model_ids (pids)."""
    err_df = compute_task_error(df)
    if err_df.empty: _blank(ax, 'No data'); return
    handles, labels = [], []
    for mt in [m for m in ALL_MODELS if m in err_df['model_type'].unique()]:
        color = palette.get(mt, '0.5')
        g = err_df[err_df['model_type']==mt]
        sns.lineplot(data=g, x='observation', y='err',
                     color=color, lw=1.8, errorbar='ci', ax=ax, legend=False)
        handles.append(Line2D([0],[0], color=color, lw=1.8)); labels.append(mt)
    if sub_df is not None:
        sub_err = compute_task_error(sub_df)
        if not sub_err.empty:
            for mt in [m for m in ALL_MODELS if m in sub_err['model_type'].unique()]:
                color = palette.get(mt, '0.5')
                g = sub_err[sub_err['model_type']==mt]
                sns.lineplot(data=g, x='observation', y='err',
                             color=color, lw=1.5, errorbar='ci', ax=ax,
                             legend=False, linestyle='--', alpha=0.7)
            if sub_label:
                handles.append(Line2D([0],[0], color='0.4', lw=1.5, ls='--'))
                labels.append(sub_label)
    ax.set_xlabel('Observation'); ax.set_ylabel('RMSE vs true mean')
    ax.set_ylim(bottom=0)
    if title: ax.set_title(title, fontsize=8, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_B(ax, df, palette, sub_df=None, sub_label=None):
    """B/G: KDE distribution of per-pid mean response variability
    (std across qid repeats within prefix region), like carrabin figure A.
    Zero for deterministic models — shown as spike at 0 (placeholder)."""
    pdf = df[(df.trial_type=='structured') &
             (df.observation < df.prefix_length)].copy()
    if pdf.empty: _blank(ax); return
    # Per-pid variability: for each (model_id, qid, obs) group, std across trials
    pv = (pdf.groupby(['model_id', 'qid', 'observation'])['response']
           .std().reset_index(name='resp_var'))
    # Mean per-pid: average over qid and obs
    pv_pid = pv.groupby('model_id')['resp_var'].mean().reset_index()
    if pv_pid.empty: _blank(ax); return

    vals = pv_pid['resp_var'].dropna().values
    if vals.std() < 1e-9:
        # All zero — deterministic model, show annotation
        ax.axvline(0, color=palette.get('RL_lambda', '0.5'), lw=2)
        ax.text(0.5, 0.6, 'zero (deterministic)', transform=ax.transAxes,
                ha='center', va='center', fontsize=7, color='0.5', style='italic')
    else:
        color = palette.get('RL_lambda', '0.5')
        sns.kdeplot(vals, ax=ax, color=color, lw=1.8, fill=True, alpha=0.2)
    ax.set_xlabel('Mean response variability')
    ax.set_ylabel('Density')
    sns.despine(ax=ax, top=True, right=True)


def plot_C(ax, df, palette, sub_df=None, sub_label=None):
    """C/H: Mean |Δresponse| vs obs."""
    dlt = compute_abs_delta(df)
    if dlt.empty: _blank(ax, 'No data'); return
    handles, labels = [], []
    for mt in [m for m in ALL_MODELS if m in dlt['model_type'].unique()]:
        color = palette.get(mt, '0.5')
        g = dlt[dlt['model_type']==mt]
        sns.lineplot(data=g, x='observation', y='delta',
                     color=color, lw=1.8, errorbar='ci', ax=ax, legend=False)
        handles.append(Line2D([0],[0], color=color, lw=1.8)); labels.append(mt)
    if sub_df is not None:
        sub_dlt = compute_abs_delta(sub_df)
        if not sub_dlt.empty:
            for mt in [m for m in ALL_MODELS if m in sub_dlt['model_type'].unique()]:
                color = palette.get(mt, '0.5')
                g = sub_dlt[sub_dlt['model_type']==mt]
                sns.lineplot(data=g, x='observation', y='delta',
                             color=color, lw=1.5, errorbar='ci', ax=ax,
                             legend=False, linestyle='--', alpha=0.7)
            if sub_label and sub_label not in labels:
                handles.append(Line2D([0],[0], color='0.4', lw=1.5, ls='--'))
                labels.append(sub_label)
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
        elif len(g) >= 1:
            ax.scatter([g['first'].mean()], [g['second'].mean()],
                       color=color, s=60, zorder=5, marker='D',
                       edgecolors='white', lw=0.5)
            handles.append(Line2D([0],[0], color=color, lw=0, marker='D',
                                  ms=6, markeredgecolor='white'))
            labels.append(f'{mt} (var≈0)')
            any_plotted = True
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
                else:
                    ax.scatter([g['first'].mean()], [g['second'].mean()],
                               color=color, s=30, marker='D', alpha=0.5,
                               edgecolors='white', lw=0.5)
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
        elif len(g) >= 1:
            ax.scatter([g['first'].mean()], [g['second'].mean()],
                       color=color, s=60, zorder=5, marker='D',
                       edgecolors='white', lw=0.5)
            handles.append(Line2D([0],[0], color=color, lw=0, marker='D',
                                  ms=6, markeredgecolor='white'))
            labels.append(f'{mt} (λ≈0)')
            any_plotted = True
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
                else:
                    ax.scatter([g['first'].mean()], [g['second'].mean()],
                               color=color, s=30, marker='D', alpha=0.5,
                               edgecolors='white', lw=0.5)
    ax.set_xlabel('λ (1st half)'); ax.set_ylabel('λ (2nd half)')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Cross-task panels ───────────────────────────────────────────────────────

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
        else:
            ax.scatter([xv.mean()], [yv.mean()], color=color, s=60, zorder=5,
                       marker='D', edgecolors='white', lw=0.5)
            handles.append(Line2D([0],[0], color=color, lw=0, marker='D',
                                  ms=6, markeredgecolor='white'))
            labels.append(f'{mt} (λ≈0)')
            any_plotted = True
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
        else:
            ax.scatter([xv.mean()], [yv.mean()], color=color, s=60, zorder=5,
                       marker='D', edgecolors='white', lw=0.5)
            handles.append(Line2D([0],[0], color=color, lw=0, marker='D',
                                  ms=6, markeredgecolor='white'))
            labels.append(f'{mt} (var≈0)')
            any_plotted = True
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
        else:
            ax.scatter([xv.mean()], [yv.mean()], color=color, s=60, zorder=5,
                       marker='D', edgecolors='white', lw=0.5)
            handles.append(Line2D([0],[0], color=color, lw=0, marker='D',
                                  ms=6, markeredgecolor='white'))
            labels.append(f'{mt} (λ≈0)')
            any_plotted = True
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
    parser.add_argument('--data_path', default=None,
                        help='Path to responses pkl (default: auto)')
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
    print(f"Loading {pkl}...")
    df = pd.read_pickle(pkl)
    df = df[df['model_type'] == 'RL_lambda'].copy()
    print(f"  {len(df)} rows after filtering to RL_lambda, "
          f"tasks={df['task'].unique().tolist()}, "
          f"model_ids={df['model_id'].nunique()}")

    df_bin  = df[df.task=='binary'].copy()
    df_cont = df[df.task=='continuous'].copy()

    # Subsampled versions (None if flag not set)
    if args.subsample:
        sub_bin  = subsample_df(df_bin,  args.n_trials, args.n_obs, args.sub_seed)
        sub_cont = subsample_df(df_cont, args.n_trials, args.n_obs, args.sub_seed)
        sub_label = f'n={args.n_trials}t×{args.n_obs}obs'
        print(f"  Subsample: {len(sub_bin['trial'].unique())} trials × {args.n_obs} obs per task")
    else:
        sub_bin = sub_cont = None
        sub_label = None

    # Palette keyed by model_type
    models  = sorted(df['model_type'].unique().tolist())
    pal     = get_palette(len(models) + 1)
    palette = {m: pal[i] for i, m in enumerate(models)}

    # ── Layout: 3 rows × 5 cols
    # Row 1: Binary (A–E)
    # Row 2: Continuous (F–J)
    # Row 3: Cross-task K (λ corr), L (var corr), M (λ→late err binary),
    #                  N (λ→late err continuous), O (spare / future)
    fig = plt.figure(figsize=(FIGURE_SIZE[0] * 1.3, FIGURE_SIZE[1] * 1.5))
    gs  = gridspec.GridSpec(3, 5, figure=fig,
                            hspace=0.55, wspace=0.4,
                            left=0.06, right=0.98,
                            top=0.94, bottom=0.06)

    # Row 1: Binary (A–E)
    ax_bin = [fig.add_subplot(gs[0, c]) for c in range(5)]
    # Row 2: Continuous (F–J)
    ax_cont = [fig.add_subplot(gs[1, c]) for c in range(5)]
    # Row 3: Cross-task panels
    ax_K   = fig.add_subplot(gs[2, 0])   # cross-task λ correlation
    ax_L   = fig.add_subplot(gs[2, 1])   # cross-task prefix var correlation
    ax_Mb  = fig.add_subplot(gs[2, 2])   # λ → late error, binary
    ax_Mc  = fig.add_subplot(gs[2, 3])   # λ → late error, continuous
    ax_O   = fig.add_subplot(gs[2, 4])   # spare

    # ── Plot binary row ──────────────────────────────────────────────────────
    plot_A(ax_bin[0], df_bin,  palette, title='Binary',      sub_df=sub_bin,  sub_label=sub_label)
    plot_B(ax_bin[1], df_bin,  palette,                       sub_df=sub_bin,  sub_label=sub_label)
    plot_C(ax_bin[2], df_bin,  palette,                       sub_df=sub_bin,  sub_label=sub_label)
    plot_D(ax_bin[3], df_bin,  palette,                       sub_df=sub_bin,  sub_label=sub_label)
    plot_E(ax_bin[4], df_bin,  palette,                       sub_df=sub_bin,  sub_label=sub_label)

    # ── Plot continuous row ──────────────────────────────────────────────────
    plot_A(ax_cont[0], df_cont, palette, title='Continuous', sub_df=sub_cont, sub_label=sub_label)
    plot_B(ax_cont[1], df_cont, palette,                      sub_df=sub_cont, sub_label=sub_label)
    plot_C(ax_cont[2], df_cont, palette,                      sub_df=sub_cont, sub_label=sub_label)
    plot_D(ax_cont[3], df_cont, palette,                      sub_df=sub_cont, sub_label=sub_label)
    plot_E(ax_cont[4], df_cont, palette,                      sub_df=sub_cont, sub_label=sub_label)

    # ── Plot cross-task row ──────────────────────────────────────────────────
    plot_K(ax_K,  df_bin, df_cont, palette, sub_bin=sub_bin,  sub_cont=sub_cont, sub_label=sub_label)
    plot_L(ax_L,  df_bin, df_cont, palette, sub_bin=sub_bin,  sub_cont=sub_cont, sub_label=sub_label)
    plot_M(ax_Mb, df_bin,          palette, task_label='binary',      sub_df=sub_bin,  sub_label=sub_label)
    plot_M(ax_Mc, df_cont,         palette, task_label='continuous',  sub_df=sub_cont, sub_label=sub_label)
    _blank(ax_O, 'Future panel')

    # ── Label panels A–M ────────────────────────────────────────────────────
    all_axes = ax_bin + ax_cont + [ax_K, ax_L, ax_Mb, ax_Mc, ax_O]
    for i, ax in enumerate(all_axes):
        ax.text(-0.12, 1.05, chr(65+i), transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES_DIR / 'test_sequences_combined.pdf'
    plt.savefig(out_pdf)
    print(f"Saved: {out_pdf}")
    print("JOB_COMPLETE")


if __name__ == '__main__':
    main()
