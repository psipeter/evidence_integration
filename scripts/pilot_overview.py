"""
scripts/pilot_overview.py
=========================
Pilot overview: all models × both tasks × 3 panels each.
No parameter scans — single param set per model.

Layout: 2 rows (Binary / Continuous) × 3 cols
  Col A/D: RMSE vs observation
  Col B/E: |Δresponse| vs observation
  Col C/F: KDE of within-prefix response variability (NEF + human only)

Usage:
    python scripts/pilot_overview.py
    python scripts/pilot_overview.py --lambda_ 0.7
    python scripts/pilot_overview.py --n_neurons 100 --n_neurons_counting 200
    python scripts/pilot_overview.py --skip_nef
    python scripts/pilot_overview.py --force_nef
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.math_models import _run_primacy_recency
from utils.paths import FIGURES_DIR, data_path, resolve_run_folder
from utils.plot_style import apply_style, get_palette
from utils.binary_transform import apply_binary_transform, nef_obs_values, nef_response_to_model_scale

PREFIX   = 4          # prefix length in observations (1-indexed: obs 1–4)
HUM_COL  = 'k'
HUM_LW   = 2.5


# ── Scale helpers ─────────────────────────────────────────────────────────────

def _gt(task, tm, tp):
    return float(tm) / 100.0 if task == 'continuous' else float(tp) * 2.0 - 1.0

def _obs_norm(v, task):
    return float(v) / 100.0 if task == 'continuous' else float(v)

def _clip(x, task):
    lo, hi = (0.0, 1.0) if task == 'continuous' else (-1.0, 1.0)
    return float(np.clip(x, lo, hi))


# ── Math models ───────────────────────────────────────────────────────────────

def _run_agents(seq_df, task, alpha_0, lambda_, gamma, eps_p, eps_r):
    dataset = f'task_{task}'
    rows = []
    for tid in sorted(seq_df['trial'].unique()):
        g    = seq_df[seq_df['trial'] == tid].sort_values('observation')
        vals = g['value'].tolist()
        tm   = float(g['true_mean'].iloc[0])
        tp   = float(g['true_p'].iloc[0])
        qid  = int(g['qid'].iloc[0])

        estimates = {
            'Bayes':     0.5 if task == 'continuous' else 0.0,
            'RL_lambda': 0.5 if task == 'continuous' else 0.0,
            'LI':        0.0,
        }
        for n, (_, row) in enumerate(g.iterrows(), 1):
            v = _obs_norm(vals[n-1], task)
            estimates['Bayes']     = _clip(estimates['Bayes']     + (1.0/n)                * (v - estimates['Bayes']),     task)
            estimates['RL_lambda'] = _clip(estimates['RL_lambda'] + (alpha_0/n**lambda_)   * (v - estimates['RL_lambda']), task)
            estimates['LI']        = _clip(gamma * estimates['LI'] + (1-gamma) * v,         task)
            obs_so_far = [_obs_norm(vals[i], task) for i in range(n)]
            pr = _clip(float(_run_primacy_recency(
                {'eps_p': eps_p, 'eps_r': eps_r, 'eta': 0.01},
                np.asarray(obs_so_far), n, 0)), task)
            for name in ('Bayes', 'RL_lambda', 'LI'):
                rows.append({'model_type': name, 'trial': int(tid),
                             'observation': int(row['observation']),
                             'response': estimates[name],
                             'true_mean': tm, 'true_p': tp, 'qid': qid,
                             'prefix_length': PREFIX, 'task': task})
            rows.append({'model_type': 'PR', 'trial': int(tid),
                         'observation': int(row['observation']),
                         'response': pr,
                         'true_mean': tm, 'true_p': tp, 'qid': qid,
                         'prefix_length': PREFIX, 'task': task})
    df = pd.DataFrame(rows)
    return apply_binary_transform(df, dataset)


def run_nef(seq_df, task, alpha_0, lambda_, cache_path,
            n_neurons=None, n_neurons_counting=None):
    if cache_path.exists():
        print(f'[nef] Loading cache: {cache_path}')
        return pd.read_pickle(cache_path)

    from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
    from models.counting_integrator import fast_decode, load_activities
    from fitting.model_params import MODEL_PARAMS
    from tqdm import tqdm

    dataset = f'task_{task}'
    fixed   = MODEL_PARAMS[dataset]['NEF']['fixed']
    params  = {**PARAM_DEFAULTS, **fixed, 'model_type': 'NEF', 'dataset': dataset,
               'pid': 0, 'alpha_0': alpha_0, 'lambda_': lambda_}
    if n_neurons is not None:
        params['n_neurons'] = n_neurons
    if n_neurons_counting is not None:
        params['n_neurons_counting'] = n_neurons_counting
    print(f'[nef] {task}: n_neurons={params["n_neurons"]} '
          f'nc={params["n_neurons_counting"]} α={alpha_0} λ={lambda_}')

    try:
        act_map = load_activities(int(params['n_neurons']),
                                  int(params['n_neurons_counting']), dataset)
    except FileNotFoundError:
        act_map = None

    rows = []
    for tid in tqdm(sorted(seq_df['trial'].unique()), desc=f'NEF {task}'):
        g   = seq_df[seq_df['trial'] == tid].sort_values('observation')
        obs = nef_obs_values(g['value'].to_numpy(dtype=float), dataset)
        p   = {**params, 'seed': int(tid)}
        act = (act_map or {}).get(int(tid))
        dec = fast_decode(act, alpha_0=alpha_0, lambda_=lambda_) \
              if act is not None else _pretrain(p)
        resps = _simulate_trial(obs, p, dec)
        for i, (_, row) in enumerate(g.iterrows()):
            rows.append({'model_type': 'NEF', 'trial': int(tid),
                         'observation': int(row['observation']),
                         'response': nef_response_to_model_scale(float(resps[i]), dataset),
                         'true_mean': float(row['true_mean']),
                         'true_p':   float(row['true_p']),
                         'qid':      int(row['qid']),
                         'prefix_length': PREFIX, 'task': task})

    df = apply_binary_transform(pd.DataFrame(rows), dataset)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f'[nef] Saved: {cache_path}')
    return df


def run_noisy_counting(seq_df, task, mu, sigma_c, nu, cache_path, n_seeds=10):
    """Run NoisyCounting on binary task sequences.
    Binary-only: operates on {-1,+1} observations.
    Stochastic (sigma_c, nu > 0), so run n_seeds times and average.
    Uses carrabin-fitted median params by default.
    NoisyCounting is exempt from Laplace smoothing (self-calibrated).
    """
    if task != 'binary':
        return None
    if cache_path.exists():
        print(f'[nc] Loading cache: {cache_path}')
        return pd.read_pickle(cache_path)

    from tqdm import tqdm
    print(f'[nc] binary: mu={mu} sigma_c={sigma_c} nu={nu} seeds={n_seeds}')

    # Build a trials list in the format math_models.run expects
    from models.math_models import _trial_seed
    rows = []
    for tid in tqdm(sorted(seq_df['trial'].unique()), desc='NoisyCounting'):
        g    = seq_df[seq_df['trial'] == tid].sort_values('observation')
        vals = g['value'].tolist()
        tm   = float(g['true_mean'].iloc[0])
        tp   = float(g['true_p'].iloc[0])
        qid  = int(g['qid'].iloc[0])
        for seed_offset in range(n_seeds):
            rng = np.random.RandomState(_trial_seed(tid + seed_offset * 1000, tid))
            r = 0.0; p_hat = 0.0
            for n, (_, row) in enumerate(g.iterrows()):
                xi      = rng.normal(0.0, sigma_c)
                r       = r + float(vals[n]) * mu + xi
                epsilon = rng.normal(0.0, nu)
                p_hat   = p_hat + (r - p_hat) * float(np.exp(epsilon))
                p_hat   = float(np.clip(p_hat, -1.0, 1.0))
                rows.append({'model_type': 'NoisyCounting', 'trial': int(tid),
                             'observation': int(row['observation']),
                             'response': p_hat, 'seed_offset': seed_offset,
                             'true_mean': tm, 'true_p': tp, 'qid': qid,
                             'prefix_length': PREFIX, 'task': task})

    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Average across seeds for each (trial, obs)
    df = df.groupby(['model_type','trial','observation','true_mean','true_p','qid',
                     'prefix_length','task'])['response'].mean().reset_index()
    # NoisyCounting is exempt from Laplace smoothing
    df['response_raw'] = df['response']
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f'[nc] Saved: {cache_path}')
    return df



def prepare_human(human_pkl, task):
    df = pd.read_pickle(human_pkl)
    t  = df[(df['task'] == task) & ~df['timed_out']].copy()
    if t.empty:
        return None
    # Keep only pids with >= 10 trials
    counts = t.groupby('prolific_pid')['trial'].nunique()
    t = t[t['prolific_pid'].isin(counts[counts >= 10].index)].copy()
    if t.empty:
        return None

    # Convert slider [0,100] to model scale
    if task == 'binary':
        t['response'] = t['response'] / 100.0 * 2.0 - 1.0
    else:
        t['response'] = t['response'] / 100.0

    # Human observations are 0-indexed — shift to 1-indexed to match models
    t['observation'] = t['observation'] + 1

    t['model_type']    = 'Human'
    t['prefix_length'] = PREFIX
    t['task']          = task

    # Do NOT apply Laplace to human responses — they are real responses,
    # not model outputs. Laplace shrinks early-obs deltas differentially
    # and would artificially flatten the human delta curve.
    return t


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse_by_obs(df, task):
    rows = []
    for (mt, tid), g in df.groupby(['model_type', 'trial']):
        g  = g.sort_values('observation')
        gt = _gt(task, g['true_mean'].iloc[0], g['true_p'].iloc[0])
        for _, r in g.iterrows():
            rows.append({'model_type': mt, 'observation': int(r['observation']),
                         'err': abs(r['response'] - gt)})
    return pd.DataFrame(rows).groupby(['model_type','observation'])['err'].mean().reset_index() \
           if rows else pd.DataFrame()


def delta_by_obs(df):
    rows = []
    for (mt, tid), g in df.groupby(['model_type', 'trial']):
        g = g.sort_values('observation').copy()
        g['delta'] = g['response'].diff().abs()
        rows.append(g[['model_type','observation','delta']])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).dropna(subset=['delta']) \
             .groupby(['model_type','observation'])['delta'].mean().reset_index()


def prefix_stds(df):
    """Within-qid response std during prefix, one value per (qid, obs)."""
    pre = df[df['observation'] <= PREFIX].copy()
    if pre.empty:
        return pd.Series(dtype=float, name='std')
    return (pre.groupby(['qid','observation'])['response']
               .std()
               .dropna()
               .rename('std'))


def add_resid(df: pd.DataFrame) -> pd.DataFrame:
    """Add resid = response - mean(response | qid, observation)."""
    means = (df.groupby(['qid', 'observation'])['response']
               .mean().reset_index().rename(columns={'response': 'qid_mean'}))
    df2 = df.merge(means, on=['qid', 'observation'], how='left')
    df2['resid'] = df2['response'] - df2['qid_mean']
    return df2


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _prefix_line(ax):
    ax.axvline(PREFIX + 0.5, color='0.65', lw=0.9, ls='--', alpha=0.7)


def plot_rmse(ax, model_df, human_df, task, palette, title):
    err = rmse_by_obs(model_df, task)
    handles, labels = [], []
    for mt in err['model_type'].unique():
        c = palette.get(mt, '0.5')
        d = err[err.model_type == mt].sort_values('observation')
        ax.plot(d['observation'], d['err'], color=c, lw=1.8)
        handles.append(Line2D([0],[0], color=c, lw=1.8)); labels.append(mt)
    if human_df is not None:
        herr = rmse_by_obs(human_df, task)
        if not herr.empty:
            d = herr.sort_values('observation')
            ax.plot(d['observation'], d['err'],
                    color=HUM_COL, lw=HUM_LW, zorder=5)
            handles.append(Line2D([0],[0], color=HUM_COL, lw=HUM_LW)); labels.append('Human')
    _prefix_line(ax)
    ax.set_xlabel('Observation'); ax.set_ylabel('RMSE')
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_delta(ax, model_df, human_df, task, palette, title):
    dlt = delta_by_obs(model_df)
    handles, labels = [], []
    for mt in dlt['model_type'].unique():
        c = palette.get(mt, '0.5')
        d = dlt[dlt.model_type == mt].sort_values('observation')
        ax.plot(d['observation'], d['delta'], color=c, lw=1.8)
        handles.append(Line2D([0],[0], color=c, lw=1.8)); labels.append(mt)
    if human_df is not None:
        hdlt = delta_by_obs(human_df)
        if not hdlt.empty:
            d = hdlt.sort_values('observation')
            ax.plot(d['observation'], d['delta'],
                    color=HUM_COL, lw=HUM_LW, zorder=5)
            handles.append(Line2D([0],[0], color=HUM_COL, lw=HUM_LW)); labels.append('Human')
    _prefix_line(ax)
    ax.set_xlabel('Observation'); ax.set_ylabel('Mean |Δresponse|')
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_prefix_kde(ax, nef_df, human_df, palette, title, x_max=None):
    """KDE of within-prefix response variability (stochastic models + human)."""
    handles, labels = [], []
    all_vals = []

    sources = []
    if nef_df is not None and not nef_df.empty:
        for mt in nef_df['model_type'].unique():
            stds = prefix_stds(nef_df[nef_df['model_type'] == mt])
            if len(stds) >= 2:
                sources.append((mt, stds.values))
                all_vals.extend(stds.values)
    if human_df is not None:
        stds = prefix_stds(human_df)
        if len(stds) >= 2:
            sources.append(('Human', stds.values))
            all_vals.extend(stds.values)

    if not sources:
        ax.text(0.5, 0.5, 'Insufficient data\n(need ≥2 qid repeats)',
                ha='center', va='center', transform=ax.transAxes,
                color='0.5', style='italic', fontsize=8)
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_title(title, fontsize=9, fontweight='bold')
        return

    if x_max is None:
        x_max = np.quantile(all_vals, 0.99) * 1.15
    x = np.linspace(0, x_max, 300)

    for name, vals in sources:
        c   = palette.get(name, HUM_COL)
        lw  = HUM_LW if name == 'Human' else 1.8
        try:
            kde = gaussian_kde(vals, bw_method='scott')
            y   = kde(x)
            y   = y / y.max()       # normalise to peak = 1
            ax.plot(x, y, color=c, lw=lw)
            ax.fill_between(x, y, color=c, alpha=0.12)
        except Exception:
            pass
        handles.append(Line2D([0],[0], color=c, lw=lw)); labels.append(name)

    ax.set_xlabel('Within-prefix response std')
    ax.set_ylabel('Density (normalised)')
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


def plot_autocorr(ax, nef_df, human_df, palette, title, max_lag=4):
    """Within-trial residual autocorrelation vs lag (analogous to carrabin T4).

    resid = response - mean(response | qid, observation)

    For a deterministic model all repeats of the same qid give identical
    responses, so resid = 0 and autocorrelation is undefined. This panel
    is therefore only plotted for stochastic models (NEF, NoisyCounting) and human data.

    With only one participant we pool residuals across all trials rather
    than averaging per-pid correlations as in carrabin.
    """
    lags = list(range(1, max_lag + 1))
    handles, labels = [], []

    sources = []
    if nef_df is not None and not nef_df.empty:
        if 'qid' in nef_df.columns and not nef_df['qid'].isna().all():
            for mt in nef_df['model_type'].unique():
                sub = nef_df[nef_df['model_type'] == mt]
                sources.append((mt, add_resid(sub.copy()),
                                palette.get(mt, '0.5'), 1.8))
    if human_df is not None and not human_df.empty:
        if 'qid' in human_df.columns and not human_df['qid'].isna().all():
            sources.append(('Human', add_resid(human_df.copy()),
                             HUM_COL, HUM_LW))

    if not sources:
        ax.text(0.5, 0.5, 'NEF + Human only\n(deterministic models: resid = 0)',
                ha='center', va='center', transform=ax.transAxes,
                color='0.5', style='italic', fontsize=8)
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_title(title, fontsize=9, fontweight='bold')
        return

    for mt, df, color, lw in sources:
        lag_rs = []
        for lag in lags:
            pairs = []
            for (qid, obs_i), grp in df.groupby(['qid', 'observation']):
                # Partner observations: same qid, observation = obs_i + lag
                partner = df[(df['qid'] == qid) &
                             (df['observation'] == obs_i + lag)]['resid'].values
                r_i = grp['resid'].values
                # Pair up by trial order within qid
                n = min(len(r_i), len(partner))
                if n >= 2:
                    pairs.extend(zip(r_i[:n], partner[:n]))
            if len(pairs) < 3:
                lag_rs.append(np.nan)
                continue
            arr = np.array(pairs)
            if arr[:, 0].std() < 1e-10 or arr[:, 1].std() < 1e-10:
                lag_rs.append(np.nan)
                continue
            rv, _ = pearsonr(arr[:, 0], arr[:, 1])
            lag_rs.append(rv)
        if all(np.isnan(v) for v in lag_rs):
            continue
        ax.plot(lags, lag_rs, 'o-', color=color, lw=lw, ms=5)
        handles.append(Line2D([0],[0], color=color, lw=lw)); labels.append(mt)

    ax.axhline(0, color='0.7', lw=0.8, ls='--')
    ax.set_xlabel('Lag (observations)')
    ax.set_ylabel('Autocorrelation of residuals')
    ax.set_xticks(lags)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(handles, labels, fontsize=6, frameon=True, framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alpha_0',             type=float, default=0.3)
    p.add_argument('--lambda_',             type=float, default=0.5)
    p.add_argument('--gamma',               type=float, default=0.9)
    p.add_argument('--eps_p',               type=float, default=0.1)
    p.add_argument('--eps_r',               type=float, default=0.9)
    p.add_argument('--nc_mu',               type=float, default=0.164)
    p.add_argument('--nc_sigma_c',          type=float, default=0.001)
    p.add_argument('--nc_nu',               type=float, default=0.058)
    p.add_argument('--nc_seeds',            type=int,   default=10)
    p.add_argument('--skip_nc',             action='store_true')
    p.add_argument('--n_neurons',           type=int,   default=100)
    p.add_argument('--n_neurons_counting',  type=int,   default=200)
    p.add_argument('--seq_dir',             default='task/sequences')
    p.add_argument('--human_pkl',           default='data/task_results.pkl')
    p.add_argument('--skip_nef',            action='store_true')
    p.add_argument('--force_nef',           action='store_true')
    p.add_argument('--out',                 default=None)
    args = p.parse_args()

    apply_style()
    a0, lam     = args.alpha_0, args.lambda_
    seq_dir     = Path(args.seq_dir)
    nef_run_dir = resolve_run_folder('pilot_nef')

    MODEL_NAMES = ['Bayes', 'RL_lambda', 'LI', 'PR', 'NEF', 'NoisyCounting']
    colors  = get_palette(len(MODEL_NAMES))
    palette = dict(zip(MODEL_NAMES, colors))
    palette['Human'] = HUM_COL

    DISPLAY = {
        'Bayes':     'Bayes',
        'RL_lambda': f'RL_λ (α={a0},λ={lam})',
        'LI':        f'LI (γ={args.gamma})',
        'PR':        f'PR (εp={args.eps_p},εr={args.eps_r})',
        'NEF':       f'NEF (α={a0},λ={lam})',
    }
    disp_palette = {DISPLAY.get(k, k): v for k, v in palette.items()}

    human_pkl = Path(args.human_pkl)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Per-task data collection ──────────────────────────────────────────────
    task_data   = {}   # stores all dfs per task for second-pass plotting
    kde_vals_all = []  # accumulate prefix stds for shared x_max

    for task in ['binary', 'continuous']:
        seq_df = pd.read_pickle(seq_dir / f'{task}_sequences.pkl')

        # Math models
        math_df = _run_agents(seq_df, task, a0, lam,
                               args.gamma, args.eps_p, args.eps_r)
        math_df['model_type'] = math_df['model_type'].map(
            lambda x: DISPLAY.get(x, x))

        # NEF
        nef_raw = None
        nef_df  = None
        if not args.skip_nef:
            n_suf = (f'_n{args.n_neurons}' if args.n_neurons else '') +                     (f'_nc{args.n_neurons_counting}' if args.n_neurons_counting else '')
            cache         = nef_run_dir / f'pilot_nef_{task}_a{a0:.4f}_l{lam:.4f}{n_suf}.pkl'
            inspect_cache = FIGURES_DIR / f'inspect_nef_a{a0:.4f}_l{lam:.4f}{n_suf}.pkl'
            if args.force_nef:
                cache.unlink(missing_ok=True)
            elif not cache.exists() and inspect_cache.exists():
                print(f'[nef] Using inspect cache: {inspect_cache}')
                payload  = pd.read_pickle(inspect_cache)
                resp     = payload['responses'][task].copy()
                seq_meta = seq_df[['trial','observation','true_mean','true_p','qid']].drop_duplicates()
                resp = resp.merge(seq_meta, on=['trial','observation'], how='left')
                resp['model_type']    = 'NEF'
                resp['task']          = task
                resp['prefix_length'] = PREFIX
                resp.to_pickle(cache)
            nef_raw = run_nef(seq_df, task, a0, lam, cache,
                              args.n_neurons, args.n_neurons_counting)
            nef_df = nef_raw.copy()
            nef_df['model_type'] = DISPLAY['NEF']

        # NoisyCounting — binary only
        nc_raw = None
        nc_df  = None
        if task == 'binary' and not args.skip_nc:
            nc_suf   = f'_mu{args.nc_mu:.3f}_sc{args.nc_sigma_c:.4f}_nu{args.nc_nu:.3f}'
            nc_cache = nef_run_dir / f'pilot_nc_binary{nc_suf}.pkl'
            nc_raw   = run_noisy_counting(seq_df, task, args.nc_mu,
                                          args.nc_sigma_c, args.nc_nu,
                                          nc_cache, args.nc_seeds)
            if nc_raw is not None:
                nc_df = nc_raw.copy()
                nc_df['model_type'] = 'NoisyCounting'

        human_df = prepare_human(human_pkl, task) if human_pkl.exists() else None

        # Restrict models to the trials present in human data (if available)
        if human_df is not None:
            human_trials = set(human_df['trial'].unique())
            math_df  = math_df[math_df['trial'].isin(human_trials)]
            if nef_df  is not None: nef_df  = nef_df[nef_df['trial'].isin(human_trials)]
            if nef_raw is not None: nef_raw = nef_raw[nef_raw['trial'].isin(human_trials)]
            if nc_df   is not None: nc_df   = nc_df[nc_df['trial'].isin(human_trials)]
            if nc_raw  is not None: nc_raw  = nc_raw[nc_raw['trial'].isin(human_trials)]

        # Combine all models for RMSE / delta panels
        all_df = pd.concat(
            [math_df]
            + ([nef_df] if nef_df is not None else [])
            + ([nc_df]  if nc_df  is not None else []),
            ignore_index=True)

        # Stochastic models for KDE / autocorr panels
        stoch_df = pd.concat(
            [d for d in [nef_raw, nc_raw] if d is not None],
            ignore_index=True) if any(d is not None for d in [nef_raw, nc_raw]) else None

        # Collect prefix stds for shared x_max
        for _df in [nef_raw, nc_raw, human_df]:
            if _df is not None and not _df.empty:
                _stds = prefix_stds(_df)
                if len(_stds) >= 2:
                    kde_vals_all.extend(_stds.values.tolist())

        label = 'Binary' if task == 'binary' else 'Continuous'
        task_data[task] = dict(seq_df=seq_df, all_df=all_df, stoch_df=stoch_df,
                               human_df=human_df, label=label, disp_palette=disp_palette)

    # Shared x_max for KDE panels
    shared_x_max = (np.quantile(kde_vals_all, 0.99) * 1.15 if kde_vals_all else None)

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)

    for row, task in enumerate(['binary', 'continuous']):
        d = task_data[task]

        plot_rmse(axes[row,0], d['all_df'], d['human_df'], task, d['disp_palette'],
                  f'{d["label"]} — RMSE vs obs')
        plot_delta(axes[row,1], d['all_df'], d['human_df'], task, d['disp_palette'],
                   f'{d["label"]} — |Δresponse| vs obs')
        plot_prefix_kde(axes[row,2], d['stoch_df'], d['human_df'], palette,
                        f'{d["label"]} — Prefix variability (KDE)',
                        x_max=shared_x_max)
        plot_autocorr(axes[row,3], d['stoch_df'], d['human_df'], palette,
                      f'{d["label"]} — Residual autocorrelation')

    for i, ax in enumerate(axes.flat):
        ax.text(-0.10, 1.05, chr(65+i), transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

    fig.suptitle(
        f'Pilot overview  |  RL/NEF α={a0} λ={lam}  '
        f'LI γ={args.gamma}  PR εp={args.eps_p} εr={args.eps_r}',
        fontsize=10, fontweight='bold')

    out = Path(args.out) if args.out else FIGURES_DIR / 'pilot_overview.pdf'
    plt.savefig(out)
    print(f'Saved: {out}')
    print('JOB_COMPLETE')


if __name__ == '__main__':
    main()
