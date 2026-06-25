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
from scipy.stats import gaussian_kde

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


def plot_prefix_kde(ax, nef_df, human_df, palette, title):
    """KDE of within-prefix response variability (NEF + human only)."""
    handles, labels = [], []
    all_vals = []

    sources = []
    if nef_df is not None and not nef_df.empty:
        stds = prefix_stds(nef_df)
        if len(stds) >= 2:
            sources.append(('NEF', stds.values))
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

    x_max = np.quantile(all_vals, 0.99) * 1.15
    x     = np.linspace(0, x_max, 300)

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alpha_0',             type=float, default=1.0)
    p.add_argument('--lambda_',             type=float, default=0.5)
    p.add_argument('--gamma',               type=float, default=0.9)
    p.add_argument('--eps_p',               type=float, default=0.5)
    p.add_argument('--eps_r',               type=float, default=0.5)
    p.add_argument('--n_neurons',           type=int,   default=None)
    p.add_argument('--n_neurons_counting',  type=int,   default=None)
    p.add_argument('--seq_dir',             default='task/sequences')
    p.add_argument('--human_pkl',           default='data/task_results.pkl')
    p.add_argument('--skip_nef',            action='store_true')
    p.add_argument('--force_nef',           action='store_true')
    p.add_argument('--out',                 default=None)
    args = p.parse_args()

    apply_style()
    a0, lam    = args.alpha_0, args.lambda_
    seq_dir    = Path(args.seq_dir)
    nef_run_dir = resolve_run_folder('pilot_nef')

    MODEL_NAMES = ['Bayes', 'RL_lambda', 'LI', 'PR', 'NEF']
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
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    for row, task in enumerate(['binary', 'continuous']):
        seq_df = pd.read_pickle(seq_dir / f'{task}_sequences.pkl')

        # Math models
        math_df = _run_agents(seq_df, task, a0, lam,
                               args.gamma, args.eps_p, args.eps_r)
        math_df['model_type'] = math_df['model_type'].map(lambda x: DISPLAY.get(x, x))

        # NEF
        nef_df = None
        if not args.skip_nef:
            n_suf = (f'_n{args.n_neurons}'          if args.n_neurons          else '') + \
                    (f'_nc{args.n_neurons_counting}' if args.n_neurons_counting else '')
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

        all_df = pd.concat(
            [math_df] + ([nef_df] if nef_df is not None else []),
            ignore_index=True)

        human_df = prepare_human(human_pkl, task) if human_pkl.exists() else None

        label = 'Binary' if task == 'binary' else 'Continuous'

        plot_rmse(axes[row,0], all_df, human_df, task, disp_palette,
                  f'{label} — RMSE vs obs')
        plot_delta(axes[row,1], all_df, human_df, task, disp_palette,
                   f'{label} — |Δresponse| vs obs')
        # For KDE: use raw NEF (un-display-renamed) so model_type='NEF'
        plot_prefix_kde(axes[row,2],
                        nef_raw if nef_df is not None else None,
                        human_df, palette,
                        f'{label} — Prefix variability (KDE)')

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
