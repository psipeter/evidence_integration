"""
scripts/inspect_sequences.py
=============================
Diagnostic figure: 2 agents (Bayesian mean + RL_lambda) × 4 panels.

  A  Binary     — RMSE vs observation
  B  Binary     — |Δresponse| vs observation
  C  Continuous — RMSE vs observation
  D  Continuous — |Δresponse| vs observation

Usage:
    python scripts/inspect_sequences.py
    python scripts/inspect_sequences.py --alpha_0 1.0 --rl_lambda 0.5
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES_DIR


def _bayes_responses(values, task, tm, tp):
    resps, running = [], 0.5
    if task == 'continuous':
        for n, v in enumerate(values, 1):
            running += (v / 100.0 - running) / n
            resps.append(float(np.clip(running, 0.0, 1.0)))
    else:
        n_pos = 0
        for n, v in enumerate(values, 1):
            n_pos += (1 if v == 1 else 0)
            resps.append((n_pos + 1) / (n + 2))
    return resps


def _rl_responses(values, task, tm, tp, alpha_0=1.0, lambda_=0.5):
    resps, running = [], 0.5
    for n, v in enumerate(values, 1):
        alpha = alpha_0 / (n ** lambda_)
        obs_n = v / 100.0 if task == 'continuous' else (1.0 if v == 1 else 0.0)
        running = float(np.clip(running + alpha * (obs_n - running), 0.0, 1.0))
        resps.append(running)
    return resps


def run_agents(seq_df, task, alpha_0, rl_lambda):
    agents = {
        'Bayes': lambda vals, tm, tp: _bayes_responses(vals, task, tm, tp),
        f'RL(α={alpha_0},λ={rl_lambda})': lambda vals, tm, tp: _rl_responses(vals, task, tm, tp, alpha_0, rl_lambda),
    }
    results = {}
    for name, fn in agents.items():
        rows = []
        for tid in seq_df['trial'].unique():
            g = seq_df[seq_df['trial'] == tid].sort_values('observation')
            vals = g['value'].tolist()
            tm = float(g['true_mean'].iloc[0])
            tp = float(g['true_p'].iloc[0]) if task == 'binary' else float('nan')
            gt = tm / 100.0 if task == 'continuous' else tp
            resp = fn(vals, tm, tp)
            prev = None
            for obs, r in zip(g['observation'].tolist(), resp):
                rows.append({'observation': int(obs), 'err': abs(r - gt),
                             'delta': abs(r - prev) if prev is not None else np.nan})
                prev = r
        results[name] = pd.DataFrame(rows)
    return results


def _plot_panel(ax, agent_data, metric, title, ylabel):
    colors = ['#2563eb', '#dc2626']
    for (name, df), color in zip(agent_data.items(), colors):
        curve = df.dropna(subset=[metric]).groupby('observation')[metric].mean()
        ax.plot(curve.index, curve.values, color=color, lw=1.8, label=name)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Observation', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(left=1); ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def make_figure(seq_dir, alpha_0, rl_lambda, out_path):
    seq_dir = Path(seq_dir)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for row, task in enumerate(['binary', 'continuous']):
        pkl = seq_dir / f'{task}_sequences.pkl'
        if not pkl.exists():
            print(f"[skip] {pkl}"); continue
        seq_df = pd.read_pickle(pkl)
        agent_data = run_agents(seq_df, task, alpha_0, rl_lambda)
        prefix = int(seq_df['prefix_length'].iloc[0])
        label = task.capitalize()
        _plot_panel(axes[row,0], agent_data, 'err',
                    f'{label} — RMSE (prefix={prefix})', 'RMSE vs true param')
        _plot_panel(axes[row,1], agent_data, 'delta',
                    f'{label} — |Δresponse| (prefix={prefix})', 'Mean |Δresponse|')
        for ax in axes[row]:
            ax.axvline(prefix + 0.5, color='#999', lw=0.8, ls='--', alpha=0.6)
    fig.suptitle(f'Sequence diagnostics  |  RL α={alpha_0} λ={rl_lambda}',
                 fontsize=10, fontweight='bold')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq_dir',   default='task/sequences')
    p.add_argument('--alpha_0',   type=float, default=1.0)
    p.add_argument('--rl_lambda', type=float, default=0.5)
    p.add_argument('--out',       default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out  = Path(args.out) if args.out else FIGURES_DIR / 'inspect_sequences.pdf'
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_figure(args.seq_dir, args.alpha_0, args.rl_lambda, out)
    print("JOB_COMPLETE")
