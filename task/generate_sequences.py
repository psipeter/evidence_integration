"""
generate_sequences.py
=====================
Generate stimulus sequences for the continuous and/or binary evidence
integration tasks.  Optionally runs a seed search (--n_tries > 1) to find
the seed producing the best-shaped decay curves.

All trials are structured (shared prefix + randomised tail across repeats).
There are no random trials.

Trial structure
---------------
  Each trial has `seq_length` observations. The first `prefix_length`
  observations are fixed and shared across all `n_repeats` repeats of a
  given unique sequence (qid). The tail is freshly randomised each repeat.
  prefix_length == seq_length → entire sequence is identical across repeats.

Generative distributions
------------------------
  continuous : Normal(true_mean, std_fixed)
               true_mean ~ Uniform(mean_range)
               values are integers clipped to [value_min, value_max]
  binary     : Bernoulli(true_p), true_p ~ Uniform(p_range)
               values in {-1, +1}

Seed search (--n_tries > 1)
----------------------------
  Scores each seed on two criteria:
    1. RMSE-decay quality (panel A): fits A/t^b to the Bayesian agent RMSE
       curve, penalises deviations and early-obs rises.
    2. Delta-decay quality (panel C): fits A/t to mean |Δresponse|, penalises
       deviations.
  The seed with the lowest combined score is saved. Diagnostic PDFs written
  to {output_dir}/diagnostics/ every 50 attempts.

Output (single master copy in task/sequences/)
----------------------------------------------
  {task}_sequences.json  — loaded by jsPsych via task/src/{task}/config.js
  {task}_sequences.pkl   — loaded by NEF.py and simulation scripts

Usage
-----
  # Single seed (fast)
  python task/generate_sequences.py --task both --seed 42

  # Seed search
  python task/generate_sequences.py --task both --n_tries 500 \\
      --n_unique_sequences 8 --n_repeats 5 --seq_length 15 \\
      --prefix_length 4 --mean_range -60 60 --std_fixed 20 \\
      --p_range 0.2 0.8
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALUE_MIN = -100
VALUE_MAX = 100
MAX_REJECTION_ATTEMPTS = 10_000


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------
def continuous_param_grid(n: int, mean_range, std_fixed, rng):
    """n (true_mean, true_std) pairs; true_mean tiled evenly, std fixed."""
    edges = np.linspace(mean_range[0], mean_range[1], n + 1)
    means = [round(rng.uniform(edges[i], edges[i + 1]), 4) for i in range(n)]
    params = [(mu, std_fixed) for mu in means]
    rng.shuffle(params)
    return params


def binary_param_grid(n: int, p_range, rng):
    """n true_p values tiled evenly across p_range."""
    edges = np.linspace(p_range[0], p_range[1], n + 1)
    params = [round(rng.uniform(edges[i], edges[i + 1]), 4) for i in range(n)]
    rng.shuffle(params)
    return params


# ---------------------------------------------------------------------------
# Observation generators
# ---------------------------------------------------------------------------
def draw_continuous_obs(rng, true_mean, true_std, n,
                        value_min=VALUE_MIN, value_max=VALUE_MAX):
    vals, attempts = [], 0
    while len(vals) < n:
        attempts += 1
        v = int(np.round(rng.normal(true_mean, true_std)))
        if value_min <= v <= value_max:
            vals.append(v)
        elif attempts > MAX_REJECTION_ATTEMPTS * n:
            vals.append(int(np.clip(np.round(rng.normal(true_mean, true_std)),
                                    value_min, value_max)))
    return vals


def draw_binary_obs(rng, true_p, n):
    return [1 if rng.random() < true_p else -1 for _ in range(n)]


# ---------------------------------------------------------------------------
# Rejection sampling (binary only)
# ---------------------------------------------------------------------------
def check_prefix_plausibility_binary(prefix, true_p, threshold=1e-4):
    if not prefix:
        return True
    n_blue = sum(v == 1 for v in prefix)
    n_red  = len(prefix) - n_blue
    log_p  = n_blue * math.log(true_p + 1e-12) + n_red * math.log(1 - true_p + 1e-12)
    return log_p >= math.log(threshold + 1e-12)


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------
def build_sequence(rng, task, true_mean, true_std, true_p,
                   seq_length, prefix_length):
    """Returns (sequence, prefix). Binary uses prefix plausibility check."""
    full_repeat = (prefix_length == 0 or prefix_length >= seq_length)
    eff_pfx     = seq_length if full_repeat else prefix_length

    for _ in range(MAX_REJECTION_ATTEMPTS):
        if task == 'continuous':
            prefix = draw_continuous_obs(rng, true_mean, true_std, eff_pfx)
        else:
            prefix = draw_binary_obs(rng, true_p, eff_pfx)

        if task == 'binary' and not check_prefix_plausibility_binary(prefix, true_p):
            continue

        if full_repeat:
            return prefix, prefix

        tail_len = seq_length - eff_pfx
        tail = (draw_continuous_obs(rng, true_mean, true_std, tail_len)
                if task == 'continuous'
                else draw_binary_obs(rng, true_p, tail_len))
        return prefix + tail, prefix

    raise RuntimeError(
        f"Could not generate valid sequence after {MAX_REJECTION_ATTEMPTS} "
        f"attempts (task={task}, true_mean={true_mean}, true_p={true_p})."
    )


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def generate_task_sequences(task, args, rng):
    """Generate all trials for one task. Returns (DataFrame, json_trials)."""
    seq_length    = args.seq_length
    prefix_length = args.prefix_length if not args.full_repeat else 0
    full_repeat   = args.full_repeat or (prefix_length >= seq_length)
    eff_pfx       = seq_length if full_repeat else prefix_length

    n_unique = args.n_unique_sequences
    n_repeats = args.n_repeats
    n_total   = n_unique * n_repeats

    print(f"\n{'='*60}")
    print(f"Task: {task.upper()}")
    print(f"  Total trials       : {n_total} ({n_unique} seqs × {n_repeats} reps)")
    print(f"  Sequence length    : {seq_length}")
    print(f"  Prefix length      : {'full' if full_repeat else eff_pfx}")

    # ── Parameter sets ───────────────────────────────────────────────────────
    if task == 'continuous':
        param_sets = continuous_param_grid(n_unique, args.mean_range, args.std_fixed, rng)
        print(f"  Std fixed          : {args.std_fixed}")
        print(f"  Mean range         : {args.mean_range}")
    else:
        param_sets = binary_param_grid(n_unique, args.p_range, rng)
        print(f"  true_p range       : {args.p_range}")

    # ── Build templates (one per unique sequence / qid) ──────────────────────
    templates = []
    for qid, ps in enumerate(param_sets):
        if task == 'continuous':
            true_mean, true_std = ps
            true_p = float('nan')
        else:
            true_p    = ps
            true_mean = true_p
            true_std  = float('nan')

        sequence, prefix = build_sequence(
            rng, task, true_mean, true_std, true_p, seq_length, prefix_length)
        sc = float(true_std) if task == 'continuous' and not math.isnan(true_std) else float('nan')
        templates.append({
            'qid': qid, 'true_mean': true_mean, 'true_std': true_std,
            'std_condition': sc, 'true_p': true_p,
            'prefix': prefix, 'sequence': sequence,
        })
        pfx_str = ','.join(map(str, prefix[:5])) + ('...' if len(prefix) > 5 else '')
        label   = f'mean={true_mean:.1f} std={true_std:.1f}' if task == 'continuous' else f'p={true_p:.3f}'
        print(f"  qid {qid:3d}: params={label}  prefix=[{pfx_str}]")

    # ── Expand to trials (n_repeats per template) ────────────────────────────
    trials = []
    for tmpl in templates:
        for _ in range(n_repeats):
            if full_repeat:
                values = tmpl['sequence']
            else:
                tail_len = seq_length - eff_pfx
                tail = (draw_continuous_obs(rng, tmpl['true_mean'], tmpl['true_std'], tail_len)
                        if task == 'continuous'
                        else draw_binary_obs(rng, tmpl['true_p'], tail_len))
                values = tmpl['prefix'] + tail
            trials.append({**tmpl, 'values': values})

    rng.shuffle(trials)

    # ── Build DataFrame + JSON ───────────────────────────────────────────────
    records, json_trials = [], []
    for t, trial in enumerate(trials):
        sc = trial['std_condition']
        for o, v in enumerate(trial['values'], start=1):
            records.append({
                'task':          task,
                'trial':         t,
                'qid':           trial['qid'],
                'observation':   o,
                'value':         v,
                'true_mean':     trial['true_mean'],
                'true_std':      trial['true_std'],
                'std_condition': sc,
                'true_p':        trial['true_p'],
                'prefix_length': eff_pfx,
            })
        json_trials.append({
            'trial':         t,
            'qid':           trial['qid'],
            'true_mean':     None if math.isnan(trial['true_mean']) else trial['true_mean'],
            'true_std':      None if math.isnan(trial['true_std'])  else trial['true_std'],
            'true_p':        None if math.isnan(trial['true_p'])    else trial['true_p'],
            'std_condition': None if (sc != sc) else sc,
            'values':        trial['values'],
            'prefix_length': eff_pfx,
        })

    df = pd.DataFrame(records)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert len(df) == n_total * seq_length
    if task == 'continuous':
        assert df['value'].between(VALUE_MIN, VALUE_MAX).all()
    else:
        assert df['value'].isin([-1, 1]).all()
    assert df['trial'].nunique() == n_total
    qid_counts = df.groupby('qid')['trial'].nunique()
    assert (qid_counts == n_repeats).all()

    print(f"\n  ✓ {len(df)} rows | {n_total} trials | {n_unique} unique seqs × {n_repeats} reps")
    if task == 'continuous':
        print(f"  Value range  : {df['value'].min()} – {df['value'].max()}")
        print(f"  true_mean    : {df['true_mean'].min():.1f} – {df['true_mean'].max():.1f}")
    else:
        print(f"  true_p range : {df['true_p'].min():.3f} – {df['true_p'].max():.3f}")

    return df, json_trials


# ---------------------------------------------------------------------------
# Seed-search scoring
# ---------------------------------------------------------------------------
def _bayesian_continuous(values_raw):
    resps, running = [], 0.0
    for t, v in enumerate(values_raw, 1):
        running += (v / 100.0 - running) / t
        resps.append(running * (t + 1) / (t + 3))
    return resps


def _bayesian_binary(values):
    resps, n_pos = [], 0
    for t, v in enumerate(values, 1):
        n_pos += (1 if v == 1 else 0)
        resps.append((n_pos + 1) / (t + 2) * 2 - 1)
    return resps


def _power_law_A(t, A):
    return A / np.asarray(t, float)


def score_sequences(seq_df, task):
    rmse_by_obs, delta_curves = {}, []
    for trial_id in seq_df['trial'].unique():
        tdf    = seq_df[seq_df['trial'] == trial_id].sort_values('observation')
        values = tdf['value'].tolist()
        if task == 'continuous':
            gt    = tdf['true_mean'].iloc[0] / 100.0
            resps = _bayesian_continuous(values)
        else:
            gt    = tdf['true_p'].iloc[0] * 2 - 1
            resps = _bayesian_binary(values)
        for obs, r in zip(tdf['observation'].tolist(), resps):
            rmse_by_obs.setdefault(obs, []).append(abs(r - gt))
        delta_curves.append([abs(resps[i] - resps[i-1]) for i in range(1, len(resps))])

    obs_sorted = sorted(rmse_by_obs)
    rmse_curve = np.array([np.mean(rmse_by_obs[o]) for o in obs_sorted])
    t_vals     = np.arange(1, len(rmse_curve) + 1, dtype=float)

    try:
        popt, _ = curve_fit(lambda t, A, b: A / t**b, t_vals, rmse_curve,
                             p0=[rmse_curve[0], 0.5], bounds=([0, 0.01], [np.inf, 2.0]))
        score_shape = float(np.sqrt(np.mean((rmse_curve - popt[0] / t_vals**popt[1])**2)))
    except Exception:
        score_shape = float(np.std(np.diff(rmse_curve)))

    early_rise = float(np.sum(np.maximum(np.diff(rmse_curve[:min(5, len(rmse_curve))]), 0)))

    mean_delta = np.mean(delta_curves, axis=0)
    t_d        = np.arange(2, len(mean_delta) + 2, dtype=float)
    try:
        popt2, _ = curve_fit(_power_law_A, t_d, mean_delta,
                              p0=[mean_delta[0]], bounds=([0], [np.inf]))
        score_delta = float(np.sqrt(np.mean((mean_delta - _power_law_A(t_d, popt2[0]))**2)))
    except Exception:
        score_delta = float(np.std(mean_delta))

    return score_shape, score_delta, score_shape + early_rise + score_delta


def _plot_diagnostic(seq_df, task, attempt, out_dir, score):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    fig.suptitle(f"{task} | attempt={attempt} | score={score:.5f}", fontsize=9)

    rmse_by_obs, delta_curves = {}, []
    for trial_id in seq_df['trial'].unique():
        tdf    = seq_df[seq_df['trial'] == trial_id].sort_values('observation')
        values = tdf['value'].tolist()
        gt     = tdf['true_mean'].iloc[0] / 100.0 if task == 'continuous' else tdf['true_p'].iloc[0] * 2 - 1
        resps  = _bayesian_continuous(values) if task == 'continuous' else _bayesian_binary(values)
        for obs, r in zip(tdf['observation'].tolist(), resps):
            rmse_by_obs.setdefault(obs, []).append(abs(r - gt))
        delta_curves.append([abs(resps[i] - resps[i-1]) for i in range(1, len(resps))])

    obs_sorted = sorted(rmse_by_obs)
    rmse_curve = np.array([np.mean(rmse_by_obs[o]) for o in obs_sorted])
    mean_delta = np.mean(delta_curves, axis=0)
    t_d        = np.arange(2, len(mean_delta) + 2, dtype=float)

    axes[0].plot(obs_sorted, rmse_curve, 'b-o', ms=3, lw=1.5)
    axes[0].set_title('RMSE vs true target'); axes[0].set_xlabel('obs')
    axes[1].plot(obs_sorted, rmse_curve, 'b-o', ms=3, lw=1.5, label='data')
    try:
        t_v = np.array(obs_sorted, float)
        popt, _ = curve_fit(lambda t, A, b: A / t**b, t_v, rmse_curve,
                             p0=[rmse_curve[0], 0.5], bounds=([0, 0.01], [np.inf, 2.0]))
        axes[1].plot(obs_sorted, popt[0] / t_v**popt[1], 'r--', lw=1.5, label=f'A/t^{popt[1]:.2f}')
    except Exception:
        pass
    axes[1].set_title('RMSE + power-law fit'); axes[1].set_xlabel('obs'); axes[1].legend(fontsize=7)
    axes[2].plot(t_d, mean_delta, 'g-o', ms=3, lw=1.5, label='data')
    try:
        popt2, _ = curve_fit(_power_law_A, t_d, mean_delta, p0=[mean_delta[0]], bounds=([0], [np.inf]))
        axes[2].plot(t_d, _power_law_A(t_d, popt2[0]), 'r--', lw=1.5, label='A/t fit')
    except Exception:
        pass
    axes[2].set_title('Mean |Δresponse|'); axes[2].set_xlabel('obs'); axes[2].legend(fontsize=7)
    if task == 'binary':
        axes[3].hist(seq_df[seq_df.observation == 1]['true_p'].values, bins=10, color='purple', alpha=0.7)
        axes[3].set_title('true_p distribution'); axes[3].set_xlabel('true_p')
    else:
        axes[3].hist(seq_df[seq_df.observation == 1]['true_mean'].values, bins=10, color='teal', alpha=0.7)
        axes[3].set_title('true_mean distribution'); axes[3].set_xlabel('true_mean')

    diag_dir = pathlib.Path(out_dir) / 'diagnostics'
    diag_dir.mkdir(exist_ok=True)
    plt.savefig(diag_dir / f'{task}_attempt{attempt:04d}.pdf')
    plt.close(fig)


def _save_sequences(df, json_trials, task, out_dir):
    out_dir = pathlib.Path(out_dir)
    json_path = out_dir / f'{task}_sequences.json'
    pkl_path  = out_dir / f'{task}_sequences.pkl'
    with open(json_path, 'w') as f:
        json.dump(json_trials, f, indent=2)
    df.to_pickle(pkl_path)
    return pkl_path, json_path


def _seed_search(task, args, out_dir):
    print(f"\n{'='*55}\nSeed search: {task.upper()} ({args.n_tries} tries)")
    best_score, best_scores, best_seed, best_df = np.inf, (np.inf, np.inf), None, None
    all_scores = []
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    try:
        for attempt in range(args.n_tries):
            task_seed = attempt if task == 'continuous' else attempt + 1000
            rng = make_rng(task_seed)
            try:
                df, json_trials = generate_task_sequences(task, args, rng)
            except Exception:
                continue
            r_shape, r_delta, combined = score_sequences(df, task)
            all_scores.append(combined)
            if combined < best_score:
                best_score, best_scores, best_seed, best_df = combined, (r_shape, r_delta), attempt, df.copy()
                _save_sequences(df, json_trials, task, out_dir)
            if (attempt + 1) % 50 == 0:
                if best_df is not None:
                    _plot_diagnostic(best_df, task, attempt + 1, out_dir, best_score)
                print(f"  [{attempt+1:4d}/{args.n_tries}]  best seed={best_seed}  combined={best_score:.5f}  "
                      f"(shape={best_scores[0]:.5f}  delta={best_scores[1]:.5f})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    arr = np.array(all_scores)
    print(f"\nResults ({args.n_tries} attempts):")
    print(f"  Best seed        : {best_seed}")
    print(f"  Best combined    : {best_score:.5f}")
    print(f"  Best rmse_shape  : {best_scores[0]:.5f}")
    print(f"  Best delta_score : {best_scores[1]:.5f}")
    print(f"  Median combined  : {np.median(arr):.5f}")
    print(f"  Best percentile  : {np.mean(arr >= best_score)*100:.1f}th")
    print(f"  Saved to         : {out_dir}/{task}_sequences.{{pkl,json}}")
    if best_df is not None:
        _plot_diagnostic(best_df, task, args.n_tries, out_dir, best_score)
        print(f"  Diagnostics      : {out_dir}/diagnostics/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Generate (and optionally seed-search) task sequences.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--task',               choices=['continuous', 'binary', 'both'], default='both')
    p.add_argument('--n_tries',            type=int,   default=1,
                   help='Seeds to try; best saved. 1 = single run with --seed.')
    p.add_argument('--n_unique_sequences', type=int,   default=8)
    p.add_argument('--n_repeats',          type=int,   default=5)
    p.add_argument('--seq_length',         type=int,   default=15)
    p.add_argument('--prefix_length',      type=int,   default=4)
    p.add_argument('--full_repeat',        action='store_true')
    p.add_argument('--mean_range',         type=float, nargs=2, default=[-60.0, 60.0], metavar=('LO', 'HI'))
    p.add_argument('--std_fixed',          type=float, default=20.0)
    p.add_argument('--p_range',            type=float, nargs=2, default=[0.2, 0.8],   metavar=('LO', 'HI'))
    p.add_argument('--output_dir',         default='task/sequences')
    p.add_argument('--seed',               type=int,   default=42,
                   help='Seed for single generation (ignored when n_tries > 1)')
    p.add_argument('--overwrite',          action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n_unique_sequences > 0
    assert args.n_repeats > 0
    assert args.seq_length > 0
    assert 0 <= args.prefix_length <= args.seq_length
    assert args.mean_range[0] < args.mean_range[1]
    assert args.std_fixed > 0
    assert 0 < args.p_range[0] < args.p_range[1] < 1
    assert args.n_tries >= 1

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = ['continuous', 'binary'] if args.task == 'both' else [args.task]

    if args.n_tries > 1:
        print(f"Seed search | {args.n_tries} tries per task")
        for task in tasks:
            _seed_search(task, args, out_dir)
    else:
        print(f"Generating sequences | seed={args.seed}")
        all_dfs = []
        for task in tasks:
            task_seed = args.seed if task == 'continuous' else args.seed + 1000
            rng = make_rng(task_seed)
            df, json_trials = generate_task_sequences(task, args, rng)
            all_dfs.append(df)
            pkl_path, json_path = _save_sequences(df, json_trials, task, out_dir)
            print(f"\n  Saved: {json_path}\n  Saved: {pkl_path}")
        if len(all_dfs) == 2:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_pickle(out_dir / 'all_sequences.pkl')
            print(f"\n  Saved combined: {out_dir / 'all_sequences.pkl'}")

    print("\nJOB_COMPLETE")


if __name__ == '__main__':
    main()
