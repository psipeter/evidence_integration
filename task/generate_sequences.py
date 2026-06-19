"""
generate_sequences.py
=====================
Generate stimulus sequences for the continuous and/or binary evidence
integration tasks.

Trial structure
---------------
  Each trial has `seq_length` observations split into two blocks:

  Prefix block (obs 1..prefix_length):
    Shared across all n_repeats repeats of a given qid.
    Generated JOINTLY across all n_unique sequences, one position at a
    time.  Each position is redrawn until the aggregate Bayesian RMSE
    decreases — guaranteeing smooth convergence in the early curve by
    construction, cheaply (median ~5–8 draws total).

  Suffix block (obs prefix_length+1..seq_length):
    Generated JOINTLY across all n_unique sequences, one position at a
    time, independently for each of the n_repeats repeats.  Same
    position-wise rejection as the prefix: each position is redrawn
    until the aggregate Bayesian RMSE does not rise.  The completed
    prefix+suffix must also pass the per-sequence plausibility check
    (k≤1) for every qid.  Suffix varies across repeats (different
    random draws each time), providing the within-qid variability
    needed for PTN/response-variability analysis.

Mirrored sequences
------------------
  n_unique must be even.  The first n_unique/2 sequences are base
  sequences drawn from the lower half of mean_range / p_range.  The
  second half are exact mirrors (v → 100−v for continuous, sign-flip
  for binary).  Mirror true_mean = 100 − base_true_mean; mirror
  true_p = 1 − base_true_p.  Mirroring doubles sequence count without
  extra parameter draws, and ensures the aggregate parameter distribution
  is symmetric around the midpoint.

Per-sequence plausibility (k=1.0)
----------------------------------
  Applied to: (a) the prefix alone, (b) the full prefix+suffix sequence.
  Continuous: |sample_mean − true_mean| ≤ 1.0 × std/√n
  Binary:     |fraction_blue − true_p|  ≤ 1.0 × √(p(1−p)/n)

Generative distributions
------------------------
  continuous : Normal(true_mean, std_fixed), clipped to [0, 100]
  binary     : Bernoulli(true_p), values in {−1, +1}

Usage
-----
  python task/generate_sequences.py --task both --seed 42
  python task/generate_sequences.py --task both --seed 42 \\
      --n_unique_sequences 10 --n_repeats 4 --prefix_length 3
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
VALUE_MIN = 0
VALUE_MAX = 100
MAX_REJECTION_ATTEMPTS = 10_000


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Parameter grids  (base sequences only — lower half of range)
# ---------------------------------------------------------------------------
def continuous_param_grid(n: int, mean_range, std_fixed, rng):
    lo, hi = mean_range[0], mean_range[1]
    mid    = (lo + hi) / 2.0
    edges  = np.linspace(lo, mid, n + 1)
    means  = [round(rng.uniform(edges[i], edges[i + 1]), 4) for i in range(n)]
    params = [(mu, std_fixed) for mu in means]
    rng.shuffle(params)
    return params


def binary_param_grid(n: int, p_range, rng):
    lo, hi = p_range[0], p_range[1]
    mid    = (lo + hi) / 2.0
    edges  = np.linspace(lo, mid, n + 1)
    params = [round(rng.uniform(edges[i], edges[i + 1]), 4) for i in range(n)]
    rng.shuffle(params)
    return params


# ---------------------------------------------------------------------------
# Mirror helpers
# ---------------------------------------------------------------------------
def mirror_sequence(sequence, task):
    return [100 - v for v in sequence] if task == 'continuous' else [-v for v in sequence]


def mirror_params(true_mean, true_std, true_p, task):
    if task == 'continuous':
        return 100.0 - true_mean, true_std, float('nan')
    else:
        return true_p, float('nan'), round(1.0 - true_p, 4)


# ---------------------------------------------------------------------------
# Observation samplers
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
            vals.append(int(np.clip(
                np.round(rng.normal(true_mean, true_std)), value_min, value_max)))
    return vals


def draw_binary_obs(rng, true_p, n):
    return [1 if rng.random() < true_p else -1 for _ in range(n)]


# ---------------------------------------------------------------------------
# Bayesian agent helpers
# ---------------------------------------------------------------------------
def _bayesian_errors(sequence, task, true_mean, true_p):
    """Per-observation |posterior − true parameter| for a Bayesian agent."""
    errs = []
    if task == 'continuous':
        gt, running = true_mean / 100.0, 0.5
        for n, v in enumerate(sequence, 1):
            running += (v / 100.0 - running) / n
            running  = float(np.clip(running, 0.0, 1.0))
            errs.append(abs(running - gt))
    else:
        gt, n_blue = true_p, 0
        for n, v in enumerate(sequence, 1):
            n_blue += (1 if v == 1 else 0)
            errs.append(abs((n_blue + 1) / (n + 2) - gt))
    return errs


# ---------------------------------------------------------------------------
# Per-sequence plausibility check
# ---------------------------------------------------------------------------
def check_sequence_plausibility(sequence, task, true_mean, true_std, true_p,
                                k: float = 1.0) -> bool:
    """True iff sample statistics are within k × SE of the true parameter."""
    n = len(sequence)
    if n == 0:
        return True
    if task == 'continuous':
        return abs(np.mean(sequence) - true_mean) <= k * true_std / np.sqrt(n)
    else:
        return abs(sum(v == 1 for v in sequence) / n - true_p) \
               <= k * np.sqrt(true_p * (1 - true_p) / n)


# ---------------------------------------------------------------------------
# PREFIX BLOCK  — joint generation across all qids
# ---------------------------------------------------------------------------
def build_prefix_block(rng, task, param_list, prefix_length):
    """Generate prefixes for all base qids jointly, one position at a time.

    At each position p, draw one observation per qid and accept the full
    position only if the aggregate Bayesian RMSE across all qids does not
    rise from position p−1 to p.  Redraw the entire position if it rises.

    This guarantees monotone aggregate RMSE across the prefix by
    construction.  Also rejects any prefix whose per-sequence plausibility
    (k=1) fails at the end.

    Returns: list of prefixes (one per entry in param_list).
    """
    n_qids = len(param_list)

    for _outer in range(MAX_REJECTION_ATTEMPTS):
        prefixes   = [[] for _ in range(n_qids)]
        prev_agg_err = float('inf')
        success = True

        for pos in range(prefix_length):
            for _pos_attempt in range(MAX_REJECTION_ATTEMPTS):
                # Draw one observation per qid at this position
                if task == 'continuous':
                    new_obs = [draw_continuous_obs(rng, tm, ts, 1)[0]
                               for tm, ts, _ in param_list]
                else:
                    new_obs = [draw_binary_obs(rng, tp, 1)[0]
                               for _, _, tp in param_list]

                # Compute aggregate Bayesian error after adding these obs
                agg_err = np.mean([
                    _bayesian_errors(prefixes[q] + [new_obs[q]], task,
                                     param_list[q][0], param_list[q][2])[-1]
                    for q in range(n_qids)
                ])

                if agg_err <= prev_agg_err:
                    for q in range(n_qids):
                        prefixes[q].append(new_obs[q])
                    prev_agg_err = agg_err
                    break
            else:
                # Couldn't find a non-rising position — restart from scratch
                success = False
                break

        if not success:
            continue

        # Per-sequence plausibility check on the completed prefixes
        all_ok = all(
            check_sequence_plausibility(
                prefixes[q], task, param_list[q][0], param_list[q][1], param_list[q][2])
            for q in range(n_qids)
        )
        if all_ok:
            return prefixes

    raise RuntimeError(
        f"build_prefix_block failed after {MAX_REJECTION_ATTEMPTS} attempts "
        f"(task={task})."
    )


# ---------------------------------------------------------------------------
# SUFFIX BLOCK  — joint across all qids, one position at a time, per repeat
# ---------------------------------------------------------------------------
def build_suffix_block(rng, task, all_templates, suffix_length):
    """Generate one suffix per qid for a single repeat, jointly.

    Mirrors build_prefix_block: sample one observation per qid at each
    suffix position, rejecting the position if the aggregate Bayesian RMSE
    (computed over the full prefix+suffix-so-far) rises.  This guarantees
    monotone aggregate RMSE throughout the suffix as well as the prefix.

    Additionally, the completed full sequence (prefix+suffix) must pass
    the per-sequence plausibility check (k=1) for every qid.  If not,
    the entire suffix block is redrawn.

    Returns: list of suffixes (one per template in all_templates).
    """
    n_qids = len(all_templates)

    for _outer in range(MAX_REJECTION_ATTEMPTS):
        suffixes    = [[] for _ in range(n_qids)]
        # Start error from end of prefix for each qid
        prev_errors = [
            _bayesian_errors(t['prefix'], task, t['true_mean'], t['true_p'])[-1]
            if t['prefix'] else float('inf')
            for t in all_templates
        ]
        prev_agg = float(np.mean(prev_errors))
        success  = True

        for pos in range(suffix_length):
            for _pos_attempt in range(MAX_REJECTION_ATTEMPTS):
                if task == 'continuous':
                    new_obs = [draw_continuous_obs(
                                   rng, t['true_mean'], t['true_std'], 1)[0]
                               for t in all_templates]
                else:
                    new_obs = [draw_binary_obs(rng, t['true_p'], 1)[0]
                               for t in all_templates]

                # Aggregate error after adding this position
                agg_err = float(np.mean([
                    _bayesian_errors(
                        all_templates[q]['prefix'] + suffixes[q] + [new_obs[q]],
                        task,
                        all_templates[q]['true_mean'],
                        all_templates[q]['true_p']
                    )[-1]
                    for q in range(n_qids)
                ]))

                if agg_err <= prev_agg:
                    for q in range(n_qids):
                        suffixes[q].append(new_obs[q])
                    prev_agg = agg_err
                    break
            else:
                success = False
                break

        if not success:
            continue

        # Per-sequence plausibility check on prefix+suffix for every qid
        all_ok = all(
            check_sequence_plausibility(
                all_templates[q]['prefix'] + suffixes[q],
                task,
                all_templates[q]['true_mean'],
                all_templates[q]['true_std'],
                all_templates[q]['true_p'])
            for q in range(n_qids)
        )
        if all_ok:
            return suffixes

    raise RuntimeError(
        f"build_suffix_block failed after {MAX_REJECTION_ATTEMPTS} attempts "
        f"(task={task})."
    )


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def generate_task_sequences(task, args, rng):
    """Generate all trials for one task. Returns (DataFrame, json_trials)."""
    seq_length    = args.seq_length
    prefix_length = args.prefix_length
    suffix_length = seq_length - prefix_length
    full_repeat   = (prefix_length >= seq_length)

    n_unique  = args.n_unique_sequences
    assert n_unique % 2 == 0, "n_unique_sequences must be even"
    n_base    = n_unique // 2
    n_repeats = args.n_repeats
    n_total   = n_unique * n_repeats

    print(f"\n{'='*60}")
    print(f"Task: {task.upper()}")
    print(f"  Total trials   : {n_total} ({n_unique} seqs × {n_repeats} reps)")
    print(f"  Seq / prefix / suffix length: {seq_length} / {prefix_length} / {suffix_length}")

    # ── Base parameter sets ──────────────────────────────────────────────────
    if task == 'continuous':
        param_sets = continuous_param_grid(n_base, args.mean_range, args.std_fixed, rng)
        print(f"  std_fixed={args.std_fixed}  mean_range={args.mean_range}")
    else:
        param_sets = binary_param_grid(n_base, args.p_range, rng)
        print(f"  p_range={args.p_range}")

    # param_list: list of (true_mean, true_std, true_p) for each base qid
    if task == 'continuous':
        param_list = [(mu, sd, float('nan')) for mu, sd in param_sets]
    else:
        param_list = [(tp, float('nan'), tp) for tp in param_sets]

    # ── Build prefix block jointly across all base qids ──────────────────────
    print(f"  Building prefix block ({prefix_length} obs, joint across {n_base} base qids)...")
    base_prefixes = build_prefix_block(rng, task, param_list, prefix_length)

    # ── Build base templates ─────────────────────────────────────────────────
    templates = []
    for qid, (params, prefix) in enumerate(zip(param_list, base_prefixes)):
        true_mean, true_std, true_p = params
        sc = float(true_std) if task == 'continuous' and not math.isnan(true_std) else float('nan')
        templates.append({
            'qid': qid, 'true_mean': true_mean, 'true_std': true_std,
            'std_condition': sc, 'true_p': true_p, 'prefix': prefix,
        })
        pfx_str = ','.join(map(str, prefix))
        label   = f'mean={true_mean:.1f}' if task == 'continuous' else f'p={true_p:.3f}'
        print(f"  qid {qid:3d}: {label}  prefix=[{pfx_str}]")

    # ── Add mirror templates ─────────────────────────────────────────────────
    for base in templates[:n_base]:
        m_mean, m_std, m_p = mirror_params(
            base['true_mean'], base['true_std'], base['true_p'], task)
        m_prefix = mirror_sequence(base['prefix'], task)
        m_qid    = len(templates)
        m_sc     = float(m_std) if task == 'continuous' and not math.isnan(m_std) else float('nan')
        templates.append({
            'qid': m_qid, 'true_mean': m_mean, 'true_std': m_std,
            'std_condition': m_sc, 'true_p': m_p, 'prefix': m_prefix,
        })
        pfx_str = ','.join(map(str, m_prefix))
        label   = f'mean={m_mean:.1f}' if task == 'continuous' else f'p={m_p:.3f}'
        print(f"  qid {m_qid:3d}: {label}  prefix=[{pfx_str}]  [mirror of qid {base['qid']}]")

    # ── Expand to trials: shared prefix + jointly-sampled suffix per repeat ──
    # For each repeat, sample one suffix block jointly across ALL qids so that
    # the aggregate RMSE is non-increasing at every suffix position.
    trials = []
    if full_repeat:
        for tmpl in templates:
            for _ in range(n_repeats):
                trials.append({**tmpl, 'values': tmpl['prefix']})
    else:
        for _rep in range(n_repeats):
            suffixes = build_suffix_block(rng, task, templates, suffix_length)
            for q, tmpl in enumerate(templates):
                trials.append({**tmpl, 'values': tmpl['prefix'] + suffixes[q]})

    rng.shuffle(trials)

    # ── Build DataFrame + JSON ───────────────────────────────────────────────
    records, json_trials = [], []
    for t, trial in enumerate(trials):
        sc = trial['std_condition']
        for o, v in enumerate(trial['values'], start=1):
            records.append({
                'task': task, 'trial': t, 'qid': trial['qid'],
                'observation': o, 'value': v,
                'true_mean': trial['true_mean'], 'true_std': trial['true_std'],
                'std_condition': sc, 'true_p': trial['true_p'],
                'prefix_length': prefix_length,
            })
        json_trials.append({
            'trial': t, 'qid': trial['qid'],
            'true_mean': None if math.isnan(trial['true_mean']) else trial['true_mean'],
            'true_std':  None if math.isnan(trial['true_std'])  else trial['true_std'],
            'true_p':    None if math.isnan(trial['true_p'])    else trial['true_p'],
            'std_condition': None if (sc != sc) else sc,
            'values': trial['values'], 'prefix_length': prefix_length,
        })

    df = pd.DataFrame(records)

    # ── Sanity checks ────────────────────────────────────────────────────────
    assert len(df) == n_total * seq_length
    if task == 'continuous':
        assert df['value'].between(VALUE_MIN, VALUE_MAX).all()
    else:
        assert df['value'].isin([-1, 1]).all()
    assert df['trial'].nunique() == n_total
    assert (df.groupby('qid')['trial'].nunique() == n_repeats).all()

    print(f"\n  ✓ {len(df)} rows | {n_total} trials | {n_unique} seqs × {n_repeats} reps")
    if task == 'continuous':
        print(f"  Value range : {df['value'].min()} – {df['value'].max()}")
        print(f"  true_mean   : {df['true_mean'].min():.1f} – {df['true_mean'].max():.1f}")
    else:
        print(f"  true_p range: {df['true_p'].min():.3f} – {df['true_p'].max():.3f}")

    return df, json_trials


# ---------------------------------------------------------------------------
# Seed-search scoring  (unchanged)
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
        gt     = tdf['true_mean'].iloc[0] / 100.0 if task == 'continuous' \
                 else tdf['true_p'].iloc[0] * 2 - 1
        resps  = _bayesian_continuous(values) if task == 'continuous' \
                 else _bayesian_binary(values)
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

    early_rise  = float(np.sum(np.maximum(np.diff(rmse_curve[:min(5, len(rmse_curve))]), 0)))
    mean_delta  = np.mean(delta_curves, axis=0)
    t_d         = np.arange(2, len(mean_delta) + 2, dtype=float)
    try:
        popt2, _ = curve_fit(_power_law_A, t_d, mean_delta,
                              p0=[mean_delta[0]], bounds=([0], [np.inf]))
        score_delta = float(np.sqrt(np.mean((mean_delta - _power_law_A(t_d, popt2[0]))**2)))
    except Exception:
        score_delta = float(np.std(mean_delta))

    return score_shape, score_delta, score_shape + early_rise + score_delta


def _save_sequences(df, json_trials, task, out_dir):
    out_dir   = pathlib.Path(out_dir)
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
                best_score, best_scores, best_seed, best_df = \
                    combined, (r_shape, r_delta), attempt, df.copy()
                _save_sequences(df, json_trials, task, out_dir)
            if (attempt + 1) % 50 == 0:
                print(f"  [{attempt+1:4d}/{args.n_tries}]  best={best_seed}  "
                      f"combined={best_score:.5f}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    arr = np.array(all_scores)
    print(f"\n  Best seed={best_seed}  combined={best_score:.5f}  "
          f"median={np.median(arr):.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--task',               choices=['continuous', 'binary', 'both'], default='both')
    p.add_argument('--n_tries',            type=int,   default=1)
    p.add_argument('--n_unique_sequences', type=int,   default=10)
    p.add_argument('--n_repeats',          type=int,   default=4)
    p.add_argument('--seq_length',         type=int,   default=15)
    p.add_argument('--prefix_length',      type=int,   default=3)
    p.add_argument('--mean_range',         type=float, nargs=2, default=[20.0, 80.0])
    p.add_argument('--std_fixed',          type=float, default=20.0)
    p.add_argument('--p_range',            type=float, nargs=2, default=[0.2, 0.8])
    p.add_argument('--output_dir',         default='task/sequences')
    p.add_argument('--seed',               type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n_unique_sequences % 2 == 0, 'n_unique_sequences must be even'
    assert args.n_repeats > 0
    assert args.seq_length > 0
    assert 0 <= args.prefix_length <= args.seq_length
    assert args.mean_range[0] < args.mean_range[1]
    assert args.std_fixed > 0
    assert 0 < args.p_range[0] < args.p_range[1] < 1

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = ['continuous', 'binary'] if args.task == 'both' else [args.task]

    if args.n_tries > 1:
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
    print("\nJOB_COMPLETE")


if __name__ == '__main__':
    main()
