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
    Drawn freely (i.i.d.) from the true distribution — no convergence
    constraint.  Participants cannot easily infer the true parameter
    from the prefix alone.

  Suffix block (obs prefix_length+1..seq_length):
    Drawn freely per repeat, accepted when the full sequence
    (prefix+suffix) passes the plausibility check.  Varies across
    repeats, providing within-qid variability for PTN analysis.

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

Best seeds (from 200-seed search minimising weighted RL |Δ| score)
------------------------------------------------------------------
  continuous : seed=51  prefix_length=4
  binary     : TBD after rerun with new scoring  prefix_length=4

Scoring (3-pass):
  Pass 1 (structural): Bayesian RMSE non-rising, k=1 plausibility
  Pass 2 (objective):  Σ_obs w(obs)×max(0, |Δ|(obs)−|Δ|(obs−1)) for Bayesian agent
    w(obs) = exp(-0.5*(obs-2)); score=0 iff |Δ| curve is perfectly monotone
    Binary: same metric using RL_lambda(λ=0.5) since bay/rl ratio varies
  Pass 2 gate: bay_delta == 0
  Pass 3 objective: minimize rl_delta

Usage
-----
  # Single seed (fast, use best known seeds)
  python task/generate_sequences.py --task continuous --seed 51
  python task/generate_sequences.py --task binary --seed 42

  # Seed search (pass 2 gate + pass 3 RL objective)
  python task/generate_sequences.py --task both --n_tries 200 \\
      --rl_alpha_0 1.0 --rl_lambda 0.5 --prefix_length 4
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import pandas as pd

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
# RL_lambda agent helper
# ---------------------------------------------------------------------------
def _rl_lambda_errors(sequence, task, true_mean, true_p,
                      alpha_0=1.0, lambda_=0.5):
    """Per-observation |response − true parameter| for RL_lambda agent.
    alpha_t = alpha_0 / n^lambda_  (decaying learning rate)
    Continuous: response tracks normalised [0,1] mean.
    Binary:     response tracks p directly (initialised at 0.5).
    """
    errs    = []
    running = 0.5   # prior at midpoint for both tasks
    if task == 'continuous':
        gt = true_mean / 100.0
        for n, v in enumerate(sequence, 1):
            alpha    = alpha_0 / (n ** lambda_)
            running += alpha * (v / 100.0 - running)
            running  = float(np.clip(running, 0.0, 1.0))
            errs.append(abs(running - gt))
    else:
        gt = true_p
        for n, v in enumerate(sequence, 1):
            alpha    = alpha_0 / (n ** lambda_)
            obs_norm = 1.0 if v == 1 else 0.0   # map {-1,+1} → {0,1}
            running += alpha * (obs_norm - running)
            running  = float(np.clip(running, 0.0, 1.0))
            errs.append(abs(running - gt))
    return errs




# ---------------------------------------------------------------------------
# Per-sequence plausibility check
# ---------------------------------------------------------------------------
def check_sequence_plausibility(sequence, task, true_mean, true_std, true_p,
                                k: float = 1.0) -> bool:
    """True iff full-sequence mean AND std are within k × SE of true values.
    Continuous: mean check + std check (std SE = std*sqrt(2/(n-1))).
    Binary:     fraction check only (std is determined by p).
    """
    n = len(sequence)
    if n == 0:
        return True
    if task == 'continuous':
        mean_ok = abs(np.mean(sequence) - true_mean) <= k * true_std / np.sqrt(n)
        std_ok  = abs(np.std(sequence) - true_std) \
                  <= k * true_std * np.sqrt(2.0 / max(n - 1, 1))
        return mean_ok and std_ok
    else:
        return abs(sum(v == 1 for v in sequence) / n - true_p) \
               <= k * np.sqrt(true_p * (1 - true_p) / n)


# ---------------------------------------------------------------------------
# PREFIX BLOCK  — joint generation across all qids
# ---------------------------------------------------------------------------
def build_prefix_block(rng, task, param_list, prefix_length):
    """Draw prefix observations freely from the generative distribution.

    No per-position constraint.  Observations are i.i.d. from the true
    distribution — the prefix is not constrained to converge early, so
    participants cannot easily infer the true parameter from the prefix
    alone.  Smoothness of model curves is enforced by seed search (pass 2/3)
    rather than by construction.

    Returns: list of prefixes (one per entry in param_list).
    """
    n_qids = len(param_list)
    if task == 'continuous':
        return [draw_continuous_obs(rng, param_list[q][0], param_list[q][1],
                                    prefix_length)
                for q in range(n_qids)]
    else:
        return [draw_binary_obs(rng, param_list[q][2], prefix_length)
                for q in range(n_qids)]


# ---------------------------------------------------------------------------
# SUFFIX BLOCK  — joint across all qids, one position at a time, per repeat
# ---------------------------------------------------------------------------
def build_suffix_block(rng, task, all_templates, suffix_length):
    """Draw suffix observations freely, accept when full sequence passes
    plausibility check (mean + std within k=1 SE of true values).

    Returns: list of suffixes (one per template in all_templates).
    """
    n_qids = len(all_templates)

    for _outer in range(MAX_REJECTION_ATTEMPTS):
        if task == 'continuous':
            suffixes = [draw_continuous_obs(rng, all_templates[q]['true_mean'],
                                            all_templates[q]['true_std'],
                                            suffix_length)
                        for q in range(n_qids)]
        else:
            suffixes = [draw_binary_obs(rng, all_templates[q]['true_p'],
                                        suffix_length)
                        for q in range(n_qids)]

        # Accept if every full sequence passes mean + std plausibility
        k_plaus = 0.7 if task == 'binary' else 1.0
        if all(
            check_sequence_plausibility(
                all_templates[q]['prefix'] + suffixes[q],
                task,
                all_templates[q]['true_mean'],
                all_templates[q]['true_std'],
                all_templates[q]['true_p'],
                k=k_plaus)
            for q in range(n_qids)
        ):
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
    # ITI condition: randomly assign n_repeats//2 repeats per qid to short ITI,
    # the rest to long ITI. Randomization is per-qid using a fixed seed so
    # the assignment is reproducible but not systematically ordered.
    ITI_SHORT_MS = 1000
    ITI_LONG_MS  = 5000
    iti_rng = np.random.default_rng(int(rng.integers(2**31)))  # derive from main rng
    # Pre-assign ITI schedule for each qid: shuffled list of short/long labels
    iti_schedule = {}
    for tmpl in templates:
        schedule = ([ITI_SHORT_MS] * (n_repeats // 2) +
                    [ITI_LONG_MS]  * (n_repeats - n_repeats // 2))
        iti_rng.shuffle(schedule)
        iti_schedule[tmpl['qid']] = schedule
    rep_count = {}  # tracks how many repeats each qid has seen
    trials = []
    if full_repeat:
        for tmpl in templates:
            qid = tmpl['qid']
            for _ in range(n_repeats):
                rep_idx = rep_count.get(qid, 0)
                iti_ms  = iti_schedule[qid][rep_idx]
                rep_count[qid] = rep_idx + 1
                trials.append({**tmpl, 'values': tmpl['prefix'], 'iti_ms': iti_ms})
    else:
        for _rep in range(n_repeats):
            suffixes = build_suffix_block(rng, task, templates, suffix_length)
            for q, tmpl in enumerate(templates):
                qid     = tmpl['qid']
                rep_idx = rep_count.get(qid, 0)
                iti_ms  = iti_schedule[qid][rep_idx]
                rep_count[qid] = rep_idx + 1
                trials.append({**tmpl, 'values': tmpl['prefix'] + suffixes[q],
                               'iti_ms': iti_ms})

    # Shuffle trials such that:
    #   1. No two consecutive trials share the same qid (same prefix).
    #   2. The first trial's distribution is sufficiently different from
    #      the tutorial's, so participants don't confuse the two.
    #      Continuous: |true_mean - 50| > 10  (avoid near-centre means)
    #      Binary:     |true_p - 0.5| > 0.15  (avoid near-0.5 probabilities)
    def shuffle_no_consecutive_qid(trials, rng, task):
        trials = list(trials)
        rng.shuffle(trials)
        result = []
        remaining = list(trials)
        while remaining:
            last_qid = result[-1]['qid'] if result else None
            is_first = len(result) == 0
            def ok(t):
                if t['qid'] == last_qid:
                    return False
                if is_first:
                    if task == 'continuous':
                        return abs(t['true_mean'] - 50.0) > 10.0
                    else:
                        return abs(t['true_p'] - 0.5) > 0.15
                return True
            candidates = [i for i, t in enumerate(remaining) if ok(t)]
            if not candidates:  # relax constraints if stuck
                candidates = [i for i, t in enumerate(remaining)
                              if t['qid'] != last_qid]
            if not candidates:
                candidates = list(range(len(remaining)))
            idx = candidates[int(rng.integers(len(candidates)))]
            result.append(remaining.pop(idx))
        return result
    trials = shuffle_no_consecutive_qid(trials, rng, task)

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
                'iti_ms': trial['iti_ms'],
            })
        json_trials.append({
            'trial': t, 'qid': trial['qid'],
            'true_mean': None if math.isnan(trial['true_mean']) else trial['true_mean'],
            'true_std':  None if math.isnan(trial['true_std'])  else trial['true_std'],
            'true_p':    None if math.isnan(trial['true_p'])    else trial['true_p'],
            'std_condition': None if (sc != sc) else sc,
            'values': trial['values'], 'prefix_length': prefix_length,
            'iti_ms': trial['iti_ms'],
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
# Seed-search scoring
# ---------------------------------------------------------------------------




def _weighted_delta_score(seq_df, task, agent_fn, gamma=0.5):
    """Exponentially weighted sum of RISES in the mean |Δresponse| curve.

    Builds the aggregate mean |Δresponse| curve across all trials, then
    sums only upward steps (rises), weighted by position:

        score = Σ_obs  w(obs) × max(0, curve(obs) − curve(obs−1))

    w(obs) = exp(-gamma*(obs-2)): early rises penalised most.
    Score = 0 iff the |Δ| curve is perfectly monotone non-rising.

    agent_fn(values, task, true_mean, true_p) → list of responses
    """
    obs_deltas = {}
    for trial_id in seq_df['trial'].unique():
        tdf   = seq_df[seq_df['trial'] == trial_id].sort_values('observation')
        tm    = tdf['true_mean'].iloc[0]
        tp    = tdf['true_p'].iloc[0] if task == 'binary' else float('nan')
        resps = agent_fn(tdf['value'].tolist(), task, tm, tp)
        for i in range(1, len(resps)):
            obs = int(tdf['observation'].iloc[i])
            obs_deltas.setdefault(obs, []).append(abs(resps[i] - resps[i - 1]))
    obs_sorted = sorted(obs_deltas)
    curve = [float(np.mean(obs_deltas[o])) for o in obs_sorted]
    total = 0.0
    for i in range(1, len(curve)):
        rise = curve[i] - curve[i - 1]
        if rise > 0:
            obs = obs_sorted[i]
            w   = float(np.exp(-gamma * (obs - 2)))
            total += w * rise
    return total


def _weighted_rmse_score(seq_df, task, agent_fn, gamma=0.5):
    """Exponentially weighted rises in the mean RMSE curve.

    Mirrors _weighted_delta_score but operates on |response − true_param|
    (RMSE) rather than |Δresponse|.  Score = 0 iff RMSE curve is monotone.

    agent_fn(values, task, true_mean, true_p) → list of responses
    """
    obs_errs = {}
    for trial_id in seq_df['trial'].unique():
        tdf  = seq_df[seq_df['trial'] == trial_id].sort_values('observation')
        tm   = tdf['true_mean'].iloc[0]
        tp   = tdf['true_p'].iloc[0] if task == 'binary' else float('nan')
        gt   = tm / 100.0 if task == 'continuous' else tp
        resp = agent_fn(tdf['value'].tolist(), task, tm, tp)
        for obs, r in zip(tdf['observation'].tolist(), resp):
            obs_errs.setdefault(int(obs), []).append(abs(r - gt))
    obs_sorted = sorted(obs_errs)
    curve  = [float(np.mean(obs_errs[o])) for o in obs_sorted]
    total  = 0.0
    for i in range(1, len(curve)):
        rise = curve[i] - curve[i - 1]
        if rise > 0:
            obs = obs_sorted[i]
            w   = float(np.exp(-gamma * (obs - 1)))
            total += w * rise
    return total


def _bayesian_responses(values, task, true_mean, true_p):
    """Bayesian agent responses (running posterior mean, normalised)."""
    resps = []
    if task == 'continuous':
        running = 0.5
        for n, v in enumerate(values, 1):
            running += (v / 100.0 - running) / n
            resps.append(float(np.clip(running, 0.0, 1.0)))
    else:
        n_pos = 0
        for n, v in enumerate(values, 1):
            n_pos += (1 if v == 1 else 0)
            resps.append((n_pos + 1) / (n + 2))  # Laplace smoothing → [0,1]
    return resps


def _rl_responses(values, task, true_mean, true_p,
                  alpha_0=1.0, lambda_=0.5):
    """RL_lambda agent responses (running estimate, normalised to [0,1])."""
    running = 0.5
    resps   = []
    for n, v in enumerate(values, 1):
        alpha   = alpha_0 / (n ** lambda_)
        obs_n   = v / 100.0 if task == 'continuous' else (1.0 if v == 1 else 0.0)
        running = float(np.clip(running + alpha * (obs_n - running), 0.0, 1.0))
        resps.append(running)
    return resps


def score_sequences(seq_df, task, rl_alpha_0=1.0, rl_lambda=0.5, gamma=0.5):
    """Three-pass scoring for seed search.

    Pass 1 (generation constraint, already applied):
      Bayesian RMSE non-rising — enforced structurally in block builders.
      k=1 plausibility — enforced structurally in block builders.
      All candidates reaching this function already satisfy pass 1.

    Pass 2 — Bayesian weighted |Δresponse|:
      Exponentially weighted sum of mean |ΔBayesian_response| per obs.
      Ensures smooth Bayesian curve shape.

    Pass 3 — RL_lambda weighted |Δresponse|  (primary objective):
      Same metric using RL_lambda(alpha_0, lambda_) responses.
      Directly targets what panels C/J show — smooth per-quartile
      |Δ| decay.  Pass 3 score is the combined score returned;
      pass 2 is reported for diagnostics.

    Pass 2 gate: bay_delta must be 0 (Bayesian |Δ| curve perfectly monotone).
    Pass 3 objective: minimize rl_delta among seeds that pass gate.
    Returns (bay_delta, rl_delta, combined) where combined = rl_delta.
    """
    bay_fn    = _bayesian_responses
    rl_fn     = lambda vals, tsk, tm, tp: _rl_responses(
        vals, tsk, tm, tp, alpha_0=rl_alpha_0, lambda_=rl_lambda)
    bay_delta = _weighted_delta_score(seq_df, task, bay_fn)
    bay_rmse  = _weighted_rmse_score(seq_df,  task, bay_fn)
    rl_delta  = _weighted_delta_score(seq_df, task, rl_fn)
    rl_rmse   = _weighted_rmse_score(seq_df,  task, rl_fn)
    bay_score = bay_delta + bay_rmse
    rl_score  = rl_delta  + rl_rmse
    return bay_score, rl_score, rl_score  # combined = rl_score


def _save_sequences(df, json_trials, task, out_dir):
    out_dir   = pathlib.Path(out_dir)
    json_path = out_dir / f'{task}_sequences.json'
    pkl_path  = out_dir / f'{task}_sequences.pkl'
    with open(json_path, 'w') as f:
        json.dump(json_trials, f, indent=2)
    df.to_pickle(pkl_path)
    return pkl_path, json_path


def _seed_search(task, args, out_dir):
    import io, contextlib, signal

    def _timeout_handler(signum, frame):
        raise TimeoutError()
    print(f"\n{'='*55}")
    print(f"Seed search: {task.upper()} | {args.n_tries} tries | "
          f"RL_lambda α={args.rl_alpha_0} λ={args.rl_lambda}")
    print(f"Score = weighted |Δresponse| rises (lower → more monotone decay)")
    print(f"  w(obs) = exp(-0.5*(obs-2));  score=0 iff perfectly monotone")
    print(f"  Pass 1 (structural): full-seq mean+std within k=1 SE of true values")
    print(f"  Pass 2 (objective):  bay_delta rises (continuous), rl_delta rises (binary)")
    best_score, best_scores, best_seed, best_df, best_json = \
        np.inf, (np.inf, np.inf), None, None, None
    all_scores, n_ok, n_pass2 = [], 0, 0
    for attempt in range(args.n_tries):
        task_seed = attempt if task == 'continuous' else attempt + 1000
        rng = make_rng(task_seed)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)  # 30s per seed — skips hangers
            # Use reduced inner attempts during search for speed
            _orig_max = generate_task_sequences.__globals__['MAX_REJECTION_ATTEMPTS']
            generate_task_sequences.__globals__['MAX_REJECTION_ATTEMPTS'] = 200
            with contextlib.redirect_stdout(io.StringIO()):
                df, json_trials = generate_task_sequences(task, args, rng)
            generate_task_sequences.__globals__['MAX_REJECTION_ATTEMPTS'] = _orig_max
            signal.alarm(0)
        except Exception:
            signal.alarm(0)
            generate_task_sequences.__globals__['MAX_REJECTION_ATTEMPTS'] = 10_000
            continue
        n_ok += 1
        bay, rl, combined = score_sequences(
            df, task, rl_alpha_0=args.rl_alpha_0, rl_lambda=args.rl_lambda)
        if bay > 2e-2:
            continue  # pass 2 gate: bay_score must be below 0.02
        n_pass2 += 1
        all_scores.append(combined)
        if combined < best_score:
            best_score  = combined
            best_scores = (bay, rl)
            best_seed   = attempt
            best_df     = df.copy()
            best_json   = json_trials
            _save_sequences(df, json_trials, task, out_dir)
        if (attempt + 1) % 10 == 0 or attempt == args.n_tries - 1:
            print(f"  [{attempt+1:4d}/{args.n_tries}]  ok={n_ok}  pass2={n_pass2}  "
                  f"best_seed={best_seed}  "
                  f"rl_delta={best_scores[1]:.5f}  "
                  f"combined={best_score:.5f}")
    arr = np.array(all_scores) if all_scores else np.array([np.inf])
    print(f"\n  Best seed      : {best_seed}")

    print(f"  Bay score      : {best_scores[0]:.6f}  (delta+rmse, gate: < 0.02)")
    print(f"  RL  score      : {best_scores[1]:.6f}  (delta+rmse)")
    print(f"  Combined       : {best_score:.5f}  (median {np.median(arr):.5f})")
    print(f"  Saved to       : {out_dir}/{task}_sequences.{{pkl,json}}")


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
    p.add_argument('--prefix_length',      type=int,   default=4)
    p.add_argument('--mean_range',         type=float, nargs=2, default=[20.0, 80.0])
    p.add_argument('--std_fixed',          type=float, default=20.0)
    p.add_argument('--p_range',            type=float, nargs=2, default=[0.2, 0.8])
    p.add_argument('--rl_alpha_0',         type=float, default=1.0,
                   help='RL_lambda alpha_0 for dual-agent constraint')
    p.add_argument('--rl_lambda',          type=float, default=0.5,
                   help='RL_lambda lambda for dual-agent constraint')
    p.add_argument('--output_dir',         default='task/sequences')
    p.add_argument('--seed',               type=int,   default=42,
                   help='Best known: continuous=51, binary=42')
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
