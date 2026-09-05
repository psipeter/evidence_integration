"""
generate_sequences.py
=====================
SHARED UTILITIES ONLY -- this file has no CLI/main() and is not meant to be
run directly. It exists purely as an import target for
generate_sequences_iid.py and generate_sequences_momentmatch.py, which
import RNG/param-grid/observation-drawing/plausibility-checking/scoring
helpers from here rather than each duplicating their own copies.

This file's OWN generation method (rejection sampling: draw prefix/suffix
freely, redraw whole blocks until the realized sequence passes a
plausibility check) was REMOVED in a cleanup pass once it was confirmed
nothing still called it directly -- it was never promoted to production
(the pilot uses generate_sequences_momentmatch.py) and had real, documented
scaling problems: the joint multi-qid rejection constraint collapsed the
binary pass rate from ~12% to ~0% going from 6 to 10 qids, and rejection
sampling structurally can't reach std_fixed near the [0,100] boundary the
way moment-matching's iterative rescale can (see CLAUDE.md's "Sequence
generation methods" section for the full history and numbers). The method
is still described there for context; only the runnable code lived here,
and it's gone now. Recoverable from git history (this file, pre-cleanup)
if ever needed again.

What's actually here now -- every one of these is imported by at least one
of the other two scripts; check both before removing or renaming anything:
  - make_rng
  - continuous_param_grid, binary_param_grid        (generate_sequences_iid.py)
  - mirror_sequence, mirror_params                  (generate_sequences_iid.py)
  - draw_continuous_obs, draw_binary_obs            (generate_sequences_iid.py)
  - check_sequence_plausibility                     (both other scripts)
  - _weighted_delta_score, _weighted_rmse_score,
    score_sequences                                 (generate_sequences_momentmatch.py, --score_mode bump)
  - _bayesian_responses, _rl_responses               (generate_sequences_momentmatch.py)
  - _save_sequences                                 (both other scripts)

See generate_sequences_iid.py and generate_sequences_momentmatch.py for the
two generation methods actually in use or under active consideration --
each has its own module docstring describing its role, status, and
rationale.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALUE_MIN = 0
VALUE_MAX = 100
MAX_REJECTION_ATTEMPTS = 10_000  # still used by draw_continuous_obs below


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
    """Draw n integer observations from Normal(mean, std), bounded to [0,100].

    Uses fast rejection sampling: draw one at a time, keep if in bounds.
    Clipping only applies if rejection fails after MAX_REJECTION_ATTEMPTS
    (effectively never for typical params).  The resulting distribution is
    equivalent to a truncated Normal — clipping at 0/100 does not bias the
    std meaningfully since truncation itself reduces std at extreme means
    by the same amount.
    """
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
# Per-sequence plausibility check
# ---------------------------------------------------------------------------
def check_sequence_plausibility(sequence, task, true_mean, true_std, true_p,
                                k: float = 1.0,
                                k_std: float = 0.7) -> bool:
    """True iff full-sequence mean AND std are within k × SE of true values.

    Continuous:
      mean check: |sample_mean - true_mean| <= k     × std/sqrt(n)
      std  check: |sample_std  - true_std|  <= k_std × std × sqrt(2/(n-1))
    Binary:
      proportion check: |sample_p - true_p| <= k × sqrt(p(1-p)/n)
    """
    n = len(sequence)
    if n == 0:
        return True
    if task == 'continuous':
        mean_ok = abs(np.mean(sequence) - true_mean) <= k * true_std / np.sqrt(n)
        std_ok  = abs(np.std(sequence) - true_std) \
                  <= k_std * true_std * np.sqrt(2.0 / max(n - 1, 1))
        return mean_ok and std_ok
    else:
        return abs(sum(v == 1 for v in sequence) / n - true_p) \
               <= k * np.sqrt(true_p * (1 - true_p) / n)


# ---------------------------------------------------------------------------
# Seed-search scoring (used by generate_sequences_momentmatch.py's
# --score_mode bump; the isotonic score_mode, the current default, does not
# use these -- see score_sequences_isotonic in that file instead)
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


def _weighted_rmse_score(seq_df, task, agent_fn, gamma=0.0):
    """Uniformly weighted rises in the mean RMSE curve.

    Uses gamma=0 (uniform weights) so rises anywhere in the curve are
    penalised equally — RMSE should be monotone throughout, not just early.
    Score = 0 iff RMSE curve is perfectly monotone non-rising.

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


def score_sequences(seq_df, task, rl_alpha_0=1.0, rl_lambda=0.5,
                    gamma_bay_delta_cont=0.7, gamma_bay_rmse_cont=0.3,
                    gamma_rl_delta_cont=0.7,  gamma_rl_rmse_cont=0.3,
                    gamma_bay_delta_bin=0.3,  gamma_bay_rmse_bin=0.0,
                    gamma_rl_delta_bin=0.3,   gamma_rl_rmse_bin=0.0):
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
    if task == 'continuous':
        gbd, gbr, grd, grr = (gamma_bay_delta_cont, gamma_bay_rmse_cont,
                               gamma_rl_delta_cont,  gamma_rl_rmse_cont)
    else:
        gbd, gbr, grd, grr = (gamma_bay_delta_bin,  gamma_bay_rmse_bin,
                               gamma_rl_delta_bin,   gamma_rl_rmse_bin)
    bay_delta = _weighted_delta_score(seq_df, task, bay_fn, gamma=gbd)
    bay_rmse  = _weighted_rmse_score( seq_df, task, bay_fn, gamma=gbr)
    rl_delta  = _weighted_delta_score(seq_df, task, rl_fn,  gamma=grd)
    rl_rmse   = _weighted_rmse_score( seq_df, task, rl_fn,  gamma=grr)
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
