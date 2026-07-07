"""
generate_sequences_momentmatch.py
==================================
Alternative sequence generator using MOMENT MATCHING instead of rejection
sampling. Kept fully separate from generate_sequences.py so the two
strategies can be compared side by side.

Motivation
----------
generate_sequences.py draws i.i.d. observations from the true generative
distribution, then repeatedly redraws whole blocks until the realized
sample statistics happen to fall within k x SE of the true parameter
(rejection sampling). This has two costs:
  1. The joint constraint (ALL qids must pass simultaneously in one draw)
     scales very badly with n_unique_sequences -- going from 6 to 10 qids
     collapsed the binary pass rate from ~12% to ~0% at k=0.5, and even at
     k=0.7 to only ~6% (see chat history).
  2. At extreme means (e.g. mean=10 or 90 with std=15), the [0,100] bound
     truncates the achievable std so far below the nominal value that NO
     amount of resampling can pass a tight k -- it's not a sampling problem,
     it's a structural mismatch between the target and the bound.

This script instead CONSTRUCTS each block (prefix or suffix) to hit the
target sample mean/std (continuous) or exact blue/red quota (binary)
directly, then only randomizes order / residual shape. This sidesteps both
problems: there is no joint multi-qid rejection loop, and the target
statistics are hit by construction rather than by chance, so it scales to
any n_unique_sequences for free.

Method
------
Continuous (per block, prefix or suffix, independently):
  1. Draw n raw i.i.d. values from Normal(true_mean, true_std) (no rejection).
  2. Affinely rescale so the block's OWN sample mean/std exactly match the
     target: v' = true_mean + (v - mean(v)) * (true_std / std(v)).
  3. Clip to [0,100], then repeat the rescale+clip step a few times
     (max_rescale_iters) to claw back most of the bias clipping introduces
     at extreme means -- this does NOT fully solve the mean=10/90 + std=15
     structural mismatch (clipping still truncates), but it converges much
     closer than a single pass, and unlike generate_sequences.py it never
     needs to reject and redraw.
  4. Round to nearest int.

  Because two blocks that each independently hit the same (mean, std) have
  a pooled mean/std approximately equal to that same (mean, std) -- the
  between-block variance term vanishes when both blocks share a mean --
  the prefix and suffix can each target the SAME global (true_mean,
  true_std) independently. No "residual target" bookkeeping is needed.

Binary (per block, prefix or suffix, independently):
  Exact quota: round(n * true_p) blue, the rest red, order shuffled.
  This is exact by construction (up to rounding) -- no correction needed.

Prefix / suffix structure (preserved from generate_sequences.py)
------------------------------------------------------------------
  Prefix: built once per qid (shared across all n_repeats repeats of that
    qid), via moment_match_continuous/binary.
  Suffix: built FRESH per repeat via a new raw draw + rescale (continuous)
    or a fresh shuffle of the same quota (binary) -- so repeats differ in
    actual realized values, giving genuine within-qid variability rather
    than just reordering a fixed multiset.

What this does NOT fix
-----------------------
  Clipping after rescaling still truncates the distribution at extreme
  means, which still biases the achieved std down somewhat (just less
  severely than generate_sequences.py's rejection sampling, since we
  actively correct for it via iterative rescale). If mean_range pushes
  all the way to the [0,100] edges with a large std, the plausibility
  check (report only, not gating -- see below) may still show a small
  residual gap at the extremes. Use --report to check this directly
  before trusting a given mean_range/std combination.

Plausibility checking here is DIAGNOSTIC ONLY (reported, not gating) --
generation always succeeds by construction, so there is no rejection loop
and no seed can "fail" the way it could in generate_sequences.py.

Usage
-----
  # Single generation (fast, no search)
  python task/generate_sequences_momentmatch.py --task continuous --seed 0

  # Seed search over orderings/realizations for smoothness
  python task/generate_sequences_momentmatch.py --task both --n_tries 200 \\
      --rl_alpha_0 1.0 --rl_lambda 0.5 --score_mode isotonic

  # Diagnostic report of achieved vs target moments (no file written)
  python task/generate_sequences_momentmatch.py --task continuous --report

Scoring modes (--score_mode)
-----------------------------
  'bump' (original): penalizes only upward steps in the aggregate
    |Delta response| curve; any non-increasing curve scores as perfect
    regardless of how irregularly it decreases. Includes an RMSE-vs-
    ground-truth component and a bay_score<0.02 gate.
  'isotonic' (default): Pool-Adjacent-Violators residual -- penalizes ANY
    deviation from the best-fitting smooth non-increasing curve, in either
    direction, with no assumption about the decay's functional form
    (power-law, exponential, etc.). No RMSE component (curve shape is
    independent of which ground truth downstream analyses use), no gate
    (every seed is ranked by bay_resid + rl_resid, lowest wins).

Output
------
  Saved to task/sequences/{task}_momentmatch_sequences.{pkl,json} by default
  -- deliberately NOT the production filenames, so this never clobbers the
  rejection-sampling sequences already in use. Pass --output_dir / rename
  manually once you've decided to promote this method to production.
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import pandas as pd

# Reuse generic, method-agnostic utilities from generate_sequences.py rather
# than duplicating them -- these are not tied to rejection sampling.
from generate_sequences import (
    VALUE_MIN, VALUE_MAX,
    make_rng, mirror_sequence, mirror_params,
    continuous_param_grid, binary_param_grid,
    check_sequence_plausibility,
    score_sequences,
    _save_sequences,
    _bayesian_responses,
    _rl_responses,
)

SCORE_MODES = ('bump', 'isotonic')


# ---------------------------------------------------------------------------
# Moment-matched block builders
# ---------------------------------------------------------------------------
def moment_match_continuous(rng, true_mean, true_std, n,
                            value_min=VALUE_MIN, value_max=VALUE_MAX,
                            max_rescale_iters=4):
    """n continuous obs whose sample mean/std match target by construction.

    Draws raw i.i.d. Normal(true_mean, true_std), then iteratively rescales
    + clips to converge the block's own sample mean/std onto the target.
    No rejection -- always returns n values.
    """
    vals = rng.normal(true_mean, true_std, size=n)
    for _ in range(max_rescale_iters):
        cur_mean, cur_std = float(vals.mean()), float(vals.std())
        if cur_std < 1e-9:
            vals = vals + rng.normal(0.0, max(true_std, 1.0), size=n)
            continue
        vals = true_mean + (vals - cur_mean) * (true_std / cur_std)
        vals = np.clip(vals, value_min, value_max)
    vals = np.clip(np.round(vals), value_min, value_max).astype(int)
    return vals.tolist()


def moment_match_binary(rng, true_p, n):
    """n binary obs {-1,+1} with exact quota round(n*true_p) blue, shuffled."""
    n_blue = int(round(true_p * n))
    n_blue = min(max(n_blue, 0), n)
    vals = [1] * n_blue + [-1] * (n - n_blue)
    perm = rng.permutation(n)
    return [vals[i] for i in perm]


# ---------------------------------------------------------------------------
# Isotonic-regression shape score (Pool Adjacent Violators, non-increasing)
# ---------------------------------------------------------------------------
def _pava_nonincreasing(y):
    """Least-squares best-fitting non-increasing curve to y (Pool Adjacent
    Violators Algorithm). No parametric assumption about the decay's shape
    -- only that it should not increase. O(n), exact, no hyperparameters.

    Standard stack-based PAVA: scan left to right, merging adjacent blocks
    (weighted-average value, summed count) whenever a block's value exceeds
    the previous block's -- i.e. whenever monotonicity is violated -- then
    expand blocks back into a full-length fitted array.
    """
    y = [float(v) for v in y]
    vals, cnts = [], []
    for yi in y:
        vals.append(yi)
        cnts.append(1)
        while len(vals) >= 2 and vals[-1] > vals[-2]:
            v2, c2 = vals.pop(), cnts.pop()
            v1, c1 = vals.pop(), cnts.pop()
            merged_c = c1 + c2
            merged_v = (v1 * c1 + v2 * c2) / merged_c
            vals.append(merged_v)
            cnts.append(merged_c)
    fit = []
    for v, c in zip(vals, cnts):
        fit.extend([v] * c)
    return np.array(fit)


def _isotonic_residual_score(seq_df, task, agent_fn):
    """Mean squared residual between the empirical aggregate |Delta response|
    curve and the best-fitting non-increasing curve (PAVA).

    Unlike a rise-only penalty (which only counts upward steps and treats
    any non-increasing curve as perfect regardless of how it decreases),
    this penalizes ANY deviation from a smooth monotone trend in either
    direction -- a curve that decreases in big irregular jumps scores
    worse than one that decreases smoothly, even though both are
    technically non-increasing. Makes no assumption about the decay's
    functional form (power-law, exponential, etc.) -- only that it
    shouldn't oscillate.
    """
    obs_deltas = {}
    for trial_id in seq_df['trial'].unique():
        tdf = seq_df[seq_df['trial'] == trial_id].sort_values('observation')
        tm = tdf['true_mean'].iloc[0]
        tp = tdf['true_p'].iloc[0] if task == 'binary' else float('nan')
        resps = agent_fn(tdf['value'].tolist(), task, tm, tp)
        for i in range(1, len(resps)):
            obs = int(tdf['observation'].iloc[i])
            obs_deltas.setdefault(obs, []).append(abs(resps[i] - resps[i - 1]))
    obs_sorted = sorted(obs_deltas)
    curve = np.array([np.mean(obs_deltas[o]) for o in obs_sorted])
    fit = _pava_nonincreasing(curve)
    return float(np.mean((curve - fit) ** 2))


def score_sequences_isotonic(seq_df, task, rl_alpha_0=1.0, rl_lambda=0.5):
    """Shape-only scoring: isotonic residual of the Bayesian and RL_lambda
    agents' |Delta response| curves. No RMSE-to-ground-truth component --
    curve shape is independent of which ground truth (true param vs running
    mean) downstream analyses use, so it's deliberately left out here.

    Returns (bay_resid, rl_resid, combined) where combined = bay_resid + rl_resid
    (both curves matter -- unlike the bump-score's gate+rank split, there's
    no natural threshold for a residual, so no gating is applied; every
    seed is just ranked by this sum).
    """
    bay_fn = _bayesian_responses
    rl_fn  = lambda vals, tsk, tm, tp: _rl_responses(
        vals, tsk, tm, tp, alpha_0=rl_alpha_0, lambda_=rl_lambda)
    bay_resid = _isotonic_residual_score(seq_df, task, bay_fn)
    rl_resid  = _isotonic_residual_score(seq_df, task, rl_fn)
    combined  = bay_resid + rl_resid
    return bay_resid, rl_resid, combined


# ---------------------------------------------------------------------------
# Trial-assembly helpers (duplicated in minimal form from generate_sequences.py
# -- kept local rather than importing private helpers, per "cleanly separate")
# ---------------------------------------------------------------------------
def _build_iti_schedule(rng, templates, n_repeats):
    iti_rng = np.random.default_rng(int(rng.integers(2**31)))
    schedule = {}
    for tmpl in templates:
        s = (['control'] * (n_repeats // 2) +
             ['distract'] * (n_repeats - n_repeats // 2))
        iti_rng.shuffle(s)
        schedule[tmpl['qid']] = s
    return schedule


def _shuffle_no_consecutive_qid(trials, rng, task):
    trials = list(trials)
    rng.shuffle(trials)
    result, remaining = [], list(trials)
    while remaining:
        last_qid = result[-1]['qid'] if result else None
        is_first = len(result) == 0

        def ok(t):
            if t['qid'] == last_qid:
                return False
            if is_first:
                if t.get('iti_condition') != 'distract':
                    return False
                if task == 'continuous':
                    return abs(t['true_mean'] - 50.0) > 10.0
                else:
                    return abs(t['true_p'] - 0.5) > 0.15
            return True

        candidates = [i for i, t in enumerate(remaining) if ok(t)]
        if not candidates:
            candidates = [i for i, t in enumerate(remaining) if t['qid'] != last_qid]
        if not candidates:
            candidates = list(range(len(remaining)))
        idx = candidates[int(rng.integers(len(candidates)))]
        result.append(remaining.pop(idx))
    return result


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def generate_task_sequences_momentmatch(task, args, rng):
    """Generate all trials for one task via moment matching.

    Mirrors generate_task_sequences()'s trial structure (prefix/suffix
    split, mirrored qids, ITI scheduling, no-consecutive-qid shuffle) but
    builds blocks via moment matching instead of rejection sampling.
    Returns (DataFrame, json_trials), same schema as generate_sequences.py.
    """
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
    print(f"Task: {task.upper()}  (moment-matched)")
    print(f"  Total trials   : {n_total} ({n_unique} seqs x {n_repeats} reps)")
    print(f"  Seq / prefix / suffix length: {seq_length} / {prefix_length} / {suffix_length}")

    # -- Base parameter sets (reused from generate_sequences.py) ------------
    if task == 'continuous':
        param_sets = continuous_param_grid(n_base, args.mean_range, args.std_fixed, rng)
        print(f"  std_fixed={args.std_fixed}  mean_range={args.mean_range}")
    else:
        param_sets = binary_param_grid(n_base, args.p_range, rng)
        print(f"  p_range={args.p_range}")

    if task == 'continuous':
        param_list = [(mu, sd, float('nan')) for mu, sd in param_sets]
    else:
        param_list = [(tp, float('nan'), tp) for tp in param_sets]

    # -- Build prefix block (once per base qid, moment-matched) -------------
    templates = []
    for qid, (true_mean, true_std, true_p) in enumerate(param_list):
        if task == 'continuous':
            prefix = moment_match_continuous(rng, true_mean, true_std, prefix_length,
                                              max_rescale_iters=args.max_rescale_iters)
        else:
            prefix = moment_match_binary(rng, true_p, prefix_length)
        templates.append({
            'qid': qid, 'true_mean': true_mean, 'true_std': true_std,
            'true_p': true_p, 'prefix': prefix,
        })
        label = f'mean={true_mean:.1f}' if task == 'continuous' else f'p={true_p:.3f}'
        print(f"  qid {qid:3d}: {label}  prefix=[{','.join(map(str, prefix))}]")

    # -- Mirror templates -----------------------------------------------------
    for base in templates[:n_base]:
        m_mean, m_std, m_p = mirror_params(base['true_mean'], base['true_std'],
                                           base['true_p'], task)
        m_prefix = mirror_sequence(base['prefix'], task)
        m_qid    = len(templates)
        templates.append({
            'qid': m_qid, 'true_mean': m_mean, 'true_std': m_std,
            'true_p': m_p, 'prefix': m_prefix,
        })
        label = f'mean={m_mean:.1f}' if task == 'continuous' else f'p={m_p:.3f}'
        print(f"  qid {m_qid:3d}: {label}  prefix=[{','.join(map(str, m_prefix))}]  "
              f"[mirror of qid {base['qid']}]")

    # -- ITI schedule ---------------------------------------------------------
    ITI_MS   = 1000
    iti_sched = _build_iti_schedule(rng, templates, n_repeats)

    # -- Build trials: shared prefix + fresh moment-matched suffix per repeat -
    rep_count = {}
    trials = []
    if full_repeat:
        for tmpl in templates:
            qid = tmpl['qid']
            for _ in range(n_repeats):
                rep_idx = rep_count.get(qid, 0)
                iti_condition = iti_sched[qid][rep_idx]
                rep_count[qid] = rep_idx + 1
                trials.append({**tmpl, 'values': tmpl['prefix'],
                               'iti_ms': ITI_MS, 'iti_condition': iti_condition})
    else:
        for _rep in range(n_repeats):
            for tmpl in templates:
                qid = tmpl['qid']
                if task == 'continuous':
                    suffix = moment_match_continuous(
                        rng, tmpl['true_mean'], tmpl['true_std'], suffix_length,
                        max_rescale_iters=args.max_rescale_iters)
                else:
                    suffix = moment_match_binary(rng, tmpl['true_p'], suffix_length)
                rep_idx = rep_count.get(qid, 0)
                iti_condition = iti_sched[qid][rep_idx]
                rep_count[qid] = rep_idx + 1
                trials.append({**tmpl, 'values': tmpl['prefix'] + suffix,
                               'iti_ms': ITI_MS, 'iti_condition': iti_condition})

    trials = _shuffle_no_consecutive_qid(trials, rng, task)

    # -- Build DataFrame + JSON (same schema as generate_sequences.py) -------
    records, json_trials = [], []
    for t, trial in enumerate(trials):
        for o, v in enumerate(trial['values'], start=1):
            records.append({
                'trial': t, 'qid': trial['qid'],
                'observation': o, 'value': v,
                'true_mean': trial['true_mean'],
                'true_std': trial['true_std'],
                'true_p': trial['true_p'],
                'iti_ms': trial['iti_ms'],
                'iti_condition': trial['iti_condition'],
            })
        json_trials.append({
            'trial': t, 'qid': trial['qid'],
            'true_mean': None if math.isnan(trial['true_mean']) else trial['true_mean'],
            'true_std': None if math.isnan(trial['true_std']) else trial['true_std'],
            'true_p':    None if math.isnan(trial['true_p'])    else trial['true_p'],
            'values': trial['values'], 'prefix_length': prefix_length,
            'iti_ms': trial['iti_ms'],
            'iti_condition': trial['iti_condition'],
        })

    df = pd.DataFrame(records)

    # -- Sanity checks (same as generate_sequences.py) ------------------------
    assert len(df) == n_total * seq_length
    if task == 'continuous':
        assert df['value'].between(VALUE_MIN, VALUE_MAX).all()
    else:
        assert df['value'].isin([-1, 1]).all()
    assert df['trial'].nunique() == n_total
    assert (df.groupby('qid')['trial'].nunique() == n_repeats).all()

    print(f"\n  {len(df)} rows | {n_total} trials | {n_unique} seqs x {n_repeats} reps")
    if task == 'continuous':
        print(f"  Value range : {df['value'].min()} - {df['value'].max()}")
        print(f"  true_mean   : {df['true_mean'].min():.1f} - {df['true_mean'].max():.1f}")
    else:
        print(f"  true_p range: {df['true_p'].min():.3f} - {df['true_p'].max():.3f}")

    return df, json_trials


# ---------------------------------------------------------------------------
# Diagnostic report: achieved vs target moments, per qid
# ---------------------------------------------------------------------------
def report_moments(task, args, rng):
    """Print achieved vs target mean/std (continuous) or p (binary) per qid,
    plus the k=1.0 plausibility verdict from generate_sequences.py's own
    check (diagnostic only -- this script never gates on it)."""
    df, _ = generate_task_sequences_momentmatch(task, args, rng)
    print(f"\n{'='*60}\nMoment report ({task}):")
    for qid, g in df.groupby('qid'):
        for trial_id, tdf in g.groupby('trial'):
            vals = tdf.sort_values('observation')['value'].tolist()
            tm, tp = tdf['true_mean'].iloc[0], tdf['true_p'].iloc[0]
            true_std_arg = args.std_fixed if task == 'continuous' else float('nan')
            ok = check_sequence_plausibility(
                vals, task, tm, true_std_arg, tp,
                k=1.0, k_std=args.k_std_cont)
            if task == 'continuous':
                print(f"  qid={qid:2d} trial={trial_id:3d}  target mean={tm:6.1f} "
                      f"std={args.std_fixed:5.1f}  achieved mean={np.mean(vals):6.2f} "
                      f"std={np.std(vals):5.2f}  k=1.0 pass={ok}")
            else:
                print(f"  qid={qid:2d} trial={trial_id:3d}  target p={tp:.3f}  "
                      f"achieved p={np.mean(np.array(vals) == 1):.3f}  k=1.0 pass={ok}")


# ---------------------------------------------------------------------------
# Seed search -- score_mode selects 'bump' (original, gated) or 'isotonic'
# (default, ungated PAVA-residual ranking)
# ---------------------------------------------------------------------------
def _seed_search_momentmatch(task, args, out_dir):
    print(f"\n{'='*55}")
    print(f"Seed search (moment-matched, score_mode={args.score_mode}): {task.upper()} | "
          f"{args.n_tries} tries | RL_lambda alpha={args.rl_alpha_0} lambda={args.rl_lambda}")
    if args.score_mode == 'bump':
        print("Score = weighted |Delta response| rises (lower -> more monotone decay)")
        print("  Generation always succeeds by construction -- searching over "
              "realizations/orderings for smoothness only, no structural gate.")
    else:
        print("Score = isotonic-regression residual (lower -> closer to a smooth "
              "monotone decay of ANY shape, not just non-rising)")
        print("  No gate -- every seed ranked by bay_resid + rl_resid, lowest wins.")

    best_score, best_scores, best_seed, best_df, best_json = \
        np.inf, (np.inf, np.inf), None, None, None
    all_scores = []
    for attempt in range(args.n_tries):
        task_seed = attempt if task == 'continuous' else attempt + 1000
        rng = make_rng(task_seed)
        df, json_trials = generate_task_sequences_momentmatch(task, args, rng)

        if args.score_mode == 'bump':
            bay, rl, combined = score_sequences(
                df, task,
                rl_alpha_0=args.rl_alpha_0, rl_lambda=args.rl_lambda,
                gamma_bay_delta_cont=args.gamma_bay_delta_cont,
                gamma_bay_rmse_cont=args.gamma_bay_rmse_cont,
                gamma_rl_delta_cont=args.gamma_rl_delta_cont,
                gamma_rl_rmse_cont=args.gamma_rl_rmse_cont,
                gamma_bay_delta_bin=args.gamma_bay_delta_bin,
                gamma_bay_rmse_bin=args.gamma_bay_rmse_bin,
                gamma_rl_delta_bin=args.gamma_rl_delta_bin,
                gamma_rl_rmse_bin=args.gamma_rl_rmse_bin)
            all_scores.append(combined)
            if bay > 2e-2:
                continue  # pass-2 smoothness gate, bump mode only
        else:
            bay, rl, combined = score_sequences_isotonic(
                df, task, rl_alpha_0=args.rl_alpha_0, rl_lambda=args.rl_lambda)
            all_scores.append(combined)
            # no gate in isotonic mode -- ranking alone selects smooth curves

        if combined < best_score:
            best_score, best_scores, best_seed = combined, (bay, rl), attempt
            best_df, best_json = df.copy(), json_trials
            _save_sequences(df, json_trials, f'{task}_momentmatch', out_dir)
        if (attempt + 1) % 20 == 0 or attempt == args.n_tries - 1:
            print(f"  [{attempt+1:4d}/{args.n_tries}]  best_seed={best_seed}  "
                  f"combined={best_score:.5f}")

    arr = np.array(all_scores) if all_scores else np.array([np.inf])
    print(f"\n  Best seed  : {best_seed}")
    if args.score_mode == 'bump':
        print(f"  Bay score  : {best_scores[0]:.6f}  (delta+rmse, gate: < 0.02)")
        print(f"  RL  score  : {best_scores[1]:.6f}  (delta+rmse)")
        print(f"  Combined   : {best_score:.5f}  (median {np.median(arr):.5f})")
    else:
        print(f"  Bay resid  : {best_scores[0]:.3e}  (isotonic, no gate)")
        print(f"  RL  resid  : {best_scores[1]:.3e}  (isotonic)")
        print(f"  Combined   : {best_score:.3e}  (median {np.median(arr):.3e})")
    print(f"  Saved to   : {out_dir}/{task}_momentmatch_sequences.{{pkl,json}}")


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
    p.add_argument('--std_fixed',          type=float, default=15.0)
    p.add_argument('--p_range',            type=float, nargs=2, default=[0.2, 0.8])
    p.add_argument('--max_rescale_iters',  type=int,   default=4,
                   help='Rescale+clip iterations to converge onto target moments')
    p.add_argument('--rl_alpha_0',         type=float, default=1.0)
    p.add_argument('--rl_lambda',          type=float, default=0.5)
    p.add_argument('--k_std_cont',         type=float, default=0.7,
                   help='Diagnostic only (--report) -- not used to gate generation')
    p.add_argument('--gamma_bay_delta_cont', type=float, default=0.7)
    p.add_argument('--gamma_bay_rmse_cont',  type=float, default=0.3)
    p.add_argument('--gamma_rl_delta_cont',  type=float, default=0.7)
    p.add_argument('--gamma_rl_rmse_cont',   type=float, default=0.3)
    p.add_argument('--gamma_bay_delta_bin',  type=float, default=0.3)
    p.add_argument('--gamma_bay_rmse_bin',   type=float, default=0.0)
    p.add_argument('--gamma_rl_delta_bin',   type=float, default=0.3)
    p.add_argument('--gamma_rl_rmse_bin',    type=float, default=0.0)
    p.add_argument('--score_mode', choices=list(SCORE_MODES), default='isotonic',
                   help="'bump' = penalize rises only (original); "
                        "'isotonic' = penalize any deviation from a smooth "
                        "monotone decay of any shape (no gate, pure ranking)")
    p.add_argument('--output_dir',         default='task/sequences')
    p.add_argument('--seed',               type=int,   default=0)
    p.add_argument('--report',             action='store_true',
                   help='Print achieved-vs-target moments per trial; no file written')
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

    if args.report:
        for task in tasks:
            rng = make_rng(args.seed if task == 'continuous' else args.seed + 1000)
            report_moments(task, args, rng)
    elif args.n_tries > 1:
        for task in tasks:
            _seed_search_momentmatch(task, args, out_dir)
    else:
        print(f"Generating sequences (moment-matched) | seed={args.seed}")
        for task in tasks:
            task_seed = args.seed if task == 'continuous' else args.seed + 1000
            rng = make_rng(task_seed)
            df, json_trials = generate_task_sequences_momentmatch(task, args, rng)
            pkl_path, json_path = _save_sequences(df, json_trials, f'{task}_momentmatch', out_dir)
            print(f"\n  Saved: {json_path}\n  Saved: {pkl_path}")
    print("\nJOB_COMPLETE")


if __name__ == '__main__':
    main()
