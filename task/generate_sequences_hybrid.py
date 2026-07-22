"""
generate_sequences_hybrid.py
==============================
ROLE: the PRODUCTION method chosen after PI discussion, replacing a single
uniform method with per-task choices -- see docs/sequence_design_open_
questions.md for the full investigation this decision is based on.

  Binary:     unchanged quota/momentmatch construction (prefix/target
              independence, optimal matching, exact-quota suffix fill),
              but NO seed search -- single draw, no curve-shape selection.
  Continuous: same prefix/target construction and optimal matching, but
              the suffix is a genuine, UNRESCALED i.i.d. draw centered on
              the algebraic residual mean needed to keep the pooled
              sequence unbiased -- not forced to hit it via the iterative
              rescale generate_sequences_momentmatch.py uses. No seed
              search either.

Why binary keeps quota's construction but continuous doesn't
---------------------------------------------------------------
This is a deliberate, asymmetric choice, not an oversight: the PI decided
quota's construction is acceptable for binary (kept as-is) but "too
heavy-handed" for continuous specifically, given concern about biasing
participants toward a specific cognitive strategy (see chat history).

Confirmed empirically before this script was written (see docs/
sequence_design_open_questions.md, and the chat that led to this file):
  - Dropping ONLY the seed search (keeping the rescale) does almost
    nothing -- single-shot moment-matched continuous sequences showed
    split-half reliability r=0.995 and a back-half-corrects-front-half
    correlation of r=0.996, both nearly identical to the full 1000-try
    seed-searched production (r=1.000 / r=0.998). The seed search was
    never the primary source of either effect.
  - The actual source is the iterative RESCALE step in
    suffix_for_continuous_target -- forcing the suffix's own sample mean/
    std to hit the residual target is what makes the trial's ending
    correct toward the truth almost regardless of the front half. Removing
    the rescale (this file's suffix_for_continuous_target_iid) while
    keeping the residual-mean CENTERING (so the pooled sequence stays
    unbiased in expectation, just genuinely noisy) dropped that
    correlation to r=0.821 -- a real, substantial reduction, confirmed
    directly, not assumed.
  - Split-half reliability itself (sweeping many hypothetical true-lambda
    values across the SAME fixed 40-trial set) does NOT improve with this
    change (stayed at r=0.9994) -- that number reflects the sequence set's
    own target-diversity/balance across trials, which is untouched by how
    any single suffix gets filled in. Accepted explicitly by the PI as an
    unavoidable property of true i.i.d. sampling, not something this
    design needs to fix.
  - Binary's exact-quota construction has NO equivalent "rescale" step to
    remove at all -- the suffix already fills an exact residual quota by
    construction, so there's nothing softer to fall back to; the only
    knob available for binary was the seed search, which is what's being
    dropped here.

What this means concretely: continuous trials are no longer guaranteed to
land close to their stated true_mean/true_std -- they land there in
EXPECTATION (unbiased, since prefix mismatch is still corrected for via
residual centering) but with genuine sampling noise on top, comparable in
spirit to true i.i.d.. Binary trials remain EXACT, as they always have
been (quota is inherently exact regardless of search).

Reuses generate_sequences_momentmatch.py's prefix/target generation,
optimal matching, ITI scheduling, and trial-shuffling machinery directly
(imported, not duplicated) -- only the continuous suffix-construction step
is new. See that file's own module docstring for the full rationale behind
everything reused here (prefix/target independence, the collision bug it
fixes, why optimal matching replaced greedy selection, etc.) -- this file
assumes that context rather than repeating it.

No seed search anywhere in this file, for either task, by design -- if
one is ever wanted again, generate_sequences_momentmatch.py (quota, both
tasks) and generate_sequences_iid.py (pure i.i.d., both tasks) remain
fully intact and re-runnable; this file is a third, separate option, not
a replacement for either.

Usage
-----
  python task/generate_sequences_hybrid.py --task both --seed 0
  python task/generate_sequences_hybrid.py --task continuous --seed 0 \\
      --n_prefix 10 --mean_range 15 85 --n_repeats 4 --boundary_margin 1
  python task/generate_sequences_hybrid.py --task binary --seed 0 \\
      --n_prefix 10 --blue_range 2 13 --n_repeats 4
  python task/generate_sequences_hybrid.py --task continuous --report
      # diagnostic: achieved-vs-target moments per trial, no file written

Output
------
  Saved to task/sequences/{task}_hybrid_sequences.{pkl,json} by default --
  deliberately NOT the production filenames, matching the same
  never-clobber-silently convention as generate_sequences_momentmatch.py.
  Promote by copying over {task}_sequences.{pkl,json} once verified.
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import pandas as pd

from generate_sequences import VALUE_MIN, VALUE_MAX, make_rng, _save_sequences
from generate_sequences_momentmatch import (
    build_continuous_prefixes,
    build_binary_prefixes,
    continuous_target_means,
    binary_target_blue_counts,
    binary_feasible_target_range,
    optimal_matching,
    suffix_for_binary_target,
    moment_match_binary,
    _build_iti_schedule,
    _shuffle_no_consecutive_qid,
)


# ---------------------------------------------------------------------------
# The genuinely new piece: i.i.d. (unrescaled) continuous suffix, with an
# analytical variance correction plus a loose std safety-net rejection.
# ---------------------------------------------------------------------------
def suffix_for_continuous_target_iid(rng, prefix_values, target_mean, std_fixed,
                                     suffix_length, value_min=VALUE_MIN, value_max=VALUE_MAX,
                                     std_tolerance_frac=0.25, max_std_attempts=20):
    """Suffix's OWN target mean = the same algebraic residual
    generate_sequences_momentmatch.py's suffix_for_continuous_target uses
    (so the pooled sequence stays unbiased in expectation regardless of
    how far the prefix's own mean is from the trial's target) --

        residual_mean = (target_mean * seq_length - prefix_sum) / suffix_length

    but UNLIKE that function, the raw draw is taken essentially as-is --
    NO iterative rescale of the suffix's own realized values, ever. That
    single difference is what actually restores genuine trial-ending
    uncertainty (confirmed empirically, see module docstring): the
    back-half-corrects-front-half correlation dropped from quota's
    ~0.996-0.998 to i.i.d.-like ~0.51-0.82 once the rescale was removed,
    and stayed there even with the two guards below added -- neither guard
    touches the mean-convergence property, only the variance.

    Two guards on the SUFFIX'S OWN VARIANCE PARAMETER (not on individual
    realized values, and not a rescale of the result):

    1. Analytical bias correction. A prefix far from target_mean inflates
       the pooled sequence's expected variance (a between-block mean-shift
       contribution, same phenomenon generate_sequences_momentmatch.py's own
       docstring flags for its rescale-based suffix). Solving
       E[pooled_var] = std_fixed**2 for the suffix's OWN variance parameter
       sigma2**2, given the prefix is already fixed/known and the residual
       mean is already chosen:

         sigma2**2 = (seq_length/suffix_length) * std_fixed**2
                     - prefix_sse/suffix_length - (residual_mean - target_mean)**2

       where prefix_sse = sum((x - target_mean)**2 for x in prefix). Floored
       at a small positive minimum (1.0) if a badly-matched prefix would
       otherwise drive this negative -- confirmed via testing that optimal
       matching keeps this rare, not a routine fallback. This is a
       first-moment-style correction (fixes the EXPECTED value, like the
       mean's residual centering already does) -- confirmed empirically to
       only partially close the achieved-std gap (~11-13% reduction in mean/
       max |error| in testing), because estimating a standard deviation from
       only suffix_length observations is inherently noisy regardless of how
       well its expectation is centered -- this guard fixes bias, not spread.

    2. Loose safety-net rejection on the FULL 15-observation achieved std
       (not the mean, not any per-observation property): redraw the whole
       suffix (fresh i.i.d. draw from the SAME sigma2, not a rescale of the
       rejected one) up to max_std_attempts times if the achieved std falls
       outside std_fixed * (1 +/- std_tolerance_frac). Confirmed empirically
       across 5 seeds that this does NOT meaningfully affect the mean-related
       confound at the default (average correlation 0.54-0.57 at +/-25-33.3%,
       vs quota's ~0.996-0.998 and pure i.i.d.'s own ~0.61) -- it's an
       orthogonal intervention on a different aggregate statistic, not the
       same lever in disguise. std_tolerance_frac is a real, tunable knob
       with a genuine trade-off, checked directly across +/-33.3%/25%/15%
       (5 seeds each, chat history has the full table): tightening from
       +/-33.3% to +/-25% is close to free (correlation +0.03 on average,
       barely above seed-to-seed noise) while meaningfully tightening std
       control (mean |std error| improves ~12%, max ~18%); tightening
       further to +/-15% costs much more (correlation +0.13 on average,
       every single seed tested moved in the same direction) for a smaller
       marginal std improvement. +/-25% (this default) was chosen as the
       point where the curve's cost/benefit bends -- not the tightest
       option tested, deliberately. If max_std_attempts is exhausted, the
       LAST attempt is used as-is (no infinite loop, no silent failure) --
       intentionally rare at this tolerance; if --report ever shows this
       triggering often, that's a sign of a mismatched prefix/target
       pairing, not something to loosen the tolerance further to paper over.
    """
    prefix_sum = sum(prefix_values)
    seq_length = len(prefix_values) + suffix_length
    residual_mean = (target_mean * seq_length - prefix_sum) / suffix_length

    prefix_sse = sum((x - target_mean) ** 2 for x in prefix_values)
    sigma2_sq = ((seq_length / suffix_length) * std_fixed ** 2
                 - prefix_sse / suffix_length
                 - (residual_mean - target_mean) ** 2)
    sigma2 = float(np.sqrt(max(sigma2_sq, 1.0)))

    std_lo = std_fixed * (1.0 - std_tolerance_frac)
    std_hi = std_fixed * (1.0 + std_tolerance_frac)
    vals = None
    for _attempt in range(max_std_attempts):
        raw = rng.normal(residual_mean, sigma2, size=suffix_length)
        vals = np.clip(np.round(raw), value_min, value_max).astype(int)
        full_seq = list(prefix_values) + vals.tolist()
        if std_lo <= float(np.std(full_seq)) <= std_hi:
            break
    return vals.tolist()


# ---------------------------------------------------------------------------
# NO-PREFIX branch (binary only, chat history) -- a SEPARATE generation
# path alongside generate_task_sequences_hybrid's existing prefix/qid-
# repeat structure, not a replacement for it. Motivation: that structure's
# prefix-composition allocation (_allocate_binary_composition_counts) is
# DETERMINISTIC -- every pool member gets the exact same {0:1,1:2,2:2,2:2,
# 4:1} blue-count split across its 8 prefixes, only the specific
# arrangements within each level vary. This collapses the |Delta response|
# curve's between-participant diversity on the prefix portion (observation
# <= prefix_length) almost down to the handful of distinct values that
# fixed split can produce, confirmed empirically (chat history) to jump
# from 3 distinct per-member averages at obs=4 (deterministic allocator) to
# 9 (randomized allocator, same capacity constraints). This branch goes
# further: no prefix/qid-repeat concept AT ALL. Each of n_total trials
# gets its OWN independently-drawn true_p (still spread evenly across
# blue_range via binary_target_blue_counts, unchanged) and its OWN
# independently-built, exact-quota FULL seq_length sequence
# (moment_match_binary) -- no qid ever repeats, so there's no "prefix" to
# even define length-4 uniqueness against in the first place (see the
# separate per-qid prefix-repetition investigation this was built to
# support).
# ---------------------------------------------------------------------------
def generate_binary_sequences_no_prefix(args, rng, verbose=True):
    """Binary, no-prefix branch. Returns (DataFrame, json_trials), SAME
    schema as generate_task_sequences_hybrid -- qid is set to the trial's
    own index (never repeated) and prefix_length is recorded as 0, so
    every existing consumer of this schema (verify_pool's prefix-
    uniqueness check, parse_results.py, tutorial-example selection, etc.)
    keeps working unmodified: a qid with exactly one trial and an empty
    (`values[:0]`) "prefix" is trivially unique, not a special case that
    needs separate handling downstream.
    """
    seq_length = args.seq_length
    n_total    = args.n_prefix * args.n_repeats  # same trial-count formula
    # as the prefix branch, so a no-prefix pool member has the same number
    # of trials as a prefix-branch one -- apples-to-apples pool sizing,
    # not a different experiment length by accident.

    if verbose:
        print(f"\n{'='*60}")
        print("Task: BINARY  (hybrid, NO-PREFIX branch: every trial independent)")
        print(f"  Total trials   : {n_total} (no prefix/qid-repeat structure)")
        print(f"  Seq length     : {seq_length}")

    target_blue = binary_target_blue_counts(rng, n_total, args.blue_range)
    if verbose:
        from collections import Counter
        print(f"  target blue-count distribution: {dict(sorted(Counter(target_blue).items()))}")

    ITI_MS = 1000
    # No qid/prefix identity to key an ITI schedule off of anymore --
    # _build_iti_schedule assumes per-qid repeats, so build a flat
    # half-control/half-distract schedule directly instead.
    iti_pool = (['control'] * (n_total // 2) + ['distract'] * (n_total - n_total // 2))
    rng.shuffle(iti_pool)

    trials = []
    for i, target in enumerate(target_blue):
        true_p = round(target / seq_length, 6)
        values = moment_match_binary(rng, true_p, seq_length)
        trials.append({
            'qid': i, 'true_mean': float('nan'), 'true_std': float('nan'),
            'true_p': true_p, 'values': values,
            'iti_ms': ITI_MS, 'iti_condition': iti_pool[i],
        })

    # Every qid is already unique, so there's no "no consecutive qid"
    # constraint to enforce -- a plain shuffle is sufficient.
    rng.shuffle(trials)

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
            'true_mean': None, 'true_std': None,
            'true_p': trial['true_p'],
            'values': trial['values'], 'prefix_length': 0,
            'iti_ms': trial['iti_ms'],
            'iti_condition': trial['iti_condition'],
        })

    df = pd.DataFrame(records)

    assert len(df) == n_total * seq_length
    assert df['value'].isin([-1, 1]).all()
    assert df['trial'].nunique() == n_total
    assert df['qid'].nunique() == n_total, "no-prefix branch must give every trial its own unique qid"
    achieved_blue = df.groupby('trial')['value'].apply(lambda s: (s == 1).sum())
    expected_blue = (df.groupby('trial')['true_p'].first() * seq_length).round()
    assert np.array_equal(achieved_blue.reindex(expected_blue.index).values,
                          expected_blue.values), (
        "binary quota mismatch -- moment_match_binary should be exact by construction.")

    if verbose:
        print(f"\n  {len(df)} rows | {n_total} trials, each with its own unique qid")
        print(f"  true_p range: {df['true_p'].min():.3f} - {df['true_p'].max():.3f}")

    return df, json_trials


# ---------------------------------------------------------------------------
# Core generation -- mirrors generate_task_sequences_momentmatch's structure
# exactly (prefix/target generation, optimal matching, ITI scheduling, trial
# shuffling all reused unchanged); only the continuous suffix step differs.
# ---------------------------------------------------------------------------
def generate_task_sequences_hybrid(task, args, rng, verbose=True):
    """Generate all trials for one task via the hybrid method:
    quota/momentmatch construction for binary, i.i.d.-suffix construction
    for continuous -- see module docstring for the full rationale. No seed
    search for either task. Returns (DataFrame, json_trials), same schema
    as every other sequence generator in this project.
    """
    seq_length    = args.seq_length
    prefix_length = args.prefix_length
    suffix_length = seq_length - prefix_length
    if prefix_length >= seq_length:
        raise NotImplementedError(
            "prefix_length >= seq_length (no suffix at all) is not supported -- "
            "see generate_sequences_momentmatch.py's identical guard for why.")

    n_prefix  = args.n_prefix
    n_repeats = args.n_repeats
    n_total   = n_prefix * n_repeats

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}  (hybrid: "
              f"{'quota, no search' if task == 'binary' else 'i.i.d. suffix, no rescale'})")
        print(f"  Total trials   : {n_total} ({n_prefix} prefixes x {n_repeats} reps)")
        print(f"  Seq / prefix / suffix length: {seq_length} / {prefix_length} / {suffix_length}")

    # -- Build n_prefix DISTINCT prefixes, independent of any target --------
    if task == 'continuous':
        cont_value_min = VALUE_MIN + args.boundary_margin
        cont_value_max = VALUE_MAX - args.boundary_margin
        prefixes = build_continuous_prefixes(rng, n_prefix, prefix_length,
                                             args.mean_range, args.std_fixed,
                                             value_min=cont_value_min, value_max=cont_value_max)
    else:
        prefixes = build_binary_prefixes(rng, n_prefix, prefix_length)

    if verbose:
        for i, pfx in enumerate(prefixes):
            print(f"  prefix {i}: [{','.join(map(str, pfx))}]")

    # -- Build n_total target values, independent of prefix, no forced repeats
    if task == 'continuous':
        target_means = continuous_target_means(n_total, args.mean_range)
        if verbose:
            print(f"  target means ({n_total}): {target_means}")
    else:
        target_blue = binary_target_blue_counts(rng, n_total, args.blue_range)
        if verbose:
            from collections import Counter
            print(f"  target blue-count distribution: {dict(sorted(Counter(target_blue).items()))}")

    # -- Prefix-slots + optimal matching (unchanged from momentmatch) -------
    prefix_slot_idx = [i for i in range(n_prefix) for _ in range(n_repeats)]
    rng.shuffle(prefix_slot_idx)

    if task == 'continuous':
        prefix_means = [float(np.mean(pfx)) for pfx in prefixes]
        prefix_slot_values = [prefix_means[i] for i in prefix_slot_idx]
        matched_targets = optimal_matching(prefix_slot_values, target_means)
    else:
        prefix_slot_blue = [sum(1 for v in prefixes[i] if v == 1) for i in prefix_slot_idx]
        matched_targets = optimal_matching(
            prefix_slot_blue, target_blue,
            feasible_fn=lambda pv, tv: (
                binary_feasible_target_range(pv, suffix_length, args.blue_range)[0]
                <= tv <=
                binary_feasible_target_range(pv, suffix_length, args.blue_range)[1]))

    # -- ITI schedule (unchanged from momentmatch) ---------------------------
    ITI_MS = 1000
    prefix_templates = [{'qid': i} for i in range(n_prefix)]
    iti_sched = _build_iti_schedule(rng, prefix_templates, n_repeats)
    rep_count = {}

    # -- Build each trial: prefix + task-specific suffix ---------------------
    trials = []
    for pfx_idx, target in zip(prefix_slot_idx, matched_targets):
        prefix_vals = prefixes[pfx_idx]
        rep_idx = rep_count.get(pfx_idx, 0)
        iti_condition = iti_sched[pfx_idx][rep_idx]
        rep_count[pfx_idx] = rep_idx + 1

        if task == 'continuous':
            true_mean, true_std, true_p = target, args.std_fixed, float('nan')
            # THE difference from generate_sequences_momentmatch.py: i.i.d.
            # suffix, no rescale -- plus bias-correction and a loose std
            # safety net -- see module docstring.
            suffix = suffix_for_continuous_target_iid(
                rng, prefix_vals, true_mean, true_std, suffix_length,
                value_min=VALUE_MIN + args.boundary_margin,
                value_max=VALUE_MAX - args.boundary_margin,
                std_tolerance_frac=args.std_tolerance_frac)
        else:
            true_p = round(target / seq_length, 6)
            true_mean, true_std = float('nan'), float('nan')
            # Unchanged from generate_sequences_momentmatch.py: exact quota.
            suffix = suffix_for_binary_target(rng, prefix_vals, target, suffix_length)

        trials.append({
            'qid': pfx_idx, 'true_mean': true_mean, 'true_std': true_std,
            'true_p': true_p, 'values': prefix_vals + suffix,
            'iti_ms': ITI_MS, 'iti_condition': iti_condition,
        })

    trials = _shuffle_no_consecutive_qid(trials, rng, task)

    # -- Build DataFrame + JSON (same schema as every other generator) ------
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

    # -- Sanity checks --------------------------------------------------------
    # Same structural checks as generate_sequences_momentmatch.py. NOTE: no
    # achieved-mean/std tolerance check for continuous -- that's the whole
    # point of this file, the achieved value is now genuinely noisy, not a
    # bug to assert against. Binary's exact-quota check is unchanged and
    # still fully deterministic.
    assert len(df) == n_total * seq_length
    if task == 'continuous':
        assert df['value'].between(VALUE_MIN, VALUE_MAX).all()
    else:
        assert df['value'].isin([-1, 1]).all()
    assert df['trial'].nunique() == n_total
    assert (df.groupby('qid')['trial'].nunique() == n_repeats).all()
    n_distinct_prefixes = len(set(tuple(p) for p in prefixes))
    assert n_distinct_prefixes == n_prefix, (
        f"prefix collision: {n_distinct_prefixes}/{n_prefix} distinct -- "
        f"should be unreachable given build_{task}_prefixes' own dedup.")
    if task == 'binary':
        achieved_blue = df.groupby('trial')['value'].apply(lambda s: (s == 1).sum())
        expected_blue = (df.groupby('trial')['true_p'].first() * seq_length).round()
        assert np.array_equal(achieved_blue.reindex(expected_blue.index).values,
                              expected_blue.values), (
            "binary quota mismatch -- suffix residual math has a bug, this should "
            "be impossible given exact-quota construction.")

    if verbose:
        print(f"\n  {len(df)} rows | {n_total} trials | {n_prefix} prefixes x {n_repeats} reps")
        if task == 'continuous':
            achieved_means = df.groupby('trial')['value'].mean()
            target_by_trial = df.groupby('trial')['true_mean'].first()
            errs = (achieved_means - target_by_trial).abs()
            print(f"  Value range : {df['value'].min()} - {df['value'].max()}")
            print(f"  true_mean   : {df['true_mean'].min():.1f} - {df['true_mean'].max():.1f}")
            print(f"  achieved-vs-target |mean error|: mean={errs.mean():.3f} max={errs.max():.3f} "
                  f"(genuinely noisy by design -- see module docstring)")
        else:
            print(f"  true_p range: {df['true_p'].min():.3f} - {df['true_p'].max():.3f}")

    return df, json_trials


# ---------------------------------------------------------------------------
# Diagnostic report: achieved vs target moments, per trial
# ---------------------------------------------------------------------------
def report_moments(task, args, rng):
    """Print achieved vs target mean/std (continuous) or p (binary) per
    trial. Purely diagnostic -- generation never gates on this, for either
    task (binary is exact by construction; continuous is deliberately
    noisy by design -- there is nothing to gate)."""
    df, _ = generate_task_sequences_hybrid(task, args, rng)
    print(f"\n{'='*60}\nMoment report ({task}):")
    for qid, g in df.groupby('qid'):
        for trial_id, tdf in g.groupby('trial'):
            vals = tdf.sort_values('observation')['value'].tolist()
            tm, tp = tdf['true_mean'].iloc[0], tdf['true_p'].iloc[0]
            if task == 'continuous':
                print(f"  qid={qid:2d} trial={trial_id:3d}  target mean={tm:6.1f} "
                      f"std={args.std_fixed:5.1f}  achieved mean={np.mean(vals):6.2f} "
                      f"std={np.std(vals):5.2f}")
            else:
                print(f"  qid={qid:2d} trial={trial_id:3d}  target p={tp:.3f}  "
                      f"achieved p={np.mean(np.array(vals) == 1):.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--task',          choices=['continuous', 'binary', 'both'], default='both')
    p.add_argument('--n_repeats',     type=int,   default=4,
                   help='How many times each of the n_prefix distinct prefixes is used. '
                        'n_trials = n_prefix * n_repeats.')
    p.add_argument('--seq_length',    type=int,   default=15)
    p.add_argument('--prefix_length', type=int,   default=4)
    p.add_argument('--n_prefix',      type=int,   default=10,
                   help='Number of DISTINCT prefixes -- see '
                        'generate_sequences_momentmatch.py\'s module docstring for '
                        'the full rationale (reused unchanged here).')
    p.add_argument('--mean_range',    type=float, nargs=2, default=[15.0, 85.0])
    p.add_argument('--std_fixed',     type=float, default=15.0)
    p.add_argument('--blue_range',    type=int,   nargs=2, default=[2, 13])
    p.add_argument('--boundary_margin', type=float, default=1.0,
                   help='Continuous only: insets the [0,100] clip range to '
                        '[margin, 100-margin]. Matters more here than in '
                        'generate_sequences_momentmatch.py, since there is no '
                        'iterative rescale to claw back boundary-clipping bias -- '
                        'default changed to 1.0 (vs that file\'s 0.0) accordingly.')
    p.add_argument('--std_tolerance_frac', type=float, default=0.25,
                   help='Continuous only: loose safety-net rejection tolerance on the '
                        'FULL 15-obs achieved std, as a fraction of std_fixed (default '
                        '0.25 = +/-25%%, chosen as the cost/benefit bend point -- see '
                        'suffix_for_continuous_target_iid\'s docstring for the full '
                        '+/-33.3%%/25%%/15%% comparison). Redraws the whole suffix (fresh '
                        'i.i.d., not a rescale) if achieved std falls outside this band -- '
                        'confirmed empirically not to reintroduce the mean-related '
                        'confound this design otherwise avoids.')
    p.add_argument('--output_dir',    default='task/sequences')
    p.add_argument('--seed',          type=int,   default=0)
    p.add_argument('--report',        action='store_true',
                   help='Print achieved-vs-target moments per trial; no file written')
    p.add_argument('--no_prefix',     action='store_true',
                   help='Binary ONLY (chat history) -- use generate_binary_sequences_no_prefix '
                        'instead of the default prefix/qid-repeat structure. Every trial gets '
                        'its own independent true_p and its own independent exact-quota full-'
                        'length sequence; no prefix, no qid repeats at all. A SEPARATE branch, '
                        'not a replacement -- omit this flag to keep using the existing prefix '
                        'scheme. Errors if --task is continuous or both (this branch has no '
                        'continuous equivalent).')
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n_repeats > 0
    assert args.seq_length > 0
    assert 0 <= args.prefix_length <= args.seq_length
    assert args.std_fixed > 0
    assert args.n_prefix >= 1
    assert args.mean_range[0] < args.mean_range[1]
    assert 0 <= args.mean_range[0] and args.mean_range[1] <= 100
    assert args.blue_range[0] < args.blue_range[1]
    assert 1 <= args.blue_range[0] and args.blue_range[1] <= args.seq_length - 1, \
        'blue_range must fall within [1, seq_length-1] (exclude degenerate all-one-color quotas)'
    assert 0 <= args.boundary_margin < (VALUE_MAX - VALUE_MIN) / 2, \
        'boundary_margin must leave a non-empty usable range'

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = ['continuous', 'binary'] if args.task == 'both' else [args.task]

    if args.no_prefix:
        assert args.task == 'binary', (
            "--no_prefix is binary-only (chat history) -- no continuous equivalent exists. "
            "Pass --task binary explicitly.")
        assert not args.report, "--report is not supported for --no_prefix yet"
        print(f"Generating sequences (hybrid, NO-PREFIX branch) | seed={args.seed}")
        rng = make_rng(args.seed + 1000)
        df, json_trials = generate_binary_sequences_no_prefix(args, rng)
        pkl_path, json_path = _save_sequences(df, json_trials, 'binary_hybrid_noprefix', out_dir)
        print(f"\n  Saved: {json_path}\n  Saved: {pkl_path}")
        print("\nJOB_COMPLETE")
        return

    if args.report:
        for task in tasks:
            rng = make_rng(args.seed if task == 'continuous' else args.seed + 1000)
            report_moments(task, args, rng)
    else:
        print(f"Generating sequences (hybrid) | seed={args.seed}")
        for task in tasks:
            task_seed = args.seed if task == 'continuous' else args.seed + 1000
            rng = make_rng(task_seed)
            df, json_trials = generate_task_sequences_hybrid(task, args, rng)
            pkl_path, json_path = _save_sequences(df, json_trials, f'{task}_hybrid', out_dir)
            print(f"\n  Saved: {json_path}\n  Saved: {pkl_path}")
    print("\nJOB_COMPLETE")


if __name__ == '__main__':
    main()
