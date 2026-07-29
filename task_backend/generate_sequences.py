"""
generate_sequences.py
======================
Builds the pool of trial sequences served to participants. Each participant
is assigned one independently-generated pool member (deterministic hash of
their ID -- see src/shared/timeline-builder.js's poolIndexForParticipant),
so no two participants see the exact same 32 trials even though every
member is drawn from the same underlying design.

Two tasks, two different construction strategies -- read top to bottom for
the full pipeline:

  NUMBERS ("Numbers"): each trial = a short, shared 4-observation PREFIX
    (one of 8 fixed prefixes, spread across the target mean range, each
    reused across 4 trials) + an 11-observation SUFFIX drawn to steer the
    trial's overall sample mean toward an assigned target -- via a genuine
    unbiased random draw, NOT forced to hit the target exactly. A trial's
    running average genuinely gets more accurate as it goes on (real
    evidence accumulation), rather than being artificially perfect by the
    end.

  COLORS ("Colors"): each of the 32 trials is fully independent -- its own
    target proportion, its own EXACT-quota 15-observation sequence (e.g.
    target 60% blue -> exactly 9 of 15 observations are blue, order
    shuffled). No prefix, no repeated qid.

Consolidated from five older scripts that had accumulated several
abandoned methods (pure i.i.d. sampling, quota+seed-search, moment-
matching+iterative-rescale) and a seed-search/smoothness-scoring path
never used in production. This file keeps ONLY the method actually live
in the real pool data today -- no seed search, no smoothness/"bump"
scoring. (One correction made while consolidating: the old scripts'
comments called the colors no-prefix method "not production" / "point at
a different directory" -- the actual pool data contradicted that. Every
trial in the real production colors pool had prefix_length=0, confirming
no-prefix *is* what's live. Comments here reflect the real, verified
behavior, not the old scripts' stale claims.)

Usage
-----
    python generate_sequences.py --task both --n_pool 200 --pool_dir .

Output
------
    {pool_dir}/sequences_numbers.json, {pool_dir}/sequences_colors.json
    -- each a single JSON array of n_pool members, each member itself an
    array of trial dicts. Index i in the array is pool member i (see
    poolIndexForParticipant) -- the client indexes directly, no glob or
    concatenation step needed.
"""

import argparse
import json
import pathlib
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Shared basics
# ---------------------------------------------------------------------------
VALUE_MIN = 0     # numbers slider range
VALUE_MAX = 100
SEQ_LENGTH = 15   # observations per trial, both tasks

# Fixed design parameters -- see module docstring. These are constants, not
# CLI flags: nothing about this pipeline calls for regenerating the pool
# with different values casually, and the one time a value drifted
# unnoticed during the original task/ -> task_backend consolidation
# (std_fixed defaulted to 15 in the generator script but real deployed
# production data showed 10) it was only possible because it was an
# overridable default rather than a single source of truth -- hence
# keeping it a hardcoded constant here rather than a CLI flag. Deliberately
# changed back to 15 (from 10): a std=10 numbers pool produces prefixes
# with enough internal variance that generate_numbers_suffix's analytical
# variance-correction formula squeezes the SUFFIX down to its 1.0-std
# floor whenever a high-variance prefix is chosen (e.g. for a tutorial
# example with large early updates) -- std=15 gives the suffix enough
# variance budget that a high-variance prefix no longer flatlines the
# rest of the trial. This is a genuine, deliberate reversal of that
# earlier std=10 confirmation, not a re-drift -- change here directly (and
# re-run --tutorial afterward) if this ever needs revisiting again.
NUMBERS_N_PREFIX = 8
NUMBERS_N_REPEATS = 4
NUMBERS_PREFIX_LENGTH = 4
NUMBERS_MEAN_RANGE = (15.0, 85.0)
NUMBERS_STD_FIXED = 15.0
NUMBERS_BOUNDARY_MARGIN = 1.0
NUMBERS_STD_TOLERANCE_FRAC = 0.25

COLORS_N_TRIALS = 32
COLORS_BLUE_RANGE = (2, 13)

# Small TEST-variant constants -- a SEPARATE set, never touched by the
# production ones above. Gated behind an explicit --name flag (see CLI):
# passing --name is the only way to reach these, and doing so also forces
# a distinctly-suffixed output filename, so there's no way to accidentally
# produce a real-shaped production file with a tiny trial count, or vice
# versa. Exists so a full session (through Prolific or locally) can be
# driven end-to-end in seconds/minutes instead of ~15-30 minutes, for
# testing -- same statistical design (mean_range, std_fixed, blue_range),
# just far fewer trials per member.
TEST_NUMBERS_N_PREFIX = 2
TEST_NUMBERS_N_REPEATS = 1
TEST_COLORS_N_TRIALS = 2


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def save_pool(pool_members, name: str, out_dir) -> pathlib.Path:
    """Write the WHOLE pool (all n_pool members) for one task to a single
    {out_dir}/{name}.json -- a list of pool members, each itself a list
    of trial dicts (the same per-trial schema as before). One file, one
    call, no per-member files to keep track of or glob together later.
    The client imports this file directly and indexes into it by
    poolIndexForParticipant -- no client-side concatenation step needed."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.json'
    with open(path, 'w') as f:
        json.dump(pool_members, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# NUMBERS: prefix + suffix construction
# ---------------------------------------------------------------------------
def build_numbers_prefixes(rng, n_prefix, prefix_length, mean_range, std_fixed,
                              value_min, value_max):
    """n_prefix DISTINCT short sequences, each centered on an evenly-spaced
    point across mean_range -- spreading prefixes across the full range
    (rather than clustering near the midpoint) ensures every target mean
    has a reasonably close prefix available once matched below."""
    centers = np.linspace(mean_range[0], mean_range[1], n_prefix)
    prefixes = []
    for center in centers:
        vals = rng.normal(center, std_fixed, size=prefix_length)
        vals = np.clip(np.round(vals), value_min, value_max).astype(int).tolist()
        prefixes.append(vals)
    assert len({tuple(p) for p in prefixes}) == n_prefix, (
        "two numbers prefixes came out identical by chance -- re-run with a different seed")
    return prefixes


def spread_numbers_targets(n_trials, mean_range):
    """n_trials DISTINCT target means, evenly spaced across mean_range --
    one per trial, independent of which prefix a trial ends up paired with."""
    return [round(float(v), 4) for v in np.linspace(mean_range[0], mean_range[1], n_trials)]


def match_prefixes_to_targets(prefix_values, target_values):
    """Globally-optimal (Hungarian algorithm) one-to-one pairing that
    minimizes total |prefix_value - target_value| mismatch across ALL pairs
    jointly -- not a greedy nearest-neighbor pass, which can leave an
    arbitrarily bad single pair once a popular region runs out of supply.
    Returns target_values reordered to align 1:1 with prefix_values."""
    pv = np.asarray(prefix_values, dtype=float)
    tv = np.asarray(target_values, dtype=float)
    cost = np.abs(pv[:, None] - tv[None, :])
    row_idx, col_idx = linear_sum_assignment(cost)
    result = [None] * len(prefix_values)
    for r, c in zip(row_idx, col_idx):
        result[r] = target_values[c]
    return result


def build_numbers_suffix(rng, prefix_values, target_mean, std_fixed, suffix_length,
                            value_min, value_max, std_tolerance_frac, max_attempts=20):
    """Draws the suffix as a genuine, unrescaled random sample, centered so
    the FULL prefix+suffix trial's sample mean lands on target_mean in
    EXPECTATION (algebraic residual centering) -- never forced there
    exactly. This is what makes a trial's running average genuinely become
    more accurate as it progresses, rather than being artificially perfect.

    Two guards on the suffix's own spread only -- never on individual drawn
    values, never a rescale of the result:
      1. an analytical correction to the suffix's variance, so the pooled
         (prefix+suffix) sequence's EXPECTED std still comes out at
         std_fixed despite the prefix being an unrelated, independent block;
      2. a loose safety-net redraw (up to max_attempts, a fresh draw each
         time, never a rescale) if the achieved 15-observation std falls
         outside std_fixed * (1 +/- std_tolerance_frac).
    """
    prefix_sum = sum(prefix_values)
    seq_length = len(prefix_values) + suffix_length
    residual_mean = (target_mean * seq_length - prefix_sum) / suffix_length

    prefix_sse = sum((x - target_mean) ** 2 for x in prefix_values)
    suffix_variance = ((seq_length / suffix_length) * std_fixed ** 2
                       - prefix_sse / suffix_length
                       - (residual_mean - target_mean) ** 2)
    suffix_std = float(np.sqrt(max(suffix_variance, 1.0)))

    std_lo = std_fixed * (1.0 - std_tolerance_frac)
    std_hi = std_fixed * (1.0 + std_tolerance_frac)
    values = None
    for _ in range(max_attempts):
        raw = rng.normal(residual_mean, suffix_std, size=suffix_length)
        values = np.clip(np.round(raw), value_min, value_max).astype(int)
        if std_lo <= float(np.std(list(prefix_values) + values.tolist())) <= std_hi:
            break
    return values.tolist()


def build_iti_schedule(rng, n_prefix, n_repeats):
    """Half of each prefix's repeats get a 'control' inter-trial-interval,
    half get 'distract' -- shuffled independently per prefix."""
    schedule = {}
    for qid in range(n_prefix):
        conditions = ['control'] * (n_repeats // 2) + ['distract'] * (n_repeats - n_repeats // 2)
        rng.shuffle(conditions)
        schedule[qid] = conditions
    return schedule


def shuffle_avoiding_consecutive_repeats(trials, rng, ground_truth_key):
    """Random trial order, but the same prefix (qid) never appears twice in
    a row, and the very first trial is a 'distract'-ITI trial with a
    clearly off-center target (>10pt / >15pp from the middle) -- so the
    first thing a participant sees isn't an ambiguous, easy-to-misjudge
    case."""
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
                if ground_truth_key == 'true_mean':
                    return abs(t['true_mean'] - 50.0) > 10.0
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


def generate_numbers_trials(rng, n_prefix, n_repeats, prefix_length, seq_length,
                               mean_range, std_fixed, boundary_margin, std_tolerance_frac,
                               verbose=True):
    """Full numbers pool member: n_prefix x n_repeats trials, each a
    prefix + steered-but-noisy suffix. Returns a list of trial dicts -- the
    schema every consumer (client bundle, verify_numbers_trials,
    tutorial-example selection) expects."""
    suffix_length = seq_length - prefix_length
    n_trials = n_prefix * n_repeats
    value_min, value_max = VALUE_MIN + boundary_margin, VALUE_MAX - boundary_margin

    prefixes = build_numbers_prefixes(rng, n_prefix, prefix_length, mean_range, std_fixed,
                                         value_min, value_max)
    target_means = spread_numbers_targets(n_trials, mean_range)

    prefix_slot_idx = [i for i in range(n_prefix) for _ in range(n_repeats)]
    rng.shuffle(prefix_slot_idx)
    prefix_means = [float(np.mean(p)) for p in prefixes]
    matched_targets = match_prefixes_to_targets(
        [prefix_means[i] for i in prefix_slot_idx], target_means)

    iti_schedule = build_iti_schedule(rng, n_prefix, n_repeats)
    rep_count = {}
    trials = []
    for pfx_idx, target_mean in zip(prefix_slot_idx, matched_targets):
        prefix_vals = prefixes[pfx_idx]
        rep = rep_count.get(pfx_idx, 0)
        rep_count[pfx_idx] = rep + 1
        suffix = build_numbers_suffix(rng, prefix_vals, target_mean, std_fixed,
                                         suffix_length, value_min, value_max, std_tolerance_frac)
        trials.append({
            'qid': pfx_idx, 'true_mean': target_mean, 'true_std': std_fixed, 'true_p': None,
            'values': prefix_vals + suffix, 'prefix_length': prefix_length,
            'iti_ms': 1000, 'iti_condition': iti_schedule[pfx_idx][rep],
        })

    trials = shuffle_avoiding_consecutive_repeats(trials, rng, ground_truth_key='true_mean')
    for i, t in enumerate(trials):
        t['trial'] = i

    if verbose:
        print(f"[numbers] {n_trials} trials ({n_prefix} prefixes x {n_repeats} reps), "
              f"mean_range={mean_range}, std_fixed={std_fixed}")
    return trials


def verify_numbers_trials(trials, n_prefix, n_repeats, seq_length):
    """Structural checks only -- no smoothness/bump scoring, matching the
    hybrid method's own design (achieved values are genuinely noisy by
    construction, not a bug to gate on)."""
    n_trials = n_prefix * n_repeats
    assert len(trials) == n_trials
    for t in trials:
        assert len(t['values']) == seq_length
        assert all(VALUE_MIN <= v <= VALUE_MAX for v in t['values'])
    prefix_by_qid = {}
    for t in trials:
        prefix_by_qid.setdefault(t['qid'], tuple(t['values'][:t['prefix_length']]))
    assert len(prefix_by_qid) == n_prefix, "expected exactly one distinct prefix per qid"
    assert len(set(prefix_by_qid.values())) == n_prefix, "two different qids share an identical prefix"
    qid_counts = Counter(t['qid'] for t in trials)
    assert all(c == n_repeats for c in qid_counts.values()), "each qid should repeat exactly n_repeats times"


# ---------------------------------------------------------------------------
# COLORS: fully independent trials, exact quota
# ---------------------------------------------------------------------------
def spread_colors_targets(rng, n_trials, blue_range):
    """n_trials target blue-counts, covering EVERY integer level in
    blue_range as evenly as possible (not an evenly-spaced subset) --
    remainder trials assigned to a random subset of levels so there's no
    systematic bias toward one end of the range."""
    levels = list(range(blue_range[0], blue_range[1] + 1))
    base, remainder = divmod(n_trials, len(levels))
    counts = {lvl: base for lvl in levels}
    if remainder:
        for lvl in rng.choice(levels, size=remainder, replace=False):
            counts[lvl] += 1
    out = [lvl for lvl in levels for _ in range(counts[lvl])]
    rng.shuffle(out)
    return out


def build_exact_quota_sequence(rng, true_p, seq_length):
    """A {-1,+1} sequence of length seq_length with EXACTLY round(true_p *
    seq_length) values equal to +1 ('blue') -- exact by construction, not
    approximate. Order shuffled. Assumes true_p already corresponds to a
    blue count in [0, seq_length] -- true of the only real caller
    (spread_colors_targets always produces an in-range count). No clamp
    here if that's ever violated -- verify_colors_trials's own quota
    check downstream would catch the resulting mismatch."""
    n_blue = int(round(true_p * seq_length))
    values = [1] * n_blue + [-1] * (seq_length - n_blue)
    order = rng.permutation(seq_length)
    return [values[i] for i in order]


def generate_colors_trials(rng, n_trials, seq_length, blue_range, verbose=True):
    """Full colors pool member: n_trials fully independent trials, each
    with its own target proportion and its own exact-quota sequence. No
    prefix, no repeated qid -- every trial's qid is just its own index."""
    target_blue = spread_colors_targets(rng, n_trials, blue_range)

    iti_pool = ['control'] * (n_trials // 2) + ['distract'] * (n_trials - n_trials // 2)
    rng.shuffle(iti_pool)

    trials = []
    for i, blue_count in enumerate(target_blue):
        true_p = round(blue_count / seq_length, 6)
        trials.append({
            'qid': i, 'true_mean': None, 'true_std': None, 'true_p': true_p,
            'values': build_exact_quota_sequence(rng, true_p, seq_length),
            'prefix_length': 0, 'iti_ms': 1000, 'iti_condition': iti_pool[i],
        })
    rng.shuffle(trials)  # every qid is already unique -- no consecutive-repeat constraint needed
    for i, t in enumerate(trials):
        t['trial'] = i

    if verbose:
        print(f"[colors] {n_trials} independent trials, blue_range={blue_range}")
    return trials


def verify_colors_trials(trials, n_trials, seq_length):
    assert len(trials) == n_trials
    assert len({t['qid'] for t in trials}) == n_trials, "colors trials must have unique qids"
    for t in trials:
        assert len(t['values']) == seq_length
        assert all(v in (-1, 1) for v in t['values'])
        achieved_blue = sum(1 for v in t['values'] if v == 1)
        expected_blue = round(t['true_p'] * seq_length)
        assert achieved_blue == expected_blue, "colors quota mismatch -- should be impossible by construction"


# ---------------------------------------------------------------------------
# Pool assembly
# ---------------------------------------------------------------------------
def _build_numbers_pool_impl(n_pool, pool_dir, name, n_prefix, n_repeats, base_seed):
    members = []
    for i in range(n_pool):
        rng = make_rng(base_seed + i * 100_000)
        trials = generate_numbers_trials(
            rng, n_prefix, n_repeats, NUMBERS_PREFIX_LENGTH, SEQ_LENGTH,
            NUMBERS_MEAN_RANGE, NUMBERS_STD_FIXED, NUMBERS_BOUNDARY_MARGIN,
            NUMBERS_STD_TOLERANCE_FRAC, verbose=False)
        verify_numbers_trials(trials, n_prefix, n_repeats, SEQ_LENGTH)
        members.append(trials)
        if (i + 1) % 20 == 0 or i == n_pool - 1:
            print(f"  [numbers] {i + 1}/{n_pool} pool members built")
    return save_pool(members, name, pool_dir)


def build_numbers_pool(n_pool, pool_dir, base_seed=0):
    """Builds all n_pool independent numbers members (fixed design --
    see NUMBERS_* constants above) in memory, then writes them as ONE
    file: {pool_dir}/sequences_numbers.json."""
    return _build_numbers_pool_impl(n_pool, pool_dir, 'sequences_numbers',
                                     NUMBERS_N_PREFIX, NUMBERS_N_REPEATS, base_seed)


def build_test_numbers_pool(n_pool, pool_dir, name, base_seed=0):
    """Small (TEST_NUMBERS_N_PREFIX x TEST_NUMBERS_N_REPEATS trials/member,
    default 2x1=2) variant for testing -- see the TEST_* constants' own
    comment above for why. Written to {pool_dir}/{name}.json."""
    return _build_numbers_pool_impl(n_pool, pool_dir, name,
                                     TEST_NUMBERS_N_PREFIX, TEST_NUMBERS_N_REPEATS, base_seed)


def _build_colors_pool_impl(n_pool, pool_dir, name, n_trials, base_seed):
    members = []
    for i in range(n_pool):
        rng = make_rng(base_seed + i * 100_000)
        trials = generate_colors_trials(rng, n_trials, SEQ_LENGTH, COLORS_BLUE_RANGE, verbose=False)
        verify_colors_trials(trials, n_trials, SEQ_LENGTH)
        members.append(trials)
        if (i + 1) % 20 == 0 or i == n_pool - 1:
            print(f"  [colors] {i + 1}/{n_pool} pool members built")
    return save_pool(members, name, pool_dir)


def build_colors_pool(n_pool, pool_dir, base_seed=50_000):
    """Builds all n_pool independent colors members (fixed design -- see
    COLORS_* constants above) in memory, then writes them as ONE file:
    {pool_dir}/sequences_colors.json."""
    return _build_colors_pool_impl(n_pool, pool_dir, 'sequences_colors', COLORS_N_TRIALS, base_seed)


def build_test_colors_pool(n_pool, pool_dir, name, base_seed=50_000):
    """Small (TEST_COLORS_N_TRIALS trials/member, default 2) variant for
    testing. Written to {pool_dir}/{name}.json."""
    return _build_colors_pool_impl(n_pool, pool_dir, name, TEST_COLORS_N_TRIALS, base_seed)


# ---------------------------------------------------------------------------
# Tutorial sequences: ONE fixed trial per task, shared by every participant
# ---------------------------------------------------------------------------
def _running_mean_deltas(values):
    """Sum of |consecutive running-mean changes| across `values` -- the
    same |Δresponse|-style metric this project already uses elsewhere
    (see scripts/plot_sequences.py) to characterize "how much did each
    update move", applied here to the RAW STIMULUS stream itself rather
    than to a model's response -- i.e. how dramatic the running average/
    ratio's own on-screen updates would look, independent of any agent.
    Colors' {-1,+1} values work directly: a color SWITCH between adjacent
    observations drives the single biggest possible running-mean step, so
    this naturally rewards alternation there with no task-specific casing
    needed."""
    running, total = [], 0.0
    for i, v in enumerate(values, 1):
        total += v
        running.append(total / i)
    return sum(abs(running[i] - running[i - 1]) for i in range(1, len(running)))


def _has_repeated_values(values):
    """True if any value in `values` appears more than once. Used to
    exclude numbers candidates with a stretch of identical observations
    (e.g. the 48,48,48... flat run an earlier std=10 pool produced) --
    even a single exact repeat means the running mean visibly pauses at
    that step, undercutting the "keep it moving" goal this whole
    selection function exists for.

    NUMBERS ONLY -- never applied to colors. Colors values are {-1,+1},
    so by the pigeonhole principle EVERY 15-observation colors trial has
    repeated values (there are only 2 distinct values to draw from at
    all) -- applying this filter there would exclude the entire colors
    pool, not just the flat/boring ones."""
    return len(set(values)) != len(values)


def choose_tutorial_sequences(pool_dir='.', out_dir=None, n_score=5,
                               prefix_percentile_lo=75, prefix_percentile_hi=95):
    """Selects ONE fixed trial per task from the real production pool
    (sequences_numbers.json / sequences_colors.json under pool_dir) to
    serve as every participant's tutorial example -- the same trial for
    everyone, rather than a per-load dynamic pick (the client-side
    pickTutorialExample in src/shared/config-base.js, whose own docstring
    already argues for deriving the example from real data -- this keeps
    that principle but fixes WHICH real trial, instead of re-deriving it
    differently depending on which pool member happens to load first).

    Selection criterion (two stages, deliberately NOT a single global
    maximum -- see below for why), plus a hard pre-filter for numbers:
      0. (numbers only) Discard any trial containing a repeated exact
         value anywhere in its 15 observations (_has_repeated_values) --
         a repeat means the running mean visibly PAUSES at that step,
         which undercuts the whole point of this selection. Not applied
         to colors -- see that function's own docstring for why it
         can't be (every colors trial has repeats by construction).
      1. Restrict candidates to trials whose PREFIX score falls between
         the prefix_percentile_lo and prefix_percentile_hi percentiles of
         the WHOLE pool's prefix-score distribution -- "high but not
         maximal": still a big early swing, but excluding the very top
         tail.
      2. Among that band, pick the trial with the HIGHEST SUFFIX score --
         i.e. the one that ALSO keeps the running average/ratio visibly
         moving after the prefix, not just during it.
    (PREFIX score = _running_mean_deltas on the first n_score
    observations; SUFFIX score = the same metric on the remaining ones --
    both the same |Δresponse|-style measure, applied to the raw stimulus
    stream rather than a model's response.)
    This two-stage design exists because taking the single global-max
    PREFIX score (an earlier version of this function did exactly that)
    reliably picked a dead-flat suffix for numbers specifically: a
    maximally spread-out prefix has high internal variance, and
    generate_numbers_suffix's analytical variance-correction formula
    squeezes the SUFFIX's variance down (often to its 1.0-std floor) to
    compensate, so the trial's overall std still lands near std_fixed --
    confirmed directly, not assumed. Restricting to a percentile BAND
    rather than the single max, then choosing by suffix score within it,
    naturally avoids that outlier without needing to special-case it.

    Writes {out_dir}/tutorial_sequence_{numbers,colors}.json -- each a
    single JSON object: the winning trial dict (same schema as every
    pool member's own trials -- qid, true_mean, true_std, true_p, values,
    prefix_length, iti_ms, iti_condition, trial) plus a `pool_index`
    field recording provenance (which of the pool's members it came
    from). out_dir defaults to pool_dir.

    NOTE: this reads the ALREADY-BUILT pool files -- it does not
    regenerate them. If the production pool is ever regenerated with a
    different design (mean_range, std_fixed, blue_range, etc.), re-run
    this afterward -- otherwise the tutorial's fixed example could
    silently drift out of sync with the real task's parameters, exactly
    the failure mode pickTutorialExample's own docstring describes
    (see "Sequences" in CLAUDE.md's task_backend section).
    """
    pool_dir = pathlib.Path(pool_dir)
    out_dir = pathlib.Path(out_dir) if out_dir else pool_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    file_for_task = {'numbers': 'sequences_numbers.json', 'colors': 'sequences_colors.json'}
    chosen = {}
    for task, fname in file_for_task.items():
        path = pool_dir / fname
        with open(path) as f:
            pool = json.load(f)

        scored = []
        for pool_idx, member in enumerate(pool):
            for trial in member:
                if task == 'numbers' and _has_repeated_values(trial['values']):
                    continue
                prefix_score = _running_mean_deltas(trial['values'][:n_score])
                suffix_score = _running_mean_deltas(trial['values'][n_score:])
                scored.append((prefix_score, suffix_score, pool_idx, trial))

        prefix_scores = np.array([s[0] for s in scored])
        lo = float(np.percentile(prefix_scores, prefix_percentile_lo))
        hi = float(np.percentile(prefix_scores, prefix_percentile_hi))
        band = [s for s in scored if lo <= s[0] <= hi]
        assert band, (f"[{task}] no candidates fell inside the prefix-score percentile band "
                      f"[{prefix_percentile_lo}, {prefix_percentile_hi}] -- widen it")

        best_prefix_score, best_suffix_score, best_pool_idx, best_trial = max(
            band, key=lambda s: s[1])

        out = dict(best_trial)
        out['pool_index'] = best_pool_idx
        chosen[task] = out

        out_path = out_dir / f'tutorial_sequence_{task}.json'
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"[{task}] chosen from pool_index={best_pool_idx}, trial={best_trial['trial']}, "
              f"qid={best_trial['qid']}")
        print(f"  true_mean={best_trial['true_mean']}  true_std={best_trial['true_std']}  "
              f"true_p={best_trial['true_p']}")
        print(f"  prefix-score band: [{lo:.3f}, {hi:.3f}] (percentiles "
              f"{prefix_percentile_lo}-{prefix_percentile_hi} of {len(scored)} trials, "
              f"{len(band)} candidates in band)")
        print(f"  first {n_score} observations: {best_trial['values'][:n_score]}  "
              f"(prefix score={best_prefix_score:.3f})")
        print(f"  remaining observations: {best_trial['values'][n_score:]}  "
              f"(suffix score={best_suffix_score:.3f})")
        print(f"  full {len(best_trial['values'])} observations: {best_trial['values']}")
        print(f"  Saved: {out_path}")

    return chosen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--task', choices=['numbers', 'colors', 'both'], default='both')
    p.add_argument('--n_pool', type=int, default=200)
    p.add_argument('--pool_dir', default='.')
    p.add_argument('--base_seed', type=int, default=0)
    p.add_argument('--name', default=None,
                   help="If given, builds a SMALL TEST variant "
                        f"({TEST_NUMBERS_N_PREFIX * TEST_NUMBERS_N_REPEATS} trials/member for numbers, "
                        f"{TEST_COLORS_N_TRIALS} for colors) instead of the full production pool, written "
                        "to sequences_<task>_<name>.json (e.g. --name test2trial). Omit for the real "
                        "production pool (sequences_<task>.json) -- this flag not being passed at all "
                        "leaves production behavior completely unaffected.")
    p.add_argument('--tutorial', action='store_true',
                   help="Instead of building the pool, select and save the fixed tutorial "
                        "sequences (tutorial_sequence_{numbers,colors}.json) from the ALREADY-BUILT "
                        "production pool under --pool_dir. See choose_tutorial_sequences' own "
                        "docstring for the selection criterion.")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n_pool > 0
    tasks = ['numbers', 'colors'] if args.task == 'both' else [args.task]

    if args.tutorial:
        choose_tutorial_sequences(pool_dir=args.pool_dir)
        print("\nJOB_COMPLETE")
        return

    if 'numbers' in tasks:
        if args.name:
            name = f'sequences_numbers_{args.name}'
            print(f"Building TEST numbers pool ({TEST_NUMBERS_N_PREFIX}x{TEST_NUMBERS_N_REPEATS} "
                  f"trials/member): {args.n_pool} members -> {args.pool_dir}/{name}.json")
            build_test_numbers_pool(args.n_pool, args.pool_dir, name, base_seed=args.base_seed)
        else:
            print(f"Building numbers pool: {args.n_pool} members -> {args.pool_dir}")
            build_numbers_pool(args.n_pool, args.pool_dir, base_seed=args.base_seed)

    if 'colors' in tasks:
        if args.name:
            name = f'sequences_colors_{args.name}'
            print(f"Building TEST colors pool ({TEST_COLORS_N_TRIALS} trials/member): "
                  f"{args.n_pool} members -> {args.pool_dir}/{name}.json")
            build_test_colors_pool(args.n_pool, args.pool_dir, name, base_seed=args.base_seed + 50_000)
        else:
            print(f"Building colors pool: {args.n_pool} members -> {args.pool_dir}")
            build_colors_pool(args.n_pool, args.pool_dir, base_seed=args.base_seed + 50_000)

    print("\nJOB_COMPLETE")


if __name__ == '__main__':
    main()
