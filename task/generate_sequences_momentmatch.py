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
any number of parameter levels for free.

Prefix generation (IMPORTANT -- read this before touching prefix logic)
-------------------------------------------------------------------------
As of the latest redesign, PREFIX IDENTITY and TARGET LEVEL are
INDEPENDENT axes. This is a deliberate change from an earlier version of
this script (and from generate_sequences.py) where each qid meant exactly
one (prefix, target) pair, repeated n_repeats times -- which had a real,
confirmed bug: two DIFFERENT qids could (and, in the production 6x4 pilot
sequences, actually did) end up with an IDENTICAL realized prefix purely by
chance, since binary's small combinatorial space at prefix_length=4 (only
2**4=16 possible sequences total, and as few as 4 arrangements for a given
exact quota) made collisions likely once enough qids shared a quota.

The corrected design:
  - `n_prefix` DISTINCT prefixes are generated first, independent of any
    target. This is what "qid" and repeat structure now track: each of the
    n_prefix prefixes is repeated `n_repeats` times, giving n_total =
    n_prefix * n_repeats trials. See build_continuous_prefixes /
    build_binary_prefixes.
      - Continuous: real-valued draws essentially never coincide exactly,
        so no active dedup is needed -- just a safety-net assertion.
      - Binary: genuinely needs active dedup, since the combinatorial space
        is small. Compositions (blue-ball counts out of prefix_length) are
        spread across the FULL range from all-red to all-blue via an
        evenly-spaced grid (same linspace pattern used elsewhere in this
        script), and arrangements within a repeated composition are drawn
        WITHOUT replacement. Fails loudly if n_prefix exceeds what's
        structurally possible at a given prefix_length.
  - `n_total` TARGET values (true_mean for continuous, true_p for binary)
    are generated SEPARATELY, with NO forced repeat structure at all --
    each trial gets its own target, decoupled from which prefix it uses.
      - Continuous: n_total DISTINCT true_mean values, evenly spaced across
        mean_range (nothing here needs to repeat -- see
        continuous_target_means).
      - Binary: the FULL native integer granularity of blue_range (every
        integer level, not an evenly-spaced SUBSET the way the old
        --n_levels design worked), distributed as evenly as possible across
        n_total trials -- when n_total doesn't divide evenly by the number
        of levels, the remainder goes to a RANDOM subset of levels (not
        deterministically the first few), so there's no systematic bias
        toward one end of the range. See binary_target_blue_counts.
  - Prefix-slots and target-slots are then matched via a GLOBALLY OPTIMAL
    assignment (optimal_matching, the Hungarian algorithm via
    scipy.optimize.linear_sum_assignment) that minimizes total
    |prefix_value - target_value| mismatch across all pairs jointly -- NOT
    a greedy or randomized heuristic. A greedy "closest available" approach
    was tried first and rejected after confirming empirically that it can
    leave an arbitrarily bad single-pair mismatch (tested down to fully
    greedy -- still left a >40-point mismatch on one pair) purely from
    unlucky processing order once a popular region's limited supply runs
    out; the Hungarian algorithm has no such failure mode since it
    considers the whole assignment jointly.
      - Continuous: no HARD feasibility constraint at all -- any prefix can
        be rescaled toward any target via the residual suffix math below.
      - Binary: a real hard feasibility constraint exists (see "Residual
        math" below) -- an all-red prefix, for instance, cannot reach a
        target near the top of blue_range no matter what the suffix
        contains; infeasible pairs are given effectively-infinite cost, and
        the function raises loudly if the optimal solution still has to use
        one (meaning no feasible assignment exists at all).
    No extra randomization is layered on top of the solver itself --
    prefixes (and, for binary, targets) are already freshly drawn per
    --seed, so different seeds already explore different assignments. Not
    having an obvious, learnable prefix->target mapping (the "soft" pairing
    goal from the design discussion) comes from the prefix's own sampling
    noise -- SEM = std_fixed/sqrt(prefix_length), e.g. 7.5 at the
    production defaults -- and from a prefix's n_repeats occurrences
    generally landing on different targets, not from randomness inside the
    matching step.
  - Continuous prefixes are drawn with their CENTERS spread evenly across
    mean_range (build_continuous_prefixes), not all centered on the range
    midpoint -- an earlier version did the latter and, confirmed
    empirically, left extreme targets with no genuinely close prefix
    available no matter how the matching was tuned (a supply problem, not a
    selection problem). Binary's compositions were already spread this way
    from the start (see build_binary_prefixes), so it never had this issue.

A consequence worth knowing about, not a bug: under this design, a given
prefix's n_repeats occurrences will generally each pair with a DIFFERENT
true_mean/true_p. "qid repeats" here means "same literal prefix shown
multiple times", NOT "same hidden generative parameter shown multiple
times" the way carrabin/yoo's qid works. If a downstream analysis or
config-base.js's pickTutorialExample assumes a qid's repeats share a target
(some do), that assumption no longer holds under this design and needs
revisiting there, not here.

Residual math (why this is needed at all)
-------------------------------------------
generate_sequences.py's prefix is drawn freely (not matched to anything),
and its suffix is built via REJECTION SAMPLING on the FULL pooled sequence
-- the rejection loop implicitly "corrects" for whatever the prefix already
contributed. This script has no rejection loop at all (that is the entire
point of moment-matching), so with a now-generic prefix, the suffix must
explicitly target the RESIDUAL needed to bring the pooled sequence to the
trial's actual target, not the target directly:

  Continuous (suffix_for_continuous_target): exact algebra --
    given prefix_sum = sum(prefix), and wanting the pooled sample mean over
    all seq_length observations to equal target_mean:
      residual_suffix_mean = (target_mean * seq_length - prefix_sum) / suffix_length
    The suffix's OWN std target is left at std_fixed (not similarly
    residual-adjusted -- doing that exactly would require accounting for
    the between-block mean-shift term, i.e. Var(pooled) includes a
    contribution from how far apart the two blocks' means are, which is
    no longer ~0 now that the prefix's mean is unrelated to the target).
    Expect pooled std to run slightly ABOVE std_fixed as a result, more so
    for trials where the prefix's own realized mean happens to be far from
    that trial's target -- check via --report if this matters for a given
    mean_range/std_fixed combination.

  Binary (suffix_for_binary_target): exact by construction, no
    approximation needed -- total blue balls needed = round(true_p *
    seq_length); suffix needs exactly (that minus however many blue balls
    the prefix already has) blue balls among its suffix_length
    observations. This is well-defined only when that residual falls in
    [0, suffix_length], which is exactly what
    match_prefixes_to_targets_binary's feasibility constraint guarantees
    before this function is ever called.

Method (per-block construction, unchanged from before)
---------------------------------------------------------
Continuous (per block):
  1. Draw n raw i.i.d. values from Normal(target_mean, target_std) (no rejection).
  2. Affinely rescale so the block's OWN sample mean/std exactly match the
     target: v' = target_mean + (v - mean(v)) * (target_std / std(v)).
  3. Clip to [0,100], then repeat the rescale+clip step a few times
     (max_rescale_iters) to claw back most of the bias clipping introduces
     at extreme means -- does NOT fully solve the mean=10/90 + std=15
     structural mismatch (clipping still truncates), but converges much
     closer than a single pass, and never needs to reject and redraw.
  4. Round to nearest int.

Binary (per block): exact quota: round(n * true_p) blue, the rest red,
  order shuffled -- exact by construction, no correction needed.

What this does NOT fix
-----------------------
  Clipping after rescaling still truncates the distribution at extreme
  means, which still biases the achieved std down somewhat (just less
  severely than generate_sequences.py's rejection sampling, since we
  actively correct for it via iterative rescale). If mean_range pushes all
  the way to the [0,100] edges with a large std, the plausibility check
  (report only, not gating -- see below) may still show a small residual
  gap at the extremes. Use --report to check this directly before trusting
  a given mean_range/std combination.

Plausibility checking here is DIAGNOSTIC ONLY (reported, not gating) --
generation always succeeds by construction, so there is no rejection loop
and no seed can "fail" the way it could in generate_sequences.py.

Usage
-----
  # Single generation (fast, no search) -- default: 6 prefixes, continuous
  # 15..85, binary blue-range 2..13 out of 15
  python task/generate_sequences_momentmatch.py --task continuous --seed 0

  # Custom range / prefix count
  python task/generate_sequences_momentmatch.py --task continuous --seed 0 \\
      --n_prefix 6 --mean_range 15 85 --n_repeats 4
  python task/generate_sequences_momentmatch.py --task binary --seed 0 \\
      --n_prefix 6 --blue_range 2 13 --n_repeats 4

  # Seed search over orderings/realizations for smoothness (this is the
  # actual production 6x4 pilot search)
  python task/generate_sequences_momentmatch.py --task both --n_tries 1000 \\
      --n_prefix 6 --n_repeats 4 --rl_alpha_0 1.0 --rl_lambda 0.5

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
from collections import Counter

import numpy as np
import pandas as pd

# Reuse generic, method-agnostic utilities from generate_sequences.py rather
# than duplicating them -- these are not tied to rejection sampling.
from generate_sequences import (
    VALUE_MIN, VALUE_MAX,
    make_rng,
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
# Prefix generation: n_prefix DISTINCT prefixes, independent of any target
# (see module docstring's "Prefix generation" section for full rationale)
# ---------------------------------------------------------------------------
def build_continuous_prefixes(rng, n_prefix, prefix_length, mean_range, std_fixed,
                              value_min=VALUE_MIN, value_max=VALUE_MAX):
    """n_prefix distinct length-prefix_length integer sequences, one drawn
    per evenly-spaced CENTER across mean_range (via linspace, the same
    pattern used for binary's composition grid and continuous's own
    target-level grid) -- NOT all drawn from a single central distribution.

    Why this matters: prefix identity and target level are independent
    axes (see module docstring), and top_k_random_matching can only PREFER
    a good pairing among whatever prefixes actually exist -- if all
    n_prefix prefixes were centered on the range midpoint (an earlier
    version of this function did exactly that), there would reliably be
    nothing close to extreme targets no matter how the matching is
    written. Confirmed empirically: with all 6 prefixes centered at 50,
    max achieved-mean error was ~4.7 and max achieved std ~28.5 against a
    target of 15, even with the closeness preference in place -- a supply
    problem, not a selection problem. Spreading the prefixes' own centers
    across the same range as the targets ensures there's always something
    reasonably close available.

    This does NOT make true_mean obvious from the prefix alone: each
    prefix is only prefix_length=4 observations, so its OWN sample mean's
    standard error is std_fixed/sqrt(prefix_length) = 15/2 = 7.5 for the
    production defaults -- a prefix centered near 85 can still look like an
    fairly ordinary 4-observation run by chance, and the ACTUAL paired
    target (chosen by the soft top-k preference, not a forced 1:1 tie) will
    often differ from any given prefix's own center anyway. The randomness
    inherent in drawing only 4 observations is doing real work here, not
    just decoration.
    """
    centers = np.linspace(mean_range[0], mean_range[1], n_prefix)
    prefixes = []
    for center in centers:
        vals = rng.normal(center, std_fixed, size=prefix_length)
        vals = np.clip(np.round(vals), value_min, value_max).astype(int).tolist()
        prefixes.append(vals)
    seen = set(tuple(p) for p in prefixes)
    assert len(seen) == n_prefix, (
        f"Two continuous prefixes came out identical by chance (n_prefix={n_prefix}, "
        f"prefix_length={prefix_length}) -- astronomically unlikely; if this ever "
        f"fires for real, just re-run with a different --seed.")
    return prefixes


def build_binary_prefixes(rng, n_prefix, prefix_length):
    """n_prefix distinct length-prefix_length binary ({-1,+1}) prefixes,
    spanning diverse compositions from all-red to all-blue -- NOT all
    forced to the maximally-flexible 50/50 composition, per the "method 2"
    design decision (diverse compositions + constrained random matching to
    targets, rather than trivial-but-uninteresting uniform composition).

    Compositions (blue-ball counts out of prefix_length) are spread via
    linspace(0, prefix_length, n_prefix), same pattern as this script's
    other level-grids elsewhere. Arrangements within a repeated composition
    (linspace can round to the same integer more than once) are drawn
    WITHOUT replacement so no two prefixes are literally identical --
    THIS is the actual fix for the collision bug that motivated this
    whole redesign (two different qids in the production 6x4 pilot ended
    up with an identical realized prefix by chance).

    Fails loudly (not a plausible-looking silent fallback) if n_prefix
    exceeds the number of distinct length-prefix_length binary sequences
    that exist at all (2**prefix_length), or if a specific composition's
    own arrangement pool (C(prefix_length, k)) is exhausted.
    """
    max_possible = 2 ** prefix_length
    assert n_prefix <= max_possible, (
        f"n_prefix={n_prefix} exceeds the {max_possible} distinct binary sequences "
        f"that exist at prefix_length={prefix_length} -- reduce n_prefix or "
        f"increase prefix_length.")

    compositions = [int(round(v)) for v in np.linspace(0, prefix_length, n_prefix)]

    prefixes = []
    used_per_composition: dict[int, set] = {}
    for k in compositions:
        pool_size = math.comb(prefix_length, k)
        already_used = used_per_composition.setdefault(k, set())
        assert len(already_used) < pool_size, (
            f"composition k={k} blue out of prefix_length={prefix_length} only has "
            f"{pool_size} distinct arrangements, and n_prefix={n_prefix}'s evenly-"
            f"spaced composition grid ({compositions}) asks for more of them than "
            f"exist -- reduce n_prefix or widen prefix_length.")
        for _attempt in range(10_000):
            positions = rng.choice(prefix_length, size=k, replace=False)
            arrangement = [-1] * prefix_length
            for pos in positions:
                arrangement[pos] = 1
            key = tuple(arrangement)
            if key not in already_used:
                already_used.add(key)
                prefixes.append(arrangement)
                break
        else:
            raise RuntimeError(
                f"Could not find a fresh arrangement for composition k={k} after "
                f"10000 attempts -- should be unreachable given the pool_size "
                f"assertion above; something is wrong if this actually fires.")
    return prefixes


# ---------------------------------------------------------------------------
# Target-level generation: independent of prefix identity, no forced repeats
# (see module docstring's "Prefix generation" section)
# ---------------------------------------------------------------------------
def continuous_target_means(n_trials, mean_range):
    """n_trials DISTINCT true_mean values, evenly spaced across mean_range --
    one per trial. No repeat structure here at all (unlike prefix identity,
    which is what carries repeats now) -- these are independent axes."""
    return [round(float(v), 4) for v in np.linspace(mean_range[0], mean_range[1], n_trials)]


def binary_target_blue_counts(rng, n_trials, blue_range):
    """Full native integer granularity across blue_range (EVERY integer
    blue count, not an evenly-spaced SUBSET the way the old --n_levels
    design worked), distributed as evenly as possible across n_trials
    trials. When n_trials doesn't divide evenly by the number of levels,
    the remainder goes to a RANDOM subset of levels (not deterministically
    the first few), so there is no systematic bias toward one end of the
    range. Returns a shuffled list of length n_trials (blue counts, with
    multiplicity)."""
    levels = list(range(blue_range[0], blue_range[1] + 1))
    n_levels = len(levels)
    base, remainder = divmod(n_trials, n_levels)
    counts = {lvl: base for lvl in levels}
    if remainder:
        bonus_levels = rng.choice(levels, size=remainder, replace=False)
        for lvl in bonus_levels:
            counts[lvl] += 1
    out = []
    for lvl in levels:
        out.extend([lvl] * counts[lvl])
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Preference-weighted random matching: prefix-slots <-> target-slots.
# Prefers CLOSER pairs (|prefix's own mean/p - target|) without collapsing
# to a fully deterministic nearest-neighbor assignment -- see
# top_k_random_matching's docstring for why.
# ---------------------------------------------------------------------------
def binary_feasible_target_range(prefix_blue, suffix_length, blue_range):
    """A prefix with `prefix_blue` blue balls (out of prefix_length) can
    only reach total-sequence blue counts in [prefix_blue, prefix_blue +
    suffix_length] -- the suffix alone can add between 0 and suffix_length
    more blue balls, nothing else is possible. Intersected with blue_range
    since targets outside it are never generated anyway."""
    lo = max(blue_range[0], prefix_blue)
    hi = min(blue_range[1], prefix_blue + suffix_length)
    return lo, hi


def optimal_matching(prefix_slot_values, target_slot_values, feasible_fn=None):
    """Finds the GLOBALLY minimum-total-mismatch one-to-one assignment
    between prefix-slots and target-slots (sum of |prefix_value -
    target_value| over all pairs), via the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment) -- NOT a greedy/random-order
    heuristic.

    This replaced an earlier greedy "random among the top_k closest
    available" approach after confirming empirically that greedy selection
    can leave an arbitrarily bad single-pair mismatch even when every step
    picks the closest currently-available option (tested down to top_k=1,
    i.e. fully greedy -- still left a >40-point mismatch on one pair,
    because "closest available right now" can get forced into a bad corner
    by unlucky processing order once a popular region's supply runs out).
    The Hungarian algorithm has no such failure mode: it considers the full
    assignment jointly, so no single pair can be arbitrarily bad while a
    better OVERALL assignment existed.

    No extra randomization layer is added here on top of scipy's solver:
    prefixes (and, for binary, targets) are already freshly drawn per
    --seed, so different seeds already explore genuinely different
    assignments -- see build_continuous_prefixes / build_binary_prefixes.
    This keeps prefix identity and target level on independent axes (a
    prefix's n_repeats occurrences generally still land on different
    targets) without needing the assignment step itself to inject
    additional noise. "Soft" in the sense the design discussion wanted --
    not an obvious, rigid prefix->target mapping a participant could
    learn -- comes from the prefix's own sampling noise (SEM =
    std_fixed/sqrt(prefix_length), e.g. 7.5 at the production defaults)
    and from repeats of the same prefix generally pairing with different
    targets, not from randomness inside this function.

    feasible_fn(prefix_value, target_value) -> bool marks infeasible pairs
    with a very large cost rather than filtering them out structurally (the
    solver needs a square cost matrix) -- used for binary's exact quota-
    reachability constraint; None for continuous, which has no such hard
    constraint. Raises loudly if the optimal solution still has to use an
    infeasible pair, meaning no feasible assignment exists at all (the
    chosen prefix compositions cannot structurally supply the requested
    target distribution) -- never silently returns a broken pairing.

    Returns a list of target_slot_values reordered to align 1:1 with
    prefix_slot_values (result[i] is the target paired with
    prefix_slot_values[i]).
    """
    from scipy.optimize import linear_sum_assignment

    n = len(prefix_slot_values)
    assert len(target_slot_values) == n
    pv = np.asarray(prefix_slot_values, dtype=float)
    tv = np.asarray(target_slot_values, dtype=float)
    cost = np.abs(pv[:, None] - tv[None, :])

    BIG = 1e6
    if feasible_fn is not None:
        infeasible = np.array([[not feasible_fn(p, t) for t in tv] for p in pv])
        cost = np.where(infeasible, BIG, cost)

    row_idx, col_idx = linear_sum_assignment(cost)
    if feasible_fn is not None and (cost[row_idx, col_idx] >= BIG).any():
        raise RuntimeError(
            "No feasible prefix<->target assignment exists even at the OPTIMAL "
            "solution -- the chosen prefix compositions cannot structurally supply "
            "the requested target distribution (e.g. too many extreme-composition "
            "all-red/all-blue prefixes competing for the same scarce end of the "
            "range). Adjust n_prefix or the composition grid (see "
            "build_binary_prefixes).")

    result = [None] * n
    for r, c in zip(row_idx, col_idx):
        result[r] = target_slot_values[c]
    return result


# ---------------------------------------------------------------------------
# Residual-suffix builders (see module docstring's "Residual math" section)
# ---------------------------------------------------------------------------
def suffix_for_continuous_target(rng, prefix_values, target_mean, std_fixed,
                                 suffix_length, max_rescale_iters,
                                 value_min=VALUE_MIN, value_max=VALUE_MAX):
    """Suffix's OWN target mean = the algebraic residual needed so the FULL
    (prefix+suffix) pooled sequence's sample mean hits target_mean exactly
    (by construction, via the same rescale mechanism as
    moment_match_continuous) -- necessary now that the prefix is generic
    and not already near the target. See module docstring for why the
    suffix's std target is left at std_fixed rather than similarly
    residual-adjusted (expect pooled std to run slightly above std_fixed
    as a result)."""
    prefix_sum = sum(prefix_values)
    seq_length = len(prefix_values) + suffix_length
    residual_mean = (target_mean * seq_length - prefix_sum) / suffix_length
    return moment_match_continuous(rng, residual_mean, std_fixed, suffix_length,
                                   value_min=value_min, value_max=value_max,
                                   max_rescale_iters=max_rescale_iters)


def suffix_for_binary_target(rng, prefix_values, target_blue, suffix_length):
    """Exact residual quota: the suffix needs exactly (target_blue minus
    however many blue balls the prefix already has) blue balls among its
    suffix_length observations. Guaranteed to land in [0, suffix_length] by
    construction AS LONG AS the prefix<->target pairing came from
    match_prefixes_to_targets_binary (which only ever proposes feasible
    pairings) -- this function trusts that contract rather than re-deriving
    it, so the assertion below is a bug detector, not a user-facing error
    path."""
    prefix_blue = sum(1 for v in prefix_values if v == 1)
    suffix_blue = target_blue - prefix_blue
    assert 0 <= suffix_blue <= suffix_length, (
        f"suffix_blue={suffix_blue} out of range [0,{suffix_length}] -- an "
        f"infeasible (prefix,target) pair reached suffix construction, which "
        f"should be impossible if match_prefixes_to_targets_binary was used "
        f"upstream. This indicates a real bug, not a data issue.")
    vals = [1] * suffix_blue + [-1] * (suffix_length - suffix_blue)
    perm = rng.permutation(suffix_length)
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
def generate_task_sequences_momentmatch(task, args, rng, verbose=True):
    """Generate all trials for one task via moment matching.

    Prefix identity (n_prefix distinct prefixes, each repeated n_repeats
    times) and target level (n_total independently-generated true_mean/
    true_p values, no forced repeats) are INDEPENDENT axes -- see module
    docstring's "Prefix generation" section for the full rationale and the
    bug this fixes. Returns (DataFrame, json_trials), same schema as
    generate_sequences.py.

    verbose controls the per-call header/per-prefix/summary prints -- this
    function is called once per attempt during a seed search (potentially
    thousands of times), so _seed_search_momentmatch passes verbose=False
    to keep output to the periodic progress line only. --report and
    single-generation mode want the full detail, so they leave it True.
    """
    seq_length    = args.seq_length
    prefix_length = args.prefix_length
    suffix_length = seq_length - prefix_length
    if prefix_length >= seq_length:
        raise NotImplementedError(
            "prefix_length >= seq_length (no suffix at all) is not supported under "
            "the independent-prefix/target design -- there would be nothing left "
            "for the residual-suffix math to adjust, so a trial's realized value "
            "couldn't be steered toward its assigned target at all. If this is ever "
            "genuinely needed, it needs its own dedicated code path, not a silent "
            "fallback here.")

    n_prefix  = args.n_prefix
    n_repeats = args.n_repeats
    n_total   = n_prefix * n_repeats

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}  (moment-matched; prefix identity and target level are independent axes)")
        print(f"  Total trials   : {n_total} ({n_prefix} prefixes x {n_repeats} reps)")
        print(f"  Seq / prefix / suffix length: {seq_length} / {prefix_length} / {suffix_length}")

    # -- Build n_prefix DISTINCT prefixes, independent of any target --------
    if task == 'continuous':
        prefixes = build_continuous_prefixes(rng, n_prefix, prefix_length,
                                             args.mean_range, args.std_fixed)
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
            print(f"  target blue-count distribution: {dict(sorted(Counter(target_blue).items()))}")

    # -- Prefix-slots: each of the n_prefix prefixes repeated n_repeats times,
    #    shuffled -- this is what carries "qid" identity/repeats now.
    prefix_slot_idx = [i for i in range(n_prefix) for _ in range(n_repeats)]
    rng.shuffle(prefix_slot_idx)

    # -- Match prefix-slots to target-slots via globally optimal assignment
    #    (see optimal_matching's docstring for why greedy selection was
    #    tried first and rejected)
    if task == 'continuous':
        prefix_means = [float(np.mean(pfx)) for pfx in prefixes]
        prefix_slot_values = [prefix_means[i] for i in prefix_slot_idx]
        # No hard feasibility constraint for continuous -- any prefix can be
        # rescaled toward any target via the residual suffix.
        matched_targets = optimal_matching(prefix_slot_values, target_means)
    else:
        prefix_slot_blue = [sum(1 for v in prefixes[i] if v == 1) for i in prefix_slot_idx]
        matched_targets = optimal_matching(
            prefix_slot_blue, target_blue,
            feasible_fn=lambda pv, tv: (
                binary_feasible_target_range(pv, suffix_length, args.blue_range)[0]
                <= tv <=
                binary_feasible_target_range(pv, suffix_length, args.blue_range)[1]))

    # -- ITI schedule (per prefix identity, same mechanism as before) --------
    ITI_MS = 1000
    prefix_templates = [{'qid': i} for i in range(n_prefix)]
    iti_sched = _build_iti_schedule(rng, prefix_templates, n_repeats)
    rep_count = {}

    # -- Build each trial: prefix + residual-matched suffix ------------------
    trials = []
    for pfx_idx, target in zip(prefix_slot_idx, matched_targets):
        prefix_vals = prefixes[pfx_idx]
        rep_idx = rep_count.get(pfx_idx, 0)
        iti_condition = iti_sched[pfx_idx][rep_idx]
        rep_count[pfx_idx] = rep_idx + 1

        if task == 'continuous':
            true_mean, true_std, true_p = target, args.std_fixed, float('nan')
            suffix = suffix_for_continuous_target(
                rng, prefix_vals, true_mean, true_std, suffix_length,
                args.max_rescale_iters)
        else:
            true_p = round(target / seq_length, 6)
            true_mean, true_std = float('nan'), float('nan')
            suffix = suffix_for_binary_target(rng, prefix_vals, target, suffix_length)

        trials.append({
            'qid': pfx_idx, 'true_mean': true_mean, 'true_std': true_std,
            'true_p': true_p, 'values': prefix_vals + suffix,
            'iti_ms': ITI_MS, 'iti_condition': iti_condition,
        })

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

    # -- Sanity checks (same spirit as generate_sequences.py) ----------------
    assert len(df) == n_total * seq_length
    if task == 'continuous':
        assert df['value'].between(VALUE_MIN, VALUE_MAX).all()
    else:
        assert df['value'].isin([-1, 1]).all()
    assert df['trial'].nunique() == n_total
    assert (df.groupby('qid')['trial'].nunique() == n_repeats).all()
    if task == 'binary':
        # Exact quota by construction -- verify no rounding slop crept in.
        achieved_blue = df.groupby('trial')['value'].apply(lambda s: (s == 1).sum())
        expected_blue = (df.groupby('trial')['true_p'].first() * seq_length).round()
        assert np.array_equal(achieved_blue.reindex(expected_blue.index).values,
                              expected_blue.values), (
            "binary quota mismatch -- suffix residual math has a bug, this should "
            "be impossible given exact-quota construction.")

    if verbose:
        print(f"\n  {len(df)} rows | {n_total} trials | {n_prefix} prefixes x {n_repeats} reps")
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
        df, json_trials = generate_task_sequences_momentmatch(task, args, rng, verbose=False)

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
    p.add_argument('--n_repeats',          type=int,   default=4,
                   help='How many times each of the n_prefix distinct prefixes is used. '
                        'n_trials = n_prefix * n_repeats.')
    p.add_argument('--seq_length',         type=int,   default=15)
    p.add_argument('--prefix_length',      type=int,   default=4)
    p.add_argument('--n_prefix',           type=int,   default=6,
                   help='Number of DISTINCT prefixes -- this is what "qid" and repeat '
                        'structure track now (see module docstring\'s "Prefix generation" '
                        'section). Independent of the target-level distribution below.')
    p.add_argument('--mean_range',         type=float, nargs=2, default=[15.0, 85.0],
                   help='Continuous: true_mean range. n_trials DISTINCT values are spread '
                        'evenly across this range, one per trial -- not tied to n_prefix.')
    p.add_argument('--std_fixed',          type=float, default=15.0)
    p.add_argument('--blue_range',         type=int,   nargs=2, default=[2, 13],
                   help='Binary: blue-ball-count range (out of --seq_length). EVERY integer '
                        'level in this range is used (not an evenly-spaced subset), '
                        'distributed as evenly as possible across n_trials trials -- not '
                        'tied to n_prefix.')
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
