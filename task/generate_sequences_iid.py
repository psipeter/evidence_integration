"""
generate_sequences_iid.py
==========================
ROLE: one of two generation methods under active consideration for the
full 10x4 experiment (see CLAUDE.md's "PI decision pending" note) -- NOT
currently promoted to production. The current 6x4 pilot uses
generate_sequences_momentmatch.py instead. Do not delete this file or its
output ({task}_iid_sequences.*) -- the i.i.d.-vs-moment-matched choice is
still open and this is one of the two live candidates being compared.

Known bug, fixed: binary prefixes were originally drawn independently per
qid with no uniqueness check -- confirmed empirically that 9 of 10 random
seeds produced a real prefix collision at n_unique_sequences=10 (only
6-8 distinct prefixes out of 10 expected; only 2**prefix_length=16 possible
binary sequences exist at prefix_length=4, so this was likely, not a fluke).
This silently affected every binary result computed via
scripts/inspect_iid_sequences.py before the fix -- see
docs/sequence_design_open_questions.md for the full investigation this was
found during. Fixed via _draw_unique_binary_prefix (active dedup, matching
in spirit -- though not code, since the generation path differs -- the fix
already applied to generate_sequences_momentmatch.py's build_binary_prefixes
for the same underlying reason).

The "pure i.i.d." branch: genuinely unconstrained sampling. No k-based
plausibility check, no rejection loop, no smoothing/gating, and -- per
explicit decision -- NO seed search or best-of-N ranking either. Any
best-of-N selection by an outcome-dependent score is itself a form of
conditioning on the realized sequence; cherry-picking a "smooth" seed out
of many draws pulls back toward exactly the same finite-population/
conditioned-sampling structure this branch exists to avoid (see chat
history -- k-constrained rejection sampling and quota sampling turned out
to be the same underlying object at different points on one continuum).
So this script is deliberately minimal: draw once, save, done.

Why this exists as its own script
----------------------------------
generate_sequences.py's rejection sampling and generate_sequences_momentmatch.py's
quota construction are the SAME underlying object at different points on
one continuum: i.i.d. sampling conditioned on the final composition falling
within k x SE of the target. Exact enumeration showed that even
generate_sequences.py's production k=0.7 already reduces the variance of
the LAST observation's predictability to ~10% of true i.i.d., 90% of the
way to quota's hard zero. There is no way to "tighten k for smoothness"
without buying into finite-population predictability -- they are the same
lever, not independent choices.

Given that, this project maintains two genuinely distinct branches:

  Branch A (this script): true i.i.d., zero smoothing, zero cherry-picking.
    Accepts whatever sampling-noise bumps occur, on the first and only
    draw. This is what the closest published precedent (Nassar/Behrens/
    Glaze-style predictive-inference tasks) actually does -- outcomes
    drawn directly from the generative distribution, no correction, no
    selection.

  Branch B (generate_sequences_momentmatch.py): exact quota / moment
    matching, explicitly minimizing bumps via seed search. Introduces
    finite-population predictability by design, in exchange for smoothness
    -- matching how the gambler's-fallacy / probability-matching literature
    manipulates sequences deliberately, expecting it to be behaviorally
    consequential.

Method
------
Continuous block: n draws from Normal(true_mean, true_std), rejection-sampled
  ONLY to keep individual values within [0,100] (a bound, not a smoothing
  check -- matches ordinary truncated-Normal practice). No block-level
  accept/reject on the resulting sample mean/std, no re-draw.
Binary block: n draws from Bernoulli(true_p), values in {-1,+1}. No
  block-level accept/reject on the resulting count, no re-draw.

Usage
-----
  # The only mode: single generation from a given seed. No --n_tries,
  # no search, no scoring -- pick a seed (or leave the default) and go.
  python task/generate_sequences_iid.py --task continuous --seed 0
  python task/generate_sequences_iid.py --task both --seed 0

  # Diagnostic report of realized vs target moments (no file written) --
  # informational only, does not feed back into generation.
  python task/generate_sequences_iid.py --task continuous --report

Output
------
  Saved to task/sequences/{task}_iid_sequences.{pkl,json} by default --
  deliberately not the production filenames.
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import pandas as pd

from generate_sequences import (
    VALUE_MIN, VALUE_MAX,
    make_rng, mirror_sequence, mirror_params,
    continuous_param_grid, binary_param_grid,
    draw_continuous_obs, draw_binary_obs,
    check_sequence_plausibility,
    _save_sequences,
)


# ---------------------------------------------------------------------------
# Prefix uniqueness fix (real bug, confirmed empirically -- see module
# docstring's "Known bug, fixed" note below)
# ---------------------------------------------------------------------------
def _draw_unique_binary_prefix(rng, true_p, prefix_length, used_prefixes, max_attempts=10_000):
    """draw_binary_obs, retried until the result hasn't already been used by
    an earlier qid in THIS generation call. Fixes a real, confirmed bug:
    with prefix_length=4 there are only 2**4=16 possible binary sequences
    total, so independently drawing one per qid with no uniqueness check
    collides often -- confirmed empirically, 9 of 10 random seeds produced a
    real collision at n_unique_sequences=10 (only 6-8 distinct prefixes out
    of 10 expected). Exactly the same class of bug
    generate_sequences_momentmatch.py's build_binary_prefixes was redesigned
    to fix earlier this project, just reached via a different generation
    path here (that fix doesn't apply directly since this script's prefixes
    aren't drawn from a shared composition grid -- each qid has its own
    true_p).

    Fails loudly rather than silently returning a duplicate -- if
    n_unique_sequences approaches or exceeds 2**prefix_length this is
    structurally impossible, not just unlucky, and retrying forever
    wouldn't help.
    """
    for _ in range(max_attempts):
        candidate = tuple(draw_binary_obs(rng, true_p, prefix_length))
        if candidate not in used_prefixes:
            used_prefixes.add(candidate)
            return list(candidate)
    raise RuntimeError(
        f"Could not draw a unique {prefix_length}-length binary prefix after "
        f"{max_attempts} attempts (true_p={true_p}). {len(used_prefixes)} prefixes "
        f"already used out of {2**prefix_length} possible at this prefix_length -- "
        f"if n_unique_sequences approaches or exceeds 2**prefix_length, this is "
        f"structurally impossible, not just unlucky; reduce n_unique_sequences or "
        f"increase prefix_length.")


# ---------------------------------------------------------------------------
# Trial-assembly helpers (same minimal local copies as
# generate_sequences_momentmatch.py, kept separate per "cleanly separate")
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
# Core generation -- genuinely unconstrained i.i.d., single draw, no search
# ---------------------------------------------------------------------------
def generate_task_sequences_iid(task, args, rng):
    """Generate all trials for one task via plain i.i.d. sampling.

    Same trial structure (prefix/suffix split, mirrored qids, ITI schedule,
    no-consecutive-qid shuffle) as generate_sequences.py and
    generate_sequences_momentmatch.py, but blocks are drawn once, with no
    accept/reject step of any kind on the resulting composition, and no
    outer search/selection over seeds. Returns (DataFrame, json_trials),
    same schema as the other two scripts.
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
    print(f"Task: {task.upper()}  (pure i.i.d., minimal, no search)")
    print(f"  Total trials   : {n_total} ({n_unique} seqs x {n_repeats} reps)")
    print(f"  Seq / prefix / suffix length: {seq_length} / {prefix_length} / {suffix_length}")

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

    # -- Build prefix block (once per base qid, plain i.i.d.) ----------------
    # used_prefixes tracks every prefix assigned so far in THIS call (base
    # AND mirrored) -- binary needs active dedup (see
    # _draw_unique_binary_prefix's docstring for why); continuous gets only
    # a safety-net assertion since real-valued draws essentially never
    # collide exactly.
    used_prefixes = set()
    templates = []
    for qid, (true_mean, true_std, true_p) in enumerate(param_list):
        if task == 'continuous':
            prefix = draw_continuous_obs(rng, true_mean, true_std, prefix_length)
            used_prefixes.add(tuple(prefix))
        else:
            prefix = _draw_unique_binary_prefix(rng, true_p, prefix_length, used_prefixes)
        templates.append({
            'qid': qid, 'true_mean': true_mean, 'true_std': true_std,
            'true_p': true_p, 'prefix': prefix,
        })
        label = f'mean={true_mean:.1f}' if task == 'continuous' else f'p={true_p:.3f}'
        print(f"  qid {qid:3d}: {label}  prefix=[{','.join(map(str, prefix))}]")

    # -- Mirror templates -----------------------------------------------------
    # Mirroring is deterministic (not a fresh draw), so on the rare chance a
    # mirrored prefix collides with something already used, there's nothing
    # to "retry" -- fall back to a fresh unique draw from the MIRRORED
    # target's own true_p instead of the mirror transform, for that one qid
    # only. Expected to essentially never trigger in practice (mirroring a
    # specific base prefix landing exactly on an already-used one is a rare
    # coincidence on top of an already-rare event), but silently allowing a
    # duplicate here would undo the whole point of the fix above.
    for base in templates[:n_base]:
        m_mean, m_std, m_p = mirror_params(base['true_mean'], base['true_std'],
                                           base['true_p'], task)
        m_prefix = mirror_sequence(base['prefix'], task)
        mirror_note = ''
        if task == 'continuous':
            used_prefixes.add(tuple(m_prefix))
        else:
            if tuple(m_prefix) in used_prefixes:
                m_prefix = _draw_unique_binary_prefix(rng, m_p, prefix_length, used_prefixes)
                mirror_note = ' [mirror collided -- redrawn fresh instead]'
            else:
                used_prefixes.add(tuple(m_prefix))
        m_qid    = len(templates)
        templates.append({
            'qid': m_qid, 'true_mean': m_mean, 'true_std': m_std,
            'true_p': m_p, 'prefix': m_prefix,
        })
        label = f'mean={m_mean:.1f}' if task == 'continuous' else f'p={m_p:.3f}'
        print(f"  qid {m_qid:3d}: {label}  prefix=[{','.join(map(str, m_prefix))}]  "
              f"[mirror of qid {base['qid']}]{mirror_note}")

    # -- ITI schedule ---------------------------------------------------------
    ITI_MS   = 1000
    iti_sched = _build_iti_schedule(rng, templates, n_repeats)

    # -- Build trials: shared prefix + fresh i.i.d. suffix per repeat ---------
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
                    suffix = draw_continuous_obs(rng, tmpl['true_mean'],
                                                 tmpl['true_std'], suffix_length)
                else:
                    suffix = draw_binary_obs(rng, tmpl['true_p'], suffix_length)
                rep_idx = rep_count.get(qid, 0)
                iti_condition = iti_sched[qid][rep_idx]
                rep_count[qid] = rep_idx + 1
                trials.append({**tmpl, 'values': tmpl['prefix'] + suffix,
                               'iti_ms': ITI_MS, 'iti_condition': iti_condition})

    trials = _shuffle_no_consecutive_qid(trials, rng, task)

    # -- Build DataFrame + JSON (same schema as the other two scripts) -------
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

    # -- Sanity checks ---------------------------------------------------------
    assert len(df) == n_total * seq_length
    if task == 'continuous':
        assert df['value'].between(VALUE_MIN, VALUE_MAX).all()
    else:
        assert df['value'].isin([-1, 1]).all()
    assert df['trial'].nunique() == n_total
    assert (df.groupby('qid')['trial'].nunique() == n_repeats).all()
    n_distinct_prefixes = len(set(tuple(t['prefix']) for t in templates))
    assert n_distinct_prefixes == n_unique, (
        f"prefix collision: {n_distinct_prefixes}/{n_unique} distinct prefixes -- "
        f"should be unreachable given _draw_unique_binary_prefix's own dedup; "
        f"indicates a real bug if this ever fires.")

    print(f"\n  {len(df)} rows | {n_total} trials | {n_unique} seqs x {n_repeats} reps")
    if task == 'continuous':
        print(f"  Value range : {df['value'].min()} - {df['value'].max()}")
        print(f"  true_mean   : {df['true_mean'].min():.1f} - {df['true_mean'].max():.1f}")
    else:
        print(f"  true_p range: {df['true_p'].min():.3f} - {df['true_p'].max():.3f}")

    return df, json_trials


# ---------------------------------------------------------------------------
# Diagnostic report: realized vs target, per trial (informational only --
# nothing here gates or influences generation)
# ---------------------------------------------------------------------------
def report_moments(task, args, rng):
    df, _ = generate_task_sequences_iid(task, args, rng)
    print(f"\n{'='*60}\nRealized-vs-target report ({task}, pure i.i.d.):")
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
                      f"std={args.std_fixed:5.1f}  realized mean={np.mean(vals):6.2f} "
                      f"std={np.std(vals):5.2f}  k=1.0 pass={ok}")
            else:
                print(f"  qid={qid:2d} trial={trial_id:3d}  target p={tp:.3f}  "
                      f"realized p={np.mean(np.array(vals) == 1):.3f}  k=1.0 pass={ok}")


# ---------------------------------------------------------------------------
# CLI -- no --n_tries, no search: one seed, one draw
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--task',               choices=['continuous', 'binary', 'both'], default='both')
    p.add_argument('--n_unique_sequences', type=int,   default=10)
    p.add_argument('--n_repeats',          type=int,   default=4)
    p.add_argument('--seq_length',         type=int,   default=15)
    p.add_argument('--prefix_length',      type=int,   default=4)
    p.add_argument('--mean_range',         type=float, nargs=2, default=[20.0, 80.0])
    p.add_argument('--std_fixed',          type=float, default=15.0)
    p.add_argument('--p_range',            type=float, nargs=2, default=[0.2, 0.8])
    p.add_argument('--k_std_cont',         type=float, default=0.7,
                   help='Diagnostic only (--report) -- never used to gate generation')
    p.add_argument('--output_dir',         default='task/sequences')
    p.add_argument('--seed',               type=int,   default=0)
    p.add_argument('--report',             action='store_true',
                   help='Print realized-vs-target per trial; no file written')
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
    else:
        print(f"Generating sequences (pure i.i.d., minimal, single draw) | seed={args.seed}")
        for task in tasks:
            task_seed = args.seed if task == 'continuous' else args.seed + 1000
            rng = make_rng(task_seed)
            df, json_trials = generate_task_sequences_iid(task, args, rng)
            pkl_path, json_path = _save_sequences(df, json_trials, f'{task}_iid', out_dir)
            print(f"\n  Saved: {json_path}\n  Saved: {pkl_path}")
    print("\nJOB_COMPLETE")


if __name__ == '__main__':
    main()
