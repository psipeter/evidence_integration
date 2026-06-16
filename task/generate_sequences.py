"""
generate_sequences.py
=====================
Generate stimulus sequences for the continuous and/or binary evidence
integration tasks.

Trial structure
---------------
Each trial has `seq_length` observations drawn from a generative distribution.
Two trial types are produced:

  structured  — the first `prefix_length` observations are fixed and shared
                across `n_repeats` trials; the tail (obs prefix_length+1 ..
                seq_length) is freshly randomised each repeat.
                If prefix_length == seq_length (or prefix_length == 0 with
                --full_repeat), the entire sequence is repeated identically.

  random      — all observations freshly randomised; qid = None.

Generative distributions
------------------------
  continuous : Normal(true_mean, true_std)
               true_mean ~ Uniform(mean_range)
               true_std  ~ Uniform(std_range)
               values clipped/redrawn to [value_min, value_max] (integers)

  binary     : Bernoulli(true_p)
               true_p ~ Uniform(p_range)
               values in {0, 1}

Rejection sampling criteria (applied to structured sequences only)
-------------------------------------------------------------------
  1. Prefix plausibility
       continuous : |mean(prefix) - true_mean| <= plausibility_k * true_std
       binary     : log P(prefix | true_p) >= log(plausibility_threshold)
                    (i.e. P(prefix|true_p) >= plausibility_threshold)
  2. Outlier cap
       continuous : count(|obs - true_mean| > outlier_sigma * true_std)
                    <= max_outliers  (applied over full sequence)
       binary     : not applicable (no concept of outlier for Bernoulli)

Parameter distribution coverage
---------------------------------
  n_unique_sequences are drawn such that their generative parameters tile the
  parameter space evenly. For continuous: a 2D grid over (true_mean, true_std)
  is used; for binary: a 1D grid over true_p. Each cell is filled with exactly
  one sequence, with parameters drawn uniformly within that cell.

Output
------
  sequences/{task}_sequences.json   — consumed by jsPsych (Vite import)
  sequences/{task}_sequences.pkl    — consumed by analysis pipeline

DataFrame columns (pkl):
  task, trial, qid, observation, value, true_mean, true_std (NaN for binary),
  true_p (NaN for continuous), trial_type ('structured'|'random'), prefix_length

JSON structure:
  list of trial objects, each with:
    { trial, qid, trial_type, true_mean, true_std (or true_p),
      values: [v0, v1, ...], prefix_length }

Usage examples
--------------
  # Default continuous (100 trials: 20 seqs × 5 reps)
  python task/generate_sequences.py --task continuous

  # Binary, more trials
  python task/generate_sequences.py --task binary --n_unique_sequences 16 \\
      --n_repeats 6 --n_random 4 --p_range 0.25 0.75

  # Both tasks, full-sequence repeats, custom output dir
  python task/generate_sequences.py --task both --prefix_length 0 \\
      --output_dir task/sequences/

  # Relaxed rejection sampling
  python task/generate_sequences.py --task continuous \\
      --plausibility_k 2.0 --max_outliers 4
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALUE_MIN = -100
VALUE_MAX = 100
MAX_REJECTION_ATTEMPTS = 10_000   # hard limit per sequence to avoid infinite loops


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Parameter grids (for even coverage)
# ---------------------------------------------------------------------------
def continuous_param_grid(n: int, mean_range, std_range, rng):
    """
    Tile (true_mean, true_std) space into n cells and sample one point per cell.
    Uses a 2D grid; if n is not a perfect square, uses ceil(sqrt(n)) × ceil(n/cols).
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    mean_edges = np.linspace(mean_range[0], mean_range[1], cols + 1)
    std_edges  = np.linspace(std_range[0],  std_range[1],  rows + 1)

    params = []
    indices = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(indices)
    for r, c in indices[:n]:
        mu  = rng.uniform(mean_edges[c], mean_edges[c + 1])
        sig = rng.uniform(std_edges[r],  std_edges[r + 1])
        params.append((round(mu, 4), round(sig, 4)))
    return params


def binary_param_grid(n: int, p_range, rng):
    """
    Tile true_p space into n equal bins and sample one point per bin.
    """
    edges = np.linspace(p_range[0], p_range[1], n + 1)
    params = []
    for i in range(n):
        p = rng.uniform(edges[i], edges[i + 1])
        params.append(round(p, 4))
    rng.shuffle(params)
    return params


# ---------------------------------------------------------------------------
# Observation generators
# ---------------------------------------------------------------------------
def draw_continuous_obs(rng, true_mean, true_std, n,
                        value_min=VALUE_MIN, value_max=VALUE_MAX):
    """Draw n integer observations from Normal(true_mean, true_std), in range."""
    vals = []
    attempts = 0
    while len(vals) < n:
        attempts += 1
        if attempts > MAX_REJECTION_ATTEMPTS * n:
            # Fallback: clip
            v = int(np.clip(np.round(rng.normal(true_mean, true_std)),
                            value_min, value_max))
        else:
            v = int(np.round(rng.normal(true_mean, true_std)))
            if not (value_min <= v <= value_max):
                continue
        vals.append(v)
    return vals


def draw_binary_obs(rng, true_p, n):
    """Draw n binary observations from Bernoulli(true_p): +1 or -1."""
    return [1 if rng.random() < true_p else -1 for _ in range(n)]


# ---------------------------------------------------------------------------
# Rejection sampling checks
# ---------------------------------------------------------------------------
def check_prefix_plausibility(prefix, true_mean, true_std, true_p,
                               task, plausibility_k, plausibility_threshold):
    """
    Returns True if the prefix is plausible given the generative parameters.

    Continuous: |mean(prefix) - true_mean| <= plausibility_k * true_std
    Binary:     P(prefix | true_p) >= plausibility_threshold
    """
    if task == 'continuous':
        if len(prefix) == 0:
            return True
        prefix_mean = np.mean(prefix)
        return abs(prefix_mean - true_mean) <= plausibility_k * true_std
    else:  # binary
        if len(prefix) == 0:
            return True
        n_blue = sum(prefix)
        n_red  = len(prefix) - n_blue
        # Probability of this exact prefix given true_p
        log_p = (n_blue * math.log(true_p + 1e-12) +
                 n_red  * math.log(1 - true_p + 1e-12))
        return log_p >= math.log(plausibility_threshold + 1e-12)


def check_outlier_cap(sequence, true_mean, true_std, outlier_sigma, max_outliers, task):
    """
    Returns True if the number of outliers in the full sequence <= max_outliers.
    Only meaningful for continuous task.
    """
    if task == 'binary':
        return True
    n_outliers = sum(
        abs(v - true_mean) > outlier_sigma * true_std for v in sequence
    )
    return n_outliers <= max_outliers


# ---------------------------------------------------------------------------
# Sequence builders
# ---------------------------------------------------------------------------
def build_structured_sequence(rng, task, true_mean, true_std, true_p,
                               seq_length, prefix_length,
                               plausibility_k, plausibility_threshold,
                               outlier_sigma, max_outliers):
    """
    Build one structured sequence with rejection sampling.
    Returns list of values of length seq_length, or raises RuntimeError.
    """
    full_repeat = (prefix_length == 0 or prefix_length >= seq_length)
    effective_prefix_len = seq_length if full_repeat else prefix_length

    for attempt in range(MAX_REJECTION_ATTEMPTS):
        # Draw the prefix
        if task == 'continuous':
            prefix = draw_continuous_obs(rng, true_mean, true_std, effective_prefix_len)
        else:
            prefix = draw_binary_obs(rng, true_p, effective_prefix_len)

        # Check prefix plausibility
        if not check_prefix_plausibility(
                prefix, true_mean, true_std, true_p,
                task, plausibility_k, plausibility_threshold):
            continue

        if full_repeat:
            sequence = prefix
        else:
            # Draw the random tail
            tail_len = seq_length - effective_prefix_len
            if task == 'continuous':
                tail = draw_continuous_obs(rng, true_mean, true_std, tail_len)
            else:
                tail = draw_binary_obs(rng, true_p, tail_len)
            sequence = prefix + tail

        # Check outlier cap on full sequence
        if not check_outlier_cap(sequence, true_mean, true_std,
                                  outlier_sigma, max_outliers, task):
            continue

        return sequence, prefix

    raise RuntimeError(
        f"Could not generate a valid structured sequence after "
        f"{MAX_REJECTION_ATTEMPTS} attempts "
        f"(task={task}, true_mean={true_mean}, true_std={true_std}, "
        f"true_p={true_p}). Consider relaxing rejection criteria."
    )


def build_random_sequence(rng, task, true_mean, true_std, true_p,
                          seq_length, outlier_sigma, max_outliers):
    """Build one fully random sequence (no prefix constraints, light outlier check)."""
    for attempt in range(MAX_REJECTION_ATTEMPTS):
        if task == 'continuous':
            sequence = draw_continuous_obs(rng, true_mean, true_std, seq_length)
        else:
            sequence = draw_binary_obs(rng, true_p, seq_length)

        if check_outlier_cap(sequence, true_mean, true_std,
                              outlier_sigma, max_outliers, task):
            return sequence

    # Fallback without outlier cap
    if task == 'continuous':
        return draw_continuous_obs(rng, true_mean, true_std, seq_length)
    else:
        return draw_binary_obs(rng, true_p, seq_length)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_task_sequences(task, args, rng):
    """
    Generate all trials for one task. Returns (DataFrame, list_of_dicts_for_json).
    """
    seq_length    = args.seq_length
    prefix_length = args.prefix_length if not args.full_repeat else 0
    full_repeat   = args.full_repeat or (prefix_length >= seq_length)
    effective_prefix_len = seq_length if full_repeat else prefix_length

    n_unique  = args.n_unique_sequences
    n_repeats = args.n_repeats
    n_structured = n_unique * n_repeats
    n_random  = args.n_random
    n_total   = n_structured + n_random

    print(f"\n{'='*60}")
    print(f"Task: {task.upper()}")
    print(f"  Total trials       : {n_total}")
    print(f"  Structured trials  : {n_structured} ({n_unique} seqs × {n_repeats} reps)")
    print(f"  Random trials      : {n_random}")
    print(f"  Sequence length    : {seq_length}")
    print(f"  Prefix length      : {'full' if full_repeat else effective_prefix_len}")
    print(f"  Rejection criteria :")
    print(f"    plausibility_k   : {args.plausibility_k}")
    print(f"    plausibility_thr : {args.plausibility_threshold}")
    print(f"    outlier_sigma    : {args.outlier_sigma}")
    print(f"    max_outliers     : {args.max_outliers}")

    # ── Generate generative parameter sets ──────────────────────────────────
    if task == 'continuous':
        param_sets = continuous_param_grid(
            n_unique, args.mean_range, args.std_range, rng)
        print(f"  Mean range         : {args.mean_range}")
        print(f"  Std range          : {args.std_range}")
    else:
        param_sets = binary_param_grid(n_unique, args.p_range, rng)
        print(f"  true_p range       : {args.p_range}")

    # ── Build structured sequences ───────────────────────────────────────────
    structured_templates = []  # (true_mean, true_std, true_p, prefix, qid)
    for qid, ps in enumerate(param_sets):
        if task == 'continuous':
            true_mean, true_std = ps
            true_p = float('nan')
        else:
            true_p = ps
            true_mean = true_p   # store true_p in true_mean column for binary
            true_std  = float('nan')

        sequence, prefix = build_structured_sequence(
            rng, task, true_mean, true_std, true_p,
            seq_length, prefix_length,
            args.plausibility_k, args.plausibility_threshold,
            args.outlier_sigma, args.max_outliers,
        )
        structured_templates.append({
            'qid':       qid,
            'true_mean': true_mean,
            'true_std':  true_std,
            'true_p':    true_p,
            'prefix':    prefix,
            'sequence':  sequence,  # the fixed part (prefix or full)
        })
        print(f"  qid {qid:3d}: params={'mean={:.1f} std={:.1f}'.format(true_mean,true_std) if task=='continuous' else 'p={:.3f}'.format(true_p)}"
              f"  prefix=[{','.join(map(str,prefix[:5]))}{'...' if len(prefix)>5 else ''}]")

    # ── Expand structured trials (one row per repeat) ────────────────────────
    structured_trials = []
    for tmpl in structured_templates:
        for rep in range(n_repeats):
            if full_repeat:
                values = tmpl['sequence']
            else:
                # Fresh random tail each repeat, same fixed prefix
                if task == 'continuous':
                    tail = draw_continuous_obs(
                        rng, tmpl['true_mean'], tmpl['true_std'],
                        seq_length - effective_prefix_len)
                else:
                    tail = draw_binary_obs(
                        rng, tmpl['true_p'],
                        seq_length - effective_prefix_len)
                values = tmpl['prefix'] + tail
            structured_trials.append({
                'qid':        tmpl['qid'],
                'trial_type': 'structured',
                'true_mean':  tmpl['true_mean'],
                'true_std':   tmpl['true_std'],
                'true_p':     tmpl['true_p'],
                'values':     values,
            })

    # ── Generate random trials ────────────────────────────────────────────────
    random_trials = []
    for _ in range(n_random):
        if task == 'continuous':
            true_mean = round(rng.uniform(*args.mean_range), 4)
            true_std  = round(rng.uniform(*args.std_range), 4)
            true_p    = float('nan')
        else:
            true_p    = round(rng.uniform(*args.p_range), 4)
            true_mean = true_p
            true_std  = float('nan')

        values = build_random_sequence(
            rng, task, true_mean, true_std, true_p,
            seq_length, args.outlier_sigma, args.max_outliers)

        random_trials.append({
            'qid':        None,
            'trial_type': 'random',
            'true_mean':  true_mean,
            'true_std':   true_std,
            'true_p':     true_p,
            'values':     values,
        })

    # ── Shuffle and assign trial numbers ─────────────────────────────────────
    all_trials = structured_trials + random_trials
    rng.shuffle(all_trials)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    records = []
    json_trials = []

    for t, trial in enumerate(all_trials):
        for o, v in enumerate(trial['values'], start=1):
            records.append({
                'task':         task,
                'trial':        t,
                'qid':          trial['qid'],
                'trial_type':   trial['trial_type'],
                'observation':  o,
                'value':        v,
                'true_mean':    trial['true_mean'],
                'true_std':     trial['true_std'],
                'true_p':       trial['true_p'],
                'prefix_length': effective_prefix_len if trial['trial_type'] == 'structured' else 0,
            })

        json_trial = {
            'trial':        t,
            'qid':          trial['qid'],
            'trial_type':   trial['trial_type'],
            'true_mean':    None if math.isnan(trial['true_mean']) else trial['true_mean'],
            'true_std':     None if math.isnan(trial['true_std'])  else trial['true_std'],
            'true_p':       None if math.isnan(trial['true_p'])    else trial['true_p'],
            'values':       trial['values'],
            'prefix_length': effective_prefix_len if trial['trial_type'] == 'structured' else 0,
        }
        json_trials.append(json_trial)

    df = pd.DataFrame(records)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert len(df) == n_total * seq_length, "Row count mismatch"
    if task == 'continuous':
        assert df['value'].between(VALUE_MIN, VALUE_MAX).all(), "Out-of-range values"
    else:
        assert df['value'].isin([-1, 1]).all(), "Non-binary values"
    n_struct_check = df[df['trial_type']=='structured']['trial'].nunique()
    n_rand_check   = df[df['trial_type']=='random']['trial'].nunique()
    assert n_struct_check == n_structured, f"Expected {n_structured} structured trials, got {n_struct_check}"
    assert n_rand_check   == n_random,     f"Expected {n_random} random trials, got {n_rand_check}"

    # Check that each qid appears exactly n_repeats times
    if n_unique > 0:
        qid_counts = (df[df['trial_type']=='structured']
                      .groupby('qid')['trial'].nunique())
        assert (qid_counts == n_repeats).all(), \
            f"Uneven repeat counts: {qid_counts.to_dict()}"

    print(f"\n  ✓ {len(df)} rows | {n_struct_check} structured + {n_rand_check} random trials")
    if task == 'continuous':
        print(f"  Value range: {df['value'].min()} – {df['value'].max()}")
        print(f"  true_mean range: {df['true_mean'].min():.1f} – {df['true_mean'].max():.1f}")
        print(f"  true_std range:  {df['true_std'].min():.1f} – {df['true_std'].max():.1f}")
    else:
        print(f"  true_p range: {df['true_p'].min():.3f} – {df['true_p'].max():.3f}")

    return df, json_trials


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Generate stimulus sequences for the evidence integration task.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task selection
    p.add_argument('--task', choices=['continuous', 'binary', 'both'],
                   default='both',
                   help='Which task(s) to generate sequences for')

    # Trial counts
    p.add_argument('--n_unique_sequences', type=int, default=20,
                   help='Number of distinct repeated sequences')
    p.add_argument('--n_repeats', type=int, default=5,
                   help='Number of repeats per unique sequence')
    p.add_argument('--n_random', type=int, default=0,
                   help='Number of fully random (non-repeated) trials')

    # Sequence structure
    p.add_argument('--seq_length', type=int, default=15,
                   help='Total observations per trial')
    p.add_argument('--prefix_length', type=int, default=8,
                   help='Number of fixed leading observations (0 or >= seq_length = full repeat)')
    p.add_argument('--full_repeat', action='store_true',
                   help='Repeat the entire sequence (overrides prefix_length)')

    # Generative parameters — continuous
    p.add_argument('--mean_range', type=float, nargs=2, default=[-60.0, 60.0],
                   metavar=('LO', 'HI'),
                   help='Range of true_mean for continuous task')
    p.add_argument('--std_range', type=float, nargs=2, default=[10.0, 20.0],
                   metavar=('LO', 'HI'),
                   help='Range of true_std for continuous task')

    # Generative parameters — binary
    p.add_argument('--p_range', type=float, nargs=2, default=[0.25, 0.75],
                   metavar=('LO', 'HI'),
                   help='Range of true_p for binary task')

    # Rejection sampling
    p.add_argument('--plausibility_k', type=float, default=1.5,
                   help='Continuous: max |prefix_mean - true_mean| / true_std')
    p.add_argument('--plausibility_threshold', type=float, default=1e-4,
                   help='Binary: minimum P(prefix | true_p) to accept. '
                        'Default 1e-4 allows rare-but-valid prefixes for near-0.5 true_p. '
                        'Scales naturally: stricter values (e.g. 0.01) work for short prefixes.')
    p.add_argument('--outlier_sigma', type=float, default=2.0,
                   help='Outlier threshold in units of true_std')
    p.add_argument('--max_outliers', type=int, default=3,
                   help='Maximum outliers allowed per sequence')

    # Output
    p.add_argument('--output_dir', type=str, default='task/sequences',
                   help='Directory to write output files')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing output files')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Validate
    assert args.n_unique_sequences > 0, "--n_unique_sequences must be > 0"
    assert args.n_repeats > 0, "--n_repeats must be > 0"
    assert args.seq_length > 0, "--seq_length must be > 0"
    assert 0 <= args.prefix_length <= args.seq_length, \
        f"--prefix_length must be in [0, seq_length={args.seq_length}]"
    assert args.mean_range[0] < args.mean_range[1], "--mean_range lo must be < hi"
    assert args.std_range[0]  < args.std_range[1],  "--std_range lo must be < hi"
    assert 0 < args.p_range[0] < args.p_range[1] < 1, \
        "--p_range must be strictly within (0, 1)"

    n_structured = args.n_unique_sequences * args.n_repeats
    n_total = n_structured + args.n_random
    print(f"Generating sequences | seed={args.seed} | total trials={n_total}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ['continuous', 'binary'] if args.task == 'both' else [args.task]
    all_dfs = []

    for task in tasks:
        # Use task-offset seeds so continuous and binary sequences are independent
        task_seed = args.seed if task == 'continuous' else args.seed + 1000
        rng = make_rng(task_seed)

        df, json_trials = generate_task_sequences(task, args, rng)
        all_dfs.append(df)

        # ── Save JSON (consumed by jsPsych) ───────────────────────────────────
        json_path = out_dir / f'{task}_sequences.json'
        if json_path.exists() and not args.overwrite:
            print(f"\n  Skipping {json_path} (already exists; use --overwrite)")
        else:
            with open(json_path, 'w') as f:
                json.dump(json_trials, f, indent=2)
            print(f"\n  Saved: {json_path}")

        # ── Save PKL (consumed by analysis pipeline) ──────────────────────────
        pkl_path = out_dir / f'{task}_sequences.pkl'
        if pkl_path.exists() and not args.overwrite:
            print(f"  Skipping {pkl_path} (already exists; use --overwrite)")
        else:
            df.to_pickle(pkl_path)
            print(f"  Saved: {pkl_path}")

        # ── Also copy JSON to src/{task}/ for Vite ────────────────────────────
        vite_dir = pathlib.Path('task/src') / task
        if vite_dir.exists():
            vite_path = vite_dir / 'sequences.json'
            with open(vite_path, 'w') as f:
                json.dump(json_trials, f, indent=2)
            print(f"  Copied to Vite: {vite_path}")

    # ── Combined pkl (both tasks together) ────────────────────────────────────
    if len(all_dfs) == 2:
        combined_path = out_dir / 'all_sequences.pkl'
        combined = pd.concat(all_dfs, ignore_index=True)
        if combined_path.exists() and not args.overwrite:
            print(f"\n  Skipping {combined_path} (use --overwrite)")
        else:
            combined.to_pickle(combined_path)
            print(f"\n  Saved combined: {combined_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
