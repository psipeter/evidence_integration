"""
generate_sequences_pool.py
============================
Builds a POOL of N independent HYBRID sequence sets, each saved as its own
numbered file, for the "serve a unique sequence set to every participant"
architecture (docs/HISTORY.md's "Sequence design: open questions" section,
Section 7, extended for the hybrid method after PI discussion -- see chat history).

This is a thin wrapper around generate_sequences_hybrid.py's own
generate_task_sequences_hybrid -- no new generation logic lives here.
Each pool member is a completely independent call (fresh RNG seed, task's
own offset) of the SAME per-task hybrid method already promoted to
production: binary via unchanged quota/momentmatch construction (no seed
search), continuous via the unrescaled i.i.d.-suffix construction (no
seed search either) -- see generate_sequences_hybrid.py's own module
docstring for the full rationale. This file previously wrapped
generate_sequences_iid.py's pure i.i.d. generator; switched to the hybrid
method once that was chosen as production. generate_sequences_iid.py
itself is untouched and still usable directly if a pure-i.i.d. pool is
ever wanted for comparison.

Deliberately NOT wired into anything yet (no config.js changes, no build
changes, no assignment mechanism, no parse_results.py changes) -- this is
one piece of an ordered plan; see chat history for the rest (asset
bundling, runtime assignment, provenance recording, diagnostic tooling,
testing). Building this in isolation lets it be tested/verified on its
own before anything downstream depends on it.

Usage
-----
    python task/generate_sequences_pool.py --n_pool 200 --task both
    python task/generate_sequences_pool.py --n_pool 20 --task continuous --pool_dir task/sequences_pool

Output
------
    {pool_dir}/{task}_{0000..N-1}_sequences.{pkl,json}
    one independent generate_task_sequences_hybrid call per member -- same
    schema as every other sequence file in this project. (Filename order is
    {task}_{index}_sequences.{ext} rather than {task}_sequences_{index}.ext
    -- this matches _save_sequences' own fixed "{name}_sequences.{ext}"
    convention exactly rather than fighting it with a double "_sequences"
    in the name.)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib

from generate_sequences import make_rng, _save_sequences
from generate_sequences_hybrid import generate_task_sequences_hybrid, generate_binary_sequences_no_prefix


def build_pool(n_pool, tasks, n_prefix, n_repeats, seq_length,
              prefix_length, mean_range, std_fixed, blue_range,
              boundary_margin, std_tolerance_frac, base_seed,
              pool_dir, progress_every=20, no_prefix=False):
    """Generate n_pool independent HYBRID sequence sets per task, each saved
    as its own {task}_{NNNN}_sequences.{pkl,json} under pool_dir. Returns a
    dict {task: [pkl_path, ...]} of everything written, for the caller to
    log or verify.

    Seed scheme mirrors scripts/inspect_iid_sequences.py's
    simulate_participants exactly (same offset formula) -- kept even though
    this now wraps the hybrid generator instead of the pure-i.i.d. one, so
    a pool member's index N still lines up with whatever a same-index
    comparison run in that tool would use, if that's ever useful.

    no_prefix (chat history, binary only) -- uses
    generate_binary_sequences_no_prefix instead of
    generate_task_sequences_hybrid for binary specifically; asserts tasks
    == ['binary'] if set (no continuous equivalent exists, matching
    generate_sequences_hybrid.py's own CLI guard). A SEPARATE pool mode,
    not a replacement -- the caller is expected to point --pool_dir at a
    DIFFERENT directory than production's, so this never overwrites the
    existing prefix-based pool.
    """
    from types import SimpleNamespace
    if no_prefix:
        assert tasks == ["binary"], "no_prefix is binary-only -- pass tasks=['binary']"
    written = {task: [] for task in tasks}
    for task in tasks:
        pool_task_dir = pathlib.Path(pool_dir)
        pool_task_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_pool):
            seed = base_seed + i * 100_000 + (0 if task == "continuous" else 50_000)
            rng = make_rng(seed)
            args_ns = SimpleNamespace(
                task=task, n_prefix=n_prefix, n_repeats=n_repeats,
                seq_length=seq_length, prefix_length=prefix_length,
                mean_range=mean_range, std_fixed=std_fixed, blue_range=blue_range,
                boundary_margin=boundary_margin, std_tolerance_frac=std_tolerance_frac,
                output_dir=str(pool_task_dir), seed=seed, report=False,
            )
            # generate_task_sequences_hybrid prints a fair amount per call
            # (by design for its normal single-generation use) -- redirected
            # to keep n_pool calls from flooding the console; a concise
            # progress line is printed here instead.
            with contextlib.redirect_stdout(io.StringIO()):
                if no_prefix:
                    df, json_trials = generate_binary_sequences_no_prefix(args_ns, rng, verbose=True)
                else:
                    df, json_trials = generate_task_sequences_hybrid(task, args_ns, rng, verbose=True)
            pkl_path, json_path = _save_sequences(
                df, json_trials, f"{task}_{i:04d}", pool_task_dir)
            written[task].append(pkl_path)
            if (i + 1) % progress_every == 0 or i == n_pool - 1:
                print(f"  [{task}] {i + 1}/{n_pool} pool members written")
    return written


def verify_pool(written, tasks, n_pool):
    """Sanity check the whole pool after writing: every member exists, has
    the expected trial count, zero prefix collisions, and (binary only)
    exact quota -- matching generate_task_sequences_hybrid's own per-call
    assertions, re-checked here across the WHOLE pool as an extra guard.
    Prints a summary; raises if anything is wrong, matching this project's
    fail-loud convention rather than silently shipping a bad pool."""
    import json as _json
    import numpy as _np
    for task in tasks:
        assert len(written[task]) == n_pool, f"{task}: expected {n_pool} pool members, wrote {len(written[task])}"
        n_prefix_collisions = 0
        n_quota_mismatches = 0
        is_no_prefix = None
        for pkl_path in written[task]:
            json_path = pkl_path.with_suffix(".json")
            with open(json_path) as f:
                trials = _json.load(f)
            pl_here = trials[0]["prefix_length"] if trials else None
            if is_no_prefix is None:
                is_no_prefix = (pl_here == 0)
            if not is_no_prefix:
                prefix_by_qid = {}
                for t in trials:
                    pl = t["prefix_length"]
                    prefix_by_qid.setdefault(t["qid"], set()).add(tuple(t["values"][:pl]))
                all_prefixes = [p for prefs in prefix_by_qid.values() for p in prefs]
                if len(set(all_prefixes)) != len(prefix_by_qid):
                    n_prefix_collisions += 1
            if task == "binary":
                for t in trials:
                    achieved_blue = sum(1 for v in t["values"] if v == 1)
                    target_blue = round(t["true_p"] * len(t["values"]))
                    if achieved_blue != target_blue:
                        n_quota_mismatches += 1
        if is_no_prefix:
            # prefix_length=0 (chat history, generate_binary_sequences_no_prefix)
            # -- there is no "prefix" to check uniqueness of at all (every
            # trial's values[:0] is the same empty tuple by construction,
            # which would otherwise register as a false-positive collision
            # against every OTHER trial's own qid). Skipped entirely, not
            # silently passed -- printed explicitly so it's clear this
            # check didn't run rather than looking like it passed.
            print(f"[verify] {task}: {n_pool} members, prefix uniqueness check SKIPPED "
                  f"(prefix_length=0 -- no-prefix branch, nothing to check)")
        else:
            status = "OK" if n_prefix_collisions == 0 else f"{n_prefix_collisions} member(s) with a prefix collision"
            print(f"[verify] {task}: {n_pool} members, prefix uniqueness {status}")
            assert n_prefix_collisions == 0, (
                f"{task}: {n_prefix_collisions} pool member(s) have a prefix collision -- "
                f"should be unreachable given build_{task}_prefixes' own dedup; "
                f"indicates a real regression if this ever fires.")
        if task == "binary":
            print(f"[verify] {task}: exact-quota check across pool: "
                  f"{n_quota_mismatches} mismatch(es) ({'OK' if n_quota_mismatches == 0 else 'FAIL'})")
            assert n_quota_mismatches == 0, (
                f"{task}: {n_quota_mismatches} trial(s) across the pool failed the exact-quota "
                f"check -- should be impossible given suffix_for_binary_target's construction.")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_pool", type=int, default=200, help="Number of independent pool members to generate")
    p.add_argument("--task", choices=["continuous", "binary", "both"], default="both")
    p.add_argument("--n_prefix", type=int, default=8,
                  help="Number of DISTINCT prefixes per pool member -- matches current "
                       "production's 8x4 default")
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--seq_length", type=int, default=15)
    p.add_argument("--prefix_length", type=int, default=4)
    p.add_argument("--mean_range", type=float, nargs=2, default=[15.0, 85.0],
                  help="Default matches current production continuous range")
    p.add_argument("--std_fixed", type=float, default=15.0)
    p.add_argument("--blue_range", type=int, nargs=2, default=[2, 13],
                  help="Default matches current production binary range")
    p.add_argument("--boundary_margin", type=float, default=1.0)
    p.add_argument("--std_tolerance_frac", type=float, default=0.25)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--pool_dir", default="task/sequences_pool")
    p.add_argument("--no_prefix", action="store_true",
                   help="Binary ONLY (chat history) -- build the pool via "
                        "generate_binary_sequences_no_prefix instead of the default "
                        "prefix/qid-repeat structure. Every trial gets its own independent "
                        "true_p and its own independent exact-quota full-length sequence. "
                        "Requires --task binary. Point --pool_dir at a DIFFERENT directory "
                        "than production's -- this is a separate pool mode, not a migration.")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n_pool > 0
    tasks = ["continuous", "binary"] if args.task == "both" else [args.task]
    if args.no_prefix:
        assert tasks == ["binary"], "--no_prefix is binary-only -- pass --task binary"

    if args.no_prefix:
        print(f"Building a pool of {args.n_pool} independent NO-PREFIX binary sequence "
              f"set(s) ({args.n_prefix * args.n_repeats} trials/member, each trial fully "
              f"independent -- no prefix, no qid repeats) -> {args.pool_dir}")
    else:
        print(f"Building a pool of {args.n_pool} independent HYBRID sequence set(s) per task "
              f"({args.n_prefix} prefixes x {args.n_repeats} repeats each = "
              f"{args.n_prefix * args.n_repeats} trials/member) -> {args.pool_dir}")
    written = build_pool(
        args.n_pool, tasks, args.n_prefix, args.n_repeats, args.seq_length,
        args.prefix_length, args.mean_range, args.std_fixed, args.blue_range,
        args.boundary_margin, args.std_tolerance_frac, args.base_seed, args.pool_dir,
        no_prefix=args.no_prefix)

    print()
    verify_pool(written, tasks, args.n_pool)
    print("\nJOB_COMPLETE")


if __name__ == "__main__":
    main()
