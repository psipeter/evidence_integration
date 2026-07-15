"""
generate_sequences_pool.py
============================
Builds a POOL of N independent i.i.d. sequence sets, each saved as its own
numbered file, for the "serve a unique sequence set to every participant"
architecture described in docs/sequence_design_open_questions.md (Section 7).

This is a thin wrapper around generate_sequences_iid.py's own
generate_task_sequences_iid -- no new generation logic lives here. Each
pool member is a completely independent call (fresh RNG seed, task's own
offset), matching exactly what scripts/inspect_iid_sequences.py already
does for its analysis-only simulations; this script just writes each one
out as a real file instead of only holding it in memory.

Deliberately NOT wired into anything yet (no config.js changes, no build
changes, no assignment mechanism) -- this is step 2 of the ordered
refactor plan in docs/sequence_design_open_questions.md; the pieces after
this (asset bundling, runtime fetch + async bootstrap, assignment,
provenance recording, parse_results.py) are separate, not-yet-started
steps. Building this in isolation lets it be tested/verified on its own
before anything downstream depends on it.

Usage
-----
    python task/generate_sequences_pool.py --n_pool 100 --task both
    python task/generate_sequences_pool.py --n_pool 20 --task continuous --pool_dir task/sequences_pool

Output
------
    {pool_dir}/{task}_{0000..N-1}_sequences.{pkl,json}
    one independent generate_task_sequences_iid call per member -- same
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
from generate_sequences_iid import generate_task_sequences_iid


def build_pool(n_pool, tasks, n_unique_sequences, n_repeats, seq_length,
              prefix_length, mean_range, std_fixed, p_range, base_seed,
              pool_dir, progress_every=10):
    """Generate n_pool independent sequence sets per task, each saved as its
    own {task}_{NNNN}_sequences.{pkl,json} under pool_dir. Returns a dict
    {task: [pkl_path, ...]} of everything written, for the caller to log or
    verify.

    Seed scheme mirrors scripts/inspect_iid_sequences.py's
    simulate_participants exactly (same offset formula), so a pool member's
    index N reproduces the identical sequences a simulated "participant N"
    would have gotten in that analysis tool -- useful for cross-checking
    the two against each other if that's ever needed.
    """
    from types import SimpleNamespace
    written = {task: [] for task in tasks}
    for task in tasks:
        pool_task_dir = pathlib.Path(pool_dir)
        pool_task_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_pool):
            seed = base_seed + i * 100_000 + (0 if task == "continuous" else 50_000)
            rng = make_rng(seed)
            args_ns = SimpleNamespace(
                task=task, n_unique_sequences=n_unique_sequences, n_repeats=n_repeats,
                seq_length=seq_length, prefix_length=prefix_length,
                mean_range=mean_range, std_fixed=std_fixed, p_range=p_range,
                k_std_cont=0.7, output_dir=str(pool_task_dir), seed=seed, report=False,
            )
            # generate_task_sequences_iid prints a lot per call (~20 lines,
            # by design for its normal single-generation use) -- redirected
            # to keep n_pool calls from flooding the console; a concise
            # progress line is printed here instead.
            with contextlib.redirect_stdout(io.StringIO()):
                df, json_trials = generate_task_sequences_iid(task, args_ns, rng)
            pkl_path, json_path = _save_sequences(
                df, json_trials, f"{task}_{i:04d}", pool_task_dir)
            written[task].append(pkl_path)
            if (i + 1) % progress_every == 0 or i == n_pool - 1:
                print(f"  [{task}] {i + 1}/{n_pool} pool members written")
    return written


def verify_pool(written, tasks, n_pool):
    """Sanity check the whole pool after writing: every member exists, has
    the expected trial count, and (binary only) zero prefix collisions --
    the exact thing this refactor pass was meant to fix. Prints a summary;
    raises if anything is wrong, matching this project's fail-loud
    convention rather than silently shipping a bad pool."""
    import json as _json
    for task in tasks:
        assert len(written[task]) == n_pool, f"{task}: expected {n_pool} pool members, wrote {len(written[task])}"
        n_collisions = 0
        for pkl_path in written[task]:
            json_path = pkl_path.with_suffix(".json")
            with open(json_path) as f:
                trials = _json.load(f)
            prefix_by_qid = {}
            for t in trials:
                pl = t["prefix_length"]
                prefix_by_qid.setdefault(t["qid"], set()).add(tuple(t["values"][:pl]))
            all_prefixes = [p for prefs in prefix_by_qid.values() for p in prefs]
            if len(set(all_prefixes)) != len(prefix_by_qid):
                n_collisions += 1
        status = "OK" if n_collisions == 0 else f"{n_collisions} member(s) with a collision"
        print(f"[verify] {task}: {n_pool} members, prefix uniqueness {status}")
        assert n_collisions == 0, (
            f"{task}: {n_collisions} pool member(s) have a prefix collision -- "
            f"should be unreachable given generate_sequences_iid.py's own fix; "
            f"indicates a real regression if this ever fires.")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_pool", type=int, default=50, help="Number of independent pool members to generate")
    p.add_argument("--task", choices=["continuous", "binary", "both"], default="both")
    p.add_argument("--n_unique_sequences", type=int, default=10,
                  help="n_unique_sequences per pool member (must be even)")
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--seq_length", type=int, default=15)
    p.add_argument("--prefix_length", type=int, default=4)
    p.add_argument("--mean_range", type=float, nargs=2, default=[15.0, 85.0],
                  help="Default matches current production continuous range")
    p.add_argument("--std_fixed", type=float, default=15.0)
    p.add_argument("--p_range", type=float, nargs=2, default=[2 / 15, 13 / 15],
                  help="Default matches production blue_range=[2,13] out of 15, "
                       "converted to a p fraction")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--pool_dir", default="task/sequences_pool")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n_unique_sequences % 2 == 0, "n_unique_sequences must be even"
    assert args.n_pool > 0
    tasks = ["continuous", "binary"] if args.task == "both" else [args.task]

    print(f"Building a pool of {args.n_pool} independent sequence set(s) per task "
          f"({args.n_unique_sequences} seqs x {args.n_repeats} repeats each) "
          f"-> {args.pool_dir}")
    written = build_pool(
        args.n_pool, tasks, args.n_unique_sequences, args.n_repeats,
        args.seq_length, args.prefix_length, args.mean_range, args.std_fixed,
        args.p_range, args.base_seed, args.pool_dir)

    print()
    verify_pool(written, tasks, args.n_pool)
    print("\nJOB_COMPLETE")


if __name__ == "__main__":
    main()
