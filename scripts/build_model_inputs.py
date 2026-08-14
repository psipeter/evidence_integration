#!/usr/bin/env python3
"""scripts/build_model_inputs.py — build data/soltani_numbers.pkl and
data/soltani_colors.pkl from a parsed task_results pilot file, in the same
tidy, per-dataset-pkl shape carrabin.pkl/yoo.pkl already use, so the
existing fitting/models/* pipeline can be pointed at this task with no
further schema-specific changes.

Steps (see the conversation that produced this script for full rationale):
  1. Load the parsed pilot data and apply utils.participant_filters
     (excludes (pid, task) pairs that fail no_integration /
     noncontingent_sign / noncontingent_magnitude) -- this is the one
     deliberate departure from carrabin/yoo's own pipeline, which has no
     analogous exclusion step of its own.
  2. Keep successful attempts only (timed_out == False), dedup to one row
     per (prolific_pid, task, trial, observation).
  3. Map prolific_pid (str) -> a stable integer `pid`, since every
     downstream piece of fitting/* (fit.py, submit.py, losses.py) assumes
     `pid` is an int and queries with `==`. The mapping is built ONCE
     across both tasks together (sorted prolific_pid -> 1, 2, 3, ...) so a
     participant who did both tasks keeps the same integer pid in both
     output files.
  4. Rescale value/response from this task's native [0, 100] scale to the
     canonical [-1, 1] scale carrabin.pkl and yoo.pkl both already use
     (verified directly: both files' value/response columns range exactly
     -1..1). Numbers: value and response both go through x' = x/50 - 1.
     Colors: `value` is already +-1 (blue/red), so only `response` gets
     rescaled the same way. Doing this means models/math_models.py's
     EXISTING _run_carrabin/_run_yoo model code (including clip bounds)
     can be reused verbatim for this task's data -- no scale-specific
     branching needed anywhere downstream.
  5. Keep qid (both tasks) and true_p (colors)/true_mean (numbers) as
     supplementary columns beyond carrabin/yoo's own minimal schemas --
     harmless extras, not read by any existing dispatch code, but useful
     for later validation. true_p is left on its native [0,1] probability
     scale (matching carrabin's own true_p convention exactly); true_mean
     is rescaled the same way as value/response for consistency.

Run:
    python scripts/build_model_inputs.py
    python scripts/build_model_inputs.py --results_file task_results_pilot2.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import data_path
from utils.participant_filters import DEFAULT_EXCLUSION_METHOD, filter_participants


def _dedup_successful(df: pd.DataFrame, task: str) -> pd.DataFrame:
    sub = df[(df["task"] == task) & (df["timed_out"] == False)]
    return sub.drop_duplicates(subset=["prolific_pid", "trial", "observation"])


def _rescale_0_100_to_neg1_1(x: pd.Series) -> pd.Series:
    return x / 50.0 - 1.0


def build_from_df(df, out_name_numbers="soltani_numbers", out_name_colors="soltani_colors",
                  apply_filters=True, exclusion_method=DEFAULT_EXCLUSION_METHOD,
                  max_error_sd=None, min_skill=None,
                  require_both_tasks=True):
    """Core logic, extracted from build() so a different raw-data SOURCE
    (e.g. scripts/build_task_backend_inputs.py, pulling from Supabase
    instead of a parsed JATOS-era task_results pilot file) can reuse this
    exact filter+rescale+anonymize+save pipeline without duplicating it --
    the only thing that differs between sources is how `df` itself gets
    built; everything from here on is source-agnostic as long as `df` has
    the columns this function expects (see module docstring: prolific_pid,
    task ('numbers'/'colors'), trial, observation, value, response,
    timed_out, qid, true_p, true_mean -- value/response on their NATIVE
    [0,100] (numbers) / already-{-1,+1} (colors) scale, i.e. BEFORE any
    rescaling -- that happens in here, not before calling this).

    `exclusion_method` selects WHICH criterion set drives exclusion --
    'contingency' (the Cohen's f2 tests) or 'performance' (carrabin's model-free
    gross-outlier rule). See utils/participant_filters.py; both are always
    computed and reported, only the decision differs.

    apply_filters=False skips utils.participant_filters entirely, keeping every
    participant. Intended for diagnosing how much the exclusion criteria
    actually change a result -- build both versions under different out_names
    and compare. NOTE the pid mapping is derived from whoever survives
    filtering, so an unfiltered build assigns DIFFERENT integer pids to the
    same people than a filtered build does; the two are not comparable
    pid-by-pid, only in aggregate.
    """
    if apply_filters:
        kw = {"method": exclusion_method}
        if max_error_sd is not None:
            kw["max_error_sd"] = max_error_sd
        if min_skill is not None:
            kw["min_skill"] = min_skill
        df = filter_participants(df, verbose=True,
                                 require_both_tasks=require_both_tasks, **kw)
    else:
        n_pid = df["prolific_pid"].nunique()
        print(f"\n*** FILTERS DISABLED -- keeping all {n_pid} participants "
              f"({len(df)} rows). Do NOT use this build for published results. ***")

    # One pid mapping shared across both tasks, so a participant who did
    # both tasks gets the same integer pid in both output files.
    all_pids = sorted(df["prolific_pid"].unique())
    pid_map = {p: i + 1 for i, p in enumerate(all_pids)}
    print(f"\nBuilt pid mapping for {len(pid_map)} prolific_pids "
          f"(1..{len(pid_map)})")

    # ── numbers -> data/soltani_numbers.pkl ──────────────────────────
    cont = _dedup_successful(df, "numbers").copy()
    cont["pid"] = cont["prolific_pid"].map(pid_map).astype("int64")
    cont["value"] = _rescale_0_100_to_neg1_1(cont["value"])
    cont["response"] = _rescale_0_100_to_neg1_1(cont["response"])
    cont["true_mean"] = _rescale_0_100_to_neg1_1(cont["true_mean"])
    cont = cont[["pid", "trial", "observation", "qid", "value", "response", "true_mean"]]
    for col in ["trial", "observation", "qid"]:
        cont[col] = cont[col].astype("int64")
    cont = cont.sort_values(["pid", "trial", "observation"]).reset_index(drop=True)

    # ── colors -> data/soltani_colors.pkl ──────────────────────────────────
    binr = _dedup_successful(df, "colors").copy()
    binr["pid"] = binr["prolific_pid"].map(pid_map).astype("int64")
    binr["response"] = _rescale_0_100_to_neg1_1(binr["response"])
    # value is already +-1 (blue/red) -- no rescale needed
    binr = binr[["pid", "trial", "observation", "qid", "value", "response", "true_p"]]
    for col in ["trial", "observation", "qid", "value"]:
        binr[col] = binr[col].astype("int64")
    binr = binr.sort_values(["pid", "trial", "observation"]).reset_index(drop=True)

    for name, out in [(out_name_numbers, cont), (out_name_colors, binr)]:
        if out.empty:
            # E.g. a pilot that only touched one task -- writing an empty
            # file for the other is pure junk output, not a real result,
            # and could be mistaken for "this pilot had zero colors data"
            # rather than "this pilot never ran that task at all".
            print(f"\n{name}: 0 rows -- SKIPPED (this call's input had no rows for this task)")
            continue
        path = data_path(f"{name}.pkl")
        out.to_pickle(path)
        print(f"\n{name}: {len(out)} rows, {out['pid'].nunique()} pids "
              f"-> {path}")
        print(f"  value range:    [{out['value'].min():.3f}, {out['value'].max():.3f}]")
        print(f"  response range: [{out['response'].min():.3f}, {out['response'].max():.3f}]")
        print(out.head(3).to_string(index=False))


def build(results_file: str) -> None:
    df = pd.read_pickle(data_path(results_file))
    build_from_df(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="task_results_pilot2.pkl",
                        help="Filename under data/ produced by task/parse_results.py")
    args = parser.parse_args()
    build(args.results_file)
