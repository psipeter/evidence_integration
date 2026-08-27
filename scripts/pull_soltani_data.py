#!/usr/bin/env python3
"""scripts/pull_soltani_data.py
=========================================
Steps 1-3 of the soltani data pipeline (step 4 -- rescale + anonymize +
save the canonical .pkl -- lives in scripts/build_model_inputs.py's
build_from_df, which this script calls into rather than duplicating):

  1. PULL the full raw event log for a task straight from Supabase's
     `events` table (_fetch_all_events) -- every phase, every attempt, not
     pre-filtered in any way.
  2. FILTER to an explicit list of finished prolific_pids (either passed by
     hand via --numbers_pids/--colors_pids, or auto-derived by
     --complete_pairs -- see that flag's own comment for why it's derived
     live rather than from a stored list), then hand off to build_from_df,
     which applies the REAL statistical exclusion criteria
     (utils/participant_filters.py) on top of that -- currently 45 pids
     survive both stages for --complete_pairs.
  3. Persistent PID REGISTRY: build_from_df resolves each surviving
     prolific_pid to its stable integer pid via utils/pid_registry.py
     (append-only -- an existing participant's pid never changes, no
     matter how many times this script runs or how the pool grows; see
     that module's own docstring for the bug this replaced).

Downloads real, finished task_backend participant data directly from
Supabase for an EXPLICIT list of prolific_pids, reformats it to match
exactly what build_from_df() expects, then calls that SAME shared
filter+rescale+anonymize+save pipeline carrabin/yoo/the old JATOS-pilot
data all already go through -- see build_model_inputs.py's own
build_from_df docstring for why this is reused rather than reimplemented.

WHY AN EXPLICIT PID LIST, NOT "everyone finished so far"
-----------------------------------------------------------
An earlier version of this script grabbed every finished real participant
it could find and wrote them all into ONE pair of files. That breaks the
moment there's more than one pilot round with different generative
parameters (e.g. numbers' std_fixed=15 vs std_fixed=10) -- there's no way
to compare pilot 4 against pilot 5 if they're silently merged into the
same file every time this runs. Different pilots are different PEOPLE
(no participant did both), so there's no need for cross-pilot pid-number
consistency the way cross-TASK consistency matters within one pilot
(build_from_df's own pid mapping already handles that correctly -- and,
via the persistent registry, now also stays consistent ACROSS separate
runs of this script, not just within one call).

Usage:
    # Probe which real, finished pids exist right now (for building a list):
    python scripts/pull_soltani_data.py --list_candidates numbers

    # Canonical production build (steps 1-4 end to end):
    python scripts/pull_soltani_data.py --complete_pairs

    # Build one PILOT's files from an explicit pid list instead:
    python scripts/pull_soltani_data.py --pilot pilot4 \\
        --numbers_pids 670bd903349d5d24bc92dcb0,69163607e65df2b5dbe294fa,... \\
        --colors_pids 670bd903349d5d24bc92dcb0,69163607e65df2b5dbe294fa,...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_model_inputs import build_from_df
from utils.participant_filters import (
    DEFAULT_EXCLUSION_METHOD,
    EXCLUSION_METHODS,
)

TASK_BACKEND_DIR = Path(__file__).resolve().parents[1] / "task_backend"
TASK_INTERNAL = {"numbers": "numbers", "colors": "colors"}

# CURRENT numbers-task generative std, mirroring task_backend/
# generate_sequences.py's own NUMBERS_STD_FIXED -- duplicated here rather
# than imported (task_backend/ is primarily a JS/Vite app with
# generate_sequences.py as its one standalone Python utility, not a
# package this analysis pipeline otherwise reaches into; no other script
# under scripts/ or utils/ imports from task_backend/ either). MUST be
# kept in sync BY HAND if that constant ever changes again -- it already
# has, twice (see that file's own history comment: 10 -> 15 -> 10).
#
# --complete_pairs uses this to exclude any pid whose numbers session used
# a DIFFERENT std -- an older pilot round's participants, still sitting in
# Supabase's append-only `events` table with a perfectly genuine 'finished'
# row. Confirmed as a REAL bug this session, not hypothetical:
# --complete_pairs (before this check existed) pulled in 5 pilot-4
# participants (true_std=15) alongside 46 current-round ones (true_std=10)
# into ONE canonical file, silently mixing two incompatible generative-
# parameter regimes. colors has no equivalent risk -- checked directly:
# every pid's true_p range is identical ([0.1333, 0.8667]) regardless of
# round, so no analogous filter is applied there. This check is intentionally
# --complete_pairs-only, not applied to the explicit --numbers_pids/
# --colors_pids path: that path is how a SPECIFIC pilot round (e.g. pilot 4
# itself) gets rebuilt on purpose, where every pid SHOULD have the old std.
CURRENT_NUMBERS_STD_FIXED = 10.0

# Real Prolific IDs are 24-character lowercase hex strings -- used only by
# --list_candidates (a probing aid), never to silently decide who's
# "real" for an actual build (that's what the explicit pid list is for).
REAL_PID_PATTERN = re.compile(r"^[0-9a-f]{24}$")


def _load_env(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _fetch_all_events(task_backend_dir: Path, task: str) -> list[dict]:
    """Every row (all phases) for one task_backend task, paginated
    (PostgREST caps a single response at 1000 rows -- confirmed directly
    this session, not assumed)."""
    env = {**_load_env(task_backend_dir / ".env"), **_load_env(task_backend_dir / ".env.test")}
    url_base = env.get("VITE_SUPABASE_URL")
    secret_key = env.get("SUPABASE_SECRET_KEY")
    if not url_base or not secret_key:
        raise RuntimeError(
            "Need VITE_SUPABASE_URL (task_backend/.env) and SUPABASE_SECRET_KEY "
            "(task_backend/.env.test, gitignored -- see .env.test.example)."
        )
    cols = ("prolific_pid,phase,trial_index,observation_index,attempt,response,"
            "timed_out,value,true_mean,true_std,true_p,qid,created_at")
    all_rows, offset, page_size = [], 0, 1000
    while True:
        url = (f"{url_base}/rest/v1/events?task=eq.{task}&select={cols}"
               f"&order=id.asc&limit={page_size}&offset={offset}")
        req = urllib.request.Request(url, headers={"apikey": secret_key, "Authorization": f"Bearer {secret_key}"})
        with urllib.request.urlopen(req) as resp:
            page = json.loads(resp.read())
        all_rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return all_rows


def _trial_rows_to_df(rows: list[dict], finished_pids: set[str], task_internal: str) -> pd.DataFrame:
    """Dedup to the highest `attempt` per (prolific_pid, trial_index,
    observation_index) -- same 'latest state wins' convention used
    throughout this project -- keep only the requested finished pids and
    phase='trial' rows, then rename to build_from_df's expected schema."""
    trial_rows = [r for r in rows if r["phase"] == "trial" and r["prolific_pid"] in finished_pids]
    if not trial_rows:
        return pd.DataFrame()
    df = pd.DataFrame(trial_rows)
    df = (df.sort_values("attempt")
          .groupby(["prolific_pid", "trial_index", "observation_index"], as_index=False)
          .last())
    df = df.rename(columns={"trial_index": "trial", "observation_index": "observation"})
    df["task"] = task_internal
    # Explicit numeric casts -- see chat history: values arrive via
    # json.loads(), which can leave true_p/true_mean/value/response as
    # dtype=object (plain Python floats) rather than float64, invisible
    # until a numpy ufunc call on them crashes downstream.
    for col in ("value", "response", "true_p", "true_mean"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cols = ["prolific_pid", "task", "trial", "observation", "value", "response",
            "timed_out", "qid", "true_p", "true_mean"]
    return df[cols]


def list_candidates(pool_root: Path, task: str) -> None:
    """Probing aid: shows every real-Prolific-format pid for `task`, its
    current status (finished/terminated/in progress), and (for numbers)
    the true_std its session used -- so you can build an explicit pid
    list without guessing who's actually done yet."""
    rows = _fetch_all_events(pool_root, task)
    by_pid: dict[str, list[dict]] = {}
    for r in rows:
        if REAL_PID_PATTERN.match(r["prolific_pid"]):
            by_pid.setdefault(r["prolific_pid"], []).append(r)

    print(f"{'PID':<28} {'status':<12} {'trials':<8} {'true_std'}")
    for pid, prows in sorted(by_pid.items(), key=lambda kv: kv[1][0]["created_at"]):
        phases = {r["phase"] for r in prows}
        status = "FINISHED" if "finished" in phases else ("TERMINATED" if "terminated" in phases else "in progress")
        trial_rows = [r for r in prows if r["phase"] == "trial"]
        n_trials = len({r["trial_index"] for r in trial_rows})
        stds = {r["true_std"] for r in trial_rows if r.get("true_std") is not None}
        print(f"{pid:<28} {status:<12} {n_trials:<8} {stds or ''}")


def build_pilot(pool_root: Path, out_prefix: str | None, pids_by_task: dict[str, list[str]],
                apply_filters: bool = True,
                exclusion_method: str = "contingency",
                require_both_tasks: bool = True) -> None:
    """pids_by_task: {'numbers': [...], 'colors': [...]} -- either list can
    be empty/omitted if this pilot didn't touch that task. Requires every
    listed pid to actually have a 'finished' row for that task -- reports
    (does not silently drop) any that don't, since an explicit list means
    you expected them to be ready."""
    frames = []
    for task_backend_task, requested_pids in pids_by_task.items():
        if not requested_pids:
            continue
        task_internal = TASK_INTERNAL[task_backend_task]
        print(f"Fetching {task_backend_task} for {len(requested_pids)} requested pid(s)...")
        rows = _fetch_all_events(pool_root, task_backend_task)

        finished_here = {r["prolific_pid"] for r in rows if r["phase"] == "finished"}
        requested = set(requested_pids)
        not_finished = requested - finished_here
        if not_finished:
            print(f"  WARNING: {len(not_finished)} requested pid(s) do NOT have a "
                  f"'finished' row for {task_backend_task} -- excluded: {sorted(not_finished)}")

        df = _trial_rows_to_df(rows, requested & finished_here, task_internal)
        print(f"  {df['prolific_pid'].nunique() if not df.empty else 0} pid(s) included, "
              f"{len(df)} trial-observation rows after dedup")
        frames.append(df)

    if not frames or all(f.empty for f in frames):
        print("Nothing to build -- no requested pids were both listed and finished.")
        return

    combined = pd.concat(frames, ignore_index=True)
    # out_prefix=None writes the CANONICAL unsuffixed data/soltani_{task}.pkl --
    # the production dataset, which is what the figures read when --datafile is
    # omitted. Pass a prefix only for comparison builds (a pilot, or an
    # alternative exclusion method); see utils.paths.dataset_stem.
    suffix = f"_{out_prefix}" if out_prefix else ""
    build_from_df(combined, out_name_numbers=f"soltani_numbers{suffix}",
                 out_name_colors=f"soltani_colors{suffix}",
                 apply_filters=apply_filters,
                 exclusion_method=exclusion_method,
                 require_both_tasks=require_both_tasks)
    print("\nJOB_COMPLETE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_root", default=str(TASK_BACKEND_DIR))
    p.add_argument("--list_candidates", choices=["numbers", "colors"], default=None,
                   help="Probe current real-pid status for one task, then exit -- doesn't build anything.")
    p.add_argument("--pilot", default=None,
                   # Omit for the canonical production build.
                   help="Output name suffix, e.g. 'pilot4' -> soltani_numbers_pilot4.pkl / soltani_colors_pilot4.pkl")
    p.add_argument("--numbers_pids", default="", help="Comma-separated real prolific_pids for the numbers task")
    p.add_argument("--colors_pids", default="", help="Comma-separated real prolific_pids for the colors task")
    p.add_argument("--exclusion_method", choices=EXCLUSION_METHODS,
                   default=DEFAULT_EXCLUSION_METHOD,
                   help="Which criterion set decides exclusion. 'non_integrator' "
                        "(default): observations before the most recent make no "
                        "RELIABLE contribution to predicting the response, by "
                        "trial-clustered bootstrap -- definition-first, model-free, "
                        "no magnitude threshold; excludes 19/61 numbers and 17/61 "
                        "colors of complete_pairs (~30%%). 'contingency': the three "
                        "Cohen's f2 tests -- model-BASED, excludes 42%%/32%%; kept "
                        "mainly as a diagnostic, since recency_only tests the same "
                        "construct as non_integrator by a different method and "
                        "their agreement is what validates the exclusions. Both are "
                        "always COMPUTED and appear in the report; only the "
                        "decision differs. Two further criteria were tested and "
                        "archived -- see archive/utils/archive_exclusion_criteria.py "
                        "and utils/participant_filters.py.")
    p.add_argument("--per_task_exclusion", action="store_true",
                   help="Exclude per (pid, task) instead of per SUBJECT. Default "
                        "is subject-level: a participant failing in either task is "
                        "dropped from BOTH, so every task keeps the same people. "
                        "Per-task exclusion degrades within-subject cross-task "
                        "panels whenever the criterion is not equally strict in "
                        "both tasks -- see filter_participants' docstring.")
    p.add_argument("--no_filter", action="store_true",
                   help="Skip utils.participant_filters entirely -- keep every "
                        "participant. For diagnosing how much the exclusion "
                        "criteria change a result. Integer pids will NOT match a "
                        "filtered build of the same people.")
    p.add_argument("--complete_pairs", action="store_true",
                   help="Auto-select the pids with a 'finished' row in BOTH tasks, "
                        "instead of passing --numbers_pids/--colors_pids by hand.")
    args = p.parse_args()

    pool_root = Path(args.pool_root)

    if args.list_candidates:
        list_candidates(pool_root, args.list_candidates)
        return

    if args.complete_pairs:
        # Participants with a 'finished' row in BOTH tasks. Derived live rather
        # than from a stored list, because no cohort pid lists are recorded in
        # the repo -- so this is the only reproducible way to re-select the same
        # people. Both tasks get the SAME pid set by construction.
        finished = {}
        for task in ("numbers", "colors"):
            rows = _fetch_all_events(pool_root, task)
            task_finished = {r["prolific_pid"] for r in rows
                             if r["phase"] == "finished"
                             and REAL_PID_PATTERN.match(r["prolific_pid"])}

            if task == "numbers":
                # See CURRENT_NUMBERS_STD_FIXED's own module-level comment
                # for why this exists -- excludes anyone whose numbers
                # session used a DIFFERENT std (a stale pilot round still
                # sitting in Supabase's append-only events table).
                std_by_pid: dict[str, set] = {}
                for r in rows:
                    if r["phase"] == "trial" and r.get("true_std") is not None:
                        std_by_pid.setdefault(r["prolific_pid"], set()).add(r["true_std"])
                stale = {pid for pid in task_finished
                        if std_by_pid.get(pid, set()) != {CURRENT_NUMBERS_STD_FIXED}}
                if stale:
                    print(f"  numbers: excluding {len(stale)} pid(s) with a stale/"
                          f"mismatched true_std (not {CURRENT_NUMBERS_STD_FIXED}, "
                          f"e.g. an older pilot round): {sorted(stale)}")
                task_finished -= stale

            finished[task] = task_finished
            print(f"  {task}: {len(finished[task])} finished real pids")
        both = sorted(finished["numbers"] & finished["colors"])
        print(f"  -> {len(both)} pids finished BOTH tasks")
        pids_by_task = {"numbers": both, "colors": both}
    else:
        pids_by_task = {
            "numbers": [p.strip() for p in args.numbers_pids.split(",") if p.strip()],
            "colors": [p.strip() for p in args.colors_pids.split(",") if p.strip()],
        }
    build_pilot(pool_root, args.pilot, pids_by_task,
                apply_filters=not args.no_filter,
                exclusion_method=args.exclusion_method,
                require_both_tasks=not args.per_task_exclusion)


if __name__ == "__main__":
    main()
