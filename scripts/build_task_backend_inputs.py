#!/usr/bin/env python3
"""scripts/build_task_backend_inputs.py
=========================================
Downloads real, finished task_backend participant data directly from
Supabase's `events` table for an EXPLICIT list of prolific_pids, reformats
it to match exactly what scripts/build_model_inputs.py's build_from_df()
expects, then calls that SAME shared filter+rescale+anonymize+save
pipeline carrabin/yoo/the old JATOS-pilot data all already go through --
see build_model_inputs.py's own build_from_df docstring for why this is
reused rather than reimplemented.

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
(build_from_df's own pid mapping already handles that correctly, is
unchanged, and is computed fresh -- and independently -- for each call
this script makes).

Usage:
    # Probe which real, finished pids exist right now (for building a list):
    python scripts/build_task_backend_inputs.py --list_candidates numbers

    # Build one pilot's files from an explicit pid list:
    python scripts/build_task_backend_inputs.py --pilot pilot4 \\
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


def build_pilot(pool_root: Path, out_prefix: str, pids_by_task: dict[str, list[str]],
                apply_filters: bool = True,
                exclusion_method: str = "contingency",
                max_error_sd: float | None = None,
                min_skill: float | None = None,
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
    build_from_df(combined, out_name_numbers=f"soltani_numbers_{out_prefix}",
                 out_name_colors=f"soltani_colors_{out_prefix}",
                 apply_filters=apply_filters,
                 exclusion_method=exclusion_method,
                 max_error_sd=max_error_sd,
                 min_skill=min_skill,
                 require_both_tasks=require_both_tasks)
    print("\nJOB_COMPLETE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_root", default=str(TASK_BACKEND_DIR))
    p.add_argument("--list_candidates", choices=["numbers", "colors"], default=None,
                   help="Probe current real-pid status for one task, then exit -- doesn't build anything.")
    p.add_argument("--pilot", default=None,
                   help="Output name suffix, e.g. 'pilot4' -> soltani_numbers_pilot4.pkl / soltani_colors_pilot4.pkl")
    p.add_argument("--numbers_pids", default="", help="Comma-separated real prolific_pids for the numbers task")
    p.add_argument("--colors_pids", default="", help="Comma-separated real prolific_pids for the colors task")
    p.add_argument("--exclusion_method", choices=EXCLUSION_METHODS,
                   default=DEFAULT_EXCLUSION_METHOD,
                   help="Which criterion set decides exclusion. 'contingency' "
                        "(default): the three Cohen's f2 tests -- model-BASED, and "
                        "excludes 42%%/32%% of complete_pairs numbers/colors. "
                        "'integration': no evidence of integrating beyond the "
                        "most recent observation (model-free, threshold read off an "
                        "empirical void). 'performance': carrabin's own rule, a gross outlier on "
                        "mean absolute error vs the true generative parameter -- "
                        "model-FREE, and closer to the 16%%/17%% rates carrabin and "
                        "yoo report. Both are always computed; only the decision "
                        "differs. See utils/participant_filters.py.")
    p.add_argument("--max_error_sd", type=float, default=None,
                   help="Threshold for --exclusion_method performance: SDs above "
                        "the retained group's mean absolute error. Default 2.0. "
                        "carrabin's literal >6 SD excludes ZERO participants here, "
                        "because our error distribution is continuous where theirs "
                        "had a 6-SD gap.")
    p.add_argument("--min_skill", type=float, default=None,
                   help="Threshold for --exclusion_method integration: skill below "
                        "this is excluded. Default 0.10, meaning 'moved at least "
                        "slightly toward the true mean, relative to copying the "
                        "latest observation'. Read off a 0.29-wide empirical void "
                        "in the skill distribution -- any value in (0.041, 0.334) "
                        "gives the identical partition -- rather than tuned.")
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

    if not args.pilot:
        print("Need --pilot <name> (or --list_candidates <task> to probe first).")
        return

    if args.complete_pairs:
        # Participants with a 'finished' row in BOTH tasks. Derived live rather
        # than from a stored list, because no cohort pid lists are recorded in
        # the repo -- so this is the only reproducible way to re-select the same
        # people. Both tasks get the SAME pid set by construction.
        finished = {}
        for task in ("numbers", "colors"):
            rows = _fetch_all_events(pool_root, task)
            finished[task] = {r["prolific_pid"] for r in rows
                              if r["phase"] == "finished"
                              and REAL_PID_PATTERN.match(r["prolific_pid"])}
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
                max_error_sd=args.max_error_sd,
                min_skill=args.min_skill,
                require_both_tasks=not args.per_task_exclusion)


if __name__ == "__main__":
    main()
