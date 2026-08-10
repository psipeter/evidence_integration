#!/usr/bin/env python3
"""scripts/build_task_backend_inputs.py
=========================================
Downloads real, finished task_backend participant data directly from
Supabase's `events` table, reformats it to match exactly what
scripts/build_model_inputs.py's build_from_df() expects (same columns,
same native scale), then calls that SAME shared filter+rescale+anonymize+
save pipeline carrabin/yoo/the old JATOS-pilot data all already go
through -- see build_model_inputs.py's own build_from_df docstring for
why this is reused rather than reimplemented.

Writes to the CANONICAL data/task_continuous.pkl / data/task_binary.pkl
paths -- NOT a separate task_backend-specific filename -- because
fitting.submit's job resolution (and MODEL_PARAMS' own keys) are hardcoded
to exactly those two filenames; using a different name would require
deeper changes throughout the fitting pipeline for no real benefit. This
DELIBERATELY OVERWRITES whatever was there before (the old, much smaller
JATOS-era pilot data -- a different population under a retired pipeline,
not meant to be merged with this). Back up first if that old data still
matters to you (not tracked in git -- data/ is gitignored, see chat
history for how this session verified that before overwriting).

ANONYMIZATION: this script never itself maps prolific_pid -> int pid --
build_from_df already does that (sorted prolific_pid -> 1, 2, 3, ...) and
DROPS the real prolific_pid entirely from what gets saved to the pkl. The
only anonymization work THIS script does is filtering down to real
Prolific-format IDs in the first place (see REAL_PID_PATTERN below) --
excluding every test/dev/student/PI id we've used this session, so none
of that non-Prolific traffic (which was never meant to be real data
anyway) ends up in the analysis pipeline at all.

COMPLETENESS: only (prolific_pid, task) pairs with an actual 'finished'
phase row are included -- an abandoned or in-progress session has no
guaranteed trial-count to speak of and shouldn't silently become partial
rows in a pkl carrabin/yoo/the fitting pipeline all assume is complete
per participant.

Usage:
    python scripts/build_task_backend_inputs.py
    python scripts/build_task_backend_inputs.py --pool_root task_backend
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

TASK_BACKEND_DIR = Path(__file__).resolve().parents[1] / "task_backend"
TASK_INTERNAL = {"numbers": "continuous", "colors": "binary"}

# Real Prolific IDs are 24-character lowercase hex strings. Every test/dev/
# student/PI id used against this backend this session (f00xxxx student
# ids, f007qzn -- the PI's own test, dev_<timestamp>, test*/testabc/
# testpid, dethiers*, check*/debug*/verify_*) fails this pattern -- a
# POSITIVE filter rather than a maintained exclusion list, so a NEW kind
# of test id invented later still gets excluded automatically rather than
# needing this list updated every time.
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


def _finished_real_pids(rows: list[dict]) -> set[str]:
    """Real-Prolific-format pids with an actual 'finished' phase row."""
    return {
        r["prolific_pid"] for r in rows
        if r["phase"] == "finished" and REAL_PID_PATTERN.match(r["prolific_pid"])
    }


def _trial_rows_to_df(rows: list[dict], finished_pids: set[str], task_internal: str) -> pd.DataFrame:
    """Dedup to the highest `attempt` per (prolific_pid, trial_index,
    observation_index) -- same 'latest state wins' convention used
    throughout this project -- keep only finished real participants and
    phase='trial' rows, then rename to build_from_df's expected schema.
    Native scale preserved exactly as task_backend stores it: numbers
    value/response already [0,100]; colors value already {-1,+1},
    response already [0,100] -- build_from_df does its own [0,100]->[-1,1]
    rescale AFTER filter_participants runs, so nothing here should
    pre-rescale anything (see that function's own docstring)."""
    trial_rows = [r for r in rows if r["phase"] == "trial" and r["prolific_pid"] in finished_pids]
    if not trial_rows:
        return pd.DataFrame()
    df = pd.DataFrame(trial_rows)
    df = (df.sort_values("attempt")
          .groupby(["prolific_pid", "trial_index", "observation_index"], as_index=False)
          .last())
    df = df.rename(columns={"trial_index": "trial", "observation_index": "observation"})
    df["task"] = task_internal
    # Explicit numeric casts -- values arrived via json.loads() (real
    # Supabase JSON), which deserializes numbers to plain Python
    # int/float, not numpy dtypes. Without this, true_p/true_mean/value/
    # response can silently end up as dtype=object (holding Python floats)
    # rather than float64 -- harmless-looking until something downstream
    # calls a numpy ufunc on them directly (np.sqrt(series) fails on
    # object-dtype with "float has no callable sqrt method", since numpy
    # tries to call .sqrt() as a METHOD on each element rather than
    # applying the ufunc elementwise) -- caught by actually running
    # figure_soltani_performance.py against this data, not by inspection.
    for col in ("value", "response", "true_p", "true_mean"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cols = ["prolific_pid", "task", "trial", "observation", "value", "response",
            "timed_out", "qid", "true_p", "true_mean"]
    return df[cols]


def download_and_reformat(pool_root) -> pd.DataFrame:
    """Full pipeline: download both tasks, filter to finished real
    participants, dedup, reformat -- returns ONE combined DataFrame (both
    tasks) ready to pass to build_model_inputs.py's build_from_df."""
    frames = []
    for task_backend_task, task_internal in TASK_INTERNAL.items():
        print(f"Fetching {task_backend_task} from Supabase...")
        rows = _fetch_all_events(pool_root, task_backend_task)
        finished = _finished_real_pids(rows)
        print(f"  {len(finished)} finished real-Prolific participant(s): {sorted(finished)}")
        df = _trial_rows_to_df(rows, finished, task_internal)
        print(f"  {len(df)} trial-observation rows after dedup")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool_root", default=str(TASK_BACKEND_DIR),
                   help="Directory containing task_backend's .env/.env.test (default: task_backend/)")
    args = p.parse_args()

    combined = download_and_reformat(Path(args.pool_root))
    if combined.empty:
        print("No finished real-Prolific data found -- nothing to build.")
        return

    print(f"\nCombined: {len(combined)} rows across "
          f"{combined.groupby('task')['prolific_pid'].nunique().to_dict()}")
    build_from_df(combined)
    print("\nJOB_COMPLETE")


if __name__ == "__main__":
    main()
