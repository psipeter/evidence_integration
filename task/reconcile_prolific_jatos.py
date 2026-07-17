"""
reconcile_prolific_jatos.py
============================
Cross-references Prolific's exported submission list against JATOS's own
per-participant result files, to answer: "for every Prolific submission,
what does JATOS actually show for that person -- and for every JATOS
participant, are they even in this Prolific export?"

This is the tool for the "leaked through" workflow from chat history: a
participant shows as started/awaiting-review/completed on Prolific with no
(or only partial) matching data in JATOS. It does NOT fix that gap (see
finish-session.js / timeline-builder.js for the actual save-then-end
gating and incremental per-trial appends) -- it's the manual-review report
that sits downstream of those fixes, for whatever still slips through
(pilot participants, runs from before those fixes were deployed, genuine
client-side failures no code can catch).

Usage
-----
    python task/reconcile_prolific_jatos.py \\
        --jatos_dir <path_to_jatos_export> \\
        --prolific_csv <path_to_prolific_export.csv> \\
        --output reconciliation_report.csv

Inputs
------
JATOS side: same export format as parse_results.py's --jatos_dir --
"Export results in JATOS plain text format" from MindProbe, a directory
where each file is one participant's JSON array of jsPsych trial objects
(or JATOS's newline-separated-JSON-arrays plain-text variant). UNLIKE
parse_results.py, this reads EVERY row (not just screen == 'observation'),
since the whole point here is the 'started'/'finished'/'terminated'
progress markers and the last screen reached -- see finish-session.js and
timeline-builder.js's on_trial_finish/started-marker for where those come
from. Files/participants with no `progress` field at all (i.e. exports
predating that change) still work -- last_progress will just show as the
last raw `screen` value instead, with a note.

Prolific side: the CSV from a study's "Export results" button on Prolific.
Column names have changed across Prolific's own UI versions, so this
DETECTS the relevant columns by fuzzy substring match (participant id,
status, started/completed timestamps, completion code) rather than
hardcoding one exact header row -- check the printed "Detected Prolific
columns" lines if detection picks the wrong one, and override with
--prolific-id-col etc.

Output
------
One row per participant seen in EITHER source (outer join on participant
ID), with a plain-language `recommendation` column. Written to --output as
CSV, and a summary breakdown printed to the console.
"""

import argparse
import json
import pathlib
import sys

import pandas as pd


# ── JATOS side ────────────────────────────────────────────────────────────

def parse_jatos_participant_file(fpath: pathlib.Path) -> list[dict]:
    """
    Parse one JATOS export file into a flat list of ALL trial-row dicts
    (not filtered by screen, unlike parse_results.py's parse_participant_file
    -- reconciliation needs the 'started'/'finished'/'terminated' markers
    and every screen in between, not just observation rows). Same
    single-JSON-array-with-newline-separated-fallback parsing as
    parse_results.py, kept in sync deliberately -- if that file's parsing
    logic changes to handle some new JATOS export quirk, mirror it here too.
    """
    try:
        raw = fpath.read_text(encoding='utf-8').strip()
        try:
            trials = json.loads(raw)
            if not isinstance(trials, list):
                trials = [trials]
        except json.JSONDecodeError:
            trials = []
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith('[') or line.startswith('{'):
                    try:
                        block = json.loads(line)
                        trials.extend(block if isinstance(block, list) else [block])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"  Warning: could not parse {fpath.name}: {e}")
        return []

    return [t for t in trials if isinstance(t, dict)]


def summarize_jatos(jatos_dir: pathlib.Path) -> pd.DataFrame:
    """
    One row per prolific_pid seen anywhere in the JATOS export, summarizing
    what's known about that participant's session: whether a 'started'
    marker exists, the last progress/screen reached (by time_elapsed order
    within whichever file(s) mention them -- a participant CAN legitimately
    span multiple files/study-results, e.g. the General-Single-link
    reload-then-restart pattern from chat history, so rows are pooled
    across all files before picking the last one, not assumed to live in
    a single file).
    """
    json_files = sorted(jatos_dir.rglob('*.json'))
    if not json_files:
        json_files = sorted(jatos_dir.rglob('*.txt'))
    if not json_files:
        sys.exit(f"No JATOS export files (*.json or *.txt) found under {jatos_dir}")

    all_rows = []
    for fpath in json_files:
        for t in parse_jatos_participant_file(fpath):
            all_rows.append({
                'prolific_pid': t.get('prolific_pid', 'unknown'),
                'task':         t.get('task'),
                'screen':       t.get('screen'),
                # Only present on rows appended after this session's
                # timeline-builder.js/finish-session.js changes -- absent
                # (None) for older exports, handled below.
                'progress':     t.get('progress'),
                'time_elapsed': t.get('time_elapsed'),
                'source_file':  fpath.name,
            })

    if not all_rows:
        sys.exit(f"Found {len(json_files)} file(s) under {jatos_dir} but none contained parseable trial rows.")

    df = pd.DataFrame(all_rows)
    df['time_elapsed'] = pd.to_numeric(df['time_elapsed'], errors='coerce')
    # Sort so "last" (via .iloc[-1] below) means chronologically last within
    # whatever file(s) a participant appears in. Rows missing time_elapsed
    # sort first (NaN), which only matters for the tiny number of legacy
    # rows predating time_elapsed being recorded on every screen.
    df = df.sort_values(['prolific_pid', 'time_elapsed'], na_position='first')

    summary_rows = []
    for pid, g in df.groupby('prolific_pid'):
        last = g.iloc[-1]
        has_progress_field = g['progress'].notna().any()
        summary_rows.append({
            'prolific_pid':     pid,
            'jatos_task':       ';'.join(sorted(g['task'].dropna().unique())),
            'jatos_n_rows':     len(g),
            'jatos_n_files':    g['source_file'].nunique(),
            'jatos_has_started_marker': bool((g['progress'] == 'started').any()),
            'jatos_last_progress': (
                last['progress'] if has_progress_field and pd.notna(last['progress'])
                else f"(no progress field -- last screen: {last['screen']})"
            ),
            'jatos_last_screen': last['screen'],
        })

    return pd.DataFrame(summary_rows)


# ── Prolific side ─────────────────────────────────────────────────────────

def find_col(columns: list[str], *keywords: str) -> str | None:
    """First column whose lowercased name contains every keyword given."""
    for col in columns:
        lc = col.lower()
        if all(k in lc for k in keywords):
            return col
    return None


def load_prolific_csv(csv_path: pathlib.Path, overrides: dict) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(csv_path)
    cols = list(raw.columns)

    detected = {
        'id':        overrides.get('id')        or find_col(cols, 'participant', 'id') or find_col(cols, 'participant'),
        'status':    overrides.get('status')     or find_col(cols, 'status'),
        'started':   overrides.get('started')    or find_col(cols, 'started'),
        'completed': overrides.get('completed')  or find_col(cols, 'completed'),
        'code':      overrides.get('code')       or find_col(cols, 'completion', 'code'),
    }

    missing_required = [k for k in ('id', 'status') if detected[k] is None]
    if missing_required:
        sys.exit(
            f"Could not detect required Prolific column(s): {missing_required}.\n"
            f"Columns actually present in {csv_path.name}:\n  " + '\n  '.join(cols) +
            "\nRe-run with --prolific-id-col / --prolific-status-col to specify them explicitly."
        )

    out = pd.DataFrame({
        'prolific_pid':     raw[detected['id']].astype(str).str.strip(),
        'prolific_status':  raw[detected['status']],
    })
    if detected['started']:
        out['prolific_started_at'] = raw[detected['started']]
    if detected['completed']:
        out['prolific_completed_at'] = raw[detected['completed']]
    if detected['code']:
        out['prolific_completion_code'] = raw[detected['code']]

    return out, detected


# ── Recommendation logic ────────────────────────────────────────────────

def recommend(row: pd.Series) -> str:
    in_prolific = pd.notna(row.get('prolific_status'))
    in_jatos    = pd.notna(row.get('jatos_last_progress'))
    pid         = str(row.get('prolific_pid', ''))

    if in_jatos and not in_prolific:
        if pid.startswith('pilot_'):
            return 'pilot/test run -- not a real Prolific participant, ignore'
        return ('in JATOS but NOT in this Prolific export -- check you exported '
                'the right study/date range before assuming this is a real gap')

    if in_prolific and not in_jatos:
        return ('NO JATOS DATA AT ALL -- Prolific shows a submission but nothing '
                'in JATOS matches this participant ID. Verify manually before '
                'approving; see chat history\'s "leaked through" investigation. '
                'Prolific supports rejecting for "gave no study data" if caught '
                'before the 21-day auto-approval window.')

    progress = row.get('jatos_last_progress')
    if progress == 'finished':
        return 'OK -- normal completion recorded in both Prolific and JATOS'
    if progress == 'terminated':
        return ('TERMINATED (timeout budget exhausted) -- this is the early-exit '
                'path, not a crash. Confirm the partial/early-exit completion '
                'code was actually issued/entered on Prolific.')
    if progress == 'started':
        return ('STUCK immediately after starting -- no further than the initial '
                '"started" marker. Consider rejecting (no study data) or a small '
                'partial payment per your own judgment.')
    return f"STUCK mid-session (last progress: '{progress}') -- manual review; consider partial payment / request return."


def build_report(jatos_dir: pathlib.Path, prolific_csv: pathlib.Path, overrides: dict) -> pd.DataFrame:
    jatos_summary = summarize_jatos(jatos_dir)
    prolific_df, detected_cols = load_prolific_csv(prolific_csv, overrides)

    print("Detected Prolific columns:")
    for k, v in detected_cols.items():
        print(f"    {k:10s} -> {v!r}")

    merged = pd.merge(prolific_df, jatos_summary, on='prolific_pid', how='outer')
    merged['recommendation'] = merged.apply(recommend, axis=1)
    merged = merged.sort_values('prolific_pid').reset_index(drop=True)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Reconcile a Prolific submission export against JATOS result files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--jatos_dir', required=True,
                        help='Directory of JATOS plain-text export files (one per participant/run)')
    parser.add_argument('--prolific_csv', required=True,
                        help='CSV exported from Prolific\'s study "Export results" button')
    parser.add_argument('--output', default='reconciliation_report.csv',
                        help='Output CSV path')
    parser.add_argument('--prolific-id-col', default=None, help='Override auto-detected participant-ID column')
    parser.add_argument('--prolific-status-col', default=None, help='Override auto-detected status column')
    parser.add_argument('--prolific-started-col', default=None, help='Override auto-detected "started at" column')
    parser.add_argument('--prolific-completed-col', default=None, help='Override auto-detected "completed at" column')
    parser.add_argument('--prolific-code-col', default=None, help='Override auto-detected completion-code column')
    args = parser.parse_args()

    overrides = {
        'id':        args.prolific_id_col,
        'status':    args.prolific_status_col,
        'started':   args.prolific_started_col,
        'completed': args.prolific_completed_col,
        'code':      args.prolific_code_col,
    }

    report = build_report(pathlib.Path(args.jatos_dir), pathlib.Path(args.prolific_csv), overrides)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    n_total = len(report)
    n_ok = (report['recommendation'].str.startswith('OK')).sum()
    n_no_data = report['recommendation'].str.contains('NO JATOS DATA').sum()
    n_stuck = report['recommendation'].str.contains('STUCK').sum()
    n_terminated = report['recommendation'].str.contains('TERMINATED').sum()
    n_pilot = report['recommendation'].str.contains('pilot/test').sum()
    n_export_mismatch = report['recommendation'].str.contains('NOT in this Prolific export').sum()

    print(f"\nWrote {output_path}  ({n_total} participant rows)")
    print(f"  OK (normal completion)          : {n_ok}")
    print(f"  TERMINATED (timeout early exit)  : {n_terminated}")
    print(f"  STUCK (needs manual review)      : {n_stuck}")
    print(f"  NO JATOS DATA AT ALL             : {n_no_data}")
    print(f"  In JATOS, not in this export      : {n_export_mismatch}")
    print(f"  Pilot/test runs (ignored)        : {n_pilot}")
    if n_no_data or n_stuck:
        print(f"\n  -> {n_no_data + n_stuck} row(s) need manual attention. See --output for detail.")


if __name__ == '__main__':
    main()
