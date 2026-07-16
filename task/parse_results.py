"""
parse_results.py
================
Converts raw JATOS JSON exports into a single tidy DataFrame saved as .pkl.

Usage
-----
    python task/parse_results.py --input_dir <path_to_jatos_export> [--output <path.pkl>]

JATOS export structure
----------------------
Download results from MindProbe as "Export results in JATOS plain text format".
This produces a directory (or zip) where each file is one participant's JSON array
of jsPsych trial objects. Both continuous and binary exports can be pointed at the
same --input_dir, or separate dirs can be merged by running twice and concatenating.

Design note: only genuinely participant-generated fields are kept here
(prolific_pid, task, trial, observation, value, response, timed_out, rt,
time_elapsed). `value` isn't in the raw export either (build-trial-timeline.js
only exports trial/observation for observation rows), but it's simple enough
to look up from task/sequences/{task}_sequences.json (trial order is identical
for every participant, nothing is shuffled) that it's worth including directly
rather than requiring a second join step downstream.

Everything else about a given (task, trial) — true_mean, true_p, true_std,
qid, prefix_length, iti_ms, iti_condition — is intentionally left out of this
pickle. All of it is still fully recoverable later the same way (a join
against the sequence files on (task, trial)) if a given analysis needs it;
it just doesn't need to live in every copy of this file.

Output columns (observation rows only)
---------------------------------------
    prolific_pid   : str    Prolific participant ID
    task           : str    'continuous' or 'binary'
    trial          : int    0-indexed trial number
    observation    : int    0-indexed observation within trial
    value          : int    Stimulus value (0..100 continuous; -1/1 binary) —
                            looked up from the sequence file, not the raw export
    response       : float  Participant estimate (NaN if timed out or no response)
    timed_out      : bool   True if response deadline elapsed
    rt             : float  Response time in ms (NaN if timed out)
    time_elapsed   : int    ms since experiment start

Duplicate rows per (task, trial, observation) are INTENTIONAL, not a bug —
read this before touching row-uniqueness assumptions anywhere downstream.
When an observation times out, build-trial-timeline.js replays the SAME
observation index (up to MAX_TIMEOUTS_PER_TRIAL times), and each attempt is
its own jsPsych trial with its own row here — so a slot that timed out once
or twice before eventually succeeding produces 1-2 extra timed_out=True rows
under the same (task, trial, observation) key as the eventual real response
(or, if the budget is exhausted, up to MAX_TIMEOUTS_PER_TRIAL timed_out=True
rows and no successful one at all — see dev-results/pilot8fail.txt for a
real example, confirmed via a real MindProbe run). This is deliberate: EVERY
attempt is kept here for participant-level data-quality auditing (e.g. flagging
someone who timed out repeatedly), not filtered at parse time. Any analysis
treating (pid, task, trial, observation) as a unique key MUST filter first —
typically to timed_out == False, or to the last attempt per slot if timeout
counts themselves matter — before doing so; this script deliberately does not
pick one, since which filtering is correct depends on the analysis. This also
means the printed "Timed out" percentage below counts ATTEMPTS, not unique
slots — a slot that timed out twice then succeeded contributes 2 to the
timed_out count and 1 to the total denominator's success side, not 1-in-3.
"""

import json
import argparse
import pathlib
import pandas as pd
import numpy as np

TASK_DIR      = pathlib.Path(__file__).resolve().parent
SEQUENCES_DIR = TASK_DIR / 'sequences'


def load_values_lookup(task: str) -> pd.DataFrame:
    """Load task/sequences/{task}_sequences.json into a per-trial lookup of
    just the `values` list, for the per-observation value lookup below."""
    seq_path = SEQUENCES_DIR / f'{task}_sequences.json'
    with open(seq_path) as f:
        seqs = json.load(f)
    df = pd.DataFrame(seqs)
    df['task'] = task
    return df[['task', 'trial', 'values']]


def parse_participant_file(fpath: pathlib.Path) -> pd.DataFrame:
    """Parse one participant's JSON file into a DataFrame of observation rows.
    Only genuinely participant-generated fields are extracted here — see
    module docstring for what's looked up afterward instead."""
    try:
        raw = fpath.read_text(encoding='utf-8').strip()
        # JATOS sometimes wraps multiple result sets with a separator line
        # Try parsing as a single JSON array first, then fall back
        try:
            trials = json.loads(raw)
        except json.JSONDecodeError:
            # JATOS plain text format: multiple JSON arrays separated by newlines
            records = []
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith('[') or line.startswith('{'):
                    try:
                        block = json.loads(line)
                        if isinstance(block, list):
                            records.extend(block)
                        else:
                            records.append(block)
                    except json.JSONDecodeError:
                        continue
            trials = records
    except Exception as e:
        print(f"  Warning: could not parse {fpath.name}: {e}")
        return pd.DataFrame()

    if not trials:
        return pd.DataFrame()

    rows       = []
    for t in trials:
        if not isinstance(t, dict):
            continue
        if t.get('screen') != 'observation':
            continue

        rows.append({
            'prolific_pid':  t.get('prolific_pid', 'unknown'),
            'task':          t.get('task', 'unknown'),
            'trial':         t.get('trial', np.nan),
            'observation':   t.get('observation', np.nan),
            'response':      t.get('response', np.nan) if t.get('response') is not None else np.nan,
            'timed_out':     bool(t.get('timed_out', False)),
            'rt':            t.get('rt', np.nan) if t.get('rt') is not None else np.nan,
            'time_elapsed':  t.get('time_elapsed', np.nan),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Parse JATOS JSON exports to .pkl')
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing JATOS result JSON files')
    parser.add_argument('--files', nargs='+', default=None,
                        help='Specific filenames within input_dir to parse (default: all)')
    parser.add_argument('--output', default='data/task_results.pkl',
                        help='Output .pkl path (default: data/task_results.pkl)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output    = pathlib.Path(args.output)

    # Find all JSON files recursively
    if args.files:
        json_files = sorted(input_dir / f for f in args.files
                            if (input_dir / f).exists())
    else:
        json_files = sorted(input_dir.rglob('*.json'))
        if not json_files:
            # Also try plain text files JATOS sometimes exports
            json_files = sorted(input_dir.rglob('*.txt'))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} file(s) in {input_dir}")

    dfs = []
    for fpath in json_files:
        if args.verbose:
            print(f"  Parsing {fpath.name}...")
        df = parse_participant_file(fpath)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No observation rows found. Check that files are valid JATOS exports.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Cast participant-generated columns
    for col in ['trial', 'observation']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce').astype('Int64')
    for col in ['response', 'rt']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')

    # ── Look up `value` from the saved sequence files — see module docstring.
    #    Only look up per task actually present in the data.
    tasks_present = [t for t in combined['task'].unique() if t in ('continuous', 'binary')]
    lookups = {task: load_values_lookup(task) for task in tasks_present}

    parts = []
    for task in tasks_present:
        sub    = combined[combined['task'] == task].copy()
        sub    = sub.merge(lookups[task], on=['task', 'trial'], how='left')

        # Per-observation value lookup from the trial's `values` list —
        # vectorised column-merge can't index into a list column, so this
        # needs an explicit row-wise lookup.
        def _lookup_value(row):
            vals = row['values']
            obs  = row['observation']
            if isinstance(vals, list) and pd.notna(obs) and int(obs) < len(vals):
                return vals[int(obs)]
            return np.nan
        sub['value'] = sub.apply(_lookup_value, axis=1)
        sub = sub.drop(columns=['values'])

        parts.append(sub)

    combined = pd.concat(parts, ignore_index=True)
    combined['value'] = pd.to_numeric(combined['value'], errors='coerce')

    # Column order matches the documented schema above
    combined = combined[['prolific_pid', 'task', 'trial', 'observation', 'value',
                          'response', 'timed_out', 'rt', 'time_elapsed']]

    # Sort
    combined = combined.sort_values(
        ['prolific_pid', 'task', 'trial', 'observation']
    ).reset_index(drop=True)

    # Summary
    n_pids   = combined['prolific_pid'].nunique()
    n_tasks  = combined['task'].nunique()
    n_rows   = len(combined)
    timeout_pct = combined['timed_out'].mean() * 100

    print(f"\nParsed {n_rows} observation rows")
    print(f"  Participants : {n_pids}")
    print(f"  Tasks        : {combined['task'].unique().tolist()}")
    print(f"  Timed out    : {timeout_pct:.1f}%")
    print(f"\nSample:\n{combined.head(6).to_string(index=False)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(output)
    print(f"\nSaved: {output}  ({n_rows} rows × {len(combined.columns)} cols)")


if __name__ == '__main__':
    main()
