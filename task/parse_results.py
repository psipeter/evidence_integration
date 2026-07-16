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

Design note: value/true_mean/true_std/true_p/qid/pool_index are extracted
DIRECTLY from each row here -- no lookup/join against a saved sequence file.
This is a deliberate change (see chat history) from an earlier design that
looked `value` up from task/sequences/{task}_sequences.json via a (task,
trial) join, which relied on every participant sharing exactly one file (so
(task, trial) alone determined `value`). That stopped being true once
per-participant pool assignment was introduced -- (task, trial) is no
longer enough; you'd also need to know which pool member that participant
got. Rather than doing a THREE-key join against the right pool member's
file, build-trial-timeline.js now records these fields directly on every
observation row, so every participant's raw export is fully self-contained:
no lookup, no join, no dependency on the pool files still existing/matching
later. `pool_index` is kept in the output specifically so which pool member
a given participant saw is always traceable, even without the pool files.

Output columns (observation rows only)
---------------------------------------
    prolific_pid   : str    Prolific participant ID
    task           : str    'continuous' or 'binary'
    pool_index     : int    Which of the 200 pool members this participant
                            was assigned (see timeline-builder.js's
                            poolIndexForParticipant) -- same index for both
                            tasks for a given participant, by design
    trial          : int    0-indexed trial number (within that pool member)
    observation    : int    0-indexed observation within trial
    qid            : int    Which of that pool member's distinct prefixes
                            this trial used
    value          : int    Stimulus value (0..100 continuous; -1/1 binary)
    true_mean      : float  Continuous only; NaN for binary rows
    true_std       : float  Continuous only; NaN for binary rows
    true_p         : float  Binary only; NaN for continuous rows
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


def parse_participant_file(fpath: pathlib.Path) -> pd.DataFrame:
    """Parse one participant's JSON file into a DataFrame of observation rows.
    Every field here is read directly off the row -- see module docstring
    for why no lookup/join against a saved sequence file happens anymore."""
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

    rows = []
    for t in trials:
        if not isinstance(t, dict):
            continue
        if t.get('screen') != 'observation':
            continue

        rows.append({
            'prolific_pid':  t.get('prolific_pid', 'unknown'),
            'task':          t.get('task', 'unknown'),
            'pool_index':    t.get('pool_index', np.nan),
            'trial':         t.get('trial', np.nan),
            'observation':   t.get('observation', np.nan),
            'qid':           t.get('qid', np.nan),
            'value':         t.get('value', np.nan),
            'true_mean':     t.get('true_mean', np.nan) if t.get('true_mean') is not None else np.nan,
            'true_std':      t.get('true_std', np.nan) if t.get('true_std') is not None else np.nan,
            'true_p':        t.get('true_p', np.nan) if t.get('true_p') is not None else np.nan,
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

    # Cast columns
    for col in ['trial', 'observation', 'qid', 'pool_index']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce').astype('Int64')
    for col in ['response', 'rt', 'value', 'true_mean', 'true_std', 'true_p']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')

    # Column order matches the documented schema above
    combined = combined[['prolific_pid', 'task', 'pool_index', 'trial', 'observation',
                          'qid', 'value', 'true_mean', 'true_std', 'true_p',
                          'response', 'timed_out', 'rt', 'time_elapsed']]

    # Sort
    combined = combined.sort_values(
        ['prolific_pid', 'task', 'trial', 'observation']
    ).reset_index(drop=True)

    # Summary
    n_pids   = combined['prolific_pid'].nunique()
    n_rows   = len(combined)
    timeout_pct = combined['timed_out'].mean() * 100
    n_pool_missing = combined['pool_index'].isna().sum()

    print(f"\nParsed {n_rows} observation rows")
    print(f"  Participants : {n_pids}")
    print(f"  Tasks        : {combined['task'].unique().tolist()}")
    print(f"  Timed out    : {timeout_pct:.1f}%")
    if n_pool_missing:
        print(f"  WARNING: {n_pool_missing} row(s) missing pool_index -- likely an export "
              f"from before pool-based assignment was introduced; check the source file(s).")
    print(f"\nSample:\n{combined.head(6).to_string(index=False)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(output)
    print(f"\nSaved: {output}  ({n_rows} rows × {len(combined.columns)} cols)")


if __name__ == '__main__':
    main()
