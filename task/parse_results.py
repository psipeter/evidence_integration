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

Output columns (observation rows only)
---------------------------------------
    prolific_pid   : str    Prolific participant ID
    task           : str    'continuous' or 'binary'
    trial          : int    0-indexed trial number
    observation    : int    0-indexed observation within trial
    value          : int    Stimulus value (-100..100 continuous; -1/1 binary)
    true_mean      : float  Generative mean (continuous)
    true_std       : float  Generative std (continuous); NaN for binary
    true_p         : float  True Bernoulli probability (binary); NaN for continuous
    qid            : int    Unique sequence ID for each repeated sequence
    prefix_length  : int    Number of fixed prefix observations
    std_condition  : float  Observation std value (continuous); NaN for binary
    response       : float  Participant estimate (NaN if timed out or no response)
    timed_out      : bool   True if response deadline elapsed
    rt             : float  Response time in ms (NaN if timed out)
    time_elapsed   : int    ms since experiment start
"""

import json
import argparse
import pathlib
import pandas as pd
import numpy as np


def parse_participant_file(fpath: pathlib.Path) -> pd.DataFrame:
    """Parse one participant's JSON file into a DataFrame of observation rows."""
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
            'trial':         t.get('trial', np.nan),
            'observation':   t.get('observation', np.nan),
            'value':         t.get('value', np.nan),
            'true_mean':     t.get('true_mean', np.nan),
            'true_std':      t.get('true_std', np.nan),
            'true_p':        t.get('true_p', np.nan) if t.get('true_p') is not None else np.nan,
            'qid':           t.get('qid', np.nan),
            'prefix_length': t.get('prefix_length', np.nan),
            'std_condition': t.get('std_condition', np.nan) if t.get('std_condition') is not None else np.nan,
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
    parser.add_argument('--output', default='data/task_results.pkl',
                        help='Output .pkl path (default: data/task_results.pkl)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output    = pathlib.Path(args.output)

    # Find all JSON files recursively
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

    # Cast types
    for col in ['trial', 'observation', 'value', 'qid', 'prefix_length']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce').astype('Int64')
    for col in ['true_mean', 'true_std', 'true_p', 'std_condition', 'response', 'rt']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')

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
