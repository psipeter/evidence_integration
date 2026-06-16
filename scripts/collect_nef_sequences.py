"""
scripts/collect_nef_sequences.py
=================================
Collect all per-param NEF run pkls from data/runs/test_sequences/nef_runs/
and append them to the main test_sequences_responses.pkl.

Usage:
    venv/bin/python scripts/collect_nef_sequences.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from utils.paths import resolve_run_folder


def main():
    out_folder = resolve_run_folder('test_sequences')
    nef_dir    = out_folder / 'nef_runs'
    pkl_path   = out_folder / 'test_sequences_responses.pkl'

    nef_files = sorted(nef_dir.glob('nef_*.pkl'))
    if not nef_files:
        print(f"No NEF run files found in {nef_dir}")
        return

    print(f"Found {len(nef_files)} NEF run files:")
    nef_dfs = []
    for f in nef_files:
        df = pd.read_pickle(f)
        print(f"  {f.name}: {len(df)} rows  "
              f"params={df['params_str'].iloc[0]}  "
              f"tasks={df['task'].unique().tolist()}")
        nef_dfs.append(df)

    df_nef = pd.concat(nef_dfs, ignore_index=True)

    # Re-index model_id as NEF[0], NEF[1], ... per unique param set
    param_sets = df_nef['params_str'].unique()
    id_map = {ps: f'NEF[{i}]' for i, ps in enumerate(sorted(param_sets))}
    df_nef['model_id']     = df_nef['params_str'].map(id_map)
    df_nef['param_set_id'] = df_nef['params_str'].map(
        {ps: i for i, ps in enumerate(sorted(param_sets))})

    # Load existing and strip any old NEF rows to avoid duplicates
    df_existing = pd.read_pickle(pkl_path)
    df_existing = df_existing[df_existing['model_type'] != 'NEF'].copy()
    print(f"\nExisting (non-NEF) rows: {len(df_existing)}")

    df_combined = pd.concat([df_existing, df_nef], ignore_index=True)
    df_combined.to_pickle(pkl_path)
    print(f"Saved {len(df_combined)} total rows "
          f"({len(df_nef)} NEF) to {pkl_path}")
    print(f"Model types: {df_combined['model_type'].unique().tolist()}")
    print("JOB_COMPLETE")


if __name__ == '__main__':
    main()
