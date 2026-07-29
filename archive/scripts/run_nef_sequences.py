"""
scripts/run_nef_sequences.py
============================
Run NEF on test sequences for one lambda_ value (fixed alpha_0=1.0).
Lambda is specified by index into a uniform grid matching RL_lambda simulation.
Saves to a separate per-index pkl so parallel runs don't conflict.
Collect results with scripts/collect_nef_sequences.py afterwards.

Usage:
    venv/bin/python scripts/run_nef_sequences.py --lambda_index 0
    venv/bin/python scripts/run_nef_sequences.py --lambda_index 49
    ...etc  (indices 0-99 for n_lambdas=100)

Output: data/runs/test_sequences/nef_runs/nef_l{index:03d}.pkl
"""

import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from models.NEF import run as nef_run
from fitting.model_params import MODEL_PARAMS
from utils.paths import resolve_run_folder

ALPHA_0   = 1.0
N_LAMBDAS = 100
LAMBDAS   = np.linspace(0.01, 0.99, N_LAMBDAS)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lambda_index', type=int, required=True,
                        help=f'Index into lambda grid (0-{N_LAMBDAS-1})')
    parser.add_argument('--tasks', nargs='+',
                        default=['task_continuous', 'task_binary'])
    args = parser.parse_args()

    idx      = args.lambda_index
    lambda_  = float(LAMBDAS[idx])
    alpha_0  = ALPHA_0
    params_str = f'alpha_0={alpha_0:.3f} lambda_={lambda_:.3f}'
    model_id   = f'NEF[{idx}]'

    out_folder = resolve_run_folder('test_sequences') / 'nef_runs'
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f'nef_l{idx:03d}.pkl'

    if out_path.exists():
        print(f"Already exists: {out_path} — skipping.")
        return

    print(f"Running NEF [{idx}]: alpha_0={alpha_0}  lambda_={lambda_:.4f}")
    print(f"Output: {out_path}")

    all_rows = []

    for task in args.tasks:
        task_label = task.replace('task_', '')
        fixed      = MODEL_PARAMS[task]['NEF']['fixed']

        seq_df   = pd.read_pickle(f'task/sequences/{task_label}_sequences.pkl')
        seq_meta = seq_df[['trial','qid','trial_type','prefix_length',
                            'observation','value','true_mean','true_std',
                            'true_p','std_condition']].copy()
        seq_meta['value_norm'] = (seq_meta['value'] / 100.0
                                   if task_label == 'continuous'
                                   else seq_meta['value'].astype(float))

        print(f"\n  Task: {task_label.upper()} ({seq_meta['trial'].nunique()} trials)")

        params = {
            **fixed,
            'model_type': 'NEF',
            'dataset':    task,
            'pid':        0,
            'alpha_0':    alpha_0,
            'lambda_':    lambda_,
        }

        t0 = time.time()
        try:
            nef_df = nef_run(params)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print(f"  Done in {time.time()-t0:.1f}s ({len(nef_df)} rows)")

        nef_df  = nef_df.rename(columns={'response': '_nef_response'})
        merged  = nef_df.merge(seq_meta, on=['trial','observation'], how='left')

        for _, row in merged.iterrows():
            all_rows.append({
                'task':           task_label,
                'model_type':     'NEF',
                'model_id':       model_id,
                'param_set_id':   idx,
                'params_str':     params_str,
                'trial':          int(row['trial']),
                'qid':            row.get('qid'),
                'trial_type':     row.get('trial_type'),
                'prefix_length':  row.get('prefix_length', 0),
                'observation':    int(row['observation']),
                'value':          row['value'],
                'value_norm':     row['value_norm'],
                'response':       float(row['_nef_response']),
                'response_raw':   float(row.get('response_raw', row['_nef_response'])),
                'true_mean':      row.get('true_mean'),
                'true_std':       row.get('true_std'),
                'true_p':         row.get('true_p'),
                'std_condition':  row.get('std_condition'),
            })

    if not all_rows:
        print("No rows generated — exiting.")
        return

    df = pd.DataFrame(all_rows)
    df.to_pickle(out_path)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print("JOB_COMPLETE")


if __name__ == '__main__':
    main()
