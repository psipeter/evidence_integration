"""
scripts/counting_accuracy_scan.py

Compare counting circuit accuracy and trial-to-trial variance across
n_neurons_counting values, using precomputed activity files.

Run:
  # carrabin (radius_c=5)
  venv/bin/python scripts/counting_accuracy_scan.py --dataset carrabin

  # yoo (radius_c=30)
  venv/bin/python scripts/counting_accuracy_scan.py --dataset yoo
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.counting_integrator import load_activities, fast_decode, _eval_idx
from fitting.model_params import _NEF_FIXED, MODEL_PARAMS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="carrabin", choices=["carrabin","yoo"])
parser.add_argument("--alpha_0", type=float, default=None)
parser.add_argument("--lambda_", type=float, default=None)
parser.add_argument("--n_neurons", type=int, default=100)
args = parser.parse_args()

DATASET   = args.dataset
N_NEURONS = args.n_neurons

# Dataset-specific defaults
DATASET_CFG = {
    "carrabin": {"alpha_0": 0.776, "lambda_": 0.927, "n_obs": 5,  "radius_c": 5},
    "yoo":      {"alpha_0": 0.500, "lambda_": 0.500, "n_obs": 30, "radius_c": 30},
}
cfg      = DATASET_CFG[DATASET]
ALPHA_0  = args.alpha_0 if args.alpha_0 is not None else cfg["alpha_0"]
LAMBDA_  = args.lambda_ if args.lambda_ is not None else cfg["lambda_"]
N_OBS    = cfg["n_obs"]
RADIUS_C = cfg["radius_c"]

# Check which activity files exist for this dataset config
data_dir  = Path("data")
available = []
for f in sorted(data_dir.glob(f"counting_activities_n{N_NEURONS}_nc*.pkl")):
    # Verify the file has the right n_obs by checking the timeseries length
    try:
        import pickle
        with open(f, "rb") as fh:
            acts = pickle.load(fh)
        first = acts[1]
        T = first["ideal_count_filt"].shape[0]
        dt = _NEF_FIXED["dt"]
        t_step = _NEF_FIXED["t_obs"] + _NEF_FIXED["t_iti"]
        n_obs_file = round(T * dt / t_step)
        if n_obs_file == N_OBS:
            nc = int(f.stem.split("_nc")[1])
            available.append(nc)
    except Exception:
        pass

if not available:
    print(f"No activity files found for dataset={DATASET} "
          f"(n_neurons={N_NEURONS}, n_obs={N_OBS}).")
    print(f"Generate with:")
    print(f"  venv/bin/python models/counting_integrator.py "
          f"--precompute_activities --n_trials 200 "
          f"--n_neurons {N_NEURONS} --n_neurons_counting <NC> --dataset {DATASET}")
    sys.exit(1)

NC_VALUES  = sorted(available)
CHECK_OBS  = [1, 2, 3, 5, 10, 15, 20, 25, 30] if N_OBS >= 30 else [1, 2, 3, 4, 5]

params_base = {
    **_NEF_FIXED,
    "n_neurons":      N_NEURONS,
    "n_obs":          N_OBS,
    "radius_c":       RADIUS_C,
}

print(f"dataset={DATASET}  n_neurons={N_NEURONS}  radius_c={RADIUS_C}  "
      f"n_obs={N_OBS}  alpha_0={ALPHA_0}  lambda_={LAMBDA_}")
print(f"Found activity files for nc values: {NC_VALUES}\n")


def analyse_nc(nc: int) -> pd.DataFrame:
    path = data_dir / f"counting_activities_n{N_NEURONS}_nc{nc}.pkl"
    if not path.exists():
        return pd.DataFrame()
    acts   = load_activities(path=path)
    params = {**params_base, "n_neurons_counting": nc}
    rows   = []
    for trial, act in acts.items():
        dec = fast_decode(act, alpha_0=ALPHA_0, lambda_=LAMBDA_)
        ic  = act["ideal_count_filt"]
        iw  = ALPHA_0 / np.maximum(ic, 1.0) ** LAMBDA_
        mem = act["mem_filt_T"]
        cd  = (dec["W_count"]  @ mem).ravel()
        wd  = (dec["W_weight"] @ mem).ravel()
        idx = _eval_idx(params, len(ic))
        for i, obs in enumerate(range(1, N_OBS + 1)):
            if i >= len(idx):
                continue
            k = idx[i]
            rows.append({
                "nc":          nc,
                "trial":       trial,
                "obs":         obs,
                "count_dec":   float(cd[k]),
                "weight_dec":  float(wd[k]),
                "count_ideal": float(ic[k]),
                "weight_ideal":float(iw[k]),
            })
    return pd.DataFrame(rows)


all_dfs = []
for nc in NC_VALUES:
    print(f"Analysing nc={nc}...", flush=True)
    df = analyse_nc(nc)
    if not df.empty:
        all_dfs.append(df)

if not all_dfs:
    print("No data loaded.")
    sys.exit(1)

combined = pd.concat(all_dfs, ignore_index=True)

# ── Summary tables ────────────────────────────────────────────────────────────
for signal, ideal_col, dec_col in [
    ("Count",  "count_ideal",  "count_dec"),
    ("Weight", "weight_ideal", "weight_dec"),
]:
    print(f"\n=== {signal}: mean decoded, std, relative error ===")
    header = f"{'obs':>4}" + "".join(
        f"  {'nc='+str(nc):>8} {'std':>7} {'rel_err':>8}"
        for nc in NC_VALUES
    )
    print(header)
    obs_to_show = [o for o in CHECK_OBS if o <= N_OBS]
    for obs in obs_to_show:
        row_str = f"{obs:>4}"
        for nc in NC_VALUES:
            sub = combined[(combined["nc"]==nc) & (combined["obs"]==obs)]
            if sub.empty:
                row_str += "  " + "-"*25
                continue
            ideal = sub[ideal_col].mean()
            dm    = sub[dec_col].mean()
            ds    = sub[dec_col].std()
            rel   = abs(dm - ideal) / max(abs(ideal), 1e-6)
            row_str += f"  {dm:>8.3f} {ds:>7.4f} {rel:>8.3f}"
        print(row_str)

# ── Key metrics ───────────────────────────────────────────────────────────────
print("\n=== Key metrics per nc ===")
print(f"{'nc':>6}  {'count_rmse':>12}  {'weight_rmse':>13}  "
      f"{'weight_std_mean':>16}  {'count_at_maxobs':>16}")
for nc in NC_VALUES:
    sub = combined[combined["nc"]==nc]
    if sub.empty:
        continue
    c_rmse     = np.sqrt(((sub["count_dec"]  - sub["count_ideal"])**2).mean())
    w_rmse     = np.sqrt(((sub["weight_dec"] - sub["weight_ideal"])**2).mean())
    w_std_mean = sub.groupby("obs")["weight_dec"].std().mean()
    sat_val    = sub[sub["obs"]==N_OBS]["count_dec"].mean()
    print(f"{nc:>6}  {c_rmse:>12.4f}  {w_rmse:>13.4f}  "
          f"{w_std_mean:>16.4f}  {sat_val:>16.3f}")
