"""
scripts/counting_accuracy_scan.py

Compare counting circuit accuracy and trial-to-trial variance across
n_neurons_counting values, using precomputed activity files.

Run: venv/bin/python scripts/counting_accuracy_scan.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.counting_integrator import load_activities, fast_decode, _eval_idx
from fitting.model_params import _NEF_FIXED

ALPHA_0  = 0.603
LAMBDA_  = 0.735
N_NEURONS = 100
NC_VALUES = [100, 200]   # extend as more activity files are generated
CHECK_OBS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]

params_base = {
    **_NEF_FIXED,
    "n_neurons":          N_NEURONS,
    "n_neurons_counting": N_NEURONS,   # overridden per nc below
    "n_obs":              _NEF_FIXED["lmu_n_obs_max"],
}


def analyse_nc(nc: int) -> pd.DataFrame:
    path = Path("data") / f"counting_activities_n{N_NEURONS}_nc{nc}.pkl"
    if not path.exists():
        print(f"  Missing: {path.name} — skipping nc={nc}")
        return pd.DataFrame()

    acts   = load_activities(n_neurons=N_NEURONS, n_neurons_counting=nc)
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

        for i, obs in enumerate(range(1, int(params["n_obs"]) + 1)):
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


print(f"alpha_0={ALPHA_0}  lambda_={LAMBDA_}  n_neurons={N_NEURONS}\n")

all_dfs = []
for nc in NC_VALUES:
    print(f"Loading nc={nc}...")
    df = analyse_nc(nc)
    if not df.empty:
        all_dfs.append(df)

if not all_dfs:
    print("No data found.")
    sys.exit(1)

combined = pd.concat(all_dfs, ignore_index=True)

# ── Print summary tables ──────────────────────────────────────────────────────
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
    for obs in CHECK_OBS:
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

# ── Key metrics summary ───────────────────────────────────────────────────────
print("\n=== Key metrics per nc ===")
print(f"{'nc':>6}  {'count_rmse':>12}  {'weight_rmse':>13}  "
      f"{'weight_std_mean':>16}  {'count_saturation':>17}")
for nc in NC_VALUES:
    sub = combined[combined["nc"]==nc]
    if sub.empty:
        continue
    # RMSE at observation midpoints
    c_rmse = np.sqrt(((sub["count_dec"]  - sub["count_ideal"])**2).mean())
    w_rmse = np.sqrt(((sub["weight_dec"] - sub["weight_ideal"])**2).mean())
    # Mean std of weight across trials (noise indicator)
    w_std_mean = sub.groupby("obs")["weight_dec"].std().mean()
    # Count saturation: decoded count at obs=30 vs ideal=30
    sat_val = sub[sub["obs"]==30]["count_dec"].mean()
    print(f"{nc:>6}  {c_rmse:>12.4f}  {w_rmse:>13.4f}  "
          f"{w_std_mean:>16.4f}  {sat_val:>17.3f}")
