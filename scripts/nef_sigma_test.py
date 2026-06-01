"""
scripts/nef_sigma_test.py

Test sigma_NEF with the new architecture:
  - n_neurons=100, n_neurons_counting=100, radius_c=5
  - Precomputed counting activities (fast_decode per trial)
  - seed = trial number directly

Run: venv/bin/python scripts/nef_sigma_test.py
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from fitting.losses import compute_loss
from fitting.model_params import _NEF_FIXED
from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
from models.counting_integrator import load_activities, fast_decode
from models.RNN import fit as rnn_fit
from utils.carrabin_transform import apply_carrabin_transform
from utils.paths import data_path

PID       = 18
ALPHA_0   = 0.603
LAMBDA_   = 0.735
LABEL     = "NEF_n100_nc100_rc5"
RUN_DIR   = Path("data/runs/carrabin")
RUN_FOLDER = "carrabin"

human    = pd.read_pickle(data_path("carrabin.pkl"))
h_pid    = human[human["pid"] == PID]

sigma_df      = pd.read_pickle(RUN_DIR / "RNN_sigma_carrabin_sigma.pkl")
human_sigma   = float(sigma_df[(sigma_df["source"]=="human")  & (sigma_df["pid"]==PID)]["sigma"].iloc[0])
nef_old_sigma = float(sigma_df[(sigma_df["source"]=="NEF")    & (sigma_df["pid"]==PID)]["sigma"].iloc[0])
print(f"pid={PID}  alpha_0={ALPHA_0}  lambda_={LAMBDA_}")
print(f"Reference:  human sigma={human_sigma:.4f}  NEF(n=200,nc=2000) sigma={nef_old_sigma:.4f}")
print()

# ── Simulate ──────────────────────────────────────────────────────────────────
resp_path = RUN_DIR / f"{LABEL}_carrabin_responses.pkl"

if resp_path.exists():
    print(f"Loading saved: {resp_path.name}")
    resp = pd.read_pickle(resp_path)
else:
    print(f"Simulating {LABEL}  (n=100, nc=100, radius_c=5, precomputed activities)...")

    try:
        activity_map = load_activities(n_neurons=100, n_neurons_counting=100)
        print(f"  Loaded {len(activity_map)} precomputed activity sets")
    except FileNotFoundError:
        print("  ERROR: counting_activities_n100_nc100.pkl not found")
        print("  Run: venv/bin/python models/counting_integrator.py "
              "--precompute_activities --n_neurons 100 --n_neurons_counting 100 --dataset carrabin")
        sys.exit(1)

    pfull = {
        **PARAM_DEFAULTS,
        **_NEF_FIXED,
        "model_type":         "NEF",
        "dataset":            "carrabin",
        "pid":                PID,
        "alpha_0":            ALPHA_0,
        "lambda_":            LAMBDA_,
        "n_neurons":          100,
        "n_neurons_counting": 100,
        "radius_c":           5,
    }

    n_trials = h_pid["trial"].nunique()
    t0   = time.time()
    rows = []
    for ti, (trial, trial_data) in enumerate(h_pid.groupby("trial"), 1):
        t_trial  = time.time()
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        p          = {**pfull, "seed": int(trial)}

        activity = activity_map.get(int(trial))
        if activity is not None:
            decoders = fast_decode(activity, alpha_0=ALPHA_0, lambda_=LAMBDA_)
        else:
            decoders = _pretrain({**p, "base_seed": int(trial)})

        responses  = _simulate_trial(obs_values, p, decoders)
        t_elapsed  = time.time() - t0
        eta        = t_elapsed / ti * (n_trials - ti)
        sys.stdout.write(
            f"\r  trial {ti:3d}/{n_trials}  "
            f"{time.time()-t_trial:.1f}s/trial  "
            f"elapsed={t_elapsed:.0f}s  ETA={eta:.0f}s   "
        )
        sys.stdout.flush()

        for i, (_, row) in enumerate(trial_data.iterrows()):
            rows.append({
                "model_type":  "NEF",
                "pid":         PID,
                "trial":       int(trial),
                "observation": int(row["observation"]),
                "response":    float(responses[i]),
            })

    print()  # newline after progress bar
    resp = apply_carrabin_transform(pd.DataFrame(rows), "carrabin")
    resp.to_pickle(resp_path)
    print(f"Elapsed: {time.time()-t0:.1f}s  Saved: {resp_path.name}")

# ── Variance analysis ─────────────────────────────────────────────────────────
print(f"\nresponse std (transformed): {resp['response'].std():.4f}")
print(f"response_raw std:           {resp['response_raw'].std():.4f}")

qid3 = h_pid[h_pid["observation"]==3][["trial","qid"]].set_index("trial")["qid"]
prefix_groups = {}
for trial, qid in qid3.items():
    prefix_groups.setdefault(qid, []).append(trial)
repeated = {q: ts for q, ts in prefix_groups.items() if len(ts) >= 5}

stds = []
for qid, trials in repeated.items():
    for obs in [1, 2, 3]:
        vals = resp[(resp["trial"].isin(trials)) & (resp["observation"]==obs)]["response"].values
        if len(vals) >= 3:
            stds.append(vals.std())
prefix_std = float(np.mean(stds)) if stds else float("nan")
print(f"Mean std across same-prefix trials: {prefix_std:.4f}")

rmse = compute_loss({"dataset": "carrabin"}, resp, h_pid)
print(f"RMSE vs human pid={PID}: {rmse:.4f}")

# ── RNN fit ───────────────────────────────────────────────────────────────────
source_resp_path = RUN_DIR / f"{LABEL}_carrabin_responses.pkl"
resp.to_pickle(source_resp_path)

print(f"\nFitting RNN to {LABEL} responses...  (this takes ~5 min)", flush=True)
rnn_result = rnn_fit(
    pid=PID,
    source=LABEL,
    run_folder=RUN_FOLDER,
    max_epochs=5000,
    patience=300,
    verbose=False,
)
sigma_new = float(rnn_result["sigma"]["sigma"].iloc[0])
cv_rmse   = float(rnn_result["params"]["cv_rmse"].iloc[0])
print(f"RNN CV RMSE: {cv_rmse:.4f}   sigma: {sigma_new:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Summary for pid={PID} ===")
print(f"  {'Source':<40} {'sigma':>8} {'rmse':>8} {'prefix_std':>12} {'ratio':>8}")
print(f"  {'-'*75}")
print(f"  {'human':<40} {human_sigma:>8.4f} {'—':>8} {'—':>12} {'1.00':>8}")
print(f"  {'NEF n=200 nc=2000 (old)':<40} {nef_old_sigma:>8.4f} {'—':>8} {'—':>12} {nef_old_sigma/human_sigma:>8.2f}")
print(f"  {LABEL:<40} {sigma_new:>8.4f} {rmse:>8.4f} {prefix_std:>12.4f} {sigma_new/human_sigma:>8.2f}")
