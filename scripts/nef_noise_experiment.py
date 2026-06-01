"""
scripts/nef_noise_experiment.py

Test whether per-trial counting decoder retraining increases NEF response noise.

Configuration: n_neurons=100, n_neurons_counting=100, counting decoders
retrained per trial using the trial-specific seed (rather than the fixed
base_seed used in the standard pipeline).

This means the weight decoder W_weight varies trial-to-trial, randomly
modulating the effective learning rate and introducing noise into updates.

Run: venv/bin/python scripts/nef_noise_experiment.py
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
from models.RNN import fit as rnn_fit
from utils.carrabin_transform import apply_carrabin_transform
from utils.paths import data_path
from utils.run_params import trial_seed as _trial_seed

PID       = 18
RUN_DIR   = Path("data/runs/carrabin")
RUN_FOLDER = "carrabin"
ALPHA_0   = 0.603
LAMBDA_   = 0.735
BEST_TRIAL_NUMBER = 1
LABEL     = "NEF_n100_nc100_pertrial"

seed = abs(hash((PID, BEST_TRIAL_NUMBER))) % (2**31)

human = pd.read_pickle(data_path("carrabin.pkl"))
h_pid = human[human["pid"] == PID]

# Reference sigmas
sigma_df    = pd.read_pickle(RUN_DIR / "RNN_sigma_carrabin_sigma.pkl")
human_sigma = float(sigma_df[(sigma_df["source"]=="human") &
                              (sigma_df["pid"]==PID)]["sigma"].iloc[0])
nef_sigma   = float(sigma_df[(sigma_df["source"]=="NEF") &
                              (sigma_df["pid"]==PID)]["sigma"].iloc[0])

print(f"pid={PID}  alpha_0={ALPHA_0}  lambda_={LAMBDA_}  seed={seed}")
print(f"Reference: human sigma={human_sigma:.4f}  NEF(n=200) sigma={nef_sigma:.4f}")
print()

# ── Simulate ──────────────────────────────────────────────────────────────────
resp_path = RUN_DIR / f"{LABEL}_carrabin_responses.pkl"

if resp_path.exists():
    print(f"Loading saved: {resp_path.name}")
    resp = pd.read_pickle(resp_path)
else:
    print(f"Simulating {LABEL}  (n_neurons=100, n_neurons_counting=100, per-trial pretrain)...")
    pfull = {
        **PARAM_DEFAULTS,
        **_NEF_FIXED,
        "model_type":         "NEF",
        "dataset":            "carrabin",
        "pid":                PID,
        "alpha_0":            ALPHA_0,
        "lambda_":            LAMBDA_,
        "seed":               seed,
        "base_seed":          seed,
        "n_obs":              5,
        "n_neurons":          100,
        "n_neurons_counting": 100,
    }

    t0   = time.time()
    rows = []
    for trial, trial_data in h_pid.groupby("trial"):
        trial_data = trial_data.sort_values("observation")
        obs_values = trial_data["value"].to_numpy(dtype=float)
        trial_s    = _trial_seed(int(pfull["seed"]), int(trial))
        p          = {**pfull, "seed": trial_s, "base_seed": trial_s}
        t_pre = time.time()
        decoders   = _pretrain(p)   # retrain counting decoders with trial seed
        responses  = _simulate_trial(obs_values, p, decoders)
        print(f"  trial {int(trial)}: {time.time()-t_pre:.1f}s", flush=True)
        for i, (_, row) in enumerate(trial_data.iterrows()):
            rows.append({
                "model_type":  "NEF",
                "pid":         PID,
                "trial":       int(trial),
                "observation": int(row["observation"]),
                "response":    float(responses[i]),
            })

    resp = apply_carrabin_transform(pd.DataFrame(rows), "carrabin")
    resp.to_pickle(resp_path)
    print(f"Elapsed: {time.time()-t0:.1f}s  Saved: {resp_path.name}")

# ── Variance analysis ─────────────────────────────────────────────────────────
print(f"\nresponse std (transformed): {resp['response'].std():.4f}")
print(f"response_raw std:           {resp['response_raw'].std():.4f}")

# Variance across trials with identical length-3 prefixes
qid3 = h_pid[h_pid["observation"]==3][["trial","qid"]].set_index("trial")["qid"]
prefix_groups = {}
for trial, qid in qid3.items():
    prefix_groups.setdefault(qid, []).append(trial)
repeated = {q: ts for q, ts in prefix_groups.items() if len(ts) >= 5}

stds = []
for qid, trials in repeated.items():
    for obs in [1, 2, 3]:
        vals = resp[(resp["trial"].isin(trials)) &
                    (resp["observation"]==obs)]["response"].values
        if len(vals) >= 3:
            stds.append(vals.std())
prefix_std = float(np.mean(stds)) if stds else float("nan")
print(f"Mean std across same-prefix trials: {prefix_std:.4f}")

# RMSE vs human
rmse = compute_loss({"dataset":"carrabin"}, resp, h_pid)
print(f"RMSE vs human pid={PID}: {rmse:.4f}")

# ── RNN fit for sigma ─────────────────────────────────────────────────────────
source_resp_path = RUN_DIR / f"{LABEL}_carrabin_responses.pkl"
resp.to_pickle(source_resp_path)  # already done above, but ensure correct name

print(f"\nFitting RNN to {LABEL} responses...")
rnn_result = rnn_fit(
    pid=PID,
    source=LABEL,
    run_folder=RUN_FOLDER,
    max_epochs=5000,
    patience=300,
    verbose=False,
)
sigma_new  = float(rnn_result["sigma"]["sigma"].iloc[0])
cv_rmse    = float(rnn_result["params"]["cv_rmse"].iloc[0])
print(f"RNN CV RMSE: {cv_rmse:.4f}   sigma: {sigma_new:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Summary for pid={PID} ===")
print(f"  {'Source':<35} {'sigma':>8} {'rmse':>8} {'prefix_std':>12} {'ratio':>8}")
print(f"  {'-'*70}")
print(f"  {'human':<35} {human_sigma:>8.4f} {'—':>8} {'—':>12} {'1.00':>8}")
print(f"  {'NEF n=200 nc=2000 (fitted)':<35} {nef_sigma:>8.4f} {'—':>8} {'—':>12} {nef_sigma/human_sigma:>8.2f}")
print(f"  {LABEL:<35} {sigma_new:>8.4f} {rmse:>8.4f} {prefix_std:>12.4f} {sigma_new/human_sigma:>8.2f}")
