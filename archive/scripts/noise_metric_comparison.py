"""
scripts/noise_metric_comparison.py

Compare two per-participant noise metrics in the carrabin dataset:

  1. qid_noise   : mean_qid_std() — mean std of responses across repetitions
                   of the same sequence prefix (carrabin figure_carrabin metric)
  2. model_noise : std of residuals (human_response - RL_lambda_response)
                   using best-fit RL_lambda params from data/runs/refit/

Then regress qid_noise ~ model_noise and report correlation.

Run from project root: venv/bin/python scripts/noise_metric_comparison.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.plot_style import mean_qid_std
from utils.paths import data_path, resolve_run_folder

RUN_FOLDER = "refit"

# ── load data ────────────────────────────────────────────────────────────────
human = pd.read_pickle(data_path("carrabin.pkl"))
run_dir = resolve_run_folder(RUN_FOLDER)

resp_path = run_dir / "RL_lambda_carrabin_responses.pkl"
params_path = run_dir / "RL_lambda_carrabin_params.pkl"

if not resp_path.exists():
    print(f"ERROR: {resp_path} not found. Run RL_lambda fits first.")
    sys.exit(1)

model_resp = pd.read_pickle(resp_path)
model_params = pd.read_pickle(params_path)

print(f"Human pids: {sorted(human['pid'].unique())}")
print(f"Model pids: {sorted(model_resp['pid'].unique())}")
print(f"Model params cols: {model_params.columns.tolist()}")
print()

# ── per-pid metrics ───────────────────────────────────────────────────────────
rows = []
for pid in sorted(human["pid"].unique()):
    h = human[human["pid"] == pid].copy()
    m = model_resp[model_resp["pid"] == pid].copy()

    if m.empty:
        print(f"  pid={pid}: no model responses, skipping")
        continue

    # metric 1: qid_noise — requires qid column, uses >=10 repeats per qid
    qid_noise = mean_qid_std(h)

    # metric 2: model_noise — std of residuals across all (trial, obs) pairs
    merged = h.merge(
        m[["trial", "observation", "response"]],
        on=["trial", "observation"],
        suffixes=("_human", "_model"),
    )
    if merged.empty:
        print(f"  pid={pid}: merge empty, skipping")
        continue

    residuals = merged["response_human"] - merged["response_model"]
    model_noise = float(residuals.std())

    # also compute per-observation residual std to check obs profile
    obs_noise = merged.groupby("observation").apply(
        lambda g: (g["response_human"] - g["response_model"]).std()
    )

    # fetch fitted params
    p = model_params[model_params["pid"] == pid]
    alpha_0 = float(p["alpha_0"].iloc[0]) if not p.empty else np.nan
    lambda_ = float(p["lambda_"].iloc[0]) if not p.empty else np.nan

    rows.append({
        "pid": pid,
        "qid_noise": qid_noise,
        "model_noise": model_noise,
        "alpha_0": alpha_0,
        "lambda_": lambda_,
        "n_trials": h["trial"].nunique(),
        "obs_noise_early": float(obs_noise.iloc[:3].mean()),   # obs 1-3
        "obs_noise_late": float(obs_noise.iloc[-3:].mean()),   # obs 3-5
    })

df = pd.DataFrame(rows)
print("=" * 60)
print("Per-pid noise metrics")
print("=" * 60)
print(df[["pid", "qid_noise", "model_noise", "alpha_0", "lambda_"]].to_string(index=False))
print()

# ── summary stats ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Summary statistics")
print("=" * 60)
for col in ["qid_noise", "model_noise"]:
    v = df[col].dropna()
    print(f"{col:>15}: mean={v.mean():.4f}  SD={v.std():.4f}  "
          f"median={v.median():.4f}  "
          f"[{v.quantile(0.25):.4f}, {v.quantile(0.75):.4f}] IQR")
print()

# ── regression: qid_noise ~ model_noise ──────────────────────────────────────
print("=" * 60)
print("Regression: qid_noise ~ model_noise")
print("=" * 60)
valid = df.dropna(subset=["qid_noise", "model_noise"])
print(f"N pids with both metrics: {len(valid)}")

r_p, p_p = pearsonr(valid["model_noise"], valid["qid_noise"])
r_s, p_s = spearmanr(valid["model_noise"], valid["qid_noise"])
print(f"Pearson  r = {r_p:.3f}  p = {p_p:.4f}")
print(f"Spearman r = {r_s:.3f}  p = {p_s:.4f}")
print()

# simple OLS slope and intercept
x = valid["model_noise"].values
y = valid["qid_noise"].values
slope = np.cov(x, y)[0,1] / np.var(x)
intercept = y.mean() - slope * x.mean()
print(f"OLS: qid_noise = {slope:.3f} * model_noise + {intercept:.4f}")
print()

# ── check confounds: does correlation survive partialling out alpha_0/lambda_? ─
print("=" * 60)
print("Partial correlations (controlling for alpha_0 and lambda_)")
print("=" * 60)
from numpy.linalg import lstsq

def partial_r(df, x_col, y_col, controls):
    """Pearson r between residuals of x and y after regressing out controls."""
    sub = df.dropna(subset=[x_col, y_col] + controls)
    X = np.column_stack([sub[c].values for c in controls] + [np.ones(len(sub))])
    res_x = sub[x_col].values - X @ lstsq(X, sub[x_col].values, rcond=None)[0]
    res_y = sub[y_col].values - X @ lstsq(X, sub[y_col].values, rcond=None)[0]
    r, p = pearsonr(res_x, res_y)
    return r, p

r_partial, p_partial = partial_r(df, "model_noise", "qid_noise", ["alpha_0", "lambda_"])
print(f"Partial r (controlling alpha_0, lambda_): r = {r_partial:.3f}  p = {p_partial:.4f}")
print()

# ── obs profile: does model_noise increase with obs index? ────────────────────
print("=" * 60)
print("Residual std by observation index (mean across pids)")
print("=" * 60)
all_residuals = []
for pid in sorted(human["pid"].unique()):
    h = human[human["pid"] == pid]
    m = model_resp[model_resp["pid"] == pid]
    if m.empty:
        continue
    merged = h.merge(m[["trial","observation","response"]],
                     on=["trial","observation"], suffixes=("_h","_m"))
    merged["resid"] = merged["response_h"] - merged["response_m"]
    merged["pid"] = pid
    all_residuals.append(merged[["pid","observation","resid"]])

resid_df = pd.concat(all_residuals)
obs_profile = resid_df.groupby("observation")["resid"].std()
print(f"{'obs':>6}  {'resid_std':>10}")
for obs, val in obs_profile.items():
    print(f"{obs:>6}  {val:>10.4f}")
