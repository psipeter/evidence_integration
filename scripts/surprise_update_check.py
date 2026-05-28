"""
scripts/surprise_update_check.py

Test whether 'surprising' observations (large prediction errors) produce
disproportionately large updates, over and above the optimal 1/t scaling.

For each (pid, trial, observation t >= 2):
  - v_optimal(t-1): running mean of x_1..x_{t-1}  (= optimal prior estimate)
  - prediction_error: |x_t - v_optimal(t-1)|
  - optimal_update:   prediction_error / t          (what 1/n rule predicts)
  - actual_update:    |response(t) - response(t-1)| (what participant did)
  - surprise_ratio:   actual_update / optimal_update (>1 = amplified)

If surprise-driven updating is present, surprise_ratio should increase
monotonically with prediction_error, even after controlling for t.

Run from project root: venv/bin/python scripts/surprise_update_check.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import data_path


def build_surprise_df(dataset: str) -> pd.DataFrame:
    df = pd.read_pickle(data_path(f"{dataset}.pkl"))
    df = df.sort_values(["pid", "trial", "observation"])

    rows = []
    for (pid, trial), grp in df.groupby(["pid", "trial"]):
        grp = grp.sort_values("observation").reset_index(drop=True)
        values    = grp["value"].to_numpy(dtype=float)
        responses = grp["response"].to_numpy(dtype=float)
        n = len(values)

        for i in range(1, n):
            t           = i + 1                          # observation number (1-indexed)
            v_opt_prev  = np.mean(values[:i])            # optimal prior = mean of x_1..x_{t-1}
            pred_error  = abs(values[i] - v_opt_prev)    # |x_t - optimal prior|
            opt_update  = pred_error / t                 # optimal 1/t update
            act_update  = abs(responses[i] - responses[i - 1])  # actual |Δresponse|

            if opt_update < 1e-8:
                continue  # avoid division by zero for zero-error obs

            surprise_ratio = act_update / opt_update     # amplification factor

            rows.append({
                "pid": pid, "trial": trial, "observation": t,
                "pred_error": pred_error,
                "opt_update": opt_update,
                "act_update": act_update,
                "surprise_ratio": surprise_ratio,
            })

    return pd.DataFrame(rows)


def report(dataset: str) -> None:
    print(f"\n{'='*62}")
    print(f"Dataset: {dataset}")
    print(f"{'='*62}")

    sdf = build_surprise_df(dataset)

    # --- 1. Raw correlation: pred_error vs actual_update ---
    r_raw, p_raw = pearsonr(sdf["pred_error"], sdf["act_update"])
    print(f"\n1. Pearson r(pred_error, actual_update): "
          f"r={r_raw:.3f}  p={p_raw:.4f}  (n={len(sdf):,})")
    print("   (Expected >0 regardless of mechanism — just checks data quality)")

    # --- 2. Correlation: pred_error vs surprise_ratio (key test) ---
    # If surprise amplification exists, large pred_error → surprise_ratio > 1
    r_sr, p_sr = pearsonr(sdf["pred_error"], sdf["surprise_ratio"])
    r_sr_sp, _ = spearmanr(sdf["pred_error"], sdf["surprise_ratio"])
    print(f"\n2. Key test — pred_error vs surprise_ratio "
          f"(actual / optimal update):")
    print(f"   Pearson  r={r_sr:.3f}  p={p_sr:.4f}")
    print(f"   Spearman r={r_sr_sp:.3f}")
    print(f"   Mean surprise_ratio:   {sdf['surprise_ratio'].mean():.3f}")
    print(f"   Median surprise_ratio: {sdf['surprise_ratio'].median():.3f}")
    print("   (r > 0 → surprise amplification; r < 0 → dampening)")

    # --- 3. Residual correlation after partialling out observation index ---
    # Regress pred_error on t, take residual pred_error; same for surprise_ratio
    sdf["t_log"] = np.log(sdf["observation"])  # log(t) captures 1/t scaling
    from numpy.linalg import lstsq

    def residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        coef, *_ = lstsq(X, y, rcond=None)
        return y - X @ coef

    X = np.column_stack([sdf["t_log"].values, np.ones(len(sdf))])
    res_pe  = residualise(sdf["pred_error"].values, X)
    res_sr  = residualise(sdf["surprise_ratio"].values, X)
    r_part, p_part = pearsonr(res_pe, res_sr)
    print(f"\n3. Partial r(pred_error, surprise_ratio | log(t)): "
          f"r={r_part:.3f}  p={p_part:.4f}")

    # --- 4. Binned: surprise_ratio by pred_error quintile ---
    sdf["pe_quintile"] = pd.qcut(sdf["pred_error"], q=5, labels=False, duplicates="drop")
    binned = sdf.groupby("pe_quintile")["surprise_ratio"].agg(["mean","median","std"])
    print(f"\n4. Mean surprise_ratio by pred_error quintile:")
    print(f"   {'Quintile':>10} {'mean_SR':>10} {'median_SR':>10} {'SD':>8}")
    for q, row in binned.iterrows():
        pe_lo = sdf.loc[sdf["pe_quintile"]==q, "pred_error"].min()
        pe_hi = sdf.loc[sdf["pe_quintile"]==q, "pred_error"].max()
        print(f"   Q{q+1} [{pe_lo:.2f},{pe_hi:.2f}]  "
              f"{row['mean']:>10.3f} {row['median']:>10.3f} {row['std']:>8.3f}")

    # --- 5. Per-pid summary ---
    pid_corrs = []
    for pid, grp in sdf.groupby("pid"):
        if len(grp) < 10:
            continue
        r, p = pearsonr(grp["pred_error"], grp["surprise_ratio"])
        pid_corrs.append(r)
    pid_corrs = np.array(pid_corrs)
    from scipy.stats import ttest_1samp
    t_stat, t_p = ttest_1samp(pid_corrs, 0)
    print(f"\n5. Per-pid r(pred_error, surprise_ratio):")
    print(f"   Mean r={pid_corrs.mean():.3f}  SD={pid_corrs.std():.3f}  "
          f"t-test vs 0: t={t_stat:.2f}  p={t_p:.4f}")
    print(f"   % pids with r>0: {(pid_corrs>0).mean()*100:.0f}%")


if __name__ == "__main__":
    report("carrabin")
    report("yoo")
