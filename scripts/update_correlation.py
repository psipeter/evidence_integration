"""
scripts/update_correlation.py

Compute per-participant Pearson and Spearman correlation between model and
human signed updates (response(t) - response(t-1)) across all (trial, obs)
pairs, skipping obs=1.

Datasets:
  carrabin: Mean, NoisyCounting, NEF vs human
  yoo:      Mean, PrimacyRecency, NEF vs human

Run from project root: venv/bin/python scripts/update_correlation.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import data_path, resolve_run_folder

RUN_FOLDER = "refit"


def compute_updates(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed update column: response(t) - response(t-1) within each
    (pid, trial). Obs 1 gets NaN and is dropped by caller."""
    df = df.sort_values(["pid", "trial", "observation"]).copy()
    df["update"] = df.groupby(["pid", "trial"])["response"].diff()
    return df[df["update"].notna()]


def update_corr_per_pid(
    human_upd: pd.DataFrame,
    model_upd: pd.DataFrame,
) -> pd.DataFrame:
    """Per-pid Pearson and Spearman r between model and human updates."""
    rows = []
    merged = human_upd.merge(
        model_upd[["pid", "trial", "observation", "update"]],
        on=["pid", "trial", "observation"],
        suffixes=("_h", "_m"),
    )
    for pid, grp in merged.groupby("pid"):
        h = grp["update_h"].to_numpy(dtype=float)
        m = grp["update_m"].to_numpy(dtype=float)
        if len(h) < 4:
            continue
        r_p, p_p = pearsonr(h, m)
        r_s, p_s = spearmanr(h, m)
        rows.append({
            "pid":       pid,
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_r": float(r_s),
            "spearman_p": float(p_s),
            "n_pairs":   len(h),
        })
    return pd.DataFrame(rows)


def report(dataset: str, models: list[str]) -> dict:
    run_dir  = resolve_run_folder(RUN_FOLDER)
    human    = pd.read_pickle(data_path(f"{dataset}.pkl"))
    human_upd = compute_updates(human)

    results = {}
    print(f"\n{'='*62}")
    print(f"Dataset: {dataset}")
    print(f"{'='*62}")
    print(f"  {'Model':<16} {'Pearson r':>10} {'SD':>7} {'Spearman r':>11} {'SD':>7}  {'N pids':>7}")

    for mt in models:
        f = run_dir / f"{mt}_{dataset}_responses.pkl"
        if not f.exists():
            print(f"  {mt:<16} MISSING")
            continue
        model_resp = pd.read_pickle(f)
        model_upd  = compute_updates(model_resp)
        corr_df    = update_corr_per_pid(human_upd, model_upd)
        results[mt] = corr_df

        pr = corr_df["pearson_r"]
        sr = corr_df["spearman_r"]
        print(f"  {mt:<16} {pr.mean():>10.3f} {pr.std():>7.3f} "
              f"{sr.mean():>11.3f} {sr.std():>7.3f}  {len(corr_df):>7}")

    # pairwise Wilcoxon vs NEF on Pearson r
    if "NEF" in results:
        nef_r = results["NEF"].set_index("pid")["pearson_r"]
        print(f"\n  Wilcoxon signed-rank vs NEF (Pearson r):")
        for mt in models:
            if mt == "NEF" or mt not in results:
                continue
            other_r = results[mt].set_index("pid")["pearson_r"]
            shared  = nef_r.index.intersection(other_r.index)
            if len(shared) < 4:
                continue
            diff = nef_r.loc[shared].values - other_r.loc[shared].values
            if np.all(diff == 0) or np.nanstd(diff) == 0:
                continue
            stat, p = wilcoxon(nef_r.loc[shared].values,
                               other_r.loc[shared].values)
            direction = "NEF > other" if diff.mean() > 0 else "NEF < other"
            print(f"    NEF vs {mt:<14}: p={p:.4f}  mean Δr={diff.mean():+.3f}  ({direction})")

    # per-observation breakdown (pooled across pids and trials)
    print(f"\n  Pearson r by observation index (pooled across pids/trials):")
    print(f"  {'obs':>5}", end="")
    for mt in models:
        print(f"  {mt:>14}", end="")
    print()

    # build merged table once
    human_upd_obs = human_upd.copy()
    for mt in models:
        if mt not in results:
            continue
        f = run_dir / f"{mt}_{dataset}_responses.pkl"
        model_resp = pd.read_pickle(f)
        model_upd  = compute_updates(model_resp)
        human_upd_obs = human_upd_obs.merge(
            model_upd[["pid","trial","observation","update"]].rename(
                columns={"update": f"update_{mt}"}),
            on=["pid","trial","observation"], how="left",
        )

    obs_vals = sorted(human_upd_obs["observation"].unique())
    for obs in obs_vals:
        sub = human_upd_obs[human_upd_obs["observation"] == obs]
        h   = sub["update"].to_numpy(dtype=float)
        print(f"  {obs:>5}", end="")
        for mt in models:
            col = f"update_{mt}"
            if col not in sub.columns:
                print(f"  {'—':>14}", end="")
                continue
            m = sub[col].to_numpy(dtype=float)
            valid = ~np.isnan(m)
            if valid.sum() < 4:
                print(f"  {'—':>14}", end="")
                continue
            r, _ = pearsonr(h[valid], m[valid])
            print(f"  {r:>14.3f}", end="")
        print()

    return results


if __name__ == "__main__":
    report("carrabin", ["Mean", "NoisyCounting", "NEF"])
    report("yoo",      ["Mean", "LeakyIntegrator", "PrimacyRecency", "NEF"])
