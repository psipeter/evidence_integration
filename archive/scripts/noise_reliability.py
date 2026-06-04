#!/usr/bin/env python3
"""
Estimate empirical response noise from repeated carrabin sequences (qid)
and bootstrap reliability of noise estimates vs number of repetitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_PATH = Path("data/carrabin.pkl")
K_VALUES = list(range(1, 6))
MIN_REPS_BY_K = {1: 8, 2: 8, 3: 8, 4: 5, 5: 3}
N_BOOT = 500
N_REPS_VALUES = [2, 3, 4, 5, 6, 8]
SPEARMAN_TARGET = 0.90
PREFIX_COUNTS = [8, 16, 32]


def load_carrabin() -> pd.DataFrame:
    df = pd.read_pickle(DATA_PATH)
    df = df.copy()
    df["qid"] = df["qid"].astype(str)
    df["prefix_len"] = df["qid"].str.len()
    return df


def step1_empirical_noise(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Step 1 — Empirical noise vs prefix length")
    print("=" * 60)
    for k in K_VALUES:
        sub = df.loc[df["prefix_len"] == k]
        if sub.empty:
            print(f"k={k}: no groups")
            continue
        noise = sub.groupby(["pid", "qid"], sort=False)["response"].std(ddof=1)
        noise = noise.dropna()
        print(
            f"k={k}: mean ± SD empirical noise = "
            f"{noise.mean():.4f} ± {noise.std():.4f}  (n_groups={len(noise)})"
        )


def build_group_responses(df: pd.DataFrame, k: int, min_reps: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Return list of response arrays and ground-truth std per (pid, qid) group."""
    sub = df.loc[df["prefix_len"] == k]
    grouped = sub.groupby(["pid", "qid"], sort=False)["response"]
    counts = grouped.size()
    valid_keys = counts[counts >= min_reps].index

    responses: list[np.ndarray] = []
    ground_truth: list[float] = []
    for key in valid_keys:
        vals = grouped.get_group(key).to_numpy(dtype=float)
        responses.append(vals)
        ground_truth.append(np.std(vals, ddof=1) if len(vals) > 1 else np.nan)

    return responses, np.asarray(ground_truth, dtype=float)


def bootstrap_spearman_batch(
    responses: list[np.ndarray],
    ground_truth: np.ndarray,
    n_reps: int,
    rng: np.random.Generator,
    n_boot: int,
) -> np.ndarray:
    """Bootstrap subset stds; vectorized over draws within each group."""
    n_valid = np.isfinite(ground_truth)
    gt = ground_truth[n_valid]
    resps = [responses[i] for i in range(len(responses)) if n_valid[i]]
    n_groups = len(resps)

    boot_stds = np.empty((n_boot, n_groups), dtype=np.float64)
    for i, r in enumerate(resps):
        n = len(r)
        if n < n_reps:
            boot_stds[:, i] = np.nan
            continue
        idx = np.stack(
            [rng.permutation(n)[:n_reps] for _ in range(n_boot)], axis=0
        )
        boot_stds[:, i] = r[idx].std(axis=1, ddof=1)

    rhos = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        mask = np.isfinite(boot_stds[b]) & np.isfinite(gt)
        if mask.sum() < 3:
            rhos[b] = np.nan
            continue
        rho, _ = stats.spearmanr(gt[mask], boot_stds[b, mask])
        rhos[b] = float(rho) if not np.isnan(rho) else np.nan
    return rhos


def step2_bootstrap_reliability(df: pd.DataFrame) -> dict[int, dict[int, np.ndarray]]:
    print("\n" + "=" * 60)
    print("Step 2 — Bootstrap reliability (Spearman r vs N_reps)")
    print("=" * 60)
    print(
        f"{'k':>4} {'N_reps':>6} {'median_r':>10} "
        f"{'p10':>8} {'p90':>8} {'n_groups':>10}"
    )
    print("-" * 60)

    results: dict[int, dict[int, np.ndarray]] = {k: {} for k in K_VALUES}

    for k in K_VALUES:
        min_reps = MIN_REPS_BY_K[k]
        responses, gt = build_group_responses(df, k, min_reps)
        if len(responses) == 0:
            print(f"k={k}: no qualifying groups")
            continue

        results[k] = {}
        for n_reps in N_REPS_VALUES:
            rng = np.random.default_rng(42 + k * 1000 + n_reps)
            rhos = bootstrap_spearman_batch(responses, gt, n_reps, rng, N_BOOT)
            valid = rhos[np.isfinite(rhos)]
            if valid.size == 0:
                results[k][n_reps] = np.full(N_BOOT, np.nan)
                continue
            results[k][n_reps] = rhos

            med = np.nanmedian(valid)
            p10, p90 = np.nanpercentile(valid, [10, 90])
            print(
                f"{k:4d}  {n_reps:6d}  {med:10.3f}  "
                f"{p10:8.3f}  {p90:8.3f}  {len(responses):10d}"
            )

    return results


def min_n_reps_for_target(rhos_by_n: dict[int, np.ndarray]) -> int | None:
    for n_reps in N_REPS_VALUES:
        med = np.nanmedian(rhos_by_n[n_reps])
        if np.isfinite(med) and med >= SPEARMAN_TARGET:
            return n_reps
    return None


def step3_trial_implications(results: dict[int, dict[int, np.ndarray]]) -> None:
    print("\n" + "=" * 60)
    print("Step 3 — Trial-count implications (median r >= 0.90)")
    print("=" * 60)
    header = (
        f"{'k':>4} {'N_reps_min':>10} "
        + " ".join(f"{'P='+str(p):>10}" for p in PREFIX_COUNTS)
    )
    print(header)
    print("-" * len(header))

    for k in K_VALUES:
        if not results[k]:
            print(f"{k:4d}  {'—':>10}  " + "  ".join(f"{'—':>10}" for _ in PREFIX_COUNTS))
            continue
        min_n = min_n_reps_for_target(results[k])
        min_str = str(min_n) if min_n is not None else "—"
        trial_cols = []
        for p in PREFIX_COUNTS:
            if min_n is None:
                trial_cols.append("—")
            else:
                trial_cols.append(str(min_n * p))
        print(f"{k:4d}  {min_str:>10}  " + "  ".join(f"{t:>10}" for t in trial_cols))


def main() -> None:
    df = load_carrabin()
    print(f"Loaded {DATA_PATH}: {len(df)} rows, {df['pid'].nunique()} pids, "
          f"{df['trial'].nunique()} trials")

    step1_empirical_noise(df)
    results = step2_bootstrap_reliability(df)
    step3_trial_implications(results)


if __name__ == "__main__":
    main()
