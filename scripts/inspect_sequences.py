"""
scripts/inspect_sequences.py
=============================
Diagnostic figure: math models + optional NEF × 4 panels
(binary + continuous × RMSE + |Δresponse|).

Usage:
    python scripts/inspect_sequences.py
    python scripts/inspect_sequences.py --alpha_0 1.0 --rl_lambda 0.5
    python scripts/inspect_sequences.py --gamma 0.9 --eps_p 0.3 --eps_r 0.7
    python scripts/inspect_sequences.py --skip_nef
    python scripts/inspect_sequences.py --force_nef
    python scripts/inspect_sequences.py --nef_cache figures/inspect_nef_a1.0000_l0.5000.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting.model_params import MODEL_PARAMS
from models.counting_integrator import (
    fast_decode as fast_decode_counting,
    load_activities as load_counting_activities,
)
from models.math_models import _run_primacy_recency
from models.NEF import PARAM_DEFAULTS, _pretrain, _simulate_trial
from utils.binary_transform import (
    apply_binary_transform,
    nef_obs_values,
    nef_response_to_model_scale,
)
from utils.paths import FIGURES_DIR
from utils.plot_style import get_palette


# ── Agent simulations ─────────────────────────────────────────────────────────
# Scale conventions (match NEF.run, run_nef_sequences.py, test_sequences.py):
#   continuous — stimulus [0, 100]; NEF I/O internally [-1, 1];
#                stored/compared on [0, 1] via denormalise_continuous_response
#   binary     — inputs {-1, +1}; responses on [-1, 1] after Laplace smoothing;
#                GT = true_p * 2 - 1


def _ground_truth(task: str, true_mean: float, true_p: float) -> float:
    if task == "continuous":
        return float(true_mean) / 100.0
    return float(true_p) * 2.0 - 1.0


# gt_mode='true'         -> constant ground truth per trial (_ground_truth above)
# gt_mode='running_mean' -> per-observation moving target = the running sample
#   mean of the observed stimulus stream so far. Note _bayes_responses (below)
#   is EXACTLY this: the incremental-mean update running += (obs-running)/n,
#   starting from any prior at n=1, fully overwrites the prior and reduces to
#   the plain running mean of the actual observations -- no separate tracker
#   needed, we just reuse _bayes_responses as the moving-target trajectory.
GT_MODES = ("true", "running_mean")


def _obs_norm(value, task: str) -> float:
    if task == "continuous":
        return value / 100.0
    return float(value)


def _clip_response(task: str, x: float) -> float:
    if task == "continuous":
        return float(np.clip(x, 0.0, 1.0))
    return float(np.clip(x, -1.0, 1.0))


def _bayes_responses(values, task):
    if task == "continuous":
        resps, running = [], 0.5
        for n, v in enumerate(values, 1):
            running += (_obs_norm(v, task) - running) / n
            resps.append(_clip_response(task, running))
        return resps
    resps = []
    for n in range(1, len(values) + 1):
        resps.append(_clip_response(task, float(np.mean(values[:n]))))
    return resps


def _rl_responses(values, task, alpha_0=1.0, lambda_=0.5):
    running = 0.5 if task == "continuous" else 0.0
    resps = []
    for n, v in enumerate(values, 1):
        alpha = alpha_0 / (n ** lambda_)
        obs_n = _obs_norm(v, task)
        running = _clip_response(task, running + alpha * (obs_n - running))
        resps.append(running)
    return resps


def _li_responses(values, task, gamma=0.9):
    resps, running = [], 0.0
    for v in values:
        obs_n = _obs_norm(v, task)
        running = gamma * running + (1.0 - gamma) * obs_n
        resps.append(_clip_response(task, running))
    return resps


def _pr_responses(values, task, eps_p=0.5, eps_r=0.5, eta=0.01):
    obs = [_obs_norm(v, task) for v in values]
    resps = []
    for n in range(1, len(obs) + 1):
        arr = np.asarray(obs[:n], dtype=float)
        resps.append(
            _clip_response(
                task,
                float(
                    _run_primacy_recency(
                        {"eps_p": eps_p, "eps_r": eps_r, "eta": eta},
                        arr,
                        n,
                        0,
                    )
                ),
            )
        )
    return resps


def _trial_metrics(vals, task, tm, tp, resp_fn):
    gt = _ground_truth(task, tm, tp)
    resp = resp_fn(vals, tm, tp)
    rows = []
    prev = None
    for obs, r in zip(range(1, len(resp) + 1), resp):
        rows.append(
            {
                "observation": int(obs),
                "err": abs(r - gt),
                "delta": abs(r - prev) if prev is not None else np.nan,
            }
        )
        prev = r
    return rows


def _metrics_from_responses(seq_df, task, response_df, gt_mode="true"):
    """Build err/delta metrics from NEF responses (post apply_binary_transform).

    gt_mode='true': err = |response - true_mean/true_p| (constant per trial).
    gt_mode='running_mean': err = |response - running_mean(observations so far)|
      (moving target, recomputed per observation -- see GT_MODES note above).
      For binary, the running-mean trajectory is passed through the SAME
      apply_binary_transform as the agent responses -- otherwise Bayes
      (which literally IS the running mean) would show a spurious nonzero
      error against an untransformed target, unlike continuous where Bayes
      correctly flatlines at zero.
    """
    dataset = f"task_{task}"
    rows = []
    for trial in sorted(seq_df["trial"].unique()):
        g = seq_df[seq_df["trial"] == trial].sort_values("observation")
        nef_t = response_df[response_df["trial"] == trial].sort_values("observation")
        tm = float(g["true_mean"].iloc[0])
        tp = float(g["true_p"].iloc[0]) if task == "binary" else float("nan")
        if gt_mode == "running_mean":
            gt_traj = _bayes_responses(g["value"].tolist(), task)
            if task == "binary":
                gt_df = pd.DataFrame({"observation": list(range(len(gt_traj))),
                                     "response": gt_traj})
                gt_traj = apply_binary_transform(gt_df, dataset)["response"].tolist()
        else:
            gt_const = _ground_truth(task, tm, tp)
        prev = None
        for i, (_, nrow) in enumerate(nef_t.iterrows()):
            r = float(nrow["response"])
            gt = gt_traj[i] if gt_mode == "running_mean" else gt_const
            rows.append(
                {
                    "observation": int(nrow["observation"]),
                    "err": abs(r - gt),
                    "delta": abs(r - prev) if prev is not None else np.nan,
                }
            )
            prev = r
    return pd.DataFrame(rows)


def simulate_nef_task(
    seq_df: pd.DataFrame,
    task: str,
    alpha_0: float,
    lambda_: float,
    *,
    n_neurons: int | None = None,
    n_neurons_counting: int | None = None,
    show_progress: bool = True,
    gt_mode: str = "true",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run NEF on one task's sequences; return (metrics_df, response_df).

    Output handling matches models.NEF.run:
    continuous stimuli → [-1, 1] for NEF, responses back to [0, 1];
    binary inputs stay {-1, +1}; responses pass through apply_binary_transform
    (Laplace smoothing for task_binary only).
    """
    dataset = f"task_{task}"
    fixed = MODEL_PARAMS[dataset]["NEF"]["fixed"]
    params = {
        **PARAM_DEFAULTS,
        **fixed,
        "model_type": "NEF",
        "dataset": dataset,
        "pid": 0,
        "alpha_0": alpha_0,
        "lambda_": lambda_,
    }
    if n_neurons is not None:
        params["n_neurons"] = n_neurons
    if n_neurons_counting is not None:
        params["n_neurons_counting"] = n_neurons_counting
    print(
        f"[nef] {task}: n_neurons={params['n_neurons']}, "
        f"n_neurons_counting={params['n_neurons_counting']}, "
        f"radius_c={params['radius_c']}"
    )

    try:
        activity_map = load_counting_activities(
            n_neurons=int(params["n_neurons"]),
            n_neurons_counting=int(params["n_neurons_counting"]),
            dataset=dataset,
        )
    except FileNotFoundError:
        activity_map = None

    trials = sorted(seq_df["trial"].unique())
    iterator = trials
    if show_progress:
        iterator = tqdm(trials, desc=f"NEF {task}", unit="trial")

    rows: list[dict] = []
    for trial in iterator:
        trial_data = seq_df[seq_df["trial"] == trial].sort_values("observation")
        obs_values = nef_obs_values(
            trial_data["value"].to_numpy(dtype=float), dataset
        )

        p = {**params, "seed": int(trial)}
        if activity_map is not None:
            activity = activity_map.get(int(trial))
            if activity is not None:
                decoders = fast_decode_counting(
                    activity,
                    alpha_0=alpha_0,
                    lambda_=lambda_,
                )
            else:
                decoders = _pretrain({**p, "base_seed": int(trial)})
        else:
            decoders = _pretrain({**p, "base_seed": int(trial)})

        responses = _simulate_trial(obs_values, p, decoders)
        for i, (_, row) in enumerate(trial_data.iterrows()):
            rows.append(
                {
                    "trial": int(trial),
                    "observation": int(row["observation"]),
                    "response": nef_response_to_model_scale(
                        float(responses[i]), dataset
                    ),
                }
            )

    response_df = apply_binary_transform(pd.DataFrame(rows), dataset)
    metrics_df = _metrics_from_responses(seq_df, task, response_df, gt_mode=gt_mode)
    return metrics_df, response_df


def default_nef_cache_path(
    alpha_0: float, lambda_: float,
    n_neurons: int | None = None,
    n_neurons_counting: int | None = None,
    gt_mode: str = "true",
) -> Path:
    suffix = ""
    if n_neurons is not None:
        suffix += f"_n{n_neurons}"
    if n_neurons_counting is not None:
        suffix += f"_nc{n_neurons_counting}"
    if gt_mode != "true":
        suffix += f"_{gt_mode}"
    return FIGURES_DIR / f"inspect_nef_a{alpha_0:.4f}_l{lambda_:.4f}{suffix}.pkl"


def load_nef_cache(
    cache_path: Path,
    alpha_0: float,
    lambda_: float,
) -> dict[str, pd.DataFrame] | None:
    if not cache_path.exists():
        return None
    payload = pd.read_pickle(cache_path)
    if not np.isclose(payload.get("alpha_0"), alpha_0) or not np.isclose(
        payload.get("lambda_"), lambda_
    ):
        print(
            f"[nef] Cache params mismatch in {cache_path} "
            f"(cached α={payload.get('alpha_0')} λ={payload.get('lambda_')})"
        )
        return None
    print(f"[nef] Loaded cache: {cache_path}")
    return payload["metrics"]


def save_nef_cache(
    cache_path: Path,
    alpha_0: float,
    lambda_: float,
    metrics: dict[str, pd.DataFrame],
    responses: dict[str, pd.DataFrame],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {
            "alpha_0": alpha_0,
            "lambda_": lambda_,
            "metrics": metrics,
            "responses": responses,
        },
        cache_path,
    )
    print(f"[nef] Saved cache: {cache_path}")


def load_or_simulate_nef(
    seq_dir: Path,
    alpha_0: float,
    lambda_: float,
    *,
    cache_path: Path,
    force: bool = False,
    show_progress: bool = True,
    n_neurons: int | None = None,
    n_neurons_counting: int | None = None,
    gt_mode: str = "true",
) -> pd.DataFrame | None:
    if not force:
        cached = load_nef_cache(cache_path, alpha_0, lambda_)
        if cached is not None:
            return cached

    metrics: dict[str, pd.DataFrame] = {}
    responses: dict[str, pd.DataFrame] = {}
    for task in ("binary", "continuous"):
        pkl = seq_dir / f"{task}_sequences.pkl"
        if not pkl.exists():
            print(f"[nef skip] {pkl}")
            continue
        seq_df = pd.read_pickle(pkl)
        metrics[task], responses[task] = simulate_nef_task(
            seq_df,
            task,
            alpha_0,
            lambda_,
            n_neurons=n_neurons,
            n_neurons_counting=n_neurons_counting,
            show_progress=show_progress,
            gt_mode=gt_mode,
        )

    if not metrics:
        return None

    save_nef_cache(cache_path, alpha_0, lambda_, metrics, responses)
    return metrics


def run_agents(seq_df, task, alpha_0, rl_lambda, gamma, eps_p, eps_r, gt_mode="true"):
    from utils.binary_transform import apply_binary_transform

    agents = {
        "Bayes": lambda vals, tm, tp: _bayes_responses(vals, task),
        f"RL(α={alpha_0},λ={rl_lambda})": lambda vals, tm, tp: _rl_responses(
            vals, task, alpha_0, rl_lambda
        ),
        f"LI(γ={gamma})": lambda vals, tm, tp: _li_responses(vals, task, gamma),
        f"PR(εp={eps_p},εr={eps_r})": lambda vals, tm, tp: _pr_responses(
            vals, task, eps_p, eps_r
        ),
    }
    dataset = f"task_{task}"
    results = {}
    for name, fn in agents.items():
        rows = []
        for tid in seq_df["trial"].unique():
            g = seq_df[seq_df["trial"] == tid].sort_values("observation")
            vals = g["value"].tolist()
            tm = float(g["true_mean"].iloc[0])
            tp = float(g["true_p"].iloc[0]) if task == "binary" else float("nan")
            if gt_mode == "running_mean":
                gt_traj = _bayes_responses(vals, task)
                if task == "binary":
                    gt_df = pd.DataFrame({"observation": list(range(len(gt_traj))),
                                         "response": gt_traj})
                    gt_traj = apply_binary_transform(gt_df, dataset)["response"].tolist()
            else:
                gt_const = _ground_truth(task, tm, tp)
            resp = fn(vals, tm, tp)
            # Build a minimal DataFrame so apply_binary_transform can operate
            resp_df = pd.DataFrame({
                "observation": list(range(len(resp))),
                "response": resp,
            })
            resp_df = apply_binary_transform(resp_df, dataset)
            prev = None
            for obs_i, r in enumerate(resp_df["response"].tolist()):
                gt = gt_traj[obs_i] if gt_mode == "running_mean" else gt_const
                rows.append({
                    "observation": obs_i + 1,
                    "err": abs(r - gt),
                    "delta": abs(r - prev) if prev is not None else np.nan,
                })
                prev = r
        results[name] = pd.DataFrame(rows)
    return results


# ── Human-readable inspection CSV ──────────────────────────────────────────────
def build_inspection_csv(seq_dir: Path, out_path: Path) -> pd.DataFrame:
    """Human-readable, ONE-ROW-PER-TRIAL CSV. Each column obs_1..obs_N is one
    observation's raw value (so a whole trial's stimulus sequence reads left
    to right on a single line), followed by true_mean/true_p/true_std and
    their achieved (obs_mean/obs_p/obs_std) counterparts. A blank '|'
    column sits right after the prefix (obs_1..obs_prefix_length) to make
    the prefix/suffix boundary visually obvious in a spreadsheet -- see
    generate_sequences_momentmatch.py's module docstring for what that
    split means and why it exists.

    Simpler, wide-format replacement for an earlier long-format (one row
    per observation, trial-level fields repeated on every row) version --
    moved to one row per TRIAL instead, since that's the natural unit for a
    human scanning by eye: an entire trial's sequence and its
    target-vs-achieved summary now sit on one line, not spread across 15.

    Reads directly from {task}_sequences.json -- no model/NEF dependency at
    all, so this is fast and always available regardless of --skip_nef.
    """
    rows = []
    console_summary = []

    for task in ("continuous", "binary"):
        json_path = seq_dir / f"{task}_sequences.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            trials = json.load(f)

        # -- Global prefix-uniqueness check across all qids in this task --
        # (the actual bug this whole redesign fixed: two different qids
        # ending up with an identical realized prefix by chance -- see
        # generate_sequences_momentmatch.py's module docstring)
        prefix_by_qid: dict = {}
        for t in trials:
            pl = t["prefix_length"]
            prefix_by_qid.setdefault(t["qid"], set()).add(tuple(t["values"][:pl]))
        all_prefixes = [p for prefs in prefix_by_qid.values() for p in prefs]
        n_distinct = len(set(all_prefixes))
        n_qids = len(prefix_by_qid)
        one_prefix_per_qid = all(len(v) == 1 for v in prefix_by_qid.values())
        prefix_ok = (n_distinct == n_qids) and one_prefix_per_qid
        console_summary.append(
            f"[{task}] prefix uniqueness: {n_distinct}/{n_qids} distinct "
            f"({'OK' if prefix_ok else 'COLLISION DETECTED'})"
        )

        # -- iti_condition balance per qid (should be within 1 of even split) --
        iti_by_qid: dict = {}
        for t in trials:
            iti_by_qid.setdefault(t["qid"], []).append(t.get("iti_condition"))
        imbalanced = [q for q, conds in iti_by_qid.items()
                      if abs(conds.count("control") - conds.count("distract")) > 1]
        console_summary.append(
            f"[{task}] iti_condition balance: "
            + (f"{len(imbalanced)} qid(s) off by >1 ({imbalanced})"
               if imbalanced else "OK (every qid within 1 of an even control/distract split)")
        )

        n_quota_bad = 0
        for t in trials:
            pl = t["prefix_length"]
            vals = t["values"]
            n = len(vals)

            row = {"task": task, "trial": t["trial"], "qid": t["qid"]}
            for i, v in enumerate(vals, start=1):
                row[f"obs_{i}"] = v
                if i == pl:
                    row["|"] = ""  # separator column, right after the prefix

            target_mean = t.get("true_mean")
            target_std  = t.get("true_std")
            target_p    = t.get("true_p")

            if task == "continuous":
                obs_mean, obs_std, obs_p = float(np.mean(vals)), float(np.std(vals)), None
            else:
                obs_p, obs_mean, obs_std = float(np.mean([v == 1 for v in vals])), None, None
                achieved_blue = sum(1 for v in vals if v == 1)
                target_blue = round(target_p * n) if target_p is not None else None
                if target_blue is not None and achieved_blue != target_blue:
                    n_quota_bad += 1

            row.update({
                "true_mean": target_mean, "true_p": target_p, "true_std": target_std,
                "obs_mean": obs_mean, "obs_p": obs_p, "obs_std": obs_std,
            })
            rows.append(row)

        if task == "binary":
            console_summary.append(
                f"[{task}] exact-quota check: {n_quota_bad} trial(s) with "
                f"achieved blue count != round(true_p * seq_length) "
                f"({'OK' if n_quota_bad == 0 else 'MISMATCH'})"
            )

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[csv skip] no {{task}}_sequences.json files found in {seq_dir}")
        return df
    df = df.sort_values(["task", "trial"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df)} rows)")
    for line in console_summary:
        print(" ", line)
    return df


# ── Split-half λ-recovery reliability (comparable to test_sequences.pdf's
# panels E/L, computed on THIS figure's own --seq_dir sequences) ────────
def compute_split_half_reliability(seq_df, task, alpha_0=1.0, n_lambdas=50):
    """Split-half λ-recovery reliability: sweep n_lambdas different RL_lambda
    ground-truth parameterizations (lambda_ in [0.01, 0.99], alpha_0 fixed)
    across the SAME sequence set -- standing in for "many different
    participants, each with their own true decay rate", the same role
    test_sequences.py's own lambda sweep plays in its panels E/L (which
    already default to these exact production sequences via
    --seq_dir task/sequences, alpha_0=1.0 by CLI default -- matched here).
    Fits lambda separately on the first half vs second half of trials for
    each swept value and correlates the two halves across the swept
    population.

    Reuses fit_lambda_mid/split_half_lambda directly from test_sequences.py
    (not reimplemented) so this number means exactly the same thing as it
    does there -- packaged into THIS figure specifically so quota's
    reliability sits side by side with scripts/inspect_iid_sequences.py's
    i.i.d. reliability panel for direct comparison (see chat history for
    why that comparison matters -- it's what motivated adding this panel
    at all, after fixing a real prefix-collision bug in
    generate_sequences_iid.py revealed that i.i.d.'s split-half reliability
    had been artificially inflated by that bug).

    Returns (rel_df, r, p) where rel_df has one row per swept lambda value
    with columns [model_id, first, second] (dropna'd), r/p from pearsonr
    (nan if too few finite points or zero variance to correlate).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_sequences import fit_lambda_mid, split_half_lambda  # noqa: F401 (fit_lambda_mid re-exported for callers)

    dataset = f"task_{task}"
    rows = []
    for i, lam in enumerate(np.linspace(0.01, 0.99, n_lambdas)):
        model_id = f"RL_lambda[{i}]"
        for tid in seq_df["trial"].unique():
            g = seq_df[seq_df["trial"] == tid].sort_values("observation")
            vals = g["value"].tolist()
            resp = _rl_responses(vals, task, alpha_0, float(lam))
            resp_df = pd.DataFrame({"observation": list(range(len(resp))), "response": resp})
            resp_df = apply_binary_transform(resp_df, dataset)
            for obs_i, r in enumerate(resp_df["response"].tolist()):
                rows.append({
                    "model_id": model_id, "model_type": "RL_lambda",
                    "trial": int(tid), "observation": obs_i + 1, "response": float(r),
                })
    sweep_df = pd.DataFrame(rows)
    half_wide = split_half_lambda(sweep_df)
    rel = half_wide.dropna(subset=["first", "second"])
    if len(rel) >= 3 and rel["first"].std() > 1e-9:
        from scipy.stats import pearsonr
        r_val, p_val = pearsonr(rel["first"], rel["second"])
    else:
        r_val, p_val = float("nan"), float("nan")
    return rel, r_val, p_val


def _plot_reliability_panel(ax, rel, r, p, title, color):
    from utils.plot_style import pvalue_to_stars
    if not rel.empty:
        ax.scatter(rel["first"], rel["second"], s=14, alpha=0.6, color=color, zorder=3)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--", zorder=1)
    note = f"r={r:.2f}{pvalue_to_stars(p)}  (n={len(rel)})" if np.isfinite(r) else "insufficient data"
    ax.text(0.05, 0.95, note, transform=ax.transAxes, ha="left", va="top", fontsize=7,
           bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("First-half fitted \u03bb", fontsize=8)
    ax.set_ylabel("Second-half fitted \u03bb", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_panel(ax, agent_data, metric, title, ylabel, colors):
    for (name, df), color in zip(agent_data.items(), colors):
        curve = df.dropna(subset=[metric]).groupby("observation")[metric].mean()
        ax.plot(curve.index, curve.values, color=color, lw=1.8, label=name)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Observation", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(
    seq_dir,
    alpha_0,
    rl_lambda,
    gamma,
    eps_p,
    eps_r,
    out_path,
    *,
    skip_nef: bool = False,
    force_nef: bool = False,
    nef_cache: Path | None = None,
    n_neurons: int | None = None,
    n_neurons_counting: int | None = None,
    gt_mode: str = "running_mean",
):
    assert gt_mode in GT_MODES, f"gt_mode must be one of {GT_MODES}"
    seq_dir = Path(seq_dir)
    cache_path = nef_cache or default_nef_cache_path(alpha_0, rl_lambda, gt_mode=gt_mode)
    nef_metrics = None
    if not skip_nef:
        nef_metrics = load_or_simulate_nef(
            seq_dir,
            alpha_0,
            rl_lambda,
            cache_path=cache_path,
            force=force_nef,
            n_neurons=n_neurons,
            n_neurons_counting=n_neurons_counting,
            gt_mode=gt_mode,
        )

    n_models = 5 if nef_metrics else 4
    colors = get_palette(n_models)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)

    gt_label = "RMSE vs running mean" if gt_mode == "running_mean" else "RMSE vs true param"
    reliability_color = get_palette(3)[2]

    for row, task in enumerate(["binary", "continuous"]):
        pkl = seq_dir / f"{task}_sequences.pkl"
        if not pkl.exists():
            print(f"[skip] {pkl}")
            continue
        seq_df = pd.read_pickle(pkl)
        agent_data = run_agents(seq_df, task, alpha_0, rl_lambda, gamma, eps_p, eps_r,
                                gt_mode=gt_mode)
        if nef_metrics and task in nef_metrics:
            agent_data[f"NEF(α={alpha_0},λ={rl_lambda})"] = nef_metrics[task]

        with open(seq_dir / f"{task}_sequences.json") as f:
            prefix = int(json.load(f)[0]["prefix_length"])
        label = task.capitalize()

        _plot_panel(
            axes[row, 0],
            agent_data,
            "err",
            f"{label} — RMSE (prefix={prefix})",
            gt_label,
            colors,
        )
        _plot_panel(
            axes[row, 1],
            agent_data,
            "delta",
            f"{label} — |Δresponse| (prefix={prefix})",
            "Mean |Δresponse|",
            colors,
        )

        for ax in axes[row, :2]:
            ax.axvline(prefix + 0.5, color="#999", lw=0.8, ls="--", alpha=0.6)

        # -- Column 2: split-half λ-recovery reliability, comparable to
        # test_sequences.pdf's panels E/L, computed on THESE sequences --
        # see compute_split_half_reliability's docstring for why this was
        # added (comparing quota vs i.i.d. split-half reliability directly)
        rel, r_val, p_val = compute_split_half_reliability(seq_df, task)
        if np.isfinite(r_val):
            print(f"[reliability] {task}: split-half λ r={r_val:.3f} p={p_val:.4g} (n={len(rel)})")
        else:
            print(f"[reliability] {task}: insufficient data")
        _plot_reliability_panel(
            axes[row, 2], rel, r_val, p_val,
            f"{label} — split-half λ reliability (quota)",
            reliability_color,
        )

    fig.suptitle(
        f"Sequence diagnostics ({gt_mode})  |  RL/NEF α={alpha_0} λ={rl_lambda}  "
        f"LI γ={gamma}  PR εp={eps_p} εr={eps_r}",
        fontsize=10,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_dir", default="task/sequences")
    p.add_argument("--alpha_0", type=float, default=1.0)
    p.add_argument("--rl_lambda", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.9, help="LeakyIntegrator γ")
    p.add_argument("--eps_p", type=float, default=0.5, help="PrimacyRecency ε_p")
    p.add_argument("--eps_r", type=float, default=0.5, help="PrimacyRecency ε_r")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--nef_cache",
        default=None,
        help="Path to save/load NEF metrics (default: figures/inspect_nef_a{A}_l{L}.pkl)",
    )
    p.add_argument(
        "--skip_nef",
        action="store_true",
        help="Skip NEF simulation and plot math models only",
    )
    p.add_argument(
        "--force_nef",
        action="store_true",
        help="Re-run NEF simulation even if cache exists",
    )
    p.add_argument(
        "--nef_only",
        action="store_true",
        help="Run/save NEF simulation only; skip figure generation",
    )
    p.add_argument(
        "--n_neurons", type=int, default=None,
        help="Override n_neurons for NEF (default: from MODEL_PARAMS)",
    )
    p.add_argument(
        "--n_neurons_counting", type=int, default=None,
        help="Override n_neurons_counting for NEF (default: from MODEL_PARAMS)",
    )
    p.add_argument(
        "--gt_mode", choices=list(GT_MODES), default="running_mean",
        help="Ground truth for RMSE panels: 'running_mean' (default) or 'true'",
    )
    p.add_argument(
        "--csv_out", default=None,
        help="Path for the human-readable inspection CSV (default: "
             "figures/inspect_sequences.csv)",
    )
    p.add_argument(
        "--skip_csv", action="store_true",
        help="Skip building the inspection CSV (built by default, alongside the figure)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cache_path = Path(args.nef_cache) if args.nef_cache else default_nef_cache_path(
        args.alpha_0, args.rl_lambda,
        n_neurons=args.n_neurons,
        n_neurons_counting=args.n_neurons_counting,
        gt_mode=args.gt_mode,
    )

    if args.nef_only:
        load_or_simulate_nef(
            args.seq_dir,
            args.alpha_0,
            args.rl_lambda,
            cache_path=cache_path,
            force=args.force_nef,
            n_neurons=args.n_neurons,
            n_neurons_counting=args.n_neurons_counting,
            gt_mode=args.gt_mode,
        )
        print("JOB_COMPLETE")
        raise SystemExit(0)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_csv:
        csv_out = Path(args.csv_out) if args.csv_out else FIGURES_DIR / "inspect_sequences.csv"
        build_inspection_csv(Path(args.seq_dir), csv_out)

    out = Path(args.out) if args.out else FIGURES_DIR / "inspect_sequences.pdf"
    make_figure(
        args.seq_dir,
        args.alpha_0,
        args.rl_lambda,
        args.gamma,
        args.eps_p,
        args.eps_r,
        out,
        skip_nef=args.skip_nef,
        force_nef=args.force_nef,
        nef_cache=cache_path,
        n_neurons=args.n_neurons,
        n_neurons_counting=args.n_neurons_counting,
        gt_mode=args.gt_mode,
    )
    print("JOB_COMPLETE")
