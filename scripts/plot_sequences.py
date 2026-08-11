"""
scripts/plot_sequences.py
==========================
Consolidates the two former inspect_sequences.py / inspect_iid_sequences.py
scripts into one file with two branches, both reading exclusively from
task_backend's real, deployed sequence pool (sequences_numbers.json /
sequences_colors.json) -- task/ 's old file layout (a single shared
reference file PLUS a separately-numbered 200-file pool directory) and its
now-closed iid-vs-quota-vs-momentmatch generation-method debate are BOTH
retired; see docs/HISTORY.md's "Sequence design: open questions" section
and git history for
that investigation if it's ever relevant again. Old file names
(inspect_sequences.py, inspect_iid_sequences.py) didn't reflect what either
script actually did once task_backend consolidated everything down to one
generation method and one pool per task -- this file is named for what it
does (plot sequences), with the two branches named for the AXIS each one
varies (matching the two output PDFs' own names), not for the mechanism
underneath (their earlier names -- running_agents/meanagent -- described
HOW each figure was built, not WHAT question it answers).

Internal `task` argument values remain "continuous"/"binary" (NOT renamed
to "numbers"/"colors") deliberately -- MODEL_PARAMS, NEF.py, and
counting_integrator.py's activity-file naming all still key on
task_continuous/task_binary throughout the fitting pipeline; renaming that
deeper vocabulary is a separate decision this file does not make. Only the
FILE name used to find task_backend's sequences maps through TASK_FILE.

Branch 1: across_models -- "how well does each candidate MODEL recover
  the ground truth, given these sequences?" Fixes the sequences (aggregated
  across all 200 real pool members by default), varies the MODEL (Bayes,
  RL_lambda, LeakyIntegrator, PrimacyRecency, optionally NEF) -- i.e. one
  curve per model, aggregated across pids. Trusts
  task_backend/generate_sequences.py's own verify_numbers_trials/
  verify_colors_trials asserts to have already caught any prefix-
  collision/quota-mismatch problem at generation time -- this file only
  plots, it does not re-audit the pool's own correctness (an earlier
  version bundled a human-readable audit CSV into this branch; dropped
  deliberately, not an oversight -- see git history if that's ever needed
  again).
    python scripts/plot_sequences.py across_models
    python scripts/plot_sequences.py across_models --alpha_0 1.0 --rl_lambda 0.5
    python scripts/plot_sequences.py across_models --skip_nef
  Output: figures/inspect_pool_sequences_across_models.pdf

Branch 2: across_pids -- "how much does natural sequence-to-sequence
  variance across real participants itself contribute to noise?" Fixes the
  MODEL (Mean by default, or RL_lambda), varies the PID (each of the 200
  real pool members individually, standing in for a real participant) --
  shows one thin line per pid plus the mean+95% CI, which is the whole
  point of this branch (across_models' pooled mode only ever shows the
  aggregate, never the spread). Ground truth is true_mean/true_p (not
  running_mean) -- a deliberate, historical choice for this specific
  investigation (see git history), distinct from across_models' own
  --gt_mode toggle.
    python scripts/plot_sequences.py across_pids
    python scripts/plot_sequences.py across_pids --agent RL_lambda --alpha_0 1.0 --rl_lambda 0.3
    python scripts/plot_sequences.py across_pids --n_pool 50
  Output: figures/inspect_pool_sequences_across_pids.pdf
          data/runs/plot_sequences/across_pids_{raw,fits}_{n}p.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as scipy_curve_fit
from scipy.stats import pearsonr, spearmanr
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
from utils.paths import FIGURES_DIR, resolve_run_folder
from utils.plot_style import get_palette, pvalue_to_stars

sys.path.insert(0, str(Path(__file__).resolve().parent))  # (reserved for any future script-local imports)


# ── Split-half power-law fitting utilities (shared by both branches) ───────
# Inlined from the now-archived scripts/test_sequences.py (see
# archive/archive_readme.md) rather than imported cross-file -- that
# script's OWN main()/figure pipeline read a STALE task/ pkl schema
# (trial_type/std_condition columns that don't exist in ANY current
# schema, task/'s or task_backend's) and is genuinely superseded by this
# file's own across_models/across_pids figures, but these three specific
# helpers are pure DataFrame utilities (model_id/model_type/trial/
# observation/response columns only) with no dependency on that stale
# schema at all -- confirmed directly before inlining, not assumed.
def fit_lambda_mid(g, min_obs=2):
    """Fit power-law A*n^(-lambda) to the |Δresponse| curve for one
    model_id. Returns (lambda, p) where p is from a Spearman correlation
    of the fitted curve (a rough significance proxy, not a formal test)."""
    rows = []
    for trial, tg in g.groupby("trial"):
        tg = tg.sort_values("observation")
        delta = tg["response"].diff().abs()
        for obs, d in zip(tg["observation"], delta):
            if pd.notna(d) and obs >= min_obs:
                rows.append({"observation": int(obs), "delta": float(d)})
    if not rows:
        return np.nan, np.nan
    ddf = pd.DataFrame(rows)
    curve = ddf.groupby("observation")["delta"].mean().sort_index()
    if len(curve) < 3:
        return np.nan, np.nan
    n_arr, y_arr = curve.index.values.astype(float), curve.values.astype(float)
    try:
        popt, _ = scipy_curve_fit(lambda n, A, lam: A * n ** (-lam), n_arr, y_arr,
                                  p0=[0.1, 0.5], bounds=([0, 0], [2, 2]), maxfev=2000)
        lam = float(popt[1])
    except Exception:
        return np.nan, np.nan
    _, p = spearmanr(n_arr, y_arr)
    return lam, float(p)


def compute_abs_delta(df):
    """Per-(model_id, trial, observation): |Δresponse|."""
    rows = []
    for (mid, trial), g in df.groupby(["model_id", "trial"]):
        g = g.sort_values("observation").copy()
        g["delta"] = g["response"].diff().abs()
        rows.append(g[["model_id", "model_type", "trial", "observation", "delta"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).dropna(subset=["delta"])


def split_half_lambda(df):
    """Split-half fitted lambda per model_id -> (first, second). Splits
    by ODD/EVEN trial index, not first-half/second-half -- a strict
    chronological split confounds genuine estimation noise (what split-
    half reliability is meant to measure) with any systematic drift over
    the sequence (e.g. simulated-agent behavior that itself depends on
    trial order); interleaving odd/even trials samples both halves from
    the same span, isolating noise from drift (see chat history)."""
    rows = []
    for mid, g in df.groupby("model_id"):
        trials = sorted(g["trial"].unique())
        halves = {"first": trials[0::2], "second": trials[1::2]}
        if min(len(halves["first"]), len(halves["second"])) < 3:
            continue
        for half, tset in halves.items():
            lam, _ = fit_lambda_mid(g[g["trial"].isin(tset)])
            if np.isfinite(lam):
                rows.append({"model_id": mid, "model_type": g["model_type"].iloc[0],
                            "half": half, "lambda_": lam})
    if not rows:
        return pd.DataFrame()
    wide = (pd.DataFrame(rows)
            .pivot_table(index=["model_id", "model_type"], columns="half", values="lambda_")
            .reset_index())
    wide.columns.name = None
    return wide.dropna(subset=["first", "second"])


# ── task_backend pool loading (shared by both branches) ────────────────────
TASK_FILE = {"continuous": "numbers", "binary": "colors"}


def _pool_path(pool_root, task: str) -> Path:
    return Path(pool_root) / f"sequences_{TASK_FILE[task]}.json"


def load_pool(pool_root, task: str) -> list:
    """Loads task_backend's sequences_{numbers,colors}.json -- a JSON array
    of 200 independent pool members, each a list of 32 trial dicts (qid,
    true_mean, true_std, true_p, values, prefix_length, iti_ms,
    iti_condition, trial)."""
    path = _pool_path(pool_root, task)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- pass --pool_root pointing at a checkout of "
            f"task_backend/ (see CLAUDE.md's task_backend section)."
        )
    with open(path) as f:
        return json.load(f)


def member_to_df(member: list) -> pd.DataFrame:
    """Explodes one pool member's list of trial dicts into a long-format
    per-observation DataFrame (trial, observation, value, true_mean,
    true_std, true_p, qid, prefix_length, iti_condition).

    true_mean/true_std/true_p come through as JSON null -> Python None for
    whichever task doesn't use them -- explicitly coerced to float('nan')
    here, NOT left as None: a plain DataFrame built from dicts containing a
    real None keeps that column as dtype=object, and float(None) raises
    TypeError where float(nan) would have silently succeeded, and every
    float(g["true_mean"].iloc[0])-style call below relies on that.
    """
    rows = []
    for t in member:
        tm = float("nan") if t["true_mean"] is None else t["true_mean"]
        ts = float("nan") if t["true_std"] is None else t["true_std"]
        tp = float("nan") if t["true_p"] is None else t["true_p"]
        for obs, v in enumerate(t["values"]):
            rows.append({
                "trial": t["trial"], "observation": obs, "value": v,
                "true_mean": tm, "true_std": ts, "true_p": tp, "qid": t["qid"],
                "prefix_length": t["prefix_length"],
                "iti_condition": t.get("iti_condition"),
            })
    return pd.DataFrame(rows)


def _ground_truth(task: str, true_mean: float, true_p: float) -> float:
    if task == "continuous":
        return float(true_mean) / 100.0
    return float(true_p) * 2.0 - 1.0


def _obs_norm(value, task: str) -> float:
    if task == "continuous":
        return value / 100.0
    return float(value)


def _clip_response(task: str, x: float) -> float:
    if task == "continuous":
        return float(np.clip(x, 0.0, 1.0))
    return float(np.clip(x, -1.0, 1.0))


def _bayes_responses(values, task):
    """The running mean of observed values -- called 'Bayes' in the
    across_models branch (optimal estimator of the fixed generative
    parameter under a uniform prior, once Laplace-smoothed for binary) and
    'Mean' in the across_pids branch (same computation, framed there as
    the reference 'no forgetting' agent) -- ONE function, two names for
    two contexts, not two independently-maintained implementations as the
    pre-consolidation scripts had."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Branch 1: across_models -- vary the MODEL, fix the sequences (pooled)
# ═══════════════════════════════════════════════════════════════════════════

GT_MODES = ("true", "running_mean")


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
                float(_run_primacy_recency({"eps_p": eps_p, "eps_r": eps_r, "eta": eta}, arr, n, 0)),
            )
        )
    return resps


def _metrics_from_responses(seq_df, task, response_df, gt_mode="true"):
    """Build err/delta metrics from NEF responses (post apply_binary_transform).
    gt_mode='true': err = |response - true_mean/true_p| (constant per trial).
    gt_mode='running_mean': err = |response - running_mean(observations so far)|.
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
                gt_df = pd.DataFrame({"observation": list(range(len(gt_traj))), "response": gt_traj})
                gt_traj = apply_binary_transform(gt_df, dataset)["response"].tolist()
        else:
            gt_const = _ground_truth(task, tm, tp)
        prev = None
        for i, (_, nrow) in enumerate(nef_t.iterrows()):
            r = float(nrow["response"])
            gt = gt_traj[i] if gt_mode == "running_mean" else gt_const
            rows.append({
                "observation": int(nrow["observation"]),
                "err": abs(r - gt),
                "delta": abs(r - prev) if prev is not None else np.nan,
            })
            prev = r
    return pd.DataFrame(rows)


def simulate_nef_task(seq_df, task, alpha_0, lambda_, *, n_neurons=None,
                       n_neurons_counting=None, show_progress=True, gt_mode="true"):
    """Run NEF on one task's sequences; return (metrics_df, response_df)."""
    dataset = f"task_{task}"
    fixed = MODEL_PARAMS[dataset]["NEF"]["fixed"]
    params = {**PARAM_DEFAULTS, **fixed, "model_type": "NEF", "dataset": dataset,
              "pid": 0, "alpha_0": alpha_0, "lambda_": lambda_}
    if n_neurons is not None:
        params["n_neurons"] = n_neurons
    if n_neurons_counting is not None:
        params["n_neurons_counting"] = n_neurons_counting
    print(f"[nef] {task}: n_neurons={params['n_neurons']}, "
          f"n_neurons_counting={params['n_neurons_counting']}, radius_c={params['radius_c']}")

    try:
        activity_map = load_counting_activities(
            n_neurons=int(params["n_neurons"]), n_neurons_counting=int(params["n_neurons_counting"]),
            dataset=dataset)
    except FileNotFoundError:
        activity_map = None

    trials = sorted(seq_df["trial"].unique())
    iterator = tqdm(trials, desc=f"NEF {task}", unit="trial") if show_progress else trials

    rows = []
    for trial in iterator:
        trial_data = seq_df[seq_df["trial"] == trial].sort_values("observation")
        obs_values = nef_obs_values(trial_data["value"].to_numpy(dtype=float), dataset)
        p = {**params, "seed": int(trial)}
        if activity_map is not None:
            activity = activity_map.get(int(trial))
            decoders = (fast_decode_counting(activity, alpha_0=alpha_0, lambda_=lambda_)
                        if activity is not None else _pretrain({**p, "base_seed": int(trial)}))
        else:
            decoders = _pretrain({**p, "base_seed": int(trial)})
        responses = _simulate_trial(obs_values, p, decoders)
        for i, (_, row) in enumerate(trial_data.iterrows()):
            rows.append({
                "trial": int(trial), "observation": int(row["observation"]),
                "response": nef_response_to_model_scale(float(responses[i]), dataset),
            })

    response_df = apply_binary_transform(pd.DataFrame(rows), dataset)
    metrics_df = _metrics_from_responses(seq_df, task, response_df, gt_mode=gt_mode)
    return metrics_df, response_df


def default_nef_cache_path(alpha_0, lambda_, n_neurons=None, n_neurons_counting=None, gt_mode="true"):
    suffix = ""
    if n_neurons is not None:
        suffix += f"_n{n_neurons}"
    if n_neurons_counting is not None:
        suffix += f"_nc{n_neurons_counting}"
    if gt_mode != "true":
        suffix += f"_{gt_mode}"
    return FIGURES_DIR / f"plot_sequences_nef_a{alpha_0:.4f}_l{lambda_:.4f}{suffix}.pkl"


def load_nef_cache(cache_path, alpha_0, lambda_):
    if not cache_path.exists():
        return None
    payload = pd.read_pickle(cache_path)
    if not np.isclose(payload.get("alpha_0"), alpha_0) or not np.isclose(payload.get("lambda_"), lambda_):
        print(f"[nef] Cache params mismatch in {cache_path}")
        return None
    print(f"[nef] Loaded cache: {cache_path}")
    return payload["metrics"]


def save_nef_cache(cache_path, alpha_0, lambda_, metrics, responses):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"alpha_0": alpha_0, "lambda_": lambda_, "metrics": metrics, "responses": responses}, cache_path)
    print(f"[nef] Saved cache: {cache_path}")


def load_or_simulate_nef(pool_root, alpha_0, lambda_, *, cache_path, force=False,
                          show_progress=True, n_neurons=None, n_neurons_counting=None,
                          gt_mode="true", pool_index=0):
    if not force:
        cached = load_nef_cache(cache_path, alpha_0, lambda_)
        if cached is not None:
            return cached
    metrics, responses = {}, {}
    for task in ("binary", "continuous"):
        try:
            pool = load_pool(pool_root, task)
        except FileNotFoundError as e:
            print(f"[nef skip] {e}")
            continue
        seq_df = member_to_df(pool[pool_index])
        metrics[task], responses[task] = simulate_nef_task(
            seq_df, task, alpha_0, lambda_, n_neurons=n_neurons,
            n_neurons_counting=n_neurons_counting, show_progress=show_progress, gt_mode=gt_mode)
    if not metrics:
        return None
    save_nef_cache(cache_path, alpha_0, lambda_, metrics, responses)
    return metrics


def run_models(seq_df, task, alpha_0, rl_lambda, gamma, eps_p, eps_r, gt_mode="true"):
    models = {
        "Bayes": lambda vals, tm, tp: _bayes_responses(vals, task),
        f"RL(\u03b1={alpha_0},\u03bb={rl_lambda})": lambda vals, tm, tp: _rl_responses(vals, task, alpha_0, rl_lambda),
        f"LI(\u03b3={gamma})": lambda vals, tm, tp: _li_responses(vals, task, gamma),
        f"PR(\u03b5p={eps_p},\u03b5r={eps_r})": lambda vals, tm, tp: _pr_responses(vals, task, eps_p, eps_r),
    }
    use_raw_models = set()
    if task == "binary":
        models["Running ratio (optimal, no Laplace)"] = lambda vals, tm, tp: _bayes_responses(vals, task)
        use_raw_models.add("Running ratio (optimal, no Laplace)")

    dataset = f"task_{task}"
    results = {}
    for name, fn in models.items():
        use_raw = name in use_raw_models
        rows = []
        for tid in seq_df["trial"].unique():
            g = seq_df[seq_df["trial"] == tid].sort_values("observation")
            vals = g["value"].tolist()
            tm = float(g["true_mean"].iloc[0])
            tp = float(g["true_p"].iloc[0]) if task == "binary" else float("nan")
            gt_traj = gt_traj_raw = gt_const = None
            if gt_mode == "running_mean":
                base_traj = _bayes_responses(vals, task)
                if task == "binary":
                    gt_df = pd.DataFrame({"observation": list(range(len(base_traj))), "response": base_traj})
                    gt_df = apply_binary_transform(gt_df, dataset)
                    gt_traj = gt_df["response"].tolist()
                    gt_traj_raw = gt_df["response_raw"].tolist()
                else:
                    gt_traj = gt_traj_raw = base_traj
            else:
                gt_const = _ground_truth(task, tm, tp)
            resp = fn(vals, tm, tp)
            resp_df = pd.DataFrame({"observation": list(range(len(resp))), "response": resp})
            resp_df = apply_binary_transform(resp_df, dataset)
            col = "response_raw" if use_raw else "response"
            active_gt_traj = gt_traj_raw if use_raw else gt_traj
            prev = None
            for obs_i, r in enumerate(resp_df[col].tolist()):
                gt = active_gt_traj[obs_i] if gt_mode == "running_mean" else gt_const
                rows.append({
                    "observation": obs_i + 1, "err": abs(r - gt),
                    "delta": abs(r - prev) if prev is not None else np.nan,
                })
                prev = r
        results[name] = pd.DataFrame(rows)
    return results


def run_models_pooled(pool_root, task, alpha_0, rl_lambda, gamma, eps_p, eps_r, gt_mode="true",
                       n_pool=None):
    """Aggregates run_models over EVERY real pool member's own trials --
    one mean-per-observation curve PER POOL MEMBER, then mean + 95% CI
    (via SEM) ACROSS the population of pool members."""
    pool = load_pool(pool_root, task)
    members = pool if n_pool is None else pool[:n_pool]
    assert members, f"empty pool for {task} at {pool_root}"

    member_curves = {}
    for member in members:
        seq_df = member_to_df(member)
        model_data = run_models(seq_df, task, alpha_0, rl_lambda, gamma, eps_p, eps_r, gt_mode=gt_mode)
        for name, df in model_data.items():
            d = member_curves.setdefault(name, {"err": [], "delta": []})
            d["err"].append(df.dropna(subset=["err"]).groupby("observation")["err"].mean())
            d["delta"].append(df.dropna(subset=["delta"]).groupby("observation")["delta"].mean())

    results = {}
    for name, d in member_curves.items():
        agg = {}
        for metric in ("err", "delta"):
            wide = pd.concat(d[metric], axis=1)
            mean_c = wide.mean(axis=1)
            n_eff = wide.notna().sum(axis=1).clip(lower=1)
            ci = 1.96 * wide.std(axis=1) / np.sqrt(n_eff)
            agg[metric] = (mean_c, (mean_c - ci).clip(lower=0), mean_c + ci)
        results[name] = agg
    return results


def _plot_panel(ax, model_data, metric, title, ylabel, colors):
    for (name, df), color in zip(model_data.items(), colors):
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


def _plot_panel_pooled(ax, pooled, metric, title, ylabel, colors):
    for (name, agg), color in zip(pooled.items(), colors):
        mean_c, lo, hi = agg[metric]
        ax.fill_between(mean_c.index, lo, hi, color=color, alpha=0.15, zorder=1)
        ax.plot(mean_c.index, mean_c.values, color=color, lw=1.8, label=name, zorder=3)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Observation", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def compute_split_half_reliability(seq_df, task, alpha_0=1.0, n_lambdas=50):
    """Split-half \u03bb-recovery reliability: sweep n_lambdas different
    RL_lambda ground-truth parameterizations across the SAME sequence set
    -- standing in for "many different participants, each with their own
    true decay rate", the same role test_sequences.py's own lambda sweep
    plays in its panels E/L. Reuses fit_lambda_mid/split_half_lambda
    directly from test_sequences.py so this number means exactly the same
    thing as it does there."""
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
                rows.append({"model_id": model_id, "model_type": "RL_lambda",
                            "trial": int(tid), "observation": obs_i + 1, "response": float(r)})
    sweep_df = pd.DataFrame(rows)
    half_wide = split_half_lambda(sweep_df)
    rel = half_wide.dropna(subset=["first", "second"])
    if len(rel) >= 3 and rel["first"].std() > 1e-9:
        r_val, p_val = pearsonr(rel["first"], rel["second"])
    else:
        r_val, p_val = float("nan"), float("nan")
    return rel, r_val, p_val


def _plot_reliability_panel(ax, rel, r, p, title, color):
    if not rel.empty:
        ax.scatter(rel["first"], rel["second"], s=14, alpha=0.6, color=color, zorder=3)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--", zorder=1)
    note = f"r={r:.2f}{pvalue_to_stars(p)}  (n={len(rel)})" if np.isfinite(r) else "insufficient data"
    ax.text(0.05, 0.95, note, transform=ax.transAxes, ha="left", va="top", fontsize=7,
           bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Odd-trial fitted \u03bb", fontsize=8)
    ax.set_ylabel("Even-trial fitted \u03bb", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure_across_models(pool_root, alpha_0, rl_lambda, gamma, eps_p, eps_r, out_path, *,
                               skip_nef=False, force_nef=False, nef_cache=None, n_neurons=None,
                               n_neurons_counting=None, gt_mode="running_mean", pooled=True,
                               n_pool=None, pool_index=0):
    assert gt_mode in GT_MODES, f"gt_mode must be one of {GT_MODES}"
    if pooled:
        assert skip_nef, ("pooled aggregation isn't wired up for NEF -- NEF metrics would still "
                          "come from a single pool member, mismatched against the pooled model "
                          "curves. Pass --skip_nef.")
    cache_path = nef_cache or default_nef_cache_path(alpha_0, rl_lambda, gt_mode=gt_mode)
    nef_metrics = None
    if not skip_nef:
        nef_metrics = load_or_simulate_nef(pool_root, alpha_0, rl_lambda, cache_path=cache_path,
                                           force=force_nef, n_neurons=n_neurons,
                                           n_neurons_counting=n_neurons_counting, gt_mode=gt_mode,
                                           pool_index=pool_index)

    n_models = (5 if nef_metrics else 4) + 1  # +1: binary's extra "Running ratio" model
    colors = get_palette(n_models)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    gt_label = "RMSE vs running mean" if gt_mode == "running_mean" else "RMSE vs true param"
    reliability_color = get_palette(3)[2]

    for row, task in enumerate(["binary", "continuous"]):
        try:
            pool = load_pool(pool_root, task)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue

        if pooled:
            pooled_models = run_models_pooled(pool_root, task, alpha_0, rl_lambda, gamma, eps_p, eps_r,
                                              gt_mode=gt_mode, n_pool=n_pool)
            n_members = len(pool) if n_pool is None else min(n_pool, len(pool))
            prefix = int(pool[0][0]["prefix_length"])
            label = task.capitalize()
            _plot_panel_pooled(axes[row, 0], pooled_models, "err",
                               f"{label} \u2014 RMSE (prefix={prefix}, n={n_members} pool members)",
                               gt_label, colors)
            _plot_panel_pooled(axes[row, 1], pooled_models, "delta",
                               f"{label} \u2014 |\u0394response| (prefix={prefix}, n={n_members} pool members)",
                               "Mean |\u0394response|", colors)
            for ax in axes[row, :2]:
                ax.axvline(prefix + 0.5, color="#999", lw=0.8, ls="--", alpha=0.6)
            seq_df = member_to_df(pool[pool_index])
        else:
            seq_df = member_to_df(pool[pool_index])
            model_data = run_models(seq_df, task, alpha_0, rl_lambda, gamma, eps_p, eps_r, gt_mode=gt_mode)
            if nef_metrics and task in nef_metrics:
                model_data[f"NEF(\u03b1={alpha_0},\u03bb={rl_lambda})"] = nef_metrics[task]
            prefix = int(pool[pool_index][0]["prefix_length"])
            label = task.capitalize()
            _plot_panel(axes[row, 0], model_data, "err", f"{label} \u2014 RMSE (prefix={prefix})", gt_label, colors)
            _plot_panel(axes[row, 1], model_data, "delta", f"{label} \u2014 |\u0394response| (prefix={prefix})",
                       "Mean |\u0394response|", colors)
            for ax in axes[row, :2]:
                ax.axvline(prefix + 0.5, color="#999", lw=0.8, ls="--", alpha=0.6)

        rel, r_val, p_val = compute_split_half_reliability(seq_df, task)
        if np.isfinite(r_val):
            print(f"[reliability] {task}: split-half \u03bb r={r_val:.3f} p={p_val:.4g} (n={len(rel)})")
        else:
            print(f"[reliability] {task}: insufficient data")
        _plot_reliability_panel(axes[row, 2], rel, r_val, p_val, f"{label} \u2014 split-half \u03bb reliability",
                                reliability_color)

    fig.suptitle(f"Sequence diagnostics ({gt_mode})  |  RL/NEF \u03b1={alpha_0} \u03bb={rl_lambda}  "
                f"LI \u03b3={gamma}  PR \u03b5p={eps_p} \u03b5r={eps_r}", fontsize=10, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Branch 2: across_pids -- fix the MODEL, vary the pids (per pool member)
# ═══════════════════════════════════════════════════════════════════════════

def _rl_lambda_responses(values, task, alpha_0, lambda_):
    """RL_lambda agent for THIS branch specifically: alpha(n)=alpha_0/n**lambda_,
    starting at est=0.0 for BOTH tasks -- matches test_sequences.py's own
    _run_model('RL_lambda', ...) exactly (Mean is this agent's special case
    alpha_0=1, lambda_=1), so across_pids' own reliability numbers stay
    comparable to figures/test_sequences.pdf's panels E/L. Deliberately a
    SEPARATE function from across_models' own _rl_responses (which starts
    continuous at 0.5, matching THAT branch's own historical convention) --
    not unified, since the two branches were calibrated against different
    reference points and unifying them would silently change one branch's
    numbers."""
    resps, est = [], 0.0
    for n, v in enumerate(values, 1):
        obs_n = v / 100.0 if task == "continuous" else float(v)
        alpha = alpha_0 / (n ** lambda_)
        est = float(np.clip(est + alpha * (obs_n - est), -1.0, 1.0))
        resps.append(est)
    return resps


def _rows_for_sequences(df, task, model_id, agent_name, agent_fn, skip_binary_transform):
    rows = []
    for trial_id, g in df.groupby("trial"):
        g = g.sort_values("observation")
        vals = g["value"].tolist()
        resp = agent_fn(vals, task)
        resp_df = pd.DataFrame({"observation": g["observation"].tolist(), "response": resp})
        if task == "binary" and not skip_binary_transform:
            resp_df = apply_binary_transform(resp_df, f"task_{task}")
        tm, ts, tp, qid = g["true_mean"].iloc[0], g["true_std"].iloc[0], g["true_p"].iloc[0], g["qid"].iloc[0]
        for obs, r in zip(g["observation"].tolist(), resp_df["response"].tolist()):
            rows.append({
                "model_id": model_id, "model_type": agent_name, "task": task, "trial": int(trial_id),
                "qid": int(qid), "observation": int(obs), "response": float(r),
                "true_mean": tm, "true_std": ts, "true_p": tp,
            })
    return rows


def load_pool_for_across_pids(tasks, pool_root, agent_name, agent_fn, skip_binary_transform=False,
                               n_pool=None):
    """Loads every (or the first n_pool) real member of task_backend's pool
    and builds one long-format row set per member (per pid) -- the REAL,
    deployed sequences, not a simulation of what a pool COULD look like."""
    all_rows = []
    for task in tasks:
        pool = load_pool(pool_root, task)
        members = pool if n_pool is None else pool[:n_pool]
        for idx, member in enumerate(members):
            df = member_to_df(member)
            all_rows.extend(_rows_for_sequences(df, task, f"p{idx:04d}", agent_name, agent_fn,
                                                skip_binary_transform))
    return pd.DataFrame(all_rows)


def _fit_A_lambda(g):
    """Same fit as test_sequences.py's fit_lambda_mid, but also keeping A
    (needed to draw the fitted curve) -- duplicated inline rather than
    changing fit_lambda_mid's own (lambda, p) return shape, which other
    code depends on."""
    dlt = compute_abs_delta(g)
    if dlt.empty:
        return np.nan, np.nan
    curve = dlt.groupby("observation")["delta"].mean().sort_index()
    curve = curve[curve.index >= 2]
    if len(curve) < 3:
        return np.nan, np.nan
    n_arr, y_arr = curve.index.values.astype(float), curve.values.astype(float)
    try:
        popt, _ = scipy_curve_fit(lambda n, A, lam: A * n ** (-lam), n_arr, y_arr,
                                  p0=[0.1, 0.5], bounds=([0, 0], [2, 2]), maxfev=2000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan


def compute_fits(raw_df, tasks):
    """Per (pid, task): full-data fitted (A, lambda) plus split-half
    (first-half-trials vs second-half-trials) lambda, via test_sequences.py's
    own fit_lambda_mid/split_half_lambda."""
    rows = []
    for task in tasks:
        df_task = raw_df[raw_df.task == task]
        half_wide = split_half_lambda(df_task)
        half_by_mid = half_wide.set_index("model_id") if not half_wide.empty else pd.DataFrame()
        for model_id, g in df_task.groupby("model_id"):
            lam_full, p_full = fit_lambda_mid(g)
            A_full, _ = _fit_A_lambda(g)
            if model_id in half_by_mid.index:
                first_half, second_half = half_by_mid.loc[model_id, "first"], half_by_mid.loc[model_id, "second"]
            else:
                first_half = second_half = np.nan
            rows.append({"model_id": model_id, "task": task, "lambda_full": lam_full, "A_full": A_full,
                        "p_full": p_full, "lambda_first_half": first_half, "lambda_second_half": second_half})
    return pd.DataFrame(rows)


def _rmse_curve(g, task):
    rows = []
    for _, tg in g.groupby("trial"):
        tg = tg.sort_values("observation")
        gt = _ground_truth(task, tg["true_mean"].iloc[0], tg["true_p"].iloc[0])
        for obs, r in zip(tg["observation"], tg["response"]):
            rows.append({"observation": int(obs), "err": abs(r - gt)})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).groupby("observation")["err"].mean().sort_index()


def _delta_curve(g):
    dlt = compute_abs_delta(g)
    if dlt.empty:
        return pd.Series(dtype=float)
    return dlt.groupby("observation")["delta"].mean().sort_index()


def make_figure_across_pids(raw_df, fits_df, tasks, out_path, n_members, agent_label="Mean"):
    from utils.plot_style import apply_style
    apply_style()
    n_rows = len(tasks)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.2 * n_rows), squeeze=False)
    palette = get_palette(3)
    thin_color, thin_lw, thin_alpha = "0.55", 0.5, 0.35

    for row, task in enumerate(tasks):
        df_task, fits_task = raw_df[raw_df.task == task], fits_df[fits_df.task == task]
        model_ids = sorted(df_task["model_id"].unique())
        ax_rmse, ax_delta, ax_fit = axes[row]

        all_rmse = []
        for mid in model_ids:
            curve = _rmse_curve(df_task[df_task.model_id == mid], task)
            if curve.empty:
                continue
            ax_rmse.plot(curve.index, curve.values, color=thin_color, lw=thin_lw, alpha=thin_alpha, zorder=1)
            all_rmse.append(curve)
        if all_rmse:
            wide = pd.concat(all_rmse, axis=1)
            mean_curve = wide.mean(axis=1)
            n_eff = wide.notna().sum(axis=1).clip(lower=1)
            ci = 1.96 * wide.std(axis=1) / np.sqrt(n_eff)
            ax_rmse.fill_between(mean_curve.index, (mean_curve - ci).clip(lower=0), mean_curve + ci,
                                 color=palette[0], alpha=0.2, zorder=4, label="95% CI")
            ax_rmse.plot(mean_curve.index, mean_curve.values, color=palette[0], lw=2.2, zorder=5,
                        label="Mean across pids")
        ax_rmse.set_xlabel("Observation")
        ax_rmse.set_ylabel("RMSE vs true mean/p")
        ax_rmse.set_title(f"{task.capitalize()} \u2014 RMSE (n={len(model_ids)} pids)",
                          fontsize=9, fontweight="bold")
        ax_rmse.set_ylim(bottom=0)
        ax_rmse.legend(fontsize=7, frameon=False)
        ax_rmse.spines[["top", "right"]].set_visible(False)

        all_delta = []
        for mid in model_ids:
            curve = _delta_curve(df_task[df_task.model_id == mid])
            if curve.empty:
                continue
            ax_delta.plot(curve.index, curve.values, color=thin_color, lw=thin_lw, alpha=thin_alpha, zorder=1)
            all_delta.append(curve)
        if all_delta:
            wide = pd.concat(all_delta, axis=1)
            mean_curve = wide.mean(axis=1)
            n_eff = wide.notna().sum(axis=1).clip(lower=1)
            ci = 1.96 * wide.std(axis=1) / np.sqrt(n_eff)
            ax_delta.fill_between(mean_curve.index, (mean_curve - ci).clip(lower=0), mean_curve + ci,
                                  color=palette[1], alpha=0.2, zorder=4, label="95% CI")
            ax_delta.plot(mean_curve.index, mean_curve.values, color=palette[1], lw=2.2, zorder=5,
                         label="Mean across pids")
        ax_delta.set_xlabel("Observation")
        ax_delta.set_ylabel("Mean |\u0394response|")
        ax_delta.set_title(f"{task.capitalize()} \u2014 |\u0394response|", fontsize=9, fontweight="bold")
        ax_delta.set_ylim(bottom=0)
        ax_delta.legend(fontsize=7, frameon=False)
        ax_delta.spines[["top", "right"]].set_visible(False)

        n_grid = np.linspace(2, 15, 100)
        for _, frow in fits_task.iterrows():
            A, lam = frow["A_full"], frow["lambda_full"]
            if not (np.isfinite(A) and np.isfinite(lam)):
                continue
            ax_fit.plot(n_grid, A * n_grid ** (-lam), color=thin_color, lw=thin_lw, alpha=thin_alpha, zorder=1)
        valid = fits_task.dropna(subset=["A_full", "lambda_full"])
        if not valid.empty:
            mean_A, mean_lam = valid["A_full"].mean(), valid["lambda_full"].mean()
            ax_fit.plot(n_grid, mean_A * n_grid ** (-mean_lam), color=palette[2], lw=2.2, zorder=5,
                       label="Mean fitted curve")
        rel = fits_task.dropna(subset=["lambda_first_half", "lambda_second_half"])
        if len(rel) >= 3 and rel["lambda_first_half"].std() > 1e-9:
            r, p = pearsonr(rel["lambda_first_half"], rel["lambda_second_half"])
            note = f"split-half \u03bb reliability:\nr={r:.2f}{pvalue_to_stars(p)} (n={len(rel)})"
        else:
            note = "split-half \u03bb reliability:\nnot enough variance to compute"
        ax_fit.text(0.97, 0.95, note, transform=ax_fit.transAxes, ha="right", va="top", fontsize=7,
                   bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax_fit.set_xlabel("Observation")
        ax_fit.set_ylabel("Fitted A\u00b7n^(-\u03bb)")
        ax_fit.set_title(f"{task.capitalize()} \u2014 fitted power laws", fontsize=9, fontweight="bold")
        ax_fit.set_ylim(bottom=0)
        ax_fit.legend(fontsize=7, frameon=False)
        ax_fit.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Real task_backend pool ({n_members} real pids, not simulated)  |  "
                f"agent: {agent_label}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _add_pool_root_arg(p):
    p.add_argument("--pool_root", default="task_backend",
                   help="Directory containing sequences_numbers.json/sequences_colors.json "
                        "(default assumes task_backend/ is a sibling of this repo's root)")


def cli_across_models(argv):
    p = argparse.ArgumentParser(prog="plot_sequences.py across_models")
    _add_pool_root_arg(p)
    p.add_argument("--alpha_0", type=float, default=1.0)
    p.add_argument("--rl_lambda", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.9, help="LeakyIntegrator \u03b3")
    p.add_argument("--eps_p", type=float, default=0.5, help="PrimacyRecency \u03b5_p")
    p.add_argument("--eps_r", type=float, default=0.5, help="PrimacyRecency \u03b5_r")
    p.add_argument("--out", default=None)
    p.add_argument("--nef_cache", default=None)
    p.add_argument("--skip_nef", action="store_true")
    p.add_argument("--force_nef", action="store_true")
    p.add_argument("--nef_only", action="store_true")
    p.add_argument("--n_neurons", type=int, default=None)
    p.add_argument("--n_neurons_counting", type=int, default=None)
    p.add_argument("--gt_mode", choices=list(GT_MODES), default="running_mean")
    p.add_argument("--pool_index", type=int, default=0)
    p.add_argument("--no_pooled", dest="pooled", action="store_false",
                   help="Use just ONE pool member (--pool_index) instead of aggregating over "
                        "all 200 -- required alongside NEF, which isn't wired up for pooled "
                        "aggregation.")
    p.add_argument("--n_pool", type=int, default=None,
                   help="Cap pooled aggregation to the first N members instead of all 200.")
    args = p.parse_args(argv)

    cache_path = Path(args.nef_cache) if args.nef_cache else default_nef_cache_path(
        args.alpha_0, args.rl_lambda, n_neurons=args.n_neurons,
        n_neurons_counting=args.n_neurons_counting, gt_mode=args.gt_mode)

    if args.nef_only:
        load_or_simulate_nef(args.pool_root, args.alpha_0, args.rl_lambda, cache_path=cache_path,
                             force=args.force_nef, n_neurons=args.n_neurons,
                             n_neurons_counting=args.n_neurons_counting, gt_mode=args.gt_mode,
                             pool_index=args.pool_index)
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else FIGURES_DIR / "inspect_pool_sequences_across_models.pdf"
    make_figure_across_models(args.pool_root, args.alpha_0, args.rl_lambda, args.gamma, args.eps_p,
                              args.eps_r, out, skip_nef=args.skip_nef, force_nef=args.force_nef,
                              nef_cache=cache_path, n_neurons=args.n_neurons,
                              n_neurons_counting=args.n_neurons_counting, gt_mode=args.gt_mode,
                              pooled=args.pooled, n_pool=args.n_pool, pool_index=args.pool_index)


def cli_across_pids(argv):
    p = argparse.ArgumentParser(prog="plot_sequences.py across_pids")
    _add_pool_root_arg(p)
    p.add_argument("--task", choices=["continuous", "binary", "both"], default="both")
    p.add_argument("--agent", choices=["Mean", "RL_lambda"], default="Mean",
                   help="Mean = running average (Bayes' special case). RL_lambda = "
                        "alpha(n)=alpha_0/n**lambda_ delta-rule agent.")
    p.add_argument("--alpha_0", type=float, default=1.0, help="Only used when --agent RL_lambda")
    p.add_argument("--rl_lambda", type=float, default=0.5, help="Only used when --agent RL_lambda")
    p.add_argument("--skip_binary_transform", action="store_true",
                   help="Skip Laplace smoothing on binary responses -- matters for non-Mean "
                        "agents specifically (see _rl_lambda_responses docstring history). "
                        "No effect on continuous.")
    p.add_argument("--n_pool", type=int, default=None, help="Cap to the first N pool members/pids (default: all 200)")
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--run_folder", default="plot_sequences")
    args = p.parse_args(argv)

    tasks = ["continuous", "binary"] if args.task == "both" else [args.task]
    if args.agent == "RL_lambda":
        agent_name = f"RL_lambda(\u03b1={args.alpha_0},\u03bb={args.rl_lambda})"
        agent_fn = lambda vals, task: _rl_lambda_responses(vals, task, args.alpha_0, args.rl_lambda)
        stem_suffix = f"_rl_a{args.alpha_0:.2f}_l{args.rl_lambda:.2f}"
    else:
        agent_name, agent_fn, stem_suffix = "Mean", _bayes_responses, ""
    if args.skip_binary_transform:
        agent_name += " [no Laplace transform]"
        stem_suffix += "_notransform"

    print(f"Loading real task_backend pool sequences from {args.pool_root} x {len(tasks)} task(s) "
          f"| agent={agent_name}...")
    raw_df = load_pool_for_across_pids(tasks, args.pool_root, agent_name, agent_fn,
                                       skip_binary_transform=args.skip_binary_transform, n_pool=args.n_pool)
    n_members = raw_df["model_id"].nunique()

    print("Fitting power laws (full + split-half) per pid...")
    fits_df = compute_fits(raw_df, tasks)

    out_folder = resolve_run_folder(args.run_folder)
    raw_path = out_folder / f"across_pids{stem_suffix}_raw_{n_members}p.pkl"
    fits_path = out_folder / f"across_pids{stem_suffix}_fits_{n_members}p.pkl"
    raw_df.to_pickle(raw_path)
    fits_df.to_pickle(fits_path)
    print(f"Saved: {raw_path}  ({len(raw_df)} rows)")
    print(f"Saved: {fits_path}  ({len(fits_df)} rows)")

    print()
    for task in tasks:
        t = fits_df[fits_df.task == task]
        print(f"[{task}] fitted lambda: mean={t.lambda_full.mean():.4f} std={t.lambda_full.std():.4f} "
              f"range=[{t.lambda_full.min():.4f},{t.lambda_full.max():.4f}] "
              f"(n_failed_fit={t.lambda_full.isna().sum()}/{len(t)})")
    print()

    out_pdf = Path(args.out_pdf) if args.out_pdf else FIGURES_DIR / f"inspect_pool_sequences_across_pids{stem_suffix}.pdf"
    make_figure_across_pids(raw_df, fits_df, tasks, out_pdf, n_members, agent_label=agent_name)


def main():
    top = argparse.ArgumentParser(prog="plot_sequences.py")
    top.add_argument("branch", choices=["across_models", "across_pids"])
    top_args, rest = top.parse_known_args()
    if top_args.branch == "across_models":
        cli_across_models(rest)
    else:
        cli_across_pids(rest)
    print("JOB_COMPLETE")


if __name__ == "__main__":
    main()
