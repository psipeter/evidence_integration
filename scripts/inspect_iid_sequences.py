"""
scripts/inspect_iid_sequences.py
==================================
Investigates how much natural sequence-to-sequence variance the pure i.i.d.
branch (task/generate_sequences_iid.py) would introduce if every participant
saw a DIFFERENT, independently randomized set of sequences, rather than
everyone sharing one fixed, promoted sequence file (as the real pilot does).
This is purely a simulation/diagnostic tool -- it never touches
task/sequences/ or anything used in production, and it doesn't test i.i.d.
vs quota (that question lives elsewhere in this project's history); it
asks a narrower, purely statistical question: how much does re-randomizing
the sequences themselves spread out our derived curves?

For each of --n_participants simulated participants, generates an
independent n_prefix x n_repeats set of trials via generate_sequences_iid.py
(a fresh RNG seed per participant, per task -- nobody shares anything, not
even a hidden target, across participants), then simulates the "optimal"
agent -- literally the running mean of observed values, called "Bayes"
elsewhere in this codebase (see scripts/inspect_sequences.py's
_bayes_responses) but referred to as "Mean" in this investigation's own
framing. Binary gets the same Laplace-smoothing transform every other
script in this project applies (utils/binary_transform.py); continuous
does not (matches that module's own dataset list). The tiny running-mean/
ground-truth helpers below are copied locally rather than imported from
inspect_sequences.py specifically to avoid pulling in that file's full
NEF/counting_integrator dependency chain just for two small functions --
see that file for the canonical version if these ever drift.

No fitting/optimization happens on the SEQUENCES themselves here -- no
seed search, no smoothing, no moment-matching. This deliberately measures
raw i.i.d. sampling noise, not anything the momentmatch redesign
elsewhere in this project corrects for. The agent itself is fully
deterministic given a sequence, so all variance across participants in
this figure comes from exactly one source: which sequences they happened
to be given.

Ground truth for the RMSE panel is true_mean/true_p (not running_mean) --
a deliberate choice for this specific investigation, confirmed in chat
(this figure is about sequence-sampling noise, not the separate
running-mean-vs-true-mean methodological question explored elsewhere).

Reuses (imports directly, never reimplemented) fit_lambda_mid /
split_half_lambda / compute_abs_delta from scripts/test_sequences.py, so
the power-law fit and split-half reliability numbers here mean exactly
the same thing as they do in figures/test_sequences.pdf's panels E/L.

Usage:
    python scripts/inspect_iid_sequences.py --n_participants 50
    python scripts/inspect_iid_sequences.py --n_participants 100 --n_prefix 10 --n_repeats 4
    python scripts/inspect_iid_sequences.py --n_participants 50 --task continuous

Output:
    data/runs/inspect_iid_sequences/raw_{n}p.pkl   -- long-format observations+responses
    data/runs/inspect_iid_sequences/fits_{n}p.pkl  -- per-participant fitted lambda (full + split-half)
    figures/inspect_iid_sequences.pdf
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit as scipy_curve_fit
from scipy.stats import pearsonr

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))                 # utils.*
sys.path.insert(0, str(_ROOT / "task"))        # generate_sequences[_iid]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # test_sequences

from generate_sequences import make_rng
from generate_sequences_iid import generate_task_sequences_iid
from utils.binary_transform import apply_binary_transform
from utils.paths import FIGURES_DIR, resolve_run_folder
from utils.plot_style import apply_style, get_palette, pvalue_to_stars
from test_sequences import fit_lambda_mid, split_half_lambda, compute_abs_delta


# ---------------------------------------------------------------------------
# Tiny local copies of the running-mean / ground-truth helpers -- see
# module docstring for why these are copied rather than imported.
# ---------------------------------------------------------------------------
def _ground_truth(task, true_mean, true_p):
    if task == "continuous":
        return float(true_mean) / 100.0
    return float(true_p) * 2.0 - 1.0


def _running_mean_responses(values, task):
    """The 'optimal' / 'Mean' agent: literally the running mean of observed
    values, on [0,1] for continuous (no further transform) or on [-1,1]
    for binary (Laplace-smoothed separately via apply_binary_transform,
    applied by the caller -- matching how every other script in this
    project handles binary)."""
    if task == "continuous":
        resps, total = [], 0.0
        for n, v in enumerate(values, 1):
            total += v / 100.0
            resps.append(total / n)
        return resps
    resps, total = [], 0.0
    for n, v in enumerate(values, 1):
        total += float(v)
        resps.append(float(np.clip(total / n, -1.0, 1.0)))
    return resps


def _rl_lambda_responses(values, task, alpha_0, lambda_):
    """RL_lambda agent: alpha(n) = alpha_0 / n**lambda_, applied to the SAME
    delta-rule update as the Mean agent (which is just RL_lambda's special
    case alpha_0=1, lambda_=1 -- see chat history for the exact equivalence
    proof). Matches test_sequences.py's own _run_model('RL_lambda', ...)
    exactly: starts at est=0.0 (not 0.5) for BOTH tasks, obs_n on [0,1] for
    continuous / raw {-1,+1} for binary, uniform clip to [-1,1] (a no-op
    for continuous, which never leaves [0,1] under this update rule, but
    kept identical to that reference implementation rather than
    special-cased away). With lambda_ << 1, alpha decays much slower than
    the Mean agent's 1/n -- the agent keeps weighting new observations
    non-trivially even late in a trial, which is exactly why this run is
    expected to be noisier/more sequence-sensitive than the Mean agent.
    """
    resps = []
    est = 0.0
    for n, v in enumerate(values, 1):
        obs_n = v / 100.0 if task == "continuous" else float(v)
        alpha = alpha_0 / (n ** lambda_)
        est = est + alpha * (obs_n - est)
        est = float(np.clip(est, -1.0, 1.0))
        resps.append(est)
    return resps


# ---------------------------------------------------------------------------
# Simulation: one fresh, independent sequence set per participant
# ---------------------------------------------------------------------------
def simulate_participants(n_participants, tasks, n_prefix, n_repeats,
                          seq_length, prefix_length, mean_range, std_fixed,
                          p_range, base_seed, agent_name="Mean", agent_fn=None,
                          skip_binary_transform=False, progress_every=10):
    """Generate n_participants independent sets of sequences (a fresh RNG
    draw per participant, per task) via generate_sequences_iid.py, simulate
    the chosen agent on each, and return one long-format DataFrame with
    columns matching what test_sequences.py's helpers expect (model_id,
    model_type, trial, observation, response, task, true_mean, true_std,
    true_p, qid).

    agent_fn(values, task) -> list[float] selects the agent; defaults to
    _running_mean_responses (the 'Mean' agent) if not given. agent_name is
    stored in the model_type column and used in figure titles/filenames --
    keep it in sync with whatever agent_fn actually is.

    skip_binary_transform: the Laplace-smoothing transform (utils/
    binary_transform.py) is applied to every non-exempt model's binary
    output by convention elsewhere in this project (test_sequences.py,
    inspect_sequences.py's run_agents) -- but its actual justification
    ('optimal Bayesian estimate under a uniform prior') is specifically
    about a raw sample-mean estimator, and doesn't obviously carry over to
    a differently-shaped estimator like RL_lambda. Confirmed empirically
    (chat history) that it introduces a large bias in RL_lambda's
    recovered lambda specifically (e.g. true lambda=0.3 recovered as 0.16
    with the transform vs 0.32 without it) -- set True to fit/plot on the
    untransformed response instead, isolating sequence-sampling noise from
    a bias inherited from a different model's calibration formula. Has no
    effect on continuous (which never gets this transform either way).

    generate_task_sequences_iid prints a lot per call (by design, for its
    normal single-generation use) -- redirected to keep n_participants x
    n_tasks calls from flooding the console; a concise progress line is
    printed here instead.
    """
    if agent_fn is None:
        agent_fn = _running_mean_responses
    all_rows = []
    for pid in range(n_participants):
        model_id = f"p{pid:04d}"
        for task in tasks:
            # Distinct, non-overlapping seed per (participant, task) --
            # offset large enough it can never collide with
            # generate_task_sequences_iid's own internal task offset
            # (base_seed vs base_seed+1000, used when this project's main()
            # generates ONE shared file across all participants).
            seed = base_seed + pid * 100_000 + (0 if task == "continuous" else 50_000)
            rng = make_rng(seed)
            args_ns = SimpleNamespace(
                task=task, n_unique_sequences=n_prefix, n_repeats=n_repeats,
                seq_length=seq_length, prefix_length=prefix_length,
                mean_range=mean_range, std_fixed=std_fixed, p_range=p_range,
                k_std_cont=0.7, output_dir="/tmp", seed=seed, report=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                df, _ = generate_task_sequences_iid(task, args_ns, rng)

            for trial_id, g in df.groupby("trial"):
                g = g.sort_values("observation")
                vals = g["value"].tolist()
                resp = agent_fn(vals, task)
                # Feed the REAL (1-indexed) observation values into
                # apply_binary_transform, matching exactly what
                # test_sequences.py's own pipeline does -- not the
                # 0-indexed convention that module's own docstring
                # describes (a pre-existing discrepancy elsewhere in this
                # project, not something to silently "fix" here; matching
                # the ACTUAL existing pipeline behavior matters more for
                # this investigation than the documented-but-unused intent).
                resp_df = pd.DataFrame({
                    "observation": g["observation"].tolist(),
                    "response": resp,
                })
                if task == "binary" and not skip_binary_transform:
                    resp_df = apply_binary_transform(resp_df, f"task_{task}")
                tm  = g["true_mean"].iloc[0]
                ts  = g["true_std"].iloc[0]
                tp  = g["true_p"].iloc[0]
                qid = g["qid"].iloc[0]
                for obs, r in zip(g["observation"].tolist(),
                                 resp_df["response"].tolist()):
                    all_rows.append({
                        "model_id": model_id, "model_type": agent_name,
                        "task": task, "trial": int(trial_id), "qid": int(qid),
                        "observation": int(obs), "response": float(r),
                        "true_mean": tm, "true_std": ts, "true_p": tp,
                    })
        if (pid + 1) % progress_every == 0 or pid == n_participants - 1:
            print(f"  simulated {pid + 1}/{n_participants} participants")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Per-participant power-law fits: full-data + split-half (panels E/L)
# ---------------------------------------------------------------------------
def _fit_A_lambda(g):
    """Same fit as test_sequences.py's fit_lambda_mid, but also keeping A
    (fit_lambda_mid only returns lambda) -- needed to actually draw the
    fitted curve in column 3. Deliberately duplicated inline (two extra
    lines) rather than modifying fit_lambda_mid's return signature, since
    that function is imported as-is from test_sequences.py and other code
    there depends on its current (lambda, p) return shape."""
    dlt = compute_abs_delta(g)
    if dlt.empty:
        return np.nan, np.nan
    curve = dlt.groupby("observation")["delta"].mean().sort_index()
    curve = curve[curve.index >= 2]
    if len(curve) < 3:
        return np.nan, np.nan
    n_arr, y_arr = curve.index.values.astype(float), curve.values.astype(float)
    try:
        popt, _ = scipy_curve_fit(
            lambda n, A, lam: A * n ** (-lam), n_arr, y_arr,
            p0=[0.1, 0.5], bounds=([0, 0], [2, 2]), maxfev=2000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan


def compute_fits(raw_df, tasks):
    """Per (participant, task): full-data fitted (A, lambda) plus
    split-half (first-half-trials vs second-half-trials) lambda, via
    test_sequences.py's own fit_lambda_mid/split_half_lambda -- so these
    numbers mean exactly what they mean in figures/test_sequences.pdf's
    panels E/L (there, per lambda-parameter model_id; here, per simulated
    participant)."""
    rows = []
    for task in tasks:
        df_task = raw_df[raw_df.task == task]
        half_wide = split_half_lambda(df_task)
        half_by_mid = (half_wide.set_index("model_id")
                       if not half_wide.empty else pd.DataFrame())
        for model_id, g in df_task.groupby("model_id"):
            lam_full, p_full = fit_lambda_mid(g)
            A_full, _lam_check = _fit_A_lambda(g)
            if model_id in half_by_mid.index:
                first_half  = half_by_mid.loc[model_id, "first"]
                second_half = half_by_mid.loc[model_id, "second"]
            else:
                first_half = second_half = np.nan
            rows.append({
                "model_id": model_id, "task": task,
                "lambda_full": lam_full, "A_full": A_full, "p_full": p_full,
                "lambda_first_half": first_half,
                "lambda_second_half": second_half,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-participant curves (thin lines)
# ---------------------------------------------------------------------------
def _rmse_curve(g, task):
    """Per-observation mean |response - true ground truth|, for ONE
    participant's data (already filtered to one model_id, one task) --
    ground truth is true_mean/true_p, constant per trial, per this
    investigation's confirmed gt choice (not running_mean)."""
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
    """Per-observation mean |Δresponse| for one participant's data."""
    dlt = compute_abs_delta(g)
    if dlt.empty:
        return pd.Series(dtype=float)
    return dlt.groupby("observation")["delta"].mean().sort_index()


# ---------------------------------------------------------------------------
# Figure: 3 columns x len(tasks) rows
# ---------------------------------------------------------------------------
def make_figure(raw_df, fits_df, tasks, out_path, n_participants, agent_label="Mean"):
    apply_style()
    n_rows = len(tasks)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.2 * n_rows), squeeze=False)
    palette = get_palette(3)
    thin_color, thin_lw, thin_alpha = "0.55", 0.5, 0.35

    for row, task in enumerate(tasks):
        df_task   = raw_df[raw_df.task == task]
        fits_task = fits_df[fits_df.task == task]
        model_ids = sorted(df_task["model_id"].unique())
        ax_rmse, ax_delta, ax_fit = axes[row]

        # -- Col 1: RMSE vs obs, one thin line per participant -----------
        all_rmse = []
        for mid in model_ids:
            curve = _rmse_curve(df_task[df_task.model_id == mid], task)
            if curve.empty:
                continue
            ax_rmse.plot(curve.index, curve.values, color=thin_color,
                        lw=thin_lw, alpha=thin_alpha, zorder=1)
            all_rmse.append(curve)
        if all_rmse:
            mean_curve = pd.concat(all_rmse, axis=1).mean(axis=1)
            ax_rmse.plot(mean_curve.index, mean_curve.values, color=palette[0],
                        lw=2.2, zorder=5, label="Mean across participants")
        ax_rmse.set_xlabel("Observation")
        ax_rmse.set_ylabel("RMSE vs true mean/p")
        ax_rmse.set_title(f"{task.capitalize()} \u2014 RMSE "
                          f"(n={len(model_ids)} participants)",
                          fontsize=9, fontweight="bold")
        ax_rmse.set_ylim(bottom=0)
        ax_rmse.legend(fontsize=7, frameon=False)
        ax_rmse.spines[["top", "right"]].set_visible(False)

        # -- Col 2: |Δresponse| vs obs, one thin line per participant ----
        all_delta = []
        for mid in model_ids:
            curve = _delta_curve(df_task[df_task.model_id == mid])
            if curve.empty:
                continue
            ax_delta.plot(curve.index, curve.values, color=thin_color,
                         lw=thin_lw, alpha=thin_alpha, zorder=1)
            all_delta.append(curve)
        if all_delta:
            mean_curve = pd.concat(all_delta, axis=1).mean(axis=1)
            ax_delta.plot(mean_curve.index, mean_curve.values, color=palette[1],
                         lw=2.2, zorder=5, label="Mean across participants")
        ax_delta.set_xlabel("Observation")
        ax_delta.set_ylabel("Mean |\u0394response|")
        ax_delta.set_title(f"{task.capitalize()} \u2014 |\u0394response|",
                           fontsize=9, fontweight="bold")
        ax_delta.set_ylim(bottom=0)
        ax_delta.legend(fontsize=7, frameon=False)
        ax_delta.spines[["top", "right"]].set_visible(False)

        # -- Col 3: fitted power-law curves, one thin line per participant
        n_grid = np.linspace(2, 15, 100)
        for _, frow in fits_task.iterrows():
            A, lam = frow["A_full"], frow["lambda_full"]
            if not (np.isfinite(A) and np.isfinite(lam)):
                continue
            ax_fit.plot(n_grid, A * n_grid ** (-lam), color=thin_color,
                       lw=thin_lw, alpha=thin_alpha, zorder=1)
        valid = fits_task.dropna(subset=["A_full", "lambda_full"])
        if not valid.empty:
            mean_A, mean_lam = valid["A_full"].mean(), valid["lambda_full"].mean()
            ax_fit.plot(n_grid, mean_A * n_grid ** (-mean_lam), color=palette[2],
                       lw=2.2, zorder=5, label="Mean fitted curve")

        rel = fits_task.dropna(subset=["lambda_first_half", "lambda_second_half"])
        if len(rel) >= 3 and rel["lambda_first_half"].std() > 1e-9:
            r, p = pearsonr(rel["lambda_first_half"], rel["lambda_second_half"])
            note = f"split-half \u03bb reliability:\nr={r:.2f}{pvalue_to_stars(p)} (n={len(rel)})"
        else:
            note = "split-half \u03bb reliability:\nnot enough variance to compute"
        ax_fit.text(0.97, 0.95, note, transform=ax_fit.transAxes,
                   ha="right", va="top", fontsize=7,
                   bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        ax_fit.set_xlabel("Observation")
        ax_fit.set_ylabel("Fitted A\u00b7n^(-\u03bb)")
        ax_fit.set_title(f"{task.capitalize()} \u2014 fitted power laws",
                        fontsize=9, fontweight="bold")
        ax_fit.set_ylim(bottom=0)
        ax_fit.legend(fontsize=7, frameon=False)
        ax_fit.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"i.i.d. sequence-generation variance across {n_participants} "
        f"simulated participants (each with independently randomized "
        f"sequences)  |  agent: {agent_label}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n_participants", type=int, default=50)
    p.add_argument("--n_prefix", type=int, default=10,
                  help="n_unique_sequences per participant (passed straight through "
                       "to generate_sequences_iid.py; must be even)")
    p.add_argument("--n_repeats", type=int, default=4)
    p.add_argument("--seq_length", type=int, default=15)
    p.add_argument("--prefix_length", type=int, default=4)
    p.add_argument("--mean_range", type=float, nargs=2, default=[15.0, 85.0],
                  help="Default matches current production continuous range")
    p.add_argument("--std_fixed", type=float, default=15.0)
    p.add_argument("--p_range", type=float, nargs=2, default=[2 / 15, 13 / 15],
                  help="Default matches production blue_range=[2,13] out of 15, "
                       "converted to a p fraction")
    p.add_argument("--task", choices=["continuous", "binary", "both"], default="both")
    p.add_argument("--agent", choices=["Mean", "RL_lambda"], default="Mean",
                  help="Mean = running average (the 'optimal' agent used by default). "
                       "RL_lambda = alpha(n)=alpha_0/n**lambda_ delta-rule agent "
                       "(Mean is RL_lambda's special case alpha_0=1, lambda_=1).")
    p.add_argument("--alpha_0", type=float, default=1.0, help="Only used when --agent RL_lambda")
    p.add_argument("--rl_lambda", type=float, default=0.5, help="Only used when --agent RL_lambda")
    p.add_argument("--skip_binary_transform", action="store_true",
                  help="Skip the Laplace-smoothing transform on binary responses -- see "
                       "simulate_participants' docstring for why this matters for "
                       "non-Mean agents specifically. No effect on continuous.")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--run_folder", default="inspect_iid_sequences")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.n_prefix % 2 == 0, "n_prefix must be even (generate_sequences_iid.py requirement)"
    tasks = ["continuous", "binary"] if args.task == "both" else [args.task]

    if args.agent == "RL_lambda":
        agent_name = f"RL_lambda(\u03b1={args.alpha_0},\u03bb={args.rl_lambda})"
        agent_fn = lambda vals, task: _rl_lambda_responses(vals, task, args.alpha_0, args.rl_lambda)
        default_stem = f"inspect_iid_sequences_rl_a{args.alpha_0:.2f}_l{args.rl_lambda:.2f}"
    else:
        agent_name = "Mean"
        agent_fn = _running_mean_responses
        default_stem = "inspect_iid_sequences"
    if args.skip_binary_transform:
        agent_name += " [no Laplace transform]"
        default_stem += "_notransform"

    print(f"Simulating {args.n_participants} participant(s) x {len(tasks)} task(s) "
          f"({args.n_prefix} prefixes x {args.n_repeats} repeats each = "
          f"{args.n_prefix * args.n_repeats} trials/participant/task) | agent={agent_name}...")
    raw_df = simulate_participants(
        args.n_participants, tasks, args.n_prefix, args.n_repeats,
        args.seq_length, args.prefix_length, args.mean_range, args.std_fixed,
        args.p_range, args.base_seed, agent_name=agent_name, agent_fn=agent_fn,
        skip_binary_transform=args.skip_binary_transform)

    print("Fitting power laws (full + split-half) per participant...")
    fits_df = compute_fits(raw_df, tasks)

    out_folder = resolve_run_folder(args.run_folder)
    raw_path  = out_folder / f"{default_stem}_raw_{args.n_participants}p.pkl"
    fits_path = out_folder / f"{default_stem}_fits_{args.n_participants}p.pkl"
    raw_df.to_pickle(raw_path)
    fits_df.to_pickle(fits_path)
    print(f"Saved: {raw_path}  ({len(raw_df)} rows)")
    print(f"Saved: {fits_path}  ({len(fits_df)} rows)")

    print()
    for task in tasks:
        t = fits_df[fits_df.task == task]
        print(f"[{task}] fitted lambda: mean={t.lambda_full.mean():.4f} "
              f"std={t.lambda_full.std():.4f} "
              f"range=[{t.lambda_full.min():.4f},{t.lambda_full.max():.4f}] "
              f"(n_failed_fit={t.lambda_full.isna().sum()}/{len(t)})")
    print()

    out_pdf = Path(args.out_pdf) if args.out_pdf else FIGURES_DIR / f"{default_stem}.pdf"
    make_figure(raw_df, fits_df, tasks, out_pdf, args.n_participants, agent_label=agent_name)
    print("JOB_COMPLETE")


if __name__ == "__main__":
    main()
