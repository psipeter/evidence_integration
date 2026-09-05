#!/usr/bin/env python3
"""scripts/inspect_participant.py
=================================
Visualizes ONE real participant's actual behavior, pulled directly from
task_backend's Supabase `events` table -- not a simulation, and not the
model-comparison-on-generated-sequences figures scripts/plot_sequences.py
already covers. This is for looking at what a real person actually did,
one participant at a time, before there's enough completed data for the
population-level figure_soltani_*.py pipeline to mean anything.

That pipeline depends on scripts/build_model_inputs.py, which is still
pointed at the OLD JATOS-era archive/task/parse_results.py output (task/
itself is now fully retired and archived -- see
archive/HISTORY_task_legacy.md) -- no adapter
from Supabase's events table into that pipeline exists yet (see
CLAUDE.md/TODO.md's "Data pipeline (deferred on purpose)" notes). This
script is NOT that adapter -- it's a deliberately lightweight, single-
participant look, since population-level machinery (participant_filters'
exclusion criteria, split-half reliability, individual-differences plots)
is undefined or trivially degenerate at low N anyway. Building the real
adapter is a separate, larger decision for whenever N actually grows
enough for that to matter.

Dedup: keeps the HIGHEST `attempt` row per (trial_index, observation_index)
-- matches this project's own "latest state wins" convention used
elsewhere (e.g. progress-check's own `ORDER BY id DESC LIMIT 1`), since
`attempt` only increments on a timeout-retry of the SAME observation, so
the highest attempt is the authoritative final outcome for that slot
(whether it finally succeeded or is the last failed attempt before
termination).

Reference agents (Bayes, RL_lambda) are computed directly against this
participant's own REAL observed stimulus values -- not a simulated pool
member -- via tiny local copies of scripts/plot_sequences.py's
_bayes_responses/_rl_responses/_ground_truth (duplicated rather than
imported, for the same reason inspect_iid_sequences.py's own local copies
existed: importing plot_sequences.py directly pulls in its NEF/
counting_integrator/MODEL_PARAMS dependency chain for 3 tiny functions).
alpha_0/lambda_ are fixed CLI defaults here, NOT fit to this participant
-- this is a qualitative look, not a rigorous fit (that belongs in the
bigger adapter-pipeline decision above, if it's ever built).

Everything is plotted on the SAME natural 0-100 display scale the
participant's own slider used (not the internal -1..1/0..1 scales
plot_sequences.py's model functions use) -- colors' model outputs
(-1..1) are linearly rescaled to 0-100 (blue %) with NO Laplace-smoothing
transform applied (that transform is a fitting-pipeline convention for a
different dataset naming scheme, not relevant to a raw display-scale
comparison against this participant's own raw slider response).

Usage:
    python scripts/inspect_participant.py --prolific_pid f0079fb --task colors
    python scripts/inspect_participant.py --prolific_pid f0079fb --task colors --alpha_0 1.0 --rl_lambda 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FIGURES_DIR
from utils.plot_style import apply_style, get_palette

TASK_BACKEND_DIR = Path(__file__).resolve().parents[1] / "task_backend"

# task_backend's own naming ('numbers'/'colors') -> the internal
# 'continuous'/'binary' vocabulary the tiny local model functions below
# use (matching plot_sequences.py's own internal convention).
TASK_INTERNAL = {"numbers": "numbers", "colors": "colors"}


def _load_env(path: Path) -> dict:
    """Tiny .env parser (KEY=VALUE lines, '#' comments) -- avoids adding a
    python-dotenv dependency for two files' worth of lookups."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def fetch_participant_events(prolific_pid: str, task: str) -> list:
    """Pulls every phase='trial' row for one (prolific_pid, task) pair
    directly from Supabase's REST API, paginated (PostgREST caps a single
    response at 1000 rows -- confirmed directly this session, not
    assumed)."""
    env = {**_load_env(TASK_BACKEND_DIR / ".env"), **_load_env(TASK_BACKEND_DIR / ".env.test")}
    url_base = env.get("VITE_SUPABASE_URL")
    secret_key = env.get("SUPABASE_SECRET_KEY")
    if not url_base or not secret_key:
        raise RuntimeError(
            "Need VITE_SUPABASE_URL (task_backend/.env) and SUPABASE_SECRET_KEY "
            "(task_backend/.env.test, gitignored -- see .env.test.example)."
        )

    cols = "trial_index,observation_index,attempt,value,response,timed_out,true_mean,true_std,true_p,qid,created_at"
    all_rows, offset, page_size = [], 0, 1000
    while True:
        url = (f"{url_base}/rest/v1/events?prolific_pid=eq.{prolific_pid}&task=eq.{task}"
               f"&phase=eq.trial&select={cols}&order=trial_index.asc,observation_index.asc,attempt.asc"
               f"&limit={page_size}&offset={offset}")
        req = urllib.request.Request(url, headers={"apikey": secret_key, "Authorization": f"Bearer {secret_key}"})
        with urllib.request.urlopen(req) as resp:
            page = json.loads(resp.read())
        all_rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return all_rows


def rows_to_df(rows: list) -> pd.DataFrame:
    """Dedups to the highest `attempt` per (trial_index, observation_index)
    -- see module docstring for why that's the authoritative row."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("attempt").groupby(["trial_index", "observation_index"], as_index=False).last()
    df = df.rename(columns={"trial_index": "trial", "observation_index": "observation"})
    return df.sort_values(["trial", "observation"]).reset_index(drop=True)


# ── Tiny local copies of plot_sequences.py's model functions ───────────────
# Duplicated deliberately, not imported -- see module docstring.

def _obs_norm(value, task):
    return value / 100.0 if task == "continuous" else float(value)


def _clip_response(task, x):
    return float(np.clip(x, 0.0, 1.0)) if task == "continuous" else float(np.clip(x, -1.0, 1.0))


def _ground_truth(task, true_mean, true_p):
    return float(true_mean) / 100.0 if task == "continuous" else float(true_p) * 2.0 - 1.0


def _bayes_responses(values, task):
    if task == "continuous":
        resps, running = [], 0.5
        for n, v in enumerate(values, 1):
            running += (_obs_norm(v, task) - running) / n
            resps.append(_clip_response(task, running))
        return resps
    return [_clip_response(task, float(np.mean(values[:n]))) for n in range(1, len(values) + 1)]


def _rl_responses(values, task, alpha_0=1.0, lambda_=0.5):
    running = 0.5 if task == "continuous" else 0.0
    resps = []
    for n, v in enumerate(values, 1):
        alpha = alpha_0 / (n ** lambda_)
        running = _clip_response(task, running + alpha * (_obs_norm(v, task) - running))
        resps.append(running)
    return resps


def _to_display_scale(task, x):
    """[0,1] (continuous) or [-1,1] (binary) -> 0-100, the same scale the
    participant's own slider used. No Laplace-smoothing transform applied
    (see module docstring)."""
    return x * 100.0 if task == "continuous" else (x + 1.0) * 50.0


def make_figure(df: pd.DataFrame, prolific_pid: str, task: str, out_path: Path,
                 alpha_0: float, rl_lambda: float):
    task_internal = TASK_INTERNAL[task]
    trials = sorted(df["trial"].unique())
    n_trials = len(trials)
    n_cols = 4
    n_rows = int(np.ceil(n_trials / n_cols))
    colors = get_palette(4)

    apply_style()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    overall_rmse = {"Participant": [], "Bayes": [], f"RL({alpha_0},{rl_lambda})": []}

    for i, trial in enumerate(trials):
        ax = axes[i // n_cols][i % n_cols]
        g = df[df["trial"] == trial].sort_values("observation")
        values = g["value"].tolist()
        responses = g["response"].tolist()
        obs = g["observation"].tolist()
        tm, tp = g["true_mean"].iloc[0], g["true_p"].iloc[0]
        gt = _ground_truth(task_internal, tm, tp)
        gt_display = _to_display_scale(task_internal, gt)

        bayes = [_to_display_scale(task_internal, r) for r in _bayes_responses(values, task_internal)]
        rl = [_to_display_scale(task_internal, r) for r in _rl_responses(values, task_internal, alpha_0, rl_lambda)]

        ax.axhline(gt_display, color="0.4", lw=1, ls="--", label="Ground truth")
        ax.plot(obs, responses, color=colors[0], lw=2, marker="o", ms=3, label="Participant")
        ax.plot(obs, bayes, color=colors[1], lw=1.3, alpha=0.8, label="Bayes")
        ax.plot(obs, rl, color=colors[2], lw=1.3, alpha=0.8, label=f"RL({alpha_0},{rl_lambda})")
        ax.set_title(f"Trial {trial} (qid={int(g['qid'].iloc[0])})", fontsize=8)
        ax.set_ylim(-5, 105)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6, frameon=False)

        n = len(responses)
        overall_rmse["Participant"].append(np.sqrt(np.mean((np.array(responses) - gt_display) ** 2)))
        overall_rmse["Bayes"].append(np.sqrt(np.mean((np.array(bayes) - gt_display) ** 2)))
        overall_rmse[f"RL({alpha_0},{rl_lambda})"].append(np.sqrt(np.mean((np.array(rl) - gt_display) ** 2)))

    for j in range(n_trials, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"{prolific_pid} -- {task} ({n_trials} trials, real responses vs. reference agents)",
                fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    print("\nMean per-trial RMSE (0-100 display scale, vs. true_mean/true_p):")
    for name, vals in overall_rmse.items():
        print(f"  {name:<20} {np.mean(vals):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prolific_pid", required=True)
    p.add_argument("--task", required=True, choices=["numbers", "colors"])
    p.add_argument("--alpha_0", type=float, default=1.0)
    p.add_argument("--rl_lambda", type=float, default=0.5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"Fetching {args.prolific_pid} / {args.task} from Supabase...")
    rows = fetch_participant_events(args.prolific_pid, args.task)
    df = rows_to_df(rows)
    if df.empty:
        print(f"No trial-phase rows found for ({args.prolific_pid}, {args.task}).")
        return
    print(f"{len(df)} distinct (trial, observation) rows across {df['trial'].nunique()} trials")

    out = Path(args.out) if args.out else FIGURES_DIR / f"inspect_participant_{args.prolific_pid}_{args.task}.pdf"
    make_figure(df, args.prolific_pid, args.task, out, args.alpha_0, args.rl_lambda)
    print("JOB_COMPLETE")


if __name__ == "__main__":
    main()
