#!/usr/bin/env python3
"""figure_soltani_performance.py — P group figure for the soltani task/ pilot
(task-continuous + task-binary, "Human Mixed Task" / Soltani lab).

Layout: 2x3
  Row 1 = task-binary, Row 2 = task-continuous
  Col 1 : Task schematic (placeholder — no schematic PDF exists yet)
  Col 2 (P1): Estimation error — RMSE to ground truth, per pid
  Col 3 (P2): Model fit — RMSE to human responses, per pid

Sources real fitted model responses from a fitting.submit/fitting.collect
run folder (default "soltani_math_v1"), mirroring figure_carrabin_
performance.py / figure_yoo_performance.py's own loading pattern, rather
than the earlier placeholder version's ad-hoc, un-fitted Mean-model
simulation. Models plotted (per the plan that produced this integration):
Mean, LeakyIntegrator, PrimacyRecency, RL_lambda — chosen together to
capture recency-biased (non-shrinking-learning-rate) behavior; NEF is not
included (not part of this phase).

DATA SOURCE AND SCALE
----------------------
Human data comes directly from data/task_continuous.pkl / data/
task_binary.pkl (built by scripts/build_model_inputs.py), NOT from a raw
task_results_pilot*.pkl — participant filtering and the prolific_pid ->
int pid mapping already happened when those files were built, so this
script does not re-apply utils.participant_filters itself.

Those files store value/response on the canonical [-1,1] scale carrabin/
yoo also use (see build_model_inputs.py), and true_mean rescaled the same
way but true_p left on its native [0,1] probability scale. Fitted model
responses (data/runs/{run_folder}/{model_type}_{dataset}_responses.pkl)
are on that same [-1,1] scale, since models/math_models.py operates
directly on data/task_continuous.pkl/task_binary.pkl. Everything is
converted back to the original [0,100] percent scale here purely for
readability/continuity with the earlier placeholder figure's units:
  continuous : pct = (x + 1) * 50
  binary     : pct = (x + 1) / 2 * 100   (x already Laplace-transformed
               where relevant -- both Human's stored `response` and each
               model's stored `response` column are already the values to
               compare directly; ground truth true_p is *100 with no +1
               shift, since it's a genuine [0,1] probability, never put
               through the [-1,1] rescale.)

Run:
    python scripts/figure_soltani_performance.py
    python scripts/figure_soltani_performance.py --run_folder soltani_math_v1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import FIGURES_DIR, data_path, resolve_run_folder
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels

TASK_ROWS   = ["binary", "continuous"]  # row order convention: binary on top,
                                          # continuous on bottom
HUMAN_COLOR = "0.3"
MODEL_ORDER = ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda"]
DATASET_FOR_TASK = {"binary": "task_binary", "continuous": "task_continuous"}


# ── schematic placeholders (col 1) ──────────────────────────────────────────

def _plot_schematic(ax, task: str) -> None:
    """Render figures/soltani_{task}_task.pdf if it exists; otherwise a
    text placeholder. No such schematic exists yet for either task."""
    pdf_path = FIGURES_DIR / f"soltani_{task}_task.pdf"
    if not pdf_path.exists():
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 0.5, f"soltani_{task}_task.pdf\nnot found",
                ha="center", va="center", transform=ax.transAxes,
                color="0.5", style="italic", fontsize=8)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / f"soltani_{task}_task"
        cmd = ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(out_prefix)]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return
        img = mpimg.imread(img_path)

    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal"); ax.set_anchor("C")


# ── scale conversion back to [0,100] for readability ────────────────────────

def _to_pct(x: pd.Series, task: str) -> pd.Series:
    if task == "binary":
        return (x + 1.0) / 2.0 * 100.0
    return (x + 1.0) * 50.0


# ── data loading ─────────────────────────────────────────────────────────────

def _load_human(task: str) -> pd.DataFrame:
    """Human data + ground truth for one task, on the [0,100] percent scale.
    Columns: [pid, trial, observation, response, ground_truth]."""
    dataset = DATASET_FOR_TASK[task]
    df = pd.read_pickle(data_path(f"{dataset}.pkl"))
    out = df[["pid", "trial", "observation"]].copy()
    out["response"] = _to_pct(df["response"], task)
    out["ground_truth"] = (df["true_p"] * 100.0 if task == "binary"
                           else _to_pct(df["true_mean"], task))
    return out


def _load_model(task: str, model_type: str, run_dir: Path) -> pd.DataFrame | None:
    """Fitted model responses for one (task, model_type), on the [0,100]
    percent scale. Returns None if the collected responses file doesn't
    exist yet (e.g. a model still being fit). Columns:
    [pid, trial, observation, model_response]."""
    dataset = DATASET_FOR_TASK[task]
    resp_path = run_dir / f"{model_type}_{dataset}_responses.pkl"
    if not resp_path.exists():
        print(f"  (missing {resp_path.name} -- skipping {model_type} for {task})")
        return None
    df = pd.read_pickle(resp_path)
    out = df[["pid", "trial", "observation"]].copy()
    out["model_response"] = _to_pct(df["response"], task)
    return out


# ── Panel P1 — Estimation error ─────────────────────────────────────────────

def _plot_panel_p1(ax, human: pd.DataFrame, models: dict[str, pd.DataFrame],
                   palette: dict) -> None:
    """RMSE to ground truth, per pid, for Human + each fitted model."""
    rows = []
    human_rmse = (human.assign(sq_err=(human["response"] - human["ground_truth"]) ** 2)
                  .groupby("pid")["sq_err"].mean().apply(np.sqrt)
                  .reset_index(name="rmse"))
    human_rmse["source"] = "Human"
    rows.append(human_rmse)

    for model_type, mdf in models.items():
        merged = mdf.merge(human[["pid", "trial", "observation", "ground_truth"]],
                           on=["pid", "trial", "observation"])
        rmse = (merged.assign(sq_err=(merged["model_response"] - merged["ground_truth"]) ** 2)
                .groupby("pid")["sq_err"].mean().apply(np.sqrt)
                .reset_index(name="rmse"))
        rmse["source"] = model_type
        rows.append(rmse)

    plot_df = pd.concat(rows, ignore_index=True)
    order = ["Human"] + list(models.keys())
    pal = {"Human": HUMAN_COLOR, **{m: palette[m] for m in models}}

    sns.boxplot(data=plot_df, x="source", y="rmse", order=order,
                hue="source", palette=pal, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Performance error vs ground truth (RMSE)")
    ax.tick_params(axis="x", rotation=30)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel P2 — Model fit ────────────────────────────────────────────────────

def _plot_panel_p2(ax, human: pd.DataFrame, models: dict[str, pd.DataFrame],
                   palette: dict) -> None:
    """RMSE to human responses, per pid, for each fitted model."""
    rows = []
    for model_type, mdf in models.items():
        merged = mdf.merge(human[["pid", "trial", "observation", "response"]],
                           on=["pid", "trial", "observation"])
        rmse = (merged.assign(sq_err=(merged["model_response"] - merged["response"]) ** 2)
                .groupby("pid")["sq_err"].mean().apply(np.sqrt)
                .reset_index(name="rmse"))
        rmse["source"] = model_type
        rows.append(rmse)

    if not rows:
        ax.text(0.5, 0.5, "No fitted models found", ha="center", va="center",
                transform=ax.transAxes, color="0.5", style="italic")
        return

    plot_df = pd.concat(rows, ignore_index=True)
    order = list(models.keys())
    pal = {m: palette[m] for m in models}

    sns.boxplot(data=plot_df, x="source", y="rmse", order=order,
                hue="source", palette=pal, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Model fit (RMSE to human responses)")
    ax.tick_params(axis="x", rotation=30)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="soltani_math_v1",
                        help="Folder under data/runs/ with fitting.submit + "
                             "fitting.collect output")
    args = parser.parse_args()

    run_dir = resolve_run_folder(args.run_folder)
    apply_style()
    pal = get_palette(len(MODEL_ORDER))
    palette = {m: pal[i] for i, m in enumerate(MODEL_ORDER)}

    fig, axes = plt.subplots(
        2, 3,
        figsize=(FIGURE_SIZE[0] * 0.75, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    for row, task in enumerate(TASK_ROWS):
        print(f"task-{task}:")
        human = _load_human(task)
        models = {}
        for model_type in MODEL_ORDER:
            mdf = _load_model(task, model_type, run_dir)
            if mdf is not None:
                models[model_type] = mdf

        _plot_schematic(axes[row, 0], task)
        _plot_panel_p1(axes[row, 1], human, models, palette)
        _plot_panel_p2(axes[row, 2], human, models, palette)
        axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")

    label_panels(axes)

    fig.text(0.5, -0.02,
              f"Model fits: {', '.join(MODEL_ORDER)} from run '{args.run_folder}' "
              "(RMSE-based Optuna + k-fold CV). NEF not yet included.",
              ha="center", va="top", fontsize=7, style="italic", color="0.4")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_soltani_performance"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
