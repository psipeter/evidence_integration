#!/usr/bin/env python3
"""figure_soltani_performance.py — P group figure for the soltani task/ pilot
(task-continuous + task-binary, "Human Mixed Task" / Soltani lab, Prolific pilot 1).

Layout: 2x3
  Row 1 = task-binary, Row 2 = task-continuous
  Col 1 : Task schematic (placeholder — no schematic PDF exists yet)
  Col 2 (P1): Estimation error — RMSE to ground truth, per prolific_pid
  Col 3 (P2): Model fit — RMSE to human responses, per prolific_pid

PLACEHOLDER NOTE
----------------
No models have been fit to this pilot data yet. As a stand-in, the Mean
model (optimal running average of the observed stimulus stream) is
simulated directly here, since it is fully deterministic and needs no
fitting. Column 2 shows Human and Mean; column 3 shows Mean only (there is
no other fitted model yet to compare it against). This entire figure
should be regenerated from the real fitting pipeline (fitting/submit.py +
fitting/collect.py, mirroring figure_yoo_performance.py /
figure_carrabin_performance.py) once carrabin/yoo-style model fits exist
for this dataset.

Ground truth definition (per row):
  - continuous: true_mean, the fixed generative Normal mean for the trial
    (already on the same [0,100] scale as value/response — no rescaling).
  - binary: true_p (fixed generative Bernoulli probability, [0,1]) scaled
    to the response's [0,100] percent scale via true_p * 100.
This mirrors carrabin's use of a fixed per-trial generative parameter
(rather than yoo's expanding-sample-mean ground truth), since — like
carrabin — every trial here has a genuine fixed hidden parameter recorded
directly in the data (see CLAUDE.md's "Sequences.json schema" section).

Mean model definition (per row):
  - continuous: expanding (cumulative) mean of the observed `value` stream
    per (prolific_pid, trial), directly comparable to response/true_mean.
    No Laplace smoothing — utils/binary_transform.py explicitly exempts
    task_continuous ("uses raw responses like yoo").
  - binary: expanding mean of `value` (coded -1/1), then Laplace-smoothed
    toward the uninformative prior via response = raw * t/(t+2)
    (utils/binary_transform.apply_binary_transform, dataset='task_binary';
    Mean is not in that module's _EXEMPT_MODELS, so the shrinkage applies
    the same way it would for any other non-NoisyCounting model), then
    rescaled to a [0,100] percent-blue estimate via (smoothed + 1) / 2 * 100,
    comparable to response/true_p*100.
Timeout retries replay the same observation index and show the same
stimulus value, so the value stream is deduplicated to one row per
(prolific_pid, trial, observation) before computing the running mean;
human RMSE is computed only over successful (timed_out == False) responses.

Run:
    python scripts/figure_soltani_performance.py
    python scripts/figure_soltani_performance.py --results_file task_results_pilot1.pkl
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

from utils.paths import FIGURES_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels
from utils.binary_transform import apply_binary_transform

TASK_ROWS   = ["binary", "continuous"]  # row order convention: binary on top,
                                          # continuous on bottom (for this and
                                          # future soltani figures)
HUMAN_COLOR = "0.3"


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


# ── data prep ────────────────────────────────────────────────────────────────

def _prepare_task_df(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Merge human responses (successful attempts only) with a Mean-model
    running-average response and the appropriate ground truth, all on a
    common [0,100] scale. Returns one row per (prolific_pid, trial, obs)."""
    sub = df[df["task"] == task].copy()

    # Stimulus value stream: dedup so timeout retries (same obs index, same
    # value) don't get double-counted in the running mean.
    vals = (sub[["prolific_pid", "trial", "observation", "value"]]
            .drop_duplicates(subset=["prolific_pid", "trial", "observation"])
            .sort_values(["prolific_pid", "trial", "observation"]))
    vals["model_mean"] = (vals.groupby(["prolific_pid", "trial"])["value"]
                               .expanding().mean().values)
    if task == "binary":
        # Laplace-smooth the raw [-1,1] running mean toward the uninformative
        # prior of 0 before converting to a percent-blue estimate — matches
        # utils/binary_transform.py's task_binary convention (Mean is not in
        # its _EXEMPT_MODELS set, so the shrinkage applies). task_continuous
        # is explicitly exempt in that module ("uses raw responses like
        # yoo"), so the continuous branch below is left untransformed.
        smoothed = apply_binary_transform(
            vals[["observation"]].assign(response=vals["model_mean"]),
            "task_binary",
        )
        vals["model_response"] = (smoothed["response"].to_numpy() + 1) / 2 * 100
    else:
        vals["model_response"] = vals["model_mean"]

    human = (sub[sub["timed_out"] == False]
             [["prolific_pid", "trial", "observation", "response",
               "true_mean", "true_p"]]
             .drop_duplicates(subset=["prolific_pid", "trial", "observation"]))

    merged = human.merge(
        vals[["prolific_pid", "trial", "observation", "model_response"]],
        on=["prolific_pid", "trial", "observation"], how="left",
    )
    merged["ground_truth"] = (merged["true_mean"] if task == "continuous"
                               else merged["true_p"] * 100)
    return merged


# ── Panel P1 — Estimation error ─────────────────────────────────────────────

def _plot_panel_p1(ax, merged: pd.DataFrame, palette: dict) -> None:
    """RMSE to ground truth, per prolific_pid, for Human and Mean model."""
    rows = []
    human_rmse = (merged.assign(sq_err=(merged["response"] - merged["ground_truth"]) ** 2)
                  .groupby("prolific_pid")["sq_err"].mean().apply(np.sqrt)
                  .reset_index(name="rmse"))
    human_rmse["source"] = "Human"
    rows.append(human_rmse)

    mean_rmse = (merged.assign(sq_err=(merged["model_response"] - merged["ground_truth"]) ** 2)
                 .groupby("prolific_pid")["sq_err"].mean().apply(np.sqrt)
                 .reset_index(name="rmse"))
    mean_rmse["source"] = "Mean"
    rows.append(mean_rmse)

    plot_df = pd.concat(rows, ignore_index=True)
    order = ["Human", "Mean"]
    pal = {"Human": HUMAN_COLOR, "Mean": palette.get("Mean", "0.5")}

    sns.boxplot(data=plot_df, x="source", y="rmse", order=order,
                hue="source", palette=pal, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Performance error vs ground truth (RMSE)")
    sns.despine(ax=ax, top=True, right=True)


# ── Panel P2 — Model fit ────────────────────────────────────────────────────

def _plot_panel_p2(ax, merged: pd.DataFrame, palette: dict) -> None:
    """RMSE to human responses, per prolific_pid, for the Mean model only
    (placeholder — no other fitted model exists yet)."""
    fit_rmse = (merged.assign(sq_err=(merged["model_response"] - merged["response"]) ** 2)
                .groupby("prolific_pid")["sq_err"].mean().apply(np.sqrt)
                .reset_index(name="rmse"))
    fit_rmse["source"] = "Mean"

    order = ["Mean"]
    pal = {"Mean": palette.get("Mean", "0.5")}

    sns.boxplot(data=fit_rmse, x="source", y="rmse", order=order,
                hue="source", palette=pal, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Model fit (RMSE to human responses)")
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="task_results_pilot1.pkl",
                        help="Filename under data/ produced by task/parse_results.py")
    args = parser.parse_args()

    df = pd.read_pickle(data_path(args.results_file))

    apply_style()
    pal = get_palette(2)
    palette = {"Human": HUMAN_COLOR, "Mean": pal[0]}

    fig, axes = plt.subplots(
        2, 3,
        figsize=(FIGURE_SIZE[0] * 0.75, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    for row, task in enumerate(TASK_ROWS):
        merged = _prepare_task_df(df, task)
        _plot_schematic(axes[row, 0], task)
        _plot_panel_p1(axes[row, 1], merged, palette)
        _plot_panel_p2(axes[row, 2], merged, palette)
        axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")

    label_panels(axes)

    fig.text(0.5, -0.02,
              "PLACEHOLDER: only the deterministic Mean model is simulated here (no "
              "fitting run yet). Replace with the full carrabin/yoo-style model-fit "
              "pipeline once fits exist for this dataset.",
              ha="center", va="top", fontsize=7, style="italic", color="0.4")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_soltani_performance"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
