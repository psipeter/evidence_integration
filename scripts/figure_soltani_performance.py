#!/usr/bin/env python3
"""figure_soltani_performance.py — P group figure for the soltani task/
pilot (task-continuous + task-binary, "Human Mixed Task" / Soltani lab).

Layout: 2x2
  Row 1 = task-binary, Row 2 = task-continuous
  Col 1 : Task schematic (placeholder -- no schematic PDF exists yet)
  Col 2 (P1): Estimation error -- RMSE to the RUNNING MEAN of the
    observed stimulus stream (NOT the fixed generative true_mean/true_p
    -- see _add_running_mean_ground_truth's own docstring for why), per pid

HUMAN DATA ONLY, DELIBERATELY -- no model fitting or model-fit panel here
right now (see chat history). The old version of this file sourced fitted
model responses from a fitting.submit/fitting.collect run folder
(data/runs/{run_folder}/{model_type}_{dataset}_responses.pkl) and plotted
a second "model fit" panel (RMSE to human responses) alongside this one --
removed for now, not because the approach was wrong, but because model
fitting against task_backend's real data hasn't been run yet and doing so
properly (Optuna + k-fold CV per pid/model/dataset) is real, separate work
worth doing in its own pass rather than half-integrating here.

FUTURE RE-INTEGRATION: once model fitting is actually done and saved to
its own dataframes, add it back as its own loading function (mirroring
_load_human's shape below: a DataFrame with [pid, trial, observation,
<a response-like column>]) and a second panel that merges it against
_load_human's own output on [pid, trial, observation], same as the old
model-fit panel did. Whatever format the future fitting step saves to,
converting it into that shape is the only real integration work -- this
file's own plotting logic (RMSE aggregation, boxplot-by-source) doesn't
need to change.

DATA SOURCE AND SCALE
----------------------
Human data comes directly from data/task_continuous.pkl / data/
task_binary.pkl -- built by scripts/build_task_backend_inputs.py (pulls
real, finished participants directly from task_backend's Supabase
`events` table) via scripts/build_model_inputs.py's own build_from_df()
(shared filter+rescale+anonymize+save pipeline -- participant filtering
and the prolific_pid -> int pid mapping already happened when those files
were built, so this script does not re-apply utils.participant_filters or
see any real prolific_pid at all).

Those files store value/response on the canonical [-1,1] scale carrabin/
yoo also use, and true_mean rescaled the same way but true_p left on its
native [0,1] probability scale (see build_model_inputs.py's own module
docstring for the exact rescaling). Converted back to the original
[0,100] percent scale here purely for readability:
  continuous : pct = (x + 1) * 50
  binary     : pct = (x + 1) / 2 * 100
Ground truth is NOT true_mean/true_p at all (see
_add_running_mean_ground_truth's own docstring) -- it's the running mean
of `value` itself, put through the same pct conversion.

Run:
    python scripts/figure_soltani_performance.py
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
from utils.plot_style import FIGURE_SIZE, apply_style, label_panels

TASK_ROWS   = ["binary", "continuous"]  # row order convention: binary on top,
                                          # continuous on bottom
HUMAN_COLOR = "0.3"
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

def _load_human(task: str, datafile: str | None) -> pd.DataFrame:
    """Human data + ground truth for one task, on the [0,100] percent scale.
    Columns: [pid, trial, observation, value, response, ground_truth].
    ground_truth is the RUNNING mean of `value` (see
    _add_running_mean_ground_truth below), NOT the fixed true_mean/
    true_p -- see that function's own docstring for why. Kept as its own
    small function (not inlined into the panel below) because the FUTURE
    model-loading function this file will eventually get is meant to
    mirror this exact shape -- see module docstring.

    datafile: optional suffix (e.g. 'pilot4', 'pilot5') appended to the
    dataset stem -- data/task_{continuous,binary}_{datafile}.pkl instead
    of the canonical data/task_{continuous,binary}.pkl. Lets this figure
    point at any pilot round's own files (built by
    scripts/build_task_backend_inputs.py's --pilot flag) without needing
    a pilot-specific concept baked into this script -- it's just a
    filename suffix, so it works the same way once there's a real
    (non-pilot) experiment dataset to point at too."""
    dataset = DATASET_FOR_TASK[task]
    if datafile:
        dataset = f"{dataset}_{datafile}"
    path = data_path(f"{dataset}.pkl")
    if not path.exists():
        return None
    df = pd.read_pickle(path)
    out = df[["pid", "trial", "observation", "value"]].copy()
    out["response"] = _to_pct(df["response"], task)
    out = _add_running_mean_ground_truth(out, task)
    return out


def _add_running_mean_ground_truth(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Ground truth = the RUNNING mean/ratio of the observed stimulus
    stream itself, per (pid, trial) -- i.e. what a perfect 'just average
    what you've seen so far' agent would report at each observation --
    NOT the fixed generative true_mean/true_p. Matches the same
    running_mean/running_p convention already established elsewhere in
    this project (scripts/plot_sequences.py's own gt_mode='running_mean';
    task_backend's own live 'correct answer' panel shows real
    participants exactly this quantity during the actual task, never the
    fixed target). Requires `value` (raw stimulus, native pkl scale --
    NOT yet through _to_pct) already present in df."""
    df = df.sort_values(["pid", "trial", "observation"]).copy()
    running = df.groupby(["pid", "trial"])["value"].transform(lambda s: s.expanding().mean())
    df["ground_truth"] = _to_pct(running, task)
    return df


# ── Panel P1 — Estimation error ─────────────────────────────────────────────

def _plot_panel_p1(ax, human: pd.DataFrame) -> None:
    """RMSE to the running-mean ground truth, per pid, shown as a violin
    plot. See module docstring for how a model column would slot in here
    later (another violin, one per source)."""
    mean_sq_err = (human.assign(sq_err=(human["response"] - human["ground_truth"]) ** 2)
                   .groupby("pid")["sq_err"].mean())
    human_rmse = np.sqrt(mean_sq_err).reset_index(name="rmse")
    human_rmse["source"] = "Human"

    sns.violinplot(data=human_rmse, x="source", y="rmse", order=["Human"],
                  hue="source", palette={"Human": HUMAN_COLOR}, legend=False, ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("Performance error vs running mean (RMSE)")
    ax.tick_params(axis="x", rotation=30)
    sns.despine(ax=ax, top=True, right=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", default=None,
                       help="Suffix identifying which dataset to load, e.g. 'pilot4' -> "
                            "data/task_continuous_pilot4.pkl / task_binary_pilot4.pkl. "
                            "Omit to use the canonical data/task_continuous.pkl / "
                            "task_binary.pkl.")
    args = parser.parse_args()

    apply_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIGURE_SIZE[0] * 0.5, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    for row, task in enumerate(TASK_ROWS):
        print(f"task-{task}:")
        human = _load_human(task, args.datafile)
        if human is None:
            print(f"  no data file found for this task/datafile combination -- skipping")
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            axes[row, 1].text(0.5, 0.5, f"No {task} data\nfor this dataset",
                             ha="center", va="center", transform=axes[row, 1].transAxes,
                             color="0.5", style="italic")
            axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")
            continue
        print(f"  {len(human)} rows, {human['pid'].nunique()} pids")

        _plot_schematic(axes[row, 0], task)
        _plot_panel_p1(axes[row, 1], human)
        axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")

    label_panels(axes)

    fig.text(0.5, -0.02,
              "Human data only -- model fits not yet run against task_backend's "
              "real data (see this script's own module docstring).",
              ha="center", va="top", fontsize=7, style="italic", color="0.4")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_soltani_performance"
    if args.datafile:
        stem = f"{stem}_{args.datafile}"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
