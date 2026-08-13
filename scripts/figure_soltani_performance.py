#!/usr/bin/env python3
"""figure_soltani_performance.py — P group figure for the soltani experiment
(soltani_numbers + soltani_colors, "Human Mixed Task" / Soltani lab).

Layout: 2x3
  Row 1 = task-colors, Row 2 = task-numbers
  Col 1      : Task schematic (placeholder -- no schematic PDF exists yet)
  Col 2 (P1) : Estimation error -- RMSE to the RUNNING MEAN of the observed
    stimulus stream (NOT the fixed generative true_mean/true_p -- see
    _add_running_mean_ground_truth's own docstring for why), per pid, for
    Human AND each fitted model. Answers "how close does each source get to
    the optimal running-average answer?"

    EXPECTED, NOT A BUG: the Mean model scores EXACTLY 0 here. Mean *is* the
    running mean and the ground truth *is* the running mean, so its error is
    zero by construction. Kept deliberately (settled decision, not an
    oversight) -- it doubles as a live check that math_models' Mean and
    _add_running_mean_ground_truth agree to floating-point precision, so a
    non-zero Mean violin is a real signal that one of them has drifted. Note
    this differs from figure_carrabin_performance.py's P1, where the ground
    truth is the fixed true_p and Mean is therefore NOT degenerate.
  Col 3 (P2) : Model fit -- RMSE to HUMAN responses, per pid, per model.
    Answers the different question "how well does each model reproduce what
    people actually did?" Significance bars are drawn from SIG_REFERENCE
    outward (see below).

Both P1 and P2 use violins rather than boxplots/rugplots: at this sample size
(9 pids in pilot 5) a boxplot's quartiles are estimated from too few points to
read as anything but noise, while the violin's density still communicates
spread honestly. Kept consistent across both panels so they're comparable.

P1 vs P2 SHARE UNITS (percentage points)
-----------------------------------------
P1 is computed in [0,100] percent units directly (see _to_pct). P2's numbers
come from each model's own fitted CV loss, which lives on the canonical [-1,1]
scale -- multiplied by LOSS_TO_PCT to match. That factor is exactly 50 for BOTH
tasks, which is not a coincidence worth glossing over: numbers converts with
pct = (x+1)*50 and colors with pct = (x+1)/2*100 = 50x+50, so although the two
tasks have different response semantics, both are affine with the same slope of
50, and an RMSE (a difference, so the intercept drops out) therefore scales by
50 either way.

WHY P2 READS THE FITTED CV LOSS RATHER THAN RECOMPUTING RMSE
--------------------------------------------------------------
The fitted loss in {model}_{stem}_performance.pkl is the k-fold
cross-validated RMSE to human responses -- i.e. held-out. Recomputing RMSE
from the collected _responses.pkl instead would be an IN-SAMPLE number, which
flatters models with more free parameters (PrimacyRecency's 2, RL_lambda's 2)
relative to Mean's 0. Matches figure_carrabin_performance.py's own panel B,
which reads performance.pkl for the same reason. Read via _get_loss, never by
hardcoding a column name.

SIGNIFICANCE BARS
-----------------
`annotate_nef_comparisons` (utils/plot_style.py) is generic despite its name --
it draws paired-Wilcoxon bars from ONE reference model outward to the others,
taking the reference as a parameter. SIG_REFERENCE is NEF, matching the
carrabin/yoo figures. (It was RL_lambda while NEF was unwired for these
datasets -- the explicit power-law delta rule the NEF is the spiking analogue
of, and the closest standing proxy. Switching it back is a one-line change if
NEF fits are unavailable, since a missing reference simply suppresses the bars
rather than erroring.)

DATA SOURCE AND SCALE
----------------------
Human data comes from data/soltani_{numbers,colors}[_<datafile>].pkl -- built
by scripts/build_task_backend_inputs.py (pulls real, finished participants
directly from task_backend's Supabase `events` table) via
scripts/build_model_inputs.py's own build_from_df() (shared
filter+rescale+anonymize+save pipeline -- participant filtering and the
prolific_pid -> int pid mapping already happened when those files were built,
so this script does not re-apply utils.participant_filters or see any real
prolific_pid at all).

Model data comes from data/runs/{run_folder}/{model_type}_{stem}_*.pkl, where
`stem` is utils.paths.dataset_stem(dataset, datafile) -- the SAME suffix the
human data was loaded with. That is what guarantees a model's responses were
actually fit against the human data plotted beside them; see dataset_stem's own
docstring for the mismatch this prevents.

Those files store value/response on the canonical [-1,1] scale carrabin/yoo
also use, and true_mean rescaled the same way but true_p left on its native
[0,1] probability scale (see build_model_inputs.py's own module docstring for
the exact rescaling). Converted back to the original [0,100] percent scale here
purely for readability:
  numbers : pct = (x + 1) * 50
  colors  : pct = (x + 1) / 2 * 100
Ground truth is NOT true_mean/true_p at all (see
_add_running_mean_ground_truth's own docstring) -- it's the running mean of
`value` itself, put through the same pct conversion.

Run:
    python scripts/figure_soltani_performance.py
    python scripts/figure_soltani_performance.py --datafile pilot5
    python scripts/figure_soltani_performance.py --datafile pilot4 --run_folder soltani
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

from utils.paths import FIGURES_DIR, data_path, dataset_stem, resolve_run_folder
from utils.plot_style import (
    FIGURE_SIZE,
    annotate_nef_comparisons,
    apply_style,
    get_palette,
    label_panels,
)

TASK_ROWS   = ["colors", "numbers"]  # row order convention: colors on top,
                                     # numbers on bottom
HUMAN_COLOR = "0.3"
DATASET_FOR_TASK = {"colors": "soltani_colors", "numbers": "soltani_numbers"}

# Same four models, in the same order, as figure_soltani_temporal.py -- so a
# model keeps one colour across the whole soltani figure set.
MODEL_ORDER = ["Mean", "LeakyIntegrator", "PrimacyRecency", "RL_lambda", "NEF"]

# Reference model for P2's significance bars; see module docstring. NEF now that
# it is fit for these datasets -- matching the carrabin/yoo figures, where the
# spiking model is always the reference the others are compared against.
SIG_REFERENCE = "NEF"

# [-1,1]-scale RMSE -> percentage points. Exactly 50 for both tasks; see
# module docstring for why that holds despite their different response scales.
LOSS_TO_PCT = 50.0


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
    if task == "colors":
        return (x + 1.0) / 2.0 * 100.0
    return (x + 1.0) * 50.0


# ── data loading ─────────────────────────────────────────────────────────────

def _load_human(task: str, datafile: str | None) -> pd.DataFrame | None:
    """Human data + ground truth for one task, on the [0,100] percent scale.
    Columns: [pid, trial, observation, value, response, ground_truth].
    ground_truth is the RUNNING mean of `value` (see
    _add_running_mean_ground_truth below), NOT the fixed true_mean/true_p --
    see that function's own docstring for why.

    datafile: optional suffix (e.g. 'pilot4', 'pilot5') appended to the dataset
    stem -- data/soltani_{numbers,colors}_{datafile}.pkl instead of the
    canonical data/soltani_{numbers,colors}.pkl. Lets this figure point at any
    round's own files (built by scripts/build_task_backend_inputs.py's --pilot
    flag) without a round-specific concept baked into this script -- it's just
    a filename suffix, so it works the same way for the real production
    dataset. Returns None if that task has no file at all for this datafile
    (e.g. a numbers-only round has no colors file)."""
    dataset = dataset_stem(DATASET_FOR_TASK[task], datafile)
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


def _load_model_responses(task: str, model_type: str, run_dir: Path,
                          datafile: str | None) -> pd.DataFrame | None:
    """Fitted model responses for one (task, model_type), on the [0,100]
    percent scale. Columns: [pid, trial, observation, response]. Returns None
    if not yet fit/collected.

    `datafile` must be the same suffix _load_human was given -- fits are named
    after the dataset STEM, so passing it is what guarantees these responses
    were fit against the human data they'll be plotted against."""
    stem = dataset_stem(DATASET_FOR_TASK[task], datafile)
    resp_path = run_dir / f"{model_type}_{stem}_responses.pkl"
    if not resp_path.exists():
        return None
    df = pd.read_pickle(resp_path)
    out = df[["pid", "trial", "observation"]].copy()
    out["response"] = _to_pct(df["response"], task)
    return out


def _get_loss(perf_df: pd.DataFrame) -> pd.Series:
    """Never hardcode cv_loss_mean -- project convention."""
    if "loss" in perf_df.columns:
        return perf_df["loss"]
    return perf_df["cv_loss_mean"]


def _load_model_loss(task: str, model_type: str, run_dir: Path,
                     datafile: str | None) -> pd.DataFrame | None:
    """Per-pid fitted CV loss (RMSE to human responses) for one
    (task, model_type), converted to percentage points. Columns: [pid, rmse].
    Returns None if not yet fit/collected."""
    stem = dataset_stem(DATASET_FOR_TASK[task], datafile)
    perf_path = run_dir / f"{model_type}_{stem}_performance.pkl"
    if not perf_path.exists():
        return None
    perf = pd.read_pickle(perf_path).copy()
    perf["rmse"] = _get_loss(perf) * LOSS_TO_PCT
    return perf[["pid", "rmse"]]


# ── Panel P1 — Estimation error ─────────────────────────────────────────────

def _rmse_per_pid(df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Per-pid RMSE of `response` against `ground_truth`, collapsing over that
    pid's own trials and observations. Shared by Human and every model, so all
    violins in P1 use an identical hierarchy and are directly comparable."""
    merged = df.merge(
        ground_truth[["pid", "trial", "observation", "ground_truth"]],
        on=["pid", "trial", "observation"],
    )
    return (merged.assign(sq_err=(merged["response"] - merged["ground_truth"]) ** 2)
            .groupby("pid")["sq_err"].mean()
            .apply(np.sqrt).reset_index(name="rmse"))


def _plot_panel_p1(ax, human: pd.DataFrame, models: dict[str, pd.DataFrame],
                   palette: dict) -> None:
    """RMSE to the running-mean ground truth, per pid, as one violin per
    source (Human first, then each fitted model).

    Mean lands at exactly 0 by construction -- see the module docstring's own
    note under Col 2. Do not "fix" it by excluding Mean or by switching this
    panel's ground truth.
    """
    frames = []
    h = _rmse_per_pid(human[["pid", "trial", "observation", "response"]], human)
    h["source"] = "Human"
    frames.append(h)

    for model_type in MODEL_ORDER:
        mdf = models.get(model_type)
        if mdf is None:
            continue
        m = _rmse_per_pid(mdf, human)
        m["source"] = model_type
        frames.append(m)

    plot_df = pd.concat(frames, ignore_index=True)
    order = [s for s in ["Human"] + MODEL_ORDER if s in set(plot_df["source"])]

    sns.violinplot(data=plot_df, x="source", y="rmse", order=order,
                   hue="source", palette=palette, legend=False, ax=ax,
                   cut=0)

    ax.set_xlabel("")
    ax.set_ylabel("Performance error vs running mean\n(RMSE, percentage points)")
    ax.tick_params(axis="x", rotation=30)
    # RMSE is non-negative, so neither the KDE tails nor the axis should imply
    # negative error. cut=0 truncates each violin at its own observed min/max
    # instead of letting the Gaussian kernel run past the data; the explicit
    # floor then stops the axis itself from padding below 0.
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)


# ── Panel P2 — Model fit to human responses ─────────────────────────────────

def _plot_panel_p2(ax, task: str, run_dir: Path, datafile: str | None,
                   palette: dict) -> None:
    """Per-pid fitted CV RMSE to human responses, one violin per model, with
    paired-Wilcoxon significance bars drawn from SIG_REFERENCE outward."""
    frames = []
    for model_type in MODEL_ORDER:
        loss_df = _load_model_loss(task, model_type, run_dir, datafile)
        if loss_df is None:
            print(f"  (no {model_type} performance file -- skipping in P2)")
            continue
        loss_df = loss_df.copy()
        loss_df["source"] = model_type
        frames.append(loss_df)

    if not frames:
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "No fitted models\nfor this dataset",
                ha="center", va="center", transform=ax.transAxes,
                color="0.5", style="italic")
        sns.despine(ax=ax, top=True, right=True)
        return

    plot_df = pd.concat(frames, ignore_index=True)
    order = [m for m in MODEL_ORDER if m in set(plot_df["source"])]

    sns.violinplot(data=plot_df, x="source", y="rmse", order=order,
                   hue="source", palette=palette, legend=False, ax=ax,
                   cut=0)

    ax.set_xlabel("")
    ax.set_ylabel("Model fit to human responses\n(cross-validated RMSE, percentage points)")
    ax.tick_params(axis="x", rotation=30)
    # See P1: cut=0 + a hard floor at 0, since RMSE cannot be negative. Set
    # BEFORE annotate_nef_comparisons, which derives its line spacing from the
    # current y-range (it only ever adjusts `top`, so the floor survives).
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, top=True, right=True)

    if SIG_REFERENCE in order:
        annotate_nef_comparisons(
            ax, plot_df, "source", "rmse", order,
            nef_label=SIG_REFERENCE,
            compare_only=[m for m in MODEL_ORDER if m != SIG_REFERENCE],
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", default=None,
                       help="Suffix identifying which dataset to load, e.g. 'pilot5' -> "
                            "data/soltani_numbers_pilot5.pkl / soltani_colors_pilot5.pkl. "
                            "Omit to use the canonical data/soltani_numbers.pkl / "
                            "soltani_colors.pkl. Applied to the model fits too.")
    parser.add_argument("--run_folder", type=str, default="soltani",
                       help="Folder under data/runs/ with fitting.submit + "
                            "fitting.collect output. Holds BOTH tasks: each "
                            "filename carries its own dataset stem, so "
                            "soltani_numbers and soltani_colors coexist.")
    args = parser.parse_args()

    apply_style()
    run_dir = resolve_run_folder(args.run_folder)

    pal = get_palette(len(MODEL_ORDER))
    palette = {m: pal[i] for i, m in enumerate(MODEL_ORDER)}
    palette["Human"] = HUMAN_COLOR

    fig, axes = plt.subplots(
        2, 3,
        figsize=(FIGURE_SIZE[0] * 0.75, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    any_models = False
    for row, task in enumerate(TASK_ROWS):
        print(f"task-{task}:")
        human = _load_human(task, args.datafile)
        if human is None:
            print("  no data file found for this task/datafile combination -- skipping")
            for col in range(3):
                axes[row, col].axis("off")
            axes[row, 1].text(0.5, 0.5, f"No {task} data\nfor this dataset",
                             ha="center", va="center", transform=axes[row, 1].transAxes,
                             color="0.5", style="italic")
            axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")
            continue
        print(f"  {len(human)} rows, {human['pid'].nunique()} pids")

        models = {}
        for model_type in MODEL_ORDER:
            mdf = _load_model_responses(task, model_type, run_dir, args.datafile)
            if mdf is not None:
                models[model_type] = mdf
        if models:
            any_models = True
            print(f"  models loaded: {', '.join(models)}")
        else:
            print("  no fitted model responses found for this task/datafile")

        _plot_schematic(axes[row, 0], task)
        _plot_panel_p1(axes[row, 1], human, models, palette)
        _plot_panel_p2(axes[row, 2], task, run_dir, args.datafile, palette)
        axes[row, 0].set_title(f"task-{task}", loc="left", fontsize=9, style="italic")

    label_panels(axes)

    if any_models:
        footer = (f"P2 significance bars: paired Wilcoxon vs {SIG_REFERENCE} "
                 f"(NEF not yet fit for these datasets). Model fits from run "
                 f"folder '{args.run_folder}'.")
    else:
        footer = ("Human data only -- no fitted model responses found for this "
                 "datafile in run folder "
                 f"'{args.run_folder}'.")
    fig.text(0.5, -0.02, footer,
              ha="center", va="top", fontsize=7, style="italic", color="0.4")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure_soltani_performance"
    if args.datafile:
        stem = f"{stem}_{args.datafile}"
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    print(f"Saved figures/{stem}.pdf")


if __name__ == "__main__":
    main()
