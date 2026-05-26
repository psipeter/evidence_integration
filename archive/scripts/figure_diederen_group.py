"""
Group-level figure for the Diederen dataset.

Companion to figure_diederen.py but uses group-level model fits from
scripts/fit_diederen_group.py rather than per-participant fits.

Data: distribution A only, pre-first-switch, CTRL+PCB groups,
bad performers excluded (see fit_diederen_group.py for full details).

Top row (implemented here):
  Panel A: task diagram (figures/diederen_task.pdf)
  Panel B: per-sequence RMSE boxplot (one box per model)
  Panel C: group mean response vs observation (+EV / −EV), SE across sequences
  Panel D: mean |Δresponse| vs observation, SE across sequences

Usage:
    python scripts/figure_diederen_group.py
    python scripts/figure_diederen_group.py --run_folder diederen_group
    python scripts/figure_diederen_group.py --include_rl_lambda
"""

from __future__ import annotations

import argparse
import pickle
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

from utils.paths import FIGURES_DIR, RUNS_DIR, data_path
from utils.plot_style import FIGURE_SIZE, apply_style, get_palette, label_panels

EXCLUDE_PIDS: list[int] = [
    1011,
    1023,
    1027,
    1028,
    1032,
    2001,
    2029,
    2036,
    2038,
    2047,
    2048,
    2064,
    2083,
    2092,
    2099,
]

GROUP_DATASET = "diederen_group"
MIN_SEQUENCES_PLOT = 10  # min sequences required to plot an observation
MODEL_ORDER = ["Mean", "RL", "RL_lambda", "PearceHall"]


def _display(model_type: str) -> str:
    return "NEF" if model_type.startswith("NEF") else model_type


def _load_group_result(run_folder: Path, model_type: str) -> dict | None:
    """Load master output file for one model. Returns None if not found."""
    path = run_folder / f"{model_type}_diederen_group.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_human() -> pd.DataFrame:
    """Load filtered group dataset (diederen_group.pkl)."""
    path = data_path(GROUP_DATASET + ".pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"{GROUP_DATASET}.pkl not found. "
            "Run scripts/fit_diederen_group.py --rebuild_dataset first."
        )
    return pd.read_pickle(path)


def _placeholder(ax, text: str) -> None:
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
        color="0.5",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def _plot_panel_a(ax) -> None:
    """Render first page of figures/diederen_task.pdf into panel A."""
    pdf_path = FIGURES_DIR / "diederen_task.pdf"
    if not pdf_path.exists():
        _placeholder(ax, "diederen_task.pdf not found")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "diederen_task"
        cmd = ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(out_prefix)]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            _placeholder(ax, "pdftoppm failed")
            return
        img_path = out_prefix.with_suffix(".png")
        if not img_path.exists():
            _placeholder(ax, "diederen_task.pdf render failed")
            return
        img = mpimg.imread(img_path)
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    ax.set_anchor("C")


def _plot_panel_b(
    ax,
    run_folder: Path,
    palette: dict,
    model_order: list[str],
) -> None:
    """Per-sequence RMSE vs group human mean (one box per model)."""
    human = _load_human()
    human_mean_pos = human[human["ev"] > 0].groupby("observation")["response"].mean()
    human_mean_neg = human[human["ev"] < 0].groupby("observation")["response"].mean()

    rows = []
    for mt in model_order:
        result = _load_group_result(run_folder, mt)
        if result is None:
            continue
        resp = result["responses"]

        for (pid, session), grp in resp.groupby(["pid", "session"]):
            if grp.empty:
                continue
            ev_sign = grp["ev"].iloc[0] if "ev" in grp.columns else None
            if ev_sign is None:
                continue
            human_mean = human_mean_pos if ev_sign > 0 else human_mean_neg
            grp_sorted = grp.sort_values("observation")
            common_obs = grp_sorted["observation"].values
            h_vals = human_mean.reindex(common_obs).values
            m_vals = grp_sorted["response"].values
            mask = ~np.isnan(h_vals)
            if mask.sum() == 0:
                continue
            rmse = float(np.sqrt(np.mean((m_vals[mask] - h_vals[mask]) ** 2)))
            rows.append({"model": _display(mt), "rmse": rmse})

    if not rows:
        _placeholder(ax, "No fits found")
        return

    df = pd.DataFrame(rows)
    available = [_display(m) for m in model_order if _display(m) in set(df["model"])]
    pal = {_display(m): palette.get(_display(m), "0.5") for m in model_order}
    sns.boxplot(
        data=df,
        x="model",
        y="rmse",
        order=available,
        hue="model",
        palette=pal,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("RMSE vs group mean")
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_c(
    ax,
    human: pd.DataFrame,
    run_folder: Path,
    palette: dict,
    model_order: list[str],
) -> None:
    """Group mean response vs observation (+EV / −EV), SE across sequences."""

    def _prep_long(df: pd.DataFrame, source: str) -> pd.DataFrame:
        seq_means = (
            df.groupby(["pid", "session", "observation", "ev"])["response"]
            .mean()
            .reset_index()
        )
        seq_means["ev_sign"] = np.where(seq_means["ev"] > 0, "+EV", "−EV")
        seq_means["source"] = source
        return seq_means

    def _filter_obs(df: pd.DataFrame) -> pd.DataFrame:
        counts = df.groupby("observation")["pid"].count()
        valid = counts[counts >= MIN_SEQUENCES_PLOT].index
        return df[df["observation"].isin(valid)]

    human_long = _filter_obs(_prep_long(human, "Human"))

    for ev_sign, ls in [("+EV", "-"), ("−EV", "--")]:
        sub = human_long[human_long["ev_sign"] == ev_sign]
        sns.lineplot(
            data=sub,
            x="observation",
            y="response",
            color="black",
            linewidth=2.2,
            linestyle=ls,
            errorbar="se",
            label=f"Human ({ev_sign})",
            ax=ax,
        )

    for mt in model_order:
        result = _load_group_result(run_folder, mt)
        if result is None:
            continue
        model_long = _filter_obs(_prep_long(result["responses"], _display(mt)))
        col = palette.get(_display(mt), "0.5")
        for ev_sign, ls in [("+EV", "-"), ("−EV", "--")]:
            sub = model_long[model_long["ev_sign"] == ev_sign]
            label = f"{_display(mt)} ({ev_sign})" if ev_sign == "+EV" else None
            sns.lineplot(
                data=sub,
                x="observation",
                y="response",
                color=col,
                linewidth=1.8,
                linestyle=ls,
                errorbar="se",
                label=label,
                ax=ax,
            )

    ax.axhline(0, color="0.7", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Response")
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax, top=True, right=True)


def _plot_panel_d(
    ax,
    human: pd.DataFrame,
    run_folder: Path,
    palette: dict,
    model_order: list[str],
) -> None:
    """Mean |Δresponse| vs observation, SE across sequences."""

    def _prep_delta_long(df: pd.DataFrame, source: str) -> pd.DataFrame:
        df = df.copy().sort_values(["pid", "trial", "observation"])
        df["delta"] = df.groupby(["pid", "trial"])["response"].diff().abs()
        seq_means = (
            df[df["delta"].notna()]
            .groupby(["pid", "session", "observation"])["delta"]
            .mean()
            .reset_index()
        )
        seq_means["source"] = source
        return seq_means

    def _filter_obs(df: pd.DataFrame) -> pd.DataFrame:
        counts = df.groupby("observation")["pid"].count()
        valid = counts[counts >= MIN_SEQUENCES_PLOT].index
        return df[df["observation"].isin(valid)]

    h_long = _filter_obs(_prep_delta_long(human, "Human"))
    sns.lineplot(
        data=h_long,
        x="observation",
        y="delta",
        color="black",
        linewidth=2.2,
        errorbar="se",
        label="Human",
        ax=ax,
    )

    for mt in model_order:
        result = _load_group_result(run_folder, mt)
        if result is None:
            continue
        m_long = _filter_obs(_prep_delta_long(result["responses"], _display(mt)))
        col = palette.get(_display(mt), "0.5")
        sns.lineplot(
            data=m_long,
            x="observation",
            y="delta",
            color=col,
            linewidth=1.8,
            errorbar="se",
            label=_display(mt),
            ax=ax,
        )

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Response change")
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax, top=True, right=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_folder", type=str, default="diederen_group")
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument(
        "--include_rl_lambda",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    run_folder = RUNS_DIR / args.run_folder
    model_order = [
        m for m in MODEL_ORDER if args.include_rl_lambda or m != "RL_lambda"
    ]

    apply_style()
    _pal = get_palette(len(model_order))
    palette = {m: _pal[i] for i, m in enumerate(model_order)}
    palette.update({_display(m): palette[m] for m in model_order})

    human = _load_human()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FIGURE_SIZE[0] * 2, FIGURE_SIZE[1]),
        constrained_layout=True,
    )

    _plot_panel_a(axes[0])
    _plot_panel_b(axes[1], run_folder, palette, model_order)
    _plot_panel_c(axes[2], human, run_folder, palette, model_order)
    _plot_panel_d(axes[3], human, run_folder, palette, model_order)

    label_panels(list(axes))

    out_dir = Path(args.out_folder) if args.out_folder else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "figure_diederen_group.png", dpi=300)
    plt.savefig(out_dir / "figure_diederen_group.pdf")
    plt.close(fig)
    print(f"Saved {out_dir}/figure_diederen_group.{{png,pdf}}")


if __name__ == "__main__":
    main()
