"""
Utility to remove per-participant fitting artifacts from ``data/`` so you can
re-run fits and use ``jobs.resubmit_missing`` on a clean slate.

Deletes only files matching the project's fit/rerun/collect naming convention
for a given ``dataset``, ``model_type``, and ``loss_type``. The three canonical
human-data pickles (``carrabin.pkl``, ``jiang.pkl``, ``yoo.pkl``) and
``jiang_networks.npy`` are never targeted and are rejected if they would ever
appear in the deletion list.

Entry point::

    python -m jobs.clear_data all [loss_type] [--dry-run]
    python -m jobs.clear_data {dataset} {model_type} [loss_type] [--dry-run]
"""

import sys
from pathlib import Path

import pandas as pd

from fitting.fit import DEFAULT_LOSS
from jobs.resubmit_missing import ALL_MODELS
from utils.paths import DATA_DIR, data_path

PROTECTED_NAMES = frozenset(
    {"carrabin.pkl", "jiang.pkl", "yoo.pkl", "jiang_networks.npy"}
)


def clear_data(
    dataset: str,
    model_type: str,
    loss_type: str | None = None,
    dry_run: bool = False,
) -> None:
    if loss_type is None:
        loss_type = DEFAULT_LOSS.get(dataset, "mse")

    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    pids = human["pid"].unique()

    paths_to_delete: list[Path] = []
    for pid in pids:
        pid = int(pid)
        pattern = f"{model_type}_{dataset}_{pid}_{loss_type}_*.pkl"
        paths_to_delete.extend(DATA_DIR.glob(pattern))

    combined_names = (
        f"{model_type}_{dataset}_{loss_type}_responses.pkl",
        f"{model_type}_{dataset}_{loss_type}_params.pkl",
        f"{model_type}_{dataset}_{loss_type}_performance.pkl",
        f"{model_type}_{dataset}_{loss_type}_cv_folds.pkl",
    )
    for name in combined_names:
        p = data_path(name)
        if p.is_file():
            paths_to_delete.append(p)

    paths_to_delete = list(dict.fromkeys(paths_to_delete))

    for p in paths_to_delete:
        if p.name in PROTECTED_NAMES:
            raise ValueError(
                f"Refusing to delete protected file (safety check): {p}"
            )

    if dry_run:
        for p in paths_to_delete:
            print(f"would delete: {p}")
        print(f"Would delete {len(paths_to_delete)} file(s)")
        return

    deleted = 0
    for p in paths_to_delete:
        p.unlink()
        deleted += 1
    print(f"Deleted {deleted} file(s)")


def clear_all(loss_type: str | None = None, dry_run: bool = False) -> None:
    """Clear fitting outputs for all dataset/model combinations."""
    for dataset, models in ALL_MODELS.items():
        for model_type in models:
            lt = loss_type if loss_type is not None else DEFAULT_LOSS.get(dataset, "mse")
            print(f"Clearing {model_type} {dataset} {lt}...")
            clear_data(dataset, model_type, loss_type=loss_type, dry_run=dry_run)


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    pos = [a for a in sys.argv[1:] if a != "--dry-run"]

    if len(pos) == 0:
        print(
            "Usage: python -m jobs.clear_data all [loss_type] [--dry-run]\n"
            "       python -m jobs.clear_data {dataset} {model_type} "
            "[loss_type] [--dry-run]",
            file=sys.stderr,
        )
        sys.exit(1)

    if pos[0] == "all":
        loss_type = pos[1] if len(pos) > 1 else None
        clear_all(loss_type=loss_type, dry_run=dry_run)
    else:
        if len(pos) < 2:
            print(
                "Usage: python -m jobs.clear_data {dataset} {model_type} "
                "[loss_type] [--dry-run]",
                file=sys.stderr,
            )
            sys.exit(1)
        dataset, model_type = pos[0], pos[1]
        loss_type = pos[2] if len(pos) > 2 else None
        clear_data(dataset, model_type, loss_type=loss_type, dry_run=dry_run)


if __name__ == "__main__":
    main()
