#!/usr/bin/env python3
"""Utility for regenerating and saving model responses from best-fit params."""

from __future__ import annotations

import argparse
from pathlib import Path

from models import NEF
from utils.paths import dataset_stem, resolve_run_folder
from utils.run_params import load_run_params


def save(pid: int, dataset: str, run_folder: str | Path, model_type: str,
         datafile: str | None = None) -> None:
    """Regenerate and save NEF responses from a pid's best-fit params.

    `datafile` is the data-version suffix (see utils.paths.dataset_stem); it
    selects both which params file to read and which responses file to write,
    so a fit against one data build cannot overwrite another's responses.
    Defaults to None for the unsuffixed carrabin/yoo behaviour.
    """
    run_folder = resolve_run_folder(run_folder)
    params = load_run_params(pid, dataset, model_type, run_folder, datafile)
    responses = NEF.run(params)  # transform applied inside NEF.run()
    stem = dataset_stem(dataset, datafile)
    responses.to_pickle(run_folder / f"{model_type}_{stem}_{pid}_responses.pkl")
    print(f"Saved responses for {stem} {model_type} pid={pid}")


if __name__ == "__main__":
    # argparse rather than positional slicing: datafile is optional, and
    # fitting.submit builds this command as a string for SLURM.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("model_type")
    parser.add_argument("pid", type=int)
    parser.add_argument("run_folder")
    parser.add_argument(
        "--datafile",
        default=None,
        help="Data-version suffix; omit for the canonical unsuffixed dataset.",
    )
    args = parser.parse_args()
    save(args.pid, args.dataset, args.run_folder, args.model_type, args.datafile)
    print("JOB_COMPLETE")
