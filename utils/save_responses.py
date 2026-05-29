#!/usr/bin/env python3
"""Utility for regenerating and saving model responses from best-fit params."""

from __future__ import annotations

import sys
from pathlib import Path

from models import NEF
from utils.carrabin_transform import apply_carrabin_transform
from utils.paths import resolve_run_folder
from utils.run_params import load_run_params


def save(pid: int, dataset: str, run_folder: str | Path, model_type: str) -> None:
    run_folder = resolve_run_folder(run_folder)
    params = load_run_params(pid, dataset, model_type, run_folder)
    responses = NEF.run(params)
    responses = apply_carrabin_transform(responses, dataset)
    responses.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_responses.pkl")
    print(f"Saved responses for {dataset} {model_type} pid={pid}")


if __name__ == "__main__":
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    pid = int(sys.argv[3])
    run_folder = Path(sys.argv[4])
    save(pid, dataset, run_folder, model_type)
    print("JOB_COMPLETE")
