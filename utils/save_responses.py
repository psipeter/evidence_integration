#!/usr/bin/env python3
"""Utility for regenerating and saving model responses from best-fit params."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from fitting.model_params import MODEL_PARAMS
from models import NEF
from models.NEF import PARAM_DEFAULTS


def save(pid: int, dataset: str, run_folder: str | Path, model_type: str) -> None:
    run_folder = Path(run_folder)
    params = pd.read_pickle(
        run_folder / f"{model_type}_{dataset}_{pid}_params.pkl"
    ).iloc[0].to_dict()
    fixed = MODEL_PARAMS[dataset][model_type].get("fixed", {})
    params = {**PARAM_DEFAULTS, **fixed, **params}
    params["nef_type"] = "recurrent" if "recurrent" in model_type else "synaptic"
    params["dataset"] = dataset
    params["model_type"] = model_type
    responses = NEF.run(params)
    responses.to_pickle(run_folder / f"{model_type}_{dataset}_{pid}_responses.pkl")
    print(f"Saved responses for {dataset} {model_type} pid={pid}")


if __name__ == "__main__":
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    pid = int(sys.argv[3])
    run_folder = Path(sys.argv[4])
    save(pid, dataset, run_folder, model_type)
    print("JOB_COMPLETE")
