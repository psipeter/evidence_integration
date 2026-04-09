"""
Response-generation step of the fitting pipeline.

Run after ``fitting.fit`` has written best parameters for a participant.
Loads those parameters, runs the mathematical model on all trials, and saves
per-participant responses. Aggregated outputs are built by ``fitting.collect``
into dataset-level pickle files.

Entry point: ``python -m fitting.rerun {dataset} {model_type} {pid}``
"""

import sys

import pandas as pd

import models.math_models as math_models
from utils.paths import data_path


def rerun(dataset: str, model_type: str, pid: int) -> pd.DataFrame:
    params_path = data_path(f"{model_type}_{dataset}_{pid}_params.pkl")
    params_df = pd.read_pickle(params_path)
    params = params_df.loc[0].to_dict()
    df = math_models.run(params, save=False)
    out_path = data_path(f"{model_type}_{dataset}_{pid}_responses.pkl")
    df.to_pickle(out_path)
    return df


if __name__ == "__main__":
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    pid = int(sys.argv[3])
    logging.basicConfig(level=logging.INFO)
    df = rerun(dataset, model_type, pid)
    print(df)
