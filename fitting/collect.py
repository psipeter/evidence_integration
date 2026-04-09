"""
Aggregation step of the fitting pipeline.

Run after ``fitting.rerun`` has produced per-participant response files.
Concatenates responses, parameters, and performance tables across all
participants for a given dataset and model type, and writes three combined
files under ``data/``:

- ``{model_type}_{dataset}_responses.pkl``
- ``{model_type}_{dataset}_params.pkl``
- ``{model_type}_{dataset}_performance.pkl``

Entry point: ``python -m fitting.collect {dataset} {model_type}``
"""

import logging
import sys

import pandas as pd

from utils.paths import data_path


def collect(dataset: str, model_type: str) -> None:
    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    pids = sorted(human["pid"].unique())

    responses_dfs: list[pd.DataFrame] = []
    params_dfs: list[pd.DataFrame] = []
    performance_dfs: list[pd.DataFrame] = []

    for pid in pids:
        pid = int(pid)
        paths = (
            data_path(f"{model_type}_{dataset}_{pid}_responses.pkl"),
            data_path(f"{model_type}_{dataset}_{pid}_params.pkl"),
            data_path(f"{model_type}_{dataset}_{pid}_performance.pkl"),
        )
        if not all(p.exists() for p in paths):
            logging.warning(
                "Skipping pid=%s: missing one or more of responses/params/performance",
                pid,
            )
            continue
        responses_dfs.append(pd.read_pickle(paths[0]))
        params_dfs.append(pd.read_pickle(paths[1]))
        performance_dfs.append(pd.read_pickle(paths[2]))

    def _combine_and_save(dfs: list[pd.DataFrame], out_name: str) -> pd.DataFrame:
        if not dfs:
            logging.warning("No data to concatenate for %s; writing empty DataFrame", out_name)
            combined = pd.DataFrame()
        else:
            combined = pd.concat(dfs, ignore_index=True)
        combined = combined.reset_index(drop=True)
        combined.to_pickle(data_path(out_name))
        return combined

    responses = _combine_and_save(
        responses_dfs, f"{model_type}_{dataset}_responses.pkl"
    )
    params = _combine_and_save(params_dfs, f"{model_type}_{dataset}_params.pkl")
    performance = _combine_and_save(
        performance_dfs, f"{model_type}_{dataset}_performance.pkl"
    )

    print(f"{model_type}_{dataset}_responses.pkl: shape {responses.shape}")
    print(responses.head(5))
    print()
    print(f"{model_type}_{dataset}_params.pkl: shape {params.shape}")
    print(params.head(5))
    print()
    print(f"{model_type}_{dataset}_performance.pkl: shape {performance.shape}")
    print(performance.head(5))


if __name__ == "__main__":
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    logging.basicConfig(level=logging.INFO)
    collect(dataset, model_type)
