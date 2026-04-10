"""
Convenience script to aggregate per-participant fit outputs into combined
dataset-level pickles for every model and task.

Runs ``fitting.collect`` for each ``(dataset, model_type)`` pair defined in
``ALL_MODELS`` — equivalent to invoking
``python -m fitting.collect {dataset} {model_type}`` for each combination.

Entry point::

    python -m jobs.collect_all
"""

from fitting.collect import collect
from jobs.resubmit_missing import ALL_MODELS


def main() -> None:
    for dataset in sorted(ALL_MODELS.keys()):
        for model_type in ALL_MODELS[dataset]:
            print(f"Collecting {model_type} {dataset}...")
            collect(dataset, model_type)
    print("Done.")


if __name__ == "__main__":
    main()
