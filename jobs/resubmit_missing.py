"""
Utility to detect participants without saved fit outputs and resubmit their
SLURM job scripts.

Use after a batch run to catch failures: compares ``data/{dataset}.pkl`` pids
against per-participant
``{model_type}_{dataset}_{pid}_{loss_type}_params.pkl`` files, then
``sbatch`` the matching ``jobs/{model_type}_{dataset}_{pid}.sh`` scripts when
not in dry-run mode.

Entry point::

    Usage:
        python -m jobs.resubmit_missing                              # all models
        python -m jobs.resubmit_missing {dataset} {model_type}       # one model
        python -m jobs.resubmit_missing {dataset} {model_type} {loss_type}
        add --dry-run to any form to list without submitting
"""

import subprocess
import sys
import time

import pandas as pd

from fitting.fit import DEFAULT_LOSS
from fitting.param_ranges import MODEL_PARAMS
from utils.paths import PROJECT_ROOT, data_path

# All dataset/model combinations
ALL_MODELS = {
    dataset: list(models.keys()) for dataset, models in MODEL_PARAMS.items()
}


def find_missing(
    dataset: str,
    model_type: str,
    loss_type: str | None = None,
) -> list[int]:
    if loss_type is None:
        loss_type = DEFAULT_LOSS.get(dataset, "mse")
    human = pd.read_pickle(data_path(f"{dataset}.pkl"))
    pids = sorted(int(x) for x in human["pid"].unique())
    missing: list[int] = []
    for pid in pids:
        params_path = data_path(
            f"{model_type}_{dataset}_{pid}_{loss_type}_params.pkl"
        )
        if not params_path.is_file():
            missing.append(pid)
    return missing


def _usage_error() -> None:
    print(
        "Usage:\n"
        "  python -m jobs.resubmit_missing                              # all models\n"
        "  python -m jobs.resubmit_missing {dataset} {model_type}       # one model\n"
        "  python -m jobs.resubmit_missing {dataset} {model_type} {loss_type}\n"
        "  add --dry-run to any form to list without submitting",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    pos_args = [a for a in sys.argv[1:] if a != "--dry-run"]

    if len(pos_args) == 0:
        _run_all_models(dry_run)
    elif len(pos_args) == 2:
        _run_single(pos_args[0], pos_args[1], dry_run, None)
    elif len(pos_args) == 3:
        _run_single(pos_args[0], pos_args[1], dry_run, pos_args[2])
    else:
        _usage_error()


def _run_single(
    dataset: str,
    model_type: str,
    dry_run: bool,
    loss_type: str | None,
) -> None:
    missing = find_missing(dataset, model_type, loss_type)
    if not missing:
        print(f"All participants complete for {model_type} {dataset}")
        return

    n = len(missing)
    print(f"Missing {n} participants: {missing}")

    if dry_run:
        return

    jobs_dir = PROJECT_ROOT / "jobs"
    resubmitted = 0
    for pid in missing:
        script_path = jobs_dir / f"{model_type}_{dataset}_{pid}.sh"
        if not script_path.is_file():
            print(
                f"Warning: no job script for pid={pid}, skipping: {script_path}",
                file=sys.stderr,
            )
            continue
        result = subprocess.run(["sbatch", str(script_path)], check=False)
        if result.returncode == 0:
            resubmitted += 1
        else:
            print(
                f"Warning: sbatch failed for pid={pid} (exit {result.returncode})",
                file=sys.stderr,
            )
        time.sleep(0.5)

    print(f"Resubmitted {resubmitted} job(s)")


def _run_all_models(dry_run: bool) -> None:
    missing_by_pair: dict[tuple[str, str], list[int]] = {}
    missing_triples: list[tuple[str, str, int]] = []

    for dataset in sorted(ALL_MODELS.keys()):
        for model_type in ALL_MODELS[dataset]:
            missing = find_missing(dataset, model_type, None)
            missing_by_pair[(dataset, model_type)] = missing
            for pid in missing:
                missing_triples.append((dataset, model_type, pid))

    print(
        f"{'dataset':<12} {'model_type':<15} {'missing_n':>10}\n"
        f"{'-' * 12} {'-' * 15} {'-' * 10}"
    )
    for (dataset, model_type) in sorted(missing_by_pair.keys()):
        n_miss = len(missing_by_pair[(dataset, model_type)])
        print(f"{dataset:<12} {model_type:<15} {n_miss:>10}")

    if not missing_triples:
        print("All participants complete for all models (all dataset/model pairs)")
        return

    total = len(missing_triples)
    print(f"\nMissing {total} (dataset, model_type, pid) job(s):")
    for triple in missing_triples:
        print(f"  {triple}")

    if dry_run:
        return

    jobs_dir = PROJECT_ROOT / "jobs"
    resubmitted = 0
    for dataset, model_type, pid in missing_triples:
        script_path = jobs_dir / f"{model_type}_{dataset}_{pid}.sh"
        if not script_path.is_file():
            print(
                f"Warning: no job script for {dataset} {model_type} pid={pid}, "
                f"skipping: {script_path}",
                file=sys.stderr,
            )
            continue
        result = subprocess.run(["sbatch", str(script_path)], check=False)
        if result.returncode == 0:
            resubmitted += 1
        else:
            print(
                f"Warning: sbatch failed for {dataset} {model_type} pid={pid} "
                f"(exit {result.returncode})",
                file=sys.stderr,
            )
        time.sleep(0.5)

    print(f"Resubmitted {resubmitted} job(s)")


if __name__ == "__main__":
    main()
