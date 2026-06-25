#!/usr/bin/env python3
"""
Build production assets and package JATOS study archives (.jzip) for MindProbe.

Usage (from repo root or task/):
    python task/generate_jzip.py
    python task/generate_jzip.py --skip-build   # package existing dist-* only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent

STUDIES = {
    "continuous": {
        "dist_dir": "dist-continuous",
        "output": "evidence-integration-continuous.jzip",
        "jas": {
            "version": "3",
            "data": {
                "uuid": "6c48d3e3-3b34-4cb8-a840-b7922ec6ff57",
                "title": "Evidence Integration — Continuous",
                "description": "Sequential evidence integration task (continuous).",
                "groupStudy": False,
                "linearStudy": False,
                "allowPreview": False,
                "dirName": "evidence_integration_continuous",
                "comments": "",
                "jsonData": None,
                "endRedirectUrl": (
                    "https://app.prolific.com/submissions/complete?cc=C3W3TF1O"
                ),
                "studyEntryMsg": None,
                "componentList": [
                    {
                        "uuid": "df03e4d5-d810-447f-bbb1-1442b5e0260f",
                        "title": "Task",
                        "htmlFilePath": "index-continuous.html",
                        "reloadable": False,
                        "active": True,
                        "comments": "",
                        "jsonData": None,
                    }
                ],
                "batchList": [
                    {
                        "uuid": "b6c21ad2-f1fb-42a3-9c53-3dc3d702a191",
                        "title": "Default",
                        "active": True,
                        "maxActiveMembers": None,
                        "maxTotalMembers": None,
                        "maxTotalWorkers": None,
                        "allowedWorkerTypes": [
                            "GeneralMultiple",
                            "GeneralSingle",
                            "Jatos",
                            "PersonalMultiple",
                            "PersonalSingle",
                        ],
                        "comments": None,
                        "jsonData": None,
                    }
                ],
            },
        },
    },
    "binary": {
        "dist_dir": "dist-binary",
        "output": "evidence-integration-binary.jzip",
        "jas": {
            "version": "3",
            "data": {
                "uuid": "5327124f-2370-49a8-a3bc-2c0534c1fcf2",
                "title": "Evidence Integration — Binary",
                "description": "Sequential evidence integration task (binary).",
                "groupStudy": False,
                "linearStudy": False,
                "allowPreview": False,
                "dirName": "evidence_integration_binary",
                "comments": "",
                "jsonData": None,
                "endRedirectUrl": (
                    "https://app.prolific.com/submissions/complete?cc=PLACEHOLDER"
                ),
                "studyEntryMsg": None,
                "componentList": [
                    {
                        "uuid": "3a183108-0807-4921-9c3b-2df801a3a607",
                        "title": "Task",
                        "htmlFilePath": "index-binary.html",
                        "reloadable": False,
                        "active": True,
                        "comments": "",
                        "jsonData": None,
                    }
                ],
                "batchList": [
                    {
                        "uuid": "cdb5f9a9-55cb-4d00-8b05-ccc48daf4db5",
                        "title": "Default",
                        "active": True,
                        "maxActiveMembers": None,
                        "maxTotalMembers": None,
                        "maxTotalWorkers": None,
                        "allowedWorkerTypes": [
                            "GeneralMultiple",
                            "GeneralSingle",
                            "Jatos",
                            "PersonalMultiple",
                            "PersonalSingle",
                        ],
                        "comments": None,
                        "jsonData": None,
                    }
                ],
            },
        },
    },
}


def run_build() -> None:
    print("Building production assets …")
    subprocess.run(
        ["npm", "run", "build:continuous"],
        cwd=TASK_DIR,
        check=True,
    )
    subprocess.run(
        ["npm", "run", "build:binary"],
        cwd=TASK_DIR,
        check=True,
    )


def package_study(name: str, spec: dict) -> Path:
    dist_dir = TASK_DIR / spec["dist_dir"]
    if not dist_dir.is_dir():
        raise FileNotFoundError(f"Missing build output: {dist_dir}")

    dir_name = spec["jas"]["data"]["dirName"]
    jas_name = f"{dir_name}.jas"
    out_path = TASK_DIR / spec["output"]

    if out_path.exists():
        out_path.unlink()

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        jas_bytes = json.dumps(spec["jas"], indent=4).encode("utf-8")
        zf.writestr(jas_name, jas_bytes)

        for path in sorted(dist_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = f"{dir_name}/{path.relative_to(dist_dir).as_posix()}"
            zf.write(path, arcname)

    size_kb = out_path.stat().st_size / 1024
    n_files = sum(1 for _ in dist_dir.rglob("*") if _.is_file())
    print(f"  {name}: {out_path.name} ({size_kb:.0f} KiB, {n_files} assets)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and package MindProbe .jzip files")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm build; package existing dist-* directories only",
    )
    parser.add_argument(
        "--task",
        choices=["continuous", "binary", "both"],
        default="both",
        help="Which study archive(s) to generate (default: both)",
    )
    args = parser.parse_args()

    if not shutil.which("npm"):
        sys.exit("npm not found on PATH")

    if not args.skip_build:
        run_build()

    tasks = list(STUDIES) if args.task == "both" else [args.task]
    print("Packaging JATOS archives …")
    for name in tasks:
        package_study(name, STUDIES[name])

    print("Done. Import into MindProbe: Studies → + → Import Study")


if __name__ == "__main__":
    main()
