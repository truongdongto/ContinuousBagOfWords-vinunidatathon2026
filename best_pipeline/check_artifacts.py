#!/usr/bin/env python3
"""Read-only: verify artifacts for best-pipeline stages. No prompts, no writes."""
import argparse
import sys

from config import (
    DATA_ROOT,
    PICKLE_CHAIN,
    PREREQUISITES,
    SUBMISSION_EXPECTED_FILES,
    TRAIN_JOBS,
    BEST_BLEND_OUTPUT,
    V41B_SUBMISSION,
)
from run_support import check_files


def print_stage(name: str, ok: list[str], missing: list[str]) -> None:
    print(f"\n=== {name} ===")
    for r in ok:
        print(f"  OK   {r}")
    for r in missing:
        print(f"  MISS {r}")


def main() -> int:
    p = argparse.ArgumentParser(description="Check pipeline artifacts under DATA_ROOT.")
    p.add_argument("--require", choices=("all", "prereq", "features", "train", "submit", "blend"),
                   default="all", help="Which stage to validate")
    args = p.parse_args()

    print(f"DATA_ROOT = {DATA_ROOT}")

    exit_code = 0
    if args.require in ("all", "prereq"):
        ok, missing = check_files(PREREQUISITES)
        print_stage("Prerequisites (v2 + v3 promo base)", ok, missing)
        if missing:
            exit_code = 1

    if args.require in ("all", "features"):
        ok, missing = check_files(PICKLE_CHAIN)
        print_stage("Feature pickles (after running run_features.py)", ok, missing)
        if missing and args.require == "features":
            exit_code = 1

    if args.require in ("all", "train"):
        paths = [job[1] for job in TRAIN_JOBS]
        ok, missing = check_files(tuple(paths))
        print_stage("Training outputs (npz)", ok, missing)
        if missing and args.require == "train":
            exit_code = 1

    if args.require in ("all", "submit"):
        ok, missing = check_files(SUBMISSION_EXPECTED_FILES)
        print_stage("Submission CSVs (single-model)", ok, missing)
        if missing and args.require == "submit":
            exit_code = 1

    if args.require in ("all", "blend"):
        ok, missing = check_files((V41B_SUBMISSION, BEST_BLEND_OUTPUT))
        print_stage("Blend outputs (v41_B + v46 w85)", ok, missing)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
