#!/usr/bin/env python3
"""Train v37 / v40 / v45 ensembles (heavy). cwd=DATA_ROOT. Asks confirmation unless --yes."""
import argparse
import sys

from config import TRAIN_JOBS
from run_support import should_execute, run_python_script


def main() -> int:
    ap = argparse.ArgumentParser(description="Train 4-way ensembles for best pipeline.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    lines = []
    for script, npz_out in TRAIN_JOBS:
        lines.append(f"  - {script}  →  {npz_out}")
    prompt = "Heavy training (~minutes each, CPU). Will run:\n" + "\n".join(lines)

    if not should_execute(prompt, dry_run=args.dry_run, yes=args.yes):
        return 2

    for script, _ in TRAIN_JOBS:
        rc = run_python_script(script, dry_run=args.dry_run)
        if rc != 0:
            print(f"FAILED ({rc}): {script}", file=sys.stderr)
            return rc
    print("\nTraining finished. Run: python check_artifacts.py --require train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
