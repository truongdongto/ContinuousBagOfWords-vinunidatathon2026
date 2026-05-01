#!/usr/bin/env python3
"""Run build_*_submission.py for v37, v40, v45 (Nelder-Mead weight opt + CSV)."""
import argparse
import sys

from config import SUBMISSION_BUILDERS
from run_support import should_execute, run_python_script


def main() -> int:
    ap = argparse.ArgumentParser(description="Build submission CSVs for base models.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    prompt = "Will run submission builders:\n" + "\n".join(f"  - {s}" for s in SUBMISSION_BUILDERS)
    if not should_execute(prompt, dry_run=args.dry_run, yes=args.yes):
        return 2

    for script in SUBMISSION_BUILDERS:
        rc = run_python_script(script, dry_run=args.dry_run)
        if rc != 0:
            print(f"FAILED ({rc}): {script}", file=sys.stderr)
            return rc
    print("\nSubmissions saved under diag_submissions/. Next: python run_blend_best.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
