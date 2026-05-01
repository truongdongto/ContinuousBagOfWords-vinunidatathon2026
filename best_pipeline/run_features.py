#!/usr/bin/env python3
"""Run feature engineering chain in DATA_ROOT (build_v5 → v6 → v8 → v10). Requires v3_pattern.pkl."""
import argparse
import sys

from config import DATA_ROOT, FEATURE_SCRIPTS, PREREQUISITES
from run_support import should_execute, run_python_script, check_files


def main() -> int:
    ap = argparse.ArgumentParser(description="Build feature pickles (delegates to parent scripts).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    args = ap.parse_args()

    ok, missing = check_files(PREREQUISITES)
    if missing:
        print(f"Missing prerequisites under {DATA_ROOT}:")
        for m in missing:
            print(f"  - {m}")
        print("\n'enriched_features_v2.pkl' and 'enriched_features_v3_pattern.pkl' must exist "
              "(there is no build_v3 script in-repo; regenerate from your notebooks if needed).")
        return 1

    prompt = (
        "Will run sequentially:\n"
        + "\n".join(f"  - {s}" for s in FEATURE_SCRIPTS)
    )
    if not should_execute(prompt, dry_run=args.dry_run, yes=args.yes):
        return 2

    for script in FEATURE_SCRIPTS:
        rc = run_python_script(script, dry_run=args.dry_run)
        if rc != 0:
            print(f"FAILED ({rc}): {script}", file=sys.stderr)
            return rc
    print("\nFeature chain finished. Run: python check_artifacts.py --require features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
