#!/usr/bin/env python3
"""Reproduce v41_B via build_v41_blend.py, then v46 blends (includes best w85) via build_v46_v41b_grid_blend.py."""
from pathlib import Path
import argparse
import sys

from config import (
    BEST_BLEND_OUTPUT,
    V41B_SUBMISSION,
    DATA_ROOT,
    BLEND_SCRIPT_V41,
    BLEND_SCRIPT_V46_GRID,
    TRAIN_JOBS,
)
from run_support import should_execute, run_python_script, check_files


def main() -> int:
    ap = argparse.ArgumentParser(description="Blend v41_B then v41+v45 (best: w85 on v41_B).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument(
        "--skip-v41",
        action="store_true",
        help="Skip running build_v41_blend.py when v41 CSV exists.",
    )
    args = ap.parse_args()

    blend_npz = (TRAIN_JOBS[0][1], TRAIN_JOBS[1][1])
    _, miss_npz = check_files(blend_npz)
    if miss_npz:
        print("Missing NPZ:", miss_npz, file=sys.stderr)
        return 1

    _, miss45 = check_files(("diag_submissions/submission_v45_web_review.csv",))
    if miss45:
        print("Missing diag_submissions/submission_v45_web_review.csv — run run_submissions.py.", file=sys.stderr)
        return 1

    v41_exists = Path(DATA_ROOT / V41B_SUBMISSION).exists()

    if not (args.skip_v41 and v41_exists):
        p1 = "Run build_v41_blend.py (creates submission_v41_B_7030.csv among variants)."
        if not should_execute(p1, dry_run=args.dry_run, yes=args.yes):
            return 2
        rc = run_python_script(BLEND_SCRIPT_V41, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if not Path(DATA_ROOT / V41B_SUBMISSION).exists() and not args.dry_run:
        print(f"Expected {V41B_SUBMISSION} after build_v41_blend.py", file=sys.stderr)
        return 1

    p2 = "Run build_v46_v41b_grid_blend.py (writes submission_v46_v41b_v45_w*.csv; best LB: w85)."
    if not should_execute(p2, dry_run=args.dry_run, yes=args.yes):
        return 2
    rc = run_python_script(BLEND_SCRIPT_V46_GRID, dry_run=args.dry_run)
    if rc != 0:
        return rc

    print(f"\nBest-report file: {BEST_BLEND_OUTPUT}")
    print("Also check w90/w95 hedges under diag_submissions/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
