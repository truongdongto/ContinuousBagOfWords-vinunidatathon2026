#!/usr/bin/env python3
"""Single entrypoint: ordered steps with --dry-run / --yes (no accidental heavy runs)."""
import argparse
import sys

from run_features import main as feats_main
from run_train import main as train_main
from run_submissions import main as submit_main
from run_blend_best import main as blend_main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reproduce best submission pipeline (orchestrator).",
        epilog=(
            "Steps: features → train → submit → blend. "
            "Use check_artifacts.py for read-only validation."
        ),
    )
    ap.add_argument(
        "--step",
        choices=("all", "features", "train", "submit", "blend"),
        default="all",
        help="Run from this step onward (for 'all', runs features→train→submit→blend)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print intended commands only.")
    ap.add_argument("--yes", action="store_true", help="Non-interactive: answer yes to confirmations.")
    ap.add_argument(
        "--blend-skip-v41",
        action="store_true",
        help="Forwarded to blend step only: skip build_v41 if CSV exists.",
    )
    args = ap.parse_args()

    steps_order = []
    if args.step == "all":
        steps_order = ["features", "train", "submit", "blend"]
    else:
        seq = ["features", "train", "submit", "blend"]
        steps_order = seq[seq.index(args.step) :]

    # Subprocess-style argv for run_*.py parsers
    sys.argv_keep = sys.argv.copy()
    rc = 0
    try:
        for step in steps_order:
            if step == "features":
                sys.argv = ["run_features.py"] + (
                    ["--dry-run"] if args.dry_run else []
                ) + (["--yes"] if args.yes else [])
                rc = feats_main()
            elif step == "train":
                sys.argv = ["run_train.py"] + (
                    ["--dry-run"] if args.dry_run else []
                ) + (["--yes"] if args.yes else [])
                rc = train_main()
            elif step == "submit":
                sys.argv = ["run_submissions.py"] + (
                    ["--dry-run"] if args.dry_run else []
                ) + (["--yes"] if args.yes else [])
                rc = submit_main()
            elif step == "blend":
                extra = []
                if args.dry_run:
                    extra.append("--dry-run")
                if args.yes:
                    extra.append("--yes")
                if args.blend_skip_v41:
                    extra.append("--skip-v41")
                sys.argv = ["run_blend_best.py"] + extra
                rc = blend_main()
            else:
                rc = 0

            if rc != 0:
                print(f"pipeline aborted at step={step}, exit_code={rc}", file=sys.stderr)
                return rc
    finally:
        sys.argv = sys.argv_keep

    print("\nDone.", "dry-run." if args.dry_run else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
