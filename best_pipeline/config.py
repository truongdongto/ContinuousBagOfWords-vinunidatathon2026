"""Paths and constants for reproducing best submission pipeline."""
from pathlib import Path

# This folder …/final-submisison/best_pipeline/
PIPELINE_ROOT = Path(__file__).resolve().parent
# Project root: final-submisison/ (CSV in data/, scripts in prerequisite_postTraining/)
DATA_ROOT = PIPELINE_ROOT.parent

# Feature/train/submit/blend .py files live here; subprocess cwd stays DATA_ROOT so paths stay valid.
SCRIPT_DIR = Path("prerequisite_postTraining")


def _script(name: str) -> str:
    """Path relative to DATA_ROOT passed to `python <path>`."""
    return str(SCRIPT_DIR / name)


V17M_MEAN_R = 4_516_336.0
V17M_MEAN_C = 3_858_794.0

# v41_B: blended ensemble preds before mean-scale (build_v41_blend.py variant B)
W41_V37 = 0.70
W41_V40 = 0.30

# v46 best reported: weight on v41_B row (submission CSV, already scaled)
W46_WEIGHT_V41B = 0.85

# Prerequisites (no script in repo to rebuild v2/v3 from raw alone)
PREREQUISITES = (
    _script("enriched_features_v2.pkl"), 
    _script("enriched_features_v3_pattern.pkl")
)

# Sequential feature builders (cwd = DATA_ROOT); require v3_pattern + raw CSV.
FEATURE_SCRIPTS = (
    _script("build_v5_features.py"),
    _script("build_v6_features.py"),
    _script("build_v8_features.py"),
    _script("build_v10_features.py"),
)

# Train jobs: script name → expected predictions archive
TRAIN_JOBS = (
    (_script("train_v37.py"), _script("ml_v37_pattern_preds.npz")),
    (_script("train_v40.py"), _script("ml_v40_pattern_preds.npz")),
    (_script("train_v45.py"), _script("ml_v45_pattern_preds.npz")),
)

SUBMISSION_BUILDERS = (
    _script("build_v37_submission.py"),
    _script("build_v40_submission.py"),
    _script("build_v45_submission.py"),
)

# Blend chain (same SCRIPT_DIR as above)
BLEND_SCRIPT_V41 = _script("build_v41_blend.py")
BLEND_SCRIPT_V46_GRID = _script("build_v46_v41b_grid_blend.py")

SUBMISSION_EXPECTED_FILES = (
    "diag_submissions/submission_v37_cust_prod.csv",
    "diag_submissions/submission_v40_cust_prod.csv",
    "diag_submissions/submission_v45_web_review.csv",
)

PICKLE_CHAIN = (
    _script("enriched_features_v5_pattern.pkl"),
    _script("enriched_features_v6_pattern.pkl"),
    _script("enriched_features_v8_pattern.pkl"),
    _script("enriched_features_v10_pattern.pkl"),
)

BEST_BLEND_OUTPUT = "diag_submissions/submission_v46_v41b_v45_w85.csv"
V41B_SUBMISSION = "diag_submissions/submission_v41_B_7030.csv"
