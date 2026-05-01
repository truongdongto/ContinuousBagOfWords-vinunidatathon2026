"""Build enriched_features_v6_pattern.pkl: v3_pattern + only top-5 strongest v5 features.
Selected via LGBM-R importance on v36: pct_female_buyers_lag365 (#4), pct_female_buyers_roll90 (#13),
pct_streetwear_share_lag365 (#25), avg_cogs_per_unit_lag365 (#28), pct_outdoor_share_lag365 (#31).
Goal: keep only orthogonal signals (demographics, premium-product mix), drop weaker correlated ones."""
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
v3 = pd.read_pickle(SCRIPT_DIR / 'enriched_features_v3_pattern.pkl')
v5 = pd.read_pickle(SCRIPT_DIR / 'enriched_features_v5_pattern.pkl')

KEEP = ['pct_female_buyers_lag365', 'pct_female_buyers_roll90',
        'pct_streetwear_share_lag365', 'avg_cogs_per_unit_lag365',
        'pct_outdoor_share_lag365']

extra = v5[['Date'] + KEEP]
v6 = v3.merge(extra, on='Date', how='left')
print(f"v3 cols={len(v3.columns)} → v6 cols={len(v6.columns)} (added {len(KEEP)} curated)")
print(f"NA check: {v6[KEEP].isna().sum().sum()} (should be 0)")
v6.to_pickle(SCRIPT_DIR / 'enriched_features_v6_pattern.pkl')
print("✓ Saved enriched_features_v6_pattern.pkl")
