"""enriched_features_v10_pattern.pkl: v8 + 6 features from web_traffic + reviews.
Roots: bounce_rate, avg_session_duration_sec, mean_review_rating (daily).
Each: lag365 + roll90. DOY climatology 2020-2022 for Date > 2022-12-31."""
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_CSV = REPO_ROOT / "data"

RAW_END = pd.Timestamp('2022-12-31')

print("Loading web_traffic, reviews...")
wt = pd.read_csv(DATA_CSV / 'web_traffic.csv', parse_dates=['date'])
wt = wt.rename(columns={'date': 'Date'})
# one row per date
daily_wt = wt.groupby('Date', as_index=False).agg(
    bounce_rate=('bounce_rate', 'mean'),
    avg_session_duration_sec=('avg_session_duration_sec', 'mean'),
)

rev = pd.read_csv(DATA_CSV / 'reviews.csv', parse_dates=['review_date'])
rev_daily = rev.groupby('review_date').agg(
    mean_review_rating=('rating', 'mean'),
).reset_index().rename(columns={'review_date': 'Date'})

v8 = pd.read_pickle(SCRIPT_DIR / 'enriched_features_v8_pattern.pkl').sort_values('Date').reset_index(drop=True)
all_dates = pd.DatetimeIndex(v8['Date'].unique())
template = pd.DataFrame({'Date': all_dates})

df = template.merge(daily_wt, on='Date', how='left').merge(rev_daily, on='Date', how='left')
df['bounce_rate'] = df['bounce_rate'].fillna(0.0)
df['avg_session_duration_sec'] = df['avg_session_duration_sec'].fillna(0.0)
df['mean_review_rating'] = df['mean_review_rating'].fillna(0.0)

agg_cols = ['bounce_rate', 'avg_session_duration_sec', 'mean_review_rating']
df = df.sort_values('Date').reset_index(drop=True)
df['doy'] = df['Date'].dt.dayofyear
clim_src = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= RAW_END)]
doy_means = clim_src.groupby('doy')[agg_cols].mean()
mask_future = df['Date'] > RAW_END
for c in agg_cols:
    df.loc[mask_future, c] = df.loc[mask_future, 'doy'].map(doy_means[c]).values
df = df.drop(columns=['doy'])

out = df[['Date']].copy()
for c in agg_cols:
    s = df[c]
    out[f'{c}_lag365'] = s.shift(365)
    out[f'{c}_roll90'] = s.shift(1).rolling(90, min_periods=1).mean()
for c in out.columns:
    if c != 'Date' and out[c].isna().any():
        out[c] = out[c].bfill().fillna(0)

v10 = v8.merge(out, on='Date', how='left')
new_cols = [c for c in v10.columns if c not in v8.columns]
print(f"v8 → v10: +{len(new_cols)} cols: {new_cols}")
v10.to_pickle(SCRIPT_DIR / 'enriched_features_v10_pattern.pkl')
print("✓ Saved enriched_features_v10_pattern.pkl")
