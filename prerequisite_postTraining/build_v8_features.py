"""Build enriched_features_v8_pattern.pkl: v6 (v3+top5 v5) + 6 inventory features.
Source: inventory.csv (monthly snapshots 2012-07 to 2022-12, 126 snapshots).
Features: avg_sell_through_rate, avg_fill_rate, pct_stockout × {lag365, roll90}.
Projection 2023-2024: month-of-year climatology from 2020-2022."""
import os
os.environ['OMP_NUM_THREADS']='3'
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_CSV = REPO_ROOT / "data"

print("Loading inventory.csv...")
inv = pd.read_csv(DATA_CSV / 'inventory.csv', parse_dates=['snapshot_date'],
                  usecols=['snapshot_date','stockout_flag','fill_rate','sell_through_rate'])

# Monthly aggregate (avg across all products in that snapshot)
agg = inv.groupby('snapshot_date').agg(
    avg_sell_through_rate=('sell_through_rate','mean'),
    avg_fill_rate=('fill_rate','mean'),
    pct_stockout=('stockout_flag','mean'),
).reset_index()
agg = agg.rename(columns={'snapshot_date':'month_end'})
agg['year'] = agg['month_end'].dt.year
agg['month'] = agg['month_end'].dt.month
print(f"Monthly aggregates: {len(agg)} rows")
print(agg.head(3))

METRICS = ['avg_sell_through_rate', 'avg_fill_rate', 'pct_stockout']

# Project 2023-2024 monthly snapshots using 2020-2022 month climatology
clim = agg[agg['year'].between(2020, 2022)].groupby('month')[METRICS].mean()
print(f"\n2020-2022 climatology by month:")
print(clim)

future_rows = []
for y in [2023, 2024]:
    for m in range(1, 13):
        if y == 2024 and m > 7:
            continue
        last_day = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
        row = {'month_end': last_day, 'year': y, 'month': m}
        for met in METRICS:
            row[met] = clim.loc[m, met]
        future_rows.append(row)
future = pd.DataFrame(future_rows)
agg_full = pd.concat([agg, future], ignore_index=True).sort_values('month_end').reset_index(drop=True)
print(f"\nAfter projection: {len(agg_full)} monthly rows  (range: {agg_full['month_end'].min()} → {agg_full['month_end'].max()})")

# Build daily template covering 2012-07-04 to 2024-07-01
v6 = pd.read_pickle(SCRIPT_DIR / 'enriched_features_v6_pattern.pkl').sort_values('Date').reset_index(drop=True)
all_dates = pd.DatetimeIndex(v6['Date'].unique())
print(f"\nDaily template: {len(all_dates)} days, {all_dates.min()} → {all_dates.max()}")

# Forward-fill: each day uses its calendar month's snapshot value
daily = pd.DataFrame({'Date': all_dates})
daily['year'] = daily['Date'].dt.year
daily['month'] = daily['Date'].dt.month
daily = daily.merge(agg_full[['year','month'] + METRICS], on=['year','month'], how='left')
print("Daily merge NA per metric:")
print(daily[METRICS].isna().sum())
# Backfill any leading NaN (very early dates before first snapshot 2012-07)
for c in METRICS:
    daily[c] = daily[c].fillna(method='bfill').fillna(method='ffill')

# Lag365 + roll90
daily = daily.sort_values('Date').reset_index(drop=True)
out = daily[['Date']].copy()
for c in METRICS:
    s = daily[c]
    out[f'{c}_lag365'] = s.shift(365)
    out[f'{c}_roll90'] = s.shift(1).rolling(90, min_periods=1).mean()
# Fill leading NaN
for c in out.columns:
    if c != 'Date' and out[c].isna().any():
        out[c] = out[c].fillna(method='bfill').fillna(0)

print("\nNew v8 inventory features (6):")
for c in out.columns:
    if c != 'Date':
        print(f"  na={out[c].isna().sum():2d}  mean={out[c].mean():.4f}  std={out[c].std():.4f}  {c}")

# Verify 2024 has reasonable values
print("\n2024 sample:")
print(out[out['Date'].between('2024-06-01','2024-06-05')])

v8 = v6.merge(out, on='Date', how='left')
new_cols = [c for c in v8.columns if c not in v6.columns]
print(f"\nv6 cols: {len(v6.columns)} → v8 cols: {len(v8.columns)} (added {len(new_cols)} inventory feats)")
v8.to_pickle(SCRIPT_DIR / 'enriched_features_v8_pattern.pkl')
print("✓ Saved enriched_features_v8_pattern.pkl")
