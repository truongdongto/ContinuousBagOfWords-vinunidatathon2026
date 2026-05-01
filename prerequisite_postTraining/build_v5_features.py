"""Build enriched_features_v5_pattern.pkl: v3 (pattern, 136 cols) + customer/product daily aggregates."""
import os
os.environ['OMP_NUM_THREADS']='3'
os.environ['MKL_NUM_THREADS']='3'
os.environ['OPENBLAS_NUM_THREADS']='3'
import pandas as pd
import numpy as np
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_CSV = REPO_ROOT / "data"

t0 = time.time()
print("Loading raw tables...", flush=True)
orders = pd.read_csv(DATA_CSV / "orders.csv", usecols=['order_id','order_date','customer_id'],
                     parse_dates=['order_date'])
order_items = pd.read_csv(DATA_CSV / "order_items.csv", usecols=['order_id','product_id','quantity','unit_price'])
customers = pd.read_csv(DATA_CSV / "customers.csv", usecols=['customer_id','signup_date','gender'],
                        parse_dates=['signup_date'])
products = pd.read_csv(DATA_CSV / "products.csv", usecols=['product_id','category','cogs','price'])
print(f"Loaded in {time.time()-t0:.1f}s. orders={len(orders)}, oi={len(order_items)}, "
      f"customers={len(customers)}, products={len(products)}", flush=True)

# Daily template
v3 = pd.read_pickle(SCRIPT_DIR / 'enriched_features_v3_pattern.pkl').sort_values('Date').reset_index(drop=True)
all_dates = pd.DatetimeIndex(v3['Date'].unique())
print(f"Date range: {all_dates.min()} → {all_dates.max()} ({len(all_dates)} days)", flush=True)

# ============ 1. Signup features (from customers.signup_date) ============
print("\n[1] Daily signups...", flush=True)
sg = customers.groupby('signup_date').size().rename('n_signups').reset_index()
sg = sg.rename(columns={'signup_date':'Date'})
print(f"  signup days: {len(sg)}, total signups: {sg['n_signups'].sum()}")

# ============ 2. Buyer demographics (orders ⋈ customers) ============
print("\n[2] Buyer demographics...", flush=True)
oc = orders.merge(customers[['customer_id','gender']], on='customer_id', how='left')
def daily_buyer_agg(g):
    n_unique = g['customer_id'].nunique()
    pct_f = (g['gender']=='Female').mean()
    return pd.Series({'n_unique_buyers': n_unique, 'pct_female_buyers': pct_f})
buyer_daily = oc.groupby('order_date').apply(daily_buyer_agg).reset_index()
buyer_daily = buyer_daily.rename(columns={'order_date':'Date'})
print(f"  buyer_daily shape: {buyer_daily.shape}")
print(buyer_daily.head(3))

# ============ 3. Product category mix (order_items ⋈ products ⋈ orders) ============
print("\n[3] Product category mix per day...", flush=True)
oi_p = order_items.merge(products[['product_id','category','cogs']], on='product_id', how='left')
oi_p = oi_p.merge(orders[['order_id','order_date']], on='order_id', how='left')
oi_p['extended_cogs'] = oi_p['quantity'] * oi_p['cogs']

def daily_product_agg(g):
    total_qty = g['quantity'].sum()
    streetwear_qty = g[g['category']=='Streetwear']['quantity'].sum()
    outdoor_qty    = g[g['category']=='Outdoor']['quantity'].sum()
    avg_cogs_unit  = g['extended_cogs'].sum() / max(total_qty, 1)
    return pd.Series({
        'pct_streetwear_share': streetwear_qty / max(total_qty, 1),
        'pct_outdoor_share':    outdoor_qty / max(total_qty, 1),
        'avg_cogs_per_unit':    avg_cogs_unit,
    })

prod_daily = oi_p.groupby('order_date').apply(daily_product_agg).reset_index()
prod_daily = prod_daily.rename(columns={'order_date':'Date'})
print(f"  prod_daily shape: {prod_daily.shape}")
print(prod_daily.head(3))

# ============ 4. Combine and create lag/roll features ============
print("\n[4] Combine + lag365 + roll30/90...", flush=True)
template = pd.DataFrame({'Date': all_dates}).sort_values('Date').reset_index(drop=True)
df = template.merge(sg, on='Date', how='left')
df = df.merge(buyer_daily, on='Date', how='left')
df = df.merge(prod_daily, on='Date', how='left')
# Fill na (early dates before tables started)
df = df.fillna({'n_signups': 0,
                'n_unique_buyers': 0,
                'pct_female_buyers': 0.5,
                'pct_streetwear_share': 0.0,
                'pct_outdoor_share':    0.0,
                'avg_cogs_per_unit':    0.0,
               })
agg_cols = ['n_signups', 'n_unique_buyers', 'pct_female_buyers',
            'pct_streetwear_share', 'pct_outdoor_share', 'avg_cogs_per_unit']

# Sort by date for rolling
df = df.sort_values('Date').reset_index(drop=True)

# IMPORTANT: raw data ends 2022-12-31. For 2023-2024 dates the daily aggregates are
# empty (n_signups=0, etc.) which would make lag365 of 2024 → 2023 → 0.
# Solution: project daily aggregates for 2023 using DOY-aligned 2022 values
# (effectively seasonal-naive). lag365 of 2024 then resolves to the projected 2023
# = 2022 real values. This mirrors the projection strategy used for promo features.
RAW_END = pd.Timestamp('2022-12-31')
df['doy'] = df['Date'].dt.dayofyear
# Compute DOY climatology from last 3 full years of raw data (2020-2022)
clim_src = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= RAW_END)]
doy_means = clim_src.groupby('doy')[agg_cols].mean()
# Fill 2023-2024 dates with DOY climatology
mask_future = df['Date'] > RAW_END
for c in agg_cols:
    df.loc[mask_future, c] = df.loc[mask_future, 'doy'].map(doy_means[c]).values
df = df.drop(columns=['doy'])

out = df[['Date']].copy()
for c in agg_cols:
    s = df[c]
    out[f'{c}_lag365'] = s.shift(365)
    out[f'{c}_roll90'] = s.shift(1).rolling(90, min_periods=1).mean()  # shift(1) avoids leakage
print(f"  v5 new features ({len(out.columns)-1}):")
for c in out.columns:
    if c != 'Date':
        print(f"    {c}: na={out[c].isna().sum()}, mean={out[c].mean():.4f}")

# Forward fill any NaN at the start (early period not yet 365 days history)
for c in out.columns:
    if c != 'Date' and out[c].isna().any():
        out[c] = out[c].fillna(method='bfill').fillna(0)

print(f"\nMerging with v3_pattern...")
v5 = v3.merge(out, on='Date', how='left')
new_cols = [c for c in v5.columns if c not in v3.columns]
print(f"v3 cols: {len(v3.columns)} → v5 cols: {len(v5.columns)} (added {len(new_cols)} new)")
print(f"New cols: {new_cols}")
v5.to_pickle(SCRIPT_DIR / 'enriched_features_v5_pattern.pkl')
print(f"\n✓ Saved enriched_features_v5_pattern.pkl in {time.time()-t0:.1f}s total")
