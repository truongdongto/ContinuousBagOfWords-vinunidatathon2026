"""Train 4-way ensemble (LGB+XGB+CB+Ridge) for v33_safe and v33_pattern variants.

Checkpointing: saves partial results after each model type. Re-running the script
will skip already-computed sections, allowing recovery from OOM crashes.
"""
import os
# Throttle CPU to leave headroom for the system
os.environ['OMP_NUM_THREADS'] = '3'
os.environ['MKL_NUM_THREADS'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '3'
os.environ['NUMEXPR_NUM_THREADS'] = '3'
import sys
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

VARIANT = 'pattern'  # v40 builds on v33_pattern (verified best LB)

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = SCRIPT_DIR / f'ml_v40_{VARIANT}_ckpt.npz'
OUT_PATH  = SCRIPT_DIR / f'ml_v40_{VARIANT}_preds.npz'

print(f"=== Training v40_{VARIANT} (v3 promo + v6 + 6 inventory features) ===", flush=True)
t_start = time.time()

# ============ Load data ============
df = pd.read_pickle(SCRIPT_DIR / f'enriched_features_v8_{VARIANT}.pkl').sort_values('Date').reset_index(drop=True)
print(f"Loaded {len(df)} rows, {len(df.columns)} cols", flush=True)

EXCLUDE = {'Date', 'Revenue', 'COGS'}
feature_cols = [c for c in df.columns if c not in EXCLUDE]
print(f"Features: {len(feature_cols)}", flush=True)

TRAIN_START = '2014-01-01'; TRAIN_END = '2022-06-30'
VAL_START   = '2022-07-01'; VAL_END   = '2022-12-31'
TEST_START  = '2023-01-01'; TEST_END  = '2024-07-01'

m_train = (df['Date']>=TRAIN_START) & (df['Date']<=TRAIN_END)
m_val   = (df['Date']>=VAL_START)   & (df['Date']<=VAL_END)
m_full  = (df['Date']>=TRAIN_START) & (df['Date']<=VAL_END)
m_test  = (df['Date']>=TEST_START)  & (df['Date']<=TEST_END)

X_train = df.loc[m_train, feature_cols].astype(np.float32).values
X_val   = df.loc[m_val,   feature_cols].astype(np.float32).values
X_full  = df.loc[m_full,  feature_cols].astype(np.float32).values
X_test  = df.loc[m_test,  feature_cols].astype(np.float32).values
test_dates = df.loc[m_test, 'Date'].astype(str).values

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Full: {X_full.shape}, Test: {X_test.shape}", flush=True)

# Per-target labels
y_train_R = df.loc[m_train, 'Revenue'].astype(np.float64).values
y_val_R   = df.loc[m_val,   'Revenue'].astype(np.float64).values
y_full_R  = df.loc[m_full,  'Revenue'].astype(np.float64).values
y_train_C = df.loc[m_train, 'COGS'].astype(np.float64).values
y_val_C   = df.loc[m_val,   'COGS'].astype(np.float64).values
y_full_C  = df.loc[m_full,  'COGS'].astype(np.float64).values
del df; gc.collect()

LGB_SEEDS = [11, 22, 33, 44, 55]
XGB_SEEDS = [101, 202, 303, 404, 505]
CB_SEEDS  = [7, 17, 27]

# ============ Checkpoint helpers ============
def load_ckpt():
    if os.path.exists(CKPT_PATH):
        d = dict(np.load(CKPT_PATH, allow_pickle=True))
        print(f"  [ckpt] loaded {CKPT_PATH} keys={list(d.keys())}", flush=True)
        return d
    return {}

def save_ckpt(d):
    np.savez(CKPT_PATH, **d)
    print(f"  [ckpt] saved {CKPT_PATH}", flush=True)

ckpt = load_ckpt()

# ============ Section 1: Ridge (do FIRST, smallest mem footprint) ============
if 'R_ridge_val' not in ckpt:
    print("\n[1] Training Ridge (both targets)...", flush=True)
    t = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(np.nan_to_num(X_train.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    X_val_s   = scaler.transform(np.nan_to_num(X_val.astype(np.float64),   nan=0.0, posinf=0.0, neginf=0.0))
    # Pick alpha by 5-fold CV on train, but tiny grid
    from sklearn.linear_model import RidgeCV
    rid_R = RidgeCV(alphas=np.array([0.1, 1.0, 10.0, 100.0]), cv=5)
    rid_R.fit(X_train_s, y_train_R)
    val_R = rid_R.predict(X_val_s)
    alpha_R = rid_R.alpha_
    rid_C = RidgeCV(alphas=np.array([0.1, 1.0, 10.0, 100.0]), cv=5)
    rid_C.fit(X_train_s, y_train_C)
    val_C = rid_C.predict(X_val_s)
    alpha_C = rid_C.alpha_
    print(f"  Ridge val (alpha_R={alpha_R}, alpha_C={alpha_C}) in {time.time()-t:.1f}s", flush=True)
    del rid_R, rid_C, X_train_s, X_val_s, scaler; gc.collect()

    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(np.nan_to_num(X_full.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    X_test_s = scaler_full.transform(np.nan_to_num(X_test.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    rid_R_full = Ridge(alpha=alpha_R); rid_R_full.fit(X_full_s, y_full_R)
    test_R = rid_R_full.predict(X_test_s)
    rid_C_full = Ridge(alpha=alpha_C); rid_C_full.fit(X_full_s, y_full_C)
    test_C = rid_C_full.predict(X_test_s)
    del rid_R_full, rid_C_full, X_full_s, X_test_s, scaler_full; gc.collect()
    print(f"  Ridge full done in {time.time()-t:.1f}s", flush=True)

    ckpt['R_ridge_val']  = val_R
    ckpt['R_ridge_test'] = test_R
    ckpt['C_ridge_val']  = val_C
    ckpt['C_ridge_test'] = test_C
    save_ckpt(ckpt)
else:
    print("[1] Ridge already done, skipping.", flush=True)

# ============ Section 2: LGBM ============
if 'R_lgb_val' not in ckpt:
    print("\n[2] Training LGBM (both targets)...", flush=True)
    import lightgbm as lgb
    t = time.time()

    lgb_val_R = []; lgb_test_R = []; lgb_val_C = []; lgb_test_C = []
    lgb_imp_R = []; lgb_imp_C = []
    for s in LGB_SEEDS:
        params = dict(
            objective='regression_l1', learning_rate=0.03, num_leaves=63,
            min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, verbose=-1, seed=s, n_estimators=3000)
        # Revenue
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train_R, eval_set=[(X_val, y_val_R)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        bi = m.best_iteration_
        lgb_val_R.append(m.predict(X_val))
        lgb_imp_R.append(m.booster_.feature_importance(importance_type='gain'))
        del m; gc.collect()
        m_full = lgb.LGBMRegressor(**{**params, 'n_estimators': bi})
        m_full.fit(X_full, y_full_R)
        lgb_test_R.append(m_full.predict(X_test))
        del m_full; gc.collect()
        # COGS
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train_C, eval_set=[(X_val, y_val_C)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        bi = m.best_iteration_
        lgb_val_C.append(m.predict(X_val))
        lgb_imp_C.append(m.booster_.feature_importance(importance_type='gain'))
        del m; gc.collect()
        m_full = lgb.LGBMRegressor(**{**params, 'n_estimators': bi})
        m_full.fit(X_full, y_full_C)
        lgb_test_C.append(m_full.predict(X_test))
        del m_full; gc.collect()
        print(f"  lgb_{s} done ({time.time()-t:.1f}s)", flush=True)

    ckpt['R_lgb_val']  = np.stack(lgb_val_R)
    ckpt['R_lgb_test'] = np.stack(lgb_test_R)
    ckpt['C_lgb_val']  = np.stack(lgb_val_C)
    ckpt['C_lgb_test'] = np.stack(lgb_test_C)
    ckpt['R_lgb_imp']  = np.stack(lgb_imp_R).mean(axis=0)
    ckpt['C_lgb_imp']  = np.stack(lgb_imp_C).mean(axis=0)
    save_ckpt(ckpt)
    del lgb_val_R, lgb_test_R, lgb_val_C, lgb_test_C, lgb_imp_R, lgb_imp_C; gc.collect()
else:
    print("[2] LGBM already done, skipping.", flush=True)

# ============ Section 3: XGBoost ============
if 'R_xgb_val' not in ckpt:
    print("\n[3] Training XGB (both targets)...", flush=True)
    import xgboost as xgb
    t = time.time()
    xgb_val_R=[]; xgb_test_R=[]; xgb_val_C=[]; xgb_test_C=[]
    for s in XGB_SEEDS:
        params = dict(
            objective='reg:absoluteerror', learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, tree_method='hist',
            random_state=s, n_estimators=3000, early_stopping_rounds=100, verbosity=0)
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train_R, eval_set=[(X_val, y_val_R)], verbose=False)
        bi = m.best_iteration + 1
        xgb_val_R.append(m.predict(X_val))
        del m; gc.collect()
        p2 = {k:v for k,v in params.items() if k != 'early_stopping_rounds'}
        p2['n_estimators'] = bi
        m_full = xgb.XGBRegressor(**p2); m_full.fit(X_full, y_full_R, verbose=False)
        xgb_test_R.append(m_full.predict(X_test))
        del m_full; gc.collect()
        m = xgb.XGBRegressor(**params)
        m.fit(X_train, y_train_C, eval_set=[(X_val, y_val_C)], verbose=False)
        bi = m.best_iteration + 1
        xgb_val_C.append(m.predict(X_val))
        del m; gc.collect()
        p2['n_estimators'] = bi
        m_full = xgb.XGBRegressor(**p2); m_full.fit(X_full, y_full_C, verbose=False)
        xgb_test_C.append(m_full.predict(X_test))
        del m_full; gc.collect()
        print(f"  xgb_{s} done ({time.time()-t:.1f}s)", flush=True)
    ckpt['R_xgb_val']  = np.stack(xgb_val_R)
    ckpt['R_xgb_test'] = np.stack(xgb_test_R)
    ckpt['C_xgb_val']  = np.stack(xgb_val_C)
    ckpt['C_xgb_test'] = np.stack(xgb_test_C)
    save_ckpt(ckpt)
    del xgb_val_R, xgb_test_R, xgb_val_C, xgb_test_C; gc.collect()
else:
    print("[3] XGB already done, skipping.", flush=True)

# ============ Section 4: CatBoost ============
if 'R_cb_val' not in ckpt:
    print("\n[4] Training CatBoost (both targets)...", flush=True)
    from catboost import CatBoostRegressor
    t = time.time()
    cb_val_R=[]; cb_test_R=[]; cb_val_C=[]; cb_test_C=[]
    for s in CB_SEEDS:
        params = dict(loss_function='MAE', learning_rate=0.05, depth=7,
                      l2_leaf_reg=3, iterations=3000, random_seed=s, verbose=False,
                      thread_count=3)
        m = CatBoostRegressor(**params)
        m.fit(X_train, y_train_R, eval_set=(X_val, y_val_R),
              early_stopping_rounds=100, verbose=False)
        bi = m.best_iteration_ + 1
        cb_val_R.append(m.predict(X_val))
        del m; gc.collect()
        m_full = CatBoostRegressor(**{**params, 'iterations': bi})
        m_full.fit(X_full, y_full_R, verbose=False)
        cb_test_R.append(m_full.predict(X_test))
        del m_full; gc.collect()
        m = CatBoostRegressor(**params)
        m.fit(X_train, y_train_C, eval_set=(X_val, y_val_C),
              early_stopping_rounds=100, verbose=False)
        bi = m.best_iteration_ + 1
        cb_val_C.append(m.predict(X_val))
        del m; gc.collect()
        m_full = CatBoostRegressor(**{**params, 'iterations': bi})
        m_full.fit(X_full, y_full_C, verbose=False)
        cb_test_C.append(m_full.predict(X_test))
        del m_full; gc.collect()
        print(f"  cb_{s} done ({time.time()-t:.1f}s)", flush=True)
    ckpt['R_cb_val']  = np.stack(cb_val_R)
    ckpt['R_cb_test'] = np.stack(cb_test_R)
    ckpt['C_cb_val']  = np.stack(cb_val_C)
    ckpt['C_cb_test'] = np.stack(cb_test_C)
    save_ckpt(ckpt)
    del cb_val_R, cb_test_R, cb_val_C, cb_test_C; gc.collect()
else:
    print("[4] CatBoost already done, skipping.", flush=True)

# ============ Final: combine into final preds file ============
print("\n[5] Combining into final preds file...", flush=True)

def stack_for(prefix):
    keys = []
    val_arrs = []
    test_arrs = []
    # LGB
    for i, s in enumerate(LGB_SEEDS):
        keys.append(f'lgb_{s}')
        val_arrs.append(ckpt[f'{prefix}_lgb_val'][i])
        test_arrs.append(ckpt[f'{prefix}_lgb_test'][i])
    for i, s in enumerate(XGB_SEEDS):
        keys.append(f'xgb_{s}')
        val_arrs.append(ckpt[f'{prefix}_xgb_val'][i])
        test_arrs.append(ckpt[f'{prefix}_xgb_test'][i])
    for i, s in enumerate(CB_SEEDS):
        keys.append(f'cb_{s}')
        val_arrs.append(ckpt[f'{prefix}_cb_val'][i])
        test_arrs.append(ckpt[f'{prefix}_cb_test'][i])
    keys.append('ridge')
    val_arrs.append(ckpt[f'{prefix}_ridge_val'])
    test_arrs.append(ckpt[f'{prefix}_ridge_test'])
    return np.array(keys), np.stack(val_arrs), np.stack(test_arrs)

R_keys, R_val, R_test = stack_for('R')
C_keys, C_val, C_test = stack_for('C')

print(f"\nPer-model val MAE Revenue:")
for i, k in enumerate(R_keys):
    print(f"  {k}: {np.mean(np.abs(R_val[i] - y_val_R)):,.0f}")
print(f"\nPer-model val MAE COGS:")
for i, k in enumerate(C_keys):
    print(f"  {k}: {np.mean(np.abs(C_val[i] - y_val_C)):,.0f}")

np.savez(OUT_PATH,
    Revenue_keys=R_keys, Revenue_val=R_val, Revenue_test=R_test,
    Revenue_y_val=y_val_R,
    COGS_keys=C_keys, COGS_val=C_val, COGS_test=C_test,
    COGS_y_val=y_val_C,
    Revenue_lgb_imp=ckpt['R_lgb_imp'],
    COGS_lgb_imp=ckpt['C_lgb_imp'],
    feature_cols=np.array(feature_cols),
    test_dates=test_dates,
)
print(f"\n✓ Saved {OUT_PATH} (total time {time.time()-t_start:.1f}s)", flush=True)
