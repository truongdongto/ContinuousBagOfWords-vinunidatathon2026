"""Optimize ensemble weights for v45, mean-scale to v17m, build submission."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

V17M_MEAN_R = 4_516_336.0
V17M_MEAN_C = 3_858_794.0
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_CSV = REPO_ROOT / "data"
DIAG_SUB = REPO_ROOT / "diag_submissions"
SAMPLE_SUB = DATA_CSV / "sample_submission.csv"

def opt_weights(val_stack, y_val, n_starts=30, seed=42):
    n = val_stack.shape[0]
    rng = np.random.RandomState(seed)
    best_w = None; best_mae = np.inf
    def loss(w):
        w = np.maximum(w, 0); w = w / (w.sum() + 1e-12)
        return np.mean(np.abs(w @ val_stack - y_val))
    starts = [np.ones(n)/n]
    for _ in range(n_starts - 1):
        starts.append(rng.dirichlet(np.ones(n)))
    for w0 in starts:
        r = minimize(loss, w0, method='Nelder-Mead',
                     options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
        if r.fun < best_mae:
            best_mae = r.fun
            w = np.maximum(r.x, 0); w = w / (w.sum() + 1e-12)
            best_w = w
    return best_w, best_mae

print("=== Building submission for v45_pattern (v8 + web/review 6 feats) ===")
d = np.load(SCRIPT_DIR / 'ml_v45_pattern_preds.npz', allow_pickle=True)
R_keys = d['Revenue_keys']; C_keys = d['COGS_keys']
R_val = d['Revenue_val'];   R_test = d['Revenue_test']; y_val_R = d['Revenue_y_val']
C_val = d['COGS_val'];      C_test = d['COGS_test'];    y_val_C = d['COGS_y_val']
test_dates = d['test_dates']

w_R, mae_R = opt_weights(R_val, y_val_R)
w_C, mae_C = opt_weights(C_val, y_val_C)
pooled = (mae_R + mae_C) / 2
print(f"val MAE Rev={mae_R:,.0f}  COGS={mae_C:,.0f}  pooled={pooled:,.0f}")
print("w_R: " + ", ".join([f"{k}={w:.3f}" for k,w in zip(R_keys, w_R) if w > 0.01]))
print("w_C: " + ", ".join([f"{k}={w:.3f}" for k,w in zip(C_keys, w_C) if w > 0.01]))

pred_R = w_R @ R_test
pred_C = w_C @ C_test
print(f"Pre-scale mean: R={pred_R.mean():,.0f}  C={pred_C.mean():,.0f}")

scale_R = V17M_MEAN_R / pred_R.mean()
scale_C = V17M_MEAN_C / pred_C.mean()
pred_R_s = np.clip(pred_R * scale_R, 0, None)
pred_C_s = np.clip(pred_C * scale_C, 0, None)
print(f"Scale factors: R={scale_R:.3f}  C={scale_C:.3f}")

template = pd.read_csv(SAMPLE_SUB)
out = template.copy()
out['Revenue'] = out['Date'].astype(str).map(dict(zip(test_dates, pred_R_s)))
out['COGS']    = out['Date'].astype(str).map(dict(zip(test_dates, pred_C_s)))
miss = out[['Revenue','COGS']].isna().sum().sum()
print(f"Missing: {miss}")
if miss:
    out['Revenue'] = out['Revenue'].fillna(V17M_MEAN_R)
    out['COGS']    = out['COGS'].fillna(V17M_MEAN_C)

DIAG_SUB.mkdir(parents=True, exist_ok=True)
out_path = DIAG_SUB / 'submission_v45_web_review.csv'
out.to_csv(out_path, index=False)
print(f"✓ Saved {out_path}")
print(f"Final means: R={out['Revenue'].mean():,.0f}  C={out['COGS'].mean():,.0f}")
