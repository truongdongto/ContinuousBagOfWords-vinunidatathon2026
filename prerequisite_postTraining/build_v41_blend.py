"""Build blend variants v37 + v40. Find optimal blend weight on validation, then apply
to test CSVs (which are already mean-scaled). Multiple variants for robustness."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_CSV = REPO_ROOT / "data"
DIAG_SUB = REPO_ROOT / "diag_submissions"

# ---------- Step 1: Reconstruct full-ensemble val predictions for each model ----------
def get_val_test_full(npz_path):
    """Reproduce the optimal ensemble blend used in build submission scripts.
    Returns (val_R, val_C, test_R, test_C) with optimal per-target weights."""
    d = np.load(npz_path, allow_pickle=True)
    R_val = d['Revenue_val']; C_val = d['COGS_val']
    R_test = d['Revenue_test']; C_test = d['COGS_test']
    y_val_R = d['Revenue_y_val']; y_val_C = d['COGS_y_val']
    test_dates = d['test_dates']

    # Optimize ensemble weights (same as build_v3X_submission)
    from scipy.optimize import minimize
    def opt_weights(stack, y, n_starts=30, seed=42):
        n = stack.shape[0]; rng = np.random.RandomState(seed)
        best_w = None; best_mae = np.inf
        def loss(w):
            w = np.maximum(w, 0); w = w/(w.sum()+1e-12)
            return np.mean(np.abs(w@stack - y))
        starts = [np.ones(n)/n] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts-1)]
        for w0 in starts:
            r = minimize(loss, w0, method='Nelder-Mead',
                         options={'xatol':1e-6,'fatol':1e-6,'maxiter':5000})
            if r.fun < best_mae:
                best_mae = r.fun
                w = np.maximum(r.x, 0); w = w/(w.sum()+1e-12); best_w = w
        return best_w, best_mae

    w_R, mae_R = opt_weights(R_val, y_val_R)
    w_C, mae_C = opt_weights(C_val, y_val_C)
    print(f"  {npz_path}: val MAE R={mae_R:.0f}  C={mae_C:.0f}")
    val_R = w_R @ R_val; val_C = w_C @ C_val
    tst_R = w_R @ R_test; tst_C = w_C @ C_test
    return val_R, val_C, tst_R, tst_C, y_val_R, y_val_C, test_dates

print("=== Reconstructing val/test predictions for v37 and v40 ===")
v37_valR, v37_valC, v37_tR, v37_tC, y_R, y_C, test_dates = get_val_test_full(
    SCRIPT_DIR / 'ml_v37_pattern_preds.npz')
v40_valR, v40_valC, v40_tR, v40_tC, _, _, _ = get_val_test_full(
    SCRIPT_DIR / 'ml_v40_pattern_preds.npz')

# ---------- Step 2: Find optimal blend weight on val ----------
def best_blend(a_val, b_val, y, name):
    res = minimize_scalar(lambda w: np.mean(np.abs(w*a_val + (1-w)*b_val - y)),
                          bounds=(0,1), method='bounded',
                          options={'xatol':1e-6})
    w_opt = res.x
    base_a = np.mean(np.abs(a_val - y))
    base_b = np.mean(np.abs(b_val - y))
    blend_mae = res.fun
    print(f"  {name}: w(v37)={w_opt:.3f}  v37={base_a:.0f}  v40={base_b:.0f}  blend={blend_mae:.0f}  Δ={blend_mae-min(base_a,base_b):+.0f}")
    return w_opt, blend_mae

print("\n=== Optimal blend weights ===")
wR_opt, _ = best_blend(v37_valR, v40_valR, y_R, 'Revenue')
wC_opt, _ = best_blend(v37_valC, v40_valC, y_C, 'COGS')

# ---------- Step 3: Apply blends to test, mean-scale, save ----------
V17M_R = 4_516_336.0; V17M_C = 3_858_794.0

def make_blend_csv(w_R, w_C, label):
    pred_R = w_R*v37_tR + (1-w_R)*v40_tR
    pred_C = w_C*v37_tC + (1-w_C)*v40_tC
    sR = V17M_R / pred_R.mean(); sC = V17M_C / pred_C.mean()
    pred_R_s = np.clip(pred_R*sR, 0, None)
    pred_C_s = np.clip(pred_C*sC, 0, None)
    template = pd.read_csv(DATA_CSV / 'sample_submission.csv')
    out = template.copy()
    out['Revenue'] = out['Date'].astype(str).map(dict(zip(test_dates, pred_R_s)))
    out['COGS']    = out['Date'].astype(str).map(dict(zip(test_dates, pred_C_s)))
    out['Revenue'] = out['Revenue'].fillna(V17M_R)
    out['COGS']    = out['COGS'].fillna(V17M_C)
    DIAG_SUB.mkdir(parents=True, exist_ok=True)
    path = DIAG_SUB / f'submission_v41_{label}.csv'
    out.to_csv(path, index=False)
    # Compute val pooled MAE for this blend
    blend_valR = w_R*v37_valR + (1-w_R)*v40_valR
    blend_valC = w_C*v37_valC + (1-w_C)*v40_valC
    pooled = (np.mean(np.abs(blend_valR-y_R)) + np.mean(np.abs(blend_valC-y_C)))/2
    print(f"  {label}: w_R={w_R:.2f}  w_C={w_C:.2f}  val pooled={pooled:.0f}  →  {path}")
    return out

print("\n=== Building blend variants ===")
# A. Simple 50/50
make_blend_csv(0.50, 0.50, 'A_5050')
# B. 70/30 favor v37
make_blend_csv(0.70, 0.70, 'B_7030')
# C. Per-target optimal (val-derived)
make_blend_csv(wR_opt, wC_opt, 'C_optval')
# D. Asymmetric: v37 for R (best on R), v40 for C (more stable)
make_blend_csv(0.85, 0.40, 'D_asym')
# E. Inv-MAE weighted (using val ensemble MAE)
v37_R_mae = np.mean(np.abs(v37_valR - y_R)); v40_R_mae = np.mean(np.abs(v40_valR - y_R))
v37_C_mae = np.mean(np.abs(v37_valC - y_C)); v40_C_mae = np.mean(np.abs(v40_valC - y_C))
wR_inv = (1/v37_R_mae) / (1/v37_R_mae + 1/v40_R_mae)
wC_inv = (1/v37_C_mae) / (1/v37_C_mae + 1/v40_C_mae)
make_blend_csv(wR_inv, wC_inv, 'E_invmae')

# Reference: v37 alone (pooled MAE for comparison)
v37_pooled = (np.mean(np.abs(v37_valR-y_R)) + np.mean(np.abs(v37_valC-y_C)))/2
v40_pooled = (np.mean(np.abs(v40_valR-y_R)) + np.mean(np.abs(v40_valC-y_C)))/2
print(f"\nReference val pooled: v37={v37_pooled:.0f}  v40={v40_pooled:.0f}")
