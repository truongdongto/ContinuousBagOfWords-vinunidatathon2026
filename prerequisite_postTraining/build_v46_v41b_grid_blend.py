"""Grid blend v41_B with v45: w * v41_B + (1-w) * v45, w in {0.85, 0.9, 0.95}. Rescale v17m."""
from pathlib import Path

import numpy as np
import pandas as pd

V17M_R = 4_516_336.0
V17M_C = 3_858_794.0

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAG_SUB = REPO_ROOT / "diag_submissions"

v41 = pd.read_csv(DIAG_SUB / 'submission_v41_B_7030.csv')
v45 = pd.read_csv(DIAG_SUB / 'submission_v45_web_review.csv')
merged = v41.merge(v45, on='Date', suffixes=('_41', '_45'))
merged['Date'] = merged['Date'].astype(str)
r41 = merged['Revenue_41'].values.astype(np.float64)
c41 = merged['COGS_41'].values.astype(np.float64)
r45 = merged['Revenue_45'].values.astype(np.float64)
c45 = merged['COGS_45'].values.astype(np.float64)
dates = merged['Date'].values

DIAG_SUB.mkdir(parents=True, exist_ok=True)

print('MAD v45 vs v41 (test):')
print(f'  MAD R: {np.mean(np.abs(r45-r41)):,.0f}')
print(f'  MAD C: {np.mean(np.abs(c45-c41)):,.0f}')

for w in [0.85, 0.9, 0.95]:
    pred_R = w * r41 + (1 - w) * r45
    pred_C = w * c41 + (1 - w) * c45
    sR = V17M_R / pred_R.mean()
    sC = V17M_C / pred_C.mean()
    pred_R = np.clip(pred_R * sR, 0, None)
    pred_C = np.clip(pred_C * sC, 0, None)
    label = f'w{int(w*100):02d}'
    out = pd.DataFrame({'Date': dates, 'Revenue': pred_R, 'COGS': pred_C})
    path = DIAG_SUB / f'submission_v46_v41b_v45_{label}.csv'
    out.to_csv(path, index=False)
    print(f'  {path}  w_v41={w}')

print('Done.')
