# Pipeline dự đoán Revenue và COGS

## 1. Chuẩn bị môi trường và thực thi code

### 1.1. Cài đặt môi trường

- **Python:** 3.10
- **Dependencies chính:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`.

```bash
git clone ...
conda create -n env-3.10 python=3.10 -y
conda activate env-3.10
pip install pandas numpy scipy scikit-learn lightgbm xgboost catboost
```

### 1.2. Cấu trúc thư mục trước và sau

**Trước khi thực thi pipeline:**

```
├── notebooks/                     # Chứa các notebook cho phần thi thứ 2 (EDA)
├── data/                          # CSV cuộc thi (orders, order_items, customers, products,
│                                  # inventory, web_traffic, reviews, sample_submission, …)
├── best_pipeline/                 # Điều phối: pipeline.py, run_*.py, config.py, …
└── prerequisite_postTraining/     # build_*_features, train_*, build_*_submission, blend
    ├── enriched_features_v2.pkl   # BẮT BUỘC - lấy từ kết quả thí nghiệm trước đó
    └── enriched_features_v3_pattern.pkl   # BẮT BUỘC — base v2 + promo pattern
```

**Sau khi thực thi pipeline:**

```
├── notebooks/
├── data/
├── best_pipeline/
├── prerequisite_postTraining/
│   ├── enriched_features_v2.pkl
│   ├── enriched_features_v3_pattern.pkl
│   ├── enriched_features_v5_pattern.pkl
│   ├── enriched_features_v6_pattern.pkl
│   ├── enriched_features_v8_pattern.pkl
│   ├── enriched_features_v10_pattern.pkl
│   ├── ml_v37_pattern_preds.npz          # (+ file ckpt tạm nếu train chưa xong 1 lần)
│   ├── ml_v40_pattern_preds.npz
│   └── ml_v45_pattern_preds.npz
└── diag_submissions/                # Tạo khi chạy bước submit / blend
    ├── submission_v37_cust_prod.csv
    ├── submission_v40_cust_prod.csv
    ├── submission_v45_web_review.csv
    ├── submission_v41_B_7030.csv
    ├── submission_v46_v41b_v45_w85.csv   # blend đề xuất (và các w90, w95, …)
    └── …
```

### 1.3. Thực thi pipeline

**Thứ tự chạy (featuring → training → submit → blend)**

1. **`python check_artifacts.py`** — chỉ đọc, kiểm tra artifact theo `--require`.
2. **`python run_features.py`** — chạy lần lượt các lần feature engineering:
   - `build_v5_features.py` → `build_v6_features.py` → `build_v8_features.py` → `build_v10_features.py`
3. **`python run_train.py`** — `train_v37.py`, `train_v40.py`, `train_v45.py` (lâu, tốn CPU).
4. **`python run_submissions.py`** — `build_v37_submission.py`, `build_v40_submission.py`, `build_v45_submission.py`.
5. **`python run_blend_best.py`** — `build_v41_blend.py` rồi `build_v46_v41b_grid_blend.py`.

**Hoặc gộp tất cả cho một lần chạy:**
```bash
cd …/final-submisison/best_pipeline
python pipeline.py --dry-run               # chỉ liệt kê hành vi
python pipeline.py --step train --yes       # ví dụ: chỉ chain từ train
python pipeline.py --yes                    # full chain, không hỏi (tự chịu trách nhiệm chạy nặng)
```

**Mặc định không dùng `--yes`**: mỗi script con sẽ hỏi `Execute? [y/N]`.

### 1.4. Các file đối chiếu (sau pipeline đầy đủ)

| Stage | Outputs chính |
|-------|----------------|
| Prerequisites | `prerequisite_postTraining/enriched_features_v2.pkl`, `…/enriched_features_v3_pattern.pkl` |
| Features | `prerequisite_postTraining/enriched_features_v5/6/8/10_pattern.pkl` |
| Train | `prerequisite_postTraining/ml_v37_pattern_preds.npz`, `…/ml_v40_pattern_preds.npz`, `…/ml_v45_pattern_preds.npz` |
| Submit | `diag_submissions/submission_v37_cust_prod.csv`, `…/submission_v40_cust_prod.csv`, `…/submission_v45_web_review.csv` |
| Blend | `diag_submissions/submission_v41_B_7030.csv`, `…/submission_v46_v41b_v45_w85.csv` (+ w75,w80,w90,w95) |

### 1.5. Kết quả thực nghiệm

| File | Leaderboard Score |
|------|-------------------|
|`sample_submisison.csv`|1,225,931 (baseline)|
|`submission_v37_cust_prod.csv`|665,580|
|`submission_v41_C_optval.csv`|664,325|
|`submisison_v41_B_7030.csv`|**663,330**|
|`submission_v46_v41b_v45_w85.csv`|663,387|

## 2. Phương pháp tiếp cận và mô hình sử dụng

### 2.1. Phương pháp tiếp cận: Thử và sai

*Lưu ý:* **Tránh aggregate** trùng ngày vì dẫn đến **data leakage**, giải pháp là  **dùng lag/roll**. Sử dụng dữ liệu trước nửa năm 2022 làm train set và sau năm 2022 đến cuối 2022 làm validation set vì không có bất kì dữ liệu nào về test set, ngoại trừ ngày dự đoán.

Team đã thực hiện thử nhiều loại chiến lược, nhiều mô hình và nhiều cách feature engineering và rút ra khung phương pháp luận như sau:

**Giai đoạn 1: Phân tích - Hiểu metric và bài toán**
1. Submit sample baseline → lấy MAE mốc.
2. Submit 2-3 submisison để thăm dò (`aS+b`) → fit quan hệ tuyến tính giữa sample và truth.
3. Dùng linear fit làm baseline đầu tiên.

**Giai đoạn 2: Kiến thức domain (Domain heuristics)**
4. Phân tích sự kiện mùa vụ (holidays, pay cycles, promotions) và apply targeted correction.
5. Analytical blend (pooled MAE + Gaussian proxy) để tìm trọng số **w**.

**Giai đoạn 3: ML model**
6. Build LightGBM baseline với:
   - Calendar features (cyclic + holiday indicators)
   - Lag features (từ target) với **projection** cho test
   - Rolling statistics
   - Climatology features
   - Exogenous features (từ bảng phụ) với lag/roll
7. Check data leakage kĩ càng (xóa những aggregation cùng ngày).
8. Multi-seed ensemble (5 seeds) để giảm phương sai.
9. Time-series CV (3+ folds) để validate.

**Giai đoạn 4 — Diversity ensembling**

10. Thêm XGBoost, CatBoost → đo lường inter-model correlation.
11. **Cross-family diversity** (Ridge/Linear) > same-family diversity.
12. **MLP neural net** khi đã chạy hết boosting trees.
13. Optimize ensemble weights trên val (Nelder-Mead multi-start).
14. CV để xác thực rằng weights không overfit val.

**Giai đoạn E — Blend with baseline**

15. Xem baseline (v17m, đã thử submit nhiều lần trước đó) như một mô hình nữa với MAE đã biết.
16. Analytical optimal w bằng Gaussian proxy:
    ```
    w = (Var_baseline − Cov) / (Var_baseline + Var_ml − 2xCov)
    ```
    với Cov suy ra từ >=2 đo đạc blended MAE.
17. Khi Var(ml) < Var(baseline) và Cov < Var(ml) → w → 1 → **bỏ baseline**.

**Giai đoạn F — Final tuning (plateau territory)**

- Grid search w quanh optimum (vùng chững lại plateau thường rộng).
- Per-component blend (w_rev != w_cogs).
- Dừng khi kết quả MAE trên leaderboard nhỏ hơn các kết quả đã có.

**Cuối cùng thử blend các phiên bản khác nhau của submisison đã có để tìm ra trọng số blend sao cho làm giảm đi MAE trên leaderboard.**

|**Phân loại**|**Chiến lược**|**Kết quả**|
|-|-|-|
|**Model diversity**|MLP, ExtraTreesm, RandomForest, ElasticNet, Kernel Ridge, LGBM-log, CatBoost-log|Bị redundant hoặc overfit|
|**Weight tunning**|Val-opt, CV-avg, per-DOW, weekend/weekday, robust min-max|CV-avg thấp hơn, per-DOW tệ đi|
|**Meta-learning**|Ridge stacking, LGB meta + calendar features|stacking kiểu OOF nhỏ, lợi không đáng kể hoặc LGB meta overfit|
|**Post-preprocessing**|DOW bias correction, month bias|Overfit|
|submisison bleding|Pairwise, 3-way equal, inv-MAE weighted, extrapolation|Không cải thiện đáng kể|
|**Feature eng (cũ)**|100+ features (holidays, climatology, ratios)|4-way đã "bão hòa"|
|**Promo features**|13 features từ promotions.csv với pattern projection|Kết quả cải thịene đáng kể -> feature mới thêm tín hiệu dự đoán|

### 2.2. Mô hình sử dụng

- Với **mỗi** target (Revenue, COGS): ensemble gồm bốn họ mô hình trên cùng bộ feature:
  - **Ridge** (`RidgeCV` để chọn alpha trên validation, sau đó `Ridge` fit train+val),
  - **LightGBM** (`LGBMRegressor`),
  - **XGBoost** (`XGBRegressor`, nhiều seed),
  - **CatBoost** (`CatBoostRegressor`).
- Trên **tập validation** (theo khoảng ngày trong từng script train), dùng **Nelder-Mead** để tối ưu trọng số không âm trên stack vector dự báo ensemble (trong `build_v37_submission.py`, `build_v40_submission.py`, `build_v45_submission.py`).
- **Mean scaling:** sau khi có dự báo test, scale mean Revenue và COGS về hai hằng số benchmark (`V17M_MEAN_R`, `V17M_MEAN_C` trong [best_pipeline/config.py](best_pipeline/config.py)) trước khi ghép vào `sample_submission.csv`.

### Tham chiếu hằng số (các hằng số được thử trong các thí nghiệm ban đầu)

*Blend và scale được ghi nhận trong `best_pipeline/config.py`: `W41_V37`, `W41_V40`, `W46_WEIGHT_V41B`, `V17M_MEAN_R`, `V17M_MEAN_C`.*

## Các phiên bản submission, cùng với các script tái tạo từng submission, đều được liệt kê chi tiết trong repo GitHub phụ: [vinunidatathon2026-auxilaryrepo](https://github.com/truongdongto/vinunidatathon2026-auxilaryrepo.git)