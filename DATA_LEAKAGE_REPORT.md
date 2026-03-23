# Data Leakage Audit Report

**Project:** NFL Player Projections
**Date:** 2026-03-23
**Scope:** Full pipeline review — data loading, feature engineering, model training, and evaluation

---

## Executive Summary

The pipeline has **strong leakage defenses** overall, including a dedicated `leakage.py` module, season-aware temporal splits, and careful train-only fitting for scalers and embeddings. The audit identified **3 issues** — all have been resolved.

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | Utilization percentile bounds normalize a target-adjacent signal | **Medium** | **Fixed** |
| 2 | Target winsorization is asymmetric (train only, not test) | **Low** | **Fixed** |
| 3 | Season-relative normalization uses full-season lookahead stats | **Medium** | **Fixed** |

---

## Pipeline Architecture Overview

```
Data Load (nfl-data-py)
  -> Season-based Train/Test Split (train.py:171-176)
  -> Leakage Column Filter (train.py:163-169)
  -> DVP / External / Season-Long Features (with temporal context)
  -> Utilization Score Calculation + Percentile Bounds (train-only fit)
  -> Horizon Target Creation (shift-based forward windows)
  -> Utilization Weight Optimization (train-only fit)
  -> Feature Engineering (rolling/lag with .shift(1), temporal context)
  -> Bounded Scaling (MinMaxScaler, train-only fit)
  -> Player Embeddings (PCA, train-only fit)
  -> Target Winsorization (train bounds, applied to both train and test)
  -> Model Training (SeasonAwareTimeSeriesSplit CV)
```

---

## Findings

### Finding 1: Utilization Percentile Bounds Normalize a Target-Adjacent Signal

**Location:** `src/models/train.py:1219-1245`
**Severity:** Medium

**Description:**
Utilization percentile bounds are fit on training data (correct), but `utilization_score` is directly used to derive target columns:

```python
# train.py:313 — target is derived from utilization_score
out["target_util_1w"] = out.groupby(group_cols)["utilization_score"].shift(-1)
```

The percentile normalization applied to `utilization_score` at line 1231-1238 shapes the distribution of the signal that becomes the target at line 1248. While the bounds are fit on train data only (no test data leakage), this creates a subtle coupling: the normalization parameters are derived from the same column that will become labels. This means the model's targets are influenced by train-set distributional statistics of the predictor, which can inflate apparent fit on in-distribution data.

**Impact:** Modest. The bounds are fit on training data only, so this is not a classic train-test leakage. The risk is that percentile normalization compresses the utilization distribution in a way that makes the target easier to predict from the (identically normalized) utilization features, overstating in-sample metrics.

**Resolution:** Added `compute_raw_utilization_score()` in `utilization_score.py` that computes a weighted sum of raw `_pct` columns (no percentile normalization). `_create_horizon_targets()` now derives `target_util_*` from `utilization_score_raw` instead of the normalized `utilization_score`. The raw column is also blocked by `leakage.py` to prevent accidental use as a feature.

---

### Finding 2: Asymmetric Target Winsorization

**Location:** `src/models/train.py:1307-1326`
**Severity:** Low

**Description:**
Target winsorization (clipping at 1st/99th percentile) is applied only to training targets. Test targets remain unclipped. This creates a distribution mismatch: the model is trained on a clipped target distribution but evaluated on the full distribution.

**Impact:** Minimal. This is conservative — it prevents extreme values from distorting training but allows the model to be evaluated against actual outcomes. The asymmetry may cause slightly pessimistic test metrics for extreme performances.

**Resolution:** Train-derived winsorization bounds `(lo, hi)` are now stored per (position, target_column) and applied symmetrically to both train and test targets, eliminating the distribution mismatch.

---

### Finding 3: Season-Relative Normalization Within-Season Lookahead

**Location:** `src/features/feature_engineering.py:1712-1733` (`_normalize_season_relative`)
**Severity:** Medium (upgraded from Monitor after deeper analysis)

**Description:**
The `_normalize_season_relative` method used `groupby(["season", "position"]).transform("mean"/"std")` which computes statistics across ALL weeks in a season. When applied via `_apply_with_temporal_context` on the combined train+test dataframe, test rows in week N see future weeks' statistics from the same season — confirmed within-season lookahead bias.

**Impact:** Moderate. For a test row in week 10, the normalization mean/std included data from weeks 11-18 of the same season. This leaks future information into feature normalization, potentially inflating test-set metrics.

**Resolution:** Replaced full-season `groupby.transform("mean"/"std")` with expanding backward-looking statistics: `shift(1).expanding(min_periods=3).mean()/std()`. Each row now only sees strictly prior weeks within the same season/position group. Rows with fewer than 3 prior data points (early season) are left unnormalized, which is safe since the model-layer StandardScaler handles mixed scales.

---

## Verified Correct Implementations

### Train/Test Split (Correct)
- **Location:** `train.py:171-176`
- Strict season-based split with assertion that test season is not in train seasons
- No random splitting that could cause temporal leakage

### Leakage Column Detection (Correct)
- **Location:** `src/utils/leakage.py`
- Comprehensive regex/prefix-based detection of target columns, model outputs, identifiers, and forward-looking features
- Applied at data load time (`train.py:163-169`)
- Safe allowlist for schedule features that contain `_next` but are backward-looking

### Rolling/Lag Features (Correct)
- **Location:** `feature_engineering.py:272-274`
- All rolling features use `.shift(1)` before rolling window computation
- This ensures no current-week data leaks into features
- Example: `x.shift(1).rolling(window=window, min_periods=1).mean()`

### Temporal Context Application (Correct)
- **Location:** `train.py:321-367` (`_apply_with_temporal_context`)
- Train features computed from train data alone
- Test features computed from combined train+test, but only test rows kept
- This correctly allows test rows to use historical context from prior seasons

### Feature Scaling (Correct)
- **Location:** `train.py:591-615` (`_apply_bounded_scaling`)
- MinMaxScaler fit exclusively on training data (`fit_transform` on train, `transform` on test)
- Persisted as joblib artifact for serving consistency

### StandardScaler in Models (Correct)
- **Location:** `position_models.py:207-209`
- Fit on training fold only within cross-validation
- Applied to validation fold via `transform` only

### Feature Median Imputation (Correct)
- **Location:** `position_models.py:187-192`
- Medians computed from training split only
- Stored for prediction-time imputation

### Player Embeddings (Correct)
- **Location:** `train.py:1290-1305`
- PCA embeddings fit on training data only
- Applied to both train and test via `get_embedding()`

### Cross-Validation (Correct)
- **Location:** `position_models.py:214-218`
- `SeasonAwareTimeSeriesSplit` respects temporal ordering
- Optional gap seasons prevent feature leakage across fold boundaries

### Isotonic Calibration (Correct)
- **Location:** `position_models.py:308-321`
- Fit on out-of-fold predictions (no test data leakage)

### Conformal Prediction Intervals (Correct)
- **Location:** `position_models.py:353-385`
- Calibration residuals from OOF or validation set only

### Adversarial Validation (Correct)
- **Location:** `train.py:441-535`
- Distribution shift detection between train/test
- Flags features with high discriminative power (potential leakage signals)

### Schedule Score Sanitization (Correct)
- **Location:** `leakage.py:189-210`
- Final game scores stripped from schedule data used as features

### Utilization Weight Optimization (Correct)
- **Location:** `train.py:1251-1261`
- Weights fit on training data only via Ridge regression with CV
- Applied identically to both train and test

---

## Defense-in-Depth Summary

The pipeline employs multiple layers of leakage prevention:

1. **Column-level filtering** — `leakage.py` blocks targets, model outputs, identifiers, and forward-looking columns
2. **Temporal split** — Season-based train/test with assertion guard
3. **Temporal context** — `_apply_with_temporal_context` isolates train features from test influence
4. **Shift-based features** — `.shift(1)` on all rolling/lag computations
5. **Train-only fitting** — Scalers, embeddings, percentile bounds, utilization weights all fit on train
6. **Season-aware CV** — Time-series cross-validation respecting season boundaries
7. **Adversarial validation** — Automated distribution shift detection
8. **Score sanitization** — Game scores removed from schedule features

---

## Resolutions Applied

All three findings have been fixed:

| Finding | Fix | Files Changed |
|---------|-----|---------------|
| 1 | Added `compute_raw_utilization_score()` for target derivation; targets now use raw `_pct` scores | `utilization_score.py`, `train.py`, `leakage.py` |
| 2 | Train-derived winsorization bounds applied symmetrically to both train and test targets | `train.py` |
| 3 | Replaced full-season normalization with expanding backward-looking `shift(1).expanding(min_periods=3)` | `feature_engineering.py` |

## Remaining Recommendations

| Priority | Action | Effort |
|----------|--------|--------|
| 1 | Add automated leakage regression test: assert no feature correlates >0.95 with any target | Low |
