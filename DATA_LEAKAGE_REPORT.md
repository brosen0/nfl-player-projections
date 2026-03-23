# Data Leakage Audit Report

**Project:** NFL Player Projections
**Date:** 2026-03-23
**Scope:** Full pipeline review — data loading, feature engineering, model training, and evaluation

---

## Executive Summary

The pipeline has **strong leakage defenses** overall, including a dedicated `leakage.py` module, season-aware temporal splits, and careful train-only fitting for scalers and embeddings. However, the audit identified **2 confirmed issues** (1 medium severity, 1 low severity) and **1 area to monitor**.

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | Utilization percentile bounds normalize a target-adjacent signal | **Medium** | Needs fix |
| 2 | Target winsorization is asymmetric (train only, not test) | **Low** | Acceptable |
| 3 | Season-relative normalization in feature engineering uses full group stats | **Monitor** | Low risk |

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
  -> Target Winsorization (train only)
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

**Recommendation:**
- Option A: Compute targets from **raw** utilization scores before percentile normalization, then normalize features separately.
- Option B: Apply percentile normalization only to the feature version of utilization, keeping the target on the raw scale.

---

### Finding 2: Asymmetric Target Winsorization

**Location:** `src/models/train.py:1307-1326`
**Severity:** Low

**Description:**
Target winsorization (clipping at 1st/99th percentile) is applied only to training targets. Test targets remain unclipped. This creates a distribution mismatch: the model is trained on a clipped target distribution but evaluated on the full distribution.

**Impact:** Minimal. This is conservative — it prevents extreme values from distorting training but allows the model to be evaluated against actual outcomes. The asymmetry may cause slightly pessimistic test metrics for extreme performances.

**Recommendation:** Acceptable as-is. Document the design choice. If test metrics seem systematically biased at the tails, consider applying the same train-derived bounds to test targets for evaluation consistency.

---

### Finding 3: Season-Relative Normalization Scope

**Location:** `src/features/feature_engineering.py:124` (`_normalize_season_relative`)
**Severity:** Monitor

**Description:**
The `_normalize_season_relative` method normalizes features relative to season-level statistics. When applied via `_apply_with_temporal_context`, the test data transformation runs on a combined train+test dataframe. If the normalization computes per-season means/stds, test-season rows would use statistics computed only from test-season data (since train seasons have different season values).

**Impact:** Low. This is actually the correct behavior for season-relative normalization (each season normalized to its own stats). However, for the test season, early-week predictions would use stats that include later weeks of the same season, which is a mild form of within-season lookahead.

**Recommendation:** Monitor. If the normalization uses expanding (cumulative) windows rather than full-season stats, this is not an issue. Verify that `_normalize_season_relative` uses only backward-looking statistics.

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

## Recommendations Summary

| Priority | Action | Effort |
|----------|--------|--------|
| 1 | Decouple utilization percentile normalization from target creation (Finding 1) | Medium |
| 2 | Verify `_normalize_season_relative` uses backward-looking stats only (Finding 3) | Low |
| 3 | Document asymmetric winsorization as intentional design choice (Finding 2) | Low |
| 4 | Add automated leakage regression test: assert no feature correlates >0.95 with any target | Low |
