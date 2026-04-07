# Critical Limitation: Models Perform Worse Than Naive Baseline

## Summary

The #1 critical limitation of this repository is that **the ML models produce predictions with negative R², meaning a simple position-average baseline outperforms the entire pipeline**. The system has extensive infrastructure — ensemble models, feature engineering, backtesting, monitoring, CI/CD — but the core predictions it produces are less accurate than predicting the mean for every player.

## Evidence

### 2025 Season Backtest (`data/backtest_results/backtest_2025_20260215.json`)

| Metric | Overall | QB | WR | TE | RB |
|--------|---------|----|----|----|----|
| **R²** | **-0.685** | 0.135 | **-1.357** | **-1.191** | 0.009 |
| Correlation | 0.247 | 0.418 | 0.266 | 0.179 | 0.377 |
| MAPE | 146.7% | 80.7% | 182.2% | 164.8% | 75.2% |
| Spearman ρ | 0.188 | -0.025 | 0.248 | 0.102 | -0.182 |
| Avg Actual | 8.31 | 11.68 | 7.29 | 6.64 | 9.92 |
| Avg Predicted | 11.95 | 12.68 | 12.90 | 11.95 | — |

A negative R² means predicting the same constant value for every player would be **more accurate** than the model's output.

### Year-over-Year Degradation

The 2024 backtest showed R² = 0.233 and correlation = 0.533 — mediocre but positive. By 2025, these collapsed to R² = -0.685 and correlation = 0.247. The model does not generalize across seasons.

## Root Causes

### 1. Systematic Over-Prediction Bias
The model predicts 11.95 fantasy points on average when reality is 8.31 — a **44% upward bias**. For WR specifically, it predicts 12.9 vs actual 7.29 (77% over-prediction). This is not noise; it is a calibration failure where the model has learned inflated baselines from training data that don't transfer to unseen seasons.

### 2. Variance Compression
For QB, the model compresses predicted standard deviation from 7.45 (actual) to 4.21 (predicted). It cannot differentiate elite performers from replacement-level players, producing predictions clustered around an inflated mean.

### 3. Ranking Failure
Spearman rank correlation is 0.188 overall. For QB (-0.025) and RB (-0.182), rankings are zero or inversely correlated with actual outcomes — the model's top-ranked players perform at or below its bottom-ranked players in practice.

### 4. Utilization Score as Indirect Target
The model predicts a "Utilization Score" (an engineered composite metric) rather than fantasy points directly, then converts utilization to fantasy points via a secondary model (`UtilizationToFPConverter`). This two-stage indirection compounds prediction error. The utilization score itself may correlate with opportunity, but the conversion to actual fantasy output introduces a second layer of model error.

## Why This Is the #1 Limitation

The repository contains ~50 test files, an ensemble of XGBoost + LightGBM + Ridge, optional LSTM + ARIMA horizon models, dimensionality reduction (PCA, RFE), a lineup optimizer, A/B testing framework, explainability module, staged deployment pipeline, and comprehensive monitoring infrastructure.

All of this sophistication is built on a prediction foundation that **does not outperform the trivial baseline**. Every downstream feature — lineup optimization, trade evaluation, start/sit recommendations — inherits and amplifies this core inaccuracy.

No other limitation (missing injury data, no weather integration, limited data sources, etc.) matters until the core prediction accuracy is addressed. Fixing the foundation must come before extending the superstructure.

## Recommended Investigation Areas

1. **Audit the training/test split for temporal leakage** — the 2024 results (R² = 0.233) may benefit from data leakage that doesn't reproduce on truly unseen data (2025).
2. **Evaluate the utilization-to-fantasy-points conversion** — the two-stage prediction pipeline may compound errors. Compare direct fantasy point prediction against the current utilization-based approach.
3. **Benchmark against simple baselines** — rolling averages, last-N-weeks mean, and Vegas lines should be formal baselines that the model must beat before adding complexity.
4. **Investigate the over-prediction bias** — the consistent +44% bias suggests a systematic issue in feature engineering or target variable construction, not random model error.
5. **Reduce model complexity** — the current ensemble + dimensionality reduction + horizon models architecture may be overfitting to training data. A simpler model that generalizes may outperform the current approach.
