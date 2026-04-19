# Backtesting

Rigorous backtesting ensures production models excel on **truly unseen** test data. This document describes how the NFL Predictor backtest is designed to avoid leakage and match the production pipeline.

## Holdout definition

- **Test set**: One holdout season (by default the **latest** available season). This season is used only for final evaluation; it is never used for training, hyperparameter tuning, or fitting any preprocessing.
- **Train set**: All seasons strictly before the test season. Cross-validation and tuning use only train (and an internal validation split within train).

The split is enforced by `DataManager.get_train_test_seasons()` and asserted in `load_training_data()` and in `run_backtest()` so that `test_season not in train_seasons`.

## No leakage

1. **Preprocessing**: Utilization weights are fit on **train only** (and persisted to `data/models/utilization_weights.json`). When backtesting or serving, test/inference data is prepared using these train-derived weights only (no fitting on test).
2. **Scaling and feature selection**: Scalers and feature selectors are fit on training data during model training. At backtest and serve time, the same persisted artifacts are applied (transform only).
3. **Hyperparameters and model choice**: All tuning and model/ensemble decisions use only training (and internal validation) data. The test season is touched only for the final backtest report.

## Production pipeline

Backtesting uses the **same** pipeline as production:

- **Data**: Same `load_training_data()` (positions, min_games, season split) as in training.
- **Features**: Same utilization weights (loaded from disk), same feature engineering (add_engineered_features). No refitting on test.
- **Model**: Persisted production ensemble (`EnsemblePredictor` loading from `data/models/`). Predictions are generated with `predict(player_data, n_weeks=1)` (or the horizon being backtested).

So backtest performance is a direct estimate of how the model will perform in production on unseen data.

## How to run

- **Standalone backtest** (after models are trained):  
  `python -m src.evaluation.backtester --season YYYY`  
  (omit `--season` to use the latest available test season.)  
  Writes `data/advanced_model_results.json` with `backtest_results` for the app.

- **Multi-season** (stability across years):  
  `python -m src.evaluation.backtester --multi-season 3`  
  Runs backtest on the last 3 seasons and reports mean ± std of RMSE, MAE, R².

- **With training**: Running `python -m src.models.train` trains models and runs a full backtest after training, writing `data/advanced_model_results.json` so the app displays production backtest metrics. This is the single source of truth for production model quality.

## Output and reproducibility

Backtest results include:

- **Metrics**: Overall and per-position RMSE, MAE, R², directional accuracy, within-X-points rates, ranking accuracy (top 5/10/20 hit rates).
- **Baseline comparison**: Model vs. a simple rolling 4-week average baseline (RMSE/MAE/R² and % improvement).
- **Config**: `train_seasons`, `test_season`, `backtest_date`, and (when saved) `model_source` so runs are auditable and reproducible.

Results are saved under `data/backtest_results/` and, in app-compatible form, to `data/advanced_model_results.json` (per-position `backtest_results` for the UI).

## Decision quality (cash H2H win rate / ROI)

Projection accuracy (R² / MAE) is a proxy — it measures whether point estimates track actuals, not whether the lineups they imply would win a contest. The walk-forward runner (`scripts/run_ts_backtest.py`) therefore also reports **cash head-to-head decision quality** alongside the regression metrics. This was the "one thing to do first" from the 2026-03-31 council.

For each historical week in the backtest, three synthetic opponents are constructed from the same prediction/actual frame:

- **Oracle** (hardest): top-N per position by *actual* fantasy points — a perfect hindsight drafter.
- **Hindsight** (realistic): top-N per position by the *prior* week's actual points — a competent drafter chasing recent form.
- **Replacement** (easiest): median-ranked players per position — a casual drafter.

The default roster (QB:1, RB:2, WR:2, TE:1) is built greedily from the model's predictions, then compared against each opponent for that week. A one-sided binomial test (H0: 50%) is reported per tier. **ROI** is computed as `payout_multiplier × win_rate − 1`, using `config.DECISION_QUALITY["payout_multiplier"]` (default 1.8, DraftKings/FanDuel cash H2H after ~20% rake; break-even ≈ 55.6%).

Outputs:

- The `decision_quality` block in `data/backtest_results/ts_backtest_*.json` holds per-tier stats plus `weekly_results` (one row per played week, with running cumulative win rate for each tier).
- A companion `ts_backtest_*_lineup_weekly.csv` materializes the same weekly series for plotting.
- The CLI prints a three-row table plus a per-week `W/✗` string for the hindsight tier.

Flags: `--payout-multiplier FLOAT` overrides the ROI assumption for the run; `--no-decision-quality` skips the block entirely and reverts to the legacy MAE/RMSE/R² output.
