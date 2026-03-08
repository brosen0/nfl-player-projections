# System Architecture — Agent Directive V7 Role Mapping

This document maps the existing module structure to the Agent Directive V7's
multi-agent architecture roles. While the system is a monolithic Python
application, its module boundaries align with the directive's agent
specializations and can serve as a foundation for future decomposition.

---

## Module-to-Agent Role Mapping

| Directive Agent Role | Module(s) | Responsibilities |
|---------------------|-----------|------------------|
| **Research Orchestrator** | `src/pipeline.py`, `run_app.py` | End-to-end orchestration: data collection, feature engineering, training, prediction, evaluation |
| **Data Agent** | `src/data/` | Data loading (`nfl_data_loader.py`), PBP aggregation (`pbp_stats_aggregator.py`), K/DST (`kicker_dst_aggregator.py`), mid-season refresh (`auto_refresh.py`), schema validation (`schema_validator.py`) |
| **Feature Agent** | `src/features/` | Feature engineering (`feature_engineering.py`), utilization scoring (`utilization_score.py`), dimensionality reduction (`dimensionality_reduction.py`), weight optimization (`utilization_weight_optimizer.py`), QB features, rookie/injury features, season-long and multi-week features |
| **Model Agent** | `src/models/` | Position-specific training (`train.py`), model families (`position_models.py`), horizon models (`horizon_models.py`), ensemble construction (`ensemble.py`), Bayesian models, advanced techniques |
| **Ensemble Agent** | `src/models/ensemble.py` | Weighted ensemble blending, uncertainty quantification, horizon-specific model combination |
| **Decision Agent** | *(not yet implemented)* | Would handle lineup construction, draft ranking, start/sit recommendations, abstention logic |
| **Audit Agent** | `src/evaluation/`, `src/utils/leakage.py` | Metrics (`metrics.py`), backtesting (`ts_backtester.py`, `backtester.py`), monitoring (`monitoring.py`), A/B testing (`ab_testing.py`), explainability (`explainability.py`), leakage guards (`leakage.py`) |
| **Deployment Agent** | `api/main.py`, `Dockerfile`, `docker-compose.yml` | FastAPI serving, containerization, health checks |

---

## Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Agent   │────▶│ Feature Agent │────▶│ Model Agent  │
│  src/data/    │     │ src/features/│     │ src/models/  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Deployment  │◀────│  Orchestrator│◀────│ Ensemble     │
│  api/main.py │     │  pipeline.py │     │ ensemble.py  │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Audit Agent  │
                     │ evaluation/  │
                     └──────────────┘
```

### Pipeline Stages

1. **Data Collection** — `src/data/nfl_data_loader.py` fetches from nfl-data-py (nflverse): weekly stats, snap counts, play-by-play, schedules, rosters
2. **Data Processing** — `src/data/pbp_stats_aggregator.py` derives advanced features (EPA, WPA, success rate, red-zone usage)
3. **Feature Engineering** — `src/features/feature_engineering.py` creates rolling averages, trend indicators, Vegas/game-script features, matchup adjustments
4. **Utilization Scoring** — `src/features/utilization_score.py` computes composite utilization (0-100) from snap%, target%, touch%, red-zone%
5. **Model Training** — `src/models/train.py` trains position-specific ensembles across 1-week, 4-week, and 18-week horizons
6. **Ensemble Construction** — `src/models/ensemble.py` blends position models with horizon models using weighted averaging
7. **Prediction** — `src/predict.py` generates schedule-aware predictions with confidence intervals
8. **Evaluation** — `src/evaluation/` computes metrics, runs backtests, monitors drift
9. **Serving** — `api/main.py` exposes predictions via FastAPI endpoints

---

## Module Public APIs

### `src/data/` — Data Agent

| File | Key Functions/Classes |
|------|----------------------|
| `nfl_data_loader.py` | `NFLDataLoader.load_weekly_stats()`, `load_snap_counts()`, `load_pbp_data()` |
| `pbp_stats_aggregator.py` | `PBPStatsAggregator.aggregate()` — derives EPA, WPA, success rate from play-by-play |
| `auto_refresh.py` | `AutoRefresher.refresh()` — incremental mid-season data loading |
| `schema_validator.py` | `SchemaValidator.validate()` — data quality checks |
| `kicker_dst_aggregator.py` | `KickerDSTAggregator.aggregate()` — K/DST rolling averages |

### `src/features/` — Feature Agent

| File | Key Functions/Classes |
|------|----------------------|
| `feature_engineering.py` | `FeatureEngineer.create_features()` — rolling, trend, Vegas, matchup features |
| `utilization_score.py` | `UtilizationScorer.compute()` — composite 0-100 utilization score |
| `dimensionality_reduction.py` | `DimensionalityReducer.reduce()` — RFE, PCA, correlation filtering |
| `utilization_weight_optimizer.py` | Optuna-based optimization of utilization component weights |

### `src/models/` — Model Agent

| File | Key Functions/Classes |
|------|----------------------|
| `train.py` | `ModelTrainer.train()` — orchestrates position-specific training |
| `position_models.py` | `PositionModelTrainer` — Ridge, RandomForest, GradientBoosting, XGBoost, LightGBM |
| `horizon_models.py` | `Hybrid4WeekModel` (LSTM+ARIMA), `Deep18WeekModel` |
| `ensemble.py` | `EnsembleModel.predict()` — weighted blending with uncertainty |

### `src/evaluation/` — Audit Agent

| File | Key Functions/Classes |
|------|----------------------|
| `metrics.py` | `compute_all_metrics()` — RMSE, MAE, R², Spearman, tier accuracy, boom/bust, VOR |
| `ts_backtester.py` | `TimeSeriesBacktester.run()` — leakage-free walk-forward backtesting |
| `monitoring.py` | `ModelMonitor` — prediction drift, feature drift (KS test), RMSE degradation |
| `ab_testing.py` | `ABTestManager` — Wilcoxon significance test, 5% improvement gate, rollback |
| `experiment_tracker.py` | `ExperimentTracker` — JSONL experiment ledger with git hash, config, metrics |

---

## Configuration

All configurable parameters live in `config/settings.py` — the single source of truth:

- **Seasons**: 2006–present (MIN_HISTORICAL_YEAR = 2006)
- **Positions**: QB, RB, WR, TE (ML models); K, DST (rolling averages)
- **Scoring**: PPR, Half-PPR, Standard
- **Horizons**: 1-week, 4-week, 18-week
- **Model parameters**: per-position, per-horizon configuration
- **Retraining**: auto-retrain settings with degradation thresholds

---

## Storage

| Artifact | Location | Format |
|----------|----------|--------|
| Raw data | `data/nfl_data.db` | SQLite |
| Team stats | `data/raw/*.csv` | CSV |
| Trained models | `data/models/*.joblib` | Joblib |
| Model metadata | `data/models/model_metadata.json` | JSON |
| Version history | `data/models/model_version_history.json` | JSON |
| Experiment log | `data/experiments/experiment_log.jsonl` | JSONL |
| Monitoring alerts | `data/monitoring/alerts.jsonl` | JSONL |
| Monitoring metrics | `data/monitoring/metrics.jsonl` | JSONL |
| SHAP outputs | `data/models/explainability/` | Various |

---

## Future Decomposition Path

The existing module boundaries provide a natural decomposition path if the
system is migrated to a multi-agent architecture:

1. Each `src/<module>/` becomes an independent service with a defined API contract
2. `src/pipeline.py` becomes the orchestrator, communicating via message passing
3. `src/evaluation/ab_testing.py` provides the foundation for an Audit Agent veto mechanism
4. Inter-agent contracts would be formalized using the `ExperimentRecord` dataclass pattern
5. A Decision Agent (Section 9) would be added as a new module consuming predictions and producing actionable recommendations
