# Data Lineage and Provenance

Per Agent Directive V7 Section 5 — documents the data transformation
pipeline from raw sources to model-ready features.

---

## Data Sources

| Source | Module | Frequency | Availability Timing |
|--------|--------|-----------|-------------------|
| nflverse weekly stats | `src/data/nfl_data_loader.py` | Weekly | Monday night / Tuesday |
| Play-by-play data | `src/data/pbp_stats_aggregator.py` | Weekly | Monday morning |
| Snap counts | `src/data/nfl_data_loader.py` | Weekly | Tuesday |
| Schedule/matchups | `src/data/nfl_data_loader.py` | Pre-season | Available pre-season |
| ESPN Fantasy API | `src/integrations/espn_fantasy.py` | Real-time | Live during games |

## Transformation Pipeline

```
Raw Data Sources
    │
    ▼
┌─────────────────────────────────┐
│ 1. Data Loading & Validation    │  src/data/nfl_data_loader.py
│    - Schema validation          │  src/data/schema_validator.py
│    - Raw snapshot to parquet    │  data/snapshots/
│    - Freshness SLA check       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 2. Feature Engineering          │  src/features/feature_engineering.py
│    - Rolling averages (shift 1) │  Temporal: lag, roll, EWM
│    - Matchup adjustments        │  Contextual: opponent, venue
│    - Utilization scoring        │  src/features/utilization_score.py
│    - Multi-week features        │  src/features/multiweek_features.py
│    - Dimensionality reduction   │  src/features/dimensionality_reduction.py
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 3. Model Training               │  src/models/train.py
│    - Position-specific models   │  src/models/position_models.py
│    - Horizon models (1w/4w/18w) │  src/models/horizon_models.py
│    - Ensemble construction      │  src/models/ensemble.py
│    - Calibration (conformal)    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│ 4. Prediction & Decision        │  src/predict.py
│    - Point predictions          │  src/evaluation/decision_optimizer.py
│    - Uncertainty estimates      │  src/optimization/lineup_optimizer.py
│    - VOR rankings               │
│    - Start/sit recommendations  │
└─────────────────────────────────┘
```

## Data Versioning

- **Dataset hashing**: `ExperimentTracker.log_dataset_hash()` computes
  SHA-256 of each training DataFrame for reproducibility
- **Raw snapshots**: Saved to `data/snapshots/` with timestamps
- **Feature version**: Tracked in `config/settings.py:FEATURE_VERSION`

## Point-in-Time Validity

All rolling/lag features use `shift(1)` to ensure only data from prior
weeks is used. The `src/utils/leakage.py` module enforces:

- No future data in training (temporal split enforcement)
- No target encoding without proper fold isolation
- Identifier columns (`id`, `player_id`) excluded from features
- `sos_next_*` features use lagged opponent stats only

## Survivorship Bias Considerations

- Players who are injured/cut may disappear from later weeks
- Rookie projections lack historical data (handled by rookie features)
- Mid-season trades change team context (handled by `auto_refresh.py`)
