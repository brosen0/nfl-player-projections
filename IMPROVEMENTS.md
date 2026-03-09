# NFL Fantasy Predictor - Improvements & Methodology

## Utilization Score Methodology

Our implementation aligns with the **FantasyLife Utilization Score** methodology:

### Position-Specific Weights

| Position | Snap Share | Target Share | Rush Share | Air Yards | Red Zone | Other |
|----------|------------|--------------|------------|-----------|----------|-------|
| **RB** | 30% | 20% | 25% | - | 15% | 10% goalline |
| **WR** | 25% | 30% | - | 25% | 15% | 5% route participation |
| **TE** | 25% | 35% | - | 20% | 20% | - |
| **QB** | - | - | 20% rush | - | 25% | 35% dropback, 20% pressure |

### Additional Metrics Implemented

- **WOPR** (Weighted Opportunity Rating): `1.5 × Target Share + 0.7 × Air Yards Share`
- **Expected Fantasy Points**: Based on historical conversion rates
- **Weekly Volatility**: Standard deviation of recent performance
- **Consistency Score**: Inverse of coefficient of variation

---

## Model Performance (Backtested)

| Position | R² Score | RMSE | MAE | Best Model |
|----------|----------|------|-----|------------|
| **QB** | 78.5% | 4.02 | 3.14 | LightGBM |
| **RB** | 89.0% | 2.60 | 1.73 | XGBoost |
| **WR** | 91.6% | 2.24 | 1.50 | XGBoost |
| **TE** | 87.3% | 2.12 | 1.42 | GBM |

### Validation Methodology

1. **Time-Series Cross-Validation**: Train on past seasons, test on future
2. **Proper Scaling**: Scaler fit on train only, applied to test
3. **No Data Leakage**: Only historical features used for prediction
4. **Automatic Window Optimization**: 1-10 years tested, optimal selected

---

## Improvements Made

### Data Science Improvements

| Issue | Before | After |
|-------|--------|-------|
| Data Leakage | Used current-week stats | Only historical/lagged features |
| Scaling | Inconsistent | Fit on train, transform test |
| Cross-Validation | Random splits | Walk-forward time-series CV |
| Training Window | Fixed | Auto-optimized (1-10 years) |
| Uncertainty | Point predictions only | Floor/ceiling, confidence intervals |
| Data Source | nfl-data-py (reliable) | nfl-data-py (reliable) |

### NFL Analytics Improvements

| Feature | Status |
|---------|--------|
| Utilization Score | ✅ Implemented (FantasyLife methodology) |
| WOPR | ✅ Implemented |
| QB-Specific Features | ✅ Passer rating, completion %, YPA, mobility |
| TD Regression | ✅ Regress toward expected |
| Volatility Metrics | ✅ Weekly std, consistency score |
| Position Weights | ✅ Different by position |

### Web App Improvements

- ✅ Modern, clean design with better UX
- ✅ Trust metrics and backtesting stats on home page
- ✅ Weekly predictions with floor/ceiling
- ✅ Draft rankings with VOR analysis
- ✅ Start/Sit analyzer with head-to-head comparison
- ✅ Model performance page with detailed metrics
- ✅ Player lookup with historical charts

---

## Remaining Limitations & Future Improvements

### Data Limitations

1. **2025 Season Data**: Not yet available in nfl-data-py (will auto-load when published)

### Model Limitations

1. **K/DST Models Use Rolling Averages**: Kicker and defense models are not ML-based like offensive positions
2. **No News/Sentiment Integration**: Cannot capture breaking news impact on player projections
3. **No Player-Tier-Specific Uncertainty**: Same variance model applied to elite starters and backups
4. **No Coaching/Scheme Change Detection**: Team change features exist but mid-season coaching changes are not detected

### Recently Resolved

1. ~~**Real Vegas Lines API**~~: The Odds API client integrated (`VegasLinesLoader.fetch_live_lines()`)
2. ~~**Red Zone Data Estimated from TDs**~~: Now PBP-derived (red zone carries, pass attempts, team totals)
3. ~~**No Offseason Tracking**~~: Roster/depth chart sync wired into auto-refresh pipeline

### Recommended Future Improvements

1. **News/Sentiment Pipeline**: Add NLP on news feeds for breaking news impact
2. **Tier-Specific Uncertainty**: Calibrate confidence intervals by player tier
3. **Coaching Change Detection**: Detect mid-season OC/HC changes and adjust projections
4. **K/DST ML Models**: Train XGBoost/LightGBM models for kickers and defenses
5. **End-to-End Integration Tests**: Add full pipeline tests with live data

---

## How to Use

### Load Data
```bash
python3 src/data/nfl_data_loader.py --seasons 2020-2024
```

### Auto-Refresh Data
```bash
# Check for and load any new data
python3 src/data/auto_refresh.py

# Check status only
python3 src/data/auto_refresh.py --status

# Force reload all data
python3 src/data/auto_refresh.py --force
```

### Train Models
```bash
python3 src/models/train_advanced.py
```

### Run Web App
```bash
streamlit run app.py
```

### Validate Methodology
```bash
python3 src/models/validate_methodology.py
```

---

## Data Availability & Refresh

### Current Status (March 2026)

| Season | Weekly Data | Schedule | Status |
|--------|-------------|----------|--------|
| 2020-2024 | ✅ Available | ✅ Available | Loaded |
| 2025 | ⏳ Pending | ✅ Available | Schedule loaded, weekly data pending in nflverse |
| 2026 | ❌ Not yet | ❌ Not yet | Will be available ~May 2026 |

### How Data Updates Work

1. **nfl-data-py** sources data from **nflverse**, which aggregates official NFL statistics
2. Weekly player data is typically available within 24-48 hours after games
3. Full season data files are published after the season completes
4. The 2025 season weekly data will be available once nflverse publishes it

### Auto-Refresh System

The app includes an auto-refresh system that:
- Checks for new seasons/weeks available in nfl-data-py
- Automatically loads any missing data
- Shows pending data status in the sidebar
- Can be triggered manually via the "Check for Updates" button

### For the 2026-27 Season

When preparing for the 2026-27 fantasy draft:

1. **Schedule Release (~May 2026)**: Run auto-refresh to load the 2026 schedule
2. **Pre-Draft**: Use 2020-2025 historical data for projections
3. **During Season**: Auto-refresh will load new weekly data as it becomes available
4. **Re-train Models**: After loading new data, re-run training for updated predictions

```bash
# Recommended workflow for new season
python3 src/data/auto_refresh.py          # Load latest data
python3 src/models/train_advanced.py      # Re-train models
streamlit run app.py                       # Launch app
```

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `src/features/utilization.py` | Utilization score, WOPR, expected FP |
| `src/features/qb_features.py` | QB-specific features |
| `src/models/production_model.py` | Production model with uncertainty |
| `src/models/robust_validation.py` | Time-series CV with proper scaling |
| `src/models/train_advanced.py` | Training script with backtesting |
| `src/data/nfl_data_loader.py` | nfl-data-py integration |
| `app.py` | Redesigned web application |

---

*Last updated: 2026-03-09*
