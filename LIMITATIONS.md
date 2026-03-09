# Fantasy Predictor - Limitations & Improvement Opportunities

This document identifies remaining limitations for different prediction horizons (1 week, 5 weeks, 18 weeks) from both **Football SME** and **Data Scientist** perspectives.

---

## Recently Resolved (Previously Listed as Limitations)

The following items were previously documented as limitations but have since been fully implemented:

- **Data Enrichment**: Injury status integration (`InjuryDataLoader`), weather data (`WeatherDataLoader`), Vegas lines/implied totals (`VegasLinesLoader`), opponent defense rankings (`DefenseRankingsLoader`), snap count integration, air yards/aDOT features
- **Feature Engineering**: Bye week handling (`post_bye`), primetime game adjustment (`is_primetime`), regression to mean (`fp_regression_to_mean_z`), ADP integration (`ADP_POSITION_TIERS`), age/decline curves (`AGE_CURVES`), games played projection (`GAMES_PLAYED_BY_AGE`)
- **Multi-Week & Season-Long**: Schedule strength analysis (`ScheduleStrengthAnalyzer`), multi-week aggregation model (`MultiWeekAggregator`), injury probability modeling (`predict_injury_probability`), rookie projections (`AdvancedRookieProjector`)
- **Infrastructure**: Auto-refresh data updates (`NFLDataRefresher`), horizon-specific models (LSTM+ARIMA for 4w, deep residual for 18w), floor/ceiling uncertainty quantification

---

## 1-Week Prediction Horizon (Weekly Start/Sit Decisions)

### Current Capabilities
- Rolling 3/4/5/8/12-week averages for recent form
- Utilization score (target share, rush share, snap share, high-value touches)
- QB-specific features (passer rating, completion %, YPA, rushing involvement)
- Uncertainty quantification (floor/ceiling confidence intervals)
- Injury status integration from external data sources
- Weather data for outdoor game adjustments
- Vegas lines and implied totals for game script expectations
- Opponent defense-vs-position rankings for matchup difficulty
- Bye week indicator and post-bye performance adjustment
- Primetime game adjustment feature

### Remaining Limitations

#### Football SME Perspective

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **No news/sentiment integration** | Cannot capture breaking news (surprise inactives, role changes) | Add NLP pipeline on news feeds for real-time impact |
| **Red zone data sometimes estimated** | When not available from source, red zone opportunities are estimated from TDs | Integrate a dedicated red zone opportunity data source |

#### Data Scientist Perspective

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **Confidence intervals not tier-specific** | Same uncertainty model for elite starters and backups | Add player-tier-specific variance calibration |
| **K/DST uses rolling averages, not ML** | Kicker and defense models are simpler than offensive position models | Train ML models (XGBoost/LightGBM) for K/DST |

---

## 5-Week Prediction Horizon (Trade Deadline / Mid-Season)

### Current Capabilities
- Season-to-date averages with trend features (improving/declining)
- Consistency scores and volatility metrics
- Schedule strength analysis for upcoming matchups
- Multi-week aggregation model (not simple multiplication of weekly projections)
- Injury probability modeling over multi-week windows
- Regression to mean factor for longer horizons

### Remaining Limitations

#### Football SME Perspective

| Limitation | Impact | Status |
|------------|--------|--------|
| **No coaching/scheme change detection** | Mid-season OC firings affect player usage significantly | Partial — `team_change_features` exist but coaching changes are not detected |
| **No trade deadline impact features** | Players traded mid-season need usage/role resets | Partial — calendar-aware but no trade-specific feature engineering |
| **No playoff implication features** | Teams may rest starters or change game plans late season | Partial — calendar logic exists but no playoff-impact features |

#### Data Scientist Perspective

| Limitation | Impact | Status |
|------------|--------|--------|
| **TD regression implementation sparse** | TDs are high-variance; regression is referenced but thinly implemented | Partial |
| **Ensemble weighting not adaptive by horizon** | Horizon models are loaded but ensemble weights are not dynamically adjusted | Partial — fixed weights rather than learned per-horizon blending |

---

## 18-Week Prediction Horizon (Season-Long / Draft)

### Current Capabilities
- Historical season totals with Value Over Replacement (VOR) rankings
- ADP integration for draft value analysis
- Strength of schedule for full-season matchup difficulty
- Rookie projections using draft capital and comparable player analysis
- Age/decline curve modeling by position
- Games played projection based on age and position
- Deep residual feedforward network for season-long predictions

### Remaining Limitations

#### Football SME Perspective

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **No offseason changes tracking** | Free agency signings, trades, and depth chart changes not tracked | Build offseason transaction tracking system |
| **No suspension risk modeling** | Suspension status values exist in the validator but are not predicted | Add suspension probability model based on player history |

#### Data Scientist Perspective

| Limitation | Impact | Recommended Fix |
|------------|--------|-----------------|
| **Opportunity share not explicitly projected** | Snap/target usage is tracked historically but not projected forward | Add a usage change projection model for season-long estimates |
| **No player-specific uncertainty bounds** | Same uncertainty model applied across all player tiers | Add tier-based or player-specific variance adjustments |

---

## Cross-Horizon Limitations

### Data Limitations

| Issue | Current State | Needed |
|-------|---------------|--------|
| **2025 season data** | Pending in nflverse | Will auto-load when available |
| **Red zone data** | Available when in source data, sometimes estimated from TDs | Dedicated red zone opportunity data source |
| **Real Vegas lines API** | Using synthetic/estimated lines (TODO at `src/models/advanced_models.py:589`) | Integrate The Odds API or ESPN for live lines |

### Model Limitations

| Issue | Current State | Needed |
|-------|---------------|--------|
| **K/DST models use simple rolling averages** | Not ML-based like offensive positions | Train ML models for K/DST |
| **Ensemble weighting not adaptive** | Horizon models loaded with fixed weights | Adaptive ensemble weights by prediction length |
| **No player-specific uncertainty** | Same variance model for all player tiers | Tier-based variance calibration |

### Code Quality

| Issue | Current State | Recommended Fix |
|-------|---------------|-----------------|
| **requirements.txt version pinning inconsistent** | Mix of pinned (`==`) and unpinned (`>=`) versions | Standardize to pinned versions with a separate constraints file |
| **No end-to-end integration tests with live data** | Unit tests and mocked integration tests exist | Add full pipeline integration test suite |

---

## Priority Improvements by Use Case

### For Weekly Start/Sit (1 Week)
1. **HIGH**: News/sentiment integration (NLP pipeline)
2. **MEDIUM**: Tier-specific confidence intervals
3. **LOW**: Real Vegas lines API (currently synthetic)

### For Trade Analysis (5 Weeks)
1. **HIGH**: Coaching/scheme change detection
2. **MEDIUM**: Trade deadline feature engineering
3. **MEDIUM**: Adaptive ensemble weighting by horizon

### For Draft Preparation (18 Weeks)
1. **HIGH**: Offseason changes tracking
2. **MEDIUM**: Suspension risk modeling
3. **MEDIUM**: Opportunity share projection model

---

## Implementation Roadmap

### Phase 1: Remaining Data Gaps
- [ ] Integrate real Vegas lines API (replace synthetic data)
- [ ] Add dedicated red zone opportunity data source
- [ ] Build offseason transaction tracking system

### Phase 2: Model Refinements
- [ ] Add player-tier-specific uncertainty calibration
- [ ] Train ML models for K/DST positions
- [ ] Implement adaptive ensemble weighting by horizon
- [ ] Strengthen TD regression implementation

### Phase 3: Advanced Features
- [ ] News/sentiment NLP pipeline
- [ ] Coaching/scheme change detection
- [ ] Suspension risk prediction
- [ ] Opportunity share projection model
- [ ] Trade deadline feature engineering
- [ ] Playoff implication features

### Phase 4: Code Quality
- [ ] Standardize requirements.txt version pinning
- [ ] Add end-to-end integration test suite

---

*Last updated: 2026-03-09*
