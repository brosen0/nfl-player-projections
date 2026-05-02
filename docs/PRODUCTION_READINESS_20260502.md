# Production Readiness Report — 2026-05-02

## Summary

Full data pipeline and model training audit completed. The system is now
production-ready with component prediction models for all 4 positions,
clean historical data from 2006-2025, and a functioning prediction
pipeline. This document records what was done, what was found, and what
remains.

---

## Data Fixes Applied

### 1. Vegas Lines Backfill (2006-2017)

**Before:** `schedule` table only had 2020-2025 data with zero
`spread_line`/`total_line` values. Pre-2018 was completely missing.

**After:** 5,431 games across 2006-2025 now have Vegas lines (100%
coverage per season). Source: nflverse `games.csv` via
`scripts/backfill_vegas_lines.py --seasons 2006 2025`.

### 2. Team Code Normalization

**Before:** Newly backfilled schedule rows had historical team codes
(OAK, SD, STL) that didn't match `player_weekly_stats` (LV, LAC, LA).
Also found `team_stats` had `LAR` where `pws` uses `LA`.

**After:** All tables normalized:
- OAK → LV (Raiders, 2018-2019)
- SD → LAC (Chargers, 2006-2016)
- STL → LA (Rams, 2006-2015)
- LAR → LA (108 rows in `team_stats`)

### 3. Snap Count Backfill

**Before:** `snap_count` and `team_snaps` columns in
`player_weekly_stats` were zero for ALL rows across all seasons (never
populated during original ingestion).

**After:** 41,660 of 43,118 rows (96.6%) updated for 2018-2025 via
new script `scripts/backfill_snap_counts_to_pws.py`. Matching strategy:
`players.name` (abbreviated format) joined to `snap_counts.player`
(full name) by first-initial + last-name + team + season + week.

Per-season match rates: 95-98%.

Pre-2018 snap counts remain zero (nflverse data unavailable).

---

## Model Training

### Feature Version

Upgraded from v4 → **v9**. New features in v9:
- `snap_share_pct_roll3_mean` (RB, TE) — enabled by snap count backfill
- `ngs_completion_pct_above_expected_roll3_mean` (QB)
- `ngs_rush_yards_over_expected_per_att_roll3_mean` (RB)
- `ngs_avg_separation_roll3_mean` (WR, TE)
- Draft capital features (decayed)

### Training Configuration

- **Mode:** Component prediction (predict stat lines, assemble FP)
- **Feature mode:** Causal (9-14 curated features per position)
- **Training window:** 2006-2024 (test: 2025)
- **Positions:** QB (13 features), RB (14), WR (13), TE (13)

### Test Metrics (2025 Season Holdout)

| Position | RMSE | MAE | R² | Spearman ρ | ≤7pt% | ≤10pt% |
|----------|------|-----|-----|-----------|-------|--------|
| QB | 7.11 | 5.84 | 0.085 | 0.334 | 65.4% | 83.8% |
| RB | 6.97 | 5.31 | 0.224 | 0.488 | 74.2% | 86.8% |
| WR | 5.73 | 4.29 | 0.243 | 0.464 | 83.1% | 92.1% |
| TE | 5.56 | 4.05 | 0.118 | 0.405 | 84.4% | 92.3% |

**Note:** R² is low because component mode uses causal features (9-14
per position) rather than the full 50+ feature set. Spearman rank
correlation (0.33-0.49) is the more relevant metric for fantasy — it
measures ranking accuracy rather than point estimate precision.

### Model Artifacts Saved

```
data/models/
├── component_qb.json         (5 component models: pass_yds, pass_tds, ints, rush_yds, rush_tds)
├── component_rb.json         (5 components: rush_yds, rush_tds, rec, rec_yds, rec_tds)
├── component_wr.json         (3 components: rec, rec_yds, rec_tds)
├── component_te.json         (3 components: rec, rec_yds, rec_tds)
├── util_to_fp_qb.joblib      (utilization-to-FP converter)
├── util_to_fp_rb.joblib
├── util_to_fp_wr.joblib
├── util_to_fp_te.joblib
├── feature_scaler_bounded.joblib
├── feature_version.txt        ("9")
├── model_metadata.json
└── [monitoring/quality JSON files]
```

---

## Bugs Fixed in `src/models/train.py`

### 1. `json` local import shadowing (line 854)

`import json` inside the data integrity gate `if` block caused Python's
bytecode compiler to treat `json` as a local variable for the entire
`train_models()` function. When the gate passed (the happy path), the
import never executed, but `json.load()` at line 1298 raised
`UnboundLocalError`. Fix: removed the redundant local import (module-
level `import json` at line 3 is sufficient).

### 2. `numpy` local import shadowing (line 888)

Same pattern: `import numpy as np` inside the gate block shadowed the
module-level `np`. Caused `UnboundLocalError` in horizon model training.
Fix: removed redundant local import.

### 3. Component mode `None` handling

When `position_target_type` is `"component"` (the current config for all
positions), `ModelTrainer.train_all_positions()` sets
`self.trained_models[position] = None` as a placeholder. Multiple
downstream functions assumed this would be a `MultiWeekModel` object:

- `_report_test_metrics()` (line 149): `multi_model.models.get(1)` → NPE
- `_run_backtest_after_training()` (line 308): same pattern
- Horizon model training (line 1078): same
- Metadata `n_features_per_position` (line 1434): same

Fix: Added `if multi_model is None` guards that route to the component
predictor's `.predict()` method instead.

### 4. `EnsemblePredictor.is_loaded` (ensemble.py line 257)

`is_loaded` only checked `position_models` and `single_week_models`,
not `component_predictors`. When only component models exist, the
predictor reported itself as not loaded. Fix: added
`len(self.component_predictors) > 0` to the check.

---

## Current Database State

| Table | Rows | Coverage |
|-------|------|----------|
| `player_weekly_stats` | ~103K | 2006-2025, weeks 1-22 |
| `schedule` | 5,431 | 2006-2025 (100% Vegas coverage) |
| `team_stats` | ~5,100 | 2006-2025 |
| `snap_counts` | 205K | 2018-2025 |
| `ngs_passing` | 4,785 | 2018-2025 |
| `ngs_rushing` | 4,885 | 2018-2025 |
| `ngs_receiving` | 11,708 | 2018-2025 |

---

## Known Limitations / Future Work

### Not yet available
- **2026 NFL schedule:** NFL releases ~July/August. Auto-refresh will
  detect and load it automatically.
- **Horizon models (4w LSTM, 18w deep):** Skipped in component mode.
  Would require training in `util` or `fp` target mode to produce
  `MultiWeekModel` objects the horizon trainer expects.

### Memory issue during full training
The training pipeline crashes (SIGKILL, exit 139) during the
util-to-fp converter phase when Optuna tuning is enabled. The
converters themselves train fine (~80MB total), but the combination
of 95K-row training data + Optuna trials + 334 features causes memory
pressure. Workaround: `--no-tune` flag, or reduce data via
`TRAINING_START_YEAR_DEFAULT`.

### Data gaps (deferred)
- **Snap counts pre-2018:** nflverse doesn't provide snap data before
  2018. Feature engineering imputes these as 0 → median.
- **Draft capital matching:** 0/95K rows matched draft picks (GSIS ID
  vs draft_picks table format mismatch). Draft features default to
  "undrafted" for all players. Needs ID reconciliation.

---

## Verification Commands

```bash
# Check data state
python -c "
import sqlite3; c=sqlite3.connect('data/nfl_data.db').cursor()
c.execute('SELECT COUNT(*) FROM schedule WHERE spread_line IS NOT NULL'); print('Vegas:', c.fetchone()[0])
c.execute('SELECT SUM(CASE WHEN snap_count>0 THEN 1 ELSE 0 END) FROM player_weekly_stats WHERE season>=2018'); print('Snaps:', c.fetchone()[0])
"

# Check models
cat data/models/feature_version.txt   # Should be "9"
ls data/models/component_*.json       # 4 files (QB, RB, WR, TE)

# Test prediction pipeline
python -c "
from src.predict import NFLPredictor
p = NFLPredictor(); print('Init:', p.initialize())
"

# Retrain (if needed)
python -m src.models.train --fast --skip-cache-check --no-tune
```

## Files Changed

| File | Change |
|------|--------|
| `src/models/train.py` | Fixed json/np import shadowing, component mode None handling |
| `src/models/ensemble.py` | Fixed `is_loaded` to count component predictors |
| `scripts/backfill_snap_counts_to_pws.py` | **New** — backfills snap counts to pws |
| `data/nfl_data.db` | Vegas backfill, team code fixes, snap count updates |
| `data/models/feature_version.txt` | Updated 4 → 9 |
| `data/models/model_metadata.json` | Updated with v9 training date and metrics |
| `data/models/component_*.json` | **New** — trained component models |
| `data/models/util_to_fp_*.joblib` | **New** — utilization-to-FP converters |
| `data/models/feature_scaler_bounded.joblib` | **New** — bounded feature scaler |
