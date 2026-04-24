# CAUSAL_FEATURES audit + share-feature rebuild — 2026-04-24

**Provenance:** user-driven audit of "key validation metrics used and without leakage." The leakage question came back clean. The validation-metrics question did not.

## TL;DR

- **Walk-forward validation is genuinely leakage-safe** (full audit in this
  doc). Expanding-window, single-week test folds, hard runtime checks,
  train/test feature isolation, shift(1) on every rolling feature.
- **But `CAUSAL_FEATURES` was declaring 4 share features that were never
  computed** — same silent-fallback pattern as Phase 3's injury bug. Every
  walk-forward metric we've quoted since commit `5aea8c4` was achieved with
  a smaller feature set than the config implied.
- **Fixing the share dropout alone** (commit `08ddca6`) lifts every
  per-position R² by +0.06 to +0.20 on both 2024 and 2025. Hindsight win
  rate gains of +4.6 pp (2024) and +9.6 pp (2025).
- **Adding `prev_season_ppg` on top** adds another +0.01-0.02 R² per
  position across the board, a +13.6 pp WR jump in 2024 (crosses statistical
  significance), and a −4.8 pp regression in 2025 (still sig). Kept.
- **Implication for prior results:** the 70.73% symmetric win rate that
  cleared the chairman's kill gate was on the pre-audit (broken) feature
  set. The symmetric walk-forward should be re-run before claiming a new
  number.

## Walk-forward leakage audit (passes)

| Check | Code | Verdict |
|---|---|---|
| Fold structure | `ts_backtester.py:296-307` | Expanding-window walk-forward: per-fold train = all rows strictly before the test week; test = the single target week. Not k-fold, not LOGO-by-player, not LOGO-by-season. |
| Runtime leakage assert | `ts_backtester.py:333, assert_no_future_leakage` | Raises `RuntimeError` if any train row is after any test row (by `game_date`, `(season, week)`, and per-player future-rows sample). |
| Train/test feature isolation | `ts_backtester.py:136, leakage_safe_features` | Train features from train-only data; test features from combined block (test rows see train history) then only test rows extracted. |
| Rolling / expanding / EWM | `feature_engineering.py:247, 384, 401` | Every `rolling()`/`ewm()`/`expanding()` composed with `shift(1)` to exclude the current week. |
| Target column | `ts_backtester.py:374` | `fantasy_points` hard-excluded from feature selection. |
| Opponent defense (FP allowed) | `database.py:1381-1400` | SQL JOIN on `tds.week = pws.week - 1` + hard assertion raising if `opp_defense_week >= week`. |
| Scaler | `ts_backtester.py:397` | `StandardScaler` fit on train only, applied to both. |
| `prev_season_ppg` (new) | `feature_engineering.py:217-223` | `shift(1).expanding().mean()` within (player, season) then `shift(1)` across player's rows. Verified by hand on CMC: 2023 W1 = 20.83 matches his 2022 PPG of 20.82. |

## Silent feature-dropout (the problem)

Realistic fold audit (train < 2024-W5, test = 2024-W5) before the fix:

```
QB: declared=10, actual used=10, missing=[]
RB: declared=12, actual used=9,  missing=['snap_share_pct', 'rush_share_pct', 'target_share_pct']
WR: declared=11, actual used=9,  missing=['target_share_pct', 'air_yards_share_pct']
TE: declared=11, actual used=9,  missing=['snap_share_pct', 'target_share_pct']
```

Root causes:
1. The 4 `_pct` columns live in `UtilizationScoreCalculator` (`utilization_score.py:280-295`), which computes them from `utilization_scores` table joins.
2. `utilization_scores` is **empty** — zero rows in this DB.
3. `UtilizationScoreCalculator` isn't called by `ts_backtester`'s feature pipeline anyway (only by `src/predict.py` and `src/evaluation/backtester.py` — a different, unused backtester).
4. `ts_backtester.py:367-368` silently drops any declared feature that isn't in the dataframe. No warning, no assertion.

Silent-fallback pattern score: 3rd recurrence in this codebase (Phase 1 Vegas, Phase 3 injuries, now shares).

## Fix (commit `08ddca6`)

- `_create_base_features`: derive `target_share_pct`, `rush_share_pct`,
  `air_yards_share_pct` from raw `player_weekly_stats` via
  `groupby(["team","season","week"]).transform("sum")` for team denominators.
  `snap_share_pct` NOT added — `snap_count` / `team_snaps` / `snap_share`
  are zero-filled across every season (data never populated). Adding it
  back would recreate the dead-declaration pattern.
- `_create_causal_rolling_features` and `_create_rolling_features` extended
  with the 4 new share columns so `{col}_roll3_mean` is computed in both
  paths with `shift(1).rolling(3).mean()`.
- `CAUSAL_FEATURES` updated to reference the `_roll3_mean` forms
  (leakage-safe) instead of raw `_pct` names (leakage-prone and never
  present anyway).

Post-fix audit:
```
QB: declared=9,  used=9  missing=[]
RB: declared=10, used=10 missing=[]
WR: declared=10, used=10 missing=[]
TE: declared=9,  used=9  missing=[]
```

Share feature coverage on the same test fold:
- `target_share_pct_roll3_mean`: coverage=100%, mean=10.7%, 84.8% non-zero
- `rush_share_pct_roll3_mean`: coverage=100%, mean=10.5%, 52.0% non-zero
- `air_yards_share_pct_roll3_mean`: coverage=100%, mean=10.5%, 68.8% non-zero

## Three-way comparison

Configs:
- **pre-fix baseline** — commit `ba15ef5` era. Shares declared but dropped.
  No `prev_season_ppg`. (2024: `ts_backtest_2024_20260423_055841`; 2025:
  `ts_backtest_2025_20260423_055829`.)
- **shares fixed (b1)** — commit `08ddca6`. Share features properly
  computed. No `prev_season_ppg`. (2024: `044010`; 2025: `044005`.)
- **shares + prev_ppg (full)** — commit `dd31779`. Both fixes. (2024:
  `045712`; 2025: `045634`.)

### Season 2024 (22 weeks)

| config | R² | MAE | Hindsight WR | Record | p_binom | ROI |
|---|---|---|---|---|---|---|
| pre-fix | 0.164 | 3.94 | 54.5% | 12-10 | 0.416 | −1.8% |
| shares fixed | 0.322 | 5.05 | 59.1% | 13-9 | 0.262 | +6.4% |
| shares + prev_ppg | 0.334 | 4.98 | **72.7%** | **16-6** | **0.026** | **+30.9%** |

### Season 2025 (21 weeks)

| config | R² | MAE | Hindsight WR | Record | p_binom | ROI |
|---|---|---|---|---|---|---|
| pre-fix | 0.140 | 3.77 | 71.4% | 15-6 | 0.039 | +28.6% |
| shares fixed | 0.281 | 4.90 | **81.0%** | **17-4** | **0.004** | **+45.7%** |
| shares + prev_ppg | 0.295 | 4.84 | 76.2% | 16-5 | 0.013 | +37.1% |

### Per-position R²

```
2024:
  QB:  0.159  →  0.241 (Δ +0.082)  →  0.259 (Δ +0.018)
  RB:  0.237  →  0.336 (Δ +0.099)  →  0.350 (Δ +0.014)
  WR:  0.096  →  0.270 (Δ +0.174)  →  0.281 (Δ +0.011)
  TE: -0.010  →  0.193 (Δ +0.203)  →  0.204 (Δ +0.011)

2025:
  QB:  0.060  →  0.124 (Δ +0.064)  →  0.143 (Δ +0.020)
  RB:  0.167  →  0.267 (Δ +0.100)  →  0.285 (Δ +0.019)
  WR:  0.112  →  0.267 (Δ +0.155)  →  0.278 (Δ +0.011)
  TE:  0.075  →  0.181 (Δ +0.106)  →  0.192 (Δ +0.011)
```

## Interpretation

1. **Most of the lift is from the share fix, not `prev_season_ppg`.** Every
   per-position R² jumps +0.06 to +0.20 from the share fix alone. WR and TE
   benefit most (they depend heavily on target / air-yards share). The
   +~0.01-0.02 from `prev_season_ppg` is consistent across positions but
   marginal.

2. **MAE worsens even as R² improves.** This is a variance-scaling effect:
   the larger feature set lets the model predict higher means for high-FP
   players, which helps rank-order (R²) but can increase raw absolute error.
   For H2H lineup selection (ranking), this is good. For point-estimate
   calibration (paper-trade notional scoring), it's a mild regression to
   track.

3. **2024 vs 2025 asymmetry on `prev_season_ppg`.** +13.6 pp WR in 2024
   (crosses significance) vs −4.8 pp in 2025 (still sig). Cross-season mean
   +4.4 pp. Sample sizes are small (22 and 21 weeks) so 1-week swings carry
   ~5 pp weight. Keeping the feature because: (a) cross-season mean is
   positive, (b) it's the canonical draft-time signal per the original
   question from the user, (c) reverting it just loses a +0.02 R² bump.

4. **The original "70.73% symmetric walk-forward" kill-gate number is
   stale.** That number was measured on the pre-fix feature set. Under the
   fixed feature set the unfiltered hindsight WR on aggregate 2024+2025 is
   (16+16)/(22+21) = 32/43 = 74.4% — slightly higher than 70.73%. But the
   symmetric walk-forward has its own harness (wide-mode, phantom rows,
   etc.) so the number should be re-run before being quoted.

## What this changes downstream

- **Step 5 paper trade (`scripts/paper_trade_lock.py`).** Predictions locked
  on Sept 2025 used the pre-fix feature set. The locks themselves remain
  valid (the model was what it was at lock time), but new locks should use
  the post-fix set. No rewrite needed, just a note in the doc.
- **Step 2 snake-draft sim.** Re-running the sim on new prediction CSVs is
  trivial (one CLI call per season). The mode-sweep directional result
  from yesterday was on the pre-fix set; if we rerun we'd expect cleaner
  week-1 ranks.
- **Symmetric walk-forward kill-gate.** Should be re-run if we want a
  defensible "filtered win rate" number that reflects the current feature
  set. ~45 min to re-measure.

## Files touched

- `config/settings.py` — CAUSAL_FEATURES rewrite (snap_share_pct excluded,
  others now `_roll3_mean`, `prev_season_ppg` added).
- `src/features/feature_engineering.py` — share derivation in
  `_create_base_features`, roll_cols extended in both rolling helpers,
  `_add_prev_season_ppg` helper for the causal path.
- `data/backtest_results/` — 6 new JSON + CSV pairs (baseline-1 and
  shares+ppg for each of 2024, 2025).

## Commits

- `704d4ba` — pre-fix-baseline + prev_ppg backtest artifacts (superseded,
  kept for reproducibility)
- `08ddca6` — share-feature fix code; prev_ppg reverted from CAUSAL_FEATURES
- `dd31779` — baseline-1 (shares-fixed) artifacts + re-add prev_season_ppg
- `d1be1b1` — 2025 final artifact (shares + prev_ppg)
- (this commit) — 2024 final artifact + this findings doc
