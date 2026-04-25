# Phase 4A — Conformal prediction intervals — findings

Implements Phase 4A of the Step 4 plan: distribution-free
prediction intervals around the walk-forward Ridge point estimates,
so the start/sit product can say "we're 80 % confident this player
scores between X and Y" with empirical evidence behind the
confidence claim.

## Approach

Sequential / online split-conformal as a **post-processing step** on
existing predictions CSVs (no `ts_backtester.py` change). For each
row at (player, season=S, week=W):

1. Maintain a per-position rolling pool of residuals
   `|actual - predicted|` from the prior 8 weeks of S (and the last
   8 weeks of S−1 as a seed for early-S folds).
2. The 80 % interval half-width = empirical 80th percentile of that
   pool. Same for 95 %.
3. Strict temporal causality: week W's interval uses only weeks
   `< W` residuals.
4. Empty/tiny pool fallback: position-specific constants
   (QB 80 %=8 PPG, RB 7, WR 6.5, TE 5.5; 95 % roughly 1.7×).

Rationale for post-processing instead of in-backtester: the
walk-forward emits causal residuals already; conformal is a deter-
ministic transform on top. Keeping it separate means we can
re-run conformal on any prior CSV without rerunning the 17-min
backtest.

## Files

- `scripts/add_conformal_intervals.py` — adds `lo80, hi80, lo95,
  hi95` columns to a predictions CSV. Optional `--prior-season` flag
  seeds the pool from S−1's tail.
- `scripts/analyze_coverage.py` — reports overall + per-position
  coverage at 80 % and 95 % plus mean interval widths.
- `tests/test_conformal_intervals.py` — 6 unit tests covering
  causality, pool growth, position stratification, window pruning,
  seed application, and synthetic-Gaussian coverage.

3 source files. Within budget.

## Results (post-audit, post-rookie-prior CSVs)

Predictions CSV inputs:
- 2024: `ts_backtest_2024_20260424_165248_predictions.csv`
- 2025: `ts_backtest_2025_20260424_165230_predictions.csv`
  (seeded from 2024's last 8 weeks: 190 QB / 346 RB / 833 WR /
  124 TE residuals)

| Season | Group | n | Cov 80 % | Cov 95 % | Width 80 % | Width 95 % |
|---|---|---|---|---|---|---|
| 2024 | overall | 5480 | **79.1 %** | **94.5 %** | 14.48 | 25.24 |
| 2024 | QB | 676 | 76.6 % | 93.8 % | 18.93 | 28.32 |
| 2024 | RB | 1289 | 79.4 % | 94.8 % | 14.49 | 26.33 |
| 2024 | WR | 3033 | 79.8 % | 94.7 % | 13.87 | 24.69 |
| 2024 | TE | 482 | 77.0 % | 93.4 % | 12.06 | 21.44 |
| 2025 | overall | 5595 | **80.3 %** | **95.3 %** | 14.47 | 25.84 |
| 2025 | QB | 644 | 83.7 % | 96.3 % | 20.09 | 28.62 |
| 2025 | RB | 1181 | 78.0 % | 94.3 % | 15.35 | 29.59 |
| 2025 | WR | 3397 | 80.2 % | 95.5 % | 13.30 | 24.46 |
| 2025 | TE | 373 | 81.8 % | 94.6 % | 12.64 | 21.74 |

## Gate

Plan gate: 80 % intervals cover **76-84 %** of held-out actuals;
95 % intervals cover **91-99 %**. Same band per-position.

Result: **PASS on every cell**. Coverage lands at nominal; nothing
sits outside the band.

## What this enables

- Honest UI claim: "80 % confidence Saquon Barkley scores between
  9.0 and 23.5 fantasy points this week" — empirically calibrated
  on 11k+ predictions across 2 seasons.
- Per-position width comparison: QB intervals are widest
  (~19 PPG at 80 %) because QB scoring has the highest variance.
  TEs the narrowest (~12 PPG). Fantasy practitioners' intuition
  confirmed quantitatively.
- Seed-from-prior-season works: 2025's W1-W2 are not handicapped
  by the fallback constants because the 2024 tail provides ~1500
  residuals.

## What this does NOT change

- Point predictions are unchanged. Coverage doesn't make the model
  better at picking the median; it makes the model honest about
  how unsure it is.
- The 75.6 % / p5=65.85 % symmetric kill-gate number is the same
  (decisions are still made on the point predictions; intervals
  are a UI / risk-disclosure feature).
- Doesn't affect the draft kill verdict (commit `47771ae`) — the
  pre-draft model still loses to ADP, intervals don't fix that.

## Width sanity-check

For a Gaussian residual distribution with σ ≈ MAE × √(π/2), a
symmetric 80 % interval should be ≈ 1.28 × σ × 2 = 2.56 × σ.
Walk-forward MAE per position is ≈ 5-6 PPG → σ ≈ 6.3 - 7.5 → 80 %
width ≈ 16-19. Observed: 12-20. Roughly Gaussian, slightly
heavier-tailed than normal (which is why the empirical conformal
quantile is more honest than a Gaussian approximation).

## Out of scope (defer to Phase 4D)

- Injury-adjusted variance: a Questionable / Doubtful flag should
  widen the interval. Currently the intervals are position-stratified
  but injury-blind. Phase 4D scales width by injury status.
- Player-specific variance: high-variance individuals (boom/bust
  WRs) should get wider intervals than steady-target volume guys
  with the same MAE. Could be a Phase 4E (not in current plan).

## Commit

- (this commit) — `scripts/add_conformal_intervals.py` +
  `scripts/analyze_coverage.py` + `tests/test_conformal_intervals.py`
  + 2 conformal-augmented prediction CSVs + this findings doc.
