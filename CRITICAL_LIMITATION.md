# Critical Limitation: Predictive Ceiling From Weak Feature Signal

> **Status update (2026-04-22):** Step 5 Phase 3 (injury status at
> lock time) is **closed**, and the **workstream-level kill criterion
> has fired**. Per-phase bias gate passes (max \|bias\| = 9.1 % at WR
> α=10 000, well under the ±10 % threshold) and the feature stays
> wired, but Phases 1 + 2 + 3 together delivered +0.004 cumulative R²
> lift vs the plan's +0.02 workstream threshold. Two of three
> "highest-confidence" features (Vegas, injuries) turned out to be
> already-declared-but-silently-dead; fixing both added ~+0.004 R² and
> +2.4 pp cross-season hindsight (α=10 000) that was real but small.
> Opp defense was redundant with existing single-week wiring. The
> "we need more predictive features" hypothesis lacks empirical
> support at this sample size; a re-council before Phase 4 is the
> plan's own instruction. Full writeup:
> `docs/PHASE_3_INJURY_FINDINGS.md`.

> **Status update (2026-04-21, part 2):** Step 5 Phase 2 (opponent
> defense rank vs. position) is **closed as measured-and-rejected.**
> A council-spec season-to-date expanding lag-1 feature
> (`opp_fpts_allowed_s2d_lag1`) was added alongside the existing
> single-week `opp_fpts_allowed`, measured across a 2 × 2 season × α
> matrix, and reverted from `CAUSAL_FEATURES` per the plan's Fail
> branch. Max RB r lift: **+0.0015** against the +0.02 kill threshold.
> Root cause: `opp_fpts_allowed` already existed and was silently
> carrying the signal; the s2d version is collinear. Bonus structural
> fix retained: `player_weekly_stats.opponent` was 100 % empty for
> 2025, so every pre-Phase-2 2025 backtest had the existing
> opponent-defense feature silently dead for that season. Backfilled
> via the schedule table; `team_defense_stats(2025)` populated (568
> rows). Full writeup: `docs/PHASE_2_OPPONENT_DEFENSE_FINDINGS.md`.

> **Status update (2026-04-21):** Step 5 Phase 1 (Vegas implied total)
> is complete. The April 10 R²=0.269 baseline and every downstream
> number in the α sweep were computed with **Vegas features pinned at
> constants** (`implied_team_total=23.0`, `spread=0.0`) — the
> silent-fallback hypothesis in `docs/PHASE_1_VEGAS_FINDINGS.md` was
> confirmed. Backfilling real Vegas lines from the local `schedule`
> cache (2,227 games, 100 % coverage for 2018–2025) lifts overall R²
> by +0.004, QB r by +0.031–0.034 on 2025 (clearing the +0.02 kill
> criterion), and cross-season hindsight win rate at α=10 000 from
> 67.4 % → **69.8 % (30-13, p=0.007, ROI +25.6 %)**. Full writeup:
> `docs/PHASE_1_VEGAS_FINDINGS.md`.

> **Status update (2026-04-20):** Step 3 of the 2026-04-14 council
> (Ridge α sensitivity sweep) is closed. The `RIDGE_DEFAULT_ALPHA` for
> the walk-forward backtester was raised from 1.0 to **10 000** in
> `config/settings.py`. Cross-season evidence: 29-14 (67.4 %) hindsight
> win rate at α=10 000 vs 27-16 (62.8 %) at α=1, p=0.016 over 43 weeks
> (2024 + 2025). Per-position tuning was investigated and shelved —
> indistinguishable from uniform α=10 000 at this sample size. Full
> writeup: `docs/ALPHA_SWEEP_20260419.md`. Decision-quality reporting
> (per-week win rate / ROI) is now wired into every walk-forward run
> via `decision_quality` in the results JSON.

> **Status update (2026-04-11):** This document was rewritten after the
> April 10, 2026 walk-forward backtest. The previous "negative R², +44%
> systematic bias" framing was based on the February 2026 results
> (`backtest_2025_20260215.json`), which was a pre-leakage-fix run. The
> walk-forward infrastructure work on
> `claude/walk-forward-backtest-dqcBD` (commits `f5ba669`, `03dcd27`,
> `52722e1`, `5aecfed`, `0aa359f`, `de261d3`) resolved most of the
> calibration failure. The new dominant limitation is predictive ceiling,
> not bias.

## Summary

The model's bias and ranking failure are largely resolved, but its
**explanatory power is capped by weak feature signal**: the Ridge
walk-forward backtest produces overall Pearson r ≈ 0.52, R² ≈ 0.27, and
loses to a blended trailing-average heuristic. The variance-compression
symptom (predicted std ≈ 0.52 × actual std) that previously looked like
over-regularization is, for RB and WR, the mathematical consequence of
that imperfect correlation — an unbiased linear model with correlation r
will produce predictions whose standard deviation is exactly r × σ_y.
Fixing it requires more predictive features (or a stronger model class),
not less shrinkage.

## Evidence

### April 10, 2026 walk-forward backtest

Results: `data/backtest_results/ts_backtest_2025_20260410_004640.json`
Predictions: `…20260410_004640_predictions.csv`
Model: **Ridge α=1.0** (production ensemble not yet walk-forward tested,
per commit `de261d3`)
Configuration: expanding window, weekly refit, per-fold features,
leakage check passed.

| Metric | Overall | QB | RB | WR | TE |
|---|---|---|---|---|---|
| n | 5,595 | 644 | 1,181 | 3,397 | 373 |
| MAE | 4.90 | 6.16 | 5.45 | 4.56 | 4.04 |
| RMSE | 6.45 | 7.47 | 7.31 | 6.00 | 5.48 |
| R² | 0.269 | 0.092 | 0.258 | 0.257 | 0.152 |
| Pearson r | 0.520 | 0.325 | 0.512 | 0.513 | 0.399 |
| Spearman ρ | 0.534 | 0.299 | 0.539 | 0.505 | 0.470 |
| Pred mean | 8.32 | 11.76 | 9.19 | 7.55 | 6.75 |
| Actual mean | 8.07 | 11.52 | 9.70 | 7.02 | 6.49 |
| **Bias** | **+3.2%** | **+2.1%** | **−5.3%** | **+7.5%** | **+4.0%** |
| Pred std | 3.95 | 3.42 | 4.30 | 3.58 | 2.81 |
| Actual std | 7.54 | 7.84 | 8.50 | 6.97 | 5.95 |
| Std ratio | 0.524 | 0.436 | 0.506 | 0.514 | 0.473 |

### Baseline comparison (same backtest run)

| Method | R² |
|---|---|
| trailing_avg_3w | 0.164 |
| season_avg | 0.244 |
| **blended_heuristic** | **0.279** |
| Ridge model | 0.269 |

The model beats the trailing average and season average but **loses to a
blended heuristic by ~0.01 R²**. This is the core problem: with the
current feature set and Ridge, the model adds no value over a simple
blend of recent performance and season average.

## Root Cause: Variance Compression Is Correlation, Not Shrinkage

For an unbiased linear model, `std(pred) / std(actual) = correlation(pred,
actual)`. Verifying on the April 10 results:

| Position | Pearson r | std ratio | Δ |
|---|---|---|---|
| RB | 0.5118 | 0.5062 | −0.006 |
| WR | 0.5130 | 0.5139 | +0.001 |
| QB | 0.3249 | 0.4356 | +0.111 |
| TE | 0.3986 | 0.4726 | +0.074 |

For RB and WR the identity holds within 0.01 — there is no extra
shrinkage to remove. The "compressed" predictions are exactly what an
unbiased estimator with this much signal must produce. The only way to
widen the predicted distribution without adding bias is to raise the
correlation, i.e. find more predictive features or a more expressive
model.

For **QB and TE** there is a 7–11 point gap on top of the correlation
floor, consistent with additional small-sample over-regularization (Ridge
α=1.0 on ~600 QB and ~370 TE training rows per week is more aggressive
than on ~3,400 WR rows). This is a real lever for those two positions
specifically.

## What Was Investigated and Ruled Out

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Two-stage `UtilizationToFPConverter` compounding error | **Ruled out** | `src/evaluation/ts_backtester.py:357` trains directly on `fantasy_points`. The converter (`src/models/utilization_to_fp.py:139`) is not imported by the walk-forward backtest. The previous CRITICAL_LIMITATION.md framing was wrong on this point. |
| Hard prediction caps in `src/models/ensemble.py:668` (QB ≤65, RB/WR ≤55, TE ≤45) | **Ruled out (inactive)** | Empirical max predictions: QB 23.3, RB 22.8, WR 25.7, TE 17.9 — none come within 30 points of their cap. |
| Huber loss δ=5.0 (`config/settings.py:183`) flattening tails | **N/A in this run** | The April 10 backtest used Ridge α=1.0, not the production ensemble. Huber only applies to XGBoost/LightGBM (`src/models/position_models.py:958`, `:1046`), which were not exercised. Still relevant once the ensemble is walk-forward tested. |
| Feature percentile clipping / utilization bounds | **Minor contributor** | The audit (`data/audit_2025_backtest_report.json`) flagged 7 zero-width and 4 full-range features in the utilization composite — degrades input signal for TE and WR specifically, but does not mechanically cap output variance. |
| Systematic +44% over-prediction bias | **Resolved** | Per-position bias is now in [−5.3%, +7.5%] vs [+8.6%, +77%] in February. |
| QB ranking inversion (Spearman ρ < 0) | **Resolved** | QB ρ = +0.299 (was −0.025); RB ρ = +0.539 (was −0.182). |

## Top Recommendations

1. **Run the production ensemble in the walk-forward backtest.** Use
   `python scripts/run_ts_backtest.py --model ensemble --season 2025`.
   Compare R², Pearson r, and per-position correlation against the Ridge
   baseline. The ensemble adds XGBoost + LightGBM + RF + Ridge stacking
   with Huber loss; whether it raises r above 0.52 is the single most
   important open question.
2. **~~Reduce QB/TE small-sample shrinkage.~~** ✓ resolved 2026-04-20.
   The Step 3 α sweep (`docs/ALPHA_SWEEP_20260419.md`) showed the
   opposite of what this item proposed: α=1 was *under*-regularized,
   not over. Production default is now α=10 000 uniform, which closes
   the QB `std_ratio − r` gap within noise and lifts cross-season
   hindsight win rate from 62.8 % to 67.4 % (p=0.016, n=43). TE's
   gap stays within noise at every α — it is a correlation-ceiling
   problem, not a shrinkage problem, and falls under rec #3 below.
3. **Add predictive features that lift correlation, not regularization
   tweaks that shrink it less.** Highest-leverage candidates from
   `LIMITATIONS.md`: injury status (HIGH), Vegas lines / implied team
   totals, opponent defense rankings by position, weather, bye-week
   adjustments. Each of these addresses the correlation ceiling
   directly.
4. **Beat the blended heuristic.** Until the model's R² exceeds 0.279 on
   the same fold, ensemble complexity is unjustified — it is being
   beaten by a one-line average.
5. **Rerun LOYO across 2018–2025 once the ensemble result is in.** The
   LOYO scaffolding at `src/evaluation/backtester.py:1892` exists but
   has not been executed end-to-end with the production model. This is
   needed for cross-season stability metrics, not just within-season
   accuracy.

## What This Document No Longer Claims

The previous version of this file asserted (a) negative R², (b) a +44%
systematic bias, (c) QB ranking inversion, and (d) that the
`UtilizationToFPConverter` two-stage indirection was compounding error.
All four are either resolved by the walk-forward fixes or were never
true of the walk-forward pipeline in the first place. Anyone following
those claims into a "fix the bias" investigation will find no bug — the
real ceiling is feature signal, not calibration.
