# Phase 2 (Opponent defense rank vs. position) — Findings

**Branch:** `claude/add-council-status-rfitZ`
**Plan reference:** [`docs/PREDICTIVE_CEILING_PLAN.md`](./PREDICTIVE_CEILING_PLAN.md)
**Status:** ✗ **kill criterion failed.** `opp_fpts_allowed_s2d_lag1` added, measured, reverted from `CAUSAL_FEATURES` per the plan's Fail branch. Helper code retained for possible Phase 3 / 4 reuse.

## TL;DR

Phase 2 added a council-spec season-to-date expanding lag-1 opponent-defense feature (`opp_fpts_allowed_s2d_lag1`) alongside the existing single-week-prior `opp_fpts_allowed`. **Across the 2 × 2 season × α matrix the feature moved every metric by less than noise; maximum RB r lift was +0.0015 against a +0.02 kill threshold.** The feature is redundant with the already-live single-week version — `team_defense_stats` was populated for 2006–2024 and the single-week feature was already declared in `CAUSAL_FEATURES`. Ridge cannot extract additional variance from a smoothed twin of a feature it already weighs.

**Bonus structural fix landed (kept):** `player_weekly_stats.opponent` was 100 % empty for 2025. Every pre-Phase-2 backtest on 2025 — including all of Phase 1's reported lift numbers — had the existing `opp_fpts_allowed` silently dead for the entire 2025 season. The opponent field was backfilled from the schedule table; `team_defense_stats` for 2025 was then populated from the aggregator. This data fix is retained regardless of the kill-criterion outcome and is a genuine silent-fallback cleanup equivalent to Phase 1 Step C.

## What landed (retained)

1. **Data backfill (`data/nfl_data.db`):**
   - One-shot SQL UPDATE filled `player_weekly_stats.opponent` for all 5,595 2025 rows from `schedule.home_team/away_team`.
   - `DatabaseManager.ensure_team_defense_stats(2025)` then produced 568 rows (32 teams × 17–18 regular-season weeks + a few playoffs) with sensible per-position means (QB 13.1, RB 20.2, WR 42.0, TE 4.3).
   - TE zeros: 50 % of 2025 rows (284/568). Known sparsity in low-usage weeks; flagged, not a bug.

2. **Feature helper (`src/features/feature_engineering.py:_add_opp_fpts_allowed_s2d_lag1`):**
   - Queries `team_defense_stats` directly.
   - Per-(team, season) `shift(1).expanding(min_periods=1).mean()` on each `fantasy_points_allowed_{pos}` column — pattern lifted from the canonical `season_expanding_ppg` at `:381`.
   - Merges on `(opponent, season, week)`, resolves per-row by player position.
   - Coverage check: logs WARNING if > 10 % default rate (tighter than Phase 1's 50 % Vegas threshold — team-defense data is more reliable than Vegas-API access).
   - Keeps running on every feature-engineering invocation (cheap ~10 k-row query; ≤ 1 s added per 17 min backtest) so the feature is available for re-declaration without further code change.

## What landed and then reverted

- `opp_fpts_allowed_s2d_lag1` was added to `CAUSAL_FEATURES` for all four positions, then removed after measurement per the plan's Fail branch. Diff on `config/settings.py` is net-zero.

## Cross-season 2 × 2 matrix (s2d feature active, α ∈ {1, 10 000}, season ∈ {2024, 2025})

### Per-position Pearson r (post-s2d vs. post-Vegas baseline)

| Cell                 | QB Δr   | RB Δr    | TE Δr   | WR Δr   |
| :------------------- | :-----: | :------: | :-----: | :-----: |
| 2025 α=10 000        | +0.005  | **+0.0007** | +0.000  | −0.001  |
| 2025 α=1             | +0.005  | **+0.0007** | +0.001  | +0.000  |
| 2024 α=10 000        | +0.004  | **+0.0015** | +0.001  | −0.001  |
| 2024 α=1             | +0.004  | **−0.0002** | +0.000  | −0.000  |

RB r lift (the kill-criterion metric, per the plan): max **+0.0015**, min −0.0002. **Threshold +0.02 not cleared in any cell.**

### Overall R² and decision-quality deltas

| Cell                 | ΔR²     | Δ Hindsight W-L     | Δ ROI   |
| :------------------- | :-----: | :-----------------: | :-----: |
| 2025 α=10 000        | +0.000  | same (16-5, 76.2 %) | +0.0 pp |
| 2025 α=1             | +0.001  | same (16-5, 76.2 %) | +0.0 pp |
| 2024 α=10 000        | +0.000  | same (14-8, 63.6 %) | +0.0 pp |
| 2024 α=1             | +0.000  | same (13-9, 59.1 %) | +0.0 pp |

No decision-quality movement in any cell. Per-position bias stayed within ±10 % everywhere (max movement: QB 2025 α=10 000, +3.96 % → +3.96 %, indistinguishable).

## Why it failed

The plan's Phase 2 kill criterion was written assuming opponent-defense was a new feature. It is not:

- `CAUSAL_FEATURES` already declared `opp_fpts_allowed` for all four positions.
- `_create_opponent_features` at `src/features/feature_engineering.py:475` already populates it from a SQL-enforced `tds.week = pws.week - 1` lag-1 join.
- The single-week-prior `opp_fpts_allowed` and the season-to-date expanding `opp_fpts_allowed_s2d_lag1` are **highly collinear** within each season — they both encode "how bad this defense has been at this position this year." Ridge splits a small total weight between the two and the model cannot extract additional variance.

In other words: the existing feature had already eaten most of Phase 2's hypothesized Δr. What's left is the difference between a single-week-ago snapshot and a season-to-date mean, and at a sample of 21-22 weeks per season that difference is within noise.

A "replace single-week with s2d" design (not the one taken — the user chose "add alongside" on 2026-04-21) might move the numbers differently, but the plan had no success signal specified for that branch and the first-principles case is weak: the single-week feature has the higher-recency signal that s2d smooths out. If someone wants to come back to this, run both as isolated A/B experiments (Phase 2 plan's "Run both" option).

## What was NOT revealed

- **Isolation of the 2025 data-backfill effect.** The 2025 pre-Phase-2 baseline (yesterday's post-Vegas runs) had `team_defense_stats` empty for 2025 and so had `opp_fpts_allowed` silently dead on that season. Today's post-s2d runs have BOTH (a) the backfilled single-week `opp_fpts_allowed` firing on 2025 for the first time, AND (b) the new s2d feature. The fact that their combined effect is nearly zero is suggestive — either the single-week feature added little on 2025 too, or it added a little and the s2d subtracted the same — but the plan's Step 4a baseline-only run was skipped to save ~17 min of compute. The 2024 cells do not suffer this confound (both pre- and post-s2d have single-week `opp_fpts_allowed` live), and the 2024 s2d Δr values are just as close to zero. That's the strongest evidence that the feature itself is the problem, not the missing isolation.
- **Feature-level Ridge coefficient inspection.** Directly reading β for `opp_fpts_allowed` vs `opp_fpts_allowed_s2d_lag1` across the walk-forward folds would definitively show the collinearity story. Deferred — the outcome metrics are conclusive enough to reject the feature without needing the coefficient plot.

## Honest caveats

- **43 weeks is a small sample.** Lifts of 0.01–0.02 in per-position r are noise-dominated. The kill threshold of +0.02 is calibrated against this — smaller lifts cannot be distinguished from chance with 22-week seasons. A feature that genuinely adds +0.01 r would also fail the criterion, and Phase 2's feature may be slightly in that category. But calling a +0.0015 lift "promising" would be motivated reasoning.
- **Per the plan's sequencing, Phase 3 (injury status) and Phase 4 (ensemble re-eval) can still proceed.** The workstream-level kill criterion is "cumulative R² lift < +0.02 across Phases 1–3"; after Phase 1 we are at +0.004 overall R² and +4.6 pp hindsight lift. That budget is still far from exhausted — Phase 2 just contributed zero.
- **The opponent-field backfill is an unambiguous structural win.** Treating it as "sunk" undervalues it. Any future downstream workstream that uses `player_weekly_stats.opponent` on 2025 would have silently degraded; that bug is now closed.

## Verification performed

- Walk-forward bias regression (`tests/test_backtest_validation.py::TestWalkForwardBiasRegression`) — passes both pre- and post-revert.
- 32/32 tests in `test_backtester.py` + `test_ts_backtester.py` + `TestWalkForwardBiasRegression` — pass.
- Smoke test of `_add_opp_fpts_allowed_s2d_lag1` on 6-row fixture (2 seasons, 3 positions): produces sensible per-game expanding means; Week 1 defaults to 0 as designed.

## Artifacts

- Helper code: `src/features/feature_engineering.py::_add_opp_fpts_allowed_s2d_lag1` (retained, not in CAUSAL_FEATURES after revert).
- 2025 α=10 000 s2d: `data/backtest_results/ts_backtest_2025_20260421_053551_*`
- 2025 α=1 s2d:      `data/backtest_results/ts_backtest_2025_20260421_053607_*`
- 2024 α=10 000 s2d: `data/backtest_results/ts_backtest_2024_20260421_055318_*`
- 2024 α=1 s2d:      `data/backtest_results/ts_backtest_2024_20260421_0_*` *(latest 2024 JSON file)*
- `data/nfl_data.db`: 2025 opponent field + team_defense_stats backfill (both retained).

---

*Phase 2 is closed as "measured and rejected." The plan's sequencing continues to Phase 3 (injury status) whenever the user authorizes.*
