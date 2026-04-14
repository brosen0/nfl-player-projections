# Predictive Ceiling Workstream — Plan

**Council:** April 14, 2026 partial council — Step 5 (planning only; no code yet)
**Provenance:** [`council-transcript-20260414-034617.md`](../council-transcript-20260414-034617.md)
**Reference:** [`CRITICAL_LIMITATION.md`](../CRITICAL_LIMITATION.md), [`LIMITATIONS.md`](../LIMITATIONS.md)
**Status:** plan — no implementation has started.

## Goal

Beat the blended-heuristic baseline. Walk-forward Ridge α=1.0 currently gets **R² = 0.269** on the 2025 season; a one-line `0.5 × trailing_3w_avg + 0.5 × season_avg` heuristic gets **R² = 0.279**. Until the model exceeds the heuristic, ensemble complexity is unjustified.

**Success criterion:** walk-forward Ridge R² ≥ **0.300** on the 2025 season (≥ 0.279 + 2pp headroom). Per-position Pearson r ≥ 0.55 for QB and TE specifically (currently 0.325 / 0.399).

**Hard constraint:** the bias regression test added in Step 2 (`tests/test_backtest_validation.py::TestWalkForwardBiasRegression`) must continue to pass at ±10% per position. Any feature that lifts R² but breaks bias calibration is rejected.

## Where the ceiling sits today

| Position | n (2025) | Pearson r | std_ratio | gap (std_ratio − r) | Stable? |
|----------|---------:|----------:|----------:|--------------------:|---|
| QB | 644 | 0.325 | 0.436 | +0.110 ± 0.047 | yes (2.3σ) |
| RB | 1,181 | 0.512 | 0.506 | −0.015 ± 0.032 | no (at zero) |
| WR | 3,397 | 0.513 | 0.514 | −0.004 ± 0.028 | no (at zero) |
| TE | 373 | 0.399 | 0.473 | +0.130 ± 0.080 | borderline (1.6σ) |

For RB/WR the variance compression is correlation arithmetic (Step 3 baseline) — there is no "shrinkage" to remove. The only path to wider, more accurate predictions is **higher correlation**, which means **better features**, not less regularization.

## Candidate features, by hypothesized r contribution

Ranked by expected contribution × ease-of-acquisition. Estimates are deliberately conservative.

| # | Feature | Hypothesized Δr (overall) | Hardest position to lift | Data source |
|--:|---------|--------------------------:|---|---|
| 1 | **Vegas implied team total** | +0.04 to +0.06 | QB, TE | The Odds API or scrape DraftKings/FanDuel |
| 2 | **Opponent defense rank vs. position** (PFF or DVOA proxy from `nfl-data-py`) | +0.03 to +0.05 | RB, TE | Derive from `nfl_data_py.import_team_desc` + season-to-date allowed FP |
| 3 | **Injury status (Q/D/Out) at lock time** | +0.02 to +0.04 | All; biggest single per-row effect when present | ESPN injury report or `nfl-data-py.import_injuries` |
| 4 | **Game total (over/under) + spread** | +0.01 to +0.03 | QB | Same source as #1 |
| 5 | **Snap-share trend (last 3w − season)** — derivative, not level | +0.01 to +0.02 | RB, WR | Already in DB; derived feature |
| 6 | **Weather (wind, precip, dome flag)** | +0.005 to +0.02 outdoor only | QB, K (out of scope) | OpenWeatherMap historical or `nfl-data-py` schedule |
| 7 | **Bye-week return flag** | +0.005 | All | `nfl-data-py.import_schedules` |

**Total upper-bound combined Δr ≈ +0.13 to +0.22**, putting overall r in `[0.65, 0.74]` if all features hit their high-end estimates with no interaction loss. Realistically interaction loss eats half of that. Target: r ≈ +0.05 to +0.10 cumulative lift (overall r 0.57–0.62; R² 0.32–0.38).

## Phased plan

Strict sequencing — each phase verified before the next starts. Each phase commits independently and is mergeable on its own.

### Phase 1 — Vegas implied team total (highest leverage, easiest)
- 1.1 Add `vegas_implied_total` and `vegas_total` to feature schema.
- 1.2 Backfill historical Vegas lines for 2018–2025 from the chosen source (single one-time cache write).
- 1.3 Wire feature into `causal_features` mode for QB and TE first; RB/WR after correlation impact is measured.
- 1.4 Run walk-forward backtest at α=1.0; compare R², per-position r, bias regression test status.
- **Phase 1 kill criterion:** if QB r does not lift by ≥ +0.02 with bias still passing ±10%, abandon Vegas features and move to Phase 2.

### Phase 2 — Opponent defense rank vs. position
- 2.1 Compute season-to-date "fantasy points allowed by position" per defense, lag-1 (so week N uses through week N-1).
- 2.2 Add as feature for all four positions.
- 2.3 Walk-forward backtest; same checks as Phase 1.
- **Phase 2 kill criterion:** if RB r does not lift by ≥ +0.02 (RB matchup-sensitivity is the strongest first-principles case), abandon and move to Phase 3.

### Phase 3 — Injury status (Q/D/Out at lock time)
- 3.1 Pull historical injury reports (point-in-time, NOT after-the-fact). Audit for leakage with `scripts/trace_player_leakage.py` before training.
- 3.2 Encode as ordinal: Healthy=0, Q=1, D=2, Out=3.
- 3.3 Walk-forward backtest.
- **Phase 3 kill criterion:** if any position's bias breaks ±10% (most plausible failure mode: Out players get included in averaging), revert and design a player-active filter instead.

### Phase 4 — Combined model + ensemble re-eval
- 4.1 Walk-forward backtest with all Phase 1–3 features, Ridge α=1.0 baseline.
- 4.2 If R² ≥ 0.300, run alpha sweep (`scripts/run_alpha_sweep.py`) on the new feature set — Step 3's borderline TE gap may now move.
- 4.3 Walk-forward backtest with production ensemble (XGBoost + LightGBM + Ridge stack) on the new feature set. This is the first time the ensemble has been evaluated against the leakage-free walk-forward (per `CRITICAL_LIMITATION.md` Top Recommendation #1).
- 4.4 Decide: ship Ridge or ensemble based on R²/bias trade.

### Phase 5 (optional) — Game-script and weather (low-leverage, high-cost)
Only if Phases 1–3 deliver R² ≥ 0.300. Otherwise the project's bottleneck is somewhere else (likely sample size or feature interactions) and adding more features won't help.

## Validation strategy

Every phase runs the same gate:

1. `pytest tests/test_backtest_validation.py::TestWalkForwardBiasRegression` must pass.
2. `python scripts/run_ts_backtest.py --season 2025` must produce R² ≥ previous phase's R².
3. `python scripts/analyze_std_ratio.py <new_predictions.csv>` must not increase the QB/TE gap above its current value (+0.110 / +0.130). A feature that improves r but expands the shrinkage gap on QB/TE means the feature is non-causal noise.
4. Compare against the blended heuristic R² 0.279 — this is the absolute floor; below it, ensemble complexity has no justification.

## Kill criteria for the entire workstream

- After Phases 1–3 (the three highest-confidence features), if cumulative R² lift is < +0.02 (i.e., R² still < 0.289), the predictive-ceiling diagnosis is wrong — the bottleneck is sample size or feature interactions, not feature breadth. Re-council before continuing.
- If any phase regresses the bias regression test, that feature is removed before the next phase starts. No "we'll fix it later" carve-outs.
- If Vegas API quota or injury-report point-in-time data cannot be acquired without leakage, the corresponding phase is dropped and the ceiling is what it is. Don't fabricate features to hit a target.

## Out of scope

- **K/DST projection upgrades.** Council scope was QB/RB/WR/TE.
- **Distribution prediction (quantile regression).** Per the original April 1 council and the partial council's reasoning, distributional prediction comes after point estimates are unbiased AND meaningfully predictive. Step 5 is about getting r above 0.55, not about widening the prediction interval.
- **Component prediction (target-share → FP).** Already explored in commit `fb828b6` "Add component prediction." Whether to revisit is downstream of whether Phases 1–3 hit the R² target.
- **Model-architecture changes** (deep learning, transformers, etc.). The ceiling diagnosis is feature signal, not model class. Don't change two things at once.

## Estimated effort

| Phase | Code | Backtest runtime | Total elapsed |
|---|---|---|---|
| 1. Vegas total | ~half day | ~30 min/run × 2 runs | ~1 day |
| 2. Opponent defense | ~half day | ~30 min/run × 2 runs | ~1 day |
| 3. Injury status | ~1 day (data acquisition is the cost) | ~30 min/run × 2 runs | ~2 days |
| 4. Combined + ensemble | ~half day | ~2 hours (full sweep + ensemble walk-forward) | ~1 day |
| **Total** | | | **~5 days** |

## What this plan does NOT cover

- **Implementation.** No code has been written. Each phase opens its own branch off `main` (suggested naming: `claude/predictive-ceiling-phase-1-vegas`, etc.).
- **Data acquisition specifics.** API key management, rate-limiting, caching strategy — addressed at the start of each phase.
- **Ensemble re-eval cost.** The production ensemble has not been walk-forward tested. Phase 4 is its first real test; if it underperforms Ridge in walk-forward, that's its own conversation.

---

*This is a plan, not work.  No phase begins without explicit approval.*
