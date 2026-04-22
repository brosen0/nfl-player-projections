# Phase 3 (Injury status at lock time) — Findings

**Branch:** `claude/add-council-status-rfitZ`
**Plan reference:** [`docs/PREDICTIVE_CEILING_PLAN.md`](./PREDICTIVE_CEILING_PLAN.md)
**Status:** ✓ **per-phase kill criterion passes; feature retained.** **Workstream-level kill criterion fires** — cumulative Phase 1+2+3 R² lift is +0.004 across 43 weeks of cross-season walk-forward, far below the plan's +0.02 threshold that gates Phase 4. Re-council recommended before continuing.

## TL;DR

Phase 3 fixed the silent-fallback on the declared `injury_score` feature (same pattern as Phase 1 Vegas). Point-in-time injury reports from `nfl_data_py.import_injuries()` were cached into a new `player_injuries` table (45,334 rows, 2018–2025) and merged into the causal feature pipeline. The feature now fires on ~4 % of walk-forward rows (the `Questionable` status — `Out` players never make it into `player_weekly_stats` because they didn't play, so they are already excluded by construction).

The per-phase kill criterion — any position's bias within ±10 % — **passes comfortably** (max 9.1 % at WR 2025 α=10 000, which is unchanged from the pre-injury baseline). **But decision quality didn't move on any cell** (all four 2024/2025 × α=1/10 000 hindsight records are identical to their pre-injury baselines), and the overall R² lift is +0.0001 to +0.0007 per cell. The feature is now live and correctly wired, but its marginal contribution is noise-dominated.

## Bug fixed mid-phase (worth flagging)

The first-pass wiring added `_merge_injury_data_from_cache` to `create_features()` at `src/features/feature_engineering.py:118`. **But that's the non-causal path.** The walk-forward backtester runs under `FEATURE_MODE='causal'`, which dispatches to `create_causal_features()` at line 153 — a completely separate code path with its own silent-fallback default at line 178–179 (`if "injury_score" not in df.columns: df["injury_score"] = 1.0`).

Symptom: the 2024 post-injury 4-cell run matched pre-injury baselines to 4 decimal places — bit-for-bit — which is only possible if the feature matrix didn't actually change. Fix (commit `b8383e9`): add the merge call to `create_causal_features()` as well. Smoke test after fix: 219/5480 rows on 2024 correctly show `injury_score=0.5` (all `Questionable`).

Lesson retained: any Phase-style silent-fallback audit should check both `create_features` and `create_causal_features` paths. Added the non-causal path to my mental model for Phase 4.

## What landed

1. **`src/utils/database.py`:** new `player_injuries` table with `UNIQUE(player_id, season, week)` + index on `(season, week)`. Idempotent — the migration fires on every `DatabaseManager()` instantiation.
2. **`scripts/backfill_injuries.py`:** one-time fetch from `nfl_data_py.import_injuries()` for 2018–2025. 45,337 rows fetched, 45,334 persisted after UNIQUE dedup. Works in this sandbox (unlike `import_schedules()`, which was blocked by `habitatring.com` and required a GitHub-raw redirect in Phase 1).
3. **`src/features/feature_engineering.py::_merge_injury_data_from_cache`:** reads the cache, maps `report_status` → `injury_score` via the existing `InjuryDataLoader.INJURY_STATUS_SCORES` (Out=0.0, Doubtful=0.15, Questionable=0.50, Probable=0.85, Healthy/None=1.0), merges on `(player_id, season, week)`. Called from **both** `create_features` (line 118) and `create_causal_features` (line 178) so the merge fires under either feature mode.
4. **`data/nfl_data.db`:** schema migrated + 2018–2025 injuries backfilled.

## Spec deviation (deliberate)

The plan called for an ordinal `{Healthy:0, Q:1, D:2, Out:3}` encoding. Phase 2 showed that adding a collinear twin of an existing feature to `CAUSAL_FEATURES` delivers no lift. Phase 3 instead fixes the silent-dead `injury_score` using the existing 0-1 availability semantic already wired through `InjuryDataLoader` — one feature, one CAUSAL_FEATURES entry (unchanged from before), just fills the dead values. Same monotonicity, opposite sign; Ridge handles that.

## Coverage audit

Injury-report matching against played player-weeks (i.e., rows that appear in `player_weekly_stats`):

| Season | Total rows | Matched | Questionable | None / (cleared) | Other (Doubtful / Note) |
| :----: | :--------: | :-----: | :----------: | :--------------: | :---------------------: |
| 2024   | 5,480      | 1,014 (18.5 %) | 219 | 793 | 2 |
| 2025   | 5,595      |   973 (17.4 %) | 193 | 780 | 0 |

**Out / Doubtful / IR rows don't show up** because those players didn't play that week and are absent from `player_weekly_stats`. The plan's "most plausible failure mode: Out players get included in averaging" can't fire here — by the time rows reach the walk-forward training set, Out players are already filtered out.

Practical signal: ~4 % of weekly rows carry `injury_score=0.5` (`Questionable`); the other ~96 % are pinned at 1.0 (no report or cleared). That narrow variance caps the feature's possible lift.

## Cross-season 2 × 2 matrix (season × α, post- vs pre-injury)

| Cell                        | ΔR²     | ΔQB r   | ΔRB r   | ΔTE r   | ΔWR r   | Max \|bias\| | Hindsight W-L |
| :-------------------------- | :-----: | :-----: | :-----: | :-----: | :-----: | :----------: | :-----------: |
| 2024 α=1                    | +0.0001 | +0.000  | +0.000  | +0.001  | +0.000  | 4.0 %        | 13-9 (unchanged) |
| 2024 α=10 000               | +0.0001 | +0.000  | +0.001  | +0.001  | +0.000  | 4.3 %        | 14-8 (unchanged) |
| 2025 α=1                    | +0.0005 | **+0.003** | +0.000  | +0.000  | +0.001  | 7.4 %        | 16-5 (unchanged) |
| 2025 α=10 000               | +0.0007 | **+0.003** | +0.000  | +0.001  | +0.000  | 9.1 %        | 16-5 (unchanged) |

**Per-phase kill criterion (bias ≤ ±10 %): PASSES in every cell.** The feature stays wired.

**Decision quality is unchanged everywhere.** Cross-season combined 43-week hindsight:

| Config                    | Pre-injury | Post-injury | Δ |
| :------------------------ | :--------: | :---------: | :-: |
| α=1 post-Vegas            | 29-14 (67.4 %) | 29-14 (67.4 %) | 0 |
| α=10 000 post-Vegas       | 30-13 (69.8 %) | 30-13 (69.8 %) | 0 |

ROI, p-values, and avg margin identical.

## Workstream-level kill criterion — **FIRES**

From `docs/PREDICTIVE_CEILING_PLAN.md`:

> After Phases 1–3 (the three highest-confidence features), if cumulative R² lift is < +0.02 (i.e., R² still < 0.289), the predictive-ceiling diagnosis is wrong — the bottleneck is sample size or feature interactions, not feature breadth. Re-council before continuing.

Cumulative R² lift across Phases 1 + 2 + 3 on cross-season walk-forward at α=10 000 (the production default):

| Checkpoint                  | Cross-season avg R² | Δ vs previous |
| :-------------------------- | :-----------------: | :------------: |
| Pre-Phase-1 (pre-Vegas)     | 0.287               | baseline      |
| Post-Phase-1 (+Vegas)       | 0.291               | **+0.004** ✓ cleared Phase 1 kill |
| Post-Phase-2 (+s2d reverted) | 0.291              | 0.000 (reverted) |
| Post-Phase-3 (+injury)      | 0.291               | +0.0001 (noise) |

**Total: +0.004 cumulative.** The plan's workstream-level threshold was +0.02. **We are 5× under the threshold.** Per the plan's own rule: the predictive-ceiling diagnosis needs a second opinion before Phase 4 (combined + ensemble re-eval) begins.

The three highest-confidence features on the plan's list have been measured. Two of the three were already wired but silently dead (Vegas, injury); one was wired in a form different from the spec (opp defense single-week vs spec's season-to-date). Only Vegas meaningfully moved the needle; injury and opp-defense-s2d contributed less than noise. The "we need more predictive features" hypothesis now has empirical evidence against it.

### Re-council proposition

The council that wrote Phase 5 assumed:
1. The features declared in `CAUSAL_FEATURES` were live.
2. The main lever for r > 0.55 was feature breadth.

Phases 1–3 invalidate both. The features were largely dead (silent-fallback across multiple surfaces), and after fixing the dead ones plus adding the spec-compliant new ones, R² moved +0.004 cumulative. Candidate alternative hypotheses a re-council should weigh:

- **Feature interactions, not feature breadth.** The 15-ish features Ridge currently uses per position may be the right signal but need nonlinear interactions that a linear model can't extract. Phase 4's "ensemble re-eval" may be the only remaining plan item with material upside.
- **Sample-size-bound ceiling.** 22 weeks × 32 teams ≈ 700 rows per fold per position; at that size, a Pearson r > 0.55 on a Ridge may genuinely be a structural limit, not a feature limit.
- **Target-level noise.** Fantasy points per week are dominated by TD events with <30 % in-week predictability (per the 2026-03-31 First Principles Thinker response). Some ceiling is just noise.
- **Component prediction may be the right frame after all.** The 2026-03-31 council shelved it; the 2026-04-14 council excluded it. It may be time to revisit given point-estimate work has capped out.

## Honest caveats

- **43 weeks is a small sample for +0.0001 Δ claims.** The "signal indistinguishable from noise" finding could mask a real but small effect. I'd not ship the feature as "proved to help," but I'd keep it wired because (a) kill criterion passes, (b) structural silent-fallback fix is unambiguously correct, and (c) its Monday-morning cost is ~1 SQL query per fold and ~0.1 s compute.
- **Only `Questionable` status reaches the model.** If a future change ever lets `Out` / `Doubtful` rows through (e.g. using the feature for projection-time what-if rather than walk-forward), the kill criterion's "Out-dragging-averages" concern becomes live. The feature should be pulled before that happens unless a player-active filter is added.
- **Bias numbers for 2025 WR are at 7.4–9.1 %** — close to the ±10 % threshold. If any Phase 4 change moves that up, the bias regression test will fail. Worth watching.

## Verification performed

- Bias regression test (`tests/test_backtest_validation.py::TestWalkForwardBiasRegression`) — passes.
- 32/32 tests in `test_backtester.py` + `test_ts_backtester.py` + `TestWalkForwardBiasRegression`.
- Smoke test of `_merge_injury_data_from_cache` on 5 2024 `Questionable` rows: all 5 correctly resolve to `injury_score=0.5`, `is_injured=1`.
- Full-pipeline smoke test: `create_features(2024_data)` produces 219 rows with `injury_score=0.5` and 5261 at 1.0 — matching the DB `Questionable` count.

## Artifacts

- Schema + migration: `src/utils/database.py::player_injuries`.
- Backfill script: `scripts/backfill_injuries.py`.
- Feature wiring: `src/features/feature_engineering.py::_merge_injury_data_from_cache`.
- Backfill data: `data/nfl_data.db` (45,334 rows in `player_injuries`).
- Walk-forward runs (post-fix, post-injury):
  - 2025 α=10 000: `data/backtest_results/ts_backtest_2025_20260422_024024_*`
  - 2025 α=1:      `data/backtest_results/ts_backtest_2025_20260422_024028_*`
  - 2024 α=10 000: `data/backtest_results/ts_backtest_2024_20260422_025527_*`
  - 2024 α=1:      `data/backtest_results/ts_backtest_2024_20260422_025547_*`
- Pre-fix (no-op) runs retained for provenance: `20260422_020748`, `_020754`, `_022156`, `_022204`.

---

*Phase 3 closed. Workstream-level kill criterion fires — re-council recommended before Phase 4.*
