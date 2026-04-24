# Phase 4C — rookie priors — findings

Implements follow-up #4 from `docs/CAUSAL_FEATURES_AUDIT_20260424.md`.
Targeted the rookie-blindness diagnosed in the Step 2 snake-draft
`--ranking week1` mode: rookies had `NaN` `prev_season_ppg` → imputed
to 0 → Ridge penalized them → the 2025 ModelBot drafted
Etienne / Collins / Higgins over ADP's McCaffrey / Kyren Williams.

## What landed

- **4C.0:** `scripts/backfill_draft_picks.py` populated the empty
  `draft_picks` table from nflverse (5,109 rows, 2006-2025, >95 %
  gsis_id coverage from 2011 on).
- **4C.1:** `scripts/compute_rookie_priors.py` fit position ×
  round-bucket rookie-year PPG priors from 1,676 historical rookies
  (2006-2023, min 4 games). Saved to `data/rookie_priors.json`.
- **4C.2:** `_apply_rookie_prior` added to `FeatureEngineer`. For rows
  where `prev_season_ppg` is NaN AND row's season = player's earliest
  season in the frame, the prior fills the value. Module-level caches
  for priors + draft rounds; no per-fold overhead.
- **4C.3:** 2024 + 2025 walk-forward re-run; rookie-split R² measured.

Fitted priors:
```
position     rd1       rd2_3      rd4_7       UDFA
  QB      11.89 (46)  11.12 (23)   7.18 (20)  10.39 (81)
  RB      13.13 (35)   7.43 (84)   4.67 (136)  5.20 (236)
  WR      10.59 (61)   6.18 (164)  4.40 (167)  5.48 (327)
  TE       7.22 (16)   5.13 (51)   3.33 (63)   3.70 (166)
```

## Measurement

Before = commit `4ee4709` (post-audit shares-fixed, no rookie prior).
After  = commit `da139b7` (this phase).

### R² split by rookie-status

| Season | Bucket | n | MAE before → after | R² before → after | Δ R² |
|---|---|---|---|---|---|
| 2024 | all | 5480 | 4.984 → 4.976 | 0.334 → 0.336 | +0.002 |
| 2024 | rookies | 800 | 4.823 → **4.745** | 0.364 → **0.375** | **+0.011** |
| 2024 | veterans | 4680 | 5.011 → 5.015 | 0.326 → 0.326 | +0.000 |
| 2025 | all | 5595 | 4.839 → 4.832 | 0.295 → 0.296 | +0.001 |
| 2025 | rookies | 1054 | 4.545 → **4.489** | 0.143 → **0.154** | **+0.011** |
| 2025 | veterans | 4541 | 4.907 → 4.912 | 0.303 → 0.303 | +0.000 |

Rookie R² improved on both seasons by **+0.011** — small but directionally
correct. Veterans unchanged (confirms the fix is surgical).

### Decision quality (hindsight win rate)

| Season | Before | After | Δ |
|---|---|---|---|
| 2024 | 72.7 %, 16-6, p=0.026 | 72.7 %, 16-6, p=0.026 | 0 |
| 2025 | 76.2 %, 16-5, p=0.013 | 76.2 %, 16-5, p=0.013 | 0 |

Unchanged. Rookies rarely sit in top-6 hindsight lineups — the effect
is too narrow to move a 22-week record.

### Rookie-level prediction deltas (2025 W1, known ADP stars)

| Player | Pos | Pick | Before (PPG) | After (PPG) | Δ |
|---|---|---|---|---|---|
| A. Jeanty | RB | rd1 | 5.34 | 6.39 | +1.05 |
| O. Hampton | RB | rd1 | 5.26 | 6.31 | +1.05 |
| T. McMillan | WR | rd1 | 6.33 | 6.97 | +0.64 |
| T. Hunter | WR | rd1 | 6.68 | 7.32 | +0.64 |
| E. Egbuka | WR | rd1 | 6.48 | 7.11 | +0.63 |
| C. Loveland | WR | rd1 | 6.27 | 6.91 | +0.64 |

Prior moves the needle by roughly **+1 PPG** for top rookies. Consistent
with a ~10 % weight on `prev_season_ppg` in the Ridge coefficient vector
(a 10-point PPG lift on one feature → ~+1 point in FP).

### Draft-sim re-run (2025 week1 mode)

Starters are **identical** before and after. ModelBot rank: #7 of 12
both times. The +1 PPG rookie boost isn't enough to push rookies past
top-ADP veterans (who project 13-16 PPG at W1) into ModelBot's starter
slots.

## Honest interpretation

Phase 4C is a **small, correct fix** that does exactly what the audit
said it would: closes a silent NaN-fill hole. But it does **not**
unblock the draft-sim problem by itself. Reasons:

1. Only rows where `prev_season_ppg` is genuinely NaN get filled — in
   practice, only W1 and W2 of a rookie's first season. W3+ is an
   in-season running PPG (already non-NaN, no change).
2. `prev_season_ppg` is one of ten causal features. Ridge weighs it
   modestly because it correlates with volume features (targets,
   carries) that already proxy "is this a good player." Adding ~10 PPG
   to one feature moves the final prediction by ~1 PPG.
3. The top-ADP veterans still outrank rookies on the W1 model
   prediction. To actually flip draft-sim behavior, we need either:
   - **Phase 4B (VORP)** — weights rookies against position replacement
     level, which is more draft-sensitive than raw FP projection.
   - **A dedicated draft-time model** — trained specifically on
     season-long outcomes with ADP as an input feature, not next-week
     PPG aggregated across the season.

## Files touched

- `scripts/backfill_draft_picks.py` (new, ~130 LOC)
- `scripts/compute_rookie_priors.py` (new, ~150 LOC)
- `src/features/feature_engineering.py` (+ ~80 LOC: `_apply_rookie_prior`,
  `_load_rookie_priors`, `_load_draft_rounds`, `_round_bucket_for`)
- `data/rookie_priors.json` (new artifact)
- `data/nfl_data.db` (draft_picks table now populated)
- `tests/test_rookie_priors.py` (new, 5 tests)

Within CLAUDE.md's 5-source-file budget.

## Commit chain

- `da139b7` — 4C.0 + 4C.1 + 4C.2 code and unit tests
- (this commit) — 4C.3 backtest artifacts + findings doc

## Next

Phase 4C complete. The natural next move is **Phase 4B (VORP)**, which
builds directly on 4C (rookies now have positive priors that VORP can
rank against position replacement). After 4B: Phase 4A (conformal
intervals) and 4D (injury-adjusted variance).

The 70.73 % → 75.61 % kill-gate number is unchanged by 4C (rookie
prior doesn't affect veteran starter picks). That number remains the
current defensible public-facing figure.
