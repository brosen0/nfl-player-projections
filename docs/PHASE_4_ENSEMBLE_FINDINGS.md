# Phase 4 (ensemble re-eval) — Findings: KILL on runtime

**Provenance:** Phase 4 of the predictive-ceiling workstream
(`docs/PREDICTIVE_CEILING_PLAN.md`) and the sole Phase-4 gate in the
2026-04-22 re-council (`council-transcript-20260422-032550.md`).

**Rule-of-record:** `docs/PHASE_4_PREREGISTRATION.md` (committed at
`fc583e6` before any ensemble code ran). The kill-gate list was
locked BEFORE the run and is applied below without modification.

**Verdict:** ✗ **KILL on runtime.** Neither walk-forward completed;
both processes died before emitting a results JSON. Pre-registered
kill gate #3 (walk-forward runtime > 4× Ridge) fires cleanly. Ridge
α=10 000 remains the production default.

---

## What was run

```
python scripts/run_ts_backtest.py --model ensemble --season 2024 &
python scripts/run_ts_backtest.py --model ensemble --season 2025 &
```

Started 2026-04-22 at ~19:23 UTC, ran in parallel, died ~12.8 h
later without completing. Output files retained on `/tmp/`.

- Feature set: current production `CAUSAL_FEATURES`. Vegas + injury
  merges wired as of commit `fc583e6`.
- Model: `ensemble_model_factory` — XGBoost + LightGBM + RF + Ridge
  stack with Huber loss, OOF stacking, isotonic calibration.

## What happened

Per-week runtime was **~650-700 s** (≈11 min) vs Ridge's ~50 s —
**roughly 13× slower per week**. The walk-forward loop itself made
it to:

- **2024:** Week 21 of 21 (the final week in scope for the season).
  The process then entered the post-loop full-dataset retrain pass
  required to produce the `save_results()` artifact and **died
  during RB / WR retraining** without emitting any JSON or
  predictions CSV.
- **2025:** Week 20 of 21. Died during week-21 OOF stacking (5-fold
  season-aware split) before the walk-forward loop finished.

No `BACKTEST RESULTS` header, no `Done.` summary, no
`ts_backtest_*_*.json` on disk, no `decision_quality` block
produced.

The failure mode is almost certainly resource exhaustion (memory
creep across 21 weeks × 4 positions × five-fold stacking × growing
train frame), compounded by the sandbox's 12-hour-ish wall-clock
ceiling. It wasn't an exception or a bug — the process was simply
terminated mid-training. Either way, the finishing requirement is
not met.

### Pre-registered kill gate #3

From `docs/PHASE_4_PREREGISTRATION.md`:

> **Walk-forward runtime exceeds 4× Ridge runtime.** Ridge
> walk-forward on one season ≈ 17 min on the reference machine.
> Ceiling: 68 min per season = ~2 h 16 m combined. If a season
> doesn't complete in that budget, the ensemble is not a practical
> production default regardless of accuracy — kill and revisit.

12.8 hours × 2 parallel processes without either finishing is far
past the 68 min / season ceiling. The gate fires mechanically.

### Other kill gates

- **#1 (hindsight ≤ 66.28 %):** can't be evaluated — no
  `decision_quality` block produced.
- **#2 (any \|bias\| > 10 %):** can't be evaluated — no
  `_predictions.csv` produced.

Neither of the other gates can fire, but the pre-registration only
requires ANY single gate to fire for a kill. #3 suffices, and the
"no narrative rescue" clause in the pre-registration explicitly
forbids extracting partial data and cherry-picking weeks.

## Incidental findings retained for follow-up

### PositionModel's internal feature engineering is not using the Vegas cache

The partial ensemble logs showed this message during 2025's
walk-forward:

```
Vegas features defaulted on 58% of rows (>50% threshold).
implied_team_total/spread are effectively dead inputs for this
training set.  Verify nfl_data_py.import_schedules works in this
environment.
```

On a season with a fully populated Vegas cache. The Phase 1 Step C
wiring added a DB-cache-first path to
`src/features/feature_engineering.py::_create_vegas_game_script_features`,
but `PositionModel` (used inside `_EnsembleModelWrapper`) runs its
own feature-engineering pipeline at train time that bypasses that
path and calls `nfl_data_py.import_schedules()` directly —
`habitatring.com`, blocked in this sandbox, same Phase 1 silent
fallback we fixed a month ago, re-surfacing in a parallel code path.

Practical implication: **any ensemble training in this environment
sees Vegas as dead inputs regardless of cache state.** If a future
session re-runs Phase 4 on a machine where nfl_data_py can reach
its upstream, this disconnect is invisible. In the sandbox, it
silently degrades the ensemble's effective feature set. Fixing it
would require wiring PositionModel's feature pipeline to the same
DB cache the walk-forward feature code uses — flagged as a
residual-bug workstream, not in Phase 4's scope. See
`docs/SILENT_FALLBACK_AUDIT_20260422.md`'s "legacy path" flag list;
this is a new one to add to that list.

### The `is_divisional`/`is_primetime` audit fix fires as designed

Partial logs showed the structured warning from Step 2's audit fix
firing hundreds of times across per-fold feature prep. Confirms the
fix in `src/features/feature_engineering.py::_add_team_matchup_features`
is live and observable, even if both target columns are still
absent from CAUSAL_FEATURES.

## What does not change

- **Ridge α=10 000 is the production default.** `RIDGE_DEFAULT_ALPHA = 10_000`
  in `config/settings.py` stays. Phase 1 Vegas + Phase 3 injury
  merges stay live.
- **The decision-quality framework stays.** 69.77 % cross-season
  hindsight (30-13, p = 0.007, ROI +25.6 %) remains the documented
  production baseline.
- **The 2026-04-22 re-council's paper-trade protocol stays.**
  Frozen features now mean frozen *Ridge-based* features. Ensemble
  is out of scope for the paper-trade run.

## What comes next

Per the pre-registration's mechanical verdict and the re-council's
sequence, Phase 4 is closed. Remaining Phase 5 (weather /
game-script) was already deferred by the re-council. The
predictive-ceiling workstream as a whole is complete.

Open residual items — flagged, not scheduled here:

1. **PositionModel Vegas-cache disconnect.** Same silent-fallback
   pattern as Phase 1. Fix would enable a sandbox-reproducible
   ensemble result if the user ever wants to re-run Phase 4 on a
   smaller feature set or different hardware.
2. **Ensemble-on-faster-hardware re-run.** The kill was on runtime,
   not on measured accuracy. If a future session has a ≥10× faster
   machine (or runs the ensemble without the 5-fold OOF stacking
   and hyperparameter tuning disabled, which is already the case),
   a proper Phase 4 measurement remains open. The rule of record
   still applies — same kill gates — and the pre-registration is
   already committed.
3. **Fully-symmetric prospective replay.** Separately documented
   residual from Step 3; unchanged by this finding.

## Sequence status after Phase 4

| Step | Status | Artifact |
| :--: | :--- | :--- |
| Chairman 1 (bootstrap) | ✓ pass | `docs/BOOTSTRAP_H2H_20260422.md` |
| Chairman 2 (audit) | ✓ done | `docs/SILENT_FALLBACK_AUDIT_20260422.md` |
| Chairman 3 (prospective replay) | ✓ pass | `docs/PROSPECTIVE_REPLAY_20260422.md` |
| Chairman 4 (pre-reg) | ✓ done | `docs/PHASE_4_PREREGISTRATION.md` |
| **Phase 4 run** | ✗ **KILL on runtime** | this doc |
| Chairman 5 (paper-trade) | ✓ done at spec level | `docs/PAPER_TRADE_PROTOCOL_20260422.md` |

The 2026-04-22 re-council is fully discharged. No open council items.
