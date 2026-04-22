# Phase 4 pre-registration — 2026-04-22

**Provenance:** Step 4 of the chairman's verdict in
[`council-transcript-20260422-032550.md`](../council-transcript-20260422-032550.md)
(Bucket 4 re-council). Chairman's success signal: "signed-off rule
committed to the repo prior to running any ensemble code."

This document locks in what counts as success, failure, and
"ambiguous re-council" for Phase 4 (ensemble re-evaluation) BEFORE
any ensemble walk-forward is executed. Rules committed here are the
rules the eventual result is judged against. No post-hoc threshold
shifts; no cherry-picking seasons; no narrative rescue of a failing
run.

---

## The run that will be judged

One command, defined up-front:

```
python scripts/run_ts_backtest.py --model ensemble --season 2024
python scripts/run_ts_backtest.py --model ensemble --season 2025
```

- Feature set: the current production `CAUSAL_FEATURES` list
  (unchanged from this commit). Vegas + injury merges fire; opp-defense
  s2d stays reverted. `RIDGE_DEFAULT_ALPHA` is not applicable to the
  ensemble branch — `ensemble_model_factory` in
  `src/evaluation/ts_backtester.py:918` ignores it.
- Model: `ensemble_model_factory` at `src/evaluation/ts_backtester.py:918`
  — the full XGBoost + LightGBM + RF + Ridge stack with Huber loss,
  OOF stacking, isotonic calibration. Hyperparameter tuning disabled
  for speed (existing default).
- Decision-quality block: automatic per the wiring added in commit
  `7a65915` (Phase 1 of the decision-quality workstream).

Two walk-forward runs — one per season — pooled into a 43-week
cross-season record. No additional seasons, no additional α values,
no additional feature surgery. Phase 4 tests **ensemble vs Ridge on
the same feature set**; that's it.

## Reference baseline

The Ridge baseline Phase 4 is compared against is **locked as of
commit `e5aa7ed` on main**:

| Metric (cross-season, 2024 + 2025, α=10 000) | Value |
| :------------------------------------------- | :----: |
| Hindsight win rate (primary)                 | 69.77 % (30-13) |
| Hindsight Wilson 95 % CI lower bound         | 54.89 % |
| Hindsight binomial p-value                   | 0.007 |
| Hindsight ROI (1.8× payout)                  | +25.57 % |
| Overall R² (production pipeline)             | ≈ 0.291 (avg of 0.314 and 0.268) |
| Max per-position bias (%) across both seasons | 9.22 % (WR, 2025, α=10 000) |

Source: `data/backtest_results/ts_backtest_2024_20260422_025527.json` +
`ts_backtest_2025_20260422_024024.json`.

## Primary success metric: cross-season hindsight win rate

Decision quality is the terminal metric (per the 2026-04-22
First-Principles-Thinker framing). The Ridge baseline is 30-13 =
69.77 %. Phase 4 ensemble will be compared against this number using
**both sides of a two-sided kill gate**.

### Continue / ship the ensemble as the new production model

All three must hold on the 43-week cross-season record:

1. **Hindsight win rate ≥ 72.09 %** (≥ 31-12). That's ≥ +2 pp over
   30-13. Rationale: +2 pp is the smallest win-rate delta the
   binomial test can distinguish from 69.8 % at p ≤ 0.05 on n=43. Any
   smaller delta is inside noise.
2. **Binomial p-value ≤ 0.01** on the combined 43-week record.
   (Ridge baseline is already p = 0.007; the ensemble must maintain or
   improve that significance level.)
3. **Every per-position `bias_pct` within ±10 %** per both seasons'
   runs, per `tests/test_backtest_validation.py::TestWalkForwardBiasRegression`.
   No carve-outs. This test runs on the ensemble predictions exactly
   as it does on Ridge.

### Kill / revert to Ridge

Any single condition fires:

1. **Hindsight win rate ≤ 66.28 %** (≤ 28-15). That's ≥ −3 pp vs Ridge
   baseline. A drop this size against a more complex model says the
   ensemble is making worse picks, not better.
2. **Any per-position bias exceeds ±10 %** in either season. Same gate
   as Phase 1 / Phase 3 used. The Huber loss in the ensemble has
   historically flattened tails and pushed mean predictions up; a bias
   regression is the expected failure mode and it's non-negotiable.
3. **Walk-forward runtime exceeds 4× Ridge runtime.** Ridge walk-forward
   on one season ≈ 17 min on the reference machine. Ceiling: 68 min
   per season = ~2 h 16 m combined. If a season doesn't complete in
   that budget, the ensemble is not a practical production default
   regardless of accuracy — kill and revisit.

### Ambiguous → re-council

Win rate lands in `(66.28 %, 72.09 %)` with no bias/runtime failure:
neither a clear ship nor a clear kill. This is the band where the
binomial test cannot distinguish ensemble from Ridge at 43 weeks. In
that case:

- Do not promote the ensemble.
- Do not kill future ensemble work.
- Ship a Bucket 4 re-council with the full ensemble numbers, and name
  the follow-up experiment (typically: more seasons, 2022 + 2023 added
  to the matrix).

## What **does not** count for Phase 4

Explicit "do not move the goalposts" list:

- **R² alone.** R² is the proxy the re-council demoted (see the First
  Principles Thinker's response in `council-transcript-20260422-032550.md`
  and the re-label in `docs/SILENT_FALLBACK_AUDIT_20260422.md`). If
  the ensemble lifts R² by +0.05 but holds win rate flat, that is
  **not** a ship signal. Report it, note it, keep Ridge.
- **Oracle / Replacement tier comparisons.** Those are sanity checks.
  The primary kill gate is vs hindsight.
- **2024-only or 2025-only results.** Cross-season 43-week is the
  decision surface. Single-season numbers are context in the
  writeup, not a basis for shipping.
- **Sub-tier filters.** "The ensemble does better on QB-heavy weeks"
  style slicing is banned as a ship justification. If someone
  proposes it post-result, it's a new experiment, not a Phase 4 win.

## Execution checklist (lands with the ship decision)

Before claiming Phase 4 PASS:

1. Both walk-forward runs committed to `data/backtest_results/`.
2. `pytest tests/test_backtest_validation.py::TestWalkForwardBiasRegression`
   runs against the ensemble predictions and passes.
3. `scripts/bootstrap_h2h_record.py --sources <two ensemble JSONs>` runs
   with p5 ≥ 52.4 % (same bootstrap gate as the Ridge baseline cleared).
4. `scripts/prospective_opponent_replay.py --runs <two ensemble JSONs>`
   runs; prospective win rate directionally matches the Ridge direction
   (prospective ≥ hindsight, i.e., retrospective filter was helping
   the opponent on the ensemble too).
5. Before ship: one-page writeup in `docs/PHASE_4_ENSEMBLE_FINDINGS.md`
   citing each of the metrics above, with the delta-vs-baseline table
   and the kill-gate verdict explicit.

Note: writing Phase 4 passes does NOT also mean running Step 5
(forward paper-trade). Step 5 is separate — a Phase 4 ship signal
unlocks it, but does not complete it.

## One explicit edge-case pre-commitment

If the ensemble ships but the per-season split shows one season
above +2 pp vs Ridge and one below −3 pp (e.g., +5 pp on 2025, −4 pp
on 2024, combined +0.7 pp = ambiguous), the combined-43-week rule is
final: **ambiguous, re-council.** Do not ship on "the good season,"
do not revert on "the bad season." 43 weeks is the decision surface.

---

*Signed-off by the chairman of the 2026-04-22 re-council via the
committee verdict in `council-transcript-20260422-032550.md`.
Committed to repo before any ensemble code runs per the chairman's
success signal.*
