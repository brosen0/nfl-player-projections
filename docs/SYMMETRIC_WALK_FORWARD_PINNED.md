# Wide-mode symmetric walk-forward: pinned — KILL gate fires

**Provenance:** Step 1 of the chairman's verdict in
[`council-transcript-20260423-051434.md`](../council-transcript-20260423-051434.md)
(2026-04-23 Bucket 4 council on draft + user-facing readiness):

> **DO TODAY:** Finish the symmetric walk-forward re-run to collapse
> the [55.8 %, 83.7 %] prospective-replay bound to a point estimate
> with honest CI. Time: 1–2 days. Success signal: a single win-rate
> number with bootstrap CI committed to the repo; **if p5 < 52.38 %,
> everything downstream halts.**

## Verdict — KILL

| Metric | Value |
| :--- | :---: |
| Wide-mode symmetric record | **23-18** |
| Win rate | **56.10 %** |
| Wilson 95 % CI | [41.04 %, 70.11 %] |
| **Bootstrap 5th percentile (gate)** | **43.90 %** |
| -110 break-even (kill threshold) | 52.38 % |
| P(bootstrap resample > break-even) | 69.11 % |
| **Gate status** | ✗ **FAIL by 8.48 pp** |

Per the chairman's locked rule, **everything downstream halts**. No
user-facing draft tool. No user-facing start/sit tool. Model-level
diagnosis before any product work.

## What landed

- `scripts/run_ts_backtest.py --emit-inactive-predictions`: new
  CLI flag. When on, the walk-forward injects phantom test rows for
  every player in the season's cumulative-active pool (through week
  N-1) who didn't play week N. Phantoms carry last-known player-
  level features with this-week team / opponent / Vegas / injury
  context.  They get predictions (`is_active=0`, `actual=NaN`).
- `scripts/wide_symmetric_replay.py`: scores symmetric win rate
  from a wide CSV. Model picks top-N-per-position from the full
  pool by this-week predicted; picks who end up inactive score 0.
- Artifacts:
  - `data/backtest_results/ts_backtest_2024_20260423_055841_*`
    (10,685 predictions; 5,480 active + 5,205 phantom)
  - `data/backtest_results/ts_backtest_2025_20260423_055829_*`
    (10,586 predictions; 5,595 active + 4,991 phantom)

## Decomposition of the drop

The total 69.77 % → 56.10 % drop (−13.67 pp) has two components:

| Step | Hindsight win rate | Δ | Source |
| :--- | :---: | :---: | :--- |
| Narrow active-only (baseline) | 69.77 % | — | production Ridge α=10 000, narrow test set |
| **Wide-mode active-only** | **65.85 %** | **−3.92 pp** | **feature-engineering perturbation** (phantoms in the combined block shift expanding-mean / group-wise features for active rows) |
| **Wide-mode symmetric** | **56.10 %** | **−9.75 pp** | **real symmetric construction** (model drafts from the wide pool, 0-scores inactives) |

About **3.9 pp of the drop is measurement artifact**; the remaining
**9.8 pp is the honest symmetric effect**. Even deducting the
artifact cleanly (hypothetical "clean symmetric" ≈ 60 %), Wilson
lower bound would still sit near 44–46 %, well below break-even.
The kill gate fires under any reasonable deduction.

## Structural finding (real, not a proxy artifact)

With fresh per-week predictions for every cumulative-active player,
the model picks inactives on **22.4 %** of slots vs the opponent's
**15.0 %**. That 7.4 pp gap was flagged as a candidate artifact in
`docs/SYMMETRIC_PROSPECTIVE_REPLAY.md`; the wide-mode run confirms
it's real and structural — not an artifact of stale predictions.

The model's top predictions concentrate on players who disproportionately sit. Candidate explanations (not tested here):

- **Load-management selection:** premium stars are both top-predicted
  and more likely to rest.
- **Feature-driven injury proxy:** high-snap-share / high-usage
  features both predict high FP and correlate with injury risk. The
  model picks "most likely to score big IF they play," not "most
  likely to play."
- **Survivorship in training:** per-week training data
  disproportionately includes players who actually played. Rising
  stars who then sit a week aren't represented at current-week
  features; their predictions are biased upward.

This is the real-world "ceiling-chaser bias" a lineup-selection
system needs to correct before shipping to users. Correcting it is
not a few-hour fix; it's a separate workstream.

## Caveats

- **Wide-mode active-only predictions are measurably different from
  narrow-mode** (−3.92 pp on the same active-only hindsight number).
  The phantom injection perturbs feature engineering for active
  rows too. A fully clean symmetric replay would require either (a)
  feature engineering that isolates phantoms from active rows, or
  (b) saving the narrow-mode per-week models + scalers and running
  a second inference pass on the phantom pool. Both are bigger
  changes than today's budget.
- **The wide-mode symmetric 56.10 % is still the best available
  measurement** because it uses current-week features and fitted
  per-week models for the inactive pool — no stale-prediction
  proxy.
- **43 weeks is small.** Wilson CI is [41.04 %, 70.11 %]. The true
  symmetric win rate could genuinely be above break-even, but not
  at the kill gate's confidence threshold.

## Downstream halts mandated by the chairman's rule

1. **No user-facing draft tool.** The 2026-04-23 plan's step 2 (ADP
   scrape + snake-draft simulator) does not start until a symmetric
   win rate that clears 52.38 % is measured.
2. **No user-facing start/sit tool.** Same gate.
3. **No paper-trade go-live.** The 2026-04-22 paper-trade protocol's
   stop-loss gate #1 references a "running Wilson LB < 52.38 %"
   condition; the measured symmetric wide-mode baseline already
   fails that condition in backtest, so a 2026 paper-trade would
   start under a fired stop-loss.

## What to diagnose before unblocking

The 7.4 pp model-vs-opponent inactive-pick gap is the single
biggest lever to investigate:

1. **Inactive-pick rate by prediction tier.** Do the top-3 predicted
   players per position sit more than the 4th-8th? If yes, the
   effect is concentrated in the stars the model leans hardest on —
   a load-management story.
2. **Drop-out-a-player ablation.** If the model had access to a
   "player active this week" binary (which in production it would
   via injury reports at lock time), does win rate recover? If yes,
   the current pipeline's injury merge is insufficient.
3. **Position-by-position breakdown.** Is the gap worse at RB/WR
   (rest candidates) than QB (rarely sits)? That would confirm the
   load-management story.

None of these are draft-specific. They're model-level.

## What does not change

- Ridge α=10 000 with Phase 1 Vegas + Phase 3 injury merges remains
  the production default.
- Narrow active-only cross-season hindsight 30-13 = 69.77 % (p =
  0.007, ROI +25.57 %) remains the documented narrow-baseline
  number. It is not wrong — it is "the model's win rate against an
  opponent who also got a retrospective filter." It over-states
  the live deployable edge.

## Next (per chairman's verdict, step 2 gated on this)

Step 2 was "scrape ADP + stand up snake-draft simulator." That step
is **GATED** and does not begin until either (a) the inactive-pick
gap is closed and wide-mode symmetric clears break-even, or (b) the
chairman re-councils on the kill gate.

## Artifact summary

| File | Purpose |
| :--- | :--- |
| `src/evaluation/ts_backtester.py` | `emit_inactive_predictions` flag + `_build_phantom_test_rows` + NaN-safe `_phantom` column handling |
| `scripts/run_ts_backtest.py` | `--emit-inactive-predictions` CLI |
| `scripts/wide_symmetric_replay.py` | scoring over the wide CSV |
| `data/backtest_results/ts_backtest_{2024,2025}_20260423_*_*` | artifacts for this measurement |

Run:
```
python scripts/wide_symmetric_replay.py \
    --runs data/backtest_results/ts_backtest_2024_20260423_055841.json \
           data/backtest_results/ts_backtest_2025_20260423_055829.json
```
