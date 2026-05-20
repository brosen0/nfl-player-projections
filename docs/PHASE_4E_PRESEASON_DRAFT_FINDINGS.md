# Phase 4E — preseason season-total vs matchup-sum draft comparison

Audit date: 2026-05-18

This closes the gap left by Phase 4B. The repo already had two draft-sim views:

- `season_sum`: sum of weekly walk-forward predictions across the season
- `week1`: a genuine pre-draft proxy using only the week-1 walk-forward prediction

What was missing was a direct historical comparison between:

- the matchup-aware weekly-sum path (`season_sum`)
- a causal preseason season-total model (`preseason_ml`)
- a simple per-game baseline (`preseason_ppg17`)

## What changed

- `scripts/snake_draft_sim.py`
  - Added historical preseason loading that can attach realized target-season totals.
  - Added explicit preseason projection modes:
    - `preseason_ml`: trained `PreseasonProjector`
    - `preseason_ppg17`: prior-season PPG x 17
- `scripts/draft_sim_mode_sweep.py`
  - Added both preseason modes to the sweep runner.
  - Normalized the summary/JSON naming to the actual 10-team simulator.
- `tests/test_snake_draft_sim.py`
  - Added coverage for attaching historical actual totals to preseason projections.

## Run

```bash
python scripts/draft_sim_mode_sweep.py \
  --seasons 2024 2025 \
  --modes season_sum preseason_ml preseason_ppg17
```

Artifact:

- `data/draft_sim_results/mode_sweep_preseason_compare_2024_2025.json`

## Results

10-team snake, 15 rounds, averaged across all 10 model draft slots.

| Season | Mode | Mean Rank | Lift vs ADP Mean | Beat ADP Mean |
|---|---:|---:|---:|---:|
| 2024 | `season_sum` | 1.1 | +399.6 | 10/10 |
| 2024 | `preseason_ml` | 5.3 | +7.2 | 3/10 |
| 2024 | `preseason_ppg17` | 9.1 | -238.3 | 0/10 |
| 2025 | `season_sum` | 1.0 | +453.6 | 10/10 |
| 2025 | `preseason_ml` | 9.4 | -284.0 | 0/10 |
| 2025 | `preseason_ppg17` | 9.2 | -367.5 | 0/10 |

## Interpretation

1. `season_sum` remains far stronger than either preseason season-total approach.
2. That does not make `season_sum` the right draft-time answer. It still benefits from within-season learning and is therefore a hindsight structural check, not an August-usable ranking.
3. `preseason_ml` is materially better than `preseason_ppg17`. It nearly breaks even in 2024 and is much less bad than the naive baseline in 2025.
4. `preseason_ml` still does not beat ADP reliably on this two-season check. The current causal season-total model is therefore not enough to justify the draft product on its own.

## Bottom line

If the question is "does the draft tool backtest better when we use individual team matchups by week or when we ignore that and predict season totals / per-game points?", the answer is:

- Backtest winner: matchup-aware weekly sum
- Honest pre-draft answer: the causal season-total model still loses to ADP overall

So the repo now supports the direct comparison, and the evidence still points to the same strategic conclusion as Phase 4B: the current draft product is strong only in hindsight mode, not in a clean pre-draft mode.
