# Bootstrap of 43-week H2H record — 2026-04-22

**Provenance:** Step 1 of the chairman's verdict in
[`council-transcript-20260422-032550.md`](../council-transcript-20260422-032550.md)
(Bucket 4 re-council).

**Re-council kill criterion:** if the bootstrap 5th-percentile win rate
on the 43-week cross-season H2H record falls below **52.38 %** (the
-110 cash H2H break-even for a 1.8× payout), the sample is not yet a
deployable edge and Phase 4 ensemble work must wait.

## Result — PASS

| Statistic                        | Value       |
| :------------------------------- | :---------: |
| Observed record                  | 30-13 (69.77 %) |
| Wilson 95 % CI                   | [54.89 %, 81.40 %] |
| Break-even (H2H -110)            | 52.38 %     |
| **Bootstrap 5th percentile**     | **58.14 %** |
| Bootstrap median                 | 69.77 %     |
| Bootstrap 95th percentile        | 81.40 %     |
| P(bootstrap resample > 52.4 %)   | **99.19 %** |

The 5th-percentile bootstrap win rate (58.14 %) clears the -110
break-even (52.38 %) by 5.8 percentage points. **Kill criterion does
not fire.** The 43-week sample is statistically distinguishable from
a sub-break-even edge.

## Method

10 000 resamples of the 43-week sequence drawn with replacement from
the per-week `won_vs_hindsight` flags in the two production-config
walk-forward runs:

- 2024 α=10 000 post-Vegas + post-injury: 14-8 (22 weeks)
- 2025 α=10 000 post-Vegas + post-injury: 16-5 (21 weeks)

Seed = 42. Implementation: `scripts/bootstrap_h2h_record.py` (stdlib-only).

```
python scripts/bootstrap_h2h_record.py
```

## Caveats that still apply

- **This is still a hindsight-opponent backtest.** The "Hindsight" tier
  builds the opponent's lineup from the top-N-by-prior-week-actual per
  position — constructed retrospectively, with knowledge of which
  slate actually happened. The Contrarian peer reviewer flagged this
  in the re-council. The bootstrap p5 reports variance *within the
  hindsight framing*; it does not prove the edge survives a
  prospective-opponent construction. Step 3 of the chairman's verdict
  (lock-time-only replay) is the next gate for that claim.
- **43 weeks is two seasons.** The bootstrap respects that sample
  size, not the larger question of whether the 2024–2025 NFL scoring
  regime replicates in 2026. A market-regime shift isn't inside the
  bootstrap distribution.
- **Break-even is per-contest, not per-bet.** Real cash H2H on
  DraftKings/FanDuel costs rake (~10–15 %) which the 1.8× payout
  already bakes in; break-even at 1.8× payout is
  `1.0 / 1.8 ≈ 55.56 %`, not 52.38 %.  The re-council chairman used
  52.38 % as the **sportsbook -110 line** standard, which is slightly
  easier to clear than the cash DFS 1.8× standard. At the 1.8× cash
  threshold, p5 = 58.14 % clears by only 2.6 pp, not 5.8 pp — still
  passing but thinner.

## Next step in the chairman's sequence

1. ~~Bootstrap the 43-week record.~~ ✓ done.
2. **Audit every `except Exception: pass` in the causal feature
   path; re-label Phase 1's "+0.004 R²" as a bug-fix in the workstream
   record.** ← next.
3. Prospective replay against lock-time-only opponent construction.
4. Pre-register Phase 4 decision rule.
5. Forward paper-trade harness, 8-12 weeks.
