# Fully-symmetric prospective replay — 2026-04-22

**Provenance:** residual from re-council Step 3
(`docs/PROSPECTIVE_REPLAY_20260422.md`). Closes the one-sided gap
flagged in that doc's "residual concern" section — moves the model
(not just the opponent) to the cumulative-active pool and
0-penalizes inactive picks on both sides.

**TL;DR — the true symmetric number is bounded, not pinned. The
best-available approximation fails the Wilson-LB kill gate, but the
approximation is known to be biased against the model. A definitive
answer still requires a walk-forward re-run with predictions
generated for every cumulative-active player each week — not done
here. In the meantime the finding surfaces a second-order risk
worth investigating: even under a tight same-filter construction,
the model's top picks go inactive at a measurably higher rate than
the opponent's.**

---

## The three constructions

The earlier work established two points on the spectrum. This
replay adds a third:

| Construction | Opp pool | Model pool | Inactive-pick rule | Result |
| :----------- | :------: | :--------: | :----------------: | :----: |
| Hindsight (live backtest) | active-only | active-only | both dodge | **30-13 = 69.77 %** |
| Step 3 asymmetric | cumulative-active, 0-score | active-only | opp only | **36-7 = 83.72 %** |
| **Symmetric (this doc, best-effort proxy)** | cumulative-active, 0-score | cumulative-active, 0-score | both | **24-19 to 25-18 = 55.8 %–58.1 %** |

Under three freshness-window variants (how far back the model's
stale prediction can reach into the season):

| `--model-freshness` | Sym win rate | Wilson 95 % CI | Avg model drop | Model inactive picks |
| :---: | :---: | :---: | :---: | :---: |
| 0 (full cumulative history) | 24-19 = **55.81 %** | [41.11 %, 69.57 %] | −31.4 FP/wk | 68/258 = **26.4 %** |
| 1 (strict prior-week; matches opp filter) | 25-18 = **58.14 %** | [43.33 %, 71.62 %] | −23.9 FP/wk | 48/258 = **18.6 %** |
| 3 (last three weeks) | 24-19 = 55.81 % | [41.11 %, 69.57 %] | −28.8 FP/wk | 59/258 = 22.9 % |

Opponent's inactive-pick rate across all runs: 37/258 = **14.3 %**
(fixed by the Step 3 construction).

## Why all three fail the Wilson-LB kill gate

The chairman's Step 3 kill criterion: Wilson 95 % lower bound
< 52.38 % ⇒ halt forward deployment. None of the three symmetric
variants clears it:

- Full cumulative: LB 41.11 %. **FAIL.**
- Strict prior-week: LB 43.33 %. **FAIL.**
- Last three weeks: LB 41.11 %. **FAIL.**

Even the tightest symmetric construction (freshness=1, same filter
as the opponent's prior-week requirement) produces a 43.33 % lower
bound, ~9 points below break-even.

## Two reasons this is the wrong number to ship on

### 1. The proxy is structurally biased against the model

To give the model a prediction for every cumulative-active player,
this script uses each player's **last-known prediction** from the
walk-forward CSV. That prediction was generated when the player
last played. For a player who's sat 2 weeks, the "prediction" is 2
weeks stale and was computed on their then-available features.

A production pipeline would never operate this way. It would
regenerate a prediction every week using current-week features,
which typically reduce a sat-out player's predicted value (recency
features, snap share trailing-3w, etc. all decay toward 0 for
inactive players). The production pipeline's updated-each-week
prediction is what we actually care about; our last-known-value
proxy is strictly staler and picks inactives more often than a
production model would.

Concretely: at freshness=0 the model picks inactives on 26.4 % of
slots vs the opponent's 14.3 %. At freshness=1 (tight filter
matching the opponent), the gap shrinks to 18.6 % vs 14.3 % — but
doesn't close. That leftover 4 pp gap could be real (next section)
or still proxy artifact; we can't disentangle with this data.

### 2. Even under the tight filter, the model picks inactives more than the opponent does

Under freshness=1 both sides draw from exactly the same filter —
"played last week." The opponent ranks by last-week actual FP; the
model ranks by last-week predicted FP. Same temporal window, same
pool membership.

The opponent's picks go inactive at 14.3 %. The model's go inactive
at **18.6 %**. That's a structural asymmetry the proxy alone
probably can't explain — over 43 weeks × 6 slots = 258 slot-weeks,
a 4 pp difference is ~10 extra inactive model picks vs what the
opponent's process would produce from the same pool.

Candidate explanations (not tested here):

- **Load-management selection.** The model's top predictions
  disproportionately concentrate on premium stars, who are more
  likely to sit than median starters.
- **Feature-driven injury proxy.** The model's features probably
  up-weight high-usage / high-snap-share players, and those
  correlate with injury risk. The model is picking "most likely to
  score big IF they play," not "most likely to play."
- **Backtest selection effects.** The walk-forward trains on
  cohorts that disproportionately include players who did play; the
  resulting predictions are upward-biased for anyone-who-might-play
  vs. the unconditional expectation.

Any of these would be a real production concern independent of the
proxy artifact. A lineup selection process that goes inactive 4 pp
more often than the natural baseline leaves real FP on the table.

## What this doesn't close

The original question — *what is the model's true symmetric
prospective win rate?* — remains open, bounded:

```
  lower bound (this session's approximation):     55.8 %
  upper bound (Step 3 asymmetric replay):         83.7 %
  best single-season hindsight (in-sample):       69.8 %
```

The true symmetric number probably sits above 55.8 % because the
proxy overpenalizes staleness, and below 83.7 % because structural
model-picks-inactives dynamics appear to exist independent of the
proxy. Where exactly is not this session's question to close.

**To close it definitively** requires a walk-forward re-run with a
widened test scope: each week, the backtester produces predictions
for every cumulative-active player (not just players who happened
to play that week). Runtime would be roughly 2× the current
walk-forward because the test set is ~2× larger. That's a session
with dedicated time budget, not a drop-in change.

## Implication for the paper-trade protocol

Two adjustments worth considering before 2026 week 1 goes live
(`docs/PAPER_TRADE_PROTOCOL_20260422.md`):

1. **The production pipeline MUST predict for every rostered player
   each week, not just players with historical records.** If
   `PositionModel.predict` ever returns NaN for an inactive-looking
   player and the lineup selection silently skips them, we re-inherit
   the asymmetric filter the backtest used. Monitor the prediction
   coverage as part of the lock-time check.
2. **Add a "lineup inactive-pick rate" gate to the stop-loss.** If
   the live model hits > 20 % inactive picks for two consecutive
   weeks, halt and investigate — that's the structural-bias failure
   mode this replay surfaced.

Both are addition-only changes to the protocol; no code built today.

## Residual open items

- **Full walk-forward re-run with wider test scope.** Closes the
  [55.8, 83.7] band. ~2 h compute + some code to widen the test
  set. Not blocking anything today.
- **Structural-bias investigation.** Why does the model pick
  inactives 4 pp more often than opponent under same-filter? Worth
  a dedicated session if paper-trade deployment becomes imminent.
  Possible angles: prediction vs roster-active rate by tier; bias
  attribution by feature category.

## Artifact

- Script: `scripts/symmetric_prospective_replay.py` (stdlib + sqlite3)
- Reference runs: `data/backtest_results/ts_backtest_2024_20260422_025527_*`,
  `ts_backtest_2025_20260422_024024_*` (the production-config
  α=10 000 post-Vegas post-injury walk-forwards)

Run:
```
python scripts/symmetric_prospective_replay.py --model-freshness 1
```

## Chairman's sequence — no status change

This session closes a residual flagged in the findings doc but
doesn't flip a ship/kill decision. Paper-trade protocol stays the
document of record; its "frozen features" list and stop-loss gates
are unchanged. The structural bias observation is a new candidate
stop-loss condition and is documented above for later
consideration.
