# Phase 4B — VORP + name-matcher fix — findings

Implements Phase 4B of the Step 4 plan.  Added a `vorp` ranking
mode to `snake_draft_sim.py` that subtracts position-replacement
level from the projection before ranking.  While validating, found
a name-match bug in the draft-sim's predictions-index that was
distorting every prior draft-sim result — fixed in the same change.

## Before the matcher fix (suspect numbers)

Initial VORP vs season_sum on post-rookie-prior CSVs (2024:
`ts_backtest_2024_20260424_165248`, 2025: `...165230`):

| Season | season_sum actual | vorp actual | Δ |
|---|---|---|---|
| 2024 | 2220.0 | 2431.4 | +9.5 % |
| 2025 | 2016.7 | 2195.4 | +8.9 % |

VORP appeared to uniformly help.  But the season_sum rosters were
telling: 2024 ModelBot "starters" included Keilan Robinson, James
Robinson, Snoop Conner — UDFA/fringe RBs who never played material
NFL snaps.  No real model would produce that.

Root cause: `_build_pred_index` used `(normalized_last_name,
position)` as the join key with `setdefault`.  When two players
share a last name at the same position (Bijan Robinson + Keilan
Robinson, Jonnu Smith + Ainias Smith, CeeDee Lamb + nothing — OK
for unique names), only the FIRST one seen in the predictions CSV
was indexed.  CSV order put Keilan Robinson before Bijan; ADP's
"B.Robinson" matched Keilan's prediction.  Wrong player, wrong
prediction, wrong draft result — on every mode, every season.

## Matcher fix

`_build_pred_index` now keys on `(first_initial, normalized_last,
position)`.  When two records collide at the same key (truly same
initial + last + position, rare), pick the record with more weeks
played (proxy for "more established player"), breaking ties by
higher `pred_total`.

Mirror change in `build_draft_board` — the lookup uses the same
three-key pattern.

## After fixes (real numbers)

4-mode sweep on the same post-rookie-prior CSVs:

| Mode | 2024 ModelBot | 2024 vs ADP | 2025 ModelBot | 2025 vs ADP |
|---|---|---|---|---|
| `season_sum` (hindsight) | 1808.7 (#2/12) | **+13.8 %** | 2016.7 (#1/12) | **+47.3 %** |
| `vorp` (hindsight) | 2019.8 (#1/12) | **+31.5 %** | 1935.5 (#1/12) | **+37.1 %** |
| `week1` (pre-draft model) | 1476.6 (#10/12) | **−9.0 %** | 1277.7 (#10/12) | **−15.2 %** |
| `prior_season` (naive PY-actuals) | 1614.2 (#7/12) | −2.8 % | 1261.3 (#11/12) | −16.2 % |

## What this actually says

1. **The hindsight model legitimately beats ADP** (+13.8 % 2024,
   +47.3 % 2025 on `season_sum`; +31.5 % / +37.1 % on `vorp`).
   The walk-forward Ridge IS capturing real signal when it has
   full in-season context.
2. **VORP lift is real but mixed:**
   - 2024: season_sum 1808.7 → vorp 2019.8 = **+11.7 %** lift
   - 2025: season_sum 2016.7 → vorp 1935.5 = **−4.0 %** regression
   - Cross-season mean: **+3.9 %**.  Not a clear headline win on
     N=2 deterministic draws; the scarcity-based ranking is
     noise-sensitive.  Step 3's N ≥ 200 randomized sweep is where
     this gate actually settles.
3. **Pre-draft modes LOSE to ADP on both seasons.**  This is the
   honest answer to "can the current model help a user's August
   draft?" — **no, not yet**.
   - `week1`: the walk-forward W1 prediction is built from pre-
     season features only.  Our feature set knows prior-season
     stats, roster shares, and historical opponent defense.  It
     does NOT know: 2024-offseason trades, training camp injuries,
     depth chart moves, NFL Draft results (for rookie priors the
     prior gives us +1 PPG, not enough), beat reporter signal.
     ADP consensus bakes all of this in.
   - `prior_season`: same limitation, even more extreme — it's
     "last year's points" with zero model info.
4. **Phase 4C rookie priors helped rookies modestly** (+0.011 R²
   each season) but did NOT change the draft-sim rank.  Confirmed
   in the `week1` result above — 2025 `week1` still ranks #10/12.

## Success-gate reading

Phase 4B plan gate: "VORP-ranked ModelBot produces better
`actual_starter_total` than `season_sum` ModelBot on 2024 + 2025
(single deterministic draw)."

Gate: **MIXED** — passes on 2024 (+11.7 %), fails on 2025 (−4.0 %).
Cross-season mean is +3.9 %, positive but inside draft-order
noise bounds.  This is exactly the case the plan flagged: single
deterministic draws don't settle VORP — N ≥ 200 (Step 3) does.

## What this does NOT mean

- It does NOT mean the model is useless for drafts.  It means our
  current PRE-DRAFT features are inferior to ADP for ranking.  Our
  HINDSIGHT model (season_sum) is clearly strong.  The gap between
  hindsight and pre-draft says exactly what Step 4 is supposed to
  close: calibration + uncertainty + real pre-draft signals.
- It does NOT invalidate Phase 4C.  Rookies still get a real +1 PPG
  lift in their W1/W2 predictions; the rookie-only R² improvement
  is real.  What 4C doesn't do is overcome the structural pre-
  draft information gap.
- It does NOT change the symmetric walk-forward kill-gate.  The
  75.6 % / p5=65.85 % number is a WEEKLY H2H number (in-season
  start/sit), not a draft number.

## What this changes for the council's Step 3

Step 3 spec: "Run the draft-sim-vs-ADP hindsight backtest on 2024
(and 2025 where data permits). Success signal: your bot beats
ADP-bot by a statistically meaningful margin on realized season
totals across N ≥ 200 simulated league configurations; if it loses
or ties, kill the draft product and redirect those months to
start/sit only."

The matcher-fixed N=1 draws are now:
- `season_sum`: hindsight, clearly beats ADP both seasons.  Not the
  right gate — hindsight is cheating.
- `week1`: genuine pre-draft, LOSES to ADP both seasons.
- `vorp`: hindsight + scarcity, beats ADP both seasons but mixed
  vs season_sum on the VORP-specific lift.

**The direct pre-draft signal (`week1`) already loses to ADP on
two seasons.**  This is the honest directional answer to the
chairman's kill criterion.  Step 3's N ≥ 200 randomization isn't
going to flip a 2-for-2 loss into a 200-for-200 win — the model is
structurally under-informed vs. ADP for draft-time ranking.

Per the council's kill criterion: **"if it loses or ties, kill the
draft product and redirect those months to start/sit only."**

We may be there.  Specifically the August-2026 draft product as
described isn't supportable on current features.  The start/sit
product (weekly H2H, 75.6 % win rate post-audit) is supportable.

## Recommendation

Pause Phase 4 before the remaining 4A (conformal intervals) and 4D
(injury variance) work, and have a conversation about whether:

(a) The draft-product workstream should continue with Step 3's N ≥
    200 + more real pre-draft features (offseason news, camp
    signals, depth chart) — a months-long data-pipeline effort.
(b) Refocus on the start/sit product where the 75.6 % number is
    defensible, and drop the August 2026 draft feature per the
    chairman's kill criterion.

The honest version of this audit is that the ADP gate has already
spoken on the current feature set.  Further model work without new
pre-draft data sources is unlikely to change it.

## Files touched

- `scripts/snake_draft_sim.py` — added `REPLACEMENT_RANKS`,
  `_apply_vorp`, `_first_initial`, `--ranking vorp` CLI option,
  rewrote `_build_pred_index` to use 3-key matching, updated
  `build_draft_board` lookup.
- `tests/test_snake_draft_sim.py` — 2 new VORP tests; existing 7
  still pass → 9/9 green.

Within CLAUDE.md's 5-source-file budget.

## Artifacts

Post-fix JSON in `data/draft_sim_results/postfix/`:
- `draft_2024_slot6_{season_sum,vorp,week1,prior_season}_v2.json`
- `draft_2025_slot6_{season_sum,vorp,week1,prior_season}_v2.json`

The `_v2` suffix distinguishes these from the buggy-matcher runs.
The older non-`_v2` JSON files in that directory should be treated
as historical (buggy) and not quoted for any lift claim.
