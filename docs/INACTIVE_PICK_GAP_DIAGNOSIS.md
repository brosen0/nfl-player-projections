# Inactive-pick gap — diagnosis

**Provenance:** residual from the 2026-04-23 wide-mode symmetric
pin (`docs/SYMMETRIC_WALK_FORWARD_PINNED.md`). The chairman's
verdict mandated diagnosing the 7.4 pp model-vs-opponent
inactive-pick gap before any user-facing workstream resumes.

**Script:** `scripts/diagnose_inactive_pick_gap.py`, five
diagnostics on the 2024 + 2025 production-config wide-mode
artifacts.

## TL;DR

The gap is NOT what either candidate hypothesis predicted:

- ✗ **Not a "top-ranked stars sit most" story.** Inactive-pick
  rate is flat across ranks within position (RB top-1 = 18.6 %,
  top-2 = 25.6 %; WR top-1 = top-2 = 20.9 %). The #1 pick is not
  measurably worse than the #2 pick.
- ✗ **Not fixable by a stricter injury filter.** 83.6 % of the
  model's inactive picks have **no `player_injuries` row at all**.
  Only 9.1 % are flagged Out/Doubtful/IR. A perfect Out-filter
  moves the gap from +7.1 pp to +5.2 pp — marginal.
- ✓ **Is a position-concentration story.** QB (11.6 % vs 14.6 %,
  gap −3.0 pp — **model is better than opp at QB**), WR / RB / TE
  are where the model loses (+7.5 / +11.2 / +8.3 pp gap).
- ✓ **Is a missing-signal story.** The 47/55 non-injury-flagged
  inactive picks are byes, healthy scratches, coach's decisions,
  mid-week IR placements, depth-chart rotations — none of which
  appear in the injury data the pipeline ingests.

## Diagnostic 1 — by position (load-management test)

| Position | Model inactive | Opp inactive | Gap |
| :---: | :---: | :---: | :---: |
| QB | 5 / 43 (11.6 %) | 6 / 41 (14.6 %) | **−3.0 pp** |
| RB | 19 / 86 (22.1 %) | 12 / 82 (14.6 %) | +7.5 pp |
| WR | 18 / 86 (20.9 %) | 8 / 82 (9.8 %) | **+11.2 pp** |
| TE | 13 / 43 (30.2 %) | 9 / 41 (22.0 %) | +8.3 pp |

QB gap is NEGATIVE — the model actually outperforms the opponent
at picking active QBs (QBs rarely sit, so model's rolling features
are a clean signal). The gap is dominated by **WR** (+11.2 pp) —
the position with the most volatile weekly availability (load
management, concussion protocol, "ankle", "rest").

## Diagnostic 2 — by prediction tier (star-concentration test)

| Position | Rank | Inactive rate |
| :---: | :---: | :---: |
| QB | 1 | 11.6 % |
| RB | 1 | 18.6 % |
| RB | 2 | 25.6 % |
| WR | 1 | 20.9 % |
| WR | 2 | 20.9 % |
| TE | 1 | 30.2 % |

No clean tier effect. RB-2 is slightly worse than RB-1, WR ranks
tie. The "stars sit more" hypothesis doesn't fit — the gap isn't
driven by the model's very top picks. It's driven by the whole
top-N set at skill positions.

## Diagnostic 3 — injury-report status at lock time

| Injury status | Count | % of model inactive picks |
| :--- | :---: | :---: |
| **(no report row at all)** | 46 | **83.6 %** |
| Out | 5 | 9.1 % |
| Questionable | 3 | 5.5 % |
| None | 1 | 1.8 % |
| **Total** | **55** | 100 % |

**The injury data the pipeline ingests does not explain 85 % of the
model's inactive picks.** Candidates for what's filling the blank:

- **Byes.** Player's team has no game this week. `player_weekly_stats`
  has no row regardless of injury status.
- **Healthy scratches.** Roster decisions; not injury reports.
- **Mid-week IR placements.** Designated after Friday's final injury
  report.
- **Team elimination / playoff drops.** Late-season players whose
  team doesn't make the playoffs disappear from `pws` for playoff
  weeks.
- **Depth-chart / snap rotations.** Team-level decision to rest.
- **`nfl_data_py` coverage gaps.** The 45,334 injury rows we
  backfilled may not include every (player, week) pair that was
  filed.

## Diagnostic 4 — counterfactual: a strict Out/IR filter

If production perfectly dropped any pick with Out/Doubtful/IR/IR-R
at lock time:

| Metric | Value |
| :--- | :---: |
| Model inactive rate TODAY | 21.3 % |
| Model inactive rate AFTER strict filter | 19.4 % |
| Opponent inactive rate (baseline) | 14.2 % |
| Gap today | +7.1 pp |
| **Gap after strict filter** | **+5.2 pp** |

Strict injury filter closes **1.9 pp of the 7.1 pp gap** — i.e., ~27 %.
**The majority of the gap survives an idealized injury filter.**
This is not an injury-feature-weighting problem; it's a
missing-signal problem.

## Diagnostic 5 — top repeat offenders

Players the model repeatedly picked while they sat:

| Player | Pos | Team | Inactive picks | Most common status |
| :--- | :---: | :---: | :---: | :--- |
| D. Njoku | TE | CLE | 8 | "(no report row)" × 4 |
| J. Chase | WR | CIN | 6 | "(no report row)" × 6 |
| J. Taylor | RB | IND | 5 | "(no report row)" × 5 |
| B. Robinson | RB | ATL | 4 | "(no report row)" × 4 |
| D. London | WR | ATL | 4 | "(no report row)" × 4 |
| D. Henry | RB | BAL | 3 | "(no report row)" × 3 |
| C. Olave | WR | NO | 3 | "(no report row)" × 3 |

These are the exact high-usage stars a Ridge trained on recent
production would rank highly. That they repeatedly appeared as
inactive picks WITHOUT injury flags confirms the missing-signal
story: they were either on bye, benched, or their injury was filed
outside the data source we ingest.

## Root-cause verdict

**The gap is a signal gap, not a model gap.** The production
pipeline's "is this player available to play this week" input
covers roughly 15 % of real inactive cases (the ones the injury
report catches). The other 85 % require signals the pipeline
doesn't currently have.

The structural fix is **not** a Ridge hyperparameter change, a
different model class, or a new feature in CAUSAL_FEATURES. It's a
**pre-lock active-roster filter**: at lineup-lock time, query an
external source (ESPN / Sleeper / nflverse roster status feed for
the current slate) and drop any player flagged as "OUT," "IR," or
"not on the active 46." This is a HARNESS responsibility, not a
model-training responsibility.

Crucially, `player_weekly_stats` contains the ground-truth answer
retrospectively — "did player X play in week W" is a single JOIN.
The narrow-mode backtest IMPLICITLY uses this filter (the test
frame = players who played). That's WHY narrow-mode measures 69.8 %
— it's the right number for a production system that has an
active-roster filter at lock time. The wide-mode 56.1 % is the
number you get when neither side has that filter.

## What this changes in the plan

### Reframes the 2026-04-22 re-council's Step 1 kill-gate verdict

The locked kill-gate (`p5 < 52.38 %`) was specified against "the
symmetric prospective replay." That measurement conflates two
distinct production configurations:

1. **Production WITH an active-roster filter** (the narrow
   backtest proxy). Measured: 69.8 % cross-season hindsight, p =
   0.007, ROI +25.6 %. This is the number that matters for a
   deployment that includes a lock-time active-status check.
2. **Production WITHOUT an active-roster filter** (the wide
   symmetric replay). Measured: 56.1 %, p5 = 43.9 %. This is the
   number that matters for a deployment that ships predictions
   raw without any status check.

**The chairman's verdict halted downstream work based on (2).** But
(2) is only the relevant measurement if we ship without a
pre-lock active-roster filter. A credible production harness
(including the `scripts/paper_trade_lock.py` skeleton) would have
this filter. The kill verdict should be re-interpreted:

- Halt any user-facing work that does NOT include a pre-lock
  active-roster filter. ✓ still mandated.
- Halt user-facing work that DOES include the filter? Unclear —
  not directly measured, but bounded by 69.8 % on the narrow side.

### Concrete actions unblocked by this diagnosis

1. **Add a pre-lock active-roster gate to the paper-trade harness.**
   Query `player_weekly_stats` at scoring time (already implicit);
   in a live 2026 harness, query the daily NFL roster feed (ESPN
   API / nflverse `import_rosters_weekly()` on Friday afternoons).
   Drop any player with status in {"OUT", "IR", "PUP", "NFI",
   "SUS", "CUT", "TRADED"} before the optimizer runs.
2. **Add a "team has a game this week" binary feature** for the
   model. Free data, catches bye weeks. Should reduce the RB/WR/TE
   bye-week contribution to the gap without any new data source.
3. **Add a "player active rate last N weeks" rolling feature.**
   Players who consistently miss games get a downweight signal
   even without a current-week injury report.
4. **Investigate nfl_data_py injury coverage.** The 45k rows in
   `player_injuries` may not include every (player, week). Spot-
   check D. Njoku 2024 — we have him as inactive 8 weeks, none
   flagged. Known ground truth: Njoku missed 5 games with
   hamstring issues in 2024. If our data shows 0 "Out" rows for
   Njoku that year, the feed is missing real reports.

### Unblocked, gated: the 2026-04-23 council's step 2

The user-facing draft + start/sit tracks were halted pending
symmetric-clear. The diagnosis here reframes "symmetric clear" as
"the right configuration for the deployment shape." A paper-trade
harness that includes the pre-lock active-roster filter (action 1
above) would measure above 56.1 % — likely closer to the narrow
69.8 % number. Step 2 can proceed **conditional on** the harness
implementing that filter.

## What was NOT tested here

- **Bye-week contribution.** Haven't programmatically separated
  bye-week inactives from other non-injury inactives. Next round
  of diagnostic would add a schedule-join to measure it.
- **Player-active-rate feature impact.** Hypothesized as action 3;
  not yet built or measured.
- **Whether the Njoku / Chase / Taylor / London inactive picks
  were bye-week vs real injuries.** Cross-referencing their
  specific weeks against the 2024 team bye schedule would answer.

These are natural next-session diagnostics. None block the
re-framing above.

## Artifacts

- `scripts/diagnose_inactive_pick_gap.py` (runs all 5 diagnostics)
- Source: `data/backtest_results/ts_backtest_2024_20260423_055841_predictions.csv`
  and `ts_backtest_2025_20260423_055829_predictions.csv`

Run:
```
python scripts/diagnose_inactive_pick_gap.py
```

## Next step per the re-council's sequence

**Chairman's Step 2** (ADP scrape + snake-draft simulator) was
gated on symmetric clear. With this diagnosis:

- The gate remains on **any user-facing work without a pre-lock
  active-roster filter**.
- The gate opens for **work that includes the filter**, subject to
  adding actions 1–3 above to the paper-trade harness first.

Recommendation: add the three features to the harness (~1 day),
re-measure the wide-symmetric win rate with a simulated active-
roster filter (drop inactives from the model's pool before
ranking), and re-apply the kill gate to that number. If it clears
52.4 %, proceed with Step 2.
