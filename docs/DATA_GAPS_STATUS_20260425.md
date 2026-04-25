# Data-gaps status check — 2026-04-25

**Provenance:** option (3) of the post-Phase-4 punch list.  Three
data gaps were called out across earlier audits as deferred items.
This doc reconciles their actual current state against what was
claimed.

## TL;DR per gap

| Gap | Claimed status | Actual status | Fix effort |
|---|---|---|---|
| (A) Vegas features pre-2018 | "67-72 % defaulted on 2022-2023" | Real — `schedule` table has 2018-2025 only; pre-2018 missing | **~15 min** (extend `backfill_vegas_lines.py` to 2006-2017) |
| (B) Snap counts | "zero-filled across every season" | Real — `snap_count` / `team_snaps` are 0 for ALL 2006-2025, not just recent | ~1 hour (new `backfill_snap_counts.py` from nflverse) |
| (C) Pre-2010 `team_defense_stats` | "missing pre-2010" | **False alarm — 100 % coverage 2006-2025** | n/a |

Plus a small bonus bug found during this audit:

| Gap | Status | Fix |
|---|---|---|
| (D) OAK/LV team-code mismatch (2018-2019) | Real, low-impact | One UPDATE statement |

## Gap (A) — Vegas features pre-2018

### Current state

`schedule` table coverage:

```
season   games   w_spread   w_total
 2018      267        267       267
 2019      267        267       267
 2020      269        269       269
 2021      285        285       285
 2022      284        284       284
 2023      285        285       285
 2024      285        285       285
 2025      285        285       285
```

Pre-2018: **zero rows.**  This is why the walk-forward backtest on
2022-2023 reports "Vegas features defaulted on 67-72 % of rows" —
the training set spans 2006-2022, but only 4 of 17 historical
seasons have Vegas data.  The warning is honest.

### Source: nflverse has it

`https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`
has **100 % spread_line + total_line coverage going back to 2006**
(7,276 rows total, 267 games/year for 2006-2019).  We just never
backfilled it.

### Fix

```bash
# scripts/backfill_vegas_lines.py already supports a season range.
python scripts/backfill_vegas_lines.py --seasons 2006 2017
```

(Or extend the default range to 2006-2025.)

Then re-run the 8-season walk-forward.  Expected impact: pre-2018
training rows now have real Vegas signal.  Whether this lifts the
75.6 % WR is unclear — the model has been working without it.  But
the per-season R² on 2018-2020 (currently 14-6, 16-4, 14-6) might
firm up if Vegas-aware training data dominates.

## Gap (B) — snap counts

### Current state

```
season    pws_rows   nonzero   avg
 2006        5502         0     0.0
 2007        5618         0     0.0
 ...
 2024        5480         0     0.0
 2025        5595         0     0.0
```

**Zero `snap_count > 0` rows in every season 2006-2025.**  The
`player_weekly_stats.snap_count` and `team_snaps` columns were
never populated by the original ingestion.

### Why it matters

`snap_share_pct` was excluded from `CAUSAL_FEATURES` for RB and
TE in commit `08ddca6` because it would have been a dead
declaration.  Re-enabling it would require:

1. Backfilling `snap_count` and `team_snaps` from
   `https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{year}.parquet`
   (verified reachable; pattern matches `weekly_rosters` backfill).
2. Re-deriving `snap_share_pct = snap_count / team_snaps * 100` in
   `_create_base_features`.
3. Adding `snap_share_pct_roll3_mean` back to `CAUSAL_FEATURES["RB"]`
   and `CAUSAL_FEATURES["TE"]`.

### Cost

| Step | Time |
|---|---|
| Write `scripts/backfill_snap_counts.py` (mirrors `backfill_weekly_rosters.py`) | ~30 min |
| Run backfill 2006-2025 | ~5 min |
| Toggle CAUSAL_FEATURES + re-run 8-season walk-forward (wide + narrow) | ~140 min |
| Re-verify symmetric replay + continuous margin | ~5 min |
| Update findings docs | ~30 min |
| **Total** | **~3.5 hours** |

### Expected impact

Snap share is a heavy contributor to RB / TE volume prediction.
Adding this feature (with the 8-season feature set otherwise
unchanged) is the most likely lever to improve our headline +19
pts/week toward +20 or higher.  But this is speculation —
worth measuring, not predicting.

## Gap (C) — pre-2010 `team_defense_stats`

### Current state

**False alarm.**  Coverage is 100 % for every season 2006-2025:

```
season    rows   w_fp_data
 2006     534      534      ← all 32 teams × ~17 weeks (with regs)
 2007     534      534
 ...
 2024     570      570      ← 32 × 18 = 570 (reg+playoffs)
 2025     568      568
```

The "14.2 % defaulted" warning we hit in `opp_fpts_allowed_s2d_lag1`
during the audit (`docs/CAUSAL_FEATURES_AUDIT_20260424.md`) was
from week-1 / opponent-week-0 edge cases, not missing data.  The
season-to-date computation `shift(1).expanding(min_periods=1).mean()`
returns NaN for week-1 of every season because there's no prior
week within-season; that's NaN-then-defaulted, but it's
mathematically correct.

**No fix needed.**

## Gap (D) — OAK/LV team code (bonus discovery)

### Current state

```
2018: pws has team="LV", schedule has team="OAK"  (32 game-slots affected)
2019: same
```

The Raiders moved Oakland → Las Vegas in 2020.  Our
`player_weekly_stats` team column was retroactively renamed during
ingestion; `schedule` retained historical names.  Result: 2018+2019
Raiders games can't JOIN, contributing a few percent to the
Vegas-defaulted rate.

### Fix

```sql
UPDATE schedule SET home_team = 'LV' WHERE home_team = 'OAK' AND season < 2020;
UPDATE schedule SET away_team = 'LV' WHERE away_team = 'OAK' AND season < 2020;
```

Or symmetrically, normalize `pws.team` for 2018-2019 Raiders rows
to "OAK".  Either works; LV-everywhere is more consistent with
nflverse 2025 conventions.

Also worth checking: SD/LAC (Chargers moved 2017), STL/LA (Rams
moved 2016).  Likely similar issues.

### Cost

~5 min.  Negligible impact — LV+SD+STL combined is < 100
game-slots out of ~5,000+ pre-2020 games.

## Combined recommendation

| Item | Cost | Recommendation |
|---|---|---|
| (A) Vegas pre-2018 | 15 min | **Do it.** Free win — script exists, just extend range. |
| (B) Snap counts | 3.5 hrs | **Defer until post-user-session.** Largest potential lift; warrants its own focused workstream. |
| (C) team_defense_stats | n/a | **Close the ticket.** False alarm. |
| (D) OAK/LV team codes | 5 min | **Do it.** Trivial, mildly improves 2018-2019 numbers. |

If we do (A) + (D) now (~20 min), the 8-season audit numbers may
shift slightly upward for 2018-2019.  Worth re-running the
continuous-margin audit afterward to refresh the headline.

## Files referenced

- `scripts/backfill_vegas_lines.py` (existing) — extends to pre-2018.
- `scripts/backfill_snap_counts.py` (new) — for gap (B).
- `data/nfl_data.db` — UPDATE for gap (D).
- `src/utils/database.py` — no changes needed for any gap.
- `src/features/feature_engineering.py:_create_vegas_game_script_features`
  is correct as-is; the bug was upstream data, not the consumer.

## Verification SQL after any fix

```sql
-- (A) Vegas pre-2018
SELECT COUNT(*), MIN(season), MAX(season) FROM schedule WHERE spread_line IS NOT NULL;
-- Expected after fix: ~5,000 rows, MIN=2006

-- (B) Snap counts
SELECT season, SUM(CASE WHEN snap_count > 0 THEN 1 ELSE 0 END) FROM player_weekly_stats
 GROUP BY season ORDER BY season;
-- Expected after fix: nonzero rows in every season

-- (C) team_defense_stats
-- No change expected — this gap was not real

-- (D) OAK/LV
SELECT season, COUNT(*) FROM schedule WHERE home_team='OAK' OR away_team='OAK' GROUP BY season;
-- Expected after fix: 0 for all seasons (or 0 for 2020+)
```
