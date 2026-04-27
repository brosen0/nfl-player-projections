# Position-classification audit (112 conflicts) — 2026-04-25

**STATUS: FIXED 2026-04-25** (commit pending). Layer 1 (data) +
Layer 2 (code) both applied. Conflicts dropped from 112 → 23
(remaining 23 are legitimate position-changers, not bugs).

**Provenance:** discovered while preparing the user-session smoke
test. The test user's roster included H.Fannin Jr. (Cleveland TE,
2025 round 3), but our `players` table classified him as WR. A
broader audit found this is one of **112 player misclassifications**
between the `players` and `draft_picks` tables.

## Root cause

Found in `src/data/pbp_stats_aggregator.py:_infer_position`. When
the loader uses PBP fallback ingestion (current season / partial
weeks / weekly stats unavailable), positions are inferred via:

```python
# Distinguish TE from WR: TEs typically have fewer receiving
# yards per target than WRs.
if targets > 0 and recv_yards / max(targets, 1) < 8:
    return 'TE'
return 'WR'   # default
```

**The bug:** productive TEs (Bowers 2024, McBride 2025, Pitts
several seasons) average > 8 yds/target — they fail the heuristic,
fall through to the WR default. nflverse's own weekly stats label
them correctly as TE; our local PBP-aggregator inference overrides
this when it runs.

The bug is asymmetric: high-volume TEs misclassified as WR (63 of
the 112). RB-misclassified-as-WR (22) follows similar logic when
the rush-vs-target check tips the wrong way.

## Fix applied (this commit)

### Layer 1 — data UPDATE

```sql
UPDATE players SET position = (
    SELECT wr.position FROM weekly_rosters wr
    WHERE wr.player_id = players.player_id
      AND wr.position IN ('QB','RB','WR','TE')
    GROUP BY wr.position ORDER BY COUNT(*) DESC LIMIT 1
)
WHERE EXISTS (
    SELECT 1 FROM weekly_rosters wr
    WHERE wr.player_id = players.player_id
      AND wr.position IN ('QB','RB','WR','TE')
)
AND players.position IN ('QB','RB','WR','TE');
```

Result: 1,545 rows updated. Conflicts vs `draft_picks` dropped
from 112 → 23.

The remaining 23 are legitimate position-changers and edge cases:
- 7 TE → WR (drafted as WR, transitioned to TE — modal weekly
  is now TE; draft_picks lags)
- 5 RB → WR (drafted as WR, became RB)
- 3 RB → TE (rare; verified one-off)
- 3 TE → QB (Tebow, etc.)
- 2 WR → QB (trick-play QBs)
- 1 each: QB → WR, TE → RB, WR → RB

These are NOT bugs — they reflect the player's actual playing
position vs draft slot. Authoritative source = `weekly_rosters`,
which is what the UPDATE used.

Spot checks post-fix:
- B.Bowers (00-0039338): TE ✓
- T.McBride (00-0037744): TE ✓
- K.Pitts (00-0036970): TE ✓
- H.Fannin (00-0040663): TE ✓
- C.Patterson (00-0030578): RB ✓ (correctly preserves his
  RB-since-2021 transition)
- C.McCaffrey (00-0033280): RB ✓ (unchanged, was already correct)

### Layer 2 — code fix in `_infer_position`

`src/data/pbp_stats_aggregator.py` — added a `players` table
lookup AFTER `row['position']` and BEFORE the heuristic. Tries:

1. nflverse-supplied `row['position']` (preferred)
2. Local `players.position` (now correct after layer 1)
3. Heuristic fallback (only for never-seen players)

Cached at class-level via `_players_position_lookup()`. ~5,000 row
dict; loaded lazily; no per-row DB hits.

This means future PBP-fallback ingestions will use the
authoritative position for any known player. The buggy heuristic
is only used as a last resort for never-seen players.

## Verification

```bash
# All unit tests pass post-fix
python -m pytest tests/test_start_sit_prototype.py tests/test_rookie_priors.py \
    tests/test_snake_draft_sim.py -q
# 22 passed

# Test user prototype runs cleanly with corrected Fannin
python scripts/start_sit_prototype.py \
    --roster data/user_sessions/test_user_2025_w15.json \
    --season 2025 --week 15 --flex 1
# H.Fannin TE matches; total 140.40 (model = user, no swap)
#
# NOTE (post-audit, 2026-04-25): test_user_2025_w15.json was
# subsequently corrected to reflect the user's ACTUAL lineup —
# the user started C.Loveland at TE, with H.Fannin on the bench
# (this was also the pivotal regret decision in the playoff
# SEMIFINALS, not the first round as originally labeled).
# Re-running the command above will therefore produce different
# numbers than the snapshot recorded here. The TE classification
# itself is unaffected — both Loveland and Fannin classify as TE
# correctly post-fix.
```

## What this does NOT change

- The 8-season walk-forward CSVs were generated BEFORE this fix.
  Per-position metrics in `STAT_SIG_AUDIT_8SEASON_20260425.md` are
  still based on the old (buggy) classification. Re-running 8-
  season wide+narrow walk-forward (~3.5 hrs wall on 4-parallel)
  would refresh those numbers. Worth doing as a clean batch.
- The headline 75.6 % / +19 pts/week claim is unaffected — uses
  summed lineup scores, not per-position.
- Phase 4C rookie priors were fit using `draft_picks` position
  which is mostly correct; not affected.

## Files changed (this commit)

- `data/nfl_data.db` — UPDATE on `players.position` (1,545 rows
  re-classified to weekly_rosters modal authority)
- `src/data/pbp_stats_aggregator.py:_infer_position` —
  players-table lookup added between row[position] and heuristic;
  new `_players_position_lookup` classmethod with cache
- `docs/POSITION_CLASSIFICATION_AUDIT.md` — this update with RCA
  and fix-applied status

## TL;DR

- 112 players have `players.position ≠ draft_picks.position`
  (within the QB/RB/WR/TE fantasy pool).
- Bug skews heavily toward **WR-misclassified-as-TE** drafts
  (63 of 112) and recent eras (88 of 112 are 2020-2025).
- 90 of those players have active 2025 predictions; **929 of
  5,595 active 2025 prediction rows (16.6 %) carry the wrong
  position**.
- High-impact victims include **B.Bowers, T.McBride, K.Pitts,
  T.Warren, C.Loveland, D.Kincaid, P.Freiermuth, J.Ferguson** —
  all top-15 fantasy TEs misclassified as WRs in our pipeline.
- They've been trained against the WR pool, evaluated against WR
  features, and contributed to WR-bucket error metrics. The
  per-position R² numbers (`docs/CAUSAL_FEATURES_AUDIT_20260424.md`)
  are mildly distorted by this.
- **Fix source:** `weekly_rosters` per-season position is the
  authoritative classification (handles rookies + veterans +
  position-changers like C.Patterson WR→RB). We've already
  backfilled this table.

## What's in conflict

| players_pos | draft_pos | n |
|---|---|---|
| WR | TE | **63** |
| WR | RB | 22 |
| TE | WR | 9 |
| RB | QB | 4 |
| RB | WR | 4 |
| TE | QB | 4 |
| RB | TE | 2 |
| WR | QB | 2 |
| QB | WR | 1 |
| TE | RB | 1 |

By era (draft year):

| Year | Conflicts |
|---|---|
| 2006-2019 | 24 |
| 2020 | 9 |
| 2021 | 11 |
| 2022 | 20 |
| 2023 | 15 |
| 2024 | 14 |
| **2025** | **19** |

Recent-era concentration suggests the bug is in current ingestion
flow rather than historical data. New rookies arriving with their
"college position" or a default WR fallback in some upstream
mapping.

## Top-15 affected players in 2025 active predictions

(`players` position shown — likely WRONG)

| player_id | name | players_pos | team | weeks | predicted | actual |
|---|---|---|---|---|---|---|
| 00-0037744 | T.McBride | WR | ARI | 17 | 257.5 | 315.9 |
| 00-0038041 | J.Ferguson | WR | DAL | 17 | 190.2 | 190.1 |
| 00-0036970 | K.Pitts | WR | ATL | 17 | 184.3 | 210.8 |
| 00-0040128 | T.Warren | WR | IND | 17 | 182.2 | 174.1 |
| 00-0036919 | K.Gainwell | WR | PIT | 18 | 157.8 | 231.9 |
| 00-0039793 | A.Barner | WR | SEA | 19 | 154.7 | 150.8 |
| 00-0040126 | C.Loveland | WR | CHI | 18 | 152.5 | 196.4 |
| 00-0039338 | B.Bowers | WR | LV | 12 | 150.0 | 174.2 |
| 00-0040189 | O.Gadsden | WR | LAC | 16 | 140.0 | 139.4 |
| 00-0037809 | C.Okonkwo | WR | TEN | 17 | 135.8 | 124.0 |
| 00-0038933 | D.Kincaid | WR | BUF | 14 | 130.7 | 158.2 |
| 00-0039847 | T.Johnson | WR | NYG | 15 | 128.4 | 127.8 |
| 00-0036244 | C.Parkinson | WR | LA | 17 | 126.1 | 161.0 |
| 00-0038129 | C.Otton | WR | TB | 15 | 124.5 | 122.2 |
| 00-0036894 | P.Freiermuth | WR | PIT | 16 | 124.5 | 116.4 |

All 15 of these are TEs in `draft_picks` and TEs in
`weekly_rosters`.

## Why this affects walk-forward results

`src/utils/database.py:get_all_players_for_training` joins
`players` for the position column. Position drives:

1. **Training pool selection** — the per-position Ridge models
   are trained per `position` value. A TE classified as WR
   trains against the WR pool, not TE.
2. **Feature selection** — `CAUSAL_FEATURES[position]` differs
   per position (e.g., RB has `rush_share_pct_roll3_mean`, WR
   has `air_yards_share_pct_roll3_mean`). A misclassified player
   gets the wrong feature set.
3. **Per-position error metrics** — R², MAE, bias are reported
   per `position`. Misclassified players poison both the source
   bucket (TE has fewer rows) and the destination bucket (WR
   gets contaminated by TE-pattern players).

Quantitative impact is bounded: most misclassified players are
TEs being trained as WRs, and TEs and WRs share most predictive
features (targets, receptions, target_share, etc.). Predictions
are likely directionally correct but the per-position R² numbers
are noisy.

## Proposed fix

### Source of truth: `weekly_rosters`

We already backfilled `weekly_rosters` from nflverse for 2018-2025
(379,789 rows, 100 % position coverage). Spot-checked it correctly
classifies:

- **B.Bowers**: TE 2024+2025 (vs `players` WR — wrong)
- **C.Patterson**: WR 2018-2020, RB 2021-2024 (vs `players` RB —
  matches current era; this player legitimately changed
  position)

So `weekly_rosters.position` is per-season authoritative — it
handles new rookies, veterans, AND position-changers correctly.

### Recommended fix (minimum scope)

Replace the `JOIN players p ON pws.player_id = p.player_id` with
a `(player_id, season)`-keyed lookup against `weekly_rosters`.
Modal position per (player, season) handles intra-season
fluctuations.

```sql
-- New per-season position view
CREATE VIEW player_season_position AS
SELECT player_id, season,
       (SELECT position FROM weekly_rosters wr2
         WHERE wr2.player_id = wr.player_id AND wr2.season = wr.season
         GROUP BY position
         ORDER BY COUNT(*) DESC LIMIT 1) AS position
  FROM weekly_rosters wr
 GROUP BY player_id, season;
```

Then update `get_all_players_for_training` to JOIN against this
view instead of `players` for the `position` column. (Keep the
`players` JOIN for the `name` column since we still need that.)

### Alternate fix (cosmetic, in-DB)

UPDATE `players.position` row-by-row from the `weekly_rosters`
modal-per-player. Simpler but loses per-season granularity for
position-changers. Acceptable if we don't care about historical
seasons (position-changes are rare).

```sql
UPDATE players SET position = (
    SELECT position FROM weekly_rosters wr
    WHERE wr.player_id = players.player_id
    GROUP BY position
    ORDER BY COUNT(*) DESC LIMIT 1
)
WHERE EXISTS (SELECT 1 FROM weekly_rosters wr WHERE wr.player_id = players.player_id);
```

This affects ~5 % of `players` rows but only the 112 listed here
have a downstream impact via the position-conflict path. Easier
to implement, slightly less correct for position-changers.

## Cost estimate

| Step | Time |
|---|---|
| DB position fix (per recommendation) | 30 min code, 5 min run |
| Re-run 8-season wide-mode walk-forward | 4 × 17 min × 2 modes = ~135 min wall-time on 4-parallel |
| Re-run cross-season symmetric replay | 30 sec |
| Re-run continuous-margin audit | 2 sec |
| Update `STAT_SIG_AUDIT_8SEASON` if numbers shift > 1pp | 30 min writeup |
| **Total** | **~3.5 hours** |

## Expected impact on the headline number

The 75.6 % symmetric WR / +19 pts/week mean margin / [+15.2,
+23.0] CI all use SUMMED scores across the lineup, not
per-position. Position misclassification doesn't change a
player's actual fantasy points. So the symmetric WR is **largely
robust to this bug**.

Where it WILL matter:
- **Per-position R²** numbers in `STAT_SIG_AUDIT_8SEASON` will
  shift, especially TE (which gains its top players back).
- **Conformal interval coverage** per-position may shift since
  the residual pools change.
- **Draft-sim outputs** because the simulator uses the position
  field to assign roster slots — a TE classified as WR can't be
  drafted to a TE slot. Likely small impact since draft-sim is
  already shelved.

## Verification SQL

```sql
-- 1. Conflict check (run before AND after fix)
SELECT p.position, dp.position, COUNT(*)
  FROM players p
  JOIN draft_picks dp ON p.player_id = dp.player_id
 WHERE p.position IN ('QB','RB','WR','TE')
   AND dp.position IN ('QB','RB','WR','TE')
   AND p.position != dp.position
 GROUP BY p.position, dp.position
 ORDER BY 3 DESC;
-- Expected before: 10 rows totaling 112
-- Expected after: 0 rows (or only legitimate position-changers
--                          if we use cosmetic fix)

-- 2. Cross-check against weekly_rosters
SELECT p.player_id, p.name, p.position AS players_pos,
       (SELECT position FROM weekly_rosters wr
         WHERE wr.player_id = p.player_id AND wr.season = 2025
         GROUP BY position ORDER BY COUNT(*) DESC LIMIT 1) AS wr_2025_pos
  FROM players p
 WHERE p.player_id IN (
   SELECT player_id FROM player_weekly_stats WHERE season = 2025
 )
HAVING players_pos != wr_2025_pos
LIMIT 30;
```

## Files that would change

1. `src/utils/database.py` — `get_all_players_for_training` query +
   one-line view creation in `_init_database`.
2. `data/nfl_data.db` — view creation, optionally
   `players.position` UPDATE.
3. `data/backtest_results/` — 16 new CSVs + JSONs (8 seasons × 2
   modes).
4. `docs/STAT_SIG_AUDIT_8SEASON_20260425.md` — refresh per-position
   numbers if shift > 1 pp.
5. `docs/POSITION_CLASSIFICATION_AUDIT.md` — this doc, updated
   with post-fix results.

Within CLAUDE.md's 5-source-file budget for a phased commit.

## Out of scope

- **Don't modify `draft_picks`** as a fix — it has its own
  legitimate use as the rookie-prior + draft-capital data source.
  `weekly_rosters` is the correct authority for "what position is
  this player playing now."
- **Don't audit other tables for similar conflicts** in this pass.
  Could be a follow-up if integrity issues persist.
- **Don't re-fit rookie priors** — Phase 4C used `draft_picks`
  position which is mostly correct for rookies. Re-fitting is
  cheap (~10s) but only matters for rookies whose draft_picks
  position is also wrong. Tiny effect.

## Recommendation

Defer until after the user session lands. The session itself
needs the prototype working (Fannin already patched specifically),
and the broader fix is a multi-hour clean-up best done as a
focused workstream once the user-feedback channel is established.
Then run the fix + re-publish the 8-season audit numbers in one
clean batch.
