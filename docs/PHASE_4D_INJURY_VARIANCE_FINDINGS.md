# Phase 4D — injury-adjusted variance — findings (NO-OP)

Phase 4D's premise: stratify Phase 4A's conformal intervals by
injury status, because Questionable / Doubtful players have higher
residual variance than healthy players, and a one-size-fits-all
interval under-covers them.

**Verdict: abort. No code change needed.** The premise fails an
empirical pre-check.

## What we tested

Joined `player_injuries.report_status` to the post-audit predictions
CSVs (2024 + 2025; commits `0eba000` / `4ee4709`) on
`(player_id, season, week)` for all rows with `is_active=1` and
non-NaN actuals. Bucketed by status and computed `|actual −
predicted|` per (position, bucket).

```
                          n   mean  median
position bucket
QB       Healthy       1303  6.234   5.481
         Questionable    17  6.938   5.812
RB       Healthy       2370  5.212   4.141
         Questionable   100  4.999   4.575
TE       Healthy        823  4.169   3.343
         Questionable    32  4.647   4.389
WR       Healthy       6167  4.599   3.769
         Questionable   263  4.824   3.703
```

Uplift ratios (Q mean / Healthy mean):
- QB: 1.11×  (n=17, very small sample)
- RB: **0.96×** (Q residual LOWER than healthy)
- WR: 1.05×
- TE: 1.11×

Pooled across positions weighted by Q-bucket n (412 rows): **uplift ≈
1.04×**. Within sampling noise of 1.0.

## Why this happened

Two upstream filters already captured the injury signal:

1. **Active-roster filter** (`scripts/paper_trade_lock.py`,
   `scripts/wide_symmetric_replay.py:--apply-active-filter`,
   commit `83daae9`). Drops every player whose `weekly_rosters.status
   != 'ACT'`. This includes Doubtful and Out — they don't make the
   active-eligible pool. The conformal CSVs only see ACT players.
   - Empirical confirmation: zero `Doubtful` or `Out` rows in either
     2024 or 2025 prediction CSVs after the filter.
2. **`injury_score` feature** in `CAUSAL_FEATURES` (Phase 3, commit
   referenced in `docs/PHASE_3_INJURY_FINDINGS.md`). Ridge already
   shifts the point prediction down for injured players. Whatever
   variance contribution the "Questionable" tag carries is mostly
   captured in the conditional mean.

By the time a player is both `status=ACT` AND only `Questionable`
on the injury report, they've effectively been cleared to play.
The "Q" label is mostly a procedural CYA — uplift ≈ 1.04×
confirms it.

## What we'd build if the premise had held

For reference (so future work can pick this up if injury-data
quality improves and the signal returns):

- Extend `add_conformal_intervals.py` to maintain a per-(position,
  bucket) rolling residual pool.
- Bucket → look up at conformal time via `(player_id, season, week)`
  against `player_injuries.report_status`.
- Sample-size guard: if a (position, Q) bucket has <20 obs, fall
  back to healthy-pool width × bucket-specific scalar.
- Validate: stratified coverage at 80 % within [70, 90] per cell,
  with Q/D widths ≥ 1.1× healthy.

If a future Phase 4D-redux runs this with the bucket data, the
gate is the 1.1× width-uplift one — current data shows 1.04×, so
re-checking is the first step.

## What this does NOT change

- Phase 4A intervals remain valid. The 79.1 % / 80.3 % overall 80 %
  coverage already passes the gate; injury stratification was a
  refinement, not a fix.
- The 75.6 % / p5=65.85 % symmetric kill-gate is unchanged.
- The August 2026 draft kill verdict is unchanged.

## Files touched

None. This is a documentation-only commit (the empirical probe was
a one-off Python snippet; the snippet's intent is captured in this
doc).

## Status of Step 4 phases

- 4A (conformal intervals): **shipped, gate passed**
  (`docs/PHASE_4A_CONFORMAL_FINDINGS.md`)
- 4B (VORP): shipped + name-matcher fix; revealed pre-draft model
  loses to ADP, triggered draft kill criterion
  (`docs/PHASE_4B_VORP_FINDINGS.md`)
- 4C (rookie priors): shipped, +0.011 rookie R² each season
  (`docs/PHASE_4C_ROOKIE_PRIORS_FINDINGS.md`)
- 4D (injury-adjusted variance): **NO-OP — premise fails empirical
  pre-check** (this doc)

Step 4 is done. The in-season start/sit feature set is now
calibrated (4A) on top of corrected feature pipeline (audit) with
rookie priors (4C). Draft scope is shelved per the council
directive (commit `47771ae`).
