# Pre-registration — 2026 W1-W4 out-of-sample test

**Date pre-registered:** 2026-04-25
**Locked commit SHA:** `4b299a0` (HEAD of `main` at time of writing,
post-8-season audit)
**Test execution date:** Within 7 days of completion of 2026 NFL
regular season Week 4 (estimated mid-October 2026)

## Purpose

Every audit done to date has had a residual look-back caveat:
features were tuned in light of inspecting 2024-2025 data, which
was then used to score performance. The 8-season audit
(`docs/STAT_SIG_AUDIT_8SEASON_20260425.md`) reduced this concern by
showing 2018-2020 (seasons unseen during recent feature
development) maintain the same 70-80 % win rate, but did NOT
eliminate it.

This pre-registration handles the remaining look-back caveat by
committing — BEFORE 2026 data exists — to:

1. The exact codebase to run.
2. The exact commands and parameters.
3. The exact metrics to report.
4. The exact pass / fail criteria.
5. The conditions that would invalidate the test.

If the locked code passes the locked criteria on 2026 W1-W4, the
8-season historical claim earns a clean OOS confirmation.

## Locked artifacts (immutable for this test)

| Component | Locked at | Notes |
|---|---|---|
| Codebase commit | `4b299a0` | Includes Phase 4A conformal intervals, 4B VORP, 4C rookie priors, 4D NO-OP, share-feature fix, prev_season_ppg |
| Feature config | `config/settings.py:CAUSAL_FEATURES` at `4b299a0` | 9-10 features per position, no draft-product features |
| Model | Ridge with `RIDGE_DEFAULT_ALPHA = 10_000` | Per-position dispatch supported but not used in this test |
| Active-roster filter | ON (`status='ACT'` from `weekly_rosters`) | The 2026 weekly rosters must be backfilled before scoring |
| Backtester | `scripts/run_ts_backtest.py` at `4b299a0` | `--emit-inactive-predictions` for wide mode |
| Replay harness | `scripts/wide_symmetric_replay.py` at `4b299a0` | `--apply-active-filter` |
| Margin analysis | `scripts/analyze_continuous_margin.py` at `4b299a0` | All defaults |

## Pre-specified procedure

**Step 1 — backfill 2026 dependencies (in commit `4b299a0` or
later, but using the same scripts unchanged).** These are
infrastructure operations, not model changes:

```bash
python scripts/backfill_weekly_rosters.py -s 2026 2026
python scripts/backfill_injuries.py --season 2026     # if script exists at lock-commit
```

If 2026 schedule / Vegas / draft data is also needed for any
already-coded feature, run the corresponding backfill from
commit `4b299a0`. **No code changes.**

**Step 2 — run the walk-forward in wide mode for 2026:**

```bash
python scripts/run_ts_backtest.py --season 2026 --emit-inactive-predictions
```

This produces `data/backtest_results/ts_backtest_2026_<TS>.json`
covering all weeks of 2026 actually played at execution time. We
filter to weeks 1-4 only when reporting (next step).

**Step 3 — score 2026 W1-W4 against the locked baseline:**

```bash
python scripts/wide_symmetric_replay.py \
    --runs data/backtest_results/ts_backtest_2026_<TS>.json \
    --apply-active-filter \
    --bootstrap-trials 100000

python scripts/analyze_continuous_margin.py \
    --runs data/backtest_results/ts_backtest_2026_<TS>.json \
    --apply-active-filter
```

If the JSON includes weeks > 4, both scripts already
auto-iterate over `weeks` in the run; we report only the rows
where `week ∈ {1, 2, 3, 4}`.

## Pre-specified metrics

For 2026 W1-W4 only (n=4 weeks):

| Metric | Definition | What we report |
|---|---|---|
| Symmetric WR | wins / 4 weeks | record + WR + Wilson 95 % CI |
| Mean per-week margin | mean(model_lineup_actual − opponent_lineup_actual) | mean + bootstrap CI |
| ROI at -110 vig (1.8× payout) | (1.8 × wins − 4) / 4 | mean + SE |
| Sign of mean margin | mean > 0 ? | direction |
| Bootstrap p5 of mean margin | from 100k IID resamples | lower bound |

## Pass / fail criteria (locked BEFORE seeing 2026 data)

Three independent gates. **PASS** = at least the *primary* clears.

### Primary gate (continuous-margin sign test)
- **PASS:** mean weekly margin > 0 across W1-W4 AND bootstrap 5th
  percentile of the mean > -5 pts (allowing for small-n variance).
- **STRONG PASS:** mean weekly margin ≥ +10 pts (consistent with
  the 8-season historical mean of +19, allowing for season-start
  noise).
- **FAIL:** mean weekly margin ≤ 0.

### Secondary gate (binary win rate)
- **PASS:** model wins ≥ 3 of 4 weeks.
- **STRONG PASS:** 4 of 4.
- **FAIL:** < 3 of 4.

### Tertiary gate (per-week sanity)
- **PASS:** no single week loses by more than 2 standard
  deviations from the 8-season margin distribution (i.e., no week
  worse than approximately −40 pts margin).
- **FAIL:** any single week shows a margin worse than −60 pts (5+
  SD below historical mean — extreme outlier).

## Decision rules

| Outcome | Action |
|---|---|
| All 3 gates pass | Cite "pre-registered 2026 W1-W4 confirmation" alongside the 8-season number. The model claim earns a clean OOS confirmation. |
| Primary passes, secondary fails (e.g., 2-2 record but +12 mean margin) | Document the result. Cite the continuous estimator (which is the more powerful one). Note the small-n binary disagreement honestly. |
| Primary fails, secondary passes (e.g., 3-1 record but mean margin ≤ 0) | Document the result. Treat as a borderline outcome — single weeks can win narrowly while average margin is small or negative. Reconvene before any product action. |
| Both primary AND secondary fail | Treat as a meaningful negative signal. Possible causes (in priority order): (1) genuine model regression on 2026 player pool, (2) early-season high variance (W1-W4 historically more volatile than W5+), (3) data quality issue (rookie pool larger than fit, injuries not yet reported). Pause any user-facing product action. Do NOT silently retune to make the gate pass — that destroys the pre-registration. |
| Any single week fires the tertiary gate | Investigate that week specifically (data quality, lineup composition, opponent pool). Do not invalidate the test based on one bad week unless the cause is mechanical (e.g., active-roster filter broken for that week). |

## What invalidates this pre-registration

These conditions disqualify the test from being a clean OOS:

1. **Any modification to feature engineering, CAUSAL_FEATURES,
   model class, or hyperparameters between `4b299a0` and 2026 W1.**
   If we ship an improvement (e.g., snap-count backfill,
   Vegas-pre-2024 backfill, new feature) BEFORE the test runs,
   commit a SECOND pre-registration doc that locks the new
   commit. Run BOTH locked codes on 2026 W1-W4 and report both.
2. **Looking at any 2026 data before W1-W4 has fully landed and
   this gate has been scored.** No "peek and adjust." If we
   need 2026 data for backfill (e.g., 2026 schedule for Vegas
   features), that's allowed only via the unchanged backfill
   scripts at `4b299a0`.
3. **Changes to the active-roster filter, opponent baseline, or
   replay harness.** Same lock as the model.
4. **Loss of test integrity** — e.g., the predictions CSV from
   commit `4b299a0` cannot reproduce the historical 8-season
   numbers (a sanity-check we run as part of step 1).

## Sanity checks before scoring 2026 W1-W4

Before reporting the 2026 W1-W4 result, run:

```bash
# Reproduce the 8-season number on the locked commit. If this
# differs by more than 1pp WR or 1pt mean-margin from
# docs/STAT_SIG_AUDIT_8SEASON_20260425.md, the locked codebase
# has drifted and the test is invalid.
git checkout 4b299a0
python scripts/wide_symmetric_replay.py \
    --runs data/backtest_results/ts_backtest_{2018..2025}_*.json \
    --apply-active-filter
```

Expected: 124-40 = 75.6 % symmetric, p5 = 70.1 %, mean margin
+19.0 pts. Any deviation > tolerance → halt and investigate.

## Honest limits of this test

**This is a 4-week confirmation test, not a freshly-powered effect
test.** With n=4 weeks:

- We cannot distinguish 65 % from 85 % true WR.
- A 3-1 record has Wilson 95 % CI of [22 %, 96 %] — wide.
- A single bad week dominates the average.
- Rule-change effects, rookie classes, and 2026-specific data
  quality issues will move the result.

What this test CAN do:

- Confirm directional consistency with the 8-season historical.
- Detect catastrophic regression (the kind that should kill product
  plans).
- Establish a reproducible OOS protocol that can be re-run with
  more weeks as 2026 progresses (W1-W17 once the season ends).

## Extension policy

After the W1-W4 gate is scored and reported, we may run an
**extended** test on the same commit `4b299a0` covering all 2026
regular-season weeks once they're available (~early Jan 2027). That
extended test does NOT require a new pre-registration since the
codebase is the same and the extension to more weeks was
foreshadowed here.

## Timestamping integrity

This document is committed to `main` at the same SHA range as the
locked code. Git log will show:

- Commit `4b299a0`: code base at lock time
- This commit: pre-registration document
- Future scoring commits: 2026 W1-W4 results

If anyone modifies this document after 2026 W1-W4 data exists,
git history will show the post-data edit. The integrity of the
pre-registration is the immutable timestamp of this commit.

## Owner / accountability

This pre-registration is owned by the project maintainer. Scoring
in October 2026 must be performed against this document
verbatim. Any deviation gets explicitly noted in the scoring
commit's body and treated as a partial-PASS or partial-FAIL per
the decision rules above.
