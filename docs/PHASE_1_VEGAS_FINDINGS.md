# Phase 1 (Vegas implied total) — Pre-implementation findings

**Branch:** `claude/predictive-ceiling-phase-1-vegas`
**Plan reference:** [`docs/PREDICTIVE_CEILING_PLAN.md`](./PREDICTIVE_CEILING_PLAN.md)
**Status:** investigation; no new wiring landed.  Implementation is **blocked** on confirming whether the existing wiring is broken in the user's environment.

## TL;DR

Phase 1 was scoped to "wire Vegas implied team total into the causal feature pipeline." Investigation reveals the wiring already exists — but with a silent-fallback bug that may be quietly degrading the feature to a constant default in the live training run. Before adding new feature wiring, the priority is **diagnose whether the existing Vegas integration is actually live in the April 10 R²=0.269 baseline**.

## What was discovered

1. **`CAUSAL_FEATURES` already declares Vegas features** for all four positions (`config/settings.py:281–310`):
   - `"implied_team_total"`, `"spread"` for QB, RB, WR, TE.

2. **Population code already exists** at `src/features/feature_engineering.py:1565` (`_create_vegas_game_script_features`). It calls `nfl_data_py.import_schedules(...)` at runtime, merges spread/total/implied_team_total onto the player DataFrame, and falls back to constants on failure.

3. **The fallback was silent.** Lines 1633–1634 (pre-PR) had:
   ```python
   except Exception:
       pass
   ```
   No log, no warning. If `nfl_data_py.import_schedules` raised — for any reason — the feature would silently default to `implied_team_total=23.0`, `spread=0.0` for every row.

4. **No local cache for Vegas data.** The `schedule` table in `src/utils/database.py:220–228` stores `home_team`, `away_team`, scores, and `venue`, but **no** `spread_line`, `total_line`, or `total`. Every backtest invocation re-fetches Vegas data from `nfl_data_py` — there's no fallback if that source becomes unavailable.

5. **Existing audit didn't catch this.** `data/audit_2025_backtest_report.json::check_3_feature_distributions` audits 7–8 features per position, but **never includes `implied_team_total` or `spread`**. One Vegas-derived feature that *is* audited — `expected_point_diff` for WR — shows `train_mean=−3.964, test_mean=0.0, var_ratio=0.0` and was marked `status=OK`. A var_ratio of zero in the test set means the feature was constant at test time. That is precisely the silent-fallback signature.

## What this PR lands

Three additions, each safe in isolation:

1. **`src/features/feature_engineering.py`** — `_create_vegas_game_script_features` now logs a `warning` when the `nfl_data_py` fetch fails, and a separate `warning` when more than 50% of rows received a constant default. The bare `except Exception: pass` is replaced with `except Exception as e:` + structured logging. Behavior is otherwise unchanged.

2. **`scripts/check_vegas_features.py`** — stdlib + sqlite3-only diagnostic. Runs four checks:
   - Does `CAUSAL_FEATURES` declare Vegas features? (currently YES)
   - Does the `schedule` table cache `spread_line` / `total_line`? (currently NO)
   - Is `nfl_data_py` importable in this environment? (varies)
   - Has the silent-fallback been replaced with logging? (after this PR: YES)
   Returns exit 0 if all pass, exit 1 otherwise.

3. **This document.**

No feature engineering or schema changes. No effect on model output unless the silent fallback was already firing — in which case the warning will surface in the next backtest log.

## What needs to happen next, in order

The next steps depend on the diagnostic result in the user's environment.

### Step A — run the diagnostic
```
python scripts/check_vegas_features.py
```
Three of the four checks have known answers (YES / NO / YES-after-this-PR). The interesting check is `nfl_data_py_available`. If that fails, **Vegas features have been dead in every backtest** and the +3.2% bias / R²=0.269 baseline was achieved with `implied_team_total` constant at 23.0 — meaning Phase 1's R² lift hypothesis is intact and the next step is the cache backfill below.

### Step B — run a backtest with the new logging in place
```
python scripts/run_ts_backtest.py --season 2025
```
The new warning will fire at `WARNING` level if Vegas features defaulted on >50% of rows. If no warning fires, Vegas features were live in the baseline — and Phase 1's "add Vegas" hypothesis is wrong (the feature is already in the model). In that case, Phase 1 should pivot to **Phase 2 (opponent defense)** as defined in `docs/PREDICTIVE_CEILING_PLAN.md`.

### Step C — only if Steps A or B confirm silent degradation
Build the local cache:
1. Add `spread_line REAL`, `total_line REAL`, `total REAL` columns to the `schedule` table via `ALTER TABLE` migration in `src/utils/database.py`.
2. Update the `INSERT OR REPLACE INTO schedule` statement to include them.
3. Add `scripts/backfill_vegas_lines.py` — one-time script that calls `nfl_data_py.import_schedules` for 2018–2025 and writes spread/total to the local DB.
4. Modify `_create_vegas_game_script_features` to read from the local DB first, falling back to nfl_data_py only when the DB row is missing.
5. Re-run the backtest. Compare R² against the baseline. If R² lift ≥ +0.02 with bias still passing ±10%, Phase 1 is done. If not, the Phase 1 kill criterion fires and we move to Phase 2.

That sequence is intentionally NOT in this PR. Building the cache before knowing whether the silent fallback is actually firing is over-engineering for an environment where `nfl_data_py` works fine. The diagnostic decides the path.

## Why the diagnostic-first approach

Per `docs/PREDICTIVE_CEILING_PLAN.md` Phase 1's plan-as-written: "1.1 Add `vegas_implied_total` and `vegas_total` to feature schema." That plan was wrong about what was already in the codebase. The schema already declares them; the population code already runs. The actual bottleneck — if there is one — is the silent fallback, not the absence of wiring.

Per CLAUDE.md: "Don't add features... beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability." This PR fixes the silent fallback. It does not pre-build cache infrastructure for a problem we haven't confirmed exists.

## What this PR does NOT cover

- **Cache backfill.** Conditional on Step C confirming silent degradation.
- **R² lift verification.** Requires running the backtest in an environment with pandas/numpy/sklearn (not available where this PR was authored).
- **Other Vegas-derived features** (`expected_point_diff`, `win_probability`, `is_favorite`, `blowout_risk`). The audit's `var_ratio=0.0` finding for `expected_point_diff` suggests those have the same silent-fallback issue, but they're not in `CAUSAL_FEATURES` and are out of Phase 1 scope.

---

*Investigation only.  No new feature wiring landed.  The diagnostic and the warning are the deliverables.*
