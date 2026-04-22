# Silent-fallback audit — 2026-04-22

**Provenance:** Step 2 of the chairman's verdict in
[`council-transcript-20260422-032550.md`](../council-transcript-20260422-032550.md)
(Bucket 4 re-council).

**Motivation:** Phase 1 (Vegas) and Phase 3 (injury) both revealed that
features declared in `CAUSAL_FEATURES` were silently dead for months —
`except Exception: pass` swallowed the data-fetch failure and the
feature collapsed to a constant default, invisibly. The re-council
chairman called this the highest-EV audit on the board: "your feature
pipeline had dead branches masquerading as live ones."

**Scope:** every `except Exception: pass` (and equivalents) reachable
from `FeatureEngineer.create_causal_features()` and its transitive
callees.  Broader audit of the old `ModelBacktester` path is included
for the record but out of scope for this session's fixes.

---

## Causal feature path — findings

### 1. ✗ `feature_engineering.py:930` (`_add_team_matchup_features` — schedule lookup) — **FIXED**

Silently swallowed `nfl_data_py.import_schedules()` failure while
populating `is_divisional` and `is_primetime`. Same pattern as Phase 1
Vegas.

Neither column is currently declared in `CAUSAL_FEATURES`, so the
walk-forward model isn't training on dead inputs today — but any future
addition of either column to the feature set would silently collapse
without a warning. Converted the bare `pass` to a structured
`logging.warning(...)` that points at `scripts/backfill_vegas_lines.py`
(same data source) if the fetch fails.

### 2. ✓ `feature_engineering.py:1906` (`_merge_injury_data_from_cache` — InjuryDataLoader import) — **INTENTIONAL**

Import of `src.data.external_data.InjuryDataLoader` falls back to a
hardcoded `status_map` that exactly matches `INJURY_STATUS_SCORES` at
the source. Purpose: let the injury merge keep working even in
test/bench contexts where external_data's heavy deps aren't
installed. Not a data-silencing fallback — kept.

### 3. ✓ `feature_engineering.py:2270` (`_update_feature_columns` — leakage-filter import) — **INTENTIONAL**

Import failure on `filter_feature_columns` falls through to an
unfiltered column list. The leakage filter is enforced elsewhere (the
walk-forward `leakage_safe_features` pipeline + the runtime assertion
in `get_all_players_for_training`), so this particular helper's
failure is not a data-integrity risk. Kept.

### 4. ✓ `ts_backtester.py:688` (`_compute_decision_quality` — ModelBacktester import) — **INTENTIONAL**

Lazy import of `ModelBacktester` inside the decision-quality helper.
Failure returns `{}` and the caller documents that an empty
`decision_quality` block is a no-op. This is the opposite of a silent
fallback — the empty dict is an explicit signal. Kept.

### 5. ✓ `ts_backtester.py:1002` (`run_ts_backtest` — season auto-detect) — **INTENTIONAL**

Import failure on `src.utils.nfl_calendar` helpers falls back to
"latest available season." Season selection is diagnostic-logged by
the caller; this isn't a feature silently collapsing. Kept.

---

## Old `ModelBacktester` path — flagged, not fixed this session

Four instances of the same pattern exist outside the walk-forward
path. They're reachable only through `src/evaluation/backtester.py`'s
`ModelBacktester.run_backtest()` flow, which the walk-forward
`TimeSeriesBacktester` does not call. If anyone ever dusts off the
old path for a production run, they'll hit the same silent-fallback
trap Vegas + injury did. They're listed here so a future sweep can
convert them in one commit without re-auditing:

| File:line | Purpose | Risk if triggered |
| :--- | :--- | :--- |
| `src/evaluation/backtester.py:226` | `vegas_implied_baseline` import | Baseline comparison row omitted silently |
| `src/evaluation/backtester.py:1606` | `add_external_features` call (injury + Vegas merge) | **Old injury/Vegas merge collapses silently — exact Phase 1 / Phase 3 trap** |
| `src/data/external_data.py:342` | `sanitize_schedule_df` import in `get_weather_features` | Raw (unsanitized) schedule merged without warning |
| `src/data/external_data.py:551, 556` | `sanitize_schedule_df` import + outer schedule-fetch fallback in `load_vegas_lines` | Vegas lines default to empty DataFrame silently |

Not fixed today because: (a) the re-council's scope was the causal
feature path, which is the production walk-forward; (b) the old
`ModelBacktester` code path is not exercised by any current workstream;
(c) blast-radius-to-value is lower than finishing Steps 3-5.

Recommendation: fold into the "clean up legacy ModelBacktester" work
alongside any future ensemble-walk-forward migration.

---

## Intentional patterns left in place

Single-row `try/except/pass` loops inside batch inserts (e.g.
`src/utils/database.py:1192`, `:1232`) skip individual malformed rows
without halting the batch. Idiomatic and not in scope.

---

## Verification

After the `feature_engineering.py:930` fix, the relevant test suites
remain green:

```
pytest tests/test_backtester.py tests/test_ts_backtester.py \
       tests/test_backtest_validation.py::TestWalkForwardBiasRegression
# 32 passed
```

The new warning message fires in the existing sandbox (habitatring.com
blocked), which is the condition the audit is defending against.

---

## Phase 1 re-labeling (chairman's explicit ask)

The chairman asked for Phase 1's **"+0.004 R²"** to be re-labeled in
the workstream record as a **bug-fix**, not a feature. Lands alongside
this commit in `docs/PHASE_1_VEGAS_FINDINGS.md` (header status note)
and `CRITICAL_LIMITATION.md` (2026-04-22 status block appended).

The re-council's framing: Phase 1's lift was Vegas inputs finally
reaching the model after months of silent default — not a new feature
landing. This is load-bearing context for whoever reads the 69.8 %
cross-season hindsight number next; they should know most of the lift
came from fixing dead pipelines, not from adding signal.

---

## Next in the chairman's sequence

1. ~~Bootstrap the 43-week record.~~ ✓ done. `docs/BOOTSTRAP_H2H_20260422.md`.
2. ~~Audit + re-label.~~ ✓ done (this doc).
3. **Prospective replay against lock-time-only opponent construction.** ← next.
4. Pre-register Phase 4 decision rule.
5. Forward paper-trade harness, 8-12 weeks.
