# Forward paper-trade protocol — 2026-04-22

**Provenance:** Step 5 of the chairman's verdict in
[`council-transcript-20260422-032550.md`](../council-transcript-20260422-032550.md)
(Bucket 4 re-council). Chairman's requirement:

> Stand up a forward paper-trade harness with frozen features, locked
> lineup-lock entries, logged opponent pools, and a pre-committed
> stop-loss. Run for 8-12 weeks before any real-money scaling
> decision. Success signal: live sample win rate tracks
> prospective-replay win rate within CI; no silent drift.

**Scope of this document:** the protocol only. The harness code
itself is deferred — today is 2026-04-22, the 2026 NFL regular season
kicks off ~2026-09-10, so nothing runs for ~20 weeks. The protocol
must exist first so the harness is built against a committed spec,
not reverse-engineered from a running system. This matches the
lesson from Phase 1 and Phase 3: pre-written behavior contracts
prevent silent drift.

---

## What is being tested

The question this paper-trade answers is **operational**, not
statistical: *does the backtested edge survive contact with live
data the way the prospective-replay said it would?*

The bootstrap (`docs/BOOTSTRAP_H2H_20260422.md`) cleared the
43-week statistical gate. The prospective replay
(`docs/PROSPECTIVE_REPLAY_20260422.md`) showed the edge grows when
the opponent's retrospective filter is denied. Phase 4
pre-registration (`docs/PHASE_4_PREREGISTRATION.md`) locks the rule
for the ensemble upgrade. Step 5 is the bridge from "backtest says
so" to "a real week's data doesn't surprise us."

## Invariants — the feature freeze

From the moment the paper-trade starts until the stop-loss fires or
the 12-week window closes, the following files are **frozen**. Any
edit to these during the paper-trade invalidates the sample and
restarts the clock.

- `config/settings.py::CAUSAL_FEATURES`
- `config/settings.py::RIDGE_DEFAULT_ALPHA`
- `config/settings.py::DECISION_QUALITY`
- `src/features/feature_engineering.py::create_causal_features` and
  everything it transitively calls
- `src/evaluation/ts_backtester.py::_EnsembleModelWrapper` and
  `default_model_factory` (if either model type runs in the paper
  trade)
- `src/evaluation/backtester.py::backtest_lineup_decisions` (the
  opponent-tier logic is frozen — if Step 3's fully-symmetric
  replay ever gets built, that's a separate branch, not an in-flight
  change)

**Allowed during the paper-trade:**
- Weekly `ensure_team_defense_stats(season)` backfills.
- Weekly `scripts/backfill_injuries.py` runs.
- Weekly `scripts/backfill_vegas_lines.py` runs.
- Schema-compatible additions to the paper-trade log itself.
- Bug fixes in code strictly outside the prediction / decision-quality
  paths (e.g. UI, CLI ergonomics, tests of unrelated modules).

Borderline / requires explicit re-council:
- Any new feature added to `CAUSAL_FEATURES`. Equivalent to starting
  a new phase — reset the 8-12 week clock.
- Any change to `RIDGE_DEFAULT_ALPHA` or ensemble config. Same reset.
- Any new opponent construction (oracle / hindsight / replacement /
  prospective). Same reset.

## Weekly harness — what runs, what gets logged

Every Wednesday of the 2026 NFL regular season (standard lineup-lock
day for Sunday slates; adjust for Thursday slates separately), the
harness runs three steps:

### 1. Fetch-and-backfill

```
python scripts/backfill_injuries.py --seasons 2026 2026
python scripts/backfill_vegas_lines.py --seasons 2026 2026
python -c "from src.utils.database import DatabaseManager; DatabaseManager().ensure_team_defense_stats(2026)"
```

Each must succeed (non-zero row count from the verification block)
or the lock for that week aborts. A single failed backfill is a
logged event, not a kill-gate by itself.

### 2. Lineup lock (at a committed wall-clock time)

A new script, `scripts/paper_trade_lock.py` (to be written before
2026 week 1), emits:

- Full model prediction set for all rostered players that week
  (not just starters — the whole pool at prediction time).
- Model lineup selected by the existing optimizer in
  `src/optimization/lineup_optimizer.py::LineupOptimizer.optimize_lineup`
  at `strategy="cash"`.
- Opponent lineup constructed by the **prospective** method
  (`scripts/prospective_opponent_replay.py`'s pool +
  `DEFAULT_ROSTER_SLOTS`).
- Notional entry — $10 at 1.8× payout, purely for bookkeeping. No
  money moves during the paper trade.
- Lock timestamp (UTC).

### 3. Post-slate scoring (Monday morning)

Same script with `--score` flag:

- Fetches final fantasy points for all locked players.
- Computes model_actual, opponent_actual, win flag, weekly ROI.
- Appends to the paper-trade log (schema below).
- Emits a one-line summary to stdout + the weekly log file.

## Data schema

A new SQLite table in `data/nfl_data.db`:

```sql
CREATE TABLE IF NOT EXISTS paper_trade_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    lock_timestamp TEXT NOT NULL,              -- ISO 8601 UTC
    lock_git_sha TEXT NOT NULL,                -- HEAD at lock time
    model_config_json TEXT NOT NULL,           -- ridge_alpha, feature_mode, etc.
    model_lineup_json TEXT NOT NULL,           -- 6 players, positions, salaries, predicted
    opponent_lineup_json TEXT NOT NULL,        -- 6 players, opponent method
    opponent_method TEXT NOT NULL,             -- "prospective" | "hindsight" | ...
    notional_entry_usd REAL NOT NULL,          -- e.g. 10.0
    notional_payout_multiplier REAL NOT NULL,  -- e.g. 1.8
    model_actual REAL,                         -- NULL until scored
    opponent_actual REAL,                      -- NULL until scored
    won INTEGER,                               -- NULL until scored (0/1)
    score_timestamp TEXT,                      -- ISO 8601 UTC or NULL
    notes TEXT,                                -- free-form
    UNIQUE(season, week)
);
```

Git SHA at lock time is mandatory — it's the single source of truth
for "what code produced this entry." If a stop-loss fires, the SHA
is how we recover the exact state that was in play.

## Stop-loss — pre-committed, no narrative rescue

The paper-trade halts automatically and requires explicit re-council
approval to resume if ANY of the following conditions fires during
the run. These are locked before the run starts.

### Statistical stop-loss

After week N (where N ≥ 4):

- **Wilson 95 % lower bound on running win rate < 52.38 %** (-110
  break-even). The cumulative record must stay statistically
  distinguishable from sub-break-even.
- **Running win rate drops > 3 pp below the prospective-replay
  baseline of 83.72 %**, measured on a trailing-4-week window. This
  is the chairman's explicit "no silent drift" gate.

### Record-based stop-loss

- **4 consecutive losses** at any point. Hard stop. Review the four
  losing weeks for a common failure mode before any resume.

### ROI stop-loss

- **Cumulative notional ROI < 0 % after week 6.** The ROI target is
  +25.6 % per the Ridge baseline; a negative cumulative ROI after
  six weeks is two standard deviations below expectation.

### Integrity stop-loss

- **Any backfill step failed for > 1 week without manual
  remediation.** Running the model on stale data is the exact
  silent-degradation failure mode Phase 1 / 3 caught.
- **Any change to the frozen files listed above** without a
  preceding "reset the clock" decision. The harness's pre-lock hook
  should refuse to run if `git diff <lock-sha-of-week-1>..HEAD`
  touches a frozen path.

## Review cadence

- **Weekly (Monday after games):** one-line log entry written
  automatically. No human decision required unless a stop-loss
  fires.
- **Every 2 weeks:** human glance at the log. Look for silent drift
  patterns the automatic gates wouldn't catch (e.g., three straight
  weeks of wins against hindsight and losses against prospective —
  suggests the opponent construction is miscalibrated).
- **Week 8 (halfway):** go / no-go checkpoint. Is the running
  record within CI of the 83.7 % prospective baseline? If yes,
  continue. If ambiguous, extend to week 12 and make the call then.
  If failing, stop-loss by rule — not by judgment.
- **Week 12 (end):** final disposition. Three possible outcomes:
  1. **Ship to real-money with a scaling ramp.** Running record is
     within CI of prospective baseline AND Wilson LB stays above
     break-even. Start small — $10 → $25 → $50 stakes across 4
     weeks, same stop-loss rules.
  2. **Extend the paper-trade.** Running record is ambiguous (in
     the re-council band of `(66.28 %, 72.09 %)`). Extend 4 more
     weeks.
  3. **Kill the deployment plan.** Running record collapsed below
     prospective baseline or tripped a stop-loss. Re-council on
     what changed between backtest and live.

## What the paper-trade does not test

- **Phase 4 ensemble.** If Phase 4 is run during paper-trade weeks,
  its ship signal is gated on the pre-registration in
  `docs/PHASE_4_PREREGISTRATION.md`, not on the paper-trade. If
  Phase 4 passes, the paper-trade clock resets with the new model.
- **Real contest variance.** Opponent modeling by the prospective
  method is a proxy. A real DFS contest's payout distribution has
  ties, splits, minimum fills, and rake we aren't modeling. The
  paper-trade establishes that the backtested edge survives live
  data; a real-money shadow test is a separate step after ship.
- **Market adaptation.** If the 2026 season's scoring environment
  drifts materially from 2024-2025 (rule changes, pace shifts,
  regime shift), the paper-trade will catch that as a stop-loss but
  won't diagnose it. Diagnosis is a separate workstream.

## What needs to exist before 2026 week 1

Checklist, in order, with the best-effort schedule:

1. **Schema migration** (`data/nfl_data.db` → `paper_trade_entries`).
   Analogous to the Phase 3 `player_injuries` migration. ~30 min.
2. **`scripts/paper_trade_lock.py`** with `--lock` and `--score`
   modes. ~1-2 days of coding; pulls from existing optimizer +
   prospective opponent replay. Mostly composition.
3. **Pre-lock `git diff` hook** that refuses to run if any frozen
   path has been modified since week 1. ~1 hour.
4. **Backfill cron** (or manual weekly runbook) for injuries +
   Vegas + team_defense_stats. Exists as scripts; just needs
   scheduling.
5. **Dry run** in late August 2026 with a 2025 replay slate, to
   smoke-test the harness end-to-end before real weeks begin. Half
   a day.

None of these blocks ship as of today's commit. They block the week-1
lock in September.

## Chairman's sequence status after Step 5

1. ~~Bootstrap the 43-week record.~~ ✓ done — `docs/BOOTSTRAP_H2H_20260422.md`
2. ~~Silent-fallback audit + Phase 1 re-label.~~ ✓ done — `docs/SILENT_FALLBACK_AUDIT_20260422.md`
3. ~~Prospective opponent replay.~~ ✓ done — `docs/PROSPECTIVE_REPLAY_20260422.md`
4. ~~Phase 4 pre-registration.~~ ✓ done — `docs/PHASE_4_PREREGISTRATION.md`
5. ✓ **Protocol spec committed (this doc). Harness code is a ~2-day
   build that can land any time between now and late August 2026.**

All five of the chairman's Monday-morning steps are discharged. The
2026-04-22 re-council is closed.
