# Bias Attribution: +44% (Feb 15) → +3.2% (Apr 10)

**Council:** April 14, 2026 partial council — Step 4
**Provenance:** [`council-transcript-20260414-034617.md`](../council-transcript-20260414-034617.md)
**Reference:** [`CRITICAL_LIMITATION.md`](../CRITICAL_LIMITATION.md)

## TL;DR

Of the **40.8 percentage points** of bias collapse between the Feb 15 backtest (+44%) and the Apr 10 walk-forward (+3.2%):

| Mechanism | Estimated pp | Confidence |
|---|---:|---|
| 1. Leakage fix (incl. walk-forward infra) | **~38** | high |
| 2. Training window 2018+ start | ~1–2 | low (inferred) |
| 3. Causal feature mode + per-fold features | ~1–2 | medium |
| 4. Weekly refit | (counted in #1) | — |
| **Aggregate residual** | **~0–2** | within noise |

**The leakage fix is dominant by an order of magnitude.** The first walk-forward run AFTER the leakage fix (Apr 9 18:57) was already at +2.7% bias — within 0.5pp of the final +3.2%. Everything after the leakage fix was tuning around the same calibration plateau.

---

## Anchor backtests

| Run | Date | Configuration | Overall bias |
|---|---|---|---:|
| `backtest_2025_20260214.json` | Feb 14 | Pre-leakage-fix, single split | +5.4% (QB +44.9%) |
| `backtest_2025_20260215.json` | Feb 15 | Pre-leakage-fix, single split | **+44%** (TE +80%, WR +77%) |
| `ts_backtest_2025_20260409_185710.json` | Apr 9 18:57 | Walk-forward, post-leakage-fix, causal features | +2.7% |
| `ts_backtest_2025_20260409_230648.json` | Apr 9 23:06 | Walk-forward, intermediate feature config | −13.0% (likely misconfig) |
| `ts_backtest_2025_20260409_234501.json` | Apr 9 23:45 | Walk-forward, refined features | +2.7% |
| `ts_backtest_2025_20260410_004640.json` | Apr 10 00:32 | **Final** (Ridge α=1.0, causal features) | **+3.2%** |

The leakage fix landed in commit `03dcd27` (Apr 9 18:50 UTC). The first walk-forward run started 7 minutes later. The 41.3pp bias drop happened in those 7 minutes.

## Mechanism 1 — Leakage fix (~38pp, high confidence)

Commits: `61e6eb7`, `f1283b5`, `f5ba669`, `03dcd27`.

Three independent lines of evidence converge:

1. **Direction matches.** Pre-fix bias was uniformly positive (+44% overall, +77% WR, +80% TE). Future-leakage in rolling-window features (a player's "rolling 3-week receiving yards" computed across the *full season* before the train/test split) inflates training labels for early-season weeks — exactly the failure mode that produces uniform over-prediction at inference.

2. **Position pattern matches.** TE/WR had the largest pre-fix biases. These positions have the highest game-to-game variance in target/yards (a single 150-yd game is a far larger fraction of a TE's season output than a similar game for an RB). Future-leakage of season-end blowups hits TE/WR hardest, which is what we observe.

3. **Audit findings match.** [`data/audit_2025_backtest_report.json`](../data/audit_2025_backtest_report.json) (`check_5_naive_baseline`) showed the pre-fix model performing *worse than the naive baseline* — the signature of a contaminated training pipeline. The same audit flagged 7 zero-width and 4 full-range percentile bounds, which fall back to rank-based features computed against the test set.

## Mechanism 2 — Training window 2018+ (~1–2pp, low confidence, inferred)

Commit: `c0c0d50` "Address non-stationarity: training start 2018, scoring shift detection."

NFL passing efficiency drifted modestly between 2014 (peak passing era) and 2024. Training on 2006–2017 data biases predictions upward for post-2018 weeks because per-snap pass yardage was modestly higher pre-2018. Cutting training to 2018+ removes this pull.

Estimate is small because: (a) the per-snap shift is fractional fantasy points, (b) the leakage fix bundle already accounts for ~38pp leaving only 2–4pp to distribute, (c) this mechanism cannot be tested without an explicit ablation.

## Mechanism 3 — Causal features + per-fold features (~1–2pp, medium confidence)

Commits: `26128eb`, `7eacf52`, `5c2ba2d`, `7fd35e3`, `5fef9e4`, `5aecfed`.

The reduction from 100+ features to 7–8 causal features per position is variance-reducing, not bias-reducing in expectation. But two specific changes had directional bias effects:

- `5c2ba2d` "Remove TD regression double-shrinkage" → removed a downward pressure on QB TD predictions; pushes QB bias slightly up.
- `5aecfed` / `52722e1` Huber loss δ raised 1.0 → 5.0 → only affects ensemble runs (XGBoost/LightGBM use Huber); **does NOT affect the Apr 10 Ridge run**.

The spread between the four post-leakage-fix walk-forward runs (which differ only in feature config) was ±0.5pp around +2.95%, excluding the misconfigured −13% run. So feature-config refinement after the leakage fix accounts for at most ~1pp of the final +3.2%.

## Mechanism 4 — Weekly refit (counted in Mechanism 1)

Commits: `f5ba669`, `0f97255`, `0aa359f`.

Weekly refit was bundled with the leakage fix (`f5ba669` and `03dcd27` were merged together as PR #83). The walk-forward design was the *vehicle* for fixing the leakage — the old single-split design was the leakage substrate. They cannot be cleanly separated from existing artifacts; both effects are inside Mechanism 1's ~38pp.

## Residual

Aggregate level: **~0pp residual**. The Mechanism 1 anchor (+2.7% post-leakage-fix, first run) is within 0.5pp of the final +3.2%; the gap is fully explained by ~1pp from feature refinement plus ~1pp from training-window inference plus measurement noise.

Per-position level: **larger residuals**, not closed by this attribution. Notably:
- RB went from −11.4% to −5.3% (only 6pp change; RB pre-fix was UNDER-predicting, unlike all other positions).
- QB went from +8.6% to +2.1% (6.5pp; QB had a +44.9% spike on Feb 14 that already collapsed to +8.6% by Feb 15 — single-split instability).
- WR went from +77% to +7.5% (70pp; consistent with leakage dominance).
- TE went from +80% to +4% (76pp; same).

## What would close the per-position residual

Three additional backtests, each toggling exactly one mechanism off the final Apr 10 config:

1. **Pre-leakage-fix anchor reproduction** — re-run with current data on the Feb 15 single-split design. Expected: bias near +44%. Confirms leakage as dominant.
2. **Training-window ablation** — walk-forward + leakage-fixed + 2006-start. Expected: +1 to +2pp shift positive.
3. **Feature-mode ablation** — walk-forward + leakage-fixed + 2018+ + 100+ feature original config. Expected: ±1pp shift.

Estimated effort: ~3 hours of backtest runtime + analysis. Ablation flags (`--training-start-year`, `--feature-mode`) would need to be added to `scripts/run_ts_backtest.py` first; the alpha sweep landed in commit `b7b579e` is the template.

## What this attribution does NOT cover

- **Variance-side analysis** — captured separately in Step 3 (`scripts/analyze_std_ratio.py`).
- **R² changes** — overall R² went from −0.685 to +0.269; the ranking inversion (Spearman ρ < 0 for QB) was also resolved by the leakage fix.
- **Interaction effects** between the four mechanisms — they may not be additive. The ablation series above would catch this.

---

*This attribution is a reasoned reconstruction from commit history, the audit report, and the backtest result files on disk. The aggregate residual is within noise. Per-position residuals are NOT closed and should not be claimed as resolved without the three follow-up backtests.*
