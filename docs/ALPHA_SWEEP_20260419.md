# Ridge α Sweep — 2026-04-19

**Branch:** `claude/add-council-status-rfitZ`
**Council step:** 2026-04-14 Step 3 — Ridge α sensitivity sweep
**Status:** Complete, with a methodological correction that changes the conclusion.

---

## What was asked

From `council-transcript-20260414-034617.md`:

> Run a Ridge α sensitivity sweep over `{0.3, 1, 3, 10}` per position on the existing walk-forward fold structure. … Success signal: the QB/TE 7–11pt `(std_ratio − r)` gap is either reproduced across folds with a tight standard error (in which case authorize per-position α tuning as the principled fix), or it dissolves (in which case the gap was an artifact and no fix is warranted).

Kill criterion:

> If the α sensitivity sweep (Step 3) shows the QB/TE `(std_ratio − r)` gap is unstable across fold splits or has standard error larger than the gap itself, the per-position α tuning plan is invalid — re-council before authorizing any QB/TE-specific intervention.

## What was run

Season 2025, expanding-window Ridge backtest, α ∈ {0.3, 1, 3, 10}, one confirmatory run at α=10000. Output: `data/backtest_results/alpha_sweep_summary.json` and per-α `ts_backtest_2025_*_predictions.csv`.

## Headline result

**The sweep as designed is invalid — α ∈ {0.3, 1, 3, 10} is a no-op on this model's feature matrix.** Gaps, MAE, RMSE and R² are identical across all four values to within 1×10⁻³. A single confirmatory run at α=10000 moves every metric materially, proving sensitivity exists — just not in the swept range.

### Per-position gap (`std_ratio − r`)

| Position | α=0.3   | α=1.0   | α=3.0   | α=10.0  | α=10000 |
| :------: | :------:| :------:| :------:| :------:| :------:|
| QB       | +0.111  | +0.111  | +0.111  | +0.111  | **−0.010** |
| RB       | −0.006  | −0.006  | −0.006  | −0.006  | **−0.094** |
| TE       | +0.074  | +0.074  | +0.074  | +0.074  | **−0.015** |
| WR       | +0.001  | +0.001  | +0.001  | +0.001  | **−0.050** |

### Why {0.3, 1, 3, 10} was invisible

Features are standardized upstream of the model fit (`StandardScaler.fit_transform` at `src/evaluation/ts_backtester.py:370`). For standardized X with N samples per fold, Ridge coefficient shrinkage is approximately `N / (N + α)`. Per-position per-fold sample sizes in this run:

- QB: 644 (expanded over 21 weeks, so per-fold much larger than 644 when summed)
- TE: 373 per season — smallest pool
- RB: 1181, WR: 3397

With N ≥ 373, α = 10 gives shrinkage factor 373 / 383 = 0.974; α = 0.3 gives 373 / 373.3 = 0.999. The entire {0.3..10} range changes coefficients by < 3%. That's why bias, std_ratio, correlation, and the gap barely budge.

The meaningful α regime for this feature matrix begins roughly where `α ≈ N`, i.e., 10² – 10⁴. At α = 10000 coefficients shrink by `~(N / (N + 10000)) ≈ 0.04–0.25` and the gap flips sign on all four positions.

## Verdict against council thresholds

Applying the council's Step 3 thresholds to the **real-α-sensitive** values:

- **QB gap:** +0.111 at α=1.0, −0.010 at α=10000. Not stable across α — flips sign. A zero-crossing exists in `1 < α* < 10000`. Finding the per-position zero-crossing is a meaningful tuning problem.
- **TE gap:** +0.074 at α=1.0, −0.015 at α=10000. Same pattern, zero-crossing in the same range. Current per-week SE on TE (~0.08) is still larger than |gap| at α=1.0, so TE hits the kill criterion as stated ("SE > |gap_mean|").
- **RB / WR:** Already near zero at α=1.0 (identity holds). At α=10000 they become over-shrunk (−0.094, −0.050). α=1.0 is closer to optimal for these two.

**Net:** the original sweep range can't answer the council's question. A re-sweep at α ∈ {10, 100, 1000, 10000} per position is needed to find the zero-crossing for QB and TE.

## Bonus finding: decision quality diverges from regression loss

The `decision_quality` block added in commit `7a65915` produced the first-ever cash-H2H walk-forward numbers as a side effect of this sweep:

| Config         | Overall R² | vs Hindsight win rate | p-value | ROI     |
| :------------- | :--------: | :-------------------: | :-----: | :-----: |
| α ∈ {0.3..10}  | 0.269      | 61.9 % (13-8)         | 0.192   | +11.4 % |
| α = 10000      | 0.263      | **71.4 % (15-6)**     | **0.039** | **+28.6 %** |

α=10000 is **worse** on point-estimate R² (−6 bp) but **better** and statistically significant on lineup decision quality. This is exactly the dissociation the 2026-03-31 council predicted when they called "optimize the goal, not the proxy" the single most important fix. The 21-week sample is too small to treat this as a production recommendation, but it is the first empirical evidence that minimizing prediction loss is *not* the same as maximizing win rate for this model.

## Recommended next step

Either:

1. **Re-run Step 3 properly** — sweep α ∈ {10, 100, 1000, 10000, 100000} per position, locate each position's `std_ratio − r = 0` crossing, and validate the result on a second season (2024) to guard against a single-season fluke. This is the honest completion of the council's request.
2. **Re-council (Bucket 4)** — the original advisor framing ("7–11pt QB/TE gap at α=1.0") rests on an α that is effectively zero regularization here; the decision-quality dissociation is new information the 2026-04-14 council didn't weigh. A partial council (Statistician + Executor) should re-scope.

**Kill criterion fired:** per the council's own Step 3 language ("If the α sensitivity sweep ... has standard error larger than the gap itself, the per-position α tuning plan is invalid — re-council before authorizing any QB/TE-specific intervention"), TE hits this at α=1.0 (SE 0.08 > gap 0.074). **Do not start per-position α tuning without re-council.**

## Artifacts

- Sweep run CSVs: `data/backtest_results/ts_backtest_2025_2026041{9_045913,9_051533,9_053229,9_054624}_predictions.csv` (α = 0.3, 1, 3, 10 in order of run)
- Confirmatory α=10000 run: `data/backtest_results/ts_backtest_2025_20260419_060400_predictions.csv`
- Machine-readable summary: `data/backtest_results/alpha_sweep_summary.json`
- First walk-forward `decision_quality` blocks embedded in each run's `*.json`
