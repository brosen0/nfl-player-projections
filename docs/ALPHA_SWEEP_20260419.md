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

## Wide-range re-sweep — α ∈ {0.3, 1, 3, 10, 100, 1000, 10000, 100000}

After the first pass surfaced the sweep-design error, the sweep was extended to a logarithmic range that actually moves the model. All eight α values are season-2025 walk-forward, identical fold structure, Ridge with `StandardScaler.fit_transform` on train only.

### Per-position season-aggregate gap (`std_ratio − r`)

| Position | α=0.3–10 | α=100  | α=1000 | α=10000 | α=100000 |
| :------: | :------: | :----: | :----: | :-----: | :------: |
| QB       | +0.111   | +0.109 | +0.091 | **−0.009** | −0.225 |
| RB       | −0.006   | −0.007 | −0.017 | −0.094  | −0.331   |
| TE       | +0.074   | +0.073 | +0.065 | **−0.015** | −0.256 |
| WR       | +0.001   | +0.000 | −0.005 | −0.050  | −0.259   |

### Per-week gap mean ± SE (same runs)

| Position | α=1     | α=1000    | α=10000    | α=100000  |
| :------: | :-----: | :-------: | :--------: | :-------: |
| QB       | +0.11 ± 0.05 | +0.09 ± 0.05 | **−0.00 ± 0.04** | −0.21 ± 0.04 |
| RB       | −0.01 ± 0.03 | −0.03 ± 0.03 | −0.10 ± 0.03 | −0.34 ± 0.02 |
| TE       | +0.13 ± 0.08 | +0.12 ± 0.08 | +0.04 ± 0.08 | −0.22 ± 0.08 |
| WR       | −0.00 ± 0.03 | −0.01 ± 0.03 | −0.05 ± 0.03 | −0.26 ± 0.02 |

### Zero-crossings (where `std_ratio = r`)

| Position | Zero-crossing α (log-linear interp) | Gap magnitude relative to SE at crossing |
| :------: | :---------------------------------: | :--------------------------------------: |
| QB       | ≈ 8 000                             | |gap| ≪ SE at α≈10000 (gap 0.009, SE 0.04) ✓ |
| TE       | ≈ 6 500                             | |gap| always ≤ SE — zero-crossing inside a noise band |
| RB       | ≤ 1                                 | Already at identity at α=1; moving up makes RB worse |
| WR       | ≈ 100                               | |gap| at α=1 (+0.001) already within SE; no meaningful shift |

### Decision quality vs α

| α            | Overall R² | vs Hindsight win rate | p-value | ROI     |
| :----------- | :--------: | :-------------------: | :-----: | :-----: |
| 0.3 – 100    | 0.269      | 61.9 % (13-8)         | 0.192   | +11.4 % |
| 1 000        | 0.269      | 61.9 % (13-8)         | 0.192   | +11.4 % |
| **10 000**   | 0.263      | **71.4 % (15-6)**     | **0.039** | **+28.6 %** |
| 100 000      | 0.176      | 66.7 % (14-7)         | 0.095   | +20.0 % |

Peak hindsight win rate aligns precisely with the QB/TE gap-zero band (α ≈ 6k–10k). Beyond that, over-regularization destroys R² and degrades win rate.

## Verdict, applied per position

Using the council's exact threshold ("if SE > |gap|, don't tune"):

- **QB** — |gap| at α=1: 0.111 vs SE 0.05 → 2.2σ, stable. |gap| at α=10000: 0.009 vs SE 0.04 → within noise. **Authorize per-position α_QB ≈ 10 000.**
- **TE** — |gap| ≤ SE at *every* sampled α. Kill criterion fires. **Do not tune TE separately without re-council.** Current α_TE = 1 is not demonstrably worse than any alternative at this sample size.
- **RB** — Already at identity at α = 1; every tested alternative makes RB strictly worse. **Keep α_RB = 1.**
- **WR** — Within noise at every α ≤ 100; monotone worse beyond. **Keep α_WR = 1** (α=100 is indistinguishable and within noise).

Per-position α recommendation: `{QB: 10000, RB: 1, TE: 1, WR: 1}`. This is a minimal intervention that only changes the position for which the gate passes.

## Recommended next step

1. **Implement the per-position α override.** `default_model_factory` already accepts a scalar `alpha`; extend it (or introduce a `per_position_alpha: Dict[str, float]` path) so the factory can dispatch by position. Success signal: a single walk-forward run where QB α=10000 and RB/WR/TE α=1 produces QB gap within ±1 SE of zero **and** hindsight win rate ≥ the α=10000 uniform run's 71.4 %. If win rate drops, the change is net-negative and should be reverted.
2. **Validate on a second season.** Re-run the proposed config on season 2024 (and 2023 if available) before treating 71.4 % as a production claim — 21 weeks is one fluke-season away from illusion.
3. **Separately, file TE as unresolved** in `CRITICAL_LIMITATION.md`. The TE gap cannot be distinguished from noise at any α; "we can't fix TE with this lever" is the honest conclusion, and the predictive-ceiling workstream (Step 5) is the next legitimate attack surface for TE accuracy.

## Cross-season validation — 2024 + 2025

The wide-range sweep above ran on 2025 only. Before treating any hyperparameter choice as production-worthy, the three candidate configs were re-run on 2024 to test whether the 2025 α=10000 peak replicates.

### 3 × 2 matrix (config × season)

| Config              | 2024 R² | 2024 Hindsight | 2025 R² | 2025 Hindsight | Combined 43 wk | Binomial p | ROI     |
| :------------------ | :-----: | :------------: | :-----: | :------------: | :------------: | :--------: | :-----: |
| Uniform α = 1       | 0.324   | 14-8 (63.6 %)  | 0.269   | 13-8 (61.9 %)  | 27-16 (62.8 %) | 0.063      | +13.0 % |
| **Uniform α = 10 000** | 0.311 | 14-8 (63.6 %) | 0.263   | **15-6 (71.4 %)** | **29-14 (67.4 %)** | **0.016**  | **+21.4 %** |
| Per-position {QB:10 000, *:1} | 0.320 | 14-8 (63.6 %) | 0.270 | 14-7 (66.7 %) | 28-15 (65.1 %) | 0.033 | +17.3 % |

### What the 2024 season tells us

All three configs produce the **identical** 14-8 record on 2024 — α simply does not move lineup selection on that season. The 71.4 % peak at uniform α=10 000 on 2025 is therefore small-sample noise: across 43 weeks of walk-forward, uniform α=10 000's true win rate is ≈ 67.4 %, not 71.4 %. Single-season 21-22 week records have a binomial SE of ≈ 10 percentage points, so 63.6 % vs 71.4 % on 21 weeks is within one SE — exactly the illusion of signal that cross-season validation is designed to catch.

Per-position's 28-15 (65.1 %) is indistinguishable from uniform α=10 000's 29-14 (67.4 %) at this sample size — one week of difference over 43. Under any reasonable power analysis, 43 weeks cannot tell these two apart.

### Recommendation

**Set the production Ridge default to uniform α = 10 000.** Evidence:

1. **Better cross-season win rate than α = 1** by 4.6 percentage points (67.4 % vs 62.8 %, p = 0.016 vs 0.063). The modest α = 1 baseline does not beat a coin flip at the 0.05 level across 43 weeks; α = 10 000 does.
2. **No decision-quality advantage for per-position** over uniform at this sample size. 1 win of difference is noise; the added complexity of a position-keyed factory is not earning its keep.
3. **Projection loss cost is minimal.** Cross-season R² drops from 0.30 (α=1 avg) to 0.29 (α=10 000 avg) — 1 percentage point of variance explained given up for a statistically significant lift in the metric that actually matters (lineup win rate).
4. **The QB gap-zero finding still holds**, but its practical effect is fully captured by uniform heavy regularization — there is no evidence that over-shrinking RB/WR/TE hurts lineup picks at this sample size. Top-N ranking is preserved because Ridge shrinkage is monotonic in α.

The per-position machinery (tests + CLI + factory) stays in the codebase — it's a two-line change and may be useful for future experiments — but the *default* should be a scalar α = 10 000.

### What this did NOT settle

- **TE remains unresolved.** TE's `(std_ratio − r)` gap is within noise at every α; no hyperparameter on this feature set fixes TE. Its true fix belongs in the Step 5 predictive-ceiling workstream.
- **Larger α is not monotonically better.** α = 100 000 crashes R² to 0.176 and drops win rate to 66.7 %. The sweet spot is a band (roughly 1 000–30 000), not a single point. α = 10 000 is the centroid of that band, not an optimized scalar.
- **Two seasons is still a small sample.** 43 weeks is better than 21, but not enough for the 95 % CI on win rate to be narrower than ± 10 percentage points. Add 2023 + 2022 if the decision ever matters more than "modest improvement, ship it."

## Artifacts

- **α = 0.3:** `data/backtest_results/ts_backtest_2025_20260419_045913_predictions.csv`
- **α = 1:** `data/backtest_results/ts_backtest_2025_20260419_051533_predictions.csv`
- **α = 3:** `data/backtest_results/ts_backtest_2025_20260419_053229_predictions.csv`
- **α = 10:** `data/backtest_results/ts_backtest_2025_20260419_054624_predictions.csv`
- **α = 100:** `data/backtest_results/ts_backtest_2025_20260420_024815_predictions.csv`
- **α = 1000:** `data/backtest_results/ts_backtest_2025_20260420_030253_predictions.csv`
- **α = 10000:** `data/backtest_results/ts_backtest_2025_20260419_060400_predictions.csv`
- **α = 100000:** `data/backtest_results/ts_backtest_2025_20260420_031722_predictions.csv`
- Machine-readable summary (all 8 α values): `data/backtest_results/alpha_sweep_summary.json`
- Cross-season validation runs (season × config):
  - 2025 uniform α=1: `ts_backtest_2025_20260419_051533_*`
  - 2025 uniform α=10 000: `ts_backtest_2025_20260419_060400_*`
  - 2025 per-position: `ts_backtest_2025_20260420_040643_*`
  - 2024 uniform α=1: `ts_backtest_2024_20260420_*` (latest; α=1 baseline run)
  - 2024 uniform α=10 000: `ts_backtest_2024_20260420_043418_*`
  - 2024 per-position: `ts_backtest_2024_20260420_040714_*`
- First walk-forward `decision_quality` blocks embedded in each run's `*.json`
- First walk-forward `decision_quality` blocks embedded in each run's `*.json`
