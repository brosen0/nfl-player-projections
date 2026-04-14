# LLM Council Transcript — 2026-04-14 03:46:17 UTC

**Question:** Now that the leakage fix has collapsed the prediction bias from +44% to ±8%, and `CRITICAL_LIMITATION.md` shows the apparent variance compression is correlation arithmetic rather than shrinkage, what should we do with the post-prediction calibration gate the April 1 council recommended (built 2026-04-09, disabled 2026-04-10 for suppressing breakouts)?

**Session type:** Bucket 4 partial council — 2 advisors (Statistician + Executor), 2 peer reviewers (Contrarian + First Principles Thinker), 1 chairman synthesis.

---

## Chairman's Verdict

### Bottom Line
**Do:** Delete the calibration gate code path entirely (Option 1) and redirect the same engineering hours at the r ≈ 0.52 correlation ceiling, which is the actual loss center.
**Don't:** Rebuild it asymmetrically, replace it with a Vegas-total ceiling, or re-enable it at looser thresholds — every variant defends against a +44% bias that no longer exists and risks compounding the mathematically-correct compression on RB/WR.

### Critical Next Steps
1. **DO TODAY:** Delete `_apply_calibration_gate` from `src/models/ensemble.py`, the `CALIBRATION_GATE_*` constants from `config/settings.py`, and any tests that assert on the gate. Time: ~1–2 hours. Success signal: `grep -ri calibration_gate` returns zero hits in `src/`, `config/`, and `tests/`, and `pytest` stays green. ✓ done (2026-04-14, gate code + config deleted; grep returns zero source/test references; AST + py_compile verified; pytest not executed locally — pandas/numpy unavailable in this environment, must be confirmed by next CI run).
2. Add a regression test that pins per-position bias on the walk-forward backtest within ±10% (QB/RB/WR/TE separately). Time: ~half day. Success signal: the test fails if the gate-era +44% bias ever reappears, catching the next pipeline-leakage regression at PR time instead of in production. ✓ done (2026-04-14, `TestWalkForwardBiasRegression` added to `tests/test_backtest_validation.py`; auto-targets the most recent `ts_backtest_*_predictions.csv`; current backtest passes — overall +3.2%, per-position [-5.3%, +7.5%]; pytest not executed locally — pandas/numpy unavailable, must be confirmed by next CI run).
3. Run a Ridge α sensitivity sweep over `{0.3, 1, 3, 10}` per position on the existing walk-forward fold structure. Time: ~1 day. Success signal: the QB/TE 7–11pt `(std_ratio − r)` gap is either reproduced across folds with a tight standard error (in which case authorize per-position α tuning as the principled fix), or it dissolves (in which case the gap was an artifact and no fix is warranted). ⏳ infra ready, awaiting execution (2026-04-14): `--alpha` flag plumbed through `run_ts_backtest()`; `scripts/run_alpha_sweep.py` orchestrates the four runs and emits a side-by-side gap-mean ± SE table per position; `scripts/analyze_std_ratio.py` (stdlib-only) computes per-position and per-week gaps from any predictions CSV. Baseline at α=1.0 already computed and matches CRITICAL_LIMITATION.md exactly: QB gap +0.110 ± 0.047 (2.3σ, stable), TE +0.130 ± 0.080 (1.6σ, NOT stable), RB −0.015 ± 0.032, WR −0.004 ± 0.028. Run `python scripts/run_alpha_sweep.py --season 2025` in a runtime with pandas/numpy/sklearn to complete; the success signal cannot be claimed from a single α.
4. Write a one-page attribution explaining why the February 2026 backtest showed +44% bias and the April 10 walk-forward shows +3.2%, decomposed by mechanism: leakage fix, training-data window change (2018+ start), per-fold features, weekly refit. Time: ~half day. Success signal: every percentage point of the 40pp bias delta is explicitly attributed to one of those four mechanisms — no residual unexplained. ✓ done (2026-04-14, [`docs/BIAS_ATTRIBUTION_20260414.md`](../docs/BIAS_ATTRIBUTION_20260414.md)). **Aggregate-level success signal met:** ~38pp leakage fix (high confidence; first walk-forward run after the leakage fix was already at +2.7%), ~1–2pp training window 2018+ (low confidence, inferred), ~1–2pp causal features (medium confidence, derived from spread between post-fix runs), weekly refit folded into Mechanism 1. Residual: ~0pp at aggregate level. **Caveat:** per-position residuals (RB −11.4%→−5.3%, QB +8.6%→+2.1%) are NOT closed without three follow-up ablations specified in §"What would close the per-position residual" of the doc — these require backtest runtime not available in this environment.
5. Open the predictive-ceiling workstream against the r ≈ 0.52 limit: injury status, Vegas implied team totals, opponent defense rankings by position. Time: ongoing. Success signal: walk-forward R² exceeds the blended heuristic's 0.279 — until then, the model is provably worse than a one-line average. ⏳ planned, not started (2026-04-14, [`docs/PREDICTIVE_CEILING_PLAN.md`](../docs/PREDICTIVE_CEILING_PLAN.md)). 5-phase plan with explicit per-phase kill criteria, validation gates (must keep Step 2's bias regression test passing), and a workstream-level kill criterion (re-council if Phases 1–3 don't deliver +0.02 R²). No code yet — each phase opens its own branch off `main`.

### Council Convergence
- The +44% bias the gate was built to catch no longer exists; maintaining the gate is defending against the last war.
- For RB and WR specifically, `std(pred)/std(actual) ≈ correlation(pred, actual)` to within 0.01 — any post-hoc widening on those positions is mathematically equivalent to manufacturing noise.
- The only legitimately fixable shrinkage (QB/TE small-sample over-regularization) is a Ridge α tuning problem, not a downstream-blend problem; the gate is the wrong tool even for the one case where compression is real.

### Council Disagreement
**The real tradeoff:** delete *now* vs. delete *after* re-validating the new backtest.
**Side A (delete now, both advisors):** The justification for the gate has evaporated, the gate actively harmed the only metric users care about (breakout prediction), and keeping a disabled code path alive is how dead code metastasizes. Move on.
**Side B (peer reviewers' implied caution):** The new bias number rests on a single backtest configuration (Ridge α=1.0, expanding window, one fold split). Until the +44%→+3.2% delta is attributed line-by-line, deleting the safety net assumes the new regime is real and stable. A leakage fix cleaning up by 40 percentage points is large enough to deserve scrutiny on its own.
**Chairman's call:** Delete now, but only because Step 2 (the bias regression test) provides the same safety net the gate was supposed to be — without the breakout-suppression cost. Reviewer caution is right that the new backtest deserves auditing; it's wrong that the gate is the appropriate hedge against backtest unreliability.

### Blind Spots Caught in Review
- **Both advisors accepted the April 10 walk-forward backtest as ground truth** without asking how a single model can show +44% bias in February and +3.2% bias in April. Until that 40pp delta is attributed to specific mechanisms (leakage fix, training window, fold split), every downstream decision rests on an unaudited assumption. Step 4 closes this.
- **Neither advisor proposed any guardrail against the next pipeline regression.** Deleting the gate without a bias regression test means the next leakage bug rebuilds the original +44% problem with nobody to catch it. Step 2 closes this.
- **Neither addressed that Ridge currently loses to a blended heuristic** (R² 0.269 vs 0.279). Deleting the gate doesn't make this go away — it just stops papering over it. The model is provably worse than a one-line average on the same fold; this is the actual limitation, and the council's verdict on the gate is downstream of that fact.

### Kill Criteria
- If the α sensitivity sweep (Step 3) shows the QB/TE `(std_ratio − r)` gap is unstable across fold splits or has standard error larger than the gap itself, the per-position α tuning plan is invalid — re-council before authorizing any QB/TE-specific intervention.
- If the bias regression test (Step 2) fires within 30 days, the leakage fix is incomplete — pause feature work, do not add new predictors on top of an unstable bias floor, and re-audit the pipeline.
- If the bias attribution exercise (Step 4) leaves more than 5 percentage points of the 40pp delta unexplained, treat the new backtest as provisional — do not delete the gate until the residual is closed.

---

## Original Question

> Yes plz fix [the calibration-gate reversal flagged in the prior council status check].

## Framed Question

The April 1, 2026 council unanimously recommended adding a "post-prediction sanity-check / circuit breaker" to the NFL fantasy projection ensemble. Motivation at the time: a 2025 backtest showed catastrophic R²=-0.685 and a systematic +44% over-prediction (predicted 11.95 vs actual 8.31, with WR/TE +5.6/+5.3 over actuals).

**The gate, as built (commit 6ffff6a, 2026-04-09):**
- Per-player trailing 3-week fantasy-points average as baseline.
- If `|pred - baseline| / baseline > 50%`, blend `pred` 50% toward baseline.
- Min 2 games of history required. Bounds-clip after blending.
- Threshold and blend weight were hand-set, not derived.

**The gate was disabled (commit fdc3da9, 2026-04-10):** it acted as a third layer of mean-regression on top of Huber loss and TD regression. Breakout games where the player actually scored 25+ fantasy points were being predicted at a 12.6 average — every upside prediction was pulled toward trailing mean.

**Then on 2026-04-11, `CRITICAL_LIMITATION.md` was rewritten based on a clean walk-forward backtest (Ridge α=1.0, expanding window, weekly refit, leakage check passed):**

- Per-position bias is now [−5.3%, +7.5%], not +44%. The leakage fixes resolved the catastrophic over-prediction the gate was built to catch.
- Variance compression is mathematical, not a bug. For an unbiased linear model, `std(pred)/std(actual) = correlation(pred, actual)`. RB/WR walk-forward results: std_ratio 0.506/0.514 against Pearson r 0.512/0.513 — identity holds within 0.01.
- Only QB and TE show extra shrinkage on top of the correlation floor: 7–11 points of gap, consistent with small-sample Ridge α=1.0 over-regularization on ~600 QB / ~370 TE training rows per fold.
- The new dominant limitation is predictive ceiling, not calibration: Ridge model R² = 0.269 *loses* to a blended heuristic at R² = 0.279.

**Decision options:**
1. Abandon the gate entirely.
2. Rebuild asymmetrically — only flag QB/TE.
3. Keep gate disabled but introduce a different post-prediction check (Vegas total ceiling, flag-and-log).
4. Re-enable at much looser thresholds (>100% deviation, 20% blend).

---

## Advisor Responses

### The Statistician

The gate was built to fix a +44% bias that no longer exists. That's not a tuning problem — that's a dead hypothesis. Option 1.

Let me interrogate the numbers the other options rest on.

**The variance compression is not a defect.** std(pred)/std(actual) = 0.506 vs Pearson r = 0.512 for RB; 0.514 vs 0.513 for WR. That identity holds to within 0.01 on both positions. An unbiased estimator with r ≈ 0.51 *cannot* produce wider spread without introducing bias. Any gate that pushes RB/WR predictions outward is manufacturing noise and calling it signal. Reject Option 2's RB/WR leg reflexively? No — reject it because the math forbids improvement there.

**QB/TE "extra shrinkage" is a real finding but a weak one.** 7–11 points of gap on ~600 QB / ~370 TE rows per fold. Before I authorize asymmetric gating, I want to see: is this gap stable across folds, or is it two or three outlier weeks? What's the standard error on that gap? A 10-point gap with a 6-point SE is not actionable. Fix the cause (tune α per position, or use position-specific models) rather than paper over it with a downstream blend.

**Options 3 and 4 fail the base-rate test.** Current per-position bias is [−5.3%, +7.5%]. A >100% deviation flag will fire on genuine outliers — the Josh Allen 45-point weeks you want to predict. You'd be killing signal to chase a calibration problem that's already within ±8%.

**Verdict: Option 1. Delete the gate. Redirect effort at the r ≈ 0.52 ceiling — that's where the loss is.**

### The Executor

Abandon the gate. Option 1.

Monday morning: delete the gate code, delete the 50%/50% constants, delete the 3-week baseline computation that exists only to feed it. The +44% bias is gone. You are maintaining a safety net for a fall that already stopped happening.

The other three options are procrastination dressed as caution:

**Option 2 (asymmetric rebuild)** — you'd be hand-calibrating position-specific thresholds against ~600 QB / ~370 TE rows. You don't have the sample size to tune those knobs without overfitting them. The real fix for QB/TE over-shrinkage is Ridge α tuned per position (or per-position models), not a post-hoc patch. Do that instead.

**Option 3 (Vegas totals / flag-and-log)** — a new feature pretending to be a safety check. If Vegas implied points help, they're a *feature* for the model, not a ceiling bolted on after. And "flag-and-log" means building a UI surface nobody asked for to babysit a model whose actual problem is predictive ceiling, not outliers.

**Option 4 (looser thresholds)** — keeps a disabled code path alive "just in case." That's how dead code metastasizes. If you ever need an anomaly killer, write it then, against real anomalies you've observed post-leakage-fix, not the ghost of a bias that no longer exists.

Concrete Monday plan:
1. Remove gate code + baseline helper + tests. One commit.
2. Update CRITICAL_LIMITATION.md to record the gate was removed and why.
3. Open a separate ticket: "Per-position Ridge α tuning for QB/TE." That's the real work.

Stop defending against the last war. The limitation now is r ≈ 0.52 — go find features.

---

## Peer Reviews

### Anonymization Mapping
- Response A = The Executor
- Response B = The Statistician

### Review 1 (by The Contrarian)

1. **B is stronger.** It interrogates the identity std_ratio ≈ r and names the precise failure mode (manufacturing noise by widening). It also demands what A doesn't: stability of the QB/TE gap across folds and its standard error. A asserts conclusions; B shows the math forbids alternatives.

2. **A's blind spot:** it treats "Ridge loses to a blended heuristic (0.269 vs 0.279)" as just "go find features." That result is load-bearing — the production model is beaten by a heuristic *right now*. Shipping Option 1 without addressing this means shipping a known-inferior model. A's ticket list omits the blend.

3. **Both missed:** the backtest itself is one seed, one α, one window. Before deleting the gate, re-run the walk-forward with α ∈ {0.3, 1, 3, 10} and a different fold split. The +44% bias "disappeared" after a leakage fix — what other artifacts changed? Also: neither proposed a regression test pinning per-position bias within ±10%, so the next leakage bug won't be caught.

### Review 2 (by The First Principles Thinker)

1. **B is strongest.** It interrogates the numbers rather than just prescribing action. The observation that std_ratio ≈ r is a *mathematical identity* for unbiased low-r estimators reframes the entire debate: Option 2's RB/WR leg isn't bad judgment, it's forbidden by algebra. B also demands the standard error on the QB/TE gap before acting — the correct first-principles move.

2. **A's blind spot:** it treats the +44% bias as simply "gone," but the bias disappeared because a *leakage check passed* on a new backtest. A never asks whether the clean walk-forward is itself trustworthy, or whether the original +44% was the leak's inverse artifact. It accepts the new regime as ground truth.

3. **Both miss the framing error:** the real question isn't "what gate?" — it's "why did two backtests of the same model disagree by 50 percentage points of bias?" Until that's explained, every downstream decision rests on trusting the newer number by default.
