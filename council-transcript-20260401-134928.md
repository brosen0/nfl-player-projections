# LLM Council Transcript

**Date:** April 1, 2026
**Question:** What is the single most critical limitation of the NFL player projection system?

---

## Original Question

> Council this: identify most critical limitation

## Framed Question

An NFL fantasy football prediction system (ensemble ML with Ridge, ElasticNet, GBR, XGBoost, LightGBM) using utilization-score features, time-series cross-validation, and multi-horizon predictions (1w, 4w, 18w). Trains on 2006-2024 NFL data (19 seasons).

**Key backtest evidence:**
- **2024 backtest:** R²=0.233, correlation=0.533, MAE=4.84, avg_predicted=8.10 vs avg_actual=8.15 (well-calibrated)
- **2025 backtest:** R²=-0.685, correlation=0.247, MAE=7.56, avg_predicted=11.95 vs avg_actual=8.31 (catastrophic failure)
- WR 2025: R²=-1.357, predicted 12.9 vs actual 7.29 (+5.6 over-prediction)
- TE 2025: R²=-1.191, predicted 11.95 vs actual 6.64 (+5.3 over-prediction)
- RB 2025: R²=0.009, predicted 8.79 vs actual 9.92 (only position close)
- Training R² ranges 0.22-0.60 (decent in-sample), but 2025 out-of-sample collapses
- Variance compression: std_predicted < std_actual across all positions

**Architecture:** 100+ features (rolling averages, lags, opponent adjustments, game script, snap counts, target quality, TD regression, injury/rookie handling, Vegas features) reduced to ~15 per position via VIF. Recency decay halflife of 2 seasons. Purged time-series CV. Utilization two-stage for RB; direct FP prediction for WR/TE.

**Question:** What is the single most critical limitation, and what should be done about it?

---

## Advisor Responses

### 1. The Contrarian

The fatal flaw is obvious and the backtest data screams it: **your model is memorizing regime-specific correlations that don't transfer across seasons, and no amount of ensemble complexity or feature engineering will fix that.**

Look at the numbers. Training R² of 0.22-0.60, then 2025 out-of-sample R² of -0.685. That's not "degradation." That's a model that learned nothing generalizable. A negative R² means you'd be better off predicting the mean for every player. The systematic +3.6 point over-prediction tells you exactly what happened: the model learned the relationship between features and fantasy points *for a specific era of offensive production*, and when the landscape shifted even slightly, it broke completely.

100+ features reduced to ~15 via VIF is a red herring of rigor. VIF removes multicollinearity, not overfitting to temporal patterns. Your rolling averages, lag features, and recency decay all encode *recent history as if it predicts the future*, but NFL production is dominated by role changes, scheme changes, coaching changes, and injury cascades that no backward-looking feature captures.

**The single most critical limitation: you're treating a non-stationary, regime-switching process as if it's a stationary regression problem with temporal weighting.** Recency decay and purged CV are band-aids on a fundamental mismatch between your modeling assumptions and the data-generating process.

**What to do:** Strip this down to 5-10 features maximum that have demonstrated *causal* relationships to production (opportunity share, snap percentage, team pace). Drop all rolling/lag features beyond 3-week windows. Accept that your ceiling for week-ahead prediction is probably R² of 0.15-0.20 out-of-sample, and build your system around *calibrated uncertainty* rather than point prediction accuracy. A model that says "I don't know" with correct confidence intervals is infinitely more useful for fantasy decisions than one that confidently predicts 12.9 points when reality delivers 7.3.

### 2. The First Principles Thinker

The question asks about the "single most critical limitation." But the backtest numbers are already screaming the answer — you're just not hearing it because you're looking at the wrong level.

**You don't have a modeling problem. You have a prediction target problem.**

The fundamental issue: NFL fantasy points are dominated by touchdowns, which are high-variance, low-autocorrelation events. A WR catching 6 passes for 70 yards scores ~10 PPR points. Add one TD and it's 16. That TD is nearly coin-flip predictable at the individual game level. You're building an increasingly complex machine to predict something that is, at its core, substantially random week-to-week.

Your R² of 0.233 in 2024 isn't "decent" — it's the ceiling for this target variable with these features. The 2025 collapse to -0.685 isn't a bug in your architecture. It's what happens when you overfit to the predictable variance in your training window and the unpredictable variance shifts even slightly between seasons.

**The systematic +3.6 over-prediction tells you exactly what happened:** the model learned that 2024's scoring environment (which it trained on with recency weighting) would persist. It didn't. Rule changes, offensive scheme shifts, injury patterns — the non-stationarity you're trying to handle with halflife decay is the actual signal, not noise to be smoothed over.

**What to do:** Stop predicting fantasy points directly. Predict the stable components separately — targets, receptions, yards, rushing attempts — then assemble fantasy points from those predictions. Each component has higher autocorrelation and lower touchdown contamination. Your R² on yards and targets will be materially higher and more stable year-over-year.

The two-stage approach you tried for RB was the right instinct applied incompletely. Apply it everywhere. Decompose the target, don't ensemble-hammer a noisy composite.

### 3. The Expansionist

The single most critical limitation is that this system is playing small. You have an ensemble ML pipeline trained on 19 seasons of NFL data and you're using it to predict fantasy points — a single scalar output. That's like building a telescope and only looking at the moon.

**The massive upside everyone is missing: this system should be predicting distributions, not point estimates.**

Your R² collapse from 0.233 to -0.685 isn't primarily a feature engineering problem. It's a *framing* problem. Fantasy football value isn't about who scores 12.3 points. It's about who has the highest probability of outperforming their cost basis (ADP/salary). The moment you shift from point prediction to distributional prediction, everything changes:

1. **Prediction intervals become the product.** A player projected at 10 points with a tight 8-12 interval is fundamentally different from one projected at 10 with a 2-22 interval.

2. **The over-prediction bias becomes actionable.** In a distributional framework, bias in the mean matters far less because you're making decisions on percentiles and probability thresholds.

3. **You unlock draft/DFS optimization that actually works.** Quantile regression forests or conformal prediction intervals give you P10/P50/P90 projections per player.

**What to do:** Replace your Ridge/ElasticNet stack with quantile regression variants and NGBoost/LightGBM with quantile loss. Train on the same features. Instead of outputting one number, output the 10th, 50th, and 90th percentiles.

### 4. The Outsider

The single most obvious thing wrong here — and I say this as someone with no skin in this game — is that your model doesn't know it's wrong.

You trained on 19 seasons. Your model fits those 19 seasons okay. Then it hits season 20 and predicts every wide receiver will score 5.6 points more than they actually do. That's not a subtle drift. That's the model confidently asserting something that is categorically false.

And nobody built a mechanism that says "hey, the predictions coming out of this pipeline are 40% higher than any reasonable baseline — maybe pump the brakes."

From the outside, the architecture reads like a system designed to be impressive rather than to be right. 100+ features reduced to 15 via VIF. Five different ML models ensembled together. Utilization-based two-stage prediction for some positions but not others. Every one of these is a technique that sounds good in a writeup. But the result is a system that predicts a TE will score 12 points when they actually score 6.6, and has no circuit breaker.

**The single most critical limitation is the absence of post-prediction sanity checking.** Before any prediction leaves this system, it should be compared against trivially simple baselines: last season's average, position average, Vegas implied totals. If your sophisticated ensemble can't beat "just use last year's numbers," the ensemble is destroying information rather than creating it.

The fix: build a calibration gate. Compare ensemble output against a naive baseline. If the ensemble deviates more than some threshold without clear justification from the features, blend it back toward the baseline. This is a two-day fix that would have caught the entire 2025 collapse before it happened.

### 5. The Executor

The single most critical limitation is obvious from the numbers: your model is memorizing historical scoring environments instead of learning portable relationships. The +3.6 point systematic over-prediction in 2025 tells you exactly what's happening — the model learned that "a WR with these usage metrics scores X" based on 2006-2024 scoring distributions, and 2025's actual scoring environment shifted downward.

**What you do Monday morning:**

1. **Predict rank-order or z-scores within position-week, not raw fantasy points.** Raw FP prediction is a trap because league scoring rules, pace of play, and offensive philosophy shift year to year. If you predict "this WR will be the #8 WR this week" instead of "this WR will score 12.9 points," you sidestep the entire calibration problem.

2. **Post-prediction calibration layer.** Isotonic regression or Platt scaling on a rolling 3-week window of actual 2025 data. Catches the mean shift within weeks.

3. **Drop training data before 2018.** The NFL before 2018 is a different sport. Those old seasons are actively poisoning your model.

Start with item 1. It's the structural fix.

---

## Peer Review

### Anonymization Mapping
- Response A = The Contrarian
- Response B = The First Principles Thinker
- Response C = The Expansionist
- Response D = The Outsider
- Response E = The Executor

### Review Results

**Unanimous strongest: Response A (The Contrarian)** — All 5 reviewers selected it. Correctly identifies the root cause: treating a non-stationary process as stationary. Rolling averages and lag features bake in the assumption that recent history predicts forward, which fails when regimes change.

**Unanimous weakest: Response C (The Expansionist)** — All 5 reviewers identified it. Switching to quantile regression on a model that's systematically +3.6 points wrong just gives well-calibrated intervals around the wrong center. Addresses presentation while ignoring that predictions are systematically wrong.

**Unanimous blind spot: Data leakage audit** — All 5 reviewers independently flagged: nobody investigated whether the 2025 backtest has data leakage or pipeline bugs in feature construction. A +3.6 systematic shift is consistent with features encoding future information during backtesting (e.g., season-end stats leaking into rolling calculations). This should be audited before any architectural changes.

---

## Chairman Synthesis

### Where the Council Agrees

The model treats a non-stationary process as stationary. All five advisors converge on this: the NFL scoring environment shifts year-over-year in ways that backward-looking features cannot capture, and the model has no mechanism to detect or adapt to these shifts. The +3.6 systematic over-prediction is the model confidently applying 2024's regime to 2025's reality.

Unanimous agreement that R² of 0.233 is close to the ceiling for raw fantasy point prediction. Nobody believes R² > 0.4 is achievable.

Every advisor implicitly agrees the feature set is too large. 100+ features on a non-stationary target with regime shifts is a recipe for fitting noise. Whether you strip to 5-10 causal features, decompose into sub-predictions, or switch targets entirely — the prescription is: simplify.

### Where the Council Clashes

**What to predict.** Component stats and assemble FP (First Principles) vs. rank-order/z-scores (Executor) vs. accept low ceiling on raw FP (Contrarian). Component prediction is more interpretable but multiplies modeling surface. Rank-order is more stable but loses magnitude. Calibrated uncertainty is honest but harder to sell as a product.

**How aggressive to be.** Two-day sanity check (Outsider) vs. rebuild with 5-10 features (Contrarian) vs. full two-stage overhaul (First Principles). Real tension between stopping the bleeding now versus fixing the structural problem.

### Blind Spots the Council Caught

Every advisor jumped to architectural explanations, but nobody investigated whether the 2025 backtest itself is broken. A +3.6 systematic shift is consistent with a pipeline bug — season-end stats leaking into rolling feature calculations during backtesting, or future information contaminating 2025 feature construction. This is the most parsimonious explanation for the observed failure mode. Architectural overhauls built on top of a leaky backtest will produce misleading validation results.

### The Recommendation

The root cause is most likely one of two things:

1. **A data leakage or pipeline bug in the 2025 backtest** — explains the systematic shift, uniform bias direction, and catastrophic year-over-year degradation.
2. **Temporal overfitting to regime-specific correlations** — explains why 100+ features and rolling windows produce a model that cannot generalize.

Audit for (1) first. If the pipeline is clean, the Contrarian's diagnosis is correct: strip to a small set of causal features, accept a lower but stable R², and add the Outsider's sanity-check circuit breaker. The First Principles decomposition (predict components, assemble points) is the right long-term architecture but is Phase 2.

Do not pursue distributional prediction until point estimates are unbiased. Do not add complexity until the simple version works across years.

### The One Thing to Do First ✓ done (2026-04-14, audit completed via scripts/trace_player_leakage.py + CRITICAL_LIMITATION.md rewrite)

Audit the 2025 backtest pipeline for data leakage. Manually trace feature values for 3-5 players in Week 1 of 2025 and verify that no statistic from 2025 or late 2024 (after training cutoff) contaminates any rolling average, lag feature, or engineered variable. If leakage is found, fix it and re-run. If clean, proceed with feature reduction.
