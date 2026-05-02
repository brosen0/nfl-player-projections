# Council Transcript — 2026-05-01
**Question:** How to improve walk-forward accuracy and overall out-of-sample player point predictions

**Type:** Bucket 4 partial council (changed conditions — v7 lifted R² from 0.269 to 0.314, invalidating the prior feature-freeze premises)

---

## Chairman's Verdict

## Bottom Line
**Do:** Shift optimization from R² to decision-quality metrics while running a time-boxed QB rescue and targeted error analysis.
**Don't:** Chase R²=0.340 as a goal — it's a proxy that doesn't map to start/sit outcomes your user cares about.

## Critical Next Steps
1. **DO TODAY:** Run error analysis segmented by home/away, spread bucket, rest days, and position across the 8-season LOYO results. Time: 4-6 hours. Success signal: A ranked list of 3-5 situation types where model error is systematically highest, with enough sample size (n>200) to act on.
2. **Build a QB-specific model variant** that uses QB-only features (sack rate, pressure rate, scramble tendency, weather sensitivity) tested against the current ensemble on QB predictions only. Time: 3-5 days. Success signal: QB R² exceeds 0.180 in at least 6 of 8 LOYO seasons (up from current 0.228 overall, 0.121 in 2025).
3. **Run exactly 3 feature experiments** informed by step 1's error analysis — add features only where the error analysis shows systematic, reducible bias, not random variance. Time: 1 week. Success signal: Each feature group must deliver +0.005 R² on the full LOYO or gets cut.
4. **Implement decision confidence tiers** — tag each start/sit recommendation as "strong" (model edge >3 pts + historically >75% correct) or "lean" (everything else). Time: 3-4 days. Success signal: "Strong" tier covers at least 40% of weekly decisions and backtests at >80% accuracy; "lean" tier is honestly labeled.
5. **Diagnose the 4 losing seasons against the blended heuristic.** Time: 2-3 hours. Success signal: You know whether losses cluster by position, by season recency, or by game-script type — this determines whether the problem is drift, position weakness, or a fixable modeling gap.

## Council Convergence
- The 72.7% decision accuracy and 98.8% vs replacement are already strong enough to ship — further accuracy gains are gravy, not survival
- QB modeling is the clear weak link and the highest-ROI single improvement target

## Council Disagreement
**The real tradeoff:** R² optimization vs decision-quality optimization as the north star metric.
**Side A:** R² is measurable, comparable, and a proven lever — the +0.045 lift directly improved decision outcomes, so pushing to 0.330-0.340 will do the same.
**Side B:** R² improvements don't linearly translate to better start/sit calls — a model at R²=0.314 with calibrated confidence beats R²=0.340 with uncalibrated outputs for user trust and retention.
**Chairman's call:** Side B, because you have one user, one product, and 4 months — confidence tiers ship as a visible feature improvement while R² gains are invisible to the user. But do the QB and feature work anyway because it's fast and directly feeds better confidence calibration.

## Blind Spots Caught in Review
- The 4 seasons where the model loses to the blended heuristic are undiagnosed — if they're the 4 most recent seasons, you have a model decay problem that invalidates everything else. Run step 5 before committing to steps 2-4.
- Coin-flip accuracy (55%) has not been examined per-position or per-season — if it's 62% for RB and 48% for QB, that changes priority ordering entirely.

## Kill Criteria
- If QB R² does not reach 0.180 overall after the QB-specific model work, stop investing in QB prediction accuracy and instead widen the "lean" confidence tier for QB calls — accept the model is weaker there and communicate it honestly.
- If 3 feature experiments collectively fail to deliver +0.010 R² on the full LOYO, reinstate the feature freeze permanently and redirect all remaining time to product polish and confidence calibration for the September launch.

---

## Original Question
How can we improve walk-forward accuracy and overall out-of-sample player point predictions?

## Framed Question
The predictive-ceiling workstream was killed in April 2026 when Ridge alpha=10,000 achieved only +0.004 R² across three feature phases (vs a +0.02 target) and R²=0.269 lost to the blended heuristic at 0.279. Since then, v7 shipped (Huber loss, residual deep net, log1p target, isotonic calibration, stability selection, learned blend weights, player embeddings, bye/short-week features) and an 8-season LOYO (2018-2025) produces R²=0.314, beating the blended heuristic in 4 of 8 seasons. The ensemble runtime problem is resolved.

Given that the model now sits above the bar the prior councils set ("beat the blended heuristic"), and the +0.045 R² lift came from model architecture rather than features: (a) what are the highest-leverage next steps to improve walk-forward accuracy further, (b) should the feature freeze be revisited now that a more expressive model class can potentially extract more from features that were noise to Ridge, and (c) what is the realistic next R² target and what kill criterion should govern the next round of work?

---

## Advisor Responses

### The Statistician
The +0.045 R² lift is real but let's be honest about what 0.314 means: you're explaining 31% of variance in a domain where the irreducible noise floor is probably around R²=0.45-0.50. You're halfway between guessing and the ceiling.

**On (a), highest-leverage next steps:** The per-position R² spread (QB 0.228 vs RB 0.299) tells you where the marginal gains live. QB is your worst position and the one with the fattest tail outcomes. I'd want to see QB modeled separately with QB-specific features (pressure rate, time-to-throw, receiver separation) before touching anything else. The 2025 QB R²=0.121 is a red flag — that's barely above noise. Diagnose whether that's a distribution shift (new QBs, rule changes) or a systematic model weakness before adding complexity elsewhere.

**On (b), revisiting the feature freeze:** Yes, but with discipline. The Ridge couldn't extract signal from interaction-heavy features — that was the correct diagnosis for killing features then. But "features that were noise to Ridge" is a hypothesis, not a fact. Run a controlled experiment: take your v7 architecture, add one feature group at a time (e.g., advanced passing metrics), measure delta-R² on the same 8-season LOYO. If no single group moves R² by +0.005, the feature freeze was right for the wrong reasons. Budget: 3 feature groups max, kill any that don't clear +0.005.

**On (c), target and kill criterion:** Your realistic next target is R²=0.340. That's roughly +0.026, which is the same magnitude as your architecture lift but harder because of diminishing returns. Kill criterion: if after QB-specific modeling + 3 feature experiments you haven't hit 0.330, the next dollar of effort goes to product, not model. The 72.7% decision accuracy is already commercially viable — further R² gains face a steep effort-to-user-value decay curve.

The 55% on coin-flip decisions is exactly what you'd expect and not worth optimizing. Those are inherently unpredictable.

### The First Principles Thinker
The question you're asking -- "how do we push R² higher?" -- is the wrong question. Here's why.

Your actual product is start/sit decisions. You're already at 72.7% vs hindsight and 75.6% H2H. The user who tested this validated specific calls (Fannin, Loveland). R² is a continuous accuracy metric over all predictions. Your users don't care about all predictions — they care about the 15-20 roster decisions per week where the model's ranking disagrees with consensus.

**The real question is: what's the decision-accuracy ceiling, and are you bottlenecked by model quality or by something else entirely?**

Look at your "hardest calls" stat: 55% on 800 coin-flips. That's barely above random. But that's not a model failure — those are genuinely unpredictable outcomes (projection gap <1.5, actual gap >5). No feature or architecture will fix that. The signal simply doesn't exist in pre-game data for those cases.

So rebuild from the ground up:

**(a)** The highest-leverage move isn't model accuracy — it's **calibrated confidence**. When your model says "start Player A by 0.3 points," the user needs to know that's a coin flip. When it says "start by 6 points," they need to know that's bankable. Attach a reliability score to every recommendation. This changes the product without touching R².

**(b)** Don't revisit the feature freeze for R². Revisit it only if specific features improve **decision separation** — the gap between clear calls and coin flips. Test features by whether they reduce the number of "too close to call" decisions, not by R² lift.

**(c)** Kill the R² target entirely. Your next metric should be: **what percentage of start/sit decisions can the model make with >80% historical reliability?** Currently unknown. Measure it first. That's the work.

---

## Peer Reviews

**Anonymization mapping:** Response A = The Statistician, Response B = The First Principles Thinker

### Contrarian Review
1. Response B is stronger — optimizing R² is chasing a proxy. Reframing around decision confidence is the higher-leverage move.
2. Response A's blind spot: R²=0.340 as target without evidence it changes user outcomes; "commercially viable" 72.7% without knowing competitive baseline.
3. Both miss: coin-flip stability across positions/seasons unexamined. The 4-of-8 heuristic losses need diagnosis — are they recent (model decay) or position-specific?

### Executor Review
1. Response A is stronger — gives concrete executable steps for Monday morning. QB-specific model, 3 feature groups, +0.005 threshold. Clear path.
2. Response B's blind spot: "calibrated confidence" sounds great but requires a reliability scoring framework that doesn't exist yet. No implementation path — it's a research project disguised as a product recommendation.
3. Both miss: low-hanging fruit of error analysis by situation (home/away, spread bucket, rest days). One afternoon of work to determine if remaining variance is reducible.
