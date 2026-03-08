# Decision Objectives and Policy Mapping

Per Agent Directive V7 Section 1 (decision objective supremacy) and
Section 9 (decision optimization layer).

---

## Prediction vs Decision Quality

The system separates prediction quality from decision quality:

| Metric Type | Measures | Module |
|-------------|----------|--------|
| **Prediction quality** | RMSE, MAE, R², calibration | `src/evaluation/metrics.py` |
| **Decision quality** | Start hit rate, lineup score, VOR accuracy | `src/evaluation/decision_optimizer.py` |

An improvement in RMSE does not automatically improve decisions.
A model with lower RMSE but poor calibration may produce worse
lineup decisions than a slightly higher-RMSE model with good calibration.

---

## Decision Objectives by Use Case

### 1. Weekly Start/Sit Recommendations
- **Objective**: Maximize hit rate of "start" recommendations scoring > 5 pts
- **Abstention policy**: When coefficient of variation > 0.30, recommend "uncertain"
- **Evaluation**: `evaluate_decision_quality()` tracks start_hit_rate, sit_correct_rate

### 2. Draft Rankings (Season-Long)
- **Objective**: Maximize rank correlation with actual season-end VOR
- **Framework**: Value Over Replacement with positional scarcity weighting
- **Module**: `compute_vor_rankings()`

### 3. DFS Lineup Construction
- **Objective**: Maximize projected lineup score under salary cap
- **Cash games**: Maximize floor (predicted - 0.5 * std)
- **GPP tournaments**: Maximize ceiling (predicted + 1.0 * std)
- **Module**: `src/optimization/lineup_optimizer.py`

### 4. Waiver Wire Pickups
- **Objective**: Maximize VOR + upside for unrostered players
- **Priority**: 60% VOR + 40% upside score
- **Module**: `waiver_wire_priority()`

---

## Abstention Policy

Per Directive V7 Section 9: abstention is a first-class policy.

The system abstains from recommendations when:
1. Prediction std > 30% of predicted value (coefficient of variation)
2. Player has < 3 weeks of data (insufficient sample)
3. Circuit breaker is tripped (monitoring degradation detected)

Abstention is logged and evaluated: high variance in the "uncertain"
bucket validates the abstention policy.

---

## When Prediction Improvement Translates to Decision Improvement

Based on empirical analysis:
- RMSE improvements > 0.5 points reliably improve start/sit accuracy
- RMSE improvements < 0.3 points may not change any decisions
- Calibration improvements (ECE reduction) directly improve abstention quality
- Ranking improvements (Spearman rho) directly improve VOR accuracy
