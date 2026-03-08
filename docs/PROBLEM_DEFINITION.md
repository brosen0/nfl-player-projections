# Problem Definition and Utility Mapping

Per Agent Directive V7, Section 4 — this document separates the prediction
problem from the decision problem and defines the operational context.

---

## Prediction Target

| Attribute | Value |
|-----------|-------|
| **Entity** | NFL player |
| **Target variable** | PPR fantasy points per game-week |
| **Granularity** | Per-player, per-week |
| **Positions** | QB, RB, WR, TE (ML models); K, DST (rolling averages) |
| **Horizons** | 1-week ahead, 4-week ahead, 18-week (season-long) |
| **Secondary target** | Utilization score (0-100) for RB, WR, TE |

### Scoring System

PPR (Points Per Reception) scoring as defined in `config/settings.py`:

| Category | Points |
|----------|--------|
| Passing yard | 0.04 |
| Passing TD | 4 |
| Interception | -2 |
| Rushing yard | 0.1 |
| Rushing TD | 6 |
| Reception | 1 (PPR) |
| Receiving yard | 0.1 |
| Receiving TD | 6 |
| Fumble lost | -2 |
| 2-point conversion | 2 |

---

## Decision Objectives

### Primary Optimization Targets

These are the real-world outcomes the system should help users optimize:

| Decision Context | Optimization Target | Current Status |
|-----------------|---------------------|----------------|
| **Weekly lineup (season-long)** | Maximize total fantasy points scored across season | Predictions available; no lineup optimizer |
| **Start/sit decisions** | Choose the higher-scoring player at each position | Predictions + confidence intervals available; no recommendation engine |
| **Draft strategy** | Maximize total roster value via VOR-aware drafting | Draft Assistant tab exists in UI; no positional scarcity optimization |
| **Waiver wire** | Identify highest-value available pickups | Not implemented |
| **Trade evaluation** | Assess fair trade value based on remaining-season projections | Not implemented |
| **DFS lineup construction** | Maximize expected points under salary cap constraints | Not implemented |

### Action Space

| Action | Input Required | Output |
|--------|---------------|--------|
| Rank players | Predictions for all players at a position | Ordered list with confidence bands |
| Start/sit | Two or more player predictions | Recommended starter with confidence level |
| Draft pick | All player predictions + positional scarcity | VOR-ranked draft board |
| Waiver claim | Current roster + available player predictions | Prioritized pickup list |
| Trade offer | Both sides' remaining-season projections | Fair value assessment |
| DFS lineup | All predictions + salary data | Optimized lineup(s) |

---

## Prediction Quality vs Decision Quality

Per Section 4 of the directive, these are distinct concepts:

| Metric Type | What It Measures | Current Implementation |
|-------------|-----------------|----------------------|
| **Prediction quality** | How close predictions are to actual outcomes | RMSE, MAE, R², MAPE via `src/evaluation/metrics.py` |
| **Ranking quality** | How well predicted order matches actual order | Spearman rank correlation, tier accuracy |
| **Decision quality** | How much value the decisions produce | Not measured — no decision layer exists |

A model with worse RMSE but better ranking accuracy may produce better fantasy
outcomes. The system should evaluate both.

---

## Operational Constraints

### Data Availability Timeline

| Data Source | Typical Availability | Used By |
|------------|---------------------|---------|
| Game results (scores, stats) | Sunday/Monday night | Weekly stats features |
| Play-by-play data | Monday night – Tuesday | PBP-derived features (EPA, WPA) |
| Snap counts | Tuesday | Utilization score components |
| Weekly injury reports | Wednesday–Saturday | Injury features |
| Vegas lines (spread, O/U) | Tuesday–Sunday (updating) | Game-script features |
| Official inactive lists | 90 min before kickoff | Active/inactive status |

### Prediction Delivery Requirements

| Requirement | Target |
|------------|--------|
| Predictions available by | Sunday morning (before 1pm ET kickoff) |
| Data refresh window | Tuesday–Saturday |
| Model retraining frequency | Weekly (configurable in `config/settings.py`) |
| Prediction latency | < 5 seconds per player |
| Full roster prediction | < 60 seconds for top 200 players |

### Confidence and Abstention

The system provides confidence intervals at 50%, 80%, 90%, and 95% levels.

**Abstention policy** (recommended, not yet implemented):
- Flag predictions where 80% CI width exceeds 2x the position average
- Flag rookies with < 3 games of NFL data
- Flag players returning from injury with < 2 games back
- Present flagged predictions with explicit uncertainty warnings

---

## Success Criteria

Per `fantasy_football_requirements_formatted.md`, Section VII:

### Accuracy Thresholds (1-week predictions)

| Metric | Target |
|--------|--------|
| RMSE vs expert consensus | Within 10% |
| Spearman rank correlation | > 0.65 for top 50 per position |
| Improvement over naive baselines | > 25% |
| Predictions within 10 points of actual | > 80% |
| Predictions within 7 points of actual | > 70% |

### Reliability Thresholds

| Metric | Target |
|--------|--------|
| Season-long accuracy stability | No > 20% degradation across season |
| Confidence interval calibration | 90% nominal should achieve 87-93% actual coverage |

### Position-Specific RMSE Targets

| Position | 1-Week | 4-Week | 18-Week |
|----------|--------|--------|---------|
| QB | 6.0–7.5 | 8–10 | 12–15 |
| RB | 7.0–8.5 | 9–11 | 12–15 |
| WR | 6.5–8.0 | — | 12–15 |
| TE | 5.5–7.0 | — | 12–15 |
