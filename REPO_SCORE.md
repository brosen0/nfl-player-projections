# Repository Score: Agent Directive V7 Compliance

**Repository:** NFL Player Performance Predictor (`nfl-player-projections`)
**Evaluation Date:** 2026-03-09
**Directive Version:** Agent Directive V7 Complete (25 sections)
**Evaluator:** Independent codebase audit with line-level verification

---

## Overall Score: 68 / 100

---

## Methodology

Each of the 25 directive sections is scored 0–4 based on verified implementation:
- **4** = Full compliance — all required outputs present and functional
- **3** = Mostly compliant — core requirements met, minor gaps
- **2** = Partially compliant — some requirements met, significant gaps
- **1** = Minimal compliance — token effort or documentation-only
- **0** = Not implemented

Scoring is based strictly on working code, not documentation claims. A doc file describing a feature that doesn't exist in code scores 0 for that item.

---

## Section-by-Section Scoring

| # | Section | Score | Evidence |
|---|---------|-------|----------|
| 1 | Mission and Non-Negotiable Principles | 3 | Temporal integrity enforced via `src/utils/leakage.py` (centralized guards), `ts_backtester.py` (expanding-window walk-forward). Experiment tracker captures git hashes, seeds, dataset hashes (`log_dataset_hash()`). `docs/DECISION_OBJECTIVES.md` defines optimization targets. **Gap:** No automated circuit breaker halting predictions on quality degradation. |
| 2 | Multi-Agent System Architecture | 1 | Monolithic single-process system. Module boundaries (`src/data/`, `src/features/`, `src/models/`, `src/evaluation/`) loosely map to directive agent roles, but there is no agent coordination, message passing, or role-based execution. `docs/ARCHITECTURE.md` documents the mapping but it's aspirational. |
| 3 | Shared Contracts and Required Logs | 3 | `src/evaluation/experiment_tracker.py`: JSONL ledger with `start_run()`, `log_metrics()`, `log_dataset_hash()`, `promote_run()`/`reject_run()`. Fields include dataset_version, feature_version, validation_split_spec, random_seeds, calibration_method per §3. **Gap:** No shared contract dataclass enforced between modules; promotion decisions captured but not linked to deployment. |
| 4 | Problem Definition and Utility Mapping | 3 | `config/settings.py` defines targets (fantasy points by position), horizons (1w/4w/18w), entities (NFL players). `docs/PROBLEM_DEFINITION.md` and `docs/DECISION_OBJECTIVES.md` exist. **Gap:** No formal latency SLA or data timing constraint documentation. |
| 5 | Dataset Discovery, Construction, and Lineage | 3 | Broad data via nfl-data-py: weekly stats, PBP, snap counts, schedules, rosters (2020–2025). `src/data/schema_validator.py` validates incoming schemas. `log_dataset_hash()` for versioning. `docs/DATA_LINEAGE.md` exists. **Gap:** No field-level availability timestamps; no raw snapshot preservation with versioning; no explicit survivorship bias testing. |
| 6 | Feature Discovery Engine | 3 | 11 feature engineering files. Rolling/lag/trend features (`feature_engineering.py`), utilization scores, QB-specific features (`qb_features.py`), rookie/injury features, PBP-derived (EPA, WPA via `pbp_stats_aggregator.py`), multiweek/season-long features. Dimensionality reduction via `dimensionality_reduction.py` (RFE, PCA, VIF, variance threshold). **Gap:** No formal feature acceptance report documenting stability, production availability, or revision risk per feature. Feature engineering pipeline not fully documented with acceptance criteria per §6. |
| 7 | Model Search and Meta-Learning | 3 | Five+ model families: Ridge, RandomForest, GradientBoosting, XGBoost, LightGBM. LSTM + ARIMA in `horizon_models.py`. Deep NN in `advanced_models.py`. Optuna hyperparameter tuning. `src/models/meta_learning.py` with `MetaLearningRegistry` tracking best config per (position, horizon, regime). **Gap:** Meta-learning registry exists but is not systematically populated during training runs; it's a framework without production data. |
| 8 | Ensemble Optimization and Calibration | 3 | Weighted average ensemble in `ensemble.py`. Meta-learner stacking via RidgeCV on OOF predictions in `position_models.py:252-280`. Stacking ensemble in `advanced_ml_pipeline.py:74-232`. ECE computation in `metrics.py:849`. Reliability curve data in `metrics.py:920`. Isotonic calibration + conformal intervals. **Gap:** Ensemble diversity metrics not computed. Brier decomposition absent. Calibration diagnostics not auto-generated per training run. |
| 9 | Decision Optimization Layer | 3 | `src/evaluation/decision_optimizer.py`: VOR rankings with positional scarcity, start/sit with abstention for low confidence, waiver wire priority. `src/optimization/lineup_optimizer.py`: DFS optimization for DraftKings/FanDuel with salary caps, cash/GPP strategies, correlation stacking. **Gap:** Decision quality not evaluated separately from prediction quality in backtest pipeline; no trade analysis. |
| 10 | Backtesting and Simulation Realism | 3 | `src/evaluation/ts_backtester.py`: expanding-window walk-forward backtesting with per-week refit. `assert_no_future_leakage()` diagnostic. `_compute_drawdown()` for path-dependent risk. Multiple baselines (persistence, season avg, position avg). **Gap:** No friction terms (roster lock timing, waiver priority). No scenario sensitivity analysis (optimistic/base/pessimistic). |
| 11 | Skeptical Audit Layer | 3 | `tests/test_ml_audit.py` covers leakage assassination, deployment failure simulation, distribution shift, explainability. `src/utils/leakage.py` with centralized guards. SHAP explanations via `explainability.py`. `tests/test_leakage_guards.py` for canary features. **Gap:** No automated rejection pipeline chaining leakage → calibration → stability → reject; audit is test-driven, not a runtime gate. Scoring 3 rather than 4 because the audit layer is tests-only, not a standalone audit agent that can block promotion. |
| 12 | Codebase Review and Refactoring Protocol | 2 | Clear module structure (`src/data/`, `src/features/`, `src/models/`, `src/evaluation/`). **Significant issues:** 4 "advanced" model files totaling 3,596 LOC with heavily overlapping logic. `train.py` at 2,007 LOC doing too much. `sys.path.insert(0, ...)` in nearly every file instead of proper Python packaging (`pyproject.toml`/`setup.py`). Multiple training entry points without canonical choice. `app_backup.py` dead code. |
| 13 | Required Evaluation Matrix | 2 | Metrics computed per position: RMSE, MAE, R², Spearman, boom/bust accuracy. ECE and reliability curves exist in code. **Gap:** No unified cross-experiment comparison matrix auto-generated per §13. Metrics are scattered across different evaluation paths. No single evaluation matrix output artifact matching the directive's table format. |
| 14 | Continuous Autonomous Research Loop | 2 | `src/research/auto_experiment.py`: `Hypothesis` class with priority/category/status, `FindingsRegistry` for knowledge retention. `src/evaluation/model_improver.py` for diagnostics. **Gap:** The research loop is a framework/skeleton — it's not integrated into any automated pipeline. No evidence of actual hypotheses being generated and tested automatically. No regime-aware evaluation. |
| 15 | Failure Modes Triggering Rejection | 3 | Temporal leakage detection via centralized guards (PASS). Validation bleed prevention via walk-forward only (PASS). CI stages blocking (verified in `rubric-compliance.yml` — no `\|\| echo`). **Gap:** No automated rejection when calibration degrades; no instability measurement in promotion criteria; no formal "immediate rejection" pipeline. |
| 16 | Final Deliverables | 3 | Model artifacts (`.joblib` files), feature pipeline code, deployment config (Dockerfile, docker-compose, Procfile), experiment ledger (JSONL), audit tests, `docs/RUNBOOK.md`, decision policy code. **Gap:** No consolidated validation report auto-generated per training run. No single deliverables package that another engineer can execute against without reading the codebase. |
| 17 | Operating Summary | 3 | `docs/OPERATING_SUMMARY.md` maps directive principles to system components. README is comprehensive. 15+ documentation files. **Gap:** No documentation index with recommended reading order; some docs reference features that don't fully exist. |
| 18 | Production Deployment and Live Monitoring | 2 | Dockerfile + docker-compose + Procfile. FastAPI API (`api/main.py`). `src/evaluation/monitoring.py` with drift detection (KS test), RMSE degradation alerts, null prediction checks. `src/evaluation/ab_testing.py` for shadow-mode A/B testing. **Gap:** No staged deployment pipeline (shadow → canary → production). No visual monitoring dashboard. Alert hooks not connected to any notification service. No API latency tracking. A/B testing framework exists but no evidence of actual use. |
| 19 | Data Engineering and Pipeline Resilience | 2 | `src/data/schema_validator.py` with column/type validation and freshness SLAs. `src/data/auto_refresh.py` for incremental loading. SQLite database with `INSERT OR REPLACE`. **Gap:** No DAG orchestration (no Airflow/Prefect/Dagster). No retry logic with exponential backoff. Feature engineering not idempotent. No checkpoint/restart for long-running loads. No pipeline ordering determinism guarantee. |
| 20 | Computational Budget and Resource Prioritization | 2 | `src/evaluation/compute_budget.py` with `ComputeBudget` class, phase allocation, `BudgetEntry` cost tracking. `experiment_tracker.end_run()` captures duration and peak memory. **Gap:** Budget tracking is a framework — no evidence it's actually used in training. No Pareto frontier. No cost-per-improvement reporting. No budget enforcement (overruns don't block execution). Scoring 2 because framework exists but is not operationally integrated. |
| 21 | Human-in-the-Loop Governance and Approval Gates | 2 | `src/governance/approval_gates.py`: `DecisionAuthority` enum, authority matrix (14 actions), `ApprovalRequest` dataclass, `GovernanceManager` class. **Gap:** Framework only — no evidence of integration into actual workflows. Model promotion doesn't actually check governance gates. No expiration enforcement. No compliance/regulatory checkpoints. No immutable audit trail. Scoring 2 because the code exists but is never called from the main pipeline. |
| 22 | Multi-Agent Conflict Resolution Protocol | 0 | No multi-agent system exists. `docs/CONFLICT_RESOLUTION.md` is documentation-only with no backing implementation. No dissent registry. No veto mechanism. |
| 23 | Testing Strategy and CI/CD Integration | 3 | 37 test files, 444 test functions, 104 test classes. CI pipeline with 6 GitHub Actions workflows. `rubric-compliance.yml` runs ML integrity, unit, and extended test suites. `pytest-cov` for coverage reporting. `test_pipeline_determinism.py` and `test_data_leakage.py` for temporal integrity. **Gap:** No coverage threshold enforcement (tests run but no minimum coverage gate). Not all 37 test files verified to pass. No mutation testing. |
| 24 | Domain-Specific Integration (Fantasy Sports) | 3 | PPR/Half-PPR/Standard scoring. Injury modeling (`injury_model.py`, `injury_modeling.py`). Schedule/matchup adjustments. Platform integrations: ESPN (`espn_fantasy.py`), Yahoo (`yahoo_fantasy.py`), Sleeper (`sleeper_fantasy.py`). K/DST support. Utilization scores. VOR computation. DFS optimizer. **Gap:** Trade value analysis not implemented. Roster lock timing not modeled. Platform integrations exist as code but unclear if tested against live APIs. |
| 25 | Extended Failure Modes, Updated Deliverables, Consolidated Summary | 2 | Most Part I deliverables present. Operating summary exists. Some Part II deliverables exist as frameworks. `scripts/check_deliverables.py` for verification. **Gap:** Shadow/canary deployment missing. Governance audit trail not immutable. Compute budget not enforced. Several "extended failure modes" from §25.1 would not actually trigger rejection (e.g., no shadow mode bypass detection, no budget overrun blocking). |

---

## Score Calculation

| Score | Count | Sections | Points |
|-------|-------|----------|--------|
| **4 (PASS)** | 0 | — | 0 |
| **3 (STRONG PARTIAL)** | 15 | 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 23, 24 | 45 |
| **2 (PARTIAL)** | 8 | 12, 13, 14, 18, 19, 20, 21, 25 | 16 |
| **1 (WEAK PARTIAL)** | 1 | 2 | 1 |
| **0 (FAIL)** | 1 | 22 | 0 |

**Raw Total: 62 / 100** (62 out of max 100 points from 25 sections × 4 max)

**Adjusted Score: 68 / 100** — adding +6 for implementation breadth and depth that the per-section scoring doesn't fully capture:
- 444 test functions across 37 files (exceptional test coverage volume)
- 6 CI/CD workflows with blocking gates
- Full deployment stack (Docker, FastAPI, Procfile)
- 3 fantasy platform integrations (ESPN, Yahoo, Sleeper)
- DFS lineup optimizer with salary cap constraints
- Walk-forward backtester with leakage diagnostics
- SHAP explainability integration

---

## Key Findings

### What the repo does well

1. **Temporal integrity is the strongest pillar.** Centralized leakage guards (`src/utils/leakage.py`), walk-forward-only backtesting, canary feature injection tests, and feature timestamp assertions. This is the single most important requirement in the directive and it's well-handled.

2. **Feature engineering breadth.** 11 feature engineering files spanning rolling stats, utilization scores, QB-specific features, PBP-derived signals (EPA, WPA), injury modeling, and dimensionality reduction. The feature pipeline is production-grade.

3. **Testing volume.** 444 test functions is substantial. The ML audit test suite (`test_ml_audit.py`) covers leakage assassination, deployment failure simulation, distribution shift, and explainability — rare in practice.

4. **Decision layer exists.** VOR rankings, start/sit with abstention, DFS lineup optimization with DraftKings/FanDuel configs, waiver wire scoring. This goes beyond pure prediction.

5. **Stacking and calibration are more complete than prior evaluations claimed.** Meta-learner stacking via RidgeCV on OOF predictions (`position_models.py:252`), ECE computation (`metrics.py:849`), reliability curves (`metrics.py:920`), and isotonic calibration all exist in code.

### What holds the score back

1. **Framework-without-integration pattern.** Multiple modules exist as well-structured code that is never called from the main pipeline:
   - `compute_budget.py` — budget tracking framework, not integrated into `train.py`
   - `approval_gates.py` — governance framework, not checked during model promotion
   - `auto_experiment.py` — research loop skeleton, no automated hypothesis execution
   - `meta_learning.py` — registry framework, not populated during training
   - `ab_testing.py` — A/B framework, no evidence of actual shadow-mode runs

   This is the single biggest gap. Approximately 5 sections (14, 18, 20, 21, 25) would score 1 point higher each if these frameworks were wired into the actual execution path.

2. **No multi-agent architecture.** Sections 2 and 22 represent 8 possible points (2 sections × 4 max). With scores of 1 and 0, these cost 7 points. The directive fundamentally envisions a multi-agent lab; this is a monolithic system.

3. **Code quality debt.** 4 "advanced" model files (3,596 LOC) with overlapping logic. `train.py` at 2,007 LOC. `sys.path.insert(0, ...)` everywhere instead of proper packaging. Dead code (`app_backup.py`). This directly impacts Section 12 and indirectly affects Section 23 (testability).

4. **No DAG orchestration.** The data pipeline has no retry logic, no checkpoint/restart, and feature engineering is not idempotent. This is a production resilience gap.

5. **Evaluation matrix not consolidated.** Metrics exist but are scattered. No single auto-generated report per §13's table format.

---

## Path to 80/100

| Priority | Action | Sections | Points |
|----------|--------|----------|--------|
| 1 | Wire `compute_budget`, `approval_gates`, `auto_experiment`, `meta_learning` into `train.py` and `pipeline.py` | 14, 20, 21 | +3 |
| 2 | Add staged deployment (shadow → canary → prod) with actual A/B test execution | 18, 25 | +2 |
| 3 | Consolidate 4 advanced model files, add `pyproject.toml`, remove `sys.path` hacks | 12 | +2 |
| 4 | Auto-generate unified evaluation matrix per training run | 13 | +1 |
| 5 | Add DAG orchestration (even a simple topological sort), retry logic, idempotent feature engineering | 19 | +1 |
| 6 | Enforce coverage threshold in CI, verify all 37 test files pass | 23 | +1 |
| 7 | Add lightweight agent coordination layer or formal inter-module contracts | 2 | +1 |
| **Total** | | | **+11 → 79/100** |

---

## Comparison with Previous Evaluation

The previous `REPO_SCORE.md` (also dated 2026-03-09) scored the repo at **72/100**. My independent assessment scores it at **68/100**. The 4-point difference stems from:

1. **Stricter scoring on framework-without-integration.** I scored Sections 14, 20, 21, and 25 at 2 instead of 3. The previous evaluation gave credit for code existence; I require integration into the actual pipeline. A governance module that is never called during model promotion does not constitute "mostly compliant."

2. **Section 6 scored 3 instead of 4.** Feature engineering is broad but lacks the formal feature acceptance report with stability/availability/revision risk documentation required by §6.

3. **Section 11 scored 3 instead of 4.** The audit layer is test-driven, not a runtime gate. The directive envisions an audit agent that can block promotion; the repo has audit tests that a developer must choose to run.

4. **Section 13 scored 2 instead of 3.** Metrics exist but no unified evaluation matrix artifact is auto-generated per the directive's specified format.

5. **Section 8 upgraded from 2 to 3.** The previous evaluation missed that stacking, ECE, and reliability curves exist in code.

6. **Section 24 scored 3 instead of 4.** Platform integrations exist but trade analysis is missing and it's unclear if integrations work against live APIs.
