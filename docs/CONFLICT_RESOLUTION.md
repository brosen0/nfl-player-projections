# Multi-Agent Conflict Resolution Protocol

Per Agent Directive V7 Section 22 — defines how disagreements between
system components are surfaced, documented, and resolved.

---

## Conflict Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Metric disagreement** | Components disagree on which metric to optimize | Model Agent optimizes RMSE; Decision Agent prefers ranking accuracy |
| **Priority conflict** | Disagreement on what to work on next | Feature Agent wants more data; Model Agent wants more training |
| **Safety concern** | One component identifies a risk another ignores | Audit Agent flags leakage; Model Agent dismisses it |

---

## Resolution Hierarchy

Conflicts are resolved through structured escalation:

1. **Empirical evidence** — Run an experiment to settle the disagreement.
   The component with stronger data wins.
2. **Safety first** — Safety concerns always take precedence over
   performance improvements. If the Audit Agent (leakage.py) flags an
   issue, it blocks promotion.
3. **Research Orchestrator** — The pipeline orchestrator (`src/pipeline.py`)
   makes the final call on priority conflicts.
4. **Human escalation** — Unresolvable conflicts go to the governance
   framework (`src/governance/approval_gates.py`).

---

## Audit Agent Veto (Section 22.3)

The Audit Agent (`src/utils/leakage.py`, `tests/test_ml_audit.py`) holds
special veto power on safety matters:

- **Confirmed temporal leakage** → blocks promotion
- **Validation contamination** → blocks promotion
- **Any Section 15 failure mode** → blocks promotion

The veto cannot be overridden by other agents. The only path forward is
to fix the underlying issue and resubmit for audit.

The veto cannot be used for priority or resource disagreements.

---

## Dissent Registry

Any system component that disagrees with a resolution may file a dissent
in `data/governance/dissent_log.jsonl`.

Dissent format:
```json
{
  "timestamp": "2026-03-08T12:00:00",
  "component": "model_agent",
  "disagreement": "Feature set v2.3 dropped high-importance features",
  "evidence": "SHAP importance > 0.05 for 3 removed features",
  "resolution": "Feature Agent decision upheld (multicollinearity)",
  "status": "open"
}
```

The Research Orchestrator reviews open dissents at the start of each
research cycle.

---

## Module-Level Conflict Mapping

| Source Module | Target Module | Likely Conflict | Resolution |
|--------------|---------------|-----------------|------------|
| `src/features/` | `src/models/` | Feature count vs model complexity | Experiment with both |
| `src/models/` | `src/evaluation/ab_testing.py` | Model ready vs promotion criteria | A/B test decides |
| `src/utils/leakage.py` | `src/features/` | Feature flagged as leaky | Leakage guard wins |
| `src/evaluation/monitoring.py` | `src/evaluation/ab_testing.py` | Drift detected during A/B test | Pause promotion |
