"""
Compute Budget and Resource Prioritization.

Per Agent Directive V7 Section 20: without resource constraints, the agent
mandate can consume unbounded compute. This module defines a budget-aware
search protocol that maximizes discovery per unit of compute.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_BUDGET_DIR = Path(__file__).parent.parent.parent / "data" / "experiments"


# ---------------------------------------------------------------------------
# Budget allocation (Section 20.1)
# ---------------------------------------------------------------------------

DEFAULT_PHASE_ALLOCATION: Dict[str, float] = {
    "data_loading": 0.10,
    "feature_engineering": 0.15,
    "model_training": 0.40,
    "ensemble_optimization": 0.15,
    "evaluation": 0.20,
}


@dataclass
class BudgetEntry:
    """Single compute cost entry."""
    phase: str
    task: str
    duration_seconds: float
    memory_peak_mb: Optional[float] = None
    started_at: str = ""
    metric_improvement: Optional[float] = None


@dataclass
class ComputeBudget:
    """Track and enforce compute budget per Directive V7 Section 20.

    Attributes:
        total_budget_seconds: Total wall-clock budget in seconds.
        phase_allocation: Fraction of budget per phase.
    """
    total_budget_seconds: float = 3600.0  # 1 hour default
    phase_allocation: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_ALLOCATION)
    )
    entries: List[BudgetEntry] = field(default_factory=list)

    @property
    def consumed_seconds(self) -> float:
        return sum(e.duration_seconds for e in self.entries)

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.total_budget_seconds - self.consumed_seconds)

    @property
    def consumed_pct(self) -> float:
        if self.total_budget_seconds <= 0:
            return 100.0
        return (self.consumed_seconds / self.total_budget_seconds) * 100

    def consumed_by_phase(self) -> Dict[str, float]:
        """Breakdown of consumed time by phase."""
        by_phase: Dict[str, float] = {}
        for entry in self.entries:
            by_phase[entry.phase] = by_phase.get(entry.phase, 0) + entry.duration_seconds
        return by_phase

    def is_over_budget(self) -> bool:
        return self.consumed_seconds >= self.total_budget_seconds

    def phase_over_budget(self, phase: str) -> bool:
        """Check if a specific phase has exceeded its allocation."""
        allocation = self.phase_allocation.get(phase, 0.10)
        phase_budget = self.total_budget_seconds * allocation
        phase_consumed = self.consumed_by_phase().get(phase, 0)
        return phase_consumed >= phase_budget

    def log_task(
        self,
        phase: str,
        task: str,
        duration_seconds: float,
        memory_peak_mb: Optional[float] = None,
        metric_improvement: Optional[float] = None,
    ) -> None:
        """Log a completed task's compute cost."""
        self.entries.append(BudgetEntry(
            phase=phase,
            task=task,
            duration_seconds=round(duration_seconds, 2),
            memory_peak_mb=memory_peak_mb,
            started_at=datetime.now().isoformat(),
            metric_improvement=metric_improvement,
        ))

        if self.is_over_budget():
            logger.warning(
                "COMPUTE BUDGET EXCEEDED: %.1f / %.1f seconds consumed (%.1f%%)",
                self.consumed_seconds, self.total_budget_seconds, self.consumed_pct,
            )

    def compute_efficiency_ratio(self) -> Optional[float]:
        """Cost per unit improvement on primary metric.

        Per Section 20.3: compute efficiency ratio = cost / improvement.
        Lower is better.
        """
        improvements = [
            e for e in self.entries
            if e.metric_improvement is not None and e.metric_improvement > 0
        ]
        if not improvements:
            return None
        total_cost = sum(e.duration_seconds for e in improvements)
        total_improvement = sum(e.metric_improvement for e in improvements)
        return total_cost / total_improvement if total_improvement > 0 else None

    def pareto_frontier(self) -> List[Tuple[float, float]]:
        """Compute Pareto frontier of (cumulative_cost, cumulative_improvement).

        Per Section 20.3: show trade-off between compute invested and
        performance achieved.
        """
        points: List[Tuple[float, float]] = []
        cum_cost = 0.0
        cum_improvement = 0.0
        for entry in self.entries:
            cum_cost += entry.duration_seconds
            if entry.metric_improvement is not None:
                cum_improvement += max(0, entry.metric_improvement)
            points.append((round(cum_cost, 2), round(cum_improvement, 4)))
        return points

    def get_summary(self) -> Dict[str, Any]:
        """Generate budget summary report per Section 20.3."""
        by_phase = self.consumed_by_phase()
        phase_budget = {
            phase: round(self.total_budget_seconds * alloc, 1)
            for phase, alloc in self.phase_allocation.items()
        }
        return {
            "total_budget_seconds": self.total_budget_seconds,
            "consumed_seconds": round(self.consumed_seconds, 2),
            "remaining_seconds": round(self.remaining_seconds, 2),
            "consumed_pct": round(self.consumed_pct, 1),
            "is_over_budget": self.is_over_budget(),
            "by_phase": {
                phase: {
                    "consumed": round(by_phase.get(phase, 0), 2),
                    "budget": phase_budget.get(phase, 0),
                    "over_budget": by_phase.get(phase, 0) >= phase_budget.get(phase, 0),
                }
                for phase in set(list(self.phase_allocation.keys()) + list(by_phase.keys()))
            },
            "efficiency_ratio": self.compute_efficiency_ratio(),
            "n_tasks": len(self.entries),
            "timestamp": datetime.now().isoformat(),
        }

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Persist budget state to JSON."""
        output_dir = output_dir or _DEFAULT_BUDGET_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "compute_budget.json"
        data = self.get_summary()
        data["entries"] = [
            {
                "phase": e.phase,
                "task": e.task,
                "duration_seconds": e.duration_seconds,
                "memory_peak_mb": e.memory_peak_mb,
                "metric_improvement": e.metric_improvement,
                "started_at": e.started_at,
            }
            for e in self.entries
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
