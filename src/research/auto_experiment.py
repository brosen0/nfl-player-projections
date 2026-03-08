"""
Continuous Autonomous Research Loop.

Per Agent Directive V7 Section 14: the system must continuously generate
hypotheses, execute experiments, adversarially review results, and retain
knowledge.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).parent.parent.parent / "data" / "research"


class Hypothesis:
    """A research hypothesis to test."""

    def __init__(
        self,
        hypothesis_id: str,
        description: str,
        category: str = "feature",
        priority: int = 5,
        estimated_compute_seconds: float = 300,
        status: str = "pending",
    ):
        self.hypothesis_id = hypothesis_id
        self.description = description
        self.category = category  # feature, model, ensemble, data, policy
        self.priority = priority  # 1=highest, 10=lowest
        self.estimated_compute_seconds = estimated_compute_seconds
        self.status = status  # pending, running, completed, failed, rejected
        self.result: Optional[Dict[str, Any]] = None
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "estimated_compute_seconds": self.estimated_compute_seconds,
            "status": self.status,
            "result": self.result,
            "created_at": self.created_at,
        }


class FindingsRegistry:
    """Persistent knowledge retention for experiment outcomes.

    Per Directive V7 Section 14: store findings so future agents learn
    what tends to work by domain, horizon, and data regime.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or (_DEFAULT_DIR / "findings_registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.findings: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.findings = data.get("findings", [])
            except (json.JSONDecodeError, IOError):
                self.findings = []

    def save(self) -> None:
        data = {
            "updated_at": datetime.now().isoformat(),
            "n_findings": len(self.findings),
            "findings": self.findings,
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_finding(
        self,
        finding: str,
        experiment_ids: List[str],
        confidence: float = 0.8,
        category: str = "general",
        still_valid: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an experiment finding for future reference.

        Args:
            finding: Plain-language description of the finding.
            experiment_ids: IDs of experiments that support the finding.
            confidence: Confidence level (0-1).
            category: Category (model, feature, calibration, etc.).
            still_valid: Whether the finding is still believed to be valid.
            details: Additional data.
        """
        entry = {
            "finding": finding,
            "experiment_ids": experiment_ids,
            "confidence": round(confidence, 2),
            "category": category,
            "still_valid": still_valid,
            "date_discovered": datetime.now().isoformat(),
        }
        if details:
            entry["details"] = details
        self.findings.append(entry)
        self.save()
        logger.info("Finding recorded: %s (confidence: %.0f%%)", finding, confidence * 100)

    def get_valid_findings(self, category: str = "") -> List[Dict[str, Any]]:
        """Get findings that are still believed to be valid."""
        valid = [f for f in self.findings if f.get("still_valid", True)]
        if category:
            valid = [f for f in valid if f.get("category") == category]
        return valid

    def invalidate_finding(self, finding_text: str) -> None:
        """Mark a finding as no longer valid."""
        for f in self.findings:
            if f["finding"] == finding_text:
                f["still_valid"] = False
                f["invalidated_at"] = datetime.now().isoformat()
        self.save()


class ResearchLoop:
    """Manages the autonomous research cycle.

    Per Directive V7 Section 14 Loop Schema:
    1. Hypothesis generation
    2. Experiment execution
    3. Adversarial review
    4. Promotion gate
    5. Knowledge retention
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or _DEFAULT_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.queue_path = self.data_dir / "hypothesis_queue.json"
        self.findings = FindingsRegistry(self.data_dir / "findings_registry.json")
        self.queue: List[Hypothesis] = []
        self._load_queue()

    def _load_queue(self) -> None:
        if self.queue_path.exists():
            try:
                with open(self.queue_path, encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("hypotheses", []):
                    h = Hypothesis(
                        hypothesis_id=item["hypothesis_id"],
                        description=item["description"],
                        category=item.get("category", "general"),
                        priority=item.get("priority", 5),
                        estimated_compute_seconds=item.get("estimated_compute_seconds", 300),
                        status=item.get("status", "pending"),
                    )
                    h.result = item.get("result")
                    self.queue.append(h)
            except (json.JSONDecodeError, IOError):
                pass

    def save_queue(self) -> None:
        data = {
            "updated_at": datetime.now().isoformat(),
            "hypotheses": [h.to_dict() for h in self.queue],
        }
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_hypothesis(
        self,
        description: str,
        category: str = "feature",
        priority: int = 5,
        estimated_compute_seconds: float = 300,
    ) -> Hypothesis:
        """Add a new hypothesis to the research queue."""
        h_id = f"H{len(self.queue) + 1:04d}"
        h = Hypothesis(
            hypothesis_id=h_id,
            description=description,
            category=category,
            priority=priority,
            estimated_compute_seconds=estimated_compute_seconds,
        )
        self.queue.append(h)
        self.save_queue()
        return h

    def get_next_hypothesis(
        self, max_compute_seconds: float = float("inf"),
    ) -> Optional[Hypothesis]:
        """Get the highest-priority pending hypothesis within budget.

        Per Directive V7 Section 20: respect compute budget when
        scheduling experiments.
        """
        pending = [
            h for h in self.queue
            if h.status == "pending"
            and h.estimated_compute_seconds <= max_compute_seconds
        ]
        if not pending:
            return None
        return min(pending, key=lambda h: h.priority)

    def complete_hypothesis(
        self,
        hypothesis_id: str,
        result: Dict[str, Any],
        improved: bool = False,
        finding: str = "",
    ) -> None:
        """Mark a hypothesis as completed and optionally record a finding."""
        for h in self.queue:
            if h.hypothesis_id == hypothesis_id:
                h.status = "completed"
                h.result = result
                break

        if finding:
            self.findings.record_finding(
                finding=finding,
                experiment_ids=[hypothesis_id],
                confidence=0.8 if improved else 0.5,
                category="research",
            )

        self.save_queue()
