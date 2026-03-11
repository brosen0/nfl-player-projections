"""
Multi-Agent Conflict Resolution Protocol.

Per Agent Directive V7 Section 22: establishes a formal conflict resolution
protocol so that disagreements are surfaced, documented, and resolved
systematically rather than silently overridden.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_GOV_DIR = Path(__file__).parent.parent.parent / "data" / "governance"


class ConflictCategory:
    FACTUAL = "factual"
    PRIORITY = "priority"
    SAFETY = "safety"
    RESOURCE = "resource"


class ConflictStatus:
    OPEN = "open"
    RESOLVED = "resolved"
    VETOED = "vetoed"
    ESCALATED = "escalated"


@dataclass
class Conflict:
    """A logged conflict between system components."""
    conflict_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    category: str = ""
    agent_a: str = ""
    agent_b: str = ""
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: str = ConflictStatus.OPEN
    resolution: str = ""
    resolved_by: str = ""
    resolved_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Dissent:
    """A registered dissent from an agent that disagrees with a resolution."""
    dissent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    conflict_id: str = ""
    agent: str = ""
    reasoning: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ConflictResolver:
    """Manages conflict detection, resolution, and dissent tracking.

    Per Directive V7 Section 22.2: resolution hierarchy is
    1. Evidence duel
    2. Audit arbitration (Audit Agent's finding is binding on factual matters)
    3. Orchestrator decision (for priority/resource disagreements)
    4. Human escalation

    Per Section 22.3: the Audit Agent holds veto power on safety matters.
    """

    def __init__(self, gov_dir: Optional[Path] = None):
        self.gov_dir = gov_dir or _DEFAULT_GOV_DIR
        self.gov_dir.mkdir(parents=True, exist_ok=True)
        self.conflict_log = self.gov_dir / "conflict_log.jsonl"
        self.dissent_log = self.gov_dir / "dissent_registry.jsonl"

    def log_conflict(
        self,
        category: str,
        agent_a: str,
        agent_b: str,
        description: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Conflict:
        """Log a new conflict between components.

        Args:
            category: One of ConflictCategory values.
            agent_a: First component involved.
            agent_b: Second component involved.
            description: Plain-language description.
            evidence: Supporting data from both sides.

        Returns:
            Conflict record with unique ID.
        """
        conflict = Conflict(
            category=category,
            agent_a=agent_a,
            agent_b=agent_b,
            description=description,
            evidence=evidence or {},
        )
        self._append_jsonl(self.conflict_log, asdict(conflict))
        logger.info(
            "Conflict logged: %s between %s and %s — %s",
            conflict.conflict_id, agent_a, agent_b, description,
        )
        return conflict

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        resolved_by: str,
    ) -> None:
        """Mark a conflict as resolved.

        Args:
            conflict_id: ID of the conflict to resolve.
            resolution: Description of the resolution.
            resolved_by: Who resolved it (agent name or "human_operator").
        """
        entry = {
            "conflict_id": conflict_id,
            "status": ConflictStatus.RESOLVED,
            "resolution": resolution,
            "resolved_by": resolved_by,
            "resolved_at": datetime.now().isoformat(),
        }
        self._append_jsonl(self.conflict_log, entry)
        logger.info("Conflict %s resolved by %s: %s", conflict_id, resolved_by, resolution)

    def audit_veto(
        self,
        conflict_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Exercise the Audit Agent's veto power on safety matters.

        Per Section 22.3: if the Audit Agent identifies confirmed temporal
        leakage, validation contamination, or any Section 15 failure mode,
        no other agent may override. The only path is to fix and resubmit.

        Args:
            conflict_id: The conflict being vetoed.
            reason: Specific safety concern (leakage, contamination, etc.).

        Returns:
            Veto record dict.
        """
        veto = {
            "conflict_id": conflict_id,
            "status": ConflictStatus.VETOED,
            "veto_reason": reason,
            "vetoed_by": "audit_agent",
            "vetoed_at": datetime.now().isoformat(),
            "blocks_promotion": True,
        }
        self._append_jsonl(self.conflict_log, veto)
        logger.warning(
            "AUDIT VETO on conflict %s: %s — promotion blocked",
            conflict_id, reason,
        )
        return veto

    def submit_dissent(
        self,
        agent: str,
        conflict_id: str,
        reasoning: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dissent:
        """File a dissent against a resolution.

        Per Section 22.4: dissents don't block action but create a record
        that can be revisited if future evidence vindicates the position.

        Args:
            agent: The dissenting component.
            conflict_id: The conflict being dissented against.
            reasoning: Why the agent disagrees.
            evidence: Supporting evidence.

        Returns:
            Dissent record.
        """
        dissent = Dissent(
            conflict_id=conflict_id,
            agent=agent,
            reasoning=reasoning,
            evidence=evidence or {},
        )
        self._append_jsonl(self.dissent_log, asdict(dissent))
        logger.info(
            "Dissent filed by %s on conflict %s: %s",
            agent, conflict_id, reasoning,
        )
        return dissent

    def get_open_conflicts(self) -> List[Dict[str, Any]]:
        """Get all unresolved conflicts."""
        return self._read_by_status(self.conflict_log, ConflictStatus.OPEN)

    def get_open_dissents(self) -> List[Dict[str, Any]]:
        """Get all filed dissents."""
        return self._read_all(self.dissent_log)

    def has_active_veto(self) -> bool:
        """Check if any active audit veto blocks promotion."""
        if not self.conflict_log.exists():
            return False
        vetoes = self._read_by_status(self.conflict_log, ConflictStatus.VETOED)
        return any(v.get("blocks_promotion", False) for v in vetoes)

    def get_summary(self) -> Dict[str, Any]:
        """Dashboard summary of conflict state."""
        all_conflicts = self._read_all(self.conflict_log)
        all_dissents = self._read_all(self.dissent_log)
        open_conflicts = [c for c in all_conflicts if c.get("status") == ConflictStatus.OPEN]
        vetoed = [c for c in all_conflicts if c.get("status") == ConflictStatus.VETOED]
        return {
            "total_conflicts": len(all_conflicts),
            "open_conflicts": len(open_conflicts),
            "active_vetoes": len([v for v in vetoed if v.get("blocks_promotion")]),
            "total_dissents": len(all_dissents),
            "timestamp": datetime.now().isoformat(),
        }

    def _read_by_status(self, path: Path, status: str) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        results = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("status") == status:
                        results.append(entry)
                except json.JSONDecodeError:
                    continue
        return results

    def _read_all(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        results = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return results

    def _append_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write to %s: %s", path, e)
