"""
Human-in-the-Loop Governance and Approval Gates.

Per Agent Directive V7 Section 21: high-stakes prediction systems require
human oversight at critical junctures. This module defines a governance
framework that preserves agent autonomy for routine operations while
requiring human approval for significant actions.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_GOV_DIR = Path(__file__).parent.parent.parent / "data" / "governance"


class DecisionAuthority(str, Enum):
    """Authority levels per Directive V7 Section 21.1."""
    AUTONOMOUS = "autonomous"
    NEEDS_APPROVAL = "needs_approval"
    REQUIRES_ESCALATION = "requires_escalation"


# ---------------------------------------------------------------------------
# Decision Authority Matrix
# ---------------------------------------------------------------------------
AUTHORITY_MATRIX: Dict[str, DecisionAuthority] = {
    # Autonomous actions (no approval needed)
    "data_refresh": DecisionAuthority.AUTONOMOUS,
    "feature_engineering": DecisionAuthority.AUTONOMOUS,
    "model_training": DecisionAuthority.AUTONOMOUS,
    "experiment_logging": DecisionAuthority.AUTONOMOUS,
    "monitoring_alert": DecisionAuthority.AUTONOMOUS,
    "prediction_generation": DecisionAuthority.AUTONOMOUS,
    # Actions requiring approval
    "model_promotion": DecisionAuthority.NEEDS_APPROVAL,
    "config_change": DecisionAuthority.NEEDS_APPROVAL,
    "hyperparameter_override": DecisionAuthority.NEEDS_APPROVAL,
    "feature_set_change": DecisionAuthority.NEEDS_APPROVAL,
    "threshold_adjustment": DecisionAuthority.NEEDS_APPROVAL,
    # Actions requiring escalation
    "production_deployment": DecisionAuthority.REQUIRES_ESCALATION,
    "model_rollback": DecisionAuthority.REQUIRES_ESCALATION,
    "data_schema_change": DecisionAuthority.REQUIRES_ESCALATION,
    "circuit_breaker_override": DecisionAuthority.REQUIRES_ESCALATION,
}


@dataclass
class ApprovalRequest:
    """Structured approval request per Directive V7 Section 21.2."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action: str = ""
    action_summary: str = ""
    evidence_package: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: str = ""
    rollback_plan: str = ""
    requested_by: str = "system"
    requested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str = field(
        default_factory=lambda: (datetime.now() + timedelta(hours=24)).isoformat()
    )
    status: str = "pending"  # pending, approved, denied, expired
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    decision_justification: Optional[str] = None


class GovernanceManager:
    """Manages approval workflows and audit trails.

    Per Directive V7 Section 21.4: every governance action must be logged
    to an immutable audit trail.
    """

    def __init__(self, gov_dir: Optional[Path] = None):
        self.gov_dir = gov_dir or _DEFAULT_GOV_DIR
        self.gov_dir.mkdir(parents=True, exist_ok=True)
        self.requests_log = self.gov_dir / "approval_requests.jsonl"
        self.decisions_log = self.gov_dir / "approval_log.jsonl"

    def check_authority(self, action: str) -> DecisionAuthority:
        """Check what authority level an action requires."""
        return AUTHORITY_MATRIX.get(action, DecisionAuthority.NEEDS_APPROVAL)

    def request_approval(
        self,
        action: str,
        action_summary: str,
        evidence_package: Optional[Dict[str, Any]] = None,
        risk_assessment: str = "",
        rollback_plan: str = "",
    ) -> ApprovalRequest:
        """Submit a structured approval request.

        Args:
            action: Action type (must be in AUTHORITY_MATRIX).
            action_summary: Plain-language description.
            evidence_package: Links to experiment records, reports.
            risk_assessment: Quantified downside estimate.
            rollback_plan: Steps to revert the action.

        Returns:
            ApprovalRequest with unique ID for tracking.
        """
        authority = self.check_authority(action)
        if authority == DecisionAuthority.AUTONOMOUS:
            # Auto-approve autonomous actions
            req = ApprovalRequest(
                action=action,
                action_summary=action_summary,
                evidence_package=evidence_package or {},
                status="auto_approved",
                decided_at=datetime.now().isoformat(),
                decision_justification="Action classified as autonomous",
            )
        else:
            req = ApprovalRequest(
                action=action,
                action_summary=action_summary,
                evidence_package=evidence_package or {},
                risk_assessment=risk_assessment,
                rollback_plan=rollback_plan,
            )

        self._log_request(req)
        logger.info("Approval request submitted: %s (%s)", req.request_id, action)
        return req

    def approve(
        self,
        request_id: str,
        decided_by: str = "human_operator",
        justification: str = "",
    ) -> bool:
        """Approve a pending request."""
        return self._decide(request_id, "approved", decided_by, justification)

    def deny(
        self,
        request_id: str,
        decided_by: str = "human_operator",
        justification: str = "",
    ) -> bool:
        """Deny a pending request."""
        return self._decide(request_id, "denied", decided_by, justification)

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        if not self.requests_log.exists():
            return []
        pending = []
        with open(self.requests_log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    if req.get("status") == "pending":
                        # Check expiration
                        expires = req.get("expires_at", "")
                        if expires and datetime.fromisoformat(expires) < datetime.now():
                            continue
                        pending.append(req)
                except (json.JSONDecodeError, ValueError):
                    continue
        return pending

    def get_audit_trail(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Read the governance audit trail."""
        if not self.decisions_log.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(self.decisions_log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-last_n:]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decide(
        self, request_id: str, status: str, decided_by: str, justification: str
    ) -> bool:
        decision = {
            "request_id": request_id,
            "status": status,
            "decided_by": decided_by,
            "decided_at": datetime.now().isoformat(),
            "justification": justification,
        }
        self._log_decision(decision)
        logger.info("Request %s: %s by %s", request_id, status, decided_by)
        return True

    def _log_request(self, req: ApprovalRequest) -> None:
        try:
            with open(self.requests_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(req), default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to log approval request: %s", e)

    def _log_decision(self, decision: Dict[str, Any]) -> None:
        try:
            with open(self.decisions_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(decision, default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to log decision: %s", e)
