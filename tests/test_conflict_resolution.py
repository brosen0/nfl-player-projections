"""Tests for Directive V7 Section 22: Conflict Resolution Protocol."""
import json
import tempfile
from pathlib import Path

import pytest

from src.governance.conflict_resolution import (
    ConflictCategory,
    ConflictResolver,
    ConflictStatus,
)


@pytest.fixture
def resolver(tmp_path):
    return ConflictResolver(gov_dir=tmp_path)


class TestConflictLogging:
    def test_log_conflict(self, resolver):
        conflict = resolver.log_conflict(
            category=ConflictCategory.SAFETY,
            agent_a="model_agent",
            agent_b="audit_agent",
            description="Potential leakage detected in feature X",
            evidence={"feature": "X", "ks_stat": 0.95},
        )
        assert conflict.conflict_id
        assert conflict.category == ConflictCategory.SAFETY
        assert conflict.status == ConflictStatus.OPEN

    def test_open_conflicts(self, resolver):
        resolver.log_conflict(
            category=ConflictCategory.FACTUAL,
            agent_a="a",
            agent_b="b",
            description="test conflict",
        )
        open_c = resolver.get_open_conflicts()
        assert len(open_c) == 1
        assert open_c[0]["category"] == ConflictCategory.FACTUAL

    def test_resolve_conflict(self, resolver):
        conflict = resolver.log_conflict(
            category=ConflictCategory.PRIORITY,
            agent_a="a",
            agent_b="b",
            description="test",
        )
        resolver.resolve_conflict(
            conflict.conflict_id,
            resolution="Evidence supports agent_a",
            resolved_by="orchestrator",
        )
        # After resolution, it should appear in the log
        all_entries = resolver._read_all(resolver.conflict_log)
        resolved = [e for e in all_entries if e.get("status") == ConflictStatus.RESOLVED]
        assert len(resolved) == 1


class TestAuditVeto:
    def test_veto_blocks_promotion(self, resolver):
        conflict = resolver.log_conflict(
            category=ConflictCategory.SAFETY,
            agent_a="model_agent",
            agent_b="audit_agent",
            description="Temporal leakage confirmed",
        )
        veto = resolver.audit_veto(conflict.conflict_id, "Confirmed future data in features")
        assert veto["blocks_promotion"] is True
        assert resolver.has_active_veto() is True

    def test_no_veto_initially(self, resolver):
        assert resolver.has_active_veto() is False


class TestDissentRegistry:
    def test_submit_dissent(self, resolver):
        conflict = resolver.log_conflict(
            category=ConflictCategory.PRIORITY,
            agent_a="a",
            agent_b="b",
            description="test",
        )
        dissent = resolver.submit_dissent(
            agent="feature_agent",
            conflict_id=conflict.conflict_id,
            reasoning="The feature is valid based on publication timing",
            evidence={"source_timestamp": "2024-01-01T08:00:00"},
        )
        assert dissent.dissent_id
        assert dissent.conflict_id == conflict.conflict_id

        dissents = resolver.get_open_dissents()
        assert len(dissents) == 1

    def test_multiple_dissents(self, resolver):
        conflict = resolver.log_conflict(
            category=ConflictCategory.FACTUAL,
            agent_a="a",
            agent_b="b",
            description="test",
        )
        resolver.submit_dissent("agent_a", conflict.conflict_id, "reason 1")
        resolver.submit_dissent("agent_b", conflict.conflict_id, "reason 2")
        assert len(resolver.get_open_dissents()) == 2


class TestSummary:
    def test_empty_summary(self, resolver):
        summary = resolver.get_summary()
        assert summary["total_conflicts"] == 0
        assert summary["open_conflicts"] == 0
        assert summary["active_vetoes"] == 0
        assert summary["total_dissents"] == 0

    def test_summary_with_data(self, resolver):
        c = resolver.log_conflict(
            category=ConflictCategory.SAFETY,
            agent_a="a",
            agent_b="b",
            description="test",
        )
        resolver.audit_veto(c.conflict_id, "leakage")
        resolver.submit_dissent("agent_c", c.conflict_id, "disagree")

        summary = resolver.get_summary()
        assert summary["total_conflicts"] >= 1
        assert summary["active_vetoes"] >= 1
        assert summary["total_dissents"] >= 1
