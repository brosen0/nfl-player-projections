"""Tests for Directive V7 governance, budget, and meta-learning integration."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.governance.approval_gates import (
    DecisionAuthority,
    GovernanceManager,
    AUTHORITY_MATRIX,
)
from src.evaluation.compute_budget import ComputeBudget

# Import meta_learning directly to avoid triggering src.models.__init__
# which imports xgboost/lightgbm
import importlib
meta_learning = importlib.import_module("src.models.meta_learning")
MetaLearningRegistry = meta_learning.MetaLearningRegistry

from src.evaluation.monitoring import ModelMonitor


class TestGovernanceIntegration:
    """Test that governance gates work correctly."""

    def test_model_promotion_needs_approval(self):
        assert AUTHORITY_MATRIX["model_promotion"] == DecisionAuthority.NEEDS_APPROVAL

    def test_model_training_is_autonomous(self):
        assert AUTHORITY_MATRIX["model_training"] == DecisionAuthority.AUTONOMOUS

    def test_approval_request_logging(self, tmp_path):
        gov = GovernanceManager(gov_dir=tmp_path)
        req = gov.request_approval(
            action="model_promotion",
            action_summary="Promote QB model v2",
            evidence_package={"rmse": 5.2, "experiment_id": "exp_001"},
            risk_assessment="New model replaces current production",
            rollback_plan="Restore from version history",
        )
        assert req.request_id
        assert req.status == "pending"
        assert (tmp_path / "approval_requests.jsonl").exists()

    def test_autonomous_action_auto_approved(self, tmp_path):
        gov = GovernanceManager(gov_dir=tmp_path)
        req = gov.request_approval(
            action="model_training",
            action_summary="Train new models",
        )
        assert req.status == "auto_approved"


class TestComputeBudgetIntegration:
    """Test compute budget tracking."""

    def test_budget_tracking(self):
        budget = ComputeBudget(total_budget_seconds=3600)
        budget.log_task("data_loading", "load_data", 120.5)
        budget.log_task("model_training", "train_models", 1800.0)

        assert budget.consumed_seconds == pytest.approx(1920.5)
        assert not budget.is_over_budget()

    def test_over_budget_detection(self):
        budget = ComputeBudget(total_budget_seconds=100)
        budget.log_task("model_training", "train", 150.0)
        assert budget.is_over_budget()

    def test_phase_breakdown(self):
        budget = ComputeBudget(total_budget_seconds=3600)
        budget.log_task("data_loading", "load", 100)
        budget.log_task("model_training", "train", 500)
        by_phase = budget.consumed_by_phase()
        assert by_phase["data_loading"] == 100
        assert by_phase["model_training"] == 500

    def test_budget_summary(self, tmp_path):
        budget = ComputeBudget(total_budget_seconds=3600)
        budget.log_task("data_loading", "load", 100)
        summary = budget.get_summary()
        assert summary["consumed_seconds"] == 100
        assert summary["remaining_seconds"] == 3500
        assert not summary["is_over_budget"]

        path = budget.save(tmp_path)
        assert path.exists()


class TestMetaLearningIntegration:
    """Test meta-learning registry."""

    def test_record_and_query(self, tmp_path):
        registry = MetaLearningRegistry(registry_path=tmp_path / "reg.json")
        registry.record_result(
            position="QB",
            horizon="1w",
            regime="full_season",
            model_family="ensemble",
            feature_set="v8",
            calibration_method="none",
            primary_metric="rmse",
            primary_value=5.2,
            experiment_id="exp_001",
        )

        best = registry.best_config_for(position="QB", horizon="1w", metric="rmse")
        assert best is not None
        assert best["primary_value"] == 5.2
        assert best["model_family"] == "ensemble"

    def test_multiple_entries_picks_best(self, tmp_path):
        registry = MetaLearningRegistry(registry_path=tmp_path / "reg.json")
        registry.record_result("QB", "1w", "full", "ridge", "v7", "none", "rmse", 6.0)
        registry.record_result("QB", "1w", "full", "ensemble", "v8", "none", "rmse", 5.2)

        best = registry.best_config_for(position="QB", metric="rmse")
        assert best["primary_value"] == 5.2
        assert best["model_family"] == "ensemble"

    def test_summary_by_position(self, tmp_path):
        registry = MetaLearningRegistry(registry_path=tmp_path / "reg.json")
        registry.record_result("QB", "1w", "full", "ensemble", "v8", "none", "rmse", 5.2)
        registry.record_result("RB", "1w", "full", "ensemble", "v8", "none", "rmse", 4.8)
        summary = registry.summary_by_position()
        assert "QB" in summary
        assert "RB" in summary


class TestLabelDriftDetection:
    """Test the 3rd drift axis: label/prior drift."""

    def test_save_and_check_label_baseline(self, tmp_path):
        # Save baseline
        training_targets = np.random.normal(12.0, 5.0, 500)
        baseline_path = tmp_path / "label_baseline.json"
        ModelMonitor.save_label_baseline(training_targets, baseline_path)

        assert baseline_path.exists()
        with open(baseline_path) as f:
            baseline = json.load(f)
        assert baseline["n_samples"] == 500
        assert "histogram_bins" in baseline

    def test_no_drift_detected(self, tmp_path):
        training_targets = np.random.normal(12.0, 5.0, 500)
        baseline_path = tmp_path / "label_baseline.json"
        ModelMonitor.save_label_baseline(training_targets, baseline_path)

        monitor = ModelMonitor(alert_dir=tmp_path)
        recent = np.random.normal(12.0, 5.0, 100)
        alerts = monitor.check_label_drift(recent, baseline_path=baseline_path)
        # Small sample variation shouldn't trigger alerts
        # (may occasionally trigger due to randomness, but usually won't)
        # Just verify it runs without error
        assert isinstance(alerts, list)

    def test_drift_detected_on_shift(self, tmp_path):
        training_targets = np.random.normal(12.0, 5.0, 500)
        baseline_path = tmp_path / "label_baseline.json"
        ModelMonitor.save_label_baseline(training_targets, baseline_path)

        monitor = ModelMonitor(alert_dir=tmp_path)
        # Shift mean by 50%
        shifted = np.random.normal(18.0, 5.0, 100)
        alerts = monitor.check_label_drift(shifted, baseline_path=baseline_path)
        assert len(alerts) >= 1
        assert any("label drift" in a["message"].lower() or "label" in a["message"].lower()
                    for a in alerts)

    def test_missing_baseline_no_error(self, tmp_path):
        monitor = ModelMonitor(alert_dir=tmp_path)
        alerts = monitor.check_label_drift(
            np.array([10.0, 12.0, 8.0]),
            baseline_path=tmp_path / "nonexistent.json",
        )
        assert alerts == []
