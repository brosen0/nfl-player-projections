"""
Meta-Learning Registry for NFL Prediction Models.

Per Agent Directive V7 Section 7: add a meta-learning layer that learns which
feature families, models, calibration methods, and ensemble strategies perform
best by problem type, sample regime, and horizon.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = (
    Path(__file__).parent.parent.parent / "data" / "meta_learning" / "registry.json"
)


class MetaLearningRegistry:
    """Tracks which configurations win per (position, horizon, regime).

    Provides lookup for the best-performing configuration given a new
    task context, enabling transfer of knowledge across experiments.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or _DEFAULT_REGISTRY_PATH
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = data.get("entries", [])
            except (json.JSONDecodeError, IOError):
                self.entries = []

    def save(self) -> None:
        data = {
            "updated_at": datetime.now().isoformat(),
            "n_entries": len(self.entries),
            "entries": self.entries,
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_result(
        self,
        position: str,
        horizon: str,
        regime: str,
        model_family: str,
        feature_set: str,
        calibration_method: str,
        primary_metric: str,
        primary_value: float,
        experiment_id: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an experiment outcome for meta-learning.

        Args:
            position: QB, RB, WR, TE.
            horizon: 1w, 4w, 18w.
            regime: full_season, early_season, late_season, small_sample.
            model_family: ridge, gbm, xgboost, lightgbm, ensemble.
            feature_set: Description of feature set used.
            calibration_method: none, isotonic, conformal, platt.
            primary_metric: Metric name (e.g., 'rmse').
            primary_value: Metric value.
            experiment_id: Link to experiment tracker.
            extra: Additional metadata.
        """
        entry = {
            "position": position,
            "horizon": horizon,
            "regime": regime,
            "model_family": model_family,
            "feature_set": feature_set,
            "calibration_method": calibration_method,
            "primary_metric": primary_metric,
            "primary_value": round(primary_value, 4),
            "experiment_id": experiment_id,
            "recorded_at": datetime.now().isoformat(),
        }
        if extra:
            entry["extra"] = extra
        self.entries.append(entry)
        self.save()

    def best_config_for(
        self,
        position: str = "",
        horizon: str = "",
        regime: str = "",
        metric: str = "rmse",
        lower_is_better: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Query the best configuration for a given context.

        Args:
            position: Filter by position (empty = all).
            horizon: Filter by horizon (empty = all).
            regime: Filter by regime (empty = all).
            metric: Metric to optimize.
            lower_is_better: If True, lower metric = better.

        Returns:
            Best matching entry or None.
        """
        candidates = self.entries
        if position:
            candidates = [e for e in candidates if e.get("position") == position]
        if horizon:
            candidates = [e for e in candidates if e.get("horizon") == horizon]
        if regime:
            candidates = [e for e in candidates if e.get("regime") == regime]
        candidates = [e for e in candidates if e.get("primary_metric") == metric]

        if not candidates:
            return None

        if lower_is_better:
            return min(candidates, key=lambda e: e.get("primary_value", float("inf")))
        else:
            return max(candidates, key=lambda e: e.get("primary_value", float("-inf")))

    def summary_by_position(self) -> Dict[str, Dict[str, Any]]:
        """Summarize best configs per position."""
        positions = set(e.get("position", "") for e in self.entries)
        summary = {}
        for pos in sorted(positions):
            best = self.best_config_for(position=pos, metric="rmse")
            if best:
                summary[pos] = {
                    "best_model": best.get("model_family"),
                    "best_rmse": best.get("primary_value"),
                    "feature_set": best.get("feature_set"),
                    "calibration": best.get("calibration_method"),
                }
        return summary
