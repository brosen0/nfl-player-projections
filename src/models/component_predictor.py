"""Component-based fantasy point prediction.

Council Phase 2 recommendation: predict stable stat components (yards,
receptions, TDs) separately, then assemble fantasy points from those
predictions.  Each component has higher autocorrelation and lower touchdown
contamination than raw fantasy points.

The key insight: a WR catching 6 passes for 70 yards scores ~10 PPR points.
Add one TD and it's 16.  That TD is nearly coin-flip predictable at the
individual game level.  By predicting yards and receptions (stable) separately
from TDs (volatile), we get a better overall prediction even though each
individual component model is simple.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ComponentPredictor:
    """Predict individual stat components per position, assemble fantasy points.

    For each position, trains a separate Ridge regression model for each stat
    component (e.g., rushing_yards, rushing_tds for RB).  At prediction time,
    predicts each component and multiplies by PPR scoring weights to get
    total fantasy points.

    This avoids the problem of directly predicting a noisy composite (fantasy
    points) where a single high-variance component (TDs) dominates.
    """

    def __init__(self, position: str):
        self.position = position
        self.models: Dict[str, Ridge] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.feature_medians: Dict[str, float] = {}
        self.is_fitted = False

        from config.settings import COMPONENT_TARGETS, PPR_SCORING_WEIGHTS
        self.components = COMPONENT_TARGETS.get(position, [])
        self.scoring_weights = PPR_SCORING_WEIGHTS

    def fit(
        self,
        X: pd.DataFrame,
        y_components: Dict[str, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ComponentPredictor":
        """Train one Ridge model per stat component.

        Args:
            X: Feature DataFrame (same features for all components).
            y_components: Dict mapping component name -> target Series.
            sample_weight: Optional per-row weights (recency decay).
        """
        self.feature_names = list(X.columns)
        self.feature_medians = {c: float(X[c].median()) for c in X.columns
                                if pd.api.types.is_numeric_dtype(X[c])}

        X_arr = X.values.astype(np.float64)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        for component in self.components:
            if component not in y_components:
                logger.warning("Component %s not in y_components for %s, skipping",
                               component, self.position)
                continue

            y = y_components[component].values.astype(np.float64)
            valid = np.isfinite(y) & np.isfinite(X_arr).all(axis=1)
            if valid.sum() < 30:
                logger.warning("Component %s has < 30 valid rows for %s, skipping",
                               component, self.position)
                continue

            X_v = X_arr[valid]
            y_v = y[valid]
            sw = sample_weight[valid] if sample_weight is not None else None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_v)

            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y_v, sample_weight=sw)

            self.models[component] = model
            self.scalers[component] = scaler

        self.is_fitted = len(self.models) > 0
        if self.is_fitted:
            logger.info("ComponentPredictor[%s]: trained %d/%d components",
                        self.position, len(self.models), len(self.components))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fantasy points by predicting each component and assembling.

        Returns:
            Array of predicted fantasy points (same length as X).
        """
        if not self.is_fitted:
            raise ValueError(f"ComponentPredictor[{self.position}] not fitted")

        # Align features
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        for col in self.feature_names:
            if col not in X_aligned.columns:
                X_aligned[col] = self.feature_medians.get(col, 0)
        X_arr = X_aligned.values.astype(np.float64)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        total_fp = np.zeros(len(X))
        component_preds: Dict[str, np.ndarray] = {}

        for component, model in self.models.items():
            scaler = self.scalers[component]
            X_scaled = scaler.transform(X_arr)
            pred = model.predict(X_scaled)

            # Clip: stat components can't be negative (except interceptions, fumbles)
            if component not in ("interceptions", "fumbles_lost"):
                pred = np.maximum(pred, 0.0)

            component_preds[component] = pred
            weight = self.scoring_weights.get(component, 0.0)
            total_fp += pred * weight

        # Ensure non-negative total
        total_fp = np.maximum(total_fp, 0.0)

        return total_fp

    def predict_components(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict individual stat components (for inspection/debugging)."""
        if not self.is_fitted:
            raise ValueError(f"ComponentPredictor[{self.position}] not fitted")

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_arr = X_aligned.values.astype(np.float64)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        result = {}
        for component, model in self.models.items():
            scaler = self.scalers[component]
            pred = model.predict(scaler.transform(X_arr))
            if component not in ("interceptions", "fumbles_lost"):
                pred = np.maximum(pred, 0.0)
            result[component] = pred
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for saving."""
        return {
            "position": self.position,
            "components": self.components,
            "feature_names": self.feature_names,
            "feature_medians": self.feature_medians,
            "models": {k: {"coef": m.coef_.tolist(), "intercept": float(m.intercept_)}
                       for k, m in self.models.items()},
            "scalers": {k: {"mean": s.mean_.tolist(), "scale": s.scale_.tolist()}
                        for k, s in self.scalers.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentPredictor":
        """Deserialize from saved dict."""
        cp = cls(data["position"])
        cp.feature_names = data.get("feature_names", [])
        cp.feature_medians = data.get("feature_medians", {})
        cp.components = data.get("components", cp.components)

        for comp_name, model_data in data.get("models", {}).items():
            model = Ridge(alpha=1.0)
            model.coef_ = np.array(model_data["coef"])
            model.intercept_ = model_data["intercept"]
            model.n_features_in_ = len(model.coef_)
            cp.models[comp_name] = model

        for comp_name, scaler_data in data.get("scalers", {}).items():
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_data["mean"])
            scaler.scale_ = np.array(scaler_data["scale"])
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
            cp.scalers[comp_name] = scaler

        cp.is_fitted = len(cp.models) > 0
        return cp
