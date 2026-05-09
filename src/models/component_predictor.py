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

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ComponentPredictor:
    """Predict individual stat components per position, assemble fantasy points.

    For each position, trains a separate GradientBoosting model for each stat
    component (e.g., rushing_yards, rushing_tds for RB).  At prediction time,
    predicts each component and multiplies by PPR scoring weights to get
    total fantasy points.

    GBR captures non-linear feature interactions (e.g., high game_total ×
    high passing_attempts → scoring boom) that Ridge regression misses.
    """

    def __init__(self, position: str):
        self.position = position
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.feature_medians: Dict[str, float] = {}
        self.is_fitted = False

        from config.settings import COMPONENT_TARGETS, PPR_SCORING_WEIGHTS
        self.components = COMPONENT_TARGETS.get(position, [])
        self.scoring_weights = PPR_SCORING_WEIGHTS

    def _prepare_array(self, X_arr: np.ndarray) -> np.ndarray:
        """Clean feature array: replace NaN/inf with 0."""
        return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(
        self,
        X: pd.DataFrame,
        y_components: Dict[str, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ComponentPredictor":
        """Train one model per stat component.

        QB uses Ridge with symmetric augmentation (mixup + noise + mirror)
        to synthetically expand training data and improve generalization.
        Other positions use GBR.
        """
        self.feature_names = list(X.columns)
        self.feature_medians = {c: float(X[c].median()) for c in X.columns
                                if pd.api.types.is_numeric_dtype(X[c])}

        X_arr = X.values.astype(np.float64)
        X_arr = self._prepare_array(X_arr)

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

            if self.position == "QB":
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=50, max_depth=2, learning_rate=0.1,
                    subsample=0.8, min_samples_leaf=50, random_state=42,
                )
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
        X_arr = self._prepare_array(X_arr)

        total_fp = np.zeros(len(X))

        for component, model in self.models.items():
            scaler = self.scalers.get(component)
            X_input = scaler.transform(X_arr) if scaler is not None else X_arr
            pred = model.predict(X_input)

            # Clip: stat components can't be negative (except interceptions, fumbles)
            if component not in ("interceptions", "fumbles_lost"):
                pred = np.maximum(pred, 0.0)

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
        X_arr = self._prepare_array(X_arr)

        result = {}
        for component, model in self.models.items():
            scaler = self.scalers.get(component)
            X_input = scaler.transform(X_arr) if scaler is not None else X_arr
            pred = model.predict(X_input)
            if component not in ("interceptions", "fumbles_lost"):
                pred = np.maximum(pred, 0.0)
            result[component] = pred
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for saving. Uses joblib for model-agnostic serialization."""
        models_b64 = {}
        for k, m in self.models.items():
            buf = io.BytesIO()
            joblib.dump(m, buf)
            models_b64[k] = base64.b64encode(buf.getvalue()).decode("ascii")

        scalers_b64 = {}
        for k, s in self.scalers.items():
            buf = io.BytesIO()
            joblib.dump(s, buf)
            scalers_b64[k] = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "position": self.position,
            "components": self.components,
            "feature_names": self.feature_names,
            "feature_medians": self.feature_medians,
            "model_type": "GradientBoostingRegressor",
            "serialization": "joblib_b64",
            "models": models_b64,
            "scalers": scalers_b64,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentPredictor":
        """Deserialize from saved dict."""
        cp = cls(data["position"])
        cp.feature_names = data.get("feature_names", [])
        cp.feature_medians = data.get("feature_medians", {})
        cp.components = data.get("components", cp.components)

        serialization = data.get("serialization", "ridge_json")

        if serialization == "joblib_b64":
            for comp_name, b64_str in data.get("models", {}).items():
                buf = io.BytesIO(base64.b64decode(b64_str))
                cp.models[comp_name] = joblib.load(buf)
            for comp_name, b64_str in data.get("scalers", {}).items():
                buf = io.BytesIO(base64.b64decode(b64_str))
                cp.scalers[comp_name] = joblib.load(buf)
        else:
            # Legacy Ridge JSON format
            from sklearn.linear_model import Ridge
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
