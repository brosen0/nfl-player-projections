"""Season-total preseason projector.

Replaces the PPG × 17 heuristic in load_preseason_projections().

Trains one Ridge model per position on prior-season aggregated features
→ current-season total fantasy points. Strictly causal: features are
aggregated from season S-1 to predict season S total.

Typical use:
    projector = PreseasonProjector.train(seasons=range(2018, 2025))
    projector.save(path)
    # later:
    projector = PreseasonProjector.load(path)
    preds = projector.predict(prior_season_df, position="WR")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Features used per position (all derived from prior-season aggregated stats)
FEATURES_COMMON = [
    "ppg",           # prior season fantasy points per game (most predictive)
    "games_played",  # games played (availability signal)
    "snap_share",    # avg snap share pct from utilization_scores
]

FEATURES_BY_POSITION: Dict[str, List[str]] = {
    "QB": FEATURES_COMMON + [
        "passing_yards_pg", "passing_tds_pg", "interceptions_pg",
        "rushing_yards_pg", "completion_pct",
    ],
    "RB": FEATURES_COMMON + [
        "carries_pg", "targets_pg", "receptions_pg",
        "rushing_yards_pg", "receiving_yards_pg",
        "rush_share", "target_share",
    ],
    "WR": FEATURES_COMMON + [
        "targets_pg", "receptions_pg", "receiving_yards_pg",
        "air_yards_pg", "target_share",
    ],
    "TE": FEATURES_COMMON + [
        "targets_pg", "receptions_pg", "receiving_yards_pg",
        "target_share",
    ],
}

MIN_GAMES = 6        # minimum prior-season games to include a player-season pair
MIN_SAMPLES = 20     # minimum training samples per position to attempt fit


class PreseasonProjector:
    """Predict full-season fantasy points from prior-season aggregate signals."""

    def __init__(self):
        self.models: Dict[str, Ridge] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training data assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_season_pairs(db, seasons: List[int]) -> pd.DataFrame:
        """Build training pairs: (prior season features, current season target).

        For each consecutive pair (S, S+1) in seasons, aggregate player stats
        from season S as features and sum season S+1 as the target.
        """
        frames = []
        season_list = sorted(seasons)

        for i in range(len(season_list) - 1):
            prior = season_list[i]
            curr = season_list[i + 1]

            with db._get_connection() as conn:
                # Prior season: aggregate per-player stats
                prior_df = pd.read_sql_query(
                    """
                    SELECT
                        pws.player_id, p.position,
                        COUNT(*) AS games_played,
                        AVG(pws.fantasy_points) AS ppg,
                        AVG(COALESCE(pws.passing_yards, 0)) AS passing_yards_pg,
                        AVG(COALESCE(pws.passing_tds, 0)) AS passing_tds_pg,
                        AVG(COALESCE(pws.interceptions, 0)) AS interceptions_pg,
                        AVG(COALESCE(pws.rushing_yards, 0)) AS rushing_yards_pg,
                        AVG(COALESCE(pws.rushing_attempts, 0)) AS carries_pg,
                        AVG(COALESCE(pws.targets, 0)) AS targets_pg,
                        AVG(COALESCE(pws.receptions, 0)) AS receptions_pg,
                        AVG(COALESCE(pws.receiving_yards, 0)) AS receiving_yards_pg,
                        AVG(COALESCE(pws.air_yards, 0)) AS air_yards_pg,
                        AVG(CASE WHEN COALESCE(pws.passing_attempts, 0) > 0
                            THEN 100.0 * pws.passing_completions / pws.passing_attempts
                            ELSE 0 END) AS completion_pct,
                        AVG(COALESCE(us.snap_share, 0)) AS snap_share,
                        AVG(COALESCE(us.target_share, 0)) AS target_share,
                        AVG(COALESCE(us.rush_share, 0)) AS rush_share
                    FROM player_weekly_stats pws
                    JOIN players p ON pws.player_id = p.player_id
                    LEFT JOIN utilization_scores us
                        ON pws.player_id = us.player_id
                        AND pws.season = us.season
                        AND pws.week = us.week
                    WHERE pws.season = ?
                      AND p.position IN ('QB', 'RB', 'WR', 'TE')
                      AND pws.week <= 18
                    GROUP BY pws.player_id, p.position
                    HAVING COUNT(*) >= ?
                    """,
                    conn,
                    params=(prior, MIN_GAMES),
                )

                # Current season: total fantasy points (target)
                curr_df = pd.read_sql_query(
                    """
                    SELECT player_id, SUM(fantasy_points) AS season_total
                    FROM player_weekly_stats
                    WHERE season = ?
                    GROUP BY player_id
                    HAVING COUNT(*) >= 4
                    """,
                    conn,
                    params=(curr,),
                )

            if prior_df.empty or curr_df.empty:
                continue

            merged = prior_df.merge(curr_df, on="player_id")
            merged["prior_season"] = prior
            merged["curr_season"] = curr
            frames.append(merged)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, pairs_df: pd.DataFrame) -> "PreseasonProjector":
        """Train one Ridge model per position on the season-pair data."""
        for pos in ("QB", "RB", "WR", "TE"):
            pos_df = pairs_df[pairs_df["position"] == pos].dropna(subset=["season_total"])
            features = FEATURES_BY_POSITION[pos]
            available = [f for f in features if f in pos_df.columns]
            if len(pos_df) < MIN_SAMPLES or not available:
                logger.warning("PreseasonProjector[%s]: only %d samples, skipping", pos, len(pos_df))
                continue

            X = pos_df[available].fillna(0).values.astype(np.float64)
            y = pos_df["season_total"].values.astype(np.float64)
            valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[valid], y[valid]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = Ridge(alpha=10.0)
            model.fit(X_scaled, y)

            self.models[pos] = model
            self.scalers[pos] = scaler
            self.feature_names[pos] = available
            logger.info("PreseasonProjector[%s]: fitted on %d samples, features=%s", pos, len(y), available)

        self.is_fitted = len(self.models) > 0
        return self

    def predict(self, prior_season_df: pd.DataFrame, position: str) -> np.ndarray:
        """Predict season-total FP from prior-season aggregate features.

        Args:
            prior_season_df: DataFrame with one row per player, columns matching
                FEATURES_BY_POSITION[position]. Missing columns filled with 0.
            position: "QB", "RB", "WR", or "TE".

        Returns:
            Array of predicted season totals (clipped >= 0).
        """
        if position not in self.models:
            raise ValueError(f"PreseasonProjector not fitted for {position}")

        features = self.feature_names[position]
        X = prior_season_df.reindex(columns=features, fill_value=0).fillna(0).values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scalers[position].transform(X)
        preds = self.models[position].predict(X_scaled)
        return np.maximum(preds, 0.0)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize to JSON (Ridge coefficients only — no joblib dependency)."""
        data: dict = {"positions": {}}
        for pos, model in self.models.items():
            scaler = self.scalers[pos]
            data["positions"][pos] = {
                "features": self.feature_names[pos],
                "coef": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info("PreseasonProjector saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "PreseasonProjector":
        """Deserialize from JSON."""
        data = json.loads(Path(path).read_text())
        proj = cls()
        for pos, d in data.get("positions", {}).items():
            model = Ridge()
            model.coef_ = np.array(d["coef"])
            model.intercept_ = float(d["intercept"])
            model.n_features_in_ = len(model.coef_)
            scaler = StandardScaler()
            scaler.mean_ = np.array(d["scaler_mean"])
            scaler.scale_ = np.array(d["scaler_scale"])
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
            proj.models[pos] = model
            proj.scalers[pos] = scaler
            proj.feature_names[pos] = d["features"]
        proj.is_fitted = len(proj.models) > 0
        return proj

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    @classmethod
    def train(cls, seasons: List[int], db=None) -> Tuple["PreseasonProjector", pd.DataFrame]:
        """Build training data and fit. Returns (projector, pairs_df) for inspection."""
        from src.utils.database import DatabaseManager
        db = db or DatabaseManager()
        pairs_df = cls._build_season_pairs(db, seasons)
        if pairs_df.empty:
            raise ValueError("No season pairs found — check database has 2+ seasons of data")
        proj = cls()
        proj.fit(pairs_df)
        return proj, pairs_df
