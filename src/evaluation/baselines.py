"""
Strong baselines for NFL fantasy-point prediction.

Production-readiness requires demonstrating that the ML model beats realistic
alternatives. This module provides baseline strategies:

1. **Naive historical**: Player's trailing N-game rolling average.
2. **Season average**: Player's expanding season-to-date mean (shift(1) to avoid
   leakage of the current week).
3. **Market-implied / ADP-based**: Position-rank historical expectation —
   maps a player's prior-season positional rank to expected fantasy points
   from historical data.
4. **Expert consensus**: Simulates expert projections by blending prior-season
   rank with recent in-season performance, weighted by position-specific
   factors.  This is the strongest non-ML baseline and is critical for
   determining whether the model provides real value.

Each baseline produces per-player, per-week point estimates that can be
compared directly to the ML model using the same metrics (RMSE, MAE,
Spearman, tier accuracy, etc.).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def trailing_average_baseline(
    df: pd.DataFrame,
    n_weeks: int = 3,
    target_col: str = "fantasy_points",
) -> pd.Series:
    """Trailing N-game rolling average (shifted to avoid current-week leakage).

    For week W, prediction = mean(W-N .. W-1).  First N rows per player
    receive the expanding mean of available history.
    """
    return (
        df.sort_values(["player_id", "season", "week"])
        .groupby("player_id")[target_col]
        .transform(lambda x: x.shift(1).rolling(n_weeks, min_periods=1).mean())
    )


def season_average_baseline(
    df: pd.DataFrame,
    target_col: str = "fantasy_points",
) -> pd.Series:
    """Expanding season-to-date average (shifted by 1 week).

    Captures a player's mean production so far this season, excluding
    the current game.  Falls back to overall expanding mean for the
    first game of a season.
    """
    return (
        df.sort_values(["player_id", "season", "week"])
        .groupby(["player_id", "season"])[target_col]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )


def positional_rank_baseline(
    df: pd.DataFrame,
    target_col: str = "fantasy_points",
) -> pd.Series:
    """ADP / positional-rank baseline.

    Uses each player's prior-season average fantasy points (by position rank)
    as the prediction for every week of the current season.  For players
    without a prior season, falls back to the position-wide median of
    prior-season averages.

    This is equivalent to a "draft your team based on last year's stats"
    strategy and represents the information available to any participant
    before the season starts.
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()

    # Prior season average per player
    season_avg = (
        df.groupby(["player_id", "season"])[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: "_prior_season_avg"})
    )
    season_avg["_prior_season"] = season_avg["season"] + 1
    season_avg = season_avg.rename(columns={"season": "_src_season"})

    df = df.merge(
        season_avg[["player_id", "_prior_season", "_prior_season_avg"]],
        left_on=["player_id", "season"],
        right_on=["player_id", "_prior_season"],
        how="left",
    )

    # Fallback: position-wide median of available prior-season averages
    pos_median = (
        df.dropna(subset=["_prior_season_avg"])
        .groupby(["season", "position"])["_prior_season_avg"]
        .transform("median")
    )
    # Merge fallback aligned by index
    fallback = df.groupby(["season", "position"])[target_col].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    result = df["_prior_season_avg"].fillna(fallback)

    # Clean up merge columns
    df.drop(columns=["_prior_season", "_prior_season_avg", "_src_season"],
            errors="ignore", inplace=True)

    return result


# ---------------------------------------------------------------------------
# Expert consensus baseline
# ---------------------------------------------------------------------------

# Position-specific blend weights: how much weight an "expert" gives to
# prior-season rank vs recent in-season form.  QBs are more stable year-
# over-year so prior rank gets more weight; RBs are more volatile.
_EXPERT_POSITION_WEIGHTS = {
    "QB": {"prior_season": 0.40, "trailing_avg": 0.35, "season_avg": 0.25},
    "RB": {"prior_season": 0.25, "trailing_avg": 0.45, "season_avg": 0.30},
    "WR": {"prior_season": 0.30, "trailing_avg": 0.40, "season_avg": 0.30},
    "TE": {"prior_season": 0.35, "trailing_avg": 0.35, "season_avg": 0.30},
}
_EXPERT_DEFAULT_WEIGHTS = {"prior_season": 0.30, "trailing_avg": 0.40, "season_avg": 0.30}


def expert_consensus_baseline(
    df: pd.DataFrame,
    target_col: str = "fantasy_points",
    trailing_window: int = 4,
    position_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.Series:
    """Simulated expert consensus projection baseline.

    Real expert projections (e.g. FantasyPros ECR) are not freely available
    at scale for historical seasons.  This baseline approximates them by
    blending three information sources that any expert would use:

    1. **Prior-season rank**: Pre-season expectation (ADP-like).
    2. **Trailing N-game average**: Recent performance trend.
    3. **Season-to-date average**: Overall season form.

    The blend is position-specific: QBs are more predictable from prior
    season, while RBs depend more on recent usage.

    For early-season weeks where trailing/season averages are unavailable,
    the baseline falls back gracefully to prior-season expectations only.

    This is designed as the *strongest* realistic non-ML baseline. If the ML
    model cannot beat this, it is not providing meaningful value over what a
    human expert would produce.

    Args:
        df: DataFrame with player weekly data (must include player_id,
            season, week, position, and target_col).
        target_col: Column with actual fantasy points.
        trailing_window: Window for the trailing-average component (default 4).
        position_weights: Optional override for position-specific blend weights.
            Keys are positions, values are dicts with keys
            "prior_season", "trailing_avg", "season_avg" summing to 1.0.

    Returns:
        Series of expert consensus predictions aligned to df index.
    """
    weights = position_weights or _EXPERT_POSITION_WEIGHTS

    # Compute the three component predictions
    prior = positional_rank_baseline(df, target_col=target_col)
    trailing = trailing_average_baseline(df, n_weeks=trailing_window, target_col=target_col)
    season = season_average_baseline(df, target_col=target_col)

    # Build per-row weights based on position
    positions = df["position"] if "position" in df.columns else pd.Series("UNK", index=df.index)
    w_prior = positions.map(lambda p: weights.get(p, _EXPERT_DEFAULT_WEIGHTS)["prior_season"])
    w_trail = positions.map(lambda p: weights.get(p, _EXPERT_DEFAULT_WEIGHTS)["trailing_avg"])
    w_season = positions.map(lambda p: weights.get(p, _EXPERT_DEFAULT_WEIGHTS)["season_avg"])

    # Where in-season components are missing (early weeks), redistribute
    # their weight to prior-season expectations
    trail_valid = np.isfinite(trailing.values)
    season_valid = np.isfinite(season.values)
    prior_valid = np.isfinite(prior.values)

    # Start with the ideal weighted blend
    result = np.full(len(df), np.nan)

    for idx in range(len(df)):
        components = []
        total_w = 0.0

        if prior_valid[idx]:
            components.append(("prior", float(w_prior.iloc[idx]), float(prior.iloc[idx])))
            total_w += float(w_prior.iloc[idx])
        if trail_valid[idx]:
            components.append(("trail", float(w_trail.iloc[idx]), float(trailing.iloc[idx])))
            total_w += float(w_trail.iloc[idx])
        if season_valid[idx]:
            components.append(("season", float(w_season.iloc[idx]), float(season.iloc[idx])))
            total_w += float(w_season.iloc[idx])

        if total_w > 0:
            result[idx] = sum(w * v / total_w for _, w, v in components)

    return pd.Series(result, index=df.index)


def expert_consensus_baseline_vectorized(
    df: pd.DataFrame,
    target_col: str = "fantasy_points",
    trailing_window: int = 4,
    position_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.Series:
    """Vectorized version of expert_consensus_baseline for large datasets.

    Identical logic but avoids per-row Python loop for performance.
    """
    weights = position_weights or _EXPERT_POSITION_WEIGHTS

    prior = positional_rank_baseline(df, target_col=target_col).values.astype(float)
    trailing = trailing_average_baseline(df, n_weeks=trailing_window, target_col=target_col).values.astype(float)
    season = season_average_baseline(df, target_col=target_col).values.astype(float)

    positions = df["position"] if "position" in df.columns else pd.Series("UNK", index=df.index)
    w_prior = positions.map(lambda p: weights.get(p, _EXPERT_DEFAULT_WEIGHTS)["prior_season"]).values.astype(float)
    w_trail = positions.map(lambda p: weights.get(p, _EXPERT_DEFAULT_WEIGHTS)["trailing_avg"]).values.astype(float)
    w_season = positions.map(lambda p: weights.get(p, _EXPERT_DEFAULT_WEIGHTS)["season_avg"]).values.astype(float)

    # Zero out weights where component is NaN
    prior_valid = np.isfinite(prior)
    trail_valid = np.isfinite(trailing)
    season_valid = np.isfinite(season)

    ew_prior = np.where(prior_valid, w_prior, 0.0)
    ew_trail = np.where(trail_valid, w_trail, 0.0)
    ew_season = np.where(season_valid, w_season, 0.0)

    total_w = ew_prior + ew_trail + ew_season

    # Replace NaN with 0 for safe multiplication
    prior_safe = np.where(prior_valid, prior, 0.0)
    trail_safe = np.where(trail_valid, trailing, 0.0)
    season_safe = np.where(season_valid, season, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            total_w > 0,
            (ew_prior * prior_safe + ew_trail * trail_safe + ew_season * season_safe) / total_w,
            np.nan,
        )

    return pd.Series(result, index=df.index)


# ---------------------------------------------------------------------------
# Comparison framework
# ---------------------------------------------------------------------------

def compare_model_to_baselines(
    df: pd.DataFrame,
    model_predictions: pd.Series,
    target_col: str = "fantasy_points",
    trailing_windows: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare ML model predictions against multiple baselines.

    Args:
        df: DataFrame with player weekly data (must include player_id,
            season, week, position, and target_col).
        model_predictions: Series of ML model predictions aligned to df.
        target_col: Column with actual fantasy points.
        trailing_windows: Windows for trailing-average baselines (default [3, 5]).

    Returns:
        Dict keyed by baseline name, each containing RMSE, MAE, and
        improvement percentages vs the ML model.
    """
    if trailing_windows is None:
        trailing_windows = [3, 5]

    actuals = df[target_col].values
    valid = np.isfinite(actuals) & np.isfinite(model_predictions.values)

    baselines: Dict[str, np.ndarray] = {}

    # 1. Trailing average baselines
    for w in trailing_windows:
        pred = trailing_average_baseline(df, n_weeks=w, target_col=target_col)
        baselines[f"trailing_{w}g_avg"] = pred.values

    # 2. Season average baseline
    baselines["season_avg"] = season_average_baseline(df, target_col=target_col).values

    # 3. Positional rank / ADP baseline
    baselines["prior_season_rank"] = positional_rank_baseline(df, target_col=target_col).values

    # 4. Expert consensus baseline (strongest non-ML baseline)
    baselines["expert_consensus"] = expert_consensus_baseline_vectorized(
        df, target_col=target_col
    ).values

    # Evaluate each
    results: Dict[str, Dict[str, float]] = {}
    model_arr = model_predictions.values

    for name, baseline_preds in baselines.items():
        mask = valid & np.isfinite(baseline_preds)
        if mask.sum() < 20:
            logger.warning("Baseline %s has < 20 valid predictions, skipping.", name)
            continue

        y = actuals[mask]
        b = baseline_preds[mask]
        m = model_arr[mask]

        b_rmse = float(np.sqrt(mean_squared_error(y, b)))
        b_mae = float(mean_absolute_error(y, b))
        m_rmse = float(np.sqrt(mean_squared_error(y, m)))
        m_mae = float(mean_absolute_error(y, m))

        results[name] = {
            "baseline_rmse": round(b_rmse, 4),
            "baseline_mae": round(b_mae, 4),
            "model_rmse": round(m_rmse, 4),
            "model_mae": round(m_mae, 4),
            "rmse_improvement_pct": round((1 - m_rmse / b_rmse) * 100, 2) if b_rmse > 0 else 0.0,
            "mae_improvement_pct": round((1 - m_mae / b_mae) * 100, 2) if b_mae > 0 else 0.0,
            "model_beats_baseline": m_rmse < b_rmse,
            "n_compared": int(mask.sum()),
        }

    return results


def compare_by_position(
    df: pd.DataFrame,
    model_predictions: pd.Series,
    target_col: str = "fantasy_points",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run baseline comparison stratified by position.

    Returns:
        Nested dict: {position: {baseline_name: metrics}}.
    """
    positions = df["position"].unique() if "position" in df.columns else []
    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for pos in sorted(positions):
        mask = df["position"] == pos
        if mask.sum() < 30:
            continue
        out[pos] = compare_model_to_baselines(
            df[mask].copy(),
            model_predictions[mask],
            target_col=target_col,
        )

    return out


def format_baseline_report(comparison: Dict[str, Dict[str, float]]) -> str:
    """Format baseline comparison as a readable text report."""
    lines = [
        "=" * 70,
        "BASELINE COMPARISON REPORT",
        "=" * 70,
        "",
    ]

    any_beaten = False
    beats_expert = False
    for name, metrics in comparison.items():
        beaten = metrics["model_beats_baseline"]
        any_beaten = any_beaten or beaten
        if name == "expert_consensus" and beaten:
            beats_expert = True
        marker = "BEATS" if beaten else "LOSES TO"
        tag = " [CRITICAL]" if name == "expert_consensus" else ""
        lines.append(f"  {name}{tag}:")
        lines.append(f"    Baseline RMSE: {metrics['baseline_rmse']:.3f}  |  Model RMSE: {metrics['model_rmse']:.3f}")
        lines.append(f"    Improvement:   {metrics['rmse_improvement_pct']:+.1f}% RMSE  |  {metrics['mae_improvement_pct']:+.1f}% MAE")
        lines.append(f"    Verdict:       Model {marker} this baseline  (n={metrics['n_compared']})")
        lines.append("")

    lines.append("-" * 70)
    if beats_expert:
        lines.append("MODEL BEATS EXPERT CONSENSUS — provides real value over human projections.")
    elif "expert_consensus" in comparison:
        lines.append("WARNING: Model does NOT beat expert consensus baseline.")
        lines.append("The model may not provide value over human expert projections.")
    if any_beaten:
        lines.append("MODEL DEMONSTRATES EDGE over at least one strong baseline.")
    else:
        lines.append("WARNING: Model does NOT beat any strong baseline.")
        lines.append("Consider narrowing the use case or improving features.")
    lines.append("-" * 70)

    return "\n".join(lines)
