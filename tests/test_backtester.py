"""Tests for confidence-interval behavior in backtester."""

from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.backtester import ModelBacktester, check_success_criteria


def test_confidence_interval_custom_column_names():
    backtester = ModelBacktester()
    df = pd.DataFrame({
        "position": ["RB", "RB", "WR", "WR"],
        "predicted_points": [10.0, 14.0, 9.0, 13.0],
        "fantasy_points": [12.0, 13.0, 8.5, 15.0],
    })
    out = backtester.calculate_confidence_intervals(
        df,
        pred_col="predicted_points",
        actual_col="fantasy_points",
        confidence=0.8,
        lower_col="prediction_ci80_lower",
        upper_col="prediction_ci80_upper",
    )
    assert "prediction_ci80_lower" in out.columns
    assert "prediction_ci80_upper" in out.columns
    assert (out["prediction_ci80_lower"] >= 0).all()


def test_success_criteria_prefers_explicit_ci_coverage():
    payload = {
        "metrics": {
            "spearman_rho": 0.7,
            "mape": 20.0,
            "within_7_pts_pct": 75.0,
            "within_10_pts_pct": 82.0,
            "tier_classification_accuracy": 0.8,
            "std_predicted": 7.0,
            "std_actual": 8.0,
            "mae_rmse_ratio": 0.77,
            "mae_rmse_healthy": True,
        },
        "by_week": {"1": {"rmse": 7.0}, "2": {"rmse": 7.4}},
        "by_position": {},
        "multiple_baseline_comparison": {
            "model_beats_all_internal_by_20_pct": True,
            "model_beats_all_internal_by_25_pct": True,
            "baseline_season_avg": {"improvement_pct": 26.0},
            "status": {"model_has_real_edge": False},
        },
        "confidence_band_coverage_10pt": 90.0,
    }
    sc = check_success_criteria(payload)
    assert sc["confidence_band_coverage_10pt"] == 90.0
    assert sc["confidence_band_target_882"] is True


def test_compare_to_multiple_baselines_returns_expected_keys():
    backtester = ModelBacktester()
    df = pd.DataFrame({
        "player_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"],
        "position": ["RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "TE"],
        "season": [2024] * 10,
        "week": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "fantasy_points": [10, 13, 9, 8, 14, 16, 6, 7, 11, 12],
        "predicted_points": [10.5, 12.5, 9.5, 8.5, 13.5, 15.0, 6.5, 7.0, 10.5, 11.5],
    })
    out = backtester.compare_to_multiple_baselines(df)
    assert "model" in out
    assert "baseline_persistence" in out
    assert "baseline_season_avg" in out
    assert "baseline_position_avg" in out
    assert "model_beats_all_internal_by_20_pct" in out or "model_beats_all_by_20_pct" in out
    assert "status" in out


def test_backtest_lineup_decisions_basic():
    """Model that predicts perfectly should win every week."""
    backtester = ModelBacktester()
    rows = []
    for week in range(1, 6):
        # 3 QBs, 4 RBs, 4 WRs, 3 TEs per week
        for i, (pos, n) in enumerate([("QB", 3), ("RB", 4), ("WR", 4), ("TE", 3)]):
            for j in range(n):
                pts = float(10 + j * 3 + week * 0.5)
                rows.append({
                    "player_id": f"{pos}{j}",
                    "position": pos,
                    "week": week,
                    "fantasy_points": pts,
                    "predicted_points": pts,  # perfect predictions
                })
    df = pd.DataFrame(rows)
    result = backtester.backtest_lineup_decisions(df)
    assert "error" not in result
    assert result["win_rate"] == 1.0
    assert result["wins"] == 5
    assert result["losses"] == 0
    assert result["n_weeks"] == 5
    assert result["avg_margin"] > 0


def test_backtest_lineup_decisions_returns_weekly_detail():
    backtester = ModelBacktester()
    rows = []
    np.random.seed(42)
    for week in range(1, 4):
        for pos, n in [("QB", 3), ("RB", 4), ("WR", 4), ("TE", 3)]:
            for j in range(n):
                pts = float(10 + j * 2 + np.random.normal(0, 2))
                rows.append({
                    "player_id": f"{pos}{j}",
                    "position": pos,
                    "week": week,
                    "fantasy_points": pts,
                    "predicted_points": pts + np.random.normal(0, 1),
                })
    df = pd.DataFrame(rows)
    result = backtester.backtest_lineup_decisions(df)
    assert "error" not in result
    assert len(result["weekly_results"]) == 3
    for wr in result["weekly_results"]:
        assert "week" in wr
        assert "model_actual" in wr
        assert "opponent_actual" in wr
        assert "margin" in wr
        assert "won" in wr
    assert result["wins"] + result["losses"] == result["n_weeks"]


def test_backtest_lineup_decisions_missing_column():
    backtester = ModelBacktester()
    df = pd.DataFrame({"x": [1, 2]})
    result = backtester.backtest_lineup_decisions(df)
    assert "error" in result


def test_backtest_lineup_decisions_custom_roster_slots():
    backtester = ModelBacktester()
    rows = []
    for week in range(1, 4):
        for pos, n in [("QB", 3), ("RB", 5), ("WR", 5), ("TE", 3)]:
            for j in range(n):
                pts = float(10 + j * 3)
                rows.append({
                    "player_id": f"{pos}{j}",
                    "position": pos,
                    "week": week,
                    "fantasy_points": pts,
                    "predicted_points": pts,
                })
    df = pd.DataFrame(rows)
    result = backtester.backtest_lineup_decisions(
        df, roster_slots={"QB": 1, "RB": 3, "WR": 3, "TE": 1}
    )
    assert "error" not in result
    assert result["roster_slots"] == {"QB": 1, "RB": 3, "WR": 3, "TE": 1}


def test_success_criteria_includes_lineup_win_rate():
    """check_success_criteria should pick up lineup decision results."""
    payload = {
        "metrics": {
            "spearman_rho": 0.7,
            "mape": 20.0,
            "within_7_pts_pct": 75.0,
            "within_10_pts_pct": 82.0,
            "tier_classification_accuracy": 0.8,
            "std_predicted": 7.0,
            "std_actual": 8.0,
            "mae_rmse_ratio": 0.77,
            "mae_rmse_healthy": True,
        },
        "by_week": {"1": {"rmse": 7.0}, "2": {"rmse": 7.4}},
        "by_position": {},
        "multiple_baseline_comparison": {
            "model_beats_all_internal_by_20_pct": True,
            "model_beats_all_internal_by_25_pct": True,
            "baseline_season_avg": {"improvement_pct": 26.0},
            "status": {"model_has_real_edge": True},
        },
        "lineup_decisions": {
            "win_rate": 0.65,
            "wins": 13,
            "losses": 7,
            "n_weeks": 20,
            "avg_margin": 4.5,
        },
    }
    sc = check_success_criteria(payload)
    assert sc["lineup_win_rate"] == 0.65
    assert sc["lineup_win_rate_gt_55"] is True
    assert sc["lineup_wins"] == 13
    assert sc["lineup_losses"] == 7


def test_compare_to_expert_consensus_with_csv(tmp_path):
    backtester = ModelBacktester()
    preds = pd.DataFrame({
        "name": [f"Player{i}" for i in range(12)],
        "fantasy_points": np.linspace(8, 20, 12),
        "predicted_points": np.linspace(8.5, 19.5, 12),
    })
    expert_path = tmp_path / "expert.csv"
    expert = pd.DataFrame({
        "player_name": [f"Player{i}" for i in range(12)],
        "proj_points": np.linspace(7.0, 18.0, 12),
    })
    expert.to_csv(expert_path, index=False)
    out = backtester.compare_to_expert_consensus(preds, str(expert_path))
    assert "model_rmse" in out
    assert "expert_rmse" in out
    assert "model_vs_expert_pct" in out
