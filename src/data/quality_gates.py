"""Pre-modeling data quality gates for refresh and training workflows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR, POSITIONS
from src.utils.nfl_calendar import get_current_nfl_season, get_current_nfl_week

EXPECTED_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}


@dataclass
class DataQualityGateResult:
    passed: bool
    report: Dict[str, Any]


class DataQualityGates:
    """Evaluate freshness, completeness, and anomaly checks."""

    def __init__(self, expected_positions: Optional[list[str]] = None):
        self.expected_positions = expected_positions or list(POSITIONS)

    def evaluate(
        self,
        df: pd.DataFrame,
        expected_season: Optional[int] = None,
        expected_week: Optional[int] = None,
    ) -> DataQualityGateResult:
        report: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "checks": {},
        }

        if df.empty:
            report["checks"]["dataset"] = {
                "passed": False,
                "reason": "dataset is empty",
            }
            return DataQualityGateResult(passed=False, report=report)

        required_cols = {"season", "week", "team", "position", "player_id"}
        missing_cols = sorted(required_cols - set(df.columns))
        if missing_cols:
            report["checks"]["dataset"] = {
                "passed": False,
                "reason": "missing required columns",
                "missing_columns": missing_cols,
            }
            return DataQualityGateResult(passed=False, report=report)

        checks = {
            "freshness": self._check_freshness(df, expected_season, expected_week),
            "completeness": self._check_completeness(df),
            "anomalies": self._check_anomalies(df),
        }
        report["checks"] = checks

        passed = all(check.get("passed", False) for check in checks.values())
        report["status"] = "pass" if passed else "fail"
        report["latest_observed"] = {
            "season": int(df["season"].max()),
            "week": int(df.loc[df["season"] == df["season"].max(), "week"].max()),
            "rows": int(len(df)),
        }
        return DataQualityGateResult(passed=passed, report=report)

    def _check_freshness(
        self,
        df: pd.DataFrame,
        expected_season: Optional[int],
        expected_week: Optional[int],
    ) -> Dict[str, Any]:
        target_season = expected_season or get_current_nfl_season()
        target_week = expected_week
        if target_week is None:
            current_week = get_current_nfl_week().get("week_num", 0)
            target_week = int(current_week) if current_week else None

        latest_season = int(df["season"].max())
        season_slice = df[df["season"] == latest_season]
        latest_week = int(season_slice["week"].max()) if not season_slice.empty else 0

        season_ok = latest_season >= target_season
        week_ok = True if target_week is None else latest_week >= target_week

        return {
            "passed": bool(season_ok and week_ok),
            "expected": {"season": target_season, "week": target_week},
            "observed": {"season": latest_season, "week": latest_week},
        }

    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        latest_season = int(df["season"].max())
        latest_week = int(df.loc[df["season"] == latest_season, "week"].max())
        latest = df[(df["season"] == latest_season) & (df["week"] == latest_week)].copy()

        teams_present = set(latest["team"].dropna().astype(str).unique())
        missing_teams = sorted(EXPECTED_TEAMS - teams_present)

        positions_present = set(latest["position"].dropna().astype(str).unique())
        missing_positions = sorted(set(self.expected_positions) - positions_present)

        activity_metric = pd.Series(0, index=latest.index, dtype=float)
        for col in ["snap_count", "targets", "carries", "passing_attempts", "fantasy_points"]:
            if col in latest.columns:
                activity_metric = activity_metric + latest[col].fillna(0).astype(float)
        active_players = int(latest.loc[activity_metric > 0, "player_id"].nunique())

        return {
            "passed": len(missing_teams) == 0 and len(missing_positions) == 0 and active_players > 0,
            "latest_window": {"season": latest_season, "week": latest_week},
            "missing_teams": missing_teams,
            "missing_positions": missing_positions,
            "active_players": active_players,
        }

    def _check_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        weekly = (
            df.groupby(["season", "week"], as_index=False)
            .size()
            .rename(columns={"size": "row_count"})
            .sort_values(["season", "week"])
            .reset_index(drop=True)
        )

        weekly["baseline"] = (
            weekly["row_count"]
            .shift(1)
            .rolling(window=4, min_periods=2)
            .median()
        )
        weekly["ratio_vs_baseline"] = weekly["row_count"] / weekly["baseline"]

        latest = weekly.iloc[-1]
        baseline = latest.get("baseline")
        if pd.isna(baseline) or baseline <= 0:
            return {
                "passed": True,
                "reason": "insufficient history for anomaly baseline",
                "latest_row_count": int(latest["row_count"]),
            }

        ratio = float(latest["ratio_vs_baseline"])
        lower_bound, upper_bound = 0.6, 1.6
        passed = lower_bound <= ratio <= upper_bound
        return {
            "passed": passed,
            "latest_window": {
                "season": int(latest["season"]),
                "week": int(latest["week"]),
                "row_count": int(latest["row_count"]),
            },
            "baseline_row_count": float(baseline),
            "ratio_vs_baseline": round(ratio, 3),
            "bounds": {"lower": lower_bound, "upper": upper_bound},
        }


def load_player_weekly_stats(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load core columns needed by data quality gates."""
    db_file = db_path or (DATA_DIR / "nfl_data.db")
    with sqlite3.connect(str(db_file)) as conn:
        return pd.read_sql_query(
            """
            SELECT
                pws.player_id,
                pws.season,
                pws.week,
                pws.team,
                p.position,
                pws.snap_count,
                pws.targets,
                pws.carries,
                pws.passing_attempts,
                pws.fantasy_points
            FROM player_weekly_stats pws
            JOIN players p ON p.player_id = pws.player_id
            """,
            conn,
        )


def run_quality_gates(
    df: pd.DataFrame,
    *,
    expected_season: Optional[int] = None,
    expected_week: Optional[int] = None,
    report_path: Optional[Path] = None,
) -> DataQualityGateResult:
    """Run quality gates over a dataframe and optionally write JSON report."""
    runner = DataQualityGates()
    result = runner.evaluate(df, expected_season=expected_season, expected_week=expected_week)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result.report, indent=2))
    return result


def run_db_quality_gates(
    *,
    db_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    expected_season: Optional[int] = None,
    expected_week: Optional[int] = None,
) -> DataQualityGateResult:
    """Run quality gates against the project DB."""
    df = load_player_weekly_stats(db_path=db_path)
    final_report_path = report_path or (MODELS_DIR / "data_quality_gate_report.json")
    return run_quality_gates(
        df,
        expected_season=expected_season,
        expected_week=expected_week,
        report_path=final_report_path,
    )
