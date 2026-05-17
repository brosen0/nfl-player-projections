from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "generate_dashboard_html",
    PROJECT_ROOT / "scripts" / "generate_dashboard_html.py",
)
dashboard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dashboard
spec.loader.exec_module(dashboard)


def test_build_board_data_uses_season_total_projection_for_ui(monkeypatch):
    monkeypatch.setattr(dashboard, "load_adp_board", lambda season: pd.DataFrame({"name": ["Player A"]}))
    monkeypatch.setattr(dashboard, "_latest_predictions_csv", lambda season: Path("fake.csv"))
    monkeypatch.setattr(
        dashboard,
        "load_model_projections",
        lambda csv, ranking, season: pd.DataFrame({
            "name": ["Player A"],
            "pred_total": [240.0],
        }),
    )
    monkeypatch.setattr(dashboard, "build_draft_board", lambda adp_df, projections: ["board"])
    monkeypatch.setattr(dashboard, "validate_spread_direction", lambda results, min_spread=10: {"n": 0, "accuracy": 0.0})
    monkeypatch.setattr(dashboard, "_apply_vorp", lambda projections, basis_col="pred_total": pd.Series([12.3]))
    monkeypatch.setattr(dashboard, "build_usage_roles", lambda season: {})
    monkeypatch.setattr(dashboard, "build_team_tendencies", lambda season: {"KC": {"tendency": "Pass-lean", "rush_pct": 40, "pass_pct": 60}})
    monkeypatch.setattr(dashboard, "compute_team_changes", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_usage_trends", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_regression_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_injury_risk", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_breakout_candidates", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_defense_rankings", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_sos_adjustment", lambda team, position, def_rankings: (1.0, ""))
    monkeypatch.setattr(
        dashboard,
        "compute_age_adjustments",
        lambda season: {"Player A": {"age": 31.0, "mult": 0.8}},
    )
    monkeypatch.setattr(
        dashboard,
        "compute_spread",
        lambda board: [
            SimpleNamespace(
                player_id="p1",
                name="Player A",
                position="WR",
                team="KC",
                ecr=15.0,
                model_rank=8,
                rank_spread=7,
                model_projection=240.0,
                blended_projection=90.0,
                actual_total=0.0,
                model_wins=False,
            )
        ],
    )

    data = dashboard.build_board_data(2026)

    assert len(data["players"]) == 1
    player = data["players"][0]
    assert player["rawProj"] == 240.0
    assert player["proj"] == 192.0
    assert player["blendProj"] == 90.0
    assert player["vorp"] == 12.3
    assert player["why"][0] == "Model #8 vs ADP #15"
    assert "Age 31" in player["why"]


def test_build_board_data_dedupes_fallback_rows_when_ml_artifact_already_has_player(monkeypatch):
    captured = {}

    def _capture_board(adp_df, projections):
        captured["projections"] = projections.copy()
        return ["board"]

    monkeypatch.setattr(dashboard, "load_adp_board", lambda season: pd.DataFrame({"name": ["Trey McBride"]}))
    monkeypatch.setattr(dashboard, "_latest_predictions_csv", lambda season: None if season == 2026 else Path("fake-2025.csv"))
    monkeypatch.setattr(
        dashboard,
        "load_model_projections",
        lambda csv, ranking, season: pd.DataFrame([
            {
                "name": "T.McBride",
                "position": "WR",
                "team": "ARI",
                "pred_total": 178.4,
                "actual_total": 0.0,
                "weeks": 17,
                "model_rank_value": 178.4,
            }
        ]),
    )
    monkeypatch.setattr(
        dashboard,
        "load_preseason_projections",
        lambda season, adp_df=None: pd.DataFrame([
            {
                "player_id": "rookie_Trey McBride_TE",
                "name": "Trey McBride",
                "position": "TE",
                "team": "ARI",
                "pred_total": 90.0,
                "actual_total": 0.0,
                "weeks": 0,
                "model_rank_value": 90.0,
            }
        ]),
    )
    monkeypatch.setattr(
        dashboard,
        "build_draft_board",
        _capture_board,
    )
    monkeypatch.setattr(
        dashboard,
        "compute_spread",
        lambda board: [
            SimpleNamespace(
                player_id="p1",
                name="Trey McBride",
                position="TE",
                team="ARI",
                ecr=20.0,
                model_rank=4,
                rank_spread=8,
                model_projection=178.4,
                blended_projection=70.0,
                actual_total=0.0,
                model_wins=False,
            )
        ],
    )
    monkeypatch.setattr(dashboard, "validate_spread_direction", lambda results, min_spread=10: {"n": 0, "accuracy": 0.0})
    monkeypatch.setattr(dashboard, "_apply_vorp", lambda projections, basis_col="pred_total": pd.Series([20.0]))
    monkeypatch.setattr(dashboard, "build_usage_roles", lambda season: {})
    monkeypatch.setattr(dashboard, "build_team_tendencies", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_age_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_team_changes", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_usage_trends", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_regression_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_injury_risk", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_breakout_candidates", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_defense_rankings", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_sos_adjustment", lambda team, position, def_rankings: (1.0, ""))

    dashboard.build_board_data(2026)

    projections = captured["projections"]
    assert len(projections) == 1
    assert projections.iloc[0]["name"] == "T.McBride"


def test_build_board_data_keeps_distinct_fallback_row_when_player_id_differs(monkeypatch):
    captured = {}

    def _capture_board(adp_df, projections):
        captured["projections"] = projections.copy()
        return ["board"]

    monkeypatch.setattr(
        dashboard,
        "load_adp_board",
        lambda season: pd.DataFrame({"name": ["Bijan Robinson", "Brian Robinson Jr."]}),
    )
    monkeypatch.setattr(
        dashboard,
        "_latest_predictions_csv",
        lambda season: None if season == 2026 else Path("fake-2025.csv"),
    )
    monkeypatch.setattr(
        dashboard,
        "load_model_projections",
        lambda csv, ranking, season: pd.DataFrame([
            {
                "player_id": "BIJAN_PID",
                "name": "B.Robinson",
                "position": "RB",
                "team": "ATL",
                "pred_total": 305.2,
                "actual_total": 0.0,
                "weeks": 17,
                "model_rank_value": 305.2,
            }
        ]),
    )
    monkeypatch.setattr(
        dashboard,
        "load_preseason_projections",
        lambda season, adp_df=None: pd.DataFrame([
            {
                "player_id": "BRIAN_PID",
                "name": "Brian Robinson Jr.",
                "position": "RB",
                "team": "WAS",
                "pred_total": 181.4,
                "actual_total": 0.0,
                "weeks": 0,
                "model_rank_value": 181.4,
            }
        ]),
    )
    monkeypatch.setattr(dashboard, "build_draft_board", _capture_board)
    monkeypatch.setattr(
        dashboard,
        "compute_spread",
        lambda board: [
            SimpleNamespace(
                player_id="p1",
                name="Bijan Robinson",
                position="RB",
                team="ATL",
                ecr=1.0,
                model_rank=1,
                rank_spread=0,
                model_projection=305.2,
                blended_projection=305.2,
                actual_total=0.0,
                model_wins=False,
            )
        ],
    )
    monkeypatch.setattr(dashboard, "validate_spread_direction", lambda results, min_spread=10: {"n": 0, "accuracy": 0.0})
    monkeypatch.setattr(dashboard, "_apply_vorp", lambda projections, basis_col="pred_total": pd.Series([20.0] * len(projections)))
    monkeypatch.setattr(dashboard, "build_usage_roles", lambda season: {})
    monkeypatch.setattr(dashboard, "build_team_tendencies", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_age_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_team_changes", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_usage_trends", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_regression_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_injury_risk", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_breakout_candidates", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_defense_rankings", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_sos_adjustment", lambda team, position, def_rankings: (1.0, ""))

    dashboard.build_board_data(2026)

    projections = captured["projections"]
    assert set(projections["player_id"]) == {"BIJAN_PID", "BRIAN_PID"}


def test_site_source_includes_position_scarcity_strip():
    template = (PROJECT_ROOT / "_site" / "template.html").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert 'id="scarcityStrip"' in template
    assert "function renderScarcityStrip()" in app_js
    assert 'document.getElementById("scarcityStrip").innerHTML = renderScarcityStrip()' in app_js
    assert ".scarcity-strip" in style


def test_site_source_includes_hover_definitions_for_key_terms():
    template = (PROJECT_ROOT / "_site" / "template.html").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert "Projected full-season PPR fantasy points" in template
    assert "Projected full-season PPR fantasy points" in app_js
    assert "Points above a freely available replacement player at this position" in template
    assert "Points above a freely available replacement player at this position" in app_js
    assert "How many spots our model ranks them versus expert consensus. Positive means undervalued" in template
    assert "How many spots our model ranks them versus expert consensus. Positive means undervalued" in app_js
    assert ".def-term" in style


def test_site_source_includes_why_expander():
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert 'Array.isArray(p.why)' in app_js
    assert 'class="why-expander"' in app_js
    assert ".why-expander" in style
    assert ".why-list" in style


def test_site_source_includes_draft_mode_watchlist():
    template = (PROJECT_ROOT / "_site" / "template.html").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert 'id="draftModeBtn"' in template
    assert 'id="draftPanel"' in template
    assert "function toggleDraftMode()" in app_js
    assert "function toggleWatch(playerId)" in app_js
    assert 'class="queue-btn' in app_js
    assert "renderDraftPanel()" in app_js
    assert ".draft-panel" in style
    assert ".watch-chip" in style
    assert "body.draft-mode .search-box" in style
