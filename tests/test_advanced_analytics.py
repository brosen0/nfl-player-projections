"""Tests for advanced analytics features module.

Covers all five feature generators:
1. NewsSentimentAnalyzer
2. CoachingChangeDetector
3. SuspensionRiskTracker
4. TradeDeadlineFeatures
5. PlayoffFeatures
Plus the combined AdvancedAnalyticsEngine.
"""

import pandas as pd
import numpy as np
import pytest
from src.features.advanced_analytics import (
    NewsSentimentAnalyzer,
    CoachingChangeDetector,
    SuspensionRiskTracker,
    TradeDeadlineFeatures,
    PlayoffFeatures,
    AdvancedAnalyticsEngine,
    add_advanced_analytics_features,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_player_df(
    n_players: int = 2,
    n_weeks: int = 10,
    season: int = 2024,
    extra_cols: dict = None,
) -> pd.DataFrame:
    """Create a minimal player-weekly DataFrame for testing."""
    rows = []
    for pid in range(1, n_players + 1):
        team = "KC" if pid % 2 == 1 else "BUF"
        pos = ["QB", "RB", "WR", "TE"][pid % 4]
        for wk in range(1, n_weeks + 1):
            row = {
                "player_id": f"P{pid:03d}",
                "name": f"Player {pid}",
                "team": team,
                "position": pos,
                "season": season,
                "week": wk,
                "fantasy_points": np.random.uniform(5, 25),
            }
            if extra_cols:
                for col, val_fn in extra_cols.items():
                    row[col] = val_fn(pid, wk)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 1. NewsSentimentAnalyzer tests
# =============================================================================

class TestNewsSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = NewsSentimentAnalyzer()

    def test_score_text_positive(self):
        result = self.analyzer.score_text("Player is healthy and dominant this season")
        assert result["sentiment_score"] > 0
        assert result["positive_count"] >= 2
        assert result["negative_count"] == 0

    def test_score_text_negative(self):
        result = self.analyzer.score_text("Player is injured and questionable for Sunday")
        assert result["sentiment_score"] < 0
        assert result["negative_count"] >= 2

    def test_score_text_neutral(self):
        result = self.analyzer.score_text("Player attended practice today")
        assert result["sentiment_score"] == 0.0
        assert result["positive_count"] == 0
        assert result["negative_count"] == 0

    def test_score_text_empty(self):
        result = self.analyzer.score_text("")
        assert result["sentiment_score"] == 0.0

    def test_score_text_none(self):
        result = self.analyzer.score_text(None)
        assert result["sentiment_score"] == 0.0

    def test_intensifiers_amplify(self):
        base = self.analyzer.score_text("Player is healthy")
        intensified = self.analyzer.score_text("Player is extremely healthy")
        assert intensified["sentiment_score"] >= base["sentiment_score"]

    def test_add_sentiment_features_with_text(self):
        df = _make_player_df(
            n_players=2,
            n_weeks=5,
            extra_cols={
                "news_text": lambda pid, wk: (
                    "breakout healthy dominant" if wk <= 3 else "injured questionable"
                )
            },
        )
        result = self.analyzer.add_sentiment_features(df)

        assert "news_sentiment" in result.columns
        assert "news_positive_count" in result.columns
        assert "news_negative_count" in result.columns
        assert "news_sentiment_roll3" in result.columns
        assert "news_sentiment_roll5" in result.columns
        assert "news_sentiment_trend" in result.columns
        assert not result["news_sentiment"].isna().any()

    def test_add_sentiment_features_without_text(self):
        df = _make_player_df(n_players=2, n_weeks=5)
        result = self.analyzer.add_sentiment_features(df)

        assert "news_sentiment" in result.columns
        assert (result["news_sentiment"] == 0.0).all()
        assert (result["news_sentiment_roll3"] == 0.0).all()

    def test_sentiment_score_bounded(self):
        # Extreme text with many positive terms.
        text = " ".join(list(self.analyzer.positive_terms)[:20])
        result = self.analyzer.score_text(text)
        assert -1.0 <= result["sentiment_score"] <= 1.0


# =============================================================================
# 2. CoachingChangeDetector tests
# =============================================================================

class TestCoachingChangeDetector:
    def setup_method(self):
        self.detector = CoachingChangeDetector()

    def test_detect_coaching_change_from_column(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=8,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A" if wk <= 4 else "Coach B"
            },
        )
        result = self.detector.add_coaching_change_features(df)

        assert "coaching_change" in result.columns
        assert "weeks_since_coaching_change" in result.columns
        assert "coaching_adaptation_score" in result.columns
        assert "coaching_change_impact" in result.columns
        assert "new_coaching_staff" in result.columns

        # Week 5 should have coaching_change = 1.
        wk5 = result[result["week"] == 5]
        assert (wk5["coaching_change"] == 1).all()

    def test_no_change_all_same_coach(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={"head_coach": lambda pid, wk: "Coach A"},
        )
        result = self.detector.add_coaching_change_features(df)
        # First row might be 0 (no prior to compare), rest should be 0.
        assert result["coaching_change"].sum() == 0

    def test_scheme_proxy_fallback(self):
        df = _make_player_df(n_players=1, n_weeks=5)
        # No head_coach column -> falls back to scheme proxy.
        result = self.detector.add_coaching_change_features(df)
        assert "coaching_change" in result.columns

    def test_empty_df(self):
        df = pd.DataFrame()
        result = self.detector.add_coaching_change_features(df)
        assert "coaching_change" in result.columns

    def test_adaptation_score_decay(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "head_coach": lambda pid, wk: "Coach A" if wk <= 2 else "Coach B"
            },
        )
        result = self.detector.add_coaching_change_features(df)
        # Adaptation score should decrease over time.
        wk3 = result[result["week"] == 3]["coaching_adaptation_score"].iloc[0]
        wk8 = result[result["week"] == 8]["coaching_adaptation_score"].iloc[0]
        assert wk3 > wk8


# =============================================================================
# 3. SuspensionRiskTracker tests
# =============================================================================

class TestSuspensionRiskTracker:
    def setup_method(self):
        self.tracker = SuspensionRiskTracker()

    def test_no_suspension_columns(self):
        df = _make_player_df(n_players=2, n_weeks=5)
        result = self.tracker.add_suspension_features(df)

        assert "is_suspended" in result.columns
        assert "prior_suspensions" in result.columns
        assert "suspension_risk" in result.columns
        assert (result["is_suspended"] == 0).all()
        assert (result["suspension_risk"] == 0.03).all()

    def test_with_suspension_status(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=8,
            extra_cols={
                "suspension_status": lambda pid, wk: (
                    "Suspended" if 3 <= wk <= 4 else "Active"
                )
            },
        )
        result = self.tracker.add_suspension_features(df)

        # Weeks 3-4 should be flagged as suspended.
        assert result[result["week"] == 3]["is_suspended"].iloc[0] == 1
        assert result[result["week"] == 4]["is_suspended"].iloc[0] == 1
        assert result[result["week"] == 5]["is_suspended"].iloc[0] == 0

    def test_with_games_suspended(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=5,
            extra_cols={
                "games_suspended": lambda pid, wk: 2 if wk == 2 else 0,
            },
        )
        result = self.tracker.add_suspension_features(df)
        assert "career_games_suspended" in result.columns

    def test_recidivism_increases_risk(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "suspension_status": lambda pid, wk: (
                    "Suspended" if wk in (2, 6) else "Active"
                )
            },
        )
        result = self.tracker.add_suspension_features(df)

        # After two suspension events, risk should be higher.
        late_risk = result[result["week"] == 8]["suspension_risk"].iloc[0]
        assert late_risk > 0.03  # Higher than baseline

    def test_empty_df(self):
        df = pd.DataFrame()
        result = self.tracker.add_suspension_features(df)
        assert "is_suspended" in result.columns

    def test_suspension_risk_bounded(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "suspension_status": lambda pid, wk: "Suspended",
            },
        )
        result = self.tracker.add_suspension_features(df)
        assert (result["suspension_risk"] <= 1.0).all()
        assert (result["suspension_risk"] >= 0.0).all()


# =============================================================================
# 4. TradeDeadlineFeatures tests
# =============================================================================

class TestTradeDeadlineFeatures:
    def setup_method(self):
        self.trade = TradeDeadlineFeatures()

    def test_basic_features(self):
        df = _make_player_df(n_players=1, n_weeks=12)
        result = self.trade.add_trade_deadline_features(df)

        assert "weeks_to_deadline" in result.columns
        assert "past_deadline" in result.columns
        assert "deadline_proximity" in result.columns
        assert "in_trade_window" in result.columns

    def test_weeks_to_deadline(self):
        df = _make_player_df(n_players=1, n_weeks=12)
        result = self.trade.add_trade_deadline_features(df)

        wk4 = result[result["week"] == 4]["weeks_to_deadline"].iloc[0]
        assert wk4 == 4  # 8 - 4 = 4

        wk10 = result[result["week"] == 10]["weeks_to_deadline"].iloc[0]
        assert wk10 == 0  # Past deadline

    def test_past_deadline(self):
        df = _make_player_df(n_players=1, n_weeks=12)
        result = self.trade.add_trade_deadline_features(df)

        assert result[result["week"] == 6]["past_deadline"].iloc[0] == 0
        assert result[result["week"] == 10]["past_deadline"].iloc[0] == 1

    def test_trade_window(self):
        df = _make_player_df(n_players=1, n_weeks=12)
        result = self.trade.add_trade_deadline_features(df)

        # Weeks 6-9 should be in trade window (deadline_week-2 to deadline_week+1).
        for wk in [6, 7, 8, 9]:
            assert result[result["week"] == wk]["in_trade_window"].iloc[0] == 1
        for wk in [4, 5, 10, 11]:
            assert result[result["week"] == wk]["in_trade_window"].iloc[0] == 0

    def test_team_record_features(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=10,
            extra_cols={
                "team_wins": lambda pid, wk: 5,
                "team_losses": lambda pid, wk: 2,
            },
        )
        result = self.trade.add_trade_deadline_features(df)
        assert "trade_deadline_contender" in result.columns
        # With 5-2 record (71% win rate), should be contender in trade window.
        wk7 = result[result["week"] == 7]
        assert wk7["trade_deadline_contender"].iloc[0] == 1

    def test_mid_season_trade_detection(self):
        df = _make_player_df(n_players=1, n_weeks=10)
        # Simulate mid-season trade: team changes from KC to BUF at week 6.
        df.loc[df["week"] >= 6, "team"] = "BUF"
        result = self.trade.add_trade_deadline_features(df)

        assert "mid_season_trade" in result.columns
        assert result[result["week"] == 6]["mid_season_trade"].iloc[0] == 1

    def test_empty_df(self):
        df = pd.DataFrame()
        result = self.trade.add_trade_deadline_features(df)
        assert "weeks_to_deadline" in result.columns


# =============================================================================
# 5. PlayoffFeatures tests
# =============================================================================

class TestPlayoffFeatures:
    def setup_method(self):
        self.playoff = PlayoffFeatures()

    def test_basic_features(self):
        df = _make_player_df(n_players=1, n_weeks=18)
        result = self.playoff.add_playoff_features(df)

        assert "playoff_proximity" in result.columns
        assert "is_playoff_week" in result.columns
        assert "weeks_remaining" in result.columns
        assert "meaningful_game" in result.columns

    def test_playoff_proximity_increases(self):
        df = _make_player_df(n_players=1, n_weeks=18)
        result = self.playoff.add_playoff_features(df)

        prox_wk5 = result[result["week"] == 5]["playoff_proximity"].iloc[0]
        prox_wk16 = result[result["week"] == 16]["playoff_proximity"].iloc[0]
        assert prox_wk16 > prox_wk5

    def test_weeks_remaining(self):
        df = _make_player_df(n_players=1, n_weeks=18)
        result = self.playoff.add_playoff_features(df)

        assert result[result["week"] == 1]["weeks_remaining"].iloc[0] == 17
        assert result[result["week"] == 18]["weeks_remaining"].iloc[0] == 0

    def test_eliminated_proxy(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 2,
                "team_losses": lambda pid, wk: 10,
            },
        )
        result = self.playoff.add_playoff_features(df)

        # With 2-10 record, projected wins ~2.8, should be eliminated late.
        late = result[result["week"] >= 14]
        assert (late["eliminated_proxy"] == 1).any()

    def test_clinched_proxy(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 12,
                "team_losses": lambda pid, wk: 1,
            },
        )
        result = self.playoff.add_playoff_features(df)

        # With 12-1 record, should be clinched.
        late = result[result["week"] >= 14]
        assert (late["clinched_proxy"] == 1).any()

    def test_rest_risk(self):
        df = _make_player_df(
            n_players=1,
            n_weeks=18,
            extra_cols={
                "team_wins": lambda pid, wk: 13,
                "team_losses": lambda pid, wk: 1,
            },
        )
        result = self.playoff.add_playoff_features(df)

        # Clinched team with <= 2 weeks remaining should have rest risk.
        wk17 = result[result["week"] == 17]
        assert wk17["rest_risk"].iloc[0] == 1

    def test_playoff_week_detection(self):
        # Create data with week > 18 to simulate playoff.
        df = _make_player_df(n_players=1, n_weeks=2)
        df["week"] = [19, 20]
        result = self.playoff.add_playoff_features(df)
        assert (result["is_playoff_week"] == 1).all()

    def test_empty_df(self):
        df = pd.DataFrame()
        result = self.playoff.add_playoff_features(df)
        assert "playoff_proximity" in result.columns


# =============================================================================
# Combined AdvancedAnalyticsEngine tests
# =============================================================================

class TestAdvancedAnalyticsEngine:
    def setup_method(self):
        self.engine = AdvancedAnalyticsEngine()

    def test_all_features_added(self):
        df = _make_player_df(n_players=2, n_weeks=10)
        result = self.engine.add_all_features(df)

        # Check at least one column from each module.
        assert "news_sentiment" in result.columns
        assert "coaching_change" in result.columns
        assert "suspension_risk" in result.columns
        assert "weeks_to_deadline" in result.columns
        assert "playoff_proximity" in result.columns

    def test_no_nans_in_key_features(self):
        df = _make_player_df(n_players=3, n_weeks=12)
        result = self.engine.add_all_features(df)

        key_cols = [
            "news_sentiment",
            "coaching_change",
            "suspension_risk",
            "in_trade_window",
            "meaningful_game",
        ]
        for col in key_cols:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_convenience_function(self):
        df = _make_player_df(n_players=1, n_weeks=5)
        result = add_advanced_analytics_features(df)
        assert "news_sentiment" in result.columns
        assert len(result) == len(df)

    def test_empty_df_handled(self):
        df = pd.DataFrame()
        result = self.engine.add_all_features(df)
        assert result.empty

    def test_row_count_preserved(self):
        df = _make_player_df(n_players=3, n_weeks=8)
        result = self.engine.add_all_features(df)
        assert len(result) == len(df)

    def test_original_columns_preserved(self):
        df = _make_player_df(n_players=2, n_weeks=5)
        original_cols = set(df.columns)
        result = self.engine.add_all_features(df)
        assert original_cols.issubset(set(result.columns))

    def test_no_temporal_leakage_in_sentiment(self):
        """Sentiment rolling features use shift(1) so current week is excluded."""
        df = _make_player_df(
            n_players=1,
            n_weeks=6,
            extra_cols={
                "news_text": lambda pid, wk: (
                    "elite breakout dominant" if wk == 6 else "normal practice"
                )
            },
        )
        result = self.engine.add_all_features(df)

        # The rolling sentiment for week 6 should NOT include week 6's score
        # because of the shift(1) in rolling computation.
        wk5_roll = result[result["week"] == 5]["news_sentiment_roll3"].iloc[0]
        wk6_roll = result[result["week"] == 6]["news_sentiment_roll3"].iloc[0]
        # Week 6's rolling should be based on weeks 3-5 (all "normal practice"),
        # not including the "elite breakout" text from week 6.
        assert abs(wk5_roll - wk6_roll) < 0.5  # Should be similar neutral values

    def test_full_feature_integration(self):
        """Test with all optional columns present."""
        df = _make_player_df(
            n_players=2,
            n_weeks=12,
            extra_cols={
                "news_text": lambda pid, wk: "healthy starter",
                "head_coach": lambda pid, wk: "Coach A" if wk <= 6 else "Coach B",
                "suspension_status": lambda pid, wk: "Active",
                "team_wins": lambda pid, wk: wk // 2,
                "team_losses": lambda pid, wk: wk - wk // 2,
            },
        )
        result = self.engine.add_all_features(df)
        new_cols = [c for c in result.columns if c not in df.columns]
        assert len(new_cols) >= 25  # Should add many features
