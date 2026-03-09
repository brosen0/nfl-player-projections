"""
Advanced Analytics Features for NFL Player Projections.

Provides contextual features that capture situational factors beyond
raw player stats:

1. NewsSentimentAnalyzer   - NLP-based news sentiment scoring per player
2. CoachingChangeDetector  - Detects coaching changes and quantifies impact
3. SuspensionRiskTracker   - Tracks suspension history and estimates risk
4. TradeDeadlineFeatures   - Features around mid-season trade activity
5. PlayoffFeatures         - Playoff context, elimination pressure, rest patterns

All features are designed to be temporally safe (no future leakage) and
gracefully degrade when optional data columns are missing.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. NEWS / SENTIMENT NLP
# =============================================================================

# Pre-built sentiment lexicon tuned for NFL/fantasy football context.
# Positive words suggest increased opportunity or performance upside.
# Negative words suggest reduced opportunity or downside risk.
_POSITIVE_TERMS = {
    "breakout", "elite", "explosive", "healthy", "dominant", "upgrade",
    "starter", "workhorse", "bellcow", "promoted", "cleared", "return",
    "activated", "featured", "emerging", "impressive", "extension",
    "signing", "pro bowl", "all-pro", "record", "career high", "mvp",
    "confident", "strong", "leader", "chemistry", "rapport", "trust",
    "reliable", "consistent", "volume", "opportunity", "upside",
}

_NEGATIVE_TERMS = {
    "injured", "injury", "questionable", "doubtful", "out", "ir",
    "suspended", "suspension", "benched", "demoted", "limited",
    "fumble", "drop", "bust", "decline", "aging", "washed",
    "holdout", "trade request", "conflict", "arrest", "legal",
    "concussion", "hamstring", "acl", "mcl", "torn", "fracture",
    "surgery", "setback", "downgrade", "disappointing", "struggling",
    "committee", "timeshare", "reduced", "backup", "depth chart",
}

# Intensifiers scale the sentiment magnitude.
_INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "slightly": 0.5, "somewhat": 0.6,
    "significantly": 1.8, "major": 1.7, "minor": 0.4, "serious": 1.9,
}


class NewsSentimentAnalyzer:
    """Compute per-player news sentiment features using keyword-based NLP.

    When a ``news_text`` column is present in the DataFrame, each row's text
    is scored on a [-1, 1] scale.  Rolling averages capture recent sentiment
    trajectory.  When no text column exists the features default to neutral
    (0.0) so downstream models are unaffected.
    """

    def __init__(
        self,
        positive_terms: set = None,
        negative_terms: set = None,
        intensifiers: dict = None,
    ):
        self.positive_terms = positive_terms or _POSITIVE_TERMS
        self.negative_terms = negative_terms or _NEGATIVE_TERMS
        self.intensifiers = intensifiers or _INTENSIFIERS
        # Pre-compile patterns for efficiency.
        self._pos_pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in self.positive_terms) + r")\b",
            re.IGNORECASE,
        )
        self._neg_pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in self.negative_terms) + r")\b",
            re.IGNORECASE,
        )
        self._intensifier_pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in self.intensifiers) + r")\b",
            re.IGNORECASE,
        )

    def score_text(self, text: str) -> Dict[str, float]:
        """Score a single text snippet and return sentiment metrics."""
        if not isinstance(text, str) or not text.strip():
            return {"sentiment_score": 0.0, "positive_count": 0, "negative_count": 0}

        text_lower = text.lower()
        pos_matches = self._pos_pattern.findall(text_lower)
        neg_matches = self._neg_pattern.findall(text_lower)
        intensifier_matches = self._intensifier_pattern.findall(text_lower)

        # Base counts.
        pos_count = len(pos_matches)
        neg_count = len(neg_matches)

        # Average intensifier multiplier (default 1.0 when none found).
        if intensifier_matches:
            avg_intensity = np.mean(
                [self.intensifiers.get(m.lower(), 1.0) for m in intensifier_matches]
            )
        else:
            avg_intensity = 1.0

        total = pos_count + neg_count
        if total == 0:
            raw_score = 0.0
        else:
            raw_score = (pos_count - neg_count) / total

        # Scale by intensity and clamp.
        sentiment = float(np.clip(raw_score * avg_intensity, -1.0, 1.0))

        return {
            "sentiment_score": sentiment,
            "positive_count": pos_count,
            "negative_count": neg_count,
        }

    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features to the DataFrame.

        Requires a ``news_text`` column.  When absent, adds neutral defaults.
        """
        result = df.copy()

        if "news_text" in result.columns:
            scores = result["news_text"].apply(self.score_text)
            result["news_sentiment"] = scores.apply(lambda s: s["sentiment_score"])
            result["news_positive_count"] = scores.apply(
                lambda s: s["positive_count"]
            ).astype(int)
            result["news_negative_count"] = scores.apply(
                lambda s: s["negative_count"]
            ).astype(int)
            logger.info("Computed news sentiment from news_text column")
        else:
            result["news_sentiment"] = 0.0
            result["news_positive_count"] = 0
            result["news_negative_count"] = 0
            logger.info("No news_text column; defaulting sentiment to neutral")

        # Rolling sentiment (3-week and 5-week windows).
        if "player_id" in result.columns:
            result = result.sort_values(
                ["player_id", "season", "week"]
            ).reset_index(drop=True)

            for window in [3, 5]:
                col_name = f"news_sentiment_roll{window}"
                result[col_name] = (
                    result.groupby("player_id")["news_sentiment"]
                    .transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                    )
                    .fillna(0.0)
                )

            # Sentiment trend (current - rolling 5).
            result["news_sentiment_trend"] = (
                result["news_sentiment"] - result["news_sentiment_roll5"]
            ).fillna(0.0)
        else:
            result["news_sentiment_roll3"] = 0.0
            result["news_sentiment_roll5"] = 0.0
            result["news_sentiment_trend"] = 0.0

        return result


# =============================================================================
# 2. COACHING CHANGE DETECTION
# =============================================================================

class CoachingChangeDetector:
    """Detect coaching changes and quantify their impact on player production.

    When a ``head_coach`` column is present the detector identifies weeks
    where a team's head coach changed (mid-season firings or off-season
    hires) and creates features capturing the disruption and adaptation
    window.

    If no coach column exists, it falls back to detecting large shifts in
    team passing rate as a proxy for scheme change.
    """

    # Historical average fantasy impact of coaching changes by position.
    # Positive = production tends to increase; negative = tends to decrease.
    COACHING_CHANGE_IMPACT = {
        "QB": -0.05,   # QBs slightly hurt by new system learning curve
        "RB": 0.02,    # RBs roughly neutral; scheme-dependent
        "WR": -0.03,   # WRs need rapport with new QB/scheme
        "TE": 0.01,    # TEs least affected
    }

    # Weeks it typically takes players to adapt to a new coaching staff.
    ADAPTATION_WEEKS = 6

    def add_coaching_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add coaching-change features to the DataFrame."""
        result = df.copy()

        if result.empty or "team" not in result.columns:
            self._add_defaults(result)
            return result

        result = result.sort_values(
            ["player_id", "season", "week"]
        ).reset_index(drop=True)

        if "head_coach" in result.columns:
            result = self._detect_from_coach_column(result)
        else:
            result = self._detect_from_scheme_proxy(result)

        return result

    # ----- private helpers -----

    def _detect_from_coach_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use explicit head_coach column to detect changes."""
        # Per-team coach timeline.
        team_grp = df.groupby("team", sort=False)
        prev_coach = team_grp["head_coach"].shift(1)
        coach_changed = (
            (df["head_coach"] != prev_coach) & prev_coach.notna()
        ).astype(int)

        df["coaching_change"] = coach_changed

        # Weeks since most recent coaching change for this team.
        df["_cc_cumsum"] = df.groupby("team")["coaching_change"].cumsum()
        df["weeks_since_coaching_change"] = (
            df.groupby(["team", "_cc_cumsum"]).cumcount()
        )
        df.drop(columns=["_cc_cumsum"], inplace=True)

        # Adaptation score: 1.0 immediately after change, decays to 0 over
        # ADAPTATION_WEEKS.
        df["coaching_adaptation_score"] = np.clip(
            1.0 - df["weeks_since_coaching_change"] / self.ADAPTATION_WEEKS,
            0.0,
            1.0,
        )

        # Position-specific expected impact.
        df["coaching_change_impact"] = (
            df["position"].map(self.COACHING_CHANGE_IMPACT).fillna(0.0)
            * df["coaching_adaptation_score"]
        )

        # New coach indicator for first 6 weeks.
        df["new_coaching_staff"] = (
            df["weeks_since_coaching_change"] < self.ADAPTATION_WEEKS
        ).astype(int)

        logger.info("Detected coaching changes from head_coach column")
        return df

    def _detect_from_scheme_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect likely scheme changes from team passing-rate shifts."""
        # Use team-level pass rate change as a proxy for coaching/scheme change.
        if "team_a_pass_rate" in df.columns:
            team_season = df.groupby(["team", "season"], sort=False)[
                "team_a_pass_rate"
            ].transform("mean")
            prev_season_rate = df.groupby("team")["team_a_pass_rate"].shift(17)
            rate_delta = (team_season - prev_season_rate).abs().fillna(0.0)

            # A shift > 10 percentage points suggests a scheme overhaul.
            df["coaching_change"] = (rate_delta > 0.10).astype(int)
        else:
            df["coaching_change"] = 0

        df["weeks_since_coaching_change"] = 0
        df["coaching_adaptation_score"] = 0.0
        df["coaching_change_impact"] = 0.0
        df["new_coaching_staff"] = 0

        logger.info("Estimated coaching changes from scheme proxy (no coach column)")
        return df

    @staticmethod
    def _add_defaults(df: pd.DataFrame) -> None:
        """Add default (neutral) coaching-change columns."""
        df["coaching_change"] = 0
        df["weeks_since_coaching_change"] = 0
        df["coaching_adaptation_score"] = 0.0
        df["coaching_change_impact"] = 0.0
        df["new_coaching_staff"] = 0


# =============================================================================
# 3. SUSPENSION RISK TRACKER
# =============================================================================

class SuspensionRiskTracker:
    """Track suspension history and estimate future suspension risk.

    Features are derived from an optional ``suspension_status`` or
    ``games_suspended`` column.  When absent, all players receive
    a baseline low risk score.
    """

    # Base risk of suspension by category (annualized probability).
    BASE_RISK = 0.03  # ~3% of players face some suspension each year

    # Recidivism multiplier: prior suspensions increase future risk.
    RECIDIVISM_FACTOR = 2.5

    def add_suspension_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add suspension-related features."""
        result = df.copy()

        if result.empty:
            self._add_defaults(result)
            return result

        result = result.sort_values(
            ["player_id", "season", "week"]
        ).reset_index(drop=True)

        has_suspension_col = "suspension_status" in result.columns
        has_games_col = "games_suspended" in result.columns

        if has_suspension_col or has_games_col:
            result = self._compute_from_data(result, has_suspension_col, has_games_col)
        else:
            self._add_defaults(result)
            logger.info("No suspension columns; defaulting risk to baseline")

        return result

    def _compute_from_data(
        self, df: pd.DataFrame, has_status: bool, has_games: bool
    ) -> pd.DataFrame:
        """Compute suspension features from available data."""
        # Current suspension flag.
        if has_status:
            df["is_suspended"] = (
                df["suspension_status"]
                .fillna("")
                .str.lower()
                .str.contains("suspend", na=False)
            ).astype(int)
        elif has_games:
            df["is_suspended"] = (df["games_suspended"].fillna(0) > 0).astype(int)
        else:
            df["is_suspended"] = 0

        # Cumulative prior suspensions (shifted to avoid leakage).
        df["prior_suspensions"] = (
            df.groupby("player_id")["is_suspended"]
            .transform(lambda x: x.shift(1).cumsum())
            .fillna(0)
            .astype(int)
        )

        # Games missed to suspension (cumulative, shifted).
        if has_games:
            df["career_games_suspended"] = (
                df.groupby("player_id")["games_suspended"]
                .transform(lambda x: x.shift(1).cumsum())
                .fillna(0)
                .astype(int)
            )
        else:
            df["career_games_suspended"] = df["prior_suspensions"]

        # Risk score: base risk * recidivism multiplier^(prior_suspensions).
        df["suspension_risk"] = np.clip(
            self.BASE_RISK * (self.RECIDIVISM_FACTOR ** df["prior_suspensions"]),
            0.0,
            1.0,
        )

        # Returning-from-suspension indicator (first 3 weeks back).
        grp = df.groupby("player_id", sort=False)
        prev_suspended = grp["is_suspended"].shift(1).fillna(0)
        df["returning_from_suspension"] = (
            (df["is_suspended"] == 0) & (prev_suspended == 1)
        ).astype(int)

        # Propagate for 3 weeks after return.
        df["_rfs_cumsum"] = df.groupby("player_id")[
            "returning_from_suspension"
        ].cumsum()
        df["_rfs_count"] = df.groupby(["player_id", "_rfs_cumsum"]).cumcount()
        df["weeks_since_suspension_return"] = np.where(
            df["_rfs_cumsum"] > 0, df["_rfs_count"], 99
        )
        df["suspension_return_window"] = (
            df["weeks_since_suspension_return"] <= 3
        ).astype(int)
        df.drop(
            columns=["_rfs_cumsum", "_rfs_count", "weeks_since_suspension_return"],
            inplace=True,
        )

        logger.info("Computed suspension features from data columns")
        return df

    @staticmethod
    def _add_defaults(df: pd.DataFrame) -> None:
        """Add baseline suspension columns when no data is available."""
        df["is_suspended"] = 0
        df["prior_suspensions"] = 0
        df["career_games_suspended"] = 0
        df["suspension_risk"] = 0.03
        df["returning_from_suspension"] = 0
        df["suspension_return_window"] = 0


# =============================================================================
# 4. TRADE DEADLINE FEATURES
# =============================================================================

# NFL trade deadline is typically Week 8 (Tuesday after Week 8 games).
_TRADE_DEADLINE_WEEK = 8


class TradeDeadlineFeatures:
    """Features capturing trade-deadline dynamics.

    The NFL trade deadline falls around Week 8.  Players on contending
    teams may see increased usage, while players on losing teams may be
    traded.  Features encode proximity to the deadline, team record
    context, and whether a trade actually occurred.
    """

    def __init__(self, deadline_week: int = _TRADE_DEADLINE_WEEK):
        self.deadline_week = deadline_week

    def add_trade_deadline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trade-deadline related features."""
        result = df.copy()

        if result.empty or "week" not in result.columns:
            self._add_defaults(result)
            return result

        result = result.sort_values(
            ["player_id", "season", "week"]
        ).reset_index(drop=True)

        # Weeks until / since trade deadline.
        result["weeks_to_deadline"] = (self.deadline_week - result["week"]).clip(
            lower=0
        )
        result["past_deadline"] = (result["week"] > self.deadline_week).astype(int)
        result["deadline_proximity"] = np.clip(
            1.0 - abs(result["week"] - self.deadline_week) / 4.0, 0.0, 1.0
        )

        # Trade window: weeks 6-9 are the "hot" trade window.
        result["in_trade_window"] = (
            (result["week"] >= self.deadline_week - 2)
            & (result["week"] <= self.deadline_week + 1)
        ).astype(int)

        # Team record context (winning teams buy, losing teams sell).
        if "team_wins" in result.columns and "team_losses" in result.columns:
            total_games = (result["team_wins"] + result["team_losses"]).replace(0, 1)
            result["team_win_pct"] = result["team_wins"] / total_games

            # Contender vs seller designation near deadline.
            result["trade_deadline_contender"] = (
                (result["in_trade_window"] == 1)
                & (result["team_win_pct"] >= 0.5)
            ).astype(int)
            result["trade_deadline_seller"] = (
                (result["in_trade_window"] == 1)
                & (result["team_win_pct"] < 0.4)
            ).astype(int)
        else:
            result["team_win_pct"] = 0.5
            result["trade_deadline_contender"] = 0
            result["trade_deadline_seller"] = 0

        # Detect actual mid-season trades via team changes in-season.
        if "team" in result.columns and "player_id" in result.columns:
            grp = result.groupby("player_id", sort=False)
            prev_team = grp["team"].shift(1)
            prev_season = grp["season"].shift(1)
            same_season = result["season"] == prev_season
            team_changed = (result["team"] != prev_team) & prev_team.notna()

            result["mid_season_trade"] = (
                same_season & team_changed
            ).astype(int)

            # Weeks since trade (for adjustment period tracking).
            result["_trade_cumsum"] = result.groupby("player_id")[
                "mid_season_trade"
            ].cumsum()
            result["weeks_since_trade"] = result.groupby(
                ["player_id", "_trade_cumsum"]
            ).cumcount()
            result.drop(columns=["_trade_cumsum"], inplace=True)

            # Trade adjustment window (first 4 weeks on new team mid-season).
            result["trade_adjustment_window"] = (
                (result["mid_season_trade"].groupby(result["player_id"]).cumsum() > 0)
                & (result["weeks_since_trade"] <= 4)
            ).astype(int)
        else:
            result["mid_season_trade"] = 0
            result["weeks_since_trade"] = 0
            result["trade_adjustment_window"] = 0

        logger.info("Added trade deadline features")
        return result

    @staticmethod
    def _add_defaults(df: pd.DataFrame) -> None:
        """Add default trade deadline columns."""
        df["weeks_to_deadline"] = 0
        df["past_deadline"] = 0
        df["deadline_proximity"] = 0.0
        df["in_trade_window"] = 0
        df["team_win_pct"] = 0.5
        df["trade_deadline_contender"] = 0
        df["trade_deadline_seller"] = 0
        df["mid_season_trade"] = 0
        df["weeks_since_trade"] = 0
        df["trade_adjustment_window"] = 0


# =============================================================================
# 5. PLAYOFF FEATURES
# =============================================================================

# NFL regular season is 18 weeks; playoffs start week 19.
_REGULAR_SEASON_WEEKS = 18
_PLAYOFF_CLINCH_EARLIEST = 12  # Earliest a team can clinch (roughly)


class PlayoffFeatures:
    """Features capturing playoff context and implications.

    Encodes whether a team is in playoff contention, has clinched,
    or is eliminated.  Also captures rest/load management patterns
    that occur when teams have locked up seeding.
    """

    def add_playoff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add playoff-context features."""
        result = df.copy()

        if result.empty or "week" not in result.columns:
            self._add_defaults(result)
            return result

        result = result.sort_values(
            ["player_id", "season", "week"]
        ).reset_index(drop=True)

        # Playoff proximity: how close to playoffs (increases as week grows).
        result["playoff_proximity"] = np.clip(
            (result["week"] - 10) / (_REGULAR_SEASON_WEEKS - 10), 0.0, 1.0
        )

        # Is this a playoff week?
        result["is_playoff_week"] = (
            result["week"] > _REGULAR_SEASON_WEEKS
        ).astype(int)

        # Weeks remaining in regular season.
        result["weeks_remaining"] = (
            _REGULAR_SEASON_WEEKS - result["week"]
        ).clip(lower=0)

        # Meaningful games indicator: team still in contention AND
        # late enough in season for it to matter.
        if "team_wins" in result.columns and "team_losses" in result.columns:
            total_games = (result["team_wins"] + result["team_losses"]).replace(0, 1)
            win_pct = result["team_wins"] / total_games

            # Projected wins (simple linear projection).
            projected_wins = win_pct * _REGULAR_SEASON_WEEKS

            # Elimination proxy: projected to win < 7 games with < 6 weeks left.
            result["eliminated_proxy"] = (
                (projected_wins < 7) & (result["weeks_remaining"] < 6)
            ).astype(int)

            # Clinched proxy: projected to win >= 11 games with < 6 weeks left.
            result["clinched_proxy"] = (
                (projected_wins >= 11) & (result["weeks_remaining"] < 6)
            ).astype(int)

            # Meaningful game: not eliminated and not fully clinched (or playoff wk).
            result["meaningful_game"] = (
                (result["eliminated_proxy"] == 0)
                & (result["is_playoff_week"] == 0)
            ).astype(int)

            # Rest risk: team has clinched and might rest starters in final weeks.
            result["rest_risk"] = (
                (result["clinched_proxy"] == 1)
                & (result["weeks_remaining"] <= 2)
            ).astype(int)

            # Playoff push: in contention with 4-8 weeks remaining.
            result["playoff_push"] = (
                (result["eliminated_proxy"] == 0)
                & (result["weeks_remaining"] >= 2)
                & (result["weeks_remaining"] <= 8)
                & (win_pct >= 0.4)
            ).astype(int)
        else:
            result["eliminated_proxy"] = 0
            result["clinched_proxy"] = 0
            result["meaningful_game"] = 1
            result["rest_risk"] = 0
            result["playoff_push"] = 0

        # Playoff seed implications for load management (if seed data exists).
        if "playoff_seed" in result.columns:
            result["has_bye_week_seed"] = (
                result["playoff_seed"].fillna(99) <= 1
            ).astype(int)
        else:
            result["has_bye_week_seed"] = 0

        logger.info("Added playoff features")
        return result

    @staticmethod
    def _add_defaults(df: pd.DataFrame) -> None:
        """Add default playoff columns."""
        df["playoff_proximity"] = 0.0
        df["is_playoff_week"] = 0
        df["weeks_remaining"] = 18
        df["eliminated_proxy"] = 0
        df["clinched_proxy"] = 0
        df["meaningful_game"] = 1
        df["rest_risk"] = 0
        df["playoff_push"] = 0
        df["has_bye_week_seed"] = 0


# =============================================================================
# COMBINED ADVANCED ANALYTICS ENGINE
# =============================================================================

class AdvancedAnalyticsEngine:
    """Orchestrates all advanced analytics feature generators.

    Usage::

        engine = AdvancedAnalyticsEngine()
        df = engine.add_all_features(df)
    """

    def __init__(self):
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.coaching_detector = CoachingChangeDetector()
        self.suspension_tracker = SuspensionRiskTracker()
        self.trade_deadline = TradeDeadlineFeatures()
        self.playoff_features = PlayoffFeatures()

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all advanced analytics features to the DataFrame.

        Each sub-module gracefully handles missing columns by adding
        neutral defaults, so this method is always safe to call.
        """
        if df.empty:
            return df

        result = df.copy()

        logger.info("Adding advanced analytics features...")

        # 1. News sentiment NLP.
        result = self.sentiment_analyzer.add_sentiment_features(result)

        # 2. Coaching change detection.
        result = self.coaching_detector.add_coaching_change_features(result)

        # 3. Suspension risk tracking.
        result = self.suspension_tracker.add_suspension_features(result)

        # 4. Trade deadline features.
        result = self.trade_deadline.add_trade_deadline_features(result)

        # 5. Playoff features.
        result = self.playoff_features.add_playoff_features(result)

        new_cols = [c for c in result.columns if c not in df.columns]
        logger.info(f"Added {len(new_cols)} advanced analytics features")

        return result


def add_advanced_analytics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to add all advanced analytics features."""
    engine = AdvancedAnalyticsEngine()
    return engine.add_all_features(df)
