"""Data loading utilities for NFL prediction model training."""
import logging

import numpy as np
import pandas as pd

from config.settings import (
    POSITIONS,
    MODEL_CONFIG,
    MIN_TRAINING_SEASONS_1W,
    MIN_TRAINING_SEASONS_18W,
    MIN_TRAINING_SEASONS_4W,
    MIN_PLAYERS_PER_POSITION,
)
from src.utils.database import DatabaseManager
from src.utils.data_manager import DataManager, auto_refresh_data

logger = logging.getLogger(__name__)


def load_training_data(positions: list = None, min_games: int = 4,
                       test_season: int = None,
                       n_train_seasons: int = None,
                       optimize_training_years: bool = False,
                       strict_requirements: bool = False) -> tuple:
    """
    Load and prepare training data with automatic train/test split.

    Uses the latest available season as test set for out-of-sample evaluation.
    When n_train_seasons is None, uses ALL available training seasons.

    Args:
        positions: List of positions to load
        min_games: Minimum games for player inclusion
        test_season: Override test season (None = auto-select latest)
        n_train_seasons: Max training seasons (None = use all available)
        optimize_training_years: If True, dynamically select optimal years per position

    Returns:
        Tuple of (train_data, test_data, train_seasons, test_season)
    """
    # Auto-refresh and check data availability
    print("Checking data availability...")
    data_status = auto_refresh_data()
    print(f"  Latest available season: {data_status['latest_season']}")
    print(f"  Available seasons: {data_status['available_seasons']}")

    data_manager = DataManager()
    optimal_years = None

    if optimize_training_years:
        from src.utils.training_years_selector import (
            select_optimal_training_years,
            save_optimal_training_years,
            get_available_seasons_from_data,
        )
        # Load raw data first for optimization
        db = DatabaseManager()
        all_raw = []
        for pos in (positions or POSITIONS):
            d = db.get_all_players_for_training(position=pos, min_games=min_games)
            if len(d) > 0:
                all_raw.append(d)
        if all_raw:
            raw_df = pd.concat(all_raw, ignore_index=True)
            seasons = get_available_seasons_from_data(raw_df)
            test_season = test_season or max(seasons)
            # Run optimization (requires prepared data - we do a quick pass)
            # For speed, use raw data with minimal prep
            print("  Optimizing training years per position...")
            optimal_years = select_optimal_training_years(
                raw_df, positions=positions or POSITIONS,
                test_season=test_season,
            )
            save_optimal_training_years(optimal_years, test_season)
            print(f"  Optimal years: {optimal_years}")

    train_seasons, auto_test_season = data_manager.get_train_test_seasons(
        test_season=test_season,
        n_train_seasons=n_train_seasons,
        optimal_years_per_position=optimal_years,
    )

    db = DatabaseManager()
    all_data = []
    positions = positions or POSITIONS

    for position in positions:
        print(f"Loading data for {position}...")
        pos_data = db.get_all_players_for_training(position=position, min_games=min_games)

        if len(pos_data) > 0:
            all_data.append(pos_data)
            print(f"  Loaded {len(pos_data)} records for {position}")

    if not all_data:
        from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
        raise ValueError(
            "No data found in database. Please load real NFL data first using:\n"
            f"  python3 src/data/nfl_data_loader.py --seasons {MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON}\n"
            "(or omit --seasons for default: config range). This system only uses real NFL data."
        )

    combined = pd.concat(all_data, ignore_index=True)

    # Guard: training data must not contain model-output or leakage columns.
    from src.utils.leakage import find_leakage_columns
    leaked = find_leakage_columns(combined.columns, ban_utilization_score=False)
    if leaked:
        logger.warning("Dropping %d leakage-risk columns from training data: %s",
                        len(leaked), sorted(leaked)[:10])
        combined = combined.drop(columns=leaked, errors="ignore")

    # Split into train/test (strict unseen test: test season must not be in train)
    assert auto_test_season not in train_seasons, (
        f"Test season {auto_test_season} must not be in train seasons {train_seasons}"
    )
    train_data = combined[combined['season'].isin(train_seasons)]
    test_data = combined[combined['season'] == auto_test_season]

    # In-season: pipeline requires current season as test and non-empty test set
    from src.utils.nfl_calendar import get_current_nfl_season, current_season_has_weeks_played
    current_season = get_current_nfl_season()
    in_season = current_season_has_weeks_played()
    if in_season and auto_test_season != current_season:
        raise ValueError(
            "The pipeline requires the current season as test when it has started. "
            f"Expected test_season={current_season}, got {auto_test_season}. "
            "Run auto_refresh or load current season from PBP and re-run."
        )
    if in_season and len(test_data) == 0:
        raise ValueError(
            "Current season is in progress but test set is empty. "
            "Load current season from play-by-play (e.g. python -m src.data.auto_refresh) and re-run."
        )

    print(f"\nData split:")
    print(f"  Training: {len(train_data)} records from seasons {train_seasons}")
    print(f"  Testing: {len(test_data)} records from season {auto_test_season}")
    n_seasons = len(train_seasons)
    # Requirement-derived minimums: warn (or fail in strict mode) when below
    # (1w min 3, 4w min 5, 18w min 8)
    requirement_failures = []
    if n_seasons < MIN_TRAINING_SEASONS_1W:
        msg = f"1-week model requires >= {MIN_TRAINING_SEASONS_1W} training seasons (have {n_seasons})"
        print(f"  WARNING: {msg}. Accuracy may suffer.")
        requirement_failures.append(msg)
    if MODEL_CONFIG.get("use_18w_deep", True) and n_seasons < MIN_TRAINING_SEASONS_18W:
        msg = f"18-week deep model requires >= {MIN_TRAINING_SEASONS_18W} training seasons (have {n_seasons})"
        print(f"  WARNING: {msg}. Consider skipping or adding data.")
        requirement_failures.append(msg)
    if MODEL_CONFIG.get("use_4w_hybrid", True) and n_seasons < MIN_TRAINING_SEASONS_4W:
        msg = f"4-week hybrid model benefits from >= {MIN_TRAINING_SEASONS_4W} training seasons (have {n_seasons})"
        print(f"  WARNING: {msg}.")
        requirement_failures.append(msg)
    # Per-position player minimums (requirements: QB 30+, RB 60+, WR 70+, TE 30+)
    train_players_per_pos = train_data.groupby("position")["player_id"].nunique()
    for pos in POSITIONS:
        min_players = MIN_PLAYERS_PER_POSITION.get(pos, 30)
        n_players = int(train_players_per_pos.get(pos, 0))
        if n_players < min_players:
            msg = f"{pos} has {n_players} unique players in training (minimum >= {min_players})"
            print(f"  WARNING: {msg}.")
            requirement_failures.append(msg)
    if strict_requirements and requirement_failures:
        joined = "; ".join(requirement_failures)
        raise ValueError(f"Strict requirements check failed: {joined}")
    return train_data, test_data, train_seasons, auto_test_season


def create_sample_data() -> pd.DataFrame:
    """
    DEPRECATED: This function previously generated fake/synthetic data.

    This system now requires real NFL data from nfl-data-py.
    To load real data, run:
        python3 src/data/nfl_data_loader.py --seasons 2020-2024
    """
    from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
    raise NotImplementedError(
        "Synthetic data generation has been removed. "
        "This system only uses real NFL data from nfl-data-py. "
        f"Run: python3 src/data/nfl_data_loader.py (default: {MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON})"
    )
