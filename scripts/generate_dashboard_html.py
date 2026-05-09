#!/usr/bin/env python3
"""
Generate Draft Advisor HTML for GitHub Pages.

Reads model projections + ADP data, computes spread/VONA/VORP,
and produces a self-contained _site/index.html.

Usage:
    python scripts/generate_dashboard_html.py
    python scripts/generate_dashboard_html.py --season 2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.snake_draft_sim import (
    TEAMS,
    ROUNDS,
    build_draft_board,
    load_adp_board,
    load_model_projections,
    load_preseason_projections,
    _apply_vorp,
)
from scripts.draft_advisor import (
    compute_spread,
    compute_vona,
    validate_spread_direction,
    _latest_predictions_csv,
)

SITE_DIR = PROJECT_ROOT / "_site"
DRAFT_PICKS_PATH = PROJECT_ROOT / "data" / "draft_picks.parquet"

# PFR team codes → standard codes used in the rest of the system
PFR_TEAM_MAP = {
    "GNB": "GB", "KAN": "KC", "LVR": "LV", "NOR": "NO",
    "NWE": "NE", "SFO": "SF", "TAM": "TB",
}


def load_draft_class(season: int) -> pd.DataFrame:
    """Load draft picks for a season from parquet, normalize team codes."""
    if not DRAFT_PICKS_PATH.exists():
        return pd.DataFrame()
    dp = pd.read_parquet(DRAFT_PICKS_PATH)
    cls = dp[(dp["season"] == season) & (dp["position"].isin(["QB", "RB", "WR", "TE"]))].copy()
    if cls.empty:
        return cls
    cls["team"] = cls["team"].map(lambda t: PFR_TEAM_MAP.get(t, t))
    cls = cls.rename(columns={"pfr_player_name": "name"})
    return cls[["name", "position", "team", "round", "pick"]].reset_index(drop=True)


def build_rookie_projection_curve(seasons: range = range(2020, 2025)) -> dict:
    """Build position+round → expected season FP from historical data."""
    import sqlite3
    from config.settings import DB_PATH

    if not DRAFT_PICKS_PATH.exists():
        return {}

    dp = pd.read_parquet(DRAFT_PICKS_PATH)
    skill = dp[
        (dp["position"].isin(["QB", "RB", "WR", "TE"]))
        & (dp["season"].isin(seasons))
        & (dp["gsis_id"].notna())
    ]

    conn = sqlite3.connect(str(DB_PATH))
    curves: dict = {}  # (pos, round) -> avg_fp

    for pos in ["QB", "RB", "WR", "TE"]:
        for rnd in range(1, 8):
            picks = skill[(skill["position"] == pos) & (skill["round"] == rnd)]
            fps = []
            for _, r in picks.iterrows():
                row = conn.execute(
                    "SELECT SUM(fantasy_points) FROM player_weekly_stats "
                    "WHERE player_id=? AND season=?",
                    (r["gsis_id"], int(r["season"])),
                ).fetchone()
                if row[0] and row[0] > 0:
                    fps.append(row[0])
            curves[(pos, rnd)] = round(sum(fps) / len(fps), 1) if fps else 0

    conn.close()
    return curves


# ------------------------------------------------------------------
# Projection adjustments (age, team change, usage trend)
# ------------------------------------------------------------------

# Position-specific aging: (decline_start_age, pct_per_year)
AGE_CURVES = {
    "QB": (37, 0.03),
    "RB": (27, 0.05),
    "WR": (30, 0.04),
    "TE": (31, 0.03),
}

TEAM_CHANGE_PENALTY = {"QB": 0.10, "RB": 0.05, "WR": 0.12, "TE": 0.08}

# Regression to mean: how much of a career year is retained next season
# From 2020-2025 analysis of 102 career-year player-seasons
REGRESSION_RETAIN = {"QB": 0.833, "RB": 0.748, "WR": 0.763, "TE": 0.774}
CAREER_YEAR_THRESHOLD = 0.30  # 30% above career avg = career year

# Injury availability → games multiplier (from r=0.57 historical analysis)
# Key: (low_avail, high_avail) → multiplier vs baseline 13.85 GP
AVAILABILITY_CURVE = [
    (0.00, 0.40, 0.363),
    (0.40, 0.50, 0.534),
    (0.50, 0.60, 0.651),
    (0.60, 0.70, 0.721),
    (0.70, 0.80, 0.821),
    (0.80, 0.90, 0.935),
    (0.90, 1.01, 1.000),  # healthy baseline
]


def compute_age_adjustments(season: int) -> dict:
    """Return {normalized_name_key: {age, multiplier}} for players with birth dates."""
    try:
        import nfl_data_py as nfl
        rosters = nfl.import_seasonal_rosters([season - 1])
    except Exception:
        return {}

    if "birth_date" not in rosters.columns or "position" not in rosters.columns:
        return {}

    from datetime import datetime
    season_start = datetime(season, 9, 1)
    result = {}
    for _, r in rosters.iterrows():
        pos = r.get("position")
        if pos not in AGE_CURVES or pd.isna(r.get("birth_date")):
            continue
        try:
            bd = pd.to_datetime(r["birth_date"])
            age = (season_start - bd).days / 365.25
        except Exception:
            continue

        decline_start, rate = AGE_CURVES[pos]
        if age > decline_start:
            years_over = age - decline_start
            mult = max(0.5, 1.0 - rate * years_over)
        else:
            mult = 1.0

        name = r.get("player_name") or r.get("name") or ""
        if not name:
            continue
        result[name] = {"age": round(age, 1), "mult": round(mult, 3)}

    return result


def compute_team_changes(season: int) -> dict:
    """Return {player_name: {from_team, to_team, multiplier}} for players who changed teams."""
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    prior2 = season - 2
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT p.name, p.position, a.team AS team_old, b.team AS team_new
        FROM (
            SELECT player_id, team, COUNT(*) as gp
            FROM player_weekly_stats WHERE season = ?
            GROUP BY player_id
            HAVING gp >= 4
        ) a
        JOIN (
            SELECT player_id, team, COUNT(*) as gp
            FROM player_weekly_stats WHERE season = ?
            GROUP BY player_id
            HAVING gp >= 4
        ) b ON a.player_id = b.player_id
        JOIN players p ON a.player_id = p.player_id
        WHERE a.team != b.team
          AND a.team != '' AND b.team != ''
          AND p.position IN ('QB','RB','WR','TE')
    """, (prior2, prior)).fetchall()

    conn.close()

    result = {}
    for name, pos, old_team, new_team in rows:
        penalty = TEAM_CHANGE_PENALTY.get(pos, 0.05)
        result[name] = {
            "from": old_team,
            "to": new_team,
            "mult": round(1.0 - penalty, 2),
        }
    return result


def compute_usage_trends(season: int) -> dict:
    """Return {player_name: {early, late, slope, multiplier}} based on late-season usage."""
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT p.name, p.position,
               AVG(CASE WHEN pws.week <= 9 THEN pws.targets END) as early_tgt,
               AVG(CASE WHEN pws.week > 9 THEN pws.targets END) as late_tgt,
               AVG(CASE WHEN pws.week <= 9 THEN pws.rushing_attempts END) as early_car,
               AVG(CASE WHEN pws.week > 9 THEN pws.rushing_attempts END) as late_car
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season = ? AND p.position IN ('QB','RB','WR','TE')
        GROUP BY pws.player_id
        HAVING COUNT(*) >= 8
    """, (prior,)).fetchall()

    conn.close()

    result = {}
    for name, pos, early_tgt, late_tgt, early_car, late_car in rows:
        # RBs use carries, others use targets
        if pos == "RB":
            early = early_car or 0
            late = late_car or 0
            metric = "carries"
        else:
            early = early_tgt or 0
            late = late_tgt or 0
            metric = "targets"

        slope = late - early

        if slope >= 2.0:
            mult = 1.05
        elif slope <= -2.0:
            mult = 0.92
        else:
            mult = 1.0

        if mult != 1.0:
            result[name] = {
                "early": round(early, 1),
                "late": round(late, 1),
                "slope": round(slope, 1),
                "metric": metric,
                "mult": mult,
            }
    return result


def compute_injury_risk(season: int) -> dict:
    """Discount projections based on 3-year games-played availability.

    Returns {player_name: {avail_rate, expected_gp, mult}}.
    """
    import sqlite3
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    prior_seasons = (season - 3, season - 2, season - 1)

    rows = conn.execute("""
        SELECT p.name, p.position,
               COUNT(DISTINCT pws.season || ':' || pws.week) as total_games,
               COUNT(DISTINCT pws.season) as seasons_active
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season IN (?, ?, ?)
          AND p.position IN ('QB','RB','WR','TE')
          AND pws.fantasy_points > 0
        GROUP BY pws.player_id
        HAVING seasons_active >= 2
    """, prior_seasons).fetchall()

    conn.close()

    result = {}
    for name, pos, total_games, seasons_active in rows:
        possible_games = seasons_active * 17
        avail_rate = total_games / possible_games if possible_games > 0 else 1.0

        # Look up multiplier from availability curve
        mult = 1.0
        for low, high, m in AVAILABILITY_CURVE:
            if low <= avail_rate < high:
                mult = m
                break

        if mult >= 0.99:
            continue  # healthy player, no discount

        # Cap at 0.65 — don't discount more than 35% for injury alone
        mult = max(0.65, mult)
        expected_gp = round(mult * 17, 1)
        result[name] = {
            "avail_rate": round(avail_rate, 2),
            "expected_gp": expected_gp,
            "mult": round(mult, 3),
        }

    return result


def compute_breakout_candidates(season: int) -> dict:
    """Identify players with efficiency + volume + snap share momentum.

    Only flags players where all three signals align (reduces false positives).
    Returns {player_name: {eff_change, vol_change, mult}}.
    """
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    prior2 = season - 2
    conn = sqlite3.connect(str(DB_PATH))

    # Per-touch efficiency and volume for two seasons
    rows = conn.execute("""
        SELECT p.name, p.position, pws.season,
               SUM(pws.targets) + SUM(pws.rushing_attempts) as touches,
               SUM(pws.fantasy_points) as total_fp,
               COUNT(DISTINCT pws.week) as games,
               AVG(CASE WHEN pws.week > 9 THEN pws.snap_share END) as late_snap
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season IN (?, ?)
          AND p.position IN ('RB','WR','TE')
          AND pws.fantasy_points > 0
        GROUP BY pws.player_id, pws.season
        HAVING games >= 8 AND touches >= 40
    """, (prior2, prior)).fetchall()

    conn.close()

    # Group by player
    from collections import defaultdict
    player_data = defaultdict(dict)
    for name, pos, yr, touches, fp, games, late_snap in rows:
        player_data[name][yr] = {
            "pos": pos,
            "touches": touches,
            "fp": fp,
            "eff": fp / touches if touches > 0 else 0,
            "games": games,
            "late_snap": late_snap or 0,
        }

    result = {}
    for name, seasons in player_data.items():
        if prior not in seasons or prior2 not in seasons:
            continue

        cur = seasons[prior]
        prev = seasons[prior2]

        if prev["eff"] <= 0:
            continue

        eff_change = (cur["eff"] - prev["eff"]) / prev["eff"]
        vol_change = (cur["touches"] - prev["touches"]) / prev["touches"] if prev["touches"] > 0 else 0

        # Both must align:
        # 1. Efficiency improved 15%+
        # 2. Volume stable or growing (>= -10%)
        if eff_change >= 0.15 and vol_change >= -0.10:
            # Conservative boost — only +5% since 75% of efficiency jumps revert
            # But combined signals have higher persistence (~45%)
            mult = 1.05
            result[name] = {
                "eff_change": round(eff_change * 100),
                "vol_change": round(vol_change * 100),
                "mult": mult,
            }

    return result


def compute_regression_adjustments(season: int) -> dict:
    """Identify career-year players who should regress.

    Compares most recent season PPG to 3-year career avg.
    If recent season was CAREER_YEAR_THRESHOLD above career avg,
    apply position-specific regression multiplier.

    Returns {player_name: {career_ppg, recent_ppg, pct_above, mult}}.
    """
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    # Get per-season PPG for each player over the last 4 seasons
    rows = conn.execute("""
        SELECT p.name, p.position, pws.season,
               AVG(pws.fantasy_points) as ppg,
               COUNT(DISTINCT pws.week) as games
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season BETWEEN ? AND ?
          AND p.position IN ('QB','RB','WR','TE')
          AND pws.fantasy_points > 0
        GROUP BY pws.player_id, pws.season
        HAVING games >= 6
    """, (prior - 3, prior)).fetchall()

    conn.close()

    # Group by player
    from collections import defaultdict
    player_seasons = defaultdict(list)
    player_pos = {}
    for name, pos, season_yr, ppg, games in rows:
        player_seasons[name].append((season_yr, ppg, games))
        player_pos[name] = pos

    result = {}
    for name, seasons_data in player_seasons.items():
        if len(seasons_data) < 2:
            continue

        pos = player_pos[name]
        if pos not in REGRESSION_RETAIN:
            continue

        # Most recent season
        seasons_data.sort(key=lambda x: x[0])
        recent_season, recent_ppg, _ = seasons_data[-1]
        if recent_season != prior:
            continue  # player didn't play last season

        # Career avg from prior seasons (excluding most recent)
        prior_ppgs = [ppg for yr, ppg, _ in seasons_data if yr < recent_season]
        if not prior_ppgs:
            continue
        career_ppg = sum(prior_ppgs) / len(prior_ppgs)

        if career_ppg <= 0:
            continue

        pct_above = (recent_ppg - career_ppg) / career_ppg
        if pct_above < CAREER_YEAR_THRESHOLD:
            continue

        # Apply regression: project next year as blend between
        # career_year * retain_rate and career_avg * 1.17
        retain = REGRESSION_RETAIN[pos]
        # The multiplier represents how much to discount the raw projection
        # Raw proj is based on recent season. Regression pulls it toward career avg.
        mult = retain + (1.0 - retain) * (career_ppg / recent_ppg)
        mult = max(0.6, min(1.0, mult))

        result[name] = {
            "career_ppg": round(career_ppg, 1),
            "recent_ppg": round(recent_ppg, 1),
            "pct_above": round(pct_above * 100),
            "mult": round(mult, 3),
        }

    return result


def compute_defense_rankings(season: int) -> dict:
    """Compute FP allowed per game by each defense, per position.

    Returns {team: {pos: rank}} where rank 1 = easiest matchup (most FP allowed).
    Uses schedule join since opponent field is empty in player_weekly_stats.
    """
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    rankings = {}  # {pos: {defense_team: avg_fp_allowed}}
    for pos in ["QB", "RB", "WR", "TE"]:
        # FP allowed when team is away defense (opponent scores at home)
        away = conn.execute("""
            SELECT s.away_team as defense, AVG(pws.fantasy_points) as fp
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            JOIN schedule s ON pws.season = s.season AND pws.week = s.week
                AND pws.team = s.home_team
            WHERE pws.season = ? AND p.position = ?
            GROUP BY s.away_team
        """, (prior, pos)).fetchall()

        # FP allowed when team is home defense
        home = conn.execute("""
            SELECT s.home_team as defense, AVG(pws.fantasy_points) as fp
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            JOIN schedule s ON pws.season = s.season AND pws.week = s.week
                AND pws.team = s.away_team
            WHERE pws.season = ? AND p.position = ?
            GROUP BY s.home_team
        """, (prior, pos)).fetchall()

        # Combine home + away
        from collections import defaultdict
        totals = defaultdict(list)
        for team, fp in away + home:
            if team:
                totals[team].append(fp)

        avgs = {team: sum(fps) / len(fps) for team, fps in totals.items() if fps}
        # Rank: 1 = most FP allowed (easiest), 32 = least (hardest)
        sorted_teams = sorted(avgs.items(), key=lambda x: -x[1])
        rankings[pos] = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

    conn.close()

    # Build per-team summary: {team: {QB: rank, RB: rank, ...}}
    all_teams = set()
    for pos_ranks in rankings.values():
        all_teams.update(pos_ranks.keys())

    result = {}
    for team in all_teams:
        result[team] = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            result[team][pos] = rankings[pos].get(team, 16)

    return result


# NFL divisions (fixed — each team plays 6 games vs division rivals)
NFL_DIVISIONS = {
    "BUF": ["MIA", "NE", "NYJ"], "MIA": ["BUF", "NE", "NYJ"],
    "NE": ["BUF", "MIA", "NYJ"], "NYJ": ["BUF", "MIA", "NE"],
    "BAL": ["CIN", "CLE", "PIT"], "CIN": ["BAL", "CLE", "PIT"],
    "CLE": ["BAL", "CIN", "PIT"], "PIT": ["BAL", "CIN", "CLE"],
    "HOU": ["IND", "JAX", "TEN"], "IND": ["HOU", "JAX", "TEN"],
    "JAX": ["HOU", "IND", "TEN"], "TEN": ["HOU", "IND", "JAX"],
    "DEN": ["KC", "LAC", "LV"], "KC": ["DEN", "LAC", "LV"],
    "LAC": ["DEN", "KC", "LV"], "LV": ["DEN", "KC", "LAC"],
    "DAL": ["NYG", "PHI", "WAS"], "NYG": ["DAL", "PHI", "WAS"],
    "PHI": ["DAL", "NYG", "WAS"], "WAS": ["DAL", "NYG", "PHI"],
    "CHI": ["DET", "GB", "MIN"], "DET": ["CHI", "GB", "MIN"],
    "GB": ["CHI", "DET", "MIN"], "MIN": ["CHI", "DET", "GB"],
    "ATL": ["CAR", "NO", "TB"], "CAR": ["ATL", "NO", "TB"],
    "NO": ["ATL", "CAR", "TB"], "TB": ["ATL", "CAR", "NO"],
    "ARI": ["LAR", "SEA", "SF"], "LAR": ["ARI", "SEA", "SF"],
    "SEA": ["ARI", "LAR", "SF"], "SF": ["ARI", "LAR", "SEA"],
}


def compute_sos_adjustment(
    player_team: str, position: str, defense_rankings: dict
) -> tuple[float, str]:
    """Compute SOS multiplier from division rival defense quality.

    6 of 17 games are against division rivals (known before schedule).
    Average their defense rank at this position → SOS proxy.
    """
    rivals = NFL_DIVISIONS.get(player_team, [])
    if not rivals or not defense_rankings:
        return 1.0, ""

    rival_ranks = []
    for rival in rivals:
        rank = defense_rankings.get(rival, {}).get(position)
        if rank is not None:
            rival_ranks.append(rank)

    if not rival_ranks:
        return 1.0, ""

    avg_rank = sum(rival_ranks) / len(rival_ranks)
    # avg_rank 1-10 = easy division, 11-22 = neutral, 23-32 = tough
    if avg_rank <= 10:
        mult = 1.03
        label = "Easy div"
    elif avg_rank >= 23:
        mult = 0.97
        label = "Hard div"
    else:
        return 1.0, ""

    return mult, label


def check_data_sources(season: int) -> list[dict]:
    """Check which data sources are available for a season.

    Returns a list of {name, status, detail} dicts where status is
    'available', 'unavailable', or 'partial'.
    """
    import sqlite3
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    sources = []

    def _count(query, params=()):
        return conn.execute(query, params).fetchone()[0]

    # 1. Prior season stats (basis of preseason projections)
    prior = season - 1
    prior_stats = _count(
        "SELECT COUNT(*) FROM player_weekly_stats WHERE season=?", (prior,)
    )
    sources.append({
        "name": f"{prior} Player Stats",
        "status": "available" if prior_stats > 500 else "unavailable",
        "detail": f"{prior_stats:,} player-weeks" if prior_stats else "No data",
    })

    # 2. ADP / Expert Consensus Rankings
    adp_count = _count(
        "SELECT COUNT(*) FROM adp_history WHERE season=?", (season,)
    )
    sources.append({
        "name": "ADP / Expert Rankings",
        "status": "available" if adp_count > 50 else "unavailable",
        "detail": f"{adp_count:,} rankings" if adp_count else "Not yet scraped",
    })

    # 3. NFL Schedule
    sched_count = _count(
        "SELECT COUNT(*) FROM schedule WHERE season=?", (season,)
    )
    sources.append({
        "name": f"{season} NFL Schedule",
        "status": "available" if sched_count >= 256 else "unavailable",
        "detail": f"{sched_count} games" if sched_count else "Released ~May",
    })

    # 4. Vegas lines (spreads + totals)
    vegas_count = _count(
        "SELECT COUNT(*) FROM schedule WHERE season=? AND spread_line IS NOT NULL",
        (season,),
    )
    sources.append({
        "name": "Vegas Lines",
        "status": "available" if vegas_count >= 256 else (
            "partial" if vegas_count > 0 else "unavailable"
        ),
        "detail": f"{vegas_count} games" if vegas_count else "Available ~August",
    })

    # 5. Current season game stats
    curr_stats = _count(
        "SELECT COUNT(*) FROM player_weekly_stats WHERE season=?", (season,)
    )
    sources.append({
        "name": f"{season} Game Stats",
        "status": "available" if curr_stats > 500 else (
            "partial" if curr_stats > 0 else "unavailable"
        ),
        "detail": f"{curr_stats:,} player-weeks" if curr_stats else "Season hasn't started",
    })

    # 6. Rookie / NFL Draft data (from parquet, not DB)
    draft_class = load_draft_class(season)
    rookie_count = len(draft_class)
    sources.append({
        "name": f"{season} NFL Draft Picks",
        "status": "available" if rookie_count > 0 else "unavailable",
        "detail": f"{rookie_count} skill picks" if rookie_count else "Available after NFL Draft (~April)",
    })

    # 7. NGS (Next Gen Stats)
    try:
        ngs_count = _count(
            "SELECT COUNT(*) FROM ngs_passing WHERE season=?", (season,)
        )
    except sqlite3.OperationalError:
        ngs_count = 0
    sources.append({
        "name": "Next Gen Stats",
        "status": "available" if ngs_count > 100 else "unavailable",
        "detail": f"{ngs_count:,} records" if ngs_count else "In-season only (2018+)",
    })

    # 8. Injury reports
    try:
        inj_count = _count(
            "SELECT COUNT(*) FROM injuries WHERE season=?", (season,)
        )
    except sqlite3.OperationalError:
        inj_count = 0
    sources.append({
        "name": "Injury Reports",
        "status": "available" if inj_count > 50 else "unavailable",
        "detail": f"{inj_count:,} reports" if inj_count else "In-season only",
    })

    # 9. Walk-forward backtest predictions
    csv = _latest_predictions_csv(season)
    sources.append({
        "name": "ML Backtest Predictions",
        "status": "available" if csv else "unavailable",
        "detail": str(csv.name) if csv else "Requires full season of data",
    })

    conn.close()
    return sources


def build_team_tendencies(season: int) -> dict:
    """Build team-level rushing/passing tendency vectors from prior season."""
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT team,
               SUM(rushing_attempts) as rush_att,
               SUM(passing_attempts) as pass_att,
               SUM(rushing_yards) as rush_yd,
               SUM(passing_yards) as pass_yd,
               SUM(targets) as total_targets
        FROM player_weekly_stats
        WHERE season = ? AND team IS NOT NULL AND team != ''
        GROUP BY team
    """, (prior,)).fetchall()

    teams = {}
    for team, rush_att, pass_att, rush_yd, pass_yd, total_tgt in rows:
        rush_att = rush_att or 0
        pass_att = pass_att or 0
        total = rush_att + pass_att
        if total == 0:
            continue
        teams[team] = {
            "rush_pct": round(rush_att / total * 100),
            "pass_pct": round(pass_att / total * 100),
            "rush_yd": round(rush_yd or 0),
            "pass_yd": round(pass_yd or 0),
            "rush_att": round(rush_att),
            "pass_att": round(pass_att),
        }

    # Classify tendency
    for t in teams.values():
        if t["rush_pct"] >= 45:
            t["tendency"] = "Run-heavy"
        elif t["rush_pct"] >= 40:
            t["tendency"] = "Balanced"
        elif t["rush_pct"] >= 35:
            t["tendency"] = "Pass-lean"
        else:
            t["tendency"] = "Pass-heavy"

    conn.close()
    return teams


def build_usage_roles(season: int) -> dict:
    """Build player roles from actual usage data + rookie draft capital."""
    import sqlite3
    from config.settings import DB_PATH
    from collections import defaultdict

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT pws.team, p.name, p.position,
               SUM(pws.targets) as total_targets,
               SUM(pws.rushing_attempts) as total_carries,
               SUM(pws.snap_count) as total_snaps,
               SUM(pws.receptions) as total_rec,
               COUNT(DISTINCT pws.week) as games
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season = ?
          AND p.position IN ('QB', 'RB', 'WR', 'TE')
          AND pws.team IS NOT NULL AND pws.team != ''
        GROUP BY pws.team, pws.player_id
        HAVING games >= 4
        ORDER BY pws.team, total_snaps DESC
    """, (prior,)).fetchall()

    conn.close()

    team_pos = defaultdict(lambda: defaultdict(list))
    for team, name, pos, tgt, carries, snaps, rec, games in rows:
        team_pos[team][pos].append({
            "name": name,
            "targets": tgt or 0,
            "carries": carries or 0,
            "snaps": snaps or 0,
            "receptions": rec or 0,
            "games": games,
            "is_rookie": False,
            "draft_round": 0,
        })

    # Inject rookies from draft class into their teams
    draft_class = load_draft_class(season)
    rookie_curve = build_rookie_projection_curve()
    for _, rk in draft_class.iterrows():
        team = rk["team"]
        pos = rk["position"]
        rnd = int(rk["round"])
        # Estimate usage from draft capital — Round 1 gets starter-level usage
        est_fp = rookie_curve.get((pos, rnd), 50)
        team_pos[team][pos].append({
            "name": rk["name"],
            "targets": est_fp * 0.3 if pos != "RB" else est_fp * 0.1,
            "carries": est_fp * 0.5 if pos == "RB" else 0,
            "snaps": est_fp * 10,
            "receptions": 0,
            "games": 0,
            "is_rookie": True,
            "draft_round": rnd,
        })

    # Rank within team+position by primary usage metric
    roles = {}
    for team, positions in team_pos.items():
        team_targets = sum(p["targets"] for pos_list in positions.values() for p in pos_list)
        team_carries = sum(p["carries"] for pos_list in positions.values() for p in pos_list)

        for pos, players in positions.items():
            if pos == "RB":
                players.sort(key=lambda x: -(x["carries"] + x["targets"]))
            else:
                players.sort(key=lambda x: -(x["targets"] + x["snaps"]))

            for rank, p in enumerate(players, 1):
                tgt_share = round(p["targets"] / team_targets * 100) if team_targets else 0
                carry_share = round(p["carries"] / team_carries * 100) if team_carries else 0

                if p["is_rookie"]:
                    note = f"Rookie R{p['draft_round']}"
                elif pos == "RB":
                    if carry_share >= 60:
                        note = "Bellcow"
                    elif carry_share >= 35:
                        note = "Lead back"
                    elif carry_share >= 20:
                        note = "Committee"
                    else:
                        note = "Backup"
                elif pos in ("WR", "TE"):
                    if tgt_share >= 20:
                        note = "Alpha"
                    elif tgt_share >= 12:
                        note = "Starter"
                    elif tgt_share >= 6:
                        note = "Rotational"
                    else:
                        note = "Depth"
                else:
                    note = "Starter" if rank == 1 else "Backup"

                roles[p["name"]] = {
                    "role": f"{pos}{rank}",
                    "tgt_share": tgt_share,
                    "carry_share": carry_share,
                    "usage": note,
                }

    return roles


def build_board_data(season: int):
    """Build the full draft board with spread, VORP, and projections."""
    adp_df = load_adp_board(season)

    csv = _latest_predictions_csv(season)
    if csv:
        projections = load_model_projections(csv, ranking="season_sum", season=season)
    else:
        # No backtest for this season yet — use prior season's ML predictions
        # as the projection basis (much better than raw PPG * 17),
        # then fill in rookies/unmatched with ECR-implied projections
        prior_csv = _latest_predictions_csv(season - 1)
        if prior_csv:
            ml_proj = load_model_projections(
                prior_csv, ranking="season_sum", season=season - 1
            )
            ml_proj["actual_total"] = 0.0
            # Merge with preseason projections for rookies
            fallback = load_preseason_projections(season, adp_df=adp_df)
            if not fallback.empty:
                # Keep ML projections, add rookies not in ML set
                ml_names = set(ml_proj["name"].tolist())
                rookies = fallback[~fallback["name"].isin(ml_names)]
                projections = pd.concat([ml_proj, rookies], ignore_index=True)
            else:
                projections = ml_proj
        else:
            projections = load_preseason_projections(season, adp_df=adp_df)

    board = build_draft_board(adp_df, projections)
    spread_results = compute_spread(board)
    validation = validate_spread_direction(spread_results, min_spread=10)

    # Build VORP values
    if not projections.empty:
        vorp_series = _apply_vorp(projections, basis_col="pred_total")
        vorp_map = dict(zip(projections["name"], vorp_series))
    else:
        vorp_map = {}

    has_actuals = csv is not None

    # Real usage-based roles from prior season play-by-play
    raw_usage_roles = build_usage_roles(season)
    team_tendencies = build_team_tendencies(season)

    # Build normalized lookup for usage roles (first_initial, last_name, pos) -> role
    # because DB names are "T.McBride" but ADP names are "Trey McBride"
    SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "jr."}

    def _norm_key(name, pos=None):
        name = name.strip()
        if "." in name and " " not in name:
            # Abbreviated: "T.McBride"
            parts = name.split(".")
            initial = parts[0].strip()[0].lower() if parts[0].strip() else ""
            last = parts[-1].strip().lower()
        else:
            # Full: "James Cook III" or "Travis Etienne Jr."
            parts = name.split()
            initial = parts[0][0].lower() if parts else ""
            # Walk backwards past suffixes to find the real last name
            idx = len(parts) - 1
            while idx > 0 and parts[idx].lower().rstrip(".") in SUFFIXES:
                idx -= 1
            last = parts[idx].lower() if idx >= 0 else ""
        return (initial, "".join(c for c in last if c.isalnum()))

    usage_roles = {}  # normalized key -> role data
    for name, data in raw_usage_roles.items():
        pos = data["role"][:2]
        key = _norm_key(name, pos)
        # Keep the one with higher usage
        existing = usage_roles.get(key)
        if existing is None or (data["tgt_share"] + data["carry_share"]) > (existing["tgt_share"] + existing["carry_share"]):
            usage_roles[key] = data

    # Fallback: projection-based role for players with no prior-season data
    team_groups = {}
    for sr in spread_results:
        if sr.ecr > 300:
            continue
        team_groups.setdefault(sr.team, []).append(sr)
    proj_role_map = {}
    for team, group in team_groups.items():
        team_pos = {}
        for sr in group:
            team_pos.setdefault(sr.position, []).append(sr)
        for pos, pos_players in team_pos.items():
            pos_players.sort(key=lambda s: -s.model_projection)
            for rank, sr in enumerate(pos_players, 1):
                proj_role_map[sr.name] = f"{pos}{rank}"

    # Compute projection adjustments
    print("  Computing adjustments (age, team changes, usage trends, SOS)...")
    age_adj = compute_age_adjustments(season)
    team_change_adj = compute_team_changes(season)
    trend_adj = compute_usage_trends(season)
    regression_adj = compute_regression_adjustments(season)
    injury_adj = compute_injury_risk(season)
    breakout_adj = compute_breakout_candidates(season)
    def_rankings = compute_defense_rankings(season)

    # Serialize board for JS
    players = []
    for i, sr in enumerate(spread_results):
        if sr.ecr > 300:
            continue
        # Use real usage role if available, else projection-based
        ur = usage_roles.get(_norm_key(sr.name, sr.position))
        if ur:
            team_role = ur["role"]
            usage_note = ur["usage"]
            tgt_share = ur["tgt_share"]
            carry_share = ur["carry_share"]
        else:
            team_role = proj_role_map.get(sr.name, "")
            usage_note = "Rookie" if "rookie_" in str(sr.name).lower() else "Proj"
            tgt_share = 0
            carry_share = 0

        # Team tendency
        tt = team_tendencies.get(sr.team, {})

        # Collect context signals (informational — shown to user but most
        # don't modify the projection since the model + ADP blend already
        # prices them in. Only age curve applies as a multiplier since
        # ablation shows it adds +1pp accuracy).
        raw_proj = sr.blended_projection
        adj_mult = 1.0
        adj_reasons = []
        player_age = None

        # Age — only adjustment that improves accuracy on top of blend
        aa = age_adj.get(sr.name)
        if aa:
            player_age = aa["age"]
            if aa["mult"] < 1.0:
                adj_mult *= aa["mult"]
                adj_reasons.append(f"Age {aa['age']:.0f}")

        # Context signals (informational only — no projection change)
        for tc_name, tc in team_change_adj.items():
            if _norm_key(tc_name, sr.position) == _norm_key(sr.name, sr.position):
                adj_reasons.append("New team")
                break

        for tr_name, tr in trend_adj.items():
            if _norm_key(tr_name, sr.position) == _norm_key(sr.name, sr.position):
                if tr["slope"] > 0:
                    adj_reasons.append(f"Usage +{tr['slope']:.0f}")
                else:
                    adj_reasons.append(f"Usage {tr['slope']:.0f}")
                break

        ra = None
        for rname, rdata in regression_adj.items():
            if _norm_key(rname) == _norm_key(sr.name):
                ra = rdata
                break
        if ra:
            adj_reasons.append(f"Regress {ra['pct_above']}%yr")

        ia = None
        for iname, idata in injury_adj.items():
            if _norm_key(iname) == _norm_key(sr.name):
                ia = idata
                break
        if ia:
            adj_reasons.append(f"Inj {ia['expected_gp']:.0f}g")

        ba = None
        for bname, bdata in breakout_adj.items():
            if _norm_key(bname) == _norm_key(sr.name):
                ba = bdata
                break
        if ba:
            adj_reasons.append(f"Breakout +{ba['eff_change']}%eff")

        _, sos_label = compute_sos_adjustment(
            sr.team, sr.position, def_rankings
        )
        if sos_label:
            adj_reasons.append(sos_label)

        adjusted_proj = round(raw_proj * adj_mult, 1)
        adj_pct = round((adj_mult - 1.0) * 100)

        players.append({
            "id": i,
            "n": sr.name,
            "p": sr.position,
            "t": sr.team,
            "ecr": round(sr.ecr, 1),
            "mr": sr.model_rank,
            "sp": sr.rank_spread,
            "proj": adjusted_proj,
            "rawProj": round(raw_proj, 1),
            "adjPct": adj_pct,
            "adjR": ", ".join(adj_reasons) if adj_reasons else "",
            "age": player_age,
            "vorp": round(vorp_map.get(sr.name, 0), 1),
            "role": team_role,
            "usage": usage_note,
            "ts": tgt_share,
            "cs": carry_share,
            "tt": tt.get("tendency", ""),
            "trp": tt.get("rush_pct", 0),
            "tpp": tt.get("pass_pct", 0),
            "act": round(sr.actual_total, 1) if has_actuals else None,
            "w": sr.model_wins if has_actuals else None,
        })

    return {
        "players": players,
        "validation": {
            "n": validation["n"],
            "wins": validation.get("model_wins", 0),
            "acc": round(validation["accuracy"] * 100) if validation["n"] > 0 else 0,
        },
        "has_actuals": has_actuals,
        "season": season,
        "board": board,
        "adp_df": adp_df,
    }


def build_vona_data(board, adp_df, max_slots=14):
    """Pre-compute VONA for each draft slot."""
    vona_all = {}
    for slot in range(1, min(max_slots + 1, TEAMS + 1)):
        raw = compute_vona(board, adp_df, slot, teams=TEAMS, rounds=ROUNDS)
        # Group by round, keep top 5 per round
        by_round = {}
        for r in raw:
            rd = r["round"]
            if rd not in by_round:
                by_round[rd] = []
            by_round[rd].append(r)

        slot_picks = []
        for rd in sorted(by_round.keys()):
            candidates = sorted(by_round[rd], key=lambda x: -x["net_value"])[:5]
            for c in candidates:
                slot_picks.append({
                    "rd": rd,
                    "pk": c["pick"],
                    "n": c["name"],
                    "p": c["position"],
                    "t": c.get("team", ""),
                    "av": round(c["avail_pct"] * 100),
                    "proj": round(c["model_proj"], 1),
                    "vona": round(c["vona"], 1),
                    "oc": round(c["opp_cost"], 1),
                    "ocp": c["opp_cost_pos"],
                    "net": round(c["net_value"], 1),
                })
        vona_all[slot] = slot_picks
    return vona_all


def generate_html(board_data, vona_data, data_sources):
    """Generate the complete HTML page."""
    season = board_data["season"]
    players_json = json.dumps(board_data["players"], separators=(",", ":"))
    vona_json = json.dumps(vona_data, separators=(",", ":"))
    meta_json = json.dumps({
        "season": season,
        "validation": board_data["validation"],
        "has_actuals": board_data["has_actuals"],
        "teams": TEAMS,
        "rounds": ROUNDS,
        "sources": data_sources,
    }, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Draft Advisor {season}</title>
<style>
*,*::before,*::after{{box-sizing:border-box}}
body{{
  margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#f8f9fa;color:#1a1a2e;font-size:14px;line-height:1.5;
}}
.header{{
  background:#1a1a2e;color:#fff;padding:16px 20px;text-align:center;
  position:sticky;top:0;z-index:100;
}}
.header h1{{margin:0;font-size:1.2rem;font-weight:700;letter-spacing:0.5px}}
.header .sub{{font-size:0.8rem;opacity:0.6;margin-top:2px}}
.main{{max-width:900px;margin:0 auto;padding:16px}}
.card{{
  background:#fff;border-radius:12px;padding:16px;
  margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.06);
}}
.banner{{
  text-align:center;padding:14px;border-radius:12px;margin-bottom:12px;
}}
.banner-good{{background:#e8f5e9;border:1px solid #c8e6c9}}
.banner-warn{{background:#fff8e1;border:1px solid #fff3c4}}
.pills{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}}
.pill{{
  padding:8px 18px;border-radius:20px;border:1.5px solid #ddd;
  background:#fff;color:#555;font-size:0.85rem;font-weight:500;
  cursor:pointer;transition:0.15s;
}}
.pill:hover{{border-color:#4fc3f7;color:#1a1a2e}}
.pill.active{{background:#4fc3f7;color:#fff;border-color:#4fc3f7}}
table{{width:100%;border-collapse:collapse;font-size:0.85rem}}
th{{
  text-align:left;padding:10px 8px;border-bottom:2px solid #e0e0e0;
  font-weight:600;color:#666;font-size:0.75rem;text-transform:uppercase;
  letter-spacing:0.3px;cursor:pointer;user-select:none;
  position:sticky;top:0;background:#fff;z-index:1;
}}
th:hover{{color:#1a1a2e}}
th.num,td.num{{text-align:right}}
td{{padding:8px;border-bottom:1px solid #f0f0f0}}
tr:hover td{{background:#f8f9ff}}
tr.sleeper td{{background:#f9fbe7}}
tr.fade td{{background:#fef3f2}}
tr.fade-strong td{{background:#ffebee}}
.signal{{
  display:inline-block;padding:1px 6px;border-radius:3px;
  font-size:0.65rem;font-weight:600;margin-left:4px;
}}
.signal-fade-strong{{background:#c62828;color:#fff}}
.signal-fade{{background:#ffcdd2;color:#c62828}}
.signal-sleeper{{background:#e8f5e9;color:#2e7d32}}
.signal-sleeper-mild{{background:#f1f8e9;color:#558b2f}}
.pos{{
  display:inline-block;padding:2px 8px;border-radius:4px;
  font-size:0.75rem;font-weight:600;color:#fff;
}}
.pos-QB{{background:#e53935}}.pos-RB{{background:#43a047}}
.pos-WR{{background:#1e88e5}}.pos-TE{{background:#f9a825;color:#333}}
.spread-pos{{color:#2e7d32;font-weight:600}}
.spread-neg{{color:#c62828;font-weight:600}}
.role-badge{{font-weight:600;font-size:0.8rem}}
.usage-tag{{
  display:inline-block;padding:1px 6px;border-radius:3px;
  font-size:0.65rem;font-weight:500;margin-left:4px;
}}
.usage-elite{{background:#e8f5e9;color:#2e7d32}}
.usage-solid{{background:#e3f2fd;color:#1565c0}}
.usage-other{{background:#f5f5f5;color:#888}}
.tend-tag{{
  display:inline-block;padding:1px 5px;border-radius:3px;
  font-size:0.6rem;font-weight:500;margin-left:2px;
}}
.tend-run{{background:#fff3e0;color:#e65100}}
.tend-pass{{background:#e8eaf6;color:#283593}}
.tend-bal{{background:#f5f5f5;color:#666}}
.search-box{{
  width:100%;padding:10px 14px;border:1.5px solid #ddd;border-radius:8px;
  font-size:0.9rem;margin-bottom:12px;outline:none;
}}
.search-box:focus{{border-color:#4fc3f7}}
.footer{{text-align:center;padding:24px 16px;color:#aaa;font-size:0.7rem}}
@media(max-width:600px){{
  table{{font-size:0.8rem}}
  th,td{{padding:6px 4px}}
}}
</style>
</head>
<body>
<div class="header">
  <h1>Draft Advisor {season}</h1>
  <div class="sub">Model-powered fantasy draft rankings</div>
</div>
<div class="main" id="main"></div>
<div class="footer">Projections generated from walk-forward ML models trained on 2006&ndash;{season - 1} NFL data. Not financial advice.</div>

<script>
const BOARD={players_json};
const VONA={vona_json};
const META={meta_json};

let posFilter="All";
let sortCol="mr";
let sortAsc=true;
let searchQ="";

// Pills
function renderPills(){{
  const positions=["All","QB","RB","WR","TE"];
  return `<div class="pills">${{positions.map(pos=>{{
    const signals=pos==="All"
      ?BOARD.filter(p=>Math.abs(p.sp)>=10).length
      :BOARD.filter(p=>p.p===pos&&Math.abs(p.sp)>=10).length;
    const label=signals
      ?`${{pos}} <span style="font-size:0.75rem;opacity:0.75">(${{signals}})</span>`
      :pos;
    return`<button class="pill ${{posFilter===pos?"active":""}}" onclick="posFilter='${{pos}}';render()">${{label}}</button>`;
  }}).join("")}}</div>`;
}}

// Sort
function setSort(col){{
  if(sortCol===col)sortAsc=!sortAsc;
  else{{sortCol=col;sortAsc=col==="mr"||col==="ecr";}}
  renderTable();
}}

function sorted(arr){{
  return[...arr].sort((a,b)=>{{
    let va=a[sortCol],vb=b[sortCol];
    if(va==null)return 1;if(vb==null)return-1;
    return sortAsc?va-vb:vb-va;
  }});
}}

function filterPos(arr){{
  if(posFilter==="All")return arr;
  return arr.filter(p=>p.p===posFilter);
}}

// Rankings view
function renderRankings(){{
  let h="";

  // Value signal stats strip
  const v=META.validation;
  const fadesAll=BOARD.filter(p=>p.sp<=-10).length;
  const fadesStrong=BOARD.filter(p=>p.sp<=-25).length;
  const sleepersAll=BOARD.filter(p=>p.sp>=10).length;
  const sleepersStrong=BOARD.filter(p=>p.sp>=25).length;
  {{
    let sh="";
    if(v.n>0){{
      const good=v.acc>=55;
      sh+=`<div style="flex:1;min-width:130px;background:${{good?"#e8f5e9":"#fff8e1"}};border:1px solid ${{good?"#c8e6c9":"#ffe082"}};border-radius:8px;padding:10px 14px;text-align:center">
        <div style="font-size:1.4rem;font-weight:700">${{v.acc}}%</div>
        <div style="font-size:0.72rem;color:#666;margin-top:3px">${{META.has_actuals?"This season":"Historical"}} accuracy vs ADP<br><span style="color:#999">10+ rank gap · ${{v.wins}}/${{v.n}} calls</span></div>
      </div>`;
    }}
    sh+=`<div style="flex:1;min-width:110px;background:#fef3f2;border:1px solid #ffcdd2;border-radius:8px;padding:10px 14px;text-align:center">
      <div style="font-size:1.4rem;font-weight:700;color:#c62828">${{fadesAll}}</div>
      <div style="font-size:0.72rem;color:#666;margin-top:3px">Overvalued players<br><span style="color:#999">${{fadesStrong}} strong FADE</span></div>
    </div>`;
    sh+=`<div style="flex:1;min-width:110px;background:#e8f5e9;border:1px solid #c8e6c9;border-radius:8px;padding:10px 14px;text-align:center">
      <div style="font-size:1.4rem;font-weight:700;color:#2e7d32">${{sleepersAll}}</div>
      <div style="font-size:0.72rem;color:#666;margin-top:3px">Undervalued players<br><span style="color:#999">${{sleepersStrong}} strong Sleeper</span></div>
    </div>`;
    h+=`<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px">${{sh}}</div>`;
  }}

  // Data sources (collapsible)
  const src=META.sources||[];
  if(src.length){{
    const avail=src.filter(s=>s.status==="available").length;
    let dots="";
    for(const s of src){{
      const dot=s.status==="available"?`<span style="color:#4caf50">●</span>`:s.status==="partial"?`<span style="color:#ff9800">●</span>`:`<span style="color:#ccc">●</span>`;
      dots+=`<span style="margin-right:14px">${{dot}} ${{s.name}}</span>`;
    }}
    h+=`<details style="margin-bottom:12px">
      <summary style="font-size:0.75rem;color:#999;cursor:pointer;list-style:none;display:inline-flex;align-items:center;gap:5px">
        <span style="font-size:0.65rem">▶</span> Data sources (${{avail}}/${{src.length}} available)
      </summary>
      <div style="display:flex;flex-wrap:wrap;gap:6px 0;font-size:0.75rem;color:#888;margin-top:6px;padding:0 2px">${{dots}}</div>
    </details>`;
  }}

  h+=renderPills();

  // Search
  h+=`<input class="search-box" id="searchInput" placeholder="Search players… just start typing" value="${{searchQ}}" oninput="searchQ=this.value;renderTable()" autofocus>`;

  // Table container — updated independently so search box keeps focus
  h+=`<div id="tableArea"></div>`;

  return h;
}}

// Route any printable keypress to the search box
document.addEventListener("keydown",function(e){{
  const tag=document.activeElement&&document.activeElement.tagName;
  if(tag==="INPUT"||tag==="TEXTAREA"||e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key.length===1){{
    const box=document.getElementById("searchInput");
    if(box){{box.focus();}}
  }}
  if(e.key==="Escape"){{
    searchQ="";
    const box=document.getElementById("searchInput");
    if(box){{box.value="";renderTable();}}
  }}
}});

function buildTableHTML(){{
  let players=filterPos(BOARD);
  if(searchQ){{
    const q=searchQ.toLowerCase();
    players=players.filter(p=>p.n.toLowerCase().includes(q)||p.t.toLowerCase().includes(q)||(p.role&&p.role.toLowerCase().includes(q))||(p.usage&&p.usage.toLowerCase().includes(q))||(p.tt&&p.tt.toLowerCase().includes(q)));
  }}
  players=sorted(players);

  // Signal tag (hoisted — used in top picks card and rows)
  function signalTag(sp){{
    if(sp<=-25)return`<span class="signal signal-fade-strong">FADE</span>`;
    if(sp<=-10)return`<span class="signal signal-fade">Fade</span>`;
    if(sp>=25)return`<span class="signal signal-sleeper">Sleeper</span>`;
    if(sp>=10)return`<span class="signal signal-sleeper-mild">Sleeper?</span>`;
    return"";
  }}

  // Legend
  let h="";
  h+=`<div style="font-size:0.75rem;color:#888;margin-bottom:8px">
    <span class="signal signal-fade-strong" style="font-size:0.7rem">FADE</span> Overvalued (71% accurate)
    &nbsp;&nbsp;
    <span class="signal signal-fade" style="font-size:0.7rem">Fade</span> Likely overvalued
    &nbsp;&nbsp;
    <span class="signal signal-sleeper" style="font-size:0.7rem">Sleeper</span> Possible value (58% accurate)
    &nbsp;&nbsp;
    <span class="signal signal-sleeper-mild" style="font-size:0.7rem">Sleeper?</span> Mild signal
  </div>`;

  const arrow=c=>sortCol===c?(sortAsc?"\\u25B2":"\\u25BC"):"";
  const actH=META.has_actuals?`<th class="num" onclick="setSort('act')">Actual ${{arrow("act")}}</th><th onclick="setSort('w')">Hit? ${{arrow("w")}}</th>`:"";

  h+=`<div class="card" style="padding:0;overflow-x:auto"><table>
    <tr>
      <th onclick="setSort('mr')">Rank ${{arrow("mr")}}</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Role</th>
      <th>Team</th>
      <th class="num" onclick="setSort('ecr')">ADP ${{arrow("ecr")}}</th>
      <th class="num" onclick="setSort('sp')">vs ADP ${{arrow("sp")}}</th>
      <th class="num" onclick="setSort('proj')">Proj ${{arrow("proj")}}</th>
      <th class="num" onclick="setSort('vorp')">VORP ${{arrow("vorp")}}</th>
      ${{actH}}
    </tr>`;

  const usageCls=u=>u==="Bellcow"||u==="Alpha"?"usage-elite":u==="Lead back"||u==="Starter"?"usage-solid":"usage-other";
  const tendCls=t=>t==="Run-heavy"?"tend-run":t==="Pass-heavy"?"tend-pass":"tend-bal";

  for(const p of players){{
    const spCls=p.sp>5?"spread-pos":p.sp<-5?"spread-neg":"";
    const spStr=p.sp>0?"+"+p.sp:String(p.sp);
    const rowCls=p.sp<=-25?"fade-strong":p.sp<=-10?"fade":p.sp>=10?"sleeper":"";
    let actCells="";
    if(META.has_actuals){{
      const hitCls=p.w?"spread-pos":"spread-neg";
      actCells=`<td class="num">${{p.act}}</td><td class="${{hitCls}}">${{p.w?"Yes":"No"}}</td>`;
    }}
    const shareStr=p.p==="RB"&&p.cs?p.cs+"% car":p.ts?p.ts+"% tgt":"";
    h+=`<tr class="${{rowCls}}">
      <td class="num">${{p.mr}}</td>
      <td style="font-weight:500">${{p.n}}</td>
      <td><span class="pos pos-${{p.p}}">${{p.p}}</span></td>
      <td>
        <span class="role-badge">${{p.role||""}}</span>
        <span class="usage-tag ${{usageCls(p.usage)}}">${{p.usage||""}}</span>
        ${{shareStr?`<span style="color:#999;font-size:0.7rem;display:block">${{shareStr}}</span>`:""}}
      </td>
      <td>
        <span style="font-weight:500">${{p.t}}</span>
        ${{p.tt?`<span class="tend-tag ${{tendCls(p.tt)}}">${{p.tt}}</span>`:""}}
        ${{p.trp?`<span style="color:#999;font-size:0.7rem;display:block">${{p.trp}}/${{p.tpp}}</span>`:""}}
      </td>
      <td class="num">${{p.ecr}}</td>
      <td class="num ${{spCls}}">${{spStr}} ${{signalTag(p.sp)}}</td>
      <td class="num">${{p.proj}}</td>
      <td class="num">${{p.vorp}}</td>
      ${{actCells}}
    </tr>`;
  }}
  h+=`</table></div>`;

  return h;
}}

function renderTable(){{
  const el=document.getElementById("tableArea");
  if(el)el.innerHTML=buildTableHTML();
  const rl=document.getElementById("rosterArea");
}}

// Render
function render(){{
  const el=document.getElementById("main");
  el.innerHTML=renderRankings();
  renderTable();
}}

render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate Draft Advisor HTML")
    parser.add_argument("--season", type=int, default=None, help="Season year")
    args = parser.parse_args()

    season = args.season
    if season is None:
        from config.settings import CURRENT_NFL_SEASON
        # Draft advisor targets the upcoming season; nfl_calendar returns the
        # last completed season (e.g. 2025 in May 2026), so default to +1.
        season = CURRENT_NFL_SEASON + 1

    print(f"Building draft advisor for {season}...")

    print("  Loading board data...")
    board_data = build_board_data(season)
    print(f"  {len(board_data['players'])} players loaded")

    print("  Computing VONA for all slots...")
    vona_data = build_vona_data(
        board_data["board"], board_data["adp_df"], max_slots=TEAMS
    )
    print(f"  VONA computed for {len(vona_data)} slots")

    print("  Checking data sources...")
    data_sources = check_data_sources(season)
    avail = sum(1 for s in data_sources if s["status"] == "available")
    print(f"  {avail}/{len(data_sources)} sources available")

    print("  Generating HTML...")
    html = generate_html(board_data, vona_data, data_sources)

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SITE_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"  Written to {out_path} ({size_kb:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
