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

    # 6. Rookie / NFL Draft data
    try:
        rookie_count = _count(
            "SELECT COUNT(*) FROM draft_picks WHERE season=?", (season,)
        )
    except sqlite3.OperationalError:
        rookie_count = 0
    sources.append({
        "name": f"{season} NFL Draft Picks",
        "status": "available" if rookie_count > 50 else "unavailable",
        "detail": f"{rookie_count} picks" if rookie_count else "Available after NFL Draft (~April)",
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
    """Build player roles from actual usage data (targets, snaps, carries)."""
    import sqlite3
    from config.settings import DB_PATH

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

    from collections import defaultdict
    team_pos = defaultdict(lambda: defaultdict(list))
    for team, name, pos, tgt, carries, snaps, rec, games in rows:
        team_pos[team][pos].append({
            "name": name,
            "targets": tgt or 0,
            "carries": carries or 0,
            "snaps": snaps or 0,
            "receptions": rec or 0,
            "games": games,
        })

    # Rank within team+position by primary usage metric
    roles = {}  # name -> {role, target_share, snap_pct, usage_note}
    for team, positions in team_pos.items():
        # Compute team totals for share calculations
        team_targets = sum(p["targets"] for pos_list in positions.values() for p in pos_list)
        team_carries = sum(p["carries"] for pos_list in positions.values() for p in pos_list)

        for pos, players in positions.items():
            # Sort by primary metric: QB/WR/TE by targets+snaps, RB by carries+targets
            if pos == "RB":
                players.sort(key=lambda x: -(x["carries"] + x["targets"]))
            else:
                players.sort(key=lambda x: -(x["targets"] + x["snaps"]))

            for rank, p in enumerate(players, 1):
                tgt_share = round(p["targets"] / team_targets * 100) if team_targets else 0
                carry_share = round(p["carries"] / team_carries * 100) if team_carries else 0

                # Usage note
                if pos == "RB":
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
                else:  # QB
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
    def _norm_key(name, pos):
        name = name.strip()
        if "." in name:
            parts = name.split(".")
            initial = parts[0].strip()[0].lower() if parts[0].strip() else ""
            last = parts[-1].strip().lower()
        else:
            parts = name.split()
            initial = parts[0][0].lower() if parts else ""
            last = parts[-1].lower() if parts else ""
        # Strip suffixes like Jr, II, III, IV
        for suf in (" jr", " sr", " ii", " iii", " iv", " v"):
            if last.endswith(suf):
                last = last[:-len(suf)].strip()
        return (initial, "".join(c for c in last if c.isalnum()), pos)

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

        players.append({
            "id": i,
            "n": sr.name,
            "p": sr.position,
            "t": sr.team,
            "ecr": round(sr.ecr, 1),
            "mr": sr.model_rank,
            "sp": sr.rank_spread,
            "proj": round(sr.model_projection, 1),
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
.tabs{{
  display:flex;background:#252547;padding:0 12px;gap:4px;
  overflow-x:auto;scrollbar-width:none;
}}
.tabs::-webkit-scrollbar{{display:none}}
.tab{{
  flex-shrink:0;padding:12px 20px;background:none;border:none;
  border-bottom:3px solid transparent;color:rgba(255,255,255,0.5);
  font-size:0.9rem;font-weight:500;cursor:pointer;transition:0.15s;
}}
.tab:hover{{color:rgba(255,255,255,0.8)}}
.tab.active{{color:#fff;border-bottom-color:#4fc3f7}}
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
tr.sleeper td{{background:#f1f8e9}}
tr.fade td{{background:#fef3f2}}
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
.slot-picker{{
  display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:14px;
}}
.slot-btn{{
  width:40px;height:40px;border-radius:50%;border:2px solid #ddd;
  background:#fff;font-weight:600;cursor:pointer;transition:0.15s;
  display:flex;align-items:center;justify-content:center;
}}
.slot-btn:hover{{border-color:#4fc3f7}}
.slot-btn.active{{background:#4fc3f7;color:#fff;border-color:#4fc3f7}}
.round-card{{
  background:#fff;border-radius:12px;margin-bottom:10px;
  box-shadow:0 1px 4px rgba(0,0,0,0.06);overflow:hidden;
}}
.round-header{{
  padding:12px 16px;font-weight:600;font-size:0.9rem;
  background:#f8f9fa;border-bottom:1px solid #eee;
  display:flex;justify-content:space-between;
}}
.round-body{{padding:8px}}
.pick-row{{
  display:flex;align-items:center;padding:8px 12px;gap:12px;
  border-bottom:1px solid #f5f5f5;cursor:pointer;transition:0.1s;
}}
.pick-row:hover{{background:#f0f7ff}}
.pick-row:last-child{{border-bottom:none}}
.pick-row.top{{background:#e8f5e9}}
.pick-name{{font-weight:600;flex:1}}
.pick-stat{{font-size:0.8rem;color:#666;min-width:60px;text-align:right}}
.pick-net{{font-weight:700}}
.pick-net.pos{{color:#2e7d32}}.pick-net.neg{{color:#c62828}}
.draft-state{{
  display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;
}}
.roster-slot{{
  padding:8px 12px;border-radius:8px;background:#f8f9fa;
  font-size:0.85rem;display:flex;justify-content:space-between;
}}
.roster-slot.filled{{background:#e8f5e9}}
.roster-slot .label{{font-weight:600;color:#666}}
.search-box{{
  width:100%;padding:10px 14px;border:1.5px solid #ddd;border-radius:8px;
  font-size:0.9rem;margin-bottom:12px;outline:none;
}}
.search-box:focus{{border-color:#4fc3f7}}
.drafted{{opacity:0.3;text-decoration:line-through;pointer-events:none}}
.btn{{
  padding:6px 14px;border-radius:6px;border:none;font-size:0.8rem;
  font-weight:500;cursor:pointer;transition:0.15s;
}}
.btn-pick{{background:#4fc3f7;color:#fff}}.btn-pick:hover{{background:#29b6f6}}
.btn-mark{{background:#eee;color:#666}}.btn-mark:hover{{background:#ddd}}
.btn-undo{{background:#ffebee;color:#c62828;font-size:0.75rem}}
.btn-undo:hover{{background:#ffcdd2}}
.empty{{text-align:center;color:#888;padding:24px;font-size:0.9rem}}
.footer{{text-align:center;padding:24px 16px;color:#aaa;font-size:0.7rem}}
@media(max-width:600px){{
  .draft-state{{grid-template-columns:1fr}}
  table{{font-size:0.8rem}}
  th,td{{padding:6px 4px}}
}}
</style>
</head>
<body>
<div class="header">
  <h1>Draft Advisor {season}</h1>
  <div class="sub">Model-powered draft rankings &amp; live companion</div>
</div>
<div class="tabs" id="tabs"></div>
<div class="main" id="main"></div>
<div class="footer">Projections generated from walk-forward ML models trained on 2006&ndash;{season - 1} NFL data. Not financial advice.</div>

<script>
const BOARD={players_json};
const VONA={vona_json};
const META={meta_json};

let view="rankings";
let posFilter="All";
let sortCol="mr";
let sortAsc=true;
let draftSlot=1;
let drafted=new Set();
let myRoster=[];
let searchQ="";

// Tabs
function renderTabs(){{
  const t=document.getElementById("tabs");
  t.innerHTML=[
    ["rankings","Rankings"],
    ["companion","Draft Companion"],
  ].map(([k,l])=>`<button class="tab ${{view===k?"active":""}}" onclick="view='${{k}}';render()">${{l}}</button>`).join("");
}}

// Pills
function renderPills(){{
  const positions=["All","QB","RB","WR","TE"];
  return `<div class="pills">${{positions.map(p=>
    `<button class="pill ${{posFilter===p?"active":""}}" onclick="posFilter='${{p}}';render()">${{p}}</button>`
  ).join("")}}</div>`;
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

  // Accuracy banner
  const v=META.validation;
  if(v.n>0){{
    const cls=v.acc>=55?"banner-good":"banner-warn";
    const label=META.has_actuals?"This season":"Historical";
    h+=`<div class="banner ${{cls}}">
      <span style="font-size:1.3rem;font-weight:700">${{v.acc}}%</span>
      <span style="font-size:0.9rem;color:#555"> ${{label}} accuracy when model disagrees with ADP by 10+ ranks (${{v.wins}}/${{v.n}})</span>
    </div>`;
  }}

  // Data sources checklist
  const src=META.sources||[];
  if(src.length){{
    h+=`<div style="display:flex;flex-wrap:wrap;gap:6px 14px;font-size:0.75rem;color:#888;margin-bottom:12px;padding:0 2px">`;
    for(const s of src){{
      const icon=s.status==="available"?"\\u2705":s.status==="partial"?"\\u26A0\\uFE0F":"\\u2B1C";
      h+=`<span>${{icon}} ${{s.name}}</span>`;
    }}
    h+=`</div>`;
  }}

  h+=renderPills();

  // Search
  h+=`<input class="search-box" id="searchInput" placeholder="Search players..." value="${{searchQ}}" oninput="searchQ=this.value;renderTable()">`;

  // Table container — updated independently so search box keeps focus
  h+=`<div id="tableArea"></div>`;

  return h;
}}

function buildTableHTML(){{
  let players=filterPos(BOARD);
  if(searchQ){{
    const q=searchQ.toLowerCase();
    players=players.filter(p=>p.n.toLowerCase().includes(q)||p.t.toLowerCase().includes(q)||(p.role&&p.role.toLowerCase().includes(q))||(p.usage&&p.usage.toLowerCase().includes(q))||(p.tt&&p.tt.toLowerCase().includes(q)));
  }}
  players=sorted(players);

  const arrow=c=>sortCol===c?(sortAsc?"\\u25B2":"\\u25BC"):"";
  const actH=META.has_actuals?`<th class="num" onclick="setSort('act')">Actual ${{arrow("act")}}</th><th onclick="setSort('w')">Hit? ${{arrow("w")}}</th>`:"";

  let h=`<div class="card" style="padding:0;overflow-x:auto"><table>
    <tr>
      <th onclick="setSort('mr')">Rank ${{arrow("mr")}}</th>
      <th>Player</th>
      <th>Pos</th>
      <th>Role</th>
      <th>Team</th>
      <th class="num" onclick="setSort('ecr')">ADP ${{arrow("ecr")}}</th>
      <th class="num" onclick="setSort('sp')">Spread ${{arrow("sp")}}</th>
      <th class="num" onclick="setSort('proj')">Proj ${{arrow("proj")}}</th>
      <th class="num" onclick="setSort('vorp')">VORP ${{arrow("vorp")}}</th>
      ${{actH}}
    </tr>`;

  const usageCls=u=>u==="Bellcow"||u==="Alpha"?"usage-elite":u==="Lead back"||u==="Starter"?"usage-solid":"usage-other";
  const tendCls=t=>t==="Run-heavy"?"tend-run":t==="Pass-heavy"?"tend-pass":"tend-bal";

  for(const p of players){{
    const spCls=p.sp>5?"spread-pos":p.sp<-5?"spread-neg":"";
    const spStr=p.sp>0?"+"+p.sp:String(p.sp);
    const rowCls=p.sp>=10?"sleeper":p.sp<=-10?"fade":"";
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
      <td class="num ${{spCls}}">${{spStr}}</td>
      <td class="num">${{p.proj}}</td>
      <td class="num">${{p.vorp}}</td>
      ${{actCells}}
    </tr>`;
  }}
  h+=`</table></div>`;

  // Legend
  h+=`<div style="font-size:0.75rem;color:#888;margin-top:8px">
    <span style="display:inline-block;width:12px;height:12px;background:#f1f8e9;border:1px solid #c8e6c9;border-radius:2px;vertical-align:middle"></span> Sleeper (model 10+ ranks higher)
    &nbsp;&nbsp;
    <span style="display:inline-block;width:12px;height:12px;background:#fef3f2;border:1px solid #ffcdd2;border-radius:2px;vertical-align:middle"></span> Fade (model 10+ ranks lower)
  </div>`;

  return h;
}}

function renderTable(){{
  const el=document.getElementById("tableArea");
  if(el)el.innerHTML=buildTableHTML();
}}

// Draft companion
function renderCompanion(){{
  let h="";

  // Slot picker
  h+=`<div class="card">
    <div style="font-weight:600;margin-bottom:8px">Your Draft Slot</div>
    <div class="slot-picker">`;
  for(let s=1;s<=META.teams;s++){{
    h+=`<button class="slot-btn ${{draftSlot===s?"active":""}}" onclick="draftSlot=${{s}};render()">${{s}}</button>`;
  }}
  h+=`</div></div>`;

  // My roster
  const slots={{"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":1}};
  const filled={{}};
  for(const p of myRoster){{
    filled[p.p]=(filled[p.p]||0)+1;
  }}

  h+=`<div class="card">
    <div style="font-weight:600;margin-bottom:8px">My Roster</div>
    <div class="draft-state">`;
  for(const[pos,count]of Object.entries(slots)){{
    const have=pos==="FLEX"?Math.max(0,myRoster.length-7):filled[pos]||0;
    const need=count;
    const isFilled=have>=need;
    h+=`<div class="roster-slot ${{isFilled?"filled":""}}">
      <span class="label">${{pos}}</span><span>${{have}}/${{need}}</span>
    </div>`;
  }}
  h+=`</div>`;

  if(myRoster.length){{
    h+=`<div style="margin-top:8px">`;
    for(let i=myRoster.length-1;i>=0;i--){{
      const p=myRoster[i];
      h+=`<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:0.85rem">
        <span><span class="pos pos-${{p.p}}">${{p.p}}</span> ${{p.n}} <span style="color:#888">(${{p.t}})</span></span>
        <button class="btn btn-undo" onclick="undoPick(${{p.id}})">Undo</button>
      </div>`;
    }}
    h+=`</div>`;
  }}
  h+=`</div>`;

  // VONA recommendations
  const vonaData=VONA[String(draftSlot)]||[];
  if(!vonaData.length){{
    h+=`<div class="card empty">No VONA data for slot ${{draftSlot}}.</div>`;
    return h;
  }}

  const rounds=[...new Set(vonaData.map(v=>v.rd))].sort((a,b)=>a-b);

  for(const rd of rounds){{
    let picks=vonaData.filter(v=>v.rd===rd);
    // Filter out drafted players
    picks=picks.filter(v=>!drafted.has(v.n));
    if(!picks.length)continue;

    const pickNum=picks[0].pk;
    h+=`<div class="round-card">
      <div class="round-header">
        <span>Round ${{rd}}</span><span style="color:#888;font-weight:400">Pick ${{pickNum}}</span>
      </div>
      <div class="round-body">`;

    for(let i=0;i<picks.length;i++){{
      const v=picks[i];
      const netCls=v.net>0?"pos":"neg";
      const isTop=i===0;
      h+=`<div class="pick-row ${{isTop?"top":""}}">
        <span class="pos pos-${{v.p}}">${{v.p}}</span>
        <span class="pick-name">${{v.n}}</span>
        <span class="pick-stat">${{v.av}}% avail</span>
        <span class="pick-stat">Proj ${{v.proj}}</span>
        <span class="pick-stat pick-net ${{netCls}}">Net ${{v.net>0?"+":""}}${{v.net}}</span>
        <button class="btn btn-pick" onclick="draftPlayer('${{v.n.replace("'","\\\\'")}}',${{v.rd}})">Draft</button>
        <button class="btn btn-mark" onclick="markOther('${{v.n.replace("'","\\\\'")}}')" title="Mark as taken by another team">X</button>
      </div>`;
    }}
    h+=`</div></div>`;
  }}

  // Drafted by others
  if(drafted.size){{
    h+=`<div class="card">
      <div style="font-weight:600;margin-bottom:6px">Drafted (${{drafted.size}})</div>
      <div style="font-size:0.8rem;color:#888">`;
    h+=[...drafted].join(", ");
    h+=`</div>
      <button class="btn btn-undo" style="margin-top:8px" onclick="drafted.clear();myRoster=[];render()">Reset All</button>
    </div>`;
  }}

  return h;
}}

// Actions
function draftPlayer(name,round){{
  const p=BOARD.find(b=>b.n===name);
  if(p){{
    myRoster.push(p);
    drafted.add(name);
    render();
  }}
}}

function markOther(name){{
  drafted.add(name);
  render();
}}

function undoPick(id){{
  const idx=myRoster.findIndex(p=>p.id===id);
  if(idx>=0){{
    const name=myRoster[idx].n;
    myRoster.splice(idx,1);
    drafted.delete(name);
    render();
  }}
}}

// Render
function render(){{
  renderTabs();
  const el=document.getElementById("main");
  if(view==="rankings"){{
    el.innerHTML=renderRankings();
    renderTable();
  }}
  else el.innerHTML=renderCompanion();
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
        season = CURRENT_NFL_SEASON

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
