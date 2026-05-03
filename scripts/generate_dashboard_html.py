#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard from walk-forward backtest CSVs.

Reads cached leakage-free predictions from
``data/backtest_results/ts_backtest_{season}_*_predictions.csv``, pre-computes
all metrics in Python, and emits a single ``_site/index.html`` with inline
CSS + JS.  Designed for GitHub Pages deployment.

Usage::

    python scripts/generate_dashboard_html.py
    open _site/index.html
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"
SITE_DIR = PROJECT_ROOT / "_site"
SEASONS = list(range(2018, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]


# ---------------------------------------------------------------------------
# Data loading (reused from render_walk_forward_validation.py)
# ---------------------------------------------------------------------------

def latest_predictions_csv(season: int) -> Optional[Path]:
    candidates = [
        c for c in RESULTS_DIR.glob(f"ts_backtest_{season}_*_predictions.csv")
        if "_conformal" not in c.name
    ]
    if not candidates:
        return None

    def ts_key(p: Path) -> str:
        parts = p.stem.split("_")
        if len(parts) >= 5:
            return parts[3] + parts[4]
        return p.name

    return max(candidates, key=ts_key)


def load_rows(csv_path: Path, season: int) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                if int(r.get("season", -1)) != season:
                    continue
            except (TypeError, ValueError):
                continue
            if str(r.get("is_active", "1")).lower() in ("0", "false"):
                continue
            pred = r.get("predicted")
            actual = r.get("actual")
            if not pred or actual in (None, "", "nan"):
                continue
            try:
                rows.append({
                    "week": int(r["week"]),
                    "name": (r.get("name") or "").strip(),
                    "position": (r.get("position") or "").strip(),
                    "team": (r.get("team") or "").strip(),
                    "predicted": round(float(pred), 2),
                    "actual": round(float(actual), 2),
                })
            except (TypeError, ValueError):
                continue
    return rows


def compute_metrics(rows: List[Dict]) -> Dict:
    if not rows:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    n = len(rows)
    errs = [r["actual"] - r["predicted"] for r in rows]
    mae = round(sum(abs(e) for e in errs) / n, 2)
    rmse = round(math.sqrt(sum(e * e for e in errs) / n), 2)
    mean_actual = sum(r["actual"] for r in rows) / n
    ss_res = sum(e * e for e in errs)
    ss_tot = sum((r["actual"] - mean_actual) ** 2 for r in rows) or 1.0
    r2 = round(1 - ss_res / ss_tot, 3)
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": n}


# ---------------------------------------------------------------------------
# Close calls extraction
# ---------------------------------------------------------------------------

# How many starters a 12-team league typically needs per position
_STARTER_DEPTH = {"QB": 14, "RB": 28, "WR": 30, "TE": 14}
_MAX_PROJ_GAP = 1.5   # projected points — tighter = harder call
_MIN_ACTUAL_GAP = 5.0  # actual points — bigger = more consequential
_MIN_ACTUAL_PTS = 2.0   # filter out true duds / inactive


def _build_close_calls(all_data: Dict[int, List]) -> Dict[str, List]:
    """Find the tightest projected margins where the outcome was decisive.

    Only keeps the single most impactful pair per player per week to avoid
    redundant cards (e.g. Gibbs vs 4 different RBs in the same week).
    """
    calls_by_season: Dict[str, List] = {}

    for season_key, rows in all_data.items():
        season = str(season_key)
        # Group by week+position
        buckets: Dict[tuple, List] = defaultdict(list)
        for r in rows:
            buckets[(r["w"], r["p"])].append(r)

        season_calls = []
        for (week, pos), players in buckets.items():
            # Rank by projection descending, keep starter-range only
            ranked = sorted(players, key=lambda x: -x["pr"])
            depth = _STARTER_DEPTH.get(pos, 20)
            ranked = ranked[:depth]

            # Only compare adjacent pairs in the ranking — the actual
            # start/sit dilemma a manager faces
            week_pairs = []
            for i in range(len(ranked) - 1):
                a, b = ranked[i], ranked[i + 1]
                proj_gap = a["pr"] - b["pr"]
                if proj_gap > _MAX_PROJ_GAP:
                    continue
                if a["a"] < _MIN_ACTUAL_PTS or b["a"] < _MIN_ACTUAL_PTS:
                    continue
                actual_gap = abs(a["a"] - b["a"])
                if actual_gap < _MIN_ACTUAL_GAP:
                    continue

                model_right = a["a"] > b["a"]
                week_pairs.append({
                    "wk": week,
                    "pos": pos,
                    "favN": a["n"],
                    "favT": a["t"],
                    "favPr": a["pr"],
                    "favA": a["a"],
                    "dogN": b["n"],
                    "dogT": b["t"],
                    "dogPr": b["pr"],
                    "dogA": b["a"],
                    "right": model_right,
                    "delta": round(actual_gap, 1),
                    "projGap": round(proj_gap, 1),
                })

            # Deduplicate: keep only the highest-impact pair per player
            seen_players: set = set()
            week_pairs.sort(key=lambda c: -c["delta"])
            for pair in week_pairs:
                key_fav = (week, pair["favN"])
                key_dog = (week, pair["dogN"])
                if key_fav in seen_players or key_dog in seen_players:
                    continue
                seen_players.add(key_fav)
                seen_players.add(key_dog)
                season_calls.append(pair)

        # Sort by impact (actual gap) descending — biggest swings first
        season_calls.sort(key=lambda c: -c["delta"])
        # Keep top 100 per season to limit payload size while ensuring
        # enough for position filtering
        calls_by_season[season] = season_calls[:100]

    return calls_by_season


# ---------------------------------------------------------------------------
# Build JSON payloads
# ---------------------------------------------------------------------------

def build_data() -> tuple:
    """Return (metrics_json, data_json, seasons_with_data)."""
    headline = []
    by_position: Dict[int, Dict] = {}
    by_week: Dict[int, Dict] = {}
    all_data: Dict[int, List] = {}
    seasons_with_data = []

    for season in SEASONS:
        csv_path = latest_predictions_csv(season)
        if not csv_path:
            continue
        rows = load_rows(csv_path, season)
        if not rows:
            continue

        seasons_with_data.append(season)
        m = compute_metrics(rows)
        headline.append({"season": season, **m})

        # By position
        pos_groups: Dict[str, List] = defaultdict(list)
        for r in rows:
            pos_groups[r["position"]].append(r)
        by_position[season] = {
            pos: compute_metrics(pos_rows)
            for pos, pos_rows in pos_groups.items()
        }

        # By week
        week_groups: Dict[int, List] = defaultdict(list)
        for r in rows:
            week_groups[r["week"]].append(r)
        by_week[season] = {
            wk: compute_metrics(wk_rows)
            for wk, wk_rows in week_groups.items()
        }

        # Compact row data
        all_data[season] = [
            {
                "w": r["week"],
                "n": html.escape(r["name"]),
                "p": r["position"],
                "t": r["team"],
                "pr": r["predicted"],
                "a": r["actual"],
            }
            for r in sorted(rows, key=lambda x: (x["week"], -x["predicted"]))
        ]

    metrics_obj = {
        "headline": headline,
        "byPosition": {str(k): v for k, v in by_position.items()},
        "byWeek": {str(k): {str(wk): m for wk, m in v.items()} for k, v in by_week.items()},
    }
    data_obj = {str(k): v for k, v in all_data.items()}

    # Build close calls from the same loaded data
    close_calls_obj = _build_close_calls(all_data)

    # Build draft advisor data for seasons with ADP (2024, 2025)
    draft_obj = _build_draft_advisor_data()

    return metrics_obj, data_obj, close_calls_obj, draft_obj, seasons_with_data


# ---------------------------------------------------------------------------
# Draft Advisor data
# ---------------------------------------------------------------------------

def _build_draft_advisor_data() -> Dict:
    """Generate spread + VONA data for seasons with ADP data."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        import snake_draft_sim as sd
        import draft_advisor as da
    except Exception as e:
        print(f"Draft advisor import failed: {e}", file=sys.stderr)
        return {}

    result = {}
    for season in [2024, 2025, 2026]:
        try:
            csv_path = latest_predictions_csv(season)
            if csv_path:
                adp_df = sd.load_adp_board(season)
                projections = sd.load_model_projections(csv_path, ranking="week1", season=season)
            else:
                # No backtest CSV (upcoming season) — use preseason projections
                try:
                    adp_df = sd.load_adp_board(season)
                    projections = sd.load_preseason_projections(season, adp_df=adp_df)
                    if projections.empty:
                        continue
                except Exception:
                    continue
            board = sd.build_draft_board(adp_df, projections)
            matched = sum(1 for p in board if p.is_modelable)
            if matched < 50:
                continue

            # Spread data: top 80 by absolute spread + all ADP top-50 players
            spread_results = da.compute_spread(board)
            top_ecr_names = {p.name for p in board if p.ecr <= 50}
            included = set()
            spread_list = []
            for r in spread_results:
                if len(included) >= 120 and r.name not in top_ecr_names:
                    continue
                if r.name in included:
                    continue
                included.add(r.name)
                spread_list.append({
                    "n": r.name, "p": r.position, "t": r.team,
                    "ecr": round(r.ecr, 1),
                    "mr": r.model_rank, "sp": r.rank_spread,
                    "mp": round(r.model_projection, 1),
                    "ai": round(r.adp_implied, 1),
                    "act": round(r.actual_total, 1),
                    "w": r.model_wins,
                })

            # Validation at multiple thresholds — filtered to top-150 ECR
            top150 = [r for r in spread_results if r.ecr <= 150]
            validations = {}
            for thresh in [5, 10, 15, 20]:
                v = da.validate_spread_direction(top150, min_spread=thresh)
                if v["n"] > 0:
                    validations[str(thresh)] = {
                        "n": v["n"], "wins": v["model_wins"],
                        "acc": round(v["accuracy"] * 100, 1),
                    }
            # Also store whether this is a preseason (no actuals) season
            has_actuals = csv_path is not None

            # VONA for all 12 slots
            vona_by_slot = {}
            for slot in range(1, 13):
                try:
                    vona = da.compute_vona(board, adp_df, slot)
                    # Keep top 5 per round, first 5 rounds
                    vona_compact = []
                    seen_rounds = set()
                    for v in vona:
                        rd = v["round"]
                        if rd > 5:
                            continue
                        key = (rd, v["name"])
                        # Count per round
                        rd_count = sum(1 for x in vona_compact if x["rd"] == rd)
                        if rd_count >= 5:
                            continue
                        vona_compact.append({
                            "rd": rd, "pk": v["pick"],
                            "n": v["name"], "p": v["position"],
                            "av": round(v["avail_pct"] * 100),
                            "proj": round(v["model_proj"], 1),
                            "vona": round(v["vona"], 1),
                            "oc": round(v["opp_cost"], 1),
                            "ocp": v["opp_cost_pos"],
                            "net": round(v["net_value"], 1),
                        })
                    vona_by_slot[str(slot)] = vona_compact
                except Exception:
                    pass

            result[str(season)] = {
                "spread": spread_list,
                "validation": validations,
                "vona": vona_by_slot,
                "matched": matched,
                "total": len(board),
                "hasActuals": has_actuals,
            }
        except Exception as e:
            print(f"Draft advisor for {season} failed: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def build_html(metrics_json: str, data_json: str, calls_json: str,
               draft_json: str, seasons: List[int], generated_at: str) -> str:
    seasons_json = json.dumps(seasons)
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFL Walk-Forward Validation</title>
<style>
*,*::before,*::after{{box-sizing:border-box}}
body{{
  margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#f5f5f5;color:#1a1a2e;font-size:14px;line-height:1.4;
}}
.header{{
  position:sticky;top:0;z-index:100;
  background:#1a1a2e;color:#fff;padding:12px 16px;
  text-align:center;
}}
.header h1{{margin:0;font-size:1.1rem;font-weight:600;letter-spacing:0.5px}}
.header .subtitle{{font-size:0.75rem;opacity:0.7;margin-top:2px}}
.tab-bar{{
  display:flex;overflow-x:auto;-webkit-overflow-scrolling:touch;
  background:#252547;padding:0 8px;gap:4px;
  scrollbar-width:none;
}}
.tab-bar::-webkit-scrollbar{{display:none}}
.tab{{
  flex-shrink:0;padding:10px 14px;min-width:48px;
  background:none;border:none;border-bottom:3px solid transparent;
  color:rgba(255,255,255,0.6);font-size:0.85rem;font-weight:500;
  cursor:pointer;transition:all 0.15s;
}}
.tab:hover{{color:rgba(255,255,255,0.85)}}
.tab.active{{color:#fff;border-bottom-color:#4fc3f7}}
.content{{max-width:720px;margin:0 auto;padding:12px}}
.card{{
  background:#fff;border-radius:10px;padding:16px;
  margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08);
}}
.metrics-grid{{
  display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center;
}}
.metric-val{{font-size:1.4rem;font-weight:700;color:#1a1a2e}}
.metric-label{{font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:0.5px}}
.pills{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}}
.pill{{
  padding:8px 16px;border-radius:20px;border:1.5px solid #ddd;
  background:#fff;color:#555;font-size:0.8rem;font-weight:500;
  cursor:pointer;transition:all 0.15s;min-width:48px;text-align:center;
}}
.pill:hover{{border-color:#4fc3f7;color:#1a1a2e}}
.pill.active{{background:#4fc3f7;color:#fff;border-color:#4fc3f7}}
table{{width:100%;border-collapse:collapse;font-size:0.8rem}}
th{{
  text-align:left;padding:8px 6px;border-bottom:2px solid #e0e0e0;
  font-weight:600;color:#555;font-size:0.7rem;text-transform:uppercase;
  letter-spacing:0.3px;position:sticky;top:0;background:#fff;
}}
th.num,td.num{{text-align:right}}
td{{padding:6px;border-bottom:1px solid #f0f0f0}}
tr:hover td{{background:#f8f9ff}}
.pos-badge{{
  display:inline-block;padding:1px 6px;border-radius:4px;
  font-size:0.7rem;font-weight:600;color:#fff;
}}
.pos-QB{{background:#e53935}}.pos-RB{{background:#43a047}}
.pos-WR{{background:#1e88e5}}.pos-TE{{background:#f9a825;color:#333}}
.resid-pos{{color:#2e7d32}}.resid-neg{{color:#c62828}}
details{{
  background:#fff;border-radius:10px;margin-bottom:8px;
  box-shadow:0 1px 3px rgba(0,0,0,0.08);overflow:hidden;
}}
summary{{
  padding:14px 16px;cursor:pointer;display:flex;align-items:center;
  gap:8px;font-weight:500;font-size:0.9rem;
  list-style:none;-webkit-tap-highlight-color:transparent;
}}
summary::-webkit-details-marker{{display:none}}
summary::before{{
  content:"\\25B6";font-size:0.65rem;color:#888;
  transition:transform 0.2s;flex-shrink:0;
}}
details[open] summary::before{{transform:rotate(90deg)}}
.week-metrics{{font-size:0.75rem;color:#888;margin-left:auto;white-space:nowrap}}
.table-wrap{{overflow-x:auto;padding:0 12px 12px;-webkit-overflow-scrolling:touch}}
.show-all-btn{{
  display:block;width:100%;padding:10px;margin-top:4px;
  background:none;border:1px dashed #ccc;border-radius:6px;
  color:#666;font-size:0.8rem;cursor:pointer;
}}
.show-all-btn:hover{{background:#f8f9ff;border-color:#4fc3f7;color:#1a1a2e}}
.empty{{text-align:center;color:#888;padding:32px 16px;font-size:0.9rem}}
.footer{{
  text-align:center;padding:24px 16px;color:#999;font-size:0.7rem;
}}
.call-card{{
  background:#fff;border-radius:10px;padding:14px 16px;
  margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,0.08);
  border-left:4px solid #ccc;
}}
.call-card.right{{border-left-color:#43a047}}
.call-card.wrong{{border-left-color:#c62828}}
.call-header{{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:8px;
}}
.call-week{{font-size:0.75rem;color:#888;font-weight:500}}
.call-verdict{{
  font-size:0.75rem;font-weight:700;padding:2px 8px;border-radius:4px;
}}
.call-verdict.right{{background:#e8f5e9;color:#2e7d32}}
.call-verdict.wrong{{background:#ffebee;color:#c62828}}
.call-matchup{{
  display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;
  font-size:0.85rem;
}}
.call-player{{text-align:center}}
.call-player-name{{font-weight:600;font-size:0.9rem}}
.call-player-team{{font-size:0.7rem;color:#888}}
.call-player-proj{{font-size:0.75rem;color:#888;margin-top:2px}}
.call-player-actual{{font-size:1.1rem;font-weight:700;margin-top:2px}}
.call-player-actual.winner{{color:#2e7d32}}
.call-player-actual.loser{{color:#c62828}}
.call-vs{{font-size:0.75rem;color:#aaa;font-weight:700}}
.call-summary{{
  display:grid;grid-template-columns:repeat(3,1fr);gap:8px;
  text-align:center;margin-bottom:16px;
}}
.call-summary .card{{padding:12px}}
.call-stat-val{{font-size:1.6rem;font-weight:700}}
.call-stat-val.good{{color:#2e7d32}}
.call-stat-val.bad{{color:#c62828}}
.call-stat-val.neutral{{color:#1a1a2e}}
.call-stat-label{{font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:0.5px}}
.call-filter-row{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;align-items:center}}
.call-filter-label{{font-size:0.75rem;color:#888;margin-right:4px}}
@media(min-width:600px){{
  .header h1{{font-size:1.3rem}}
  .content{{padding:16px 20px}}
  .metrics-grid{{gap:12px}}
  .metric-val{{font-size:1.6rem}}
  table{{font-size:0.85rem}}
}}
</style>
</head>
<body>
<div class="header">
  <h1>NFL Walk-Forward Validation</h1>
  <div class="subtitle">Model predictions vs actual results &middot; 2018&ndash;2025</div>
</div>
<div class="tab-bar" id="tabBar"></div>
<div class="content" id="content"></div>
<div class="footer">
  Generated {generated_at} &middot; predictions use only data available before each week
</div>

<script>
const SEASONS = {seasons_json};
const METRICS = {metrics_json};
const DATA = {data_json};
const CALLS = {calls_json};
const DRAFT = {draft_json};

let selectedSeason = SEASONS[SEASONS.length - 1];
let selPos = new Set(["QB","RB","WR","TE"]);
const ALL_POS = ["QB","RB","WR","TE"];
let viewMode = "predictions"; // "predictions" or "draft"
let draftSlot = 6;

function isAllPos() {{ return selPos.size === 4; }}
function matchPos(p) {{ return selPos.has(p); }}
function fmt(v, d) {{ return v == null ? "—" : v.toFixed(d); }}
function fmtN(v) {{ return v == null ? "—" : v.toLocaleString(); }}

function togglePos(p) {{
  if (p === "ALL") {{
    if (isAllPos()) return;
    ALL_POS.forEach(x => selPos.add(x));
  }} else {{
    if (selPos.has(p)) {{
      if (selPos.size === 1) return;
      selPos.delete(p);
    }} else {{
      selPos.add(p);
    }}
  }}
  render();
}}

function computeMetrics(rows) {{
  if (!rows.length) return {{ n: 0 }};
  const n = rows.length;
  const errs = rows.map(r => r.a - r.pr);
  const mae = errs.reduce((s,e) => s + Math.abs(e), 0) / n;
  const rmse = Math.sqrt(errs.reduce((s,e) => s + e*e, 0) / n);
  const meanA = rows.reduce((s,r) => s + r.a, 0) / n;
  const ssTot = rows.reduce((s,r) => s + (r.a - meanA)**2, 0) || 1;
  const ssRes = errs.reduce((s,e) => s + e*e, 0);
  return {{ mae, rmse, r2: 1 - ssRes/ssTot, n }};
}}

function renderTabs() {{
  const bar = document.getElementById("tabBar");
  bar.innerHTML = "";
  // View mode tabs
  ["Weekly Predictions","Pre-Draft Rankings"].forEach(label => {{
    const mode = label === "Weekly Predictions" ? "predictions" : "draft";
    const btn = document.createElement("button");
    btn.className = "tab" + (viewMode === mode ? " active" : "");
    btn.textContent = label;
    btn.style.borderBottomColor = viewMode === mode ? "#ff9800" : "transparent";
    btn.onclick = () => {{ viewMode = mode; render(); }};
    bar.appendChild(btn);
  }});
  // Separator
  const sep = document.createElement("span");
  sep.style.cssText = "width:2px;background:rgba(255,255,255,0.15);margin:6px 4px;flex-shrink:0";
  bar.appendChild(sep);
  // Season tabs
  const draftSeasons = Object.keys(DRAFT).map(Number).sort();
  const seasonList = viewMode === "predictions" ? SEASONS.concat(["All"]) : draftSeasons;
  if (viewMode === "draft" && !draftSeasons.includes(selectedSeason)) selectedSeason = draftSeasons[draftSeasons.length-1] || 2025;
  seasonList.forEach(s => {{
    const btn = document.createElement("button");
    const isPreseason = viewMode === "draft" && DRAFT[s] && DRAFT[s].hasActuals === false;
    btn.className = "tab" + (s === selectedSeason ? " active" : "");
    btn.textContent = isPreseason ? s + " *" : s;
    if (isPreseason) btn.style.fontStyle = "italic";
    btn.onclick = () => {{ selectedSeason = s; selPos = new Set(ALL_POS); render(); }};
    bar.appendChild(btn);
  }});
}}

function getHeadline(season) {{
  if (season === "All") {{
    const allRows = [];
    SEASONS.forEach(s => {{ if (DATA[s]) allRows.push(...DATA[s]); }});
    const filtered = isAllPos() ? allRows : allRows.filter(r => matchPos(r.p));
    return computeMetrics(filtered);
  }}
  const data = DATA[season] || [];
  const filtered = isAllPos() ? data : data.filter(r => matchPos(r.p));
  return computeMetrics(filtered);
}}

function getPosMeta(season) {{
  const combined = {{}};
  ALL_POS.forEach(pos => {{
    let rows = [];
    if (season === "All") {{
      SEASONS.forEach(s => {{ if (DATA[s]) rows.push(...DATA[s].filter(r => r.p === pos)); }});
    }} else {{
      rows = (DATA[season] || []).filter(r => r.p === pos);
    }}
    if (rows.length) combined[pos] = computeMetrics(rows);
  }});
  return combined;
}}

function renderHeadlineCard(m) {{
  if (!m || !m.n) return '<div class="card empty">No predictions for this selection.</div>';
  return `<div class="card"><div class="metrics-grid">
    <div><div class="metric-val">${{fmt(m.mae,2)}}</div><div class="metric-label">MAE</div></div>
    <div><div class="metric-val">${{fmt(m.rmse,2)}}</div><div class="metric-label">RMSE</div></div>
    <div><div class="metric-val">${{fmt(m.r2,3)}}</div><div class="metric-label">R&sup2;</div></div>
    <div><div class="metric-val">${{fmtN(m.n)}}</div><div class="metric-label">Predictions</div></div>
  </div></div>`;
}}

function renderPills() {{
  const allActive = isAllPos();
  return `<div class="pills">
    <button class="pill${{allActive?" active":""}}" onclick="togglePos('ALL')">All</button>
    ${{ALL_POS.map(p =>
      `<button class="pill${{selPos.has(p)?" active":""}}" onclick="togglePos('${{p}}')">${{p}}</button>`
    ).join("")}}
  </div>`;
}}

function renderPosTable(posMeta) {{
  const positions = ALL_POS.filter(p => posMeta[p]);
  if (!positions.length) return "";
  return `<div class="card"><table>
    <tr><th>Pos</th><th class="num">n</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">R&sup2;</th></tr>
    ${{positions.map(p => {{
      const m = posMeta[p];
      const hl = selPos.has(p) && !isAllPos() ? ' style="background:#e3f2fd"' : '';
      return `<tr${{hl}}><td><span class="pos-badge pos-${{p}}">${{p}}</span></td><td class="num">${{fmtN(m.n)}}</td><td class="num">${{fmt(m.mae,2)}}</td><td class="num">${{fmt(m.rmse,2)}}</td><td class="num">${{fmt(m.r2,3)}}</td></tr>`;
    }}).join("")}}
  </table></div>`;
}}

function renderAllSeasonsTable() {{
  return `<div class="card"><table>
    <tr><th>Season</th><th class="num">n</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">R&sup2;</th></tr>
    ${{METRICS.headline.map(h =>
      `<tr><td>${{h.season}}</td><td class="num">${{fmtN(h.n)}}</td><td class="num">${{fmt(h.mae,2)}}</td><td class="num">${{fmt(h.rmse,2)}}</td><td class="num">${{fmt(h.r2,3)}}</td></tr>`
    ).join("")}}
  </table></div>`;
}}

function renderPlayerTable(sorted, weekId) {{
  const limit = 25;
  function row(r, extra) {{
    const resid = r.a - r.pr;
    const cls = resid >= 0 ? "resid-pos" : "resid-neg";
    const style = extra ? ` class="extra-${{weekId}}" style="display:none"` : "";
    return `<tr${{style}}><td>${{r.n}}</td><td><span class="pos-badge pos-${{r.p}}">${{r.p}}</span></td><td>${{r.t}}</td><td class="num">${{r.pr.toFixed(1)}}</td><td class="num">${{r.a.toFixed(1)}}</td><td class="num ${{cls}}">${{(resid>=0?"+":"") + resid.toFixed(1)}}</td></tr>`;
  }}
  let h = `<div class="table-wrap"><table>
    <tr><th>Player</th><th>Pos</th><th>Team</th><th class="num">Pred</th><th class="num">Act</th><th class="num">Resid</th></tr>`;
  sorted.forEach((r, i) => {{ h += row(r, i >= limit); }});
  h += `</table>`;
  if (sorted.length > limit) {{
    h += `<button class="show-all-btn" id="btn-${{weekId}}" onclick="document.querySelectorAll('.extra-${{weekId}}').forEach(r=>r.style.display='');this.style.display='none'">Show all ${{sorted.length}} players</button>`;
  }}
  h += `</div>`;
  return h;
}}

function renderWeeks(season) {{
  const data = DATA[season] || [];
  const weeks = [...new Set(data.map(r => r.w))].sort((a,b) => a - b);

  return weeks.map(wk => {{
    const weekRows = data.filter(r => r.w === wk);
    const filtered = isAllPos() ? weekRows : weekRows.filter(r => matchPos(r.p));
    const dispMeta = computeMetrics(filtered);

    const metricsStr = dispMeta.n
      ? `n=${{dispMeta.n}} &middot; MAE ${{fmt(dispMeta.mae,2)}} &middot; R&sup2; ${{fmt(dispMeta.r2,3)}}`
      : `n=0`;
    const wid = `w${{season}}-${{wk}}`;

    return `<details>
      <summary>Week ${{wk}} <span class="week-metrics">${{metricsStr}}</span></summary>
      ${{filtered.length ? renderPlayerTable(filtered.sort((a,b) => b.pr - a.pr), wid) : '<div class="empty">No predictions for this selection.</div>'}}
    </details>`;
  }}).join("");
}}

function getCallsForView() {{
  let calls = [];
  if (selectedSeason === "All") {{
    SEASONS.forEach(s => {{ if (CALLS[s]) calls.push(...CALLS[s].map(c => ({{...c, season: s}}))); }});
  }} else {{
    calls = (CALLS[selectedSeason] || []).map(c => ({{...c, season: selectedSeason}}));
  }}
  if (!isAllPos()) calls = calls.filter(c => matchPos(c.pos));
  calls.sort((a, b) => b.delta - a.delta);
  return calls;
}}

function renderCallsSummary(calls) {{
  const total = calls.length;
  if (!total) return '<div class="card empty">No close calls for this selection.</div>';
  const wins = calls.filter(c => c.right).length;
  const losses = total - wins;
  const pct = Math.round(100 * wins / total);
  const pctClass = pct >= 55 ? "good" : pct <= 45 ? "bad" : "neutral";
  return `<div class="call-summary">
    <div class="card"><div class="call-stat-val ${{pctClass}}">${{pct}}%</div><div class="call-stat-label">Accuracy</div></div>
    <div class="card"><div class="call-stat-val good">${{wins}}</div><div class="call-stat-label">Right</div></div>
    <div class="card"><div class="call-stat-val bad">${{losses}}</div><div class="call-stat-label">Wrong</div></div>
  </div>`;
}}

function renderCallCard(c) {{
  const vClass = c.right ? "right" : "wrong";
  const vLabel = c.right ? "RIGHT" : "WRONG";
  const favWon = c.favA > c.dogA;
  const seasonLabel = selectedSeason === "All" ? c.season + " " : "";
  return `<div class="call-card ${{vClass}}">
    <div class="call-header">
      <span class="call-week">${{seasonLabel}}Week ${{c.wk}} &middot; <span class="pos-badge pos-${{c.pos}}">${{c.pos}}</span> &middot; proj gap ${{c.projGap}} pts</span>
      <span class="call-verdict ${{vClass}}">${{vLabel}} &middot; ${{c.delta}} pts</span>
    </div>
    <div class="call-matchup">
      <div class="call-player">
        <div class="call-player-name">${{c.favN}}</div>
        <div class="call-player-team">${{c.favT}} &middot; model pick</div>
        <div class="call-player-proj">Proj ${{c.favPr.toFixed(1)}}</div>
        <div class="call-player-actual ${{favWon?"winner":"loser"}}">${{c.favA.toFixed(1)}}</div>
      </div>
      <div class="call-vs">vs</div>
      <div class="call-player">
        <div class="call-player-name">${{c.dogN}}</div>
        <div class="call-player-team">${{c.dogT}}</div>
        <div class="call-player-proj">Proj ${{c.dogPr.toFixed(1)}}</div>
        <div class="call-player-actual ${{favWon?"loser":"winner"}}">${{c.dogA.toFixed(1)}}</div>
      </div>
    </div>
  </div>`;
}}

function renderCallsView() {{
  const calls = getCallsForView();
  let h = renderPills();
  h += renderCallsSummary(calls);
  const limit = 50;
  const shown = calls.slice(0, limit);
  h += shown.map(renderCallCard).join("");
  if (calls.length > limit) {{
    h += `<div class="card empty">${{calls.length - limit}} more close calls not shown.</div>`;
  }}
  return h;
}}

function renderDraftView() {{
  const d = DRAFT[selectedSeason];
  if (!d) return '<div class="card empty">No ADP data for this season.</div>';
  const hasActuals = d.hasActuals !== false;
  let h = '';

  // Preseason disclaimer for upcoming season
  if (!hasActuals) {{
    const missing = [];
    missing.push("NFL schedule (available late July)");
    missing.push("Rosters & depth charts (available August)");
    missing.push("Preseason & Week 1 stats");
    h += `<div class="card" style="background:#fff8e1;border-left:4px solid #f9a825;padding:14px">
      <div style="font-weight:600;font-size:0.9rem;margin-bottom:6px">Preseason projections &mdash; data is incomplete</div>
      <div style="font-size:0.8rem;color:#555;line-height:1.5">
        Rankings based on prior-season stats + ADP consensus. Rookies use ADP-implied projections (no NFL game data yet).
        <div style="margin-top:6px"><strong>Not yet available:</strong> ${{missing.join(" &middot; ")}}</div>
        <div style="margin-top:4px">ADP data updates as FantasyPros publishes new rankings through August. Re-run <code style="background:#f5f5f5;padding:1px 4px;border-radius:3px;font-size:0.75rem">python scripts/generate_dashboard_html.py</code> to refresh.</div>
      </div>
    </div>`;
  }}

  // Compact validation banner (skip for preseason)
  const v10 = d.validation["10"];
  if (v10 && hasActuals) {{
    const cls = v10.acc >= 60 ? "good" : v10.acc >= 50 ? "neutral" : "bad";
    h += `<div class="card" style="text-align:center;padding:12px">
      <span class="call-stat-val ${{cls}}" style="font-size:1.2rem">${{v10.acc}}%</span>
      <span style="font-size:0.85rem;color:#666"> accurate among top-150 ADP players where model disagrees by 10+ ranks (${{v10.wins}}/${{v10.n}})</span>
    </div>`;
  }}

  // VONA section (the actionable tool) — first
  const vonaData = d.vona || {{}};
  const slots = Object.keys(vonaData).sort((a,b)=>a-b);
  if (slots.length) {{
    h += `<div class="card"><h3 style="margin:0 0 8px;font-size:0.9rem">Pick Recommendations by Draft Slot</h3>`;
    h += `<div class="pills" style="margin-bottom:8px">`;
    slots.forEach(s => {{
      h += `<button class="pill${{parseInt(s)===draftSlot?" active":""}}" onclick="draftSlot=${{s}};render()">Slot ${{s}}</button>`;
    }});
    h += `</div>`;
    const vona = vonaData[String(draftSlot)] || vonaData[slots[0]] || [];
    if (vona.length) {{
      const rounds = [...new Set(vona.map(v=>v.rd))].sort();
      rounds.forEach(rd => {{
        const picks = vona.filter(v=>v.rd===rd);
        const pick = picks[0]?.pk || "?";
        h += `<details${{rd<=2?" open":""}}>
          <summary>Round ${{rd}} (Pick ${{pick}})</summary>
          <div class="table-wrap"><table>
            <tr><th>Player</th><th>Pos</th><th class="num">Avail</th><th class="num">Proj</th><th class="num">VONA</th><th class="num">Opp Cost</th><th class="num">Net</th></tr>
            ${{picks.map((v,i) => `<tr${{i===0?' style="background:#e8f5e9"':""}}><td>${{v.n}}</td><td><span class="pos-badge pos-${{v.p}}">${{v.p}}</span></td><td class="num">${{v.av}}%</td><td class="num">${{v.proj}}</td><td class="num">${{v.vona>0?"+":""}}${{v.vona}}</td><td class="num">${{v.oc>0?"-"+v.oc+"\u2009("+v.ocp+")":"—"}}</td><td class="num" style="font-weight:600;color:${{v.net>0?"#2e7d32":"#c62828"}}">${{v.net>0?"+":""}}${{v.net}}</td></tr>`).join("")}}
          </table></div>
        </details>`;
      }});
    }} else {{
      h += `<div class="empty">No VONA data for slot ${{draftSlot}}.</div>`;
    }}
    h += `</div>`;
  }}

  // Spread tables — filtered to ECR ≤ 150 (draftable players)
  const spread = (d.spread || []).filter(s => s.ecr <= 150);
  const filteredSpread = isAllPos() ? spread : spread.filter(s => matchPos(s.p));
  const under = filteredSpread.filter(s => s.sp > 0).sort((a,b) => b.sp - a.sp).slice(0, 20);
  const over = filteredSpread.filter(s => s.sp < 0).sort((a,b) => a.sp - b.sp).slice(0, 20);

  h += renderPills();

  const actCols = hasActuals ? '<th class="num">Actual</th><th>Right?</th>' : '';
  function spreadRow(s, color) {{
    const spStr = s.sp > 0 ? "+" + s.sp : String(s.sp);
    let actCells = '';
    if (hasActuals) {{
      const cls = s.w ? "resid-pos" : "resid-neg";
      actCells = `<td class="num">${{s.act}}</td><td class="${{cls}}">${{s.w?"✓":"✗"}}</td>`;
    }}
    return `<tr><td>${{s.n}}</td><td><span class="pos-badge pos-${{s.p}}">${{s.p}}</span></td><td class="num">${{Math.round(s.ecr)}}</td><td class="num">${{s.mr}}</td><td class="num" style="color:${{color}};font-weight:600">${{spStr}}</td><td class="num">${{s.mp}}</td>${{actCells}}</tr>`;
  }}

  if (under.length) {{
    h += `<div class="card"><h3 style="margin:0 0 8px;font-size:0.9rem;color:#2e7d32">Model Says Higher Than ADP</h3><table>
      <tr><th>Player</th><th>Pos</th><th class="num">ADP</th><th class="num">Model</th><th class="num">Spread</th><th class="num">Proj</th>${{actCols}}</tr>
      ${{under.map(s => spreadRow(s, "#2e7d32")).join("")}}
    </table></div>`;
  }}

  if (over.length) {{
    h += `<div class="card"><h3 style="margin:0 0 8px;font-size:0.9rem;color:#c62828">Model Says Lower Than ADP</h3><table>
      <tr><th>Player</th><th>Pos</th><th class="num">ADP</th><th class="num">Model</th><th class="num">Spread</th><th class="num">Proj</th>${{actCols}}</tr>
      ${{over.map(s => spreadRow(s, "#c62828")).join("")}}
    </table></div>`;
  }}

  return h;
}}

function render() {{
  renderTabs();
  const el = document.getElementById("content");

  if (viewMode === "draft") {{
    el.innerHTML = renderDraftView();
    return;
  }}

  // Predictions view (metrics + weeks + close calls)
  const m = getHeadline(selectedSeason);
  const posMeta = getPosMeta(selectedSeason);

  let h = renderPills();
  h += renderHeadlineCard(m);

  if (selectedSeason === "All") {{
    h += renderAllSeasonsTable();
  }}

  h += renderPosTable(posMeta);

  if (selectedSeason !== "All") {{
    h += renderWeeks(selectedSeason);
  }}

  // Close calls section
  const calls = getCallsForView();
  if (calls.length) {{
    h += `<h3 style="margin:16px 0 8px;font-size:0.95rem;color:#555">Close Calls</h3>`;
    h += renderCallsSummary(calls);
    const shown = calls.slice(0, 30);
    h += shown.map(renderCallCard).join("");
    if (calls.length > 30) {{
      h += `<div class="card empty">${{calls.length - 30}} more close calls not shown.</div>`;
    }}
  }}

  el.innerHTML = h;
}}

render();
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    metrics_obj, data_obj, calls_obj, draft_obj, seasons = build_data()
    if not seasons:
        print("No backtest CSVs found in", RESULTS_DIR, file=sys.stderr)
        return 1

    metrics_json = json.dumps(metrics_obj, separators=(",", ":"))
    data_json = json.dumps(data_obj, separators=(",", ":"))
    calls_json = json.dumps(calls_obj, separators=(",", ":"))
    draft_json = json.dumps(draft_obj, separators=(",", ":"))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html_str = build_html(metrics_json, data_json, calls_json, draft_json,
                          seasons, generated_at)

    SITE_DIR.mkdir(exist_ok=True)
    out_path = SITE_DIR / "index.html"
    out_path.write_text(html_str)

    size_kb = len(html_str) / 1024
    print(f"Wrote {out_path} ({size_kb:,.0f} KB, {len(seasons)} seasons)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
