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
    return metrics_obj, data_obj, seasons_with_data


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def build_html(metrics_json: str, data_json: str, seasons: List[int], generated_at: str) -> str:
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
  <div class="subtitle">Model predictions vs actual results</div>
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

let selectedSeason = SEASONS[SEASONS.length - 1];
let selPos = new Set(["QB","RB","WR","TE"]);
const ALL_POS = ["QB","RB","WR","TE"];

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
  SEASONS.concat(["All"]).forEach(s => {{
    const btn = document.createElement("button");
    btn.className = "tab" + (s === selectedSeason ? " active" : "");
    btn.textContent = s;
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

function render() {{
  renderTabs();
  const el = document.getElementById("content");
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
    metrics_obj, data_obj, seasons = build_data()
    if not seasons:
        print("No backtest CSVs found in", RESULTS_DIR, file=sys.stderr)
        return 1

    metrics_json = json.dumps(metrics_obj, separators=(",", ":"))
    data_json = json.dumps(data_obj, separators=(",", ":"))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html_str = build_html(metrics_json, data_json, seasons, generated_at)

    SITE_DIR.mkdir(exist_ok=True)
    out_path = SITE_DIR / "index.html"
    out_path.write_text(html_str)

    size_kb = len(html_str) / 1024
    print(f"Wrote {out_path} ({size_kb:,.0f} KB, {len(seasons)} seasons)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
