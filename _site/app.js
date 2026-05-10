"use strict";

let posFilter = "All";
let sortKey   = "mr";
let searchQ   = "";

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function getSignal(sp) {
  if (sp <= -25) return { cls: "avoid", label: "FADE" };
  if (sp <= -10) return { cls: "avoid", label: "Fade" };
  if (sp >= 25)  return { cls: "value", label: "Sleeper" };
  if (sp >= 10)  return { cls: "value", label: "Sleeper?" };
  return { cls: "fair", label: "Fair" };
}

function fmtSpread(sp) {
  return sp > 0 ? "+" + sp : String(sp);
}

// ------------------------------------------------------------------
// Data filtering + sorting
// ------------------------------------------------------------------

function filteredAndSorted() {
  let players = BOARD.slice();

  if (posFilter !== "All") {
    players = players.filter(p => p.p === posFilter);
  }

  if (searchQ) {
    const q = searchQ.toLowerCase();
    players = players.filter(p =>
      p.n.toLowerCase().includes(q) ||
      p.t.toLowerCase().includes(q) ||
      (p.role  && p.role.toLowerCase().includes(q)) ||
      (p.usage && p.usage.toLowerCase().includes(q))
    );
  }

  players.sort((a, b) => {
    const va = a[sortKey], vb = b[sortKey];
    if (va == null) return  1;
    if (vb == null) return -1;
    // mr / ecr: lower rank number = better → ascending
    if (sortKey === "mr" || sortKey === "ecr") return va - vb;
    // sp / proj / vorp: higher = better → descending
    return vb - va;
  });

  return players;
}

// ------------------------------------------------------------------
// Card rendering
// ------------------------------------------------------------------

function renderCard(p) {
  const sig    = getSignal(p.sp);
  const spStr  = fmtSpread(p.sp);
  const spCls  = p.sp > 5 ? "pos" : p.sp < -5 ? "neg" : "neu";
  const cardCls = p.sp >= 10 ? "signal-value" : p.sp <= -10 ? "signal-avoid" : "";

  const roleDepthCap = { QB: 2, RB: 3, WR: 3, TE: 2 };
  const roleNum = p.role ? parseInt(p.role.replace(/\D/g, ""), 10) : 99;
  const showRole = p.role && roleNum <= (roleDepthCap[p.p] || 3);
  const roleTag  = showRole ? `<span class="role-tag">${p.role}</span>` : "";
  const usageNote = p.usage ? `<span style="font-size:0.6rem;color:var(--text-dim)">${p.usage}</span>` : "";

  return `<div class="player-card ${cardCls}">
  <div class="card-top">
    <span class="pos-badge pos-${p.p}">${p.p}</span>
    <span class="card-rank">#${p.mr}</span>
  </div>
  <div class="player-name" title="${p.n}">${p.n}</div>
  <div class="player-team">${p.t}${roleTag}</div>
  <div class="card-stats">
    <div class="stat-item" title="Projected fantasy points this season">
      <span class="stat-label">Proj Pts</span>
      <span class="stat-value">${p.proj}</span>
    </div>
    <div class="stat-item" title="Average draft position (expert consensus rank)">
      <span class="stat-label">ADP Rank</span>
      <span class="stat-value">${p.ecr}</span>
    </div>
    <div class="stat-item" title="Points above a freely available replacement player at this position">
      <span class="stat-label">Above Repl.</span>
      <span class="stat-value">${p.vorp}</span>
    </div>
    <div class="stat-item" title="How many spots our model ranks them vs expert consensus — positive means undervalued">
      <span class="stat-label">vs Experts</span>
      <span class="stat-value ${spCls}">${spStr}</span>
    </div>
  </div>
  <div class="signal-row">
    <span class="signal-badge ${sig.cls}">${sig.label}</span>
    ${usageNote}
  </div>
</div>`;
}

// ------------------------------------------------------------------
// Section renderers
// ------------------------------------------------------------------

function renderPills() {
  return ["All", "QB", "RB", "WR", "TE"].map(pos => {
    const count = pos === "All"
      ? BOARD.filter(p => Math.abs(p.sp) >= 10).length
      : BOARD.filter(p => p.p === pos && Math.abs(p.sp) >= 10).length;
    const label = count
      ? `${pos} <span style="opacity:0.6;font-size:0.75em">(${count})</span>`
      : pos;
    const active = posFilter === pos ? " active" : "";
    return `<button class="pill${active}" onclick="setPos('${pos}')">${label}</button>`;
  }).join("");
}

function renderStatsStrip() {
  const v           = META.validation || {};
  const fadesAll    = BOARD.filter(p => p.sp <= -10).length;
  const fadesStrong = BOARD.filter(p => p.sp <= -25).length;
  const sleepAll    = BOARD.filter(p => p.sp >= 10).length;
  const sleepStrong = BOARD.filter(p => p.sp >= 25).length;

  let h = "";
  if (v.n > 0) {
    const cls = v.acc >= 55 ? " value" : "";
    h += `<div class="stat-chip${cls}">
      <span class="stat-chip-value">${v.acc}%</span>
      <span class="stat-chip-label">${META.has_actuals ? "This season" : "Historical"} accuracy<br>${v.wins}/${v.n} calls</span>
    </div>`;
  }
  h += `<div class="stat-chip avoid">
    <span class="stat-chip-value">${fadesAll}</span>
    <span class="stat-chip-label">Overvalued<br>${fadesStrong} strong FADE</span>
  </div>`;
  h += `<div class="stat-chip value">
    <span class="stat-chip-value">${sleepAll}</span>
    <span class="stat-chip-label">Undervalued<br>${sleepStrong} strong Sleeper</span>
  </div>`;
  return h;
}

function renderSources() {
  const src = META.sources || [];
  if (!src.length) return "";

  const avail = src.filter(s => s.status === "available").length;
  const dots  = src.map(s => {
    const dot = s.status === "available" ? '<span style="color:var(--value)">●</span>'
              : s.status === "partial"   ? '<span style="color:var(--wr)">●</span>'
              :                           '<span style="color:var(--border)">●</span>';
    return `<span>${dot} ${s.name}</span>`;
  }).join("");

  return `<div class="sources-row">
    <details>
      <summary>▶ Data sources (${avail}/${src.length} available)</summary>
      <div class="source-dots">${dots}</div>
    </details>
  </div>`;
}

function renderCards() {
  const players = filteredAndSorted();
  if (!players.length) {
    return `<div class="no-results">No players match your search.</div>`;
  }
  return players.map(renderCard).join("");
}

// ------------------------------------------------------------------
// State setters (called from template event handlers)
// ------------------------------------------------------------------

function setPos(pos) {
  posFilter = pos;
  document.getElementById("pills").innerHTML = renderPills();
  updateCards();
}

function setSort(val) {
  sortKey = val;
  updateCards();
}

function setSearch(val) {
  searchQ = val;
  updateCards();
}

function updateCards() {
  const grid = document.getElementById("cardGrid");
  if (grid) grid.innerHTML = renderCards();
}

// ------------------------------------------------------------------
// Boot
// ------------------------------------------------------------------

function render() {
  document.getElementById("pills").innerHTML      = renderPills();
  document.getElementById("statsStrip").innerHTML = renderStatsStrip();
  updateCards();
}

// Route any printable keypress to the search box
document.addEventListener("keydown", function(e) {
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "Escape") {
    searchQ = "";
    const box = document.getElementById("searchInput");
    if (box) { box.value = ""; updateCards(); }
  } else if (e.key.length === 1) {
    const box = document.getElementById("searchInput");
    if (box) box.focus();
  }
});

render();
