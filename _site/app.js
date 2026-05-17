"use strict";

let posFilter    = "All";
let sortKey      = "mr";
let searchQ      = "";
let draftMode    = false;
let watchlistIds = [];
let draftPick    = 1;

const WATCHLIST_KEY = "draftAdvisor.watchlist";
const DRAFT_MODE_KEY = "draftAdvisor.draftMode";

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

function loadPrefs() {
  try {
    const savedWatchlist = JSON.parse(window.localStorage.getItem(WATCHLIST_KEY) || "[]");
    if (Array.isArray(savedWatchlist)) watchlistIds = savedWatchlist;
    draftMode = window.localStorage.getItem(DRAFT_MODE_KEY) === "1";
  } catch (_) {
    watchlistIds = [];
    draftMode = false;
  }
}

function saveWatchlist() {
  try {
    window.localStorage.setItem(WATCHLIST_KEY, JSON.stringify(watchlistIds));
  } catch (_) {}
}

function saveDraftMode() {
  try {
    window.localStorage.setItem(DRAFT_MODE_KEY, draftMode ? "1" : "0");
  } catch (_) {}
}

function isWatched(playerId) {
  return watchlistIds.includes(playerId);
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
  const seasonPprTitle = "Projected full-season PPR fantasy points";
  const aboveReplTitle = "Points above a freely available replacement player at this position";
  const vsExpertsTitle = "How many spots our model ranks them versus expert consensus. Positive means undervalued";

  const roleDepthCap = { QB: 2, RB: 3, WR: 3, TE: 2 };
  const roleNum = p.role ? parseInt(p.role.replace(/\D/g, ""), 10) : 99;
  const showRole = p.role && roleNum <= (roleDepthCap[p.p] || 3);
  const roleTag  = showRole ? `<span class="role-tag">${p.role}</span>` : "";
  const usageNote = p.usage ? `<span style="font-size:0.6rem;color:var(--text-dim)">${p.usage}</span>` : "";
  const adjBadge  = p.adj_note ? `<div class="adj-badge" title="Manual adjustment">✎ ${p.adj_note}</div>` : "";
  const whyItems = Array.isArray(p.why) ? p.why.filter(Boolean).slice(0, 4) : [];
  const whyBlock = whyItems.length ? `<details class="why-expander">
    <summary>Why</summary>
    <ul class="why-list">${whyItems.map(item => `<li>${item}</li>`).join("")}</ul>
  </details>` : "";
  const mktBlock = (p.mkt25 != null && p.edge != null) ? (() => {
    const edgeCls = p.edge > 0 ? "pos" : p.edge < 0 ? "neg" : "";
    const edgeStr = p.edge > 0 ? `+${p.edge}` : String(p.edge);
    return `<div class="mkt-edge-row">
      <span class="mkt-label">Mkt '25: ${p.mkt25} fp</span>
      <span class="stat-value ${edgeCls}" title="Model 2026 projection minus 2025 market-implied FP">edge ${edgeStr}</span>
    </div>`;
  })() : "";
  const watched = isWatched(p.id);
  const watchBtn = `<button class="queue-btn${watched ? " active" : ""}" type="button" onclick="toggleWatch(${p.id})">${watched ? "Queued" : "Queue"}</button>`;

  return `<div class="player-card ${cardCls}">
  <div class="card-top">
    <span class="pos-badge pos-${p.p}">${p.p}</span>
    <div class="card-actions">
      <span class="card-rank">#${p.mr}</span>
      ${watchBtn}
    </div>
  </div>
  <div class="player-name" title="${p.n}">${p.n}</div>
  <div class="player-team">${p.t}${roleTag}</div>
  <div class="card-stats">
    <div class="stat-item" title="${seasonPprTitle}">
      <span class="stat-label def-term" title="${seasonPprTitle}">Season PPR</span>
      <span class="stat-value">${p.proj}</span>
    </div>
    <div class="stat-item" title="Average draft position (expert consensus rank)">
      <span class="stat-label">ADP Rank</span>
      <span class="stat-value">${p.ecr}</span>
    </div>
    <div class="stat-item" title="${aboveReplTitle}">
      <span class="stat-label def-term" title="${aboveReplTitle}">Above Repl.</span>
      <span class="stat-value">${p.vorp}</span>
    </div>
    <div class="stat-item" title="${vsExpertsTitle}">
      <span class="stat-label def-term" title="${vsExpertsTitle}">vs Experts</span>
      <span class="stat-value ${spCls}">${spStr}</span>
    </div>
  </div>
  <div class="signal-row">
    <span class="signal-badge ${sig.cls}">${sig.label}</span>
    ${usageNote}
  </div>
  ${mktBlock}
  ${whyBlock}
  ${adjBadge}
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

function renderScarcityStrip() {
  if (!window.SCARCITY) return "";
  const POSITIONS = ["QB", "RB", "WR", "TE"];
  const teams  = SCARCITY.teams  || 10;
  const rounds = SCARCITY.rounds || 15;
  const maxPick = teams * rounds;
  const cliffs  = SCARCITY.cliffs || {};
  const pickData = (SCARCITY.by_pick || {})[String(draftPick)] || {};

  const tiles = POSITIONS.map(pos => {
    const d = pickData[pos] || { rem: 0, top: 0 };
    const picksToCliff = (cliffs[pos] || maxPick + 1) - draftPick;
    const heat = d.rem <= 3 ? "scar-red" : d.rem <= 6 ? "scar-yellow" : "scar-green";
    const cliffWarn = picksToCliff > 0 && picksToCliff <= teams
      ? `<div class="scar-cliff">cliff ~${picksToCliff}</div>` : "";
    return `<div class="scar-tile ${heat}">
      <div class="scar-pos">${pos}</div>
      <div class="scar-rem">${d.rem}</div>
      <div class="scar-label">left</div>
      ${cliffWarn}
    </div>`;
  }).join("");

  const round = Math.ceil(draftPick / teams);
  const prevPick = Math.max(1, draftPick - teams);
  const nextPick = Math.min(maxPick, draftPick + teams);

  return `<div class="scarcity-strip">
    <div class="scar-header">
      <button class="scar-btn" onclick="setDraftPick(${prevPick})" ${draftPick <= teams ? "disabled" : ""}>◀</button>
      <span class="scar-pick-label">Pick ${draftPick} · Rd ${round}</span>
      <button class="scar-btn" onclick="setDraftPick(${nextPick})" ${draftPick >= maxPick ? "disabled" : ""}>▶</button>
    </div>
    <div class="scar-tiles">${tiles}</div>
    <div class="scar-note">Starters above replacement remaining</div>
  </div>`;
}

function setDraftPick(pick) {
  draftPick = pick;
  const el = document.getElementById("scarcityStrip");
  if (el) el.innerHTML = renderScarcityStrip();
}

function renderDraftPanel() {
  const panel = document.getElementById("draftPanel");
  if (!panel) return;

  if (!draftMode) {
    panel.innerHTML = "";
    panel.classList.remove("active");
    return;
  }

  panel.classList.add("active");
  const watchedPlayers = watchlistIds
    .map(id => BOARD.find(p => p.id === id))
    .filter(Boolean);

  const chips = watchedPlayers.length
    ? watchedPlayers.map(p => `<button class="watch-chip" type="button" onclick="toggleWatch(${p.id})">
        <span>${p.n}</span>
        <span class="watch-chip-meta">${p.p} · #${p.mr}</span>
      </button>`).join("")
    : `<div class="draft-empty">Queue players here while drafting.</div>`;

  panel.innerHTML = `<div class="draft-panel-inner">
    <div class="draft-panel-top">
      <div>
        <div class="draft-panel-title">Draft Mode</div>
        <div class="draft-panel-sub">Big search, quick queue, one-page board.</div>
      </div>
      <div class="draft-panel-count">${watchedPlayers.length} queued</div>
    </div>
    <div class="watch-chip-row">${chips}</div>
  </div>`;
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

function toggleDraftMode() {
  draftMode = !draftMode;
  saveDraftMode();
  renderDraftModeState();
}

function toggleWatch(playerId) {
  if (isWatched(playerId)) {
    watchlistIds = watchlistIds.filter(id => id !== playerId);
  } else {
    watchlistIds = [playerId].concat(watchlistIds.filter(id => id !== playerId)).slice(0, 12);
  }
  saveWatchlist();
  renderDraftPanel();
  updateCards();
}

function updateCards() {
  const grid = document.getElementById("cardGrid");
  if (grid) grid.innerHTML = renderCards();
}

function renderDraftModeState() {
  document.body.classList.toggle("draft-mode", draftMode);
  const btn = document.getElementById("draftModeBtn");
  if (btn) btn.textContent = draftMode ? "Draft Mode On" : "Draft Mode Off";
  renderDraftPanel();
}

// ------------------------------------------------------------------
// Boot
// ------------------------------------------------------------------

function render() {
  loadPrefs();
  document.getElementById("pills").innerHTML         = renderPills();
  document.getElementById("statsStrip").innerHTML    = renderStatsStrip();
  document.getElementById("scarcityStrip").innerHTML = renderScarcityStrip();
  renderDraftModeState();
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
