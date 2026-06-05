const $ = (id) => document.getElementById(id);

let _data = null;

// ── random matchup on load + button ───────────────────────────────────────
function pickRandom() {
  const shuffled = [...ALL_TEAMS].sort(() => Math.random() - 0.5);
  $("away-select").value = shuffled[0];
  $("home-select").value = shuffled[1];
}

// auto-random on first load
pickRandom();
$("random-btn").addEventListener("click", pickRandom);

// ── step 1 → 2: load matchup ──────────────────────────────────────────────
$("load-btn").addEventListener("click", loadMatchup);

async function loadMatchup() {
  const home = $("home-select").value;
  const away = $("away-select").value;
  const err  = $("pick-error");

  if (home === away) {
    err.textContent = "Pick two different teams.";
    err.hidden = false;
    return;
  }
  err.hidden = true;
  $("load-btn").textContent = "Loading…";

  const fd = new FormData();
  fd.append("home_team", home);
  fd.append("away_team", away);
  const res  = await fetch("/predict", { method: "POST", body: fd });
  const data = await res.json();
  $("load-btn").textContent = "Show matchup →";

  if (!res.ok) {
    err.textContent = data.error || "Something went wrong.";
    err.hidden = false;
    return;
  }

  _data = data;
  showMatchup(data);
}

function applyTeamStyle(panelId, bgId, abbr, meta) {
  const panel = $(panelId);
  const bg    = $(bgId);
  if (meta && meta.color) {
    // darken the team color heavily for the panel background
    panel.style.background = shadeColor(meta.color, -78);
  }
  if (meta && meta.logo) {
    bg.style.backgroundImage = `url('${meta.logo}')`;
  }
}

function showMatchup(d) {
  $("header-away").textContent = d.away_team;
  $("header-home").textContent = d.home_team;
  $("away-team-name").textContent = d.away_team;
  $("home-team-name").textContent = d.home_team;

  applyTeamStyle("away-panel", "away-bg", d.away_team, d.away_meta);
  applyTeamStyle("home-panel", "home-bg", d.home_team, d.home_meta);

  // style pick buttons with team colors
  const awayBtn = document.querySelector(".away-pick-btn");
  const homeBtn = document.querySelector(".home-pick-btn");
  if (d.away_meta?.color) awayBtn.style.background = d.away_meta.color;
  if (d.home_meta?.color) homeBtn.style.background = d.home_meta.color;

  const TEASER = ["Passing yds/g", "Rushing yds/g", "Points scored/g", "Points allowed/g"];
  fillTeaser("away-teaser", d.away_stats, TEASER);
  fillTeaser("home-teaser", d.home_stats, TEASER);

  show("screen-matchup");
}

function fillTeaser(id, stats, keys) {
  $(id).innerHTML = keys
    .map(k => `<tr><td>${k}</td><td>${stats[k]}</td></tr>`)
    .join("");
}

// ── shuffle on matchup screen ──────────────────────────────────────────────
$("shuffle-btn").addEventListener("click", () => {
  pickRandom();
  show("screen-pick");
  // auto-trigger load after a tick so selects update
  setTimeout(() => $("load-btn").click(), 50);
});

// ── step 2 → 3: pick ─────────────────────────────────────────────────────
document.querySelectorAll(".pick-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    if (!_data) return;
    showReveal(_data, btn.dataset.pick);
  });
});

function showReveal(d, userPick) {
  const homePct  = Math.round(d.home_win_prob * 100);
  const awayPct  = 100 - homePct;
  const modelPick = homePct >= 50 ? "home" : "away";
  const winner   = modelPick === "home"
    ? `${d.home_team} (home)` : `${d.away_team} (away)`;

  const banner = $("verdict-banner");
  if (userPick === modelPick) {
    banner.className = "verdict-banner agree";
    banner.textContent = `✓ Model agrees — it also favors ${winner}`;
  } else {
    banner.className = "verdict-banner disagree";
    banner.textContent = `✗ Model disagrees — it favors ${winner}`;
  }

  $("rev-away").textContent = d.away_team;
  $("rev-home").textContent = d.home_team;

  $("rev-away-fill").style.width = awayPct + "%";
  $("rev-home-fill").style.width = homePct + "%";
  $("rev-away-label").textContent = `${awayPct}% ${d.away_team}`;
  $("rev-home-label").textContent = `${d.home_team} ${homePct}%`;

  // apply team colors to prob bar
  if (d.away_meta?.color) $("rev-away-fill").style.background = d.away_meta.color;
  if (d.home_meta?.color) $("rev-home-fill").style.background = d.home_meta.color;

  // card accent borders
  document.querySelector(".away-card").style.borderTopColor = d.away_meta?.color || "var(--away)";
  document.querySelector(".home-card").style.borderTopColor = d.home_meta?.color || "var(--home)";

  $("rev-away-name").textContent = `${d.away_team} (away)`;
  $("rev-home-name").textContent = `${d.home_team} (home)`;
  fillStats("rev-away-stats", d.away_stats);
  fillStats("rev-home-stats", d.home_stats);
  fillPlayers("rev-away-players", d.away_players);
  fillPlayers("rev-home-players", d.home_players);
  fillFactors(d.shap_factors);

  show("screen-reveal");
}

// ── back / reset ──────────────────────────────────────────────────────────
$("back-btn").addEventListener("click",  () => { _data = null; show("screen-pick"); });
$("reset-btn").addEventListener("click", () => { _data = null; pickRandom(); show("screen-pick"); });

// ── helpers ───────────────────────────────────────────────────────────────
function show(id) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  $(id).classList.add("active");
  window.scrollTo(0, 0);
}

function fillStats(id, stats) {
  $(id).innerHTML = Object.entries(stats)
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
}

function fillPlayers(id, players) {
  $(id).innerHTML = players
    .map(p => `<li>
      <span>${p.name} <span class="pos">${p.position}</span></span>
      <span class="stat">${p.stat}</span>
    </li>`).join("");
}

function fillFactors(factors) {
  $("rev-factors").innerHTML = factors
    .map(f => {
      const side = f.direction === "home" ? "home" : "away";
      const word = f.direction === "home" ? "toward home" : "toward away";
      return `<li><span class="chip ${side}">${word}</span><span>${f.feature}</span></li>`;
    }).join("");
}

// ── show why panel ────────────────────────────────────────────────────────
let _modelStats = null;

$("why-btn").addEventListener("click", async () => {
  const panel = $("why-panel");
  const btn   = $("why-btn");
  const open  = !panel.hidden;

  if (open) {
    panel.hidden = true;
    $("why-btn-text").textContent = "See how the model works";
    btn.setAttribute("aria-expanded", "false");
    return;
  }

  // fetch split-stat numbers once
  if (!_modelStats) {
    const res = await fetch("/model-stats");
    _modelStats = await res.json();
    $("why-trained-on").textContent = _modelStats.trained_on.split(" ")[0];
    $("why-tested-on").textContent  = _modelStats.tested_on.split(" ")[0];
  }

  panel.hidden = false;
  $("why-btn-text").textContent = "Hide the details";
  btn.setAttribute("aria-expanded", "true");
});

// ── lightbox: click a chart to enlarge ──────────────────────────────────────
document.querySelectorAll(".why-viz").forEach(fig => {
  fig.addEventListener("click", () => {
    const src = fig.dataset.zoom;
    if (!src) return;
    $("lightbox-img").src = src;
    $("lightbox").hidden = false;
  });
});
function closeLightbox() { $("lightbox").hidden = true; $("lightbox-img").src = ""; }
$("lightbox").addEventListener("click", closeLightbox);
$("lightbox-close").addEventListener("click", closeLightbox);
document.addEventListener("keydown", e => { if (e.key === "Escape") closeLightbox(); });

// darken a hex color by `pct` percent (negative = darker)
function shadeColor(hex, pct) {
  const n = parseInt(hex.replace("#",""), 16);
  const r = Math.min(255, Math.max(0, (n >> 16) + pct));
  const g = Math.min(255, Math.max(0, ((n >> 8) & 0xff) + pct));
  const b = Math.min(255, Math.max(0, (n & 0xff) + pct));
  return `rgb(${r},${g},${b})`;
}
