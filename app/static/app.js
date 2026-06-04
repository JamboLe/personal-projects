const $ = (id) => document.getElementById(id);

$("predict-btn").addEventListener("click", async () => {
  const home = $("home").value;
  const away = $("away").value;
  const errorEl = $("error");
  const resultEl = $("result");

  errorEl.hidden = true;

  const body = new FormData();
  body.append("home_team", home);
  body.append("away_team", away);

  const res = await fetch("/predict", { method: "POST", body });
  const data = await res.json();

  if (!res.ok) {
    errorEl.textContent = data.error || "Something went wrong.";
    errorEl.hidden = false;
    resultEl.hidden = true;
    return;
  }

  render(data);
  resultEl.hidden = false;
});

function render(d) {
  const homePct = Math.round(d.home_win_prob * 100);
  const awayPct = 100 - homePct;

  $("home-fill").style.width = homePct + "%";
  $("away-fill").style.width = awayPct + "%";
  $("home-prob-label").textContent = `${d.home_team} ${homePct}%`;
  $("away-prob-label").textContent = `${awayPct}% ${d.away_team}`;

  $("home-name").textContent = `${d.home_team} (home)`;
  $("away-name").textContent = `${d.away_team} (away)`;

  fillStats("home-stats", d.home_stats);
  fillStats("away-stats", d.away_stats);
  fillPlayers("home-players", d.home_players);
  fillPlayers("away-players", d.away_players);
  fillFactors(d.shap_factors);
}

function fillStats(id, stats) {
  $(id).innerHTML = Object.entries(stats)
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
    .join("");
}

function fillPlayers(id, players) {
  $(id).innerHTML = players
    .map((p) => `<li><span>${p.name} <span class="pos">${p.position}</span></span>
      <span class="stat">${p.stat}</span></li>`)
    .join("");
}

function fillFactors(factors) {
  $("shap-factors").innerHTML = factors
    .map((f) => {
      const pct = (Math.abs(f.impact) * 100).toFixed(0);
      const side = f.direction === "home" ? "home" : "away";
      const word = f.direction === "home" ? "toward home" : "toward away";
      return `<li><span class="chip ${side}">${word}</span>
        <span>${f.feature}</span></li>`;
    })
    .join("");
}
