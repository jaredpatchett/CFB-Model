#!/usr/bin/env python3
"""
Render docs/data/latest.json into a single self-contained HTML dashboard
(docs/dashboard.html) — real games, real team logos/colors, real
moneyline/spread/total lines. No external API calls at view-time (the data
is baked in as a JSON blob), since embedding live API keys in client-side
HTML would expose them to anyone who opens the page.

Regenerate whenever docs/data/latest.json is refreshed (i.e. after a new
GitHub Actions run):

  python scripts/build_dashboard.py
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "docs/data/latest.json"
OUT_PATH = "docs/dashboard.html"

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CFB Model — Live Slate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Barlow+Semi+Condensed:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --turf-black: #081810;
    --turf-dark: #0e2418;
    --turf-mid: #163a26;
    --turf-line: rgba(245,242,232,0.05);
    --chalk: #f4f1e6;
    --chalk-dim: #b8c4b9;
    --gold: #ffb81c;
    --gold-dim: #cf9a1d;
    --live-red: #e2453c;
    --card-shadow: 0 18px 40px -12px rgba(0,0,0,0.55);
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--turf-black);
    color: var(--chalk);
    font-family: 'Barlow Semi Condensed', sans-serif;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
  }

  /* Yard-line field texture */
  body::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
      repeating-linear-gradient(
        100deg,
        var(--turf-dark) 0px, var(--turf-dark) 130px,
        var(--turf-mid) 130px, var(--turf-mid) 133px,
        var(--turf-dark) 133px, var(--turf-dark) 260px
      );
    z-index: -2;
  }
  body::after {
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse at 50% -10%, rgba(255,184,28,0.10), transparent 45%),
                radial-gradient(ellipse at 100% 110%, rgba(0,0,0,0.55), transparent 60%);
    z-index: -1;
    pointer-events: none;
  }

  .grain {
    position: fixed; inset: 0; z-index: -1; opacity: 0.05; pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }

  header {
    padding: 56px 6vw 32px;
    text-align: center;
    position: relative;
  }
  .hashmark-row {
    display: flex; justify-content: center; gap: 10px; margin-bottom: 18px;
    opacity: 0.55;
  }
  .hashmark-row span { width: 3px; height: 18px; background: var(--gold-dim); }

  .kicker {
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    font-size: 13px;
    color: var(--gold);
    font-weight: 600;
  }
  h1 {
    font-family: 'Anton', sans-serif;
    font-size: clamp(42px, 8vw, 92px);
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin: 8px 0 4px;
    line-height: 0.95;
    color: var(--chalk);
    text-shadow: 0 0 40px rgba(255,184,28,0.15);
  }
  h1 .accent { color: var(--gold); }
  .subhead {
    color: var(--chalk-dim);
    font-size: 17px;
    max-width: 640px;
    margin: 12px auto 0;
    line-height: 1.5;
  }
  .meta-strip {
    display: flex; justify-content: center; gap: 28px; margin-top: 24px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--chalk-dim); flex-wrap: wrap;
  }
  .meta-strip b { color: var(--gold); font-weight: 700; }

  .controls {
    max-width: 720px; margin: 36px auto 0; padding: 0 6vw;
    display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
  }
  #search {
    flex: 1; min-width: 220px;
    background: rgba(20,45,30,0.7);
    border: 1px solid rgba(255,184,28,0.25);
    color: var(--chalk);
    padding: 13px 18px;
    border-radius: 999px;
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 15px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  #search::placeholder { color: var(--chalk-dim); }
  #search:focus { border-color: var(--gold); box-shadow: 0 0 0 4px rgba(255,184,28,0.12); }

  main { max-width: 1160px; margin: 0 auto; padding: 20px 6vw 100px; }

  .day-header {
    display: flex; align-items: baseline; gap: 14px;
    margin: 46px 0 20px;
  }
  .day-header:first-of-type { margin-top: 12px; }
  .day-header .day-title {
    font-family: 'Anton', sans-serif;
    font-size: 26px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--gold);
    white-space: nowrap;
  }
  .day-header .day-line { flex: 1; height: 2px; background: linear-gradient(90deg, rgba(255,184,28,0.5), transparent); }
  .day-header .day-count {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--chalk-dim); white-space: nowrap;
  }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }

  .card {
    background: linear-gradient(165deg, rgba(22,58,38,0.75), rgba(10,26,17,0.9));
    border: 1px solid rgba(244,241,230,0.08);
    border-radius: 14px;
    padding: 18px 20px 16px;
    box-shadow: var(--card-shadow);
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, border-color 0.18s ease;
    animation: rise 0.5s ease backwards;
  }
  .card:hover { transform: translateY(-3px); border-color: rgba(255,184,28,0.35); }
  .card .split-bar {
    position: absolute; top: 0; left: 0; right: 0; height: 5px;
    display: flex;
  }
  .card .split-bar span { flex: 1; }

  @keyframes rise {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .matchup { display: flex; align-items: center; justify-content: space-between; margin: 14px 0 12px; gap: 10px; }
  .team { display: flex; flex-direction: column; align-items: center; gap: 8px; width: 40%; text-align: center; }
  .team .badge {
    width: 58px; height: 58px; border-radius: 50%;
    background: var(--chalk);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.4), 0 0 0 3px rgba(0,0,0,0.15) inset;
    overflow: hidden;
  }
  .team .badge img { width: 40px; height: 40px; object-fit: contain; }
  .team .badge .initials {
    font-family: 'Anton', sans-serif; font-size: 17px; color: var(--turf-dark);
  }
  .team .name {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 13px;
    line-height: 1.15;
    color: var(--chalk);
  }
  .team .tag {
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--chalk-dim);
  }
  .at {
    font-family: 'Anton', sans-serif;
    color: var(--gold-dim);
    font-size: 20px;
  }

  .kickoff {
    text-align: center;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--chalk-dim);
    margin-bottom: 14px;
  }

  .lines {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    border-top: 1px dashed rgba(244,241,230,0.15);
    padding-top: 12px;
  }
  .lines .stat { text-align: center; }
  .lines .stat .label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--gold-dim); margin-bottom: 4px;
  }
  .lines .stat .value {
    font-family: 'Anton', sans-serif;
    font-size: 17px;
    color: var(--chalk);
  }
  .lines .stat .value small { font-size: 11px; color: var(--chalk-dim); font-family: 'Barlow Semi Condensed', sans-serif; }

  .book-tag {
    text-align: center; margin-top: 10px;
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--chalk-dim);
  }

  .empty-state {
    margin: 50px auto 0; max-width: 560px; text-align: center;
    background: rgba(22,58,38,0.4); border: 1px dashed rgba(255,184,28,0.3);
    border-radius: 16px; padding: 32px 28px;
  }
  .empty-state .icon { font-family: 'Anton', sans-serif; font-size: 34px; color: var(--gold); margin-bottom: 8px; }
  .empty-state p { color: var(--chalk-dim); line-height: 1.6; font-size: 15px; }

  #no-results {
    text-align: center; color: var(--chalk-dim); padding: 60px 0; display: none;
    font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.05em; text-transform: uppercase;
  }

  footer {
    text-align: center; padding: 36px 6vw 60px;
    color: var(--chalk-dim); font-size: 12px; line-height: 1.8;
    border-top: 1px solid rgba(244,241,230,0.08);
    max-width: 780px; margin: 0 auto;
  }
  footer b { color: var(--gold-dim); }
</style>
</head>
<body>
<div class="grain"></div>

<header>
  <div class="hashmark-row"><span></span><span></span><span></span><span></span><span></span></div>
  <div class="kicker">College Football &middot; Model Output</div>
  <h1>THE <span class="accent">SLATE</span></h1>
  <p class="subhead">Live market lines pulled from real sportsbooks, matched against team data. Research/decision-support only &mdash; see disclaimer below before acting on anything here.</p>
  <div class="meta-strip">
    <div>Games tracked: <b id="game-count">0</b></div>
    <div>Snapshot: <b id="generated-at">&mdash;</b></div>
  </div>
  <div class="controls">
    <input id="search" type="text" placeholder="Search a team, e.g. &quot;Michigan&quot; or &quot;Alabama&quot;&hellip;" autocomplete="off">
  </div>
</header>

<main id="main"></main>

<footer>
  Built from <b>CFBD</b> (team data), <b>The Odds API</b> (moneyline/spread/total), and <b>PrizePicks via OddsPapi</b> (player props).
  Snapshot only &mdash; regenerate after each data pull to refresh.<br><br>
  No model reliably beats a well-priced sportsbook line on every game. Treat this as research support, not a guarantee.
  Bet only what you're comfortable losing.
</footer>

<script id="cfb-data" type="application/json">__DATA_JSON__</script>
<script>
  const raw = JSON.parse(document.getElementById('cfb-data').textContent);
  const games = raw.games || [];
  const props = raw.props || [];

  document.getElementById('game-count').textContent = games.length;
  document.getElementById('generated-at').textContent = raw.generated_at
    ? new Date(raw.generated_at).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })
    : 'unknown';

  function initials(name) {
    if (!name) return '?';
    return name.split(' ').filter(w => w[0] === w[0].toUpperCase()).slice(0,2).map(w => w[0]).join('') || name[0];
  }

  function fmtOdds(v) {
    if (v === null || v === undefined || v === '' || Number.isNaN(v)) return '&mdash;';
    const n = Number(v);
    return n > 0 ? '+' + n : String(n);
  }
  function fmtSpread(v) {
    if (v === null || v === undefined || v === '' || Number.isNaN(v)) return '&mdash;';
    const n = Number(v);
    return (n > 0 ? '+' : '') + n;
  }
  function fmtTotal(v) {
    if (v === null || v === undefined || v === '') return '&mdash;';
    return 'O/U ' + v;
  }

  function teamBadge(name, logo, color) {
    const ring = color ? `box-shadow: 0 4px 14px rgba(0,0,0,0.4), 0 0 0 3px ${color}55 inset;` : '';
    if (logo) {
      return `<div class="badge" style="${ring}"><img src="${logo}" alt="${name}" onerror="this.parentElement.innerHTML='<span class=\\'initials\\'>${initials(name)}</span>'"></div>`;
    }
    return `<div class="badge" style="${ring}"><span class="initials">${initials(name)}</span></div>`;
  }

  function dayLabel(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });
  }

  function cardHTML(g) {
    const homeColor = g.home_color || '#163a26';
    const awayColor = g.away_color || '#163a26';
    const time = new Date(g.commence_time).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    return `
      <div class="card" data-teams="${(g.home_team + ' ' + g.away_team).toLowerCase()}">
        <div class="split-bar"><span style="background:${awayColor}"></span><span style="background:${homeColor}"></span></div>
        <div class="kickoff">${time} &middot; ${g.book_used ? g.book_used.toUpperCase() : 'MARKET'}</div>
        <div class="matchup">
          <div class="team">
            ${teamBadge(g.away_team, g.away_logo, g.away_color)}
            <div class="name">${g.away_team}</div>
            <div class="tag">Away</div>
          </div>
          <div class="at">@</div>
          <div class="team">
            ${teamBadge(g.home_team, g.home_logo, g.home_color)}
            <div class="name">${g.home_team}</div>
            <div class="tag">Home</div>
          </div>
        </div>
        <div class="lines">
          <div class="stat">
            <div class="label">Spread</div>
            <div class="value">${fmtSpread(g.spread_home)}<br><small>${fmtSpread(g.spread_away)}</small></div>
          </div>
          <div class="stat">
            <div class="label">Moneyline</div>
            <div class="value">${fmtOdds(g.moneyline_home)}<br><small>${fmtOdds(g.moneyline_away)}</small></div>
          </div>
          <div class="stat">
            <div class="label">Total</div>
            <div class="value">${fmtTotal(g.total_over)}</div>
          </div>
        </div>
      </div>`;
  }

  function render(filterText) {
    const main = document.getElementById('main');
    main.innerHTML = '';
    const term = (filterText || '').trim().toLowerCase();
    const filtered = games.filter(g => !term || (g.home_team + ' ' + g.away_team).toLowerCase().includes(term));

    if (!filtered.length) {
      main.innerHTML = '<div id="no-results" style="display:block">No games match that search.</div>';
      return;
    }

    const byDay = {};
    filtered.slice().sort((a,b) => new Date(a.commence_time) - new Date(b.commence_time)).forEach(g => {
      const day = g.commence_time.slice(0,10);
      (byDay[day] = byDay[day] || []).push(g);
    });

    Object.keys(byDay).sort().forEach(day => {
      const dayGames = byDay[day];
      const section = document.createElement('section');
      section.innerHTML = `
        <div class="day-header">
          <div class="day-title">${dayLabel(dayGames[0].commence_time)}</div>
          <div class="day-line"></div>
          <div class="day-count">${dayGames.length} game${dayGames.length === 1 ? '' : 's'}</div>
        </div>
        <div class="grid">${dayGames.map(cardHTML).join('')}</div>
      `;
      main.appendChild(section);
    });

    if (!props.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.innerHTML = `
        <div class="icon">PROPS</div>
        <p>No PrizePicks player props are posted yet for this slate. Prop lines for college football typically
        open closer to kickoff week &mdash; re-run the data pull nearer game day to populate this section.</p>`;
      main.appendChild(empty);
    }
  }

  document.getElementById('search').addEventListener('input', (e) => render(e.target.value));
  render('');
</script>
</body>
</html>
"""


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    data_json = json.dumps(data).replace("</", "<\\/")  # avoid breaking out of <script>
    html = TEMPLATE.replace("__DATA_JSON__", data_json)

    with open(OUT_PATH, "w") as f:
        f.write(html)

    print(f"Wrote {OUT_PATH} ({len(data.get('games', []))} games, {len(data.get('props', []))} props)")


if __name__ == "__main__":
    main()
