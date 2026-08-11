#!/usr/bin/env python3
"""
Render docs/data/latest.json into a single self-contained HTML dashboard
(docs/dashboard.html) — real games, real team logos/colors, real
moneyline/spread/total lines, plus a Props tab listing the real player-prop
market catalog (populated with real lines once PrizePicks posts them).

DESIGN v3: "Coach's whiteboard" — light background, hand-drawn marker
typography and route-line diagrams connecting each matchup, highlighter-style
favorite callouts, rubber-stamp "coming soon" props. Deliberately a different
genre from the previous two passes (dark turf/gold, then vintage parchment).

Engineering choices worth knowing about (carried over from earlier fixes):
  - Every card is rendered directly into the HTML by THIS PYTHON SCRIPT, not
    built at page-load time by JavaScript. An earlier version relied on
    client-side JS to build all content from an embedded JSON blob, and in
    some environments that render/preview HTML without executing <script>,
    the page showed up completely blank. Pre-rendering means every game,
    logo, and line is visible even with JavaScript fully disabled.
  - Tabs (Games / Prop Markets) use the CSS-only radio-input technique, not
    JS, for the same reason.

Regenerate whenever docs/data/latest.json is refreshed:

  python scripts/build_dashboard.py
"""
import html
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "docs/data/latest.json"
OUT_PATH = "docs/dashboard.html"


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def initials(name):
    if not name:
        return "?"
    words = [w for w in str(name).split(" ") if w]
    caps = [w[0] for w in words if w[0].isupper()]
    return "".join(caps[:2]) if caps else name[0].upper()


def fmt_odds(v):
    if v is None or v == "" or (isinstance(v, float) and v != v):
        return "&mdash;"
    n = float(v)
    n = int(n) if n == int(n) else n
    return f"+{n}" if n > 0 else f"{n}"


def fmt_total(v):
    if v is None or v == "":
        return "&mdash;"
    n = float(v)
    n = int(n) if n == int(n) else n
    return f"{n}"


def fmt_time(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%-I:%M %p")
    except Exception:
        return ""


def day_label(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%a %b %-d")
    except Exception:
        return "TBD"


def team_token_html(name, logo, color, side):
    color = color or ("#e0433d" if side == "away" else "#2f5fd6")
    if logo:
        inner = (
            f'<img src="{esc(logo)}" alt="{esc(name)} logo" loading="lazy" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">'
            f'<span class="initials" style="display:none">{esc(initials(name))}</span>'
        )
    else:
        inner = f'<span class="initials">{esc(initials(name))}</span>'
    return f'<div class="token token-{side}" style="border-color:{esc(color)}">{inner}</div>'


def card_html(g, idx):
    home_color = g.get("home_color") or "#2f5fd6"
    away_color = g.get("away_color") or "#e0433d"
    home_ml = g.get("moneyline_home")
    away_ml = g.get("moneyline_away")

    home_is_fav = home_ml is not None and away_ml is not None and float(home_ml) < float(away_ml)
    away_is_fav = home_ml is not None and away_ml is not None and float(away_ml) < float(home_ml)

    teams_search_key = esc(f"{str(g.get('home_team',''))} {str(g.get('away_team',''))}".lower())
    tilt = ["-1.1deg", "0.8deg", "-0.4deg", "1.2deg"][idx % 4]

    return f"""
      <div class="card game-card" data-teams="{teams_search_key}" style="--tilt:{tilt}">
        <div class="clip"></div>
        <div class="card-head">
          <span class="game-time">&#9210; {fmt_time(g.get('commence_time',''))}</span>
          <span class="game-book">{esc((g.get('book_used') or 'market').upper())}</span>
        </div>
        <div class="route-diagram">
          {team_token_html(g.get('away_team'), g.get('away_logo'), g.get('away_color'), 'away')}
          <svg class="route-line" viewBox="0 0 100 24" preserveAspectRatio="none">
            <path d="M 4 12 C 30 -6, 70 30, 96 12" fill="none" stroke="{esc(away_color)}" stroke-width="2.5" stroke-dasharray="6 5" stroke-linecap="round"/>
          </svg>
          {team_token_html(g.get('home_team'), g.get('home_logo'), g.get('home_color'), 'home')}
        </div>
        <div class="names">
          <div class="name-block">
            <span class="name">{esc(g.get('away_team'))}</span>
            <span class="role">Away</span>
          </div>
          <span class="vs">at</span>
          <div class="name-block">
            <span class="name">{esc(g.get('home_team'))}</span>
            <span class="role">Home</span>
          </div>
        </div>
        <div class="lines">
          <div class="stat">
            <div class="label">Spread</div>
            <div class="value">{fmt_odds(g.get('spread_away'))} <span class="dim">/</span> {fmt_odds(g.get('spread_home'))}</div>
          </div>
          <div class="stat">
            <div class="label">Moneyline</div>
            <div class="value-row">
              <span class="{'hl-away' if away_is_fav else ''}">{fmt_odds(away_ml)}</span>
              <span class="dim">/</span>
              <span class="{'hl-home' if home_is_fav else ''}">{fmt_odds(home_ml)}</span>
            </div>
          </div>
          <div class="stat">
            <div class="label">Total</div>
            <div class="value">O/U {fmt_total(g.get('total_over'))}</div>
          </div>
        </div>
      </div>"""


def build_game_sections(games):
    by_day = {}
    for g in sorted(games, key=lambda x: x.get("commence_time") or ""):
        day = (g.get("commence_time") or "")[:10]
        by_day.setdefault(day, []).append(g)

    sections = []
    counter = 0
    for day in sorted(by_day.keys()):
        day_games = by_day[day]
        cards = []
        for g in day_games:
            cards.append(card_html(g, counter))
            counter += 1
        sections.append(f"""
        <section>
          <div class="day-header">
            <span class="day-title">{day_label(day_games[0].get('commence_time',''))}</span>
            <span class="day-count">{len(day_games)} game{'s' if len(day_games) != 1 else ''} &#9679;</span>
          </div>
          <div class="grid">{''.join(cards)}</div>
        </section>""")
    return "\n".join(sections)


def prop_row_html(market_name, rows, idx):
    tilt = ["-0.6deg", "0.5deg", "-0.3deg"][idx % 3]
    if rows:
        detail = "".join(f"""
            <div class="prop-line-row">
              <span class="prop-player">{esc(r.get('player_name'))}</span>
              <span class="prop-num">{esc(r.get('line'))}</span>
              <span class="prop-odds">O {esc(r.get('over_price', '&mdash;'))} / U {esc(r.get('under_price', '&mdash;'))}</span>
            </div>""" for r in rows)
        return f"""
        <div class="prop-card is-live" style="--tilt:{tilt}">
          <div class="prop-head">
            <span class="prop-name">{esc(market_name)}</span>
            <span class="live-dot">&#9679; LIVE</span>
          </div>
          {detail}
        </div>"""
    return f"""
        <div class="prop-card is-pending" style="--tilt:{tilt}">
          <div class="stamp">Coming<br>Soon</div>
          <span class="prop-name">{esc(market_name)}</span>
          <p class="prop-note">Confirmed market on PrizePicks &mdash; opens closer to kickoff week.</p>
        </div>"""


def build_props_section(prop_catalog, props):
    if not prop_catalog:
        return """
        <div class="empty-state">
          <p><strong>No catalog loaded yet.</strong> Re-run the data pull to fetch the real player-prop market list.</p>
        </div>"""

    by_market = {}
    for p in props:
        by_market.setdefault(p.get("market_name"), []).append(p)

    live_count = sum(1 for m in prop_catalog if by_market.get(m))
    intro = "" if live_count else """
        <p class="props-intro">These are the <strong>real</strong> player-prop markets PrizePicks offers for college
        football &mdash; confirmed from the live catalog, not guessed. None are priced yet for this slate (normal
        this far from kickoff). Re-run the data pull closer to game week and any market with posted lines flips
        to LIVE automatically.</p>"""

    cards = "".join(prop_row_html(m, by_market.get(m, []), i) for i, m in enumerate(prop_catalog))
    return f'{intro}<div class="prop-grid">{cards}</div>'


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Slate — CFB Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Permanent+Marker&family=Sora:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --paper: #f7f5ee;
    --paper-line: rgba(23,24,26,0.06);
    --ink: #17181a;
    --ink-soft: #55575c;
    --red: #e0433d;
    --blue: #2f5fd6;
    --highlighter: #f5ef4d;
    --card-shadow: 3px 4px 0 rgba(23,24,26,0.9);
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0; background: var(--paper); color: var(--ink);
    font-family: 'Sora', sans-serif; min-height: 100vh; position: relative;
  }}
  body::before {{
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background-image:
      linear-gradient(var(--paper-line) 1px, transparent 1px),
      linear-gradient(90deg, var(--paper-line) 1px, transparent 1px);
    background-size: 34px 34px;
  }}

  #tab-games, #tab-props {{ position: absolute; opacity: 0; pointer-events: none; }}
  .games-panel {{ display: block; }}
  .props-panel {{ display: none; }}
  #tab-props:checked ~ main .games-panel {{ display: none; }}
  #tab-props:checked ~ main .props-panel {{ display: block; }}

  .tab-label {{
    font-family: 'Sora', sans-serif; font-weight: 700; font-size: 14px;
    color: var(--ink); padding: 11px 24px; border: 2.5px solid var(--ink);
    background: #fff; cursor: pointer; border-radius: 999px; box-shadow: var(--card-shadow);
    transition: transform 0.15s;
  }}
  .tab-label:hover {{ transform: translateY(-2px); }}
  #tab-games:checked ~ header .tab-label[for="tab-games"],
  #tab-props:checked ~ header .tab-label[for="tab-props"] {{
    background: var(--ink); color: var(--paper); box-shadow: 3px 4px 0 var(--red);
  }}

  header {{ padding: 54px 6vw 0; text-align: center; }}
  .kicker {{
    display: inline-block; font-weight: 800; font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--ink); background: var(--highlighter); padding: 4px 14px; border: 2px solid var(--ink);
    transform: rotate(-1.5deg); box-shadow: var(--card-shadow);
  }}
  h1 {{
    font-family: 'Permanent Marker', cursive; font-size: clamp(46px, 9vw, 104px); font-weight: 400;
    margin: 20px 0 6px; line-height: 0.95; color: var(--ink); transform: rotate(-0.6deg);
  }}
  h1 .accent {{ color: var(--red); position: relative; }}
  .subhead {{ color: var(--ink-soft); font-size: 16px; max-width: 600px; margin: 10px auto 0; line-height: 1.6; font-weight: 500; }}
  .meta-strip {{
    display: flex; justify-content: center; gap: 24px; margin-top: 22px; flex-wrap: wrap;
    font-size: 13px; font-weight: 600; color: var(--ink-soft);
  }}
  .meta-strip b {{ color: var(--ink); border-bottom: 3px solid var(--highlighter); }}
  .controls {{ max-width: 720px; margin: 28px auto 0; padding: 0 6vw; display: flex; justify-content: center; }}
  #search {{
    width: 100%; max-width: 420px; background: #fff; border: 2.5px solid var(--ink);
    color: var(--ink); padding: 12px 20px; border-radius: 999px; font-family: 'Sora', sans-serif;
    font-weight: 600; font-size: 14.5px; outline: none; box-shadow: var(--card-shadow);
  }}
  #search::placeholder {{ color: #9a9a94; font-weight: 500; }}
  #search:focus {{ transform: translateY(-1px); }}
  .tab-bar {{ margin: 28px 0 0; display: flex; justify-content: center; gap: 10px; }}

  main {{ max-width: 1200px; margin: 0 auto; padding: 40px 6vw 100px; }}
  .day-header {{ display: flex; align-items: baseline; justify-content: space-between; margin: 42px 0 18px; border-bottom: 3px solid var(--ink); padding-bottom: 8px; }}
  section:first-of-type .day-header {{ margin-top: 0; }}
  .day-title {{ font-family: 'Permanent Marker', cursive; font-size: 22px; color: var(--blue); }}
  .day-count {{ font-size: 12px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-soft); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 26px 22px; }}

  .card {{
    background: #fff; border: 2.5px solid var(--ink); border-radius: 10px; padding: 16px 18px 16px;
    box-shadow: var(--card-shadow); position: relative; transform: rotate(var(--tilt, 0deg));
    transition: transform 0.2s ease;
  }}
  .card:hover {{ transform: rotate(0deg) translateY(-3px); }}
  .clip {{
    position: absolute; top: -9px; left: 22px; width: 26px; height: 16px; background: var(--red);
    border: 2px solid var(--ink); border-radius: 3px; transform: rotate(-4deg);
  }}
  .card-head {{ display: flex; justify-content: space-between; font-size: 10.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-soft); margin-bottom: 10px; }}

  .route-diagram {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }}
  .route-line {{ flex: 1; height: 24px; margin: 0 4px; }}
  .token {{
    width: 50px; height: 50px; border-radius: 50%; background: #fff; border: 3px solid;
    display: flex; align-items: center; justify-content: center; overflow: hidden; flex-shrink: 0;
  }}
  .token img {{ width: 34px; height: 34px; object-fit: contain; }}
  .token .initials {{ font-family: 'Permanent Marker', cursive; font-size: 15px; color: var(--ink); }}

  .names {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; gap: 6px; }}
  .name-block {{ display: flex; flex-direction: column; width: 42%; text-align: center; }}
  .name {{ font-weight: 700; font-size: 12.5px; line-height: 1.25; color: var(--ink); }}
  .role {{ font-size: 10px; font-weight: 600; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.05em; }}
  .vs {{ font-family: 'Permanent Marker', cursive; font-size: 13px; color: var(--ink-soft); }}

  .lines {{ display: grid; grid-template-columns: 1fr 1fr 1fr; border-top: 2px dashed var(--ink); padding-top: 12px; gap: 4px; }}
  .lines .stat {{ text-align: center; }}
  .label {{ font-size: 9.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-soft); margin-bottom: 4px; }}
  .value, .value-row {{ font-weight: 800; font-size: 14px; color: var(--ink); }}
  .value-row span {{ padding: 1px 3px; }}
  .value .dim, .value-row .dim {{ color: #b7b6ac; font-weight: 500; }}
  .hl-away, .hl-home {{
    background: var(--highlighter); border-radius: 3px; box-shadow: 2px 2px 0 rgba(23,24,26,0.15);
  }}

  .props-intro {{ max-width: 700px; margin: 0 auto 30px; text-align: center; color: var(--ink-soft); font-size: 14.5px; line-height: 1.6; font-weight: 500; }}
  .prop-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 24px 20px; }}
  .prop-card {{
    background: #fff; border: 2.5px solid var(--ink); border-radius: 10px; padding: 18px 20px;
    box-shadow: var(--card-shadow); position: relative; transform: rotate(var(--tilt, 0deg));
  }}
  .prop-card.is-pending {{ min-height: 150px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }}
  .prop-name {{ font-weight: 700; font-size: 14px; color: var(--ink); }}
  .prop-note {{ font-size: 12px; color: var(--ink-soft); line-height: 1.5; margin: 8px 0 0; font-weight: 500; }}
  .stamp {{
    font-family: 'Permanent Marker', cursive; color: var(--red); border: 3px solid var(--red);
    border-radius: 8px; padding: 6px 14px; font-size: 13px; line-height: 1.15; text-align: center;
    transform: rotate(-9deg); margin-bottom: 10px; opacity: 0.88;
  }}
  .prop-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  .live-dot {{ font-size: 10px; font-weight: 800; color: var(--red); letter-spacing: 0.05em; }}
  .prop-line-row {{ display: flex; justify-content: space-between; gap: 6px; font-size: 12px; padding: 5px 0; border-top: 1px dashed var(--paper-line); }}
  .prop-player {{ font-weight: 700; }}
  .prop-num {{ color: var(--blue); font-weight: 700; }}
  .prop-odds {{ color: var(--ink-soft); }}

  .empty-state {{ margin: 40px auto 0; max-width: 560px; text-align: center; background: #fff; border: 2.5px solid var(--ink); border-radius: 10px; padding: 28px; box-shadow: var(--card-shadow); }}
  #no-results {{ display: none; text-align: center; color: var(--ink-soft); padding: 60px 0; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; font-size: 14px; }}

  footer {{
    text-align: center; padding: 30px 6vw 60px; color: var(--ink-soft); font-size: 12.5px; line-height: 1.8;
    border-top: 3px solid var(--ink); max-width: 800px; margin: 0 auto; font-weight: 500;
  }}
  footer b {{ color: var(--ink); }}
</style>
</head>
<body>

<input type="radio" name="tabs" id="tab-games" checked>
<input type="radio" name="tabs" id="tab-props">

<header>
  <span class="kicker">College Football &middot; Model Output</span>
  <h1>THE <span class="accent">SLATE</span></h1>
  <p class="subhead">Real market lines, matched with real team data. Research/decision-support only &mdash;
  read the disclaimer at the bottom before acting on anything here.</p>
  <div class="meta-strip">
    <div>Games: <b>{game_count}</b></div>
    <div>Prop markets: <b>{market_count}</b></div>
    <div>Snapshot: <b>{generated_at}</b></div>
  </div>
  <div class="controls">
    <input id="search" type="text" placeholder="Search a team, e.g. &quot;Michigan&quot;&hellip;" autocomplete="off" oninput="filterCards(this.value)">
  </div>
  <noscript><span style="display:block;text-align:center;color:var(--ink-soft);font-size:12px;margin-top:8px;">(Search needs JavaScript &mdash; everything else on this page works without it.)</span></noscript>
  <div class="tab-bar">
    <label class="tab-label" for="tab-games">Games</label>
    <label class="tab-label" for="tab-props">Prop Markets</label>
  </div>
</header>

<main>
  <div class="games-panel">
{game_sections}
    <p id="no-results">No games match that search.</p>
  </div>
  <div class="props-panel">
{props_section}
  </div>
</main>

<footer>
  Built from <b>CFBD</b> (team data), <b>The Odds API</b> (moneyline/spread/total), and <b>PrizePicks via OddsPapi</b> (player props).
  Static snapshot &mdash; regenerate after each data pull to refresh.<br><br>
  No model reliably beats a well-priced sportsbook line on every game. Treat this as research support, not a guarantee.
  Bet only what you're comfortable losing.
</footer>

<script>
  function filterCards(term) {{
    term = (term || '').trim().toLowerCase();
    var cards = document.querySelectorAll('.game-card');
    var visibleCount = 0;
    cards.forEach(function(card) {{
      var match = !term || card.getAttribute('data-teams').indexOf(term) !== -1;
      card.style.display = match ? '' : 'none';
      if (match) visibleCount++;
    }});
    document.querySelectorAll('.games-panel section').forEach(function(section) {{
      var visible = 0;
      section.querySelectorAll('.game-card').forEach(function(c) {{ if (c.style.display !== 'none') visible++; }});
      section.style.display = visible ? '' : 'none';
    }});
    document.getElementById('no-results').style.display = visibleCount ? 'none' : 'block';
  }}
</script>
</body>
</html>
"""


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    games = data.get("games", [])
    props = data.get("props", [])
    prop_catalog = data.get("prop_market_catalog", [])
    generated_at = data.get("generated_at")
    try:
        gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00")) if generated_at else None
        gen_label = gen_dt.strftime("%b %-d, %-I:%M %p UTC") if gen_dt else "unknown"
    except Exception:
        gen_label = generated_at or "unknown"

    game_sections_html = build_game_sections(games)
    props_section_html = build_props_section(prop_catalog, props)

    html_out = PAGE_TEMPLATE.format(
        game_count=len(games),
        market_count=len(prop_catalog),
        generated_at=esc(gen_label),
        game_sections=game_sections_html,
        props_section=props_section_html,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html_out)

    print(f"Wrote {OUT_PATH} ({len(games)} games, {len(prop_catalog)} prop markets, {len(props)} live prop rows)")


if __name__ == "__main__":
    main()
