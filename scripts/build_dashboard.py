#!/usr/bin/env python3
"""
Render docs/data/latest.json into a single self-contained HTML dashboard
(docs/dashboard.html) — real games, real team logos, real moneyline/spread/
total lines, and (new) the model's fair odds vs. the book's line with EV%,
plus a Props tab listing the real player-prop market catalog.

DESIGN v4: deliberately plain and functional after three cosmetic passes
that didn't land — clean data table layout, EV%-first, no illustration or
theming. If the numbers are useful, the design should get out of the way of
them.

Engineering choices carried over from earlier fixes (still apply):
  - Every card is rendered directly into the HTML by THIS PYTHON SCRIPT, not
    built at page-load time by JavaScript, so content is visible even if
    inline <script> doesn't execute in whatever renders this file.
  - Tabs (Games / Prop Markets) use the CSS-only radio-input technique.

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
    n = int(n) if n == int(n) else round(n, 1)
    return f"+{n}" if n > 0 else f"{n}"


def fmt_total(v):
    if v is None or v == "":
        return "&mdash;"
    n = float(v)
    n = int(n) if n == int(n) else n
    return f"{n}"


def fmt_pct(v):
    if v is None or v == "" or (isinstance(v, float) and v != v):
        return "&mdash;"
    n = float(v)
    sign = "+" if n > 0 else ""
    return f"{sign}{n:.1f}%"


def fmt_time(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%-I:%M %p")
    except Exception:
        return ""


def day_label(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%a, %b %-d")
    except Exception:
        return "TBD"


def logo_html(name, logo, side):
    if logo:
        return (
            f'<span class="logo"><img src="{esc(logo)}" alt="{esc(name)} logo" loading="lazy" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'inline-flex\'">'
            f'<span class="logo-fallback" style="display:none">{esc(initials(name))}</span></span>'
        )
    return f'<span class="logo"><span class="logo-fallback">{esc(initials(name))}</span></span>'


def ev_class(ev):
    if ev is None:
        return ""
    if ev > 0:
        return "ev-pos"
    if ev < -3:
        return "ev-neg"
    return ""


def card_html(g):
    home_ml = g.get("moneyline_home")
    away_ml = g.get("moneyline_away")
    teams_search_key = esc(f"{str(g.get('home_team',''))} {str(g.get('away_team',''))}".lower())
    has_model = g.get("has_model_line")

    if has_model:
        model_row = f"""
        <div class="model-row">
          <div class="model-col">
            <span class="model-label">Model fair ML</span>
            <span class="model-val">{fmt_odds(g.get('model_fair_ml_away'))} / {fmt_odds(g.get('model_fair_ml_home'))}</span>
          </div>
          <div class="model-col">
            <span class="model-label">Model win%</span>
            <span class="model-val">{round(g.get('model_away_win_prob',0)*100,1)}% / {round(g.get('model_home_win_prob',0)*100,1)}%</span>
          </div>
          <div class="model-col">
            <span class="model-label">EV vs. book</span>
            <span class="model-val">
              <span class="{ev_class(g.get('ev_away_pct'))}">{fmt_pct(g.get('ev_away_pct'))}</span>
              <span class="dim"> / </span>
              <span class="{ev_class(g.get('ev_home_pct'))}">{fmt_pct(g.get('ev_home_pct'))}</span>
            </span>
          </div>
        </div>"""
    else:
        reason = {
            "missing_sp_rating": "insufficient data (opponent outside FBS team database)",
            "missing_book_moneyline": "no moneyline posted yet",
            "no_prior_fitted": "model prior unavailable",
        }.get(g.get("no_line_reason"), "unavailable")
        model_row = f'<div class="model-row no-model">No model line &mdash; {esc(reason)}</div>'

    return f"""
      <div class="card game-card" data-teams="{teams_search_key}">
        <div class="card-top">
          <span class="meta">{fmt_time(g.get('commence_time',''))} &middot; {esc((g.get('book_used') or 'market').upper())}</span>
        </div>
        <div class="matchup">
          <div class="team">
            {logo_html(g.get('away_team'), g.get('away_logo'), 'away')}
            <span class="team-name">{esc(g.get('away_team'))}</span>
          </div>
          <span class="at">@</span>
          <div class="team">
            {logo_html(g.get('home_team'), g.get('home_logo'), 'home')}
            <span class="team-name">{esc(g.get('home_team'))}</span>
          </div>
        </div>
        <div class="lines">
          <div class="stat"><span class="label">Spread</span><span class="val">{fmt_odds(g.get('spread_away'))} / {fmt_odds(g.get('spread_home'))}</span></div>
          <div class="stat"><span class="label">Moneyline</span><span class="val">{fmt_odds(away_ml)} / {fmt_odds(home_ml)}</span></div>
          <div class="stat"><span class="label">Total</span><span class="val">O/U {fmt_total(g.get('total_over'))}</span></div>
        </div>
        {model_row}
      </div>"""


def build_game_sections(games):
    by_day = {}
    for g in sorted(games, key=lambda x: x.get("commence_time") or ""):
        day = (g.get("commence_time") or "")[:10]
        by_day.setdefault(day, []).append(g)

    sections = []
    for day in sorted(by_day.keys()):
        day_games = by_day[day]
        n_model = sum(1 for g in day_games if g.get("has_model_line"))
        cards = "".join(card_html(g) for g in day_games)
        sections.append(f"""
        <section>
          <div class="day-header">
            <span class="day-title">{day_label(day_games[0].get('commence_time',''))}</span>
            <span class="day-count">{len(day_games)} games &middot; {n_model} with a model line</span>
          </div>
          <div class="grid">{cards}</div>
        </section>""")
    return "\n".join(sections)


def prop_card_html(market_name, rows):
    if rows:
        detail = "".join(f"""
            <div class="prop-line-row">
              <span>{esc(r.get('player_name'))}</span>
              <span>{esc(r.get('line'))}</span>
              <span>O {esc(r.get('over_price', '&mdash;'))} / U {esc(r.get('under_price', '&mdash;'))}</span>
            </div>""" for r in rows)
        return f"""
        <div class="prop-card is-live">
          <div class="prop-head"><span>{esc(market_name)}</span><span class="live-tag">LIVE</span></div>
          {detail}
        </div>"""
    return f"""
        <div class="prop-card is-pending">
          <div class="prop-head"><span>{esc(market_name)}</span><span class="pending-tag">Not posted yet</span></div>
          <p class="prop-note">Confirmed market on PrizePicks. Opens closer to kickoff week.</p>
        </div>"""


def build_props_section(prop_catalog, props):
    if not prop_catalog:
        return '<div class="empty-state">No prop market catalog loaded yet. Re-run the data pull to fetch it.</div>'

    by_market = {}
    for p in props:
        by_market.setdefault(p.get("market_name"), []).append(p)

    live_count = sum(1 for m in prop_catalog if by_market.get(m))
    intro = "" if live_count else (
        '<p class="props-intro">Real player-prop markets confirmed available on PrizePicks for college football. '
        'None are priced for this slate yet (normal this far from kickoff) &mdash; re-run the data pull closer '
        'to game week and any market with posted lines switches to LIVE automatically.</p>')
    cards = "".join(prop_card_html(m, by_market.get(m, [])) for m in prop_catalog)
    return f'{intro}<div class="prop-grid">{cards}</div>'


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CFB Model — Fair Odds vs. Book</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #f5f6f7; --panel: #ffffff; --border: #d9dcdf; --text: #1a1d21; --text-dim: #6b7075;
    --green: #12805c; --green-bg: #e6f4ee; --red: #b3261e; --red-bg: #fbeceb; --accent: #1f5fd1;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; }}
  .mono {{ font-family: 'IBM Plex Mono', monospace; }}

  #tab-games, #tab-props {{ position: absolute; opacity: 0; pointer-events: none; }}
  .games-panel {{ display: block; }}
  .props-panel {{ display: none; }}
  #tab-props:checked ~ main .games-panel {{ display: none; }}
  #tab-props:checked ~ main .props-panel {{ display: block; }}
  .tab-label {{
    font-size: 13.5px; font-weight: 600; color: var(--text-dim); padding: 9px 18px;
    border: 1.5px solid var(--border); background: var(--panel); cursor: pointer; border-radius: 6px;
  }}
  #tab-games:checked ~ header .tab-label[for="tab-games"],
  #tab-props:checked ~ header .tab-label[for="tab-props"] {{
    background: var(--text); color: #fff; border-color: var(--text);
  }}

  header {{ padding: 36px 5vw 20px; border-bottom: 1px solid var(--border); background: var(--panel); }}
  .title-row {{ display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 12px; max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 24px; font-weight: 700; margin: 0; }}
  .subhead {{ color: var(--text-dim); font-size: 13.5px; margin: 4px 0 0; max-width: 640px; }}
  .meta-strip {{ display: flex; gap: 20px; font-size: 12.5px; color: var(--text-dim); flex-wrap: wrap; }}
  .meta-strip b {{ color: var(--text); }}
  .controls-row {{ max-width: 1200px; margin: 18px auto 0; display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; }}
  #search {{
    flex: 1; max-width: 320px; border: 1.5px solid var(--border); border-radius: 6px; padding: 8px 12px;
    font-family: 'IBM Plex Sans', sans-serif; font-size: 13.5px; outline: none;
  }}
  #search:focus {{ border-color: var(--accent); }}
  .tab-bar {{ display: flex; gap: 8px; }}
  .legend {{ font-size: 12px; color: var(--text-dim); display: flex; gap: 16px; }}
  .legend span.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }}

  main {{ max-width: 1200px; margin: 0 auto; padding: 24px 5vw 90px; }}
  .day-header {{ display: flex; justify-content: space-between; align-items: baseline; margin: 34px 0 14px; }}
  section:first-of-type .day-header {{ margin-top: 6px; }}
  .day-title {{ font-size: 15.5px; font-weight: 700; }}
  .day-count {{ font-size: 12px; color: var(--text-dim); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}

  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .card-top {{ margin-bottom: 8px; }}
  .meta {{ font-size: 10.5px; font-weight: 600; letter-spacing: 0.03em; color: var(--text-dim); text-transform: uppercase; }}
  .matchup {{ display: flex; align-items: center; justify-content: space-between; gap: 6px; margin-bottom: 10px; }}
  .team {{ display: flex; align-items: center; gap: 6px; width: 44%; overflow: hidden; }}
  .logo {{ width: 22px; height: 22px; border-radius: 50%; background: #eef0f2; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; overflow: hidden; }}
  .logo img {{ width: 16px; height: 16px; object-fit: contain; }}
  .logo-fallback {{ font-size: 9px; font-weight: 700; color: var(--text-dim); }}
  .team-name {{ font-size: 12.5px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .at {{ font-size: 11px; color: var(--text-dim); flex-shrink: 0; }}

  .lines {{ display: grid; grid-template-columns: 1fr 1fr 1fr; border-top: 1px solid var(--border); padding-top: 8px; gap: 4px; }}
  .stat {{ display: flex; flex-direction: column; }}
  .label {{ font-size: 9.5px; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 2px; }}
  .val {{ font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; font-weight: 600; }}

  .model-row {{ margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border); display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; }}
  .model-col {{ display: flex; flex-direction: column; }}
  .model-label {{ font-size: 9.5px; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; color: var(--accent); margin-bottom: 2px; }}
  .model-val {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; }}
  .model-row.no-model {{ font-size: 11.5px; color: var(--text-dim); font-style: italic; grid-template-columns: 1fr; }}
  .ev-pos {{ color: var(--green); background: var(--green-bg); padding: 1px 4px; border-radius: 3px; }}
  .ev-neg {{ color: var(--red); background: var(--red-bg); padding: 1px 4px; border-radius: 3px; }}
  .dim {{ color: var(--text-dim); }}

  .props-intro {{ max-width: 700px; margin: 0 0 20px; color: var(--text-dim); font-size: 13.5px; line-height: 1.6; }}
  .prop-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
  .prop-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }}
  .prop-head {{ display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: 700; margin-bottom: 8px; }}
  .live-tag {{ font-size: 10px; font-weight: 700; color: var(--green); background: var(--green-bg); padding: 2px 8px; border-radius: 4px; }}
  .pending-tag {{ font-size: 10px; font-weight: 700; color: var(--text-dim); background: #eef0f2; padding: 2px 8px; border-radius: 4px; }}
  .prop-note {{ font-size: 12px; color: var(--text-dim); margin: 0; }}
  .prop-line-row {{ display: flex; justify-content: space-between; gap: 8px; font-size: 12px; padding: 5px 0; border-top: 1px solid var(--border); }}

  .empty-state {{ background: var(--panel); border: 1px dashed var(--border); border-radius: 8px; padding: 24px; text-align: center; color: var(--text-dim); font-size: 13.5px; }}
  #no-results {{ display: none; text-align: center; color: var(--text-dim); padding: 50px 0; font-size: 13.5px; }}

  footer {{ text-align: center; padding: 26px 5vw 50px; color: var(--text-dim); font-size: 12px; line-height: 1.75; border-top: 1px solid var(--border); max-width: 900px; margin: 0 auto; }}
  footer b {{ color: var(--text); }}
</style>
</head>
<body>

<input type="radio" name="tabs" id="tab-games" checked>
<input type="radio" name="tabs" id="tab-props">

<header>
  <div class="title-row">
    <div>
      <h1>CFB Model &mdash; Fair Odds vs. Book</h1>
      <p class="subhead">Model's estimated fair moneyline and win probability, compared to the real book price, with EV%. Research/decision-support only.</p>
    </div>
    <div class="meta-strip">
      <div>Games: <b>{game_count}</b></div>
      <div>With model line: <b>{model_count}</b></div>
      <div>Snapshot: <b>{generated_at}</b></div>
    </div>
  </div>
  <div class="controls-row">
    <input id="search" type="text" placeholder="Search a team..." autocomplete="off" oninput="filterCards(this.value)">
    <div class="tab-bar">
      <label class="tab-label" for="tab-games">Games</label>
      <label class="tab-label" for="tab-props">Prop Markets</label>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--green)"></span>Positive EV</span>
      <span><span class="dot" style="background:var(--red)"></span>Negative EV</span>
    </div>
  </div>
  <noscript><span style="display:block;color:var(--text-dim);font-size:12px;margin-top:8px;">(Search needs JavaScript — everything else works without it.)</span></noscript>
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
  Built from <b>CFBD</b> (team data, SP+ ratings), <b>The Odds API</b> (moneyline/spread/total), and <b>PrizePicks via OddsPapi</b> (player props).
  <br><br>
  <b>How "model fair odds" works right now:</b> the 2026 season hasn't started, so there's no in-season form yet for any team.
  The model line uses each team's most recent SP+ rating (a public power rating calibrated to approximate point spread) plus a
  home-field edge and prediction uncertainty (residual std) both fit empirically from real 2021&ndash;2025 historical games &mdash;
  not assumed. Once actual 2026 games are played, in-season data will improve this. Games where either team is outside the FBS
  database (mostly FCS opponents) show "no model line" rather than a fabricated number.
  <br><br>
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

    model_count = sum(1 for g in games if g.get("has_model_line"))

    game_sections_html = build_game_sections(games)
    props_section_html = build_props_section(prop_catalog, props)

    html_out = PAGE_TEMPLATE.format(
        game_count=len(games),
        model_count=model_count,
        generated_at=esc(gen_label),
        game_sections=game_sections_html,
        props_section=props_section_html,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html_out)

    print(f"Wrote {OUT_PATH} ({len(games)} games, {model_count} with a model line, "
          f"{len(prop_catalog)} prop markets, {len(props)} live prop rows)")


if __name__ == "__main__":
    main()
