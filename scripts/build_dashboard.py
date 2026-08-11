#!/usr/bin/env python3
"""
Render docs/data/latest.json into a single self-contained HTML dashboard
(docs/dashboard.html) — real games, real team logos, real moneyline/spread/
total lines, the model's fair odds vs. the book with EV%, and a Props tab
listing the real player-prop market catalog.

DESIGN v5: real sportsbook odds-board layout (DraftKings/FanDuel/ESPN BET
genre) — dark theme, two stacked rows per game (away/home), SPREAD / TOTAL /
MONEYLINE columns rendered as odds "pills," with an added MODEL EV column
so the comparison reads as one more market column rather than a bolt-on.
Not a clone of any one book's exact branding/colors — the layout pattern and
density are what's being matched, with a neutral dark-green/blue palette.

Engineering choices carried over from earlier fixes (still apply):
  - Every row is rendered directly into the HTML by THIS PYTHON SCRIPT, not
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


def fmt_num(v, decimals=None):
    if v is None or v == "" or (isinstance(v, float) and v != v):
        return None
    n = float(v)
    if decimals is not None:
        return round(n, decimals)
    return int(n) if n == int(n) else round(n, 1)


def fmt_signed(v):
    n = fmt_num(v)
    if n is None:
        return "&mdash;"
    return f"+{n}" if n > 0 else f"{n}"


def fmt_price(v):
    n = fmt_num(v)
    if n is None:
        return ""
    return f"({'+' if n > 0 else ''}{n})"


def fmt_pct(v):
    n = fmt_num(v, 1)
    if n is None:
        return "&mdash;"
    return f"{'+' if n > 0 else ''}{n}%"


def fmt_time(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%-I:%M%p").replace("AM", "a").replace("PM", "p")
    except Exception:
        return ""


def day_label(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%a %m/%d")
    except Exception:
        return "TBD"


def logo_html(name, logo):
    if logo:
        return (
            f'<span class="logo"><img src="{esc(logo)}" alt="" loading="lazy" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">'
            f'<span class="logo-fallback" style="display:none">{esc(initials(name))}</span></span>'
        )
    return f'<span class="logo"><span class="logo-fallback">{esc(initials(name))}</span></span>'


def ev_pill(ev):
    n = fmt_num(ev, 1)
    if n is None:
        return '<span class="pill pill-flat">&mdash;</span>'
    cls = "pill-pos" if n > 0 else ("pill-neg" if n < -3 else "pill-flat")
    return f'<span class="pill {cls}">{fmt_pct(ev)}</span>'


def spread_pill(point, price):
    p = fmt_signed(point)
    price_str = fmt_price(price)
    return f'<span class="pill">{p} <small>{price_str}</small></span>'


def ml_pill(ml):
    n = fmt_num(ml)
    if n is None:
        return '<span class="pill">&mdash;</span>'
    cls = "pill-fav" if n < 0 else ""
    return f'<span class="pill {cls}">{fmt_signed(ml)}</span>'


def total_pill(label, line):
    n = fmt_num(line)
    if n is None:
        return '<span class="pill">&mdash;</span>'
    return f'<span class="pill">{label} {n}</span>'


def fair_ml_pill(ml):
    n = fmt_num(ml, 0)
    if n is None:
        return '<span class="pill pill-ghost">&mdash;</span>'
    return f'<span class="pill pill-ghost">{fmt_signed(ml)}</span>'


def game_block_html(g):
    teams_search_key = esc(f"{str(g.get('home_team',''))} {str(g.get('away_team',''))}".lower())
    has_model = g.get("has_model_line")

    if has_model:
        away_model_cell = f"""
          <div class="cell model-cell">
            {fair_ml_pill(g.get('model_fair_ml_away'))}
            {ev_pill(g.get('ev_away_pct'))}
          </div>"""
        home_model_cell = f"""
          <div class="cell model-cell">
            {fair_ml_pill(g.get('model_fair_ml_home'))}
            {ev_pill(g.get('ev_home_pct'))}
          </div>"""
    else:
        reason = {
            "missing_sp_rating": "opponent outside FBS DB",
            "missing_book_moneyline": "no moneyline yet",
            "no_prior_fitted": "prior unavailable",
        }.get(g.get("no_line_reason"), "unavailable")
        away_model_cell = f'<div class="cell model-cell"><span class="no-model">No line &mdash; {esc(reason)}</span></div>'
        home_model_cell = '<div class="cell model-cell"></div>'

    return f"""
      <div class="game-block" data-teams="{teams_search_key}">
        <div class="time-col">
          <span class="time">{fmt_time(g.get('commence_time',''))}</span>
          <span class="book">{esc((g.get('book_used') or 'MKT')[:3].upper())}</span>
        </div>
        <div class="rows">
          <div class="row">
            <div class="cell team-cell">
              {logo_html(g.get('away_team'), g.get('away_logo'))}
              <span class="team-name">{esc(g.get('away_team'))}</span>
            </div>
            <div class="cell">{spread_pill(g.get('spread_away'), g.get('spread_price_away'))}</div>
            <div class="cell">{total_pill('O', g.get('total_over'))}</div>
            <div class="cell">{ml_pill(g.get('moneyline_away'))}</div>
            {away_model_cell}
          </div>
          <div class="row">
            <div class="cell team-cell">
              {logo_html(g.get('home_team'), g.get('home_logo'))}
              <span class="team-name">{esc(g.get('home_team'))}</span>
            </div>
            <div class="cell">{spread_pill(g.get('spread_home'), g.get('spread_price_home'))}</div>
            <div class="cell">{total_pill('U', g.get('total_under'))}</div>
            <div class="cell">{ml_pill(g.get('moneyline_home'))}</div>
            {home_model_cell}
          </div>
        </div>
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
        blocks = "".join(game_block_html(g) for g in day_games)
        sections.append(f"""
        <section>
          <div class="day-header">
            <span class="day-title">{day_label(day_games[0].get('commence_time',''))}</span>
            <span class="day-count">{len(day_games)} games &middot; {n_model} with a model line</span>
          </div>
          <div class="col-labels">
            <span></span><span>SPREAD</span><span>TOTAL</span><span>MONEYLINE</span><span>MODEL / EV</span>
          </div>
          <div class="game-list">{blocks}</div>
        </section>""")
    return "\n".join(sections)


def prop_card_html(market_name, rows):
    if rows:
        detail = "".join(f"""
            <div class="prop-line-row">
              <span>{esc(r.get('player_name'))}</span>
              <span>{esc(r.get('line'))}</span>
              <span class="pill">O {esc(r.get('over_price', '&mdash;'))}</span>
              <span class="pill">U {esc(r.get('under_price', '&mdash;'))}</span>
            </div>""" for r in rows)
        return f"""
        <div class="prop-card is-live">
          <div class="prop-head"><span>{esc(market_name)}</span><span class="live-tag">LIVE</span></div>
          {detail}
        </div>"""
    return f"""
        <div class="prop-card is-pending">
          <div class="prop-head"><span>{esc(market_name)}</span><span class="pending-tag">Not posted</span></div>
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
<title>CFB Model — Odds Board</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0b0f0d; --panel: #121815; --panel-alt: #161e1a; --border: #232d28;
    --text: #eef2ef; --text-dim: #7d8a84; --green: #2ecf7a; --green-bg: rgba(46,207,122,0.14);
    --red: #ef5b53; --red-bg: rgba(239,91,83,0.14); --ghost: #5b83c9; --ghost-bg: rgba(91,131,201,0.14);
  }}
  * {{ box-sizing: border-box; font-variant-numeric: tabular-nums; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font-family: 'Manrope', sans-serif; }}

  #tab-games, #tab-props {{ position: absolute; opacity: 0; pointer-events: none; }}
  .games-panel {{ display: block; }}
  .props-panel {{ display: none; }}
  #tab-props:checked ~ main .games-panel {{ display: none; }}
  #tab-props:checked ~ main .props-panel {{ display: block; }}
  .tab-label {{
    font-size: 13px; font-weight: 700; color: var(--text-dim); padding: 8px 16px;
    border-radius: 999px; cursor: pointer;
  }}
  #tab-games:checked ~ header .tab-label[for="tab-games"],
  #tab-props:checked ~ header .tab-label[for="tab-props"] {{ background: var(--panel-alt); color: var(--text); }}

  header {{ padding: 24px 4vw 16px; border-bottom: 1px solid var(--border); }}
  .title-row {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; max-width: 1300px; margin: 0 auto; }}
  .brand {{ display: flex; align-items: baseline; gap: 10px; }}
  h1 {{ font-size: 18px; font-weight: 800; margin: 0; letter-spacing: -0.01em; }}
  h1 .dot {{ color: var(--green); }}
  .meta-strip {{ display: flex; gap: 16px; font-size: 11.5px; color: var(--text-dim); flex-wrap: wrap; }}
  .meta-strip b {{ color: var(--text); }}
  .controls-row {{ max-width: 1300px; margin: 16px auto 0; display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; }}
  #search {{
    flex: 1; max-width: 280px; background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 12px; font-family: 'Manrope', sans-serif; font-size: 13px; color: var(--text); outline: none;
  }}
  #search::placeholder {{ color: var(--text-dim); }}
  #search:focus {{ border-color: var(--green); }}
  .tab-bar {{ display: flex; gap: 4px; background: var(--panel); border-radius: 999px; padding: 3px; }}
  .legend {{ font-size: 11px; color: var(--text-dim); display: flex; gap: 14px; }}
  .legend .sw {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; }}

  main {{ max-width: 1300px; margin: 0 auto; padding: 18px 4vw 80px; }}
  .day-header {{ display: flex; justify-content: space-between; align-items: baseline; margin: 30px 0 4px; }}
  section:first-of-type .day-header {{ margin-top: 4px; }}
  .day-title {{ font-size: 14px; font-weight: 800; }}
  .day-count {{ font-size: 11px; color: var(--text-dim); }}
  .col-labels {{
    display: grid; grid-template-columns: 64px minmax(140px,1fr) 110px 100px 90px 130px; gap: 0;
    font-size: 9.5px; font-weight: 700; letter-spacing: 0.06em; color: var(--text-dim); padding: 10px 10px 6px;
  }}
  .game-list {{ display: flex; flex-direction: column; gap: 6px; }}

  .game-block {{ display: grid; grid-template-columns: 64px 1fr; background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
  .time-col {{ display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px; border-right: 1px solid var(--border); background: var(--panel-alt); padding: 6px 2px; }}
  .time {{ font-size: 11.5px; font-weight: 700; }}
  .book {{ font-size: 8.5px; color: var(--text-dim); font-weight: 700; letter-spacing: 0.04em; }}
  .rows {{ display: flex; flex-direction: column; }}
  .row {{
    display: grid; grid-template-columns: minmax(140px,1fr) 110px 100px 90px 130px;
    align-items: center; padding: 7px 10px; gap: 4px;
  }}
  .row:first-child {{ border-bottom: 1px solid var(--border); }}
  .cell {{ display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }}
  .team-cell {{ gap: 8px; overflow: hidden; }}
  .logo {{ width: 22px; height: 22px; border-radius: 50%; background: #1c2420; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; overflow: hidden; }}
  .logo img {{ width: 16px; height: 16px; object-fit: contain; }}
  .logo-fallback {{ font-size: 9px; font-weight: 800; color: var(--text-dim); }}
  .team-name {{ font-size: 12.5px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

  .pill {{
    display: inline-flex; align-items: center; gap: 3px; background: var(--panel-alt); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 8px; font-size: 12px; font-weight: 700; white-space: nowrap;
  }}
  .pill small {{ font-size: 9.5px; font-weight: 600; color: var(--text-dim); }}
  .pill-fav {{ border-color: rgba(46,207,122,0.35); }}
  .pill-pos {{ background: var(--green-bg); color: var(--green); border-color: transparent; }}
  .pill-neg {{ background: var(--red-bg); color: var(--red); border-color: transparent; }}
  .pill-flat {{ color: var(--text-dim); }}
  .pill-ghost {{ background: var(--ghost-bg); color: var(--ghost); border-color: transparent; font-size: 11px; }}
  .model-cell {{ flex-direction: column; align-items: flex-start; gap: 3px; }}
  .no-model {{ font-size: 10.5px; color: var(--text-dim); font-style: italic; line-height: 1.3; }}

  .props-intro {{ max-width: 700px; margin: 0 0 20px; color: var(--text-dim); font-size: 13px; line-height: 1.6; }}
  .prop-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; }}
  .prop-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  .prop-head {{ display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: 700; margin-bottom: 8px; }}
  .live-tag {{ font-size: 9.5px; font-weight: 800; color: var(--green); background: var(--green-bg); padding: 2px 8px; border-radius: 4px; }}
  .pending-tag {{ font-size: 9.5px; font-weight: 800; color: var(--text-dim); background: var(--panel-alt); padding: 2px 8px; border-radius: 4px; }}
  .prop-note {{ font-size: 12px; color: var(--text-dim); margin: 0; }}
  .prop-line-row {{ display: flex; justify-content: space-between; align-items: center; gap: 6px; font-size: 12px; padding: 6px 0; border-top: 1px solid var(--border); }}

  .empty-state {{ background: var(--panel); border: 1px dashed var(--border); border-radius: 10px; padding: 24px; text-align: center; color: var(--text-dim); font-size: 13px; }}
  #no-results {{ display: none; text-align: center; color: var(--text-dim); padding: 50px 0; font-size: 13px; }}

  footer {{ text-align: center; padding: 24px 4vw 50px; color: var(--text-dim); font-size: 11.5px; line-height: 1.75; border-top: 1px solid var(--border); max-width: 900px; margin: 0 auto; }}
  footer b {{ color: var(--text); }}

  @media (max-width: 720px) {{
    .col-labels {{ display: none; }}
    .row {{ grid-template-columns: 1fr; grid-auto-rows: auto; row-gap: 6px; }}
  }}
</style>
</head>
<body>

<input type="radio" name="tabs" id="tab-games" checked>
<input type="radio" name="tabs" id="tab-props">

<header>
  <div class="title-row">
    <div class="brand"><h1>CFB ODDS<span class="dot">BOARD</span></h1></div>
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
      <span><span class="sw" style="background:var(--green)"></span>+EV</span>
      <span><span class="sw" style="background:var(--red)"></span>-EV</span>
      <span><span class="sw" style="background:var(--ghost)"></span>Model fair ML</span>
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
  <b>Model / EV column:</b> since the 2026 season hasn't started, there's no in-season form yet for any team.
  The model line uses each team's most recent SP+ rating (a public power rating calibrated to approximate point spread)
  plus a home-field edge and uncertainty band both fit empirically from real 2021&ndash;2025 historical games. Games
  where either team is outside the FBS database (mostly FCS opponents) show "no line" rather than a fabricated number.
  <br><br>
  No model reliably beats a well-priced sportsbook line on every game. Treat this as research support, not a guarantee.
  Bet only what you're comfortable losing.
</footer>

<script>
  function filterCards(term) {{
    term = (term || '').trim().toLowerCase();
    var blocks = document.querySelectorAll('.game-block');
    var visibleCount = 0;
    blocks.forEach(function(b) {{
      var match = !term || b.getAttribute('data-teams').indexOf(term) !== -1;
      b.style.display = match ? '' : 'none';
      if (match) visibleCount++;
    }});
    document.querySelectorAll('.games-panel section').forEach(function(section) {{
      var visible = 0;
      section.querySelectorAll('.game-block').forEach(function(b) {{ if (b.style.display !== 'none') visible++; }});
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
