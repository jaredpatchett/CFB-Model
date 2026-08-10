#!/usr/bin/env python3
"""
Render docs/data/latest.json into a single self-contained HTML dashboard
(docs/dashboard.html) — real games, real team logos/colors, real
moneyline/spread/total lines, plus a Props tab listing the real player-prop
market catalog (populated with real lines once PrizePicks posts them).

Design choices worth knowing about:
  - Every game/prop card is rendered directly into the HTML by THIS PYTHON
    SCRIPT, not built at page-load time by JavaScript. That way all content
    is visible even in environments that don't execute inline <script>
    (some sandboxed previews). JS is only used for the search filter — a
    pure enhancement, nothing depends on it to be visible.
  - Tabs (Games / Prop Markets) use the CSS-only radio-input technique, not
    JS, for the same reason: tab switching works even with JavaScript fully
    disabled.
  - Aesthetic: vintage varsity/game-program theme (cream parchment, maroon,
    aged brass) — deliberately different from a typical dark-mode sportsbook
    palette.

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


def fmt_spread(v):
    return fmt_odds(v)


def fmt_total(v):
    if v is None or v == "":
        return "&mdash;"
    n = float(v)
    n = int(n) if n == int(n) else n
    return f"{n}"


def fmt_time(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%-I:%M %p UTC")
    except Exception:
        return ""


def day_label(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%A, %B %-d")
    except Exception:
        return "TBD"


def team_badge_html(name, logo, color):
    ring = f"box-shadow: 0 0 0 3px {esc(color)}, 0 0 0 4px var(--ink) inset, 0 3px 8px rgba(43,24,16,0.35);" if color else ""
    if logo:
        return (
            f'<div class="badge" style="{ring}">'
            f'<img src="{esc(logo)}" alt="{esc(name)} logo" loading="lazy" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">'
            f'<span class="initials" style="display:none">{esc(initials(name))}</span>'
            f'</div>'
        )
    return (
        f'<div class="badge" style="{ring}">'
        f'<span class="initials">{esc(initials(name))}</span>'
        f'</div>'
    )


def card_html(g):
    home_color = g.get("home_color") or "#7c1f2e"
    away_color = g.get("away_color") or "#7c1f2e"
    home_ml = g.get("moneyline_home")
    away_ml = g.get("moneyline_away")

    home_is_fav = home_ml is not None and away_ml is not None and float(home_ml) < float(away_ml)
    away_is_fav = home_ml is not None and away_ml is not None and float(away_ml) < float(home_ml)

    teams_search_key = esc(f"{str(g.get('home_team',''))} {str(g.get('away_team',''))}".lower())

    return f"""
      <div class="card game-card" data-teams="{teams_search_key}">
        <div class="stub-notch"></div>
        <div class="ticket-row">
          <span class="ticket-no">GATE {esc((g.get('book_used') or 'MKT').upper()[:3])}</span>
          <span class="ticket-time">{fmt_time(g.get('commence_time',''))}</span>
        </div>
        <div class="matchup">
          <div class="team">
            {team_badge_html(g.get('away_team'), g.get('away_logo'), g.get('away_color'))}
            <div class="name">{esc(g.get('away_team'))}</div>
            <div class="tag{' fav' if away_is_fav else ''}">{'&#9733; Favorite' if away_is_fav else 'Visitor'}</div>
          </div>
          <div class="at">vs</div>
          <div class="team">
            {team_badge_html(g.get('home_team'), g.get('home_logo'), g.get('home_color'))}
            <div class="name">{esc(g.get('home_team'))}</div>
            <div class="tag{' fav' if home_is_fav else ''}">{'&#9733; Favorite' if home_is_fav else 'Home'}</div>
          </div>
        </div>
        <div class="lines">
          <div class="stat">
            <div class="label">Spread</div>
            <div class="value">{fmt_spread(g.get('spread_away'))}<span class="slash">/</span>{fmt_spread(g.get('spread_home'))}</div>
          </div>
          <div class="stat">
            <div class="label">Moneyline</div>
            <div class="value{' fav' if away_is_fav else ''}">{fmt_odds(away_ml)}</div>
            <div class="value{' fav' if home_is_fav else ''}">{fmt_odds(home_ml)}</div>
          </div>
          <div class="stat">
            <div class="label">Total</div>
            <div class="value">{fmt_total(g.get('total_over'))}</div>
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
        cards = "\n".join(card_html(g) for g in day_games)
        sections.append(f"""
        <section>
          <div class="day-header">
            <div class="day-title">{day_label(day_games[0].get('commence_time',''))}</div>
            <div class="day-line"></div>
            <div class="day-count">{len(day_games)} game{'s' if len(day_games) != 1 else ''}</div>
          </div>
          <div class="grid">{cards}</div>
        </section>""")
    return "\n".join(sections)


def prop_card_html(market_name, rows):
    if rows:
        row_html = "\n".join(f"""
            <div class="prop-row">
              <span class="prop-player">{esc(r.get('player_name'))}</span>
              <span class="prop-line">{esc(r.get('line'))}</span>
              <span class="prop-price">O {esc(r.get('over_price', '&mdash;'))} / U {esc(r.get('under_price', '&mdash;'))}</span>
            </div>""" for r in rows)
        return f"""
        <div class="card prop-card is-live">
          <div class="stub-notch"></div>
          <div class="prop-header">
            <span class="prop-market">{esc(market_name)}</span>
            <span class="prop-status live">LIVE</span>
          </div>
          {row_html}
        </div>"""
    return f"""
        <div class="card prop-card is-pending">
          <div class="stub-notch"></div>
          <div class="prop-header">
            <span class="prop-market">{esc(market_name)}</span>
            <span class="prop-status">COMING SOON</span>
          </div>
          <p class="prop-note">Market confirmed available on PrizePicks &mdash; lines open closer to kickoff week.</p>
        </div>"""


def build_props_section(prop_catalog, props):
    if not prop_catalog:
        return """
        <div class="empty-state">
          <div class="icon">NO CATALOG YET</div>
          <p>Couldn't load the player-prop market catalog on the last data pull. Re-run the data pull to populate this list.</p>
        </div>"""

    by_market = {}
    for p in props:
        by_market.setdefault(p.get("market_name"), []).append(p)

    cards = "\n".join(prop_card_html(m, by_market.get(m, [])) for m in prop_catalog)
    live_count = sum(1 for m in prop_catalog if by_market.get(m))
    note = "" if live_count else """
        <p class="props-intro">These are the real player-prop markets PrizePicks offers for college football.
        None are priced yet for this slate &mdash; that's normal this far from kickoff. Re-run the data pull
        closer to game week and any market with posted lines will switch from "Coming Soon" to live odds automatically.</p>"""
    return f"{note}<div class=\"grid\">{cards}</div>"


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Slate — CFB Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rye&family=Special+Elite&family=PT+Serif:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --paper: #f2e6c9;
    --paper-dark: #e6d5a8;
    --paper-line: rgba(43,24,16,0.12);
    --ink: #2b1810;
    --ink-soft: #5b4636;
    --maroon: #7c1f2e;
    --maroon-dark: #551420;
    --brass: #a9793b;
    --brass-light: #c99a53;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0; background: var(--paper); color: var(--ink);
    font-family: 'PT Serif', serif; min-height: 100vh; position: relative;
  }}
  body::before {{
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none; opacity: 0.5;
    background-image:
      radial-gradient(circle at 20% 30%, rgba(169,121,59,0.10), transparent 45%),
      radial-gradient(circle at 85% 75%, rgba(124,31,46,0.08), transparent 40%),
      url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E");
  }}

  /* --- CSS-only tabs: no JS required --- */
  #tab-games, #tab-props {{ position: absolute; opacity: 0; pointer-events: none; }}
  .games-panel {{ display: block; }}
  .props-panel {{ display: none; }}
  #tab-props:checked ~ main .games-panel {{ display: none; }}
  #tab-props:checked ~ main .props-panel {{ display: block; }}
  .tab-label {{
    font-family: 'Special Elite', monospace; font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--ink-soft); padding: 12px 26px; border: 2px solid var(--ink); border-bottom: none;
    background: var(--paper-dark); cursor: pointer; border-radius: 8px 8px 0 0; position: relative; top: 2px;
    transition: color 0.15s, background 0.15s;
  }}
  #tab-games:checked ~ header .tab-label[for="tab-games"],
  #tab-props:checked ~ header .tab-label[for="tab-props"] {{
    background: var(--maroon); color: var(--paper); border-color: var(--maroon-dark); top: 0;
  }}

  header {{ padding: 50px 6vw 0; text-align: center; }}
  .frame-rule {{ border-top: 3px double var(--ink); max-width: 720px; margin: 0 auto 22px; }}
  .kicker {{
    font-family: 'Special Elite', monospace; letter-spacing: 0.3em; text-transform: uppercase;
    font-size: 12px; color: var(--maroon); font-weight: 400;
  }}
  h1 {{
    font-family: 'Rye', serif; font-size: clamp(42px, 8.5vw, 96px); letter-spacing: 0.01em;
    margin: 10px 0 4px; line-height: 0.95; color: var(--ink);
  }}
  h1 .accent {{ color: var(--maroon); }}
  .subhead {{ color: var(--ink-soft); font-size: 16.5px; max-width: 620px; margin: 8px auto 0; line-height: 1.6; font-style: italic; }}
  .meta-strip {{
    display: flex; justify-content: center; gap: 28px; margin-top: 20px; flex-wrap: wrap;
    font-family: 'Special Elite', monospace; font-size: 12px; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--ink-soft);
  }}
  .meta-strip b {{ color: var(--maroon); }}
  .controls {{ max-width: 720px; margin: 26px auto 0; padding: 0 6vw; display: flex; justify-content: center; }}
  #search {{
    width: 100%; max-width: 440px; background: var(--paper-dark); border: 2px solid var(--ink);
    color: var(--ink); padding: 12px 20px; border-radius: 6px; font-family: 'PT Serif', serif;
    font-size: 15px; outline: none;
  }}
  #search::placeholder {{ color: var(--ink-soft); opacity: 0.7; }}
  #search:focus {{ border-color: var(--maroon); }}
  .tab-bar {{ margin-top: 30px; display: flex; justify-content: center; gap: 6px; }}

  main {{ max-width: 1180px; margin: 0 auto; padding: 30px 6vw 100px; border-top: 2px solid var(--ink); }}
  .day-header {{ display: flex; align-items: baseline; gap: 14px; margin: 40px 0 18px; }}
  section:first-of-type .day-header {{ margin-top: 6px; }}
  .day-title {{
    font-family: 'Rye', serif; font-size: 24px; color: var(--maroon); white-space: nowrap;
  }}
  .day-line {{ flex: 1; height: 2px; background: repeating-linear-gradient(90deg, var(--brass) 0 6px, transparent 6px 10px); }}
  .day-count {{
    font-family: 'Special Elite', monospace; font-size: 11px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-soft); white-space: nowrap;
  }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 18px; }}

  .card {{
    background: var(--paper-dark); border: 2px solid var(--ink); border-radius: 4px;
    padding: 16px 18px 16px; position: relative; box-shadow: 4px 4px 0 rgba(43,24,16,0.18);
  }}
  .stub-notch {{
    position: absolute; top: -2px; left: 50%; transform: translateX(-50%);
    width: 22px; height: 11px; background: var(--paper); border: 2px solid var(--ink);
    border-top: none; border-radius: 0 0 11px 11px;
  }}
  .ticket-row {{
    display: flex; justify-content: space-between; font-family: 'Special Elite', monospace;
    font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--brass);
    margin: 6px 0 12px; border-bottom: 1px dashed var(--paper-line); padding-bottom: 8px;
  }}
  .matchup {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; gap: 6px; }}
  .team {{ display: flex; flex-direction: column; align-items: center; gap: 8px; width: 42%; text-align: center; }}
  .badge {{
    width: 56px; height: 56px; border-radius: 50%; background: var(--paper);
    display: flex; align-items: center; justify-content: center; overflow: hidden;
  }}
  .badge img {{ width: 38px; height: 38px; object-fit: contain; }}
  .badge .initials {{ font-family: 'Rye', serif; font-size: 17px; color: var(--maroon); }}
  .name {{ font-family: 'PT Serif', serif; font-weight: 700; font-size: 13px; line-height: 1.2; color: var(--ink); }}
  .tag {{ font-family: 'Special Elite', monospace; font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-soft); }}
  .tag.fav {{ color: var(--maroon); }}
  .at {{ font-family: 'Rye', serif; color: var(--brass); font-size: 16px; }}

  .lines {{ display: grid; grid-template-columns: 1fr 1fr 1fr; border-top: 2px solid var(--ink); padding-top: 12px; }}
  .lines .stat {{ text-align: center; }}
  .lines .label {{
    font-family: 'Special Elite', monospace; font-size: 9.5px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--brass); margin-bottom: 5px;
  }}
  .lines .value {{ font-family: 'Special Elite', monospace; font-size: 15px; color: var(--ink); line-height: 1.6; }}
  .lines .value .slash {{ color: var(--brass); margin: 0 3px; }}
  .lines .value.fav {{ color: var(--maroon); font-weight: 700; }}

  /* Props tab */
  .props-intro {{
    max-width: 720px; margin: 0 auto 26px; text-align: center; color: var(--ink-soft);
    font-size: 15px; line-height: 1.65; font-style: italic;
  }}
  .prop-card {{ padding: 16px 18px 18px; }}
  .prop-header {{
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 2px solid var(--ink); padding-bottom: 8px; margin-bottom: 10px;
  }}
  .prop-market {{ font-family: 'Rye', serif; font-size: 17px; color: var(--ink); }}
  .prop-status {{
    font-family: 'Special Elite', monospace; font-size: 9.5px; letter-spacing: 0.08em;
    padding: 3px 9px; border: 1.5px solid var(--brass); border-radius: 3px; color: var(--brass);
    transform: rotate(-3deg); white-space: nowrap;
  }}
  .prop-status.live {{ color: var(--paper); background: var(--maroon); border-color: var(--maroon-dark); }}
  .prop-note {{ color: var(--ink-soft); font-size: 13px; line-height: 1.55; margin: 0; font-style: italic; }}
  .prop-row {{
    display: flex; justify-content: space-between; gap: 8px; font-size: 13px;
    padding: 6px 0; border-bottom: 1px dashed var(--paper-line);
  }}
  .prop-row:last-child {{ border-bottom: none; }}
  .prop-player {{ font-weight: 700; }}
  .prop-line {{ color: var(--maroon); font-family: 'Special Elite', monospace; }}
  .prop-price {{ color: var(--ink-soft); font-family: 'Special Elite', monospace; font-size: 11.5px; }}

  .empty-state {{
    margin: 40px auto 0; max-width: 580px; text-align: center; background: var(--paper-dark);
    border: 2px dashed var(--brass); border-radius: 8px; padding: 32px 28px;
  }}
  .empty-state .icon {{ font-family: 'Special Elite', monospace; font-size: 13px; letter-spacing: 0.15em; color: var(--maroon); margin-bottom: 10px; }}
  .empty-state p {{ color: var(--ink-soft); line-height: 1.65; font-size: 15px; margin: 0; }}
  #no-results {{
    display: none; text-align: center; color: var(--ink-soft); padding: 70px 0;
    font-family: 'Special Elite', monospace; letter-spacing: 0.05em; text-transform: uppercase; font-size: 14px;
  }}

  footer {{
    text-align: center; padding: 34px 6vw 60px; color: var(--ink-soft); font-size: 12.5px; line-height: 1.85;
    border-top: 2px solid var(--ink); max-width: 800px; margin: 0 auto; font-family: 'PT Serif', serif;
  }}
  footer b {{ color: var(--maroon); }}
</style>
</head>
<body>

<input type="radio" name="tabs" id="tab-games" checked>
<input type="radio" name="tabs" id="tab-props">

<header>
  <div class="frame-rule"></div>
  <div class="kicker">College Football &middot; Official Model Program</div>
  <h1>THE <span class="accent">SLATE</span></h1>
  <p class="subhead">Real market lines matched with real team data &mdash; research/decision-support only.
  See the disclaimer at the bottom before acting on anything here.</p>
  <div class="meta-strip">
    <div>Games tracked: <b>{game_count}</b></div>
    <div>Prop markets tracked: <b>{market_count}</b></div>
    <div>Snapshot: <b>{generated_at}</b></div>
  </div>
  <div class="controls">
    <input id="search" type="text" placeholder="Search a team, e.g. &quot;Michigan&quot; or &quot;Alabama&quot;&hellip;" autocomplete="off" oninput="filterCards(this.value)">
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
        gen_label = gen_dt.strftime("%b %-d, %Y %-I:%M %p UTC") if gen_dt else "unknown"
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
