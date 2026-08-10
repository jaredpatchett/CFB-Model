#!/usr/bin/env python3
"""
Render docs/data/latest.json into a single self-contained HTML dashboard
(docs/dashboard.html) — real games, real team logos/colors, real
moneyline/spread/total lines.

IMPORTANT design choice: every game card is rendered directly into the HTML
by THIS PYTHON SCRIPT, not built at page-load time by JavaScript from an
embedded JSON blob. Some environments preview/open HTML in a way that
doesn't run inline <script> tags (sandboxed previews, some in-app viewers) —
if all the content depends on JS running first, that shows up as a "blank,
plain page where nothing works." Pre-rendering means every game, logo, and
line is visible even with JavaScript completely disabled. The only thing JS
is used for here is the search filter, which is a pure enhancement on top of
content that already exists in the page.

Regenerate whenever docs/data/latest.json is refreshed:

  python scripts/build_dashboard.py
"""
import html
import json
import os
import sys
from datetime import datetime, timezone

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
        return dt.strftime("%-I:%M %p UTC")
    except Exception:
        return ""


def day_label(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%A, %B %-d")
    except Exception:
        return "TBD"


def team_badge_html(name, logo, color, side_class):
    ring = f"box-shadow: 0 4px 14px rgba(0,0,0,0.4), 0 0 0 3px {esc(color)}88 inset;" if color else ""
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
    home_color = g.get("home_color") or "#163a26"
    away_color = g.get("away_color") or "#163a26"
    home_ml = g.get("moneyline_home")
    away_ml = g.get("moneyline_away")

    home_is_fav = home_ml is not None and away_ml is not None and float(home_ml) < float(away_ml)
    away_is_fav = home_ml is not None and away_ml is not None and float(away_ml) < float(home_ml)

    teams_search_key = esc(f"{str(g.get('home_team',''))} {str(g.get('away_team',''))}".lower())

    return f"""
      <div class="card" data-teams="{teams_search_key}">
        <div class="split-bar"><span style="background:{esc(away_color)}"></span><span style="background:{esc(home_color)}"></span></div>
        <div class="kickoff">{fmt_time(g.get('commence_time',''))} &middot; {esc((g.get('book_used') or 'market').upper())}</div>
        <div class="matchup">
          <div class="team">
            {team_badge_html(g.get('away_team'), g.get('away_logo'), g.get('away_color'), 'away')}
            <div class="name">{esc(g.get('away_team'))}</div>
            <div class="tag">{'Favorite' if away_is_fav else 'Away'}</div>
          </div>
          <div class="at">@</div>
          <div class="team">
            {team_badge_html(g.get('home_team'), g.get('home_logo'), g.get('home_color'), 'home')}
            <div class="name">{esc(g.get('home_team'))}</div>
            <div class="tag">{'Favorite' if home_is_fav else 'Home'}</div>
          </div>
        </div>
        <div class="lines">
          <div class="stat">
            <div class="label">Spread</div>
            <div class="value">{fmt_spread(g.get('spread_home'))}<br><small>{fmt_spread(g.get('spread_away'))}</small></div>
          </div>
          <div class="stat">
            <div class="label">Moneyline</div>
            <div class="value{' fav' if home_is_fav else ''}">{fmt_odds(home_ml)}</div>
            <div class="value{' fav' if away_is_fav else ''}"><small>{fmt_odds(away_ml)}</small></div>
          </div>
          <div class="stat">
            <div class="label">Total</div>
            <div class="value">O/U<br><small>{fmt_total(g.get('total_over'))}</small></div>
          </div>
        </div>
      </div>"""


def build_sections(games):
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


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CFB Model — Live Slate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Barlow+Semi+Condensed:wght@400;500;600;700&family=Barlow+Condensed:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --turf-black: #071309;
    --turf-dark: #0e2418;
    --turf-mid: #17402a;
    --chalk: #f4f1e6;
    --chalk-dim: #b8c4b9;
    --gold: #ffb81c;
    --gold-dim: #d9a422;
    --card-shadow: 0 18px 40px -12px rgba(0,0,0,0.6);
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: var(--turf-black);
    color: var(--chalk);
    font-family: 'Barlow Semi Condensed', sans-serif;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
  }}
  body::before {{
    content: ""; position: fixed; inset: 0; z-index: -2;
    background: repeating-linear-gradient(100deg,
      var(--turf-dark) 0px, var(--turf-dark) 130px,
      var(--turf-mid) 130px, var(--turf-mid) 134px,
      var(--turf-dark) 134px, var(--turf-dark) 260px);
  }}
  body::after {{
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background: radial-gradient(ellipse at 50% -10%, rgba(255,184,28,0.12), transparent 45%),
                radial-gradient(ellipse at 100% 110%, rgba(0,0,0,0.6), transparent 60%);
  }}
  header {{ padding: 56px 6vw 30px; text-align: center; }}
  .hashmark-row {{ display: flex; justify-content: center; gap: 10px; margin-bottom: 18px; opacity: 0.6; }}
  .hashmark-row span {{ width: 3px; height: 18px; background: var(--gold-dim); }}
  .kicker {{
    font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.35em; text-transform: uppercase;
    font-size: 13px; color: var(--gold); font-weight: 600;
  }}
  h1 {{
    font-family: 'Anton', sans-serif; font-size: clamp(44px, 9vw, 100px); letter-spacing: 0.02em;
    text-transform: uppercase; margin: 10px 0 6px; line-height: 0.92; color: var(--chalk);
    text-shadow: 0 0 50px rgba(255,184,28,0.18);
  }}
  h1 .accent {{ color: var(--gold); }}
  .subhead {{ color: var(--chalk-dim); font-size: 17px; max-width: 640px; margin: 10px auto 0; line-height: 1.55; }}
  .meta-strip {{
    display: flex; justify-content: center; gap: 30px; margin-top: 22px; flex-wrap: wrap;
    font-family: 'Barlow Condensed', sans-serif; font-size: 13px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--chalk-dim);
  }}
  .meta-strip b {{ color: var(--gold); font-weight: 700; }}
  .controls {{ max-width: 720px; margin: 34px auto 0; padding: 0 6vw; display: flex; justify-content: center; }}
  #search {{
    width: 100%; max-width: 460px;
    background: rgba(23,64,42,0.75); border: 1px solid rgba(255,184,28,0.3); color: var(--chalk);
    padding: 14px 22px; border-radius: 999px; font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 16px; outline: none; transition: border-color 0.2s, box-shadow 0.2s;
  }}
  #search::placeholder {{ color: var(--chalk-dim); }}
  #search:focus {{ border-color: var(--gold); box-shadow: 0 0 0 5px rgba(255,184,28,0.15); }}
  main {{ max-width: 1180px; margin: 0 auto; padding: 10px 6vw 100px; }}
  .day-header {{ display: flex; align-items: baseline; gap: 16px; margin: 48px 0 20px; }}
  section:first-of-type .day-header {{ margin-top: 10px; }}
  .day-title {{
    font-family: 'Anton', sans-serif; font-size: 27px; text-transform: uppercase;
    letter-spacing: 0.03em; color: var(--gold); white-space: nowrap;
  }}
  .day-line {{ flex: 1; height: 2px; background: linear-gradient(90deg, rgba(255,184,28,0.55), transparent); }}
  .day-count {{
    font-family: 'Barlow Condensed', sans-serif; font-size: 12px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--chalk-dim); white-space: nowrap;
  }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }}
  .card {{
    background: linear-gradient(165deg, rgba(23,64,42,0.85), rgba(9,22,14,0.95));
    border: 1px solid rgba(244,241,230,0.1); border-radius: 16px; padding: 18px 20px 18px;
    box-shadow: var(--card-shadow); position: relative; overflow: hidden;
    transition: transform 0.18s ease, border-color 0.18s ease;
  }}
  .card:hover {{ transform: translateY(-3px); border-color: rgba(255,184,28,0.4); }}
  .card.hidden {{ display: none; }}
  .split-bar {{ position: absolute; top: 0; left: 0; right: 0; height: 5px; display: flex; }}
  .split-bar span {{ flex: 1; }}
  .kickoff {{
    text-align: center; font-family: 'Barlow Condensed', sans-serif; font-size: 12px;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--chalk-dim); margin: 6px 0 14px;
  }}
  .matchup {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; gap: 8px; }}
  .team {{ display: flex; flex-direction: column; align-items: center; gap: 8px; width: 40%; text-align: center; }}
  .badge {{
    width: 60px; height: 60px; border-radius: 50%; background: var(--chalk);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.45), 0 0 0 3px rgba(0,0,0,0.15) inset; overflow: hidden;
  }}
  .badge img {{ width: 42px; height: 42px; object-fit: contain; }}
  .badge .initials {{ font-family: 'Anton', sans-serif; font-size: 18px; color: var(--turf-dark); }}
  .name {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 13.5px; line-height: 1.15; color: var(--chalk); }}
  .tag {{ font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--gold-dim); }}
  .at {{ font-family: 'Anton', sans-serif; color: var(--gold-dim); font-size: 20px; }}
  .lines {{ display: grid; grid-template-columns: 1fr 1fr 1fr; border-top: 1px dashed rgba(244,241,230,0.18); padding-top: 14px; }}
  .lines .stat {{ text-align: center; }}
  .lines .label {{
    font-family: 'Barlow Condensed', sans-serif; font-size: 10px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--gold-dim); margin-bottom: 6px;
  }}
  .lines .value {{ font-family: 'Anton', sans-serif; font-size: 19px; color: var(--chalk); line-height: 1.5; }}
  .lines .value.fav {{ color: var(--gold); }}
  .lines .value small {{ font-size: 12px; color: var(--chalk-dim); font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 600; }}
  .lines .value.fav small {{ color: var(--gold); }}
  .empty-state {{
    margin: 50px auto 0; max-width: 580px; text-align: center; background: rgba(23,64,42,0.45);
    border: 1px dashed rgba(255,184,28,0.35); border-radius: 16px; padding: 34px 30px;
  }}
  .empty-state .icon {{ font-family: 'Anton', sans-serif; font-size: 30px; letter-spacing: 0.08em; color: var(--gold); margin-bottom: 10px; }}
  .empty-state p {{ color: var(--chalk-dim); line-height: 1.65; font-size: 15px; margin: 0; }}
  #no-results {{
    display: none; text-align: center; color: var(--chalk-dim); padding: 70px 0;
    font-family: 'Barlow Condensed', sans-serif; letter-spacing: 0.06em; text-transform: uppercase; font-size: 15px;
  }}
  footer {{
    text-align: center; padding: 40px 6vw 60px; color: var(--chalk-dim); font-size: 12.5px; line-height: 1.85;
    border-top: 1px solid rgba(244,241,230,0.1); max-width: 800px; margin: 0 auto;
  }}
  footer b {{ color: var(--gold-dim); }}
  noscript .search-note {{ display: block; text-align: center; color: var(--chalk-dim); font-size: 12px; margin-top: 8px; }}
</style>
</head>
<body>

<header>
  <div class="hashmark-row"><span></span><span></span><span></span><span></span><span></span></div>
  <div class="kicker">College Football &middot; Model Output</div>
  <h1>THE <span class="accent">SLATE</span></h1>
  <p class="subhead">Live market lines pulled from real sportsbooks, matched with real team data. Research/decision-support only &mdash; see the disclaimer below before acting on anything here.</p>
  <div class="meta-strip">
    <div>Games tracked: <b>{game_count}</b></div>
    <div>Snapshot: <b>{generated_at}</b></div>
  </div>
  <div class="controls">
    <input id="search" type="text" placeholder="Search a team, e.g. &quot;Michigan&quot; or &quot;Alabama&quot;&hellip;" autocomplete="off" oninput="filterCards(this.value)">
  </div>
  <noscript><span class="search-note">(Search needs JavaScript enabled &mdash; all games below are visible either way.)</span></noscript>
</header>

<main>
{sections}
{empty_state}
<p id="no-results">No games match that search.</p>
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
    var cards = document.querySelectorAll('.card');
    var visibleCount = 0;
    cards.forEach(function(card) {{
      var match = !term || card.getAttribute('data-teams').indexOf(term) !== -1;
      card.classList.toggle('hidden', !match);
      if (match) visibleCount++;
    }});
    document.querySelectorAll('main > section').forEach(function(section) {{
      var visibleInSection = section.querySelectorAll('.card:not(.hidden)').length;
      section.style.display = visibleInSection ? '' : 'none';
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
    generated_at = data.get("generated_at")
    try:
        gen_dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00")) if generated_at else None
        gen_label = gen_dt.strftime("%b %-d, %Y %-I:%M %p UTC") if gen_dt else "unknown"
    except Exception:
        gen_label = generated_at or "unknown"

    sections_html = build_sections(games)

    if not props:
        empty_state = """
        <div class="empty-state">
          <div class="icon">NO PROPS YET</div>
          <p>No PrizePicks player props are posted for this slate yet. Prop lines for college football typically
          open closer to kickoff week &mdash; re-run the data pull nearer game day to populate this section.</p>
        </div>"""
    else:
        empty_state = ""  # TODO: render real prop cards once props are populated

    html_out = PAGE_TEMPLATE.format(
        game_count=len(games),
        generated_at=esc(gen_label),
        sections=sections_html,
        empty_state=empty_state,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html_out)

    print(f"Wrote {OUT_PATH} ({len(games)} games, {len(props)} props)")


if __name__ == "__main__":
    main()
