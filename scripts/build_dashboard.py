#!/usr/bin/env python3
"""
Render docs/data/latest.json into a single self-contained HTML dashboard
(docs/dashboard.html).

DESIGN v6: matches the reference screenshots the user provided of their
existing golf/soccer models — dark forest-green "quant terminal" aesthetic,
monospace font throughout, four tabs (Games / EV Builder / Prop Markets /
Tracker), BET / EDGE / FADE signal tags on every priced side, a
comprehensive EV Builder table (model win% vs fair odds vs book odds vs
implied% vs edge vs EV/$100), and a persistent bet tracker (localStorage)
with WIN/LOSS/PUSH grading and running P&L — mirroring their tracker screens
exactly (description / date / type / price / EV% / stake / status / profit).

Engineering choices carried over from earlier fixes (still apply to the
Games / EV Builder / Prop Markets tabs — NOT to the Tracker, which is
inherently interactive and needs JS by nature, same as their reference):
  - Every row in Games / EV Builder / Prop Markets is rendered directly into
    the HTML by THIS PYTHON SCRIPT, not built at page-load time by
    JavaScript, so that content is visible even if inline <script> doesn't
    execute in whatever renders this file.
  - Tabs use the CSS-only radio-input technique for the same reason.
  - The Tracker tab is the one place JS is required (localStorage read/
    write, grading buttons, live P&L) — same as the reference screenshots,
    where the tracker is clearly a stateful, interactive feature. This file
    is a plain HTML file opened directly in a real browser (not rendered
    inside Cowork's sandboxed artifact viewer), so localStorage works
    normally here.

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

BET_THRESHOLD = 5.0   # EV% >= this -> "BET"
EDGE_THRESHOLD = 0.0  # 0 < EV% < BET_THRESHOLD -> "EDGE"; EV% <= 0 -> "FADE"


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


def fmt_pct(v, decimals=1):
    n = fmt_num(v, decimals)
    if n is None:
        return "&mdash;"
    return f"{'+' if n > 0 else ''}{n}%"


def fmt_time(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%-I:%M%p").replace("AM", "a").replace("PM", "p")
    except Exception:
        return ""


def fmt_date(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def day_label(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%a %m/%d")
    except Exception:
        return "TBD"


def signal_for(ev):
    n = fmt_num(ev)
    if n is None:
        return None
    if n >= BET_THRESHOLD:
        return "BET"
    if n > EDGE_THRESHOLD:
        return "EDGE"
    return "FADE"


def signal_tag(ev):
    sig = signal_for(ev)
    if sig is None:
        return '<span class="tag tag-flat">&mdash;</span>'
    return f'<span class="tag tag-{sig.lower()}">{sig}</span>'


def logo_html(name, logo):
    if logo:
        return (
            f'<span class="logo"><img src="{esc(logo)}" alt="" loading="lazy" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">'
            f'<span class="logo-fallback" style="display:none">{esc(initials(name))}</span></span>'
        )
    return f'<span class="logo"><span class="logo-fallback">{esc(initials(name))}</span></span>'


# ---------------------------------------------------------------------------
# GAMES TAB
# ---------------------------------------------------------------------------

def spread_cell(point, price):
    p = fmt_signed(point)
    price_str = fmt_price(price)
    return f'<span class="odds">{p} <small>{price_str}</small></span>'


def ml_cell(ml):
    n = fmt_num(ml)
    if n is None:
        return '<span class="odds">&mdash;</span>'
    return f'<span class="odds">{fmt_signed(ml)}</span>'


def total_cell(label, line):
    n = fmt_num(line)
    if n is None:
        return '<span class="odds">&mdash;</span>'
    return f'<span class="odds">{label} {n}</span>'


def game_block_html(g):
    teams_search_key = esc(f"{str(g.get('home_team',''))} {str(g.get('away_team',''))}".lower())
    has_model = g.get("has_model_line")

    if has_model:
        away_model = f'<span class="fair">{fmt_signed(g.get("model_fair_ml_away"))}</span>{signal_tag(g.get("ev_away_pct"))}'
        home_model = f'<span class="fair">{fmt_signed(g.get("model_fair_ml_home"))}</span>{signal_tag(g.get("ev_home_pct"))}'
    else:
        reason = {
            "missing_sp_rating": "opp. outside FBS DB",
            "missing_book_moneyline": "no ML posted",
            "no_prior_fitted": "prior unavailable",
        }.get(g.get("no_line_reason"), "unavailable")
        away_model = f'<span class="no-line">no line &middot; {esc(reason)}</span>'
        home_model = ""

    return f"""
      <div class="game-block" data-teams="{teams_search_key}">
        <div class="time-col"><span class="time">{fmt_time(g.get('commence_time',''))}</span><span class="book">{esc((g.get('book_used') or 'MKT')[:3].upper())}</span></div>
        <div class="rows">
          <div class="row">
            <div class="cell team-cell">{logo_html(g.get('away_team'), g.get('away_logo'))}<span class="team-name">{esc(g.get('away_team'))}</span></div>
            <div class="cell">{spread_cell(g.get('spread_away'), g.get('spread_price_away'))}</div>
            <div class="cell">{total_cell('O', g.get('total_over'))}</div>
            <div class="cell">{ml_cell(g.get('moneyline_away'))}</div>
            <div class="cell model-cell">{away_model}</div>
          </div>
          <div class="row">
            <div class="cell team-cell">{logo_html(g.get('home_team'), g.get('home_logo'))}<span class="team-name">{esc(g.get('home_team'))}</span></div>
            <div class="cell">{spread_cell(g.get('spread_home'), g.get('spread_price_home'))}</div>
            <div class="cell">{total_cell('U', g.get('total_under'))}</div>
            <div class="cell">{ml_cell(g.get('moneyline_home'))}</div>
            <div class="cell model-cell">{home_model}</div>
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
          <div class="day-header"><span class="day-title">{day_label(day_games[0].get('commence_time',''))}</span><span class="day-count">{len(day_games)} games &middot; {n_model} priced</span></div>
          <div class="col-labels"><span></span><span>SPREAD</span><span>TOTAL</span><span>MONEYLINE</span><span>MODEL FAIR / SIGNAL</span></div>
          <div class="game-list">{blocks}</div>
        </section>""")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# EV BUILDER TAB
# ---------------------------------------------------------------------------

def ev_builder_rows(games):
    """One row per priced side (2 per game with a model line), sorted best
    EV first, so the plays the model actually likes surface at the top."""
    rows = []
    for g in games:
        if not g.get("has_model_line"):
            continue
        for side, team_key, ml_key, prob_key, fair_key, ev_key in [
            ("away", "away_team", "moneyline_away", "model_away_win_prob", "model_fair_ml_away", "ev_away_pct"),
            ("home", "home_team", "moneyline_home", "model_home_win_prob", "model_fair_ml_home", "ev_home_pct"),
        ]:
            book_ml = g.get(ml_key)
            if book_ml is None:
                continue
            model_prob = g.get(prob_key)
            implied = None
            other_key = "model_home_win_prob" if side == "away" else "model_away_win_prob"
            # book-implied fair prob, devigged, mirrors what compute_fair_odds_fields already computed
            book_prob_key = "book_implied_prob_" + side
            implied = g.get(book_prob_key)
            ev = g.get(ev_key)
            edge_pp = None
            if model_prob is not None and implied is not None:
                edge_pp = (model_prob - implied) * 100
            rows.append({
                "description": f"{esc(g.get(team_key))} ML",
                "matchup": f"{esc(g.get('away_team'))} @ {esc(g.get('home_team'))}",
                "date": fmt_date(g.get("commence_time", "")),
                "model_prob": model_prob,
                "fair_ml": g.get(fair_key),
                "book_ml": book_ml,
                "implied": implied,
                "edge_pp": edge_pp,
                "ev": ev,
                "signal": signal_for(ev),
            })
    rows.sort(key=lambda r: (r["ev"] if r["ev"] is not None else -999), reverse=True)
    return rows


def ev_builder_row_html(r, idx):
    sig = r["signal"] or "flat"
    model_pct = fmt_pct(r["model_prob"] * 100 if r["model_prob"] is not None else None, 1)
    implied_pct = fmt_pct(r["implied"] * 100 if r["implied"] is not None else None, 1)
    edge_str = fmt_pct(r["edge_pp"], 1) + "pp" if r["edge_pp"] is not None else "&mdash;"
    row_id = f"ev-{idx}"
    payload = esc(json.dumps({
        "description": f"{r['description']} — {r['matchup']}",
        "date": r["date"], "type": "Moneyline",
        "price": fmt_num(r["book_ml"]), "ev": fmt_num(r["ev"], 1),
    }))
    return f"""
        <div class="ev-row ev-{sig.lower()}" id="{row_id}">
          <div class="ev-cell ev-desc">
            <span class="ev-pick">{r['description']}</span>
            <span class="ev-sub">{r['matchup']} &middot; {r['date']}</span>
          </div>
          <div class="ev-cell">{model_pct}</div>
          <div class="ev-cell mono-num">{fmt_signed(r['fair_ml'])}</div>
          <div class="ev-cell mono-num">{fmt_signed(r['book_ml'])}</div>
          <div class="ev-cell">{implied_pct}</div>
          <div class="ev-cell">{edge_str}</div>
          <div class="ev-cell ev-val">{fmt_pct(r['ev'], 1)}</div>
          <div class="ev-cell"><span class="tag tag-{sig.lower()}">{sig.upper() if r['signal'] else '&mdash;'}</span></div>
          <div class="ev-cell"><button class="track-btn" onclick="addToTracker({payload})">+ Track</button></div>
        </div>"""


def build_ev_builder_section(games):
    rows = ev_builder_rows(games)
    if not rows:
        return '<div class="empty-state">No priced sides yet &mdash; every game in this slate is either missing SP+ data for one side or hasn\'t posted a moneyline.</div>'
    n_bet = sum(1 for r in rows if r["signal"] == "BET")
    n_edge = sum(1 for r in rows if r["signal"] == "EDGE")
    header = f"""
        <div class="ev-summary">
          <span><span class="tag tag-bet">BET</span> {n_bet} plays</span>
          <span><span class="tag tag-edge">EDGE</span> {n_edge} plays</span>
          <span class="dim">Sorted by EV, best first</span>
        </div>
        <div class="ev-col-labels">
          <span>PICK</span><span>MODEL%</span><span>FAIR</span><span>BOOK</span><span>IMPLIED%</span><span>EDGE</span><span>EV/$100</span><span>SIGNAL</span><span></span>
        </div>"""
    body = "".join(ev_builder_row_html(r, i) for i, r in enumerate(rows))
    return header + f'<div class="ev-list">{body}</div>'


# ---------------------------------------------------------------------------
# PROP MARKETS TAB
# ---------------------------------------------------------------------------

def prop_card_html(market_name, rows):
    if rows:
        detail = "".join(f"""
            <div class="prop-line-row"><span>{esc(r.get('player_name'))}</span><span>{esc(r.get('line'))}</span>
              <span class="odds">O {esc(r.get('over_price', '&mdash;'))}</span><span class="odds">U {esc(r.get('under_price', '&mdash;'))}</span></div>""" for r in rows)
        return f"""
        <div class="prop-card is-live">
          <div class="prop-head"><span>{esc(market_name)}</span><span class="tag tag-bet">LIVE</span></div>{detail}
        </div>"""
    return f"""
        <div class="prop-card is-pending">
          <div class="prop-head"><span>{esc(market_name)}</span><span class="tag tag-flat">NOT POSTED</span></div>
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
<title>CFB Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a1810; --panel: #0f2417; --panel-alt: #13301d; --border: #1e3d28;
    --text: #e7f2ea; --text-dim: #6f8c7a; --gold: #d4ad33; --gold-dim: #8a742a;
    --green: #3ecf6e; --green-bg: rgba(62,207,110,0.14);
    --red: #e2635a; --red-bg: rgba(226,99,90,0.14);
    --teal: #4fc3b0; --teal-bg: rgba(79,195,176,0.14);
  }}
  * {{ box-sizing: border-box; font-variant-numeric: tabular-nums; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 13px; }}
  button {{ font-family: 'JetBrains Mono', monospace; }}

  #tab-games, #tab-ev, #tab-props, #tab-tracker {{ position: absolute; opacity: 0; pointer-events: none; }}
  .panel {{ display: none; }}
  .games-panel {{ display: block; }}
  #tab-ev:checked ~ main .games-panel {{ display: none; }} #tab-ev:checked ~ main .ev-panel {{ display: block; }}
  #tab-props:checked ~ main .games-panel {{ display: none; }} #tab-props:checked ~ main .props-panel {{ display: block; }}
  #tab-tracker:checked ~ main .games-panel {{ display: none; }} #tab-tracker:checked ~ main .tracker-panel {{ display: block; }}
  .tab-label {{ font-size: 12px; font-weight: 700; letter-spacing: 0.04em; color: var(--text-dim); padding: 8px 16px; cursor: pointer; border-bottom: 2px solid transparent; }}
  #tab-games:checked ~ header .tab-label[for="tab-games"],
  #tab-ev:checked ~ header .tab-label[for="tab-ev"],
  #tab-props:checked ~ header .tab-label[for="tab-props"],
  #tab-tracker:checked ~ header .tab-label[for="tab-tracker"] {{ color: var(--gold); border-color: var(--gold); }}

  header {{ border-bottom: 1px solid var(--border); background: var(--panel); }}
  .brand-row {{ display: flex; align-items: center; justify-content: space-between; max-width: 1400px; margin: 0 auto; padding: 16px 4vw 12px; flex-wrap: wrap; gap: 12px; }}
  .brand {{ display: flex; align-items: center; gap: 12px; }}
  .brand-icon {{ width: 34px; height: 34px; border: 1.5px solid var(--gold); border-radius: 6px; display: flex; align-items: center; justify-content: center; color: var(--gold); font-weight: 800; font-size: 13px; }}
  .brand-title {{ font-size: 15px; font-weight: 800; letter-spacing: 0.02em; }}
  .brand-sub {{ font-size: 10.5px; color: var(--text-dim); letter-spacing: 0.04em; margin-top: 1px; }}
  .meta-strip {{ display: flex; gap: 16px; font-size: 11px; color: var(--text-dim); flex-wrap: wrap; align-items: center; }}
  .meta-strip b {{ color: var(--text); }}
  .snap-badge {{ font-size: 10px; font-weight: 800; letter-spacing: 0.06em; color: var(--gold); border: 1px solid var(--gold-dim); border-radius: 4px; padding: 2px 8px; }}
  .tab-row {{ display: flex; gap: 4px; max-width: 1400px; margin: 0 auto; padding: 0 4vw; }}
  .controls-row {{ max-width: 1400px; margin: 0 auto; padding: 10px 4vw 14px; display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }}
  #search {{ flex: 1; max-width: 260px; background: var(--panel-alt); border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text); outline: none; }}
  #search::placeholder {{ color: var(--text-dim); }}
  #search:focus {{ border-color: var(--gold); }}
  .legend {{ font-size: 10.5px; color: var(--text-dim); display: flex; gap: 12px; }}

  main {{ max-width: 1400px; margin: 0 auto; padding: 16px 4vw 90px; }}

  /* GAMES */
  .day-header {{ display: flex; justify-content: space-between; align-items: baseline; margin: 26px 0 4px; }}
  section:first-of-type .day-header {{ margin-top: 4px; }}
  .day-title {{ font-size: 13px; font-weight: 800; color: var(--gold); }}
  .day-count {{ font-size: 10.5px; color: var(--text-dim); }}
  .col-labels {{ display: grid; grid-template-columns: 60px minmax(140px,1fr) 100px 90px 80px 150px; font-size: 9px; font-weight: 700; letter-spacing: 0.06em; color: var(--text-dim); padding: 8px 8px 5px; }}
  .game-list {{ display: flex; flex-direction: column; gap: 5px; }}
  .game-block {{ display: grid; grid-template-columns: 60px 1fr; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  .time-col {{ display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px; border-right: 1px solid var(--border); background: var(--panel-alt); padding: 6px 2px; }}
  .time {{ font-size: 10.5px; font-weight: 700; }}
  .book {{ font-size: 8px; color: var(--text-dim); font-weight: 700; }}
  .rows {{ display: flex; flex-direction: column; }}
  .row {{ display: grid; grid-template-columns: minmax(140px,1fr) 100px 90px 80px 150px; align-items: center; padding: 6px 8px; gap: 4px; }}
  .row:first-child {{ border-bottom: 1px solid var(--border); }}
  .cell {{ display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }}
  .team-cell {{ gap: 7px; overflow: hidden; }}
  .logo {{ width: 20px; height: 20px; border-radius: 50%; background: var(--panel-alt); display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; overflow: hidden; }}
  .logo img {{ width: 14px; height: 14px; object-fit: contain; }}
  .logo-fallback {{ font-size: 8px; font-weight: 800; color: var(--text-dim); }}
  .team-name {{ font-size: 11.5px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .odds {{ font-size: 11.5px; font-weight: 700; }}
  .odds small {{ font-size: 9px; color: var(--text-dim); font-weight: 500; }}
  .fair {{ font-size: 11px; color: var(--teal); font-weight: 700; margin-right: 2px; }}
  .no-line {{ font-size: 10px; color: var(--text-dim); font-style: italic; }}
  .model-cell {{ gap: 4px; }}

  .tag {{ display: inline-flex; align-items: center; font-size: 9.5px; font-weight: 800; letter-spacing: 0.04em; padding: 2px 7px; border-radius: 4px; }}
  .tag-bet {{ background: var(--green-bg); color: var(--green); }}
  .tag-edge {{ background: var(--teal-bg); color: var(--teal); }}
  .tag-fade {{ background: var(--red-bg); color: var(--red); }}
  .tag-flat {{ background: var(--panel-alt); color: var(--text-dim); }}

  /* EV BUILDER */
  .ev-panel {{ }}
  .ev-summary {{ display: flex; gap: 20px; align-items: center; font-size: 11.5px; margin-bottom: 12px; color: var(--text-dim); }}
  .ev-col-labels {{ display: grid; grid-template-columns: 1.6fr 70px 70px 70px 80px 70px 80px 70px 70px; font-size: 9px; font-weight: 700; letter-spacing: 0.05em; color: var(--text-dim); padding: 0 10px 6px; }}
  .ev-list {{ display: flex; flex-direction: column; gap: 4px; }}
  .ev-row {{ display: grid; grid-template-columns: 1.6fr 70px 70px 70px 80px 70px 80px 70px 70px; align-items: center; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; gap: 4px; }}
  .ev-row.ev-bet {{ border-color: rgba(62,207,110,0.4); }}
  .ev-cell {{ font-size: 11.5px; }}
  .ev-desc {{ display: flex; flex-direction: column; }}
  .ev-pick {{ font-weight: 700; }}
  .ev-sub {{ font-size: 9.5px; color: var(--text-dim); }}
  .mono-num {{ font-weight: 600; }}
  .ev-val {{ font-weight: 800; }}
  .ev-bet .ev-val {{ color: var(--green); }}
  .ev-edge .ev-val {{ color: var(--teal); }}
  .ev-fade .ev-val {{ color: var(--red); }}
  .track-btn {{ background: var(--panel-alt); border: 1px solid var(--border); color: var(--text); border-radius: 4px; padding: 4px 8px; font-size: 10px; font-weight: 700; cursor: pointer; }}
  .track-btn:hover {{ border-color: var(--gold); color: var(--gold); }}

  /* PROPS */
  .props-intro {{ max-width: 700px; margin: 0 0 18px; color: var(--text-dim); font-size: 12px; line-height: 1.6; }}
  .prop-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; }}
  .prop-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .prop-head {{ display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; margin-bottom: 8px; }}
  .prop-note {{ font-size: 11px; color: var(--text-dim); margin: 0; }}
  .prop-line-row {{ display: flex; justify-content: space-between; align-items: center; gap: 5px; font-size: 11px; padding: 5px 0; border-top: 1px solid var(--border); }}

  /* TRACKER */
  .tracker-summary {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 16px; }}
  .tsum-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .tsum-label {{ font-size: 9px; font-weight: 700; letter-spacing: 0.05em; color: var(--text-dim); margin-bottom: 4px; }}
  .tsum-val {{ font-size: 17px; font-weight: 800; }}
  .tracker-col-labels {{ display: grid; grid-template-columns: 1.8fr 90px 90px 70px 70px 90px 180px 90px 40px; font-size: 9px; font-weight: 700; letter-spacing: 0.05em; color: var(--text-dim); padding: 0 10px 6px; }}
  #tracker-list {{ display: flex; flex-direction: column; gap: 4px; }}
  .tracker-row {{ display: grid; grid-template-columns: 1.8fr 90px 90px 70px 70px 90px 180px 90px 40px; align-items: center; background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; gap: 4px; font-size: 11.5px; }}
  .tracker-row input[type="number"] {{ width: 70px; background: var(--panel-alt); border: 1px solid var(--border); color: var(--text); border-radius: 4px; padding: 3px 6px; font-family: 'JetBrains Mono', monospace; font-size: 11px; }}
  .status-btns {{ display: flex; gap: 4px; }}
  .status-btn {{ border: 1px solid var(--border); background: var(--panel-alt); color: var(--text-dim); border-radius: 4px; padding: 4px 8px; font-size: 9.5px; font-weight: 800; cursor: pointer; }}
  .status-btn.active-win {{ background: var(--green-bg); color: var(--green); border-color: var(--green); }}
  .status-btn.active-loss {{ background: var(--red-bg); color: var(--red); border-color: var(--red); }}
  .status-btn.active-push {{ background: rgba(212,173,51,0.14); color: var(--gold); border-color: var(--gold); }}
  .remove-btn {{ background: none; border: none; color: var(--text-dim); cursor: pointer; font-size: 14px; }}
  .profit-pos {{ color: var(--green); font-weight: 700; }}
  .profit-neg {{ color: var(--red); font-weight: 700; }}
  #tracker-empty {{ color: var(--text-dim); font-size: 12px; padding: 30px; text-align: center; border: 1px dashed var(--border); border-radius: 8px; }}

  .empty-state {{ background: var(--panel); border: 1px dashed var(--border); border-radius: 8px; padding: 22px; text-align: center; color: var(--text-dim); font-size: 12px; }}
  #no-results {{ display: none; text-align: center; color: var(--text-dim); padding: 40px 0; font-size: 12px; }}

  footer {{ text-align: center; padding: 22px 4vw 46px; color: var(--text-dim); font-size: 10.5px; line-height: 1.7; border-top: 1px solid var(--border); max-width: 900px; margin: 0 auto; }}
  footer b {{ color: var(--text); }}
</style>
</head>
<body>

<input type="radio" name="tabs" id="tab-games" checked>
<input type="radio" name="tabs" id="tab-ev">
<input type="radio" name="tabs" id="tab-props">
<input type="radio" name="tabs" id="tab-tracker">

<header>
  <div class="brand-row">
    <div class="brand">
      <div class="brand-icon">CFB</div>
      <div><div class="brand-title">CFB MODEL</div><div class="brand-sub">COLLEGE FOOTBALL &middot; BETTING ANALYTICS</div></div>
    </div>
    <div class="meta-strip">
      <div>GAMES <b>{game_count}</b></div>
      <div>PRICED <b>{model_count}</b></div>
      <div>SNAPSHOT <b>{generated_at}</b></div>
      <span class="snap-badge">STATIC SNAPSHOT</span>
    </div>
  </div>
  <div class="tab-row">
    <label class="tab-label" for="tab-games">GAMES</label>
    <label class="tab-label" for="tab-ev">EV BUILDER</label>
    <label class="tab-label" for="tab-props">PROP MARKETS</label>
    <label class="tab-label" for="tab-tracker">TRACKER</label>
  </div>
  <div class="controls-row">
    <input id="search" type="text" placeholder="search a team..." autocomplete="off" oninput="filterCards(this.value)">
    <div class="legend">
      <span><span class="tag tag-bet">BET</span> EV &ge; {bet_threshold}%</span>
      <span><span class="tag tag-edge">EDGE</span> 0&ndash;{bet_threshold}%</span>
      <span><span class="tag tag-fade">FADE</span> EV &le; 0%</span>
    </div>
  </div>
  <noscript><span style="display:block;color:var(--text-dim);font-size:11px;padding:0 4vw 8px;">(Search + Tracker need JavaScript. Games, EV Builder, and Prop Markets all work without it.)</span></noscript>
</header>

<main>
  <div class="panel games-panel">
{game_sections}
    <p id="no-results">No games match that search.</p>
  </div>
  <div class="panel ev-panel">
{ev_section}
  </div>
  <div class="panel props-panel">
{props_section}
  </div>
  <div class="panel tracker-panel">
    <div class="tracker-summary">
      <div class="tsum-box"><div class="tsum-label">TRACKED</div><div class="tsum-val" id="tsum-count">0</div></div>
      <div class="tsum-box"><div class="tsum-label">RECORD</div><div class="tsum-val" id="tsum-record">0-0-0</div></div>
      <div class="tsum-box"><div class="tsum-label">STAKED</div><div class="tsum-val" id="tsum-staked">$0</div></div>
      <div class="tsum-box"><div class="tsum-label">PROFIT</div><div class="tsum-val" id="tsum-profit">$0</div></div>
      <div class="tsum-box"><div class="tsum-label">ROI</div><div class="tsum-val" id="tsum-roi">0%</div></div>
    </div>
    <div class="tracker-col-labels"><span>DESCRIPTION</span><span>DATE</span><span>TYPE</span><span>PRICE</span><span>EV%</span><span>STAKE</span><span>STATUS</span><span>PROFIT</span><span></span></div>
    <div id="tracker-list"></div>
    <p id="tracker-empty">No plays tracked yet. Go to EV Builder and click "+ Track" on any play.</p>
  </div>
</main>

<footer>
  Built from <b>CFBD</b> (team data, SP+ ratings), <b>The Odds API</b> (moneyline/spread/total), and <b>PrizePicks via OddsPapi</b> (player props).
  <br><br>
  <b>Model / EV:</b> the 2026 season hasn't started, so there's no in-season form yet for any team. The model line uses
  each team's most recent SP+ rating (calibrated to approximate point spread) plus a home-field edge and uncertainty band
  both fit empirically from real 2021&ndash;2025 games. Games where either team is outside the FBS database (mostly FCS
  opponents) show "no line" rather than a fabricated number. EV Builder currently prices moneyline only.
  <br><br>
  <b>Tracker</b> is stored only in this browser (localStorage) &mdash; it does not sync anywhere or place real bets.
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

  // ---- Tracker (localStorage-backed, this browser only) ----
  var TRACKER_KEY = 'cfb_model_tracker_v1';

  function loadTracker() {{
    try {{ return JSON.parse(localStorage.getItem(TRACKER_KEY)) || []; }} catch (e) {{ return []; }}
  }}
  function saveTracker(items) {{ localStorage.setItem(TRACKER_KEY, JSON.stringify(items)); }}

  function addToTracker(play) {{
    var items = loadTracker();
    items.unshift({{
      id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
      description: play.description, date: play.date, type: play.type,
      price: play.price, ev: play.ev, stake: 50, status: null,
    }});
    saveTracker(items);
    renderTracker();
    document.getElementById('tab-tracker').checked = true;
  }}

  function americanToDecimal(odds) {{
    odds = Number(odds);
    return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
  }}

  function computeProfit(item) {{
    if (!item.status || !item.price) return null;
    var stake = Number(item.stake) || 0;
    if (item.status === 'push') return 0;
    if (item.status === 'loss') return -stake;
    var dec = americanToDecimal(item.price);
    return stake * dec - stake;
  }}

  function updateStake(id, value) {{
    var items = loadTracker();
    var it = items.find(function(x) {{ return x.id === id; }});
    if (it) {{ it.stake = value; saveTracker(items); renderTracker(); }}
  }}
  function setStatus(id, status) {{
    var items = loadTracker();
    var it = items.find(function(x) {{ return x.id === id; }});
    if (it) {{ it.status = (it.status === status ? null : status); saveTracker(items); renderTracker(); }}
  }}
  function removeItem(id) {{
    saveTracker(loadTracker().filter(function(x) {{ return x.id !== id; }}));
    renderTracker();
  }}

  function fmtMoney(n) {{
    var sign = n < 0 ? '-' : '';
    return sign + '$' + Math.abs(n).toFixed(2).replace(/\\.00$/, '');
  }}

  function renderTracker() {{
    var items = loadTracker();
    var list = document.getElementById('tracker-list');
    var empty = document.getElementById('tracker-empty');
    list.innerHTML = '';
    empty.style.display = items.length ? 'none' : 'block';

    var wins = 0, losses = 0, pushes = 0, staked = 0, profit = 0;

    items.forEach(function(it) {{
      var p = computeProfit(it);
      if (it.status === 'win') wins++;
      if (it.status === 'loss') losses++;
      if (it.status === 'push') pushes++;
      if (it.status) staked += Number(it.stake) || 0;
      if (p !== null) profit += p;

      var row = document.createElement('div');
      row.className = 'tracker-row';
      row.innerHTML =
        '<div>' + it.description + '</div>' +
        '<div>' + it.date + '</div>' +
        '<div>' + it.type + '</div>' +
        '<div>' + (it.price > 0 ? '+' : '') + it.price + '</div>' +
        '<div>' + (it.ev > 0 ? '+' : '') + it.ev + '%</div>' +
        '<div><input type="number" value="' + it.stake + '" min="0" step="5" onchange="updateStake(\\'' + it.id + '\\', this.value)"></div>' +
        '<div class="status-btns">' +
          '<button class="status-btn' + (it.status === 'win' ? ' active-win' : '') + '" onclick="setStatus(\\'' + it.id + '\\',\\'win\\')">WIN</button>' +
          '<button class="status-btn' + (it.status === 'loss' ? ' active-loss' : '') + '" onclick="setStatus(\\'' + it.id + '\\',\\'loss\\')">LOSS</button>' +
          '<button class="status-btn' + (it.status === 'push' ? ' active-push' : '') + '" onclick="setStatus(\\'' + it.id + '\\',\\'push\\')">PUSH</button>' +
        '</div>' +
        '<div class="' + (p === null ? '' : (p >= 0 ? 'profit-pos' : 'profit-neg')) + '">' + (p === null ? '&mdash;' : fmtMoney(p)) + '</div>' +
        '<div><button class="remove-btn" onclick="removeItem(\\'' + it.id + '\\')">&times;</button></div>';
      list.appendChild(row);
    }});

    document.getElementById('tsum-count').textContent = items.length;
    document.getElementById('tsum-record').textContent = wins + '-' + losses + '-' + pushes;
    document.getElementById('tsum-staked').textContent = fmtMoney(staked);
    var profitEl = document.getElementById('tsum-profit');
    profitEl.textContent = fmtMoney(profit);
    profitEl.className = 'tsum-val ' + (profit >= 0 ? 'profit-pos' : 'profit-neg');
    document.getElementById('tsum-roi').textContent = staked > 0 ? ((profit / staked) * 100).toFixed(1) + '%' : '0%';
  }}

  renderTracker();
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
        gen_label = gen_dt.strftime("%b %-d %-I:%M%p UTC") if gen_dt else "unknown"
    except Exception:
        gen_label = generated_at or "unknown"

    model_count = sum(1 for g in games if g.get("has_model_line"))

    html_out = PAGE_TEMPLATE.format(
        game_count=len(games),
        model_count=model_count,
        generated_at=esc(gen_label),
        bet_threshold=int(BET_THRESHOLD),
        game_sections=build_game_sections(games),
        ev_section=build_ev_builder_section(games),
        props_section=build_props_section(prop_catalog, props),
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html_out)

    print(f"Wrote {OUT_PATH} ({len(games)} games, {model_count} priced, "
          f"{len(prop_catalog)} prop markets, {len(props)} live prop rows)")


if __name__ == "__main__":
    main()
