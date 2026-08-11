#!/usr/bin/env python3
"""
Render docs/data/latest.json into a single self-contained HTML dashboard
(docs/dashboard.html).

DESIGN v7: reskinned to match a reference file the user provided (angular
"GRIDIRON"-style quant terminal — Saira Condensed / Azeret Mono / Doto
fonts, skewed chips, helmet SVGs colored from real team data, an Edge
Board, Power Ratings, a Matchup Projector with a simulated margin
distribution + line decomposition, a Bet Card, Player Props, and a Tracker).

IMPORTANT — what changed vs. just reskinning, and why:
The reference file's sample data includes several things our real pipeline
does not actually produce, and rather than fabricate numbers to fill those
slots, this build either derives them honestly from data we do have, or
drops them with a disclosure in the footer:
  - Total/Team-total markets: DROPPED. We have no scoring/total model, only
    a margin model (SP+ diff -> expected margin -> spread + moneyline). The
    market switcher only offers Spread and Moneyline, both real.
  - modelSpread: DERIVED, not fabricated — it's just -model_predicted_margin
    from src/models/fair_odds.py, expressed in the reference's sign
    convention (negative = home favored, matching the book's spread_home).
  - Line decomposition: real, but only 2 components (our model only HAS 2
    inputs) — SP+ rating differential x fitted slope, and the fitted
    home-field/intercept constant. The reference's 7-component breakdown
    (EPA, travel, pace, QB posterior, etc.) assumed model internals we
    don't have.
  - sigma: our model has ONE league-wide residual std, not a true per-game
    sigma. Every game uses the same value. Disclosed in the footer.
  - Weather/venue/travel/pace/returning-production/situational flags: not
    fetched anywhere in this pipeline. Omitted rather than invented.
  - Futures markets: not fetched. Section removed entirely.
  - Closing Line Value / bankroll history: the reference's version assumes
    3 seasons of real settled bets, which we don't have (no real money has
    been wagered yet). Replaced with an actual live Tracker (localStorage,
    WIN/LOSS/PUSH grading, running P&L) that builds REAL history from here
    forward, instead of showing a fabricated backtest curve.

Regenerate whenever docs/data/latest.json is refreshed:

  python scripts/build_dashboard.py
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "docs/data/latest.json"
OUT_PATH = "docs/dashboard.html"

MIN_EDGE_POINTS = 1.9  # unified board/card threshold; ~= our 5pp moneyline edge threshold
                        # via the /2.6 rescale in ModelMath.priceGame (5 / 2.6 = 1.92)


def fmt_kickoff(iso):
    if not iso:
        return "TBD"
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return dt.strftime("%a %-m/%-d, %-I:%M%p UTC")
    except Exception:
        return "TBD"


def build_model_data(data: dict) -> dict:
    prior = data.get("preseason_prior") or {}
    slope = prior.get("slope")
    intercept = prior.get("intercept")
    residual_std = prior.get("residual_std")
    n_games_hist = prior.get("n_games")

    teams_raw = data.get("teams", [])
    teams = []
    for t in teams_raw:
        teams.append({
            "name": t["name"],
            "abbr": t["abbr"],
            "conf": t.get("conf") or "IND",
            "primary": t["primary"],
            "secondary": t["secondary"],
            "net": t["net"],
            "offSp": t.get("off_sp_rating"),
            "defSp": t.get("def_sp_rating"),
        })

    off_vals = [abs(t["offSp"]) for t in teams if t["offSp"] is not None]
    def_vals = [abs(t["defSp"]) for t in teams if t["defSp"] is not None]
    off_scale = round(max(off_vals) * 1.1, 1) if off_vals else 1.0
    def_scale = round(max(def_vals) * 1.1, 1) if def_vals else 1.0

    games_all = data.get("games", [])
    priced = [g for g in games_all if g.get("has_model_line")]

    games = []
    for g in priced:
        sp_diff = g.get("sp_rating_diff")
        model_margin = g.get("model_predicted_margin")
        model_spread = round(-model_margin, 2) if model_margin is not None else None

        mp_home = g.get("model_home_win_prob")
        ip_home = g.get("book_implied_prob_home")
        edge_pp_home = round((mp_home - ip_home) * 100, 2) if (mp_home is not None and ip_home is not None) else None

        comp_rating = round(-(slope * sp_diff), 2) if (slope is not None and sp_diff is not None) else 0.0
        comp_hfa = round(-intercept, 2) if intercept is not None else 0.0

        book = (g.get("book_used") or "market").replace("_", " ").title()

        note_parts = []
        if slope is not None and sp_diff is not None:
            note_parts.append(f"SP+ diff {sp_diff:+.1f} pts x fitted slope {slope:.2f}")
        if intercept is not None:
            note_parts.append(f"home-field constant {intercept:+.1f} pts (fit from {n_games_hist} 2021-2025 games)")
        if edge_pp_home is not None:
            note_parts.append(f"moneyline edge vs. book: {edge_pp_home:+.1f}pp on {esc_plain(g['home_team'])}")
        note = "Model: " + "; ".join(note_parts) + "." if note_parts else "Insufficient history to explain this line."

        games.append({
            "away": g["away_team"],
            "home": g["home_team"],
            "kickoff": fmt_kickoff(g.get("commence_time")),
            "book": book,
            "marketSpread": g.get("spread_home"),
            "modelSpread": model_spread,
            "marketTotal": g.get("total_over"),
            "marketMoneyline": g.get("moneyline_home"),
            "awayMoneyline": g.get("moneyline_away"),
            "sigma": residual_std,
            "components": [
                {"label": "SP+ rating differential x fitted slope", "points": comp_rating},
                {"label": "Home-field constant (fit, 2021-2025)", "points": comp_hfa},
            ],
            "flags": [],
            "note": note,
            "evHomePct": g.get("ev_home_pct"),
            "evAwayPct": g.get("ev_away_pct"),
        })

    props_catalog = data.get("prop_market_catalog", [])
    props_live = data.get("props", [])

    meta = {
        "modelName": "CFB",
        "sport": "EDGE",
        "subtitle": "SP+ preseason rating & market edge engine",
        "version": "v1.0",
        "dataAsOf": _fmt_generated_at(data.get("generated_at")),
        "gamesPriced": len(priced),
        "totalGames": len(games_all),
        "minEdge": MIN_EDGE_POINTS,
        "spSlope": slope,
        "spIntercept": intercept,
        "marginSd": residual_std,
        "spN": n_games_hist,
        "offScale": off_scale,
        "defScale": def_scale,
    }

    return {
        "meta": meta,
        "teams": teams,
        "games": games,
        "propCatalog": props_catalog,
        "propsLive": props_live,
    }


def esc_plain(s):
    return str(s) if s is not None else ""


def _fmt_generated_at(iso):
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return dt.strftime("%b %-d %-I:%M%p UTC")
    except Exception:
        return iso or "unknown"


# =============================================================================
# HTML / CSS shell (design tokens + layout). Static — no data substitution.
# =============================================================================

HEAD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CFB Model — Edge Dashboard</title>

<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@600;700;800&family=Azeret+Mono:wght@400;500;700&family=Doto:wght@600;800;900&display=swap" rel="stylesheet" />

<style>
:root {
  --bg:            #0A0D12;
  --panel:         #101722;
  --panel-deep:    #0C121B;
  --row-active:    #151E2B;
  --chip:          #161E28;

  --rule:          #212B39;
  --rule-strong:   #2A3543;
  --rule-faint:    #1A222E;
  --rule-row:      #161E28;

  --text:          #EDF2F8;
  --text-dim:      #C4CFDC;
  --muted:         #7E8C9E;
  --muted-2:       #61707F;
  --muted-3:       #5E6C7D;
  --muted-4:       #4E5C6D;

  --blue:          #2E7BFF;
  --blue-light:    #5BA4FF;
  --blue-link:     #6BA6FF;
  --green:         #17C26B;
  --red:           #FF5252;
  --amber:         #E0B44A;
  --salmon:        #FF8A7A;

  --font-display:  'Saira Condensed', 'Arial Narrow', sans-serif;
  --font-data:     'Azeret Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --font-led:      'Doto', 'Azeret Mono', monospace;

  --skew:          -11deg;
  --max-width:     1560px;
  --gutter:        32px;
}

* { box-sizing: border-box; }

html, body {
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-data);
}

a { color: var(--blue-link); text-decoration: none; }
a:hover { color: #A8C9FF; text-decoration: underline; }
::selection { background: #1D3F7A; }

.wrap { max-width: var(--max-width); margin: 0 auto; padding: 0 var(--gutter); }
.page { min-height: 100vh; padding-bottom: 80px; }

.topbar { border-bottom: 1px solid var(--rule); background: var(--bg); position: sticky; top: 0; z-index: 30; }
.topbar-inner {
  max-width: var(--max-width); margin: 0 auto; padding: 0 var(--gutter);
  display: flex; align-items: center; justify-content: space-between; gap: 28px; height: 62px; flex-wrap: wrap;
}
.brand { display: flex; align-items: center; gap: 14px; }
.brand-block { background: var(--blue); transform: skewX(var(--skew)); padding: 7px 16px; }
.brand-block > span, .brand-outline > span { display: inline-block; transform: skewX(11deg); }
.brand-block span { font-family: var(--font-display); font-weight: 800; font-size: 22px; letter-spacing: 0.06em; text-transform: uppercase; color: #fff; }
.brand-outline { border: 1px solid #38475A; transform: skewX(var(--skew)); padding: 6px 13px; }
.brand-outline span { font-family: var(--font-display); font-weight: 800; font-size: 20px; letter-spacing: 0.09em; text-transform: uppercase; }
.brand-sub { font-size: 9.5px; letter-spacing: 0.16em; text-transform: uppercase; color: #6D7C8E; line-height: 1.5; padding-left: 6px; }
.topbar-right { display: flex; align-items: center; gap: 14px; font-size: 11px; flex-wrap: wrap; }
.topbar-label { font-family: var(--font-display); font-weight: 700; font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase; color: #6D7C8E; }
.divider-v { width: 1px; height: 26px; background: var(--rule); }
.sync { display: flex; align-items: center; gap: 7px; color: #6D7C8E; }
.sync-dot { width: 7px; height: 7px; background: var(--green); display: inline-block; box-shadow: 0 0 8px var(--green); border-radius: 50%; }
#search { background: var(--chip); border: 1px solid var(--rule); color: var(--text); font-family: var(--font-data); font-size: 12px; padding: 6px 10px; border-radius: 3px; outline: none; width: 170px; }
#search::placeholder { color: var(--muted-3); }
#search:focus { border-color: var(--blue); }

.tabs { display: flex; gap: 3px; }
.tab { font-family: var(--font-display); font-weight: 700; cursor: pointer; border: none; transform: skewX(var(--skew)); padding: 6px 14px; font-size: 13px; letter-spacing: 0.09em; text-transform: uppercase; background: var(--chip); color: var(--muted); }
.tab > span { display: inline-block; transform: skewX(11deg); }
.tab.is-active { background: var(--text); color: var(--bg); }
.tab--market { padding: 6px 13px; font-size: 12.5px; letter-spacing: 0.08em; }
.tab--market.is-active { background: var(--blue); color: #fff; }

.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--rule); border-bottom: 1px solid var(--rule); }
.kpi { background: var(--panel); padding-bottom: 15px; }
.kpi-bar { height: 3px; background: var(--blue); }
.kpi.is-green .kpi-bar { background: var(--green); }
.kpi-body { padding: 13px 18px 0; }
.kpi-label { font-family: var(--font-display); font-weight: 700; font-size: 11.5px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted-2); margin-bottom: 9px; }
.kpi-value { font-family: var(--font-led); font-weight: 900; font-size: 34px; line-height: 1; letter-spacing: 0.01em; text-shadow: 0 0 16px rgba(46, 123, 255, 0.30); }
.kpi.is-green .kpi-value { color: var(--green); text-shadow: 0 0 16px rgba(23, 194, 107, 0.32); }
.kpi-sub { font-size: 10px; color: var(--muted-3); margin-top: 9px; line-height: 1.4; }

.section-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; border-bottom: 1px solid var(--rule-strong); padding-bottom: 10px; flex-wrap: wrap; }
.section-title { display: flex; align-items: center; gap: 11px; }
.section-flag { width: 5px; height: 17px; background: var(--blue); transform: skewX(var(--skew)); }
.section-flag.is-green { background: var(--green); }
.section-head h2 { font-family: var(--font-display); font-weight: 800; font-size: 20px; text-transform: uppercase; letter-spacing: 0.05em; margin: 0; }

.thead { font-family: var(--font-display); font-weight: 700; font-size: 10.5px; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted-2); padding: 10px 14px; border-bottom: 1px solid var(--rule-faint); display: grid; }
.num { text-align: right; }

.edge-grid { grid-template-columns: 1fr 72px 72px 62px 54px 46px 54px; }
.rate-grid { grid-template-columns: 26px 1fr 56px 1fr 100px; }
.card-row  { grid-template-columns: 1fr 58px 58px 46px 58px; }

.row { display: flex; align-items: stretch; border-bottom: 1px solid var(--rule-row); }
.row--click { cursor: pointer; }
.row--click.is-selected { background: var(--row-active); }
.row-accent { width: 4px; flex: none; }
.row-accent--thin { width: 3px; }
.row-body { flex: 1; display: grid; align-items: center; padding: 11px 14px; }

.matchup { display: flex; align-items: center; gap: 9px; }
.team-abbr { font-family: var(--font-display); font-weight: 700; font-size: 16px; letter-spacing: 0.03em; text-transform: uppercase; }
.at { font-size: 9.5px; color: var(--muted-4); letter-spacing: 0.1em; }
.meta { font-size: 9.5px; color: var(--muted-3); margin-top: 5px; display: flex; gap: 10px; white-space: nowrap; overflow: hidden; }

.cell-market { font-size: 12.5px; color: var(--muted); }
.cell-model  { font-size: 13px; font-weight: 700; }
.cell-edge   { font-family: var(--font-led); font-weight: 900; font-size: 17px; }
.cell-edge.is-pos { color: var(--green); }
.cell-edge.is-neg { color: var(--red); }
.cell-edge.is-off { color: var(--muted-4); }
.cell-prob   { font-size: 11.5px; color: var(--muted); }

.tier { display: inline-block; transform: skewX(var(--skew)); font-family: var(--font-display); font-weight: 800; font-size: 12.5px; letter-spacing: 0.06em; padding: 3px 9px; background: var(--text); color: var(--bg); }
.tier > span { display: inline-block; transform: skewX(11deg); }
.tier.is-off { background: var(--chip); color: var(--muted-4); }

.track-btn { font-family: var(--font-display); font-weight: 700; font-size: 10.5px; letter-spacing: 0.05em; text-transform: uppercase; background: var(--chip); color: var(--muted); border: 1px solid var(--rule); border-radius: 3px; padding: 4px 8px; cursor: pointer; }
.track-btn:hover { border-color: var(--blue); color: var(--blue-light); }

.table-foot { display: flex; justify-content: space-between; padding: 12px 14px 0; font-size: 10px; color: var(--muted-3); border-top: 1px solid var(--rule-faint); }

.helmet { display: block; flex: none; }
.helmet--flip { transform: scaleX(-1); }

.team-chip { width: 4px; height: 15px; flex: none; transform: skewX(var(--skew)); }
.diverge { position: relative; height: 13px; margin: 0 8px; }
.diverge-axis { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: var(--rule-strong); }
.diverge-bar { position: absolute; top: 2px; height: 9px; }
.diverge-def { background: #C4553F; }
.diverge-off { background: var(--blue); left: 50%; }
.scale-track { height: 7px; background: var(--rule-faint); }
.scale-fill { height: 7px; }

.projector { background: var(--panel); border-bottom: 1px solid var(--rule); }
.proj-head { padding: 20px 22px 18px; position: relative; overflow: hidden; }
.proj-head-inner { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
.proj-side { display: flex; flex-direction: column; align-items: center; gap: 8px; flex: none; }
.proj-bar { width: 58px; height: 5px; transform: skewX(var(--skew)); }
.proj-title { font-family: var(--font-display); font-weight: 800; font-size: 27px; letter-spacing: 0.02em; text-transform: uppercase; line-height: 1.05; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6); }
.proj-title .at-lg { font-size: 15px; color: rgba(255, 255, 255, 0.55); }
.proj-meta { font-size: 9.5px; color: rgba(255, 255, 255, 0.62); margin-top: 7px; letter-spacing: 0.05em; }

.scoreboard { display: flex; align-items: stretch; border-bottom: 1px solid var(--rule); background: var(--panel-deep); }
.score-cell { flex: 1; padding: 13px 18px; text-align: center; }
.score-cell--wide { flex: 1.2; }
.score-label { font-family: var(--font-display); font-weight: 700; font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted-2); margin-bottom: 7px; }
.score-value { font-family: var(--font-led); font-weight: 900; font-size: 42px; line-height: 1; text-shadow: 0 0 20px rgba(46, 123, 255, 0.32); }
.score-value.is-blue { color: var(--blue); text-shadow: 0 0 20px rgba(46, 123, 255, 0.45); }

.proj-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--rule); }
.proj-stat { background: var(--panel); padding: 12px 16px 13px; }
.proj-stat-label { font-family: var(--font-display); font-weight: 700; font-size: 10.5px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted-2); margin-bottom: 7px; }
.proj-stat-value { font-size: 15px; font-weight: 700; }
.proj-stat-sub { font-size: 9.5px; color: var(--muted-3); margin-top: 5px; }

.panel-pad { padding: 22px 22px 24px; }
.panel-pad--tight { padding: 0 22px 24px; }
.chart-head { display: flex; justify-content: space-between; align-items: baseline; font-family: var(--font-display); font-weight: 700; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted-2); margin-bottom: 11px; }
.chart-head--ruled { border-top: 1px solid var(--rule); padding-top: 18px; margin-bottom: 13px; }
.chart-head .legend { letter-spacing: 0.04em; text-transform: none; font-family: var(--font-data); font-weight: 400; font-size: 9.5px; color: var(--muted-4); }

.field { position: relative; height: 138px; background: #0D2418; background-image: repeating-linear-gradient(90deg, rgba(255,255,255,0.10) 0 1px, transparent 1px 10%); border: 1px solid #17402A; }
.field-shade { position: absolute; top: 0; bottom: 0; background: rgba(46, 123, 255, 0.14); }
.field-bars { position: absolute; inset: 0; display: flex; align-items: flex-end; gap: 2px; padding: 0 2px; }
.field-bar { flex: 1; background: rgba(255, 255, 255, 0.20); }
.field-bar.is-win { background: var(--blue-light); }
.field-zero { position: absolute; top: 0; bottom: 0; border-left: 1px dashed rgba(255, 255, 255, 0.35); }
.field-marker { position: absolute; top: 0; bottom: 0; border-left: 2px solid #FFD84D; box-shadow: 0 0 10px rgba(255, 216, 77, 0.5); }
.field-marker span { position: absolute; top: 5px; left: 4px; white-space: nowrap; font-family: var(--font-display); font-weight: 800; font-size: 11.5px; letter-spacing: 0.06em; background: #FFD84D; color: var(--bg); padding: 2px 6px; }
.axis { position: relative; height: 14px; margin-top: 6px; }
.axis span { position: absolute; transform: translateX(-50%); font-size: 9.5px; color: var(--muted-3); }
.chart-foot { display: flex; justify-content: space-between; font-size: 9.5px; color: var(--muted-3); margin-top: 4px; }
.chart-foot .cover { color: var(--blue-link); }
.chart-note { font-size: 10.5px; color: var(--muted); margin-top: 14px; line-height: 1.65; }

.decomp { position: relative; }
.decomp-axis { position: absolute; top: 0; bottom: 26px; right: 158px; width: 1px; background: #3A4757; }
.decomp-row { display: grid; grid-template-columns: 1fr 200px 46px; align-items: center; gap: 12px; padding: 4px 0; }
.decomp-label { font-size: 11px; line-height: 1.35; color: var(--text-dim); }
.decomp-track { height: 14px; position: relative; }
.decomp-base { position: absolute; top: 6.5px; left: 0; right: 0; height: 1px; background: var(--rule-faint); }
.decomp-bar { position: absolute; top: 0; bottom: 0; }
.decomp-value { font-size: 12px; text-align: right; font-weight: 700; }
.decomp-scale { position: relative; height: 12px; }
.decomp-scale span { position: absolute; transform: translateX(-50%); font-size: 9.5px; color: var(--muted-4); }
.decomp-total { display: grid; grid-template-columns: 1fr 200px 46px; gap: 12px; border-top: 1px solid var(--rule-strong); margin-top: 4px; padding-top: 11px; }
.decomp-total-label { font-family: var(--font-display); font-weight: 700; font-size: 14px; letter-spacing: 0.06em; text-transform: uppercase; }
.decomp-total-note { font-size: 10px; color: var(--muted-3); align-self: center; }
.decomp-total-value { font-size: 14px; text-align: right; font-weight: 700; }

.card-play { font-family: var(--font-display); font-weight: 700; font-size: 16px; letter-spacing: 0.02em; }
.card-note { font-size: 9.5px; color: var(--muted-3); margin-top: 4px; }
.card-price { font-size: 11px; text-align: right; color: var(--muted); }
.card-ev { font-family: var(--font-led); font-weight: 900; font-size: 16px; text-align: right; color: var(--green); }
.card-total { display: flex; justify-content: space-between; padding: 13px 22px; font-size: 11px; background: var(--panel-deep); }
.card-total-label { color: var(--muted-2); font-family: var(--font-display); font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; font-size: 11px; }

.pcard-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 8px; }
.pcard { background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 12px 14px; }
.pcard-head { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; margin-bottom: 8px; }
.pcard-note { font-size: 10.5px; color: var(--muted-3); margin: 0; line-height: 1.5; }
.pcard-line-row { display: flex; justify-content: space-between; align-items: center; gap: 5px; font-size: 11px; padding: 5px 0; border-top: 1px solid var(--rule-faint); }
.pchip { font-family: var(--font-display); font-weight: 800; font-size: 9.5px; letter-spacing: 0.06em; padding: 2px 7px; border-radius: 3px; }
.pchip.is-live { background: rgba(23,194,107,0.16); color: var(--green); }
.pchip.is-pending { background: var(--chip); color: var(--muted-4); }

.trk-summary { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: var(--rule); margin-bottom: 1px; }
.trk-box { background: var(--panel); padding: 12px 14px; }
.trk-label { font-family: var(--font-display); font-weight: 700; font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-2); margin-bottom: 6px; }
.trk-value { font-family: var(--font-led); font-weight: 900; font-size: 20px; }
.trk-value.is-pos { color: var(--green); }
.trk-value.is-neg { color: var(--red); }
.trk-grid { grid-template-columns: 1.6fr 70px 60px 60px 90px 150px 70px 34px; }
.trk-row { display: grid; grid-template-columns: 1.6fr 70px 60px 60px 90px 150px 70px 34px; align-items: center; background: var(--panel); border-bottom: 1px solid var(--rule-row); padding: 9px 14px; gap: 6px; font-size: 11px; }
.trk-row input[type=number] { width: 62px; background: var(--chip); border: 1px solid var(--rule); color: var(--text); border-radius: 3px; padding: 3px 6px; font-family: var(--font-data); font-size: 11px; }
.trk-status-btns { display: flex; gap: 3px; }
.trk-status-btn { border: 1px solid var(--rule); background: var(--chip); color: var(--muted-3); border-radius: 3px; padding: 3px 7px; font-size: 9px; font-weight: 800; cursor: pointer; font-family: var(--font-display); letter-spacing: 0.04em; }
.trk-status-btn.is-win { background: rgba(23,194,107,0.18); color: var(--green); border-color: var(--green); }
.trk-status-btn.is-loss { background: rgba(255,82,82,0.16); color: var(--red); border-color: var(--red); }
.trk-status-btn.is-push { background: rgba(224,180,74,0.16); color: var(--amber); border-color: var(--amber); }
.trk-remove { background: none; border: none; color: var(--muted-4); cursor: pointer; font-size: 15px; }
.trk-empty { color: var(--muted-3); font-size: 12px; padding: 26px; text-align: center; border: 1px dashed var(--rule); }

.note-block { font-size: 10.5px; color: var(--muted-3); margin-top: 14px; line-height: 1.7; border-top: 1px solid var(--rule-faint); padding-top: 12px; }
.empty-state { background: var(--panel); border: 1px dashed var(--rule); border-radius: 4px; padding: 20px; text-align: center; color: var(--muted-3); font-size: 12px; }

.main { display: grid; grid-template-columns: 1.3fr 1fr; gap: 26px; margin-top: 32px; align-items: start; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; margin-top: 44px; }
.mt-lg { margin-top: 44px; }
.mt-md { margin-top: 38px; }
.footer { margin-top: 48px; border-top: 1px solid var(--rule); padding-top: 16px; display: flex; justify-content: space-between; font-size: 9.5px; color: var(--muted-4); flex-wrap: wrap; gap: 8px; }

@media (max-width: 1280px) {
  .main, .split { grid-template-columns: 1fr; }
  .kpis { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
  <clipPath id="hshell">
    <path d="M25 4C36 4 43 10.5 43 19.5V23C43 25.5 41 27 38 27H14C8.5 27 5 23 5 18.5 5 10.5 14 4 25 4Z" />
  </clipPath>
</defs></svg>

<div id="app" class="page"></div>
"""


# =============================================================================
# Part 2 — pure model math (verbatim generic functions, no data-specific
# content, ported from the reference design with one real fix: the
# Moneyline branch now de-vigs using BOTH sides' prices via removeVig(),
# instead of comparing against the raw single-side implied probability
# (which still carries the book's hold) — consistent with how the rest of
# this pipeline's fair-odds math already de-vigs in src/models/fair_odds.py.
# =============================================================================

MATH_JS = """<script>
(function (root) {
  'use strict';

  function erf(x) {
    var s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  function normalCdf(x, mean, sd) { return 0.5 * (1 + erf((x - mean) / (sd * Math.SQRT2))); }
  function normalPdf(x, mean, sd) {
    return Math.exp(-Math.pow(x - mean, 2) / (2 * sd * sd)) / (sd * Math.sqrt(2 * Math.PI));
  }

  function fairAmerican(p) {
    return p >= 0.5 ? Math.round(-100 * p / (1 - p)) : Math.round(100 * (1 - p) / p);
  }

  function impliedProb(american) {
    return american < 0 ? (-american) / ((-american) + 100) : 100 / (american + 100);
  }

  function removeVig(priceA, priceB) {
    var a = impliedProb(priceA), b = impliedProb(priceB), s = a + b;
    return [a / s, b / s];
  }

  function expectedValue(p, american) {
    var payout = american > 0 ? american / 100 : 100 / (-american);
    return p * payout - (1 - p);
  }

  function signed(v, dp) { return (v > 0 ? '+' : '') + v.toFixed(dp === undefined ? 1 : dp); }
  function homeMargin(game) { return -game.modelSpread; }

  function tierFor(edge, opts) {
    var minEdge = opts && opts.minEdge != null ? opts.minEdge : 1.9;
    var e = Math.abs(edge);
    if (e < minEdge) return '\\u2014';
    if (e >= 3.2 && opts && opts.confident) return '3u';
    if (e >= 2.2) return '2u';
    return '1u';
  }

  function priceGame(game, opts) {
    opts = opts || {};
    var market  = opts.market || 'Spread';
    var minEdge = opts.minEdge != null ? opts.minEdge : 1.9;
    var abbrOf  = opts.abbrOf || function (name) { return name; };

    var sd = game.sigma;
    var hm = homeMargin(game);

    var marketLabel, modelLabel, edge, coverProb, playLabel, side = null;

    if (market === 'Moneyline') {
      var p    = 1 - normalCdf(0, hm, sd);
      var fair = removeVig(game.marketMoneyline, game.awayMoneyline);
      var vig  = fair[0];
      marketLabel = (game.marketMoneyline > 0 ? '+' : '') + game.marketMoneyline;
      modelLabel  = (fairAmerican(p) > 0 ? '+' : '') + fairAmerican(p);
      edge        = (p - vig) * 100;
      coverProb   = p;
      side        = p > vig ? game.home : game.away;
      playLabel   = abbrOf(side) + ' ML';
    } else {
      marketLabel = signed(game.marketSpread);
      modelLabel  = signed(game.modelSpread);
      edge        = game.marketSpread - game.modelSpread;
      coverProb   = edge > 0 ? 1 - normalCdf(-game.marketSpread, hm, sd)
                             :     normalCdf(-game.marketSpread, hm, sd);
      side        = edge > 0 ? game.home : game.away;
      playLabel   = abbrOf(side) + ' ' + signed(edge > 0 ? game.marketSpread : -game.marketSpread);
    }

    var confident = market === 'Spread';
    var edgeForTier = market === 'Moneyline' ? edge / 2.6 : edge;

    return {
      game: game, market: market, sd: sd,
      marketLabel: marketLabel, modelLabel: modelLabel,
      edge: edge, edgeForTier: edgeForTier,
      edgeLabel: market === 'Moneyline' ? signed(edge) + '%' : signed(edge),
      coverProb: coverProb, playLabel: playLabel, side: side,
      qualifies: Math.abs(edgeForTier) >= minEdge,
      tier: tierFor(edgeForTier, { minEdge: minEdge, confident: confident })
    };
  }

  function distribution(game, priced, opts) {
    opts = opts || {};
    var abbrOf = opts.abbrOf || function (n) { return n; };
    var bins   = opts.bins || 29;
    var market = priced.market;
    var sd     = priced.sd;

    var center = homeMargin(game), dsd = sd, threshold = -game.marketSpread;
    var signedTicks = true;
    var axisLabel = 'home margin, points';
    var markerLabel = signed(game.marketSpread);
    var title = 'Margin distribution';

    if (market === 'Moneyline') {
      threshold = 0; markerLabel = 'PK';
    }

    var betOver = priced.edge > 0;
    var lo = Math.round(center - 3.1 * dsd), hi = Math.round(center + 3.1 * dsd);
    var w = (hi - lo) / bins, out = [], peak = 0, i, mid, d;

    for (i = 0; i < bins; i++) {
      mid = lo + w * (i + 0.5);
      d = normalPdf(mid, center, dsd);
      if (d > peak) peak = d;
      out.push({ mid: mid, density: d, winning: betOver ? mid > threshold : mid < threshold });
    }
    out.forEach(function (b) { b.heightPct = Math.max(1.5, b.density / peak * 100); });

    var pos = function (v) { return Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100)); };

    return {
      title: title, axisLabel: axisLabel, lo: lo, hi: hi, center: center, sigma: dsd,
      bins: out, threshold: threshold, markerLabel: markerLabel,
      markerPct: pos(threshold), zeroPct: pos(signedTicks ? 0 : lo),
      shadeFromPct: betOver ? pos(threshold) : 0,
      shadeWidthPct: betOver ? 100 - pos(threshold) : pos(threshold),
      ticks: [0, 0.25, 0.5, 0.75, 1].map(function (f) {
        var v = lo + (hi - lo) * f;
        return { label: signedTicks ? signed(v, 0) : String(Math.round(v)), pct: pos(v) };
      })
    };
  }

  function decomposition(game) {
    var maxAbs = Math.max.apply(null, game.components.map(function (c) { return Math.abs(c.points); })) || 1;
    var step = maxAbs > 8 ? 4 : maxAbs > 4 ? 2 : 1;
    var scaleMax = Math.ceil(maxAbs / step) * step;

    return {
      scaleMax: scaleMax,
      sum: game.components.reduce(function (a, c) { return a + c.points; }, 0),
      rows: game.components.map(function (c) {
        var pct = Math.abs(c.points) / scaleMax * 50;
        return {
          label: c.label, points: c.points,
          pointsLabel: c.points === 0 ? '0.0' : signed(c.points),
          towardHome: c.points < 0,
          leftPct: c.points >= 0 ? 50 : 50 - pct,
          widthPct: Math.max(pct, 0.6)
        };
      })
    };
  }

  function buildBetCard(games, opts) {
    opts = opts || {};
    var limit = opts.limit || 10;
    return games
      .map(function (g) { return priceGame(g, opts); })
      .filter(function (p) { return p.qualifies; })
      .sort(function (a, b) { return Math.abs(b.edgeForTier) - Math.abs(a.edgeForTier); })
      .slice(0, limit);
  }

  function relativeLuminance(hex) {
    var c = [1, 3, 5].map(function (i) {
      var v = parseInt(hex.substr(i, 2), 16) / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }

  function lighten(hex, t) {
    return '#' + [1, 3, 5].map(function (i) {
      var v = parseInt(hex.substr(i, 2), 16);
      return Math.round(v + (255 - v) * t).toString(16).padStart(2, '0');
    }).join('');
  }

  function displayColor(hex) {
    if (!hex || hex[0] !== '#' || hex.length < 7) return '#8A94A3';
    var L = relativeLuminance(hex);
    return L < 0.055 ? lighten(hex, 0.52) : L < 0.14 ? lighten(hex, 0.34) : hex;
  }

  root.ModelMath = {
    erf: erf, normalCdf: normalCdf, normalPdf: normalPdf,
    fairAmerican: fairAmerican, impliedProb: impliedProb, removeVig: removeVig,
    expectedValue: expectedValue, signed: signed, homeMargin: homeMargin,
    tierFor: tierFor, priceGame: priceGame, distribution: distribution,
    decomposition: decomposition, buildBetCard: buildBetCard,
    displayColor: displayColor, lighten: lighten, relativeLuminance: relativeLuminance
  };
})(window);
</script>
"""

TAIL_HTML = """</body>
</html>
"""


# =============================================================================
# Part 3 — renderer. Builds the DOM from window.MODEL_DATA + window.ModelMath.
# Adapted from the reference design: Spread+Moneyline only (our only 2 real
# markets), no week tabs (not functionally wired even in the source design),
# no situational-flags chip (no real flags), no futures (no data fetched),
# real props "coming soon" catalog cards instead of fabricated projections,
# and a real localStorage Tracker in place of the fabricated CLV/bankroll
# history chart.
# =============================================================================

RENDERER_JS = """<script>
(function () {
  'use strict';

  var D = window.MODEL_DATA;
  var M = window.ModelMath;
  var MARKETS = ['Spread', 'Moneyline'];

  var state = { selected: 0, market: 'Moneyline', search: '' };

  var byName = {};
  D.teams.forEach(function (t) { byName[t.name] = t; });
  function team(name) { return byName[name] || { name: name, abbr: name, primary: '#8A94A3', secondary: '#E7EDF5' }; }
  function abbrOf(name) { return team(name).abbr; }

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function opts() { return { market: state.market, minEdge: D.meta.minEdge, abbrOf: abbrOf }; }

  function matchesSearch(g) {
    if (!state.search) return true;
    var q = state.search.toLowerCase();
    return g.away.toLowerCase().indexOf(q) !== -1 || g.home.toLowerCase().indexOf(q) !== -1;
  }

  function helmet(teamName, width, height, flip) {
    var t = team(teamName), p = esc(M.displayColor(t.primary)), s = esc(t.secondary || '#E7EDF5');
    return '<svg class="helmet' + (flip ? ' helmet--flip' : '') + '" viewBox="0 0 54 32" ' +
      'style="width:' + width + 'px;height:' + height + 'px">' +
      '<path d="M25 4C36 4 43 10.5 43 19.5V23C43 25.5 41 27 38 27H14C8.5 27 5 23 5 18.5 5 10.5 14 4 25 4Z" ' +
        'fill="' + p + '" stroke="rgba(255,255,255,0.52)" stroke-width="1.2"/>' +
      '<g clip-path="url(#hshell)"><path d="M8.5 17.5C12 9.5 18 6.6 25 6.6S38 9.5 41.5 17.5" fill="none" stroke="' + s + '" stroke-width="2.6"/></g>' +
      '<circle cx="16" cy="18.2" r="3.6" fill="rgba(0,0,0,0.32)"/>' +
      '<circle cx="16" cy="18.2" r="1.3" fill="rgba(0,0,0,0.5)"/>' +
      '<path d="M41.4 15.8c6.6 1.2 9 6 6.9 11.4" fill="none" stroke="' + s + '" stroke-width="2"/>' +
      '<path d="M42.8 20.4h5.9" fill="none" stroke="' + s + '" stroke-width="1.4"/>' +
      '<path d="M43.6 24.2h5" fill="none" stroke="' + s + '" stroke-width="1.4"/>' +
      '</svg>';
  }

  function renderTopbar() {
    return '' +
      '<div class="topbar"><div class="topbar-inner">' +
        '<div class="brand">' +
          '<div class="brand-block"><span>' + esc(D.meta.modelName) + '</span></div>' +
          '<div class="brand-outline"><span>' + esc(D.meta.sport) + '</span></div>' +
          '<div class="brand-sub">' + esc(D.meta.subtitle) + '<br>' + esc(D.meta.version) + '</div>' +
        '</div>' +
        '<div class="topbar-right">' +
          '<input id="search" type="text" placeholder="search a team..." autocomplete="off" oninput="window.__cfbSearch(this.value)">' +
          '<div class="divider-v"></div>' +
          '<div class="sync"><span class="sync-dot"></span><span>' + esc(D.meta.dataAsOf) + '</span></div>' +
        '</div>' +
      '</div></div>';
  }

  function renderKpis(card) {
    var k = [
      ['Games priced', String(D.meta.gamesPriced), D.meta.totalGames + ' total in slate \\u00b7 ' + (D.meta.totalGames - D.meta.gamesPriced) + ' unpriced (no FBS SP+ rating or no market line)', false],
      ['Qualifying edges', String(card.length), '\\u2265 ' + D.meta.minEdge.toFixed(1) + ' pt threshold, ' + state.market + ' market', false],
      ['Model \\u03c3 margin', D.meta.marginSd != null ? D.meta.marginSd.toFixed(1) : '\\u2014', 'league-wide residual std, fit from ' + (D.meta.spN || 0) + ' 2021-2025 games', false],
      ['SP+ slope fit', D.meta.spSlope != null ? D.meta.spSlope.toFixed(3) : '\\u2014', 'margin = slope\\u00d7SP+diff + ' + (D.meta.spIntercept != null ? D.meta.spIntercept.toFixed(2) : '\\u2014'), false]
    ];
    return '<div class="kpis">' + k.map(function (r) {
      return '<div class="kpi">' +
        '<div class="kpi-bar"></div>' +
        '<div class="kpi-body">' +
          '<div class="kpi-label">' + esc(r[0]) + '</div>' +
          '<div class="kpi-value">' + esc(r[1]) + '</div>' +
          '<div class="kpi-sub">' + esc(r[2]) + '</div>' +
        '</div></div>';
    }).join('') + '</div>';
  }

  function trackPayload(p) {
    var g = p.game;
    return esc(JSON.stringify({
      description: p.playLabel + ' \\u2014 ' + abbrOf(g.away) + ' at ' + abbrOf(g.home),
      date: g.kickoff, type: p.market,
      price: p.market === 'Moneyline' ? g.marketMoneyline : -110,
      edge: Math.round(p.edge * 10) / 10
    }));
  }

  function renderEdgeBoard(priced, card) {
    var visible = priced.filter(function (p) { return matchesSearch(p.game); });
    var head = '' +
      '<div class="section-head">' +
        '<div class="section-title"><div class="section-flag"></div><h2>Edge Board</h2></div>' +
        '<div class="tabs">' + MARKETS.map(function (m) {
          return '<button class="tab tab--market' + (m === state.market ? ' is-active' : '') + '" data-market="' + esc(m) + '"><span>' + esc(m) + '</span></button>';
        }).join('') + '</div>' +
      '</div>' +
      '<div class="thead edge-grid"><div>Matchup</div><div class="num">Market</div><div class="num">Model</div>' +
      '<div class="num">Edge</div><div class="num">Win%</div><div class="num">Play</div><div class="num"></div></div>';

    var rows = visible.map(function (p) {
      var g = p.game, a = team(g.away), h = team(g.home);
      var i = priced.indexOf(p);
      var cls = Math.abs(p.edgeForTier) < D.meta.minEdge ? 'is-off' : (p.edge > 0 ? 'is-pos' : 'is-neg');
      return '' +
        '<div class="row row--click' + (i === state.selected ? ' is-selected' : '') + '" data-game="' + i + '">' +
          '<div class="row-accent" style="background:linear-gradient(' + esc(M.displayColor(a.primary)) + ',' + esc(M.displayColor(h.primary)) + ')"></div>' +
          '<div class="row-body edge-grid">' +
            '<div><div class="matchup">' +
              helmet(g.away, 32, 19, false) +
              '<span class="team-abbr">' + esc(a.abbr) + '</span>' +
              '<span class="at">AT</span>' +
              helmet(g.home, 32, 19, true) +
              '<span class="team-abbr">' + esc(h.abbr) + '</span>' +
            '</div>' +
            '<div class="meta"><span>' + esc(g.kickoff) + '</span><span>' + esc(g.book) + '</span></div></div>' +
            '<div class="num cell-market">' + esc(p.marketLabel) + '</div>' +
            '<div class="num cell-model">' + esc(p.modelLabel) + '</div>' +
            '<div class="num cell-edge ' + cls + '">' + esc(p.edgeLabel) + '</div>' +
            '<div class="num cell-prob">' + (p.coverProb * 100).toFixed(1) + '%</div>' +
            '<div class="num"><span class="tier' + (p.tier === '\\u2014' ? ' is-off' : '') + '"><span>' + esc(p.tier) + '</span></span></div>' +
            '<div class="num"><button class="track-btn" onclick="event.stopPropagation();window.__cfbTrack(' + trackPayload(p) + ')">+TRK</button></div>' +
          '</div>' +
        '</div>';
    }).join('');

    var foot = '<div class="table-foot"><span>Edge stated in points of expected value against the posted number. ' +
      'Threshold ' + D.meta.minEdge.toFixed(1) + ' pts.</span><span>' + card.length + ' qualifying plays \\u00b7 ' + visible.length + ' shown</span></div>';

    return head + (visible.length ? rows : '<div class="empty-state">No games match that search.</div>') + foot;
  }

  function renderRatings() {
    var maxNet = Math.max.apply(null, D.teams.map(function (t) { return Math.abs(t.net); })) * 1.06 || 1;
    var head = '' +
      '<div class="section-head mt-lg"><div class="section-title"><div class="section-flag"></div>' +
      '<h2>Power Ratings \\u2014 SP+ Preseason</h2></div></div>' +
      '<div class="thead rate-grid"><div>#</div><div>Team</div><div class="num">Net</div>' +
      '<div style="text-align:center">Off \\u2190 \\u2192 Def</div><div class="num">Scale</div></div>';

    var visible = D.teams.filter(function (t) { return !state.search || t.name.toLowerCase().indexOf(state.search.toLowerCase()) !== -1; });

    var rows = visible.slice().sort(function (a, b) { return b.net - a.net; }).map(function (t) {
      var c = M.displayColor(t.primary);
      var hasSplit = t.offSp != null && t.defSp != null;
      var offW = hasSplit ? Math.min(50, Math.abs(t.offSp) / D.meta.offScale * 50).toFixed(1) : 0;
      var defW = hasSplit ? Math.min(50, Math.abs(t.defSp) / D.meta.defScale * 50).toFixed(1) : 0;
      var i = D.teams.indexOf(t);
      return '' +
        '<div class="row"><div class="row-body rate-grid" style="padding:8px 14px;font-size:12px">' +
          '<div style="color:var(--muted-4)">' + (i + 1) + '</div>' +
          '<div class="matchup" style="overflow:hidden;white-space:nowrap">' +
            '<span class="team-chip" style="background:' + esc(c) + '"></span>' +
            '<span class="team-abbr" style="font-size:15px">' + esc(t.abbr) + '</span>' +
            '<span style="color:var(--muted-4);font-size:9.5px;letter-spacing:.08em">' + esc(t.conf) + '</span>' +
          '</div>' +
          '<div class="num" style="font-weight:700">' + M.signed(t.net) + '</div>' +
          (hasSplit ?
            '<div class="diverge"><div class="diverge-axis"></div>' +
              '<div class="diverge-bar diverge-def" style="left:' + (50 - defW) + '%;width:' + defW + '%" title="Defense SP+: ' + t.defSp + '"></div>' +
              '<div class="diverge-bar diverge-off" style="width:' + offW + '%" title="Offense SP+: ' + t.offSp + '"></div>' +
            '</div>' : '<div style="color:var(--muted-4);text-align:center;font-size:10px">\\u2014</div>') +
          '<div style="padding-left:14px"><div class="scale-track">' +
            '<div class="scale-fill" style="width:' + (Math.abs(t.net) / maxNet * 100).toFixed(1) + '%;background:' + esc(c) + '"></div>' +
          '</div></div>' +
        '</div></div>';
    }).join('');

    return head + (rows || '<div class="empty-state">No teams match that search.</div>');
  }

  function renderProjector(p) {
    if (!p) return '<div class="empty-state">No priced games to project.</div>';
    var g = p.game, a = team(g.away), h = team(g.home);
    var dist = M.distribution(g, p, { abbrOf: abbrOf });
    var dec = M.decomposition(g);
    var pWin = 1 - M.normalCdf(0, M.homeMargin(g), p.sd);
    var aC = M.displayColor(a.primary), hC = M.displayColor(h.primary);
    var split = 'linear-gradient(100deg,' + aC + '55 0%,' + aC + '18 33%,' +
      'var(--panel-deep) 46%,var(--panel-deep) 54%,' + hC + '18 67%,' + hC + '55 100%)';

    return '' +
      '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Matchup Projector</h2></div></div>' +
      '<div class="projector">' +

        '<div class="proj-head" style="background:' + split + '"><div class="proj-head-inner">' +
          '<div class="proj-side">' + helmet(g.away, 86, 51, false) +
            '<div class="proj-bar" style="background:' + esc(aC) + '"></div></div>' +
          '<div style="text-align:center;flex:1">' +
            '<div class="proj-title">' + esc(a.abbr) + ' <span class="at-lg">AT</span> ' + esc(h.abbr) + '</div>' +
            '<div class="proj-meta">' + esc(g.kickoff) + ' \\u00b7 line: ' + esc(g.book) + '</div>' +
          '</div>' +
          '<div class="proj-side">' + helmet(g.home, 86, 51, true) +
            '<div class="proj-bar" style="background:' + esc(hC) + '"></div></div>' +
        '</div></div>' +

        '<div class="scoreboard">' +
          '<div class="score-cell"><div class="score-label">' + esc(a.abbr) + ' margin</div>' +
            '<div class="score-value">' + M.signed(-g.modelSpread) + '</div></div>' +
          '<div class="divider-v" style="height:auto"></div>' +
          '<div class="score-cell score-cell--wide"><div class="score-label">Win prob</div>' +
            '<div class="score-value is-blue">' + (pWin * 100).toFixed(0) + '%</div></div>' +
        '</div>' +

        '<div class="proj-pair">' +
          '<div class="proj-stat"><div class="proj-stat-label">Model spread</div>' +
            '<div class="proj-stat-value">' + M.signed(g.modelSpread) + '</div>' +
            '<div class="proj-stat-sub">market ' + M.signed(g.marketSpread) + '</div></div>' +
          '<div class="proj-stat"><div class="proj-stat-label">Fair moneyline</div>' +
            '<div class="proj-stat-value">' + (M.fairAmerican(pWin) > 0 ? '+' : '') + M.fairAmerican(pWin) + '</div>' +
            '<div class="proj-stat-sub">posted ' + (g.marketMoneyline > 0 ? '+' : '') + g.marketMoneyline + '</div></div>' +
        '</div>' +

        '<div class="panel-pad">' +
          '<div class="chart-head"><span>' + esc(dist.title) + '</span><span>\\u03c3 ' + dist.sigma.toFixed(1) + ' (league-wide)</span></div>' +
          '<div class="field">' +
            '<div class="field-shade" style="left:' + dist.shadeFromPct.toFixed(2) + '%;width:' + dist.shadeWidthPct.toFixed(2) + '%"></div>' +
            '<div class="field-bars">' + dist.bins.map(function (b) {
              return '<div class="field-bar' + (b.winning ? ' is-win' : '') + '" style="height:' + b.heightPct.toFixed(1) + '%"></div>';
            }).join('') + '</div>' +
            '<div class="field-zero" style="left:' + dist.zeroPct.toFixed(2) + '%"></div>' +
            '<div class="field-marker" style="left:' + dist.markerPct.toFixed(2) + '%"><span>' + esc(dist.markerLabel) + '</span></div>' +
          '</div>' +
          '<div class="axis">' + dist.ticks.map(function (t) {
            return '<span style="left:' + t.pct.toFixed(2) + '%">' + esc(t.label) + '</span>';
          }).join('') + '</div>' +
          '<div class="chart-foot"><span>' + esc(dist.axisLabel) + '</span>' +
            '<span class="cover">shaded: normal-model probability, ' + (p.coverProb * 100).toFixed(1) + '%</span></div>' +
          '<div class="chart-note">' + esc(g.note) + '</div>' +
        '</div>' +

        '<div class="panel-pad--tight">' +
          '<div class="chart-head chart-head--ruled"><span>Line decomposition</span>' +
            '<span class="legend">\\u2190 ' + esc(h.abbr) + ' \\u00b7 ' + esc(a.abbr) + ' \\u2192</span></div>' +
          '<div class="decomp"><div class="decomp-axis"></div>' +
            dec.rows.map(function (r) {
              var col = r.points === 0 ? '#3A4757' : (r.towardHome ? hC : aC);
              return '<div class="decomp-row">' +
                '<div class="decomp-label">' + esc(r.label) + '</div>' +
                '<div class="decomp-track"><div class="decomp-base"></div>' +
                  '<div class="decomp-bar" style="left:' + r.leftPct + '%;width:' + r.widthPct.toFixed(2) + '%;background:' + esc(col) + '"></div>' +
                '</div>' +
                '<div class="decomp-value" style="color:' + esc(col) + '">' + esc(r.pointsLabel) + '</div>' +
              '</div>';
            }).join('') +
            '<div class="decomp-row"><div></div><div class="decomp-scale">' +
              '<span style="left:0%">\\u2212' + dec.scaleMax + '</span><span style="left:50%">0</span>' +
              '<span style="left:100%">+' + dec.scaleMax + '</span></div><div></div></div>' +
          '</div>' +
          '<div class="decomp-total"><div class="decomp-total-label">Model line</div>' +
            '<div class="decomp-total-note">sum of components</div>' +
            '<div class="decomp-total-value">' + M.signed(g.modelSpread) + '</div></div>' +
        '</div>' +
      '</div>';
  }

  function renderBetCard(card) {
    return '' +
      '<div class="section-head mt-md"><div class="section-title"><div class="section-flag is-green"></div>' +
      '<h2>Bet Card</h2></div></div>' +
      '<div class="projector">' +
        (card.length ? card.map(function (c) {
          var col = c.side ? M.displayColor(team(c.side).primary) : 'var(--blue)';
          var ev = M.expectedValue(c.coverProb, c.market === 'Moneyline' ? c.game.marketMoneyline : -110) * 100;
          return '<div class="row"><div class="row-accent" style="background:' + esc(col) + '"></div>' +
            '<div class="row-body card-row">' +
              '<div><div class="card-play">' + esc(c.playLabel) + '</div>' +
                '<div class="card-note">' + esc(abbrOf(c.game.away) + ' at ' + abbrOf(c.game.home) + ' \\u00b7 ' + c.game.kickoff) + '</div></div>' +
              '<div class="card-price">' + (c.market === 'Moneyline' ? esc(c.marketLabel) : '-110') + '</div>' +
              '<div class="card-ev">' + (ev >= 0 ? '+' : '') + ev.toFixed(1) + '%</div>' +
              '<div class="num"><span class="tier"><span>' + esc(c.tier) + '</span></span></div>' +
              '<div class="num"><button class="track-btn" onclick="window.__cfbTrack(' + trackPayload(c) + ')">+TRK</button></div>' +
            '</div></div>';
        }).join('') : '<div class="empty-state">No plays clear the ' + D.meta.minEdge.toFixed(1) + '-pt threshold on ' + state.market + ' right now.</div>') +
      '</div>';
  }

  function renderProps() {
    var catalog = D.propCatalog || [];
    var live = D.propsLive || [];
    var byMarket = {};
    live.forEach(function (p) { (byMarket[p.market_name] = byMarket[p.market_name] || []).push(p); });
    var liveCount = catalog.filter(function (m) { return byMarket[m] && byMarket[m].length; }).length;

    var intro = liveCount ? '' : '<p class="pcard-note" style="margin-bottom:14px">Real player-prop markets confirmed available on PrizePicks for college football. ' +
      'None are priced for this slate yet (normal this far from kickoff) \\u2014 any market with posted lines switches to LIVE automatically on the next data refresh.</p>';

    if (!catalog.length) {
      return '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Player Props</h2></div></div>' +
        '<div class="empty-state">No prop market catalog loaded.</div>';
    }

    var cards = catalog.map(function (m) {
      var rows = byMarket[m] || [];
      if (rows.length) {
        return '<div class="pcard"><div class="pcard-head"><span>' + esc(m) + '</span><span class="pchip is-live">LIVE</span></div>' +
          rows.map(function (r) {
            return '<div class="pcard-line-row"><span>' + esc(r.player_name) + ' \\u00b7 ' + esc(r.line) + '</span>' +
              '<span>O ' + esc(r.over_price) + ' / U ' + esc(r.under_price) + '</span></div>';
          }).join('') + '</div>';
      }
      return '<div class="pcard"><div class="pcard-head"><span>' + esc(m) + '</span><span class="pchip is-pending">NOT POSTED</span></div>' +
        '<p class="pcard-note">Confirmed market on PrizePicks. Opens closer to kickoff week.</p></div>';
    }).join('');

    return '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Player Props</h2></div></div>' +
      intro + '<div class="pcard-grid">' + cards + '</div>';
  }

  /* ---- Tracker (localStorage, this browser only, real bets you log) ---- */

  var TRK_KEY = 'cfb_model_tracker_v1';
  function loadTrk() { try { return JSON.parse(localStorage.getItem(TRK_KEY)) || []; } catch (e) { return []; } }
  function saveTrk(items) { localStorage.setItem(TRK_KEY, JSON.stringify(items)); }

  window.__cfbTrack = function (play) {
    var items = loadTrk();
    items.unshift({
      id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
      description: play.description, date: play.date, type: play.type,
      price: play.price, edge: play.edge, stake: 50, status: null
    });
    saveTrk(items);
    render();
    var el = document.getElementById('trk-section');
    if (el && typeof el.scrollIntoView === 'function') el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  function americanToDecimal(odds) {
    odds = Number(odds);
    return odds > 0 ? 1 + odds / 100 : 1 + 100 / Math.abs(odds);
  }
  function computeProfit(item) {
    if (!item.status || !item.price) return null;
    var stake = Number(item.stake) || 0;
    if (item.status === 'push') return 0;
    if (item.status === 'loss') return -stake;
    return stake * americanToDecimal(item.price) - stake;
  }

  window.__cfbTrkStake = function (id, value) {
    var items = loadTrk();
    var it = items.find(function (x) { return x.id === id; });
    if (it) { it.stake = value; saveTrk(items); render(); }
  };
  window.__cfbTrkStatus = function (id, status) {
    var items = loadTrk();
    var it = items.find(function (x) { return x.id === id; });
    if (it) { it.status = (it.status === status ? null : status); saveTrk(items); render(); }
  };
  window.__cfbTrkRemove = function (id) {
    saveTrk(loadTrk().filter(function (x) { return x.id !== id; }));
    render();
  };

  function fmtMoney(n) {
    var sign = n < 0 ? '-' : '';
    return sign + '$' + Math.abs(n).toFixed(2).replace(/\\.00$/, '');
  }

  function renderTracker() {
    var items = loadTrk();
    var wins = 0, losses = 0, pushes = 0, staked = 0, profit = 0;
    items.forEach(function (it) {
      var p = computeProfit(it);
      if (it.status === 'win') wins++;
      if (it.status === 'loss') losses++;
      if (it.status === 'push') pushes++;
      if (it.status) staked += Number(it.stake) || 0;
      if (p !== null) profit += p;
    });

    var summary = '<div class="trk-summary">' +
      '<div class="trk-box"><div class="trk-label">Tracked</div><div class="trk-value">' + items.length + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Record</div><div class="trk-value">' + wins + '-' + losses + '-' + pushes + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Staked</div><div class="trk-value">' + fmtMoney(staked) + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Profit</div><div class="trk-value ' + (profit >= 0 ? 'is-pos' : 'is-neg') + '">' + fmtMoney(profit) + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">ROI</div><div class="trk-value">' + (staked > 0 ? ((profit / staked) * 100).toFixed(1) + '%' : '0%') + '</div></div>' +
    '</div>';

    if (!items.length) {
      return '<div class="section-head" id="trk-section"><div class="section-title"><div class="section-flag is-green"></div><h2>Tracker</h2></div></div>' +
        summary + '<div class="trk-empty">No plays tracked yet. Click +TRK on any Edge Board row or Bet Card play.</div>';
    }

    var rows = items.map(function (it) {
      var p = computeProfit(it);
      return '<div class="trk-row">' +
        '<div>' + esc(it.description) + '</div>' +
        '<div>' + esc(it.date) + '</div>' +
        '<div>' + esc(it.type) + '</div>' +
        '<div>' + (it.price > 0 ? '+' : '') + esc(it.price) + '</div>' +
        '<div>' + (it.edge > 0 ? '+' : '') + esc(it.edge) + (it.type === 'Moneyline' ? 'pp' : 'pt') + '</div>' +
        '<div class="trk-status-btns">' +
          '<input type="number" value="' + it.stake + '" min="0" step="5" onchange="window.__cfbTrkStake(\\'' + it.id + '\\', this.value)">' +
          '<button class="trk-status-btn' + (it.status === 'win' ? ' is-win' : '') + '" onclick="window.__cfbTrkStatus(\\'' + it.id + '\\',\\'win\\')">W</button>' +
          '<button class="trk-status-btn' + (it.status === 'loss' ? ' is-loss' : '') + '" onclick="window.__cfbTrkStatus(\\'' + it.id + '\\',\\'loss\\')">L</button>' +
          '<button class="trk-status-btn' + (it.status === 'push' ? ' is-push' : '') + '" onclick="window.__cfbTrkStatus(\\'' + it.id + '\\',\\'push\\')">P</button>' +
        '</div>' +
        '<div style="color:' + (p === null ? 'var(--muted-4)' : (p >= 0 ? 'var(--green)' : 'var(--red)')) + ';font-weight:700">' + (p === null ? '\\u2014' : fmtMoney(p)) + '</div>' +
        '<div><button class="trk-remove" onclick="window.__cfbTrkRemove(\\'' + it.id + '\\')">\\u00d7</button></div>' +
      '</div>';
    }).join('');

    return '<div class="section-head" id="trk-section"><div class="section-title"><div class="section-flag is-green"></div><h2>Tracker</h2></div></div>' +
      summary +
      '<div class="thead trk-grid"><div>Description</div><div>Date</div><div>Type</div><div>Price</div><div>Edge</div><div>Stake / Grade</div><div>Profit</div><div></div></div>' +
      rows;
  }

  /* ---- mount ------------------------------------------------------------- */

  function render() {
    var priced = D.games.map(function (g) { return M.priceGame(g, opts()); });
    var card = M.buildBetCard(D.games, opts());
    var sel = priced[state.selected] || priced[0];

    document.getElementById('app').innerHTML =
      renderTopbar() +
      '<div class="wrap">' +
        renderKpis(card) +
        '<div class="main">' +
          '<div>' + renderEdgeBoard(priced, card) + renderRatings() + '</div>' +
          '<div>' + renderProjector(sel) + renderBetCard(card) + '</div>' +
        '</div>' +
        '<div class="split"><div>' + renderProps() + '</div><div>' + renderTracker() + '</div></div>' +
        '<div class="footer">' +
          '<span>Team marks are generic color-accurate helmets, not school logos. Preseason: no in-season form exists yet for 2026, ' +
          'so every model number here comes from SP+ rating differential plus a fitted home-field constant (2 real inputs, not a full ' +
          'drive-level sim). No Total/Team-total market, weather, travel, pace, returning-production, situational flags, or futures data ' +
          'is fetched by this pipeline \\u2014 those are simply not shown rather than estimated. Sigma is one league-wide value, not per-game. ' +
          'Tracker is local to this browser only \\u2014 no sync, no real money moved. No model reliably beats a well-priced line on every game.</span>' +
        '</div>' +
      '</div>';
  }

  window.__cfbSearch = function (v) { state.search = v; render(); };

  document.addEventListener('click', function (e) {
    var g = e.target.closest('[data-game]');
    if (g) { state.selected = +g.dataset.game; return render(); }
    var m = e.target.closest('[data-market]');
    if (m) { state.market = m.dataset.market; return render(); }
  });

  render();
})();
</script>
"""


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    model_data = build_model_data(data)
    data_script = "<script>\nwindow.MODEL_DATA = " + json.dumps(model_data) + ";\n</script>\n"

    html_out = HEAD_HTML + data_script + MATH_JS + RENDERER_JS + TAIL_HTML

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html_out)

    print(f"Wrote {OUT_PATH} ({model_data['meta']['gamesPriced']} priced of "
          f"{model_data['meta']['totalGames']} total games, {len(model_data['teams'])} "
          f"teams with real ratings, {len(model_data['propCatalog'])} prop markets in catalog)")


if __name__ == "__main__":
    main()
