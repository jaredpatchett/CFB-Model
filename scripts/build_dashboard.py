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
  - Line decomposition: real, 1-3 components depending on the game and which
    of our two model paths priced it (see below) — either SP+ rating
    differential x fitted slope + the fitted home-field/intercept constant
    (preseason), or a single trained-GameMarginModel line (in-season, once
    both teams have real current-season games), plus (only when
    config/injury_overrides.csv has an active entry for either team) a
    manual injury/availability adjustment. The reference's 7-component
    breakdown (EPA, travel, pace, QB posterior, etc.) assumed model
    internals we don't have.
  - Two model paths, auto-switched per game: every game starts on the
    preseason SP+ prior (src/models/fair_odds.py). Once BOTH teams in a
    specific matchup have played enough real games this season (see
    src/features/live_features.py's MIN_GAMES_FOR_TRAINED_MODEL), THAT game
    switches to the trained GameMarginModel (rolling scoring margin/PPG +
    SP+ + home field, actually fit on real historical results) instead —
    automatically, no manual step, and independently per game (an early
    non-conference game between two 3-0 teams can be on the trained model
    while a same-week game involving a team on a bye is still on the
    preseason prior). A real "In-season model" chip shows when a specific
    game has switched over.
  - sigma: our model has ONE league-wide residual std, not a true per-game
    sigma. Every game uses the same value. Disclosed in the footer.
  - Weather/venue/travel: not fetched anywhere in this pipeline. Omitted
    rather than invented.
  - Pace and returning production: NOW fetched (CFBD's /stats/season/advanced
    and /player/returning — see src/features/team_features.py's
    build_pace_returning_features). Pace is plays-per-drive (a real,
    self-contained tempo-ADJACENT proxy — CFBD has no direct seconds-per-play
    field at this granularity, so it's labeled "plays/drive" everywhere, not
    unqualified "pace"). Returning production is CFBD's own percentPPA. Both
    feed the trained GameMarginModel's features (pace_diff/
    returning_production_diff) and show on Power Ratings (hover a team row
    for the real numbers; a "RP xx%" badge shows inline when available).
    Teams CFBD doesn't have coverage for simply show nothing here — no
    fabricated placeholder.
  - Situational flags: mostly not fetched, EXCEPT neutral site (CFBD's
    /games schedule exposes it — zeroes the home-field constant, since that
    constant was fit only from real home games) and manual injury/
    availability overrides (see src/data/injury_overrides.py — no free CFB
    injury API exists, so this is a hand-maintained, version-controlled CSV
    a person edits, not scraped data). Both show a real chip in the Matchup
    Projector when active; neither is silently baked in with no trace.
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
BACKTEST_PATH = "docs/data/backtest_results.json"
CLV_PATH = "docs/data/clv_results.json"
OUT_PATH = "docs/dashboard.html"

MIN_EDGE_POINTS = 1.9  # unified board/card threshold; ~= our 5pp moneyline edge threshold
                        # via the /2.6 rescale in ModelMath.priceGame (5 / 2.6 = 1.92)
                        # COUPLING: src/analysis/clv.py's SPREAD_EDGE_THRESHOLD_POINTS must match
                        # this number, or the CLV tracker will snapshot a different set of games
                        # than what actually shows as a flagged spread play here -- see that
                        # module's comment on its own constant.


def fmt_kickoff(iso):
    if not iso:
        return "TBD"
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return dt.strftime("%a %-m/%-d, %-I:%M%p UTC")
    except Exception:
        return "TBD"


def build_model_data(data: dict, backtest: dict = None, clv: dict = None) -> dict:
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
            "pace": t.get("pace"),
            "returningProduction": t.get("returning_production"),
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

        is_neutral = bool(g.get("neutral_site"))
        is_trained = g.get("model_source") == "trained_model"

        home_inj_pts = g.get("injury_adjustment_home") or 0.0
        away_inj_pts = g.get("injury_adjustment_away") or 0.0
        home_inj_notes = g.get("injury_notes_home") or []
        away_inj_notes = g.get("injury_notes_away") or []
        has_injury_override = bool(home_inj_pts or away_inj_pts)

        comp_rating = round(-(slope * sp_diff), 2) if (slope is not None and sp_diff is not None) else 0.0
        comp_hfa = 0.0 if is_neutral else (round(-intercept, 2) if intercept is not None else 0.0)
        # Trained-model games: back out the model's OWN base margin (before
        # the injury adjustment, which is applied on top either way) to show
        # as a single decomposition line -- the trained GBM has no simple
        # additive SP+/home-field breakdown the way the preseason prior does,
        # so showing comp_rating/comp_hfa here would misrepresent what
        # actually produced this number.
        base_margin_trained = (model_margin - home_inj_pts + away_inj_pts) if (is_trained and model_margin is not None) else None
        comp_trained = round(-base_margin_trained, 2) if base_margin_trained is not None else 0.0

        book = (g.get("book_used") or "market").replace("_", " ").title()

        note_parts = []
        preseason_cmp = g.get("preseason_comparison_margin")
        if is_trained:
            note_parts.append("trained in-season GameMarginModel (rolling scoring margin/PPG, "
                               "SP+ diff, home field) -- both teams have enough real games played "
                               "this season; preseason SP+ prior no longer used for this matchup")
            if preseason_cmp is not None:
                # Side-by-side comparison, added 9/19/2026 for this weekend's
                # first live look at the trained model taking over mid-season
                # -- lets the user actually see what the preseason-only prior
                # would have said instead of a blind cutover with nothing to
                # check it against.
                note_parts.append(f"for comparison, the preseason-only SP+ prior alone would have "
                                   f"said {preseason_cmp:+.1f} pts (spread {-preseason_cmp:+.1f})")
        elif slope is not None and sp_diff is not None:
            note_parts.append(f"SP+ diff {sp_diff:+.1f} pts x fitted slope {slope:.2f}")
        if not is_trained:
            if is_neutral:
                note_parts.append("neutral site: home-field constant not applied")
            elif intercept is not None:
                note_parts.append(f"home-field constant {intercept:+.1f} pts (fit from {n_games_hist} 2021-2025 games)")
        if has_injury_override:
            inj_bits = []
            if home_inj_pts:
                inj_bits.append(f"{esc_plain(g['home_team'])} {home_inj_pts:+.1f} pts ({'; '.join(home_inj_notes) or 'manual override'})")
            if away_inj_pts:
                inj_bits.append(f"{esc_plain(g['away_team'])} {away_inj_pts:+.1f} pts ({'; '.join(away_inj_notes) or 'manual override'})")
            note_parts.append("manual injury/availability override: " + "; ".join(inj_bits))
        if edge_pp_home is not None:
            note_parts.append(f"moneyline edge vs. book: {edge_pp_home:+.1f}pp on {esc_plain(g['home_team'])}")
        note = "Model: " + "; ".join(note_parts) + "." if note_parts else "Insufficient history to explain this line."

        if is_trained:
            components = [
                {"label": "Trained GameMarginModel (in-season form + SP+ + home field)", "points": comp_trained},
            ]
        else:
            components = [
                {"label": "SP+ rating differential x fitted slope", "points": comp_rating},
                {
                    "label": "Home-field constant (neutral site — not applied)" if is_neutral
                    else "Home-field constant (fit, 2021-2025)",
                    "points": comp_hfa,
                },
            ]
        if has_injury_override:
            # Sign convention: comp_rating/comp_hfa above are in SPREAD terms
            # (negative = home favored), but home_inj_pts/away_inj_pts are in
            # MARGIN terms (positive = that team's own margin goes up). A
            # margin hit to the home team (negative home_inj_pts) should push
            # the SPREAD toward the away side (positive) -- i.e. this needs
            # the opposite sign of (home_inj_pts - away_inj_pts), not the same
            # sign, or the decomposition bar points the wrong direction and
            # the components stop summing to modelSpread.
            components.append({
                "label": "Manual injury/availability override",
                "points": round(away_inj_pts - home_inj_pts, 2),
            })

        flags = []
        if is_trained:
            flags.append({"text": "In-season model", "level": 2})
        if is_neutral:
            flags.append({"text": "Neutral site", "level": 0})
        if home_inj_pts:
            flags.append({
                "text": f"{esc_plain(g['home_team'])} {home_inj_pts:+.1f} pts: {'; '.join(home_inj_notes) or 'manual override'}",
                "level": 1 if home_inj_pts < 0 else 2,
            })
        if away_inj_pts:
            flags.append({
                "text": f"{esc_plain(g['away_team'])} {away_inj_pts:+.1f} pts: {'; '.join(away_inj_notes) or 'manual override'}",
                "level": 1 if away_inj_pts < 0 else 2,
            })

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
            "sigma": g.get("residual_std") or residual_std,
            "components": components,
            "flags": flags,
            "note": note,
            "evHomePct": g.get("ev_home_pct"),
            "evAwayPct": g.get("ev_away_pct"),
        })

    props_catalog = data.get("prop_market_catalog", [])
    props_live = data.get("props", [])
    fantasy_out = data.get("fantasy", [])

    # backtest: real ATS win rate / margin MAE / moneyline accuracy against
    # historical CFBD closing lines (see run_backtest.py + src/backtest/
    # backtester.py). Read from a SEPARATE file (docs/data/backtest_results.json)
    # rather than latest.json, since it's produced by a different script on a
    # different cadence (only changes when the model itself is retrained,
    # not every dashboard refresh) -- None here just means that file hasn't
    # been generated yet (e.g. before scripts/run_backtest.py has ever run),
    # and the dashboard panel is simply omitted, not faked.
    backtest_out = None
    if backtest:
        sp = backtest.get("spread") or {}
        ml = backtest.get("moneyline") or {}
        backtest_out = {
            "generatedAt": _fmt_generated_at(backtest.get("generated_at")),
            "seasonsCovered": backtest.get("seasons_covered") or [],
            "spread": {
                "nGames": sp.get("n_games"),
                "atsWinRate": sp.get("ats_win_rate"),
                "marginMae": sp.get("margin_mae"),
                "breakevenAtsRate": sp.get("breakeven_ats_rate"),
                "beatMarket": sp.get("beat_market"),
            },
            "moneyline": {
                "nGames": ml.get("n_games"),
                "accuracy": ml.get("accuracy"),
                "logLoss": ml.get("log_loss"),
            },
        }

    # CLV (closing-line value) track record for the model's own flagged spread
    # picks -- see src/analysis/clv.py and scripts/compute_clv.py. Read from a
    # separate file for the same reason as backtest_results.json above: a
    # different cadence than latest.json, and None here just means no game
    # has been graded yet (normal before the season's first kickoff), not a
    # broken pipeline.
    clv_out = None
    if clv and clv.get("n_games"):
        clv_out = {
            "generatedAt": _fmt_generated_at(clv.get("generated_at")),
            "nGames": clv.get("n_games"),
            "avgClvPoints": clv.get("avg_clv_points"),
            "medianClvPoints": clv.get("median_clv_points"),
            "pctPositiveClv": clv.get("pct_positive_clv"),
        }

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
        "clv": clv_out,
        "propCatalog": props_catalog,
        "propsLive": props_live,
        "fantasy": fantasy_out,
        "backtest": backtest_out,
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

.page-nav { border-bottom: 1px solid var(--rule); background: var(--panel); }
.page-nav-inner { max-width: var(--max-width); margin: 0 auto; padding: 10px var(--gutter); display: flex; gap: 6px; flex-wrap: wrap; }
.tab--page { padding: 7px 18px; font-size: 12.5px; letter-spacing: 0.08em; }
.tab--page.is-active { background: var(--blue); color: #fff; }

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
.fantasy-grid { grid-template-columns: 26px 1fr 120px 90px; }
.card-row  { grid-template-columns: 1fr 58px 68px 46px 58px; }

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
.edge-play-pill { display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 4px; font-family: var(--font-display); font-weight: 700; font-size: 11px; letter-spacing: 0.02em; }
.edge-fade-pill { display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 4px; font-family: var(--font-display); font-weight: 700; font-size: 11px; letter-spacing: 0.02em; background: rgba(255,255,255,0.05); border: 1px solid var(--muted-3); color: var(--muted-3); }

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
.card-conf { font-size: 17px; font-weight: 900; text-align: right; color: var(--green); font-family: var(--font-led); }
.card-total { display: flex; justify-content: space-between; padding: 13px 22px; font-size: 11px; background: var(--panel-deep); }
.card-total-label { color: var(--muted-2); font-family: var(--font-display); font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; font-size: 11px; }

.pcard-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 8px; }
.pcard { background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 12px 14px; }
.pcard-head { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; margin-bottom: 8px; }
.pcard-note { font-size: 10.5px; color: var(--muted-3); margin: 0; line-height: 1.5; }
.pcard-line-row { display: flex; justify-content: space-between; align-items: center; gap: 5px; font-size: 11px; padding: 5px 0; border-top: 1px solid var(--rule-faint); }

.prop-best-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 8px; margin-bottom: 22px; }
.prop-best-card { background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; }
.prop-best-top { height: 3px; background: var(--rule); }
.prop-best-top.is-over { background: var(--green); }
.prop-best-top.is-under { background: var(--red); }
.prop-best-body { padding: 12px 14px; }
.prop-best-name { font-weight: 700; font-size: 12.5px; margin-bottom: 2px; }
.prop-best-meta { font-size: 10px; color: var(--muted-2); margin-bottom: 10px; letter-spacing: 0.02em; }
.prop-best-edge { font-family: var(--font-led); font-weight: 900; font-size: 22px; line-height: 1; }
.prop-best-edge.is-pos { color: var(--green); }
.prop-best-edge.is-neg { color: var(--red); }
.prop-best-sub { font-size: 9.5px; color: var(--muted-3); margin-top: 5px; }
.prop-best-price { font-size: 10px; color: var(--muted-2); margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--rule-faint); }
.prop-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 16px; }
.tab--prop { padding: 6px 13px; font-size: 11px; letter-spacing: 0.03em; display: inline-flex; align-items: center; gap: 6px; }
.tab--prop.is-active { background: var(--blue); color: #fff; }
.prop-tab-count { font-size: 9px; opacity: 0.8; background: rgba(255,255,255,0.14); border-radius: 8px; padding: 1px 5px; transform: skewX(11deg); display: inline-block; }
.pcard-line-row.is-scored { border-top: 1px solid var(--rule); }
.pchip.is-official { background: rgba(23,194,107,0.16); color: var(--green); }
.pchip.is-lean { background: rgba(224,180,74,0.16); color: var(--amber); }
.prop-lean-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 8px; margin-bottom: 22px; }
.prop-lean-card { background: var(--panel); border: 1px solid var(--rule-faint); border-radius: 4px; padding: 11px 13px; opacity: 0.88; }
.trk-auto-tag { font-family: var(--font-display); font-weight: 800; font-size: 8.5px; letter-spacing: 0.06em; padding: 1px 5px; border-radius: 3px; background: rgba(46,123,255,0.16); color: var(--blue); vertical-align: middle; margin-left: 6px; }
.pchip { font-family: var(--font-display); font-weight: 800; font-size: 9.5px; letter-spacing: 0.06em; padding: 2px 7px; border-radius: 3px; }
.pchip.is-live { background: rgba(23,194,107,0.16); color: var(--green); }
.pchip.is-pending { background: var(--chip); color: var(--muted-4); }

.flag-chip { display: inline-block; font-family: var(--font-display); font-weight: 800; font-size: 9px; letter-spacing: 0.07em; padding: 2px 7px; border-radius: 3px; background: rgba(230,168,45,0.16); color: var(--gold, #E6A82D); margin-left: 6px; vertical-align: middle; }
.flag-chip.is-warn { background: rgba(255,82,82,0.16); color: var(--red); }
.flag-chip.is-good { background: rgba(23,194,107,0.16); color: var(--green); }

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
.trk-summary-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--rule); margin-bottom: 1px; }
.unit-size-input { width: 108px; background: var(--chip); border: 1px solid var(--rule); color: var(--text); border-radius: 3px; padding: 6px 9px; font-family: var(--font-data); font-size: 11px; }

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

  function tierFor(winProb) {
    // Flat 1u across every qualifying play, on purpose -- confirmed with
    // the user 9/19/2026. Win probability (coverProb/sideProb) is real and
    // now shown directly (Edge Board's Win% column, Bet Card's new
    // confidence column), but it comes from ONE FIXED league-wide sigma
    // (see game.sigma / homeMargin above), not a per-game or validated-
    // per-bucket calibration -- so while it's directionally right, it
    // hasn't earned the right to size real money yet. Revisit tiered
    // sizing (58%/65% bands, or whatever bands the data supports) once
    // there's a real in-season sample to check win-prob calibration
    // against actual results, the same way the 20+pt spread call got made
    // off real tracked data rather than a theory.
    return winProb == null ? '\\u2014' : '1u';
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
      // NOTE on home vs. picked-side numbers: marketLabel/modelLabel/edge/
      // coverProb below are all deliberately kept in HOME-TEAM terms (same
      // convention Spread uses for its own marketLabel/modelLabel/edge) --
      // this is what the Edge Board grid's Market/Model/Edge/Win% columns
      // are built from, and changing that would flip its is-pos/is-neg
      // color convention and Win% meaning out from under it. Fine for a
      // grid that's always read in "home team's number" terms.
      //
      // sideMoneyline/sideProb ADDITIONALLY compute the same two things in
      // PICKED-SIDE terms (i.e. actually correct for whichever team
      // playLabel names) -- moneyline's home and away prices are two
      // genuinely independent numbers (unlike a spread, where home/away are
      // just a sign flip of the same number), so anything that names a
      // specific team -- like the Bet Card -- MUST use these, not the
      // home-only fields above, or it silently shows the wrong team's price
      // and an EV computed from the wrong team's probability entirely. This
      // was a real bug: the Bet Card was showing e.g. 'TOL ML' next to
      // Michigan State's own -410 price and using Michigan State's win
      // probability to compute Toledo's supposed EV, producing a
      // meaningless number that also happened to look enormous whenever
      // the recommended side was a big underdog (large payout multiplier
      // amplifying an already-wrong probability into a triple-digit "EV%").
      // See renderBetCard, which is the only consumer of these two fields.
      var p    = 1 - normalCdf(0, hm, sd);
      var fair = removeVig(game.marketMoneyline, game.awayMoneyline);
      var vig  = fair[0];
      marketLabel = (game.marketMoneyline > 0 ? '+' : '') + game.marketMoneyline;
      modelLabel  = (fairAmerican(p) > 0 ? '+' : '') + fairAmerican(p);
      edge        = (p - vig) * 100;
      coverProb   = p;
      side        = p > vig ? game.home : game.away;
      playLabel   = abbrOf(side) + ' ML';
      var isHomeSide  = side === game.home;
      var sideMoneyline = isHomeSide ? game.marketMoneyline : game.awayMoneyline;
      var sideProb      = isHomeSide ? p : (1 - p);
      // Real dollar EV on the picked side, at the price actually being
      // offered for that side (vig and all) -- NOT the same thing as
      // `edge` above, which only measures the model's probability against
      // the market's de-vigged fair line. A side can clear that
      // probability-edge bar and still be a losing bet once you account
      // for the vig actually baked into ITS specific price (vig isn't
      // always split evenly across both sides). Real EV is the honest bar
      // for "is this actually worth betting" -- see priceInRange/isFade
      // below for how it's used.
      var sideEV = expectedValue(sideProb, sideMoneyline);
    } else {
      marketLabel = signed(game.marketSpread);
      modelLabel  = signed(game.modelSpread);
      edge        = game.marketSpread - game.modelSpread;
      coverProb   = edge > 0 ? 1 - normalCdf(-game.marketSpread, hm, sd)
                             :     normalCdf(-game.marketSpread, hm, sd);
      side        = edge > 0 ? game.home : game.away;
      playLabel   = abbrOf(side) + ' ' + signed(edge > 0 ? game.marketSpread : -game.marketSpread);
    }

    var edgeForTier = market === 'Moneyline' ? edge / 2.6 : edge;

    // Sanity bound, Moneyline only: with ONE fixed league-wide sigma (see
    // game.sigma / the residual_std footer note), the model structurally
    // cannot express a win probability much above the low-to-mid 90s no
    // matter how lopsided a matchup actually is -- getting normalCdf(...)
    // near 95%+ needs a 30+ point predicted margin, which basically never
    // happens even in real blowouts. So against any market price that
    // implies a near-lock (a big favorite well past -450, or the
    // complementary big dog past +450), the model will ALWAYS look like it
    // disagrees with the market and flag the other side as a play -- not
    // because it found real value, but because it's mechanically unable to
    // match that level of market confidence. That's a structural blind
    // spot, not a discovered edge, so these are excluded from qualifying
    // as a play at all here (not just flagged) -- confirmed with the user
    // after real Week 1 examples (e.g. ECU +3000 vs Alabama) made this
    // visible on the Edge Board's play labels.
    var MONEYLINE_PRICE_QUALIFY_MAX = 450;
    var priceInRange  = market !== 'Moneyline' || Math.abs(sideMoneyline) <= MONEYLINE_PRICE_QUALIFY_MAX;
    var edgeClears    = Math.abs(edgeForTier) >= minEdge;
    // Underdog moneyline picks additionally require the TRAINED in-season
    // model, not just the preseason-only prior, to count as real signal.
    // The preseason model uses one fixed sigma and structurally can't
    // express strong confidence in a good favorite (see
    // MONEYLINE_PRICE_QUALIFY_MAX's note above for the extreme version of
    // this), so a preseason-only 'the market's favorite isn't as good as
    // it thinks' read is much more often the model's own resolution limit
    // than a real mispriced dog -- even at moderate, plausible-looking
    // prices where the price cap above doesn't trigger. Confirmed with
    // the user from real Week 1 examples where the preseason model's fair
    // line for the FAVORITE was well off market (Duke -142 fair vs -340
    // market; Auburn -184 fair vs -298 market) purely because neither
    // team had real season data yet, not because the favorite was
    // actually overpriced. Only applies to the underdog side -- the
    // model favoring the FAVORITE even more than market isn't subject to
    // the same bias (the fixed sigma makes the model UNDER-confident on
    // strong teams if anything, so that read fights its own conservatism
    // rather than being produced by it).
    var isUnderdog = market === 'Moneyline' && sideMoneyline > 0;
    var isTrainedGame = (game.flags || []).some(function (f) { return f.text === 'In-season model'; });
    var untrustedUnderdog = isUnderdog && !isTrainedGame;
    var isPositiveEV  = market !== 'Moneyline' || (sideEV > 0 && !untrustedUnderdog);
    // Fade spreads of 20+ points outright -- confirmed 9/17-9/19/2026 across
    // two independent looks at the data (week 1-2 spread record improved
    // 16-12/54% -> 15-6/71% once these were cut; the tracker's own recovered
    // seed log, reconciled 9/19, showed 7-6/54% official vs 15-6/71% raw once
    // the 20pt line was actually applied). The user decided these don't just
    // get excluded from the record after the fact (see autoUnofficial below,
    // which still exists for anything hand-tracked) -- the model shouldn't
    // recommend them as a PLAY at all going forward. Uses the market spread's
    // own size, not just the picked side, since the pattern showed up as a
    // property of the linear SP+ slope fit diverging at large differentials
    // in general, not specifically an underdog-side bias.
    var bigSpread = market === 'Spread' && Math.abs(game.marketSpread) >= 20;
    var qualifies = edgeClears && priceInRange && isPositiveEV && !bigSpread;
    // A moneyline pick that clears the probability-edge bar and passes the
    // price sanity check, but comes out negative on real dollar EV (or is
    // an underdog read the preseason-only model hasn't earned trust for
    // yet, per untrustedUnderdog above), isn't a play -- it just means the
    // model disagrees with the market on that side without that
    // disagreement being worth betting. Label it a FADE instead of
    // silently dropping it, so it's visibly "not a play" rather than
    // looking like the model has no opinion at all. Confirmed with the
    // user: negative-EV (or untrusted-preseason-underdog) sides should
    // never be flagged as plays, and a negative read on one side never
    // automatically makes the other side a play either (see
    // MONEYLINE_PRICE_QUALIFY_MAX above for that).
    var isFade = market === 'Moneyline' && edgeClears && priceInRange && !isPositiveEV;

    // sideEdge/sideEdgeLabel: `edge` above is signed in HOME-team terms
    // (same convention as marketLabel/modelLabel) -- but `side` is always
    // chosen as whichever team edge's sign favors, so the recommended
    // side's OWN edge is always just Math.abs(edge), regardless of market.
    // Without this, the Edge Board's Edge column read backwards any time
    // the play was on the away side (e.g. showing '-20.4' right next to a
    // 'PLAY: ECU +28.5' pill) -- caught by the user looking at real rows.
    var sideEdge = Math.abs(edge);
    var sideEdgeLabel = market === 'Moneyline' ? signed(sideEdge) + '%' : signed(sideEdge);

    return {
      game: game, market: market, sd: sd,
      marketLabel: marketLabel, modelLabel: modelLabel,
      edge: edge, edgeForTier: edgeForTier,
      edgeLabel: market === 'Moneyline' ? signed(edge) + '%' : signed(edge),
      sideEdge: sideEdge, sideEdgeLabel: sideEdgeLabel,
      coverProb: coverProb, playLabel: playLabel, side: side,
      sideMoneyline: market === 'Moneyline' ? sideMoneyline : null,
      sideProb: market === 'Moneyline' ? sideProb : coverProb,
      sideEV: market === 'Moneyline' ? sideEV : null,
      qualifies: qualifies,
      isFade: isFade,
      bigSpread: bigSpread,
      tier: qualifies ? tierFor(market === 'Moneyline' ? sideProb : coverProb) : '\\u2014'
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
      // Sort by win probability (sideProb), not raw edge points -- fixed
      // 9/19/2026. This was the last place still ranking by |edgeForTier|
      // after the 20+pt spread fade already established that raw point-
      // edge scales with spread SIZE, not confidence (a 19pt edge on a
      // blowout-line game isn't more trustworthy than a 3pt edge on a
      // pick'em -- it's just a bigger disagreement in points). Win
      // probability is the actual confidence number -- it's what tierFor
      // and both the Edge Board's/Bet Card's Win% columns already use --
      // so the curated top-10 should surface by that too, not silently
      // rank by the same flawed metric the fade logic was built to fight.
      .sort(function (a, b) { return b.sideProb - a.sideProb; })
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
# real flag chips in the projector header for the situational data we
# actually have — neutral site (from CFBD's schedule) and manual injury/
# availability overrides (from config/injury_overrides.csv) — no other
# situational flags/futures (not fetched by this pipeline), real props
# "coming soon" catalog cards instead of fabricated projections (now with a
# real trained-model prediction/edge line under any LIVE prop the pipeline
# could actually match a player + opponent-defense number for — see
# src/features/live_player_features.py; most/all rows show no overlay
# before the season has real in-season player data, by design, not a bug),
# and a real localStorage Tracker in place of the fabricated CLV/bankroll
# history chart.
# =============================================================================

RENDERER_JS = """<script>
(function () {
  'use strict';

  var D = window.MODEL_DATA;
  var M = window.ModelMath;
  var MARKETS = ['Spread', 'Moneyline'];

  var state = { selected: 0, market: 'Moneyline', search: '', page: 'edge', propMarket: 'ALL', unitSize: 50, trackerModelTab: 'all' };

  var byName = {};
  D.teams.forEach(function (t) { byName[t.name] = t; });
  function team(name) { return byName[name] || { name: name, abbr: name, primary: '#8A94A3', secondary: '#E7EDF5' }; }
  function abbrOf(name) { return team(name).abbr; }
  // Player-prop rows carry a 'team' field now (see live_player_features.py's
  // score_prop) so the dashboard/tracker can show which team a player is on
  // without the user having to look it up by hand. Falls back to just the
  // name for anything unscored (score_prop never ran, so there's no team to
  // attach -- e.g. a posted line with no model read yet), same "don't
  // fabricate what we don't know" policy as everywhere else here.
  function playerLabel(r) { return (r.team ? abbrOf(r.team) + ' ' : '') + r.player_name; }

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

  var PAGES = [
    { id: 'edge', label: 'Edge Board' },
    { id: 'ratings', label: 'Power Ratings' },
    { id: 'props', label: 'Player Props' },
    { id: 'fantasy', label: 'Fantasy' },
    { id: 'tracker', label: 'My Tracker' },
    { id: 'performance', label: 'Model Performance' },
  ];

  function renderTopbar() {
    // Search only makes sense (and only has a visible box) on the Edge
    // Board tab -- see renderPageNav for the tab switcher itself. Brand
    // and the data-freshness indicator stay visible everywhere since
    // they're not tab-specific.
    var searchHtml = state.page === 'edge'
      ? '<input id="search" type="text" placeholder="search a team..." autocomplete="off" value="' + esc(state.search) + '" oninput="window.__cfbSearch(this.value)"><div class="divider-v"></div>'
      : '';
    return '' +
      '<div class="topbar"><div class="topbar-inner">' +
        '<div class="brand">' +
          '<div class="brand-block"><span>' + esc(D.meta.modelName) + '</span></div>' +
          '<div class="brand-outline"><span>' + esc(D.meta.sport) + '</span></div>' +
          '<div class="brand-sub">' + esc(D.meta.subtitle) + '<br>' + esc(D.meta.version) + '</div>' +
        '</div>' +
        '<div class="topbar-right">' +
          searchHtml +
          '<div class="sync"><span class="sync-dot"></span><span>' + esc(D.meta.dataAsOf) + '</span></div>' +
        '</div>' +
      '</div></div>';
  }

  function renderPageNav() {
    return '<div class="page-nav"><div class="page-nav-inner">' +
      PAGES.map(function (p) {
        return '<button class="tab tab--page' + (p.id === state.page ? ' is-active' : '') + '" data-page="' + esc(p.id) + '"><span>' + esc(p.label) + '</span></button>';
      }).join('') +
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
    // Same fix as renderBetCard: g.marketMoneyline is always the HOME
    // team's price, but p.playLabel names whichever team (home or away)
    // the model actually recommends -- using g.marketMoneyline here logs
    // the wrong team's price into the user's own Tracker for any away-side
    // pick (e.g. a tracked "TOL ML" bet would silently log Michigan
    // State's -410 instead of Toledo's real +320). p.sideMoneyline is the
    // side-correct price. edge is reported as a plain positive magnitude
    // for the same reason: p.edge is signed in HOME-team terms (negative
    // whenever the away side is picked), which would show a tracked pick
    // as having "negative edge" even though it was tracked specifically
    // because the recommended side clears the edge threshold.
    // Auto-flag 20+ point underdog spreads as unofficial -- the linear
    // SP+ slope fit appears to systematically favor big dogs it shouldn't
    // (confirmed 9/17/2026: cutting these took the wk1-2 spread record from
    // 16-12/54% to 15-6/71%). Parsed straight from the pick's own line
    // (e.g. "LT +35.5") rather than a separate field, so it can't drift out
    // of sync with what's actually being tracked. User can always override
    // via the row's unofficial toggle.
    var lineMatch = /([+-]?\\d+(?:\\.\\d+)?)\\s*$/.exec(p.playLabel);
    var lineSize = lineMatch ? Math.abs(parseFloat(lineMatch[1])) : 0;
    var autoUnofficial = p.market === 'Spread' && lineSize >= 20;
    // Which model actually priced this game, captured at track time so it
    // travels with the pick permanently -- a game can switch from
    // preseason_prior to trained_model on a LATER run (more games played),
    // but the tracked play should keep recording whichever model's number
    // it was actually taken at, not whatever the game shows today. Same
    // "In-season model" flag the Edge Board/Bet Card already use elsewhere
    // (see isTrainedGame in priceGame/renderBetCard) -- single source of
    // truth, not a second parallel check that could drift out of sync.
    var modelSource = (g.flags || []).some(function (f) { return f.text === 'In-season model'; })
      ? 'trained_model' : 'preseason_prior';

    return esc(JSON.stringify({
      description: p.playLabel + ' \\u2014 ' + abbrOf(g.away) + ' at ' + abbrOf(g.home),
      date: g.kickoff, type: p.market,
      price: p.market === 'Moneyline' ? p.sideMoneyline : -110,
      edge: Math.round(Math.abs(p.edge) * 10) / 10,
      unofficial: autoUnofficial,
      modelSource: modelSource
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
      // cls now reflects "is this row actually good/bad," not raw
      // home-team sign: is-off below the edge threshold or price-cap-
      // excluded, is-neg for a confirmed-negative-EV FADE, is-pos for an
      // actual qualifying play. (See sideEdge/sideEdgeLabel in priceGame
      // for why the Edge column text itself also switched off raw p.edge.)
      var cls = Math.abs(p.edgeForTier) < D.meta.minEdge ? 'is-off'
        : p.isFade ? 'is-neg'
        : p.qualifies ? 'is-pos'
        : 'is-off';
      var qualifies = p.qualifies;
      var playPillLabel = qualifies
        ? (p.market === 'Moneyline'
            ? p.playLabel + ' ' + (p.sideMoneyline > 0 ? '+' : '') + p.sideMoneyline
            : p.playLabel)
        : null;
      var playPillColor = qualifies ? M.displayColor(team(p.side).primary) : null;
      // FADE: the model disagreed with the market on this side but it
      // isn't actually positive EV at the price offered -- shown so it's
      // visibly "not a play," not silently dropped. Never implies the
      // OTHER side is a play (see isFade in priceGame()).
      var fadePillLabel = p.isFade
        ? p.playLabel + ' ' + (p.sideMoneyline > 0 ? '+' : '') + p.sideMoneyline
        : null;
      // Same idea as the Moneyline FADE pill above, for the 20+pt spread
      // exclusion: the model still has an opinion here (often a big one --
      // that's the whole problem), but it's not a play, so say so instead
      // of the row just looking like nothing happened.
      var bigSpreadPillLabel = p.bigSpread ? p.playLabel : null;
      // Which model priced this game -- shown on EVERY row (not just
      // PLAY/FADE ones), since it's a property of the game itself, not the
      // pick. Same flag the tracker badge/priceGame's untrustedUnderdog
      // check already use elsewhere, so this can never drift out of sync
      // with what actually generated the line. Added 9/19/2026 -- the user
      // wanted this visible on the Edge Board directly, not just after
      // tracking a play.
      var isTrainedGame = (g.flags || []).some(function (f) { return f.text === 'In-season model'; });
      var modelSrcLabel = isTrainedGame ? 'TRAINED' : 'UNTRAINED';
      var modelSrcColor = isTrainedGame ? '#2ecc71' : '#8A94A3';
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
            '<div class="meta"><span>' + esc(g.kickoff) + '</span><span>' + esc(g.book) + '</span>' +
              '<span style="font-family:var(--font-display);font-weight:700;letter-spacing:.05em;color:' + modelSrcColor + '">' + modelSrcLabel + '</span></div>' +
            (qualifies ? '<div class="edge-play-pill" style="background:' + esc(playPillColor) + '26;border:1px solid ' + esc(playPillColor) + ';color:' + esc(playPillColor) + '">PLAY: ' + esc(playPillLabel) + '</div>' : '') +
            (p.isFade ? '<div class="edge-fade-pill">FADE: ' + esc(fadePillLabel) + '</div>' : '') +
            (p.bigSpread ? '<div class="edge-fade-pill">FADE: 20+PT SPREAD</div>' : '') +
            '</div>' +
            '<div class="num cell-market">' + esc(p.marketLabel) + '</div>' +
            '<div class="num cell-model">' + esc(p.modelLabel) + '</div>' +
            '<div class="num cell-edge ' + cls + '">' + esc(p.sideEdgeLabel) + '</div>' +
            '<div class="num cell-prob">' + (p.sideProb * 100).toFixed(1) + '%</div>' +
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

    var visible = D.teams; // no search box on this tab (search lives on Edge Board only)

    var rows = visible.slice().sort(function (a, b) { return b.net - a.net; }).map(function (t) {
      var c = M.displayColor(t.primary);
      var hasSplit = t.offSp != null && t.defSp != null;
      var offW = hasSplit ? Math.min(50, Math.abs(t.offSp) / D.meta.offScale * 50).toFixed(1) : 0;
      var defW = hasSplit ? Math.min(50, Math.abs(t.defSp) / D.meta.defScale * 50).toFixed(1) : 0;
      var i = D.teams.indexOf(t);
      // pace/returningProduction are real CFBD data (see export_dashboard_data.py)
      // but only shown when present -- no fabricated placeholder for teams
      // CFBD didn't have coverage for. Kept as a tooltip + compact inline
      // badge rather than new grid columns, to avoid reworking rate-grid's
      // fixed column widths for two optional stats.
      var tipBits = [];
      if (t.pace != null) tipBits.push('Pace: ' + t.pace.toFixed(1) + ' plays/drive');
      if (t.returningProduction != null) tipBits.push('Returning production: ' + (t.returningProduction * 100).toFixed(0) + '%');
      var tip = tipBits.join(' \\u00b7 ');
      var rpBadge = t.returningProduction != null
        ? '<span style="color:var(--muted-3);font-size:9px;letter-spacing:.04em;margin-left:6px" title="' + esc(tip) + '">RP ' + (t.returningProduction * 100).toFixed(0) + '%</span>'
        : '';
      return '' +
        '<div class="row"><div class="row-body rate-grid" style="padding:8px 14px;font-size:12px">' +
          '<div style="color:var(--muted-4)">' + (i + 1) + '</div>' +
          '<div class="matchup" style="overflow:hidden;white-space:nowrap" title="' + esc(tip) + '">' +
            '<span class="team-chip" style="background:' + esc(c) + '"></span>' +
            '<span class="team-abbr" style="font-size:15px">' + esc(t.abbr) + '</span>' +
            '<span style="color:var(--muted-4);font-size:9.5px;letter-spacing:.08em">' + esc(t.conf) + '</span>' +
            rpBadge +
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

    return head + (rows || '<div class="empty-state">No teams to show.</div>');
  }

  var FANTASY_STAT_LABELS = {
    pass_yds: 'pass yds', pass_tds: 'pass TD', pass_int: 'INT',
    rush_yds: 'rush yds', rush_tds: 'rush TD',
    rec_yds: 'rec yds', rec_tds: 'rec TD', receptions: 'rec',
  };

  function renderFantasy() {
    var rows = D.fantasy || [];
    if (!rows.length) {
      return '<div class="section-head mt-lg"><div class="section-title"><div class="section-flag"></div>' +
        '<h2>Fantasy Projections (PPR)</h2></div></div>' +
        '<p class="pcard-note">No players clear the real-in-season-games threshold yet \u2014 ' +
        'this fills in automatically as the season\u2019s games get played (same trained per-stat ' +
        'models used for Player Props, just summed into PPR points instead of compared to a posted line).</p>';
    }
    var head = '' +
      '<div class="section-head mt-lg"><div class="section-title"><div class="section-flag"></div>' +
      '<h2>Fantasy Projections (PPR)</h2></div></div>' +
      '<div class="thead fantasy-grid"><div>#</div><div>Player</div><div>Next Opponent</div><div class="num">Proj. Pts</div></div>';

    var visible = rows; // no search box on this tab (search lives on Edge Board only)

    var body = visible.map(function (r, i) {
      var t = team(r.team);
      var c = M.displayColor(t.primary);
      var breakdown = Object.keys(r.stat_breakdown || {})
        .filter(function (k) { return r.stat_breakdown[k]; })
        .map(function (k) { return r.stat_breakdown[k] + ' ' + (FANTASY_STAT_LABELS[k] || k); })
        .join(', ');
      // A 1-game "rolling average" is just that player's real Week 1 stat
      // line -- still real data, but a single fluke huge/tiny game hasn't
      // been smoothed by anything yet. Flag it rather than show a 1-game
      // outlier with the same visual confidence as a multi-game average.
      var lowSample = r.games_played_prior < 2;
      var sampleBadge = lowSample
        ? '<span style="color:var(--amber);font-size:9px;letter-spacing:.04em;margin-left:6px" title="Projection based on a single game so far this season -- treat as noisier than a multi-game average">1 GM</span>'
        : '';
      return '' +
        '<div class="row"><div class="row-body fantasy-grid" style="padding:8px 14px;font-size:12px">' +
          '<div style="color:var(--muted-4)">' + (i + 1) + '</div>' +
          '<div class="matchup" style="overflow:hidden;white-space:nowrap" title="' + esc(breakdown) + '">' +
            '<span class="team-chip" style="background:' + esc(c) + '"></span>' +
            '<span style="font-weight:700">' + esc(r.player_name) + '</span>' +
            '<span style="color:var(--muted-4);font-size:9.5px;letter-spacing:.08em">' + esc(r.position) + '</span>' +
            sampleBadge +
          '</div>' +
          '<div style="overflow:hidden;white-space:nowrap;color:var(--muted-3);font-size:11px">at ' + esc(abbrOf(r.opponent)) + '</div>' +
          '<div class="num" style="font-weight:700">' + r.projected_points.toFixed(1) + '</div>' +
        '</div></div>';
    }).join('');

    return head + (body || '<div class="empty-state">No players to show.</div>') +
      '<div class="table-foot"><span>PPR scoring (1 pt/reception, 1 pt/10 rush or rec yards, 1 pt/25 pass yards, ' +
      '6 pt rush/rec TD, 4 pt pass TD, -2 INT), projected from each player\u2019s own trained stat model against ' +
      'their next scheduled opponent. Hover a player for their full stat-line breakdown. Position is inferred from ' +
      'usage, not a verified roster field \u2014 treat it as a label, not a guarantee.</span></div>';
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
            '<div class="proj-meta">' + esc(g.kickoff) + ' \\u00b7 line: ' + esc(g.book) +
              (g.flags && g.flags.length ? g.flags.map(function (f) {
                var cls = f.level === 1 ? ' is-warn' : (f.level === 2 ? ' is-good' : '');
                return '<span class="flag-chip' + cls + '">' + esc(f.text) + '</span>';
              }).join('') : '') +
            '</div>' +
          '</div>' +
          '<div class="proj-side">' + helmet(g.home, 86, 51, true) +
            '<div class="proj-bar" style="background:' + esc(hC) + '"></div></div>' +
        '</div></div>' +

        '<div class="scoreboard">' +
          '<div class="score-cell"><div class="score-label">' + esc(h.abbr) + ' margin</div>' +
            '<div class="score-value">' + M.signed(-g.modelSpread) + '</div></div>' +
          '<div class="divider-v" style="height:auto"></div>' +
          '<div class="score-cell score-cell--wide"><div class="score-label">' + esc(h.abbr) + ' win prob</div>' +
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
            '<span class="cover">shaded: normal-model probability, ' + (p.sideProb * 100).toFixed(1) + '%</span></div>' +
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
          // Bug fix: this used to always read c.coverProb / c.game.marketMoneyline
          // here, which are HOME-team numbers regardless of which side c.playLabel
          // actually names (see priceGame's Moneyline branch comment) -- for any
          // row recommending the AWAY side, that silently showed the home team's
          // price next to the away team's name and computed "EV" from the home
          // team's win probability against the home team's price, producing a
          // real number that had nothing to do with the labeled bet (this is what
          // produced things like a 4-figure "-6500" price next to a 20+ point
          // underdog, and absurd +200%/+300% "edges" on the other end). c.sideProb
          // and c.sideMoneyline are the picked-side-correct versions; c.coverProb
          // stays correct as-is for Spread, where home/away are just a sign flip
          // of the same number.
          var sidePrice = c.market === 'Moneyline' ? c.sideMoneyline : -110;
          var sideProb  = c.market === 'Moneyline' ? c.sideProb : c.coverProb;
          var ev = M.expectedValue(sideProb, sidePrice) * 100;
          var sidePriceLabel = c.market === 'Moneyline' ? ((sidePrice > 0 ? '+' : '') + sidePrice) : '-110';
          // Fixing the price/EV attribution bug above (see priceGame's
          // Moneyline branch) revealed a SEPARATE, real issue underneath:
          // several preseason-only picks still show implausibly large EV
          // (100%+) once the price is correctly attributed. That's not a
          // display bug -- it's the preseason estimator (last season's SP+
          // diff run through one fixed, league-wide sigma, no in-season
          // data yet) disagreeing wildly with a much better-informed
          // market. A real, durable edge that size doesn't exist in CFB
          // betting, so this is far more likely the crude preseason
          // estimate being wrong than a hidden opportunity -- flagged
          // in-place (per user decision) rather than hidden, since it's
          // still possible, just something to treat with real skepticism
          // until the in-season trained model (real rolling data, already
          // validated via backtest) takes over for that specific game.
          var isTrainedGame = (c.game.flags || []).some(function (f) { return f.text === 'In-season model'; });
          var showCaveat = c.market === 'Moneyline' && !isTrainedGame && Math.abs(ev) >= 50;
          return '<div class="row"><div class="row-accent" style="background:' + esc(col) + '"></div>' +
            '<div class="row-body card-row">' +
              '<div><div class="card-play">' + esc(c.playLabel) + '</div>' +
                '<div class="card-note">' + esc(abbrOf(c.game.away) + ' at ' + abbrOf(c.game.home) + ' \\u00b7 ' + c.game.kickoff) +
                  ' \\u00b7 <span style="font-weight:700;color:' + (isTrainedGame ? '#2ecc71' : '#8A94A3') + '">' + (isTrainedGame ? 'TRAINED' : 'UNTRAINED') + '</span></div></div>' +
              '<div class="card-price">' + esc(sidePriceLabel) + '</div>' +
              '<div class="card-conf">' + (sideProb * 100).toFixed(1) + '%</div>' +
              '<div class="num"><span class="tier"><span>' + esc(c.tier) + '</span></span></div>' +
              '<div class="num"><button class="track-btn" onclick="window.__cfbTrack(' + trackPayload(c) + ')">+TRK</button></div>' +
              (showCaveat ? '<div style="grid-column:1/-1;font-size:11px;color:var(--amber);padding-top:6px;line-height:1.4">' +
                '\u26a0 Large model/market gap on a preseason-only estimate (no in-season data yet for this game) \u2014 likely reflects the model\u2019s limits, not a confirmed edge. Extra caution advised.</div>' : '') +
            '</div></div>';
        }).join('') : '<div class="empty-state">No plays clear the ' + D.meta.minEdge.toFixed(1) + '-pt threshold on ' + state.market + ' right now.</div>') +
      '</div>';
  }

  var PROP_BOOK_LABELS = {
    draftkings: 'DraftKings', fanduel: 'FanDuel', betmgm: 'BetMGM',
    williamhill_us: 'Caesars', espnbet: 'ESPN Bet', betrivers: 'BetRivers',
    fanatics: 'Fanatics', bovada: 'Bovada', betonlineag: 'BetOnline',
    mybookieag: 'MyBookie', ballybet: 'Bally Bet', betparx: 'betPARX',
    fliff: 'Fliff', hardrockbet: 'Hard Rock Bet',
  };
  function bookLabel(key) { return PROP_BOOK_LABELS[key] || key; }

  function renderProps() {
    var catalog = D.propCatalog || [];
    var live = D.propsLive || [];
    var byMarket = {};
    live.forEach(function (p) { (byMarket[p.market_name] = byMarket[p.market_name] || []).push(p); });
    var liveCount = catalog.filter(function (m) { return byMarket[m] && byMarket[m].length; }).length;

    var intro = liveCount ? '' : '<p class="pcard-note" style="margin-bottom:14px">Player props are sourced from real sportsbooks (DraftKings, FanDuel, BetMGM, Caesars, etc) via The Odds API, not a DFS site. ' +
      'Real books don’t post props on every game \\u2014 coverage is normal for ranked/primetime matchups and thin or empty elsewhere. Any market with posted lines switches to LIVE automatically on the next data refresh.</p>';

    if (!catalog.length) {
      return '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Player Props</h2></div></div>' +
        '<div class="empty-state">No prop market catalog loaded.</div>';
    }

    // model_predicted_value/model_edge/model_lean only exist once
    // export_dashboard_data.py (src/features/live_player_features.py)
    // actually matched this player to real in-season stats + a trained
    // model AND found a real opponent-defense number -- normal to be
    // absent for most/all rows before the season has real in-season data.
    // No fabricated overlay when it's missing. is_official_play/model_ev
    // only exist alongside that (see export_dashboard_data.py) -- a real
    // dollar-EV bar (>=3%, normal confidence, real price) against the
    // actual posted price on the model's leaned side, same philosophy as
    // the Edge Board's sideEV for moneylines. Anything scored but short of
    // that bar is a LEAN, never silently upgraded to OFFICIAL.
    function hasModel(r) { return r.model_predicted_value != null && r.model_lean && r.model_edge != null; }
    function absEdge(r) { return hasModel(r) ? Math.abs(r.model_edge) : -1; }
    function rankVal(r) { return r.model_ev != null ? r.model_ev : absEdge(r); }
    function tagHtml(r) {
      return r.is_official_play
        ? '<span class="pchip is-official">OFFICIAL</span>'
        : '<span class="pchip is-lean">LEAN</span>';
    }

    // ---- Official Plays: every row that clears the real-EV bar, ranked by
    // EV, so the plays actually worth acting on surface immediately
    // instead of being buried inside whichever market card they fall in.
    var official = live.filter(function (r) { return r.is_official_play; })
      .slice().sort(function (a, b) { return rankVal(b) - rankVal(a); });
    // ---- Leans: scored but short of the official bar -- still shown, just
    // clearly labeled, so a near-miss isn't confused with a real play.
    var leans = live.filter(function (r) { return hasModel(r) && !r.is_official_play; })
      .slice().sort(function (a, b) { return rankVal(b) - rankVal(a); });

    function bestCard(r) {
      var isOver = r.model_lean === 'over';
      var price = isOver ? r.over_price : r.under_price;
      return '<div class="prop-best-card">' +
        '<div class="prop-best-top ' + (isOver ? 'is-over' : 'is-under') + '"></div>' +
        '<div class="prop-best-body">' +
          '<div class="prop-best-name">' + esc(playerLabel(r)) + ' ' + tagHtml(r) + '</div>' +
          '<div class="prop-best-meta">' + esc(r.market_name) + ' · ' + (isOver ? 'O' : 'U') + ' ' + esc(r.line) + '</div>' +
          '<div class="prop-best-edge ' + (r.model_edge >= 0 ? 'is-pos' : 'is-neg') + '">' +
            (r.model_ev != null ? ((r.model_ev >= 0 ? '+' : '') + r.model_ev.toFixed(1) + '% EV') : ((r.model_edge >= 0 ? '+' : '') + r.model_edge.toFixed(1) + ' edge')) +
          '</div>' +
          '<div class="prop-best-sub">model ' + r.model_predicted_value.toFixed(1) + (r.model_confidence === 'low' ? ' · low confidence' : '') + '</div>' +
          '<div class="prop-best-price">' + (price != null ? esc(price) : 'no live price yet') + (r.book_used ? ' · ' + esc(bookLabel(r.book_used)) : '') + '</div>' +
        '</div></div>';
    }

    var officialHtml = '<div class="section-head"><div class="section-title"><div class="section-flag is-green"></div><h2>Official Plays</h2></div></div>' +
      (official.length
        ? '<div class="prop-best-row">' + official.slice(0, 8).map(bestCard).join('') + '</div>'
        : '<p class="pcard-note" style="margin-bottom:18px">No player props clear the official-play bar this run (≥ 3% modeled EV at a real posted price, normal confidence only — see the low-confidence note above). Check Leans below or browse By Market.</p>');

    var leansHtml = leans.length
      ? '<div class="section-head mt-lg"><div class="section-title"><div class="section-flag"></div><h2>Leans</h2></div></div>' +
        '<p class="pcard-note" style="margin-bottom:12px">Scored, but short of the official-play bar — worth a look, not a recommended play.</p>' +
        '<div class="prop-lean-row">' + leans.slice(0, 8).map(function (r) {
          var isOver = r.model_lean === 'over';
          var price = isOver ? r.over_price : r.under_price;
          return '<div class="prop-lean-card">' +
            '<div class="prop-best-name">' + esc(playerLabel(r)) + ' ' + tagHtml(r) + '</div>' +
            '<div class="prop-best-meta">' + esc(r.market_name) + ' · ' + (isOver ? 'O' : 'U') + ' ' + esc(r.line) +
              (price != null ? ' · ' + esc(price) : '') + '</div>' +
            '<div class="prop-best-sub">model ' + r.model_predicted_value.toFixed(1) +
              (r.model_ev != null ? ' · ' + (r.model_ev >= 0 ? '+' : '') + r.model_ev.toFixed(1) + '% EV' : '') +
              (r.model_confidence === 'low' ? ' · low confidence' : '') + '</div>' +
          '</div>';
        }).join('') + '</div>'
      : '';

    // ---- By-market tabs: browse one prop type at a time instead of every
    // market's card stacked on one page. ----
    var tabs = ['ALL'].concat(catalog);
    var activeMarket = tabs.indexOf(state.propMarket) === -1 ? 'ALL' : state.propMarket;
    var tabsHtml = '<div class="prop-tabs">' + tabs.map(function (m) {
      var n = m === 'ALL' ? live.length : (byMarket[m] || []).length;
      return '<button class="tab tab--prop' + (m === activeMarket ? ' is-active' : '') + '" data-prop-market="' + esc(m) + '">' +
        '<span>' + esc(m) + (n ? ' <span class="prop-tab-count">' + n + '</span>' : '') + '</span></button>';
    }).join('') + '</div>';

    var visibleMarkets = activeMarket === 'ALL' ? catalog : [activeMarket];

    var cards = visibleMarkets.map(function (m) {
      var rows = (byMarket[m] || []).slice().sort(function (a, b) { return rankVal(b) - rankVal(a); });
      if (rows.length) {
        return '<div class="pcard"><div class="pcard-head"><span>' + esc(m) + '</span><span class="pchip is-live">LIVE</span></div>' +
          rows.map(function (r) {
            var modelLine = hasModel(r)
              ? '<div class="pcard-line-row is-scored" style="border-top:none;padding-top:0">' +
                  '<span style="color:var(--muted-3);font-size:9.5px;letter-spacing:.04em">' +
                    tagHtml(r) + ' Model: ' + r.model_predicted_value.toFixed(1) + ' (' + r.model_lean.toUpperCase() + (r.model_confidence === 'low' ? ', low confidence' : '') + ')' +
                  '</span>' +
                  '<span style="color:' + (r.model_edge >= 0 ? 'var(--green)' : 'var(--red)') + ';font-weight:700;font-size:10.5px">' +
                    (r.model_ev != null ? ((r.model_ev >= 0 ? '+' : '') + r.model_ev.toFixed(1) + '% EV') : ((r.model_edge >= 0 ? '+' : '') + r.model_edge.toFixed(1) + ' edge')) +
                  '</span>' +
                '</div>'
              : '';
            var bookTag = r.book_used ? ' <span style="color:var(--muted-3);font-size:9px">(' + esc(bookLabel(r.book_used)) + ')</span>' : '';
            return '<div class="pcard-line-row"><span>' + esc(playerLabel(r)) + ' · ' + esc(r.line) + '</span>' +
              '<span>O ' + esc(r.over_price) + ' / U ' + esc(r.under_price) + bookTag + '</span></div>' + modelLine;
          }).join('') + '</div>';
      }
      return '<div class="pcard"><div class="pcard-head"><span>' + esc(m) + '</span><span class="pchip is-pending">NOT POSTED</span></div>' +
        '<p class="pcard-note">A market real sportsbooks offer for CFB, but no book has posted a line for this game yet.</p></div>';
    }).join('');

    return '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Player Props</h2></div></div>' +
      intro + officialHtml + leansHtml +
      '<div class="section-head mt-lg"><div class="section-title"><div class="section-flag"></div><h2>By Market</h2></div></div>' +
      tabsHtml + '<div class="pcard-grid">' + cards + '</div>';
  }

  /* ---- Tracker (localStorage, this browser only, real bets you log) ---- */

  var TRK_KEY = 'cfb_model_tracker_v1';
  function loadTrk() { try { return JSON.parse(localStorage.getItem(TRK_KEY)) || []; } catch (e) { return []; } }
  function saveTrk(items) { localStorage.setItem(TRK_KEY, JSON.stringify(items)); }

  // ---- Seed data baked into the repo (survives a cleared browser --
  // this is the recovered 9/3-9/19 manual spread/moneyline log, restored
  // 9/17/2026 after a browser-storage loss). Merges by id on every load,
  // so it's a no-op once a browser already has these, and self-heals any
  // browser/laptop that doesn't. ----
  var SEED_TRACKER = [{"id": "t1789600000000seed00", "description": "MD +3.0 \u2014 VT at MD", "date": "Sat 9/19, 11:30PM UTC", "type": "Spread", "price": -110, "edge": 13.6, "stake": 50, "status": null, "modelSource": "preseason_prior"}, {"id": "t1789600000137seed01", "description": "LT +19.5 \u2014 LT at BAY", "date": "Sat 9/19, 8:00PM UTC", "type": "Spread", "price": -110, "edge": 14.1, "stake": 50, "status": null, "modelSource": "preseason_prior"}, {"id": "t1789600000274seed02", "description": "MISS +2.5 \u2014 LSU at MISS", "date": "Sat 9/19, 11:30PM UTC", "type": "Spread", "price": -110, "edge": 15.1, "stake": 50, "status": null, "modelSource": "preseason_prior"}, {"id": "t1789600000411seed03", "description": "UNT +3.0 \u2014 UNLV at UNT", "date": "Sat 9/12, 7:45PM UTC", "type": "Spread", "price": -110, "edge": 12.9, "stake": 50, "status": "win", "modelSource": "preseason_prior"}, {"id": "t1789600000548seed04", "description": "CONN +11.5 \u2014 MD at CONN", "date": "Sat 9/12, 7:30PM UTC", "type": "Spread", "price": -110, "edge": 18.1, "stake": 50, "status": "loss", "modelSource": "preseason_prior"}, {"id": "t1789600000685seed05", "description": "SDSU +12.5 \u2014 SDSU at UCLA", "date": "Sat 9/12, 11:15PM UTC", "type": "Spread", "price": -110, "edge": 19, "stake": 50, "status": "loss", "modelSource": "preseason_prior"}, {"id": "t1789600000822seed06", "description": "LT +35.5 \u2014 LT at LSU", "date": "Sat 9/12, 11:30PM UTC", "type": "Spread", "price": -110, "edge": 24.3, "stake": 50, "status": "win", "unofficial": true, "modelSource": "preseason_prior"}, {"id": "t1789600000959seed07", "description": "ODU +19.5 \u2014 ODU at VT", "date": "Sat 9/12, 4:00PM UTC", "type": "Spread", "price": -110, "edge": 26.4, "stake": 50, "status": "loss", "modelSource": "preseason_prior"}, {"id": "t1789600001096seed08", "description": "MICH +5.5 \u2014 OU at MICH", "date": "Sat 9/12, 4:00PM UTC", "type": "Spread", "price": -110, "edge": 5.3, "stake": 50, "status": "win", "modelSource": "preseason_prior"}, {"id": "t1789600001233seed09", "description": "WSU +17.5 \u2014 WSU at KSU", "date": "Sat 9/12, 4:00PM UTC", "type": "Spread", "price": -110, "edge": 11.8, "stake": 50, "status": "loss", "modelSource": "preseason_prior"}, {"id": "t1789600001370seed10", "description": "USF +3.0 \u2014 USF at ARMY", "date": "Sat 9/12, 4:00PM UTC", "type": "Spread", "price": -110, "edge": 6.5, "stake": 50, "status": "win", "modelSource": "preseason_prior"}, {"id": "t1789600001507seed11", "description": "RUTG +3.0 \u2014 RUTG at BC", "date": "Fri 9/11, 11:30PM UTC", "type": "Spread", "price": -110, "edge": 5.6, "stake": 50, "status": "loss", "modelSource": "preseason_prior"}, {"id": "t1789600001644seed12", "description": "BOIS +24.5 \u2014 BOIS at ORE", "date": "Sat 9/5, 7:30PM UTC", "type": "Spread", "price": -110, "edge": 5.9, "stake": 50, "status": "win", "unofficial": true, "modelSource": "preseason_prior"}, {"id": "t1789600001781seed13", "description": "ORST +21.0 \u2014 ORST at HOU", "date": "Sat 9/5, 4:00PM UTC", "type": "Spread", "price": -110, "edge": 2.1, "stake": 50, "status": "win", "unofficial": true, "modelSource": "preseason_prior"}, {"id": "t1789600001918seed14", "description": "CCU +21.0 \u2014 CCU at WVU", "date": "Sat 9/5, 4:00PM UTC", "type": "Spread", "price": -110, "edge": 11.9, "stake": 50, "status": "win", "unofficial": true, "modelSource": "preseason_prior"}, {"id": "t1789600002055seed15", "description": "WMU +27.5 \u2014 WMU at MICH", "date": "Sat 9/5, 11:00PM UTC", "type": "Spread", "price": -110, "edge": 14.8, "stake": 50, "status": "win", "unofficial": true, "modelSource": "preseason_prior"}, {"id": "t1789600002192seed16", "description": "JMU ML \u2014 LIB at JMU", "date": "Sat 9/5, 4:00PM UTC", "type": "Moneyline", "price": -225, "edge": 17.7, "stake": 50, "status": "win", "modelSource": "preseason_prior"}, {"id": "t1789600002329seed17", "description": "TLSA +13.5 \u2014 OKST at TLSA", "date": "Sat 9/5, 7:45PM UTC", "type": "Spread", "price": -110, "edge": 20.5, "stake": 50, "status": "win", "modelSource": "preseason_prior"}, {"id": "t1789600002466seed18", "description": "TOL +10.0 \u2014 TOL at MSU", "date": "Sat 9/5, 12:00AM UTC", "type": "Spread", "price": -110, "edge": 12.5, "stake": 50, "status": "win", "modelSource": "preseason_prior"}, {"id": "t1789600002603seed19", "description": "UAB +27.5 \u2014 UAB at ILL", "date": "Fri 9/4, 1:00AM UTC", "type": "Spread", "price": -110, "edge": 5.1, "stake": 50, "status": "win", "unofficial": true, "modelSource": "preseason_prior"}, {"id": "t1789600002740seed20", "description": "GT -6.5 \u2014 COLO at GT", "date": "Fri 9/4, 12:00AM UTC", "type": "Spread", "price": -110, "edge": 8.7, "stake": 50, "status": "loss", "modelSource": "preseason_prior"}, {"id": "t1789600002877seed21", "description": "AKR +25.5 \u2014 AKR at WAKE", "date": "Thu 9/3, 11:00PM UTC", "type": "Spread", "price": -110, "edge": 9, "stake": 50, "status": "win", "unofficial": true, "modelSource": "preseason_prior"}, {"id": "t1789600003014seed22", "description": "NMSU +31.5 \u2014 NMSU at FSU", "date": "Sat 8/29, 11:00PM UTC", "type": "Spread", "price": -110, "edge": 13, "stake": 50, "status": "win", "unofficial": true, "modelSource": "preseason_prior"}, {"id": "t1789600003151seed23", "description": "UVA -4.0 \u2014 NCSU at UVA", "date": "Sat 8/29, 7:30PM UTC", "type": "Spread", "price": -110, "edge": 3.8, "stake": 50, "status": "win", "modelSource": "preseason_prior"}];
  function seedTrackerIfMissing() {
    var items = loadTrk();
    var known = {};
    items.forEach(function (it) { known[it.id] = true; });
    var added = false;
    SEED_TRACKER.forEach(function (it) {
      if (!known[it.id]) { items.push(it); known[it.id] = true; added = true; }
    });
    if (added) saveTrk(items);
  }

  // ---- Export / import (moves the tracker between browsers/laptops --
  // localStorage never syncs on its own, this is the manual bridge) ----
  window.__cfbTrkExport = function () {
    var blob = new Blob([JSON.stringify(loadTrk(), null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'cfb_tracker_export_' + new Date().toISOString().slice(0, 10) + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  window.__cfbTrkImportFile = function (input) {
    var file = input.files && input.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function (e) {
      var imported;
      try { imported = JSON.parse(e.target.result); } catch (err) {
        alert('That file is not valid tracker JSON.');
        return;
      }
      // Grades file (added 9/2026): {"grades": [...]} -- sets W/L/P on rows
      // ALREADY in the tracker instead of adding new ones, so a batch of
      // results graded from real box scores can be applied in one click
      // rather than row by row. Two match styles:
      //   game lines: {"play": "PITT ML", "status": "win"} -- matches the
      //     text before " \u2014 " in the row's description exactly.
      //   props: {"player": "Noah Kim", "market": "Pass Yards", "status": ...}
      //     -- matches any Prop row whose description contains
      //     "<player> <market>" (works with or without the newer team-
      //     abbreviation prefix, and regardless of which drifted line the
      //     row was tracked at -- the grades file only lists a player+market
      //     when the result is the same at every line that was tracked).
      // Only rows with no grade yet are touched, so a manual grade you've
      // already entered is never overwritten.
      if (imported && !Array.isArray(imported) && Array.isArray(imported.grades)) {
        var trk = loadTrk();
        var applied = 0;
        imported.grades.forEach(function (g) {
          if (!g || ['win', 'loss', 'push'].indexOf(g.status) === -1) return;
          trk.forEach(function (it) {
            if (it.status) return;
            var desc = it.description || '';
            var hit = g.play
              ? desc.split(' \\u2014 ')[0] === g.play
              : (g.player && g.market && it.type === 'Prop' && desc.indexOf(g.player + ' ' + g.market) !== -1);
            if (hit) { it.status = g.status; applied++; }
          });
        });
        saveTrk(trk);
        input.value = '';
        render();
        alert(applied + ' tracked play(s) graded from the file.');
        return;
      }
      if (!Array.isArray(imported)) {
        alert('That file is not a valid tracker export.');
        return;
      }
      // Merge by id so re-importing the same file twice (or importing on
      // a laptop that already has some overlapping plays) never duplicates
      // a row -- only genuinely new ids get added.
      var existing = loadTrk();
      var known = {};
      existing.forEach(function (it) { known[it.id] = true; });
      var merged = existing.slice();
      var added = 0;
      imported.forEach(function (it) {
        if (it && it.id && !known[it.id]) { merged.push(it); known[it.id] = true; added++; }
      });
      saveTrk(merged);
      input.value = '';
      render();
      alert(added + ' play(s) imported (' + (imported.length - added) + ' already present, skipped).');
    };
    reader.readAsText(file);
  };

  window.__cfbTrack = function (play) {
    var items = loadTrk();
    items.unshift({
      id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
      description: play.description, date: play.date, type: play.type,
      price: play.price, edge: play.edge, stake: 50, status: null,
      unofficial: !!play.unofficial,
      modelSource: play.modelSource || null
    });
    saveTrk(items);
    render();
    var el = document.getElementById('trk-gamelines-section') || document.getElementById('trk-section');
    if (el && typeof el.scrollIntoView === 'function') el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  // Manual flip for any tracked play -- overrides the auto-detection above
  // in either direction, and is the only way to mark/unmark a manually-
  // added or auto-tracked-prop play as unofficial.
  window.__cfbTrkToggleUnofficial = function (id) {
    var items = loadTrk();
    var it = items.find(function (x) { return x.id === id; });
    if (it) { it.unofficial = !it.unofficial; saveTrk(items); render(); }
  };

  // Manual add -- for plays with no model behind them yet (e.g. Totals,
  // which the live pipeline doesn't project -- see backtest_totals_quick.py,
  // whose naive average lost to market in the 2026 wk1-2 check and was
  // deliberately NOT wired into the Edge Board). Same item schema as
  // window.__cfbTrack, just filled in by hand instead of from a model row.
  window.__cfbTrkManualAdd = function () {
    var typeEl = document.getElementById('trk-manual-type');
    var descEl = document.getElementById('trk-manual-desc');
    var dateEl = document.getElementById('trk-manual-date');
    var priceEl = document.getElementById('trk-manual-price');
    var edgeEl = document.getElementById('trk-manual-edge');
    var stakeEl = document.getElementById('trk-manual-stake');
    var unofficialEl = document.getElementById('trk-manual-unofficial');
    var description = (descEl.value || '').trim();
    if (!description) { descEl.focus(); return; }
    var items = loadTrk();
    items.unshift({
      id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
      description: description,
      date: (dateEl.value || '').trim(),
      type: typeEl.value,
      price: priceEl.value !== '' ? Number(priceEl.value) : -110,
      edge: edgeEl.value !== '' ? Number(edgeEl.value) : null,
      stake: stakeEl.value !== '' ? Number(stakeEl.value) : 50,
      status: null,
      unofficial: !!(unofficialEl && unofficialEl.checked),
      modelSource: 'manual',
    });
    saveTrk(items);
    descEl.value = ''; dateEl.value = ''; edgeEl.value = '';
    if (unofficialEl) unofficialEl.checked = false;
    render();
  };

  // ---- Auto-track official player props -----------------------------
  // Every prop the pipeline itself flags as an "official play" (see
  // export_dashboard_data.py -- normal confidence + real EV against the
  // actual posted price) gets logged into this browser's Tracker
  // automatically, no +TRK click needed. Runs once per page load, keyed by
  // a stable sourceId (game + player + market + line + lean) so re-opening
  // the dashboard after a later pipeline run only adds genuinely NEW
  // official plays, never duplicates one already logged. Manually-tracked
  // rows (via +TRK) have no sourceId and are never touched by this.
  function propSourceId(r) {
    // Deliberately excludes r.line and r.model_lean -- a player's posted
    // line drifts a little between workflow runs (e.g. Noah Kim's Pass
    // Yards moving 131.5 -> 130.5 -> 136.5 across several runs), and
    // including the line here meant every drift got auto-tracked as a
    // "new" prop, producing 10 rows for what's really one ongoing read on
    // one player in one market. Fixed 9/19/2026 -- the identity of an
    // auto-tracked prop is the player+market+game, full stop; whichever
    // line/lean it had the FIRST time it cleared the official bar is what
    // gets kept (see dedupeAutoProps for the one-time cleanup of rows
    // already duplicated under the old key).
    return ['autoprop', r.fixture_id, r.player_name, r.market_name].join('|');
  }

  function autoTrackOfficialProps() {
    var official = (D.propsLive || []).filter(function (r) { return r.is_official_play; });
    if (!official.length) return;
    var items = loadTrk();
    var known = {};
    items.forEach(function (it) { if (it.sourceId) known[it.sourceId] = true; });
    var added = false;
    official.forEach(function (r) {
      var sid = propSourceId(r);
      if (known[sid]) return;
      var isOver = r.model_lean === 'over';
      var price = isOver ? r.over_price : r.under_price;
      items.unshift({
        id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
        sourceId: sid,
        description: playerLabel(r) + ' ' + r.market_name + ' ' + (isOver ? 'O' : 'U') + ' ' + r.line,
        date: (r.start_time || '').slice(0, 10),
        type: 'Prop',
        price: price != null ? price : null,
        edge: r.model_ev,
        stake: 50, status: null, auto: true,
      });
      known[sid] = true;
      added = true;
    });
    if (added) saveTrk(items);
  }

  // One-time cleanup for rows already duplicated under the OLD sourceId
  // (which included the line, so every line drift re-tracked the same
  // player+market as "new" -- see propSourceId's comment). Groups every
  // auto-tracked Prop by player+market (parsed back out of its
  // description, since older rows never stored those as separate fields),
  // and where a group has more than one row, keeps only the EARLIEST one
  // (by the timestamp embedded in its own id) and removes the rest --
  // preserving "when the model first flagged this player in this market"
  // rather than picking arbitrarily. Safe to run on every load: a clean
  // tracker (no duplicate groups) is a no-op, and the new propSourceId
  // means fresh auto-tracks won't create new duplicates for this to
  // clean up going forward.
  function dedupeAutoProps() {
    var items = loadTrk();
    var groups = {};
    items.forEach(function (it) {
      if (it.type !== 'Prop' || !it.auto) return;
      var m = /^(.*?)\\s+[OU]\\s+[\\d.]+$/.exec(it.description || '');
      var key = m ? m[1] : it.description;
      (groups[key] = groups[key] || []).push(it);
    });
    var dropIds = {};
    Object.keys(groups).forEach(function (key) {
      var g = groups[key];
      if (g.length <= 1) return;
      g.sort(function (a, b) {
        var ta = parseInt(String(a.id || '').slice(1), 10) || 0;
        var tb = parseInt(String(b.id || '').slice(1), 10) || 0;
        return ta - tb;
      });
      for (var i = 1; i < g.length; i++) dropIds[g[i].id] = true;
    });
    var dropCount = Object.keys(dropIds).length;
    if (dropCount) {
      saveTrk(items.filter(function (it) { return !dropIds[it.id]; }));
    }
    return dropCount;
  }

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

  window.__cfbUnitSize = function (value) {
    var n = Number(value);
    if (n > 0) { state.unitSize = n; render(); }
  };

  function fmtMoney(n) {
    var sign = n < 0 ? '-' : '';
    return sign + '$' + Math.abs(n).toFixed(2).replace(/\\.00$/, '');
  }

  function renderTrackerSection(items, opts) {
    var wins = 0, losses = 0, pushes = 0, staked = 0, profit = 0;
    // Units record is independent of whatever real $ stake was logged per
    // bet -- it grades every bet as a flat 1 unit won/lost/pushed (the
    // standard way bettors describe a record independent of bet-sizing:
    // "+4.3u"), then scales that unit total by a user-chosen $/unit so it's
    // meaningful at whatever bankroll they actually apply this record to.
    var unitWins = 0, unitLosses = 0, unitPushes = 0, unitsGraded = 0, totalUnits = 0;
    var unofficialCount = 0;
    items.forEach(function (it) {
      var p = computeProfit(it);
      if (it.unofficial) { unofficialCount++; return; } // excluded from every record/profit/units stat below, shown in the row list only
      if (it.status === 'win') wins++;
      if (it.status === 'loss') losses++;
      if (it.status === 'push') pushes++;
      if (it.status) staked += Number(it.stake) || 0;
      if (p !== null) profit += p;

      if (it.status && it.price != null) {
        unitsGraded++;
        if (it.status === 'win') { unitWins++; totalUnits += americanToDecimal(it.price) - 1; }
        else if (it.status === 'loss') { unitLosses++; totalUnits -= 1; }
        else if (it.status === 'push') { unitPushes++; }
      }
    });

    var head = '<div class="section-head mt-lg" id="' + opts.sectionId + '"><div class="section-title"><div class="section-flag is-green"></div><h2>' + esc(opts.title) + '</h2></div></div>' +
      (opts.extraHtml || '');

    if (!items.length) {
      return head + '<div class="trk-empty">' + esc(opts.emptyMsg) + '</div>';
    }

    var summary = '<div class="trk-summary">' +
      '<div class="trk-box"><div class="trk-label">Tracked</div><div class="trk-value">' + items.length + (unofficialCount ? ' <span style="font-size:11px;font-weight:400;color:var(--muted-4,#888)">(' + unofficialCount + ' unofficial)</span>' : '') + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Record</div><div class="trk-value">' + wins + '-' + losses + '-' + pushes + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Staked</div><div class="trk-value">' + fmtMoney(staked) + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Profit</div><div class="trk-value ' + (profit >= 0 ? 'is-pos' : 'is-neg') + '">' + fmtMoney(profit) + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">ROI</div><div class="trk-value">' + (staked > 0 ? ((profit / staked) * 100).toFixed(1) + '%' : '0%') + '</div></div>' +
    '</div>';

    // Unit size is one shared preference across both sections (state.unitSize)
    // -- switching it in one section updates the other too, which is the
    // right default for "what's my overall unit size" rather than tracking
    // two independent sizes.
    var unitPresets = [10, 50, 100];
    var unitSizeBar = '<div class="prop-tabs" style="margin:14px 0 10px">' +
      unitPresets.map(function (v) {
        return '<button class="tab tab--prop' + (state.unitSize === v ? ' is-active' : '') + '" data-unit-size="' + v + '"><span>$' + v + '/unit</span></button>';
      }).join('') +
      '<input class="unit-size-input" type="number" min="1" step="5" value="' + state.unitSize + '" ' +
        'onchange="window.__cfbUnitSize(this.value)" placeholder="custom $/unit">' +
    '</div>';

    var unitsSummary = '<div class="trk-summary-3">' +
      '<div class="trk-box"><div class="trk-label">Units Record</div><div class="trk-value">' + unitWins + '-' + unitLosses + '-' + unitPushes + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Total Units</div><div class="trk-value ' + (totalUnits >= 0 ? 'is-pos' : 'is-neg') + '">' + (totalUnits >= 0 ? '+' : '') + totalUnits.toFixed(2) + 'u</div></div>' +
      '<div class="trk-box"><div class="trk-label">Est. $ at $' + state.unitSize + '/unit</div><div class="trk-value ' + (totalUnits >= 0 ? 'is-pos' : 'is-neg') + '">' + fmtMoney(totalUnits * state.unitSize) + '</div></div>' +
    '</div>';

    var unitsHtml = '<div class="section-head" style="border-bottom:none;margin-top:14px"><div class="section-title"><div class="section-flag"></div><h2 style="font-size:15px">Units</h2></div></div>' +
      unitSizeBar +
      (unitsGraded ? unitsSummary : '<div class="trk-empty">No graded plays yet — mark a play W/L/P below to start building a units record.</div>');

    var rows = items.map(function (it) {
      var p = computeProfit(it);
      var edgeUnit = it.type === 'Moneyline' ? 'pp' : (it.type === 'Prop' ? '%' : 'pt');
      var modelTagText = it.modelSource === 'trained_model' ? 'TRAINED'
        : it.modelSource === 'preseason_prior' ? 'UNTRAINED'
        : it.modelSource === 'manual' ? 'MANUAL' : null;
      var modelTagColor = it.modelSource === 'trained_model' ? '#2ecc71' : '#8A94A3';
      return '<div class="trk-row">' +
        '<div>' + esc(it.description) + (it.auto ? ' <span class="trk-auto-tag">AUTO</span>' : '') + (it.unofficial ? ' <span class="trk-auto-tag" style="background:#ff0000;color:#fff;font-weight:700">UNOFFICIAL</span>' : '') + (modelTagText ? ' <span class="trk-auto-tag" style="background:' + modelTagColor + '26;color:' + modelTagColor + ';border:1px solid ' + modelTagColor + '">' + modelTagText + '</span>' : '') + '</div>' +
        '<div>' + esc(it.date) + '</div>' +
        '<div>' + esc(it.type) + '</div>' +
        '<div>' + (it.price > 0 ? '+' : '') + (it.price != null ? esc(it.price) : '—') + '</div>' +
        '<div>' + (it.edge != null ? ((it.edge > 0 ? '+' : '') + esc(it.edge) + edgeUnit) : '—') + '</div>' +
        '<div class="trk-status-btns">' +
          '<input type="number" value="' + it.stake + '" min="0" step="5" onchange="window.__cfbTrkStake(\\'' + it.id + '\\', this.value)">' +
          '<button class="trk-status-btn' + (it.status === 'win' ? ' is-win' : '') + '" onclick="window.__cfbTrkStatus(\\'' + it.id + '\\',\\'win\\')">W</button>' +
          '<button class="trk-status-btn' + (it.status === 'loss' ? ' is-loss' : '') + '" onclick="window.__cfbTrkStatus(\\'' + it.id + '\\',\\'loss\\')">L</button>' +
          '<button class="trk-status-btn' + (it.status === 'push' ? ' is-push' : '') + '" onclick="window.__cfbTrkStatus(\\'' + it.id + '\\',\\'push\\')">P</button>' +
        '</div>' +
        '<div style="color:' + (p === null ? 'var(--muted-4)' : (p >= 0 ? 'var(--green)' : 'var(--red)')) + ';font-weight:700">' + (p === null ? '\\u2014' : fmtMoney(p)) + '</div>' +
        '<div><button class="trk-remove" style="font-size:9px;padding:2px 5px;margin-right:4px" title="Toggle whether this play counts in the record" onclick="window.__cfbTrkToggleUnofficial(\\'' + it.id + '\\')">' + (it.unofficial ? 'MAKE OFFICIAL' : 'MAKE UNOFFICIAL') + '</button><button class="trk-remove" onclick="window.__cfbTrkRemove(\\'' + it.id + '\\')">\\u00d7</button></div>' +
      '</div>';
    }).join('');

    return head + summary + unitsHtml +
      '<div class="thead trk-grid"><div>Description</div><div>Date</div><div>Type</div><div>Price</div><div>Edge</div><div>Stake / Grade</div><div>Profit</div><div></div></div>' +
      rows;
  }

  function renderTracker() {
    var items = loadTrk();
    // Two separate trackers sharing one localStorage log: Game Lines
    // (Moneyline/Spread/Total -- manually tracked via +TRK on the Edge
    // Board or Bet Card) and Player Props (auto-tracked -- see
    // autoTrackOfficialProps). Splitting the STORAGE would risk losing
    // history on a schema change; splitting the DISPLAY gets the same
    // separation the user actually wants without that risk.
    var gameLineItems = items.filter(function (it) { return it.type === 'Moneyline' || it.type === 'Spread' || it.type === 'Total'; });
    var propItems = items.filter(function (it) { return it.type === 'Prop'; });

    // Trained vs. preseason-prior tabs -- added 9/19/2026 so the user can
    // see how each formula actually performs once tracked separately,
    // rather than one blended record hiding whether the mid-season
    // switchover is worth trusting. 'manual' (Totals added by hand, no
    // model behind them) and anything untagged (tracked before this field
    // existed) fall under All but not under either specific tab, rather
    // than being miscounted as one or the other.
    var modelTabs = [
      { id: 'all', label: 'All' },
      { id: 'trained_model', label: 'Trained Model' },
      { id: 'preseason_prior', label: 'Untrained Model' },
    ];
    var modelTabBar = '<div class="prop-tabs" style="margin:10px 0 4px">' +
      modelTabs.map(function (t) {
        return '<button class="tab tab--prop' + (state.trackerModelTab === t.id ? ' is-active' : '') + '" data-model-tab="' + t.id + '"><span>' + esc(t.label) + '</span></button>';
      }).join('') +
    '</div>';
    var gameLineItemsShown = state.trackerModelTab === 'all' ? gameLineItems
      : gameLineItems.filter(function (it) { return it.modelSource === state.trackerModelTab; });

    // Manual add-a-play form -- for anything not driven by a live model
    // signal (right now: Totals). Free text date so it can match the
    // "Sat 9/19, 11:30PM UTC" style the rest of the Tracker uses, but any
    // format you want works since it's just displayed, never parsed.
    var manualAddForm = '<div class="trk-manual-add" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 4px;padding:10px;border:1px solid var(--border-1, #2a2f3a);border-radius:8px">' +
      '<select id="trk-manual-type" style="min-width:110px">' +
        '<option value="Spread">Spread</option>' +
        '<option value="Moneyline">Moneyline</option>' +
        '<option value="Total" selected>Total</option>' +
      '</select>' +
      '<input id="trk-manual-desc" type="text" placeholder="Description, e.g. O 55.5 -- VT at MD" style="flex:2;min-width:220px">' +
      '<input id="trk-manual-date" type="text" placeholder="Date, e.g. Sat 9/19, 11:30PM UTC" style="flex:1;min-width:170px">' +
      '<input id="trk-manual-price" type="number" placeholder="Price" value="-110" style="width:90px">' +
      '<input id="trk-manual-edge" type="number" step="0.1" placeholder="Edge (pts)" style="width:100px">' +
      '<input id="trk-manual-stake" type="number" placeholder="Stake" value="50" style="width:80px">' +
      '<label style="display:flex;align-items:center;gap:4px;font-size:12px;color:var(--muted-4,#888)">' +
        '<input id="trk-manual-unofficial" type="checkbox"> Unofficial' +
      '</label>' +
      '<button class="tab tab--prop" onclick="window.__cfbTrkManualAdd()"><span>+ Add Play</span></button>' +
    '</div>';

    var exportImportBar = '<div class="prop-tabs" style="margin:10px 0 4px">' +
      '<button class="tab tab--prop" onclick="window.__cfbTrkExport()"><span>Export JSON</span></button>' +
      '<label class="tab tab--prop" style="cursor:pointer">' +
        '<span>Import JSON</span>' +
        '<input type="file" accept="application/json,.json" style="display:none" onchange="window.__cfbTrkImportFile(this)">' +
      '</label>' +
    '</div>';

    return '<div class="section-head" id="trk-section"><div class="section-title"><div class="section-flag is-green"></div><h2>Tracker</h2></div></div>' +
      manualAddForm +
      exportImportBar +
      (items.length ? '' : '<div class="trk-empty">No plays tracked yet. Click +TRK on any Edge Board row or Bet Card play, or check back after the next run — official player props get added here automatically.</div>') +
      renderTrackerSection(gameLineItemsShown, {
        title: 'Game Lines (Moneyline / Spread / Total)',
        emptyMsg: state.trackerModelTab === 'all'
          ? 'No game-line plays tracked yet. Click +TRK on any Edge Board row or Bet Card play.'
          : 'No ' + (state.trackerModelTab === 'trained_model' ? 'trained-model' : 'untrained-model') + ' plays tracked yet under this tab.',
        sectionId: 'trk-gamelines-section',
        extraHtml: modelTabBar,
      }) +
      renderTrackerSection(propItems, {
        title: 'Player Props',
        emptyMsg: 'No player-prop plays tracked yet — official props (see the Props tab) get added here automatically after each pipeline run.',
        sectionId: 'trk-props-section',
      });
  }

  function renderClv() {
    var c = D.clv;
    if (!c) return '';
    var avgCls = c.avgClvPoints > 0 ? 'is-pos' : (c.avgClvPoints < 0 ? 'is-neg' : '');
    var avgStr = c.avgClvPoints != null ? (c.avgClvPoints > 0 ? '+' : '') + c.avgClvPoints.toFixed(2) + ' pts' : '\u2014';
    var medStr = c.medianClvPoints != null ? (c.medianClvPoints > 0 ? '+' : '') + c.medianClvPoints.toFixed(2) + ' pts' : '\u2014';
    var pctStr = c.pctPositiveClv != null ? (c.pctPositiveClv * 100).toFixed(1) + '%' : '\u2014';
    var smallSample = c.nGames != null && c.nGames < 20;

    var head = '<div class="section-head mt-lg" id="clv-section"><div class="section-title">' +
      '<div class="section-flag"></div><h2>CLV Track Record</h2></div></div>';

    var summary = '<div class="trk-summary">' +
      '<div class="trk-box"><div class="trk-label">Graded Picks</div><div class="trk-value">' + (c.nGames != null ? c.nGames : '\u2014') + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Avg CLV</div><div class="trk-value ' + avgCls + '">' + avgStr + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Median CLV</div><div class="trk-value">' + medStr + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">% Positive CLV</div><div class="trk-value">' + pctStr + '</div></div>' +
    '</div>';

    var foot = '<div class="table-foot"><span>' +
      'CLV compares the line captured when a game was first flagged as a play to the real ' +
      'closing line. Positive means you\u2019d have gotten a better price than the market\u2019s ' +
      'final number \u2014 the standard forward-looking edge signal, independent of whether any ' +
      'single pick actually won. ' +
      (smallSample ? 'Under 20 graded picks \u2014 too small to read much into yet.' : '') +
      '</span><span>as of ' + esc(c.generatedAt || '') + '</span></div>';

    return head + summary + foot;
  }

  function renderBacktest() {
    var bt = D.backtest;
    if (!bt) return '';
    var sp = bt.spread || {};
    var ml = bt.moneyline || {};
    var atsPct = (sp.atsWinRate != null) ? (sp.atsWinRate * 100).toFixed(1) + '%' : '\\u2014';
    var beatCls = sp.beatMarket === true ? 'is-pos' : (sp.beatMarket === false ? 'is-neg' : '');
    var mlPct = (ml.accuracy != null) ? (ml.accuracy * 100).toFixed(1) + '%' : '\\u2014';
    var smallSample = (sp.nGames != null && sp.nGames < 200);

    var head = '<div class="section-head mt-lg" id="bt-section"><div class="section-title">' +
      '<div class="section-flag"></div><h2>Backtest Track Record</h2></div></div>';

    var summary = '<div class="trk-summary">' +
      '<div class="trk-box"><div class="trk-label">Graded Games</div><div class="trk-value">' + (sp.nGames != null ? sp.nGames : '\\u2014') + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">ATS Win Rate</div><div class="trk-value ' + beatCls + '">' + atsPct + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Breakeven</div><div class="trk-value">' + (sp.breakevenAtsRate != null ? (sp.breakevenAtsRate * 100).toFixed(1) + '%' : '52.4%') + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Margin MAE</div><div class="trk-value">' + (sp.marginMae != null ? sp.marginMae.toFixed(2) + ' pts' : '\\u2014') + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">ML Accuracy</div><div class="trk-value">' + mlPct + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">ML Log Loss</div><div class="trk-value">' + (ml.logLoss != null ? ml.logLoss.toFixed(3) : '\\u2014') + '</div></div>' +
    '</div>';

    var foot = '<div class="table-foot"><span>' +
      (sp.beatMarket === true ? 'Model beat the market\\u2019s closing spread over this sample. '
        : sp.beatMarket === false ? 'Model has NOT beaten the market\\u2019s closing spread over this sample \\u2014 not yet an edge. '
        : '') +
      (smallSample ? 'Under 200 graded games \\u2014 treat as noisy, not a verdict.' : '') +
      '</span><span>Seasons ' + esc((bt.seasonsCovered || []).join(', ')) + ' \\u00b7 as of ' + esc(bt.generatedAt || '') + '</span></div>';

    return head + summary + foot;
  }

  /* ---- mount ------------------------------------------------------------- */

  function renderPage(priced, card, sel) {
    // Each tab shows just its own section(s) now, instead of one long
    // stacked page -- grouped by what the user actually does together:
    // Edge Board keeps the grid + the curated Bet Card + the matchup
    // detail view side by side, since picking a row there is what drives
    // the Projector and the Bet Card is just the qualifying subset of
    // the same data. Model Performance combines Backtest + CLV since
    // both are "how good has this model actually been" history, not
    // something you'd act on day-to-day. Ratings/Props/Fantasy/Tracker
    // each get a page to themselves since they're independent, standalone
    // things to check.
    switch (state.page) {
      case 'ratings': return renderRatings();
      case 'props': return renderProps();
      case 'fantasy': return renderFantasy();
      case 'tracker': return renderTracker();
      case 'performance': return renderBacktest() + renderClv();
      case 'edge':
      default:
        return renderKpis(card) +
          '<div class="main">' +
            '<div>' + renderEdgeBoard(priced, card) + '</div>' +
            '<div>' + renderProjector(sel) + renderBetCard(card) + '</div>' +
          '</div>';
    }
  }

  function render() {
    var priced = D.games.map(function (g) { return M.priceGame(g, opts()); });
    var card = M.buildBetCard(D.games, opts());
    var sel = priced[state.selected] || priced[0];

    document.getElementById('app').innerHTML =
      renderTopbar() +
      renderPageNav() +
      '<div class="wrap">' +
        renderPage(priced, card, sel) +
        '<div class="footer">' +
          '<span>Team marks are generic color-accurate helmets, not school logos. Preseason: no in-season form exists yet for 2026, ' +
          'so every model number here comes from SP+ rating differential plus a fitted home-field constant, adjusted by any active ' +
          'manual injury/availability override (config/injury_overrides.csv \\u2014 hand-maintained, not scraped; no free CFB injury API ' +
          'exists). No Total/Team-total market, weather, travel, pace, returning-production, or futures data is fetched by this pipeline ' +
          '\\u2014 those are simply not shown rather than estimated. Sigma is one league-wide value, not per-game. ' +
          'Tracker is local to this browser only \\u2014 no sync, no real money moved. No model reliably beats a well-priced line on every game.</span>' +
        '</div>' +
      '</div>';
  }

  window.__cfbSearch = function (v) { state.search = v; render(); };

  document.addEventListener('click', function (e) {
    var p = e.target.closest('[data-page]');
    if (p) { state.page = p.dataset.page; return render(); }
    var g = e.target.closest('[data-game]');
    if (g) { state.selected = +g.dataset.game; return render(); }
    var m = e.target.closest('[data-market]');
    if (m) { state.market = m.dataset.market; return render(); }
    var pm = e.target.closest('[data-prop-market]');
    if (pm) { state.propMarket = pm.dataset.propMarket; return render(); }
    var us = e.target.closest('[data-unit-size]');
    if (us) { state.unitSize = Number(us.dataset.unitSize); return render(); }
    var mt = e.target.closest('[data-model-tab]');
    if (mt) { state.trackerModelTab = mt.dataset.modelTab; return render(); }
  });

  seedTrackerIfMissing();
  autoTrackOfficialProps();
  dedupeAutoProps();
  render();
})();
</script>
"""


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    backtest = None
    if os.path.exists(BACKTEST_PATH):
        with open(BACKTEST_PATH) as f:
            backtest = json.load(f)
    else:
        print(f"  [note] {BACKTEST_PATH} not found — Backtest Track Record panel will be omitted "
              f"(expected before scripts/run_backtest.py has run in this pipeline)")

    clv = None
    if os.path.exists(CLV_PATH):
        with open(CLV_PATH) as f:
            clv = json.load(f)
    else:
        print(f"  [note] {CLV_PATH} not found — CLV Track Record panel will be omitted "
              f"(expected before scripts/compute_clv.py has run in this pipeline)")

    model_data = build_model_data(data, backtest=backtest, clv=clv)
    data_script = "<script>\nwindow.MODEL_DATA = " + json.dumps(model_data) + ";\n</script>\n"

    html_out = HEAD_HTML + data_script + MATH_JS + RENDERER_JS + TAIL_HTML

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html_out)

    print(f"Wrote {OUT_PATH} ({model_data['meta']['gamesPriced']} priced of "
          f"{model_data['meta']['totalGames']} total games, {len(model_data['teams'])} "
          f"teams with real ratings, {len(model_data['propCatalog'])} prop markets in catalog, "
          f"{len(model_data['fantasy'])} fantasy projections)")


if __name__ == "__main__":
    main()
