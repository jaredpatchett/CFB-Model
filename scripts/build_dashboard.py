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
# Backtest results by market, side and size of the model's edge (10/2026),
# shown next to each prop on the Props tab. Built by
# src/analysis/prop_edge_history.py, which the tracker
# (scripts/compute_props_clv.py) also uses, so both agree on which groups are
# up. If the backtest files are missing the column just shows the edge.
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
    name_abbr = {t["name"]: t["abbr"] for t in teams}
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
        inj_home = g.get("injuries_home") or []
        inj_away = g.get("injuries_away") or []
        # QB injuries are the ones that move a line most, so they get a
        # visible warning chip; the full list is on the Injury Report page.
        for team_name, lst in ((g["home_team"], inj_home), (g["away_team"], inj_away)):
            for i in lst:
                if i.get("pos") == "QB" and i.get("status") in ("Out", "Out For Season", "IR", "Doubtful", "Questionable", "Game-Time Decision"):
                    flags.append({"text": f"QB {i['player']} ({i['status']})", "level": 1, "qb": True, "team": team_name})
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
            "kickoffIso": g.get("commence_time"),
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
            "modelVersion": g.get("model_version"),
            "injHome": inj_home,
            "injAway": inj_away,
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
        "schoolAbbr": {sch: abbr for sch, abbr in (
            [(g.get("home_school"), name_abbr.get(g.get("home_team"))) for g in games_all] +
            [(g.get("away_school"), name_abbr.get(g.get("away_team"))) for g in games_all]) if sch and abbr},
        "injuriesAsOf": _fmt_generated_at(data.get("injuries_as_of")) if data.get("injuries_as_of") else None,
        "slateSchools": sorted({x for g in games_all for x in (g.get("home_school"), g.get("away_school")) if x}),
    }

    return {
        "meta": meta,
        "teams": teams,
        "teamDir": build_team_dir(data, teams),
        "games": games,
        "clv": clv_out,
        "propCatalog": props_catalog,
        "propsLive": props_live,
        "fantasy": fantasy_out,
        "backtest": backtest_out,
        "injuries": data.get("injuries") or [],
    }


def _https(url):
    """CFBD hands some logo links back as http://, which a page served over
    https (GitHub Pages) would refuse to load."""
    if not url:
        return None
    url = str(url)
    return "https://" + url[len("http://"):] if url.startswith("http://") else url


def build_team_dir(data: dict, teams: list) -> dict:
    """abbr -> {n: name, l: logo url, p: primary, s: secondary} for the
    tracker's logos (10/2026). A tracked play is stored by abbreviation and
    outlives the slate it was tracked on, so this is keyed by abbreviation
    and built from the widest source available: the export's team_directory
    (every FBS team) when it is there, then this slate's own teams and game
    logos layered on top, since the slate's abbreviations are the exact
    strings plays get tracked with. Before the export has written
    team_directory, only this slate's teams have a logo and every other
    team falls back to the helmet -- nothing is guessed."""
    out = {}
    for t in data.get("team_directory") or []:
        abbr = t.get("abbr")
        if not abbr:
            continue
        out[abbr] = {"n": t.get("school"), "l": _https(t.get("logo")),
                     "p": t.get("primary"), "s": t.get("secondary")}
    logo_by_name = {}
    for g in data.get("games") or []:
        for side in ("home", "away"):
            if g.get(f"{side}_team") and g.get(f"{side}_logo"):
                logo_by_name[g[f"{side}_team"]] = g[f"{side}_logo"]
    for t in teams:
        prev = out.get(t["abbr"]) or {}
        out[t["abbr"]] = {"n": t["name"], "l": _https(logo_by_name.get(t["name"])) or prev.get("l"),
                          "p": t["primary"], "s": t["secondary"]}
    return out


def esc_plain(s):
    return str(s) if s is not None else ""


PROPS_TRACK_PATH = "docs/data/props_clv.json"


def build_prop_tracking(history, path: str = PROPS_TRACK_PATH, max_results: int = 150, max_upcoming: int = 150):
    """The automatic prop log for the My Tracker tab (10/2026): every prop in
    a group that is up in the backtest, logged when first flagged and graded
    from the box score by scripts/compute_props_clv.py. Returns the overall
    record, a record per kind of play (market + side) next to its backtest
    record, the latest graded plays and the upcoming ones. None if the log
    isn't there yet. Never raises."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            log = json.load(f)
        plays = [p for p in (log.get("plays") or []) if p.get("tag") == "group"]
        if not plays:
            return None

        def agg(rows):
            graded = [r for r in rows if r.get("result") in ("W", "L", "P")]
            w = sum(1 for r in graded if r["result"] == "W")
            l = sum(1 for r in graded if r["result"] == "L")
            pu = len(graded) - w - l
            units = sum(float(r.get("units") or 0) for r in graded)
            return {"logged": len(rows), "graded": len(graded), "record": f"{w}-{l}" + (f"-{pu}" if pu else ""),
                    "units": round(units, 2), "roi": round(units / len(graded) * 100, 1) if graded else None,
                    "upcoming": sum(1 for r in rows if not r.get("started"))}

        kinds = {}
        for p in plays:
            kinds.setdefault((p.get("market"), p.get("side")), []).append(p)
        by_kind = []
        for (market, side), rows in kinds.items():
            a = agg(rows)
            a["market"], a["side"] = market, side
            ups = [r for r in ((((history or {}).get("markets") or {}).get(market) or {}).get(side) or []) if r.get("up")]
            if ups:
                bw = sum(int(str(r["record"]).split("-")[0]) for r in ups)
                bl = sum(int(str(r["record"]).split("-")[1]) for r in ups)
                bu, bn = sum(r["units"] for r in ups), sum(r["n"] for r in ups)
                a["backtest"] = {"record": f"{bw}-{bl}", "units": round(bu, 1), "roi": round(bu / bn * 100, 1) if bn else None}
            by_kind.append(a)
        by_kind.sort(key=lambda a: (-a["graded"], -a["logged"]))

        def slim(p):
            return {k: p.get(k) for k in ("player", "team", "opponent", "market", "side", "line", "price", "model_ev",
                                          "group", "start_time", "result", "units", "actual")}

        results = sorted((p for p in plays if p.get("result") in ("W", "L", "P")), key=lambda p: str(p.get("start_time")), reverse=True)
        upcoming = sorted((p for p in plays if not p.get("started")), key=lambda p: str(p.get("start_time")))
        # Official plays' own live record (10/2026), for the Props tab's record line.
        live = log.get("results") or {}
        official = live.get("official")
        # Receptions unders (10/2026): plays and leans, each on its own line.
        return {"asOf": log.get("generated_at"), "overall": agg(plays), "official": official,
                "underPlay": live.get("under_play"), "underLean": live.get("under_lean"), "byKind": by_kind,
                "results": [slim(p) for p in results[:max_results]], "nResults": len(results),
                "upcoming": [slim(p) for p in upcoming[:max_upcoming]], "nUpcoming": len(upcoming)}
    except Exception as e:
        print(f"  [note] could not read {path} ({e}) -- My Tracker will not show the automatic prop log")
        return None


PROP_FLAGS_PATH = "data/clv/prop_flags.csv"


def attach_prop_first_flags(props, path: str = PROP_FLAGS_PATH):
    """Adds first_flag = {at, book, line, price} to each prop: the first time
    the line-movement log (scripts/compute_props_clv.py) recorded the side
    the model leans to. Shown in the Props tab's Details panel as the price
    the play was first logged at. Display only: never raises, and a prop
    that is not in the log simply gets nothing. Returns how many matched."""
    if not props or not os.path.exists(path):
        return 0
    try:
        import csv
        import compute_props_clv as pclv   # same key the log itself is written with

        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        first = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                when, price = r.get("flagged_at") or "", num(r.get("price"))
                if not r.get("key") or not when or price is None:
                    continue
                k = (r["key"], "under" if (r.get("side") or "over") == "under" else "over")
                if k not in first or when < first[k]["at"]:
                    first[k] = {"at": when, "book": r.get("book") or None, "line": num(r.get("line")), "price": price}
        n = 0
        for p in props:
            if p.get("model_lean") not in ("over", "under"):
                continue
            rec = first.get((pclv.prop_key(p), p["model_lean"]))
            if rec:
                p["first_flag"] = rec
                n += 1
        return n
    except Exception as e:
        print(f"  [note] could not read {path} ({e}) -- Props details will not show the first logged price")
        return 0


PROPS_LAB3_PATH = "docs/data/props_lab3.json"


def build_prop_rules(path: str = PROPS_LAB3_PATH):
    """How each receptions RULE did over every season graded in props lab 3
    (this season's saved lines plus the past seasons pulled), for the Props
    tab's Track record column: {'rec_over': official overs, 10% to 30% edge;
    'rec_under': unders at 30%+; 'rec_under_lean': unders at 10% to 30%},
    each {record, units, roi, n, seasons}. None if the lab file is not there.
    Never raises -- a display extra."""
    try:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            res = json.load(f)["results"]["pff"]
        sets = [x for x in [res.get("props_2026")] + list((res.get("props_history") or {}).values()) if x and x.get("n")]

        def total(rows):
            w = l = n = 0
            units = 0.0
            for r in rows:
                if not r or not r.get("n"):
                    continue
                a, b = (int(x) for x in str(r["record"]).split("-")[:2])
                w, l, n, units = w + a, l + b, n + int(r["n"]), units + float(r["units"])
            if not n:
                return None
            return {"record": f"{w}-{l}", "units": round(units, 1), "roi": round(units / n * 100, 1), "n": n, "seasons": len(sets)}

        def unders(x, labels):
            return [b for b in (x.get("under_edge_buckets") or []) if b.get("bucket") in labels]

        out = {"rec_over": total([x.get("rule_under_30_edge") for x in sets]),
               "rec_under": total([b for x in sets for b in unders(x, {"30%+"})]),
               "rec_under_lean": total([b for x in sets for b in unders(x, {"10-14.9%", "15-19.9%", "20-29.9%"})])}
        return {k: v for k, v in out.items() if v} or None
    except Exception as e:
        print(f"  [note] could not read {path} ({e}) -- receptions rules will show the single-season track record")
        return None


def build_prop_edge_history():
    """Backtest record for every market / side / edge size, or None. Never
    raises -- this is a display extra, not something the dashboard should
    fail over."""
    try:
        from src.analysis import prop_edge_history as peh
        return peh.build_history()
    except Exception as e:
        print(f"  [note] could not build the props track record ({e}) -- Props tab will show edges without it")
        return None


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

.cell-market { font-size: 12.5px; color: var(--muted); }
.cell-model  { font-size: 13px; font-weight: 700; }
.cell-edge   { font-family: var(--font-led); font-weight: 900; font-size: 17px; }
.cell-edge.is-pos { color: var(--green); }
.cell-edge.is-neg { color: var(--red); }
.cell-edge.is-off { color: var(--muted-4); }

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
.trk-status-btns { display: flex; gap: 3px; }
.trk-status-btn { border: 1px solid var(--rule); background: var(--chip); color: var(--muted-3); border-radius: 3px; padding: 3px 7px; font-size: 9px; font-weight: 800; cursor: pointer; font-family: var(--font-display); letter-spacing: 0.04em; }
.trk-status-btn.is-win { background: rgba(23,194,107,0.18); color: var(--green); border-color: var(--green); }
.trk-status-btn.is-loss { background: rgba(255,82,82,0.16); color: var(--red); border-color: var(--red); }
.trk-status-btn.is-push { background: rgba(224,180,74,0.16); color: var(--amber); border-color: var(--amber); }
.trk-empty { color: var(--muted-3); font-size: 12px; padding: 26px; text-align: center; border: 1px dashed var(--rule); }
/* Tracker table: search, filters, logos, and rows edited in place (10/2026) */
.trk-tools { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 12px 0 10px; }
.trk-search, .trk-sel, .trk-in { background: var(--chip); border: 1px solid var(--rule-strong); color: var(--text); border-radius: 4px; padding: 7px 10px; font-family: var(--font-data); font-size: 12px; }
.trk-search { flex: 1 1 240px; max-width: 360px; }
.trk-in, .trk-sel, .trk-search { color-scheme: dark; }
.trk-search::placeholder, .trk-in::placeholder { color: var(--muted-3); }
.trk-sel { color: var(--text-dim); cursor: pointer; }
.trk-search:focus-visible, .trk-sel:focus-visible, .trk-in:focus-visible, .trk-chip:focus-visible, .trk-btn:focus-visible, .trk-pill:focus-visible, .trk-icon:focus-visible, .trk-tagbtn:focus-visible, .trk-switch:focus-visible, .trk-status-btn:focus-visible, .trk-linkbtn:focus-visible { outline: 2px solid var(--blue-light); outline-offset: 1px; }
.trk-in.is-bad { border-color: var(--red); }
.trk-msg { color: var(--red); font-size: 10.5px; margin-top: 4px; line-height: 1.35; }
.trk-filtered { font-size: 11.5px; color: var(--text-dim); background: rgba(46,123,255,0.10); border: 1px solid rgba(46,123,255,0.35); border-radius: 4px; padding: 8px 12px; margin: 0 0 10px; }
.trk-linkbtn { background: none; border: none; color: var(--blue-light); cursor: pointer; font: inherit; padding: 0; text-decoration: underline; }
.trk-table { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; margin-top: 1px; }
.trk-g { display: grid; grid-template-columns: 84px minmax(210px, 1.5fr) minmax(128px, 0.7fr) 52px 66px 78px 98px 84px 72px 100px 56px; column-gap: 10px; align-items: center; padding: 10px 14px; }
.trk-h { font-family: var(--font-display); font-weight: 700; font-size: 10.5px; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted-2); border-bottom: 1px solid var(--rule); }
.trk-r { border-bottom: 1px solid var(--rule-row); font-size: 12px; }
.trk-r:last-child { border-bottom: none; }
.trk-r:hover { background: var(--row-active); }
.trk-when { color: var(--text-dim); }
.trk-was { font-size: 10px; color: var(--muted); margin-top: 3px; line-height: 1.35; }
.trk-dim { color: var(--muted-3); }
.trk-dim2 { color: var(--muted); font-size: 11px; }
.trk-neg { color: var(--red); }
.trk-match { display: flex; align-items: center; flex-wrap: wrap; gap: 4px 8px; }
.trk-team { display: inline-flex; align-items: center; gap: 6px; }
.trk-team b { font-family: var(--font-display); font-weight: 800; font-size: 14.5px; letter-spacing: 0.04em; }
.trk-at { color: var(--muted-3); font-size: 10.5px; }
.trk-desc { font-weight: 500; font-size: 12px; }
/* A thin light edge so a dark logo (navy, black) still reads on the dark panel. */
.trk-logo { display: block; flex: none; object-fit: contain; filter: drop-shadow(0 0 0.6px rgba(255,255,255,0.75)); }
.trk-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.trk-tag { font-family: var(--font-display); font-weight: 800; font-size: 9.5px; letter-spacing: 0.07em; text-transform: uppercase; padding: 1px 6px; border-radius: 3px; border: 1px solid var(--rule-strong); color: var(--muted); }
.trk-tag.is-model { color: var(--green); border-color: rgba(23,194,107,0.55); background: rgba(23,194,107,0.10); }
.trk-tag.is-card { color: var(--green); border-color: var(--green); background: rgba(23,194,107,0.16); }
.trk-tag.is-auto { color: var(--blue-light); border-color: rgba(46,123,255,0.5); background: rgba(46,123,255,0.12); }
.trk-tag.is-injury { color: #FF8A8A; border-color: rgba(255,82,82,0.6); background: rgba(255,82,82,0.12); }
.trk-note { font-size: 10.5px; color: var(--muted); margin-top: 5px; max-width: 420px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.trk-chip { display: inline-flex; align-items: center; gap: 7px; background: var(--chip); border: 1px solid var(--rule-strong); color: var(--text); border-radius: 4px; padding: 5px 9px; font-family: var(--font-data); font-size: 12.5px; font-weight: 700; cursor: pointer; white-space: nowrap; }
.trk-chip svg { color: var(--muted-2); flex: none; }
.trk-chip:hover { border-color: var(--blue); }
.trk-chip:hover svg { color: var(--blue-light); }
.trk-chip.is-plain { background: transparent; border-color: transparent; font-weight: 500; padding: 5px 6px; margin-left: -6px; }
.trk-chip.is-plain:hover { background: var(--chip); border-color: var(--blue); }
.trk-chip.is-static { cursor: default; }
.trk-chip.is-static:hover { border-color: var(--rule-strong); }
.trk-editor { display: flex; flex-direction: column; gap: 5px; }
.trk-editor .trk-in { width: 100%; min-width: 0; padding: 5px 8px; border-color: var(--blue); }
.trk-editor-btns { display: flex; gap: 4px; }
.trk-btn { font-family: var(--font-display); font-weight: 700; font-size: 12px; letter-spacing: 0.05em; background: transparent; color: var(--text-dim); border: 1px solid var(--rule-strong); border-radius: 4px; padding: 5px 10px; cursor: pointer; white-space: nowrap; }
.trk-btn:hover { border-color: var(--blue); color: var(--text); }
.trk-btn.is-primary { background: var(--blue); border-color: var(--blue); color: #fff; }
.trk-btn.is-primary:hover { background: var(--blue-light); border-color: var(--blue-light); }
.trk-btn.is-danger { color: #FF8A8A; border-color: rgba(255,82,82,0.45); }
.trk-btn.is-danger:hover { border-color: var(--red); color: #fff; }
.trk-in--stake { width: 62px; padding: 5px 7px; }
.trk-profit { font-weight: 700; }
.trk-pill { font-family: var(--font-display); font-weight: 800; font-size: 10.5px; letter-spacing: 0.08em; border-radius: 3px; padding: 4px 8px; cursor: pointer; background: transparent; white-space: nowrap; }
.trk-pill.is-official { color: var(--green); border: 1px solid rgba(23,194,107,0.6); }
.trk-pill.is-unofficial { color: #FF8A8A; border: 1px solid rgba(255,82,82,0.6); background: rgba(255,82,82,0.10); }
.trk-pill:hover { filter: brightness(1.25); }
.trk-acts { display: flex; align-items: center; justify-content: flex-end; gap: 2px; }
.trk-icon { background: none; border: 1px solid transparent; color: var(--muted); cursor: pointer; border-radius: 4px; width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; font-size: 17px; line-height: 1; padding: 0; }
.trk-icon:hover { color: var(--text); border-color: var(--rule-strong); background: var(--chip); }
.trk-icon.is-x:hover { color: var(--red); }
.trk-dupes { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; margin: 8px 0 4px; padding: 11px 14px; border: 1px solid rgba(224,180,74,0.5); background: rgba(224,180,74,0.08); border-radius: 6px; font-size: 12px; line-height: 1.5; color: var(--text-dim); }
.trk-dupes b { color: var(--amber); font-weight: 700; }
.trk-dupes .trk-btn { padding: 7px 14px; }
.trk-add { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; margin: 4px 0 6px; padding: 12px; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); }
.trk-add .trk-in { min-width: 0; }
.trk-add-hint { flex-basis: 100%; font-size: 10.5px; color: var(--muted); line-height: 1.5; }
.trk-backdrop { position: fixed; inset: 0; background: rgba(4,7,11,0.6); z-index: 80; }
.trk-drawer { position: fixed; top: 0; right: 0; bottom: 0; width: 400px; max-width: 100vw; background: var(--panel-deep); border-left: 1px solid var(--rule-strong); z-index: 81; display: flex; flex-direction: column; box-shadow: -18px 0 40px rgba(0,0,0,0.45); }
.trk-d-head { display: flex; align-items: center; justify-content: space-between; padding: 16px 18px 12px; border-bottom: 1px solid var(--rule); }
.trk-d-head h3 { font-family: var(--font-display); font-weight: 800; font-size: 19px; letter-spacing: 0.05em; text-transform: uppercase; margin: 0; }
.trk-d-body { flex: 1; overflow-y: auto; padding: 16px 18px; display: flex; flex-direction: column; gap: 14px; }
.trk-d-game { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; }
.trk-d-game .trk-team b { font-size: 18px; }
.trk-d-field { display: flex; flex-direction: column; gap: 6px; }
.trk-d-field > span { font-family: var(--font-display); font-weight: 700; font-size: 11.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.trk-d-field .trk-in, .trk-d-field .trk-sel { width: 100%; font-size: 13px; padding: 8px 10px; }
.trk-d-field textarea.trk-in { resize: vertical; min-height: 84px; line-height: 1.5; }
.trk-d-two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.trk-d-preview { font-size: 12.5px; line-height: 1.5; }
.trk-d-preview:empty { display: none; }
.trk-d-preview b { font-size: 14px; }
.trk-d-count { font-size: 10px; color: var(--muted-3); text-align: right; }
.trk-tagrow { display: flex; flex-wrap: wrap; gap: 6px; }
.trk-tagbtn { font-family: var(--font-display); font-weight: 700; font-size: 12.5px; letter-spacing: 0.04em; background: var(--chip); color: var(--muted); border: 1px solid var(--rule-strong); border-radius: 4px; padding: 5px 11px; cursor: pointer; }
.trk-tagbtn:hover { color: var(--text); }
.trk-tagbtn.is-on { background: rgba(46,123,255,0.16); color: var(--blue-light); border-color: var(--blue); }
.trk-tagbtn.is-on.is-injury { background: rgba(255,82,82,0.14); color: #FF8A8A; border-color: var(--red); }
.trk-d-addtag { display: flex; gap: 6px; }
.trk-switchrow { display: flex; align-items: center; gap: 10px; }
.trk-switch { position: relative; width: 38px; height: 22px; border-radius: 11px; border: 1px solid rgba(255,82,82,0.6); background: rgba(255,82,82,0.22); cursor: pointer; padding: 0; flex: none; }
.trk-switch-knob { position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: #FF8A8A; transition: left 0.12s; }
.trk-switch.is-plain:not(.is-on) { border-color: var(--rule-strong); background: var(--chip); }
.trk-switch.is-plain:not(.is-on) .trk-switch-knob { background: var(--muted-2); }
.trk-switch.is-on { border-color: rgba(23,194,107,0.7); background: rgba(23,194,107,0.22); }
.trk-switch.is-on .trk-switch-knob { left: 18px; background: var(--green); }
.trk-switch-label { font-size: 12px; color: var(--text-dim); }
.trk-d-foot { display: flex; align-items: center; gap: 8px; padding: 12px 18px; border-top: 1px solid var(--rule); background: var(--panel); }
.trk-d-foot .trk-btn { padding: 8px 14px; font-size: 13px; }
@media (max-width: 1180px) {
  .trk-table { overflow-x: auto; }
  .trk-g { min-width: 1080px; }
}
@media (prefers-reduced-motion: reduce) { .trk-switch-knob { transition: none; } }
.trk-summary-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--rule); margin-bottom: 1px; }
.unit-size-input { width: 108px; background: var(--chip); border: 1px solid var(--rule); color: var(--text); border-radius: 3px; padding: 6px 9px; font-family: var(--font-data); font-size: 11px; }

.note-block { font-size: 10.5px; color: var(--muted-3); margin-top: 14px; line-height: 1.7; border-top: 1px solid var(--rule-faint); padding-top: 12px; }
.empty-state { background: var(--panel); border: 1px dashed var(--rule); border-radius: 4px; padding: 20px; text-align: center; color: var(--muted-3); font-size: 12px; }

.main { display: grid; grid-template-columns: 1.3fr 1fr; gap: 26px; margin-top: 32px; align-items: start; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; margin-top: 44px; }
.mt-lg { margin-top: 44px; }
.mt-md { margin-top: 38px; }
.footer { margin-top: 48px; border-top: 1px solid var(--rule); padding-top: 16px; display: flex; justify-content: space-between; font-size: 9.5px; color: var(--muted-4); flex-wrap: wrap; gap: 8px; }

/* ---- Player Props tab, simplified 10/2026: a one-line live record, then
   the plays in one table (official plays first, by kickoff), a Details
   panel per row, then where the model has been right and the latest graded
   results. ---- */
.pp-recline { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 28px; margin: 14px 0 18px; padding: 12px 16px; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; }
.pp-rec { display: inline-flex; flex-wrap: wrap; align-items: baseline; gap: 4px 9px; font-size: 13.5px; }
.pp-rec b { font-weight: 700; white-space: nowrap; }
.pp-rec-label { font-family: var(--font-display); font-weight: 700; font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.pp-rec-n { font-size: 11.5px; color: var(--muted); white-space: nowrap; }
.pp-key { margin-left: auto; font-size: 11.5px; line-height: 1.55; color: var(--muted); max-width: 560px; }
.pp-key b { color: var(--text-dim); font-weight: 500; }
.pp-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 0 0 10px; }
.pp-seg { display: inline-flex; border: 1px solid var(--rule-strong); border-radius: 5px; overflow: hidden; }
.pp-seg button { font-family: var(--font-display); font-weight: 700; font-size: 13px; letter-spacing: 0.04em; background: transparent; color: var(--muted); border: none; padding: 7px 14px; cursor: pointer; }
.pp-seg button + button { border-left: 1px solid var(--rule-strong); }
.pp-seg button.is-on { background: var(--blue); color: #fff; }
.pp-seg button:focus-visible, .pp-chip:focus-visible, .pp-trk:focus-visible, .pp-link:focus-visible { outline: 2px solid var(--blue-light); outline-offset: 2px; }
.pp-select { background: var(--chip); color: var(--text-dim); border: 1px solid var(--rule-strong); border-radius: 5px; padding: 7px 10px; font-family: var(--font-data); font-size: 11.5px; }
.pp-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 14px; }
.pp-chip { font-family: var(--font-display); font-weight: 700; font-size: 12.5px; letter-spacing: 0.03em; background: var(--chip); color: var(--muted); border: 1px solid transparent; border-radius: 5px; padding: 5px 12px; cursor: pointer; }
.pp-chip.is-on { background: rgba(46,123,255,0.16); color: var(--blue-light); border-color: var(--blue); }
.pp-table { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; overflow-x: auto; }
.pp-grid { display: grid; grid-template-columns: minmax(190px, 1.4fr) minmax(190px, 1.2fr) minmax(150px, 0.9fr) minmax(170px, 1.1fr) minmax(120px, 0.8fr) 190px; align-items: center; column-gap: 16px; min-width: 1040px; }
.pp-head { font-family: var(--font-display); font-weight: 700; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted-2); padding: 11px 18px; border-bottom: 1px solid var(--rule); }
.pp-group { display: flex; align-items: baseline; gap: 10px; padding: 12px 18px 9px; background: var(--panel-deep); border-bottom: 1px solid var(--rule); border-top: 1px solid var(--rule); min-width: 1040px; }
.pp-table > .pp-group:first-of-type { border-top: none; }
.pp-group-name { font-family: var(--font-display); font-weight: 800; font-size: 15px; letter-spacing: 0.05em; text-transform: uppercase; }
.pp-group-count { font-family: var(--font-data); font-size: 11px; color: var(--muted); background: var(--chip); border-radius: 10px; padding: 1px 8px; }
.pp-group-note { font-size: 10.5px; color: var(--muted-3); }
.pp-row { padding: 12px 18px; border-bottom: 1px solid var(--rule-row); font-size: 13px; }
.pp-row.is-open { background: var(--row-active); }
.pp-row:hover { background: var(--row-active); }
.pp-name { font-family: var(--font-display); font-weight: 700; font-size: 17.5px; letter-spacing: 0.01em; color: var(--text); }
.pp-pick { font-weight: 700; font-size: 13.5px; color: var(--text); }
.pp-pick b { color: var(--blue-light); font-weight: 700; }
.pp-sub { font-size: 11.5px; color: var(--muted); margin-top: 4px; }
.pp-big { font-weight: 700; font-size: 14px; }
.pp-edge { font-weight: 700; font-size: 18px; line-height: 1.1; }
.pp-track { font-weight: 500; font-size: 13.5px; }
.pp-track.is-none { color: var(--muted); }
.pp-flags { display: flex; flex-wrap: wrap; gap: 6px; }
.pp-actions { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
.pp-detail { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px 24px; padding: 14px 18px 16px; background: var(--panel-deep); border-bottom: 1px solid var(--rule-row); position: sticky; left: 0; }
.pp-dcell.is-wide { grid-column: 1 / -1; }
.pp-dlabel { font-family: var(--font-display); font-weight: 700; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.pp-dval { font-size: 14px; font-weight: 500; color: var(--text); margin-top: 5px; }
.pp-gls { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.pp-gl { display: inline-flex; flex-direction: column; align-items: center; min-width: 46px; padding: 5px 8px; border: 1px solid var(--rule-strong); border-radius: 4px; }
.pp-gl b { font-size: 14px; font-weight: 700; color: var(--text-dim); }
.pp-gl i { font-style: normal; font-size: 10px; color: var(--muted); margin-top: 2px; }
.pp-gl.is-hit { border-color: rgba(23,194,107,0.55); background: rgba(23,194,107,0.10); }
.pp-gl.is-hit b { color: var(--green); }
.pp-wi { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; margin-top: 8px; }
.pp-wi-field { display: inline-flex; align-items: center; gap: 7px; font-size: 11.5px; color: var(--muted); }
.pp-wi-input { width: 88px; background: var(--chip); color: var(--text); border: 1px solid var(--rule-strong); border-radius: 5px; padding: 7px 9px; font-family: var(--font-data); font-size: 13px; }
.pp-wi-input:focus-visible { outline: 2px solid var(--blue-light); outline-offset: 1px; }
.pp-wi-out { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; font-size: 13px; color: var(--text); }
.pp-wi-main b { font-weight: 700; }
.pp-wi-edge { font-weight: 700; font-size: 15px; }
.pp-wi-note { font-size: 11.5px; color: var(--muted); }
.pp-wi-posted { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 10px; }
.pp-more { padding: 10px 18px; border-bottom: 1px solid var(--rule-row); min-width: 1040px; }
.pp-more .pp-link { margin-top: 0; }
.pp-tag { font-family: var(--font-display); font-weight: 800; font-size: 11px; letter-spacing: 0.07em; padding: 3px 9px; border-radius: 4px; }
.pp-tag.is-official { background: rgba(23,194,107,0.16); color: var(--green); }
.pp-tag.is-watch { background: rgba(91,164,255,0.16); color: var(--blue-light); }
.pp-tag.is-track { background: transparent; color: var(--green); border: 1px solid rgba(23,194,107,0.4); }
.pp-tag.is-lean { background: var(--chip); color: var(--muted); }
.pp-tag.is-inj { background: rgba(224,180,74,0.16); color: var(--amber); }
.pp-tag.is-pass { background: rgba(255,82,82,0.14); color: var(--salmon); }
.pp-trk { font-family: var(--font-display); font-weight: 700; font-size: 11.5px; letter-spacing: 0.05em; background: transparent; color: var(--text-dim); border: 1px solid var(--rule-strong); border-radius: 4px; padding: 7px 12px; cursor: pointer; white-space: nowrap; }
.pp-trk:hover { border-color: var(--blue); color: var(--blue-light); }
.pp-inlist { font-family: var(--font-display); font-weight: 700; font-size: 11.5px; letter-spacing: 0.04em; color: var(--blue-light); white-space: nowrap; padding: 0 4px; }
.pp-lower { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; margin-top: 26px; align-items: start; }
.pp-panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 16px 18px 14px; }
.pp-panel h3 { font-family: var(--font-display); font-weight: 800; font-size: 17px; letter-spacing: 0.05em; text-transform: uppercase; margin: 0; }
.pp-panel-note { font-size: 10.5px; color: var(--muted-3); margin: 5px 0 12px; line-height: 1.5; }
.pp-buckets { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
.pp-bucket { background: var(--panel-deep); border: 1px solid var(--rule-faint); border-radius: 5px; padding: 11px 13px; }
.pp-bucket-name { font-family: var(--font-display); font-weight: 700; font-size: 14.5px; letter-spacing: 0.02em; }
.pp-bucket-roi { font-family: var(--font-led); font-weight: 900; font-size: 20px; color: var(--green); margin-top: 5px; }
.pp-res { display: grid; grid-template-columns: minmax(0, 1.6fr) 70px 62px 64px; align-items: center; column-gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--rule-row); font-size: 12px; }
.pp-res:last-of-type { border-bottom: none; }
.pp-res-name { font-family: var(--font-display); font-weight: 700; font-size: 14.5px; }
.pp-result { font-family: var(--font-display); font-weight: 800; font-size: 11px; letter-spacing: 0.07em; padding: 2px 8px; border-radius: 4px; border: 1px solid; text-align: center; }
.pp-link { font-family: var(--font-display); font-weight: 700; font-size: 12.5px; letter-spacing: 0.04em; background: transparent; color: var(--blue-light); border: 1px solid var(--rule-strong); border-radius: 4px; padding: 6px 12px; cursor: pointer; margin-top: 12px; }
.pp-link:hover { border-color: var(--blue); }
.pp-help { margin-top: 18px; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); }
.pp-help summary { cursor: pointer; padding: 12px 18px; font-family: var(--font-display); font-weight: 700; font-size: 14px; letter-spacing: 0.04em; color: var(--text-dim); }
.pp-help-body { padding: 0 18px 16px; font-size: 11.5px; line-height: 1.65; color: var(--muted); max-width: 980px; }
.pp-help-body p { margin: 0 0 9px; }
.pp-help-body b { color: var(--text-dim); }
/* Edge Board rows + Projector header (rebuilt 10/2026) */
.eb-grid { grid-template-columns: minmax(186px, 1fr) 56px 56px 74px 48px 138px 48px; column-gap: 8px; }
.eb-tools { margin: 12px 0 2px; }
.eb-count { margin-left: auto; font-size: 11px; color: var(--muted); white-space: nowrap; }
.eb-match { min-width: 0; }
.eb-match .matchup { gap: 7px; }
.eb-src { font-family: var(--font-display); font-weight: 700; letter-spacing: .05em; color: #8A94A3; }
.eb-src.is-trained { color: #2ecc71; }
.eb-badges { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 7px; }
.eb-inj { font-family: var(--font-display); font-weight: 800; font-size: 10.5px; letter-spacing: 0.06em; padding: 1px 6px; border-radius: 3px; border: 1px solid rgba(224,180,74,0.55); color: var(--amber); background: rgba(224,180,74,0.08); cursor: help; white-space: nowrap; }
.eb-inj.is-out { border-color: rgba(255,82,82,0.6); color: #FF8A8A; background: rgba(255,82,82,0.10); }
.eb-ring { position: relative; width: 44px; height: 44px; margin-left: auto; }
.eb-ring svg { display: block; }
.eb-ring span { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 9.5px; font-weight: 700; letter-spacing: -0.03em; color: var(--text-dim); }
.eb-ring-track { stroke: var(--rule-strong); }
.eb-ring-val { stroke: #54657A; }
.eb-ring.is-play .eb-ring-val { stroke: var(--blue-light); }
.eb-ring.is-card .eb-ring-val { stroke: var(--green); }
.eb-ring.is-card span, .eb-ring.is-play span { color: var(--text); }
.eb-play { min-width: 0; }
.eb-noplay { display: block; text-align: center; color: var(--muted-4); }
.eb-pill { border: 1px solid; border-radius: 5px; padding: 5px 6px 6px; text-align: center; line-height: 1.15; }
.eb-pill-top { font-family: var(--font-display); font-weight: 800; font-size: 10.5px; letter-spacing: 0.1em; white-space: nowrap; }
.eb-pill-pick { font-family: var(--font-display); font-weight: 700; font-size: 14.5px; letter-spacing: 0.02em; margin-top: 3px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.eb-pill-pick span { font-weight: 500; color: var(--text-dim); }
.eb-pill-sub { font-size: 9.5px; margin-top: 3px; white-space: nowrap; }
.eb-pill.is-card { background: rgba(23,194,107,0.13); border-color: var(--green); color: var(--green); }
.eb-pill.is-track { background: rgba(46,123,255,0.09); border-color: rgba(91,164,255,0.55); color: var(--blue-light); }
.eb-pill.is-warn { background: rgba(224,180,74,0.07); border-color: rgba(224,180,74,0.45); color: var(--amber); }
.eb-pill.is-warn .eb-pill-pick, .eb-pill.is-track .eb-pill-pick { color: var(--text-dim); }
.proj-side { display: flex; flex-direction: column; align-items: center; gap: 6px; flex: none; width: 132px; }
.proj-team { font-family: var(--font-display); font-weight: 800; font-size: 27px; letter-spacing: 0.02em; text-transform: uppercase; line-height: 1; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6); }
.proj-school { font-size: 10px; color: rgba(255, 255, 255, 0.7); text-align: center; line-height: 1.3; }
.proj-mid { flex: 1; min-width: 0; text-align: center; }
.proj-at { font-family: var(--font-display); font-weight: 800; font-size: 13px; letter-spacing: 0.18em; color: rgba(255, 255, 255, 0.5); }
.proj-when { font-family: var(--font-display); font-weight: 700; font-size: 17px; letter-spacing: 0.03em; margin-top: 4px; text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6); }
.proj-flags { display: flex; flex-wrap: wrap; justify-content: center; gap: 5px; margin-top: 9px; }
.proj-flags .flag-chip { margin-left: 0; }
.cmp-row { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; font-size: 12px; padding: 3px 0; }
.cmp-row span { color: var(--muted); font-size: 11px; }
.cmp-row b { font-weight: 700; text-align: right; }
.cmp-row i { font-style: normal; font-weight: 400; color: var(--muted); font-size: 11px; margin-left: 4px; }
.cmp-row.is-gap { border-top: 1px solid var(--rule-faint); margin-top: 4px; padding-top: 7px; }
.cmp-row.is-gap b { color: var(--blue-light); }
.card-play { display: flex; align-items: center; gap: 8px; }
.eb-match .meta { flex-wrap: wrap; row-gap: 2px; overflow: visible; }
.eb-match .meta span { white-space: nowrap; }
/* The board needs about 690px; below this the two columns stack rather than squeeze it. */
@media (max-width: 1320px) { .main { grid-template-columns: 1fr; } }
@media (max-width: 1100px) {
  .pp-key { margin-left: 0; }
  .pp-lower { grid-template-columns: 1fr; }
}
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

  // Null-safe: a game can have a moneyline posted before its spread is up
  // (9/2026) -- formatting a missing number used to throw and blank the
  // entire page, not just that one row.
  function signed(v, dp) {
    if (v == null || isNaN(v)) return '\\u2014';
    return (v > 0 ? '+' : '') + v.toFixed(dp === undefined ? 1 : dp);
  }
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
    } else if (game.marketSpread == null || isNaN(game.marketSpread)) {
      // No spread posted yet -- show the model's number, but there's no
      // market line to have an edge against, so this can't qualify.
      marketLabel = '\\u2014';
      modelLabel  = signed(game.modelSpread);
      edge        = 0;
      coverProb   = 0.5;
      side        = null;
      playLabel   = 'No spread posted';
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

  var state = { selected: 0, market: 'Moneyline', search: '', page: 'edge', edgeSort: 'kickoff', edgeShow: 'all', edgeConf: 'ALL', propMarket: 'ALL', unitSize: 50, trackerModelTab: 'all', injScope: 'slate', propGame: 'ALL', propOfficialOnly: false, propShowPass: false, propView: 'official', propDetail: null };

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
  // Near-even moneyline target (added 10/2026): a qualifying moneyline at
  // +100 to +150 where the model gives that team 65%+ to win. Across the
  // first three trained-model weekends these went 5-2 (+4.1u) while
  // lower-confidence near-even plays went 2-5 -- promising but a small
  // sample, so it's tagged for small stakes and tracking, not a core bet.
  function isNearEvenTarget(p) {
    return p && p.market === 'Moneyline' && p.qualifies && p.sideMoneyline != null &&
      p.sideMoneyline >= 100 && p.sideMoneyline <= 150 && p.sideProb >= 0.65;
  }
  function schoolAbbr(s) { return s ? ((D.meta.schoolAbbr || {})[s] || s) : ''; }
  function playerLabel(r) { return (r.team ? schoolAbbr(r.team) + ' ' : '') + r.player_name; }

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
    var t = team(teamName);
    return helmetSvg(t.primary, t.secondary, width, height, flip);
  }
  // Same helmet from a pair of colors, for a team that isn't on this
  // week's slate (the tracker's logo fallback -- see trkMark).
  function helmetSvg(primary, secondary, width, height, flip) {
    var p = esc(M.displayColor(primary)), s = esc(secondary || '#E7EDF5');
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
    { id: 'injuries', label: 'Injury Report' },
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

  function renderKpis(card, priced) {
    // "Qualifying edges" used to show card.length, which is capped at the
    // Bet Card's ten -- it read "10" however many plays cleared the bar.
    var flagged = (priced || []).filter(function (p) { return p.qualifies; }).length;
    var k = [
      ['Games priced', String(D.meta.gamesPriced), D.meta.totalGames + ' total in slate \\u00b7 ' + (D.meta.totalGames - D.meta.gamesPriced) + ' unpriced (no FBS SP+ rating or no market line)', false],
      ['Qualifying edges', String(flagged), card.length + ' on the Bet Card \\u00b7 \\u2265 ' + D.meta.minEdge.toFixed(1) + ' pt threshold, ' + state.market + ' market', false],
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

  function trackPayload(p, onCard) {
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
    // The tracker's official record is Bet Card spreads and nothing else
    // (10/2026, the rule the user set on 10/5: "we're only officially
    // tracking spreads", wager only the Bet Card). Until now +TRK marked
    // every play official unless it was a 20+ point spread, so moneylines
    // and off-the-card spreads all counted in the record unless they were
    // flipped by hand. Now a play is tracked as official only when it is a
    // spread AND on the spread Bet Card at the moment it is tracked;
    // everything else -- every moneyline (favorites, near-even targets and
    // underdogs alike), off-the-card spreads, 20+ point spreads -- goes in
    // as unofficial: still listed and graded, just not in the record. The
    // row's Official pill still flips any single play either way.
    // betCard travels with the play, so the tracker can show the Bet Card's
    // own record later without anyone having to remember which plays it was.
    var betCard = !!onCard && p.qualifies;
    var autoUnofficial = !(p.market === 'Spread' && betCard && lineSize < 20);
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
      date: g.kickoff, kickoffIso: g.kickoffIso || null, type: p.market,
      price: p.market === 'Moneyline' ? p.sideMoneyline : -110,
      edge: Math.round(Math.abs(p.edge) * 10) / 10,
      unofficial: autoUnofficial,
      betCard: betCard,
      modelSource: modelSource,
      modelVersion: g.modelVersion || null
    }));
  }

  /* ---- Edge Board (rows rebuilt 10/2026) ------------------------------
     Same numbers as before -- every cell still comes straight from
     priceGame() -- laid out so they can be read at a glance:
       - school logos (helmet when there isn't one on file);
       - the play in its own column instead of pills stacked under the
         matchup, labelled by what to DO with it: BET CARD (wager), TRACK
         ONLY, UNOFFICIAL, FADE. The label comes from playStatusOf(), the
         same call the Projector makes, so the two can't disagree;
       - win % as a ring; QB injuries as one short badge (hover for who);
       - kickoff in your own time zone, like the Props tab and the tracker;
       - sort and filters above the table. */
  var EDGE_SORTS = [['kickoff', 'Sort: kickoff'], ['edge', 'Sort: biggest edge'], ['prob', 'Sort: highest win %']];
  var EDGE_SHOWS = [['all', 'All games'], ['card', 'Bet Card plays'], ['plays', 'All flagged plays']];

  function edgeBook(g) { return PROP_BOOK_LABELS[String(g.book || '').toLowerCase()] || g.book || ''; }
  function edgeKickoff(g) { return propKickoff(g.kickoffIso) || g.kickoff; }
  function edgeOnCard(s) { return s.kind === 'card' || (s.kind === 'target' && s.onCard); }
  function edgeFiltering() { return state.edgeShow !== 'all' || state.edgeConf !== 'ALL'; }

  function edgeRing(prob, tone) {
    var c = 2 * Math.PI * 17, pct = Math.max(0, Math.min(1, prob)), txt = (pct * 100).toFixed(1) + '%';
    return '<div class="eb-ring' + (tone ? ' ' + tone : '') + '" role="img" aria-label="' + txt + '">' +
      '<svg viewBox="0 0 44 44" width="44" height="44" aria-hidden="true">' +
        '<circle class="eb-ring-track" cx="22" cy="22" r="17" fill="none" stroke-width="4"/>' +
        '<circle class="eb-ring-val" cx="22" cy="22" r="17" fill="none" stroke-width="4" stroke-linecap="round" ' +
          'stroke-dasharray="' + (c * pct).toFixed(2) + ' ' + c.toFixed(2) + '" transform="rotate(-90 22 22)"/>' +
      '</svg><span>' + txt + '</span></div>';
  }
  // "QB L. Kienholz (Questionable)" on Louisville -> "LOU QB QUESTIONABLE",
  // with the full line on hover. Out is red, anything short of it amber.
  function edgeInjuryBadge(f) {
    var m = /\\(([^)]+)\\)\\s*$/.exec(f.text || ''), status = m ? m[1] : '';
    return '<span class="eb-inj' + (/^out/i.test(status) ? ' is-out' : '') + '" title="' + esc(abbrOf(f.team) + ' ' + f.text) + '">⚠ ' +
      esc(abbrOf(f.team)) + ' QB' + (status ? ' ' + esc(status.toUpperCase()) : '') + '</span>';
  }
  function edgePlayPill(s) {
    var top, cls, title = s.note, extra = '';
    if (edgeOnCard(s)) {
      top = 'BET CARD'; cls = 'is-card'; title = 'On the Bet Card — wager';
      if (s.kind === 'target') { extra = '<div class="eb-pill-sub">★ near-even · small stake</div>'; title = 'On the Bet Card. Near-even target: ' + s.note; }
    } else if (s.kind === 'target') { top = 'NEAR-EVEN TARGET'; cls = 'is-track'; }
    else if (s.kind === 'official') { top = 'TRACK ONLY'; cls = 'is-track'; title = 'Official play, not on the Bet Card — track only'; }
    else if (s.kind === 'bigdog') { top = 'UNOFFICIAL DOG'; cls = 'is-warn'; title = 'Underdog moneyline longer than +150 — unofficial, track only'; }
    else if (s.kind === 'bigspread') { top = 'FADE · 20+ PT'; cls = 'is-warn'; title = '20+ point spread — faded by rule'; }
    else if (s.kind === 'fade') { top = 'FADE'; cls = 'is-warn'; }
    else return '<span class="eb-noplay" title="' + esc(s.kind === 'nospread' ? 'No spread posted' : 'No play: ' + s.note) + '">—</span>';
    return '<div class="eb-pill ' + cls + '" title="' + esc(title) + '"><div class="eb-pill-top">' + top + '</div>' +
      '<div class="eb-pill-pick">' + esc(s.playText) + (cls === 'is-card' ? ' <span>· ' + esc(s.p.tier) + '</span>' : '') + '</div>' + extra + '</div>';
  }

  function renderEdgeBoard(priced, card) {
    var all = priced.map(function (p, i) {
      return { p: p, i: i, s: playStatusOf(p, card.some(function (c) { return c.game === p.game; })) };
    });
    var confs = {};
    all.forEach(function (r) { [team(r.p.game.away).conf, team(r.p.game.home).conf].forEach(function (c) { if (c) confs[c] = true; }); });
    if (state.edgeConf !== 'ALL' && !confs[state.edgeConf]) state.edgeConf = 'ALL';
    var visible = all.filter(function (r) {
      var g = r.p.game;
      if (!matchesSearch(g)) return false;
      if (state.edgeShow === 'card' && !edgeOnCard(r.s)) return false;
      if (state.edgeShow === 'plays' && !r.p.qualifies) return false;
      if (state.edgeConf !== 'ALL' && team(g.away).conf !== state.edgeConf && team(g.home).conf !== state.edgeConf) return false;
      return true;
    });
    // The slate arrives in kickoff order, so "kickoff" is simply the order
    // it came in; the other two sort on the picked side's own number.
    if (state.edgeSort === 'edge') visible.sort(function (a, b) { return (b.p.sideEdge - a.p.sideEdge) || (a.i - b.i); });
    else if (state.edgeSort === 'prob') visible.sort(function (a, b) { return (b.p.sideProb - a.p.sideProb) || (a.i - b.i); });

    var opt = function (v, label, cur) { return '<option value="' + esc(v) + '"' + (cur === v ? ' selected' : '') + '>' + esc(label) + '</option>'; };
    var head = '' +
      '<div class="section-head">' +
        '<div class="section-title"><div class="section-flag"></div><h2>Edge Board</h2></div>' +
        '<div class="tabs">' + MARKETS.map(function (m) {
          return '<button class="tab tab--market' + (m === state.market ? ' is-active' : '') + '" data-market="' + esc(m) + '"><span>' + esc(m) + '</span></button>';
        }).join('') + '</div>' +
      '</div>' +
      '<div class="trk-tools eb-tools">' +
        '<select class="trk-sel" data-edge-filter="edgeShow" aria-label="Which games to show">' + EDGE_SHOWS.map(function (o) { return opt(o[0], o[1], state.edgeShow); }).join('') + '</select>' +
        '<select class="trk-sel" data-edge-filter="edgeConf" aria-label="Filter by conference">' + opt('ALL', 'All conferences', state.edgeConf) +
          Object.keys(confs).sort().map(function (c) { return opt(c, c, state.edgeConf); }).join('') + '</select>' +
        '<select class="trk-sel" data-edge-filter="edgeSort" aria-label="Sort games">' + EDGE_SORTS.map(function (o) { return opt(o[0], o[1], state.edgeSort); }).join('') + '</select>' +
        '<span class="eb-count">' + visible.length + ' of ' + all.length + ' games' +
          (edgeFiltering() || state.search ? ' · <button class="trk-linkbtn" data-edge-clear="1">Show all</button>' : '') + '</span>' +
      '</div>' +
      '<div class="thead eb-grid"><div>Matchup</div><div class="num" title="Home team’s number">Market</div><div class="num" title="Home team’s number">Model</div>' +
      '<div class="num">Edge</div><div class="num">Win %</div><div style="text-align:center">Play</div><div></div></div>';

    var rows = visible.map(function (r) {
      var p = r.p, s = r.s, i = r.i, g = p.game, a = team(g.away), h = team(g.home);
      // cls reflects "is this row actually good/bad," not raw home-team
      // sign: is-off below the edge threshold or price-cap-excluded, is-neg
      // for a confirmed-negative-EV FADE, is-pos for an actual qualifying
      // play. (See sideEdge/sideEdgeLabel in priceGame for why the Edge
      // column text itself also switched off raw p.edge.)
      var cls = Math.abs(p.edgeForTier) < D.meta.minEdge ? 'is-off'
        : p.isFade ? 'is-neg'
        : p.qualifies ? 'is-pos'
        : 'is-off';
      // Which model priced this game -- shown on EVERY row (not just
      // PLAY/FADE ones), since it's a property of the game itself, not the
      // pick. Same flag the tracker badge/priceGame's untrustedUnderdog
      // check already use elsewhere, so this can never drift out of sync
      // with what actually generated the line. Added 9/19/2026 -- the user
      // wanted this visible on the Edge Board directly, not just after
      // tracking a play.
      var isTrainedGame = (g.flags || []).some(function (f) { return f.text === 'In-season model'; });
      var modelSrcLabel = isTrainedGame ? (g.modelVersion === 'upgraded' ? 'UPGRADED' : 'TRAINED') : 'UNTRAINED';
      var injuries = (g.flags || []).filter(function (f) { return f.qb; }).map(edgeInjuryBadge).join('');
      return '' +
        '<div class="row row--click' + (i === state.selected ? ' is-selected' : '') + '" data-game="' + i + '">' +
          '<div class="row-accent" style="background:linear-gradient(' + esc(M.displayColor(a.primary)) + ',' + esc(M.displayColor(h.primary)) + ')"></div>' +
          '<div class="row-body eb-grid">' +
            '<div class="eb-match"><div class="matchup">' +
              trkMark(a.abbr, 26) +
              '<span class="team-abbr" title="' + esc(g.away) + '">' + esc(a.abbr) + '</span>' +
              '<span class="at">AT</span>' +
              trkMark(h.abbr, 26) +
              '<span class="team-abbr" title="' + esc(g.home) + '">' + esc(h.abbr) + '</span>' +
            '</div>' +
            '<div class="meta"><span>' + esc(edgeKickoff(g)) + '</span><span>' + esc(edgeBook(g)) + '</span>' +
              '<span class="eb-src' + (isTrainedGame ? ' is-trained' : '') + '">' + modelSrcLabel + '</span></div>' +
            (injuries ? '<div class="eb-badges">' + injuries + '</div>' : '') +
            '</div>' +
            '<div class="num cell-market">' + esc(p.marketLabel) + '</div>' +
            '<div class="num cell-model">' + esc(p.modelLabel) + '</div>' +
            '<div class="num cell-edge ' + cls + '">' + esc(p.sideEdgeLabel) + '</div>' +
            '<div>' + edgeRing(p.sideProb, edgeOnCard(s) ? 'is-card' : (p.qualifies ? 'is-play' : '')) + '</div>' +
            '<div class="eb-play">' + edgePlayPill(s) + '</div>' +
            '<div class="num"><button class="track-btn" onclick="event.stopPropagation();window.__cfbTrack(' + trackPayload(p, s.onCard) + ')">+TRK</button></div>' +
          '</div>' +
        '</div>';
    }).join('');

    var foot = '<div class="table-foot"><span>Edge stated in points of expected value against the posted number. ' +
      'Threshold ' + D.meta.minEdge.toFixed(1) + ' pts. Market and Model are the home team’s number; Edge and Win % are for the side the model takes.</span>' +
      '<span>' + card.length + ' on the Bet Card · ' + visible.length + ' shown</span></div>';

    return head + (visible.length ? rows : '<div class="empty-state">No games match. <button class="trk-linkbtn" data-edge-clear="1">Show all</button></div>') + foot;
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

  // ---- Injury Report page (added 9/2026) ----
  // Source: Covers.com's free NCAAF injury page, pulled on every pipeline
  // run (src/data/injury_report.py). Entries come from team and media
  // reports, not official conference filings.
  var INJ_ORDER = { 'Out For Season': 0, 'IR': 0, 'Out': 1, 'Doubtful': 2, 'Questionable': 3, 'Game-Time Decision': 3, 'Day-To-Day': 4, 'Probable': 5 };
  function injColor(st) {
    if (st === 'Out' || st === 'Out For Season' || st === 'IR' || st === 'Doubtful') return 'var(--red)';
    if (st === 'Probable') return 'var(--green)';
    return 'var(--amber)';
  }
  function renderInjuries() {
    var all = D.injuries || [];
    var head = '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Injury Report</h2></div>' +
      '<div style="font-size:10px;color:var(--muted-3)">Source: Covers.com (team &amp; media reports, not official filings)' +
      (D.meta.injuriesAsOf ? ' \\u00b7 as of ' + esc(D.meta.injuriesAsOf) : '') + '</div></div>';
    if (!all.length) {
      return head + '<div class="empty-state">No injury data this run \\u2014 the injury page couldn\\u2019t be read. ' +
        'Picks are not being filtered for injuries until the next successful run.</div>';
    }
    var slate = {};
    (D.meta.slateSchools || []).forEach(function (x) { slate[x] = true; });
    var scope = state.injScope || 'slate';
    var rows = scope === 'slate' ? all.filter(function (i) { return slate[i.school]; }) : all;
    var tabs = '<div class="prop-tabs" style="margin:12px 0 14px">' +
      [['slate', "This week's teams"], ['all', 'All teams']].map(function (t) {
        return '<button class="tab tab--prop' + (scope === t[0] ? ' is-active' : '') + '" data-inj-scope="' + t[0] + '"><span>' + t[1] + '</span></button>';
      }).join('') + '</div>';
    var byTeam = {};
    rows.forEach(function (i) { (byTeam[i.school] = byTeam[i.school] || []).push(i); });
    var teams = Object.keys(byTeam).sort();
    if (!teams.length) return head + tabs + '<div class="empty-state">No injuries listed for teams on this week\\u2019s slate.</div>';
    var body = teams.map(function (school) {
      var list = byTeam[school].slice().sort(function (a, b) {
        return ((a.pos === 'QB') ? 0 : 1) - ((b.pos === 'QB') ? 0 : 1) || (INJ_ORDER[a.status] || 9) - (INJ_ORDER[b.status] || 9);
      });
      return '<div class="pcard" style="margin-bottom:8px"><div class="pcard-head"><span>' + esc(school) + '</span>' +
        '<span style="color:var(--muted-3);font-size:10px">' + list.length + ' listed</span></div>' +
        list.map(function (i) {
          return '<div class="pcard-line-row" title="' + esc(i.note || '') + '">' +
            '<span><b>' + esc(i.player) + '</b> <span style="color:var(--muted-3)">' + esc(i.pos || '') + '</span>' +
              (i.injury ? ' \\u00b7 <span style="color:var(--muted-3)">' + esc(i.injury) + '</span>' : '') + '</span>' +
            '<span><span style="color:' + injColor(i.status) + ';font-weight:700">' + esc(i.status) + '</span>' +
              (i.reported ? ' <span style="color:var(--muted-4);font-size:9.5px">(' + esc(i.reported) + ')</span>' : '') + '</span>' +
          '</div>';
        }).join('') + '</div>';
    }).join('');
    return head + tabs + '<div class="pcard-grid">' + body + '</div>' +
      '<div class="table-foot"><span>Hover a player for the latest note. Players listed Out or Doubtful are automatically kept off official ' +
      'prop plays; QB injuries show as warnings on the Edge Board and Bet Card. The model itself does not adjust its numbers for injuries.</span></div>';
  }

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

  // ---- How to treat one priced play, under the betting rules ------------
  // The single place that decides it (10/2026). The Projector's "Model's
  // play" panel and the Edge Board's Play column both read this, so a row
  // on the board can never say one thing while the Projector says another:
  //   Bet Card + official ................ wager
  //   official but off the card .......... track only
  //   underdog moneyline past +150 ....... unofficial, track only
  //   20+ point spread ................... faded by rule
  //   clears the edge bar but not +EV .... fade
  //   otherwise .......................... no play
  // p is a priceGame() result; onCard is whether that game is on the Bet
  // Card for p's market.
  function playStatusOf(p, onCard) {
    var market = p.market;
    var playText = market === 'Moneyline' && p.sideMoneyline != null
      ? p.playLabel + ' ' + (p.sideMoneyline > 0 ? '+' : '') + p.sideMoneyline
      : p.playLabel;
    var bigDogML = market === 'Moneyline' && p.sideMoneyline > 150;
    var s = { p: p, onCard: !!onCard, playText: playText, note: '' };
    if (market === 'Spread' && p.playLabel === 'No spread posted') {
      s.kind = 'nospread'; s.label = 'NO SPREAD POSTED'; s.color = 'var(--muted-3)'; s.playText = '—';
    } else if (isNearEvenTarget(p)) {
      s.kind = 'target'; s.label = 'TARGET · NEAR-EVEN ML'; s.color = 'var(--green)'; s.note = 'model 65%+ at +100 to +150 — small stake';
    } else if (p.qualifies && bigDogML) {
      s.kind = 'bigdog'; s.label = 'UNOFFICIAL · BIG UNDERDOG ML'; s.color = 'var(--amber)'; s.note = 'track only — longer than +150';
    } else if (p.qualifies && onCard) {
      s.kind = 'card'; s.label = 'OFFICIAL · BET CARD'; s.color = 'var(--green)'; s.note = 'wager';
    } else if (p.qualifies) {
      s.kind = 'official'; s.label = 'OFFICIAL PLAY'; s.color = 'var(--blue-light)'; s.note = 'not on the Bet Card — track only';
    } else if (p.bigSpread) {
      s.kind = 'bigspread'; s.label = 'UNOFFICIAL · 20+ PT SPREAD'; s.color = 'var(--amber)'; s.note = 'faded by rule';
    } else if (p.isFade) {
      s.kind = 'fade'; s.label = 'FADE'; s.color = 'var(--amber)'; s.note = 'model disagrees, but not +EV at this price';
    } else {
      s.kind = 'none'; s.label = 'NO PLAY'; s.color = 'var(--muted-3)';
      s.note = market === 'Spread' ? 'edge under the ' + D.meta.minEdge.toFixed(1) + '-pt threshold' : 'edge under the play threshold';
    }
    return s;
  }

  // ---- "Model's play" panel for the selected game (added 10/2026) ----
  // Spells out, for BOTH markets, what the model's play is and how to treat
  // it under the betting rules (see playStatusOf).
  function modelPlaysPanel(g) {
    var trained = (g.flags || []).some(function (f) { return f.text === 'In-season model'; });
    function row(market) {
      var o = { market: market, minEdge: D.meta.minEdge, abbrOf: abbrOf };
      var p = M.priceGame(g, o);
      var onCard = M.buildBetCard(D.games, o).some(function (c) { return c.game === g; });
      var s = playStatusOf(p, onCard);
      var label = s.label, color = s.color, note = s.note, playText = s.playText;
      var showLean = label !== 'NO SPREAD POSTED';
      return '<div style="display:grid;grid-template-columns:96px 1fr auto;gap:10px;align-items:center;padding:9px 0;border-top:1px solid var(--rule-faint)">' +
        '<div class="proj-stat-label" style="margin:0">' + esc(market) + '</div>' +
        '<div><div style="font-family:var(--font-display);font-weight:700;font-size:17px">' + esc(playText) + '</div>' +
          '<div style="font-size:10px;color:var(--muted-3);margin-top:3px">' +
            (showLean ? (market === 'Spread' ? 'cover' : 'win') + ' prob ' + (p.sideProb * 100).toFixed(1) + '% · edge ' + esc(p.sideEdgeLabel) : '') +
            (note ? (showLean ? ' · ' : '') + esc(note) : '') + '</div></div>' +
        '<div style="font-family:var(--font-display);font-weight:800;font-size:11px;letter-spacing:.06em;padding:4px 9px;border-radius:3px;' +
          'border:1px solid ' + color + ';color:' + color + ';white-space:nowrap">' + label + '</div>' +
      '</div>';
    }
    return '<div class="panel-pad" style="padding-top:14px;padding-bottom:10px;border-bottom:1px solid var(--rule)">' +
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">' +
        '<div class="chart-head" style="margin:0">Model’s play</div>' +
        '<span class="pchip" style="background:' + (trained ? 'rgba(23,194,107,0.16);color:var(--green)' : 'rgba(255,255,255,0.06);color:var(--muted-3)') + '">' +
          (trained ? (g.modelVersion === 'upgraded' ? 'UPGRADED MODEL' : 'TRAINED MODEL') : 'UNTRAINED (PRESEASON) MODEL') + '</span>' +
      '</div>' + row('Spread') + row('Moneyline') + '</div>';
  }

  function renderProjector(p) {
    if (!p) return '<div class="empty-state">No priced games to project.</div>';
    var g = p.game, a = team(g.away), h = team(g.home);
    var dist = M.distribution(g, p, { abbrOf: abbrOf });
    var dec = M.decomposition(g);
    var pWin = 1 - M.normalCdf(0, M.homeMargin(g), p.sd);
    // Everything in the scoreboard below is shown from the side of the team
    // the MODEL favors (10/2026) -- previously it was always home-team terms
    // ("TLSA margin -10.0", "TLSA win prob 28%") even when every play was on
    // the away team, which read as a contradiction.
    var favHome = M.homeMargin(g) >= 0;
    var fav = favHome ? h : a;
    var favLine = -Math.abs(M.homeMargin(g));
    var favProb = favHome ? pWin : 1 - pWin;
    var favMkt = g.marketSpread == null ? null : (favHome ? g.marketSpread : -g.marketSpread);
    var favPosted = favHome ? g.marketMoneyline : g.awayMoneyline;
    var favFair = M.fairAmerican(favProb);
    var gap = favMkt == null ? null : Math.abs(favLine - favMkt);
    // What the posted moneylines say that same team's chance is, with the
    // book's margin taken out -- the same removeVig() the Moneyline edge on
    // the board is measured against, so the gap here IS that edge, stated
    // for the model's favorite.
    var hasML = g.marketMoneyline != null && g.awayMoneyline != null && !isNaN(g.marketMoneyline) && !isNaN(g.awayMoneyline);
    var mktProbs = hasML ? M.removeVig(g.marketMoneyline, g.awayMoneyline) : null;
    var favMktProb = mktProbs ? (favHome ? mktProbs[0] : mktProbs[1]) : null;
    var probGap = favMktProb == null ? null : (favProb - favMktProb) * 100;
    function amer(v) { return v == null ? '—' : (v > 0 ? '+' : '') + v; }
    var aC = M.displayColor(a.primary), hC = M.displayColor(h.primary);
    var split = 'linear-gradient(100deg,' + aC + '55 0%,' + aC + '18 33%,' +
      'var(--panel-deep) 46%,var(--panel-deep) 54%,' + hC + '18 67%,' + hC + '55 100%)';
    var side = function (name, t) {
      return '<div class="proj-side">' + trkMark(t.abbr, 64) +
        '<div class="proj-team">' + esc(t.abbr) + '</div>' +
        (name !== t.abbr ? '<div class="proj-school">' + esc(name) + '</div>' : '') + '</div>';
    };
    var cmp = function (label, value, cls) {
      return '<div class="cmp-row' + (cls ? ' ' + cls : '') + '"><span>' + label + '</span><b>' + value + '</b></div>';
    };

    return '' +
      '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Matchup Projector</h2></div></div>' +
      '<div class="projector">' +

        '<div class="proj-head" style="background:' + split + '"><div class="proj-head-inner">' +
          side(g.away, a) +
          '<div class="proj-mid">' +
            '<div class="proj-at">AT</div>' +
            '<div class="proj-when">' + esc(edgeKickoff(g)) + '</div>' +
            '<div class="proj-meta">line: ' + esc(edgeBook(g)) + '</div>' +
            (g.flags && g.flags.length ? '<div class="proj-flags">' + g.flags.map(function (f) {
              var cls = f.level === 1 ? ' is-warn' : (f.level === 2 ? ' is-good' : '');
              return '<span class="flag-chip' + cls + '">' + (f.qb ? esc(abbrOf(f.team)) + ' ' : '') + esc(f.text) + '</span>';
            }).join('') + '</div>' : '') +
          '</div>' +
          side(g.home, h) +
        '</div></div>' +

        modelPlaysPanel(g) +
        '<div class="scoreboard">' +
          '<div class="score-cell"><div class="score-label">Model line</div>' +
            '<div class="score-value">' + esc(fav.abbr) + ' ' + M.signed(favLine) + '</div></div>' +
          '<div class="divider-v" style="height:auto"></div>' +
          '<div class="score-cell score-cell--wide"><div class="score-label">' + esc(fav.abbr) + ' win prob</div>' +
            '<div class="score-value is-blue">' + (favProb * 100).toFixed(0) + '%</div></div>' +
        '</div>' +

        // Model against market, both ways of asking the question: by how
        // many points, and how often the model's favorite wins.
        '<div class="proj-pair">' +
          '<div class="proj-stat"><div class="proj-stat-label">Spread</div>' +
            cmp('Model', esc(fav.abbr) + ' ' + M.signed(favLine)) +
            cmp('Market', favMkt == null ? '—' : esc(fav.abbr) + ' ' + M.signed(favMkt)) +
            cmp('Gap', gap == null ? 'no spread posted yet' : gap.toFixed(1) + ' pts ' + (favLine < favMkt ? 'more' : 'less') + ' on ' + esc(fav.abbr), 'is-gap') +
          '</div>' +
          '<div class="proj-stat"><div class="proj-stat-label">' + esc(fav.abbr) + ' to win</div>' +
            cmp('Model', (favProb * 100).toFixed(1) + '% <i>fair ' + amer(favFair) + '</i>') +
            cmp('<span title="What the posted moneylines imply, with the book’s margin taken out">Market</span>', favMktProb == null ? '—' : (favMktProb * 100).toFixed(1) + '% <i>posted ' + amer(favPosted) + '</i>') +
            cmp('Gap', probGap == null ? 'no moneyline posted yet' : M.signed(probGap) + ' pp', 'is-gap') +
          '</div>' +
        '</div>' +

        '<div class="panel-pad">' +
          '<div class="chart-head"><span>' + esc(dist.title) + '</span><span>σ ' + dist.sigma.toFixed(1) + ' (league-wide)</span></div>' +
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
            '<span class="legend">← ' + esc(h.abbr) + ' · ' + esc(a.abbr) + ' →</span></div>' +
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
              '<span style="left:0%">−' + dec.scaleMax + '</span><span style="left:50%">0</span>' +
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
          var qbFlags = (c.game.flags || []).filter(function (f) { return f.qb; });
          return '<div class="row"><div class="row-accent" style="background:' + esc(col) + '"></div>' +
            '<div class="row-body card-row">' +
              '<div><div class="card-play">' + (c.side ? trkMark(abbrOf(c.side), 22) : '') + '<span>' + esc(c.playLabel) + '</span></div>' +
                '<div class="card-note">' + esc(abbrOf(c.game.away) + ' at ' + abbrOf(c.game.home) + ' \\u00b7 ' + edgeKickoff(c.game)) +
                  ' \\u00b7 <span style="font-weight:700;color:' + (isTrainedGame ? '#2ecc71' : '#8A94A3') + '">' + (isTrainedGame ? (c.game.modelVersion === 'upgraded' ? 'UPGRADED' : 'TRAINED') : 'UNTRAINED') + '</span></div></div>' +
              '<div class="card-price">' + esc(sidePriceLabel) + '</div>' +
              '<div class="card-conf">' + (sideProb * 100).toFixed(1) + '%</div>' +
              '<div class="num"><span class="tier"><span>' + esc(c.tier) + '</span></span></div>' +
              '<div class="num"><button class="track-btn" onclick="window.__cfbTrack(' + trackPayload(c, true) + ')">+TRK</button></div>' +
              (isNearEvenTarget(c) ? '<div style="grid-column:1/-1;font-size:11px;color:var(--green);padding-top:6px;line-height:1.4">\u2605 Near-even target: +100 to +150 with the model at 65%+ \u2014 small stake</div>' : '') +
              (qbFlags.length ? '<div style="grid-column:1/-1;font-size:11px;color:var(--amber);padding-top:6px;line-height:1.4">\u26a0 Injury: ' +
                qbFlags.map(function (f) { return esc(abbrOf(f.team)) + ' ' + esc(f.text); }).join(' \u00b7 ') +
                ' \u2014 the model doesn\u2019t account for this. Check the latest status before betting.</div>' : '') +
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

  // ---- Player Props (rebuilt 10/2026) ----
  // One table instead of stacked cards. Each row shows the bet, the edge
  // (hit chance against what the price needs to break even) and the
  // backtest record for that edge size. propConfidence() below still rates
  // each play, but only to order the Watch / All props lists and to raise a
  // Check or Pass heads-up tag; the rating combines:
  //   * hit chance on the model's side,
  //   * how far that clears the price's breakeven,
  //   * games of data behind the projection,
  //   * injury status.
  // Only one row per player + market + game (the model's best line);
  // other books/lines for the same prop are counted, not listed.
  function propKickoff(iso) {
    if (!iso) return '';
    var d = new Date(iso);
    if (isNaN(d)) return '';
    // Shown in the viewer's own time zone (10/2026) -- it used to be UTC,
    // which put Saturday night games on "Sunday".
    var days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    var h = d.getHours(), m = d.getMinutes();
    return days[d.getDay()] + ' ' + (d.getMonth() + 1) + '/' + d.getDate() + ', ' +
      ((h % 12) || 12) + ':' + (m < 10 ? '0' : '') + m + ' ' + (h < 12 ? 'AM' : 'PM');
  }
  function propConfidence(r) {
    var lp = r._lp, be = r._be, gp = r.games_played, inj = r.injury_status;
    // Official = receptions overs at 10%+ and under 30% EV, and (10/2026)
    // receptions unders at 30%+ EV -- always listed first, as TARGET. Overs
    // at 30%+ are capped to Watch by the export.
    if (r.is_official_play) return { tier: 'TARGET', rank: 0, color: 'var(--green)' };
    if (lp == null) return { tier: 'NONE', rank: 9, color: 'var(--muted-4)' };
    var gap = be != null ? lp - be : 0;
    var injBlock = inj === 'Out' || inj === 'Out For Season' || inj === 'IR' || inj === 'Doubtful';
    if (injBlock || gap <= 0) return { tier: 'PASS', rank: 5, color: 'var(--red)' };
    // A projection wildly off the market (85%+ claimed, or 50%+ away from
    // the line) almost always means the model is missing information --
    // injury, depth-chart change, bad data match -- not a 40-yard mispricing.
    var far = r.line > 0 && Math.abs(r.model_predicted_value - r.line) >= Math.max(0.25 * r.line, 12);
    if (lp >= 0.80 || far) return { tier: 'CHECK', rank: 4, color: 'var(--amber)' };
    var t;
    if (lp >= 0.60 && gap >= 0.05 && (gp == null || gp >= 3) && !inj) t = { tier: 'HIGH', rank: 1, color: 'var(--green)' };
    else if (lp >= 0.55 && gap >= 0.02 && (gp == null || gp >= 2)) t = { tier: 'MEDIUM', rank: 2, color: 'var(--blue-light)' };
    else t = { tier: 'LOW', rank: 3, color: 'var(--muted-3)' };
    // Small whole-number stats (TDs, INTs, receptions) don't fit the model's
    // bell-curve math well, so they can't rate above Low.
    if (/touchdown|interception/i.test(r.market_name || '') && t.rank < 3) t = { tier: 'LOW', rank: 3, color: 'var(--muted-3)' };
    return t;
  }
  // ---- Hit chance at any line (10/2026) ----
  // The "Try another line" box in a prop's Details works out the model's
  // hit chance at a line the feed doesn't have. propOutcomeProbs mirrors
  // PlayerStatModel.outcome_probabilities (src/models/props_model.py): the
  // projection plus the actual holdout misses for projections that size,
  // with the bell curve as the fallback. Keep the two in step.
  var propDistCache = {};
  function propDist(stat) {
    if (propDistCache.hasOwnProperty(stat)) return propDistCache[stat];
    var d = (D.propDists || {})[stat], out = null;
    if (d) {
      out = {
        nonneg: !!d.nonneg,
        edges: (d.edges || []).map(function (e) { return e == null ? Infinity : Number(e); }),
        stds: d.stds || [],
        sd: d.sd,
        // Stored sorted, as whole numbers of 1/scale, each as the step up from the one before.
        samples: (d.samples || []).map(function (enc, b) {
          var acc = 0, a = new Array(enc.length), scale = (d.scales || [])[b] || 100;
          for (var i = 0; i < enc.length; i++) { acc += enc[i]; a[i] = acc / scale; }
          return a;
        })
      };
    }
    propDistCache[stat] = out;
    return out;
  }
  function normCdf(z) {
    // Abramowitz and Stegun 7.1.26; good to about 1e-7.
    var x = Math.abs(z) / Math.SQRT2;
    var t = 1 / (1 + 0.3275911 * x);
    var erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return z >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
  }
  function roundHalfEven(x) {   // numpy's rint
    var f = Math.floor(x), d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return f % 2 === 0 ? f : f + 1;
  }
  // {over, under, push} for a stat, an unrounded projection and a line, or
  // null when the data for it isn't there.
  function propOutcomeProbs(stat, pred, line) {
    var d = propDist(stat);
    if (!d || pred == null || isNaN(pred) || line == null || isNaN(line)) return null;
    var whole = Math.floor(line) === line;
    var n = d.edges.length, i;
    if (n && d.samples.length === n) {
      for (i = 0; i < n; i++) if (pred <= d.edges[i]) break;
      if (i >= n) i = n - 1;
      var r = d.samples[i];
      if (r.length >= 30) {
        var over = 0, under = 0;
        for (var k = 0; k < r.length; k++) {
          var o = pred + r[k];
          if (d.nonneg && o < 0) o = 0;
          if (whole) o = roundHalfEven(o);
          if (o > line) over++; else if (o < line) under++;
        }
        over /= r.length; under /= r.length;
        return whole ? { over: over, under: under, push: Math.max(0, 1 - over - under) } : { over: over, under: 1 - over, push: 0 };
      }
    }
    var sd = null;
    for (i = 0; i < d.edges.length && i < d.stds.length; i++) if (pred <= d.edges[i]) { sd = d.stds[i]; break; }
    if (sd == null) sd = d.stds.length ? d.stds[d.stds.length - 1] : d.sd;
    if (!sd || sd <= 0) return null;
    if (!whole) {
      var ov = 1 - normCdf((line - pred) / sd);
      return { over: ov, under: 1 - ov, push: 0 };
    }
    var o2 = 1 - normCdf((line + 0.5 - pred) / sd), u2 = normCdf((line - 0.5 - pred) / sd);
    return { over: o2, under: u2, push: Math.max(0, 1 - o2 - u2) };
  }
  // "-115", "+120" or "120" -> a number; null if it isn't an American price.
  function parseAmerican(txt) {
    var s = String(txt == null ? '' : txt).split(' ').join('').split(',').join('').split(String.fromCharCode(8722)).join('-');
    if (!s) return null;
    var n = Number(s);
    return (isNaN(n) || Math.abs(n) < 100) ? null : n;
  }

  function renderProps() {
    var live = D.propsLive || [];
    var head = '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Player Props</h2></div></div>';
    var scored = live.filter(function (r) { return r.model_predicted_value != null && r.model_lean; });
    if (!scored.length) {
      return head + '<div class="empty-state">' + (live.length
        ? live.length + ' prop lines are posted, but none have a model read yet (usually players without enough games this season).'
        : 'No player props posted right now. Books usually post most of the slate by Thursday or Friday.') + '</div>';
    }
    // Best line per player + market + game, by the model's hit chance on its side.
    var best = {}, counts = {}, alts = {};
    scored.forEach(function (r) {
      var isOver = r.model_lean === 'over';
      var price = isOver ? r.over_price : r.under_price;
      r._price = (price == null || isNaN(price)) ? null : Number(price);
      r._be = r._price != null ? M.impliedProb(r._price) : null;
      r._lp = r.model_lean_probability != null ? r.model_lean_probability
        : (r.model_over_probability != null ? (isOver ? r.model_over_probability : 1 - r.model_over_probability) : null);
      var key = [r.fixture_id, r.player_name, r.market_name].join('|');
      counts[key] = (counts[key] || 0) + 1;
      (alts[key] = alts[key] || []).push(r);
      var cur = best[key];
      var pri = function (x) { return x.is_official_play ? 2 : (x.is_watch_play ? 1 : 0); };
      if (!cur || pri(r) > pri(cur) || (pri(r) === pri(cur) && (r._lp || 0) > (cur._lp || 0))) best[key] = r;
    });
    var rows = Object.keys(best).map(function (k) { var r = best[k]; r._others = counts[k] - 1; r._alts = alts[k].filter(function (x) { return x !== r; }); r._conf = propConfidence(r); return r; });

    // Filters
    var markets = ['ALL'].concat(Array.from(new Set(rows.map(function (r) { return r.market_name; }))).sort());
    var mkt = markets.indexOf(state.propMarket) === -1 ? 'ALL' : state.propMarket;
    var games = {};
    rows.forEach(function (r) {
      if (!games[r.fixture_id]) {
        var a = schoolAbbr(r.team), b = schoolAbbr(r.opponent);
        games[r.fixture_id] = { label: [a, b].filter(Boolean).sort().join(' / ') || r.fixture_id, t: r.start_time || '' };
      }
    });
    var gameIds = Object.keys(games).sort(function (x, y) { return games[x].t < games[y].t ? -1 : 1; });
    var gameSel = state.propGame && games[state.propGame] ? state.propGame : 'ALL';
    // ---- Layout redesigned 10/2026 for readability ----
    var EH = D.propEdgeHistory, T = D.propTracking;
    // The backtest group a play belongs to (market + side + edge size). A
    // capped play is judged by the 30%+ group it was in when first flagged.
    var edgeHist = function (r) {
      if (!EH || r.model_ev == null || (r.model_lean !== 'over' && r.model_lean !== 'under')) return null;
      var bk = (EH.markets[r.market_name] || {})[r.model_lean];
      if (!bk) return null;
      for (var i = 0; i < bk.length; i++) if (r.edge_capped ? bk[i].lo >= 30 : (r.model_ev >= bk[i].lo && r.model_ev < bk[i].hi)) return bk[i];
      return null;
    };
    var signed = function (x, d) { var v = Number(x).toFixed(d); if (Number(v) === 0) v = (0).toFixed(d); return (Number(v) > 0 ? '+' : '') + v; };
    var tone = function (x) { return x > 0 ? 'var(--green)' : (x < 0 ? 'var(--red)' : 'var(--muted-3)'); };
    // Auto-tracked = logged and graded by the pipeline: official plays, plus
    // any play with an edge whose group is up in the backtest.
    var isTracked = function (r, h) { return !!(h && h.up && r.model_ev != null && r.model_ev >= 0); };
    var marketName = function (m) { return m === 'Reception Yards' ? 'Receiving Yards' : m; };
    rows.forEach(function (r) {
      r._h = edgeHist(r);
      r._auto = !!(r.is_official_play || isTracked(r, r._h));
      // Receptions unders at a 10% to 30% edge are leans with their own view
      // and their own record (10/2026).
      r._sec = r.is_official_play ? 'official' : (r.is_under_lean ? 'underlean' : (r.is_watch_play ? 'watch' : (r._auto ? 'tracked' : 'lean')));
    });

    // ---- Simplified 10/2026: each row answers "is there an edge?" once ----
    // Row = the bet, the edge (hit chance against what the price needs), the
    // backtest record for that edge size, and a heads-up tag only when
    // something needs a second look. The projection, games played, role
    // share, price history and game log moved into a Details panel. The tab
    // opens on Official plays, in kickoff order; Watch and All props are one
    // click away. The Conf column and the High/Medium/Low tally are gone
    // (Check and Pass still show, as heads-up tags).
    var view = (state.propView === 'watch' || state.propView === 'all' || state.propView === 'underlean') ? state.propView : 'official';
    var oneMarket = view === 'official' || view === 'underlean';   // receptions only: no market chips, every play listed
    var visible = function (r) { return state.propShowPass || r._conf.tier !== 'PASS' || r.is_official_play; };
    var mktOn = oneMarket ? 'ALL' : mkt;
    var shown = rows.filter(function (r) {
      return (mktOn === 'ALL' || r.market_name === mktOn) &&
        (gameSel === 'ALL' || r.fixture_id === gameSel) &&
        (view === 'all' || r._sec === view) && visible(r);
    });
    var byKick = function (x, y) {
      var a = x.start_time || '', b = y.start_time || '';
      return a < b ? -1 : (a > b ? 1 : (y.model_ev || 0) - (x.model_ev || 0));
    };
    var byStrength = function (x, y) { return x._conf.rank - y._conf.rank || (y._lp || 0) - (x._lp || 0); };

    // Counts on the view buttons match what each view will actually list.
    var nSec = { official: 0, underlean: 0, watch: 0, tracked: 0, lean: 0 }, nPass = 0;
    rows.forEach(function (r) {
      if (r._conf.tier === 'PASS') nPass++;
      if (visible(r)) nSec[r._sec]++;
    });

    // 1) Live record, one line, plus the key to the two numbers on each row.
    var recBit = function (label, o) {
      if (!o || !o.graded) return '<span class="pp-rec"><span class="pp-rec-label">' + label + '</span><span class="pp-rec-n">no graded plays yet</span></span>';
      return '<span class="pp-rec"><span class="pp-rec-label">' + label + '</span><b>' + esc(o.record) + '</b>' +
        '<b style="color:' + tone(o.units) + '">' + signed(o.units, 2) + 'u</b><span class="pp-rec-n">' + o.graded + ' graded</span></span>';
    };
    var strip = '<div class="pp-recline">' +
      (T ? recBit('Official overs', T.official) + recBit('Official unders', T.underPlay) + recBit('Under leans', T.underLean) + recBit('All auto-tracked', T.overall) : '') +
      '<span class="pp-key"><b>Edge</b> is the model&rsquo;s hit chance against the price. <b>Track record</b> is how plays at that edge size did in the backtest.</span>' +
    '</div>';

    // 2) Controls: which plays, which game, and (outside Official) which market.
    var seg = function (i, key, label, n) {
      return '<button class="' + (view === key ? 'is-on' : '') + '" aria-pressed="' + (view === key ? 'true' : 'false') + '" onclick="window.__cfbPropView(' + i + ')">' + label + ' (' + n + ')</button>';
    };
    var controls = '<div class="pp-controls">' +
      '<div class="pp-seg" role="group" aria-label="Which plays to show">' +
        seg(0, 'official', 'Official', nSec.official) +
        seg(3, 'underlean', 'Under leans', nSec.underlean) +
        seg(1, 'watch', 'Watch', nSec.watch) +
        seg(2, 'all', 'All props', nSec.official + nSec.underlean + nSec.watch + nSec.tracked + nSec.lean) +
      '</div>' +
      '<select class="pp-select" aria-label="Game" onchange="window.__cfbPropGame(this.value)">' +
        '<option value="ALL"' + (gameSel === 'ALL' ? ' selected' : '') + '>All games (' + gameIds.length + ')</option>' +
        gameIds.map(function (id) { return '<option value="' + esc(id) + '"' + (gameSel === id ? ' selected' : '') + '>' + esc(games[id].label) + ', ' + esc(propKickoff(games[id].t)) + '</option>'; }).join('') +
      '</select>' +
      (view === 'all' ? '<button class="pp-chip' + (state.propShowPass ? ' is-on' : '') + '" onclick="window.__cfbPropShowPass()">Show passes (' + nPass + ')</button>' : '') +
    '</div>' +
    (oneMarket ? '' : '<div class="pp-chips">' + markets.map(function (m) {
      return '<button class="pp-chip' + (m === mkt ? ' is-on' : '') + '" data-prop-market="' + esc(m) + '">' + (m === 'ALL' ? 'All markets' : esc(marketName(m))) + '</button>';
    }).join('') + '</div>');

    // 3) The plays. Official and Watch are grouped by kickoff day (soonest
    // first); All props keeps the status groups.
    var DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    var pad2 = function (n) { return (n < 10 ? '0' : '') + n; };
    var dayKey = function (iso) {
      var d = new Date(iso || '');
      return isNaN(d) ? '9999-99-99' : d.getFullYear() + '-' + pad2(d.getMonth() + 1) + '-' + pad2(d.getDate());
    };
    var dayLabel = function (iso) {
      var d = new Date(iso || '');
      if (isNaN(d)) return 'Kickoff not posted';
      var txt = DAYS[d.getDay()] + ' ' + (d.getMonth() + 1) + '/' + d.getDate();
      return (d.toDateString() === new Date().toDateString() ? 'Today, ' : '') + txt;
    };
    var kickTime = function (iso) {
      var d = new Date(iso || '');
      if (isNaN(d)) return '';
      var h = d.getHours(), m = d.getMinutes();
      return ((h % 12) || 12) + ':' + pad2(m) + ' ' + (h < 12 ? 'AM' : 'PM');
    };
    var groups = [];   // {key, name, note, cap, rows}
    if (view === 'all') {
      [['official', 'Official plays', 'Receptions overs with a 10% to 30% edge, and receptions unders with a 30%+ edge. The only props the rules call bets.', 0],
       ['underlean', 'Under leans', 'Receptions unders with a 10% to 30% edge. Winners in all three seasons tested, by a thin margin. Graded on their own line.', 12],
       ['watch', 'Watch', 'A 10%+ edge on a secondary-role player, or a receptions over capped at 30%+. Tracked, not bet.', 12],
       ['tracked', 'Auto-tracked', 'Has an edge and its kind of play is up in the backtest. Logged and graded automatically.', 12],
       ['lean', 'Other leans', 'Everything else the model has a read on.', 20]].forEach(function (sec) {
        var list = shown.filter(function (r) { return r._sec === sec[0]; }).sort(byStrength);
        if (list.length) groups.push({ key: 'all|' + sec[0], name: sec[1], note: sec[2], cap: sec[3], rows: list });
      });
    } else {
      var byDay = {};
      shown.forEach(function (r) { var k = dayKey(r.start_time); (byDay[k] = byDay[k] || []).push(r); });
      Object.keys(byDay).sort().forEach(function (k) {
        var list = byDay[k].sort(oneMarket ? byKick : byStrength);
        groups.push({ key: view + '|' + k, name: dayLabel(list[0].start_time), note: '', cap: oneMarket ? 0 : 12, rows: list });
      });
    }
    var ordered = [];
    groups.forEach(function (g) { g.start = ordered.length; ordered = ordered.concat(g.rows); });
    window.__cfbPropRows = ordered;

    var myTrk = {};
    // flaggedDesc: a play whose line you later changed in the tracker still
    // counts as the same play here, under the line it was added at.
    loadTrk().forEach(function (it) { if (it.type === 'Prop') { myTrk[it.description] = true; if (it.flaggedDesc) myTrk[it.flaggedDesc] = true; } });
    var keyOf = function (r) { return [r.fixture_id, r.player_name, r.market_name].join('|'); };
    var nowMs = Date.now();
    var priceTxt = function (x) { return x == null || isNaN(x) ? 'no price' : (x > 0 ? '+' : '') + Number(x); };
    var numTxt = function (x) { var n = Number(x); return isNaN(n) ? '' : (Math.round(n * 10) / 10).toString(); };
    var dcell = function (label, value, sub) {
      return '<div class="pp-dcell"><div class="pp-dlabel">' + label + '</div><div class="pp-dval">' + value + '</div>' + (sub ? '<div class="pp-sub">' + sub + '</div>' : '') + '</div>';
    };
    // "Try another line" (10/2026): the model's hit chance and edge at any
    // line and price, e.g. when your book hangs 82.5 and the feed has 85.5.
    // A line the feed already has uses the pipeline's own numbers; any other
    // line is worked out here from the same projection and the same misses.
    var whatIfHtml = function (r, side, lineTxt, priceIn) {
      var line = parseFloat(lineTxt), price = parseAmerican(priceIn);
      if (lineTxt === '' || lineTxt == null || isNaN(line) || line < 0) return '<span class="pp-wi-note">Enter a line, like 82.5.</span>';
      var over = side !== 'under';
      var posted = null;
      [r].concat(r._alts || []).forEach(function (x) { if (!posted && Number(x.line) === line && x.model_over_probability != null) posted = x; });
      var pr;
      if (posted) {
        var push0 = posted.model_push_probability || 0;
        pr = { over: posted.model_over_probability, push: push0,
               under: (push0 > 0 && posted.model_under_probability != null) ? posted.model_under_probability : 1 - posted.model_over_probability };
      } else {
        pr = r.model_projection_raw != null ? propOutcomeProbs(r.model_stat, Number(r.model_projection_raw), line) : null;
      }
      if (!pr) return '<span class="pp-wi-note">Not available for this play until the next model run.</span>';
      var p = over ? pr.over : pr.under, q = over ? pr.under : pr.over;
      var html = '<span class="pp-wi-main">' + (over ? 'Over ' : 'Under ') + line + ': <b>' + (p * 100).toFixed(1) + '%</b> hit chance</span>' +
        (pr.push > 0 ? '<span class="pp-wi-note">push ' + (pr.push * 100).toFixed(1) + '%</span>' : '');
      if (price == null) return html + '<span class="pp-wi-note">' + (String(priceIn || '').length ? 'Enter a price like -115 or +120.' : 'Add a price to see the edge.') + '</span>';
      var dec = price > 0 ? 1 + price / 100 : 1 + 100 / Math.abs(price);
      // Same sums as the pipeline: on a line that can push, a push returns the stake.
      var ev = pr.push > 0 ? (p * (dec - 1) - q) * 100 : (p * dec - 1) * 100;
      html += '<span class="pp-wi-note">needs ' + (100 / dec).toFixed(1) + '% at ' + priceTxt(price) + '</span>' +
        '<span class="pp-wi-edge" style="color:' + tone(ev) + '">Edge ' + signed(ev, 1) + '%</span>';
      var h = edgeHist({ market_name: r.market_name, model_lean: over ? 'over' : 'under', model_ev: ev });
      if (h) html += '<span class="pp-wi-note">' + (h.small ? 'Small backtest sample at this edge size (' + esc(h.record) + ').'
        : 'Backtest at this edge size: ' + esc(h.record) + ', ' + signed(h.roi, 0) + '% ROI.') + '</span>';
      return html;
    };
    window.__cfbWhatIfHtml = whatIfHtml;
    var whatIfCell = function (r) {
      var wi = (state.propWhatIf && state.propWhatIf.key === keyOf(r)) ? state.propWhatIf : null;
      if (!wi) wi = state.propWhatIf = { key: keyOf(r), side: r.model_lean === 'under' ? 'under' : 'over', line: String(r.line), price: r._price != null ? priceTxt(r._price) : '' };
      window.__cfbPropOpenRow = r;
      var postedRows = [r].concat(r._alts || []).sort(function (x, y) { return Number(x.line) - Number(y.line) || (y.model_ev || 0) - (x.model_ev || 0); });
      window.__cfbPropOpenAlts = postedRows;
      var postedHtml = postedRows.length > 1 ? '<div class="pp-wi-posted"><span class="pp-wi-note">Posted lines:</span>' + postedRows.map(function (x, j) {
        var xp = x.model_lean === 'over' ? x.over_price : x.under_price;
        xp = (xp == null || isNaN(xp)) ? null : Number(xp);
        return '<button class="pp-chip" title="Load this line into the box" onclick="window.__cfbWhatIfUse(' + j + ')">' + (x.model_lean === 'over' ? 'Over ' : 'Under ') + esc(x.line) +
          ' &middot; ' + priceTxt(xp) + (x.book_used ? ' ' + esc(bookLabel(x.book_used)) : '') + (x.model_ev != null ? ' &middot; ' + signed(x.model_ev, 1) + '%' : '') + '</button>';
      }).join('') + '</div>' : '';
      var sideBtn = function (i, key, label) {
        return '<button id="pp-wi-' + key + '" class="' + (wi.side === key ? 'is-on' : '') + '" aria-pressed="' + (wi.side === key ? 'true' : 'false') + '" onclick="window.__cfbWhatIfSide(' + i + ')">' + label + '</button>';
      };
      return '<div class="pp-dcell is-wide"><div class="pp-dlabel">Try another line</div>' +
        '<div class="pp-wi">' +
          '<div class="pp-seg" role="group" aria-label="Side">' + sideBtn(0, 'over', 'Over') + sideBtn(1, 'under', 'Under') + '</div>' +
          '<label class="pp-wi-field">Line <input id="pp-wi-line" class="pp-wi-input" type="number" step="0.5" min="0" inputmode="decimal" value="' + esc(wi.line) + '" oninput="window.__cfbWhatIf()"></label>' +
          '<label class="pp-wi-field">Price <input id="pp-wi-price" class="pp-wi-input" type="text" placeholder="-115" autocomplete="off" value="' + esc(wi.price) + '" oninput="window.__cfbWhatIf()"></label>' +
          '<div id="pp-wi-out" class="pp-wi-out" aria-live="polite">' + whatIfHtml(r, wi.side, wi.line, wi.price) + '</div>' +
        '</div>' + postedHtml + '</div>';
    };
    var detailHtml = function (r) {
      var isOver = r.model_lean === 'over';
      var cells = [];
      cells.push(dcell('Projection', r.model_predicted_value.toFixed(1), 'line ' + esc(r.line)));
      cells.push(dcell('Games this season', r.games_played != null ? String(r.games_played) : '&mdash;', 'with a box-score line'));
      var isRec = /recep|receiv/i.test(r.market_name || ''), isRush = /rush/i.test(r.market_name || '');
      var share = isRec ? r.rec_share : (isRush ? r.carry_share : null);
      if (share != null && !isNaN(share)) cells.push(dcell(isRec ? 'Share of team catches' : 'Share of team carries', (share * 100).toFixed(0) + '%', 'per game, this season'));
      var ff = r.first_flag;
      var now = priceTxt(r._price) + (r.book_used ? ' &middot; ' + esc(bookLabel(r.book_used)) : '');
      var was = '';
      if (ff && ff.price != null) {
        var fd = new Date(ff.at || '');
        was = 'first logged ' + (isNaN(fd) ? '' : DAYS[fd.getDay()] + ' ' + (fd.getMonth() + 1) + '/' + fd.getDate() + ' ') + 'at ' + priceTxt(Number(ff.price)) +
          (ff.line != null && Number(ff.line) !== Number(r.line) ? ', line ' + esc(ff.line) : '') + (ff.book ? ', ' + esc(bookLabel(ff.book)) : '');
      }
      cells.push(dcell('Price', now, (was || '') + (r._others ? (was ? ' &middot; ' : '') + '+' + r._others + ' other line' + (r._others > 1 ? 's' : '') + ' posted' : '')));
      var gl = (D.propGameLogs || {})[r.game_log_key] || [];   // [week, opponent, value], oldest first
      if (gl.length) {
        var hits = 0;
        var chips = gl.map(function (g) {
          var wk = g[0], oppName = g[1], v = Number(g[2]), ok = isOver ? v > r.line : v < r.line;
          if (ok) hits++;
          var opp = (D.meta.schoolAbbr || {})[oppName] || (wk != null ? 'Wk ' + wk : '');
          return '<span class="pp-gl' + (ok ? ' is-hit' : '') + '" title="' + (wk != null ? 'Week ' + esc(wk) : '') + (oppName ? ' vs ' + esc(oppName) : '') + '"><b>' + numTxt(v) + '</b><i>' + esc(opp) + '</i></span>';
        }).join('');
        cells.push('<div class="pp-dcell is-wide"><div class="pp-dlabel">' + esc(marketName(r.market_name)) + ' by game</div><div class="pp-gls">' + chips + '</div>' +
          '<div class="pp-sub">' + (isOver ? 'Over ' : 'Under ') + esc(r.line) + ' in ' + hits + ' of ' + gl.length + '. The model already counts these games.</div></div>');
      }
      var muD = matchupOf(r);
      if (muD) cells.push(dcell('Matchup', esc(muD.label), esc(muD.title)));
      else if (r.market_name === 'Receptions' && r.matchup === 'average') cells.push(dcell('Matchup', 'Average defense against ' + (MATCHUP_GROUPS[r.matchup_vs] || 'his position'), 'It has allowed about what its opponents usually get.'));
      cells.push(whatIfCell(r));
      return '<div class="pp-detail">' + cells.join('') + '</div>';
    };
    // Matchup marker on receptions props (10/2026). The export labels the
    // opposing defense tough, average or soft against this player's position
    // group (catches allowed against what those offenses usually get, earlier
    // games only). Nothing is shown for an average defense.
    //   fire   an under with a 10%+ edge against a tough defense -- the one
    //          pairing the three-season backtest supports (260-164)
    //   check  an over with a 10%+ edge against a soft defense -- agrees
    //          with the pick, but did not add anything in the backtest
    //   plain  the defense goes against the pick, or the edge is under 10%
    var MATCHUP_GROUPS = { WR: 'wide receivers', TE: 'tight ends', RB: 'running backs' };
    var matchupOf = function (r) {
      if (r.market_name !== 'Receptions' || (r.matchup !== 'tough' && r.matchup !== 'soft')) return null;
      if (r.model_lean !== 'over' && r.model_lean !== 'under') return null;
      var tough = r.matchup === 'tough', over = r.model_lean === 'over', edge = r.model_ev != null && r.model_ev >= 10;
      var name = tough ? 'Tough D' : 'Soft D';
      var what = 'This defense has allowed ' + (tough ? 'fewer' : 'more') + ' catches to ' + (MATCHUP_GROUPS[r.matchup_vs] || 'his position') + ' than its opponents usually get.';
      if (!over && tough && edge) return { kind: 'fire', cls: 'is-official', text: '&#128293; ' + name, label: 'Tough defense against ' + (MATCHUP_GROUPS[r.matchup_vs] || 'his position'),
        title: what + ' Unders with a 10%+ edge against tough defenses went 260-164 over three seasons, and were up in each one.' };
      if (over && !tough && edge) return { kind: 'check', cls: 'is-track', text: '&#10003; ' + name, label: 'Soft defense against ' + (MATCHUP_GROUPS[r.matchup_vs] || 'his position'),
        title: what + ' That agrees with the over, but it is not a proven edge: official overs against soft defenses went 33-25 over three seasons, no better than other overs.' };
      var against = over === tough;
      return { kind: against ? 'against' : 'plain', cls: 'is-lean', text: name, label: (tough ? 'Tough' : 'Soft') + ' defense against ' + (MATCHUP_GROUPS[r.matchup_vs] || 'his position'),
        title: what + (against ? ' That goes against this pick.' + (over ? ' Official overs against tough defenses went 79-72 over three seasons.' : ' Unders with a 10%+ edge against soft defenses went 160-132 over three seasons.')
                               : ' The edge here is under 10%, below where the matchup was tested.') };
    };
    var rowHtml = function (r, i, withDay) {
      var isOver = r.model_lean === 'over';
      var h = r._h;
      var isOpen = state.propDetail === keyOf(r);
      var started = r.start_time && new Date(r.start_time).getTime() < nowMs;
      // A play made under one of the receptions rules shows that RULE's
      // record over every season graded (props lab 3), not the record of
      // its narrow edge band in this season alone.
      var rule = (D.propRules || {})[r.play_kind];
      var track = rule ? '<div class="pp-track" style="color:' + tone(rule.roi) + '">' + signed(rule.roi, 0) + '% ROI &middot; ' + signed(rule.units, 1) + 'u</div>' +
          '<div class="pp-sub">' + esc(rule.record) + ' under this rule, ' + rule.seasons + ' season' + (rule.seasons === 1 ? '' : 's') + '</div>'
        : !h ? '<div class="pp-track is-none">Not in backtest</div>'
        : h.small ? '<div class="pp-track is-none">Small sample</div><div class="pp-sub">' + esc(h.record) + ' at this edge size</div>'
        : '<div class="pp-track" style="color:' + tone(h.roi) + '">' + signed(h.roi, 0) + '% ROI &middot; ' + signed(h.units, 1) + 'u</div>' +
          '<div class="pp-sub">' + esc(h.record) + ' at this edge size</div>';
      var flags = [];
      if (r.play_kind === 'rec_under') flags.push('<span class="pp-tag is-official" title="A receptions under with a 30%+ edge. An official play.">Under play</span>');
      else if (r.is_under_lean) flags.push('<span class="pp-tag is-watch" title="A receptions under with a 10% to 30% edge. A lean, graded on its own line.">Under lean</span>');
      var mu = matchupOf(r);
      if (mu) flags.push('<span class="pp-tag ' + mu.cls + '" title="' + esc(mu.title) + '">' + mu.text + '</span>');
      if (r.injury_status) flags.push('<span class="pp-tag is-inj" title="' + esc(r.injury_detail || 'On the injury report') + '">' + esc(r.injury_status) + '</span>');
      if (r.edge_capped) flags.push('<span class="pp-tag is-inj" title="A receptions over with a 30%+ edge. Those lost in the backtest, so it is watched, not bet.">30%+ edge</span>');
      else if (!r.is_official_play && r._conf.tier === 'CHECK') flags.push('<span class="pp-tag is-inj" title="The model is far from the market. Check for an injury or a role change first.">Check</span>');
      if (r._conf.tier === 'PASS') flags.push('<span class="pp-tag is-pass" title="The price needs more than the model gives, or the player is out.">Pass</span>');
      if (started) flags.push('<span class="pp-tag is-lean" title="This game has kicked off.">Started</span>');
      return '<div class="pp-grid pp-row' + (isOpen ? ' is-open' : '') + '">' +
        '<div><div class="pp-name">' + esc(r.player_name) + '</div>' +
          '<div class="pp-sub">' + esc(schoolAbbr(r.team) || '') + (r.opponent ? ' vs ' + esc(schoolAbbr(r.opponent)) : '') + ' &middot; ' + esc(withDay ? propKickoff(r.start_time) : kickTime(r.start_time)) + '</div></div>' +
        '<div><div class="pp-pick">' + esc(marketName(r.market_name)) + ' <b>' + (isOver ? 'Over' : 'Under') + ' ' + esc(r.line) + '</b></div>' +
          '<div class="pp-sub">' + priceTxt(r._price) + (r.book_used ? ' &middot; ' + esc(bookLabel(r.book_used)) : '') + '</div></div>' +
        '<div><div class="pp-edge" style="color:' + (r.model_ev != null ? tone(r.model_ev) : 'var(--muted-3)') + '">' + (r.model_ev != null ? signed(r.model_ev, 1) + '%' : '&mdash;') + '</div>' +
          '<div class="pp-sub">' + (r._lp != null ? (r._lp * 100).toFixed(0) + '%' : '&mdash;') + ' vs ' + (r._be != null ? (r._be * 100).toFixed(0) + '%' : '&mdash;') + ' needed</div></div>' +
        '<div>' + track + '</div>' +
        '<div class="pp-flags">' + flags.join('') + '</div>' +
        '<div class="pp-actions">' +
          '<button class="pp-trk" aria-expanded="' + (isOpen ? 'true' : 'false') + '" onclick="window.__cfbPropDetail(' + i + ')">' + (isOpen ? 'Hide' : 'Details') + '</button>' +
          (myTrk[propTrackLabel(r)]
            ? '<span class="pp-inlist" title="Already in your own list in My Tracker">&#10003; My list</span>'
            : '<button class="pp-trk" title="Add this play to your own list in My Tracker" onclick="window.__cfbTrackProp(' + i + ')">+ My list</button>') +
        '</div>' +
      '</div>' + (isOpen ? detailHtml(r) : '');
    };
    // Long groups start folded to their strongest plays so the page stays
    // short; one click opens them.
    var open = state.propOpen || {};
    var body = groups.map(function (g) {
      var list = g.rows.map(function (r, j) { return rowHtml(r, g.start + j, view === 'all'); });
      var folded = g.cap && list.length > g.cap && !open[g.key];
      return '<div class="pp-group"><span class="pp-group-name">' + esc(g.name) + '</span><span class="pp-group-count">' + list.length + '</span>' +
        (g.note ? '<span class="pp-group-note">' + g.note + '</span>' : '') + '</div>' + (folded ? list.slice(0, g.cap) : list).join('') +
        (g.cap && list.length > g.cap ? '<div class="pp-more"><button class="pp-link" data-prop-fold="' + esc(g.key) + '">' +
          (folded ? 'Show all ' + list.length : 'Show top ' + g.cap + ' only') + '</button></div>' : '');
    }).join('');
    var emptyMsg = view === 'official' ? 'No official plays right now. Under leans, Watch and All props show everything else the model has a read on.'
      : view === 'underlean' ? 'No under leans right now. They are receptions unders with a 10% to 30% edge.'
      : 'No plays match these filters. Try another market or game.';
    var table = '<div class="pp-table">' +
      '<div class="pp-grid pp-head"><div>Player</div><div>Pick</div><div>Edge</div><div>Track record</div><div>Heads up</div><div></div></div>' +
      (body || '<div class="empty-state" style="border:none">' + emptyMsg + '</div>') +
    '</div>';

    // 4) Where the model has been right (backtest), and the latest results.
    var lower = '';
    var kinds = [];
    if (EH) {
      Object.keys(EH.markets).forEach(function (m) {
        ['over', 'under'].forEach(function (side) {
          var ups = ((EH.markets[m] || {})[side] || []).filter(function (g) { return g.up; });
          if (!ups.length) return;
          var w = 0, l = 0, u = 0, n = 0;
          ups.forEach(function (g) { var p = String(g.record).split('-'); w += Number(p[0]) || 0; l += Number(p[1]) || 0; u += g.units; n += g.n; });
          kinds.push({ name: marketName(m) + ' ' + side + 's', roi: n ? u / n * 100 : 0, units: u, record: w + '-' + l,
                       edges: ups.map(function (g) { return g.label; }).join(', ') });
        });
      });
      kinds.sort(function (x, y) { return y.units - x.units; });
    }
    var bucketPanel = kinds.length
      ? '<div class="pp-panel"><h3>Where the model has been right</h3>' +
          '<div class="pp-panel-note">Backtest' + (EH.nWeekends ? ', ' + EH.nWeekends + ' weekends' + (EH.span ? ' (' + esc(EH.span) + ')' : '') : '') +
            ', 1 unit a play. Only the edge sizes that finished up are counted, so these numbers flatter each kind of play.</div>' +
          '<div class="pp-buckets">' + kinds.slice(0, 8).map(function (k) {
            return '<div class="pp-bucket"><div class="pp-bucket-name">' + esc(k.name) + '</div><div class="pp-bucket-roi">' + signed(k.roi, 0) + '% ROI</div>' +
              '<div class="pp-sub" style="color:var(--text-dim)">' + signed(k.units, 1) + 'u &middot; ' + esc(k.record) + '</div>' +
              '<div class="pp-sub">edges ' + esc(k.edges) + '</div></div>';
          }).join('') + '</div></div>'
      : '';
    var resultsPanel = '';
    if (T && T.results && T.results.length) {
      resultsPanel = '<div class="pp-panel"><h3>Latest results</h3>' +
        '<div class="pp-panel-note">Auto-tracked plays, graded from the box score. Newest first.</div>' +
        T.results.slice(0, 8).map(function (p) {
          var rc = p.result === 'W' ? 'var(--green)' : (p.result === 'L' ? 'var(--red)' : 'var(--amber)');
          return '<div class="pp-res"><div><div class="pp-res-name">' + esc(p.player) + '</div>' +
              '<div class="pp-sub">' + esc(marketName(p.market)) + ' ' + (p.side === 'over' ? 'Over' : 'Under') + ' ' + esc(p.line) + ' &middot; final ' + (p.actual != null ? Number(p.actual) : '?') + '</div></div>' +
            '<div class="num">' + (p.model_ev != null ? signed(p.model_ev, 1) + '%' : '') + '</div>' +
            '<div class="pp-result" style="color:' + rc + ';border-color:' + rc + '">' + (p.result === 'W' ? 'WIN' : (p.result === 'L' ? 'LOSS' : 'PUSH')) + '</div>' +
            '<div class="num" style="font-weight:700;color:' + tone(p.units || 0) + '">' + signed(p.units || 0, 2) + 'u</div></div>';
        }).join('') +
        '<button class="pp-link" onclick="window.__cfbGoTracker()">See every result in My Tracker</button></div>';
    }
    if (bucketPanel || resultsPanel) lower = '<div class="pp-lower">' + bucketPanel + resultsPanel + '</div>';

    // 5) The fine print, folded away until wanted.
    var help = '<details class="pp-help"><summary>How to read this page</summary><div class="pp-help-body">' +
      '<p><b>Official</b> plays are the only props the rules call bets: receptions overs with a 10% to 30% edge, and receptions unders with a 30%+ edge, at the best price across books. ' +
        '<b>Under leans</b> are receptions unders with a 10% to 30% edge: they won in all three seasons tested, but by a thin margin, so they are graded on their own line. ' +
        'An under is a play or a lean by its edge at the best price right now, so it can move between the two; the record grades each one once, at the price and label it was first flagged with. ' +
        '<b>Watch</b> plays are logged to see whether they hold up, not bet. <b>All props</b> lists everything the model has a read on.</p>' +
      '<p><b>Edge</b> is what the model expects the bet to return. Under it is the model&rsquo;s chance the pick wins, against the chance the price needs to break even.</p>' +
      '<p><b>Track record</b> is how plays of the same kind and edge size did in the backtest' +
        (EH && EH.nWeekends ? ' (' + EH.nWeekends + ' weekends, 1 unit a play)' : '') + '. Green made money, red lost, and &ldquo;small sample&rdquo; means fewer than ' +
        ((EH && EH.minBets) || 15) + ' bets. Every play in the same edge range shows the same record. Groups are small, so read it as a guide, not a promise. ' +
        'Official plays and under leans show their whole rule&rsquo;s record over every season tested instead.</p>' +
      '<p><b>Heads up</b> marks unders as an <b>Under play</b> or an <b>Under lean</b>. Otherwise it is empty unless something needs a second look: an injury listing, a <b>30%+ edge</b> on an over (those lost in the backtest, so they are capped to Watch), ' +
        '<b>Check</b> (the model is far from the market, which usually means it is missing an injury or a role change), ' +
        '<b>Pass</b> (the price needs more than the model gives, or the player is out), or a game that has already started.</p>' +
      '<p><b>&#128293; Tough D</b> and <b>&#10003; Soft D</b> appear on receptions props when the opposing defense has allowed clearly fewer, or clearly more, catches to that player&rsquo;s position than its opponents usually get (earlier games this season only). ' +
        'The fire marks an under with a 10%+ edge against a tough defense: those went 260-164 over three seasons. The check marks an over with a 10%+ edge against a soft defense: it agrees with the pick, but overs there did no better than other overs, so treat it as context. ' +
        'A plain grey Tough D or Soft D means the defense goes against the pick. Hover any of them for the detail.</p>' +
      '<p><b>Details</b> opens the projection, games played, share of the team&rsquo;s catches or carries, the price when the play was first logged, and the player&rsquo;s numbers by game. ' +
        'The model already counts those games, so a good run there is not extra evidence. ' +
        '<b>Try another line</b>, in the same panel, gives the model&rsquo;s hit chance and edge at any line and price you type, for when your book has a different number.</p>' +
      '<p><b>+ My list</b> adds any play to your own list in My Tracker. Kickoff times are in your own time zone.</p>' +
    '</div></details>';

    return head + strip + controls + table + lower + help;
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
    var known = {}, bets = {};
    // bets: every play already here, by what the bet IS (trkDupKey), not
    // just by id (10/2026). Without it, removing the restored copy of a bet
    // that also exists as the original row brought the copy straight back
    // on the next page load, since its id was "missing" again.
    items.forEach(function (it) { known[it.id] = true; bets[trkDupKey(it)] = true; });
    var added = false;
    SEED_TRACKER.forEach(function (it) {
      if (!known[it.id] && !bets[trkDupKey(it)]) { items.push(it); known[it.id] = true; added = true; }
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
        var applied = 0, skippedEdited = {};
        imported.grades.forEach(function (g) {
          if (!g || ['win', 'loss', 'push'].indexOf(g.status) === -1) return;
          trk.forEach(function (it) {
            if (it.status) return;
            var desc = it.description || '';
            var hit = g.play
              ? desc.split(' \\u2014 ')[0] === g.play
              : (g.player && g.market && it.type === 'Prop' && desc.indexOf(g.player + ' ' + g.market) !== -1);
            // A prop grade is matched by player + market only, on the
            // understanding that the row sits at the line the model
            // flagged. Once you've changed a prop's line or side in the
            // tracker (10/2026) that no longer holds -- the Over can win
            // where your Under lost -- so those rows are left for you to
            // grade. Game lines need no such guard: their match includes
            // the number, so an edited spread simply doesn't match.
            if (hit && !g.play && it.flaggedDesc && it.flaggedDesc !== desc) { skippedEdited[it.id] = true; return; }
            if (hit) { it.status = g.status; it.updatedAt = Date.now(); applied++; }
          });
        });
        saveTrk(trk);
        input.value = '';
        render();
        var nSkipped = Object.keys(skippedEdited).length;
        alert(applied + ' tracked play(s) graded from the file.' +
          (nSkipped ? '\\n\\n' + nSkipped + ' prop(s) whose line or side you changed were left ungraded \\u2014 grade those yourself.' : ''));
        return;
      }
      if (!Array.isArray(imported)) {
        alert('That file is not a valid tracker export.');
        return;
      }
      // Merge by id so re-importing the same file twice (or importing on
      // a laptop that already has some overlapping plays) never duplicates
      // a row. An auto-tracked prop is also matched by its sourceId: each
      // browser gives the same official prop its own random id, so matching
      // on id alone added a second copy of every one.
      //
      // A play that IS already here (10/2026, now that plays can be edited):
      //  - whichever copy was changed more recently wins -- every edit,
      //    tag, note and grade stamps updatedAt, and a copy with no stamp
      //    counts as oldest -- so a spread fixed on one laptop carries to
      //    the other and re-importing an old file can't roll an edit back;
      //  - a grade is never dropped by that. If the copy that wins has no
      //    grade and the other one does, the grade is kept, as long as both
      //    copies are the same bet (same description);
      //  - if they are NOT the same bet (the line was changed on one
      //    laptop after the other graded it at the old number), the old
      //    grade can't be trusted at the new number, so the play is left
      //    ungraded and counted in the message as needing a grade.
      var existing = loadTrk();
      // atBet: the same bet under a different id -- the other laptop's copy
      // of a play both laptops tracked, or the copy that Remove duplicates
      // already folded away here. Matched last, after id and sourceId, so
      // importing a file can't bring removed duplicates back.
      var at = Object.create(null), atSource = Object.create(null), atBet = Object.create(null);
      existing.forEach(function (it, i) { at[it.id] = i; if (it.sourceId) atSource[it.sourceId] = i; if (atBet[trkDupKey(it)] === undefined) atBet[trkDupKey(it)] = i; });
      var merged = existing.slice();
      var added = 0, updated = 0, gradesKept = 0, regrade = 0;
      imported.forEach(function (it) {
        if (!it || !it.id) return;
        var i = at[it.id];
        if (i === undefined && it.sourceId) i = atSource[it.sourceId];
        if (i === undefined) i = atBet[trkDupKey(it)];
        if (i === undefined) {
          at[it.id] = merged.length;
          atBet[trkDupKey(it)] = merged.length;
          if (it.sourceId) atSource[it.sourceId] = merged.length;
          merged.push(it); added++;
          return;
        }
        var mine = merged[i], sameBet = mine.description === it.description;
        if ((Number(it.updatedAt) || 0) > (Number(mine.updatedAt) || 0)) {
          var next = JSON.parse(JSON.stringify(it));
          next.id = mine.id;   // one row per play in this browser, under the id it already has here
          if (!next.status && mine.status) {
            if (sameBet) next.status = mine.status; else regrade++;
          }
          merged[i] = next; updated++;
        } else if (!mine.status && it.status && sameBet) {
          mine.status = it.status; gradesKept++;
        }
      });
      saveTrk(merged);
      input.value = '';
      render();
      alert(added + ' play(s) added, ' + updated + ' updated with newer edits' +
        (gradesKept ? ', ' + gradesKept + ' grade(s) brought over' : '') + '.' +
        (regrade ? '\\n\\n' + regrade + ' play(s) had their line changed after being graded at the old number. They are now ungraded \\u2014 grade them at the new line.' : ''));
    };
    reader.readAsText(file);
  };

  window.__cfbTrack = function (play) {
    var items = loadTrk();
    var item = {
      id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
      description: play.description, date: play.date, type: play.type,
      price: play.price, edge: play.edge, stake: 50, status: null,
      unofficial: !!play.unofficial,
      modelSource: play.modelSource || null,
      modelVersion: play.modelVersion || null
    };
    if (play.kickoffIso) item.kickoffIso = play.kickoffIso;   // real kickoff, so the row can show it in your own time zone
    if (play.betCard) item.betCard = true;                    // was on the Bet Card for its market when tracked
    items.unshift(item);
    saveTrk(items);
    render();
    var el = document.getElementById('trk-gamelines-section') || document.getElementById('trk-section');
    if (el && typeof el.scrollIntoView === 'function') el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  // + My list on a Props-tab row. Same description format the auto-added
  // official props use, so a play can't land in the list twice.
  function propTrackLabel(r) {
    return playerLabel(r) + ' ' + r.market_name + ' ' + (r.model_lean === 'over' ? 'O' : 'U') + ' ' + r.line;
  }
  window.__cfbTrackProp = function (i) {
    var r = (window.__cfbPropRows || [])[i];
    if (!r) return;
    var desc = propTrackLabel(r);
    var items = loadTrk();
    if (items.some(function (it) { return it.type === 'Prop' && (it.description === desc || it.flaggedDesc === desc); })) return;
    items.unshift({
      id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
      description: desc, date: (r.start_time || '').slice(0, 10), kickoffIso: r.start_time || undefined, type: 'Prop',
      price: r.model_lean === 'over' ? r.over_price : r.under_price,
      edge: r.model_ev, stake: 50, status: null
    });
    saveTrk(items);
    render();
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
    // The date box is a date-and-time picker in your own time zone. Stored
    // both as the real moment (kickoffIso) and as the same "Sat 9/19,
    // 11:30PM UTC" text every other play carries.
    var when = dateEl.value ? new Date(dateEl.value) : null, dateText = '';
    if (when && !isNaN(when)) {
      var hh = when.getUTCHours(), mm = when.getUTCMinutes();
      dateText = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][when.getUTCDay()] + ' ' + (when.getUTCMonth() + 1) + '/' + when.getUTCDate() + ', ' +
        ((hh % 12) || 12) + ':' + (mm < 10 ? '0' : '') + mm + (hh < 12 ? 'AM' : 'PM') + ' UTC';
    } else { when = null; }
    var items = loadTrk();
    items.unshift({
      id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
      description: description,
      date: dateText,
      kickoffIso: when ? when.toISOString() : undefined,
      type: trkGuessType(description, typeEl.value),
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
        kickoffIso: r.start_time || undefined,
        type: 'Prop',
        price: price != null ? price : null,
        edge: r.model_ev,
        stake: 50, status: null, auto: true,
      });
      // Official unders carry a tag (10/2026), so the tag filter in My Player
      // Props shows their record apart from the overs.
      if (!isOver) items[0].tags = ['Rec under'];
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

  // One-click cleanup for the auto-tracked prop backlog (added 9/2026).
  // Before the top-5 official-prop filter, every pipeline run auto-logged
  // every prop that cleared a miscalibrated 3%-EV bar, leaving hundreds of
  // pending rows in localStorage. Removes ONLY auto-tracked Prop rows that
  // have no grade yet -- graded props (the real record) and anything added
  // manually via +TRK or the add-a-play form are never touched.
  window.__cfbTrkClearPendingAutoProps = function () {
    var items = loadTrk();
    var drop = items.filter(function (it) { return it.type === 'Prop' && it.auto && !it.status; });
    if (!drop.length) { alert('No ungraded auto-tracked props to remove.'); return; }
    if (!confirm('Remove ' + drop.length + ' ungraded auto-tracked prop(s)? Graded props and manually tracked plays stay.')) return;
    saveTrk(items.filter(function (it) { return !(it.type === 'Prop' && it.auto && !it.status); }));
    render();
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

  /* ---- Tracked plays you can edit in place (10/2026) ------------------
     Before this, a tracked play was one line of text ("MD +3.0 -- VT at
     MD") and the only way to fix a spread you actually got at a different
     number was to add the play again by hand. Now the spread, odds, stake,
     tags and notes are all editable on the row or in the Edit panel.

     The description stays the one stored source of truth for the pick and
     the line -- the grades file, the export and the prop de-duplication all
     read it -- so the helpers below read the pick and line back OUT of the
     description (trkParse) and write an edited one back INTO it in the same
     format (trkDescribe). Nothing in a browser's saved tracker is rewritten
     until a row is actually edited, so there is no migration to go wrong. */

  var TRK_TAG_PRESETS = ['Injury', 'Fade', 'Degen'];
  var TRK_NOTE_MAX = 500;

  function trkNum(v) {
    if (v === '' || v == null) return null;
    var n = Number(v);
    return isNaN(n) ? null : n;
  }
  // "+3.0", "-2.5" -- one decimal, like the Edge Board writes them -- but a
  // quarter-point or alt line keeps every digit ("+3.25"), so opening and
  // saving a play can never round its number.
  function trkFmtLine(v) {
    v = Number(v);
    return (v > 0 ? '+' : '') + (Math.round(v * 10) / 10 === v ? v.toFixed(1) : String(v));
  }
  function trkHasTag(tags, t) { t = String(t).toLowerCase(); return tags.some(function (x) { return String(x).toLowerCase() === t; }); }
  // Which kind of bet a hand-typed description is, when it's written in one
  // of the shapes trkParse reads: "TOL ML ..." / "O 55.5 ..." / "PITT -2.5 ...".
  // The add-a-play form uses it so a spread typed with the Type box still on
  // "Total" is saved as a spread, not as a total nothing can read.
  function trkGuessType(desc, chosen) {
    var cut = /\\s+(?:—|–|--)\\s+/.exec(desc), label = (cut ? desc.slice(0, cut.index) : desc).trim();
    if (/^\\S.*?\\s+ML$/i.test(label)) return 'Moneyline';
    if (/^(O|U|Over|Under)\\s*\\d+(?:\\.\\d+)?$/i.test(label)) return 'Total';
    if (/^\\S.*?\\s+([+-]?\\d+(?:\\.\\d+)?|PK)$/i.test(label)) return 'Spread';
    return chosen;
  }
  function trkFmtPrice(v) { return (v == null || v === '') ? '—' : ((Number(v) > 0 ? '+' : '') + v); }
  function trkTagsOf(it) { return Array.isArray(it.tags) ? it.tags : []; }
  function trkFind(items, id) { for (var i = 0; i < items.length; i++) if (items[i].id === id) return items[i]; return null; }

  // Pick + line + matchup, read back out of the stored description.
  //   game lines: "<pick> <line|ML> — <away> at <home>"   (also "--", "@", "vs")
  //   props:      "<team> <player> <market> <O|U> <line>"
  // ok is false when the text doesn't fit (a hand-typed description in some
  // other shape): the row then just shows the text as written, and the line
  // can't be edited on its own -- the Edit panel offers the whole
  // description instead.
  function trkParse(it) {
    var desc = String(it.description || ''), m;
    var out = { pick: null, line: null, away: null, home: null, head: null, tail: '', ok: false };
    if (it.type === 'Prop') {
      m = /^(.*?)\\s+([OU])\\s+(\\d+(?:\\.\\d+)?)$/.exec(desc);
      if (m) { out.head = m[1]; out.pick = m[2]; out.line = Number(m[3]); out.ok = true; }
      return out;
    }
    var cut = /\\s+(?:—|–|--)\\s+/.exec(desc);
    var label = cut ? desc.slice(0, cut.index) : desc;
    out.tail = cut ? desc.slice(cut.index + cut[0].length) : '';
    m = /^(\\S.*?)\\s+(?:at|@|vs\\.?)\\s+(\\S.*)$/i.exec(out.tail);
    if (m) { out.away = m[1].trim(); out.home = m[2].trim(); }
    if (it.type === 'Moneyline') {
      m = /^(\\S.*?)\\s+ML$/i.exec(label);
      if (m) { out.pick = m[1]; out.ok = true; }
    } else if (it.type === 'Total') {
      m = /^(O|U|Over|Under)\\s*(\\d+(?:\\.\\d+)?)$/i.exec(label);
      if (m) { out.pick = m[1].charAt(0).toUpperCase(); out.line = Number(m[2]); out.ok = true; }
    } else {
      m = /^(\\S.*?)\\s+([+-]?\\d+(?:\\.\\d+)?|PK)$/i.exec(label);
      if (m) { out.pick = m[1]; out.line = /^pk$/i.test(m[2]) ? 0 : Number(m[2]); out.ok = true; }
    }
    return out;
  }
  function trkLabel(kind, pick, line) {
    if (kind === 'Moneyline') return pick + ' ML';
    if (kind === 'Total' || kind === 'Prop') return pick + ' ' + Number(line);
    return pick + ' ' + trkFmtLine(line);
  }
  // Same text shapes trackPayload / propTrackLabel write, so an edited row
  // is indistinguishable from one tracked at that number in the first place.
  function trkDescribe(kind, p, pick, line) {
    if (kind === 'Prop') return p.head + ' ' + trkLabel(kind, pick, line);
    return trkLabel(kind, pick, line) + (p.tail ? ' — ' + p.tail : '');
  }

  function trkImplied(price) {
    price = Number(price);
    return price > 0 ? 100 / (price + 100) : -price / (-price + 100);
  }
  // What the play looked like when its edge was measured. Frozen onto the
  // row (orig*) the first time anything about the bet is changed, so every
  // later edit is worked out from the original numbers rather than stacking
  // rounding on top of the last edit.
  function trkBase(it, p) {
    var frozen = it.origEdge !== undefined;
    return {
      frozen: frozen,
      pick: frozen ? it.origPick : p.pick,
      line: frozen ? it.origLine : p.line,
      price: frozen ? it.origPrice : it.price,
      edge: frozen ? it.origEdge : it.edge
    };
  }
  // The edge at the number you actually bet, from the edge at the number
  // the play was flagged at. Returns { edge, stale }; stale means "this is
  // still the flagged edge, it could not be moved to your number".
  //   Spread: a point of line is a point of edge. Flagged PITT -3 at +9.0
  //     and bet -2.5 is +9.5; the other side at +3 is -9.0.
  //   Total:  same, in whichever direction helps the side you took.
  //   Moneyline: the edge is win chance minus the price's implied chance,
  //     so a different price moves it by the change in implied chance.
  //     Close, not exact -- the flagged edge had the book's margin taken
  //     out using BOTH sides' prices and only one is stored.
  //   Prop: the edge is expected return at the price. A new price on the
  //     same half-point line is exact; a different line needs the model's
  //     projection, which the tracker doesn't keep, so it stays stale.
  function trkEdgeAt(kind, base, pick, line, price) {
    var e = trkNum(base.edge);
    if (e == null) return { edge: base.edge == null ? null : base.edge, stale: false };
    var r1 = function (x) { return Math.round(x * 10) / 10; };
    var flipped = base.pick != null && pick != null && pick !== base.pick;
    if (kind === 'Spread') {
      if (base.line == null || line == null) return { edge: e, stale: false };
      return { edge: r1(flipped ? -e + line + base.line : e + line - base.line), stale: false };
    }
    if (kind === 'Total') {
      if (base.line == null || line == null) return { edge: e, stale: false };
      var modelOver = (base.pick === 'U' ? -e : e) + base.line - line;   // model total minus the new line
      return { edge: r1(pick === 'U' ? -modelOver : modelOver), stale: false };
    }
    var p0 = trkNum(base.price), p1 = trkNum(price);
    if (kind === 'Moneyline') {
      if (flipped) return { edge: r1(-e), stale: false };
      if (p0 == null || p1 == null || p0 === p1) return { edge: e, stale: false };
      return { edge: r1(e + (trkImplied(p0) - trkImplied(p1)) * 100), stale: false };
    }
    // Prop
    if (flipped || line !== base.line) return { edge: e, stale: true };
    if (p0 == null || p1 == null || p0 === p1) return { edge: e, stale: false };
    if (line == null || Math.round(line * 2) % 2 === 0) return { edge: e, stale: true };   // whole-number line can push
    return { edge: r1(((1 + e / 100) / americanToDecimal(p0) * americanToDecimal(p1) - 1) * 100), stale: false };
  }

  // Reads what was typed into a line / odds / stake box. Lenient about a
  // leading "+" and "pk", strict about the result being a real number.
  function trkCheck(kind, field, raw) {
    var txt = String(raw == null ? '' : raw).trim().replace(/^\\+/, '');
    var bad = function (msg) { return { ok: false, msg: msg }; };
    if (field === 'line' && kind === 'Spread' && /^pk$/i.test(txt)) return { ok: true, value: 0 };
    var n = trkNum(txt);
    if (field === 'line') {
      if (n == null) return bad(kind === 'Spread' ? 'Enter a spread, like -2.5' : 'Enter a line, like 55.5');
      if (kind === 'Spread' ? Math.abs(n) > 80 : (n < 0 || n > 2000)) return bad('That line looks off');
      return { ok: true, value: n };
    }
    if (field === 'price') {
      if (n == null || n !== Math.round(n) || Math.abs(n) < 100 || Math.abs(n) > 100000) return bad('Use American odds, like -110 or +120');
      return { ok: true, value: n };
    }
    if (n == null || n < 0) return bad('Enter a stake, like 50');
    return { ok: true, value: n };
  }

  // The one place a tracked play is changed. ch may carry any of: pick,
  // line, price, stake, tags, notes, unofficial, description. Every change
  // is stamped (updatedAt) so Import can tell which copy of a play is newer.
  function trkCommit(id, ch) {
    var items = loadTrk();
    var it = trkFind(items, id);
    if (!it) return false;
    var p = trkParse(it);
    var pick = ch.pick != null ? ch.pick : p.pick;
    var line = ch.line != null ? ch.line : p.line;
    var price = ch.price !== undefined ? ch.price : it.price;
    var betChanged = p.ok && (pick !== p.pick || (it.type !== 'Moneyline' && line !== p.line));
    var priceChanged = trkNum(price) !== trkNum(it.price);
    if (betChanged || priceChanged) {
      if (it.origEdge === undefined) {
        it.origPick = p.pick; it.origLine = p.line;
        it.origPrice = it.price == null ? null : it.price;
        it.origEdge = it.edge == null ? null : it.edge;
        if (it.type === 'Prop') it.flaggedDesc = it.description;   // keeps "in My list" on the Props tab pointing at this row
      }
      var res = trkEdgeAt(it.type, trkBase(it, p), pick, line, price);
      it.edge = res.edge;
      if (res.stale) it.edgeStale = true; else delete it.edgeStale;
      if (betChanged) {
        // The 20+ point fade rule follows the number you actually took.
        if (it.type === 'Spread') {
          var wasBig = Math.abs(p.line) >= 20, isBig = Math.abs(line) >= 20;
          if (isBig && !wasBig) it.unofficial = true;
          else if (wasBig && !isBig && it.unofficial) it.unofficial = false;
        }
        it.description = trkDescribe(it.type, p, pick, line);
      }
      it.price = price;
      // Edited back to exactly what was flagged: it's the original play again.
      if (pick === it.origPick && line === it.origLine && trkNum(price) === trkNum(it.origPrice)) {
        it.edge = it.origEdge;
        delete it.origPick; delete it.origLine; delete it.origPrice; delete it.origEdge; delete it.edgeStale; delete it.flaggedDesc;
      }
    }
    if (ch.description != null && !p.ok && String(ch.description).trim() && String(ch.description).trim() !== it.description) {
      it.description = String(ch.description).trim();
      // Rewritten by hand: whatever was frozen while the text was unreadable
      // (no pick, no line) says nothing about the play as it now reads.
      delete it.origPick; delete it.origLine; delete it.origPrice; delete it.origEdge; delete it.edgeStale; delete it.flaggedDesc;
    }
    if (ch.stake !== undefined) it.stake = ch.stake;
    if (ch.tags !== undefined) {
      var seen = {}, tags = [];
      (ch.tags || []).forEach(function (t) {
        t = String(t || '').trim().slice(0, 24);
        TRK_TAG_PRESETS.forEach(function (pre) { if (pre.toLowerCase() === t.toLowerCase()) t = pre; });   // "injury" is the Injury tag
        if (t && !seen[t.toLowerCase()]) { seen[t.toLowerCase()] = true; tags.push(t); }
      });
      if (tags.length) it.tags = tags; else delete it.tags;
    }
    if (ch.notes !== undefined) {
      var note = String(ch.notes || '').trim().slice(0, TRK_NOTE_MAX);
      if (note) it.notes = note; else delete it.notes;
    }
    if (ch.unofficial !== undefined) it.unofficial = !!ch.unofficial;
    if (ch.betCard !== undefined) { if (ch.betCard) it.betCard = true; else delete it.betCard; }
    if (ch.status !== undefined) it.status = ch.status;
    it.updatedAt = Date.now();
    saveTrk(items);
    return true;
  }

  /* ---- Duplicate plays (10/2026) ----------------------------------------
     The same bet can end up in the tracker twice: the September log that
     was restored after a browser lost its storage (SEED_TRACKER) sits next
     to the original rows on any browser that still had them, and a slate
     tracked on two laptops and then imported lands twice as well. Every
     copy counts in the record, so the record reads better than it is.

     Two rows are the same bet only when ALL of these match: type, pick,
     line, game, kickoff, and the edge the play was flagged at. A row whose
     line or odds were later edited is compared on what it was flagged at
     (orig*), so an edited copy still matches its untouched twin. The edge
     is part of the test on purpose: the same pick tracked once under the
     untrained model and once under the trained one are two different
     reads, and are left alone. */
  function trkDupKey(it) {
    var p = trkParse(it), base = trkBase(it, p), w = trkWhen(it), e = trkNum(base.edge);
    var bet = p.ok
      ? [base.pick != null ? base.pick : p.pick,
         it.type === 'Moneyline' ? 'ML' : Number(base.line != null ? base.line : p.line),
         p.head || (p.away && p.home ? p.away + '@' + p.home : p.tail)].join('|')
      : String(it.description || '').trim();
    return [it.type, bet.toLowerCase(), w.ts != null ? w.ts : String(it.date || '').trim().toLowerCase(), e == null ? '' : e].join('||');
  }
  // How much of YOUR input a row carries -- the copy that gets kept.
  function trkRichness(it) {
    return (it.updatedAt ? 4 : 0) + (it.notes ? 2 : 0) + (trkTagsOf(it).length ? 2 : 0) + (it.origEdge !== undefined ? 2 : 0) + (it.modelSource ? 1 : 0) + (it.status ? 1 : 0);
  }
  // { groups: [[keep, copy, ...], ...], extra: rows that would go, conflicts: sets left alone }
  // A set whose copies are graded differently (one a win, one a loss) is a
  // conflict: there is no telling which is right, so it is never touched.
  function trkFindDuplicates(items) {
    var by = Object.create(null), groups = [], conflicts = 0, extra = 0;
    items.forEach(function (it) { var k = trkDupKey(it); (by[k] = by[k] || []).push(it); });
    Object.keys(by).forEach(function (k) {
      var g = by[k];
      if (g.length < 2) return;
      var grades = {};
      g.forEach(function (it) { if (it.status) grades[it.status] = true; });
      if (Object.keys(grades).length > 1) { conflicts++; return; }
      g = g.slice().sort(function (a, b) { return trkRichness(b) - trkRichness(a); });   // stable: ties keep tracker order
      groups.push(g); extra += g.length - 1;
    });
    return { groups: groups, extra: extra, conflicts: conflicts };
  }
  // Folds each set into its kept copy, then drops the rest. Nothing a copy
  // knew is lost: a grade, tags, a note, which model made the play and the
  // kickoff time all carry over, and the bet stays unofficial if ANY copy
  // was (the restored September rows carry the 20+ point flag that the
  // originals were tracked before).
  function trkRemoveDuplicates() {
    var items = loadTrk(), found = trkFindDuplicates(items), gone = [];
    found.groups.forEach(function (g) {
      var keep = g[0];
      g.slice(1).forEach(function (o) {
        if (!keep.status && o.status) keep.status = o.status;
        if (o.unofficial) keep.unofficial = true;
        if (o.betCard) keep.betCard = true;
        if (!keep.notes && o.notes) keep.notes = o.notes;
        var tags = trkTagsOf(keep).slice();
        trkTagsOf(o).forEach(function (t) { if (!trkHasTag(tags, t)) tags.push(t); });
        if (tags.length) keep.tags = tags;
        ['modelSource', 'modelVersion', 'kickoffIso', 'sourceId'].forEach(function (f) { if (keep[f] == null && o[f] != null) keep[f] = o[f]; });
        gone.push(o);
      });
      // Stamped no older than any copy it absorbed, so a file exported or
      // prepared slightly "ahead" of this clock can't look newer than the
      // merged row and be re-applied over it on the next import.
      keep.updatedAt = Math.max.apply(null, [Date.now()].concat(g.map(function (x) { return Number(x.updatedAt) || 0; })));
    });
    // Dropped by identity of the row object, not by id: two copies could in
    // principle share an id, and dropping by id would take both.
    saveTrk(items.filter(function (it) { return gone.indexOf(it) === -1; }));
    return found;
  }
  function trkDuplicateBanner(items) {
    var found = trkFindDuplicates(items);
    if (!found.extra) return '';
    return '<div class="trk-dupes" role="status"><div><b>' + found.extra + ' ' + (found.extra === 1 ? 'bet is' : 'bets are') + ' in the tracker twice.</b> ' +
      'Each copy counts in the record below, so it reads better or worse than it really is.</div>' +
      '<button class="trk-btn is-primary" data-trk-act="dedupe">Remove duplicates</button></div>';
  }
  function trkDedupeClick() {
    var found = trkFindDuplicates(loadTrk());
    if (!found.extra) return render();
    var names = found.groups.slice(0, 6).map(function (g) { return '  ' + g[0].description; });
    if (found.groups.length > 6) names.push('  ...and ' + (found.groups.length - 6) + ' more');
    if (!confirm('Remove ' + found.extra + ' duplicate ' + (found.extra === 1 ? 'play' : 'plays') + '?\\n\\n' +
      'These bets are in the tracker more than once with the same pick, line, game and edge:\\n' + names.join('\\n') + '\\n\\n' +
      'One copy of each is kept, with its grade, tags and notes. If either copy was marked unofficial, the kept one stays unofficial.\\n\\n' +
      'A backup of the tracker downloads first.')) return;
    window.__cfbTrkExport();
    var done = trkRemoveDuplicates();
    render();
    alert('Removed ' + done.extra + ' duplicate ' + (done.extra === 1 ? 'play' : 'plays') + '.' +
      (done.conflicts ? '\\n\\n' + done.conflicts + ' set(s) were left alone because their copies are graded differently. Fix the grade on one, then remove duplicates again.' : ''));
  }

  // When the game is, for display and for sorting. New plays carry the real
  // kickoff (kickoffIso); older ones only have the text the dashboard
  // showed at the time ("Sat 9/19, 11:30PM UTC", or "2026-10-11" for a
  // prop), so that text is read back. Shown in your own time zone, like the
  // Props tab. Anything unreadable (a hand-typed date) is shown as typed.
  function trkWhen(it) {
    var raw = String(it.date || ''), d = null, dateOnly = false, m;
    if (it.kickoffIso) {
      d = new Date(it.kickoffIso);
    } else if ((m = /^(\\d{4})-(\\d\\d)-(\\d\\d)$/.exec(raw))) {
      d = new Date(+m[1], +m[2] - 1, +m[3]); dateOnly = true;
    } else if ((m = /(\\d{1,2})\\/(\\d{1,2}),\\s*(\\d{1,2}):(\\d\\d)\\s*(AM|PM)\\s*UTC/i.exec(raw))) {
      // No year in that text. A play is tracked close to its game, so use
      // the year it was tracked in (the first 13 digits of its id are the
      // moment it was added), stepping a year for a bowl game in January.
      var added = parseInt(String(it.id || '').slice(1, 14), 10);
      var ref = added > 1e12 ? new Date(added) : new Date();
      var yr = ref.getUTCFullYear(), mo = +m[1], refMo = ref.getUTCMonth() + 1;
      if (mo - refMo > 6) yr--; else if (refMo - mo > 6) yr++;
      d = new Date(Date.UTC(yr, mo - 1, +m[2], (+m[3] % 12) + (/pm/i.test(m[5]) ? 12 : 0), +m[4]));
    }
    if (!d || isNaN(d)) return { top: raw, sub: '', ts: null };
    var days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    var months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    var h = d.getHours(), mi = d.getMinutes();
    return {
      top: days[d.getDay()] + ' ' + months[d.getMonth()] + ' ' + d.getDate(),
      sub: dateOnly ? '' : ((h % 12) || 12) + ':' + (mi < 10 ? '0' : '') + mi + ' ' + (h < 12 ? 'AM' : 'PM'),
      ts: d.getTime()
    };
  }

  // Team logo for a tracked play, by the abbreviation the play was saved
  // with. Real school logos (CFBD's), falling back to the same color-
  // accurate helmet the rest of the dashboard draws when there is no logo
  // for that team or the image won't load.
  var trkLogoBad = {};
  function trkTeam(abbr) { return (D.teamDir || {})[abbr] || null; }
  function trkHelmet(abbr, size) {
    var t = trkTeam(abbr);
    return helmetSvg(t && t.p ? t.p : '#8A94A3', t && t.s ? t.s : '#E7EDF5', Math.round(size * 1.3), Math.round(size * 0.77), false);
  }
  function trkMark(abbr, size) {
    var t = trkTeam(abbr);
    if (t && t.l && !trkLogoBad[t.l]) {
      return '<img class="trk-logo" src="' + esc(t.l) + '" alt="" width="' + size + '" height="' + size + '" data-trk-abbr="' + esc(abbr) + '" data-trk-size="' + size + '">';
    }
    return trkHelmet(abbr, size);
  }
  function trkTeamHtml(abbr, size) {
    var t = trkTeam(abbr);
    return '<span class="trk-team"' + (t && t.n ? ' title="' + esc(t.n) + '"' : '') + '>' + trkMark(abbr, size) + '<b>' + esc(abbr) + '</b></span>';
  }
  function trkMatchHtml(it, p, size) {
    if (it.type === 'Prop') {
      if (!p.ok) return '<span class="trk-desc">' + esc(it.description) + '</span>';
      // "MINN Javon Tracy Receptions": the team tag leads when the model knew the team.
      var sp = p.head.indexOf(' '), first = sp > 0 ? p.head.slice(0, sp) : '';
      if (first && trkTeam(first)) return '<span class="trk-team" title="' + esc(trkTeam(first).n || first) + '">' + trkMark(first, size) + '<b>' + esc(first) + '</b></span><span class="trk-desc">' + esc(p.head.slice(sp + 1)) + '</span>';
      return '<span class="trk-desc">' + esc(p.head) + '</span>';
    }
    // Only when the pick reads too. A matchup with a pick that doesn't fit
    // ("Over 55.5 1H -- VT at MD") shows the whole text as written, since
    // the pick column has nothing to show for it.
    if (p.ok && p.away && p.home) return trkTeamHtml(p.away, size) + '<span class="trk-at">at</span>' + trkTeamHtml(p.home, size);
    return '<span class="trk-desc">' + esc(it.description) + '</span>';
  }

  /* ---- Tracker table: search, filters, and rows edited in place ------- */

  function trkPencil(size) {
    return '<svg viewBox="0 0 16 16" width="' + size + '" height="' + size + '" aria-hidden="true"><path d="M11.4 1.6l3 3L5 14H2v-3z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>';
  }
  var TRK_PENCIL = trkPencil(11);

  function trkFilterState(sec) {
    state.trkF = state.trkF || {};
    return state.trkF[sec] || (state.trkF[sec] = { q: '', status: 'all', tag: 'all', sort: 'added' });
  }
  function trkFiltering(f) { return !!f.q || f.status !== 'all' || f.tag !== 'all'; }
  function trkFiltered(items, f) {
    var q = (f.q || '').toLowerCase().trim();
    var out = items.filter(function (it) {
      if (f.status === 'pending' && it.status) return false;
      if ((f.status === 'win' || f.status === 'loss' || f.status === 'push') && it.status !== f.status) return false;
      if (f.status === 'official' && it.unofficial) return false;
      if (f.status === 'unofficial' && !it.unofficial) return false;
      if (f.status === 'betcard' && !it.betCard) return false;
      if (f.tag !== 'all' && trkTagsOf(it).indexOf(f.tag) === -1) return false;
      if (q) {
        var p = trkParse(it), names = [p.away, p.home].map(function (a) { var t = a && trkTeam(a); return t && t.n ? t.n : ''; });
        var hay = [it.description, it.type, it.notes || '', trkTagsOf(it).join(' '), names.join(' ')].join(' ').toLowerCase();
        if (hay.indexOf(q) === -1) return false;
      }
      return true;
    });
    if (f.sort === 'added') return out;   // the order plays were added in, newest first -- how the list has always read
    var key = f.sort === 'edge' ? function (it) { return trkNum(it.edge); }
      : f.sort === 'profit' ? function (it) { return computeProfit(it); }
      : function (it) { return trkWhen(it).ts; };
    var dir = f.sort === 'date_asc' ? 1 : -1;
    // Rows with nothing to sort on (no edge, ungraded, unreadable date) go last either way.
    return out.map(function (it, i) { return { it: it, k: key(it), i: i }; }).sort(function (a, b) {
      if (a.k == null && b.k == null) return a.i - b.i;
      if (a.k == null) return 1;
      if (b.k == null) return -1;
      return a.k === b.k ? a.i - b.i : (a.k - b.k) * dir;
    }).map(function (x) { return x.it; });
  }

  function trkEditorHtml(ed, label) {
    return '<div class="trk-editor" data-trk-editing>' +
      '<input id="trk-edit-input" class="trk-in' + (ed.msg ? ' is-bad' : '') + '" type="text" autocomplete="off" aria-label="' + label + '" value="' + esc(ed.value) + '">' +
      (ed.msg ? '<div class="trk-msg" role="alert">' + esc(ed.msg) + '</div>' : '') +
      '<div class="trk-editor-btns"><button class="trk-btn is-primary" data-trk-act="save">Save</button>' +
      '<button class="trk-btn" data-trk-act="cancel">Cancel</button></div></div>';
  }
  function trkTagChip(t) { return '<span class="trk-tag' + (t === 'Injury' ? ' is-injury' : '') + '">' + esc(t) + '</span>'; }

  function trkRowHtml(it) {
    var p = trkParse(it), base = trkBase(it, p), profit = computeProfit(it), when = trkWhen(it), id = esc(it.id);
    var ed = state.trkEdit && state.trkEdit.id === it.id ? state.trkEdit : null;
    var edgeUnit = it.type === 'Moneyline' ? 'pp' : (it.type === 'Prop' ? '%' : 'pt');
    var was = it.modelSource === 'manual' ? 'was ' : 'flagged ';
    var modelTag = it.modelSource === 'trained_model' ? (it.modelVersion === 'upgraded' ? 'UPGRADED' : 'TRAINED')
      : it.modelSource === 'preseason_prior' ? 'UNTRAINED'
      : it.modelSource === 'manual' ? 'MANUAL' : null;

    var tags = (modelTag ? '<span class="trk-tag' + (it.modelSource === 'trained_model' ? ' is-model' : '') + '">' + modelTag + '</span>' : '') +
      (it.betCard ? '<span class="trk-tag is-card" title="On the Bet Card when it was tracked">BET CARD</span>' : '') +
      (it.auto ? '<span class="trk-tag is-auto">AUTO</span>' : '') +
      trkTagsOf(it).map(trkTagChip).join('');
    var matchCell = '<div><div class="trk-match">' + trkMatchHtml(it, p, 22) + '</div>' +
      (tags ? '<div class="trk-tags">' + tags + '</div>' : '') +
      (it.notes ? '<div class="trk-note" title="' + esc(it.notes) + '">' + esc(it.notes) + '</div>' : '') + '</div>';

    var pickText = !p.ok ? '' : it.type === 'Prop' ? (p.pick === 'O' ? 'Over ' : 'Under ') + p.line : trkLabel(it.type, p.pick, p.line);
    var pickCell;
    if (ed && ed.field === 'line') {
      pickCell = trkEditorHtml(ed, it.type === 'Spread' ? 'Spread' : 'Line');
    } else if (!p.ok) {
      pickCell = '<span class="trk-dim">—</span>';
    } else if (it.type === 'Moneyline') {
      pickCell = '<span class="trk-chip is-static">' + esc(pickText) + '</span>';
    } else {
      pickCell = '<button class="trk-chip" data-trk-act="edit" data-trk-field="line" data-trk-id="' + id + '" title="Change the ' + (it.type === 'Spread' ? 'spread' : 'line') + ' to the number you got">' +
        '<span>' + esc(pickText) + '</span>' + TRK_PENCIL + '</button>';
    }
    if (p.ok && base.frozen && base.pick != null && (base.pick !== p.pick || base.line !== p.line)) {
      pickCell += '<div class="trk-was">' + was + esc(it.type === 'Prop' ? base.pick + ' ' + base.line : trkLabel(it.type, base.pick, base.line)) + '</div>';
    }

    var oddsCell = (ed && ed.field === 'price') ? trkEditorHtml(ed, 'Odds')
      : '<button class="trk-chip is-plain" data-trk-act="edit" data-trk-field="price" data-trk-id="' + id + '" title="Change the odds to the price you got">' +
        '<span>' + esc(trkFmtPrice(it.price)) + '</span>' + TRK_PENCIL + '</button>';
    if (base.frozen && trkNum(base.price) !== trkNum(it.price) && base.price != null) {
      oddsCell += '<div class="trk-was">' + was + esc(trkFmtPrice(base.price)) + '</div>';
    }

    var edgeCell = it.edge == null ? '<span class="trk-dim">—</span>'
      : '<span class="' + (it.edgeStale ? 'trk-dim' : (Number(it.edge) < 0 ? 'trk-neg' : '')) + '"' +
        (it.edgeStale ? ' title="Measured at the flagged line. A prop edge is not recalculated for a different line."' : '') + '>' +
        (it.edge > 0 ? '+' : '') + esc(it.edge) + edgeUnit + '</span>';
    if (it.edgeStale) edgeCell += '<div class="trk-was">at flagged line</div>';
    else if (base.frozen && base.edge != null && Number(base.edge) !== Number(it.edge)) {
      edgeCell += '<div class="trk-was">was ' + (base.edge > 0 ? '+' : '') + esc(base.edge) + edgeUnit + '</div>';
    }

    var grade = ['win', 'loss', 'push'].map(function (s) {
      return '<button class="trk-status-btn' + (it.status === s ? ' is-' + s : '') + '" data-trk-act="grade" data-trk-status="' + s + '" data-trk-id="' + id + '" ' +
        'aria-pressed="' + (it.status === s ? 'true' : 'false') + '" title="' + (s === 'win' ? 'Win' : s === 'loss' ? 'Loss' : 'Push') + '">' + s.charAt(0).toUpperCase() + '</button>';
    }).join('');

    return '<div class="trk-g trk-r' + (it.unofficial ? ' is-unofficial' : '') + '">' +
      '<div class="trk-when"><div>' + (when.top ? esc(when.top) : '<span class="trk-dim">\\u2014</span>') + '</div>' + (when.sub ? '<div class="trk-was">' + esc(when.sub) + '</div>' : '') + '</div>' +
      matchCell +
      '<div>' + pickCell + '</div>' +
      '<div class="trk-dim2">' + (it.type === 'Moneyline' ? 'ML' : esc(it.type)) + '</div>' +
      '<div><input id="trk-stake-' + id + '" class="trk-in trk-in--stake" type="number" min="0" step="5" value="' + esc(it.stake == null ? '' : it.stake) + '" data-trk-stake="' + id + '" aria-label="Stake"></div>' +
      '<div>' + oddsCell + '</div>' +
      '<div>' + edgeCell + '</div>' +
      '<div class="trk-status-btns">' + grade + '</div>' +
      '<div class="trk-profit" style="color:' + (profit === null ? 'var(--muted-4)' : (profit >= 0 ? 'var(--green)' : 'var(--red)')) + '">' + (profit === null ? '—' : fmtMoney(profit)) + '</div>' +
      '<div><button class="trk-pill ' + (it.unofficial ? 'is-unofficial' : 'is-official') + '" data-trk-act="official" data-trk-id="' + id + '" ' +
        'title="Click to switch. Unofficial plays stay in the list but are left out of the record.">' + (it.unofficial ? 'UNOFFICIAL' : 'OFFICIAL') + '</button></div>' +
      '<div class="trk-acts"><button class="trk-icon" data-trk-act="drawer" data-trk-id="' + id + '" aria-label="Edit play" title="Edit play: pick, tags, notes">' + trkPencil(13) + '</button>' +
        '<button class="trk-icon is-x" data-trk-act="remove" data-trk-id="' + id + '" aria-label="Remove play" title="Remove play">×</button></div>' +
    '</div>';
  }

  function renderTrackerSection(allItems, opts) {
    var f = trkFilterState(opts.sec);
    var tagsInUse = {};
    allItems.forEach(function (it) { trkTagsOf(it).forEach(function (t) { tagsInUse[t] = true; }); });
    if (f.tag !== 'all' && !tagsInUse[f.tag]) f.tag = 'all';   // the last play with that tag was edited or removed
    var items = trkFiltered(allItems, f);
    var filtering = trkFiltering(f);

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

    if (!allItems.length) {
      return head + '<div class="trk-empty">' + esc(opts.emptyMsg) + '</div>';
    }

    var sec = esc(opts.sec);
    var opt = function (v, label, cur) { return '<option value="' + esc(v) + '"' + (cur === v ? ' selected' : '') + '>' + esc(label) + '</option>'; };
    var tools = '<div class="trk-tools">' +
      '<input id="trk-q-' + sec + '" class="trk-search" type="search" placeholder="Search teams, tags or notes" autocomplete="off" value="' + esc(f.q) + '" data-trk-filter="q" data-trk-sec="' + sec + '" aria-label="Search tracked plays">' +
      '<select class="trk-sel" data-trk-filter="status" data-trk-sec="' + sec + '" aria-label="Filter by status">' +
        [['all', 'All statuses'], ['pending', 'Not graded yet'], ['win', 'Wins'], ['loss', 'Losses'], ['push', 'Pushes'], ['official', 'Official only'], ['unofficial', 'Unofficial only'], ['betcard', 'Bet Card plays']]
          .map(function (o) { return opt(o[0], o[1], f.status); }).join('') + '</select>' +
      '<select class="trk-sel" data-trk-filter="tag" data-trk-sec="' + sec + '" aria-label="Filter by tag">' +
        opt('all', Object.keys(tagsInUse).length ? 'All tags' : 'No tags yet', f.tag) +
        Object.keys(tagsInUse).sort().map(function (t) { return opt(t, t, f.tag); }).join('') + '</select>' +
      '<select class="trk-sel" data-trk-filter="sort" data-trk-sec="' + sec + '" aria-label="Sort plays">' +
        [['added', 'Sort: newest added'], ['date_desc', 'Sort: latest game first'], ['date_asc', 'Sort: earliest game first'], ['edge', 'Sort: biggest edge'], ['profit', 'Sort: biggest profit']]
          .map(function (o) { return opt(o[0], o[1], f.sort); }).join('') + '</select>' +
    '</div>';
    var banner = filtering
      ? '<div class="trk-filtered">Showing ' + items.length + ' of ' + allItems.length + ' plays. The record, profit and units below count only these. ' +
        '<button class="trk-linkbtn" data-trk-act="clearf" data-trk-sec="' + sec + '">Clear filters</button></div>'
      : '';

    if (!items.length) {
      return head + tools + banner + '<div class="trk-empty">No plays match these filters.</div>';
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

    var isProps = opts.sec === 'props';
    var table = '<div class="trk-table">' +
      '<div class="trk-g trk-h"><div>Date</div><div>' + (isProps ? 'Player' : 'Matchup') + '</div><div>' + (isProps ? 'Pick / Line' : 'Pick / Spread') + '</div><div>Type</div><div>Stake</div><div>Odds</div><div>Edge</div><div>Grade</div><div>Profit</div><div>Status</div><div></div></div>' +
      items.map(trkRowHtml).join('') +
    '</div>';

    return head + tools + banner + summary + unitsHtml + table;
  }

  /* ---- Edit panel: everything about one play, in one place ------------
     Slides in over the right edge. Lives outside #app (its own node on
     <body>) and keeps what's been typed in trkDraft, so the page redrawing
     underneath never wipes a half-typed note. Nothing is saved until
     Save changes. */
  var trkDraft = null;

  function trkDraftCtx() {
    var it = trkDraft && trkFind(loadTrk(), trkDraft.id);
    if (!it) return null;
    var p = trkParse(it);
    return { it: it, p: p, base: trkBase(it, p), hasLine: p.ok && it.type !== 'Moneyline' };
  }
  // The typed line and odds, checked. An empty odds box is fine on a play
  // that never had a price (some props are flagged before one is posted).
  function trkDraftVals(c) {
    var line = c.hasLine ? trkCheck(c.it.type, 'line', trkDraft.line) : { ok: true, value: c.p.line };
    var price = (String(trkDraft.price).trim() === '' && c.it.price == null) ? { ok: true, value: null } : trkCheck(c.it.type, 'price', trkDraft.price);
    var stake = (String(trkDraft.stake).trim() === '' && (c.it.stake == null || c.it.stake === '')) ? { ok: true, value: undefined } : trkCheck(c.it.type, 'stake', trkDraft.stake);
    return { line: line, price: price, stake: stake };
  }
  // Official or not as it will be saved: your own switch if you touched it,
  // otherwise whatever the 20+ point rule says about the line now typed.
  function trkDraftUnofficial(c, v) {
    if (trkDraft.unofficialTouched) return trkDraft.unofficial;
    if (c.it.type === 'Spread' && c.p.ok && v.line.ok) {
      var wasBig = Math.abs(c.p.line) >= 20, isBig = Math.abs(v.line.value) >= 20;
      if (isBig && !wasBig) return true;
      if (wasBig && !isBig) return false;
    }
    return !!c.it.unofficial;
  }
  function trkPreviewHtml(c) {
    var v = trkDraftVals(c), kind = c.it.type, unit = kind === 'Moneyline' ? 'pp' : (kind === 'Prop' ? '%' : 'pt');
    if (!v.line.ok || !v.price.ok) return '';
    var sg = function (x) { return (x > 0 ? '+' : '') + esc(x) + unit; };
    var res = trkEdgeAt(kind, c.base, trkDraft.pick, v.line.value, v.price.value);
    var moved = c.p.ok && c.base.pick != null && (trkDraft.pick !== c.base.pick || (c.hasLine && v.line.value !== c.base.line));
    var repriced = trkNum(v.price.value) !== trkNum(c.base.price);
    var out = [];
    if (res.edge != null) {
      var baseLabel = !c.p.ok ? '' : kind === 'Prop' ? c.base.pick + ' ' + c.base.line : trkLabel(kind, c.base.pick, c.base.line);
      out.push('<div>Edge <b' + (Number(res.edge) < 0 ? ' class="trk-neg"' : '') + '>' + sg(res.edge) + '</b>' +
        ((moved || repriced) && c.base.edge != null
          ? ' <span class="trk-was">' + sg(c.base.edge) + ' when flagged' + (baseLabel ? ' at ' + esc(baseLabel) : '') + (c.base.price != null ? ' ' + esc(trkFmtPrice(c.base.price)) : '') + '</span>'
          : '') + '</div>');
    }
    if (res.stale) out.push('<div class="trk-was">A prop edge can’t be moved to a different line here, so it stays at the flagged number.</div>');
    else if (kind === 'Moneyline' && repriced && res.edge != null && trkDraft.pick === c.base.pick) out.push('<div class="trk-was">Moneyline edge at a new price is close, not exact.</div>');
    if (kind === 'Spread' && c.p.ok && !trkDraft.unofficialTouched) {
      var wasBig = Math.abs(c.p.line) >= 20, isBig = Math.abs(v.line.value) >= 20;
      if (isBig && !wasBig) out.push('<div class="trk-was">20+ point spreads are faded by rule, so this saves as unofficial.</div>');
      else if (wasBig && !isBig && c.it.unofficial) out.push('<div class="trk-was">Under 20 points now, so this saves as official.</div>');
    }
    return out.join('');
  }
  function trkSwitchHtml(c) {
    var off = trkDraftUnofficial(c, trkDraftVals(c));
    return '<button id="trk-d-official" class="trk-switch' + (off ? '' : ' is-on') + '" role="switch" aria-checked="' + (off ? 'false' : 'true') + '" data-trk-act="dofficial">' +
      '<span class="trk-switch-knob"></span></button><span class="trk-switch-label">' + (off ? 'Unofficial — left out of the record' : 'Official — counts in the record') + '</span>';
  }

  function trkCardSwitchHtml() {
    var on = !!trkDraft.betCard;
    return '<button id="trk-d-betcard" class="trk-switch is-plain' + (on ? ' is-on' : '') + '" role="switch" aria-checked="' + (on ? 'true' : 'false') + '" data-trk-act="dbetcard">' +
      '<span class="trk-switch-knob"></span></button><span class="trk-switch-label">' + (on ? 'On the Bet Card when it was tracked' : 'Not a Bet Card play') + '</span>';
  }

  function trkDrawerHtml() {
    var c = trkDraftCtx();
    if (!c) return '';
    var it = c.it, p = c.p, d = trkDraft, when = trkWhen(it);
    var field = function (label, key, extra) {
      return '<label class="trk-d-field"><span>' + label + '</span>' +
        '<input id="trk-d-' + key + '" class="trk-in' + (d.msg[key] ? ' is-bad' : '') + '" type="text" autocomplete="off" value="' + esc(d[key]) + '" data-trk-d="' + key + '"' + (extra || '') + '>' +
        (d.msg[key] ? '<div class="trk-msg" role="alert">' + esc(d.msg[key]) + '</div>' : '') + '</label>';
    };
    var pickField = '';
    if (p.ok) {
      var opts = (it.type === 'Total' || it.type === 'Prop') ? [['O', 'Over'], ['U', 'Under']]
        : [p.away, p.home].filter(function (a) { return !!a; }).map(function (a) { var t = trkTeam(a); return [a, a + (t && t.n ? ' — ' + t.n : '')]; });
      if (!opts.some(function (o) { return o[0] === d.pick; })) opts.unshift([d.pick, d.pick]);
      pickField = '<label class="trk-d-field"><span>' + (it.type === 'Total' || it.type === 'Prop' ? 'Side' : 'Pick') + '</span>' +
        '<select id="trk-d-pick" class="trk-sel" data-trk-d="pick">' +
        opts.map(function (o) { return '<option value="' + esc(o[0]) + '"' + (o[0] === d.pick ? ' selected' : '') + '>' + esc(o[1]) + '</option>'; }).join('') +
        '</select></label>';
    }
    // Presets first, then any tag already used on some other play, then this play's own.
    var pool = TRK_TAG_PRESETS.slice(), seen = {};
    loadTrk().forEach(function (x) { trkTagsOf(x).forEach(function (t) { pool.push(t); }); });
    d.tags.forEach(function (t) { pool.push(t); });
    var chips = pool.filter(function (t) { var k = t.toLowerCase(); if (seen[k]) return false; seen[k] = true; return true; })
      .map(function (t, i) {
        var on = trkHasTag(d.tags, t);
        return '<button id="trk-d-tag-' + i + '" class="trk-tagbtn' + (on ? ' is-on' : '') + (t === 'Injury' ? ' is-injury' : '') + '" aria-pressed="' + (on ? 'true' : 'false') + '" data-trk-act="dtag" data-trk-tag="' + esc(t) + '">' + esc(t) + '</button>';
      }).join('');

    return '<div class="trk-backdrop" data-trk-act="dclose"></div>' +
      '<aside class="trk-drawer" role="dialog" aria-label="Edit play">' +
        '<div class="trk-d-head"><h3>Edit play</h3><button class="trk-icon is-x" data-trk-act="dclose" aria-label="Close">×</button></div>' +
        '<div class="trk-d-body">' +
          '<div class="trk-d-game"><div class="trk-match">' + trkMatchHtml(it, p, 30) + '</div>' +
            '<div class="trk-was">' + esc(it.type) + (when.top ? ' · ' + esc(when.top) : '') + (when.sub ? ', ' + esc(when.sub) : '') + '</div></div>' +
          (p.ok ? '' : field('Description', 'description')) +
          pickField +
          (c.hasLine ? field(it.type === 'Spread' ? 'Spread' : 'Line', 'line') : '') +
          '<div class="trk-d-two">' + field('Odds', 'price') + field('Stake', 'stake') + '</div>' +
          '<div id="trk-d-preview" class="trk-d-preview">' + trkPreviewHtml(c) + '</div>' +
          '<div class="trk-d-field"><span>Tags</span><div class="trk-tagrow">' + chips + '</div>' +
            '<div class="trk-d-addtag"><input id="trk-d-newTag" class="trk-in" type="text" maxlength="24" autocomplete="off" placeholder="Add your own tag" value="' + esc(d.newTag) + '" data-trk-d="newTag" aria-label="New tag">' +
            '<button class="trk-btn" data-trk-act="dtagadd">Add</button></div></div>' +
          '<div class="trk-d-field"><span>Official play</span><div id="trk-d-switch" class="trk-switchrow">' + trkSwitchHtml(c) + '</div></div>' +
          '<div class="trk-d-field"><span>Bet Card play</span><div id="trk-d-card" class="trk-switchrow">' + trkCardSwitchHtml() + '</div></div>' +
          '<label class="trk-d-field"><span>Notes</span>' +
            '<textarea id="trk-d-notes" class="trk-in" rows="4" maxlength="' + TRK_NOTE_MAX + '" placeholder="Why you took it, what moved, anything worth remembering" data-trk-d="notes">' + esc(d.notes) + '</textarea>' +
            '<div id="trk-d-count" class="trk-d-count">' + d.notes.length + '/' + TRK_NOTE_MAX + '</div></label>' +
        '</div>' +
        '<div class="trk-d-foot"><button class="trk-btn is-danger" data-trk-act="ddelete">Remove play</button><span style="flex:1"></span>' +
          '<button class="trk-btn" data-trk-act="dclose">Cancel</button><button class="trk-btn is-primary" data-trk-act="dsave">Save changes</button></div>' +
      '</aside>';
  }
  function trkDrawerRender(focusId) {
    var root = document.getElementById('trk-drawer-root');
    if (!root) { root = document.createElement('div'); root.id = 'trk-drawer-root'; document.body.appendChild(root); }
    root.innerHTML = trkDraft ? trkDrawerHtml() : '';
    if (!root.innerHTML) trkDraft = null;
    var el = focusId && document.getElementById(focusId);
    if (el) el.focus();
  }
  // Just the live read-out and the official switch -- redrawing the whole
  // panel on every keystroke would drop the cursor out of the box being typed in.
  function trkDrawerRefresh() {
    var c = trkDraftCtx();
    if (!c) return;
    var pv = document.getElementById('trk-d-preview'), sw = document.getElementById('trk-d-switch'), n = document.getElementById('trk-d-count');
    if (pv) pv.innerHTML = trkPreviewHtml(c);
    if (sw) sw.innerHTML = trkSwitchHtml(c);
    if (n) n.textContent = trkDraft.notes.length + '/' + TRK_NOTE_MAX;
  }
  function trkOpenDrawer(id) {
    var it = trkFind(loadTrk(), id);
    if (!it) return;
    var p = trkParse(it);
    state.trkEdit = null;
    trkDraft = {
      id: id, pick: p.pick,
      line: p.line == null ? '' : (it.type === 'Spread' ? trkFmtLine(p.line) : String(p.line)),
      price: it.price == null ? '' : trkFmtPrice(it.price),
      stake: it.stake == null ? '' : String(it.stake),
      tags: trkTagsOf(it).slice(), notes: it.notes || '', newTag: '',
      unofficial: !!it.unofficial, unofficialTouched: false, betCard: !!it.betCard,
      description: it.description || '', msg: {}
    };
    render();
    trkDrawerRender(!p.ok ? 'trk-d-description' : (it.type === 'Moneyline' ? 'trk-d-price' : 'trk-d-line'));
  }
  function trkCloseDrawer() { trkDraft = null; trkDrawerRender(); }
  function trkDraftAddTag() {
    var t = String(trkDraft.newTag || '').trim().slice(0, 24);
    if (!t) return;
    if (!trkHasTag(trkDraft.tags, t)) trkDraft.tags.push(t);
    trkDraft.newTag = '';
    trkDrawerRender('trk-d-newTag');
  }
  function trkDrawerSave() {
    var c = trkDraftCtx();
    if (!c) return trkCloseDrawer();
    var v = trkDraftVals(c);
    trkDraft.msg = {};
    if (!v.line.ok) trkDraft.msg.line = v.line.msg;
    if (!v.price.ok) trkDraft.msg.price = v.price.msg;
    if (!v.stake.ok) trkDraft.msg.stake = v.stake.msg;
    if (!c.p.ok && !String(trkDraft.description).trim()) trkDraft.msg.description = 'Enter a description';
    var firstBad = ['description', 'line', 'price', 'stake'].filter(function (k) { return trkDraft.msg[k]; })[0];
    if (firstBad) return trkDrawerRender('trk-d-' + firstBad);
    var typed = String(trkDraft.newTag || '').trim();   // a tag typed but not yet added still counts
    var ch = { stake: v.stake.value, notes: trkDraft.notes, tags: typed ? trkDraft.tags.concat([typed]) : trkDraft.tags };
    if (c.p.ok) { ch.pick = trkDraft.pick; if (c.hasLine) ch.line = v.line.value; }
    else ch.description = trkDraft.description;
    if (v.price.value != null) ch.price = v.price.value;
    if (trkDraft.unofficialTouched) ch.unofficial = trkDraft.unofficial;
    if (!!trkDraft.betCard !== !!c.it.betCard) ch.betCard = !!trkDraft.betCard;
    trkCommit(trkDraft.id, ch);
    trkCloseDrawer();
    render();
  }

  /* ---- Tracker events -------------------------------------------------
     One listener per kind of event for every tracker control, keyed off
     data-trk-* attributes, since rows are redrawn on every change. */
  function renderKeep() {
    var a = document.activeElement, id = a && a.id, s = null, e = null;
    try { s = a.selectionStart; e = a.selectionEnd; } catch (x) {}
    render();
    var n = id && document.getElementById(id);
    if (n) { n.focus(); try { if (s != null) n.setSelectionRange(s, e); } catch (x) {} }
  }
  function trkFocusEditor() {
    var n = document.getElementById('trk-edit-input');
    if (n) { n.focus(); n.select(); }
  }
  function trkSaveInline() {
    var ed = state.trkEdit;
    if (!ed) return;
    var it = trkFind(loadTrk(), ed.id);
    if (!it) { state.trkEdit = null; return render(); }
    var chk = trkCheck(it.type, ed.field, ed.value);
    if (!chk.ok) { ed.msg = chk.msg; render(); return trkFocusEditor(); }
    var ch = {};
    ch[ed.field] = chk.value;
    trkCommit(ed.id, ch);
    state.trkEdit = null;
    render();
  }

  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-page]')) { state.trkEdit = null; if (trkDraft) trkCloseDrawer(); return; }   // leaving the tab drops a half-typed edit
    var el = e.target.closest('[data-trk-act]');
    if (!el) return;
    var act = el.dataset.trkAct, id = el.dataset.trkId, it;
    if (act === 'edit') {
      it = trkFind(loadTrk(), id);
      if (!it) return;
      var p = trkParse(it);
      state.trkEdit = { id: id, field: el.dataset.trkField, msg: '',
        value: el.dataset.trkField === 'price' ? (it.price == null ? '' : trkFmtPrice(it.price))
          : (it.type === 'Spread' ? trkFmtLine(p.line) : String(p.line)) };
      render();
      return trkFocusEditor();
    }
    if (act === 'save') return trkSaveInline();
    if (act === 'cancel') { state.trkEdit = null; return render(); }
    if (act === 'grade') {
      it = trkFind(loadTrk(), id);
      if (it) { trkCommit(id, { status: it.status === el.dataset.trkStatus ? null : el.dataset.trkStatus }); render(); }
      return;
    }
    if (act === 'official') {
      it = trkFind(loadTrk(), id);
      if (it) { trkCommit(id, { unofficial: !it.unofficial }); render(); }
      return;
    }
    if (act === 'remove') {
      it = trkFind(loadTrk(), id);
      if (it && confirm('Remove this play from the tracker?\\n\\n' + it.description)) window.__cfbTrkRemove(id);
      return;
    }
    if (act === 'drawer') return trkOpenDrawer(id);
    if (act === 'clearf') { var f = trkFilterState(el.dataset.trkSec); f.q = ''; f.status = 'all'; f.tag = 'all'; return render(); }
    if (act === 'addtoggle') { state.trkAddOpen = !state.trkAddOpen; return render(); }
    if (act === 'dedupe') return trkDedupeClick();
    if (!trkDraft) return;
    if (act === 'dclose') return trkCloseDrawer();
    if (act === 'dsave') return trkDrawerSave();
    if (act === 'ddelete') {
      it = trkFind(loadTrk(), trkDraft.id);
      if (it && confirm('Remove this play from the tracker?\\n\\n' + it.description)) { var rid = trkDraft.id; trkCloseDrawer(); window.__cfbTrkRemove(rid); }
      return;
    }
    if (act === 'dtag') {
      var t = el.dataset.trkTag;
      if (trkHasTag(trkDraft.tags, t)) trkDraft.tags = trkDraft.tags.filter(function (x) { return String(x).toLowerCase() !== t.toLowerCase(); });
      else trkDraft.tags.push(t);
      return trkDrawerRender(el.id);
    }
    if (act === 'dtagadd') return trkDraftAddTag();
    if (act === 'dbetcard') {
      trkDraft.betCard = !trkDraft.betCard;
      var cardRow = document.getElementById('trk-d-card');
      if (cardRow) cardRow.innerHTML = trkCardSwitchHtml();
      var cardBtn = document.getElementById('trk-d-betcard');
      if (cardBtn) cardBtn.focus();
      return;
    }
    if (act === 'dofficial') {
      var c = trkDraftCtx();
      if (!c) return;
      trkDraft.unofficial = !trkDraftUnofficial(c, trkDraftVals(c));
      trkDraft.unofficialTouched = true;
      trkDrawerRefresh();
      var sw = document.getElementById('trk-d-official');
      if (sw) sw.focus();
    }
  });

  document.addEventListener('input', function (e) {
    var el = e.target;
    if (el.id === 'trk-edit-input' && state.trkEdit) { state.trkEdit.value = el.value; return; }
    if (el.dataset && el.dataset.trkFilter === 'q') { trkFilterState(el.dataset.trkSec).q = el.value; return renderKeep(); }
    if (el.dataset && el.dataset.trkD && trkDraft && el.tagName !== 'SELECT') { trkDraft[el.dataset.trkD] = el.value; trkDrawerRefresh(); }
  });

  document.addEventListener('change', function (e) {
    var el = e.target;
    if (!el.dataset) return;
    if (el.dataset.trkFilter && el.tagName === 'SELECT') { trkFilterState(el.dataset.trkSec)[el.dataset.trkFilter] = el.value; return render(); }
    if (el.dataset.trkStake) {
      var chk = trkCheck('', 'stake', el.value);
      if (chk.ok) trkCommit(el.dataset.trkStake, { stake: chk.value });
      return renderKeep();
    }
    if (el.dataset.trkD === 'pick' && trkDraft) {
      var c = trkDraftCtx();
      // Switching sides on a spread flips the sign of the number in the box:
      // PITT -2.5 becomes UNC +2.5, which is almost always what was meant.
      if (c && c.it.type === 'Spread' && el.value !== trkDraft.pick) {
        var cur = trkCheck('Spread', 'line', trkDraft.line);
        if (cur.ok) trkDraft.line = trkFmtLine(-cur.value);
      }
      trkDraft.pick = el.value;
      trkDrawerRender('trk-d-pick');
    }
  });

  document.addEventListener('keydown', function (e) {
    var el = e.target;
    if (el.id === 'trk-edit-input' && state.trkEdit) {
      if (e.key === 'Enter') { e.preventDefault(); return trkSaveInline(); }
      if (e.key === 'Escape') { e.preventDefault(); state.trkEdit = null; return render(); }
      // Up / down arrows walk a spread or line by half a point.
      if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && state.trkEdit.field === 'line') {
        var it = trkFind(loadTrk(), state.trkEdit.id), cur = it && trkCheck(it.type, 'line', el.value);
        if (cur && cur.ok) {
          e.preventDefault();
          var nx = cur.value + (e.key === 'ArrowUp' ? 0.5 : -0.5);
          el.value = state.trkEdit.value = it.type === 'Spread' ? trkFmtLine(nx) : String(Math.max(0, nx));
        }
      }
      return;
    }
    if (!trkDraft) return;
    if (e.key === 'Escape') { e.preventDefault(); return trkCloseDrawer(); }
    if (e.key === 'Enter' && el.dataset && el.dataset.trkD && el.tagName === 'INPUT') {
      e.preventDefault();
      return el.dataset.trkD === 'newTag' ? trkDraftAddTag() : trkDrawerSave();
    }
  });

  // A logo that won't load (no logo on file under that link, or offline)
  // is swapped for the team's helmet, and not asked for again this visit.
  document.addEventListener('error', function (e) {
    var el = e.target;
    if (!el || el.tagName !== 'IMG' || !el.classList.contains('trk-logo')) return;
    trkLogoBad[el.getAttribute('src')] = true;
    var box = document.createElement('span');
    box.innerHTML = trkHelmet(el.dataset.trkAbbr, Number(el.dataset.trkSize) || 22);
    if (el.parentNode && box.firstChild) el.parentNode.replaceChild(box.firstChild, el);
  }, true);

  // The automatic prop log (10/2026). Every prop in a group that is up in
  // the backtest is logged the first time the model flags it and graded
  // from the box score -- nothing to click. Comes from the pipeline
  // (D.propTracking), not from this browser, so it is the same on every
  // device. Kept apart from the user's own plays so the two records
  // never mix.
  function renderAutoTrackedProps() {
    var T = D.propTracking;
    if (!T) return '';
    var sg = function (x, d) { return (x > 0 ? '+' : '') + Number(x).toFixed(d); };
    var cls = function (x) { return x > 0 ? 'is-pos' : (x < 0 ? 'is-neg' : ''); };
    var col = function (x) { return x > 0 ? 'var(--green)' : (x < 0 ? 'var(--red)' : 'var(--muted-3)'); };
    var kind = function (m, s) { return (m === 'Reception Yards' ? 'Receiving Yards' : m) + ' ' + s + 's'; };
    var price = function (x) { return x == null ? '' : (x > 0 ? '+' : '') + x; };
    var o = T.overall;
    var head = '<div class="section-head" id="trk-auto-section" style="margin-top:26px"><div class="section-title"><div class="section-flag is-green"></div><h2>Auto-Tracked Props</h2></div></div>' +
      '<div style="font-size:11px;color:var(--muted-3);line-height:1.55;margin:4px 0 12px">Automatic. Every prop whose kind of play is up in the backtest is logged the first time the model flags it, ' +
      'then graded from the box score at 1 unit, at the price it was flagged at. These are the model&rsquo;s plays, not bets you placed. Your own plays are in the other sections; use + My list on the Props tab to add any play to them.</div>';
    var summary = '<div class="trk-summary">' +
      '<div class="trk-box"><div class="trk-label">Record</div><div class="trk-value">' + esc(o.record) + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Units</div><div class="trk-value ' + cls(o.units) + '">' + sg(o.units, 2) + 'u</div></div>' +
      '<div class="trk-box"><div class="trk-label">ROI</div><div class="trk-value ' + cls(o.roi || 0) + '">' + (o.roi != null ? sg(o.roi, 1) + '%' : '&mdash;') + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Graded</div><div class="trk-value">' + o.graded + '</div></div>' +
      '<div class="trk-box"><div class="trk-label">Upcoming</div><div class="trk-value">' + o.upcoming + '</div></div>' +
    '</div>';

    var kc = 'grid-template-columns:1.5fr 0.7fr 0.7fr 0.7fr 0.7fr 1.5fr;';
    // The receptions rules, each on its own line (10/2026): live record since
    // the log started, next to the rule's record over every season tested.
    var R = D.propRules || {};
    var ruleRow = function (name, live, bt) {
      var g = live && live.graded;
      return '<div class="row"><div class="row-accent" style="background:' + (g ? col(live.units) : 'var(--rule)') + '"></div>' +
        '<div class="row-body" style="' + kc + 'padding:8px 14px;font-size:11.5px">' +
          '<div style="font-weight:700">' + name + '</div>' +
          '<div class="num">' + (g ? esc(live.record) : '&mdash;') + '</div>' +
          '<div class="num" style="color:' + (g ? col(live.units) : 'var(--muted-3)') + ';font-weight:700">' + (g ? sg(live.units, 2) + 'u' : '&mdash;') + '</div>' +
          '<div class="num" style="color:' + (g ? col(live.roi) : 'var(--muted-3)') + '">' + (g && live.roi != null ? sg(live.roi, 0) + '%' : '&mdash;') + '</div>' +
          '<div class="num">' + (g ? live.graded : 0) + '</div>' +
          '<div class="num" style="color:var(--muted-3);font-size:10.5px">' + (bt ? sg(bt.roi, 0) + '% &middot; ' + sg(bt.units, 1) + 'u &middot; ' + esc(bt.record) : '&mdash;') + '</div>' +
        '</div></div>';
    };
    var rules = '<div class="thead" style="display:grid;' + kc + 'margin-top:14px"><div>Receptions rule</div><div class="num">Record</div><div class="num">Units</div><div class="num">ROI</div><div class="num">Graded</div><div class="num">' +
        (R.rec_over ? R.rec_over.seasons + '-season backtest' : 'Backtest') + '</div></div>' +
      ruleRow('Official overs, 10% to 30% edge', T.official, R.rec_over) +
      ruleRow('Official unders, 30%+ edge', T.underPlay, R.rec_under) +
      ruleRow('Under leans, 10% to 30% edge', T.underLean, R.rec_under_lean);
    var kinds = '<div class="thead" style="display:grid;' + kc + 'margin-top:14px"><div>Kind of play</div><div class="num">Record</div><div class="num">Units</div><div class="num">ROI</div><div class="num">Upcoming</div><div class="num">Backtest</div></div>' +
      T.byKind.map(function (k) {
        var b = k.backtest;
        return '<div class="row"><div class="row-accent" style="background:' + (k.graded ? col(k.units) : 'var(--rule)') + '"></div>' +
          '<div class="row-body" style="' + kc + 'padding:8px 14px;font-size:11.5px">' +
            '<div style="font-weight:700">' + esc(kind(k.market, k.side)) + '</div>' +
            '<div class="num">' + (k.graded ? esc(k.record) : '&mdash;') + '</div>' +
            '<div class="num" style="color:' + (k.graded ? col(k.units) : 'var(--muted-3)') + ';font-weight:700">' + (k.graded ? sg(k.units, 2) + 'u' : '&mdash;') + '</div>' +
            '<div class="num" style="color:' + (k.graded ? col(k.roi) : 'var(--muted-3)') + '">' + (k.roi != null ? sg(k.roi, 0) + '%' : '&mdash;') + '</div>' +
            '<div class="num">' + k.upcoming + '</div>' +
            '<div class="num" style="color:var(--muted-3);font-size:10.5px">' + (b ? sg(b.roi, 0) + '% &middot; ' + sg(b.units, 1) + 'u &middot; ' + esc(b.record) : '&mdash;') + '</div>' +
          '</div></div>';
      }).join('');

    var pc = 'grid-template-columns:1.5fr 1.3fr 0.7fr 1.1fr;';
    var playRow = function (p) {
      var done = p.result === 'W' || p.result === 'L' || p.result === 'P';
      var rc = p.result === 'W' ? 'var(--green)' : (p.result === 'L' ? 'var(--red)' : (p.result === 'P' ? 'var(--amber)' : 'var(--rule)'));
      var res = done
        ? '<span class="pchip" style="color:' + rc + ';border:1px solid ' + rc + '">' + (p.result === 'W' ? 'WIN' : (p.result === 'L' ? 'LOSS' : 'PUSH')) + '</span>' +
          '<div style="font-size:9.5px;color:var(--muted-3);margin-top:3px">final ' + (p.actual != null ? Number(p.actual) : '?') + ' &middot; ' + sg(p.units || 0, 2) + 'u</div>'
        : '<span class="pchip is-lean">UPCOMING</span>';
      return '<div class="row"><div class="row-accent" style="background:' + rc + '"></div>' +
        '<div class="row-body" style="' + pc + 'padding:8px 14px;font-size:11.5px">' +
          '<div><div style="font-weight:700">' + esc(p.player) + '</div><div style="font-size:9.5px;color:var(--muted-3);margin-top:3px">' +
            esc(schoolAbbr(p.team) || '') + (p.opponent ? ' vs ' + esc(schoolAbbr(p.opponent)) : '') + ' &middot; ' + esc(propKickoff(p.start_time)) + '</div></div>' +
          '<div><div style="font-weight:700">' + esc(p.market) + ' ' + (p.side === 'over' ? 'O' : 'U') + ' ' + esc(p.line) + '</div>' +
            '<div style="font-size:9.5px;color:var(--muted-3);margin-top:3px">' + price(p.price) + '</div></div>' +
          '<div class="num">' + (p.model_ev != null ? sg(p.model_ev, 1) + '%' : '&mdash;') + '</div>' +
          '<div class="num">' + res + '</div>' +
        '</div></div>';
    };
    var playHead = function (title, note) {
      return '<div style="margin-top:16px;font-family:var(--font-display);font-weight:800;font-size:12px;letter-spacing:.06em;text-transform:uppercase">' + title +
        (note ? ' <span style="font-weight:400;color:var(--muted-3);text-transform:none;letter-spacing:0">' + note + '</span>' : '') + '</div>' +
        '<div class="thead" style="display:grid;' + pc + 'margin-top:6px"><div>Player</div><div>Pick</div><div class="num">Edge when flagged</div><div class="num">Result</div></div>';
    };
    var more = function (shown, total) { return total > shown ? '<div class="table-foot"><span>Showing ' + shown + ' of ' + total + '.</span></div>' : ''; };
    var results = T.results.length
      ? playHead('Results', 'newest first') + T.results.map(playRow).join('') + more(T.results.length, T.nResults)
      : '<div class="trk-empty" style="margin-top:14px">No graded plays yet. They appear here after their games finish and the next pipeline run.</div>';
    var showUp = !!state.trkAutoShowUpcoming;
    var upcoming = T.upcoming.length
      ? '<div class="prop-tabs" style="margin:16px 0 4px"><button class="tab tab--prop' + (showUp ? ' is-active' : '') + '" onclick="window.__cfbTrkAutoUpcoming()"><span>' +
          (showUp ? 'Hide' : 'Show') + ' ' + T.nUpcoming + ' upcoming</span></button></div>' +
        (showUp ? playHead('Upcoming', 'earliest kickoff first') + T.upcoming.map(playRow).join('') + more(T.upcoming.length, T.nUpcoming) : '')
      : '';
    var foot = '<div class="table-foot"><span>A few days of results is a very small sample, so the record will swing. Backtest = how the same kinds of play did before the log started.</span></div>';
    return head + summary + rules + kinds + results + upcoming + foot;
  }
  window.__cfbTrkAutoUpcoming = function () { state.trkAutoShowUpcoming = !state.trkAutoShowUpcoming; render(); };

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

    // Add-a-play form -- for anything not driven by a live model signal
    // (right now: Totals). Folded away behind a button (10/2026) now that a
    // tracked spread can be corrected on its own row, which is what the
    // form mostly got used for.
    var manualAddForm = !state.trkAddOpen ? '' : '<div class="trk-add">' +
      '<select id="trk-manual-type" class="trk-sel" aria-label="Type">' +
        '<option value="Spread">Spread</option>' +
        '<option value="Moneyline">Moneyline</option>' +
        '<option value="Total" selected>Total</option>' +
      '</select>' +
      '<input id="trk-manual-desc" class="trk-in" type="text" placeholder="O 55.5 -- VT at MD" style="flex:2 1 240px" aria-label="Play">' +
      '<input id="trk-manual-date" class="trk-in" type="datetime-local" style="flex:0 1 200px" aria-label="Kickoff, your time" title="Kickoff, in your own time zone">' +
      '<input id="trk-manual-price" class="trk-in" type="number" placeholder="Odds" value="-110" style="width:90px" aria-label="Odds">' +
      '<input id="trk-manual-edge" class="trk-in" type="number" step="0.1" placeholder="Edge (pts)" style="width:112px" aria-label="Edge in points">' +
      '<input id="trk-manual-stake" class="trk-in" type="number" placeholder="Stake" value="50" style="width:84px" aria-label="Stake">' +
      '<label style="display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted);padding:7px 2px">' +
        '<input id="trk-manual-unofficial" type="checkbox"> Unofficial' +
      '</label>' +
      '<button class="trk-btn is-primary" style="padding:7px 14px" onclick="window.__cfbTrkManualAdd()">Add play</button>' +
      '<div class="trk-add-hint">Write the pick, two dashes, then the game: <b>O 55.5 -- VT at MD</b>, <b>PITT -2.5 -- UNC at PITT</b> or <b>TOL ML -- TOL at MSU</b>. ' +
        'Written that way the row gets team logos and a line you can click to change. Anything else is kept exactly as typed.</div>' +
    '</div>';

    var exportImportBar = '<div class="prop-tabs" style="margin:12px 0 4px">' +
      '<button class="tab tab--prop' + (state.trkAddOpen ? ' is-active' : '') + '" data-trk-act="addtoggle" aria-expanded="' + (state.trkAddOpen ? 'true' : 'false') + '"><span>+ Add a play</span></button>' +
      '<button class="tab tab--prop" onclick="window.__cfbTrkExport()"><span>Export JSON</span></button>' +
      '<label class="tab tab--prop" style="cursor:pointer">' +
        '<span>Import JSON</span>' +
        '<input type="file" accept="application/json,.json" style="display:none" onchange="window.__cfbTrkImportFile(this)">' +
      '</label>' +
    '</div>';

    return '<div class="section-head" id="trk-section"><div class="section-title"><div class="section-flag is-green"></div><h2>Tracker</h2></div></div>' +
      exportImportBar +
      trkDuplicateBanner(items) +
      manualAddForm +
      (items.length ? '' : '<div class="trk-empty">No plays tracked yet. Click +TRK on any Edge Board row or Bet Card play, or check back after the next run — official player props get added here automatically.</div>') +
      renderTrackerSection(gameLineItemsShown, {
        title: 'Game Lines (Moneyline / Spread / Total)',
        emptyMsg: state.trackerModelTab === 'all'
          ? 'No game-line plays tracked yet. Click +TRK on any Edge Board row or Bet Card play.'
          : 'No ' + (state.trackerModelTab === 'trained_model' ? 'trained-model' : 'untrained-model') + ' plays tracked yet under this tab.',
        sectionId: 'trk-gamelines-section',
        sec: 'lines',
        extraHtml: modelTabBar,
      }) +
      renderAutoTrackedProps() +
      renderTrackerSection(propItems, {
        title: 'My Player Props',
        emptyMsg: 'No player-prop plays in your own list yet — official props get added here automatically after each pipeline run, and + My list on the Props tab adds any other play.',
        sectionId: 'trk-props-section',
        sec: 'props',
        extraHtml: '<div class="prop-tabs" style="margin:10px 0 4px">' +
          '<button class="tab tab--prop" onclick="window.__cfbTrkClearPendingAutoProps()"><span>Clear ungraded auto-props</span></button>' +
        '</div>',
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
      case 'injuries': return renderInjuries();
      case 'tracker': return renderTracker();
      case 'performance': return renderBacktest() + renderClv();
      case 'edge':
      default:
        return renderKpis(card, priced) +
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
          '<span>Team marks are school logos where one is on file (Edge Board, Projector, Bet Card, My Tracker) and a generic color-accurate helmet otherwise. Preseason: no in-season form exists yet for 2026, ' +
          'so every model number here comes from SP+ rating differential plus a fitted home-field constant, adjusted by any active ' +
          'manual injury/availability override (config/injury_overrides.csv \\u2014 hand-maintained, not scraped; no free CFB injury API ' +
          'exists). No Total/Team-total market, weather, travel, pace, returning-production, or futures data is fetched by this pipeline ' +
          '\\u2014 those are simply not shown rather than estimated. Sigma is one league-wide value, not per-game. ' +
          'Tracker is local to this browser only \\u2014 no sync, no real money moved. No model reliably beats a well-priced line on every game.</span>' +
        '</div>' +
      '</div>';
  }

  // renderKeep, not render: redrawing the page rebuilt the search box too,
  // so the cursor fell out of it after every single letter typed.
  window.__cfbSearch = function (v) { state.search = v; renderKeep(); };
  window.__cfbPropGame = function (v) { state.propGame = v; render(); };
  window.__cfbPropOfficial = function () { state.propOfficialOnly = !state.propOfficialOnly; render(); };
  window.__cfbPropShowPass = function () { state.propShowPass = !state.propShowPass; render(); };
  window.__cfbPropView = function (i) { state.propView = ['official', 'watch', 'all', 'underlean'][i] || 'official'; render(); };
  // Details panel on a Props-tab row: one open at a time.
  window.__cfbPropDetail = function (i) {
    var r = (window.__cfbPropRows || [])[i];
    if (!r) return;
    var k = [r.fixture_id, r.player_name, r.market_name].join('|');
    state.propDetail = state.propDetail === k ? null : k;
    state.propWhatIf = null;   // the "Try another line" box starts from the posted line each time
    render();
  };
  // "Try another line" in a prop's Details. Recalculates in place as the
  // line or price is typed, without redrawing the page, so the box you are
  // typing in keeps its focus.
  window.__cfbWhatIf = function () {
    var r = window.__cfbPropOpenRow, w = state.propWhatIf, out = document.getElementById('pp-wi-out');
    var lineEl = document.getElementById('pp-wi-line'), priceEl = document.getElementById('pp-wi-price');
    if (!r || !w || !out || !lineEl || !priceEl || !window.__cfbWhatIfHtml) return;
    w.line = lineEl.value;
    w.price = priceEl.value;
    out.innerHTML = window.__cfbWhatIfHtml(r, w.side, w.line, w.price);
  };
  window.__cfbWhatIfSide = function (i) {
    var w = state.propWhatIf;
    if (!w) return;
    w.side = i === 1 ? 'under' : 'over';
    ['over', 'under'].forEach(function (k) {
      var b = document.getElementById('pp-wi-' + k);
      if (b) { b.className = w.side === k ? 'is-on' : ''; b.setAttribute('aria-pressed', w.side === k ? 'true' : 'false'); }
    });
    window.__cfbWhatIf();
  };
  // Load one of the posted lines (its side, line and price) into the box.
  window.__cfbWhatIfUse = function (j) {
    var x = (window.__cfbPropOpenAlts || [])[j];
    var lineEl = document.getElementById('pp-wi-line'), priceEl = document.getElementById('pp-wi-price');
    if (!x || !lineEl || !priceEl) return;
    var p = x.model_lean === 'over' ? x.over_price : x.under_price;
    lineEl.value = x.line;
    priceEl.value = (p == null || isNaN(p)) ? '' : (p > 0 ? '+' : '') + Number(p);
    window.__cfbWhatIfSide(x.model_lean === 'under' ? 1 : 0);
  };
  window.__cfbGoTracker = function () { state.page = 'tracker'; render(); window.scrollTo(0, 0); };
  window.__cfbPropOpen = function (i) {
    var k = ['official', 'watch', 'tracked', 'lean'][i];
    state.propOpen = state.propOpen || {};
    state.propOpen[k] = !state.propOpen[k];
    render();
  };

  document.addEventListener('click', function (e) {
    var p = e.target.closest('[data-page]');
    if (p) { state.page = p.dataset.page; return render(); }
    var g = e.target.closest('[data-game]');
    if (g) { state.selected = +g.dataset.game; return render(); }
    var m = e.target.closest('[data-market]');
    if (m) { state.market = m.dataset.market; return render(); }
    if (e.target.closest('[data-edge-clear]')) { state.edgeShow = 'all'; state.edgeConf = 'ALL'; state.search = ''; return render(); }
    var pm = e.target.closest('[data-prop-market]');
    if (pm) { state.propMarket = pm.dataset.propMarket; return render(); }
    var pf = e.target.closest('[data-prop-fold]');
    if (pf) { state.propOpen = state.propOpen || {}; state.propOpen[pf.dataset.propFold] = !state.propOpen[pf.dataset.propFold]; return render(); }
    var us = e.target.closest('[data-unit-size]');
    if (us) { state.unitSize = Number(us.dataset.unitSize); return render(); }
    var mt = e.target.closest('[data-model-tab]');
    if (mt) { state.trackerModelTab = mt.dataset.modelTab; return render(); }
    var is = e.target.closest('[data-inj-scope]');
    if (is) { state.injScope = is.dataset.injScope; return render(); }
  });

  // Edge Board sort / filter dropdowns.
  document.addEventListener('change', function (e) {
    var k = e.target && e.target.dataset && e.target.dataset.edgeFilter;
    if (k === 'edgeSort' || k === 'edgeShow' || k === 'edgeConf') { state[k] = e.target.value; render(); }
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
    model_data["propEdgeHistory"] = build_prop_edge_history()
    model_data["propRules"] = build_prop_rules()
    attach_prop_first_flags(model_data.get("propsLive") or [])
    # Game-by-game box-score numbers for the Details panel (10/2026); absent
    # until scripts/export_dashboard_data.py writes them, and the panel just
    # leaves that part out.
    model_data["propGameLogs"] = data.get("prop_game_logs") or {}
    # What the "Try another line" box needs to price a line the feed does
    # not have (10/2026); same story -- absent until the export writes it.
    model_data["propDists"] = data.get("prop_distributions") or {}
    model_data["propTracking"] = build_prop_tracking(model_data["propEdgeHistory"])
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
