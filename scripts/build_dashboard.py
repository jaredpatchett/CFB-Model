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


# 11/2026: the spread model's inputs for each game on the slate, from the
# export's input log (the latest run), for the game breakdown's "why it's a
# play" read. {(home school, away school, kickoff iso): {input: value}}.
MODEL_INPUTS_LOG_PATH = "docs/data/model_inputs_log.csv"
GAME_INPUT_KEYS = ("home_field", "roll_margin_diff", "roll_ppg_for_diff", "roll_ppg_against_diff", "sp_rating_diff",
                   "pace_diff", "returning_production_diff", "core_overall_diff", "ewm_adj_margin_diff",
                   "ewm_net_ppa_diff", "ewm_off_ppa_diff", "ewm_def_ppa_diff", "ewm_off_sr_diff", "ewm_def_sr_diff",
                   "rest_diff", "home_qb_out", "away_qb_out", "temperature", "wind_speed", "precipitation")


def load_game_inputs(path: str = MODEL_INPUTS_LOG_PATH) -> dict:
    """Never raises; {} if the log is missing."""
    import csv
    try:
        if not os.path.exists(path):
            return {}
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        out, when = {}, {}
        for r in rows:
            if r.get("model_source") != "trained_model":
                continue
            k = (r.get("home"), r.get("away"), r.get("commence_time"))
            if r.get("run_at", "") < when.get(k, ""):
                continue
            vals = {}
            for c in GAME_INPUT_KEYS:
                try:
                    if r.get(c) not in (None, ""):
                        vals[c] = round(float(r[c]), 4)
                except ValueError:
                    pass
            when[k], out[k] = r.get("run_at", ""), vals
        return out
    except Exception as e:
        print(f"  [note] model input log not read ({e})")
        return {}


# 11/2026: line movement on flagged spread plays. The export logs the line
# every time a game qualifies as a flagged spread play (data/clv/
# line_snapshots.csv, also used for CLV). The first snapshot is the line
# when the model first flagged it; the Bet Card warns when the current line
# has moved against the pick since then.
LINE_SNAPSHOTS_PATH = "data/clv/line_snapshots.csv"


def load_line_history(path: str = LINE_SNAPSHOTS_PATH) -> dict:
    """{(home_team, away_team, commence_time): {first, at, side, edge, path}}.
    path = the line each time it changed. Never raises."""
    import csv
    try:
        if not os.path.exists(path):
            return {}
        with open(path, newline="") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: r.get("captured_at", ""))
        out = {}
        for r in rows:
            try:
                k = (r["home_team"], r["away_team"], r["commence_time"])
                line = float(r["spread_home_at_capture"])
            except (KeyError, ValueError, TypeError):
                continue
            e = out.get(k)
            if e is None:
                e = out[k] = {"first": line, "at": r.get("captured_at"), "side": r.get("side"),
                              "edge": round(float(r.get("edge_points_at_capture") or 0), 2), "path": []}
            if not e["path"] or e["path"][-1][1] != line:
                e["path"].append([r.get("captured_at"), line])
        return out
    except Exception as ex:
        print(f"  [note] line history not read ({ex})")
        return {}


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
    game_inputs = load_game_inputs()
    line_hist = load_line_history()
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
        # 11/2026: when each player first showed up on the report, so the
        # prop breakdown can tell which of a teammate's games came after.
        _since = {(i.get("school"), i.get("player")): i.get("reported") for i in (data.get("injuries") or [])}
        inj_home = [dict(i, since=_since.get((g.get("home_school"), i.get("player")))) for i in inj_home]
        inj_away = [dict(i, since=_since.get((g.get("away_school"), i.get("player")))) for i in inj_away]
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
            "awaySchool": g.get("away_school"),
            "homeSchool": g.get("home_school"),
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
            "flagLine": line_hist.get((g.get("home_team"), g.get("away_team"), g.get("commence_time"))),
            "injHome": inj_home,
            "injAway": inj_away,
            # 11/2026, game breakdown "why it's a play": what moves the
            # model's number (export) and its raw inputs (input log).
            "drivers": g.get("model_drivers"),
            "inputs": game_inputs.get((g.get("home_school"), g.get("away_school"), g.get("commence_time"))),
        })

    props_catalog = data.get("prop_market_catalog", [])
    props_live = data.get("props", [])
    # 11/2026: the spread Bet Card the previous run logged (scripts/card_watch.py),
    # so the card can tag NEW plays and list ones that dropped off.
    card_prev = None
    try:
        if os.path.exists("docs/data/card_log.json"):
            _runs = (json.load(open("docs/data/card_log.json")).get("runs") or [])
            if _runs:
                card_prev = {"at": _runs[-1].get("at"), "plays": _runs[-1].get("plays") or []}
    except Exception as e:
        print(f"  [note] card log not read ({e})")
    # The model's inputs ride along only on official props (game breakdown).
    for r in props_live:
        if not r.get("is_official_play"):
            r.pop("model_inputs", None)
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
            "overall": ((backtest.get("splits") or {}).get("overall") or []),
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
        # 11/2026 reskin: the latest flagged picks with their closing-line
        # value, and average CLV by week (the Edge Board's bottom panels).
        try:
            gl = [g for g in (clv.get("games") or []) if g.get("commence_time")]
            gl.sort(key=lambda g: g["commence_time"], reverse=True)
            clv_out["recent"] = [{"date": g["commence_time"], "home": g.get("home_team"), "away": g.get("away_team"),
                                  "side": g.get("side"), "line": g.get("recommended_spread_home"),
                                  "close": g.get("closing_spread_home"), "clv": g.get("clv_points")} for g in gl[:6]]
            wk = {}
            for g in gl:
                if g.get("clv_points") is not None and g.get("week") is not None:
                    wk.setdefault((g.get("season"), g["week"]), []).append(float(g["clv_points"]))
            clv_out["byWeek"] = [{"week": k[1], "avg": round(sum(v) / len(v), 3), "n": len(v)} for k, v in sorted(wk.items())]
        except Exception as e:
            print(f"  [note] CLV recent picks not built ({e})")

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
        "confBaseline": _load_conf_baseline(),
        "propCatalog": props_catalog,
        "propsLive": props_live,
        "cardPrev": card_prev,
        "fantasy": fantasy_out,
        "anytimeTd": data.get("anytime_td") or [],
        "anytimeTdNote": data.get("anytime_td_note"),
        "tdAllowed": data.get("td_allowed") or [],
        "tdAllowedNote": data.get("td_allowed_note"),
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
            # Every edge size of 0%+, not just the ones that finished up (10/23/2026):
            # counting only the winning sizes made every kind look better than it was.
            ups = [r for r in ((((history or {}).get("markets") or {}).get(market) or {}).get(side) or []) if r.get("lo", -1) >= 0]
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
                "underPlay": live.get("under_play"), "underLean": live.get("under_lean"),
                "recYdsPlay": live.get("rec_yds_play"), "rushUnderPlay": live.get("rush_under_play"), "byKind": by_kind,
                # 11/2/2026 receiving-model retrain: those rules' live records restart.
                "recYdsPlayNew": live.get("rec_yds_play_since_update"), "recYdsPlayPrev": live.get("rec_yds_play_prev"),
                "underPlayNew": live.get("under_play_since_update"), "underPlayPrev": live.get("under_play_prev"),
                "rushUnderPlayNew": live.get("rush_under_play_since_update"), "rushUnderPlayPrev": live.get("rush_under_play_prev"),
                "byMatchup": live.get("by_matchup_grade"), "byMatchupSince": live.get("by_matchup_grade_since"),
                "officialNow": live.get("official_now"), "officialPlays": (log.get("official_plays") or [])[:200],
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
# 11/2026: backtests rerun with played-zero games (ran routes, no box-score
# line) in training and grading. Used when present.
PROPS_LAB3_PZ_PATH = "docs/data/props_lab3_pz.json"
PROPS_LAB2_PZ_DETAIL = "docs/data/props_lab2_detail_pz.csv"
RUSH_LAB_PATH = "docs/data/rush_lab.json"


def build_prop_rules(path: str = None):
    """How each receptions RULE did over every season graded in props lab 3
    (this season's saved lines plus the past seasons pulled), for the Props
    tab's Track record column: {'rec_over': official overs, 10% to 30% edge;
    'rec_under': unders at 30%+; 'rec_under_lean': unders at 10% to 30%},
    each {record, units, roi, n, seasons}. None if the lab file is not there.
    Never raises -- a display extra."""
    try:
        path = path or (PROPS_LAB3_PZ_PATH if os.path.exists(PROPS_LAB3_PZ_PATH) else PROPS_LAB3_PATH)
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
        # 10/23/2026 official rules, from the edge history (this season's
        # backtest weekends only -- no past-season lines for these markets).
        try:
            from src.analysis import prop_edge_history as peh
            hist = (peh.build_history() or {}).get("markets") or {}

            def from_hist(market, side, lo):
                rows = [r for r in ((hist.get(market) or {}).get(side) or []) if r["lo"] >= lo]
                t = total(rows)
                if t:
                    t["seasons"] = 1
                return t
            out["rec_yds_over"] = from_hist("Reception Yards", "over", 15)
            out["rush_under"] = from_hist("Rush Yards", "under", 0)
        except Exception as e:
            print(f"  [note] new prop rules' backtest skipped ({e})")
        # Rec yds overs 15%+ on the retrained (played-zeros) model, this
        # season's lines -- the model now live.
        try:
            if os.path.exists(PROPS_LAB2_PZ_DETAIL):
                import pandas as pd
                d = pd.read_csv(PROPS_LAB2_PZ_DETAIL, low_memory=False)
                # 11/3/2026 rule: 15-30% edge on a line of 25.5+.
                d = d[(d["version"] == "pff") & (d["stat"] == "rec_yds") & (d["over_ev"] >= 0.15) & (d["over_ev"] < 0.30)
                      & (d["over_line"] >= 25.5) & d["over_res"].isin(["W", "L", "P"])]
                if len(d):
                    u = float(d["over_units"].sum())
                    out["rec_yds_over"] = {"record": f"{int((d['over_res'] == 'W').sum())}-{int((d['over_res'] == 'L').sum())}",
                                           "units": round(u, 1), "roi": round(u / len(d) * 100, 1), "n": int(len(d)), "seasons": 1}
        except Exception as e:
            print(f"  [note] retrained rec yds backtest not read ({e})")
        # Rush yds unders over 2024, 2025 and 2026 lines (rush lab, 11/3/2026),
        # the live (current) rush model, replacing the Sept-only number.
        try:
            if os.path.exists(RUSH_LAB_PATH):
                with open(RUSH_LAB_PATH) as f:
                    rl = json.load(f)
                cur = ((rl.get("results") or {}).get("current") or {})
                # 11/3/2026 rule: unders vs tough run defenses only.
                a = ((cur.get("by_run_defense") or {}).get("tough")) or {}
                if a.get("n"):
                    out["rush_under"] = {"record": a["record"], "units": a["units"], "roi": a["roi"], "n": a["n"],
                                         "seasons": len(rl.get("seasons") or [])}
        except Exception as e:
            print(f"  [note] rush lab not read ({e})")
        # The three official rules together.
        parts = [out.get(k) for k in ("rec_yds_over", "rush_under", "rec_under")]
        if all(parts):
            w = sum(int(x["record"].split("-")[0]) for x in parts)
            l = sum(int(x["record"].split("-")[1]) for x in parts)
            n = sum(x["n"] for x in parts)
            u = sum(x["units"] for x in parts)
            out["official_total"] = {"record": f"{w}-{l}", "units": round(u, 1), "roi": round(u / n * 100, 1), "n": n,
                                     "seasons": max(x["seasons"] for x in parts)}
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
<link href="https://fonts.googleapis.com/css2?family=Saira+Condensed:ital,wght@0,600;0,700;0,800;1,800;1,900&family=Azeret+Mono:wght@400;500;700&family=Doto:wght@600;800;900&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet" />

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
.tda-grid { grid-template-columns: 30px minmax(130px,1fr) minmax(150px,1.1fr) 40px 104px 74px 74px 74px 74px; }
.tdaw-grid { grid-template-columns: minmax(150px,1.1fr) minmax(190px,1.3fr) 92px 92px 140px 2.1fr; gap: 6px; }
.tda-atd { text-align: right; line-height: 1.35; padding-right: 14px; }
.tda-atd-book { font-size: 9.5px; color: var(--muted-3); letter-spacing: .05em; }
.tda-atd-best { font-size: 9.5px; color: var(--green); }
.tda-atd-sub { font-size: 9.5px; color: var(--muted-3); }
.tda-row { cursor: pointer; }
.tda-row:hover { background: var(--chip); }
.tda-row.is-open { background: var(--chip); }
.tda-rk { display: inline-block; min-width: 18px; margin-left: 4px; font-size: 9px; color: var(--muted-4); text-align: left; }
.tda-big { display: inline-block; min-width: 52px; text-align: center; padding: 2px 6px; border-radius: 3px; color: #0B0F15; font-weight: 800; }
.tda-search { background: var(--chip); border: 1px solid var(--rule-faint); color: var(--text); font-size: 11px; padding: 5px 9px; border-radius: 3px; width: 150px; }
.tda-detail { background: var(--panel-deep); border-bottom: 1px solid var(--rule); padding: 12px 18px 16px 44px; font-size: 11.5px; }
.tda-detail-title { font-family: var(--font-display); font-weight: 700; font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted-2); margin: 10px 0 6px; }
.tda-detail-grid { display: grid; grid-template-columns: 40px repeat(8, 1fr); gap: 4px; padding: 3px 0; border-bottom: 1px solid var(--rule-faint); }
.tda-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
.tda-log { display: grid; grid-template-columns: 44px 1.4fr 1fr 1.2fr; gap: 8px; padding: 3px 0; border-bottom: 1px solid var(--rule-faint); }
.tda-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.tda-chip { background: var(--chip); border-radius: 3px; padding: 2px 6px; font-size: 10.5px; white-space: nowrap; }
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
.conf-tier { display: block; margin-top: 3px; font-family: var(--font-body, inherit); font-size: 9px; font-weight: 800; letter-spacing: .06em; text-align: right; }
.conf-tier.is-high { color: var(--green); }
.conf-tier.is-mid { color: var(--amber); }
.conf-tier.is-low { color: var(--muted); }
.conf-grid { display: grid; grid-template-columns: 1.1fr 1fr .8fr .8fr; gap: 2px 10px; font-size: 13px; align-items: center; }
.conf-grid .h { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.conf-grid .r { text-align: right; font-variant-numeric: tabular-nums; }
.conf-mkt { font-size: 12px; font-weight: 800; margin: 10px 0 4px; color: var(--text, inherit); }
.conf-grid .conf-tier { text-align: left; margin-top: 0; font-size: 10px; }
.conf-live { font-size: 10px; color: var(--muted); display: block; }
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
.trk-tag.is-watch { color: var(--blue-light); border-color: var(--blue-light); background: rgba(90,160,255,0.14); }
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
.pp-grid { display: grid; grid-template-columns: minmax(190px, 1.4fr) minmax(190px, 1.2fr) minmax(130px, 0.8fr) minmax(120px, 0.75fr) minmax(170px, 1.1fr) minmax(120px, 0.8fr) 190px; align-items: center; column-gap: 16px; min-width: 1180px; }
.pp-mu { display: inline-flex; align-items: center; justify-content: center; min-width: 34px; font-family: var(--font-display); font-weight: 800; font-size: 15px; padding: 3px 8px; border-radius: 4px; border: 1px solid currentColor; }
.pp-mu.is-a { color: var(--green); background: rgba(23,194,107,0.12); }
.pp-mu.is-b { color: #8FD694; background: rgba(143,214,148,0.10); }
.pp-mu.is-c { color: var(--muted); background: var(--chip); border-color: var(--rule-strong); }
.pp-mu.is-d { color: var(--amber); background: rgba(224,180,74,0.12); }
.pp-mu.is-f { color: var(--salmon); background: rgba(255,82,82,0.12); }
.pp-mu-list { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 5px; font-size: 12.5px; color: var(--text-dim); }
.pp-mu-list b { font-weight: 700; }
.pp-head { font-family: var(--font-display); font-weight: 700; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted-2); padding: 11px 18px; border-bottom: 1px solid var(--rule); }
.pp-group { display: flex; align-items: baseline; gap: 10px; padding: 12px 18px 9px; background: var(--panel-deep); border-bottom: 1px solid var(--rule); border-top: 1px solid var(--rule); min-width: 1180px; }
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
.pp-more { padding: 10px 18px; border-bottom: 1px solid var(--rule-row); min-width: 1180px; }
.pp-more .pp-link { margin-top: 0; }
.pp-tag { font-family: var(--font-display); font-weight: 800; font-size: 11px; letter-spacing: 0.07em; padding: 3px 9px; border-radius: 4px; }
.pp-tag.is-official { background: rgba(23,194,107,0.16); color: var(--green); }
.pp-tag.is-watch { background: rgba(91,164,255,0.16); color: var(--blue-light); }
.pp-tag.is-newmodel { background: rgba(45,212,191,0.14); color: #2dd4bf; }
.pp-model { flex-basis: 100%; font-size: 11px; color: var(--muted); margin-top: 6px; }
.pp-model b { color: #2dd4bf; }
.pp-tag.is-track { background: transparent; color: var(--green); border: 1px solid rgba(23,194,107,0.4); }
.pp-tag.is-lean { background: var(--chip); color: var(--muted); }
.pp-tag.is-inj { background: rgba(224,180,74,0.16); color: var(--amber); }
.pp-tag.is-pass { background: rgba(255,82,82,0.14); color: var(--salmon); }
.pp-trk { font-family: var(--font-display); font-weight: 700; font-size: 11.5px; letter-spacing: 0.05em; background: transparent; color: var(--text-dim); border: 1px solid var(--rule-strong); border-radius: 4px; padding: 7px 12px; cursor: pointer; white-space: nowrap; }
.pp-trk:hover { border-color: var(--blue); color: var(--blue-light); }
.pp-inlist { font-family: var(--font-display); font-weight: 700; font-size: 11.5px; letter-spacing: 0.04em; color: var(--blue-light); white-space: nowrap; padding: 0 4px; }
.pp-lower { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; margin-top: 26px; align-items: start; }
/* Sort & Filter panel on the Props tab (10/2026) */
.pp-filter { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px 12px; margin: 0 0 14px; }
.pp-ftop { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; margin-bottom: 12px; }
.pp-ftitle { font-family: var(--font-display); font-weight: 800; font-size: 16px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text); }
.pp-fnote { font-size: 11.5px; color: var(--muted); }
.pp-fbtns { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.pp-frow { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px 18px; margin-top: 10px; }
.pp-fgroup { display: flex; flex-direction: column; gap: 5px; }
.pp-flabel { font-family: var(--font-display); font-weight: 700; font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-2); }
.pp-search { min-width: 220px; background: var(--chip); color: var(--text); border: 1px solid var(--rule-strong); border-radius: 5px; padding: 7px 10px; font-family: var(--font-data); font-size: 12.5px; }
.pp-search:focus-visible, .pp-select:focus-visible { outline: 2px solid var(--blue-light); outline-offset: 1px; }
.pp-chip .pp-cn { font-family: var(--font-data); font-weight: 400; font-size: 10.5px; opacity: 0.75; margin-left: 5px; }
.pp-chip.is-dim { opacity: 0.45; }
.pp-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
.pp-mkt { display: inline-block; font-family: var(--font-display); font-weight: 800; font-size: 10.5px; letter-spacing: 0.06em; padding: 2px 7px; border-radius: 3px; margin-right: 7px; vertical-align: 1px; white-space: nowrap; }
.pp-mkt.is-rec, .pp-dot.is-rec { background: rgba(32,201,201,0.16); color: #3FD6D6; }
.pp-mkt.is-rush, .pp-dot.is-rush { background: rgba(255,146,64,0.16); color: #FF9F5A; }
.pp-mkt.is-pass, .pp-dot.is-pass { background: rgba(167,123,255,0.18); color: #B796FF; }
.pp-mkt.is-other, .pp-dot.is-other { background: var(--chip); color: var(--muted); }
.pp-dot.is-rec { background: #3FD6D6; } .pp-dot.is-rush { background: #FF9F5A; } .pp-dot.is-pass { background: #B796FF; } .pp-dot.is-other { background: var(--muted); }
.pp-toggle { display: inline-flex; align-items: center; gap: 8px; font-family: var(--font-display); font-weight: 700; font-size: 12.5px; color: var(--muted); background: transparent; border: 1px solid var(--rule-strong); border-radius: 5px; padding: 6px 11px; cursor: pointer; }
.pp-toggle i { width: 26px; height: 14px; border-radius: 8px; background: var(--rule-strong); position: relative; }
.pp-toggle i::after { content: ''; position: absolute; top: 2px; left: 2px; width: 10px; height: 10px; border-radius: 50%; background: #fff; transition: left 0.15s; }
.pp-toggle.is-on { color: var(--blue-light); border-color: var(--blue); }
.pp-toggle.is-on i { background: var(--blue); }
.pp-toggle.is-on i::after { left: 14px; }
.pp-btn { font-family: var(--font-display); font-weight: 700; font-size: 12px; letter-spacing: 0.04em; border-radius: 5px; padding: 7px 13px; cursor: pointer; border: 1px solid var(--rule-strong); background: transparent; color: var(--text-dim); }
.pp-btn.is-primary { background: var(--blue); border-color: var(--blue); color: #fff; }
.pp-btn:focus-visible, .pp-toggle:focus-visible, .pp-sorth:focus-visible { outline: 2px solid var(--blue-light); outline-offset: 2px; }
.pp-saved { font-size: 11.5px; color: var(--green); }
.pp-sorth { all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 4px; }
.pp-sorth:hover, .pp-sorth.is-on { color: var(--blue-light); }
.pp-sorth .pp-arr { font-size: 9px; opacity: 0.6; }
.pp-sorth.is-on .pp-arr { opacity: 1; }
.pp-who { display: flex; align-items: center; gap: 10px; }
.pp-who .trk-logo, .pp-who svg { flex: none; }
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

/* ===================================================================
   11/2026 reskin ("CFB Edge" look): near-black + neon green, rounded
   cards, icon nav, no slanted shapes. Layered over the rules above so
   every page picks it up; page-specific layouts follow below.
   =================================================================== */
:root {
  --bg:            #07090A;
  --panel:         #0E1311;
  --panel-deep:    #0A0E0C;
  --row-active:    #121A16;
  --chip:          #141B18;
  --rule:          #1C2621;
  --rule-strong:   #26332C;
  --rule-faint:    #151D19;
  --rule-row:      #131A16;
  --text:          #F1F5F2;
  --text-dim:      #C9D3CD;
  --muted:         #8B9A92;
  --muted-2:       #6E7D75;
  --muted-3:       #64736B;
  --muted-4:       #4F5C55;
  --neon:          #39F07A;
  --neon-dim:      #1FB85A;
  --neon-glow:     rgba(57,240,122,0.35);
  --blue:          #2BD96B;
  --blue-light:    #5EF09A;
  --blue-link:     #6CF2A4;
  --green:         #39F07A;
  --font-body:     'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-num:      'JetBrains Mono', 'Azeret Mono', ui-monospace, monospace;
  --skew:          0deg;
  --radius:        12px;
  --radius-sm:     8px;
}
html, body { font-family: var(--font-body); background: radial-gradient(1200px 500px at 20% -10%, rgba(57,240,122,0.06), transparent 60%), var(--bg); background-attachment: fixed; }
::selection { background: rgba(57,240,122,0.3); }
a:hover { color: #B6FFD3; }

/* Top bar: brand + icon nav in one row */
.topbar { background: rgba(7,9,10,0.92); backdrop-filter: blur(10px); border-bottom: 1px solid var(--rule); }
.topbar-inner { height: 64px; gap: 22px; flex-wrap: nowrap; }
.brand { gap: 10px; flex-shrink: 0; }
.brand-mark { display: inline-flex; gap: 3px; transform: skewX(-18deg); }
.brand-mark i { display: block; width: 6px; height: 22px; background: var(--neon); border-radius: 1px; box-shadow: 0 0 12px var(--neon-glow); }
.brand-mark i:nth-child(2) { opacity: 0.55; }
.brand-word { font-family: var(--font-display); font-style: italic; font-weight: 900; font-size: 26px; letter-spacing: 0.02em; text-transform: uppercase; line-height: 1; }
.brand-word b { color: var(--neon); font-weight: 900; }
.brand-sub { font-size: 9px; letter-spacing: 0.14em; color: var(--muted-3); padding-left: 10px; border-left: 1px solid var(--rule-strong); }
.navx { display: flex; align-items: stretch; gap: 0; overflow-x: auto; scrollbar-width: none; flex: 1; justify-content: flex-start; height: 64px; min-width: 0; }
.navx::-webkit-scrollbar { display: none; }
.navx-item { display: inline-flex; align-items: center; gap: 6px; padding: 0 9px; background: none; border: none; border-bottom: 2px solid transparent; color: var(--muted); font-family: var(--font-body); font-weight: 600; font-size: 12.5px; cursor: pointer; white-space: nowrap; transition: color .15s, border-color .15s; }
.navx-item svg { width: 16px; height: 16px; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; opacity: 0.85; }
.navx-item:hover { color: var(--text); }
.navx-item.is-active { color: var(--neon); border-bottom-color: var(--neon); }
.page-nav { display: none; }
#search { background: var(--chip); border: 1px solid var(--rule-strong); border-radius: 999px; padding: 7px 14px; width: 170px; font-family: var(--font-body); }
@media (max-width: 1500px) { .brand-sub, .sync span:last-child { display: none; } }
@media (max-width: 1280px) { .navx-item span { display: none; } .navx-item { padding: 0 11px; } }
#search:focus { border-color: var(--neon); box-shadow: 0 0 0 3px rgba(57,240,122,0.12); }
.sync { font-size: 11px; font-family: var(--font-num); }
.sync-dot { background: var(--neon); box-shadow: 0 0 8px var(--neon); }

/* Buttons / tabs: rounded, no skew */
.tab, .tab--market, .tab--page { border-radius: var(--radius-sm); transform: none; font-family: var(--font-body); font-weight: 700; letter-spacing: 0.03em; text-transform: none; font-size: 13px; }
.tab > span { transform: none; }
.tab.is-active, .tab--market.is-active { background: var(--neon); color: #04130A; box-shadow: 0 0 16px rgba(57,240,122,0.25); }
.section-flag { transform: none; width: 4px; border-radius: 2px; background: var(--neon); box-shadow: 0 0 10px var(--neon-glow); }
.section-head { border-bottom-color: var(--rule); }
.section-head h2, .pp-ftitle, .pp-panel h3 { font-family: var(--font-display); font-style: italic; font-weight: 900; letter-spacing: 0.02em; }

/* Stat cards */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; background: none; border: none; margin: 20px 0 6px; }
.kpi { background: linear-gradient(180deg, #111814, #0C110F); border: 1px solid var(--rule); border-radius: var(--radius); padding: 16px 18px 14px; position: relative; overflow: hidden; }
.kpi-bar { display: none; }
.kpi-body { padding: 0; }
.kpi-label { font-family: var(--font-body); font-weight: 700; font-size: 10.5px; letter-spacing: 0.12em; color: var(--muted-2); margin-bottom: 8px; }
.kpi-value { font-family: var(--font-body); font-weight: 800; font-size: 30px; letter-spacing: -0.01em; text-shadow: none; display: flex; align-items: center; gap: 10px; }
.kpi.is-green .kpi-value, .kpi-value.is-neon { color: var(--neon); text-shadow: 0 0 18px rgba(57,240,122,0.25); }
.kpi-sub { font-size: 11px; color: var(--muted-3); margin-top: 8px; }
.kpi-pill { font-size: 12px; font-weight: 800; color: var(--neon); background: rgba(57,240,122,0.12); border: 1px solid rgba(57,240,122,0.3); border-radius: 999px; padding: 2px 9px; }
.kpi-spark { position: absolute; right: 14px; bottom: 14px; width: 92px; height: 34px; opacity: 0.95; }
.kpi-spark path.l { fill: none; stroke: var(--neon); stroke-width: 2; filter: drop-shadow(0 0 4px rgba(57,240,122,0.5)); }
.kpi-spark path.a { fill: url(#sparkfill); stroke: none; }
.kpi-spark rect { fill: var(--neon); opacity: 0.85; rx: 1; }
.kpi-icon { width: 34px; height: 34px; border-radius: 9px; display: inline-flex; align-items: center; justify-content: center; background: rgba(57,240,122,0.1); border: 1px solid rgba(57,240,122,0.25); color: var(--neon); }
.kpi-icon svg { width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }

/* Generic panels and rows get rounded corners */
.pp-panel, .pp-filters, .pp-recline, .pp-table, .matchup, .proj, .card, .trk-summary, .trk-box { border-radius: var(--radius); }
.row-accent { border-radius: 2px; }
.pp-tag, .pchip, .trk-tag, .flag-chip { border-radius: 999px; }
.pp-tag.is-official { background: rgba(57,240,122,0.14); color: var(--neon); border: 1px solid rgba(57,240,122,0.3); }
input, select, textarea { border-radius: var(--radius-sm) !important; }
.footer { color: var(--muted-4); }

/* Edge Matrix */
.em-sub { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted-3); font-weight: 700; margin-left: 6px; }
.em-wrap { margin-top: 18px; }
.em-table { background: var(--panel); border: 1px solid var(--rule); border-radius: var(--radius); overflow: hidden; margin-top: 12px; }
.em-grid { display: grid; grid-template-columns: 30px minmax(170px, 1.3fr) 112px 64px 64px 76px minmax(130px, 1fr) minmax(150px, 1.1fr) minmax(130px, 1fr) 52px; column-gap: 12px; align-items: center; }
.em-table .thead { border-bottom: 1px solid var(--rule); padding: 11px 16px; font-family: var(--font-body); letter-spacing: 0.1em; font-size: 10px; }
.em-row { border-bottom: 1px solid var(--rule-row); }
.em-row .row-body { padding: 9px 16px; }
.em-row:hover { background: #111a15; }
.em-row.is-selected { background: #122019; box-shadow: inset 3px 0 0 var(--neon); }
.em-n { color: var(--muted-3); font-family: var(--font-num); font-size: 12px; }
.em-game { display: flex; align-items: center; gap: 7px; white-space: nowrap; }
.em-game .team-abbr { font-family: var(--font-body); font-weight: 800; font-size: 13.5px; }
.em-game .at { color: var(--muted-4); font-size: 11px; }
.em-kick { font-size: 11.5px; color: var(--text-dim); line-height: 1.35; }
.em-kick div { color: var(--muted-3); font-size: 10.5px; }
.em-grid .cell-market, .em-grid .cell-model { font-family: var(--font-num); font-size: 13px; }
.em-grid .cell-model { color: var(--text); font-weight: 700; }
.em-edge { display: inline-block; min-width: 62px; text-align: center; font-family: var(--font-num); font-weight: 700; font-size: 14px; padding: 6px 8px; border-radius: 6px; }
.em-edge.is-pos { color: #EFFFF5; border: 1px solid rgba(57,240,122,0.45); }
.em-edge.is-neg { color: var(--red); background: rgba(255,82,82,0.1); }
.em-edge.is-off { color: var(--muted-4); }
.em-prob { display: flex; align-items: center; gap: 10px; }
.em-prob span { font-family: var(--font-num); font-size: 12.5px; min-width: 44px; }
.em-prob i { flex: 1; height: 7px; background: #1A2420; border-radius: 99px; overflow: hidden; }
.em-prob b { display: block; height: 100%; background: linear-gradient(90deg, #1FB85A, var(--neon)); border-radius: 99px; box-shadow: 0 0 8px rgba(57,240,122,0.4); }
.em-play { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--rule-strong); border-radius: 7px; padding: 6px 10px; font-weight: 700; font-size: 12.5px; color: var(--text); white-space: nowrap; max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
.em-play svg { width: 11px; height: 11px; fill: var(--muted); flex-shrink: 0; }
.em-play em { font-style: normal; color: var(--neon); font-weight: 800; }
.em-play.is-card { border-color: rgba(57,240,122,0.6); background: rgba(57,240,122,0.08); box-shadow: 0 0 14px rgba(57,240,122,0.12); }
.em-play.is-card svg { fill: var(--neon); }
.em-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.em-tag { font-size: 9.5px; font-weight: 800; letter-spacing: 0.07em; padding: 3px 7px; border-radius: 5px; border: 1px solid var(--rule-strong); color: var(--muted); white-space: nowrap; }
.em-tag.is-card { color: var(--neon); border-color: rgba(57,240,122,0.45); background: rgba(57,240,122,0.08); }
.em-tag.is-track { color: #8FD3FF; border-color: rgba(143,211,255,0.35); }
.em-tag.is-warn { color: var(--amber); border-color: rgba(224,180,74,0.45); }
.em-tag.is-inj { color: #FFB84D; border-color: rgba(255,184,77,0.45); background: rgba(255,184,77,0.07); }
.em-tag.is-src { color: var(--muted-3); }
.em-tag.is-src.is-trained { color: var(--muted); }
.em-below { margin-top: 22px; }
/* Edge Board bottom panels */
.ep-grid { display: grid; grid-template-columns: 1.15fr 1fr 1fr; gap: 14px; margin-top: 18px; }
.ep-card { background: var(--panel); border: 1px solid var(--rule); border-radius: var(--radius); padding: 14px 16px; position: relative; }
.ep-head { display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px; }
.ep-head svg { width: 16px; height: 16px; stroke: var(--neon); fill: none; stroke-width: 1.8; }
.ep-head em { margin-left: auto; font-style: normal; font-weight: 600; font-size: 10.5px; letter-spacing: 0.02em; text-transform: none; color: var(--muted-3); }
.ep-row { display: grid; grid-template-columns: 1.6fr 0.9fr 0.7fr 0.6fr; gap: 8px; font-size: 12px; padding: 6px 0; border-bottom: 1px solid var(--rule-row); align-items: center; }
.ep-row b { font-family: var(--font-num); font-weight: 700; text-align: right; }
.ep-row span { color: var(--text-dim); }
.ep-th span { font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-3); text-align: right; }
.ep-th span:first-child { text-align: left; }
.ep-pick { grid-template-columns: 0.5fr 1.4fr 1fr 0.6fr; }
.ep-date { color: var(--muted-3) !important; font-family: var(--font-num); }
.ep-clv { display: flex; justify-content: space-between; align-items: flex-end; margin-top: 12px; position: relative; min-height: 46px; }
.ep-clv .kpi-spark { position: static; }
.ep-k { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-3); font-weight: 700; }
.ep-v { font-size: 22px; font-weight: 800; margin-top: 3px; }
.ep-k2 { font-size: 11px; color: var(--muted-3); margin-top: 2px; }
.ep-notes { list-style: none; margin: 0; padding: 0; }
.ep-notes li { position: relative; padding: 6px 0 6px 14px; font-size: 12px; color: var(--text-dim); line-height: 1.45; border-bottom: 1px solid var(--rule-row); }
.ep-notes li::before { content: ''; position: absolute; left: 0; top: 12px; width: 6px; height: 6px; border-radius: 2px; background: var(--neon); }
.ep-notes b { color: var(--text); }
.ep-foot { margin-top: 10px; font-size: 11.5px; }
.ep-empty { color: var(--muted-3); font-size: 12px; padding: 10px 0; }
@media (max-width: 1100px) { .ep-grid { grid-template-columns: 1fr; } .em-table { overflow-x: auto; } .em-grid { min-width: 1060px; } }

/* Player Props reskin */
.pp-hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; margin: 24px 0 16px; flex-wrap: wrap; }
.pp-hero h1 { font-family: var(--font-display); font-style: italic; font-weight: 900; font-size: 40px; letter-spacing: 0.01em; text-transform: uppercase; margin: 0; line-height: 1; }
.pp-hero-sub { color: var(--muted); font-size: 13.5px; margin-top: 6px; }
.pp-hero-card { display: flex; gap: 0; background: linear-gradient(180deg, #111814, #0C110F); border: 1px solid var(--rule); border-radius: var(--radius); }
.pp-hero-card > div { padding: 12px 22px; border-left: 1px solid var(--rule); }
.pp-hero-card > div:first-child { border-left: none; }
.pp-hk { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted-2); }
.pp-hv { font-size: 24px; font-weight: 800; margin-top: 4px; font-family: var(--font-body); }
.pp-hv.is-neon { color: var(--neon); }
.pp-cats { display: flex; gap: 6px; background: var(--panel); border: 1px solid var(--rule); border-radius: var(--radius); padding: 6px; overflow-x: auto; }
.pp-cat { display: inline-flex; align-items: center; gap: 8px; background: none; border: none; color: var(--muted); font-family: var(--font-body); font-weight: 700; font-size: 12.5px; letter-spacing: 0.06em; text-transform: uppercase; padding: 10px 18px; border-radius: 9px; cursor: pointer; white-space: nowrap; }
.pp-cat svg { width: 16px; height: 16px; stroke: currentColor; fill: none; stroke-width: 1.8; }
.pp-cat em { font-style: normal; font-size: 11px; color: var(--muted-3); background: var(--chip); border-radius: 99px; padding: 1px 7px; }
.pp-cat:hover { color: var(--text); }
.pp-cat.is-on { background: var(--neon); color: #04130A; box-shadow: 0 0 18px rgba(57,240,122,0.25); }
.pp-cat.is-on em { background: rgba(4,19,10,0.18); color: #04130A; }
.pp-layout { display: grid; grid-template-columns: 270px minmax(0, 1fr); gap: 16px; margin-top: 16px; align-items: start; }
.pp-side { position: sticky; top: 78px; max-height: calc(100vh - 92px); overflow-y: auto; scrollbar-width: thin; }
.pp-side .pp-controls { margin: 0 0 10px; }
.pp-side .pp-controls .pp-seg { display: grid; grid-template-columns: 1fr 1fr; width: 100%; }
.pp-side .pp-filter { margin: 0 0 12px; padding: 14px; }
.pp-side .pp-ftop { flex-wrap: wrap; gap: 6px; }
.pp-side .pp-fbtns { width: 100%; display: flex; gap: 6px; }
.pp-side .pp-fbtns .pp-btn { flex: 1; }
.pp-side .pp-frow { flex-direction: column; align-items: stretch; gap: 12px; }
.pp-side .pp-fgroup { width: 100%; }
.pp-side .pp-seg { display: flex; flex-wrap: wrap; }
.pp-side .pp-seg button { flex: 1 1 auto; }
.pp-side .pp-select, .pp-side .pp-search { width: 100%; }
.pp-side .pp-chips { display: flex; flex-direction: column; gap: 4px; }
.pp-side .pp-chip { display: flex; align-items: center; gap: 7px; justify-content: flex-start; text-align: left; }
.pp-side .pp-chip .pp-cn { margin-left: auto; }
.pp-side .pp-recline { flex-direction: column; align-items: flex-start; gap: 8px; margin: 0; }
.pp-side .pp-key { margin-left: 0; }
.pp-main-head { display: flex; align-items: center; gap: 10px; font-family: var(--font-display); font-style: italic; font-weight: 900; font-size: 20px; text-transform: uppercase; margin: 2px 0 10px; }
.pp-main-head em { font-style: normal; font-family: var(--font-body); font-size: 12px; color: var(--muted); background: var(--chip); border-radius: 99px; padding: 2px 9px; }
.pp-table { border-radius: var(--radius); }
.pp-grid { grid-template-columns: minmax(170px, 1.5fr) minmax(100px, 0.8fr) 58px 72px 44px 66px 70px minmax(110px, 0.95fr) minmax(110px, 0.95fr) 84px; column-gap: 12px; min-width: 1100px; }
.pp-head { font-family: var(--font-body); letter-spacing: 0.1em; font-size: 10px; }
.pp-row { padding: 10px 16px; }
.pp-row.is-official { box-shadow: inset 3px 0 0 var(--neon); }
.pp-name { font-family: var(--font-body); font-weight: 700; font-size: 14px; }
.pp-pick { font-size: 12px; color: var(--muted); }
.pp-ou { font-family: var(--font-num); font-weight: 700; font-size: 14px; margin-top: 3px; }
.pp-proj { font-family: var(--font-num); font-weight: 700; font-size: 15px; }
.pp-evpill { display: inline-block; min-width: 64px; text-align: center; font-family: var(--font-num); font-weight: 700; font-size: 13px; padding: 6px 6px; border-radius: 7px; border: 1px solid rgba(57,240,122,0.35); }
.pp-ring { position: relative; display: inline-flex; width: 40px; height: 40px; align-items: center; justify-content: center; }
.pp-ring svg { position: absolute; inset: 0; }
.pp-ring .v { stroke: #5E6C66; }
.pp-ring.is-good .v { stroke: var(--neon); filter: drop-shadow(0 0 3px rgba(57,240,122,0.6)); }
.pp-ring b { font-family: var(--font-num); font-size: 11px; font-weight: 700; }
.pp-odds { display: inline-block; min-width: 54px; text-align: center; font-family: var(--font-num); font-weight: 700; font-size: 13px; border: 1px solid rgba(57,240,122,0.45); color: var(--text); border-radius: 7px; padding: 5px 6px; }
.pp-actions { flex-direction: column; align-items: stretch; gap: 5px; }
.pp-actions .pp-trk { padding: 5px 8px; font-size: 11px; }
.pp-flags { display: flex; flex-wrap: wrap; gap: 4px; }
.pp-tag { font-family: var(--font-body); font-size: 10px; letter-spacing: 0.03em; }
@media (max-width: 1100px) { .pp-layout { grid-template-columns: 1fr; } .pp-side { position: static; max-height: none; } }

.mu-panel { margin: 18px 0 14px; }
.mu-grid { display: grid; grid-template-columns: minmax(170px, 1.4fr) repeat(5, minmax(90px, 1fr)); gap: 10px; align-items: center; }
.mu-head { padding: 8px 0; border-bottom: 1px solid var(--rule); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-3); font-weight: 700; }
.mu-head > div:not(:first-child), .mu-c { text-align: center; }
.mu-row { padding: 9px 0; border-bottom: 1px solid var(--rule-row); font-size: 12.5px; }
.mu-name { font-weight: 700; }
.mu-c b { display: block; font-family: var(--font-num); font-size: 13px; }
.mu-c span { font-family: var(--font-num); font-size: 11.5px; }
.mu-c em { display: block; font-style: normal; font-size: 10px; color: var(--muted-3); }
.mu-c.is-empty { color: var(--muted-4); font-size: 11px; }

/* Team matchup */
.tm-card { background: var(--panel); border: 1px solid var(--rule); border-radius: var(--radius); padding: 14px 16px 12px; margin-top: 16px; }
.tm-top { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; align-items: center; }
.tm-title { display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; }
.tm-title svg { width: 16px; height: 16px; stroke: var(--neon); fill: none; stroke-width: 1.8; }
.tm-title em { font-style: normal; font-weight: 600; font-size: 10.5px; letter-spacing: 0.02em; text-transform: none; color: var(--muted-3); }
.tm-tabs { display: flex; gap: 4px; background: var(--chip); border-radius: 9px; padding: 3px; }
.tm-tabs button { background: none; border: none; color: var(--muted); font-family: var(--font-body); font-weight: 700; font-size: 11.5px; padding: 6px 10px; border-radius: 7px; cursor: pointer; }
.tm-tabs button.is-on { background: var(--neon); color: #04130A; }
.tm-sum { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0 6px; }
.tm-sum > div { background: var(--panel-deep); border: 1px solid var(--rule); border-radius: 9px; padding: 8px 10px; text-align: center; }
.tm-sum b { display: block; font-size: 20px; font-weight: 800; }
.tm-sum b.is-o { color: var(--neon); } .tm-sum b.is-d { color: #7FD8FF; }
.tm-sum span { font-size: 10.5px; color: var(--muted-3); }
.tm-row { display: grid; grid-template-columns: minmax(120px, 1.25fr) 92px minmax(70px, 1fr) 92px minmax(110px, 1.1fr) 52px; gap: 8px; align-items: center; padding: 5px 0; border-bottom: 1px solid var(--rule-row); font-size: 12px; }
.tm-headrow { border-bottom: 1px solid var(--rule); font-weight: 800; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; }
.tm-headrow .tm-lab { display: flex; align-items: center; gap: 6px; color: var(--text); }
.tm-headrow .tm-lab.is-r { justify-content: flex-end; }
.tm-group { font-size: 10px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: var(--neon-dim); margin: 10px 0 2px; }
.tm-lab { color: var(--text-dim); }
.tm-lab.is-r { text-align: right; color: var(--muted); }
.tm-src { font-size: 8.5px; font-weight: 800; letter-spacing: 0.06em; color: var(--muted-3); border: 1px solid var(--rule-strong); border-radius: 4px; padding: 0 4px; }
.tm-v { display: flex; align-items: center; gap: 6px; font-family: var(--font-num); font-size: 11.5px; color: var(--muted); }
.tm-v.is-r { justify-content: flex-end; }
.tm-pill { font-family: var(--font-num); font-weight: 700; font-size: 11.5px; min-width: 40px; text-align: center; padding: 3px 5px; border-radius: 6px; background: var(--chip); color: var(--text); border: 1px solid var(--rule-strong); }
.tm-pill.is-top { color: var(--neon); border-color: rgba(57,240,122,0.5); background: rgba(57,240,122,0.08); }
.tm-pill.is-bot { color: var(--red); border-color: rgba(255,82,82,0.45); background: rgba(255,82,82,0.07); }
.tm-pill.is-na { color: var(--muted-4); }
.tm-bar { position: relative; height: 8px; background: #17201C; border-radius: 99px; }
.tm-bar .tm-mid { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--rule-strong); }
.tm-bar b { position: absolute; top: 0; bottom: 0; border-radius: 99px; }
.tm-bar b.is-o { background: linear-gradient(270deg, var(--neon), rgba(57,240,122,0.3)); }
.tm-bar b.is-d { background: linear-gradient(90deg, #7FD8FF, rgba(127,216,255,0.3)); }
.tm-e { text-align: right; }
.tm-edge { font-size: 9.5px; font-weight: 800; letter-spacing: 0.06em; padding: 2px 6px; border-radius: 5px; color: var(--muted-3); border: 1px solid var(--rule-strong); }
.tm-edge.is-o { color: var(--neon); border-color: rgba(57,240,122,0.45); }
.tm-edge.is-d { color: #7FD8FF; border-color: rgba(127,216,255,0.45); }
.tm-foot { font-size: 10.5px; color: var(--muted-3); margin-top: 10px; }
.gb-card { margin-bottom: 14px; }
.bc-new { margin-left: 8px; font-size: 10px; font-weight: 900; letter-spacing: 0.06em; color: #06120A; background: var(--neon); border-radius: 999px; padding: 2px 7px; vertical-align: middle; }
.bc-dropped { border: 1px solid rgba(255,122,89,0.45); background: rgba(255,122,89,0.06); border-radius: 10px; padding: 10px 12px; margin-top: 10px; font-size: 12.5px; color: var(--muted-2, #B6C2CF); }
.bc-dropped .bc-note-h { color: #FF7A59; }
.bc-drop-row { margin: 4px 0; }
.bc-drop-row b { color: #fff; }
.bc-drop-row em { color: #FF7A59; font-style: normal; font-weight: 700; }
.bc-note { border: 1px solid rgba(57,240,122,0.35); background: rgba(57,240,122,0.05); border-radius: 10px; padding: 10px 12px; margin: 8px 0 10px; font-size: 12.5px; line-height: 1.5; color: var(--muted-2, #B6C2CF); }
.bc-note-h { font-weight: 800; color: var(--neon); margin-bottom: 4px; letter-spacing: 0.02em; }
.bc-note p { margin: 4px 0; }
.bc-note b { color: #fff; }
.bc-note .bc-note-s { font-size: 11px; color: var(--muted-3, #8B98A5); }
.lm-warn { font-size: 12px; line-height: 1.45; color: #F5B83D; background: rgba(245,184,61,0.08); border: 1px solid rgba(245,184,61,0.4); border-radius: 8px; padding: 6px 9px; margin-top: 6px; }
.lm-warn b { color: #FFD27A; }
.lm-good { font-size: 11.5px; color: var(--neon); margin-top: 6px; }
/* Player card (11/2026) */
.pp-namebtn, .pc-open { background: none; border: 0; padding: 0; font: inherit; color: inherit; cursor: pointer; text-align: left; text-decoration: underline; text-decoration-color: rgba(57,240,122,0.45); text-underline-offset: 3px; }
#pc-overlay { position: fixed; inset: 0; z-index: 9999; background: rgba(0,0,0,0.72); display: none; align-items: flex-start; justify-content: center; overflow-y: auto; padding: 18px 10px 90px; }
#pc-overlay.is-open { display: flex; }
body.pc-lock { overflow: hidden; }
.pc-card { width: 100%; max-width: 520px; background: #0B110D; border: 1px solid rgba(57,240,122,0.45); border-radius: 16px; padding: 16px; box-shadow: 0 0 28px rgba(57,240,122,0.15); }
.pc-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.pc-name { font-family: var(--font-display); font-weight: 800; font-size: 24px; color: #fff; line-height: 1.1; }
.pc-sub { font-family: var(--font-data); font-size: 12px; color: var(--muted-3, #8B98A5); margin-top: 3px; }
.pc-x { background: none; border: 1px solid rgba(255,255,255,0.2); color: #fff; border-radius: 999px; width: 32px; height: 32px; font-size: 20px; line-height: 1; cursor: pointer; flex: none; }
.pc-pick { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 10px; margin: 12px 0 6px; }
.pc-side { font-family: var(--font-display); font-weight: 900; font-size: 40px; line-height: 1; letter-spacing: 0.01em; }
.pc-side.is-under { color: #7FD8FF; text-shadow: 0 0 14px rgba(127,216,255,0.35); }
.pc-side.is-over { color: var(--neon); text-shadow: 0 0 14px rgba(57,240,122,0.35); }
.pc-pickline { font-family: var(--font-display); font-weight: 800; font-size: 22px; color: #fff; }
.pc-price { font-family: var(--font-data); font-size: 12.5px; color: var(--muted-2, #B6C2CF); }
.pc-off { font-size: 11px; font-weight: 800; color: #06120A; background: var(--neon); border-radius: 999px; padding: 2px 8px; }
.pc-mu { display: inline-block; font-family: var(--font-data); font-size: 13px; color: var(--neon); border: 1px solid rgba(57,240,122,0.55); border-radius: 10px; padding: 5px 10px; margin-bottom: 10px; }
.pc-mu b { color: #fff; }
.pc-tile { border: 1px solid rgba(57,240,122,0.22); border-radius: 12px; padding: 12px; margin-top: 10px; background: rgba(255,255,255,0.015); }
.pc-lab { font-family: var(--font-data); font-size: 12px; color: var(--muted-2, #B6C2CF); letter-spacing: 0.03em; }
.pc-lab em { font-style: normal; color: var(--neon); margin-left: 6px; font-size: 11px; }
.pc-big { font-family: var(--font-display); font-weight: 900; font-size: 34px; line-height: 1.1; margin: 4px 0 10px; }
.pc-big small { font-size: 16px; font-weight: 700; }
.pc-big.is-good, .pc-hr-pct.is-good { color: var(--neon); text-shadow: 0 0 12px rgba(57,240,122,0.3); }
.pc-big.is-bad, .pc-hr-pct.is-bad { color: #FF7A59; text-shadow: 0 0 12px rgba(255,122,89,0.3); }
.pc-hr-pct.is-even { color: #fff; }
.pc-bar { position: relative; height: 16px; border-radius: 999px; background: rgba(255,255,255,0.06); margin: 18px 0 8px; }
.pc-bar-fill { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 999px; }
.pc-bar-fill.is-good { background: linear-gradient(90deg, #22b35e, var(--neon)); box-shadow: 0 0 10px rgba(57,240,122,0.4); }
.pc-bar-fill.is-bad { background: linear-gradient(90deg, #d9603f, #FF7A59); box-shadow: 0 0 10px rgba(255,122,89,0.4); }
.pc-bar-line { position: absolute; top: -5px; bottom: -5px; width: 3px; margin-left: -1.5px; background: #fff; border-radius: 2px; }
.pc-bar-line span, .pc-bar-mid span { position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); font-family: var(--font-data); font-size: 10.5px; color: var(--muted-2, #B6C2CF); white-space: nowrap; margin-bottom: 2px; }
.pc-bar-c .pc-bar-fill { left: auto; }
.pc-bar-mid { position: absolute; left: 50%; top: -5px; bottom: -5px; width: 3px; margin-left: -1.5px; background: #fff; border-radius: 2px; z-index: 1; }
.pc-note { font-family: var(--font-data); font-size: 12px; color: var(--muted-2, #B6C2CF); line-height: 1.5; }
.pc-note b { color: #fff; }
.pc-ends { display: flex; justify-content: space-between; font-size: 10.5px; margin-top: -4px; }
.pc-key { margin-top: 6px; font-size: 11px; color: var(--muted-3, #8B98A5); }
.pc-svg { width: 100%; height: auto; display: block; margin-top: 6px; }
.pc-svg .pc-hit { fill: var(--neon); opacity: 0.9; }
.pc-svg .pc-miss { fill: #FF7A59; opacity: 0.85; }
.pc-svg .pc-gv { fill: #fff; font-family: var(--font-data); font-size: 11px; text-anchor: middle; }
.pc-svg .pc-gl { fill: #8B98A5; font-family: var(--font-data); font-size: 9.5px; text-anchor: middle; }
.pc-svg .pc-lineL { stroke: #fff; stroke-width: 1.5; stroke-dasharray: 5 4; }
.pc-svg .pc-lineT { fill: #fff; font-family: var(--font-data); font-size: 10px; text-anchor: start; paint-order: stroke; stroke: #0B110D; stroke-width: 3px; }
.pc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
.pc-grid > .pc-hr:last-child:nth-child(odd) { grid-column: span 2; }
.pc-hr { border: 1px solid rgba(57,240,122,0.22); border-radius: 12px; padding: 10px; }
.pc-hr-top { display: flex; align-items: baseline; gap: 8px; }
.pc-hr-pct { font-family: var(--font-display); font-weight: 900; font-size: 30px; line-height: 1; }
.pc-hr-odds { font-family: var(--font-data); font-size: 13px; color: var(--muted-2, #B6C2CF); }
.pc-hr-lab { font-family: var(--font-data); font-size: 11.5px; color: var(--muted-2, #B6C2CF); margin: 6px 0 8px; }
.pc-hr-bar { position: relative; height: 7px; border-radius: 999px; background: rgba(255,255,255,0.07); }
.pc-hr-bar div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #22b35e, var(--neon)); }
.pc-hr-bar i { position: absolute; top: -3px; bottom: -3px; width: 2px; background: #fff; }
.pc-go { margin-top: 14px; width: 100%; }
.gb-back { display: inline-block; margin-bottom: 10px; background: transparent; border: 1px solid rgba(57,240,122,0.45); color: var(--neon); border-radius: 999px; font: inherit; font-size: 12px; font-weight: 700; padding: 5px 12px; cursor: pointer; }
.pp-gotobd { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
.pp-gobtn { background: var(--neon); color: #06120A; border: 0; border-radius: 999px; font: inherit; font-size: 13px; font-weight: 800; padding: 7px 14px; cursor: pointer; }
.gb-card .tm-title { margin-bottom: 10px; }
.gb-p { font-size: 13px; line-height: 1.6; color: var(--text, #E6EDF3); margin: 0 0 9px; }
.gb-p b { color: #fff; }
.gb-rk { font-family: var(--font-mono, monospace); font-size: 11px; color: var(--muted-3); white-space: nowrap; }
.gb-playbox { background: var(--panel-deep); border: 1px solid rgba(57,240,122,0.35); border-radius: 10px; padding: 10px 12px; margin-top: 4px; }
.gb-lab { font-size: 10px; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: var(--neon); margin-bottom: 6px; }
.gb-play { display: inline-flex; align-items: center; gap: 6px; font-weight: 800; font-size: 15px; color: #fff; vertical-align: middle; }
.gb-play.is-lean { color: var(--muted-2, #B6C2CF); }
.gb-off { font-size: 11px; color: var(--muted-3); }
.gb-info { font-size: 11.5px; color: var(--muted-2, #B6C2CF); background: var(--panel-deep); border: 1px solid var(--rule); border-radius: 8px; padding: 7px 10px; margin-bottom: 12px; line-height: 1.6; }
.gb-sec { border-top: 1px solid var(--rule); padding-top: 10px; margin-bottom: 10px; }
.gb-lab2 { font-size: 10px; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-2, #B6C2CF); margin-bottom: 7px; }
.gb-lab2 span { text-transform: none; letter-spacing: 0; font-weight: 500; color: var(--muted-3); }
.gb-injs { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 5px; }
.gb-inj, .gb-prop { font-size: 11.5px; border: 1px solid var(--rule-strong); border-radius: 6px; padding: 3px 7px; color: var(--text, #E6EDF3); }
.gb-inj i { font-style: normal; color: #F5B942; }
.gb-inj.is-out i { color: #FF6B6B; }
.gb-inj em { font-style: normal; font-weight: 800; font-size: 10px; color: #fff; background: rgba(255,107,107,0.18); border-radius: 4px; padding: 0 4px; margin-left: 2px; }
.gb-inj.is-key { border-color: rgba(255,107,107,0.45); }
.gb-prop { border-color: rgba(57,240,122,0.45); }
.gb-note { font-size: 11px; color: var(--muted-3); }
.gb-risk { font-size: 12px; color: #F5B942; margin-top: 7px; line-height: 1.5; }
.gb-why { font-size: 12.5px; line-height: 1.55; color: var(--muted-2, #B6C2CF); margin-top: 6px; }
.gb-why-card { background: var(--panel-deep); border: 1px solid rgba(57,240,122,0.25); border-radius: 9px; padding: 8px 11px; margin-bottom: 8px; }
.gb-why-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 4px; }
.gb-why-head .gb-prop { font-weight: 800; }
.gb-why-list { margin: 4px 0 0; padding-left: 17px; font-size: 12.5px; line-height: 1.55; color: var(--muted-2, #B6C2CF); }
.gb-why-list li { margin-bottom: 4px; }
.gb-why-list b { color: #fff; }
.gb-why-list .gb-note { display: block; }
.gb-why-lab { margin-top: 10px; color: var(--neon); }
.gb-read { font-size: 13.5px; line-height: 1.6; color: #E6EDF3; margin: 6px 0 8px; }
.gb-read b { color: var(--neon); display: block; font-size: 11.5px; letter-spacing: .05em; text-transform: uppercase; }
.gb-read p { margin: 4px 0 6px; }
.gb-read2 p { margin: 0 0 10px; }
.gb-read b.gb-h { display: block; margin-bottom: 2px; }
.gb-inj { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 10px; margin: 6px 0 8px; }
.gb-inj-tog { display: inline-flex; border: 1px solid rgba(57,240,122,0.4); border-radius: 999px; overflow: hidden; margin-bottom: 6px; }
.gb-inj-tog button { background: transparent; color: var(--muted-2, #B6C2CF); border: 0; font: inherit; font-size: 12px; font-weight: 700; padding: 5px 12px; cursor: pointer; }
.gb-inj-tog button.is-on { background: var(--neon); color: #06120A; }
.gb-inj-pane { display: none; }
.gb-inj-pane.is-on { display: block; }
.gb-inj-row { font-size: 13px; color: #E6EDF3; margin: 2px 0; }
.gb-inj-row b { color: #fff; }
.gb-inj-big { font-size: 20px; font-weight: 800; color: #fff; }
.gb-inj-note { font-size: 11.5px; color: var(--muted-3, #8B98A5); margin-top: 3px; }
.gb-read b.gb-inline { display: inline; font-size: inherit; text-transform: none; letter-spacing: 0; color: #fff; }
.gb-read.is-risk b { color: #F5B83D; }
.gb-nums { margin-top: 4px; }
.gb-nums summary { cursor: pointer; font-size: 11.5px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--muted-3, #8B98A5); padding: 4px 0; }
.gb-nums[open] summary { color: #fff; }

.pp-grade { display: flex; gap: 3px; }
.pp-gbtn { flex: 1; border: 1px solid var(--rule-strong); background: transparent; color: var(--muted-3); border-radius: 5px; font-family: var(--font-body); font-weight: 800; font-size: 10.5px; padding: 3px 0; cursor: pointer; }
.pp-gbtn:hover { color: var(--text); }
.pp-gbtn.is-win.is-on { background: rgba(57,240,122,0.16); color: var(--neon); border-color: var(--neon); }
.pp-gbtn.is-loss.is-on { background: rgba(255,82,82,0.16); color: var(--red); border-color: var(--red); }
.pp-gbtn.is-push.is-on { background: rgba(224,180,74,0.16); color: var(--amber); border-color: var(--amber); }
.my-props { margin-top: 16px; }
.my-props-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-top: 8px; }
.my-res { display: grid; grid-template-columns: 1fr 52px 62px; gap: 8px; font-size: 12px; padding: 6px 0; border-bottom: 1px solid var(--rule-row); align-items: center; }
.my-res span { color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.my-res em { font-style: normal; color: var(--muted-3); }
.my-res .my-d { font-style: normal; color: var(--muted-3); font-family: var(--font-num); font-size: 11px; margin-right: 4px; }
.my-res b { font-family: var(--font-num); text-align: right; font-size: 11.5px; }
@media (max-width: 900px) { .my-props-grid { grid-template-columns: 1fr; } }

.td-grid { display: grid; grid-template-columns: minmax(190px, 1.6fr) 90px 52px 80px 96px 130px minmax(150px, 1.1fr); column-gap: 14px; align-items: center; min-width: 900px; }
.td-controls { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 12px; }
.td-note { margin-top: 16px; background: var(--panel); border: 1px solid var(--rule); border-left: 3px solid var(--amber); border-radius: var(--radius); padding: 10px 14px; font-size: 12px; color: var(--text-dim); line-height: 1.5; }
.td-def { display: inline-block; font-family: var(--font-num); font-weight: 700; font-size: 13px; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--rule-strong); }
.td-def.is-soft { color: var(--neon); border-color: rgba(57,240,122,0.5); background: rgba(57,240,122,0.08); }
.td-def.is-tough { color: var(--red); border-color: rgba(255,82,82,0.45); background: rgba(255,82,82,0.07); }

/* ===== Phones (11/2026): bottom tab bar, cards instead of wide tables ===== */
.navm { display: none; }
@media (max-width: 760px) {
  :root { --gutter: 14px; }
  html, body { overflow-x: hidden; }
  .page { padding-bottom: 96px; }
  .topbar-inner { height: 56px; gap: 10px; }
  .brand-word { font-size: 21px; }
  .brand-mark i { height: 18px; width: 5px; }
  .navx, .sync { display: none !important; }
  .topbar-right { margin-left: auto; }
  #search { width: 150px; font-size: 13px; padding: 6px 12px; }
  .navm { display: flex; position: fixed; left: 0; right: 0; bottom: 0; z-index: 60; background: rgba(9,12,11,0.97); backdrop-filter: blur(10px);
          border-top: 1px solid var(--rule-strong); overflow-x: auto; scrollbar-width: none; padding: 4px 4px calc(4px + env(safe-area-inset-bottom)); }
  .navm::-webkit-scrollbar { display: none; }
  .navm-item { flex: 0 0 auto; min-width: 68px; display: flex; flex-direction: column; align-items: center; gap: 3px; background: none; border: none; color: var(--muted);
               font-family: var(--font-body); font-weight: 700; font-size: 10.5px; padding: 7px 6px; border-radius: 10px; cursor: pointer; }
  .navm-item svg { width: 20px; height: 20px; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
  .navm-item.is-active { color: var(--neon); background: rgba(57,240,122,0.10); }

  .main, .split, .em-below, .ep-grid, .pp-layout, .my-props-grid, .pp-lower { grid-template-columns: minmax(0, 1fr) !important; }
  .main > div, .split > div { min-width: 0; }
  .section-head { flex-wrap: wrap; gap: 10px; }
  .section-head h2 { font-size: 18px; }
  .em-sub { display: none; }

  /* stat cards: two per row */
  .kpis { grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
  .kpi { padding: 12px 12px 10px; }
  .kpi-value { font-size: 22px; gap: 7px; flex-wrap: wrap; }
  .kpi-icon { width: 26px; height: 26px; border-radius: 7px; }
  .kpi-icon svg { width: 14px; height: 14px; }
  .kpi-sub { font-size: 10px; }
  .kpi-spark { display: none; }

  /* Edge Matrix rows as cards */
  .em-table { overflow: visible; }
  .em-table .thead { display: none; }
  .em-grid { min-width: 0 !important; grid-template-columns: 1fr auto; row-gap: 8px; }
  .em-row .row-body { padding: 12px 12px; }
  .em-grid > .em-n, .em-grid > .cell-market, .em-grid > .cell-model { display: none; }
  .em-grid > .em-game { grid-column: 1; grid-row: 1; }
  .em-grid > div:nth-child(6) { grid-column: 2; grid-row: 1; }
  .em-grid > .em-kick { grid-column: 1; grid-row: 2; }
  .em-grid > .em-prob { grid-column: 2; grid-row: 2; min-width: 120px; }
  .em-grid > div:nth-child(8) { grid-column: 1; grid-row: 3; min-width: 0; }
  .em-grid > div:nth-child(10) { grid-column: 2; grid-row: 3; }
  .em-grid > .em-tags { grid-column: 1 / -1; grid-row: 4; }
  .em-play { max-width: 100%; }

  /* projector, team matchup, bet card */
  .proj-head { padding: 14px 12px; }
  .proj-head-inner { gap: 8px; }
  .tm-card { padding: 12px 10px; }
  .tm-tabs { width: 100%; }
  .tm-tabs button { flex: 1; font-size: 11px; padding: 7px 4px; }
  .tm-row { grid-template-columns: 52px 1fr 52px 46px; grid-template-areas: "lab lab lab lab" "op bar dp e"; row-gap: 4px; padding: 7px 0; }
  .tm-row > .tm-lab:first-child { grid-area: lab; }
  .tm-row > .tm-v:not(.is-r) { grid-area: op; }
  .tm-row > .tm-bar { grid-area: bar; }
  .tm-row > .tm-v.is-r { grid-area: dp; }
  .tm-row > .tm-lab.is-r { display: none; }
  .tm-row > .tm-e { grid-area: e; }
  .tm-v span:not(.tm-pill) { display: none; }
  .tm-headrow { display: none; }

  /* Player Props */
  .pp-hero h1 { font-size: 30px; }
  .pp-hero-card { width: 100%; }
  .pp-hero-card > div { padding: 10px 12px; flex: 1; }
  .pp-hv { font-size: 18px; }
  .pp-cat { padding: 9px 12px; font-size: 11.5px; }
  .pp-side { position: static; max-height: none; }
  .pp-side .pp-filter { display: none; }
  .pp-side.is-filters .pp-filter { display: block; }
  .pp-ftoggle { display: inline-flex !important; }
  .pp-table { overflow: visible; border: none; background: none; }
  .pp-head { display: none; }
  .pp-grid, .td-grid { min-width: 0 !important; }
  .pp-grid.pp-row { grid-template-columns: 1fr 1fr 1fr; row-gap: 10px; column-gap: 8px; background: var(--panel); border: 1px solid var(--rule); border-radius: 12px; margin-bottom: 10px; padding: 12px; }
  .pp-grid.pp-row > .pp-who { grid-column: 1 / -1; }
  .pp-grid.pp-row > div:nth-child(2) { grid-column: 1 / 3; }
  .pp-grid.pp-row > div:nth-child(3) { grid-column: 3; text-align: right; }
  .pp-grid.pp-row > div:nth-child(4) { grid-column: 1; text-align: left; }
  .pp-grid.pp-row > div:nth-child(5) { grid-column: 2; text-align: center; }
  .pp-grid.pp-row > div:nth-child(6) { grid-column: 3; text-align: right; }
  .pp-grid.pp-row > div:nth-child(7) { grid-column: 1; }
  .pp-grid.pp-row > div:nth-child(8) { grid-column: 2 / -1; }
  .pp-grid.pp-row > .pp-flags { grid-column: 1 / -1; }
  .pp-grid.pp-row > .pp-actions { grid-column: 1 / -1; flex-direction: row; flex-wrap: wrap; align-items: center; }
  .pp-actions .pp-grade { flex: 1 1 100%; }
  .pp-actions .pp-gbtn { padding: 8px 0; font-size: 12px; }
  .pp-actions .pp-trk { flex: 1; padding: 8px; }
  .pp-group { padding: 10px 2px; }
  .td-grid.pp-row { grid-template-columns: 1fr 1fr; row-gap: 10px; background: var(--panel); border: 1px solid var(--rule); border-radius: 12px; margin-bottom: 10px; padding: 12px; }
  .td-grid.pp-row > .pp-who { grid-column: 1 / -1; }
  .td-grid.pp-row > div:nth-child(n+2) { text-align: left; }
  .td-controls .pp-seg { flex-wrap: wrap; }
  .mu-grid { grid-template-columns: 110px repeat(5, minmax(54px, 1fr)); gap: 4px; font-size: 11px; }
  .mu-panel { overflow-x: auto; }
  .my-res { grid-template-columns: 1fr 48px 54px; }

  /* tracker and other wide tables scroll sideways inside their box */
  .trk-table { max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .trk-table .trk-g { min-width: 1080px; }
  .trk-summary { grid-template-columns: 1fr 1fr !important; }
  .footer { font-size: 10px; }
}
.pp-ftoggle { display: none; }
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
    // Big-dog watch (10/2026): the one 20+ pocket the big-spread lab found
    // worth following -- the model's side is the UNDERDOG getting 28 to 34.5
    // points and clears the normal edge bar. Walk-forward 2024-2026: 29-16
    // (+23%) at the opening line, only 30-26 (+2%) at the close, so the value
    // is in getting the early number. Watch only: never official, never on
    // the Bet Card, the 20+ fade above still applies (see big_spread_lab.json).
    var sideLine = market === 'Spread' && side ? (side === game.home ? game.marketSpread : -game.marketSpread) : null;
    var bigDogWatch = bigSpread && edgeClears && sideLine != null && sideLine >= 28 && sideLine < 35;
    // 11/2026: only games priced by the in-season trained model can be
    // plays. A game falls back to the preseason SP+ prior when an input the
    // trained model needs isn't out yet -- usually next week's games, which
    // have no weather forecast until a few days out -- and that prior throws
    // 20+ point "edges" that were crowding real plays off the Bet Card.
    var waitingModel = edgeClears && priceInRange && isPositiveEV && !bigSpread && !isTrainedGame;
    var qualifies = edgeClears && priceInRange && isPositiveEV && !bigSpread && isTrainedGame;
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
      waitingModel: waitingModel,
      isFade: isFade,
      bigSpread: bigSpread,
      bigDogWatch: bigDogWatch,
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

  // Confidence tier (11/2026). The win % itself runs well above real hit
  // rates (one fixed sigma -- see tierFor), so the tier is a RANKING label,
  // not a probability or a sizing rule: every play is still 1u. Cutoffs
  // split the weeks 4-10 Bet Card into rough thirds (see
  // docs/data/bet_card_confidence.json); MLs sit higher because a moneyline
  // win % is naturally bigger than the same game's cover %.
  var CONF_CUTS = { Spread: [0.77, 0.71], Moneyline: [0.81, 0.775] };
  function confTier(market, winProb) {
    if (winProb == null || isNaN(winProb)) return null;
    var c = CONF_CUTS[market === 'Moneyline' ? 'Moneyline' : 'Spread'];
    return winProb >= c[0] ? 'High' : winProb >= c[1] ? 'Mid' : 'Low';
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
    decomposition: decomposition, buildBetCard: buildBetCard, confTier: confTier, CONF_CUTS: CONF_CUTS,
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

  var state = { selected: 0, market: 'Moneyline', search: '', page: 'edge', edgeSort: 'kickoff', edgeShow: 'all', edgeConf: 'ALL', propMarket: 'ALL', unitSize: 50, trackerModelTab: 'all', injScope: 'slate', propGame: 'ALL', propOfficialOnly: false, propShowPass: false, propView: 'official', propDetail: null,
    propSort: 'kickoff', propSort2: 'edge', propSortRev: false, propDir: 'all', propBook: 'ALL', propMinEdge: 0, propQ: '', propMyList: false,
    atdView: 'all', atdBasis: 'best', atdSort: 'edge', atdGame: 'ALL' };
  // Props tab Sort & Filter preset (10/2026): saved per browser, applied on load.
  var PROP_PRESET_KEY = 'cfb_props_preset_v1';
  var PROP_PRESET_FIELDS = ['propCat', 'propView', 'propMarket', 'propGame', 'propSort', 'propSort2', 'propSortRev', 'propDir', 'propBook', 'propMinEdge', 'propMyList', 'propShowPass', 'propMinMu'];
  var PROP_DEFAULTS = { propCat: 'all', propView: 'official', propMarket: 'ALL', propGame: 'ALL', propSort: 'kickoff', propSort2: 'edge', propSortRev: false, propDir: 'all', propBook: 'ALL', propMinEdge: 0, propMyList: false, propShowPass: false, propMinMu: 0 };
  try {
    var pp0 = JSON.parse(localStorage.getItem(PROP_PRESET_KEY) || 'null');
    if (pp0) PROP_PRESET_FIELDS.forEach(function (k) { if (pp0.hasOwnProperty(k)) state[k] = pp0[k]; });
  } catch (e) {}

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
    { id: 'anytime', label: 'Anytime TD' },
    { id: 'tdallowed', label: 'TDs Allowed' },
    { id: 'fantasy', label: 'Fantasy' },
    { id: 'injuries', label: 'Injury Report' },
    { id: 'tracker', label: 'My Tracker' },
    { id: 'performance', label: 'Model Performance' },
  ];

  // Icon set for the nav and stat cards (11/2026 reskin). Simple line
  // icons drawn here, 24x24 grid.
  var ICONS = {
    edge: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    ratings: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    props: '<path d="M4 15l5-5 4 4 7-7"/><path d="M14 7h6v6"/>',
    anytime: '<path d="M12 2c1 4-3 5-3 9a3 3 0 0 0 6 0c0-1.5-.5-2.5-1-3.5 2 1 4 3.5 4 6.5a6 6 0 0 1-12 0c0-5 5-7 6-12z"/>',
    tdallowed: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
    fantasy: '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
    injuries: '<rect x="3" y="3" width="18" height="18" rx="4"/><path d="M12 8v8M8 12h8"/>',
    tracker: '<rect x="3" y="3" width="18" height="18" rx="4"/><path d="M8 12l3 3 5-6"/>',
    performance: '<path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 6-7"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    bars: '<path d="M5 20v-6M10 20V9M15 20v-9M20 20V4"/>',
    check: '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
    trend: '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    sigma: '<path d="M18 4H6l7 8-7 8h12"/>',
    slope: '<path d="M4 20L20 4"/><circle cx="7" cy="16" r="1.4"/><circle cx="12" cy="11" r="1.4"/><circle cx="17" cy="8" r="1.4"/>',
    games: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M12 5v14M3 12h18"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>'
  };
  function icon(name) { return '<svg viewBox="0 0 24 24" aria-hidden="true">' + (ICONS[name] || '') + '</svg>'; }
  // Sparkline from a real series (cumulative units etc.). Returns '' for
  // fewer than 3 points -- never drawn from made-up numbers.
  function spark(vals) {
    vals = (vals || []).filter(function (v) { return v != null && isFinite(v); });
    if (vals.length < 3) return '';
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), w = 92, h = 34;
    var xs = function (i) { return (i / (vals.length - 1)) * w; };
    var ys = function (v) { return hi === lo ? h / 2 : h - 2 - ((v - lo) / (hi - lo)) * (h - 4); };
    var d = vals.map(function (v, i) { return (i ? 'L' : 'M') + xs(i).toFixed(1) + ' ' + ys(v).toFixed(1); }).join(' ');
    return '<svg class="kpi-spark" viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none"><defs><linearGradient id="sparkfill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#39F07A" stop-opacity=".35"/><stop offset="1" stop-color="#39F07A" stop-opacity="0"/></linearGradient></defs>' +
      '<path class="a" d="' + d + ' L' + w + ' ' + h + ' L0 ' + h + ' Z"/><path class="l" d="' + d + '"/></svg>';
  }
  // Stat card: [label, value, sub, opts{neon, icon, pill, spark}]
  function statCards(items) {
    return '<div class="kpis">' + items.map(function (r) {
      var o = r[3] || {};
      return '<div class="kpi">' +
        '<div class="kpi-body">' +
          '<div class="kpi-label">' + esc(r[0]) + '</div>' +
          '<div class="kpi-value' + (o.neon ? ' is-neon' : '') + '">' + (o.icon ? '<span class="kpi-icon">' + icon(o.icon) + '</span>' : '') + esc(r[1]) +
            (o.pill ? '<span class="kpi-pill">' + esc(o.pill) + '</span>' : '') + '</div>' +
          '<div class="kpi-sub">' + esc(r[2]) + '</div>' +
        '</div>' + (o.spark ? spark(o.spark) : '') + '</div>';
    }).join('') + '</div>';
  }

  var SHORT_LABEL = { edge: 'Edge', ratings: 'Ratings', props: 'Props', anytime: 'TDs', tdallowed: 'TDs vs D', fantasy: 'Fantasy', injuries: 'Injuries', tracker: 'Tracker', performance: 'Model' };
  function renderTopbar() {
    // Search only on the Edge Board tab.
    var searchHtml = state.page === 'edge'
      ? '<input id="search" type="text" placeholder="Search team, matchup..." autocomplete="off" value="' + esc(state.search) + '" oninput="window.__cfbSearch(this.value)">'
      : '';
    return '' +
      '<div class="topbar"><div class="topbar-inner">' +
        '<div class="brand">' +
          '<span class="brand-mark"><i></i><i></i></span>' +
          '<span class="brand-word">' + esc(D.meta.modelName) + ' <b>' + esc(D.meta.sport) + '</b></span>' +
          '<span class="brand-sub">' + esc(D.meta.version) + '</span>' +
        '</div>' +
        '<nav class="navx">' + PAGES.map(function (p) {
          return '<button class="navx-item' + (p.id === state.page ? ' is-active' : '') + '" data-page="' + esc(p.id) + '">' + icon(p.id) + '<span>' + esc(p.label) + '</span></button>';
        }).join('') + '</nav>' +
        '<div class="topbar-right">' +
          searchHtml +
          '<div class="sync"><span class="sync-dot"></span><span>' + esc(D.meta.dataAsOf) + '</span></div>' +
        '</div>' +
      '</div></div>';
  }

  // Phones (11/2026): every tab in a bar fixed to the bottom of the screen.
  // Rendered outside the top bar, whose blur would otherwise pin it there.
  function renderPageNav() {
    return '<nav class="navm" aria-label="Pages">' + PAGES.map(function (p) {
      return '<button class="navm-item' + (p.id === state.page ? ' is-active' : '') + '" data-page="' + esc(p.id) + '">' + icon(p.id) + '<span>' + esc(SHORT_LABEL[p.id] || p.label) + '</span></button>';
    }).join('') + '</nav>';
  }

  function renderKpis(card, priced) {
    // "Qualifying edges" used to show card.length, which is capped at the
    // Bet Card's ten -- it read "10" however many plays cleared the bar.
    var flagged = (priced || []).filter(function (p) { return p.qualifies; }).length;
    var pct = D.meta.gamesPriced ? Math.round(flagged / D.meta.gamesPriced * 100) + '%' : null;
    var c = D.clv || {};
    var items = [
      ['Games priced', String(D.meta.gamesPriced), D.meta.totalGames + ' in slate \u00b7 ' + (D.meta.totalGames - D.meta.gamesPriced) + ' unpriced', { icon: 'games' }],
      ['Qualifying edges', String(flagged), card.length + ' on the Bet Card \u00b7 \u2265 ' + D.meta.minEdge.toFixed(1) + ' pts \u00b7 ' + state.market, { icon: 'target', neon: true, pill: pct }],
      ['Model \u03c3 margin', D.meta.marginSd != null ? D.meta.marginSd.toFixed(1) : '\u2014', 'league-wide residual std \u00b7 ' + (D.meta.spN || 0) + ' games', { icon: 'sigma' }],
      ['SP+ slope fit', D.meta.spSlope != null ? D.meta.spSlope.toFixed(3) : '\u2014', 'margin = slope\u00d7SP+ diff + ' + (D.meta.spIntercept != null ? D.meta.spIntercept.toFixed(2) : '\u2014'), { icon: 'slope' }]
    ];
    if (c.nGames) items.push(['Avg CLV', (c.avgClvPoints > 0 ? '+' : '') + Number(c.avgClvPoints).toFixed(2) + ' pts',
      (c.pctPositiveClv != null ? (c.pctPositiveClv * 100).toFixed(0) + '% beat the close' : '') + ' \u00b7 ' + c.nGames + ' graded picks', { icon: 'trend', neon: c.avgClvPoints > 0 }]);
    return statCards(items);
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
      dogWatch: !!p.bigDogWatch,
      // Win % and its confidence tier at the moment it was tracked, so the
      // Tracker can keep a record by tier (11/2026).
      winProb: Math.round(p.sideProb * 1000) / 1000,
      confTier: M.confTier(p.market, p.sideProb),
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
  var EDGE_SHOWS = [['all', 'All games'], ['card', 'Bet Card plays'], ['plays', 'All flagged plays'], ['dogwatch', 'Big-dog watch']];

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
    else if (s.kind === 'dogwatch') { top = 'WATCH · BIG DOG'; cls = 'is-track'; title = '28–34.5 point underdog the model likes — watch only, track at the opening line'; }
    else if (s.kind === 'bigspread') { top = 'FADE · 20+ PT'; cls = 'is-warn'; title = '20+ point spread — faded by rule'; }
    else if (s.kind === 'fade') { top = 'FADE'; cls = 'is-warn'; }
    else if (s.kind === 'waiting') { top = 'WAITING ON MODEL'; cls = 'is-track'; title = 'Preseason ratings only so far: ' + s.note; }
    else return '<span class="eb-noplay" title="' + esc(s.kind === 'nospread' ? 'No spread posted' : 'No play: ' + s.note) + '">—</span>';
    return '<div class="eb-pill ' + cls + '" title="' + esc(title) + '"><div class="eb-pill-top">' + top + '</div>' +
      '<div class="eb-pill-pick">' + esc(s.playText) + (cls === 'is-card' ? ' <span>· ' + esc(s.p.tier) + '</span>' : '') + '</div>' + extra + '</div>';
  }

  // Edge Matrix (11/2026 reskin): the play as a button-style pill, its
  // status (what to do with it) as a tag. Same playStatusOf() verdicts.
  function emStatus(s) {
    if (edgeOnCard(s)) return ['BET CARD', 'is-card', 'On the Bet Card — wager'];
    if (s.kind === 'target') return ['NEAR-EVEN', 'is-track', 'Near-even target: ' + s.note];
    if (s.kind === 'official') return ['TRACK ONLY', 'is-track', 'Official play, not on the Bet Card — track only'];
    if (s.kind === 'bigdog') return ['UNOFFICIAL DOG', 'is-warn', 'Underdog moneyline longer than +150 — unofficial, track only'];
    if (s.kind === 'dogwatch') return ['BIG-DOG WATCH', 'is-track', '28–34.5 point underdog the model likes — watch only, at the opening line'];
    if (s.kind === 'bigspread') return ['FADE 20+', 'is-warn', '20+ point spread — faded by rule'];
    if (s.kind === 'fade') return ['FADE', 'is-warn', s.note || 'Fade'];
    if (s.kind === 'waiting') return ['WAITING ON MODEL', '', 'Preseason ratings only so far: ' + s.note];
    return null;
  }
  function emPlay(s) {
    if (!emStatus(s)) return '<span class="eb-noplay" title="' + esc(s.kind === 'nospread' ? 'No spread posted' : 'No play: ' + (s.note || '')) + '">—</span>';
    var card = edgeOnCard(s);
    return '<span class="em-play' + (card ? ' is-card' : '') + '" title="' + esc(emStatus(s)[2]) + '"><svg viewBox="0 0 24 24"><path d="M8 5l11 7-11 7z"/></svg>' +
      esc(s.playText) + (card && s.p && s.p.tier ? ' <em>' + esc(s.p.tier) + '</em>' : '') + '</span>';
  }

  // Bottom of the Edge Board: backtest record, latest picks vs the close,
  // and notes pulled from this slate. Real numbers only.
  function renderEdgePanels(priced) {
    var bt = D.backtest || {}, c = D.clv || {};
    var sg = function (x, d) { return (x > 0 ? '+' : '') + Number(x).toFixed(d); };
    var col = function (x) { return x > 0 ? 'var(--neon)' : (x < 0 ? 'var(--red)' : 'var(--muted)'); };
    // Live record from My Tracker (11/3/2026): game-line plays only, official
    // only (unofficial plays -- moneylines, 20+ spreads -- are left out),
    // 1 unit a play, the same way the tracker's Units record counts them.
    var trk = loadTrk().filter(function (it) { return it.type !== 'Prop' && !it.unofficial && it.status && it.price != null; });
    var when = function (it) {
      var t = Date.parse(it.kickoffIso || '');
      if (isNaN(t)) { var m = /^t(\\d{13})/.exec(it.id || ''); t = m ? Number(m[1]) : 0; }
      return t;
    };
    trk.sort(function (x, y) { return when(x) - when(y); });
    var unitOf = function (it) { return it.status === 'win' ? americanToDecimal(it.price) - 1 : (it.status === 'loss' ? -1 : 0); };
    var tally = function (list) {
      var w = 0, l = 0, pu = 0, u = 0;
      list.forEach(function (it) { if (it.status === 'win') w++; else if (it.status === 'loss') l++; else if (it.status === 'push') pu++; u += unitOf(it); });
      return { n: list.length, rec: w + '-' + l + '-' + pu, units: u, roi: list.length ? u / list.length * 100 : 0 };
    };
    var line = function (label, t) {
      return '<div class="ep-row"><span>' + esc(label) + '</span><b>' + esc(t.rec) + '</b><b style="color:' + col(t.units) + '">' + sg(t.units, 2) + 'u</b><b style="color:' + col(t.roi) + '">' + sg(t.roi, 1) + '%</b></div>';
    };
    var perf = '';
    if (trk.length) {
      perf = line('All official plays', tally(trk));
      var bc = trk.filter(function (it) { return it.betCard; });
      if (bc.length) perf += line('Bet Card plays', tally(bc));
      var sp = trk.filter(function (it) { return it.type === 'Spread'; });
      if (sp.length && sp.length !== trk.length) perf += line('Spreads', tally(sp));
    }
    var cum = [], run = 0; trk.forEach(function (it) { run += unitOf(it); cum.push(run); });
    var totU = trk.length ? tally(trk).units : 0;
    var p1 = '<div class="ep-card"><div class="ep-head">' + icon('performance') + '<span>Model performance</span><em>live &middot; My Tracker, official plays, 1u each</em></div>' +
      (perf ? '<div class="ep-row ep-th"><span></span><span>Record</span><span>Units</span><span>ROI</span></div>' + perf
            : '<div class="ep-empty">No graded official plays in My Tracker in this browser yet.</div>') +
      (trk.length ? '<div class="ep-clv"><div><div class="ep-k">Units, season</div><div class="ep-v" style="color:' + col(totU) + '">' + sg(totU, 2) + 'u</div>' +
        '<div class="ep-k2">' + trk.length + ' graded official plays' + (c.nGames ? ' &middot; avg CLV ' + sg(c.avgClvPoints, 2) + ' pts' : '') + '</div></div>' + spark(cum) + '</div>' : '') +
      '<div class="ep-foot" style="color:var(--muted-3)">Moneylines and other unofficial plays are not counted. Backtest is on the Model Performance tab.</div></div>';
    var recent = (c.recent || []).map(function (g) {
      var side = g.side === 'home' ? g.home : g.away, ln = g.side === 'home' ? g.line : -g.line;
      var d = new Date(g.date), ds = isNaN(d) ? '' : (d.getMonth() + 1) + '/' + d.getDate();
      return '<div class="ep-row ep-pick"><span class="ep-date">' + ds + '</span><span>' + esc(abbrOf(g.away)) + ' at ' + esc(abbrOf(g.home)) + '</span><b>' + esc(abbrOf(side)) + ' ' + (ln > 0 ? '+' : '') + ln + '</b>' +
        '<b style="color:' + col(g.clv) + '">' + (g.clv != null ? sg(g.clv, 1) : '—') + '</b></div>';
    }).join('');
    var p2 = '<div class="ep-card"><div class="ep-head">' + icon('tracker') + '<span>Recent picks vs the close</span><em>CLV in points</em></div>' +
      (recent ? '<div class="ep-row ep-th ep-pick"><span>Date</span><span>Game</span><span>Pick</span><span>CLV</span></div>' + recent : '<div class="ep-empty">No graded picks yet.</div>') +
      '<div class="ep-foot"><button class="trk-linkbtn" onclick="window.__cfbGoTracker()">Your graded bets are in My Tracker →</button></div></div>';
    var notes = [];
    priced.forEach(function (p) {
      (p.game.flags || []).filter(function (f) { return f.qb; }).forEach(function (f) {
        if (notes.length < 6) notes.push('<b>' + esc(abbrOf(p.game.away)) + ' at ' + esc(abbrOf(p.game.home)) + ':</b> ' + esc(abbrOf(f.team) + ' ' + f.text));
      });
    });
    var untrained = priced.filter(function (p) { return !(p.game.flags || []).some(function (f) { return f.text === 'In-season model'; }); }).length;
    var flagged = priced.filter(function (p) { return p.qualifies; }).length;
    notes.push('<b>Market:</b> ' + flagged + ' of ' + priced.length + ' priced games clear the ' + D.meta.minEdge.toFixed(1) + '-pt bar (' + state.market + ').');
    if (untrained) notes.push('<b>Model:</b> ' + untrained + ' game' + (untrained === 1 ? '' : 's') + ' still priced by the preseason prior.');
    var p3 = '<div class="ep-card"><div class="ep-head">' + icon('injuries') + '<span>Model notes</span><em>this slate</em></div><ul class="ep-notes">' +
      notes.map(function (n) { return '<li>' + n + '</li>'; }).join('') + '</ul></div>';
    return '<div class="ep-grid">' + p1 + p2 + p3 + '</div>';
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
      if (state.edgeShow === 'dogwatch' && r.s.kind !== 'dogwatch') return false;
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
        '<div class="section-title"><div class="section-flag"></div><h2>Edge Matrix</h2><span class="em-sub">Model edges &amp; recommended plays</span></div>' +
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
      '<div class="em-table"><div class="thead em-grid"><div>#</div><div>Game</div><div>Kickoff</div><div class="num" title="Home team’s number">Market</div><div class="num" title="Home team’s number">Model</div>' +
      '<div class="num">Edge</div><div>Probability</div><div>Recommended play</div><div>Tags</div><div></div></div>';

    var rows = visible.map(function (r, n) {
      var p = r.p, s = r.s, i = r.i, g = p.game, a = team(g.away), h = team(g.home);
      var cls = Math.abs(p.edgeForTier) < D.meta.minEdge ? 'is-off' : p.isFade ? 'is-neg' : p.qualifies ? 'is-pos' : 'is-off';
      var isTrainedGame = (g.flags || []).some(function (f) { return f.text === 'In-season model'; });
      var modelSrcLabel = isTrainedGame ? (g.modelVersion === 'upgraded' ? 'UPGRADED' : 'TRAINED') : 'UNTRAINED';
      var qb = (g.flags || []).filter(function (f) { return f.qb; });
      var tags = [];
      var st = emStatus(s);
      if (st) tags.push('<span class="em-tag ' + st[1] + '" title="' + esc(st[2]) + '">' + st[0] + '</span>');
      var lm = state.market === 'Spread' && p.qualifies ? lineMove(p) : null;
      if (lm && lm.worse > 0) tags.push('<span class="em-tag is-inj" title="Worse number than when first flagged: the market moved ' + lm.worse.toFixed(1) + ' toward the model\u2019s side">LINE ' + lm.worse.toFixed(1) + ' WORSE</span>');
      if (qb.length) tags.push('<span class="em-tag is-inj" title="' + esc(qb.map(function (f) { return abbrOf(f.team) + ' ' + f.text; }).join(' · ')) + '">INJURY</span>');
      tags.push('<span class="em-tag is-src' + (isTrainedGame ? ' is-trained' : '') + '" title="Which model priced this game">' + modelSrcLabel + '</span>');
      // Edge heat: brighter green the bigger a qualifying edge.
      var e = Math.abs(p.sideEdge || 0), heat = cls === 'is-pos' ? Math.min(0.55, 0.12 + e / 30) : 0;
      var prob = Math.max(0, Math.min(1, p.sideProb || 0));
      return '' +
        '<div class="row row--click em-row' + (i === state.selected ? ' is-selected' : '') + '" data-game="' + i + '">' +
          '<div class="row-body em-grid">' +
            '<div class="em-n">' + (n + 1) + '</div>' +
            '<div class="em-game">' + trkMark(a.abbr, 24) + '<span class="team-abbr" title="' + esc(g.away) + '">' + esc(a.abbr) + '</span>' +
              '<span class="at">at</span>' + trkMark(h.abbr, 24) + '<span class="team-abbr" title="' + esc(g.home) + '">' + esc(h.abbr) + '</span></div>' +
            '<div class="em-kick">' + esc(edgeKickoff(g)) + '<div>' + esc(edgeBook(g)) + '</div></div>' +
            '<div class="num cell-market">' + esc(p.marketLabel) + '</div>' +
            '<div class="num cell-model">' + esc(p.modelLabel) + '</div>' +
            '<div class="num"><span class="em-edge ' + cls + '" style="' + (heat ? 'background:rgba(57,240,122,' + heat.toFixed(2) + ')' : '') + '">' + esc(p.sideEdgeLabel) + '</span></div>' +
            '<div class="em-prob"><span>' + (prob * 100).toFixed(1) + '%</span><i><b style="width:' + (prob * 100).toFixed(1) + '%"></b></i></div>' +
            '<div>' + emPlay(s) + '</div>' +
            '<div class="em-tags">' + tags.join('') + '</div>' +
            '<div class="num"><button class="track-btn" title="Add to My Tracker" onclick="event.stopPropagation();window.__cfbTrack(' + trackPayload(p, s.onCard) + ')">+TRK</button></div>' +
          '</div>' +
        '</div>';
    }).join('') + '</div>';

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

  // ---- Anytime TD (10/2026, display only) ----
  // The model's chance each player scores, the fair odds that chance implies,
  // and every book's price next to it. Nothing here is an official play or
  // tracked -- see scripts/anytime_td.py for how the chance is worked out and
  // what the backtest found.
  // ---- TDs Allowed page (10/30/2026, display only) ----
  // What each FBS defense has given up to QBs / RBs / WRs / TEs this season
  // (src/features/td_allowed.py): TDs, catches, rec yds, rush yds -- per game
  // and vs what those offenses usually get. "This week" lines up this week's
  // props and anytime TD prices against the opponent's numbers.
  var TDA_STATS = [['td', 'TDs'], ['rec', 'Catches'], ['recyds', 'Rec yds'], ['rushyds', 'Rush yds']];
  var TDA_POS = [['qb', 'QB'], ['rb', 'RB'], ['wr', 'WR'], ['te', 'TE']];
  function tdaVal(r, stat, pos, basis) {
    if (pos === 'all') {
      if (basis === 'adj') return null;
      return TDA_POS.reduce(function (t, p) { return t + (r[stat + '_' + p[0] + '_pg'] || 0); }, 0);
    }
    return basis === 'adj' ? r[stat + '_' + pos + '_adj'] : r[stat + '_' + pos + '_pg'];
  }
  function tdaRanks(all) {
    // rank 1 = allows the MOST of that stat to that position (per game and vs usual)
    if (D._tdaRanks) return D._tdaRanks;
    var out = {};
    ['pg', 'adj'].forEach(function (b) {
      TDA_STATS.forEach(function (s) {
        TDA_POS.concat([['all', 'All']]).forEach(function (p) {
          var k = s[0] + '_' + p[0] + '_' + b;
          var vals = all.map(function (r) { return [r.team, tdaVal(r, s[0], p[0], b === 'adj' ? 'adj' : 'pg')]; })
            .filter(function (x) { return x[1] != null; })
            .sort(function (a, c) { return c[1] - a[1]; });
          out[k] = {};
          vals.forEach(function (x, i) { out[k][x[0]] = i > 0 && vals[i - 1][1] === x[1] ? out[k][vals[i - 1][0]] : i + 1; });
        });
      });
    });
    D._tdaRanks = out;
    return out;
  }
  function tdaHeat(rank, n) {
    // green = gives up a lot (good for overs / TDs), red = stingy
    if (rank == null) return 'var(--muted-3)';
    var f = rank / n;
    if (f <= 0.15) return 'var(--green)';
    if (f <= 0.33) return '#7FD8A4';
    if (f >= 0.85) return 'var(--red)';
    if (f >= 0.67) return '#FF9C9C';
    return 'var(--muted-2)';
  }
  function tdaFmt(v, stat, basis) {
    if (v == null) return '&ndash;';
    if (basis === 'adj') { var p = Math.round((v - 1) * 100); return (p > 0 ? '+' : '') + p + '%'; }
    return stat === 'td' || stat === 'rec' ? Number(v).toFixed(2) : Number(v).toFixed(1);
  }
  function tdaKick(t) {
    if (!t) return '';
    var d = new Date(t);
    if (isNaN(d)) return '';
    return d.toLocaleString(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' });
  }

  function renderTdAllowed() {
    var all = D.tdAllowed || [];
    var view = state.tdaView === 'week' ? 'week' : 'rank';
    var seg = function (key, val, cur, label) {
      return '<button class="tab tab--prop' + (cur === val ? ' is-active' : '') + '" data-atd-set="' + key + '" data-atd-val="' + val + '"><span>' + label + '</span></button>';
    };
    var head = '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Defense vs Position</h2></div>' +
      '<div style="font-size:10px;color:var(--muted-3)">FBS defenses &middot; this season &middot; display only, not official plays</div></div>' +
      '<div class="prop-tabs" style="margin:12px 0 4px">' + seg('tdaView', 'rank', view, 'Defense rankings') + seg('tdaView', 'week', view, 'This week\\u2019s matchups') + '</div>';
    if (!all.length) {
      return head + '<div class="empty-state">No numbers yet' + (D.tdAllowedNote ? ' (' + esc(D.tdAllowedNote) + ')' : '') + '.</div>';
    }
    return head + (view === 'week' ? renderTdaWeek(all, seg) : renderTdaRank(all, seg));
  }

  function renderTdaRank(all, seg) {
    var n = all.length, R = tdaRanks(all);
    var pos = ['qb', 'rb', 'wr', 'te', 'all'].indexOf(state.tdaPos) === -1 ? 'rb' : state.tdaPos;
    var stat = ['td', 'rec', 'recyds', 'rushyds'].indexOf(state.tdaStat) === -1 ? 'td' : state.tdaStat;
    var basis = state.tdaBasis === 'adj' && pos !== 'all' ? 'adj' : 'pg';
    var dir = state.tdaDir === 'least' ? 'least' : 'most';
    var q = (state.tdaQ || '').toLowerCase();
    var filters = '<div class="prop-tabs" style="margin:10px 0 6px">' +
      TDA_POS.concat([['all', 'All']]).map(function (p) { return seg('tdaPos', p[0], pos, p[1]); }).join('') +
      '<span style="width:12px"></span>' +
      TDA_STATS.map(function (s) { return seg('tdaStat', s[0], stat, s[1]); }).join('') + '</div>' +
      '<div class="prop-tabs" style="margin:0 0 8px">' +
      seg('tdaBasis', 'pg', basis, 'Per game') + (pos !== 'all' ? seg('tdaBasis', 'adj', basis, 'vs usual (schedule-adjusted)') : '') +
      '<span style="width:12px"></span>' + seg('tdaDir', 'most', dir, 'Most allowed') + seg('tdaDir', 'least', dir, 'Fewest allowed') +
      '<span style="width:12px"></span><input class="tda-search" placeholder="Find a team" value="' + esc(state.tdaQ || '') + '" oninput="window.__cfbTdaQ(this.value)">' +
      '</div>';
    var rows = all.filter(function (r) { return !q || r.team.toLowerCase().indexOf(q) !== -1 || (r.next && (r.next.opp || '').toLowerCase().indexOf(q) !== -1); })
      .sort(function (a, b) {
        var va = tdaVal(a, stat, pos, basis), vb = tdaVal(b, stat, pos, basis);
        var d = (vb == null ? -1e9 : vb) - (va == null ? -1e9 : va);
        return dir === 'least' ? -d : d;
      });
    var label = (pos === 'all' ? 'All' : pos.toUpperCase()) + ' ' + TDA_STATS.filter(function (s) { return s[0] === stat; })[0][1] + (basis === 'adj' ? ' vs usual' : '/gm');
    var p2 = pos === 'all' ? 'all' : pos;
    var thead = '<div class="thead tda-grid"><div>#</div><div>Defense</div><div>Next</div><div class="num">Gms</div>' +
      '<div class="num">' + esc(label) + '</div>' +
      TDA_STATS.map(function (s) { return '<div class="num">' + s[1] + '</div>'; }).join('') + '</div>';
    var cell = function (r, s) {
      var b = basis === 'adj' ? 'adj' : 'pg';
      var v = tdaVal(r, s, p2, b), rk = (R[s + '_' + p2 + '_' + b] || {})[r.team];
      return '<div class="num" title="' + (rk ? '#' + rk + ' of ' + n + ' in allowing' : '') + '"><span style="color:' + tdaHeat(rk, n) + (s === stat ? ';font-weight:700' : '') + '">' + tdaFmt(v, s, b) + '</span>' +
        (rk ? '<span class="tda-rk">' + rk + '</span>' : '') + '</div>';
    };
    var body = rows.map(function (r, i) {
      var open = state.tdaOpen === r.team;
      var v = tdaVal(r, stat, pos, basis), rk = (R[stat + '_' + p2 + '_' + (basis === 'adj' ? 'adj' : 'pg')] || {})[r.team];
      var nx = r.next ? '<span style="color:var(--muted-2)">' + esc(r.next.opp) + '</span> <span style="color:var(--muted-4);font-size:9.5px">' + esc(tdaKick(r.next.start)) + '</span>' : '<span style="color:var(--muted-4)">&ndash;</span>';
      return '<div class="row tda-row' + (open ? ' is-open' : '') + '" data-atd-set="tdaOpen" data-atd-val="' + (open ? '' : esc(r.team)) + '">' +
        '<div class="row-body tda-grid" style="padding:8px 14px;font-size:12px">' +
        '<div style="color:var(--muted-4)">' + (i + 1) + '</div>' +
        '<div style="overflow:hidden;white-space:nowrap;font-weight:600">' + (open ? '\\u25BE ' : '\\u25B8 ') + esc(r.team) + '</div>' +
        '<div style="overflow:hidden;white-space:nowrap;font-size:11px">' + nx + '</div>' +
        '<div class="num" style="color:var(--muted-3)">' + r.games + '</div>' +
        '<div class="num"><span class="tda-big" style="background:' + tdaHeat(rk, n) + '">' + tdaFmt(v, stat, basis) + '</span></div>' +
        TDA_STATS.map(function (s) { return cell(r, s[0]); }).join('') +
        '</div></div>' + (open ? tdaDetail(r, R, n) : '');
    }).join('');
    var foot = '<div class="table-foot"><span>Green = gives up a lot to that position (friendly for overs and TDs), red = stingy. Small number = national rank in allowing (1 = most). ' +
      '&ldquo;vs usual&rdquo; compares what a defense allowed with what those same offenses get against everyone else, so a soft schedule doesn&rsquo;t flatter it (+25% = 25% more than usual). ' +
      'TDs = rushing + receiving (QB = QB rushing TDs; passing TDs count for the receiver). Positions from CFBD rosters. Click a defense for its full breakdown.' +
      (D.tdAllowedNote ? ' ' + esc(D.tdAllowedNote) + '.' : '') + '</span></div>';
    return filters + thead + (body || '<div class="empty-state">No defenses match.</div>') + foot;
  }

  function tdaDetail(r, R, n) {
    var grid = '<div class="tda-detail-grid"><div></div>' + TDA_STATS.map(function (s) { return '<div class="num">' + s[1] + '/gm</div><div class="num">vs usual</div>'; }).join('') + '</div>' +
      TDA_POS.map(function (p) {
        return '<div class="tda-detail-grid"><div style="font-weight:700">' + p[1] + '</div>' + TDA_STATS.map(function (s) {
          var k = s[0] + '_' + p[0];
          var r1 = (R[k + '_pg'] || {})[r.team], r2 = (R[k + '_adj'] || {})[r.team];
          return '<div class="num"><span style="color:' + tdaHeat(r1, n) + '">' + tdaFmt(r[k + '_pg'], s[0], 'pg') + '</span><span class="tda-rk">' + (r1 || '') + '</span></div>' +
            '<div class="num"><span style="color:' + tdaHeat(r2, n) + '">' + tdaFmt(r[k + '_adj'], s[0], 'adj') + '</span><span class="tda-rk">' + (r2 || '') + '</span></div>';
        }).join('') + '</div>';
      }).join('');
    var log = (r.log || []).map(function (g) {
      var tds = TDA_POS.map(function (p) { return g[p[0]] ? p[1] + ' ' + g[p[0]] : ''; }).filter(Boolean).join(', ') || 'no rush/rec TDs';
      return '<div class="tda-log"><span style="color:var(--muted-4)">Wk ' + g.wk + '</span><span>vs ' + esc(g.vs) + '</span><span>' + tds + '</span>' +
        '<span style="color:var(--muted-3)">' + g.recyds + ' rec yds &middot; ' + g.rushyds + ' rush yds</span></div>';
    }).join('');
    var sc = (r.scorers || []).map(function (s) {
      var how = [s.rush ? s.rush + ' rush' : '', s.rec ? s.rec + ' rec' : ''].filter(Boolean).join(', ');
      return '<div class="tda-log"><span style="color:var(--muted-4)">Wk ' + s.wk + '</span><span><b>' + esc(s.player) + '</b> <span style="color:var(--muted-3)">' + esc(s.pos) + ', ' + esc(s.team) + '</span></span><span>' + how + ' TD</span><span></span></div>';
    }).join('');
    return '<div class="tda-detail">' +
      '<div class="tda-detail-title">' + esc(r.team) + ' defense by position</div>' + grid +
      '<div class="tda-cols"><div><div class="tda-detail-title">Game log (allowed)</div>' + (log || '<div style="color:var(--muted-3)">&ndash;</div>') + '</div>' +
      '<div><div class="tda-detail-title">Most recent TDs allowed</div>' + (sc || '<div style="color:var(--muted-3)">None yet.</div>') + '</div></div>' +
      '</div>';
  }

  function renderTdaWeek(all, seg) {
    var n = all.length, R = tdaRanks(all);
    var byTeam = {};
    all.forEach(function (r) { byTeam[r.team] = r; });
    var MK = { 'Receptions': 'Rec', 'Reception Yards': 'Rec yds', 'Rush Yards': 'Rush yds', 'Reception Touchdowns': 'Rec TD', 'Rush Touchdowns': 'Rush TD', 'Rush Attempts': 'Carries', 'Pass Touchdowns': 'Pass TD', 'Pass Yards': 'Pass yds' };
    var players = {};
    var key = function (x) { return (x.team || '') + '|' + (x.player_name || ''); };
    (D.propsLive || []).forEach(function (p) {
      if (!p.pos || !MK[p.market_name]) return;
      var k = key(p);
      var e = players[k] || (players[k] = { name: p.player_name, team: p.team, opp: p.opponent, pos: p.pos, start: p.start_time, props: [], atd: null });
      e.props.push(p);
    });
    (D.anytimeTd || []).forEach(function (a) {
      if (!a.pos) return;
      var k = key(a);
      var e = players[k] || (players[k] = { name: a.player_name, team: a.team, opp: a.opponent, pos: a.pos, start: a.start_time, props: [], atd: null });
      e.atd = a;
    });
    var pos = ['qb', 'rb', 'wr', 'te', 'all'].indexOf(state.tdaWpos) === -1 ? 'all' : state.tdaWpos;
    var top = ['10', '25', '50', 'all'].indexOf(state.tdaTop) === -1 ? '25' : state.tdaTop;
    var sortBy = ['td', 'yds', 'kick'].indexOf(state.tdaWsort) === -1 ? 'td' : state.tdaWsort;
    var basis = state.tdaWbasis === 'adj' ? 'adj' : 'pg';
    var ydsOf = function (p) { return p === 'WR' || p === 'TE' ? 'recyds' : 'rushyds'; };
    var list = Object.keys(players).map(function (k) { return players[k]; }).filter(function (e) { return byTeam[e.opp]; });
    list.forEach(function (e) {
      var pp = e.pos.toLowerCase(), o = byTeam[e.opp];
      e.tdRk = (R['td_' + pp + '_' + basis] || {})[e.opp];
      e.ydRk = (R[ydsOf(e.pos) + '_' + pp + '_' + basis] || {})[e.opp];
      e.tdV = o['td_' + pp + '_' + basis]; e.ydV = o[ydsOf(e.pos) + '_' + pp + '_' + basis];
    });
    var lim = top === 'all' ? 1e9 : Number(top);
    var rows = list.filter(function (e) {
      if (pos !== 'all' && e.pos.toLowerCase() !== pos) return false;
      var rk = sortBy === 'yds' ? e.ydRk : e.tdRk;
      return rk != null && rk <= lim;
    }).sort(function (a, b) {
      if (sortBy === 'kick') return (a.start || '') < (b.start || '') ? -1 : 1;
      var ka = sortBy === 'yds' ? a.ydRk : a.tdRk, kb = sortBy === 'yds' ? b.ydRk : b.tdRk;
      return (ka - kb) || ((b.atd && b.atd.model_prob || 0) - (a.atd && a.atd.model_prob || 0));
    });
    var filters = '<div class="prop-tabs" style="margin:10px 0 6px">' +
      [['all', 'All']].concat(TDA_POS).map(function (p) { return seg('tdaWpos', p[0], pos, p[1]); }).join('') +
      '<span style="width:12px"></span>' + seg('tdaWsort', 'td', sortBy, 'Opp rank: TDs') + seg('tdaWsort', 'yds', sortBy, 'Opp rank: yards') + seg('tdaWsort', 'kick', sortBy, 'Kickoff') + '</div>' +
      '<div class="prop-tabs" style="margin:0 0 8px">' +
      seg('tdaTop', '10', top, 'Opp top 10') + seg('tdaTop', '25', top, 'Top 25') + seg('tdaTop', '50', top, 'Top 50') + seg('tdaTop', 'all', top, 'Everyone') +
      '<span style="width:12px"></span>' + seg('tdaWbasis', 'pg', basis, 'Per game') + seg('tdaWbasis', 'adj', basis, 'vs usual') + '</div>';
    var odds = function (x) { if (x == null || isNaN(x)) return '&ndash;'; x = Math.round(Number(x)); return (x > 0 ? '+' : '') + x; };
    var thead = '<div class="thead tdaw-grid"><div>Player</div><div>Matchup</div><div class="num">Opp TDs to pos</div><div class="num">Opp yds to pos</div><div class="num">Anytime TD odds</div><div>Props (line &middot; model lean)</div></div>';
    var body = rows.slice(0, 300).map(function (e) {
      // One chip per market: the official play if there is one, else the line with the biggest model edge.
      var best = {};
      e.props.forEach(function (p) {
        var c = best[p.market_name];
        var score = function (x) { return (x.is_official_play ? 1e6 : 0) + (x.model_ev == null ? -1e5 : x.model_ev); };
        if (!c || score(p) > score(c)) best[p.market_name] = p;
      });
      var order = ['Receptions', 'Reception Yards', 'Reception Touchdowns', 'Rush Yards', 'Rush Attempts', 'Rush Touchdowns', 'Pass Yards', 'Pass Touchdowns'];
      var chips = order.filter(function (m) { return best[m]; }).map(function (m) { return best[m]; }).map(function (p) {
        var ev = p.model_ev, lean = p.model_lean;
        var col = ev == null ? 'var(--muted-3)' : (ev >= 10 ? 'var(--green)' : (ev > 0 ? 'var(--text)' : 'var(--muted-3)'));
        return '<span class="tda-chip" title="' + esc(p.market_name + ' ' + p.line + ' (' + (p.book_used || '') + ')' + (p.is_official_play ? ' -- official play' : '')) + '">' +
          (p.is_official_play ? '\\u2605 ' : '') + MK[p.market_name] + ' ' + p.line +
          (lean ? ' <span style="color:' + col + '">' + (lean === 'over' ? 'over' : 'under') + (ev != null ? ' ' + (ev > 0 ? '+' : '') + Math.round(ev) + '%' : '') + '</span>' : '') + '</span>';
      }).join('');
      var a = e.atd;
      var atd = '<span style="color:var(--muted-4)">&ndash;</span>';
      if (a) {
        var px = a.dk_price != null ? a.dk_price : a.best_price;
        var bk = a.dk_price != null ? 'DK' : (a.best_book_label || 'Best');
        var impl = px == null ? null : (px > 0 ? 100 / (px + 100) : -px / (-px + 100));
        var mp = a.model_prob;
        var better = a.best_price != null && a.dk_price != null && a.best_price > a.dk_price;
        atd = '<div class="tda-atd" title="' + esc('Best price: ' + odds(a.best_price) + ' (' + (a.best_book_label || '') + ')') + '">' +
          '<div><span class="tda-atd-book">' + esc(bk) + '</span> <b>' + odds(px) + '</b>' +
          (better ? ' <span class="tda-atd-best">best ' + odds(a.best_price) + '</span>' : '') + '</div>' +
          '<div class="tda-atd-sub">book ' + (impl == null ? '&ndash;' : Math.round(impl * 100) + '%') +
          ' &middot; model <span style="color:' + (mp != null && impl != null && mp > impl ? 'var(--green)' : 'var(--muted-2)') + '">' +
          (mp == null ? '&ndash;' : Math.round(mp * 100) + '%') + '</span></div></div>';
      }
      return '<div class="row"><div class="row-body tdaw-grid" style="padding:8px 14px;font-size:12px">' +
        '<div style="overflow:hidden;white-space:nowrap"><b>' + esc(e.name) + '</b> <span style="color:var(--muted-3);font-size:10px">' + esc(e.pos) + '</span></div>' +
        '<div style="overflow:hidden;white-space:nowrap;font-size:11px">' + esc(e.team) + ' <span style="color:var(--muted-4)">vs</span> <span data-atd-set="tdaJump" data-atd-val="' + esc(e.opp) + '" style="cursor:pointer;text-decoration:underline dotted">' + esc(e.opp) + '</span>' +
          ' <span style="color:var(--muted-4);font-size:9.5px">' + esc(tdaKick(e.start)) + '</span></div>' +
        '<div class="num"><span style="color:' + tdaHeat(e.tdRk, n) + ';font-weight:700">' + tdaFmt(e.tdV, 'td', basis) + '</span><span class="tda-rk">#' + e.tdRk + '</span></div>' +
        '<div class="num"><span style="color:' + tdaHeat(e.ydRk, n) + ';font-weight:700">' + tdaFmt(e.ydV, 'recyds', basis) + '</span><span class="tda-rk">#' + (e.ydRk || '') + '</span></div>' +
        '<div class="num">' + atd + '</div>' +
        '<div class="tda-chips">' + (chips || '<span style="color:var(--muted-4)">no yardage/TD props posted</span>') + '</div>' +
        '</div></div>';
    }).join('');
    var foot = '<div class="table-foot"><span>Every player with a prop or anytime TD price this week whose opponent ranks in the chosen top group for giving up TDs (or yards) to his position. ' +
      'Yards = receiving yards for WRs/TEs, rush yards for RBs/QBs. Prop chips show the line and the model&rsquo;s lean and edge (green = 10%+); \\u2605 = official play. Anytime TD odds = DraftKings price (or the best book if DK has none, plus the best price when another book beats DK); &ldquo;book&rdquo; = the chance that price implies, &ldquo;model&rdquo; = the model&rsquo;s chance he scores (green when the model is higher than the book). ' +
      'Click an opponent to open its breakdown. For fun / research &mdash; a soft matchup is not itself a tested edge.' +
      (rows.length > 300 ? ' Showing 300 of ' + rows.length + '.' : '') + '</span></div>';
    return filters + thead + (body || '<div class="empty-state">No players match. Props usually post a day or two before kickoff.</div>') + foot;
  }

  function renderAnytime() {
    var all = D.anytimeTd || [];
    // Tag from the player's role (also set by the export; recomputed here so
    // older data reads the same way).
    all.forEach(function (r) {
      if (r.model_prob == null) return;
      if (r.role === 'QB' || r.role === 'Lead RB') r.tag = 'caution';
      else if (r.role === 'Minor role' || r.role === 'Backup RB') r.tag = 'longshot';
    });
    var okTag = function (r) { return r.tag !== 'caution' && r.tag !== 'longshot'; };
    var head = '<div class="section-head"><div class="section-title"><div class="section-flag"></div><h2>Anytime TD</h2></div>' +
      '<div style="font-size:10px;color:var(--muted-3)">Model odds vs sportsbook odds &middot; not official plays</div></div>';
    if (!all.length) {
      return head + '<div class="empty-state">No anytime TD lines posted right now' +
        (D.anytimeTdNote ? ' (' + esc(D.anytimeTdNote) + ')' : '') + '. Books usually post them a day or two before kickoff.</div>';
    }
    var odds = function (x) { if (x == null || isNaN(x)) return '&ndash;'; x = Math.round(Number(x)); return (x > 0 ? '+' : '') + x; };
    var pct = function (x) { return x == null ? '&ndash;' : (x * 100).toFixed(1) + '%'; };
    var sgn = function (x) { return x == null ? '&ndash;' : (x > 0 ? '+' : '') + Number(x).toFixed(1) + '%'; };
    var tone = function (x) { return x == null ? 'var(--muted-3)' : (x > 0 ? 'var(--green)' : (x < 0 ? 'var(--red)' : 'var(--muted-3)')); };
    var basis = state.atdBasis === 'dk' ? 'dk' : 'best';
    var edgeOf = function (r) { return basis === 'dk' ? r.edge_dk : r.edge_best; };
    var priceOf = function (r) { return basis === 'dk' ? r.dk_price : r.best_price; };

    var games = {};
    all.forEach(function (r) {
      if (!games[r.fixture_id]) games[r.fixture_id] = { label: (r.away_team || '') + ' @ ' + (r.home_team || ''), t: r.start_time || '' };
    });
    var gameIds = Object.keys(games).sort(function (a, b) { return games[a].t < games[b].t ? -1 : 1; });
    var gameSel = state.atdGame && games[state.atdGame] ? state.atdGame : 'ALL';
    var view = ['all', 'beats', 'value'].indexOf(state.atdView) === -1 ? 'all' : state.atdView;

    var rows = all.filter(function (r) {
      if (gameSel !== 'ALL' && r.fixture_id !== gameSel) return false;
      var e = edgeOf(r);
      if (view === 'beats') return r.model_prob != null && e != null && e > 0;
      if (view === 'value') return r.model_prob != null && e != null && e >= 20 && okTag(r);
      return true;
    });
    var sort = ['edge', 'chance', 'kickoff'].indexOf(state.atdSort) === -1 ? 'edge' : state.atdSort;
    rows.sort(function (a, b) {
      if (sort === 'chance') return (b.model_prob || -1) - (a.model_prob || -1);
      if (sort === 'kickoff') return (a.start_time || '') < (b.start_time || '') ? -1 : ((a.start_time || '') > (b.start_time || '') ? 1 : ((edgeOf(b) || -999) - (edgeOf(a) || -999)));
      var ea = edgeOf(a), eb = edgeOf(b);
      return (eb == null ? -9999 : eb) - (ea == null ? -9999 : ea);
    });

    var seg = function (key, val, cur, label) {
      return '<button class="tab tab--prop' + (cur === val ? ' is-active' : '') + '" data-atd-set="' + key + '" data-atd-val="' + val + '"><span>' + label + '</span></button>';
    };
    var nBeats = all.filter(function (r) { var e = edgeOf(r); return r.model_prob != null && e != null && e > 0; }).length;
    var nValue = all.filter(function (r) { var e = edgeOf(r); return r.model_prob != null && e != null && e >= 20 && okTag(r); }).length;
    var filters = '<div class="prop-tabs" style="margin:12px 0 8px">' +
        seg('atdView', 'all', view, 'All (' + all.length + ')') +
        seg('atdView', 'beats', view, 'Price better than fair (' + nBeats + ')') +
        seg('atdView', 'value', view, '20%+ edge, main roles (' + nValue + ')') +
      '</div>' +
      '<div class="prop-tabs" style="margin:0 0 14px;align-items:flex-end">' +
        '<div class="pp-fgroup"><span class="pp-flabel">Price</span><div style="display:flex;gap:6px">' +
          seg('atdBasis', 'best', basis, 'Best legal book') + seg('atdBasis', 'dk', basis, 'DraftKings') + '</div></div>' +
        '<div class="pp-fgroup"><span class="pp-flabel">Sort</span><div style="display:flex;gap:6px">' +
          seg('atdSort', 'edge', sort, 'Edge') + seg('atdSort', 'chance', sort, 'Chance') + seg('atdSort', 'kickoff', sort, 'Kickoff') + '</div></div>' +
        '<div class="pp-fgroup" style="min-width:0;max-width:100%"><span class="pp-flabel">Game</span><select class="pp-select" style="max-width:100%" aria-label="Game" onchange="window.__cfbAtdGame(this.value)">' +
          '<option value="ALL"' + (gameSel === 'ALL' ? ' selected' : '') + '>All games (' + gameIds.length + ')</option>' +
          gameIds.map(function (id) { return '<option value="' + esc(id) + '"' + (gameSel === id ? ' selected' : '') + '>' + esc(games[id].label) + ', ' + esc(propKickoff(games[id].t)) + '</option>'; }).join('') +
        '</select></div>' +
      '</div>';

    if (!rows.length) {
      return head + filters + '<div class="empty-state">No players match this view.</div>' + atdFoot();
    }
    var cards = rows.slice(0, 300).map(function (r) {
      var chip = '';
      var e0 = edgeOf(r);
      if (r.tag === 'longshot') chip = '<span style="color:var(--amber);font-size:9.5px;letter-spacing:.08em" title="Small role: often does not touch the ball. In the backtest this group only looked profitable when players without a box-score line were left out">LONG SHOT</span>';
      else if (r.tag !== 'caution' && r.model_prob != null && e0 != null && e0 >= 20) chip = '<span style="color:var(--green);font-size:9.5px;letter-spacing:.08em">20%+ EDGE</span>';
      else if (r.tag === 'caution') chip = '<span style="color:var(--amber);font-size:9.5px;letter-spacing:.08em" title="QBs and lead running backs lost at every edge size in the backtest">CAUTION</span>';
      var inj = r.injury_status ? ' <span style="color:var(--red);font-size:9.5px">' + esc(r.injury_status) + '</span>' : '';
      var sub = [r.role, r.team ? schoolAbbr(r.team) + (r.opponent ? ' vs ' + schoolAbbr(r.opponent) : '') : (r.away_team + ' @ ' + r.home_team), propKickoff(r.start_time)]
        .filter(Boolean).map(esc).join(' &middot; ');
      var model = r.model_prob != null
        ? '<div class="pcard-line-row"><span>Model chance</span><span><b>' + pct(r.model_prob) + '</b> &rarr; fair <b>' + odds(r.fair_odds) + '</b></span></div>'
        : '<div class="pcard-line-row"><span style="color:var(--muted-3)">No model read (needs 2+ games this season)</span><span></span></div>';
      var line = function (label, price, book, e) {
        if (price == null) return '';
        return '<div class="pcard-line-row"><span>' + label + ' <b>' + odds(price) + '</b>' + (book ? ' <span style="color:var(--muted-3)">' + esc(book) + '</span>' : '') +
          ' <span style="color:var(--muted-4);font-size:9.5px">(' + pct(M.impliedProb(price)) + ')</span></span>' +
          '<span style="color:' + tone(e) + ';font-weight:700">' + (r.model_prob != null ? sgn(e) : '') + '</span></div>';
      };
      var others = (r.books || []).filter(function (b) { return b.book !== r.best_book && b.book !== 'draftkings'; })
        .map(function (b) { return esc(b.label) + ' ' + odds(b.price) + (b.offshore ? '*' : ''); }).join(' &middot; ');
      var proj = r.model_prob != null
        ? '<div style="font-size:9.5px;color:var(--muted-4);padding-top:5px">Projected ' + Number(r.rush_td_proj).toFixed(2) + ' rush TD + ' + Number(r.rec_td_proj).toFixed(2) + ' rec TD' +
          (r.games_played ? ' &middot; ' + r.games_played + ' games' : '') + '</div>' : '';
      return '<div class="pcard"><div class="pcard-head"><span>' + esc(r.player_name) + inj + '</span>' + chip + '</div>' +
        '<div style="font-size:10px;color:var(--muted-3);margin:-4px 0 6px">' + sub + '</div>' +
        model +
        line('Best', r.best_price, r.best_book_label, r.edge_best) +
        (r.best_book !== 'draftkings' ? line('DraftKings', r.dk_price, '', r.edge_dk) : '') +
        (others ? '<div style="font-size:9.5px;color:var(--muted-3);padding-top:5px">' + others + '</div>' : '') +
        proj + '</div>';
    }).join('');
    return head + filters + '<div class="pcard-grid">' + cards + '</div>' +
      (rows.length > 300 ? '<div class="table-foot"><span>Showing 300 of ' + rows.length + '. Pick a game to narrow it down.</span></div>' : '') +
      atdFoot();
  }

  function atdFoot() {
    return '<div class="table-foot"><span>Model chance = the rush TD and receiving TD projections added together and turned into the chance of at least one touchdown. ' +
      'Fair = the odds where that chance breaks even; a book price longer than fair is in your favor, and Edge is the expected return per 1u at that price. ' +
      'In the 2024&ndash;26 backtest these chances matched reality (27.7% said, 27.3% scored), but books price Yes well above the real rate, so most lines lose. ' +
      'QBs and lead running backs (CAUTION) lost at every edge size. Small-role players (LONG SHOT) show the biggest edges, but about a third of their lines had no box-score line (sat, or never touched the ball) and the edge disappeared once those were counted. ' +
      'Main receivers at a 20%+ edge were about break-even at DraftKings. ' +
      'Best legal book leaves out offshore books (marked *). Display only &mdash; not official plays and not tracked.</span></div>';
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
    } else if (p.waitingModel) {
      s.kind = 'waiting'; s.label = 'WAITING ON MODEL'; s.color = 'var(--muted-3)';
      s.note = 'priced by preseason ratings only; becomes a play only if the in-season model still shows an edge once it can score this game';
    } else if (isNearEvenTarget(p)) {
      s.kind = 'target'; s.label = 'TARGET · NEAR-EVEN ML'; s.color = 'var(--green)'; s.note = 'model 65%+ at +100 to +150 — small stake';
    } else if (p.qualifies && bigDogML) {
      s.kind = 'bigdog'; s.label = 'UNOFFICIAL · BIG UNDERDOG ML'; s.color = 'var(--amber)'; s.note = 'track only — longer than +150';
    } else if (p.qualifies && onCard) {
      s.kind = 'card'; s.label = 'OFFICIAL · BET CARD'; s.color = 'var(--green)'; s.note = 'wager';
    } else if (p.qualifies) {
      s.kind = 'official'; s.label = 'OFFICIAL PLAY'; s.color = 'var(--blue-light)'; s.note = 'not on the Bet Card — track only';
    } else if (p.bigDogWatch) {
      s.kind = 'dogwatch'; s.label = 'WATCH · BIG DOG 28–34.5'; s.color = 'var(--blue-light)';
      s.note = 'watch only — not official. The edge is at the opening line, so track it when lines first post';
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


  // ---- Team matchup (11/2026): every offensive and defensive category,
  // national ranks (1 = best for that unit), one team's offense against
  // the other's defense. PFF categories show ranks only (licensed data).
  function tmFmt(v, fmt) {
    if (v == null || !fmt) return '';
    if (fmt.indexOf('%') >= 0) return (v * 100).toFixed(1) + '%';
    var m = /\\.(\\d)f/.exec(fmt), d = m ? +m[1] : 1, txt = Number(v).toFixed(d);
    return fmt.indexOf('+') >= 0 && v > 0 ? '+' + txt : txt;
  }
  function tmPill(e) {
    if (!e) return '<span class="tm-pill is-na">—</span>';
    var q = e.rank / e.of, cls = q <= 0.25 ? 'is-top' : (q > 0.75 ? 'is-bot' : '');
    return '<span class="tm-pill ' + cls + '" title="' + e.rank + ' of ' + e.of + ' FBS teams">#' + e.rank + '</span>';
  }
  function renderTeamMatchup(p) {
    var TM = D.teamMatchup || {}, meta = D.teamMatchupMeta || {};
    if (!p || !meta.cats) return '';
    var g = p.game, A = team(g.away), H = team(g.home);
    var pa = TM[g.awaySchool], ph = TM[g.homeSchool];
    var head = '<div class="tm-card"><div class="tm-top"><div class="tm-title">' + icon('ratings') + '<span>Team matchup</span><em>national ranks among FBS &middot; 1 = best for that unit</em></div>';
    if (!pa || !ph) return head + '<div class="ep-empty">No season ranks for ' + esc(!pa ? A.abbr : H.abbr) + ' yet (FBS teams only).</div></div>';
    var view = state.tmView === 'home' ? 'home' : 'away';
    var offT = view === 'away' ? A : H, defT = view === 'away' ? H : A;
    var offP = view === 'away' ? pa : ph, defP = view === 'away' ? ph : pa;
    var cats = {}; meta.cats.forEach(function (c) { cats[c.key] = c; });
    // Who wins each category: offense rank vs defense rank, as a share of the field.
    var winsO = 0, winsD = 0, n = 0;
    var body = (meta.groups || []).map(function (grp) {
      var rows = grp.keys.map(function (k) {
        var c = cats[k], o = offP.off[k], d = defP.def[k];
        if (!c || (!o && !d)) return '';
        var diff = (o && d) ? (d.rank - o.rank) / Math.max(o.of, d.of) : 0;   // + = offense better ranked
        if (o && d) { n++; if (diff >= 0.2) winsO++; else if (diff <= -0.2) winsD++; }
        var w = Math.min(50, Math.abs(diff) * 60).toFixed(1);
        var bar = '<div class="tm-bar"><i class="tm-mid"></i>' + (o && d ? '<b class="' + (diff >= 0 ? 'is-o' : 'is-d') + '" style="' + (diff >= 0 ? 'right:50%' : 'left:50%') + ';width:' + w + '%"></b>' : '') + '</div>';
        var edge = !(o && d) ? '' : (diff >= 0.2 ? '<span class="tm-edge is-o">' + esc(offT.abbr) + '</span>' : (diff <= -0.2 ? '<span class="tm-edge is-d">' + esc(defT.abbr) + '</span>' : '<span class="tm-edge">even</span>'));
        return '<div class="tm-row">' +
          '<div class="tm-lab">' + esc(c.off) + (c.src === 'pff' ? ' <span class="tm-src">PFF</span>' : '') + '</div>' +
          '<div class="tm-v">' + tmPill(o) + '<span>' + esc(o ? tmFmt(o.value, c.fmt) : '') + '</span></div>' +
          bar +
          '<div class="tm-v is-r"><span>' + esc(d ? tmFmt(d.value, c.fmt) : '') + '</span>' + tmPill(d) + '</div>' +
          '<div class="tm-lab is-r">' + esc(c.def) + '</div>' +
          '<div class="tm-e">' + edge + '</div></div>';
      }).join('');
      return rows ? '<div class="tm-group">' + esc(grp.name) + '</div>' + rows : '';
    }).join('');
    var tabs = '<div class="tm-tabs">' +
      '<button class="' + (view === 'away' ? 'is-on' : '') + '" onclick="window.__cfbTmView(\\'away\\')">' + esc(A.abbr) + ' offense vs ' + esc(H.abbr) + ' defense</button>' +
      '<button class="' + (view === 'home' ? 'is-on' : '') + '" onclick="window.__cfbTmView(\\'home\\')">' + esc(H.abbr) + ' offense vs ' + esc(A.abbr) + ' defense</button></div>';
    var sum = '<div class="tm-sum"><div><b class="is-o">' + winsO + '</b><span>' + esc(offT.abbr) + ' offense edges</span></div>' +
      '<div><b>' + (n - winsO - winsD) + '</b><span>even</span></div>' +
      '<div><b class="is-d">' + winsD + '</b><span>' + esc(defT.abbr) + ' defense edges</span></div></div>';
    var cols = '<div class="tm-row tm-headrow"><div class="tm-lab">' + trkMark(offT.abbr, 20) + ' ' + esc(offT.abbr) + ' offense</div><div></div><div></div><div></div>' +
      '<div class="tm-lab is-r">' + esc(defT.abbr) + ' defense ' + trkMark(defT.abbr, 20) + '</div><div></div></div>';
    return head + tabs + '</div>' + sum + cols + body +
      '<div class="tm-foot">An edge is shown when one side ranks at least 20% of the field better. Ranks are this season, games so far. PFF categories show ranks only.</div></div>';
  }
  window.__cfbTmView = function (v) { state.tmView = v; render(); };
  // ---- Game breakdown (11/2026): a plain-English read of the team
  // matchup ranks above, both directions, ending with the model's play.
  // Descriptive only -- the play itself still comes from the model's
  // number against the market, not from these ranks.
  function tmCompare(offP, defP) {
    var meta = D.teamMatchupMeta || {}, out = [];
    (meta.cats || []).forEach(function (c) {
      var o = offP.off[c.key], d = defP.def[c.key];
      if (!o || !d) return;
      out.push({ c: c, o: o, d: d, diff: (d.rank - o.rank) / Math.max(o.of, d.of) });
    });
    return out;
  }
  function gbGroupLean(rows, keys) {
    var r = rows.filter(function (x) { return keys.indexOf(x.c.key) >= 0; });
    if (!r.length) return null;
    return r.reduce(function (s, x) { return s + x.diff; }, 0) / r.length;
  }
  function gbList(rows, n) {
    return rows.slice(0, n).map(function (x) {
      return esc(x.c.off.replace(/ \\(.*\\)$/, '')) + ' <span class="gb-rk">#' + x.o.rank + ' vs #' + x.d.rank + '</span>';
    }).join(', ');
  }
  function gbSide(offT, defT, rows) {
    var oE = rows.filter(function (x) { return x.diff >= 0.2; }).sort(function (a, b) { return b.diff - a.diff; });
    var dE = rows.filter(function (x) { return x.diff <= -0.2; }).sort(function (a, b) { return a.diff - b.diff; });
    var meta = D.teamMatchupMeta || {}, leans = [];
    (meta.groups || []).forEach(function (g) {
      if (g.name.indexOf('Situational') === 0) return;
      var v = gbGroupLean(rows, g.keys);
      if (v != null) leans.push({ name: ({ 'Run game': 'the run game', 'Pass game': 'the pass game' })[g.name] || 'scoring and efficiency', v: v });
    });
    var txt = '<b>' + esc(offT.abbr) + ' offense vs ' + esc(defT.abbr) + ' defense:</b> ';
    txt += esc(offT.abbr) + ' wins ' + oE.length + ' of ' + rows.length + ' categories, ' +
      esc(defT.abbr) + '&rsquo;s defense wins ' + dE.length + '. ';
    if (oE.length) txt += 'Biggest ' + esc(offT.abbr) + ' edges: ' + gbList(oE, 3) + '. ';
    if (dE.length) txt += 'Biggest ' + esc(defT.abbr) + ' defense edges: ' + gbList(dE, 2) + '. ';
    var best = leans.slice().sort(function (a, b) { return b.v - a.v; })[0];
    var worst = leans.slice().sort(function (a, b) { return a.v - b.v; })[0];
    if (best && best.v >= 0.12) txt += esc(offT.abbr) + '&rsquo;s clearest advantage is ' + esc(best.name) + '.';
    else if (worst && worst.v <= -0.12) txt += esc(defT.abbr) + '&rsquo;s defense has the edge in ' + esc(worst.name) + '.';
    else txt += 'No unit has a clear advantage here.';
    // Net score for this direction: + = offense, - = defense.
    var net = rows.reduce(function (s, x) { return s + x.diff; }, 0) / Math.max(1, rows.length);
    return { html: txt, net: net, oE: oE.length, dE: dE.length };
  }
  // Game breakdown extras (11/2026): game info, injuries (with whether the
  // player is one of the team's main producers), players to know, and any
  // official props in the game.
  function gbNameKey(n) {
    var parts = String(n || '').replace(/[.]/g, ' ').trim().split(/ +/);
    var suf = { jr: 1, sr: 1, ii: 1, iii: 1, iv: 1 };
    while (parts.length > 1 && suf[parts[parts.length - 1].toLowerCase()]) parts.pop();
    return (parts[0] || '').charAt(0).toLowerCase() + '|' + (parts[parts.length - 1] || '').toLowerCase();
  }
  function gbFantasy(school) {
    return (D.fantasy || []).filter(function (f) { return f.team === school; })
      .sort(function (a, b) { return (b.projected_points || 0) - (a.projected_points || 0); });
  }
  function gbRole(fan, inj) {
    // Rank the injured player within his position on his team by projected points.
    var k = gbNameKey(inj.player), pos = inj.pos;
    var grp = (pos === 'WR' || pos === 'TE') ? 'WR/TE' : pos;
    var same = fan.filter(function (f) { return f.position === grp; });
    for (var i = 0; i < same.length; i++) {
      if (gbNameKey(same[i].player_name) === k) {
        if (grp === 'QB') return i === 0 ? 'starting QB' : 'backup QB';
        if (grp === 'WR/TE') return i === 0 ? 'top pass catcher' : (i < 3 ? 'No. ' + (i + 1) + ' pass catcher' : null);
        if (grp === 'RB') return i === 0 ? 'lead back' : (i === 1 ? 'No. 2 back' : null);
        return null;
      }
    }
    return null;
  }
  function gbInjuries(T, school, list) {
    var fan = gbFantasy(school);
    var rows = (list || []).slice().sort(function (a, b) {
      return ((a.pos === 'QB') ? 0 : 1) - ((b.pos === 'QB') ? 0 : 1) || (INJ_ORDER[a.status] || 9) - (INJ_ORDER[b.status] || 9);
    });
    if (!rows.length) return { html: '<b>' + esc(T.abbr) + ':</b> nothing listed.', key: [] };
    var key = [], bits = rows.map(function (i) {
      var role = gbRole(fan, i), st = i.status || '';
      var out = /^(Out|IR|Doubtful)/.test(st);
      if (role || i.pos === 'QB') key.push({ player: i.player, pos: i.pos, status: st, role: role || (i.pos === 'QB' ? 'QB' : null) });
      return '<span class="gb-inj' + (out ? ' is-out' : '') + (role ? ' is-key' : '') + '">' + esc(i.player) + ' ' + esc(i.pos || '') +
        ' <i>' + esc(st) + (i.injury && i.injury !== 'Undisclosed' ? ', ' + esc(i.injury.toLowerCase()) : '') + '</i>' +
        (role ? ' <em>' + esc(role) + '</em>' : '') + '</span>';
    });
    var nOut = rows.filter(function (i) { return /^(Out|IR|Doubtful)/.test(i.status || ''); }).length;
    return { html: '<b>' + esc(T.abbr) + '</b> <span class="gb-rk">' + rows.length + ' listed, ' + nOut + ' out/doubtful</span><div class="gb-injs">' + bits.join('') + '</div>', key: key };
  }
  function gbPlayers(T, school) {
    var fan = gbFantasy(school), pick = [], seen = {};
    [['QB', 1], ['RB', 1], ['WR/TE', 2]].forEach(function (pp) {
      fan.filter(function (f) { return f.position === pp[0]; }).slice(0, pp[1]).forEach(function (f) {
        if ((f.projected_points || 0) >= 6 && !seen[f.player_name]) { seen[f.player_name] = 1; pick.push(f); }
      });
    });
    if (!pick.length) return '';
    return '<b>' + esc(T.abbr) + ':</b> ' + pick.map(function (f) {
      var s = f.stat_breakdown || {}, line = [];
      if (f.position === 'QB') { line.push(Math.round(s.pass_yds || 0) + ' pass yds'); if ((s.rush_yds || 0) >= 20) line.push(Math.round(s.rush_yds) + ' rush'); }
      else if (f.position === 'RB') { line.push(Math.round(s.rush_yds || 0) + ' rush yds'); if ((s.rec_yds || 0) >= 15) line.push(Math.round(s.rec_yds) + ' rec'); }
      else line.push((s.receptions || 0).toFixed(1) + ' rec, ' + Math.round(s.rec_yds || 0) + ' yds');
      return esc(f.player_name) + ' <span class="gb-rk">' + esc(f.position === 'WR/TE' ? 'WR/TE' : f.position) + ' &middot; ' + line.join(', ') + '</span>';
    }).join('; ') + '.';
  }
  function gbProps(g) {
    var mine = (D.propsLive || []).filter(function (r) {
      return r.is_official_play && (r.team === g.awaySchool || r.team === g.homeSchool) &&
        (!g.kickoffIso || !r.start_time || Math.abs(new Date(r.start_time) - new Date(g.kickoffIso)) < 864e5);
    });
    // 11/2026: each official prop now carries its "why it's a play" read.
    var seen = {}, out = [];
    mine.sort(function (a, b) { return (b.model_ev || 0) - (a.model_ev || 0); }).forEach(function (r) {
      var k = r.player_name + '|' + r.market_name;
      if (seen[k]) return; seen[k] = 1;
      out.push(gbWhyProp(r, g));
    });
    return out;
  }
  function gbInfo(p) {
    var g = p.game, A = team(g.away), H = team(g.home), bits = [];
    bits.push(esc(edgeKickoff(g)));
    var neutral = (g.flags || []).some(function (f) { return f.text === 'Neutral site'; });
    if (neutral) bits.push('neutral site');
    if (g.marketTotal != null && g.marketSpread != null) {
      var hT = (g.marketTotal - g.marketSpread) / 2, aT = g.marketTotal - hT;
      bits.push('total ' + g.marketTotal + ' (implied ' + esc(A.abbr) + ' ' + aT.toFixed(1) + ', ' + esc(H.abbr) + ' ' + hT.toFixed(1) + ')');
    } else if (g.marketTotal != null) bits.push('total ' + g.marketTotal);
    var trained = (g.flags || []).some(function (f) { return f.text === 'In-season model'; });
    bits.push(trained ? (g.modelVersion === 'upgraded' ? 'upgraded in-season model' : 'trained in-season model') : 'preseason ratings only (no in-season model yet)');
    var hp = 1 - M.normalCdf(0, M.homeMargin(g), p.sd);
    bits.push('model win chance ' + esc(A.abbr) + ' ' + ((1 - hp) * 100).toFixed(0) + '% / ' + esc(H.abbr) + ' ' + (hp * 100).toFixed(0) + '%');
    return bits.join(' &middot; ');
  }
  // ---- "Why it's a play" (11/2026) ----
  // For every official prop in the game and the game's Bet Card side: the
  // rule it qualifies under and how that rule has done, the model's number
  // against the price, what the projection or line is built on, the matchup
  // and what could break it. Everything here is read from data the model
  // already uses; nothing here changes which plays are official.
  function gbOrd(n) { var s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); }
  function gbSigned(x, d) { var v = Number(x).toFixed(d); if (Number(v) === 0) v = (0).toFixed(d); return (Number(v) > 0 ? '+' : '') + v; }
  function gbPrice(x) { return x == null || isNaN(x) ? 'no price' : (x > 0 ? '+' : '') + Number(x); }
  function gbNum(x, d) { return x == null || isNaN(x) ? '&mdash;' : Number(x).toFixed(d == null ? 1 : d); }
  function gbRec(r) { return r ? esc(r.record) + ', ' + gbSigned(r.roi, 1) + '% ROI, ' + gbSigned(r.units, 1) + 'u' : null; }
  var GB_RULES = {
    rush_under: { key: 'rush_under', what: 'Rush yards unders are official at any edge, but only against a tough run defense (top third of FBS).' },
    rec_under: { key: 'rec_under', what: 'Receptions unders are official at a 30%+ edge.' },
    rec_yds_over: { key: 'rec_yds_over', what: 'Receiving yards overs are official at a 15% to 30% edge on a line of 25.5+ (a player with a real role in the offense).' }
  };
  var GB_STAT = { 'Rush Yards': 'rush yds', 'Receptions': 'catches', 'Reception Yards': 'rec yds' };
  // 11/2026: injury-adjusted rush props carry both models' reads. The
  // toggle switches the card between them; the official play is always the
  // injury-aware read (the live rule).
  function gbInjToggle(r) {
    var ni = r.no_injury;
    if (!ni || !(r.injury_adjusted && r.injury_adjusted.length)) return '';
    var unit = r.market_name === 'Rush Yards' ? 'rush yds' : String(r.market_name || '').toLowerCase();
    var tough = r.play_kind === 'rush_under' || r.run_def === 'tough';
    function pane(v, isInj, on) {
      var lean = v.model_lean, lp = v.model_lean_probability, ev = v.model_ev;
      var play = lean === 'under' && ev != null && ev >= 0 && tough;
      var verdict = isInj ? (r.is_official_play ? 'Official play' : 'Not a play')
        : play ? (lean === r.model_lean ? 'Would also be a play' : 'Would be a play') : lean !== r.model_lean ? 'Would lean the other way (' + esc(lean) + ')' : 'Would not be a play';
      return '<div class="gb-inj-pane' + (on ? ' is-on' : '') + '" data-pane="' + (isInj ? 'inj' : 'base') + '">' +
        '<div class="gb-inj-row"><span class="gb-inj-big">' + gbNum(v.model_predicted_value) + '</span> <span class="gb-rk">' + esc(unit) + ' projected vs ' + r.line + '</span></div>' +
        '<div class="gb-inj-row">' + esc(lean) + ' ' + (lp != null ? (lp * 100).toFixed(1) + '%' : '&mdash;') + ' &middot; edge ' + (ev != null ? gbSigned(ev, 1) + '%' : '&mdash;') +
        ' &middot; <b>' + verdict + '</b></div>' +
        '<div class="gb-inj-note">' + (isInj ? 'Accounts for ' + esc(r.injury_adjusted.join(', ')) + ' being out: uses his share of the carries in the games without them.'
          : 'The regular model: season and last-2-game averages, including games those teammates played.') + '</div></div>';
    }
    var injV = { model_predicted_value: r.model_predicted_value, model_lean: r.model_lean, model_lean_probability: r.model_lean_probability, model_ev: r.model_ev };
    return '<div class="gb-inj"><div class="gb-inj-tog"><button type="button" class="is-on" data-pane="inj">With injuries</button><button type="button" data-pane="base">Without injuries</button></div>' +
      pane(injV, true, true) + pane(ni, false, false) + '</div>';
  }
  document.addEventListener('click', function (e) {
    var b = e.target && e.target.closest ? e.target.closest('.gb-inj-tog button') : null;
    if (!b) return;
    var box = b.closest('.gb-inj'), want = b.getAttribute('data-pane');
    box.querySelectorAll('.gb-inj-tog button').forEach(function (x) { x.classList.toggle('is-on', x === b); });
    box.querySelectorAll('.gb-inj-pane').forEach(function (x) { x.classList.toggle('is-on', x.getAttribute('data-pane') === want); });
  });
  function gbWhyProp(r, g) {
    var over = r.model_lean === 'over', stat = GB_STAT[r.market_name] || String(r.market_name || '').toLowerCase();
    var oppAb = schoolAbbr(r.opponent), meAb = schoolAbbr(r.team);
    var rule = GB_RULES[r.play_kind], ruleRec = rule ? (D.propRules || {})[rule.key] : null;
    var price = over ? r.over_price : r.under_price, be = price != null ? M.impliedProb(Number(price)) : null;
    var lp = r.model_lean_probability, out = [];
    // 1) The rule.
    if (rule) out.push('<b>The rule:</b> ' + rule.what + (ruleRec ? ' Backtest for this rule: ' + gbRec(ruleRec) +
      ' (' + ruleRec.seasons + ' season' + (ruleRec.seasons === 1 ? '' : 's') + ').' : ''));
    // 2) The number.
    out.push('<b>The number:</b> the model projects ' + gbNum(r.model_predicted_value) + ' ' + stat + ' against a line of ' + r.line +
      (lp != null ? ', a ' + (lp * 100).toFixed(1) + '% chance of the ' + (over ? 'over' : 'under') : '') +
      (be != null ? '. At ' + gbPrice(price) + (r.book_used ? ' (' + esc(bookLabel(r.book_used)) + ')' : '') + ' it needs ' + (be * 100).toFixed(1) + '%' : '') +
      (r.model_ev != null ? ', so the edge is ' + gbSigned(r.model_ev, 1) + '%.' : '.'));
    // 3) What the projection is built on.
    var gl = (D.propGameLogs || {})[r.game_log_key] || [], built = [];
    if (gl.length) {
      var vals = gl.map(function (x) { return Number(x[2]); });
      var avg = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
      var l2 = vals.slice(-2), l2avg = l2.reduce(function (a, b) { return a + b; }, 0) / l2.length;
      var hits = vals.filter(function (v) { return over ? v > r.line : v < r.line; }).length;
      built.push('he&rsquo;s averaged ' + avg.toFixed(1) + ' ' + stat + ' in ' + vals.length + ' game' + (vals.length === 1 ? '' : 's') +
        ' (' + l2avg.toFixed(1) + ' over his last ' + l2.length + ') and went ' + (over ? 'over ' : 'under ') + r.line + ' in ' + hits + ' of ' + vals.length);
    }
    var mi = r.model_inputs || {};
    var isRush = r.market_name === 'Rush Yards';
    if (isRush) {
      var cs = mi.roll_carry_share != null ? mi.roll_carry_share : r.carry_share;
      if (cs != null) built.push('he gets ' + (cs * 100).toFixed(0) + '% of ' + esc(meAb) + '&rsquo;s carries' +
        (mi.last2_carry_share != null ? ' (' + (mi.last2_carry_share * 100).toFixed(0) + '% the last 2)' : '') +
        (mi.roll_rush_att != null ? ', about ' + mi.roll_rush_att.toFixed(1) + ' a game' : ''));
      if (mi.roll_team_rush_att != null) built.push(esc(meAb) + ' runs it ' + mi.roll_team_rush_att.toFixed(1) + ' times a game');
    } else {
      var rs = mi.roll_rec_share != null ? mi.roll_rec_share : r.rec_share;
      if (rs != null) built.push('he catches ' + (rs * 100).toFixed(0) + '% of ' + esc(meAb) + '&rsquo;s receptions' +
        (mi.last2_rec_share != null ? ' (' + (mi.last2_rec_share * 100).toFixed(0) + '% the last 2)' : ''));
      if (mi.pff_matched === 1 && mi.roll_pff_routes != null) {
        built.push('PFF: ' + mi.roll_pff_routes.toFixed(1) + ' routes a game' +
          (mi.roll_pff_route_part != null ? ' (on the field for ' + (mi.roll_pff_route_part * 100).toFixed(0) + '% of pass plays' +
            (mi.last2_pff_route_part != null ? ', ' + (mi.last2_pff_route_part * 100).toFixed(0) + '% the last 2' : '') + ')' : '') +
          (mi.roll_pff_target_share != null ? ', ' + (mi.roll_pff_target_share * 100).toFixed(0) + '% target share' : '') +
          (mi.roll_pff_tprr != null ? ', ' + mi.roll_pff_tprr.toFixed(2) + ' targets per route' : '') +
          (mi.roll_pff_adot != null && r.market_name === 'Reception Yards' ? ', ' + mi.roll_pff_adot.toFixed(1) + '-yard average depth of target' : ''));
      }
      if (mi.roll_team_pass_att != null) built.push(esc(meAb) + ' throws ' + mi.roll_team_pass_att.toFixed(1) + ' passes a game');
    }
    if (mi.team_implied_total != null) built.push('the model also counts the game script: ' + esc(meAb) + ' is ' +
      (mi.team_spread != null ? (mi.team_spread < 0 ? 'a ' + Math.abs(mi.team_spread) + '-pt favorite' : mi.team_spread > 0 ? 'a ' + mi.team_spread + '-pt underdog' : 'a pick&rsquo;em') + ' with ' : '') +
      'an implied ' + mi.team_implied_total.toFixed(1) + ' points');
    if (built.length) out.push('<b>Built on:</b> ' + built.join('; ') + '.');
    // 4) Matchup.
    var mu = [];
    if (r.mu_grade) {
      var allowed = Object.keys(r.mu_allowed || {}).map(function (k) {
        var v = r.mu_allowed[k];
        return 'allows ' + Math.abs(v) + '% ' + (v < 0 ? 'fewer' : 'more') + ' ' + esc(k) + ' than its opponents usually get';
      });
      mu.push(esc(oppAb) + '&rsquo;s defense ranks ' + gbOrd(r.mu_rank) + ' of ' + r.mu_of + ' against ' + esc(r.mu_vs) + ' (1st = toughest)' +
        (allowed.length ? ' and ' + allowed.join(', ') : '') + '. Matchup grade <b>' + esc(r.mu_grade) + '</b> for this ' + (over ? 'over' : 'under'));
      var byG = ((D.propTracking || {}).byMatchup || {}).official || {}, gr = byG[String(r.mu_grade).charAt(0)];
      if (gr && gr.graded) mu.push('official props graded ' + esc(String(r.mu_grade).charAt(0)) + ' are ' + esc(gr.record) + ' (' + gbSigned(gr.units, 1) + 'u) since ' +
        ((D.propTracking || {}).byMatchupSince ? esc(String(D.propTracking.byMatchupSince).slice(5, 10).replace('-', '/')) : 'tracking began') + (gr.graded < 20 ? ', a small sample' : ''));
      if (/^[DF]/.test(r.mu_grade)) mu.push('the defense works against this pick, so the play rests on the projection, not the matchup');
    } else if (r.matchup === 'tough' || r.matchup === 'soft') {
      mu.push(esc(oppAb) + ' has been ' + r.matchup + ' on ' + esc(r.matchup_vs || 'his position') + (r.matchup === 'tough' && !over ? ', which fits an under (unders at 10%+ vs tough defenses went 260-164 over three seasons)' : ''));
    }
    if (mu.length) out.push('<b>Matchup:</b> ' + mu.join('; ') + '.');
    // 5) What could break it.
    var risk = [];
    if (r.injury_status) risk.push('he&rsquo;s listed ' + esc(String(r.injury_status).toLowerCase()) + ' himself');
    // Teammates at his position on the report: out = more work for him,
    // questionable = his role could shrink if they play.
    var mates = (r.team === g.awaySchool ? g.injAway : g.injHome) || [], myGrp = r.pos === 'TE' ? 'WR' : r.pos;
    var last = (function () { var k = gbNameKey(r.player_name).split('|')[1] || ''; return k.charAt(0).toUpperCase() + k.slice(1); })();
    var mOut = [], mQ = [];
    mates.forEach(function (i) {
      var grp = i.pos === 'TE' ? 'WR' : i.pos;
      if (grp !== myGrp || gbNameKey(i.player) === gbNameKey(r.player_name)) return;
      (/^(Out|IR|Doubtful)/.test(i.status || '') ? mOut : mQ).push(esc(i.player) + ' (' + esc(i.pos) + ')');
    });
    if (mOut.length && !gbRoleChange(r, g)) risk.push(mOut.join(', ') + (mOut.length > 1 ? ' are' : ' is') + ' out at his position, so more work for ' + esc(last) +
      (over ? ' (helps the over)' : ' (the risk to the under)'));
    if (mQ.length) risk.push(mQ.join(', ') + (mQ.length > 1 ? ' are' : ' is') + ' questionable at his position; if ' + (mQ.length > 1 ? 'they play' : 'he plays') + ', ' +
      esc(last) + '&rsquo;s role could shrink' + (over ? ' (the risk to the over)' : ' (helps the under)'));
    if (gl.length >= 4) {
      var hr = gl.filter(function (x) { return over ? Number(x[2]) > r.line : Number(x[2]) < r.line; }).length;
      if (hr / gl.length < 0.5) risk.push('he&rsquo;s cleared this ' + (over ? 'over' : 'under') + ' in only ' + hr + ' of ' + gl.length + ' games, so the pick leans on the model&rsquo;s read of this matchup and his recent games more than his season line');
    }
    var ff = r.first_flag;
    if (ff && ff.price != null && price != null && (Number(ff.price) !== Number(price) || (ff.line != null && Number(ff.line) !== Number(r.line)))) {
      var better = Number(price) > Number(ff.price);
      risk.push('first logged at ' + gbPrice(Number(ff.price)) + (ff.line != null && Number(ff.line) !== Number(r.line) ? ' on ' + ff.line : '') +
        ', now ' + gbPrice(Number(price)) + ' on ' + r.line + (better ? ' (moved away from the play)' : ' (the market has moved toward the play)'));
    }
    if (risk.length) out.push('<b>Watch:</b> ' + risk.join('; ') + '.');
    return '<div class="gb-why-card"><div class="gb-why-head"><span class="gb-prop">&#9733; <button type="button" class="pc-open"' + pcOpenAttr(r) + 'title="Open the player card">' + esc(r.player_name) + '</button> ' + esc(r.model_lean) + ' ' + r.line + ' ' +
      esc(String(r.market_name || '').toLowerCase()) + '</span> <span class="gb-rk">' + esc(meAb) + ' vs ' + esc(oppAb) + '</span></div>' +
      gbInjToggle(r) +
      gbFootballProp(r, g) +
      '<details class="gb-nums"><summary>The numbers behind it</summary><ul class="gb-why-list"><li>' + out.join('</li><li>') + '</li></ul></details></div>';
  }
  // Side plays: the inputs of the spread model, in the picked team's favor
  // (+) or against it (-). Diffs are home minus away; for defensive
  // per-play numbers lower is better.
  var GB_GROUPS = ['form', 'ratings', 'scoring', 'eff', 'pace', 'returning', 'home', 'rest', 'qb', 'weather'];
  function gbInputText(key, v, S, O, sgn, vals) {
    var x = function (k) { return vals[k] == null ? null : sgn * vals[k]; };
    var more = function (n, u, d) { return Math.abs(n).toFixed(d) + ' ' + u + ' ' + (n >= 0 ? 'better' : 'worse'); };
    if (key === 'form') { var f = x('ewm_adj_margin_diff'); return f == null ? null : esc(S) + '&rsquo;s opponent-adjusted scoring margin, recent games weighted most, is ' + more(f, 'pts a game', 1) + ' than ' + esc(O) + '&rsquo;s'; }
    if (key === 'ratings') { var sp = x('sp_rating_diff'), co = x('core_overall_diff'), bits = [];
      if (sp != null) bits.push('SP+ ' + gbSigned(sp, 1)); if (co != null) bits.push('CORE ' + gbSigned(co, 1));
      return bits.length ? 'Team ratings, ' + esc(S) + ' minus ' + esc(O) + ': ' + bits.join(', ') : null; }
    if (key === 'scoring') { var m = x('roll_margin_diff'), pf = x('roll_ppg_for_diff'), pa = x('roll_ppg_against_diff');
      var sc = [];
      if (pf != null) sc.push('scores ' + Math.abs(pf).toFixed(1) + (pf >= 0 ? ' more' : ' fewer'));
      if (pa != null) sc.push('allows ' + Math.abs(pa).toFixed(1) + (pa <= 0 ? ' fewer' : ' more'));
      return m == null ? null : 'Season scoring margin ' + more(m, 'pts a game', 1) + (sc.length ? ' (' + esc(S) + ' ' + sc.join(', ') + ')' : ''); }
    if (key === 'eff') { var n = x('ewm_net_ppa_diff'), o = x('ewm_off_ppa_diff'), d = vals.ewm_def_ppa_diff == null ? null : -sgn * vals.ewm_def_ppa_diff, sr = x('ewm_off_sr_diff');
      return n == null ? null : 'Per play (EPA, recent games weighted most): net ' + gbSigned(n, 2) + (o != null ? ', offense ' + gbSigned(o, 2) : '') + (d != null ? ', defense ' + gbSigned(d, 2) : '') +
        (sr != null ? ', offensive success rate ' + gbSigned(sr * 100, 1) + ' pts' : '') + ' for ' + esc(S); }
    if (key === 'pace') { var pc = x('pace_diff'); return pc == null ? null : 'Tempo (plays per drive) ' + gbSigned(pc, 2) + ' for ' + esc(S); }
    if (key === 'returning') { var rp = x('returning_production_diff'); return rp == null ? null : 'Returning production ' + gbSigned(rp * 100, 0) + ' pts for ' + esc(S); }
    if (key === 'home') return vals.home_field ? (sgn > 0 ? esc(S) + ' is at home' : esc(O) + ' is at home') : 'Neutral site';
    if (key === 'rest') { var rd = x('rest_diff'); return rd == null || rd === 0 ? null : esc(S) + ' has ' + Math.abs(rd) + ' day' + (Math.abs(rd) === 1 ? '' : 's') + ' ' + (rd > 0 ? 'more' : 'less') + ' rest'; }
    if (key === 'qb') { var mine = sgn > 0 ? vals.home_qb_out : vals.away_qb_out, theirs = sgn > 0 ? vals.away_qb_out : vals.home_qb_out;
      return mine ? esc(S) + '&rsquo;s starting QB is out' : theirs ? esc(O) + '&rsquo;s starting QB is out' : null; }
    if (key === 'weather') return vals.temperature != null ? 'Forecast ' + Math.round(vals.temperature) + '&deg;F, wind ' + Math.round(vals.wind_speed || 0) + ' mph' : null;
    return null;
  }
  var GB_GROUP_KEYS = { form: ['ewm_adj_margin_diff'], ratings: ['sp_rating_diff', 'core_overall_diff'], scoring: ['roll_margin_diff', 'roll_ppg_for_diff', 'roll_ppg_against_diff'],
    eff: ['ewm_net_ppa_diff', 'ewm_off_ppa_diff', 'ewm_def_ppa_diff', 'ewm_off_sr_diff', 'ewm_def_sr_diff'], pace: ['pace_diff'], returning: ['returning_production_diff'],
    home: ['home_field'], rest: ['rest_diff'], qb: ['home_qb_out', 'away_qb_out'], weather: ['temperature', 'wind_speed', 'precipitation'] };
  function gbSideDrivers(p) {
    var g = p.game, sideIsHome = p.side === g.home, sgn = sideIsHome ? 1 : -1;
    var S = team(p.side).abbr, O = team(sideIsHome ? g.away : g.home).abbr;
    var inputs = g.inputs || {}, rows = [];
    if (g.drivers && g.drivers.length) {
      // Measured: each group set to even and the game re-scored (export).
      g.drivers.forEach(function (d) {
        var vals = {}; Object.keys(inputs).forEach(function (k) { vals[k] = inputs[k]; });
        Object.keys(d.vals || {}).forEach(function (k) { vals[k] = d.vals[k]; });
        var pts = sgn * d.pts, txt = gbInputText(d.key, null, S, O, sgn, vals);
        if (Math.abs(pts) >= 0.25 && txt) rows.push({ pts: pts, txt: txt, label: d.label });
      });
      return { measured: true, rows: rows };
    }
    // Fallback (no measured pushes yet): the raw inputs, in order of how much
    // the model leans on them, with a direction where one is clear.
    var dir = {
      form: function () { return inputs.ewm_adj_margin_diff == null || Math.abs(inputs.ewm_adj_margin_diff) < 1 ? 0 : sgn * inputs.ewm_adj_margin_diff; },
      ratings: function () { return inputs.sp_rating_diff == null || Math.abs(inputs.sp_rating_diff) < 1 ? 0 : sgn * inputs.sp_rating_diff; },
      scoring: function () { return inputs.roll_margin_diff == null || Math.abs(inputs.roll_margin_diff) < 1 ? 0 : sgn * inputs.roll_margin_diff; },
      eff: function () { return inputs.ewm_net_ppa_diff == null || Math.abs(inputs.ewm_net_ppa_diff) < 0.03 ? 0 : sgn * inputs.ewm_net_ppa_diff; },
      home: function () { return inputs.home_field ? sgn : 0; },
      rest: function () { return inputs.rest_diff == null || Math.abs(inputs.rest_diff) < 2 ? 0 : sgn * inputs.rest_diff; },
      qb: function () { return (sideIsHome ? inputs.away_qb_out : inputs.home_qb_out) ? 1 : (sideIsHome ? inputs.home_qb_out : inputs.away_qb_out) ? -1 : 0; }
    };
    ['form', 'ratings', 'scoring', 'eff', 'home', 'rest', 'qb'].forEach(function (k) {
      var v = dir[k](), txt = v ? gbInputText(k, null, S, O, sgn, inputs) : null;
      if (txt) rows.push({ pts: v > 0 ? 1 : -1, txt: txt, raw: true });
    });
    return { measured: false, rows: rows };
  }
  // ---- The football read (11/2026) ----
  // Turns the model's inputs and the unit-vs-unit ranks into how the game
  // should actually play out: where the edge shows up on the field, why that
  // matters for this number, and what has to go wrong for it to lose.
  // Plain-language only; it reads the same data, it does not change plays.
  function gbUnits(sSchool, oSchool) {
    var TM = D.teamMatchup || {}, sP = TM[sSchool], oP = TM[oSchool], o = {}, d = {};
    if (!sP || !oP) return null;
    tmCompare(sP, oP).forEach(function (x) { o[x.c.key] = x; });   // S offense vs O defense, + = S
    tmCompare(oP, sP).forEach(function (x) { d[x.c.key] = x; });   // O offense vs S defense, + = O
    return { o: o, d: d };
  }
  function gbAvg(map, keys) {
    var xs = keys.map(function (k) { return map[k]; }).filter(Boolean);
    if (!xs.length) return null;
    var best = xs.slice().sort(function (a, b) { return b.diff - a.diff; })[0];
    var worst = xs.slice().sort(function (a, b) { return a.diff - b.diff; })[0];
    return { v: xs.reduce(function (s, x) { return s + x.diff; }, 0) / xs.length, best: best, worst: worst };
  }
  function gbLab(t) { return esc(String(t).replace(/ \\(.*\\)$/, '')); }
  function gbCite(x) { return x ? gbLab(x.c.off) + ' #' + x.o.rank + ' vs ' + (x.c.off === x.c.def.replace(' allowed', '') || /allowed/.test(x.c.def) ? '#' + x.d.rank + ' allowed' : gbLab(x.c.def) + ' #' + x.d.rank) : ''; }
  function gbCiteD(x) { return x ? gbLab(x.c.def) + ' #' + x.d.rank + ' vs ' + gbLab(x.c.off) + ' #' + x.o.rank : ''; }
  var GB_KEYS = {
    run: ['rush_epa', 'line', 'stuff', 'pff_rblk', 'rush_ypg'],
    pass: ['pass_epa', 'pff_pass', 'pff_recv', 'pass_ypg'],
    protect: ['pff_pblk', 'pff_pres', 'sacks', 'havoc'],
    downs: ['pd_sr', 'third'],
    finish: ['ppo'],
    expl: ['expl'],
    to: ['to']
  };
  // Story lines. "for" = the picked side has the edge, "risk" = the
  // opponent has it. fav = the picked side is laying points.
  function gbStories(u, S, O, fav) {
    var out = [], add = function (score, kind, txt) { out.push({ s: Math.abs(score), kind: kind, txt: txt }); };
    var a;
    // S offense vs O defense
    if ((a = gbAvg(u.o, GB_KEYS.run)) && a.v >= 0.2) add(a.v, 'for', esc(S) + ' should run the ball (' + gbCite(a.best) + '). A team that runs well stays ahead of the chains &mdash; 2nd-and-4 instead of 2nd-and-9 &mdash; and ' +
      (fav ? 'can milk the clock once it has the lead, which is how favorites protect a number.' : 'shortens the game: fewer possessions means fewer chances for the favorite to pull away.'));
    if ((a = gbAvg(u.o, GB_KEYS.pass)) && a.v >= 0.2) add(a.v, 'for', esc(S) + '&rsquo;s passing game should find room (' + gbCite(a.best) + '). That&rsquo;s where explosive plays and quick scores come from.');
    if ((a = gbAvg(u.o, GB_KEYS.protect)) && a.v >= 0.25) add(a.v * 0.9, 'for', esc(S) + '&rsquo;s line should keep its QB clean (' + gbCite(a.best) + '), so the offense can get through its reads instead of throwing it away or taking drive-killing sacks.');
    if ((a = gbAvg(u.o, GB_KEYS.finish)) && a.v >= 0.25) add(a.v * 0.9, 'for', esc(S) + ' finishes drives (' + gbCite(a.best) + '). Touchdowns instead of field goals inside the 40 is a 4-point swing every trip, and that&rsquo;s how covers get decided.');
    if ((a = gbAvg(u.o, GB_KEYS.downs)) && a.v >= 0.25) add(a.v * 0.8, 'for', esc(S) + ' converts on passing downs and third down (' + gbCite(a.best) + '), which keeps drives alive and keeps the other offense on the sideline.');
    // O offense vs S defense (low diff = S defense wins)
    if ((a = gbAvg(u.d, GB_KEYS.run)) && a.v <= -0.2) add(a.v, 'for', esc(S) + '&rsquo;s front should take away ' + esc(O) + '&rsquo;s run game (' + gbCiteD(a.worst) + '). Once ' + esc(O) +
      ' can&rsquo;t run, it&rsquo;s throwing on 2nd and 3rd-and-long, the downs where offenses stall and turn it over.');
    if ((a = gbAvg(u.d, GB_KEYS.protect)) && a.v <= -0.25) add(a.v, 'for', esc(S) + '&rsquo;s pass rush against ' + esc(O) + '&rsquo;s protection (' + gbCiteD(a.worst) + ') is the drive-killer. Pressure means sacks, hurried throws and turnovers, and one sack usually ends a college drive.');
    if ((a = gbAvg(u.d, GB_KEYS.pass)) && a.v <= -0.2) add(a.v, 'for', esc(S) + '&rsquo;s secondary should hold up (' + gbCiteD(a.worst) + '), so ' + esc(O) + ' has to string together long drives instead of hitting shots.');
    if ((a = gbAvg(u.d, GB_KEYS.downs)) && a.v <= -0.25) add(a.v * 0.8, 'for', esc(S) + '&rsquo;s defense gets off the field on passing downs and third down (' + gbCiteD(a.worst) + '), which ends drives before they reach scoring range.');
    if ((a = gbAvg(u.o, GB_KEYS.to)) && a.v >= 0.3 && u.d.to) add(a.v * 0.5, 'for', 'Turnovers lean ' + esc(S) + '&rsquo;s way on paper, but they&rsquo;re the noisiest stat in football, so treat it as a bonus, not the reason.');
    // Risks: O has the edge
    if ((a = gbAvg(u.d, GB_KEYS.run)) && a.v >= 0.2) add(a.v, 'risk', esc(O) + ' can run on ' + esc(S) + ' (' + gbCite(a.best) + '). If ' + esc(O) + ' controls the ball on the ground, ' +
      (fav ? 'it shortens the game and keeps ' + esc(S) + '&rsquo;s offense on the sideline &mdash; the classic way an underdog stays inside the number.' : 'it can protect a lead and run the clock out on ' + esc(S) + '.'));
    if ((a = gbAvg(u.o, GB_KEYS.protect)) && a.v <= -0.25) add(a.v, 'risk', esc(O) + '&rsquo;s pass rush against ' + esc(S) + '&rsquo;s protection (' + gbCiteD(a.worst) + '). If ' + esc(S) + '&rsquo;s QB is running for his life, drives stall and turnovers follow.');
    if ((a = gbAvg(u.d, GB_KEYS.pass)) && a.v >= 0.2) add(a.v, 'risk', esc(O) + ' can throw on ' + esc(S) + '&rsquo;s secondary (' + gbCite(a.best) + '). Big plays through the air are ' +
      (fav ? 'how an underdog hangs around and steals a backdoor cover.' : 'how a favorite pulls away fast.'));
    if ((a = gbAvg(u.o, GB_KEYS.run)) && a.v <= -0.2) add(a.v * 0.8, 'risk', esc(O) + '&rsquo;s front can stuff ' + esc(S) + '&rsquo;s run game (' + gbCiteD(a.worst) + '), so ' + esc(S) + ' may have to win by throwing it.');
    if ((a = gbAvg(u.d, GB_KEYS.expl)) && a.v >= 0.3) add(a.v * 0.7, 'risk', esc(O) + ' hits big plays (' + gbCite(a.best) + ') &mdash; one or two explosives can flip a game ' + esc(S) + ' is controlling.');
    return out.sort(function (x, y) { return y.s - x.s; });
  }
  // What kind of bet this is, in football terms: the number itself.
  function gbNumberTalk(S, O, spr, model, total) {
    var t = '', a = Math.abs(spr), need = a % 1 === 0 ? a + 1 : Math.ceil(a), push = a % 1 === 0 && spr !== 0 ? ' (a ' + a + '-point win is a push)' : '';
    if (spr < 0) {
      if (a <= 3) t = 'At ' + M.signed(spr) + ' this is close to a pick-the-winner bet: ' + esc(S) + ' just has to win' + (a > 1 ? ' by ' + Math.ceil(a) + '+' : '') + '.';
      else if (a <= 7.5) t = esc(S) + ' has to win by ' + need + '+' + (need > 3 ? ' &mdash; more than a field goal' + (need > 7 ? ' and more than a touchdown' : '') : '') + push + '.';
      else if (a <= 17) t = 'Laying ' + a + ' means ' + esc(S) + ' has to win by ' + need + '+' + push + ', so it needs to keep scoring in the second half instead of sitting on a lead.';
      else t = 'Laying ' + a + ' is a lot: ' + esc(S) + ' has to keep its foot down, and a late score against the backups is the classic backdoor risk.';
      if (a === 2.5 || a === 6.5) t += ' The line sits just under a key number (' + (a + 0.5) + '), so a ' + (a + 0.5) + '-point win still cashes.';
      if (a === 3.5 || a === 7.5) t += ' Heads up: it&rsquo;s a hook past a key number, so a ' + (a - 0.5) + '-point win &mdash; one of the most common margins &mdash; loses.';
    } else if (spr > 0) {
      t = esc(S) + ' can lose by up to ' + Math.floor(spr) + ' and still cash' + (model < 0 ? ', and the model has ' + esc(S) + ' winning outright, so the points are a cushion, not the bet.' : '.');
      if (spr === 3.5 || spr === 7.5) t += ' Getting the hook past a key number (' + (spr - 0.5) + ') is worth a lot.';
    } else t = 'A pick&rsquo;em: ' + esc(S) + ' just has to win.';
    if (total != null && total <= 44) t += ' The total is low (' + total + '), so possessions and points are scarce &mdash; every point of the spread matters more.';
    else if (total != null && total >= 62) t += ' The total is ' + total + ', a shootout look: lots of possessions, so expect swings.';
    return t;
  }
  // ---- Side read v2 (11/12/2026) ----
  // Every Bet Card side gets: the disagreement with the market, what the
  // market is pricing vs what the model is pricing (from the measured
  // pushes), a trust check (do the model's own inputs agree?), how the
  // pick cashes and how it loses in football terms (unit ranks), whether
  // the injuries are new or already in the numbers, and the number itself.
  function gbRk(school, side, key) {
    var t = (D.teamMatchup || {})[school], c = ((D.teamMatchupMeta || {}).cats || []).filter(function (x) { return x.key === key; })[0];
    var v = t && t[side] && t[side][key];
    return v && v.rank ? { r: v.rank, lab: c ? c[side] : key } : null;
  }
  var GB_N = 136;
  function gbGood(x) { return x && x.r <= GB_N / 3; }
  function gbBad(x) { return x && x.r >= GB_N * 2 / 3; }
  function gbR(x) { return x ? '#' + x.r : ''; }
  // Driver groups in football words, from the picked side's view.
  function gbDriverWords(key, S, O, sgn, v) {
    var x = function (k) { return v[k] == null ? null : sgn * v[k]; };
    var lead = function (n) { return n >= 0 ? esc(S) : esc(O); };
    if (key === 'form') { var f = x('ewm_adj_margin_diff'); return f == null ? null : 'recent form: ' + lead(f) + ' has been ' + Math.abs(f).toFixed(0) + ' pts a game better lately, adjusted for opponents'; }
    if (key === 'scoring') { var m = x('roll_margin_diff'), pf = x('roll_ppg_for_diff'), pa = v.roll_ppg_against_diff == null ? null : -sgn * v.roll_ppg_against_diff;
      return m == null ? null : 'season scoring: ' + lead(m) + '&rsquo;s margin is ' + Math.abs(m).toFixed(0) + (Math.round(Math.abs(m)) === 1 ? ' pt' : ' pts') + ' a game better' +
        (pf != null && pa != null ? ' (' + esc(S) + ' scores ' + Math.abs(pf).toFixed(0) + (pf >= 0 ? ' more' : ' fewer') + ', allows ' + Math.abs(pa).toFixed(0) + (pa >= 0 ? ' fewer' : ' more') + ')' : ''); }
    if (key === 'eff') { var n = x('ewm_net_ppa_diff'); return n == null ? null : Math.abs(n) < 0.02 ? 'per-play efficiency: about even (the push comes from how offense and defense split)' : 'per-play efficiency: ' + lead(n) + ' is ' + Math.abs(n).toFixed(2) + ' EPA a play better, net of offense and defense'; }
    if (key === 'ratings') { var sp = x('sp_rating_diff'), co = x('core_overall_diff');
      if (sp == null) return null;
      var same = co == null || (sp >= 0) === (co >= 0);
      return 'roster ratings: SP+ ' + (sp >= 0 ? 'rates ' + esc(S) + ' ' + sp.toFixed(1) + ' pts better' : 'rates ' + esc(O) + ' ' + (-sp).toFixed(1) + ' pts better') +
        (co != null ? (same ? ' and CORE agrees (' + gbSigned(co, 1) + ')' : ', CORE disagrees (' + gbSigned(co, 1) + ')') : ''); }
    if (key === 'home') return v.home_field ? 'home field (' + (sgn > 0 ? esc(S) : esc(O)) + ' at home)' : null;
    if (key === 'pace') { var pc = x('pace_diff'); return pc == null ? null : 'drive length: ' + esc(S) + ' ran ' + Math.abs(pc).toFixed(1) + (pc >= 0 ? ' more' : ' fewer') + ' plays per drive than ' + esc(O) + ' last season (longer drives = sustained offense, fewer possessions)'; }
    if (key === 'weather') return v.temperature != null ? (v.temperature < 55 ? 'the cold forecast (' + Math.round(v.temperature) + '&deg;F' : 'the forecast (' + Math.round(v.temperature) + '&deg;F') +
      ', wind ' + Math.round(v.wind_speed || 0) + ' mph) &mdash; a weather effect the model learned from past games' : 'the forecast';
    if (key === 'rest') { var rd = x('rest_diff'); return rd ? 'rest (' + esc(S) + ' ' + (rd > 0 ? '+' : '') + rd + ' days)' : null; }
    if (key === 'qb') return 'the QB-out input';
    if (key === 'returning') return 'returning production';
    return null;
  }
  function gbCase(p, S, O, sgn) {
    var g = p.game, inp = g.inputs || {}, pro = [], con = [];
    (g.drivers || []).forEach(function (d) {
      var vals = {}; Object.keys(inp).forEach(function (k) { vals[k] = inp[k]; }); Object.keys(d.vals || {}).forEach(function (k) { vals[k] = d.vals[k]; });
      var pts = sgn * d.pts, w = gbDriverWords(d.key, S, O, sgn, vals);
      if (!w || Math.abs(pts) < 0.4) return;
      // Skip a push whose input points the other way (tree-model quirk);
      // it would read as a contradiction.
      var dir = { form: vals.ewm_adj_margin_diff, scoring: vals.roll_margin_diff, eff: vals.ewm_net_ppa_diff, ratings: vals.sp_rating_diff, rest: vals.rest_diff }[d.key];
      if (dir != null && dir !== 0 && (sgn * dir > 0) !== (pts > 0)) return;
      (pts > 0 ? pro : con).push({ pts: pts, w: w, k: d.key });
    });
    var on = 0, rat = 0;
    pro.concat(con).forEach(function (x) { if (/^(recent form|season scoring|per-play)/.test(x.w)) on += x.pts; if (/^roster ratings/.test(x.w)) rat += x.pts; });
    return { on: on, rat: rat, pro: pro, con: con, sp: pro.reduce(function (a, x) { return a + x.pts; }, 0), sc: -con.reduce(function (a, x) { return a + x.pts; }, 0) };
  }
  function gbPaths(S, O, sSch, oSch, dog, spr, total) {
    var cov = [], lose = [], add = function (arr, s, t) { arr.push({ s: s, t: t }); };
    var r = function (sch, sd, k) { return gbRk(sch, sd, k); };
    var F = dog ? O : S, Fs = dog ? oSch : sSch, Dg = dog ? S : O, Ds = dog ? sSch : oSch;
    var fExpl = r(Fs, 'off', 'expl'), dExplD = r(Ds, 'def', 'expl'), fPpo = r(Fs, 'off', 'ppo'), dPpoD = r(Ds, 'def', 'ppo');
    var fTo = r(Fs, 'off', 'to'), fTk = r(Fs, 'def', 'to'), dTo = r(Ds, 'off', 'to'), dTk = r(Ds, 'def', 'to');
    var dPpg = r(Ds, 'off', 'ppg'), fPpgD = r(Fs, 'def', 'ppg'), dExpl = r(Ds, 'off', 'expl'), fExplD = r(Fs, 'def', 'expl');
    var fPass = r(Fs, 'off', 'pass_epa'), dCov = r(Ds, 'def', 'pff_pass'), fRush = r(Fs, 'off', 'rush_epa'), dRushD = r(Ds, 'def', 'rush_epa');
    var dThird = r(Ds, 'off', 'third'), fThirdD = r(Fs, 'def', 'third');
    // --- paths for the underdog to stay inside / for the favorite to pull away ---
    var dogPaths = [], favPaths = [];
    if (gbGood(dExplD)) dogPaths.push({ s: GB_N - dExplD.r, t: esc(Dg) + ' doesn&rsquo;t give up big plays (explosives allowed ' + gbR(dExplD) + (fExpl ? ', vs ' + esc(F) + '&rsquo;s ' + gbR(fExpl) + ' offense' : '') +
      '). Big numbers get covered with explosives; without them ' + esc(F) + ' has to string together 10-play drives, which burns clock and gives ' + esc(Dg) + ' more chances to get a stop.' });
    if (fPpo && !gbGood(fPpo)) dogPaths.push({ s: fPpo.r, t: esc(F) + ' stalls in scoring range (' + gbR(fPpo) + ' in points per trip inside the 40). Field goals instead of touchdowns is how a favorite wins by 10 instead of 20.' });
    if (gbGood(dCov) && fPass && gbGood(fPass)) dogPaths.push({ s: GB_N - dCov.r, t: esc(F) + '&rsquo;s best weapon is the pass (' + gbR(fPass) + ' in passing EPA), and ' + esc(Dg) + '&rsquo;s coverage grades well (PFF ' + gbR(dCov) + '). That&rsquo;s the matchup that decides how many points ' + esc(F) + ' can pile up.' });
    else if (gbGood(dCov)) dogPaths.push({ s: (GB_N - dCov.r) * 0.6, t: esc(Dg) + '&rsquo;s coverage grades well (PFF ' + gbR(dCov) + '), so ' + esc(F) + ' has to earn it on the ground.' });
    if (gbGood(dRushD) && fRush && !gbGood(fRush)) dogPaths.push({ s: GB_N - dRushD.r, t: esc(Dg) + ' can hold up against the run (' + gbR(dRushD) + ' in rush EPA allowed vs ' + esc(F) + '&rsquo;s ' + gbR(fRush) + ' run game), which takes away the clock-killing drives favorites use to put games away.' });
    if (dExpl && !gbBad(dExpl)) dogPaths.push({ s: (GB_N - dExpl.r) * 0.7, t: esc(Dg) + ' can hit a big play of its own (' + gbR(dExpl) + ' in explosiveness' + (fExplD ? ' vs ' + esc(F) + '&rsquo;s ' + gbR(fExplD) + ' defense' : '') + '). One or two of those keeps it within the number, or gets a backdoor score late.' });
    if (dThird && gbGood(dThird)) dogPaths.push({ s: GB_N - dThird.r, t: esc(Dg) + ' converts third downs (' + gbR(dThird) + '), so it can keep the ball and shorten the game.' });
    if (dTo && gbGood(dTo)) dogPaths.push({ s: GB_N - dTo.r, t: esc(Dg) + ' takes care of the ball (' + gbR(dTo) + ' in turnovers), so it&rsquo;s not handing ' + esc(F) + ' short fields.' });
    if (Math.abs(spr) >= 13.5) dogPaths.push({ s: 30, t: 'At this size, if ' + esc(F) + ' gets comfortable its starters sit in the 4th quarter &mdash; the window when big underdogs get garbage-time scores.' });
    if (gbGood(fPpo) && gbBad(dPpoD)) favPaths.push({ s: dPpoD.r, t: esc(F) + ' turns trips into touchdowns (' + gbR(fPpo) + ' in points per trip) against a defense that gives them up (' + gbR(dPpoD) + '). That&rsquo;s how a lead grows fast.' });
    if (gbGood(fTk) && gbBad(dTo)) favPaths.push({ s: dTo.r + (GB_N - fTk.r), t: esc(F) + '&rsquo;s defense takes the ball away (' + gbR(fTk) + ' in takeaways) and ' + esc(Dg) + ' gives it up (' + gbR(dTo) + ' in turnovers). Short fields are the quickest way for a favorite to run up the margin.' });
    if (gbBad(dPpg)) favPaths.push({ s: dPpg.r, t: esc(Dg) + '&rsquo;s offense struggles to score (' + gbR(dPpg) + ' in points a game' + (fPpgD ? ', against ' + esc(F) + '&rsquo;s ' + gbR(fPpgD) + ' scoring defense' : '') + '). If it can&rsquo;t get to 20, the number is a lot easier to cover.' });
    if (gbGood(fExpl) && gbBad(dExplD)) favPaths.push({ s: dExplD.r, t: esc(F) + ' hits big plays (' + gbR(fExpl) + ') against a defense that allows them (' + gbR(dExplD) + ').' });
    if (fPass && gbGood(fPass) && gbBad(dCov)) favPaths.push({ s: dCov.r, t: esc(F) + '&rsquo;s passing game (' + gbR(fPass) + ' in EPA) should be open against ' + esc(Dg) + '&rsquo;s coverage (PFF ' + gbR(dCov) + ').' });
    if (fRush && gbGood(fRush) && gbBad(dRushD)) favPaths.push({ s: dRushD.r, t: esc(F) + ' should run through ' + esc(Dg) + ' (' + gbR(fRush) + ' run game vs ' + gbR(dRushD) + ' run defense), the way favorites bleed out games.' });
    if (fThirdD && gbGood(fThirdD) && dThird && !gbGood(dThird)) favPaths.push({ s: GB_N - fThirdD.r, t: esc(F) + '&rsquo;s defense gets off the field on third down (' + gbR(fThirdD) + '), so ' + esc(Dg) + '&rsquo;s drives end early.' });
    if (total != null && total <= 44) dogPaths.push({ s: 40, t: 'A low total (' + total + ') means fewer possessions, and fewer chances for the favorite to build a big lead.' });
    if (total != null && total >= 62) favPaths.push({ s: 35, t: 'A ' + total + ' total means a lot of possessions, room for the better team to pull away.' });
    var by = function (a, b) { return b.s - a.s; };
    dogPaths.sort(by); favPaths.sort(by);
    if (dog) { cov = dogPaths; lose = favPaths; } else { cov = favPaths; lose = dogPaths; }
    return { cov: cov, lose: lose };
  }
  function gbWeeksSince(since, iso) {
    var m = /([A-Z][a-z]{2}) (\\d{1,2})/.exec(String(since || ''));
    if (!m || GB_MON[m[1]] == null) return null;
    var k = new Date(iso), y = k.getUTCFullYear(), d = Date.UTC(y, GB_MON[m[1]], +m[2]);
    if (d > k.getTime()) d = Date.UTC(y - 1, GB_MON[m[1]], +m[2]);
    return Math.floor((k.getTime() - d) / (7 * 86400000));
  }
  // The picked side's biggest unit-vs-unit edges, in football terms, for
  // games where no single matchup clears the story thresholds.
  var GB_WHY = { ppg: 'points', epa: 'yards and points per play', sr: 'staying on schedule', expl: 'big plays', ppo: 'finishing drives in scoring range',
    rush_ypg: 'the run game', rush_epa: 'the run game', line: 'winning up front on runs', stuff: 'avoiding runs stuffed at the line', pass_ypg: 'the passing game',
    pass_epa: 'the passing game', pd_sr: 'converting passing downs', havoc: 'avoiding (or creating) havoc plays', sacks: 'sacks', third: 'third downs', to: 'turnovers',
    pff_all: 'overall PFF grade', pff_pass: 'passing (PFF)', pff_recv: 'receivers (PFF)', pff_pblk: 'pass protection (PFF)', pff_pres: 'pressure', pff_rblk: 'run blocking (PFF)', pff_rush: 'running (PFF)' };
  function gbBestEdges(sSch, oSch, S, O) {
    var u = gbUnits(sSch, oSch); if (!u) return [];
    var xs = [];
    Object.keys(u.o).forEach(function (k) { var y = u.o[k]; if (y.diff > 0.15) xs.push({ s: y.diff, t: esc(S) + '&rsquo;s offense has the edge in ' + (GB_WHY[k] || gbLab(y.c.off)) + ' (' + gbCite(y) + ')' }); });
    Object.keys(u.d).forEach(function (k) { var y = u.d[k]; if (y.diff < -0.15) xs.push({ s: -y.diff, t: esc(S) + '&rsquo;s defense has the edge on ' + esc(O) + '&rsquo;s ' + (GB_WHY[k] || gbLab(y.c.off)) + ' (' + gbCiteD(y) + ')' }); });
    xs.sort(function (a, b) { return b.s - a.s; });
    if (!xs.length) return [];
    return [{ s: 1, t: 'No single matchup is a big mismatch, so this rides on the number. ' + esc(S) + '&rsquo;s best spots: ' + xs.slice(0, 3).map(function (x) { return x.t; }).join('; ') + '.' }];
  }
  function gbFootballSide(p, injOpp, injOwn) {
    var g = p.game, sideIsHome = p.side === g.home, sgn = sideIsHome ? 1 : -1;
    var S = team(p.side).abbr, O = team(sideIsHome ? g.away : g.home).abbr;
    var sSch = sideIsHome ? g.homeSchool : g.awaySchool, oSch = sideIsHome ? g.awaySchool : g.homeSchool;
    var spr = p.market === 'Spread' ? (sideIsHome ? g.marketSpread : -g.marketSpread) : (sideIsHome ? g.marketSpread : -g.marketSpread);
    var model = sideIsHome ? g.modelSpread : -g.modelSpread;
    var dog = p.market === 'Spread' ? spr > 0 : p.sideMoneyline > 0;
    var sec = [];
    // 1) The disagreement.
    var gap = Math.abs(spr - model);
    var mTxt = function (v) { return v < 0 ? esc(S) + ' by ' + Math.abs(v).toFixed(1) : v > 0 ? esc(O) + ' by ' + v.toFixed(1) : 'a pick&rsquo;em'; };
    sec.push(['The disagreement', 'The market has ' + mTxt(spr) + '. The model has ' + mTxt(model) + '. ' +
      (p.market === 'Spread' ? 'That ' + gap.toFixed(1) + '-point gap is the whole bet' + (dog && model > 0 ? ': the model doesn&rsquo;t think ' + esc(S) + ' wins, it thinks the line is too big.' : dog && model <= 0 ? ': the model has ' + esc(S) + ' winning outright, so the points are a cushion.' : ': the model thinks ' + esc(S) + ' wins by more than the market does.')
        : 'Moneyline: the model gives ' + esc(S) + ' ' + (p.sideProb * 100).toFixed(0) + '% to win.')]);
    // 2) What each side is pricing + trust check.
    var c = gbCase(p, S, O, sgn);
    if (c.pro.length || c.con.length) {
      var fmt = function (x) { return x.w + ' <span class="gb-rk">' + gbSigned(x.pts, 1) + '</span>'; };
      sec.push(['What the model is buying', c.pro.length ? c.pro.slice(0, 4).map(fmt).join('; ') + '.' : 'Nothing big pushes it toward ' + esc(S) + '.']);
      sec.push(['What it&rsquo;s fighting', c.con.length ? c.con.slice(0, 4).map(fmt).join('; ') + '.' : 'Nothing big pushes it toward ' + esc(O) + '.']);
      var verdict, top = c.pro[0];
      var situ = c.pro.length && c.pro.every(function (x) { return /^(home|weather|rest|pace)$/.test(x.k); }) && c.on <= 0;
      if (top && top.pts >= 5 && top.pts >= 0.6 * c.sp) {
        verdict = '<b class="gb-inline">One-input play.</b> ' + gbSigned(top.pts, 1) + ' of the model&rsquo;s ' + gbSigned(c.sp, 1) + ' pts toward ' + esc(S) + ' comes from one thing: ' + top.w.split(':')[0] + '. ' +
          (top.k === 'pace' ? 'That&rsquo;s last season&rsquo;s plays per drive, and the model can overreact to an extreme offense (option teams like the service academies run very long drives).'
           : top.k === 'ratings' ? 'The roster ratings see a big gap' + (c.on < 0 ? ' that this season&rsquo;s results don&rsquo;t back up' : '') + '. If the ratings are stale, so is the edge.'
           : top.k === 'weather' ? 'That&rsquo;s a weather effect the model learned, not a football mismatch.'
           : 'If that one read is wrong, there&rsquo;s nothing else holding the play up.');
      } else if (situ) verdict = '<b class="gb-inline">Situational only.</b> The lean comes from ' + c.pro.map(function (x) { return x.w.split(' (')[0].split(':')[0]; }).join(' and ') +
        ', not from ' + esc(S) + ' being the better team &mdash; the on-field numbers favor ' + esc(O) + '.';
      else if (c.rat >= 1 && c.on <= -1) verdict = '<b class="gb-inline">Ratings vs results.</b> The model is siding with the roster ratings (SP+ / CORE), which like ' + esc(S) + ', over this season&rsquo;s results, which favor ' + esc(O) +
        '. This is a bet that the results so far understate ' + esc(S) + ' &mdash; it loses if ' + esc(O) + '&rsquo;s season is the real version of these teams.';
      else if (c.sp >= 2 * c.sc) verdict = '<b class="gb-inline">Solid.</b> The model&rsquo;s inputs mostly agree with each other on ' + esc(S) + '.';
      else if (c.sp >= c.sc) verdict = '<b class="gb-inline">Mixed.</b> More of the model&rsquo;s inputs point to ' + esc(S) + ' than away, but there&rsquo;s real pushback.';
      else verdict = '<b class="gb-inline">Thin.</b> Most of what the model measures points to ' + esc(O) + ' (' + c.sc.toFixed(1) + ' pts of push vs ' + c.sp.toFixed(1) + ' for ' + esc(S) + '). ' +
        'The play rests on ' + (c.pro[0] ? c.pro[0].w.split(':')[0] : 'a few inputs') + (c.pro[1] ? ' and ' + c.pro[1].w.split(':')[0] : '') +
        ', while what&rsquo;s happened on the field says ' + esc(O) + '.';
      // 11/12/2026 trust lab (2024-26, walk-forward): these labels did NOT
      // sort Bet Card winners from losers (weak labels +2.9% vs strong -9.0%
      // at the close; +7.4% vs +4.5% at the open; flips by season). So the
      // label explains what the bet rests on; it is not a grade.
      sec.push(['What the bet rests on', verdict + ' <span class="gb-rk">This describes the case, it doesn&rsquo;t grade it: in the 2024&ndash;26 backtest these labels didn&rsquo;t separate winners from losers. pts = how far the model&rsquo;s line moves if that input were even.</span>']);
    }
    // 3) How it cashes / how it loses.
    var paths = gbPaths(S, O, sSch, oSch, dog, spr, g.marketTotal);
    if (Math.abs(spr) <= 4) {
      // A small number is about who wins, not margin: unit-vs-unit edges.
      var u = gbUnits(sSch, oSch), st = u ? gbStories(u, S, O, !dog) : [];
      paths = { cov: st.filter(function (x) { return x.kind === 'for'; }).map(function (x) { return { s: x.s, t: x.txt }; }),
                lose: st.filter(function (x) { return x.kind === 'risk'; }).map(function (x) { return { s: x.s, t: x.txt }; }) };
    }
    if (!paths.cov.length) paths.cov = gbBestEdges(sSch, oSch, S, O);
    if (paths.cov.length) sec.push(['How ' + esc(S) + ' cashes', paths.cov.slice(0, 3).map(function (x) { return x.t; }).join(' ')]);
    else sec.push(['How ' + esc(S) + ' cashes', 'No unit of ' + esc(S) + '&rsquo;s beats ' + esc(O) + '&rsquo;s on paper, so this is purely a bet that the number is wrong.']);
    if (paths.lose.length) sec.push(['How it loses', paths.lose.slice(0, 2).map(function (x) { return x.t; }).join(' ')]);
    // 4) Injuries: new, or already in the numbers?
    var injTxt = [];
    var allInj = (g.injAway || []).concat(g.injHome || []);
    var qbLine = function (k, who) {
      if (!k.since) { var m0 = allInj.filter(function (x) { return x.player === k.player; })[0]; if (m0) k = Object.assign({}, k, { since: m0.since }); }
      var wk = gbWeeksSince(k.since, g.kickoffIso);
      return esc(who) + '&rsquo;s ' + (k.role === 'starting QB' ? 'starting QB ' : 'QB ') + esc(k.player) + ' is ' + esc(String(k.status).toLowerCase()) +
        (/^Questionable/.test(k.status || '') ? (wk != null && wk >= 2 ? ' (listed since ' + esc(String(k.since).replace(/^[A-Za-z]+, /, '')) + ')' : '') + '. If he sits, that&rsquo;s not in the model&rsquo;s line &mdash; check his status before kickoff.'
        : wk != null && wk >= 2 ? ' &mdash; on the report since ' + esc(String(k.since).replace(/^[A-Za-z]+, /, '')) + ', so about ' + wk + ' weeks of ' + esc(who) + '&rsquo;s numbers already come without him. Not new information; both the market and the model mostly have it.'
          : wk != null ? ' &mdash; a new injury. The model&rsquo;s numbers still include games he played, so this one isn&rsquo;t fully in its line.' : '.');
    };
    (injOpp || []).concat([]).forEach(function (k) { if (/QB/.test(k.pos || '') || k.role === 'starting QB') { if (/^(Out|IR|Doubtful)/.test(k.status || '')) injTxt.push(qbLine(k, O)); } });
    (injOwn || []).forEach(function (k) { if ((/QB/.test(k.pos || '') || k.role === 'starting QB') && /^(Out|IR|Doubtful|Questionable)/.test(k.status || '')) injTxt.push(qbLine(k, S)); });
    var keyOut = function (lst, who) { return (lst || []).filter(function (k) { return k.role && !/QB/.test(k.role) && !/QB/.test(k.pos || '') && /^(Out|IR|Doubtful)/.test(k.status || ''); })
      .map(function (k) { return esc(k.player) + ' (' + esc(k.role) + ')'; }); };
    var oo = keyOut(injOpp, O), so = keyOut(injOwn, S);
    if (oo.length) injTxt.push(esc(O) + ' is also without ' + oo.join(', ') + '.');
    if (so.length) injTxt.push(esc(S) + ' is without ' + so.join(', ') + ', which works against this play.');
    if (injTxt.length) sec.push(['Injuries', injTxt.join(' ')]);
    // 5) The number.
    if (p.market === 'Spread') sec.push(['The number', gbNumberTalk(S, O, spr, model, g.marketTotal)]);
    var inp = g.inputs || {};
    if (inp.wind_speed != null && inp.wind_speed >= 15) sec[sec.length - 1][1] += ' Wind around ' + Math.round(inp.wind_speed) + ' mph hurts the deep ball and the kickers, which favors the team that can run it and keeps scoring down.';
    return '<div class="gb-read gb-read2">' + sec.map(function (x) { return '<p><b class="gb-h">' + x[0] + '</b>' + x[1] + '</p>'; }).join('') + '</div>';
  }
  // Props: the position-specific football logic behind each official rule.
  // Role change from injuries (11/2026). The props model has no injury
  // input: it only sees a teammate's absence through the player's last-2-
  // game inputs, while its season inputs still include games the teammate
  // played. This reads the report against his game log to say how much of
  // his line came after the teammate went out, and whether the projection
  // sits closer to his old role or his new one.
  var GB_MON = { Jan: 0, Feb: 1, Mar: 2, Apr: 3, May: 4, Jun: 5, Jul: 6, Aug: 7, Sep: 8, Oct: 9, Nov: 10, Dec: 11 };
  function gbSinceWeek(since, season) {
    var m = /([A-Z][a-z]{2}) (\\d{1,2})/.exec(String(since || ''));
    if (!m || GB_MON[m[1]] == null) return null;
    var dt = Date.UTC(season, GB_MON[m[1]], +m[2]);
    if (GB_MON[m[1]] < 6) dt = Date.UTC(season + 1, GB_MON[m[1]], +m[2]);
    // Week 1 Saturday ~ the last Saturday of August; weeks run Sun-Sat.
    var w1 = Date.UTC(season, 7, 29);
    return Math.floor((dt - w1) / (7 * 86400000)) + 2;   // first week played after the report date
  }
  function gbRoleChange(r, g) {
    var mates = (r.team === g.awaySchool ? g.injAway : g.injHome) || [], over = r.model_lean === 'over';
    var myGrp = r.pos === 'TE' ? 'WR' : r.pos, isRush = r.market_name === 'Rush Yards';
    var gone = mates.filter(function (i) {
      var grp = i.pos === 'TE' ? 'WR' : i.pos;
      return /^(Out|IR|Doubtful)/.test(i.status || '') && gbNameKey(i.player) !== gbNameKey(r.player_name) &&
        (grp === myGrp || (isRush && r.pos === 'QB' && i.pos === 'RB' && gbRole(gbFantasy(r.team), i) === 'lead back'));
    });
    if (!gone.length) return null;
    var season = new Date(g.kickoffIso).getUTCFullYear(), fan = gbFantasy(r.team);
    var names = gone.map(function (i) { var role = gbRole(fan, i); return esc(i.player) + (role ? ' (' + role + ')' : ' (' + esc(i.pos) + ')'); });
    var starts = gone.map(function (i) { return gbSinceWeek(i.since, season); }).filter(function (w) { return w != null; });
    var gl = (D.propGameLogs || {})[r.game_log_key] || [], mi = r.model_inputs || {};
    var last = (function () { var k = gbNameKey(r.player_name).split('|')[1] || ''; return k.charAt(0).toUpperCase() + k.slice(1); })();
    var txt = names.join(', ') + (gone.length > 1 ? ' are' : ' is') + ' out', after = [], before = [];
    if (starts.length && gl.length) {
      var wk = Math.max.apply(null, starts);   // the most recent loss is the current role
      gl.forEach(function (x) { (x[0] >= wk ? after : before).push(Number(x[2])); });
      var since = gone.filter(function (i) { return gbSinceWeek(i.since, season) === wk; })[0].since;
      txt += ' (on the report since ' + esc(String(since).replace(/^[A-Za-z]+, /, '')) + ').';
    } else txt += '.';
    var avg = function (a) { return a.reduce(function (s2, v) { return s2 + v; }, 0) / a.length; };
    var proj = Number(r.model_predicted_value), unit = isRush ? 'rush yds' : r.market_name === 'Receptions' ? 'catches' : 'rec yds';
    var shareK = isRush ? ['roll_carry_share', 'last2_carry_share', 'carries'] : ['roll_rec_share', 'last2_rec_share', 'catches'];
    var s0 = mi[shareK[0]], s2 = mi[shareK[1]];
    var role = s0 != null && s2 != null && s2 - s0 >= 0.05;
    if (after.length) {
      txt += ' Since then ' + esc(last) + ' has ' + after.length + ' game' + (after.length === 1 ? '' : 's') + ': ' + after.map(function (v) { return v; }).join(', ') + ' ' + unit +
        ' (' + avg(after).toFixed(1) + ' a game' + (before.length ? ', vs ' + avg(before).toFixed(1) + ' before' : '') + ').';
    }
    if (role) txt += ' His share of the ' + shareK[2] + ' is up to ' + (s2 * 100).toFixed(0) + '% the last 2 games from ' + (s0 * 100).toFixed(0) + '% on the season' +
      (isRush && mi.last2_rush_att != null && mi.roll_rush_att != null ? ' (' + mi.last2_rush_att.toFixed(0) + ' carries a game, up from ' + mi.roll_rush_att.toFixed(0) + ')' : '') + '.';
    // Does the new role cut for or against this pick?
    var newAvg = after.length ? avg(after) : null, oldAvg = before.length ? avg(before) : null;
    var grew = newAvg != null && oldAvg != null && Math.abs(newAvg - oldAvg) >= Math.max(3, 0.2 * Math.max(newAvg, oldAvg)) ? newAvg > oldAvg : (role ? true : null);
    var lean = grew == null ? null : (grew === over ? 'help' : 'risk');
    var adj = r.injury_adjusted && r.injury_adjusted.length;
    if (adj) {
      var nw = mi.n_without_carry_share, sw = mi.carry_share_without, vac = mi.vac_carry_share;
      txt += ' <b class="gb-inline">The model accounts for it.</b> With ' + esc(r.injury_adjusted.join(', ')) + ' out' +
        (vac != null ? ' (' + (vac * 100).toFixed(0) + '% of the carries freed up)' : '') + ', this projection comes from the injury-aware model, built on ' +
        (nw ? 'his ' + (sw * 100).toFixed(0) + '% share of the carries in the ' + nw + ' game' + (nw === 1 ? '' : 's') + ' he&rsquo;s played without them'
            : 'his season share, since he hasn&rsquo;t played a game without them yet') + ': ' + gbNum(proj) + ' ' + unit + '.';
      if (newAvg != null && after.length) txt += (Math.abs(proj - newAvg) <= Math.max(5, 0.15 * newAvg)
        ? ' That lines up with what he&rsquo;s done since.' : proj < newAvg
        ? ' Still below his ' + newAvg.toFixed(1) + ' a game since, so the under is betting that number comes down' + (after.length <= 2 ? ' (only ' + after.length + ' game' + (after.length === 1 ? '' : 's') + ' to go on)' : '') + '.'
        : ' Above his ' + newAvg.toFixed(1) + ' a game since.');
      return { html: txt, lean: (newAvg != null && proj < newAvg * 0.8 && !over) ? 'risk' : null };
    }
    txt += ' The model has no injury input &mdash; it only sees this through his last-2-games numbers, while his season numbers still include games ' +
      (gone.length > 1 ? 'they' : 'he') + ' played. ';
    if (lean === 'risk') txt += grew
      ? 'His role has grown, and the projection (' + gbNum(proj) + ')' + (newAvg != null ? ' is ' + (proj < newAvg ? 'below' : 'near') + ' his ' + newAvg.toFixed(1) + ' a game since' : ' still leans on his season numbers') +
        ' &mdash; the model may be underrating his new workload. That&rsquo;s the biggest risk to this under' + (after.length && after.length <= 2 ? ' (only ' + after.length + ' game' + (after.length === 1 ? '' : 's') + ' of the new role, so it&rsquo;s still a small read)' : '') + '.'
      : 'His production has dropped since, which works against this over.';
    else if (lean === 'help') txt += grew ? 'The bigger role points the same way as this over.' : 'His numbers have dropped since' + (newAvg != null ? ' (' + newAvg.toFixed(1) + ' a game)' : '') + ', which backs this under.';
    else txt += 'His numbers haven&rsquo;t moved much since, so the injuries don&rsquo;t change the read.';
    return { html: txt, lean: lean };
  }
  function gbFootballProp(r, g) {
    var over = r.model_lean === 'over', me = schoolAbbr(r.team), opp = schoolAbbr(r.opponent);
    var u = gbUnits(r.team, r.opponent) || { o: {}, d: {} }, mi = r.model_inputs || {};
    var tSpr = r.team === g.homeSchool ? g.marketSpread : -g.marketSpread;   // < 0 = his team favored
    var last = (function () { var k = gbNameKey(r.player_name).split('|')[1] || ''; return k.charAt(0).toUpperCase() + k.slice(1); })();
    var parts = [], risk = [], o = u.o;
    var script = tSpr == null || isNaN(tSpr) ? null : tSpr <= -7 ? 'bigfav' : tSpr < 0 ? 'fav' : tSpr >= 7 ? 'bigdog' : 'dog';
    if (r.market_name === 'Rush Yards' && r.pos === 'QB') {
      var prs = gbAvg(u.o, ['pff_pres', 'sacks', 'havoc']);
      parts.push('QB rushing yards are a weird stat in college: sacks count against them. ' + esc(last) + '&rsquo;s total is scrambles and designed runs minus every sack.' +
        (prs && prs.v <= -0.1 ? ' ' + esc(opp) + ' gets after the passer (' + gbCiteD(prs.worst) + '), so a couple of sacks can erase a scramble &mdash; that&rsquo;s built-in help for the under.' : ''));
      if (o.rush_epa && o.rush_epa.diff <= -0.15) parts.push(esc(opp) + ' also tackles well against the run (' + gbCiteD(o.rush_epa) + '), which limits how far his scrambles go.');
      risk.push('one long scramble or a broken designed run can cover the under by itself');
      if (script === 'dog' || script === 'bigdog') risk.push('if ' + esc(me) + ' falls behind, QBs tend to take off more when they&rsquo;re pressing');
    } else if (r.market_name === 'Rush Yards') {
      var front = gbAvg(u.o, ['line', 'stuff', 'pff_rblk', 'rush_epa']);
      parts.push((front && front.v <= -0.1 ? esc(opp) + '&rsquo;s front wins at the line of scrimmage (' + gbCiteD(front.worst) + '), so backs get hit at or behind the line instead of getting to the second level. '
        : esc(opp) + ' is tough against the run. ') + 'Backs usually beat a rushing under with one or two chunk runs; a defense that keeps everything in front of it takes those away.');
      if (mi.roll_carry_share != null || r.carry_share != null) {
        var cs = mi.roll_carry_share != null ? mi.roll_carry_share : r.carry_share;
        var cs2 = mi.last2_carry_share;
        if (cs2 != null && cs2 - cs >= 0.05) parts.push('His workload has grown lately (' + (cs2 * 100).toFixed(0) + '% of the carries the last 2 games, up from ' + (cs * 100).toFixed(0) + '% on the season), so this under has to beat a bigger role than his season numbers show.');
        else if (cs < 0.45) parts.push('He&rsquo;s in a committee (' + (cs * 100).toFixed(0) + '% of the carries), so his volume isn&rsquo;t guaranteed &mdash; fewer carries means fewer chances to break one.');
        else parts.push('He&rsquo;s the lead back (' + (cs * 100).toFixed(0) + '% of the carries), so the under is about efficiency, not volume.');
      }
      if (script === 'dog' || script === 'bigdog') parts.push(esc(me) + ' is the underdog: if it falls behind, it throws more and his carries dry up in the second half.');
      if (script === 'bigfav') risk.push(esc(me) + ' is a big favorite, so he could get a pile of 4th-quarter carries running out the clock &mdash; the main risk to the under');
      else if (script === 'fav') risk.push('if ' + esc(me) + ' gets an early lead he&rsquo;ll get clock-killing carries late');
      risk.push('a single long run can blow up any rushing under');
    } else if (r.market_name === 'Receptions') {
      var glR = ((D.propGameLogs || {})[r.game_log_key] || []).map(function (q) { return Number(q[2]); });
      var avgR = glR.length ? glR.reduce(function (a2, b2) { return a2 + b2; }, 0) / glR.length : null;
      parts.push('Receptions are a pure volume stat: no targets, no catches. ' + (mi.roll_team_pass_att != null ? esc(me) + ' throws about ' + mi.roll_team_pass_att.toFixed(0) + ' times a game' +
        (r.rec_share != null ? ' and ' + esc(last) + ' catches about ' + (r.rec_share * 100).toFixed(0) + '% of the team&rsquo;s receptions' : '') + '. ' : '') +
        (avgR == null ? '' : over
          ? 'That&rsquo;s worked out to ' + avgR.toFixed(1) + ' catches a game.'
          : avgR < r.line ? 'That&rsquo;s worked out to ' + avgR.toFixed(1) + ' catches a game, so getting past ' + r.line + ' takes a bigger role than he usually gets.'
          : 'He&rsquo;s averaged ' + avgR.toFixed(1) + ' catches, above the line, so this under is the model saying this game shrinks his usual share (it projects ' + gbNum(r.model_predicted_value) + ').'));
      var cov = gbAvg(u.o, ['pass_epa', 'pff_pass', 'pff_recv', 'pd_sr']);
      if (cov && cov.v <= -0.15) parts.push(esc(opp) + '&rsquo;s coverage is good (' + gbCiteD(cov.worst) + '), which squeezes the windows and spreads the ball around.');
      if (script === 'fav' || script === 'bigfav') parts.push(esc(me) + ' is favored, and teams playing with a lead run it more in the second half &mdash; fewer late targets.');
      if (script === 'dog' || script === 'bigdog') risk.push(esc(me) + ' is the underdog; if it&rsquo;s chasing points, the volume goes up for everyone');
      risk.push('a short-area, quick-game plan can pile up catches even against good coverage');
    } else if (r.market_name === 'Reception Yards') {
      var aDot = mi.pff_matched === 1 ? mi.roll_pff_adot : null;
      parts.push(aDot != null && aDot >= 12 ? 'He&rsquo;s a downfield target (' + aDot.toFixed(1) + '-yard average depth), so one catch can get him halfway to ' + r.line + '.'
        : aDot != null && aDot < 8 ? 'He works underneath (' + aDot.toFixed(1) + '-yard average depth), so this over is built on volume and yards after the catch.'
        : 'He&rsquo;s a real part of the passing game, which is the point of the 25.5+ line rule: no part-time players.');
      if (mi.roll_pff_route_part != null) parts.push('He&rsquo;s running a route on ' + (mi.roll_pff_route_part * 100).toFixed(0) + '% of dropbacks &mdash; an every-down role, not a rotation guy.');
      var weak = gbAvg(u.o, ['pass_epa', 'pff_pass', 'pff_recv', 'expl']);
      if (weak && weak.v >= 0.15) parts.push(esc(opp) + '&rsquo;s pass defense is soft (' + gbCite(weak.best) + '), so the targets should come with yards.');
      else if (weak && weak.v <= -0.15) risk.push(esc(opp) + '&rsquo;s coverage is good (' + gbCiteD(weak.worst) + ') &mdash; the projection is betting on his role, not the matchup');
      if (script === 'dog' || script === 'bigdog') parts.push(esc(me) + ' is the underdog, so it should be throwing to catch up &mdash; more targets.');
      if (script === 'bigfav') risk.push(esc(me) + ' is a big favorite; if it&rsquo;s a blowout, the starters sit and the ball stays on the ground');
      risk.push('one quiet half or a game plan built around someone else');
    }
    var html = '<div class="gb-read"><b>The football read</b><p>' + parts.join(' ') + '</p></div>';
    var rc = gbRoleChange(r, g);
    if (rc) html += '<div class="gb-read' + (rc.lean === 'risk' ? ' is-risk' : '') + '"><b>Injuries at his position</b><p>' + rc.html + '</p></div>';
    if (risk.length) html += '<div class="gb-read"><b>What beats it</b><p>' + risk.join('; ').replace(/^./, function (c) { return c.toUpperCase(); }) + '.</p></div>';
    return html;
  }
  function gbWhySide(p, injKeyOpp, injKeyOwn) {
    var g = p.game, S = team(p.side).abbr, sideIsHome = p.side === g.home, out = [];
    var tier = M.confTier(p.market, p.sideProb), B = (D.confBaseline || {}).rows || [];
    var tr = B.filter(function (x) { return x.market === p.market && x.tier === tier; })[0];
    // 1) The number.
    if (p.market === 'Moneyline') {
      var be = M.impliedProb(p.sideMoneyline);
      out.push('<b>The number:</b> the model gives ' + esc(S) + ' ' + (p.sideProb * 100).toFixed(1) + '% to win; ' + gbPrice(p.sideMoneyline) + ' needs ' + (be * 100).toFixed(1) +
        '%, an edge of ' + gbSigned((p.sideProb - be) * 100, 1) + ' pts of win chance. Moneylines are unofficial.');
    } else {
      var mS = sideIsHome ? g.modelSpread : -g.modelSpread, kS = sideIsHome ? g.marketSpread : -g.marketSpread;
      out.push('<b>The number:</b> the model makes it ' + esc(S) + ' ' + M.signed(mS) + ', the market ' + esc(S) + ' ' + M.signed(kS) +
        ' &mdash; ' + Math.abs(mS - kS).toFixed(1) + ' pts of value, ' + (p.sideProb * 100).toFixed(1) + '% to cover where -110 needs 52.4%.');
    }
    if (tier) out.push('<b>Confidence:</b> ' + esc(tier) + ' tier' + (tr ? '. ' + esc(p.market) + ' plays in this tier went ' + tr.w + '-' + tr.l + (tr.p ? '-' + tr.p : '') +
      ' (' + gbSigned(tr.units, 1) + 'u) on the rebuilt weeks 4&ndash;10 card' : '') + '.');
    // 2) What drives the line.
    var dv = gbSideDrivers(p), pro = dv.rows.filter(function (x) { return x.pts > 0; }), con = dv.rows.filter(function (x) { return x.pts < 0; });
    var fmt = function (x) { return x.txt + (dv.measured ? ' <span class="gb-rk">' + gbSigned(x.pts, 1) + ' pts</span>' : ''); };
    if (pro.length) out.push('<b>Driving the model toward ' + esc(S) + ':</b> ' + pro.slice(0, 4).map(fmt).join('; ') + '.');
    if (con.length) out.push('<b>Working against it:</b> ' + con.slice(0, 3).map(fmt).join('; ') + '.');
    if (!dv.rows.length) out.push('<b>Driving the model:</b> ' + esc(g.note || 'inputs not logged for this game yet.'));
    if (dv.rows.length) out.push('<span class="gb-note">' + (dv.measured
      ? 'Points = how far the model&rsquo;s number moves if that input were even between the teams (boosted-tree model, so they don&rsquo;t add up exactly).'
      : 'Biggest inputs first (recent form carries about 39% of the model, team ratings 25%). Point pushes per input arrive with the next model run.') + '</span>');
    // 3) The QB input against the injury report.
    var inputs = g.inputs || {};
    (injKeyOpp || []).forEach(function (k) {
      if (k.role !== 'starting QB' || !/^(Out|IR|Doubtful)/.test(k.status || '')) return;
      var oppHome = !sideIsHome, flag = oppHome ? inputs.home_qb_out : inputs.away_qb_out;
      if (g.inputs && !flag) out.push('<b>QB note:</b> ' + esc(k.player) + ' is listed ' + esc(k.status.toLowerCase()) +
        ', but the model&rsquo;s QB-out input is off for that team (it only fires when the season&rsquo;s pass-attempts leader is out), so the line does not price his absence.');
    });
    return gbFootballSide(p, injKeyOpp, injKeyOwn) +
      '<details class="gb-nums"><summary>The numbers behind it</summary><ul class="gb-why-list">' + out.map(function (x) { return '<li>' + x + '</li>'; }).join('') + '</ul></details>';
  }
  // The Edge Board game a prop belongs to (index into D.games), or -1.
  // ---- Player card (11/2026) ----
  // A visual read of one prop: projection and season average against the
  // line, game-by-game bars, hit rates over recent windows (with the odds
  // each one is "worth"), and the opposing defense against the league.
  // Opens from a player's name on Player Props or in a game breakdown.
  var PC_REG = [];
  function pcOpenAttr(r) { PC_REG.push(r); return ' onclick="window.__cfbPlayerCard(' + (PC_REG.length - 1) + ')" '; }
  function pcFair(p) {
    if (p == null || p <= 0 || p >= 1) return '&mdash;';
    return p >= 0.5 ? '-' + Math.round(100 * p / (1 - p)) : '+' + Math.round(100 * (1 - p) / p);
  }
  function pcBar(val, line, max, good) {
    var w = Math.max(0, Math.min(100, val / max * 100)), lp = Math.max(0, Math.min(100, line / max * 100));
    return '<div class="pc-bar"><div class="pc-bar-fill ' + (good ? 'is-good' : 'is-bad') + '" style="width:' + w.toFixed(1) + '%"></div>' +
      '<div class="pc-bar-line" style="left:' + lp.toFixed(1) + '%"><span>line ' + line + '</span></div></div>';
  }
  function pcShort(x) {
    var a = schoolAbbr(x[1]);
    if (a && a.length <= 6) return a;
    var w = String(x[1] || '').split(/\\s+/).filter(Boolean);
    return w.length > 1 ? w.map(function (t) { return t.charAt(0); }).join('').toUpperCase().slice(0, 5) : (w[0] || 'Wk ' + x[0]).slice(0, 4).toUpperCase();
  }
  function pcGames(gl, line, under) {
    if (!gl.length) return '';
    var vals = gl.map(function (x) { return Number(x[2]); }), max = Math.max.apply(null, vals.concat([line])) * 1.15 || 1;
    var W = 320, H = 150, pad = 18, bw = Math.min(42, (W - 10) / gl.length - 8), step = (W - 10) / gl.length;
    var y = function (v) { return H - pad - (v / max) * (H - pad - 14); };
    var bars = gl.map(function (x, i) {
      var v = Number(x[2]), hit = under ? v < line : v > line, cx = 5 + step * i + step / 2;
      return '<rect x="' + (cx - bw / 2).toFixed(1) + '" y="' + y(v).toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + Math.max(2, H - pad - y(v)).toFixed(1) + '" rx="3" class="' + (hit ? 'pc-hit' : 'pc-miss') + '"/>' +
        '<text x="' + cx.toFixed(1) + '" y="' + (y(v) - 4).toFixed(1) + '" class="pc-gv">' + v + '</text>' +
        '<text x="' + cx.toFixed(1) + '" y="' + (H - 4) + '" class="pc-gl">' + esc(pcShort(x)) + '</text>';
    }).join('');
    var ly = y(line).toFixed(1);
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" class="pc-svg" role="img" aria-label="Game by game">' + bars +
      '<line x1="0" x2="' + W + '" y1="' + ly + '" y2="' + ly + '" class="pc-lineL"/></svg>';
  }
  window.__cfbPlayerCard = function (i) {
    var r = PC_REG[i];
    if (!r) return;
    var under = r.model_lean === 'under', side = under ? 'Under' : 'Over', line = Number(r.line);
    var unit = { 'Rush Yards': 'rush yds', 'Receptions': 'catches', 'Reception Yards': 'rec yds' }[r.market_name] || String(r.market_name || '').toLowerCase();
    var price = under ? r.under_price : r.over_price, be = price != null ? M.impliedProb(Number(price)) : null;
    var gl = (D.propGameLogs || {})[r.game_log_key] || [], vals = gl.map(function (x) { return Number(x[2]); });
    var proj = Number(r.model_predicted_value), avg = vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : null;
    var max = Math.max.apply(null, [line * 1.6, proj * 1.1, (avg || 0) * 1.1].concat(vals.map(function (v) { return v * 1.05; }))) || 1;
    var good = function (v) { return under ? v < line : v > line; };
    var tone = function (v) { return good(v) ? 'is-good' : 'is-bad'; };
    var html = '';
    // Header
    html += '<div class="pc-head"><div><div class="pc-name">' + esc(r.player_name) + '</div><div class="pc-sub">' + esc(schoolAbbr(r.team)) + ' vs ' + esc(schoolAbbr(r.opponent)) +
      ' &middot; ' + esc(propKickoff(r.start_time)) + (r.pos ? ' &middot; ' + esc(r.pos) : '') + '</div></div><button class="pc-x" onclick="window.__cfbPlayerCardClose()" aria-label="Close">&times;</button></div>';
    html += '<div class="pc-pick"><span class="pc-side ' + (under ? 'is-under' : 'is-over') + '">' + side.toUpperCase() + '</span><span class="pc-pickline">' + line + ' ' + esc(unit) + '</span>' +
      (price != null ? '<span class="pc-price">' + (price > 0 ? '+' : '') + Number(price) + (r.book_used ? ' &middot; ' + esc(bookLabel(r.book_used)) : '') + '</span>' : '') +
      (r.is_official_play ? '<span class="pc-off">&#9733; Official</span>' : '') + '</div>';
    if (r.mu_grade) html += '<div class="pc-mu">Matchup <b>' + esc(r.mu_grade) + '</b>' + (r.mu_score != null ? ' &middot; ' + r.mu_score + '/100' : '') + ' for the ' + side.toLowerCase() + '</div>';
    // Projection
    if (!isNaN(proj)) {
      var d1 = proj - line;
      html += '<div class="pc-tile"><div class="pc-lab">Model projection' + (r.injury_adjusted && r.injury_adjusted.length ? ' <em>injury-adjusted</em>' : '') + '</div>' +
        '<div class="pc-big ' + tone(proj) + '">' + proj.toFixed(1) + ' <small>vs ' + line + ' line &middot; ' + (d1 > 0 ? '+' : '') + d1.toFixed(1) + '</small></div>' +
        pcBar(proj, line, max, good(proj)) +
        '<div class="pc-note">' + side + ' hit chance <b>' + (r.model_lean_probability != null ? (r.model_lean_probability * 100).toFixed(1) + '%' : '&mdash;') + '</b>' +
        (be != null ? ' &middot; price needs ' + (be * 100).toFixed(1) + '%' : '') + (r.model_ev != null ? ' &middot; edge <b>' + gbSigned(r.model_ev, 1) + '%</b>' : '') +
        (r.no_injury && r.no_injury.model_predicted_value != null ? '<br>Without injuries: ' + gbNum(r.no_injury.model_predicted_value) : '') + '</div></div>';
    }
    // Season average
    if (avg != null) {
      var d2 = avg - line, hitsN = vals.filter(good).length;
      html += '<div class="pc-tile"><div class="pc-lab">Season average (' + vals.length + ' game' + (vals.length === 1 ? '' : 's') + ')</div>' +
        '<div class="pc-big ' + tone(avg) + '">' + avg.toFixed(1) + ' <small>vs ' + line + ' line &middot; ' + (d2 > 0 ? '+' : '') + d2.toFixed(1) + '</small></div>' +
        pcBar(avg, line, max, good(avg)) +
        '<div class="pc-note">Averaging ' + (avg < line ? 'BELOW' : 'ABOVE') + ' the line &middot; ' + side.toLowerCase() + ' in ' + hitsN + ' of ' + vals.length + '</div></div>';
      html += '<div class="pc-tile"><div class="pc-lab">Game by game <em>green = ' + side.toLowerCase() + ' hit &middot; dashed = line ' + line + '</em></div>' + pcGames(gl, line, under) + '</div>';
    }
    // Hit-rate windows
    if (vals.length) {
      var wins = [], n = vals.length;
      [3, 5, 10].forEach(function (k) { if (k < n) wins.push(k); });
      wins.push(n);
      html += '<div class="pc-grid">' + wins.map(function (k) {
        var sl = vals.slice(-k), h = sl.filter(good).length, p = h / sl.length, o = sl.length - h;
        var cls = be == null ? '' : p >= be + 0.05 ? 'is-good' : p <= be - 0.05 ? 'is-bad' : 'is-even';
        return '<div class="pc-hr"><div class="pc-hr-top"><span class="pc-hr-pct ' + cls + '">' + Math.round(p * 100) + '%</span><span class="pc-hr-odds">' + pcFair(p) + '</span></div>' +
          '<div class="pc-hr-lab">' + (k === n ? 'Season' : 'Last ' + k) + ': ' + side + ' ' + h + '/' + sl.length + ' &middot; ' + (under ? 'Over' : 'Under') + ' ' + o + '/' + sl.length + '</div>' +
          '<div class="pc-hr-bar"><div style="width:' + (p * 100).toFixed(0) + '%"></div>' + (be != null ? '<i style="left:' + (be * 100).toFixed(0) + '%"></i>' : '') + '</div></div>';
      }).join('') + '</div><div class="pc-note pc-key">Big number = how often the ' + side.toLowerCase() + ' hit; next to it, the odds that hit rate is worth. The tick on each bar is what ' +
        (price != null ? (price > 0 ? '+' : '') + Number(price) : 'the price') + ' needs.</div>';
    }
    // Opposing defense
    var allowKey = Object.keys(r.mu_allowed || {})[0];
    if (allowKey != null || r.mu_rank) {
      var pct = allowKey != null ? Number(r.mu_allowed[allowKey]) : null;
      var fav = pct == null ? null : (under ? pct < 0 : pct > 0);
      html += '<div class="pc-tile"><div class="pc-lab">' + esc(schoolAbbr(r.opponent)) + ' defense' + (r.mu_vs ? ' vs ' + esc(r.mu_vs) : '') + '</div>' +
        (pct == null && r.mu_rank && r.mu_of ? (function () {
          var soft = r.mu_rank / r.mu_of, f2 = under ? soft <= 0.5 : soft > 0.5;
          return '<div class="pc-big ' + (f2 ? 'is-good' : 'is-bad') + '">' + gbOrd(r.mu_rank) + ' <small>of ' + r.mu_of + '</small></div>' +
            '<div class="pc-bar"><div class="pc-bar-fill ' + (f2 ? 'is-good' : 'is-bad') + '" style="width:' + (soft * 100).toFixed(1) + '%"></div>' +
            '<div class="pc-bar-line" style="left:50%"><span>middle</span></div></div><div class="pc-note pc-ends"><span>toughest</span><span>softest</span></div>';
        })() : '') +
        (pct != null ? '<div class="pc-big ' + (fav ? 'is-good' : 'is-bad') + '">' + (pct > 0 ? '+' : '') + pct + '% <small>' + esc(allowKey) + ' vs what offenses usually get</small></div>' +
          '<div class="pc-bar pc-bar-c"><div class="pc-bar-mid"><span>league avg</span></div><div class="pc-bar-fill ' + (fav ? 'is-good' : 'is-bad') + '" style="' +
          (pct < 0 ? 'right:50%;width:' : 'left:50%;width:') + Math.min(50, Math.abs(pct) * 0.8).toFixed(1) + '%"></div></div>' : '') +
        '<div class="pc-note">' + (r.mu_rank ? 'Ranks ' + gbOrd(r.mu_rank) + ' of ' + r.mu_of + ' (1st = toughest). ' : '') +
        (pct != null ? (fav ? 'Favors the ' : 'Works against the ') + side.toLowerCase() + '.' : '') + '</div></div>';
    }
    var gi = gameIndexForProp(r);
    if (gi >= 0) html += '<button class="pp-gobtn pc-go" onclick="window.__cfbPlayerCardClose(); window.__cfbGoBreakdown(' + gi + ', null)">Game breakdown &rarr;</button>';
    var ov = document.getElementById('pc-overlay');
    if (!ov) { ov = document.createElement('div'); ov.id = 'pc-overlay'; document.body.appendChild(ov);
      ov.addEventListener('click', function (e) { if (e.target === ov) window.__cfbPlayerCardClose(); }); }
    ov.innerHTML = '<div class="pc-card" role="dialog" aria-label="' + esc(r.player_name) + '">' + html + '</div>';
    ov.classList.add('is-open');
    document.body.classList.add('pc-lock');
  };
  window.__cfbPlayerCardClose = function () {
    var ov = document.getElementById('pc-overlay');
    if (ov) ov.classList.remove('is-open');
    document.body.classList.remove('pc-lock');
  };
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') window.__cfbPlayerCardClose(); });
  function gameIndexForProp(r) {
    var t = r.start_time ? Date.parse(r.start_time) : NaN, best = -1, bestGap = Infinity;
    (D.games || []).forEach(function (g, i) {
      var sides = [g.homeSchool, g.awaySchool];
      if (sides.indexOf(r.team) < 0 || (r.opponent && sides.indexOf(r.opponent) < 0)) return;
      var gap = isNaN(t) ? 0 : Math.abs(Date.parse(g.kickoffIso) - t);
      if (gap < bestGap) { best = i; bestGap = gap; }
    });
    return bestGap <= 36 * 3600 * 1000 ? best : -1;
  }
  function renderGameBreakdown(p) {
    var TM = D.teamMatchup || {}, meta = D.teamMatchupMeta || {};
    if (!p || !meta.cats) return '';
    var g = p.game, A = team(g.away), H = team(g.home);
    var pa = TM[g.awaySchool], ph = TM[g.homeSchool];
    if (!pa || !ph) return '';
    var aOff = gbSide(A, H, tmCompare(pa, ph)), hOff = gbSide(H, A, tmCompare(ph, pa));
    // Matchup score per team: its offense's net edge plus its defense's net edge.
    var aScore = aOff.net - hOff.net, hScore = -aScore;
    var aEdges = aOff.oE + hOff.dE, hEdges = hOff.oE + aOff.dE;
    var eGap = aEdges - hEdges;
    var better = Math.abs(eGap) >= 3 ? (eGap > 0 ? A : H) : (Math.abs(aScore) >= 0.08 ? (aScore > 0 ? A : H) : null);
    var bothOff = aOff.oE >= aOff.dE + 6 && hOff.oE >= hOff.dE + 6, bothDef = aOff.dE >= aOff.oE + 6 && hOff.dE >= hOff.oE + 6;
    var flavor = bothOff ? ' Both offenses have the upper hand on the other&rsquo;s defense, which points to a high-scoring game.'
      : (bothDef ? ' Both defenses grade ahead of the other&rsquo;s offense, which points to a lower-scoring game.' : '');
    var bottom = '<b>Bottom line:</b> ' + (better
      ? esc(better.abbr) + ' has the better matchup overall' + (Math.abs(eGap) >= 3
          ? ' (' + (better === A ? aEdges + ' to ' + hEdges : hEdges + ' to ' + aEdges) + ' in category edges across both sides of the ball).'
          : ': category edges are close (' + aEdges + '&ndash;' + hEdges + '), but ' + esc(better.abbr) + '&rsquo;s edges come by wider rank gaps.')
      : 'the units grade out close to even (' + aEdges + ' edges for ' + esc(A.abbr) + ', ' + hEdges + ' for ' + esc(H.abbr) + ').') + flavor;
    var injAForWhy = gbInjuries(A, g.awaySchool, g.injAway).key, injHForWhy = gbInjuries(H, g.homeSchool, g.injHome).key;
    var play;
    if (!p.side) {
      play = 'No ' + esc(p.market === 'Moneyline' ? 'moneyline' : 'spread') + ' posted yet, so there&rsquo;s no play.';
    } else {
      var S = team(p.side), sideIsA = p.side === g.away;
      var sideScore = sideIsA ? aScore : hScore;
      var desc = p.market === 'Moneyline'
        ? 'The model gives ' + esc(S.abbr) + ' ' + (p.sideProb * 100).toFixed(1) + '% to win at ' + (p.sideMoneyline > 0 ? '+' : '') + p.sideMoneyline + ' (moneylines are unofficial)'
        : 'The model has ' + esc(S.abbr) + ' ' + M.signed(sideIsA ? -g.modelSpread : g.modelSpread) + ', the market ' + esc(S.abbr) + ' ' +
          M.signed(sideIsA ? -g.marketSpread : g.marketSpread) + ', ' + (p.sideProb * 100).toFixed(1) + '% to cover';
      var fit;
      if (better && better.abbr === S.abbr) fit = 'The matchup backs it: ' + esc(S.abbr) + ' has the better units.';
      else if (better) fit = p.market === 'Spread' && /\\+/.test(p.playLabel)
        ? esc(S.abbr) + ' has the weaker units, so this is a points play: the model thinks the number is too big, not that ' + esc(S.abbr) + ' is the better team.'
        : 'Heads up: the matchup leans the other way. The edge comes from the model&rsquo;s number, not the unit ranks.';
      else fit = 'The matchup is close to even, so the play rests on the model&rsquo;s number.';
      play = (p.qualifies
        ? '<span class="gb-play">' + trkMark(S.abbr, 18) + ' ' + esc(p.playLabel) + '</span> <span class="tier"><span>' + esc(p.tier) + '</span></span> '
        : '<span class="gb-play is-lean">Lean ' + esc(p.playLabel) + '</span> <em class="gb-off">not on the Bet Card (edge below threshold)</em> ') +
        '<div class="gb-why">' + desc + '. ' + fit + '</div>';
      play += lineMoveHtml(p, false);
      var onCard = p.qualifies && M.buildBetCard(D.games, opts()).some(function (c) { return c.game.home === g.home && c.game.away === g.away && c.game.kickoffIso === g.kickoffIso; });
      if (onCard) play += '<div class="gb-lab2 gb-why-lab">Why it&rsquo;s a play</div>' + gbWhySide(p, p.side === g.away ? injHForWhy : injAForWhy, p.side === g.away ? injAForWhy : injHForWhy);
    }
    var sec = function (lab, html) { return html ? '<div class="gb-sec"><div class="gb-lab2">' + lab + '</div>' + html + '</div>' : ''; };
    var injA = gbInjuries(A, g.awaySchool, g.injAway), injH = gbInjuries(H, g.homeSchool, g.injHome);
    var adj = (g.flags || []).filter(function (f) { return / pts: /.test(f.text); }).map(function (f) { return esc(f.text); });
    var injNote = adj.length ? 'Model adjustment applied: ' + adj.join('; ') + '.'
      : 'The model line does not adjust for these.';
    var injHtml = '<p class="gb-p">' + injA.html + '</p><p class="gb-p">' + injH.html + '</p>' +
      '<div class="gb-note">' + injNote + (D.meta.injuriesAsOf ? ' Report as of ' + esc(D.meta.injuriesAsOf) + ' (Covers.com, team and media reports).' : '') + '</div>';
    var players = [gbPlayers(A, g.awaySchool), gbPlayers(H, g.homeSchool)].filter(Boolean);
    var props = gbProps(g);
    // Risk line for the play: key players listed on either side.
    var risk = '';
    if (p.side) {
      var S2 = team(p.side), sideKey = p.side === g.away ? injA.key : injH.key, oppKey = p.side === g.away ? injH.key : injA.key;
      var fmtK = function (k) { return esc(k.player) + ' (' + esc(k.role || k.pos) + ', ' + esc(k.status.toLowerCase()) + ')'; };
      var parts = [];
      if (sideKey.length) parts.push('the play is on ' + esc(S2.abbr) + ' with ' + sideKey.map(fmtK).join(', ') + ' on the report');
      if (oppKey.length) parts.push('the other side has ' + oppKey.map(fmtK).join(', ') + ' listed');
      if (parts.length) risk = '<div class="gb-risk">&#9888; Injury check: ' + parts.join('; ') + '. Confirm status before kickoff.</div>';
    }
    var back = state.bdFrom ? '<button type="button" class="gb-back" onclick="window.__cfbBackToProps()">&larr; Back to Player Props</button>' : '';
    return '<div class="tm-card gb-card">' + back + '<div class="tm-title">' + icon('edge') + '<span>Game breakdown</span><em>' +
      esc(A.abbr) + ' at ' + esc(H.abbr) + '</em></div>' +
      '<div class="gb-info">' + gbInfo(p) + '</div>' +
      sec('Matchup', '<p class="gb-p">' + aOff.html + '</p><p class="gb-p">' + hOff.html + '</p><p class="gb-p">' + bottom + '</p>') +
      sec('Injury report', injHtml) +
      sec('Players to know <span>(model projections this week)</span>', players.length ? '<p class="gb-p">' + players.join('</p><p class="gb-p">') + '</p>' : '') +
      sec('Official props in this game <span>(why each one is a play)</span>', props.length ? props.join('') : '') +
      '<div class="gb-playbox"><div class="gb-lab">The play</div>' + play + risk + '</div>' +
      '<div class="tm-foot">Matchup text comes from the team ranks below. The play comes from the model&rsquo;s number against the market; the matchup and injuries are context, not extra filters (not yet backtested).</div></div>';
  }
  window.__cfbPpFilters = function () { state.ppFilters = !state.ppFilters; render(); };

  function confChip(t) {
    return t ? '<span class="conf-tier is-' + t.toLowerCase() + '" title="Confidence tier: ' + t + ' (ranks the play by model win %; still 1u)">' + t.toUpperCase() + '</span>' : '';
  }

  // ---- Line movement since first flagged (11/2026) ----
  // From the export's line snapshots: the spread when the model first
  // flagged the game vs the spread now, from the picked side's view.
  function lineMove(c) {
    var g = c.game, fl = g.flagLine;
    if (!fl || c.market !== 'Spread' || g.marketSpread == null || !c.side) return null;
    var home = c.side === g.home, ab = abbrOf(c.side);
    var first = home ? fl.first : -fl.first, now = home ? g.marketSpread : -g.marketSpread;
    var worse = first - now;            // + = the number got worse for the pick
    if (Math.abs(worse) < 0.25) return null;
    var lo = Math.min(Math.abs(first), Math.abs(now)), hi = Math.max(Math.abs(first), Math.abs(now));
    var keys = [3, 7, 10, 14].filter(function (k) { return k > lo && k < hi + 0.01 && (first < 0) === (now < 0); });
    var d = fl.at ? new Date(fl.at) : null;
    var when = d ? ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.getDay()] + ' ' + (d.getMonth() + 1) + '/' + d.getDate() : '';
    var fmt = function (v) { return (v > 0 ? '+' : '') + v; };
    return { worse: worse, first: first, now: now, keys: keys, when: when, ab: ab,
      txt: esc(ab) + ' ' + fmt(first) + ' &rarr; ' + fmt(now) + (when ? ' since first flagged ' + when : '') };
  }
  function lineMoveHtml(c, compact) {
    var m = lineMove(c);
    if (!m) return '';
    // 11/12/2026 check (rebuilt 2026 card, weeks 4-10, graded at the locked
    // line): plays whose line moved TOWARD the picked team after it was
    // flagged went 16-4 (+10.5u); lines that moved away went 17-15 (+0.5u).
    // A worse number here means the market came to the model -- a price
    // warning, not a reason to fade. Small sample; revisit as weeks grade.
    if (m.worse > 0) return '<div class="lm-warn">&#9888; Worse number now: ' + m.txt + ' (' + m.worse.toFixed(1) + ' pts).' +
      (m.keys.length ? ' <b>It crossed ' + (m.keys.length > 1 ? 'key numbers ' : 'the key number ') + m.keys.join(' and ') + '</b>, so a ' + m.keys[0] + '-point ' + (m.now < 0 ? 'win' : 'loss') + ' that used to cash no longer does.' : '') +
      ' The market has moved toward ' + esc(m.ab) + ', the same way the model leans &mdash; that&rsquo;s the market agreeing, not a reason to fade (2026: plays like this went 16-4). Shop for the best number left.' + '</div>';
    return '<div class="lm-good">&#10003; Better number now: ' + m.txt + ' (' + (-m.worse).toFixed(1) + ' pts). Heads up: that means the market moved away from the model’s side (2026: plays like this went 17-15).</div>';
  }
  // 11/2026: what changed since the last model run (spread card only).
  function cardKey(g) { return g.home + '|' + g.away + '|' + g.kickoffIso; }
  function cardChanges(card) {
    var prev = D.cardPrev;
    if (state.market !== 'Spread' || !prev || !prev.plays) return null;
    var now = {}, pk = {};
    card.forEach(function (c) { now[cardKey(c.game)] = c; });
    prev.plays.forEach(function (p) { pk[p.key] = p; });
    var byKey = {};
    D.games.forEach(function (g) { byKey[cardKey(g)] = g; });
    var dropped = prev.plays.filter(function (p) { return !now[p.key] && new Date(p.kickoff) > new Date(); }).map(function (p) {
      var g = byKey[p.key], pr = g ? M.priceGame(g, opts()) : null;
      return { p: p, why: !pr ? 'no line posted now' : pr.qualifies ? 'still a play, ranked out of the top 10' : 'edge at the current line is under ' + D.meta.minEdge.toFixed(1) + ' pts' +
        (g ? ' (' + esc(abbrOf(p.side === abbrOf(g.home) ? g.home : g.away)) + ' now ' + (function () { var v = p.side === abbrOf(g.home) ? g.marketSpread : -g.marketSpread; return (v > 0 ? '+' : '') + v; })() + ')' : '') };
    });
    var d = new Date(prev.at);
    return { isNew: function (c) { return !pk[cardKey(c.game)]; }, dropped: dropped,
      since: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.getDay()] + ' ' + (d.getMonth() + 1) + '/' + d.getDate() + ' ' + d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) };
  }
  function renderBetCard(card) {
    var chg = cardChanges(card);
    return '' +
      '<div class="section-head mt-md"><div class="section-title"><div class="section-flag is-green"></div>' +
      '<h2>Bet Card</h2></div></div>' +
      // 11/12/2026: timing and line-move notes (trust lab + rebuilt-card check).
      '<div class="bc-note"><div class="bc-note-h">&#9201; Bet early &mdash; and how to read line moves</div>' +
        '<p><b>Timing matters more than anything else we&rsquo;ve tested.</b> 2024&ndash;26 backtest of this card: <b>+5.8%</b> at the opening line, <b>&minus;3.0%</b> at the closing line. The edge is biggest when the model first flags a play, before the market moves.</p>' +
        '<p><b>&#9888; Worse number now</b> (e.g. IU &minus;7.5 &rarr; &minus;8.5): the market moved toward our side &mdash; it&rsquo;s agreeing with the model. Not a fade. 2026: <b>16-4</b>.</p>' +
        '<p><b>&#10003; Better number now</b> (e.g. SJSU +10.5 &rarr; +12.5): the market moved away from our side &mdash; it disagrees. Not a reason to bet more. 2026: <b>17-15</b>.</p>' +
        '<p class="bc-note-s">2026 records: rebuilt card, weeks 4&ndash;10, graded at the locked line, 68 plays &mdash; a small sample, not a rule yet.</p>' +
        '<p><b>Every play on this card is still a bet at the line shown</b> &mdash; the card is rebuilt from the current line every run. If a play stops being a bet, it drops off and shows up below the card.</p></div>' +
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
              '<div><div class="card-play">' + (c.side ? trkMark(abbrOf(c.side), 22) : '') + '<span>' + esc(c.playLabel) + '</span>' + (chg && chg.isNew(c) ? '<span class="bc-new">NEW</span>' : '') + '</div>' +
                '<div class="card-note">' + esc(abbrOf(c.game.away) + ' at ' + abbrOf(c.game.home) + ' \\u00b7 ' + edgeKickoff(c.game)) +
                  ' \\u00b7 <span style="font-weight:700;color:' + (isTrainedGame ? '#2ecc71' : '#8A94A3') + '">' + (isTrainedGame ? (c.game.modelVersion === 'upgraded' ? 'UPGRADED' : 'TRAINED') : 'UNTRAINED') + '</span></div></div>' +
              '<div class="card-price">' + esc(sidePriceLabel) + '</div>' +
              '<div class="card-conf">' + (sideProb * 100).toFixed(1) + '%' + confChip(M.confTier(c.market, sideProb)) + '</div>' +
              '<div class="num"><span class="tier"><span>' + esc(c.tier) + '</span></span></div>' +
              '<div class="num"><button class="track-btn" onclick="window.__cfbTrack(' + trackPayload(c, true) + ')">+TRK</button></div>' +
              (lineMoveHtml(c, true) ? '<div style="grid-column:1/-1;padding-top:6px">' + lineMoveHtml(c, true) + '</div>' : '') +
              (isNearEvenTarget(c) ? '<div style="grid-column:1/-1;font-size:11px;color:var(--green);padding-top:6px;line-height:1.4">\u2605 Near-even target: +100 to +150 with the model at 65%+ \u2014 small stake</div>' : '') +
              (qbFlags.length ? '<div style="grid-column:1/-1;font-size:11px;color:var(--amber);padding-top:6px;line-height:1.4">\u26a0 Injury: ' +
                qbFlags.map(function (f) { return esc(abbrOf(f.team)) + ' ' + esc(f.text); }).join(' \u00b7 ') +
                ' \u2014 the model doesn\u2019t account for this. Check the latest status before betting.</div>' : '') +
              (showCaveat ? '<div style="grid-column:1/-1;font-size:11px;color:var(--amber);padding-top:6px;line-height:1.4">' +
                '\u26a0 Large model/market gap on a preseason-only estimate (no in-season data yet for this game) \u2014 likely reflects the model\u2019s limits, not a confirmed edge. Extra caution advised.</div>' : '') +
            '</div></div>';
        }).join('') : '<div class="empty-state">No plays clear the ' + D.meta.minEdge.toFixed(1) + '-pt threshold on ' + state.market + ' right now.</div>') +
      '</div>' +
      (chg && chg.dropped.length ? '<div class="bc-dropped"><div class="bc-note-h">Dropped off since the last run (' + esc(chg.since) + ')</div>' +
        chg.dropped.map(function (x) { return '<div class="bc-drop-row"><b>' + esc(x.p.label) + '</b> <span>' + esc(x.p.away) + ' at ' + esc(x.p.home) + '</span> &mdash; ' + x.why + '. <em>Not a bet now.</em></div>'; }).join('') + '</div>' : '');
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
    var bookKeys = {};
    scored.forEach(function (r) { if (r.book_used) (bookKeys[r.book_used] = bookKeys[r.book_used] || {})[[r.fixture_id, r.player_name, r.market_name].join('|')] = 1; });
    var bookSel = state.propBook && bookKeys[state.propBook] ? state.propBook : 'ALL';
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
      // Sportsbook filter (10/2026): the row shown is the best line on that book.
      if (bookSel !== 'ALL' && r.book_used !== bookSel) return;
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
    var visible = function (r) { return state.propShowPass || r._conf.tier !== 'PASS' || r.is_official_play; };

    // ---- Sort & Filter (10/2026) ----
    // Every filter applies in every view. Counts on each button are what
    // that button would show with the other filters left as they are.
    var myTrk = {};
    // flaggedDesc: a play whose line you later changed in the tracker still
    // counts as the same play here, under the line it was added at.
    var myStatus = {}, myProps = [];
    loadTrk().forEach(function (it) {
      if (it.type !== 'Prop') return;
      myTrk[it.description] = true; myStatus[it.description] = it.status || null;
      if (it.flaggedDesc) { myTrk[it.flaggedDesc] = true; myStatus[it.flaggedDesc] = it.status || null; }
      myProps.push(it);
    });
    var F = {
      dir: state.propDir === 'over' || state.propDir === 'under' ? state.propDir : 'all',
      minEdge: Number(state.propMinEdge) || 0,
      q: String(state.propQ || '').trim().toLowerCase(),
      mine: !!state.propMyList,
      minMu: Number(state.propMinMu) || 0
    };
    var passes = function (r, skip) {
      if (skip !== 'view' && view !== 'all' && r._sec !== view) return false;
      if (skip !== 'market' && mkt !== 'ALL' && r.market_name !== mkt) return false;
      if (skip !== 'cat' && state.propCat && state.propCat !== 'all' && state.propCat !== 'td' && mktCat(r.market_name) !== state.propCat) return false;
      if (skip !== 'game' && gameSel !== 'ALL' && r.fixture_id !== gameSel) return false;
      if (skip !== 'dir' && F.dir !== 'all' && r.model_lean !== F.dir) return false;
      if (skip !== 'edge' && F.minEdge && !(r.model_ev != null && r.model_ev >= F.minEdge)) return false;
      if (F.q) {
        var hay = [r.player_name, r.team, schoolAbbr(r.team), r.opponent, schoolAbbr(r.opponent)].join(' ').toLowerCase();
        if (hay.indexOf(F.q) === -1) return false;
      }
      if (F.mine && !myTrk[propTrackLabel(r)]) return false;
      if (skip !== 'mu' && F.minMu && !(r.mu_score != null && r.mu_score >= F.minMu)) return false;
      return visible(r);
    };
    var countBy = function (skip, test) { var n = 0; rows.forEach(function (r) { if (passes(r, skip) && test(r)) n++; }); return n; };
    var shown = rows.filter(function (r) { return passes(r); });

    // Sorting: a main sort and a tiebreaker. Clicking a column header sets
    // the main sort; clicking it again flips the direction.
    var SORTS = [['kickoff', 'Kickoff'], ['edge', 'Highest edge'], ['hit', 'Hit chance'], ['roi', 'Track record ROI'], ['matchup', 'Matchup grade'], ['player', 'Player A-Z'], ['market', 'Market']];
    var sortOk = function (k) { return SORTS.some(function (x) { return x[0] === k; }); };
    var sort1 = sortOk(state.propSort) ? state.propSort : 'kickoff';
    var sort2 = sortOk(state.propSort2) ? state.propSort2 : 'edge';
    var trackRoi = function (r) {
      var ru = (D.propRules || {})[r.play_kind];
      if (ru) return ru.roi;
      return r._h && !r._h.small ? r._h.roi : -1e9;
    };
    var num = function (x) { return x == null || isNaN(x) ? -1e9 : Number(x); };
    var cmpOf = function (k) {
      switch (k) {
        case 'edge': return function (a, b) { return num(b.model_ev) - num(a.model_ev); };
        case 'hit': return function (a, b) { return num(b._lp) - num(a._lp); };
        case 'roi': return function (a, b) { return trackRoi(b) - trackRoi(a); };
        case 'matchup': return function (a, b) { return num(b.mu_score) - num(a.mu_score); };
        case 'player': return function (a, b) { return String(a.player_name || '').localeCompare(String(b.player_name || '')); };
        case 'market': return function (a, b) { return marketName(a.market_name).localeCompare(marketName(b.market_name)) || (a.model_lean < b.model_lean ? -1 : a.model_lean > b.model_lean ? 1 : 0); };
        default: return function (a, b) { var x = a.start_time || '9', y = b.start_time || '9'; return x < y ? -1 : (x > y ? 1 : 0); };
      }
    };
    var rev = state.propSortRev ? -1 : 1;
    var c1 = cmpOf(sort1), c2 = cmpOf(sort2);
    var cmp = function (a, b) { return rev * c1(a, b) || c2(a, b) || num(b.model_ev) - num(a.model_ev); };

    // Prop type badge: color by family (receiving teal, rushing orange,
    // passing purple) so the kind of prop reads at a glance.
    var isRecvModel = function (r) { return r.market_name === 'Receptions' || r.market_name === 'Reception Yards'; };
    function mktCat(m) { m = m || ''; return /pass/i.test(m) ? 'pass' : (/rush/i.test(m) ? 'rush' : (/recep/i.test(m) ? 'rec' : 'other')); }
    var MKT_SHORT = { 'Receptions': 'REC', 'Reception Yards': 'REC YDS', 'Reception Touchdowns': 'REC TD', 'Rush Yards': 'RUSH YDS', 'Rush Attempts': 'RUSH ATT',
      'Rush Touchdowns': 'RUSH TD', 'Pass Yards': 'PASS YDS', 'Pass Touchdowns': 'PASS TD', 'Pass Attempts': 'PASS ATT', 'Pass Completions': 'COMP', 'Pass Interceptions': 'INT' };
    var mktBadge = function (m) { return '<span class="pp-mkt is-' + mktCat(m) + '">' + esc(MKT_SHORT[m] || marketName(m)) + '</span>'; };

    // Counts on the view buttons match what each view will actually list.
    var nSec = { official: 0, underlean: 0, watch: 0, tracked: 0, lean: 0 }, nPass = 0;
    rows.forEach(function (r) {
      if (r._conf.tier === 'PASS') nPass++;
      if (passes(r, 'view')) nSec[r._sec]++;
    });

    // 1) Live record, one line, plus the key to the two numbers on each row.
    var recBit = function (label, o) {
      if (!o || !o.graded) return '<span class="pp-rec"><span class="pp-rec-label">' + label + '</span><span class="pp-rec-n">no graded plays yet</span></span>';
      return '<span class="pp-rec"><span class="pp-rec-label">' + label + '</span><b>' + esc(o.record) + '</b>' +
        '<b style="color:' + tone(o.units) + '">' + signed(o.units, 2) + 'u</b><span class="pp-rec-n">' + o.graded + ' graded</span></span>';
    };
    var nRecv = 0, nRecvNew = 0, recvVer = null;
    rows.forEach(function (r) { if (isRecvModel(r) && r.model_ev != null) { nRecv++; if (r.model_version) { nRecvNew++; recvVer = r.model_version; } } });
    var modelLine = '<div class="pp-model">Model: receptions &amp; rec yds props &mdash; ' +
      (!nRecv ? 'none scored yet' : (nRecvNew === nRecv ? '<b>' + esc(recvVer) + '</b>, all ' + nRecv + ' scored by it'
        : '<span style="color:var(--red)">' + nRecvNew + ' of ' + nRecv + ' on the retrained model, the rest on the box-score fallback</span>')) +
      ' &middot; rush yds and other props &mdash; unchanged model</div>';
    var strip = '<div class="pp-recline">' +
      (T ? recBit('Rec yds overs', T.recYdsPlayNew || T.recYdsPlay) + recBit('Rush yds unders', T.rushUnderPlayNew || T.rushUnderPlay) + recBit('Rec unders', T.underPlayNew || T.underPlay) + recBit('Rec overs (watch)', T.official) + recBit('All auto-tracked', T.overall) : '') +
      '<span class="pp-key"><b>Edge</b> is the model&rsquo;s hit chance against the price. <b>Track record</b> is how plays at that edge size did in the backtest.</span>' +
      modelLine +
    '</div>';

    // 2) Controls: which plays, then the Sort & Filter panel.
    var seg = function (i, key, label, n) {
      return '<button class="' + (view === key ? 'is-on' : '') + '" aria-pressed="' + (view === key ? 'true' : 'false') + '" onclick="window.__cfbPropView(' + i + ')">' + label + ' (' + n + ')</button>';
    };
    var segOpt = function (field, value, cur, label, n) {
      return '<button class="' + (cur === value ? 'is-on' : '') + '" aria-pressed="' + (cur === value ? 'true' : 'false') + '" data-prop-set="' + field + '" data-prop-val="' + esc(value) + '">' + label + (n != null ? ' (' + n + ')' : '') + '</button>';
    };
    var sortSel = function (field, cur) {
      return '<select class="pp-select" data-prop-select="' + field + '" aria-label="' + (field === 'propSort' ? 'Sort by' : 'Then by') + '">' +
        SORTS.map(function (x) { return '<option value="' + x[0] + '"' + (x[0] === cur ? ' selected' : '') + '>' + x[1] + '</option>'; }).join('') + '</select>';
    };
    var mktCounts = {};
    rows.forEach(function (r) { if (passes(r, 'market')) mktCounts[r.market_name] = (mktCounts[r.market_name] || 0) + 1; });
    var mktOrder = markets.slice(1).sort(function (a, b) {
      var ca = ['rec', 'rush', 'pass', 'other'].indexOf(mktCat(a)), cb = ['rec', 'rush', 'pass', 'other'].indexOf(mktCat(b));
      return ca - cb || marketName(a).localeCompare(marketName(b));
    });
    var nAllMkt = 0; Object.keys(mktCounts).forEach(function (k) { nAllMkt += mktCounts[k]; });
    var books = Object.keys(bookKeys).sort(function (a, b) { return bookLabel(a).localeCompare(bookLabel(b)); });
    var edgeOpt = function (v, label) { return segOpt('propMinEdge', String(v), String(F.minEdge), label, countBy('edge', function (r) { return !v || (r.model_ev != null && r.model_ev >= v); })); };
    var nActive = (mkt !== 'ALL') + (gameSel !== 'ALL') + (F.dir !== 'all') + (bookSel !== 'ALL') + (F.minEdge > 0) + (F.q ? 1 : 0) + (F.mine ? 1 : 0) + (F.minMu > 0);
    var controls = '<div class="pp-controls">' +
      '<div class="pp-seg" role="group" aria-label="Which plays to show">' +
        seg(0, 'official', 'Official', nSec.official) +
        seg(3, 'underlean', 'Under leans', nSec.underlean) +
        seg(1, 'watch', 'Watch', nSec.watch) +
        seg(2, 'all', 'All props', nSec.official + nSec.underlean + nSec.watch + nSec.tracked + nSec.lean) +
      '</div>' +
      (view === 'all' ? '<button class="pp-chip' + (state.propShowPass ? ' is-on' : '') + '" onclick="window.__cfbPropShowPass()">Show passes (' + nPass + ')</button>' : '') +
    '</div>' +
    '<div class="pp-filter">' +
      '<div class="pp-ftop"><span class="pp-ftitle">Sort &amp; Filter</span>' +
        '<span class="pp-fnote">' + shown.length + ' prop' + (shown.length === 1 ? '' : 's') + ' shown' + (nActive ? ' &middot; ' + nActive + ' filter' + (nActive === 1 ? '' : 's') + ' on' : '') + '</span>' +
        '<div class="pp-fbtns">' + (state.propPresetMsg ? '<span class="pp-saved">' + esc(state.propPresetMsg) + '</span>' : '') +
          '<button class="pp-btn" onclick="window.__cfbPropReset()">Reset all</button>' +
          '<button class="pp-btn is-primary" title="Open the Props tab with these filters and this sort every time, in this browser" onclick="window.__cfbPropSavePreset()">Save preset</button>' +
        '</div></div>' +
      '<div class="pp-frow">' +
        '<div class="pp-fgroup"><span class="pp-flabel">Sort by</span><div style="display:flex;gap:6px">' + sortSel('propSort', sort1) +
          '<button class="pp-btn" title="Flip the direction" aria-label="Flip sort direction" data-prop-flip="1">' + (state.propSortRev ? '&uarr;' : '&darr;') + '</button></div></div>' +
        '<div class="pp-fgroup"><span class="pp-flabel">Then by</span>' + sortSel('propSort2', sort2) + '</div>' +
        '<div class="pp-fgroup"><span class="pp-flabel">Player or team</span><input id="pp-search" class="pp-search" type="search" placeholder="e.g. Jeanty, BSU" autocomplete="off" value="' + esc(state.propQ || '') + '" oninput="window.__cfbPropQ(this.value)"></div>' +
      '</div>' +
      '<div class="pp-frow"><div class="pp-fgroup"><span class="pp-flabel">Market</span><div class="pp-chips" style="margin:0">' +
        '<button class="pp-chip' + (mkt === 'ALL' ? ' is-on' : '') + '" data-prop-market="ALL">All markets<span class="pp-cn">' + nAllMkt + '</span></button>' +
        mktOrder.map(function (m) {
          var n = mktCounts[m] || 0;
          return '<button class="pp-chip' + (m === mkt ? ' is-on' : '') + (n ? '' : ' is-dim') + '" data-prop-market="' + esc(m) + '"><span class="pp-dot is-' + mktCat(m) + '"></span>' + esc(marketName(m)) + '<span class="pp-cn">' + n + '</span></button>';
        }).join('') +
      '</div></div></div>' +
      '<div class="pp-frow">' +
        '<div class="pp-fgroup"><span class="pp-flabel">Direction</span><div class="pp-seg" role="group" aria-label="Direction">' +
          segOpt('propDir', 'all', F.dir, 'All') +
          segOpt('propDir', 'over', F.dir, 'Overs', countBy('dir', function (r) { return r.model_lean === 'over'; })) +
          segOpt('propDir', 'under', F.dir, 'Unders', countBy('dir', function (r) { return r.model_lean === 'under'; })) +
        '</div></div>' +
        '<div class="pp-fgroup"><span class="pp-flabel">Min edge</span><div class="pp-seg" role="group" aria-label="Minimum edge">' +
          edgeOpt(0, 'Any') + edgeOpt(10, '10%+') + edgeOpt(15, '15%+') + edgeOpt(20, '20%+') + edgeOpt(30, '30%+') +
        '</div></div>' +
        '<div class="pp-fgroup"><span class="pp-flabel">Matchup grade</span><div class="pp-seg" role="group" aria-label="Minimum matchup grade">' +
          [[0, 'Any'], [60, 'B or better'], [80, 'A or better']].map(function (o) {
            return segOpt('propMinMu', String(o[0]), String(F.minMu), o[1], countBy('mu', function (r) { return !o[0] || (r.mu_score != null && r.mu_score >= o[0]); }));
          }).join('') +
        '</div></div>' +
        '<div class="pp-fgroup"><span class="pp-flabel">Sportsbook</span><select class="pp-select" data-prop-select="propBook" aria-label="Sportsbook">' +
          '<option value="ALL"' + (bookSel === 'ALL' ? ' selected' : '') + '>All books (best price)</option>' +
          books.map(function (b) { return '<option value="' + esc(b) + '"' + (bookSel === b ? ' selected' : '') + '>' + esc(bookLabel(b)) + ' (' + Object.keys(bookKeys[b]).length + ')</option>'; }).join('') +
        '</select></div>' +
        '<div class="pp-fgroup"><span class="pp-flabel">Game</span><select class="pp-select" aria-label="Game" onchange="window.__cfbPropGame(this.value)">' +
          '<option value="ALL"' + (gameSel === 'ALL' ? ' selected' : '') + '>All games (' + gameIds.length + ')</option>' +
          gameIds.map(function (id) { return '<option value="' + esc(id) + '"' + (gameSel === id ? ' selected' : '') + '>' + esc(games[id].label) + ', ' + esc(propKickoff(games[id].t)) + '</option>'; }).join('') +
        '</select></div>' +
        '<div class="pp-fgroup"><span class="pp-flabel">&nbsp;</span><button class="pp-toggle' + (F.mine ? ' is-on' : '') + '" aria-pressed="' + (F.mine ? 'true' : 'false') + '" data-prop-set="propMyList" data-prop-val="' + (F.mine ? '' : '1') + '">&#9733; My list only <i></i></button></div>' +
      '</div>' +
    '</div>';

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
    if (sort1 !== 'kickoff') {
      // Any sort other than kickoff: one flat list in that order.
      var sname = SORTS.filter(function (x) { return x[0] === sort1; })[0][1];
      if (shown.length) groups.push({ key: 'sorted|' + view + '|' + sort1, name: 'Sorted by ' + sname + (state.propSortRev ? ' (reversed)' : ''), note: '', cap: 60, rows: shown.slice().sort(cmp) });
    } else if (view === 'all') {
      [['official', 'Official plays', 'Receiving yards overs with a 15% to 30% edge on a line of 25.5+, rush yards unders with any edge against a tough run defense, and receptions unders with a 30%+ edge. The only props the rules call bets.', 0],
       ['underlean', 'Under leans', 'Receptions unders with a 10% to 30% edge. Winners in all three seasons tested, by a thin margin. Graded on their own line.', 12],
       ['watch', 'Watch', 'Receptions overs with a 10%+ edge, or a 10%+ edge on a secondary-role player. Tracked, not bet.', 12],
       ['tracked', 'Auto-tracked', 'Has an edge and its kind of play is up in the backtest. Logged and graded automatically.', 12],
       ['lean', 'Other leans', 'Everything else the model has a read on.', 20]].forEach(function (sec) {
        var list = shown.filter(function (r) { return r._sec === sec[0]; }).sort(cmp);
        if (list.length) groups.push({ key: 'all|' + sec[0], name: sec[1], note: sec[2], cap: sec[3], rows: list });
      });
    } else {
      var byDay = {};
      shown.forEach(function (r) { var k = dayKey(r.start_time); (byDay[k] = byDay[k] || []).push(r); });
      Object.keys(byDay).sort(function (a, b) { return rev * (a < b ? -1 : a > b ? 1 : 0); }).forEach(function (k) {
        var list = byDay[k].sort(cmp);
        groups.push({ key: view + '|' + k, name: dayLabel(list[0].start_time), note: '', cap: view === 'official' || view === 'underlean' ? 0 : 30, rows: list });
      });
    }
    var ordered = [];
    groups.forEach(function (g) { g.start = ordered.length; ordered = ordered.concat(g.rows); });
    window.__cfbPropRows = ordered;

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
      if (r.mu_grade) cells.push(muDetail(r));
      cells.push(whatIfCell(r));
      // 11/2026: jump to this game's breakdown on the Edge Board.
      var gi = gameIndexForProp(r);
      var go = gi < 0 ? '' : '<div class="pp-dcell is-wide pp-gotobd"><button type="button" class="pp-gobtn" onclick="window.__cfbGoBreakdown(' + gi + ', ' + JSON.stringify(keyOf(r)).replace(/"/g, '&quot;') + ')">' +
        'Game breakdown: ' + esc(schoolAbbr(r.team) || '') + ' vs ' + esc(schoolAbbr(r.opponent) || '') + ' &rarr;</button>' +
        '<span class="pp-sub">Matchup, injuries, and why each play in this game is a play.</span></div>';
      return '<div class="pp-detail">' + go + cells.join('') + '</div>';
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
    // Matchup grade (10/2026, src/features/defense_grades.py): how the
    // opposing defense rates against what this pick needs. A = the defense
    // helps the pick (tough D for an under, soft D for an over). Context
    // only -- not part of the edge or the official rules.
    var ordinal = function (n) { var s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
    var muCls = function (g) { return 'is-' + String(g || 'c').charAt(0).toLowerCase(); };
    var muTitle = function (r) {
      return (r.mu_def === 'tough' ? 'Tough' : r.mu_def === 'soft' ? 'Soft' : 'Average') + ' defense vs ' + r.mu_vs + ': ' + ordinal(r.mu_rank) + ' of ' + r.mu_of +
        ' (1st = toughest). Grade ' + r.mu_grade + ' for this ' + (r.model_lean === 'over' ? 'over' : 'under') + ' (' + r.mu_score + '/100).';
    };
    var muCell = function (r) {
      if (!r.mu_grade) return '<span class="pp-track is-none">&mdash;</span>';
      return '<span class="pp-mu ' + muCls(r.mu_grade) + '" title="' + esc(muTitle(r)) + '">' + esc(r.mu_grade) + '</span>' +
        '<div class="pp-sub">D ' + ordinal(r.mu_rank) + ' vs ' + esc(r.mu_vs) + '</div>';
    };
    var muDetail = function (r) {
      if (!r.mu_grade) return '';
      var bits = [];
      Object.keys(r.mu_allowed || {}).forEach(function (k) {
        var v = r.mu_allowed[k];
        bits.push('<span>Allows <b style="color:' + (v > 0 ? 'var(--salmon)' : v < 0 ? 'var(--green)' : 'var(--muted)') + '">' + (v > 0 ? '+' : '') + v + '%</b> ' + esc(k) + ' vs usual</span>');
      });
      Object.keys(r.mu_pff || {}).forEach(function (k) { bits.push('<span>PFF ' + esc(k) + ' <b>' + Number(r.mu_pff[k]).toFixed(1) + '</b></span>'); });
      return '<div class="pp-dcell is-wide"><div class="pp-dlabel">Matchup grade</div>' +
        '<div class="pp-dval"><span class="pp-mu ' + muCls(r.mu_grade) + '">' + esc(r.mu_grade) + '</span> &nbsp;' + esc(muTitle(r)) + '</div>' +
        (bits.length ? '<div class="pp-mu-list">' + bits.join('') + '</div>' : '') +
        '<div class="pp-sub">Context, not part of the edge. "vs usual" = against what the same offenses get in their other games.</div></div>';
    };
    // Hit-chance ring: the model's chance the pick wins; the tick marks what
    // the price needs to break even.
    var ppRing = function (p, be) {
      if (p == null) return '<span class="pp-track is-none">&mdash;</span>';
      var c = 2 * Math.PI * 16, pct = Math.max(0, Math.min(1, p)), good = be == null || p > be;
      var tick = be != null ? (function () { var a = (be * 360 - 90) * Math.PI / 180; return '<line x1="' + (20 + 12 * Math.cos(a)).toFixed(1) + '" y1="' + (20 + 12 * Math.sin(a)).toFixed(1) + '" x2="' + (20 + 20 * Math.cos(a)).toFixed(1) + '" y2="' + (20 + 20 * Math.sin(a)).toFixed(1) + '" stroke="#F1F5F2" stroke-width="1.6"/>'; })() : '';
      return '<span class="pp-ring' + (good ? ' is-good' : '') + '"><svg viewBox="0 0 40 40" width="40" height="40"><circle cx="20" cy="20" r="16" fill="none" stroke="#1E2924" stroke-width="3.5"/>' +
        '<circle cx="20" cy="20" r="16" fill="none" stroke-width="3.5" stroke-linecap="round" class="v" stroke-dasharray="' + (c * pct).toFixed(1) + ' ' + c.toFixed(1) + '" transform="rotate(-90 20 20)"/>' + tick +
        '</svg><b>' + Math.round(pct * 100) + '</b></span>';
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
      // 11/2026: which model scored a receiving prop. "New model" = the
      // receptions / rec yds model retrained Nov 2 with games a player had no
      // catch; "Fallback" = the box-score backup used when PFF is down.
      if (isRecvModel(r) && r.model_ev != null) flags.push(r.model_version
        ? '<span class="pp-tag is-newmodel" title="Scored by the ' + esc(r.model_version) + ' model: trained on, and averaging in, games where he played with no catch.">New model</span>'
        : '<span class="pp-tag is-inj" title="Scored by the box-score backup model (PFF unavailable this run), not the retrained model.">Fallback model</span>');
      // 11/2026: a regular rusher on his team is out, so this rush prop was
      // scored by the injury-aware model (his share without that teammate).
      if (r.injury_adjusted && r.injury_adjusted.length) flags.push('<span class="pp-tag is-newmodel" title="' + esc(r.injury_adjusted.join(', ')) +
        ' out: scored by the injury-aware rush yds model, using his share of the carries in the games without them.">Injury-adjusted</span>');
      if (r.play_kind === 'rec_under') flags.push('<span class="pp-tag is-official" title="A receptions under with a 30%+ edge. An official play.">Under play</span>');
      else if (r.play_kind === 'rec_yds_over') flags.push('<span class="pp-tag is-official" title="A receiving yards over with a 15% to 30% edge on a line of 25.5+. An official play.">Rec yds play</span>');
      else if (r.play_kind === 'rec_yds_watch') flags.push('<span class="pp-tag is-watch" title="A receiving yards over with a 15%+ edge, but ' + esc(r.rec_yds_watch_reason || 'outside the rule') + ' -- official only at 15% to 30% on a line of 25.5+ (a player with a real role). Watch only.">Rec yds watch &middot; ' + esc(r.rec_yds_watch_reason || '') + '</span>');
      else if (r.play_kind === 'rush_under') flags.push('<span class="pp-tag is-official" title="A rush yards under with an edge, against a tough run defense (top third). An official play.">Rush under play</span>');
      else if (r.play_kind === 'rush_under_watch') flags.push('<span class="pp-tag is-watch" title="A rush yards under with an edge, but the run defense is ' + esc(r.run_def || 'not graded') + ' -- official only vs a tough run defense (top third). Watch only.">Rush under watch &middot; ' + esc(r.run_def || 'no grade') + ' run D</span>');
      else if (r.play_kind === 'rec_over' && !r.is_official_play && !r.edge_capped) flags.push('<span class="pp-tag is-watch" title="A receptions over with a 10% to 30% edge. Watch only since 10/23/2026.">Rec over watch</span>');
      else if (r.is_under_lean) flags.push('<span class="pp-tag is-watch" title="A receptions under with a 10% to 30% edge. A lean, graded on its own line.">Under lean</span>');
      var mu = matchupOf(r);
      if (mu) flags.push('<span class="pp-tag ' + mu.cls + '" title="' + esc(mu.title) + '">' + mu.text + '</span>');
      if (r.injury_status) flags.push('<span class="pp-tag is-inj" title="' + esc(r.injury_detail || 'On the injury report') + '">' + esc(r.injury_status) + '</span>');
      if (r.edge_capped) flags.push('<span class="pp-tag is-inj" title="A receptions over with a 30%+ edge. Those lost in the backtest, so it is watched, not bet.">30%+ edge</span>');
      else if (!r.is_official_play && r._conf.tier === 'CHECK') flags.push('<span class="pp-tag is-inj" title="The model is far from the market. Check for an injury or a role change first.">Check</span>');
      if (r._conf.tier === 'PASS') flags.push('<span class="pp-tag is-pass" title="The price needs more than the model gives, or the player is out.">Pass</span>');
      if (started) flags.push('<span class="pp-tag is-lean" title="This game has kicked off.">Started</span>');
      var ev = r.model_ev, evHeat = ev != null && ev > 0 ? Math.min(0.5, 0.1 + ev / 80) : 0;
      return '<div class="pp-grid pp-row' + (isOpen ? ' is-open' : '') + (r.is_official_play ? ' is-official' : '') + '">' +
        '<div class="pp-who">' + (r.team ? trkMark(schoolAbbr(r.team), 30) : '') + '<div><div class="pp-name"><button type="button" class="pp-namebtn"' + pcOpenAttr(r) + 'title="Open the player card">' + esc(r.player_name) + '</button></div>' +
          '<div class="pp-sub">' + esc(schoolAbbr(r.team) || '') + (r.opponent ? ' vs ' + esc(schoolAbbr(r.opponent)) : '') + ' &middot; ' + esc(withDay ? propKickoff(r.start_time) : kickTime(r.start_time)) + '</div></div></div>' +
        '<div title="' + esc(marketName(r.market_name)) + '"><div class="pp-pick">' + mktBadge(r.market_name) + '</div>' +
          '<div class="pp-ou">' + (isOver ? 'Over' : 'Under') + ' ' + esc(r.line) + '</div></div>' +
        '<div class="num pp-proj" style="color:' + (isOver ? 'var(--neon)' : '#7FD8FF') + '">' + (r.model_predicted_value != null ? r.model_predicted_value.toFixed(1) : '&mdash;') + '</div>' +
        '<div class="num"><span class="pp-evpill" style="' + (evHeat ? 'background:rgba(57,240,122,' + evHeat.toFixed(2) + ');' : '') + 'color:' + (ev != null ? (ev > 0 ? '#EFFFF5' : tone(ev)) : 'var(--muted-3)') + '">' + (ev != null ? signed(ev, 1) + '%' : '&mdash;') + '</span></div>' +
        '<div title="' + (r._lp != null ? 'Model hit chance ' + (r._lp * 100).toFixed(1) + '%; the price needs ' + (r._be != null ? (r._be * 100).toFixed(1) : '?') + '%' : '') + '">' + ppRing(r._lp, r._be) + '</div>' +
        '<div class="num"><span class="pp-odds">' + priceTxt(r._price) + '</span><div class="pp-sub">' + (r.book_used ? esc(bookLabel(r.book_used)) : '') + '</div></div>' +
        '<div>' + muCell(r) + '</div>' +
        '<div>' + track + '</div>' +
        '<div class="pp-flags">' + flags.join('') + '</div>' +
        '<div class="pp-actions">' +
          '<div class="pp-grade" role="group" aria-label="Grade this play">' + ['win', 'loss', 'push'].map(function (st) {
            var on = myStatus[propTrackLabel(r)] === st;
            return '<button class="pp-gbtn is-' + st + (on ? ' is-on' : '') + '" title="' + (on ? 'Clear the grade' : 'Mark as a ' + st + ' (adds it to your list)') + '" ' +
              'aria-pressed="' + on + '" onclick="window.__cfbPropGrade(' + i + ',\\'' + st + '\\')">' + st.charAt(0).toUpperCase() + '</button>';
          }).join('') + '</div>' +
          '<button class="pp-trk" aria-expanded="' + (isOpen ? 'true' : 'false') + '" onclick="window.__cfbPropDetail(' + i + ')">' + (isOpen ? 'Hide' : 'Details') + '</button>' +
          (myTrk[propTrackLabel(r)]
            ? '<span class="pp-inlist" title="Already in your own list in My Tracker">&#10003; Listed</span>'
            : '<button class="pp-trk" title="Add this play to your own list in My Tracker" onclick="window.__cfbTrackProp(' + i + ')">+ List</button>') +
        '</div>' +
      '</div>' + (isOpen ? detailHtml(r) : '');
    };
    // Long groups start folded to their strongest plays so the page stays
    // short; one click opens them.
    var open = state.propOpen || {};
    var body = groups.map(function (g) {
      var list = g.rows.map(function (r, j) { return rowHtml(r, g.start + j, view === 'all' || sort1 !== 'kickoff'); });
      var folded = g.cap && list.length > g.cap && !open[g.key];
      return '<div class="pp-group"><span class="pp-group-name">' + esc(g.name) + '</span><span class="pp-group-count">' + list.length + '</span>' +
        (g.note ? '<span class="pp-group-note">' + g.note + '</span>' : '') + '</div>' + (folded ? list.slice(0, g.cap) : list).join('') +
        (g.cap && list.length > g.cap ? '<div class="pp-more"><button class="pp-link" data-prop-fold="' + esc(g.key) + '">' +
          (folded ? 'Show all ' + list.length : 'Show top ' + g.cap + ' only') + '</button></div>' : '');
    }).join('');
    var emptyMsg = nActive ? 'No plays match these filters. Loosen one, or hit Reset all.'
      : view === 'official' ? 'No official plays right now. Under leans, Watch and All props show everything else the model has a read on.'
      : view === 'underlean' ? 'No under leans right now. They are receptions unders with a 10% to 30% edge.'
      : 'No plays match these filters. Loosen one, or hit Reset all.';
    var table = '<div class="pp-table">' +
      '<div class="pp-grid pp-head">' + [['player', 'Player / matchup'], ['market', 'Prop'], [null, 'Proj'], ['edge', 'Edge'], ['hit', 'Hit'], [null, 'Odds'], ['matchup', 'Matchup'], ['roi', 'Track record']].map(function (c, j) {
        var al = (j >= 2 && j <= 5) ? ' class="num"' : '';
        if (!c[0]) return '<div' + al + '>' + c[1] + '</div>';
        var on = sort1 === c[0];
        return '<div' + al + '><button class="pp-sorth' + (on ? ' is-on' : '') + '" data-prop-sorth="' + c[0] + '" title="Sort by ' + c[1] + '">' + c[1] + ' <span class="pp-arr">' + (on ? (state.propSortRev ? '&#9650;' : '&#9660;') : '&#8597;') + '</span></button></div>';
      }).join('') + '<div>Heads up</div><div></div></div>' +
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
    // Record by matchup grade (11/3/2026): each rule's graded plays split by
    // the grade the pick had when it was first flagged, games from Nov 3 on.
    var muPanel = '';
    if (T && T.byMatchup) {
      var MU_ROWS = [['rec_yds_play', 'Rec yds overs', true], ['rush_under_play', 'Rush yds unders', true], ['under_play', 'Receptions unders', true],
                     ['official', 'Rec overs (watch)', false], ['under_lean', 'Under leans', false], ['group', 'All auto-tracked', false]];
      var cell = function (o) {
        if (!o || (!o.graded && !o.pending)) return '<div class="mu-c is-empty">&mdash;</div>';
        if (!o.graded) return '<div class="mu-c is-empty">' + o.pending + ' pending</div>';
        return '<div class="mu-c"><b>' + esc(o.record) + '</b><span style="color:' + tone(o.units) + '">' + signed(o.units, 1) + 'u</span>' +
          (o.pending ? '<em>+' + o.pending + ' pending</em>' : '') + '</div>';
      };
      var since = T.byMatchupSince ? new Date(T.byMatchupSince) : null;
      muPanel = '<div class="pp-panel mu-panel"><h3>Record by matchup grade</h3>' +
        '<div class="pp-panel-note">Graded plays from games ' + (since && !isNaN(since) ? 'starting ' + (since.getMonth() + 1) + '/' + since.getDate() : 'this week') +
          ' on, split by the matchup grade each pick had when it was first flagged (A = the defense helps the pick). 1 unit a play.</div>' +
        '<div class="mu-grid mu-head"><div>Rule</div>' + ['A', 'B', 'C', 'D', 'F'].map(function (g) { return '<div><span class="pp-mu is-' + g.toLowerCase() + '">' + g + '</span></div>'; }).join('') + '</div>' +
        MU_ROWS.map(function (r) {
          var o = T.byMatchup[r[0]] || {};
          return '<div class="mu-grid mu-row"><div class="mu-name">' + esc(r[1]) + (r[2] ? ' <span class="pp-tag is-official">Official</span>' : '') + '</div>' +
            ['A', 'B', 'C', 'D', 'F'].map(function (g) { return cell(o[g]); }).join('') + '</div>';
        }).join('') + '</div>';
    }
    if (bucketPanel || resultsPanel) lower = '<div class="pp-lower">' + bucketPanel + resultsPanel + '</div>';
    lower = muPanel + lower;

    // 5) The fine print, folded away until wanted.
    var help = '<details class="pp-help"><summary>How to read this page</summary><div class="pp-help-body">' +
      '<p><b>Official</b> plays are the only props the rules call bets: receiving yards overs with a 15% to 30% edge on a line of 25.5+ (a player with a real role), rush yards unders with any edge against a tough run defense, and receptions unders with a 30%+ edge, at the best price across books. Receptions overs moved to Watch on 10/23/2026. ' +
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
      '<p><b>Sort &amp; Filter</b> works in every view. Kickoff order groups plays by day; any other sort lists them in one run (click a column header to sort by it, again to flip it). ' +
        '<b>Sportsbook</b> shows each prop at that book&rsquo;s line, so the edge is the one you can actually bet there. The colored tag on each pick is the kind of prop: teal receiving, orange rushing, purple passing. ' +
        '<b>Save preset</b> opens the tab with your filters and sort every time, in this browser.</p>' +
      '<p><b>Matchup</b> grades the opposing defense against what the pick needs, A+ to F: A means the defense helps the pick (a tough defense for an under, a soft one for an over). ' +
        'Receiving props are graded against the player&rsquo;s position (catches and yards allowed to WRs, TEs or RBs, plus coverage), rushing props against the run (rush yards allowed, run defense, tackling), passing props against the pass (pass yards allowed, coverage, pass rush). ' +
        '&ldquo;D 12th vs RBs&rdquo; is the defense&rsquo;s national rank, 1st being the toughest. Attempts props get no grade. ' +
        'It is context, not part of the edge: the model already counts yards allowed, and only tough-defense receptions unders have held up in the backtest so far.</p>' +
      '<p><b>+ My list</b> adds any play to your own list in My Tracker. Kickoff times are in your own time zone.</p>' +
    '</div></details>';

    // ---- 11/2026 reskin: hero, category tabs, stat cards, sidebar ----
    var R = D.propRules || {}, tot = R.official_total;
    var winPct = function (rec) { var p = String(rec || '').split('-'); var w = +p[0] || 0, l = +p[1] || 0; return w + l ? (w / (w + l) * 100).toFixed(1) + '%' : '\u2014'; };
    var hero = '<div class="pp-hero"><div><h1>Player Props</h1><div class="pp-hero-sub">Model-driven props with an edge across the key markets</div></div>' +
      (tot ? '<div class="pp-hero-card"><div><div class="pp-hk">Backtest &middot; official rules</div><div class="pp-hv">' + esc(tot.record) + '</div></div>' +
        '<div><div class="pp-hk">Win %</div><div class="pp-hv is-neon">' + winPct(tot.record) + '</div></div>' +
        '<div><div class="pp-hk">Units</div><div class="pp-hv is-neon">' + signed(tot.units, 1) + 'u</div></div></div>' : '') + '</div>';
    var CATS = [['all', 'All props', 'props'], ['pass', 'Passing', 'target'], ['rush', 'Rushing', 'trend'], ['rec', 'Receiving', 'bars']];
    var catCount = function (c) { return countBy('cat', function (r) { return c === 'all' || mktCat(r.market_name) === c; }); };
    var cur = state.propCat || 'all';
    var cats = '<div class="pp-cats">' + CATS.map(function (c) {
      return '<button class="pp-cat' + (cur === c[0] ? ' is-on' : '') + '" data-prop-set="propCat" data-prop-val="' + c[0] + '">' + icon(c[2]) + '<span>' + c[1] + '</span><em>' + catCount(c[0]) + '</em></button>';
    }).join('') + '<button class="pp-cat' + (cur === 'td' ? ' is-on' : '') + '" data-prop-set="propCat" data-prop-val="td">' + icon('anytime') + '<span>Anytime TD</span><em>' + (D.anytimeTd || []).filter(function (a) { return a.model_prob != null; }).length + '</em></button></div>';
    var off = rows.filter(function (r) { return r.is_official_play && r.model_ev != null; });
    var avg = function (xs) { return xs.length ? xs.reduce(function (a, b) { return a + b; }, 0) / xs.length : null; };
    var avgEv = avg(off.map(function (r) { return r.model_ev; })), avgHit = avg(off.filter(function (r) { return r._lp != null; }).map(function (r) { return r._lp * 100; }));
    var res = (T && T.results) || [], last = res.slice(0, 30), lw = last.filter(function (p) { return p.result === 'W'; }).length, ll = last.filter(function (p) { return p.result === 'L'; }).length;
    var cum = [], t = 0; res.slice(0, 60).reverse().forEach(function (p) { t += Number(p.units || 0); cum.push(t); });
    var cards = statCards([
      ['Model plays', String(off.length), 'official plays on the board right now', { icon: 'target', neon: true }],
      ['Avg edge', avgEv != null ? signed(avgEv, 1) + '%' : '\u2014', 'mean model edge, official plays', { icon: 'bars', neon: avgEv > 0 }],
      ['Avg hit chance', avgHit != null ? avgHit.toFixed(1) + '%' : '\u2014', 'model chance the pick wins, official plays', { icon: 'check' }],
      ['Hit rate (L30)', lw + ll ? (lw / (lw + ll) * 100).toFixed(1) + '%' : '\u2014', 'last ' + (lw + ll) + ' graded auto-tracked plays', { icon: 'trend', neon: lw > ll, spark: cum }]
    ]);
    // Official record (11/4/2026): only the plays that meet the current
    // official rules, graded automatically from the box score, 1u each at
    // the price they were flagged at. Your own W / L / P grades on other
    // props stay in My Tracker and do not count here.
    var myPanel = '';
    if (T) {
      var OP = T.officialPlays || [];
      var RL = { rec_yds_play: 'Rec yds overs', rush_under_play: 'Rush yds unders', under_play: 'Receptions unders' };
      var NEWK = { rec_yds_play: T.recYdsPlayNew, rush_under_play: T.rushUnderPlayNew, under_play: T.underPlayNew };
      var lineO = function (label, o) {
        var g = o && o.graded;
        return '<div class="ep-row"><span>' + esc(label) + '</span><b>' + (g ? esc(o.record) : '&mdash;') + '</b><b style="color:' + tone(g ? o.units : 0) + '">' + (g ? signed(o.units, 2) + 'u' : '&mdash;') + '</b>' +
          '<b style="color:' + tone(g ? o.roi : 0) + '">' + (g && o.roi != null ? signed(o.roi, 1) + '%' : '&mdash;') + '</b></div>';
      };
      var nPend = OP.filter(function (x) { return !x.result; }).length;
      var dOf = function (x) { var d = new Date(x.start_time || ''); return isNaN(d) ? '' : (d.getMonth() + 1) + '/' + d.getDate() + ' '; };
      var gr = OP.filter(function (x) { return x.result; }), pe = OP.filter(function (x) { return !x.result; }).reverse();
      OP = gr.concat(pe);
      var listO = OP.slice(0, 16).map(function (x) {
        var rc = x.result === 'W' ? 'var(--neon)' : (x.result === 'L' ? 'var(--red)' : (x.result === 'P' ? 'var(--amber)' : 'var(--muted-3)'));
        var lab = x.result === 'W' ? 'WIN' : x.result === 'L' ? 'LOSS' : x.result === 'P' ? 'PUSH' : 'PENDING';
        return '<div class="my-res"><span title="' + esc(RL[x.tag] || '') + '"><i class="my-d">' + dOf(x) + '</i>' + esc(schoolAbbr(x.team) || '') + ' ' + esc(x.player) + ' &middot; ' + esc(marketName(x.market)) + ' ' + (x.side === 'over' ? 'O' : 'U') + ' ' + esc(x.line) +
          (x.actual != null && x.result ? ' <em>(' + Number(x.actual) + ')</em>' : '') + '</span>' +
          '<b style="color:' + rc + '">' + lab + '</b><b style="color:' + tone(x.units || 0) + '">' + (x.result ? signed(x.units || 0, 2) + 'u' : '') + '</b></div>';
      }).join('');
      myPanel = '<div class="pp-panel my-props"><h3>Official record</h3>' +
        '<div class="pp-panel-note">Only plays that meet the current official rules, graded automatically from the box score, 1 unit each at the price they were flagged at. ' +
          'Records start when each rule last changed (rec yds overs and receptions unders Nov 2, rush yds unders Nov 3). Your W / L / P on other props stays in My Tracker and does not count here.</div>' +
        '<div class="my-props-grid"><div>' +
          '<div class="ep-row ep-th"><span></span><span>Record</span><span>Units</span><span>ROI</span></div>' +
          lineO('All official plays', T.officialNow) +
          ['rec_yds_play', 'rush_under_play', 'under_play'].map(function (k) { return lineO(RL[k], NEWK[k]); }).join('') +
          '<div class="ep-k2" style="margin-top:8px">' + OP.length + ' official plays logged &middot; ' + nPend + ' still to be graded</div>' +
        '</div><div>' + (listO || '<div class="ep-empty">No official plays logged yet under the current rules.</div>') +
          (OP.length > 16 ? '<div class="ep-k2" style="margin-top:6px">Showing 16 of ' + OP.length + ' (graded first, then upcoming).</div>' : '') + '</div></div></div>';
    }
    if (cur === 'td') return hero + cats + renderPropsTd() + help;
    return hero + cats + cards +
      '<div class="pp-layout"><aside class="pp-side' + (state.ppFilters ? ' is-filters' : '') + '">' +
        '<button class="pp-btn pp-ftoggle" style="width:100%;justify-content:center;margin-bottom:10px" onclick="window.__cfbPpFilters()">' + (state.ppFilters ? 'Hide' : 'Show') + ' sort &amp; filter' + (nActive ? ' (' + nActive + ' on)' : '') + '</button>' +
        controls + strip + '</aside>' +
      '<div class="pp-main"><div class="pp-main-head"><span>Model plays</span><em>' + shown.length + '</em></div>' + table + '</div></div>' +
      lower + myPanel + help;
  }


  // ---- Anytime TD on the Player Props page (11/4/2026) -----------------
  // Every anytime TD line with the model's chance, the price at DraftKings
  // and the best price, and where the opposing defense ranks in TDs allowed
  // to the player's position (1 = gives up the most). Watch only: the
  // 10/24 backtest found no edge after the books' margin at a legal book.
  function renderPropsTd() {
    var all = (D.anytimeTd || []).filter(function (a) { return a.model_prob != null; });
    var TA = D.tdAllowed || [];
    if (!all.length) return '<div class="empty-state">No anytime TD lines with a model read right now' + (D.anytimeTdNote ? ' (' + esc(D.anytimeTdNote) + ')' : '') + '.</div>';
    var POS = ['qb', 'rb', 'wr', 'te'];
    // Defense ranks: TDs allowed per game to each position, and overall. Rank 1 = most allowed.
    var def = {}, nDef = TA.length;
    var rankBy = function (key) {
      var srt = TA.slice().sort(function (x, y) { return (y[key] || 0) - (x[key] || 0); });
      srt.forEach(function (r, i) { (def[r.team] = def[r.team] || { row: r })[key] = i + 1; });
    };
    POS.forEach(function (g) { rankBy('td_' + g + '_pg'); });
    rankBy('td_all_pg');
    var posOf = function (a) {
      var p = String(a.pos || '').toLowerCase();
      if (POS.indexOf(p) >= 0) return p;
      var r = String(a.role || '');            // no roster position: the model's role label
      return /\bQB\b/.test(r) ? 'qb' : (/\bRB\b/.test(r) ? 'rb' : null);
    };
    var tdPos = state.tdPos || 'all', tdSort = state.tdSort || 'dk';
    var rows = all.filter(function (a) { return tdPos === 'all' || posOf(a) === tdPos; }).map(function (a) {
      var d = def[a.opponent], g = posOf(a);
      a._dr = d ? (g ? d['td_' + g + '_pg'] : d.td_all_pg) : null;
      a._dpg = d ? (g ? d.row['td_' + g + '_pg'] : d.row.td_all_pg) : null;
      a._dall = d ? d.td_all_pg : null;
      return a;
    });
    var num = function (x) { return x == null || isNaN(x) ? -1e9 : Number(x); };
    var cmp = {
      dk: function (x, y) { return num(y.edge_dk) - num(x.edge_dk) || num(y.edge_best) - num(x.edge_best); },
      best: function (x, y) { return num(y.edge_best) - num(x.edge_best); },
      chance: function (x, y) { return num(y.model_prob) - num(x.model_prob); },
      def: function (x, y) { return (x._dr || 999) - (y._dr || 999) || num(y.model_prob) - num(x.model_prob); },
      kick: function (x, y) { return String(x.start_time) < String(y.start_time) ? -1 : 1; }
    }[tdSort] || function () { return 0; };
    rows.sort(cmp);
    var cap = state.tdAll ? rows.length : 60;
    var am = function (x) { return x == null || isNaN(x) ? '&mdash;' : (x > 0 ? '+' : '') + Math.round(x); };
    var sg = function (x) { return x == null || isNaN(x) ? '&mdash;' : (x > 0 ? '+' : '') + Number(x).toFixed(1) + '%'; };
    var tone = function (x) { return x > 0 ? 'var(--neon)' : (x < 0 ? 'var(--red)' : 'var(--muted-3)'); };
    var seg = function (field, val, cur, label) { return '<button class="' + (cur === val ? 'is-on' : '') + '" data-prop-set="' + field + '" data-prop-val="' + val + '">' + label + '</button>'; };
    var defCell = function (a) {
      if (a._dr == null) return '<span class="pp-track is-none">&mdash;</span>';
      var q = a._dr / nDef, cls = q <= 0.25 ? 'is-soft' : (q > 0.75 ? 'is-tough' : '');
      var g = posOf(a);
      return '<span class="td-def ' + cls + '" title="' + esc(a.opponent) + ' allows ' + (a._dpg != null ? a._dpg : '?') + ' TDs a game to ' + (g ? g.toUpperCase() + 's' : 'all positions') + ' (#' + a._dr + ' of ' + nDef + ', 1 = most). Overall #' + a._dall + '.">#' + a._dr + '</span>' +
        '<div class="pp-sub">' + (a._dpg != null ? a._dpg + '/g' : '') + ' to ' + (g ? g.toUpperCase() + 's' : 'all') + ' &middot; all #' + a._dall + '</div>';
    };
    var ring = function (p, mk) {
      var c = 2 * Math.PI * 16, pct = Math.max(0, Math.min(1, p || 0)), good = mk == null || p > mk;
      return '<span class="pp-ring' + (good ? ' is-good' : '') + '"><svg viewBox="0 0 40 40" width="40" height="40"><circle cx="20" cy="20" r="16" fill="none" stroke="#1E2924" stroke-width="3.5"/>' +
        '<circle cx="20" cy="20" r="16" fill="none" stroke-width="3.5" stroke-linecap="round" class="v" stroke-dasharray="' + (c * pct).toFixed(1) + ' ' + c.toFixed(1) + '" transform="rotate(-90 20 20)"/></svg><b>' + Math.round(pct * 100) + '</b></span>';
    };
    var body = rows.slice(0, cap).map(function (a) {
      var g = posOf(a);
      return '<div class="td-grid pp-row">' +
        '<div class="pp-who">' + (a.team ? trkMark(schoolAbbr(a.team), 30) : '') + '<div><div class="pp-name">' + esc(a.player_name) + '</div>' +
          '<div class="pp-sub">' + esc(schoolAbbr(a.team) || a.team || '') + ' vs ' + esc(schoolAbbr(a.opponent) || a.opponent || '') + ' &middot; ' + esc(propKickoff(a.start_time)) + '</div></div></div>' +
        '<div><span class="pp-mkt is-' + (g === 'qb' ? 'pass' : g === 'rb' ? 'rush' : 'rec') + '">' + esc(g ? g.toUpperCase() : '?') + '</span><div class="pp-sub">' + esc(a.role || '') + '</div></div>' +
        '<div title="Model chance ' + (a.model_prob * 100).toFixed(1) + '%; best price implies ' + (a.market_implied != null ? (a.market_implied * 100).toFixed(1) + '%' : '?') + '">' + ring(a.model_prob, a.market_implied) + '</div>' +
        '<div class="num pp-proj">' + am(a.fair_odds) + '</div>' +
        '<div class="num"><span class="pp-odds">' + am(a.dk_price) + '</span><div class="pp-sub" style="color:' + tone(a.edge_dk) + '">' + sg(a.edge_dk) + '</div></div>' +
        '<div class="num"><span class="pp-odds">' + am(a.best_price) + '</span><div class="pp-sub">' + esc(a.best_book_label || '') + (a.edge_best != null ? ' &middot; <span style="color:' + tone(a.edge_best) + '">' + sg(a.edge_best) + '</span>' : '') + '</div></div>' +
        '<div>' + defCell(a) + '</div>' +
      '</div>';
    }).join('');
    var head = '<div class="td-grid pp-head"><div>Player / matchup</div><div>Pos</div><div>Model</div><div class="num">Fair odds</div><div class="num">DraftKings</div><div class="num">Best price</div><div>Opp. D, TDs allowed</div></div>';
    var controls = '<div class="td-controls">' +
      '<div class="pp-seg">' + seg('tdPos', 'all', tdPos, 'All') + POS.map(function (g) { return seg('tdPos', g, tdPos, g.toUpperCase() + 's'); }).join('') + '</div>' +
      '<div class="pp-seg">' + seg('tdSort', 'dk', tdSort, 'Edge at DK') + seg('tdSort', 'best', tdSort, 'Edge, best price') + seg('tdSort', 'chance', tdSort, 'Model chance') + seg('tdSort', 'def', tdSort, 'Softest defense') + seg('tdSort', 'kick', tdSort, 'Kickoff') + '</div>' +
    '</div>';
    var note = '<div class="td-note"><b>Watch only.</b> Anytime TDs are not official plays: the Oct 24 backtest found the model prices scoring well, but not by enough to beat the books&rsquo; margin at a single legal book. ' +
      '<b>Opp. D</b> ranks every FBS defense by TDs allowed per game to that position (1 = gives up the most, green = bottom quarter of defenses, red = top quarter); &ldquo;all&rdquo; is total TDs allowed. ' +
      'The 10/31 matchup watch found that soft TD defenses are already in the price.</div>';
    return note + controls + '<div class="pp-table">' + head + body + '</div>' +
      (rows.length > cap ? '<div class="pp-more"><button class="pp-link" data-prop-set="tdAll" data-prop-val="1">Show all ' + rows.length + '</button></div>' : '');
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
    if (play.dogWatch) item.dogWatch = true;                  // big-dog watch play (28-34.5 underdog) when tracked
    if (play.betCard) item.betCard = true;                    // was on the Bet Card for its market when tracked
    if (play.winProb != null) item.winProb = play.winProb;    // model win % when tracked
    if (play.confTier) item.confTier = play.confTier;         // High / Mid / Low when tracked
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

  // Grade a prop straight from its row (11/2026): adds it to your own list
  // first if it isn't there, then sets Win / Loss / Push (click again to
  // clear). Same item and grade as the My Tracker tab.
  window.__cfbPropGrade = function (i, st) {
    var r = (window.__cfbPropRows || [])[i];
    if (!r) return;
    var desc = propTrackLabel(r);
    var items = loadTrk();
    var it = items.filter(function (x) { return x.type === 'Prop' && (x.description === desc || x.flaggedDesc === desc); })[0];
    if (!it) {
      items.unshift({
        id: 't' + Date.now() + Math.random().toString(36).slice(2, 7),
        description: desc, date: (r.start_time || '').slice(0, 10), kickoffIso: r.start_time || undefined, type: 'Prop',
        price: r.model_lean === 'over' ? r.over_price : r.under_price,
        edge: r.model_ev, stake: 50, status: st, market: r.market_name, official: !!r.is_official_play
      });
      saveTrk(items);
    } else {
      trkCommit(it.id, { status: it.status === st ? null : st });
    }
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
      var kindTag = { rec_under: 'Rec under', rec_yds_over: 'Rec yds over', rush_under: 'Rush under' }[r.play_kind];
      if (kindTag) items[0].tags = [kindTag];
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
        if (o.dogWatch) keep.dogWatch = true;
        if (!keep.notes && o.notes) keep.notes = o.notes;
        var tags = trkTagsOf(keep).slice();
        trkTagsOf(o).forEach(function (t) { if (!trkHasTag(tags, t)) tags.push(t); });
        if (tags.length) keep.tags = tags;
        ['modelSource', 'modelVersion', 'kickoffIso', 'sourceId', 'winProb', 'confTier'].forEach(function (f) { if (keep[f] == null && o[f] != null) keep[f] = o[f]; });
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
    return state.trkF[sec] || (state.trkF[sec] = { q: '', status: 'all', tag: 'all', sort: 'date_desc' });
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
      if (f.status === 'dogwatch' && !it.dogWatch) return false;
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
      (it.dogWatch ? '<span class="trk-tag is-watch" title="Big-dog watch: 28–34.5 point underdog, watch only">BIG-DOG WATCH</span>' : '') +
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
        [['all', 'All statuses'], ['pending', 'Not graded yet'], ['win', 'Wins'], ['loss', 'Losses'], ['push', 'Pushes'], ['official', 'Official only'], ['unofficial', 'Unofficial only'], ['betcard', 'Bet Card plays'], ['dogwatch', 'Big-dog watch']]
          .map(function (o) { return opt(o[0], o[1], f.status); }).join('') + '</select>' +
      '<select class="trk-sel" data-trk-filter="tag" data-trk-sec="' + sec + '" aria-label="Filter by tag">' +
        opt('all', Object.keys(tagsInUse).length ? 'All tags' : 'No tags yet', f.tag) +
        Object.keys(tagsInUse).sort().map(function (t) { return opt(t, t, f.tag); }).join('') + '</select>' +
      '<select class="trk-sel" data-trk-filter="sort" data-trk-sec="' + sec + '" aria-label="Sort plays">' +
        [['date_desc', 'Sort: latest game first'], ['added', 'Sort: newest added'], ['date_asc', 'Sort: earliest game first'], ['edge', 'Sort: biggest edge'], ['profit', 'Sort: biggest profit']]
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
    // Receiving rules' live record restarts at the Nov 2, 2026 model retrain;
    // the old model's record stays visible underneath.
    var prevNote = function (prev, now, what) {
      if (!now) return '';
      var rule = what === 'Nov 3 rule change';
      return '<div style="font-size:9.5px;color:var(--muted-3);font-weight:400;margin-top:2px">live since ' + (what || 'Nov 2 model update') +
        (prev && prev.graded ? ' &middot; ' + (rule ? 'old rule' : 'old model') + ': ' + esc(prev.record) + ', ' + sg(prev.units, 1) + 'u' : '') + '</div>';
    };
    var btLabel = function (bt) { return bt ? ' <span style="color:var(--muted-3);font-weight:400">(' + bt.seasons + ' season' + (bt.seasons === 1 ? '' : 's') + ')</span>' : ''; };
    var rules = '<div class="thead" style="display:grid;' + kc + 'margin-top:14px"><div>Official rules</div><div class="num">Record</div><div class="num">Units</div><div class="num">ROI</div><div class="num">Graded</div><div class="num">Backtest</div></div>' +
      ruleRow('Receiving yards overs, 15% to 30% edge, line 25.5+' + btLabel(R.rec_yds_over) + prevNote(T.recYdsPlayPrev, T.recYdsPlayNew), T.recYdsPlayNew || T.recYdsPlay, R.rec_yds_over) +
      ruleRow('Rush yards unders, any edge, vs a tough run defense' + btLabel(R.rush_under) + prevNote(T.rushUnderPlayPrev, T.rushUnderPlayNew, 'Nov 3 rule change'), T.rushUnderPlayNew || T.rushUnderPlay, R.rush_under) +
      ruleRow('Receptions unders, 30%+ edge' + btLabel(R.rec_under) + prevNote(T.underPlayPrev, T.underPlayNew), T.underPlayNew || T.underPlay, R.rec_under) +
      (R.official_total ? '<div class="row"><div class="row-accent" style="background:var(--rule)"></div>' +
        '<div class="row-body" style="' + kc + 'padding:8px 14px;font-size:11.5px">' +
          '<div style="font-weight:700">Official rules, backtest total</div><div></div><div></div><div></div><div></div>' +
          '<div class="num" style="font-weight:700;color:' + col(R.official_total.units) + '">' + sg(R.official_total.roi, 0) + '% &middot; ' +
            sg(R.official_total.units, 1) + 'u &middot; ' + esc(R.official_total.record) + '</div>' +
        '</div></div>' : '') +
      '<div class="thead" style="display:grid;' + kc + 'margin-top:10px"><div>Watch rules</div><div></div><div></div><div></div><div></div><div></div></div>' +
      ruleRow('Receptions overs, 10% to 30% edge' + btLabel(R.rec_over), T.official, R.rec_over) +
      ruleRow('Receptions under leans, 10% to 30% edge' + btLabel(R.rec_under_lean), T.underLean, R.rec_under_lean);
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

  // Bet Card record by confidence tier (11/2026). Weeks 4-10 come from the
  // rebuilt card (D.confBaseline -- this browser's tracker has no win % for
  // plays tracked before the tier existed); everything after that is Bet
  // Card plays tracked here with +TRK, graded, 1u each, MLs included even
  // though they're unofficial so their tiers can be watched too.
  function renderConfidenceRecord(items) {
    var B = D.confBaseline || { rows: [] };
    var live = items.filter(function (it) { return it.betCard && it.confTier && it.status && (it.type === 'Spread' || it.type === 'Moneyline'); });
    var uOf = function (it) { return it.status === 'win' ? americanToDecimal(it.price) - 1 : (it.status === 'loss' ? -1 : 0); };
    var sg = function (v, d) { return (v > 0 ? '+' : '') + v.toFixed(d); };
    var col = function (v) { return v > 0.005 ? 'var(--green)' : v < -0.005 ? 'var(--red)' : 'var(--muted)'; };
    var block = function (mkt) {
      var cuts = M.CONF_CUTS[mkt];
      var rows = ['High', 'Mid', 'Low'].map(function (t) {
        var b = B.rows.filter(function (r) { return r.market === mkt && r.tier === t; })[0] || { w: 0, l: 0, p: 0, units: 0, n: 0 };
        var lv = live.filter(function (it) { return it.type === mkt && it.confTier === t; });
        var lw = 0, ll = 0, lp = 0, lu = 0;
        lv.forEach(function (it) { if (it.status === 'win') lw++; else if (it.status === 'loss') ll++; else if (it.status === 'push') lp++; lu += uOf(it); });
        var w = b.w + lw, l = b.l + ll, p = b.p + lp, u = b.units + lu, n = b.n + lv.length;
        var roi = n ? u / n * 100 : 0;
        var range = t === 'High' ? (cuts[0] * 100).toFixed(1).replace('.0', '') + '%+' : t === 'Mid' ? (cuts[1] * 100).toFixed(1).replace('.0', '') + '–' + (cuts[0] * 100).toFixed(1).replace('.0', '') + '%' : 'under ' + (cuts[1] * 100).toFixed(1).replace('.0', '') + '%';
        return '<span>' + confChip(t) + '<span class="conf-live">' + range + '</span></span>' +
          '<span class="r"><b>' + w + '-' + l + (p ? '-' + p : '') + '</b>' + (lv.length ? '<span class="conf-live">live ' + lw + '-' + ll + (lp ? '-' + lp : '') + '</span>' : '') + '</span>' +
          '<span class="r" style="color:' + col(u) + '"><b>' + sg(u, 1) + 'u</b></span>' +
          '<span class="r" style="color:' + col(roi) + '">' + sg(roi, 1) + '%</span>';
      }).join('');
      return '<div class="conf-mkt">' + (mkt === 'Spread' ? 'Spreads (official)' : 'Moneylines (unofficial)') + '</div>' +
        '<div class="conf-grid"><span class="h">Tier</span><span class="h r">Record</span><span class="h r">Units</span><span class="h r">ROI</span>' + rows + '</div>';
    };
    return '<div class="ep-card" style="margin:10px 0 6px"><div class="ep-head">' + icon('performance') + '<span>Bet Card by confidence</span><em>1u each</em></div>' +
      block('Spread') + block('Moneyline') +
      '<div class="ep-foot" style="color:var(--muted-3);margin-top:8px">Weeks ' + esc(B.weeks || '4-10') + ' are the Bet Card rebuilt from saved runs; after that it adds the Bet Card plays you track here once they’re graded. ' +
      'The tier ranks plays by the model’s win %, which runs high (Bet Card spreads have averaged 75% shown vs 58% actual), so it’s a ranking, not a real probability.</div></div>';
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
        extraHtml: renderConfidenceRecord(gameLineItems) + modelTabBar,
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
      case 'anytime': return renderAnytime();
      case 'tdallowed': return renderTdAllowed();
      case 'fantasy': return renderFantasy();
      case 'injuries': return renderInjuries();
      case 'tracker': return renderTracker();
      case 'performance': return renderBacktest() + renderClv();
      case 'edge':
      default:
        return renderKpis(card, priced) +
          '<div class="em-wrap">' + renderEdgeBoard(priced, card) + '</div>' +
          renderEdgePanels(priced) +
          '<div class="main em-below" id="em-projector">' +
            '<div>' + renderProjector(sel) + renderGameBreakdown(sel) + renderTeamMatchup(sel) + '</div>' +
            '<div>' + renderBetCard(card) + '</div>' +
          '</div>';
    }
  }

  function render() {
    PC_REG.length = 0;
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
  window.__cfbAtdGame = function (v) { state.atdGame = v; render(); };
  window.__cfbTdaQ = function (v) {
    state.tdaQ = v; render();
    var el = document.querySelector('.tda-search');
    if (el) { el.focus(); el.setSelectionRange(v.length, v.length); }
  };
  window.__cfbPropGame = function (v) { state.propGame = v; state.propPresetMsg = ''; render(); };
  window.__cfbPropQ = function (v) { state.propQ = v; state.propPresetMsg = ''; renderKeep(); };
  window.__cfbPropReset = function () {
    Object.keys(PROP_DEFAULTS).forEach(function (k) { state[k] = PROP_DEFAULTS[k]; });
    state.propQ = ''; state.propPresetMsg = ''; render();
  };
  window.__cfbPropSavePreset = function () {
    var o = {};
    PROP_PRESET_FIELDS.forEach(function (k) { o[k] = state[k]; });
    try { localStorage.setItem(PROP_PRESET_KEY, JSON.stringify(o)); state.propPresetMsg = 'Preset saved'; }
    catch (e) { state.propPresetMsg = 'Could not save in this browser'; }
    render();
  };
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
  // 11/2026: Player Props -> this game's breakdown on the Edge Board, and back.
  window.__cfbGoBreakdown = function (gi, propKey) {
    state.selected = gi;
    state.bdFrom = (propKey || state.page === 'props') ? { key: propKey, y: window.scrollY } : null;
    state.page = 'edge';
    render();
    var el = document.querySelector('.gb-card') || document.getElementById('em-projector');
    if (el) el.scrollIntoView({ behavior: 'auto', block: 'start' });
  };
  window.__cfbBackToProps = function () {
    var from = state.bdFrom || {};
    state.page = 'props';
    if (from.key) state.propDetail = from.key;
    state.bdFrom = null;
    render();
    window.scrollTo(0, from.y || 0);
  };
  window.__cfbPropOpen = function (i) {
    var k = ['official', 'watch', 'tracked', 'lean'][i];
    state.propOpen = state.propOpen || {};
    state.propOpen[k] = !state.propOpen[k];
    render();
  };

  document.addEventListener('click', function (e) {
    var p = e.target.closest('[data-page]');
    if (p) { var changed = state.page !== p.dataset.page; state.page = p.dataset.page; state.bdFrom = null; render(); if (changed) window.scrollTo(0, 0); return; }
    var g = e.target.closest('[data-game]');
    if (g) {
      state.selected = +g.dataset.game; state.bdFrom = null; render();
      var pj = document.getElementById('em-projector');
      if (pj && g.classList.contains('em-row')) pj.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    var m = e.target.closest('[data-market]');
    if (m) { state.market = m.dataset.market; return render(); }
    if (e.target.closest('[data-edge-clear]')) { state.edgeShow = 'all'; state.edgeConf = 'ALL'; state.search = ''; return render(); }
    var pm = e.target.closest('[data-prop-market]');
    if (pm) { state.propMarket = pm.dataset.propMarket; state.propPresetMsg = ''; return render(); }
    var ps = e.target.closest('[data-prop-set]');
    if (ps) {
      var f = ps.dataset.propSet, v = ps.dataset.propVal;
      state[f] = (f === 'propMinEdge' || f === 'propMinMu') ? Number(v) || 0 : ((f === 'propMyList' || f === 'tdAll') ? !!v : v);
      state.propPresetMsg = ''; return render();
    }
    var sh = e.target.closest('[data-prop-sorth]');
    if (sh) {
      var k = sh.dataset.propSorth;
      if (state.propSort === k) state.propSortRev = !state.propSortRev;
      else { state.propSort = k; state.propSortRev = false; }
      state.propPresetMsg = ''; return render();
    }
    if (e.target.closest('[data-prop-flip]')) { state.propSortRev = !state.propSortRev; state.propPresetMsg = ''; return render(); }
    var pf = e.target.closest('[data-prop-fold]');
    if (pf) { state.propOpen = state.propOpen || {}; state.propOpen[pf.dataset.propFold] = !state.propOpen[pf.dataset.propFold]; return render(); }
    var us = e.target.closest('[data-unit-size]');
    if (us) { state.unitSize = Number(us.dataset.unitSize); return render(); }
    var mt = e.target.closest('[data-model-tab]');
    if (mt) { state.trackerModelTab = mt.dataset.modelTab; return render(); }
    var is = e.target.closest('[data-inj-scope]');
    if (is) { state.injScope = is.dataset.injScope; return render(); }
    var atd = e.target.closest('[data-atd-set]');
    if (atd && atd.dataset.atdSet === 'tdaJump') {
      e.stopPropagation();
      state.tdaView = 'rank'; state.tdaOpen = atd.dataset.atdVal; state.tdaQ = atd.dataset.atdVal;
      render(); window.scrollTo(0, 0); return;
    }
    if (atd) { state[atd.dataset.atdSet] = atd.dataset.atdVal; return render(); }
  });

  // Edge Board sort / filter dropdowns.
  document.addEventListener('change', function (e) {
    var k = e.target && e.target.dataset && e.target.dataset.edgeFilter;
    if (k === 'edgeSort' || k === 'edgeShow' || k === 'edgeConf') { state[k] = e.target.value; render(); }
    var pk = e.target && e.target.dataset && e.target.dataset.propSelect;
    if (pk === 'propSort' || pk === 'propSort2' || pk === 'propBook') {
      state[pk] = e.target.value;
      if (pk === 'propSort') state.propSortRev = false;
      state.propPresetMsg = ''; render();
    }
  });

  seedTrackerIfMissing();
  autoTrackOfficialProps();
  dedupeAutoProps();
  render();
})();
</script>
"""


def _load_conf_baseline():
    """Weeks 4-10 Bet Card record by confidence tier (rebuilt 11/10/2026).
    Static file -- the Tracker adds live tracked Bet Card plays on top."""
    try:
        with open("docs/data/bet_card_confidence.json") as f:
            b = json.load(f)
        return {"rows": b.get("rows", []), "weeks": b.get("weeks"), "through": b.get("through"), "cutoffs": b.get("cutoffs")}
    except Exception as e:
        print(f"  [note] bet card confidence baseline not loaded ({e})")
        return None


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
    # Team-vs-team national ranks for the Matchup Projector (11/2026).
    model_data["teamMatchup"] = data.get("team_matchup") or {}
    model_data["teamMatchupMeta"] = data.get("team_matchup_meta") or {}
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
