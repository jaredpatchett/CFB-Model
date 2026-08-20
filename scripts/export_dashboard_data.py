#!/usr/bin/env python3
"""
Build one consolidated, dashboard-ready JSON file from everything the
pipeline has already fetched: current game lines, current player props, and
team metadata (logos, colors, conference) from CFBD.

Written to docs/data/latest.json — deliberately OUTSIDE the gitignored
data/ directories, since this file (unlike raw pulls) is meant to be
committed to the repo so it can be read without needing GitHub API/artifact
access.

Usage:
  python scripts/export_dashboard_data.py --year 2025
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data import cfbd_client as cfbd
from src.data import prizepicks_client
from src.data.injury_overrides import load_injury_overrides, get_team_override
from src.models import fair_odds as fo


def _normalize_team_name(name: str) -> str:
    """Best-effort normalization so 'Ohio State Buckeyes' (Odds API style)
    can match 'Ohio State' (CFBD 'school' field). Strips accents (e.g. CFBD's
    'San José State' -> 'san jose state') before stripping punctuation/case,
    then relies on substring/alias matching in build_team_lookup below."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", ascii_name.lower()).strip()


# Hand-maintained aliases for cases where Odds API and CFBD naming diverge
# enough that substring matching won't catch it. Add to this as mismatches
# are found in real data. Keys are normalized Odds API names.
TEAM_ALIASES = {
    "ole miss rebels": "ole miss",
    "usc trojans": "usc",
    "smu mustangs": "smu",
    "ucf knights": "ucf",
    "lsu tigers": "lsu",
    "byu cougars": "byu",
    "tcu horned frogs": "tcu",
    "unlv rebels": "unlv",
    "umass minutemen": "massachusetts",
}


def build_team_lookup(teams_df: pd.DataFrame) -> dict:
    """normalized CFBD school name -> {logo, color, alt_color, conference,
    mascot, abbreviation}. Indexed under BOTH the school-only form ('ohio
    state') and the school+mascot form ('ohio state buckeyes'), since the
    Odds API sends team names with the mascot attached but CFBD's 'school'
    field doesn't include it. See the note on match_team/match_sp_rating
    below for why this replaced substring matching."""
    lookup = {}
    for _, row in teams_df.iterrows():
        school = row.get("school")
        if not school:
            continue
        logos = row.get("logos")
        logo_url = None
        if isinstance(logos, list) and logos:
            logo_url = logos[0]
        mascot = row.get("mascot")
        record = {
            "school": school,
            "logo": logo_url,
            "color": row.get("color"),
            "alt_color": row.get("alt_color"),
            "conference": row.get("conference"),
            "mascot": mascot,
            "abbreviation": row.get("abbreviation"),
        }
        lookup[_normalize_team_name(school)] = record
        if mascot:
            lookup[_normalize_team_name(f"{school} {mascot}")] = record
    return lookup


def match_team(name: str, lookup: dict) -> dict:
    """EXACT match only (after normalizing + applying TEAM_ALIASES) against
    a lookup already indexed under both 'school' and 'school mascot' forms.

    History: an earlier version fell back to substring matching ('key in
    norm or norm in key') when an exact match failed, on the theory that it
    would only ever catch legitimate mascot-suffix mismatches (e.g. 'Ohio
    State Buckeyes' vs CFBD's 'Ohio State'). In real data it did much worse
    than that: 'houston' (Houston Cougars, FBS) is a substring of 'houston
    baptist huskies' (a different, non-FBS school); 'georgia' is a substring
    of 'west georgia wolves' (D-II, not FBS); 'north carolina' is a
    substring of 'north carolina at aggies' (NC A&T, FCS). Substring
    matching silently attached the wrong (unrelated) school's real data to
    each of these, which for the SP+ rating case (see match_sp_rating)
    produced wildly wrong model probabilities on exactly the long-shot lines
    where it's easiest to not notice. There is no reliable way to
    distinguish a legitimate mascot suffix from an unrelated school that
    happens to share a leading word using string heuristics alone — so
    instead of guessing, this only trusts an exact match against the known
    school/mascot forms (or a hand-verified TEAM_ALIASES entry). A team not
    found under either form is reported as unmatched, which is correct: if
    it's not FBS or CFBD spells it differently, we don't actually know its
    data and shouldn't fabricate a match."""
    if not name:
        return {}
    norm = _normalize_team_name(name)
    if norm in TEAM_ALIASES:
        norm = TEAM_ALIASES[norm]
    if norm in lookup:
        return lookup[norm]
    return {"school": name, "logo": None, "color": None, "alt_color": None,
            "conference": None, "mascot": None, "unmatched": True}


def build_sp_lookup(sp_df: pd.DataFrame, team_lookup: dict = None) -> dict:
    """normalized team name -> SP+ rating, for the fair-odds/EV preseason
    estimate. Indexed under both 'school' and 'school mascot' forms (using
    team_lookup's mascot data, since CFBD's SP+ endpoint returns bare school
    names with no mascot) so exact matching works the same way as
    build_team_lookup — see match_team's docstring for why this replaced
    substring matching."""
    lookup = {}
    if sp_df is None or sp_df.empty or "team" not in sp_df.columns:
        return lookup
    for _, row in sp_df.iterrows():
        team = row.get("team")
        rating = row.get("rating")
        if not team or pd.isna(rating):
            continue
        norm_school = _normalize_team_name(team)
        lookup[norm_school] = float(rating)
        if team_lookup is not None:
            meta = team_lookup.get(norm_school)
            mascot = meta.get("mascot") if meta else None
            if mascot:
                lookup[_normalize_team_name(f"{team} {mascot}")] = float(rating)
    return lookup


def match_sp_rating(name: str, lookup: dict):
    """EXACT match only — see match_team's docstring for why."""
    if not name:
        return None
    norm = _normalize_team_name(name)
    if norm in TEAM_ALIASES:
        norm = TEAM_ALIASES[norm]
    return lookup.get(norm)


def build_sp_splits_lookup(sp_df: pd.DataFrame, team_lookup: dict = None) -> dict:
    """normalized team name -> {off, def} SP+ sub-ratings, for the Power
    Ratings table's offense/defense diverging bar. CFBD's /ratings/sp
    endpoint returns these as nested 'offense.rating'/'defense.rating'
    fields (flattened by pd.json_normalize when fetch_historical_data.py
    saved the raw CSV) — no new API call needed, just reading columns that
    were already being fetched and ignored. Same school/mascot dual-keying
    as build_sp_lookup, for the same reason."""
    lookup = {}
    if sp_df is None or sp_df.empty or "team" not in sp_df.columns:
        return lookup
    has_off = "offense.rating" in sp_df.columns
    has_def = "defense.rating" in sp_df.columns
    if not (has_off and has_def):
        return lookup
    for _, row in sp_df.iterrows():
        team = row.get("team")
        off_r, def_r = row.get("offense.rating"), row.get("defense.rating")
        if not team or pd.isna(off_r) or pd.isna(def_r):
            continue
        norm_school = _normalize_team_name(team)
        splits = {"off": float(off_r), "def": float(def_r)}
        lookup[norm_school] = splits
        if team_lookup is not None:
            meta = team_lookup.get(norm_school)
            mascot = meta.get("mascot") if meta else None
            if mascot:
                lookup[_normalize_team_name(f"{team} {mascot}")] = splits
    return lookup


def match_sp_splits(name: str, lookup: dict):
    """EXACT match only — see match_team's docstring for why."""
    if not name:
        return None
    norm = _normalize_team_name(name)
    if norm in TEAM_ALIASES:
        norm = TEAM_ALIASES[norm]
    return lookup.get(norm)


def derive_abbr(name: str, cfbd_abbr: str = None) -> str:
    """Short team tag for the helmet/power-ratings UI. Prefers CFBD's own
    'abbreviation' field (e.g. 'OSU'); falls back to a simple
    first-letters-of-each-word heuristic if CFBD didn't provide one, rather
    than leaving the UI with nothing to show."""
    if cfbd_abbr:
        return str(cfbd_abbr).upper()
    if not name:
        return "?"
    words = [w for w in re.sub(r"[^A-Za-z0-9 ]", "", name).split() if w]
    if len(words) == 1:
        return words[0][:4].upper()
    return "".join(w[0] for w in words[:4]).upper()


def build_teams_export(games_out: list, team_lookup: dict, sp_lookup: dict, sp_splits_lookup: dict) -> list:
    """One entry per distinct team that appears in games_out AND has a real
    matched SP+ rating (i.e. is actually in the FBS SP+ database — the same
    scope as the Power Ratings table). Keyed by the EXACT team-name string
    used in games_out's home_team/away_team fields (the Odds API's
    'School Mascot' form), so the dashboard's own team lookup can match by
    plain string equality with no extra normalization step on the JS side.
    Teams without a real rating are deliberately left out rather than
    padded with fabricated numbers — the dashboard's own fallback (generic
    gray helmet, full name as the tag) handles that case."""
    seen = {}
    for g in games_out:
        for side in ("home", "away"):
            name = g.get(f"{side}_team")
            if not name or name in seen:
                continue
            rating = match_sp_rating(name, sp_lookup)
            if rating is None:
                continue  # not in the FBS SP+ database — excluded, not faked
            meta = match_team(name, team_lookup)
            splits = match_sp_splits(name, sp_splits_lookup) or {}
            # Default to a neutral gray if CFBD didn't have a color on file —
            # keeps the helmet SVG from breaking on a null hex rather than
            # fabricating a team color.
            seen[name] = {
                "name": name,
                "abbr": derive_abbr(meta.get("school") or name, meta.get("abbreviation")),
                "conf": meta.get("conference"),
                "primary": meta.get("color") or "#8A94A3",
                "secondary": meta.get("alt_color") or "#E7EDF5",
                "net": round(rating, 2),
                "off_sp_rating": round(splits["off"], 2) if "off" in splits else None,
                "def_sp_rating": round(splits["def"], 2) if "def" in splits else None,
            }
    return list(seen.values())


def build_neutral_site_lookup(games_df: pd.DataFrame) -> dict:
    """frozenset({normalized_home_school, normalized_away_school}) -> bool
    neutralSite, from CFBD's OWN /games endpoint (which knows the full
    season's schedule, including neutral-site openers, well before kickoff
    — this isn't derived from results). Keyed by an unordered pair (not
    home/away-specific) since we're only matching THIS pipeline's home/away
    labeling (from The Odds API) against CFBD's schedule to ask 'is this
    matchup on a neutral field', not trying to reconcile which provider's
    home/away designation is authoritative.

    The Odds API itself doesn't expose a neutral-site flag at all — that's
    the actual gap this closes. Without it, the preseason model was
    applying a home-field edge to every game uniformly, including true
    neutral-site openers (e.g. season-opening 'showcase' games), which
    misattributes an advantage to a team that doesn't have one there."""
    lookup = {}
    if games_df is None or games_df.empty:
        return lookup
    for _, row in games_df.iterrows():
        home, away = row.get("homeTeam"), row.get("awayTeam")
        neutral = row.get("neutralSite")
        if not home or not away or pd.isna(neutral):
            continue
        key = frozenset({_normalize_team_name(str(home)), _normalize_team_name(str(away))})
        lookup[key] = bool(neutral)
    return lookup


def match_neutral_site(home_school: str, away_school: str, lookup: dict):
    """Returns True/False if this matchup is found in CFBD's schedule,
    None if unknown (falls back to 'assume normal home game' at the call
    site — the pre-existing, documented behavior — rather than guessing)."""
    if not home_school or not away_school:
        return None
    key = frozenset({_normalize_team_name(home_school), _normalize_team_name(away_school)})
    return lookup.get(key)


def compute_fair_odds_fields(home_rating, away_rating, moneyline_home, moneyline_away, prior,
                              neutral_site=False, home_injury_adj=0.0, away_injury_adj=0.0):
    """Returns a dict of model/EV fields, or a dict with has_model_line=False
    and a reason if either team's SP+ rating (or the book's moneyline) is
    missing — never fabricates a number when the inputs aren't there.

    home_injury_adj/away_injury_adj: manual points added directly to that
    team's own expected margin (see src/data/injury_overrides.py) — applied
    on top of the SP+-diff-and-home-field base margin, since neither of
    those inputs has any awareness of who's actually available to play.
    Defaults to 0.0 (no override), so existing callers are unaffected."""
    if home_rating is None or away_rating is None:
        return {"has_model_line": False, "no_line_reason": "missing_sp_rating"}
    if moneyline_home is None or moneyline_away is None or pd.isna(moneyline_home) or pd.isna(moneyline_away):
        return {"has_model_line": False, "no_line_reason": "missing_book_moneyline"}

    home_injury_adj = home_injury_adj or 0.0
    away_injury_adj = away_injury_adj or 0.0

    sp_diff = home_rating - away_rating
    base_margin = fo.preseason_predicted_margin(sp_diff, prior, neutral_site=neutral_site)
    model_margin = base_margin + home_injury_adj - away_injury_adj
    model_home_win_prob = fo.margin_to_win_prob(model_margin, prior["residual_std"])
    model_away_win_prob = 1 - model_home_win_prob

    book_home_raw = fo.american_to_implied_prob(float(moneyline_home))
    book_away_raw = fo.american_to_implied_prob(float(moneyline_away))
    book_home_fair, book_away_fair = fo.devig_two_way(book_home_raw, book_away_raw)

    return {
        "has_model_line": True,
        "neutral_site": bool(neutral_site),
        "injury_adjustment_home": round(home_injury_adj, 2),
        "injury_adjustment_away": round(away_injury_adj, 2),
        "home_sp_rating": round(home_rating, 2),
        "away_sp_rating": round(away_rating, 2),
        "sp_rating_diff": round(sp_diff, 2),
        "model_predicted_margin": round(model_margin, 2),
        "model_home_win_prob": round(model_home_win_prob, 4),
        "model_away_win_prob": round(model_away_win_prob, 4),
        "model_fair_ml_home": round(fo.prob_to_american(model_home_win_prob), 1),
        "model_fair_ml_away": round(fo.prob_to_american(model_away_win_prob), 1),
        "book_implied_prob_home": round(book_home_fair, 4) if book_home_fair is not None else None,
        "book_implied_prob_away": round(book_away_fair, 4) if book_away_fair is not None else None,
        "ev_home_pct": round(fo.ev_percent(model_home_win_prob, float(moneyline_home)), 2),
        "ev_away_pct": round(fo.ev_percent(model_away_win_prob, float(moneyline_away)), 2),
    }


def main(year: int):
    print(f"Fetching {year} team metadata (logos, colors)...")
    teams_df = cfbd.get_fbs_teams(year)
    team_lookup = build_team_lookup(teams_df)

    print(f"Fetching {year} schedule for neutral-site flags (The Odds API doesn't expose this)...")
    try:
        schedule_df = cfbd.get_games(year)
        neutral_site_lookup = build_neutral_site_lookup(schedule_df)
        print(f"  {len(neutral_site_lookup)} scheduled matchups with a known neutral-site flag")
    except Exception as e:
        print(f"  [warn] could not fetch {year} schedule for neutral-site flags: {e}")
        neutral_site_lookup = {}

    print("Loading manual injury/starter-availability overrides (config/injury_overrides.csv)...")
    injury_overrides = load_injury_overrides()
    if injury_overrides:
        unknown_teams = [ov["school"] for ov in injury_overrides.values()
                          if _normalize_team_name(ov["school"]) not in team_lookup]
        print(f"  {len(injury_overrides)} team(s) flagged")
        if unknown_teams:
            print(f"  [warn] these override team names don't match any known FBS school "
                  f"(check spelling against CFBD's 'school' field, e.g. 'Ohio State' not "
                  f"'Ohio State Buckeyes'): {unknown_teams}")
    else:
        print("  none active (config/injury_overrides.csv empty or not found — normal/default)")

    print(f"Loading {year} SP+ ratings for the preseason fair-odds estimate...")
    sp_path = f"{config.DATA_RAW_DIR}/sp_ratings_{year}.csv"
    sp_lookup = {}
    sp_splits_lookup = {}
    if os.path.exists(sp_path):
        sp_df = pd.read_csv(sp_path)
        sp_lookup = build_sp_lookup(sp_df, team_lookup=team_lookup)
        sp_splits_lookup = build_sp_splits_lookup(sp_df, team_lookup=team_lookup)
        if not sp_splits_lookup:
            print(f"  [warn] {sp_path} has no offense.rating/defense.rating columns — "
                  f"power-ratings off/def split will be omitted")
    else:
        print(f"  [warn] {sp_path} not found — model fair odds/EV will be skipped for all games")

    print("Fitting preseason prior (slope, intercept, residual std) from real historical games...")
    prior = None
    features_path = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    if os.path.exists(features_path) and sp_lookup:
        try:
            prior = fo.fit_preseason_prior(pd.read_csv(features_path))
            print(f"  margin ~= {prior['slope']:.3f} * sp_rating_diff + {prior['intercept']:.2f}, "
                  f"residual_std={prior['residual_std']:.2f} pts, n={prior['n_games']} historical games")
            if abs(prior["slope"] - 1.0) > 0.15:
                print(f"  [note] fitted slope ({prior['slope']:.3f}) is meaningfully off from 1.0 — "
                      f"SP+ rating diff does not translate 1:1 to point margin in this data, "
                      f"which is exactly why this is fit rather than assumed.")
        except ValueError as e:
            print(f"  [warn] {e}")
    elif not os.path.exists(features_path):
        print(f"  [warn] {features_path} not found — run build_features.py first. Skipping fair odds/EV.")

    game_lines_path = f"{config.DATA_CURRENT_DIR}/game_lines.csv"
    props_path = f"{config.DATA_CURRENT_DIR}/player_props.csv"

    games_out = []
    n_with_model_line = 0
    n_neutral_unknown = 0
    n_injury_applications = 0
    if os.path.exists(game_lines_path):
        lines = pd.read_csv(game_lines_path)
        for _, g in lines.iterrows():
            home_meta = match_team(g.get("home_team"), team_lookup)
            away_meta = match_team(g.get("away_team"), team_lookup)
            ml_home = g.get(f"ml_{str(g.get('home_team')).replace(' ', '_')}")
            ml_away = g.get(f"ml_{str(g.get('away_team')).replace(' ', '_')}")

            game_out = {
                "commence_time": g.get("commence_time"),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "home_logo": home_meta.get("logo"),
                "away_logo": away_meta.get("logo"),
                "home_color": home_meta.get("color"),
                "away_color": away_meta.get("color"),
                "book_used": g.get("book_used"),
                "moneyline_home": ml_home,
                "moneyline_away": ml_away,
                "spread_home": g.get(f"spread_{str(g.get('home_team')).replace(' ', '_')}"),
                "spread_away": g.get(f"spread_{str(g.get('away_team')).replace(' ', '_')}"),
                "spread_price_home": g.get(f"spread_price_{str(g.get('home_team')).replace(' ', '_')}"),
                "spread_price_away": g.get(f"spread_price_{str(g.get('away_team')).replace(' ', '_')}"),
                "total_over": g.get("total_over"),
                "total_under": g.get("total_under"),
                "home_unmatched": home_meta.get("unmatched", False),
                "away_unmatched": away_meta.get("unmatched", False),
            }

            if prior is not None:
                home_rating = match_sp_rating(g.get("home_team"), sp_lookup)
                away_rating = match_sp_rating(g.get("away_team"), sp_lookup)
                neutral = match_neutral_site(home_meta.get("school"), away_meta.get("school"), neutral_site_lookup)
                home_ov = get_team_override(home_meta.get("school"), injury_overrides)
                away_ov = get_team_override(away_meta.get("school"), injury_overrides)
                fair_fields = compute_fair_odds_fields(
                    home_rating, away_rating, ml_home, ml_away, prior,
                    neutral_site=bool(neutral),  # None (unknown) -> False, same as pre-existing default
                    home_injury_adj=(home_ov["points"] if home_ov else 0.0),
                    away_injury_adj=(away_ov["points"] if away_ov else 0.0),
                )
                game_out.update(fair_fields)
                if home_ov:
                    game_out["injury_notes_home"] = home_ov["notes"]
                    n_injury_applications += 1
                if away_ov:
                    game_out["injury_notes_away"] = away_ov["notes"]
                    n_injury_applications += 1
                if neutral is None:
                    n_neutral_unknown += 1
                if fair_fields.get("has_model_line"):
                    n_with_model_line += 1
            else:
                game_out["has_model_line"] = False
                game_out["no_line_reason"] = "no_prior_fitted"

            games_out.append(game_out)
    else:
        print(f"  [warn] {game_lines_path} not found, skipping games")

    props_out = []
    if os.path.exists(props_path):
        props = pd.read_csv(props_path)
        props_out = props.to_dict(orient="records")
    else:
        print(f"  [warn] {props_path} not found, skipping props")

    print("Fetching real player-prop market catalog (for placeholder tab)...")
    try:
        prop_market_catalog = prizepicks_client.get_prop_market_catalog()
    except Exception as e:
        print(f"  [warn] could not fetch prop market catalog: {e}")
        prop_market_catalog = []

    teams_out = build_teams_export(games_out, team_lookup, sp_lookup, sp_splits_lookup)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "team_metadata_year": year,
        "preseason_prior": prior,
        "teams": teams_out,
        "games": games_out,
        "props": props_out,
        "prop_market_catalog": prop_market_catalog,
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/latest.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    n_unmatched = sum(1 for g in games_out if g["home_unmatched"] or g["away_unmatched"])
    n_no_model = len(games_out) - n_with_model_line
    n_neutral = sum(1 for g in games_out if g.get("neutral_site"))
    print(f"Wrote docs/data/latest.json: {len(games_out)} games ({n_unmatched} with an "
          f"unmatched team logo), {n_with_model_line} with a model fair-odds/EV line, "
          f"{n_no_model} without (insufficient data), {n_neutral} flagged neutral-site "
          f"({n_neutral_unknown} had no neutral-site match in CFBD's schedule, "
          f"assumed a normal home game), {n_injury_applications} manual injury-override "
          f"adjustment(s) applied (of {len(injury_overrides)} teams flagged in "
          f"config/injury_overrides.csv), {len(teams_out)} teams with a real "
          f"SP+ rating exported, {len(props_out)} prop rows, "
          f"{len(prop_market_catalog)} prop market types in catalog.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True,
                         help="Season to pull team metadata (logos/colors) for.")
    args = parser.parse_args()
    main(args.year)
