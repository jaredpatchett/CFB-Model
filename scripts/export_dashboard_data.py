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
    mascot}."""
    lookup = {}
    for _, row in teams_df.iterrows():
        school = row.get("school")
        if not school:
            continue
        logos = row.get("logos")
        logo_url = None
        if isinstance(logos, list) and logos:
            logo_url = logos[0]
        lookup[_normalize_team_name(school)] = {
            "school": school,
            "logo": logo_url,
            "color": row.get("color"),
            "alt_color": row.get("alt_color"),
            "conference": row.get("conference"),
            "mascot": row.get("mascot"),
        }
    return lookup


def _best_substring_match(norm: str, lookup: dict):
    """Return the LOOKUP KEY with the longest substring match against norm,
    or None. Longest match, not first match, matters here: dict iteration
    order is arbitrary (CFBD's own row order), and picking the first
    substring hit is a real bug — e.g. 'new mexico' (a separate real FBS
    team, the New Mexico Lobos) is a substring of 'new mexico state aggies'
    (New Mexico State), so 'first match wins' silently attaches the WRONG
    team's data. Preferring the longest matching key picks 'new mexico
    state' correctly instead."""
    best_key, best_len = None, 0
    for key in lookup.keys():
        if (key in norm or norm in key) and len(key) > best_len:
            best_key, best_len = key, len(key)
    return best_key


def match_team(name: str, lookup: dict) -> dict:
    if not name:
        return {}
    norm = _normalize_team_name(name)
    if norm in TEAM_ALIASES:
        norm = TEAM_ALIASES[norm]
    if norm in lookup:
        return lookup[norm]
    best_key = _best_substring_match(norm, lookup)
    if best_key is not None:
        return lookup[best_key]
    return {"school": name, "logo": None, "color": None, "alt_color": None,
            "conference": None, "mascot": None, "unmatched": True}


def build_sp_lookup(sp_df: pd.DataFrame) -> dict:
    """normalized team name -> SP+ rating, for the fair-odds/EV preseason
    estimate. Same name-matching approach as build_team_lookup, since SP+
    ratings and CFBD team logos come from the same 'school' naming
    convention."""
    lookup = {}
    if sp_df is None or sp_df.empty or "team" not in sp_df.columns:
        return lookup
    for _, row in sp_df.iterrows():
        team = row.get("team")
        rating = row.get("rating")
        if not team or pd.isna(rating):
            continue
        lookup[_normalize_team_name(team)] = float(rating)
    return lookup


def match_sp_rating(name: str, lookup: dict):
    if not name:
        return None
    norm = _normalize_team_name(name)
    if norm in TEAM_ALIASES:
        norm = TEAM_ALIASES[norm]
    if norm in lookup:
        return lookup[norm]
    best_key = _best_substring_match(norm, lookup)
    return lookup[best_key] if best_key is not None else None


def compute_fair_odds_fields(home_rating, away_rating, moneyline_home, moneyline_away, prior):
    """Returns a dict of model/EV fields, or a dict with has_model_line=False
    and a reason if either team's SP+ rating (or the book's moneyline) is
    missing — never fabricates a number when the inputs aren't there."""
    if home_rating is None or away_rating is None:
        return {"has_model_line": False, "no_line_reason": "missing_sp_rating"}
    if moneyline_home is None or moneyline_away is None or pd.isna(moneyline_home) or pd.isna(moneyline_away):
        return {"has_model_line": False, "no_line_reason": "missing_book_moneyline"}

    sp_diff = home_rating - away_rating
    model_margin = fo.preseason_predicted_margin(sp_diff, prior)
    model_home_win_prob = fo.margin_to_win_prob(model_margin, prior["residual_std"])
    model_away_win_prob = 1 - model_home_win_prob

    book_home_raw = fo.american_to_implied_prob(float(moneyline_home))
    book_away_raw = fo.american_to_implied_prob(float(moneyline_away))
    book_home_fair, book_away_fair = fo.devig_two_way(book_home_raw, book_away_raw)

    return {
        "has_model_line": True,
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

    print(f"Loading {year} SP+ ratings for the preseason fair-odds estimate...")
    sp_path = f"{config.DATA_RAW_DIR}/sp_ratings_{year}.csv"
    sp_lookup = {}
    if os.path.exists(sp_path):
        sp_lookup = build_sp_lookup(pd.read_csv(sp_path))
    else:
        print(f"  [warn] {sp_path} not found — model fair odds/EV will be skipped for all games")

    print("Fitting preseason prior (home-field edge, residual std) from real historical games...")
    prior = None
    features_path = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    if os.path.exists(features_path) and sp_lookup:
        try:
            prior = fo.fit_preseason_prior(pd.read_csv(features_path))
            print(f"  home_field_adv={prior['home_field_adv']:.2f} pts, "
                  f"residual_std={prior['residual_std']:.2f} pts, n={prior['n_games']} historical games")
        except ValueError as e:
            print(f"  [warn] {e}")
    elif not os.path.exists(features_path):
        print(f"  [warn] {features_path} not found — run build_features.py first. Skipping fair odds/EV.")

    game_lines_path = f"{config.DATA_CURRENT_DIR}/game_lines.csv"
    props_path = f"{config.DATA_CURRENT_DIR}/player_props.csv"

    games_out = []
    n_with_model_line = 0
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
                fair_fields = compute_fair_odds_fields(home_rating, away_rating, ml_home, ml_away, prior)
                game_out.update(fair_fields)
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

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "team_metadata_year": year,
        "preseason_prior": prior,
        "games": games_out,
        "props": props_out,
        "prop_market_catalog": prop_market_catalog,
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/latest.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    n_unmatched = sum(1 for g in games_out if g["home_unmatched"] or g["away_unmatched"])
    n_no_model = len(games_out) - n_with_model_line
    print(f"Wrote docs/data/latest.json: {len(games_out)} games ({n_unmatched} with an "
          f"unmatched team logo), {n_with_model_line} with a model fair-odds/EV line, "
          f"{n_no_model} without (insufficient data), {len(props_out)} prop rows, "
          f"{len(prop_market_catalog)} prop market types in catalog.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True,
                         help="Season to pull team metadata (logos/colors) for.")
    args = parser.parse_args()
    main(args.year)
