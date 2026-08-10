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
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data import cfbd_client as cfbd


def _normalize_team_name(name: str) -> str:
    """Best-effort normalization so 'Ohio State Buckeyes' (Odds API style)
    can match 'Ohio State' (CFBD 'school' field). Strips common mascot-style
    trailing words is too error-prone (mascots vary wildly), so instead this
    strips punctuation/case only and relies on substring/alias matching in
    build_team_lookup below."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


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


def match_team(name: str, lookup: dict) -> dict:
    if not name:
        return {}
    norm = _normalize_team_name(name)
    if norm in TEAM_ALIASES:
        norm = TEAM_ALIASES[norm]
    if norm in lookup:
        return lookup[norm]
    # fallback: substring match (e.g. "ohio state buckeyes" contains "ohio state")
    for key, meta in lookup.items():
        if key in norm or norm in key:
            return meta
    return {"school": name, "logo": None, "color": None, "alt_color": None,
            "conference": None, "mascot": None, "unmatched": True}


def main(year: int):
    print(f"Fetching {year} team metadata (logos, colors)...")
    teams_df = cfbd.get_fbs_teams(year)
    team_lookup = build_team_lookup(teams_df)

    game_lines_path = f"{config.DATA_CURRENT_DIR}/game_lines.csv"
    props_path = f"{config.DATA_CURRENT_DIR}/player_props.csv"

    games_out = []
    if os.path.exists(game_lines_path):
        lines = pd.read_csv(game_lines_path)
        for _, g in lines.iterrows():
            home_meta = match_team(g.get("home_team"), team_lookup)
            away_meta = match_team(g.get("away_team"), team_lookup)
            games_out.append({
                "commence_time": g.get("commence_time"),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "home_logo": home_meta.get("logo"),
                "away_logo": away_meta.get("logo"),
                "home_color": home_meta.get("color"),
                "away_color": away_meta.get("color"),
                "book_used": g.get("book_used"),
                "moneyline_home": g.get(f"ml_{str(g.get('home_team')).replace(' ', '_')}"),
                "moneyline_away": g.get(f"ml_{str(g.get('away_team')).replace(' ', '_')}"),
                "spread_home": g.get(f"spread_{str(g.get('home_team')).replace(' ', '_')}"),
                "spread_away": g.get(f"spread_{str(g.get('away_team')).replace(' ', '_')}"),
                "total_over": g.get("total_over"),
                "total_under": g.get("total_under"),
                "home_unmatched": home_meta.get("unmatched", False),
                "away_unmatched": away_meta.get("unmatched", False),
            })
    else:
        print(f"  [warn] {game_lines_path} not found, skipping games")

    props_out = []
    if os.path.exists(props_path):
        props = pd.read_csv(props_path)
        props_out = props.to_dict(orient="records")
    else:
        print(f"  [warn] {props_path} not found, skipping props")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "team_metadata_year": year,
        "games": games_out,
        "props": props_out,
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/latest.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    n_unmatched = sum(1 for g in games_out if g["home_unmatched"] or g["away_unmatched"])
    print(f"Wrote docs/data/latest.json: {len(games_out)} games ({n_unmatched} with an "
          f"unmatched team logo), {len(props_out)} prop rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True,
                         help="Season to pull team metadata (logos/colors) for.")
    args = parser.parse_args()
    main(args.year)
