#!/usr/bin/env python3
"""
One-off audit: re-scores a specific, hand-entered list of already-completed
games using TODAY'S model code -- specifically to check whether the
opponent-adjustment fix to build_current_season_form (added 9/2026) would
have caught the Week 3 misses (Texas A&M, Louisiana, South Alabama, UCF,
Western Michigan, Florida Atlantic, Toledo, Texas Tech) if it had been live
at the time.

Reconstructs each team's pre-game "current form" using ONLY games that
were actually completed BEFORE the audited weekend started (9/17/2026) --
not today's full schedule, which by now includes several more weeks of
games these teams have since played. Using today's full schedule would
leak information the model never actually had at kickoff into its own
prediction, which would make this comparison meaningless.

Runs the SAME game twice per row -- once through the OLD (unadjusted)
formula and once through the NEW (opponent-adjusted) one -- so the two
columns are a clean apples-to-apples comparison of the fix's real effect,
not just two different single numbers with no baseline.

Usage:
  python scripts/audit_predictions.py --year 2025
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd

import config
from src.data import cfbd_client as cfbd
from src.features.live_features import (
    build_current_season_form, score_with_trained_model,
    build_current_pace_returning, build_current_core_ratings,
)
from src.models.game_model import GameMarginModel
from scripts.export_dashboard_data import (
    build_team_lookup, build_sp_lookup, match_sp_rating, match_team,
    build_weather_lookup, match_weather, current_cfb_season_year,
)

# Cutoff before the audited weekend started -- every game below is reduced
# to "what did the model know as of this date," not today's full schedule.
ASOF_DATE = "2026-09-17"

# The 27 games from the Week 3 (9/19-9/20/2026) review, hand-entered from
# the actual tracked/reviewed slate: (home, away, home_pts, away_pts,
# market_home_spread). Real final scores, not re-fetched, since these
# markets have long since closed and aren't in any live feed anymore.
AUDIT_GAMES = [
    ("UCLA", "Purdue", 52, 38, -14.0),
    ("San Jose State", "Fresno State", 10, 26, 6.5),
    ("San Diego State", "James Madison", 13, 26, -1.5),
    ("Louisiana", "UAB", 21, 14, -7.5),
    ("Maryland", "Virginia Tech", 26, 35, 3.0),
    ("Ole Miss", "LSU", 32, 24, 3.0),
    ("Colorado State", "BYU", 23, 41, 17.5),
    ("Rice", "Western Michigan", 21, 28, 9.5),   # WMU -9.5 as the away favorite
    ("South Alabama", "Ohio", 41, 36, -7.0),
    ("Middle Tennessee", "Nevada", 27, 20, 3.5),
    ("UCF", "Georgia State", 44, 30, -17.5),
    ("Jacksonville State", "Georgia Southern", 31, 27, -3.5),
    ("Auburn", "Florida", 39, 44, 2.5),
    ("Missouri State", "Marshall", 24, 30, 4.0),
    ("Florida Atlantic", "FIU", 16, 10, -6.5),
    ("Old Dominion", "East Carolina", 17, 20, -3.0),
    ("South Carolina", "Mississippi State", 34, 41, -4.0),
    ("Duke", "Stanford", 35, 7, -10.0),
    ("Liberty", "Ball State", 51, 15, -14.5),    # BALL +14.5 as the away dog
    ("Louisville", "SMU", 41, 31, -1.5),
    ("Texas A&M", "Kentucky", 21, 31, -16.5),
    ("Toledo", "Temple", 49, 48, -6.0),
    ("Central Michigan", "Wyoming", 24, 10, -1.5),
    ("Clemson", "North Carolina", 28, 20, -3.0),
    ("Texas State", "North Texas", 49, 35, -2.5),  # UNT +2.5 as the away dog
    ("Delaware", "Coastal Carolina", 22, 14, -5.5),  # CCU +5.5 as the away dog
    ("Texas Tech", "Houston", 28, 26, -7.5),
]


def main(year: int):
    season_year = current_cfb_season_year()
    print(f"Fetching {year} team metadata and {season_year} full-season schedule...")
    teams_df = cfbd.get_fbs_teams(year)
    team_lookup = build_team_lookup(teams_df)
    schedule_df = cfbd.get_games(season_year)

    print(f"Loading {year} SP+ ratings...")
    sp_path = f"{config.DATA_RAW_DIR}/sp_ratings_{year}.csv"
    if not os.path.exists(sp_path):
        print(f"  [error] {sp_path} not found -- run the normal pipeline's SP+ fetch first")
        return
    sp_df = pd.read_csv(sp_path)
    sp_lookup = build_sp_lookup(sp_df, team_lookup=team_lookup)

    model_path = f"{config.MODELS_DIR}/game_model.joblib"
    if not os.path.exists(model_path):
        print(f"  [error] {model_path} not found -- run scripts/train_game_model.py first")
        return
    trained_model = GameMarginModel.load(model_path)

    print(f"Fetching {season_year} pace/returning-production and CORE ratings...")
    adv_stats_df = cfbd.get_advanced_team_stats(season_year)
    returning_df = cfbd.get_returning_production(season_year)
    pace_returning_lookup = build_current_pace_returning(adv_stats_df, returning_df)
    core_ratings_df = cfbd.get_core_ratings(year)
    core_ratings_lookup = build_current_core_ratings(core_ratings_df)
    weather_df = cfbd.get_weather(season_year)
    weather_lookup = build_weather_lookup(weather_df)

    if "startDate" in schedule_df.columns:
        prior_games = schedule_df[schedule_df["startDate"] < ASOF_DATE]
        print(f"\n{len(prior_games)} games completed before {ASOF_DATE} used to reconstruct 'current form' "
              f"as it actually stood before this weekend -- not today's full schedule.")
    else:
        print(f"\n[warn] no startDate column found on the schedule -- falling back to the full current "
              f"schedule, which may leak later weeks' results into 'current form' and make this comparison "
              f"less accurate than intended.")
        prior_games = schedule_df

    # OLD: no opponent adjustment (exactly what was live at kickoff that
    # weekend). NEW: today's opponent-adjusted formula. Both built from the
    # SAME pre-weekend game set, so the only thing that differs between
    # them is the fix itself.
    old_form = build_current_season_form(prior_games)
    new_form = build_current_season_form(prior_games, sp_lookup=sp_lookup)

    print(f"\nRe-scoring {len(AUDIT_GAMES)} completed games\n")
    header = (f"{'Matchup':<34} {'Actual':>7} {'Market':>7} "
              f"{'OLD pred':>9} {'OLD miss':>9}   {'NEW pred':>9} {'NEW miss':>9}   {'Better?'}")
    print(header)
    print("-" * len(header))

    old_total_miss, new_total_miss, n_scored = 0.0, 0.0, 0
    for home, away, home_pts, away_pts, market_home_spread in AUDIT_GAMES:
        home_meta = match_team(home, team_lookup)
        away_meta = match_team(away, team_lookup)
        home_school = home_meta.get("school") or home
        away_school = away_meta.get("school") or away

        home_rating = match_sp_rating(home, sp_lookup)
        away_rating = match_sp_rating(away, sp_lookup)
        game_weather = match_weather(home_school, away_school, weather_lookup)
        actual_margin = home_pts - away_pts

        old_pred = score_with_trained_model(
            home_school, away_school, home_rating, away_rating, False,
            old_form, trained_model, pace_returning=pace_returning_lookup,
            core_ratings=core_ratings_lookup, weather=game_weather,
        )
        new_pred = score_with_trained_model(
            home_school, away_school, home_rating, away_rating, False,
            new_form, trained_model, pace_returning=pace_returning_lookup,
            core_ratings=core_ratings_lookup, weather=game_weather,
        )

        matchup = f"{away} @ {home}"
        if old_pred is None or new_pred is None:
            print(f"{matchup:<34} {actual_margin:>+7.0f} {-market_home_spread:>+7.1f}   "
                  f"[missing feature(s) for this game -- couldn't re-score]")
            continue

        old_miss = abs(old_pred - actual_margin)
        new_miss = abs(new_pred - actual_margin)
        old_total_miss += old_miss
        new_total_miss += new_miss
        n_scored += 1
        better = "YES" if new_miss < old_miss - 0.5 else ("no" if new_miss > old_miss + 0.5 else "~same")

        print(f"{matchup:<34} {actual_margin:>+7.0f} {-market_home_spread:>+7.1f}   "
              f"{old_pred:>+9.1f} {old_miss:>9.1f}   {new_pred:>+9.1f} {new_miss:>9.1f}   {better}")

    if n_scored:
        print("\n" + "-" * len(header))
        print(f"Average absolute miss -- OLD formula: {old_total_miss / n_scored:.2f} pts   "
              f"NEW formula: {new_total_miss / n_scored:.2f} pts   "
              f"({n_scored} of {len(AUDIT_GAMES)} games scored)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True,
                         help="Season to pull team metadata/SP+ ratings for (same as the normal pipeline).")
    args = parser.parse_args()
    main(args.year)
