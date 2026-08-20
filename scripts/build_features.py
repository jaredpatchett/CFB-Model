#!/usr/bin/env python3
"""
Turn raw CFBD CSVs (from fetch_historical_data.py) into model-ready feature
tables: data/processed/team_game_features.csv and player_game_features.csv.

Usage:
  python scripts/build_features.py --years 2022 2023 2024
"""
import argparse
import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.features.team_features import build_game_team_features, build_pace_returning_features
from src.features.player_features import pivot_player_game_stats, build_rolling_player_features


def load_years(pattern: str, years: list) -> pd.DataFrame:
    frames = []
    for year in years:
        path = pattern.format(year=year)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            print(f"  [warn] missing {path}, skipping")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", required=True)
    args = parser.parse_args()

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)

    print("Loading raw data...")
    games = load_years(config.DATA_RAW_DIR + "/games_{year}.csv", args.years)
    sp = load_years(config.DATA_RAW_DIR + "/sp_ratings_{year}.csv", args.years)
    adv_stats = load_years(config.DATA_RAW_DIR + "/adv_stats_{year}.csv", args.years)
    returning = load_years(config.DATA_RAW_DIR + "/returning_production_{year}.csv", args.years)
    player_stats_long = load_years(config.DATA_RAW_DIR + "/player_game_stats_{year}.csv", args.years)

    pace_returning = build_pace_returning_features(adv_stats, returning)
    if not pace_returning.empty:
        print(f"  pace/returning-production: {len(pace_returning)} team-year rows")
    else:
        print("  [warn] no pace/returning-production data found — pace_diff/returning_production_diff "
              "will be all-null for this build (run fetch_historical_data.py to get it)")

    print("Building team features...")
    team_features = build_game_team_features(games, sp, pace_returning)
    team_features.to_csv(f"{config.DATA_PROCESSED_DIR}/team_game_features.csv", index=False)
    print(f"  wrote {len(team_features)} rows")

    if not player_stats_long.empty:
        print("Building player features...")
        wide = pivot_player_game_stats(player_stats_long, games)
        player_features = build_rolling_player_features(wide)
        player_features.to_csv(f"{config.DATA_PROCESSED_DIR}/player_game_features.csv", index=False)
        print(f"  wrote {len(player_features)} rows")
    else:
        print("  [warn] no player stats found, skipping player features")

    print("Done.")
