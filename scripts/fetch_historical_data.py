#!/usr/bin/env python3
"""
Pull historical CFBD data (games, team stats, SP+ ratings, player game stats,
historical betting lines) for one or more seasons, needed for training and
backtesting. Saves raw CSVs to data/raw/.

Usage:
  python scripts/fetch_historical_data.py --years 2022 2023 2024
"""
import argparse
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data import cfbd_client as cfbd


def year_already_cached(year: int) -> bool:
    """True if this year's core historical data was already pulled by an
    earlier run (data/raw/ restored from the workflow's actions/cache step,
    or just re-running this script locally). A past, completed CFB season
    doesn't change, so once it's been fetched there's no reason to spend
    CFBD's scarce free-tier budget (1,000 calls/month — confirmed hit
    mid-run on a real Action run, a training pull across 4 years is ~80-90
    calls on its own) re-pulling the same season every single run.

    Checks the two priciest files as a proxy for 'this year fully
    completed': games (1 call) and player_game_stats (~15 calls, the bulk
    of a year's cost). adv_stats/returning_production are already
    best-effort/non-fatal elsewhere so aren't required here — a year missing
    only those will still be treated as cached; re-run with --force to
    retry them specifically if needed."""
    return (os.path.exists(f"{config.DATA_RAW_DIR}/games_{year}.csv")
            and os.path.exists(f"{config.DATA_RAW_DIR}/player_game_stats_{year}.csv"))


def fetch_year(year: int):
    print(f"\n--- Fetching {year} ---")
    games = cfbd.get_games(year)
    games.to_csv(f"{config.DATA_RAW_DIR}/games_{year}.csv", index=False)
    print(f"  games: {len(games)} rows")

    sp = cfbd.get_sp_ratings(year)
    sp.to_csv(f"{config.DATA_RAW_DIR}/sp_ratings_{year}.csv", index=False)
    print(f"  sp_ratings: {len(sp)} rows")

    try:
        adv_stats = cfbd.get_advanced_team_stats(year)
        adv_stats.to_csv(f"{config.DATA_RAW_DIR}/adv_stats_{year}.csv", index=False)
        print(f"  adv_stats (pace source): {len(adv_stats)} rows")
    except Exception as e:
        print(f"  [warn] adv_stats fetch failed: {e} — pace_diff will be unavailable for {year}")

    try:
        returning = cfbd.get_returning_production(year)
        returning.to_csv(f"{config.DATA_RAW_DIR}/returning_production_{year}.csv", index=False)
        print(f"  returning_production: {len(returning)} rows")
    except Exception as e:
        print(f"  [warn] returning_production fetch failed: {e} — returning_production_diff will be unavailable for {year}")

    team_stats = cfbd.get_team_season_stats(year)
    team_stats.to_csv(f"{config.DATA_RAW_DIR}/team_season_stats_{year}.csv", index=False)
    print(f"  team_season_stats: {len(team_stats)} rows")

    raw_lines = cfbd.get_historical_lines(year)
    lines = cfbd.historical_lines_to_dataframe(raw_lines)
    lines.to_csv(f"{config.DATA_RAW_DIR}/lines_{year}.csv", index=False)
    print(f"  historical lines: {len(lines)} games with a usable market line "
          f"(of {len(raw_lines)} games returned; the rest had no sportsbook coverage)")

    # Player game stats: fetched week by week (CFBD requires a week param here).
    # A small proactive delay between these ~15 calls/year keeps this well
    # under CFBD's free-tier rate limit in the first place (cfbd_client._get
    # still retries with backoff if a 429 slips through anyway).
    max_week = int(games["week"].max()) if "week" in games.columns and len(games) else 15
    all_player_stats = []
    for week in range(1, max_week + 1):
        try:
            wk = cfbd.get_player_game_stats(year, week)
            if not wk.empty:
                all_player_stats.append(wk)
        except Exception as e:
            print(f"  [warn] week {week} player stats failed: {e}")
        time.sleep(0.5)
    if all_player_stats:
        player_stats = pd.concat(all_player_stats, ignore_index=True)
        player_stats.to_csv(f"{config.DATA_RAW_DIR}/player_game_stats_{year}.csv", index=False)
        print(f"  player_game_stats: {len(player_stats)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", required=True,
                         help="Season(s) to fetch, e.g. --years 2022 2023 2024")
    parser.add_argument("--force", action="store_true",
                         help="Re-fetch every requested year even if already cached in data/raw/ "
                              "(normally a completed season already on disk is skipped to save "
                              "CFBD's limited free-tier call budget).")
    args = parser.parse_args()

    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    for year in args.years:
        if not args.force and year_already_cached(year):
            print(f"\n--- {year}: already cached in {config.DATA_RAW_DIR}/, skipping "
                  f"(~20+ CFBD calls saved — use --force to re-fetch anyway) ---")
            continue
        fetch_year(year)
    print("\nDone.")
