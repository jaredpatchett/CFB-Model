#!/usr/bin/env python3
"""
Pull TODAY's live NCAAF lines (moneyline/spread/totals AND player props,
both from real sportsbooks via The Odds API) and save to data/current/.

Player props previously came from PrizePicks via OddsPapi (see
src/data/prizepicks_client.py) — that path is no longer used. Real
sportsbooks (DraftKings, FanDuel, BetMGM, Caesars, ESPN Bet, etc) don't
guarantee prop coverage on every game the way a DFS site does, so expect
props to show up mainly for ranked/nationally-relevant matchups, and be
thin or empty elsewhere — that's real book coverage, not a bug.

Usage:
  python scripts/fetch_current_lines.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data import odds_api_client

if __name__ == "__main__":
    os.makedirs(config.DATA_CURRENT_DIR, exist_ok=True)

    print("Fetching game lines (The Odds API)...")
    raw_odds = odds_api_client.get_ncaaf_odds()
    odds_df = odds_api_client.odds_to_dataframe(raw_odds)
    odds_df.to_csv(f"{config.DATA_CURRENT_DIR}/game_lines.csv", index=False)
    print(f"  {len(odds_df)} games")

    print("Fetching player props (real sportsbooks via The Odds API)...")
    try:
        props_df = odds_api_client.get_ncaaf_player_props(raw_odds)
        props_df.to_csv(f"{config.DATA_CURRENT_DIR}/player_props.csv", index=False)
        print(f"  {len(props_df)} prop lines")
    except Exception as e:
        print(f"  [warn] player props fetch failed: {e}")

    print("Done.")
