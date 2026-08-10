"""
Client for The Odds API (the-odds-api.com) — used for CURRENT/upcoming CFB
moneyline, spread, and totals lines from real sportsbooks (DraftKings, FanDuel,
etc). Free tier does not include historical odds, so this is only used for
"what are the live lines right now" — backtesting uses CFBD's historical lines
instead (see cfbd_client.get_historical_lines).

Docs: https://the-odds-api.com/liveapi/guides/v4/
"""
import requests
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config


def get_ncaaf_odds(regions: str = "us", markets: str = "h2h,spreads,totals",
                    odds_format: str = "american") -> list:
    """Raw list of upcoming/live NCAAF games with bookmaker odds."""
    config.require_keys("ODDS_API_KEY")
    url = f"{config.ODDS_API_BASE_URL}/sports/{config.ODDS_API_NCAAF_KEY}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    # The Odds API returns remaining quota in response headers — useful to log
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None:
        print(f"[odds_api] requests used: {used}, remaining: {remaining}")
    return resp.json()


def odds_to_dataframe(raw_odds: list, preferred_book: str = "draftkings") -> pd.DataFrame:
    """Flatten the nested Odds API response into one row per game with the
    preferred book's line (falls back to the first book listed if the
    preferred one isn't offering this game)."""
    rows = []
    for game in raw_odds:
        bookmakers = {b["key"]: b for b in game.get("bookmakers", [])}
        book = bookmakers.get(preferred_book) or (list(bookmakers.values())[0] if bookmakers else None)
        row = {
            "game_id": game.get("id"),
            "commence_time": game.get("commence_time"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "book_used": book.get("key") if book else None,
        }
        if book:
            for market in book.get("markets", []):
                key = market.get("key")
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "").replace(" ", "_")
                    if key == "h2h":
                        row[f"ml_{name}"] = outcome.get("price")
                    elif key == "spreads":
                        row[f"spread_{name}"] = outcome.get("point")
                        row[f"spread_price_{name}"] = outcome.get("price")
                    elif key == "totals":
                        row[f"total_{name.lower()}"] = outcome.get("point")
        rows.append(row)
    return pd.DataFrame(rows)
