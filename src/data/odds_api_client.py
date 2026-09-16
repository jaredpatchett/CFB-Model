"""
Client for The Odds API (the-odds-api.com) — used for CURRENT/upcoming CFB
moneyline, spread, totals, AND player prop lines from real sportsbooks
(DraftKings, FanDuel, BetMGM, Caesars, ESPN Bet, etc). Free tier does not
include historical odds, so this is only used for "what are the live lines
right now" — backtesting uses CFBD's historical lines instead (see
cfbd_client.get_historical_lines).

Player props intentionally come from real sportsbooks only, NOT daily
fantasy sites like PrizePicks/Underdog — see src/data/prizepicks_client.py
(no longer called by scripts/fetch_current_lines.py) for the old DFS path.
Note "Onyx Odds" is not one of The Odds API's covered bookmakers (checked
their full bookmaker list across every region) — it simply isn't available
through this provider, on any plan.

Docs: https://the-odds-api.com/liveapi/guides/v4/
      https://the-odds-api.com/sports-odds-data/betting-markets.html#player-props-api-markets
      https://the-odds-api.com/sports-odds-data/bookmaker-apis.html
"""
import requests
import pandas as pd
import sys
import os
from datetime import datetime, timezone, timedelta

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


# ---------------------------------------------------------------------------
# Player props (real sportsbooks)
# ---------------------------------------------------------------------------

# Real-sportsbook priority order for player props. DFS sites (prizepicks,
# underdog, pick6, dabble_us_dfs) are deliberately excluded from this list —
# the user only wants "real sportsbook" lines. First book in this list that
# has posted a given player/market/side wins; later books only fill in gaps
# the higher-priority books didn't post. See
# https://the-odds-api.com/sports-odds-data/bookmaker-apis.html for the full
# bookmaker list this was drawn from — note "onyx"/"Onyx Odds" is not a
# bookmaker The Odds API covers at all, on any plan.
PROP_BOOK_PRIORITY = [
    "draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet",
    "betrivers", "fanatics", "bovada", "betonlineag", "mybookieag",
    "ballybet", "betparx", "fliff", "hardrockbet",
]

# Player-prop market keys The Odds API defines for NCAAF (shared with the
# NFL/CFL prop catalog — see the "NFL, NCAAF, CFL Player Props API" table at
# https://the-odds-api.com/sports-odds-data/betting-markets.html). Restricted
# to true Over/Under numeric markets so every row fits the dashboard's
# existing Over/Under prop-card layout; Yes/No markets like
# player_anytime_td are intentionally left out — they'd need a different UI.
PROP_MARKET_NAMES = {
    "player_pass_yds": "Pass Yards",
    "player_pass_tds": "Pass Touchdowns",
    "player_pass_completions": "Pass Completions",
    "player_pass_attempts": "Pass Attempts",
    "player_pass_interceptions": "Pass Interceptions",
    "player_rush_yds": "Rush Yards",
    "player_rush_tds": "Rush Touchdowns",
    "player_rush_attempts": "Rush Attempts",
    "player_receptions": "Receptions",
    "player_reception_yds": "Reception Yards",
    "player_reception_tds": "Reception Touchdowns",
}
PROP_MARKET_KEYS = list(PROP_MARKET_NAMES.keys())


def get_prop_market_catalog() -> list:
    """Static catalog of the player-prop markets requested from real
    sportsbooks via The Odds API — used to render 'NOT POSTED' placeholders
    for markets no book has priced yet. Replaces the old PrizePicks/OddsPapi
    catalog call (see prizepicks_client.get_prop_market_catalog, no longer
    used) — The Odds API doesn't have a live "markets" endpoint to query, so
    this list is drawn directly from their published market docs instead."""
    return sorted(PROP_MARKET_NAMES.values())


def get_event_player_props(event_id: str, regions: str = "us,us2",
                            odds_format: str = "american") -> dict:
    """Player props for ONE NCAAF game from real sportsbooks. Unlike game
    lines, player props are NOT included in the bulk /sports/{sport}/odds
    response — The Odds API only returns them one event at a time from this
    endpoint. Returns {} if this event has no props posted by any covered
    book yet (normal outside the days right before kickoff, and normal for
    lower-profile matchups that never get props at all)."""
    config.require_keys("ODDS_API_KEY")
    url = f"{config.ODDS_API_BASE_URL}/sports/{config.ODDS_API_NCAAF_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": config.ODDS_API_KEY,
        "regions": regions,
        "markets": ",".join(PROP_MARKET_KEYS),
        "oddsFormat": odds_format,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"[odds_api] (props) requests remaining: {remaining}")
    return resp.json()


def get_ncaaf_player_props(events: list, regions: str = "us,us2",
                            max_days_out: float = 8.0) -> pd.DataFrame:
    """Player props for every upcoming NCAAF game from real sportsbooks,
    flattened to one row per player/market with over_price/under_price side
    by side — the same shape prizepicks_client.get_prizepicks_props() used
    to produce, so nothing downstream (score_prop, renderProps) needed to
    change. `events` should be the raw list from get_ncaaf_odds() — reused
    here instead of a second bulk call, just to read each game's id and
    commence_time.

    Real sportsbooks only post CFB props a few days out at most, so games
    further than `max_days_out` away are skipped rather than burning a
    per-event API call on a game nothing has been posted for yet."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_days_out)
    rows = []
    n_checked = 0
    n_with_props = 0

    for game in events:
        commence = game.get("commence_time")
        commence_dt = None
        if commence:
            try:
                commence_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            except ValueError:
                commence_dt = None
        if commence_dt and (commence_dt < now or commence_dt > cutoff):
            continue
        event_id = game.get("id")
        if not event_id:
            continue

        n_checked += 1
        try:
            event_data = get_event_player_props(event_id, regions=regions)
        except requests.exceptions.RequestException as e:
            print(f"  [warn] props fetch failed for {game.get('away_team')} @ {game.get('home_team')}: {e}")
            continue

        books_by_key = {b.get("key"): b for b in event_data.get("bookmakers", [])}
        if not books_by_key:
            continue

        # (player_name, market_name, line) -> {"over_price", "under_price", "book_used"}
        merged = {}
        for book_key in PROP_BOOK_PRIORITY:
            book = books_by_key.get(book_key)
            if not book:
                continue
            for market in book.get("markets", []):
                mname = PROP_MARKET_NAMES.get(market.get("key"))
                if not mname:
                    continue
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description")  # player name lives here for prop markets
                    side = outcome.get("name")  # "Over" / "Under"
                    line = outcome.get("point")
                    if not player or side not in ("Over", "Under") or line is None:
                        continue
                    slot = merged.setdefault((player, mname, line), {})
                    price_field = "over_price" if side == "Over" else "under_price"
                    if price_field not in slot:  # a higher-priority book already filled this side
                        slot[price_field] = outcome.get("price")
                        slot["book_used"] = book_key

        if merged:
            n_with_props += 1
        for (player, mname, line), slot in merged.items():
            rows.append({
                "fixture_id": event_id,
                "start_time": commence,
                "player_name": player,
                "market_name": mname,
                "line": line,
                "over_price": slot.get("over_price"),
                "under_price": slot.get("under_price"),
                "book_used": slot.get("book_used"),
            })

    print(f"[odds_api] checked {n_checked} upcoming game(s) for player props "
          f"({regions}), {n_with_props} had at least one real-sportsbook prop posted")
    return pd.DataFrame(rows)
