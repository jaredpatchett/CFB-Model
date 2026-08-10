"""
Client for OddsPapi (oddspapi.io) — used to pull PrizePicks player prop lines
for college football. PrizePicks doesn't offer a public API of its own;
OddsPapi proxies it.

IMPORTANT: OddsPapi's sportId and marketId values are NOT fixed constants —
they're assigned dynamically per their system (see their docs: "This is the
value you should use when sending the sport parameter to any other endpoint").
So this client resolves them at runtime by name/slug instead of hardcoding
numbers that could be wrong or change. Results are cached in-memory per run.

Docs: https://oddspapi.io/us/docs
"""
import requests
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

_cache = {}


def _get(endpoint: str, params: dict = None) -> list:
    config.require_keys("ODDSPAPI_API_KEY")
    params = dict(params or {})
    params["apiKey"] = config.ODDSPAPI_API_KEY
    url = f"{config.ODDSPAPI_BASE_URL}{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def resolve_football_sport_id() -> int:
    """Find the sportId for American football."""
    if "sport_id" in _cache:
        return _cache["sport_id"]
    sports = _get("/sports")
    candidates = [s for s in sports if "american" in s.get("sportName", "").lower()
                  or "american" in s.get("slug", "").lower()
                  or s.get("slug") == "american-football"]
    if not candidates:
        raise ValueError(
            f"Could not find an American football sportId in OddsPapi's /sports "
            f"response. Available sports: {[s.get('sportName') for s in sports]}"
        )
    sport_id = candidates[0]["sportId"]
    _cache["sport_id"] = sport_id
    return sport_id


def resolve_ncaaf_tournament_id(sport_id: int) -> int:
    """Find the tournamentId for NCAAF / college football within American football."""
    key = f"tournament_id_{sport_id}"
    if key in _cache:
        return _cache[key]
    tournaments = _get("/tournaments", {"sportId": sport_id})
    candidates = [
        t for t in tournaments
        if "ncaa" in t.get("tournamentName", "").lower()
        or "college" in t.get("tournamentName", "").lower()
        or "ncaa" in t.get("tournamentSlug", "").lower()
        or "college" in t.get("tournamentSlug", "").lower()
    ]
    if not candidates:
        raise ValueError(
            f"Could not find an NCAAF tournamentId. Available tournaments for "
            f"sportId={sport_id}: {[t.get('tournamentName') for t in tournaments]}"
        )
    tournament_id = candidates[0]["tournamentId"]
    _cache[key] = tournament_id
    return tournament_id


def get_player_prop_market_ids(sport_id: int) -> pd.DataFrame:
    """List all markets flagged playerProp=True for this sport, so we know
    which marketIds in the odds response correspond to player props (vs
    team-level markets like moneyline/spread)."""
    markets = _get("/markets", {"sportId": sport_id})
    df = pd.json_normalize(markets)
    if df.empty:
        return df
    return df[df["playerProp"] == True] if "playerProp" in df.columns else df


def get_prizepicks_props(bookmaker: str = "prizepicks") -> pd.DataFrame:
    """Fetch current PrizePicks player prop lines for NCAAF, flattened to one
    row per player/market/outcome."""
    sport_id = resolve_football_sport_id()
    tournament_id = resolve_ncaaf_tournament_id(sport_id)
    prop_markets = get_player_prop_market_ids(sport_id)
    prop_market_ids = set(prop_markets["marketId"]) if not prop_markets.empty else set()

    fixtures = _get("/odds-by-tournaments", {
        "bookmaker": bookmaker,
        "tournamentIds": tournament_id,
    })

    rows = []
    for fixture in fixtures:
        book_data = fixture.get("bookmakerOdds", {}).get(bookmaker, {})
        markets = book_data.get("markets", {})
        for market_id, market in markets.items():
            if prop_market_ids and int(market_id) not in prop_market_ids:
                continue  # skip non-player-prop markets (moneyline, spread, etc.)
            for outcome_id, outcome in market.get("outcomes", {}).items():
                for _, player_line in outcome.get("players", {}).items():
                    rows.append({
                        "fixture_id": fixture.get("fixtureId"),
                        "start_time": fixture.get("startTime"),
                        "market_id": market_id,
                        "outcome_id": outcome_id,
                        "player_name": player_line.get("playerName"),
                        "price": player_line.get("price"),
                        "bookmaker_outcome_id": player_line.get("bookmakerOutcomeId"),
                    })
    return pd.DataFrame(rows)
