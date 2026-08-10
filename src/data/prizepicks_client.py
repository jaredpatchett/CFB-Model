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


def get_player_prop_markets(sport_id: int) -> pd.DataFrame:
    """List all markets flagged playerProp=True for this sport, keeping the
    full metadata (marketName, handicap i.e. the O/U line, marketType,
    outcomes) — not just the IDs — so prop rows can carry a human-readable
    stat name and line instead of opaque numeric IDs."""
    markets = _get("/markets", {"sportId": sport_id})
    df = pd.json_normalize(markets)
    if df.empty:
        return df
    return df[df["playerProp"] == True] if "playerProp" in df.columns else df


def get_prop_market_catalog() -> list:
    """Real list of player-prop market names OddsPapi/PrizePicks support for
    football (e.g. 'Passing Yards', 'Receptions'), independent of whether any
    fixture currently has odds posted for them. Used to show 'coming soon'
    placeholders for markets that exist but aren't priced yet this far out
    from kickoff, instead of guessing at market names."""
    sport_id = resolve_football_sport_id()
    markets_df = get_player_prop_markets(sport_id)
    if markets_df.empty or "marketName" not in markets_df.columns:
        return []
    names = sorted(set(markets_df["marketName"].dropna().tolist()))
    return names


def _build_outcome_name_lookup(markets_raw: list) -> dict:
    """Map (marketId, outcomeId) -> outcomeName (e.g. 'Over'/'Under'), since
    outcome names live nested inside each market's own 'outcomes' list in the
    /v4/markets response, not in the /odds-by-tournaments response."""
    lookup = {}
    for m in markets_raw:
        for o in m.get("outcomes", []):
            lookup[(m.get("marketId"), o.get("outcomeId"))] = o.get("outcomeName")
    return lookup


def get_prizepicks_props(bookmaker: str = "prizepicks") -> pd.DataFrame:
    """Fetch current PrizePicks player prop lines for NCAAF, flattened to one
    row per player/market/outcome, WITH the stat name and line (handicap)
    attached — e.g. market_name='Receiving Yards', line=64.5, outcome='Over'.
    Without this join, rows would just be opaque numeric IDs and unusable for
    a dashboard or any human-facing output."""
    sport_id = resolve_football_sport_id()
    tournament_id = resolve_ncaaf_tournament_id(sport_id)

    markets_raw = _get("/markets", {"sportId": sport_id})
    prop_markets_df = pd.json_normalize(markets_raw)
    prop_market_ids = set()
    market_meta = {}
    if not prop_markets_df.empty and "playerProp" in prop_markets_df.columns:
        prop_rows = prop_markets_df[prop_markets_df["playerProp"] == True]
        prop_market_ids = set(prop_rows["marketId"])
        for _, row in prop_rows.iterrows():
            market_meta[row["marketId"]] = {
                "market_name": row.get("marketName"),
                "line": row.get("handicap"),
                "market_type": row.get("marketType"),
                "period": row.get("period"),
            }
    outcome_name_lookup = _build_outcome_name_lookup(
        [m for m in markets_raw if m.get("marketId") in prop_market_ids]
    )

    fixtures = _get("/odds-by-tournaments", {
        "bookmaker": bookmaker,
        "tournamentIds": tournament_id,
    })

    rows = []
    for fixture in fixtures:
        book_data = fixture.get("bookmakerOdds", {}).get(bookmaker, {})
        markets = book_data.get("markets", {})
        for market_id_str, market in markets.items():
            market_id = int(market_id_str)
            if prop_market_ids and market_id not in prop_market_ids:
                continue  # skip non-player-prop markets (moneyline, spread, etc.)
            meta = market_meta.get(market_id, {})
            for outcome_id_str, outcome in market.get("outcomes", {}).items():
                outcome_id = int(outcome_id_str)
                outcome_name = outcome_name_lookup.get((market_id, outcome_id))
                for _, player_line in outcome.get("players", {}).items():
                    rows.append({
                        "fixture_id": fixture.get("fixtureId"),
                        "start_time": fixture.get("startTime"),
                        "player_name": player_line.get("playerName"),
                        "market_name": meta.get("market_name"),
                        "line": meta.get("line"),
                        "outcome": outcome_name,
                        "price": player_line.get("price"),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # One row per (player, market) with Over and Under side by side is far
    # more useful than one row per outcome — pivot so callers get
    # over_price/under_price directly instead of having to self-join.
    pivoted = df.pivot_table(
        index=["fixture_id", "start_time", "player_name", "market_name", "line"],
        columns="outcome", values="price", aggfunc="first"
    ).reset_index()
    pivoted.columns = [c if isinstance(c, str) and c not in ("Over", "Under")
                        else f"{c.lower()}_price" for c in pivoted.columns]
    return pivoted
