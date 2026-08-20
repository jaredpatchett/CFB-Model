"""
Client for CollegeFootballData.com (CFBD) — free API for team/player stats,
schedules, rosters, SP+ ratings, and (usefully) HISTORICAL betting lines.

CFBD's /lines endpoint gives us historical closing lines for backtesting, which
sidesteps the fact that The Odds API's free tier doesn't include historical odds.
The Odds API is used instead for CURRENT/upcoming lines (see odds_api_client.py).

Docs: https://api.collegefootballdata.com/api/docs/?url=/api-docs.json
"""
import requests
import pandas as pd
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# CFBD's free tier rate-limits (429 Too Many Requests) once a script makes
# many calls back-to-back -- e.g. fetch_historical_data.py's per-week player
# stats loop is ~15 calls/year, and across 4 training years plus the
# per-year games/sp/adv/returning/team-stats/lines calls that's ~90+ calls
# with zero delay between them. Confirmed for real: a live Action run threw
# "429 Client Error: Too Many Requests" partway through 2024 after 2022/2023
# had already succeeded (so it's a rate limit, not an auth/key problem).
# Retries with exponential backoff, honoring the server's Retry-After header
# when present (CFBD sends one), so a transient rate-limit or hiccup doesn't
# kill an otherwise-good run.
MAX_RETRIES = 6
BASE_BACKOFF_SECONDS = 5


def _get(endpoint: str, params: dict = None) -> list:
    config.require_keys("CFBD_API_KEY")
    headers = {"Authorization": f"Bearer {config.CFBD_API_KEY}"}
    url = f"{config.CFBD_BASE_URL}{endpoint}"
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == MAX_RETRIES:
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            else:
                wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
            print(f"  [warn] {resp.status_code} from {endpoint} (attempt {attempt + 1}/"
                  f"{MAX_RETRIES + 1}) — waiting {wait:.0f}s before retrying...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()


def get_fbs_teams(year: int) -> pd.DataFrame:
    """All FBS teams for a season, with conference/division info."""
    data = _get("/teams/fbs", {"year": year})
    return pd.json_normalize(data)


def get_games(year: int, week: int = None, season_type: str = "regular") -> pd.DataFrame:
    """Games/schedule + final scores for a season (or a single week)."""
    params = {"year": year, "seasonType": season_type}
    if week is not None:
        params["week"] = week
    data = _get("/games", params)
    return pd.json_normalize(data)


def get_team_season_stats(year: int) -> pd.DataFrame:
    """Basic season team stats (yards, plays, turnovers, etc.) by team."""
    data = _get("/stats/season", {"year": year})
    return pd.json_normalize(data)


def get_advanced_team_stats(year: int, week: int = None) -> pd.DataFrame:
    """Advanced efficiency stats (success rate, PPA, explosiveness) by team.
    week=None (the default) returns SEASON-level aggregates (AdvancedSeasonStat
    per CFBD's schema) — nested 'offense'/'defense' objects, flattened by
    json_normalize into dotted columns like 'offense.plays', 'offense.drives'.
    This is also where our pace proxy comes from — see
    team_features.build_pace_returning_features's docstring for why it's
    plays-per-drive, not a true seconds-per-play tempo stat."""
    params = {"year": year}
    if week is not None:
        params["week"] = week
    data = _get("/stats/season/advanced", params)
    return pd.json_normalize(data)


def get_returning_production(year: int) -> pd.DataFrame:
    """Percent of last season's total production (PPA-based) that's back on
    the roster this season, per team — CFBD's own computed metric (GET
    /player/returning, despite living under the 'player' API group, this is
    team-level aggregated data), not derived here. Real signal for "is this
    team the same team as last year's SP+ rating implies," which the
    preseason SP+ prior has no way to know on its own."""
    data = _get("/player/returning", {"year": year})
    return pd.json_normalize(data)


def get_sp_ratings(year: int) -> pd.DataFrame:
    """Bill Connelly's SP+ ratings — a strong single-number team strength prior."""
    data = _get("/ratings/sp", {"year": year})
    return pd.json_normalize(data)


def get_player_season_stats(year: int, category: str = None) -> pd.DataFrame:
    """Season-level player stats. category e.g. 'passing','rushing','receiving'."""
    params = {"year": year}
    if category:
        params["category"] = category
    data = _get("/stats/player/season", params)
    return pd.json_normalize(data)


def get_player_game_stats(year: int, week: int, season_type: str = "regular") -> pd.DataFrame:
    """Box-score-level player stats for a given week. Needed to build rolling
    per-player usage features (targets/game, carries/game, etc.) without leakage."""
    params = {"year": year, "week": week, "seasonType": season_type}
    data = _get("/games/players", params)
    # This endpoint returns a nested structure: games -> teams -> categories -> types -> athletes
    rows = []
    for game in data:
        for team in game.get("teams", []):
            for category in team.get("categories", []):
                for stat_type in category.get("types", []):
                    for athlete in stat_type.get("athletes", []):
                        rows.append({
                            "gameId": game.get("id"),
                            "team": team.get("team"),
                            "category": category.get("name"),
                            "statType": stat_type.get("name"),
                            "athleteId": athlete.get("id"),
                            "player": athlete.get("name"),
                            "stat": athlete.get("stat"),
                        })
    return pd.DataFrame(rows)


def get_roster(year: int, team: str = None) -> pd.DataFrame:
    """Roster with position, so we can filter player features to relevant positions
    (QB/RB/WR/TE) for props modeling."""
    params = {"year": year}
    if team:
        params["team"] = team
    data = _get("/roster", params)
    return pd.json_normalize(data)


def get_historical_lines(year: int, week: int = None, season_type: str = "regular") -> list:
    """Historical betting lines (spread, over/under, moneyline) by game and
    sportsbook. This is CFBD's free historical odds — used for backtesting
    since The Odds API's free tier has no historical odds.

    Returns the RAW list of per-game objects, each shaped like:
      {id, season, seasonType, week, startDate, homeTeamId, homeTeam,
       homeConference, homeScore, awayTeamId, awayTeam, awayConference,
       awayScore, lines: [{provider, spread, formattedSpread, spreadOpen,
       overUnder, overUnderOpen, homeMoneyline, awayMoneyline}, ...]}
    (per CFBD's own OpenAPI schema for this endpoint — BettingGame/GameLine).

    Deliberately NOT run through pd.json_normalize() here: normalize() does
    not explode a nested LIST field like 'lines' — it would leave it as a
    column of raw Python list objects, which silently turns into an
    unusable stringified list ("[{'provider': ...}]" as literal text) the
    moment it's saved to CSV. That's exactly why the backtest was never
    wired up: lines_{year}.csv had per-game rows but no actual usable
    spread/moneyline columns. Call historical_lines_to_dataframe() on this
    return value to get a real one-row-per-game DataFrame instead.
    """
    params = {"year": year, "seasonType": season_type}
    if week is not None:
        params["week"] = week
    return _get("/lines", params)


# Provider preference order for picking ONE sportsbook's line per game out
# of however many CFBD has for it. 'consensus' (a blended line CFBD
# computes itself) is preferred when present since it doesn't depend on any
# single book covering a given game — useful for backtesting since
# lower-tier matchups often have thin book coverage. DraftKings is next,
# matching the book this pipeline already prefers for CURRENT lines (see
# odds_api_client.odds_to_dataframe's preferred_book="draftkings"), so
# backtested and live picks are graded against a consistent kind of source
# where possible.
DEFAULT_LINE_PROVIDER_PREFERENCE = ["consensus", "DraftKings", "Bovada", "Caesars"]


def historical_lines_to_dataframe(raw_games: list, provider_preference: list = None) -> pd.DataFrame:
    """Flatten CFBD's nested /lines response (see get_historical_lines'
    docstring) into ONE row per game, picking a single provider's numbers
    per game via provider_preference (falling back to whichever provider is
    first in that specific game's own 'lines' array if none of the
    preferred ones covered it).

    Games with an empty 'lines' array (no sportsbook covered them — not
    uncommon for lower-tier matchups) are dropped entirely rather than
    given a fabricated line.

    NOTE on sign convention: CFBD's numeric 'spread' field is documented as
    the home team's spread using the standard book convention (negative =
    home favored) — the same convention this pipeline already assumes
    everywhere else (spread_home, backtester.py's market_home_spread). This
    hasn't been independently verified against a live sample since this
    dev sandbox can't reach api.collegefootballdata.com directly; the first
    real fetch should spot-check a game's numeric 'spread' against its own
    human-readable 'formattedSpread' string before trusting a backtest run
    on it.
    """
    prefs = provider_preference or DEFAULT_LINE_PROVIDER_PREFERENCE
    rows = []
    for g in raw_games:
        lines = g.get("lines") or []
        if not lines:
            continue
        by_provider = {l.get("provider"): l for l in lines if l.get("provider")}
        chosen = None
        for p in prefs:
            if p in by_provider:
                chosen = by_provider[p]
                break
        if chosen is None:
            chosen = lines[0]
        rows.append({
            "id": g.get("id"),
            "season": g.get("season"),
            "week": g.get("week"),
            "seasonType": g.get("seasonType"),
            "startDate": g.get("startDate"),
            "homeTeam": g.get("homeTeam"),
            "awayTeam": g.get("awayTeam"),
            "homeScore": g.get("homeScore"),
            "awayScore": g.get("awayScore"),
            "line_provider": chosen.get("provider"),
            "market_spread_home": chosen.get("spread"),
            "market_over_under": chosen.get("overUnder"),
            "market_moneyline_home": chosen.get("homeMoneyline"),
            "market_moneyline_away": chosen.get("awayMoneyline"),
        })
    return pd.DataFrame(rows)
