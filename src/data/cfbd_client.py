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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config


def _get(endpoint: str, params: dict = None) -> list:
    config.require_keys("CFBD_API_KEY")
    headers = {"Authorization": f"Bearer {config.CFBD_API_KEY}"}
    url = f"{config.CFBD_BASE_URL}{endpoint}"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
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
    """Advanced efficiency stats (success rate, PPA, explosiveness) by team."""
    params = {"year": year}
    if week is not None:
        params["week"] = week
    data = _get("/stats/season/advanced", params)
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


def get_historical_lines(year: int, week: int = None, season_type: str = "regular") -> pd.DataFrame:
    """Historical betting lines (spread, over/under, moneyline) by game and
    sportsbook. This is CFBD's free historical odds — used for backtesting
    since The Odds API's free tier has no historical odds.
    """
    params = {"year": year, "seasonType": season_type}
    if week is not None:
        params["week"] = week
    data = _get("/lines", params)
    return pd.json_normalize(data)
