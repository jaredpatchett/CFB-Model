"""
PFF Developer API client (added 10/2026) -- routes run and other charted
receiving data for the props model.

Needs a paid PFF Pro subscription and an API key (created at
pff.com/account/api-keys, starts with "ak_live_"). The key is read from the
PFF_API_KEY environment variable; in GitHub Actions it comes from a repo
secret. It is never printed or written to a file.

API facts this client relies on (from PFF's public docs / OpenAPI spec):
  - Base URL https://api.pff.com, key sent as "Authorization: Bearer <key>".
  - /v1 endpoints take snake_case query parameters.
  - GET /v1/auth/whoami        -> who the key belongs to; never rate-counted.
  - GET /v1/leagues            -> leagues with their seasons and weeks.
  - GET /v1/facet/<report>/summary?league=ncaa&season=YYYY&week=N
                               -> one row per player for that week.
  - Budget: 100 reads per minute per account. Every counted response carries
    x-ratelimit-remaining / x-ratelimit-reset; over budget is a 429 with
    Retry-After.

IMPORTANT: this repo is public. PFF data is licensed to the subscriber, so
raw PFF numbers must stay in data/raw (gitignored) and never be committed or
written into docs/.
"""
import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.pff.com"
LEAGUE = "ncaa"
MAX_ATTEMPTS = 4


class PFFError(RuntimeError):
    """A PFF API failure, carrying the API's own error code and reason."""

    def __init__(self, status, code=None, reason=None, message=None):
        self.status, self.code, self.reason = status, code, reason
        super().__init__(f"PFF API {status} {code or ''} {('(' + reason + ')') if reason else ''}: {message or ''}".strip())


def _key() -> str:
    key = (os.environ.get("PFF_API_KEY") or "").strip()
    if not key:
        raise EnvironmentError(
            "Missing PFF_API_KEY. Add your PFF Pro API key as a GitHub Actions secret "
            "(or to .env locally)."
        )
    return key


def _get(path: str, params: dict = None) -> dict:
    """GET one endpoint, waiting out rate limits and brief outages."""
    headers = {"Authorization": f"Bearer {_key()}", "Accept": "application/json"}
    last = None
    for attempt in range(MAX_ATTEMPTS):
        resp = requests.get(f"{BASE_URL}{path}", headers=headers, params=params or {}, timeout=60)
        if resp.status_code == 200:
            # Slow down before the budget runs out rather than after.
            try:
                if int(resp.headers.get("x-ratelimit-remaining", "99")) <= 2:
                    reset = float(resp.headers.get("x-ratelimit-reset", "0"))
                    time.sleep(min(max(reset - time.time(), 1), 65))
            except (TypeError, ValueError):
                pass
            return resp.json()
        try:
            err = (resp.json() or {}).get("error") or {}
        except ValueError:
            err = {}
        last = PFFError(resp.status_code, err.get("code"), (err.get("details") or {}).get("reason"), err.get("message"))
        if resp.status_code in (429, 503, 502, 504) and attempt < MAX_ATTEMPTS - 1:
            try:
                wait = float(resp.headers.get("Retry-After", "") or 0)
            except ValueError:
                wait = 0
            time.sleep(min(max(wait, 5 * (attempt + 1)), 65))
            continue
        raise last
    raise last


def whoami() -> dict:
    """What the API thinks of this key: tier, entitled, credential type."""
    return _get("/v1/auth/whoami")


def get_leagues() -> list:
    return _get("/v1/leagues").get("leagues") or []


def ncaa_calendar() -> dict:
    """{'seasons': [...], 'weeks': [week ids]} for college football."""
    for lg in get_leagues():
        if str(lg.get("slug", "")).lower() == LEAGUE:
            return {"seasons": sorted(lg.get("seasons") or []),
                    "weeks": sorted(w.get("id") for w in (lg.get("weeks") or []) if w.get("id") is not None)}
    return {"seasons": [], "weeks": []}


def _rows(payload: dict) -> list:
    """Reports come back as {'<report>_summary': [rows]} -- take the row list
    whatever the key is called."""
    if isinstance(payload, list):
        return payload
    for value in (payload or {}).values():
        if isinstance(value, list):
            return value
    return []


def get_facet_summary(report: str, season: int, week: int) -> pd.DataFrame:
    """One row per player for one college week of a league-wide report
    ('receiving', 'rushing', 'passing', ...). One week = one game per team,
    so this is game-level data."""
    payload = _get(f"/v1/facet/{report}/summary", {"league": LEAGUE, "season": season, "week": week})
    df = pd.DataFrame(_rows(payload))
    if not df.empty:
        df["season"], df["week"] = season, week
    return df


def get_receiving_summary(season: int, week: int) -> pd.DataFrame:
    """Charted receiving data for one week -- includes routes run and targets."""
    return get_facet_summary("receiving", season, week)
