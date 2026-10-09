#!/usr/bin/env python3
"""
Pull past-season receptions prop lines from The Odds API (added 10/2026).

Run from the Actions tab. Feeds props lab 3, which grades the receptions
model and its matchup versions against these lines for profit. Does not
touch the live model, the dashboard or the daily run.

For every college game of each season asked for, this reads the receptions
lines that were posted a set time before kickoff (default 2 hours) and
saves every book's line and both prices.

Cost: 10 credits for each game that had receptions lines, nothing for a game
that had none, and 1 credit a day to list that day's games. A full season is
roughly 4,000-8,500 credits. Two safety limits, both checked before every
request:
  --max-credits   stop once this run has spent this much
  --reserve       stop if the key's remaining balance would fall below this,
                  so the daily model runs are never starved
A run that stops early loses nothing: every game already read is recorded,
and the next run picks up where this one stopped without paying again.

Saved (prices from sportsbooks, the same kind of data the repo already keeps
in data/clv)
  data/props_history/receptions_<season>.csv   one row per game, book, player, line
  data/props_history/events_<season>.csv       every game looked at, and what came back

Usage:
  python scripts/pull_props_history.py --seasons 2024 2025
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import requests

import config

OUT_DIR = "data/props_history"
MARKET = "player_receptions"
REGION = "us"
FMT = "%Y-%m-%dT%H:%M:%SZ"
SEASON_START, SEASON_END = (8, 20), (12, 16)         # regular season and conference championships
DAY_STARTS_HOUR = 10                                 # a "day" of games runs 10:00 UTC to 10:00 UTC (6am Eastern)
SAVE_EVERY = 40
LINE_COLS = ["season", "event_id", "commence_time", "home_team", "away_team", "snapshot", "book", "player", "line", "over", "under"]
EVENT_COLS = ["season", "event_id", "commence_time", "home_team", "away_team", "asked_for", "status", "lines", "credits"]


class Stop(Exception):
    """Ends the run early, keeping everything pulled so far."""


class Budget:
    def __init__(self, max_credits, reserve):
        self.max_credits, self.reserve = max_credits, reserve
        self.spent, self.remaining = 0, None

    def check(self, next_cost):
        if self.spent + next_cost > self.max_credits:
            raise Stop(f"this run's limit of {self.max_credits:,} credits is reached")
        if self.remaining is not None and self.remaining - next_cost < self.reserve:
            raise Stop(f"the key is down to {self.remaining:,} credits and {self.reserve:,} are held back for the daily runs")

    def record(self, headers):
        try:
            self.spent += int(float(headers.get("x-requests-last") or 0))
        except (TypeError, ValueError):
            pass
        try:
            self.remaining = int(float(headers.get("x-requests-remaining")))
        except (TypeError, ValueError):
            pass


def get(path, params, budget, cost, soft=()):
    """One GET. Returns the parsed body, or None when there is nothing for
    this request. Raises Stop when the run should end."""
    budget.check(cost)
    params = dict(params, apiKey=config.ODDS_API_KEY)
    for attempt in range(5):
        try:
            resp = requests.get(f"{config.ODDS_API_BASE_URL}{path}", params=params, timeout=30)
        except requests.RequestException as e:
            if attempt == 4:
                raise Stop(f"The Odds API could not be reached ({type(e).__name__})")
            time.sleep(3 * (attempt + 1))
            continue
        if resp.status_code == 200:
            budget.record(resp.headers)
            try:
                return resp.json()
            except ValueError:
                return None
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
            time.sleep(3 * (attempt + 1))
            continue
        if resp.status_code in soft:
            # for one game: no such game at that time, or nothing saved for it
            budget.record(resp.headers)
            return None
        try:
            msg = (resp.json() or {}).get("message")
        except ValueError:
            msg = (resp.text or "")[:160]
        raise Stop(f"The Odds API refused the request (status {resp.status_code}: {msg})")
    return None


def game_days(season):
    d = datetime(season, *SEASON_START, DAY_STARTS_HOUR, tzinfo=timezone.utc)
    end = min(datetime(season, *SEASON_END, DAY_STARTS_HOUR, tzinfo=timezone.utc), datetime.now(timezone.utc) - timedelta(hours=6))
    while d < end:
        yield d
        d += timedelta(days=1)


def parse_lines(data, season, snapshot):
    """Event odds -> one row per (book, player, line) with both prices."""
    rows = {}
    for bk in (data or {}).get("bookmakers") or []:
        for m in bk.get("markets") or []:
            if m.get("key") != MARKET:
                continue
            for o in m.get("outcomes") or []:
                player, side, point, price = o.get("description"), str(o.get("name") or "").lower(), o.get("point"), o.get("price")
                if not player or side not in ("over", "under") or point is None or price is None:
                    continue
                key = (bk.get("key"), player, float(point))
                row = rows.setdefault(key, {"season": season, "event_id": data.get("id"), "commence_time": data.get("commence_time"),
                                            "home_team": data.get("home_team"), "away_team": data.get("away_team"), "snapshot": snapshot,
                                            "book": bk.get("key"), "player": player, "line": float(point), "over": None, "under": None})
                row[side] = float(price)
    return list(rows.values())


def load(path, cols):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame(columns=cols)


def pull_season(season, hours_before, budget):
    """Returns (lines, events, finished). Saves as it goes."""
    lpath, epath = f"{OUT_DIR}/receptions_{season}.csv", f"{OUT_DIR}/events_{season}.csv"
    lines, events = load(lpath, LINE_COLS), load(epath, EVENT_COLS)
    done = set(events["event_id"].astype(str))
    new_lines, new_events, finished, why = [], [], True, ""

    def save():
        nonlocal lines, events, new_lines, new_events
        if new_events:
            events = pd.concat([events, pd.DataFrame(new_events, columns=EVENT_COLS)], ignore_index=True)
        if new_lines:
            lines = pd.concat([lines, pd.DataFrame(new_lines, columns=LINE_COLS)], ignore_index=True)
        new_lines, new_events = [], []
        events.to_csv(epath, index=False)
        lines.to_csv(lpath, index=False)

    try:
        for day in game_days(season):
            listing = get(f"/historical/sports/{config.ODDS_API_NCAAF_KEY}/events",
                          {"date": day.strftime(FMT), "commenceTimeFrom": day.strftime(FMT),
                           "commenceTimeTo": (day + timedelta(days=1) - timedelta(seconds=1)).strftime(FMT)}, budget, 1)
            for ev in (listing or {}).get("data") or []:
                eid = str(ev.get("id"))
                if not ev.get("id") or eid in done or not ev.get("commence_time"):
                    continue
                kick = datetime.fromisoformat(str(ev["commence_time"]).replace("Z", "+00:00"))
                asked = (kick - timedelta(hours=hours_before)).strftime(FMT)
                before = budget.spent
                body = get(f"/historical/sports/{config.ODDS_API_NCAAF_KEY}/events/{eid}/odds",
                           {"date": asked, "regions": REGION, "markets": MARKET, "oddsFormat": "american"}, budget, 10, soft=(404, 422))
                rows = parse_lines((body or {}).get("data") or {}, season, (body or {}).get("timestamp"))
                for r in rows:                                  # the listing is the reliable source for who played
                    r.update(event_id=eid, commence_time=ev["commence_time"], home_team=ev.get("home_team"), away_team=ev.get("away_team"))
                new_lines.extend(rows)
                new_events.append({"season": season, "event_id": eid, "commence_time": ev["commence_time"], "home_team": ev.get("home_team"),
                                   "away_team": ev.get("away_team"), "asked_for": asked, "status": "lines" if rows else "none",
                                   "lines": len(rows), "credits": budget.spent - before})
                done.add(eid)
                if len(new_events) >= SAVE_EVERY:
                    save()
                time.sleep(0.05)
    except Stop as e:
        finished, why = False, str(e)
    save()
    return lines, events, finished, why


def main(seasons, hours_before, max_credits, reserve):
    config.require_keys("ODDS_API_KEY")
    os.makedirs(OUT_DIR, exist_ok=True)
    budget = Budget(max_credits, reserve)
    print(f"Reading receptions lines {hours_before:g}h before kickoff. Limits: {max_credits:,} credits this run, {reserve:,} held back.")
    stopped = ""
    for season in seasons:
        if stopped:
            print(f"{season}: not started")
            continue
        lines, events, finished, why = pull_season(season, hours_before, budget)
        with_lines = int((events["status"] == "lines").sum()) if len(events) else 0
        players = lines.groupby("event_id")["player"].nunique().mean() if len(lines) else 0
        print(f"{season}: {len(events)} games looked at, {with_lines} had receptions lines "
              f"({lines['player'].nunique() if len(lines) else 0} different players, {players:.1f} per game, "
              f"{len(lines)} book lines){'' if finished else ' -- NOT FINISHED'}")
        if not finished:
            stopped = why
    print(f"\nCredits spent this run: {budget.spent:,}   remaining on the key: "
          f"{budget.remaining if budget.remaining is None else format(budget.remaining, ',')}")
    if stopped:
        print(f"STOPPED EARLY: {stopped}. Everything read so far is saved; run again to continue from here.")
    else:
        print("FINISHED: every game of the seasons asked for has been read.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--hours-before", type=float, default=2.0)
    ap.add_argument("--max-credits", type=int, default=20000)
    ap.add_argument("--reserve", type=int, default=25000)
    a = ap.parse_args()
    main(sorted(a.seasons), a.hours_before, a.max_credits, a.reserve)
